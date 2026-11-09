//! The private transport runtime and a completion cell that sync and async
//! callers can wait on without touching the caller's runtime (D29).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

use opentelemetry::context::FutureExt as _;
use opentelemetry::Context;
use tokio::runtime::{Builder, Handle, Runtime};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::error::{Error, Result};

/// Name of every thread the transport runtime starts.
pub(crate) const THREAD_NAME: &str = "xybrid-observability-transport";

/// Grace period for runtime teardown (D29.4).
pub(crate) const TEARDOWN_GRACE: Duration = Duration::from_millis(100);

/// Blocking threads of the transport runtime. reqwest resolves host names on
/// this pool (`spawn_blocking`), and the pipeline parks at most two blocking
/// SDK calls on it at a time: one serialized metrics flush and the metrics
/// shutdown (provider shutdowns run only after all transport work ended).
/// Four threads therefore always leave at least two for name resolution, so
/// lifecycle calls can never starve the exports they wait for.
pub(crate) const MAX_BLOCKING_THREADS: usize = 4;

/// Far enough in the future to mean "no deadline", near enough that adding it
/// to any monotonic instant cannot overflow (the same horizon Tokio uses).
const FAR_FUTURE: Duration = Duration::from_secs(86_400 * 365 * 30);

/// `now + duration`, saturating: a `Duration` such as `Duration::MAX` is a
/// valid limit or lifecycle timeout and means "practically never", never a
/// panic (`Instant + Duration` panics on overflow).
pub(crate) fn after(now: Instant, duration: Duration) -> Instant {
    now.checked_add(duration.min(FAR_FUTURE))
        .or_else(|| now.checked_add(Duration::from_secs(86_400)))
        .unwrap_or(now)
}

/// [`after`] for Tokio instants.
pub(crate) fn tokio_after(now: tokio::time::Instant, duration: Duration) -> tokio::time::Instant {
    tokio::time::Instant::from_std(after(now.into_std(), duration))
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A pipeline's (or standalone exporter set's) own Tokio runtime.
///
/// Only user-facing handles own it; tasks hold a [`Handle`] and tokens, so
/// the runtime is never dropped from inside one of its own tasks.
pub(crate) struct TransportRuntime {
    runtime: Mutex<Option<Runtime>>,
    handle: Handle,
    root: CancellationToken,
}

impl TransportRuntime {
    /// Starts the runtime: one worker thread plus at most
    /// [`MAX_BLOCKING_THREADS`] blocking threads, all named [`THREAD_NAME`].
    /// Call only after validation.
    pub(crate) fn start() -> Result<Arc<Self>> {
        let runtime = Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(MAX_BLOCKING_THREADS)
            .thread_name(THREAD_NAME)
            .enable_io()
            .enable_time()
            .build()
            .map_err(|_| Error::Lifecycle {
                operation: "build",
                message: "could not start the transport runtime".into(),
            })?;
        let handle = runtime.handle().clone();
        Ok(Arc::new(Self {
            runtime: Mutex::new(Some(runtime)),
            handle,
            root: CancellationToken::new(),
        }))
    }

    pub(crate) fn handle(&self) -> &Handle {
        &self.handle
    }

    /// A token cancelled when the runtime is torn down.
    pub(crate) fn token(&self) -> CancellationToken {
        self.root.child_token()
    }

    /// Cancels all transport work and stops the runtime. Waits at most
    /// [`TEARDOWN_GRACE`] for its threads, and never blocks inside an async
    /// context (where it lets them stop in the background).
    pub(crate) fn teardown(&self) {
        self.root.cancel();
        let runtime = lock(&self.runtime).take();
        if let Some(runtime) = runtime {
            if Handle::try_current().is_ok() || std::thread::panicking() {
                runtime.shutdown_background();
            } else {
                runtime.shutdown_timeout(TEARDOWN_GRACE);
            }
        }
    }
}

impl Drop for TransportRuntime {
    fn drop(&mut self) {
        self.teardown();
    }
}

impl std::fmt::Debug for TransportRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportRuntime")
            .field("running", &lock(&self.runtime).is_some())
            .finish()
    }
}

/// Spawns transport work with telemetry suppressed, so exports never trace
/// themselves (D26.8).
pub(crate) fn spawn<F>(handle: &Handle, future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    handle.spawn(future.with_context(Context::new().with_telemetry_suppressed()))
}

/// A write-once value that threads wait on with a timeout and futures await
/// on any executor.
#[derive(Debug)]
pub(crate) struct Completion<T> {
    value: Mutex<Option<T>>,
    ready: Condvar,
    notify: Notify,
}

impl<T: Clone> Completion<T> {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            value: Mutex::new(None),
            ready: Condvar::new(),
            notify: Notify::new(),
        })
    }

    /// Stores `value` unless a value is already set; returns whether it won.
    pub(crate) fn complete(&self, value: T) -> bool {
        let mut slot = lock(&self.value);
        if slot.is_some() {
            return false;
        }
        *slot = Some(value);
        drop(slot);
        self.ready.notify_all();
        self.notify.notify_waiters();
        true
    }

    pub(crate) fn get(&self) -> Option<T> {
        lock(&self.value).clone()
    }

    /// Blocks the calling thread for at most `timeout`.
    pub(crate) fn wait_timeout(&self, timeout: Duration) -> Option<T> {
        let deadline = after(Instant::now(), timeout);
        let mut slot = lock(&self.value);
        loop {
            if let Some(value) = slot.as_ref() {
                return Some(value.clone());
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            slot = self
                .ready
                .wait_timeout(slot, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Waits without a timeout; runtime-agnostic.
    pub(crate) async fn wait(&self) -> T {
        loop {
            let mut notified = std::pin::pin!(self.notify.notified());
            notified.as_mut().enable();
            if let Some(value) = self.get() {
                return value;
            }
            notified.await;
        }
    }
}

/// A spawned task that is aborted when its handle is dropped, so work a
/// caller stopped waiting for does not keep running.
pub(crate) struct AbortOnDrop<T>(pub(crate) tokio::task::JoinHandle<T>);

impl<T> Future for AbortOnDrop<T> {
    type Output = std::result::Result<T, tokio::task::JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx)
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Completes a [`Completion`] with a fallback value if the task holding it
/// ends (or is dropped by a stopped runtime) without completing it.
pub(crate) struct CompleteOnDrop<T: Clone> {
    completion: Arc<Completion<T>>,
    fallback: Option<T>,
}

impl<T: Clone> CompleteOnDrop<T> {
    pub(crate) fn new(completion: Arc<Completion<T>>, fallback: T) -> Self {
        Self {
            completion,
            fallback: Some(fallback),
        }
    }

    pub(crate) fn complete(mut self, value: T) {
        self.fallback = None;
        self.completion.complete(value);
    }
}

impl<T: Clone> Drop for CompleteOnDrop<T> {
    fn drop(&mut self) {
        if let Some(fallback) = self.fallback.take() {
            self.completion.complete(fallback);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_is_write_once_and_waitable_from_threads() {
        let c = Completion::<u32>::new();
        assert_eq!(c.wait_timeout(Duration::from_millis(5)), None);
        let writer = c.clone();
        let t = std::thread::spawn(move || writer.complete(7));
        assert_eq!(c.wait_timeout(Duration::from_secs(5)), Some(7));
        assert!(t.join().unwrap());
        assert!(!c.complete(8));
        assert_eq!(c.get(), Some(7));
    }

    #[test]
    fn complete_on_drop_uses_the_fallback() {
        let c = Completion::<&str>::new();
        drop(CompleteOnDrop::new(c.clone(), "dropped"));
        assert_eq!(c.get(), Some("dropped"));
        let c = Completion::<&str>::new();
        CompleteOnDrop::new(c.clone(), "dropped").complete("done");
        assert_eq!(c.get(), Some("done"));
    }

    #[test]
    fn deadlines_saturate_instead_of_panicking() {
        let now = Instant::now();
        assert!(after(now, Duration::MAX) > now + Duration::from_secs(86_400 * 365));
        assert_eq!(
            after(now, Duration::from_millis(5)),
            now + Duration::from_millis(5)
        );
        let t = tokio::time::Instant::now();
        assert!(tokio_after(t, Duration::MAX) > t);
        let c = Completion::<u8>::new();
        c.complete(1);
        assert_eq!(c.wait_timeout(Duration::MAX), Some(1));
    }

    #[test]
    fn runtime_teardown_cancels_tokens_and_is_idempotent() {
        let rt = TransportRuntime::start().unwrap();
        let token = rt.token();
        let c = Completion::<bool>::new();
        let done = c.clone();
        let t = token.clone();
        spawn(rt.handle(), async move {
            t.cancelled().await;
            done.complete(true);
        });
        rt.teardown();
        assert!(token.is_cancelled());
        rt.teardown();
        // The task either observed the cancellation or was dropped.
        let _ = c.wait_timeout(Duration::from_millis(200));
    }
}

//! The pipeline's bounded batch processors for spans and log records:
//! admission by item and byte budget (D32), flush barriers and shutdown
//! (D30), per-signal counters (D31). Exports run on the transport runtime;
//! the hot path only takes a lock and pushes.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use opentelemetry::logs::Severity;
use opentelemetry::{Context, InstrumentationScope};
use opentelemetry_proto::tonic::logs::v1::LogRecord as ProtoLogRecord;
use opentelemetry_proto::tonic::trace::v1::Span as ProtoSpan;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::logs::{LogProcessor, SdkLogRecord};
use opentelemetry_sdk::trace::{Span, SpanData, SpanProcessor};
use opentelemetry_sdk::Resource;
use tokio::runtime::Handle;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::config::Limits;
use super::defaults;
use super::encode::{self, Estimator, Grouped, ResourceInfo, Scoped};
use super::export::{ExportCore, SETTLE_GRACE};
use super::runtime::{self, after, lock, Completion};
use super::stats::{Outcome, SignalStats};

/// A batched item type.
pub(crate) trait BatchCodec: Send + Sync + 'static {
    type Item: Send + 'static;
    type Proto: Grouped;
    fn prepare(items: Vec<Self::Item>) -> Scoped<Self::Proto>;
}

/// Spans (moved from the SDK, converted at export time).
#[derive(Debug)]
pub(crate) struct Spans;

impl BatchCodec for Spans {
    type Item = SpanData;
    type Proto = ProtoSpan;

    fn prepare(items: Vec<SpanData>) -> Scoped<ProtoSpan> {
        encode::prepare_spans(items)
    }
}

/// Log records (cloned at emit, as the SDK batch processor does).
#[derive(Debug)]
pub(crate) struct Logs;

impl BatchCodec for Logs {
    type Item = (SdkLogRecord, InstrumentationScope);
    type Proto = ProtoLogRecord;

    fn prepare(items: Vec<Self::Item>) -> Scoped<ProtoLogRecord> {
        encode::prepare_logs(items.iter().map(|(r, s)| (r, s)))
    }
}

/// Losses among the items a flush covered (or, conservatively, exported in
/// the same batches).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Losses {
    pub failed: u64,
    pub dropped: u64,
}

impl Losses {
    pub(crate) fn is_empty(&self) -> bool {
        self.failed == 0 && self.dropped == 0
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.failed += other.failed;
        self.dropped += other.dropped;
    }
}

/// Result of a signal's shutdown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SignalShutdown {
    /// Work was cancelled at the deadline (in flight failed, queued dropped).
    pub cancelled: bool,
    /// Losses among items pending when admission closed.
    pub losses: Losses,
}

pub(crate) enum FlushTicket {
    Done(Losses),
    Wait(Arc<Completion<Losses>>),
    Closed,
}

struct Waiter {
    barrier: u64,
    failed: u64,
    dropped: u64,
    done: Arc<Completion<Losses>>,
}

struct State<T> {
    queue: VecDeque<(T, u64)>,
    stats: SignalStats,
    /// Queued plus in-flight estimated bytes.
    bytes: u64,
    /// Sequence number of the next admitted item.
    next_seq: u64,
    /// Every item with a lower sequence number is terminal.
    settled_seq: u64,
    /// Drops after admission (oversized singletons, cancelled at shutdown,
    /// queued when the destination was disabled).
    lost_dropped: u64,
    flush_target: u64,
    waiters: Vec<Waiter>,
    closed: bool,
    draining: bool,
    deadline_cap: Option<Instant>,
    in_flight: Option<(u64, u64)>,
    /// Counters were settled without the worker; it must not settle again.
    abandoned: bool,
    worker_done: bool,
    /// `(failed, lost_dropped)` when admission closed.
    baseline: Option<(u64, u64)>,
    /// An export capped by the shutdown deadline failed when it was reached:
    /// work was cut off by the shutdown (D30.4 timeout), not by its own
    /// export deadline.
    cut_at_deadline: bool,
}

impl<T> State<T> {
    fn losses_since(&self, failed: u64, dropped: u64) -> Losses {
        Losses {
            failed: self.stats.failed - failed,
            dropped: self.lost_dropped - dropped,
        }
    }

    fn resolve_waiters(&mut self, all: bool) {
        let settled = self.settled_seq;
        let (failed, dropped) = (self.stats.failed, self.lost_dropped);
        self.waiters.retain(|w| {
            if all || w.barrier <= settled {
                w.done.complete(Losses {
                    failed: failed - w.failed,
                    dropped: dropped - w.dropped,
                });
                false
            } else {
                true
            }
        });
    }

    fn close(&mut self) {
        self.closed = true;
        if self.baseline.is_none() {
            self.baseline = Some((self.stats.failed, self.lost_dropped));
        }
    }
}

/// One signal's bounded queue and its export worker.
pub(crate) struct Batcher<C: BatchCodec> {
    state: Mutex<State<C::Item>>,
    wake: Notify,
    finished: Notify,
    limits: Limits,
    core: Arc<ExportCore>,
    resource: Mutex<Arc<ResourceInfo>>,
    handle: Handle,
    cancel: CancellationToken,
    worker: Mutex<Option<JoinHandle<()>>>,
    shutdown: Mutex<Option<Arc<Completion<SignalShutdown>>>>,
    estimator: Estimator,
}

impl<C: BatchCodec> Batcher<C> {
    /// Creates the batcher and starts its worker on the transport runtime.
    pub(crate) fn start(
        core: Arc<ExportCore>,
        limits: Limits,
        resource: Arc<ResourceInfo>,
        handle: Handle,
        cancel: CancellationToken,
    ) -> Arc<Self> {
        let estimator = Estimator::new(core.protocol());
        let batcher = Arc::new(Self {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                stats: SignalStats::default(),
                bytes: 0,
                next_seq: 0,
                settled_seq: 0,
                lost_dropped: 0,
                flush_target: 0,
                waiters: Vec::new(),
                closed: false,
                draining: false,
                deadline_cap: None,
                in_flight: None,
                abandoned: false,
                worker_done: false,
                baseline: None,
                cut_at_deadline: false,
            }),
            wake: Notify::new(),
            finished: Notify::new(),
            limits,
            core,
            resource: Mutex::new(resource),
            handle,
            cancel,
            worker: Mutex::new(None),
            shutdown: Mutex::new(None),
            estimator,
        });
        let worker = runtime::spawn(&batcher.handle, batcher.clone().run());
        *lock(&batcher.worker) = Some(worker);
        batcher
    }

    pub(crate) fn stats(&self) -> SignalStats {
        lock(&self.state).stats
    }

    pub(crate) fn is_closed(&self) -> bool {
        lock(&self.state).closed
    }

    fn set_resource(&self, resource: &Resource) {
        *lock(&self.resource) = Arc::new(ResourceInfo::from_sdk(resource));
    }

    /// Admits one item if the item and byte budgets allow it and the
    /// destination is enabled; `make` runs only for admitted items. Never
    /// blocks on export.
    pub(crate) fn offer(&self, bytes: u64, make: impl FnOnce() -> C::Item) {
        let mut st = lock(&self.state);
        st.stats.accepted += 1;
        if st.closed
            || self.core.is_disabled()
            || st.stats.pending >= self.limits.max_queue_size as u64
            || st.bytes.saturating_add(bytes) > self.limits.max_buffered_bytes
        {
            st.stats.dropped += 1;
            return;
        }
        st.stats.pending += 1;
        st.bytes += bytes;
        st.next_seq += 1;
        st.queue.push_back((make(), bytes));
        let len = st.queue.len();
        drop(st);
        if len == 1 || len == self.limits.batch_size {
            self.wake.notify_one();
        }
    }

    /// Registers a flush barrier over the items pending now.
    pub(crate) fn flush_ticket(&self) -> FlushTicket {
        let mut st = lock(&self.state);
        if st.closed {
            return FlushTicket::Closed;
        }
        let barrier = st.next_seq;
        if st.settled_seq >= barrier {
            return FlushTicket::Done(Losses::default());
        }
        let done = Completion::new();
        let waiter = Waiter {
            barrier,
            failed: st.stats.failed,
            dropped: st.lost_dropped,
            done: done.clone(),
        };
        st.waiters.push(waiter);
        st.flush_target = st.flush_target.max(barrier);
        drop(st);
        self.wake.notify_one();
        FlushTicket::Wait(done)
    }

    /// Closes admission; later items are dropped and counted.
    pub(crate) fn close(&self) {
        lock(&self.state).close();
    }

    /// Starts (once) the signal's shutdown: drain until `deadline`, then
    /// cancel in-flight work and drop what is still queued.
    pub(crate) fn shutdown(self: &Arc<Self>, deadline: Instant) -> Arc<Completion<SignalShutdown>> {
        let mut slot = lock(&self.shutdown);
        if let Some(existing) = slot.as_ref() {
            return existing.clone();
        }
        let completion = Completion::new();
        *slot = Some(completion.clone());
        drop(slot);
        if self.cancel.is_cancelled() {
            // The transport is gone; settle synchronously.
            let result = self.settle_remaining();
            completion.complete(result);
            return completion;
        }
        let guard = ShutdownGuard {
            batcher: self.clone(),
            completion: completion.clone(),
            done: false,
        };
        runtime::spawn(&self.handle, guard.drive(deadline));
        completion
    }

    /// Pipeline dropped without shutdown: cancel at once, no flush.
    pub(crate) fn abandon(&self) {
        let mut slot = lock(&self.shutdown);
        lock(&self.state).close();
        self.cancel.cancel();
        let result = self.settle_remaining();
        let completion = slot.get_or_insert_with(Completion::new);
        completion.complete(result);
    }

    async fn wait_finished(&self) {
        loop {
            let mut notified = std::pin::pin!(self.finished.notified());
            notified.as_mut().enable();
            if lock(&self.state).worker_done {
                return;
            }
            notified.await;
        }
    }

    /// Drops what is still queued, fails what is still in flight, resolves
    /// every flush waiter, and reports the losses since admission closed.
    fn settle_remaining(&self) -> SignalShutdown {
        let mut st = lock(&self.state);
        st.close();
        let mut cancelled = st.cut_at_deadline;
        let queued = st.queue.len() as u64;
        if queued > 0 {
            let bytes: u64 = st.queue.iter().map(|(_, b)| *b).sum();
            st.queue.clear();
            st.stats.settle(Outcome::dropped(queued));
            st.lost_dropped += queued;
            st.bytes = st.bytes.saturating_sub(bytes);
            st.settled_seq += queued;
            cancelled = true;
        }
        if let Some((n, bytes)) = st.in_flight.take() {
            st.stats.settle(Outcome::failed(n));
            st.bytes = st.bytes.saturating_sub(bytes);
            st.settled_seq += n;
            st.abandoned = true;
            cancelled = true;
        }
        st.worker_done = true;
        st.resolve_waiters(true);
        let (failed, dropped) = st.baseline.unwrap_or((st.stats.failed, st.lost_dropped));
        let losses = st.losses_since(failed, dropped);
        drop(st);
        self.finished.notify_waiters();
        SignalShutdown { cancelled, losses }
    }

    async fn run(self: Arc<Self>) {
        let mut last_export = Instant::now();
        loop {
            enum Step<T> {
                Export(Vec<T>, u64, u64),
                Wait(Option<Instant>),
                Again,
                Exit,
            }
            let step = {
                let mut st = lock(&self.state);
                if self.cancel.is_cancelled() || st.abandoned {
                    Step::Exit
                } else if st.queue.is_empty() {
                    if st.draining {
                        Step::Exit
                    } else {
                        Step::Wait(None)
                    }
                } else if self.core.is_disabled() {
                    // The response hook disabled the destination: nothing
                    // queued will ever be sent. Drop it all now, so pending
                    // counters and flush waiters settle without a request.
                    let n = st.queue.len() as u64;
                    let bytes: u64 = st.queue.drain(..).map(|(_, b)| b).sum();
                    st.stats.settle(Outcome::dropped(n));
                    st.bytes = st.bytes.saturating_sub(bytes);
                    st.lost_dropped += n;
                    st.settled_seq += n;
                    st.resolve_waiters(false);
                    Step::Again
                } else if st.draining
                    || st.flush_target > st.settled_seq
                    || st.queue.len() >= self.limits.batch_size
                    || last_export.elapsed() >= self.limits.scheduled_delay
                {
                    let n = st.queue.len().min(self.limits.batch_size);
                    let mut items = Vec::with_capacity(n);
                    let mut bytes = 0;
                    for (item, b) in st.queue.drain(..n) {
                        items.push(item);
                        bytes += b;
                    }
                    st.in_flight = Some((n as u64, bytes));
                    Step::Export(items, n as u64, bytes)
                } else {
                    Step::Wait(Some(after(last_export, self.limits.scheduled_delay)))
                }
            };
            match step {
                Step::Exit => break,
                Step::Again => {}
                Step::Wait(until) => {
                    let notified = self.wake.notified();
                    match until {
                        Some(until) => {
                            let until = tokio::time::Instant::from_std(until);
                            let _ = self
                                .cancel
                                .run_until_cancelled(tokio::time::timeout_at(until, notified))
                                .await;
                        }
                        None => {
                            let _ = self.cancel.run_until_cancelled(notified).await;
                        }
                    }
                }
                Step::Export(items, n, bytes) => {
                    let own = after(Instant::now(), self.limits.export_timeout);
                    let cap = lock(&self.state).deadline_cap;
                    let capped = cap.is_some_and(|cap| cap < own);
                    let deadline = cap.map_or(own, |cap| own.min(cap));
                    let resource = lock(&self.resource).clone();
                    let scoped = C::prepare(items);
                    let outcome = self
                        .core
                        .send_split::<C::Proto>(
                            &resource,
                            scoped,
                            tokio::time::Instant::from_std(deadline),
                            &self.cancel,
                        )
                        .await;
                    debug_assert_eq!(outcome.total(), n, "every item reaches one outcome");
                    last_export = Instant::now();
                    let mut st = lock(&self.state);
                    if st.abandoned {
                        break;
                    }
                    st.in_flight = None;
                    if capped && outcome.failed > 0 && Instant::now() >= deadline {
                        st.cut_at_deadline = true;
                    }
                    st.stats.settle(outcome);
                    st.bytes = st.bytes.saturating_sub(bytes);
                    st.lost_dropped += outcome.dropped;
                    st.settled_seq += n;
                    st.resolve_waiters(false);
                }
            }
        }
        let mut st = lock(&self.state);
        st.worker_done = true;
        if st.queue.is_empty() && st.in_flight.is_none() {
            st.resolve_waiters(true);
        }
        drop(st);
        self.finished.notify_waiters();
    }

    fn flush_blocking(&self, timeout: Duration) -> OTelSdkResult {
        match self.flush_ticket() {
            FlushTicket::Done(losses) => losses_result(losses),
            FlushTicket::Closed => Err(OTelSdkError::AlreadyShutdown),
            FlushTicket::Wait(done) => match done.wait_timeout(timeout) {
                Some(losses) => losses_result(losses),
                None => Err(OTelSdkError::Timeout(timeout)),
            },
        }
    }

    fn shutdown_blocking(self: &Arc<Self>, timeout: Duration) -> OTelSdkResult {
        let completion = self.shutdown(after(Instant::now(), timeout));
        match completion.wait_timeout(timeout.saturating_add(SETTLE_GRACE + SETTLE_GRACE)) {
            Some(result) if result.cancelled => Err(OTelSdkError::Timeout(timeout)),
            Some(result) => losses_result(result.losses),
            None => Err(OTelSdkError::Timeout(timeout)),
        }
    }
}

fn losses_result(losses: Losses) -> OTelSdkResult {
    if losses.is_empty() {
        Ok(())
    } else {
        Err(OTelSdkError::InternalFailure(format!(
            "{} failed, {} dropped",
            losses.failed, losses.dropped
        )))
    }
}

impl<C: BatchCodec> fmt::Debug for Batcher<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Batcher")
            .field("signal", &self.core.signal())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// Runs a signal shutdown; if the task is dropped (runtime stopped) before
/// finishing, settles synchronously instead.
struct ShutdownGuard<C: BatchCodec> {
    batcher: Arc<Batcher<C>>,
    completion: Arc<Completion<SignalShutdown>>,
    done: bool,
}

impl<C: BatchCodec> ShutdownGuard<C> {
    async fn drive(mut self, deadline: Instant) {
        let batcher = self.batcher.clone();
        {
            let mut st = lock(&batcher.state);
            st.close();
            st.draining = true;
            st.deadline_cap = Some(deadline);
        }
        batcher.wake.notify_one();
        let drained = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            batcher.wait_finished(),
        )
        .await
        .is_ok();
        let mut had_work = false;
        if !drained {
            {
                let st = lock(&batcher.state);
                had_work = st.in_flight.is_some() || !st.queue.is_empty();
            }
            batcher.cancel.cancel();
        }
        let worker = lock(&batcher.worker).take();
        if let Some(worker) = worker {
            let _ = tokio::time::timeout(SETTLE_GRACE, worker).await;
        }
        let mut result = batcher.settle_remaining();
        result.cancelled |= had_work;
        self.done = true;
        self.completion.complete(result);
    }
}

impl<C: BatchCodec> Drop for ShutdownGuard<C> {
    fn drop(&mut self) {
        if !self.done {
            let result = self.batcher.settle_remaining();
            self.completion.complete(result);
        }
    }
}

/// The pipeline's span processor: sampled spans that end are offered to the
/// bounded batcher; nothing else happens on the application's thread.
#[derive(Debug)]
pub(crate) struct PipelineSpanProcessor {
    pub(crate) batcher: Arc<Batcher<Spans>>,
}

impl SpanProcessor for PipelineSpanProcessor {
    fn on_start(&self, _span: &mut Span, _cx: &Context) {}

    fn on_end(&self, span: SpanData) {
        if !span.span_context.is_sampled() {
            return;
        }
        let bytes = self.batcher.estimator.span(&span);
        self.batcher.offer(bytes, || span);
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.batcher.flush_blocking(defaults::FLUSH_TIMEOUT)
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.batcher.shutdown_blocking(timeout)
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.batcher.set_resource(resource);
    }
}

/// The pipeline's log processor (replaces the SDK batch log processor).
#[derive(Debug)]
pub(crate) struct PipelineLogProcessor {
    pub(crate) batcher: Arc<Batcher<Logs>>,
}

impl LogProcessor for PipelineLogProcessor {
    fn emit(&self, record: &mut SdkLogRecord, scope: &InstrumentationScope) {
        let bytes = self.batcher.estimator.log(record, scope);
        self.batcher
            .offer(bytes, || (record.clone(), scope.clone()));
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.batcher.flush_blocking(defaults::FLUSH_TIMEOUT)
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.batcher.shutdown_blocking(timeout)
    }

    fn event_enabled(&self, _level: Severity, _target: &str, _name: Option<&str>) -> bool {
        !self.batcher.is_closed()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.batcher.set_resource(resource);
    }
}

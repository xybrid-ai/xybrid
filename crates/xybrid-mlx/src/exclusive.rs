//! Process-wide serialization of MLX work.
//!
//! xybrid evaluates graphs on one stream per device that every thread in the
//! process shares (see [`crate::MlxStream`]), and MLX does not synchronize
//! those streams. Evaluating from two threads at once corrupts their state:
//! two models generating at the same time end in a segfault inside MLX,
//! whichever architectures they are. One model used from many threads in turn
//! is fine.
//!
//! [`exclusive`] is the one lock every MLX-evaluating operation takes: a model
//! load (weights are materialized into MLX arrays), a generation, an embedding
//! run. Concurrent operations queue; they never interleave.
//!
//! The evaluating calls of this crate ([`MlxArray::eval`](crate::MlxArray::eval),
//! [`async_eval`](crate::async_eval) and the readbacks built on them) take the
//! lock themselves, so no use of the safe API can evaluate from two threads at
//! once. An operation that spans several evaluations, with lazy graphs in
//! flight between them, still takes it once around the whole run.
//!
//! # Nothing outlives an operation
//!
//! When the outermost guard on a thread drops, it first waits for whatever
//! the operation left in flight (the evaluation a generation scheduled for a
//! token it never needed, say). Two things follow. The lock being free means
//! MLX is idle, not merely unowned. And an evaluation that failed after the
//! operation stopped waiting for it (the GPU rejecting a command buffer)
//! cannot surface in the next operation: MLX keeps such an error on the
//! stream and raises it from the next call that touches it, so it is taken
//! here and logged. An evaluation the operation did wait for reports its
//! failure to the operation itself, as an error.
//!
//! [`async_eval`](crate::async_eval) called outside an operation therefore
//! completes before it returns; it runs in the background only under a guard
//! held around the whole run.
//!
//! A generation holds the lock for its whole run, streaming callbacks
//! included: it has the next token's graph in flight while a callback runs,
//! so the lock cannot be dropped between tokens. A callback may start MLX
//! work on its own thread (the lock is reentrant there). It must not block on
//! MLX work that another thread has yet to start, which would wait for this
//! lock forever.
//!
//! # Giving memory back
//!
//! MLX frees an array's memory into a pool it keeps for reuse, so a dropped
//! model stays allocated to the process until the pool is cleared.
//! [`release_pooled_memory`] clears it without ever waiting for the lock: at
//! once when MLX is idle, otherwise as the operation in flight lets go.
//! [`ReleasePooledMemory`] does that on drop, for the types that own a model's
//! arrays.
//!
//! A static is the right owner here even though statics are otherwise avoided
//! for correctness-relevant state: the state being protected is MLX's own
//! process-global runtime, and `xybrid-mlx-sys`'s `links` key guarantees one copy
//! of that runtime per process.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};

static MLX_WORK: Mutex<()> = Mutex::new(());

/// Set by [`release_pooled_memory`]; cleared by whoever clears the pool.
static POOL_RELEASE_REQUESTED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// How many [`MlxExclusive`] guards this thread holds. Only the outermost
    /// owns the lock, so a streaming callback that starts another MLX
    /// operation on its own thread nests instead of deadlocking.
    static HELD: Cell<usize> = const { Cell::new(0) };
}

/// Exclusive access to the MLX runtime for the current thread, released on
/// drop. See the [module docs](self).
#[must_use = "MLX work is serialized only while the guard is alive"]
pub struct MlxExclusive {
    /// Held by the outermost guard on a thread only; released when that guard
    /// drops, after its `Drop` has counted it out.
    _lock: Option<MutexGuard<'static, ()>>,
    /// The guard counts per thread, so it must stay on the thread that took it.
    _not_send: PhantomData<*const ()>,
}

/// Take the process-wide MLX lock, waiting for any MLX operation in flight on
/// another thread. Reentrant on one thread.
///
/// A panic while the lock is held (a streaming callback that panics, say) does
/// not poison MLX for the rest of the process: per-run state is rebuilt at the
/// start of every operation, so the next caller proceeds.
pub fn exclusive() -> MlxExclusive {
    let outermost = HELD.with(|held| {
        let depth = held.get();
        held.set(depth + 1);
        depth == 0
    });
    MlxExclusive {
        _lock: outermost.then(|| MLX_WORK.lock().unwrap_or_else(PoisonError::into_inner)),
        _not_send: PhantomData,
    }
}

impl Drop for MlxExclusive {
    fn drop(&mut self) {
        HELD.with(|held| held.set(held.get() - 1));
        if let Some(lock) = self._lock.take() {
            // Outermost guard: the operation is over. Leave nothing in
            // flight behind it, then clear the pool if a model was dropped
            // while it ran.
            settle();
            clear_pool_if_requested();
            drop(lock);
            // A request can arrive after the check above and find the lock
            // still held; it left the job to this thread.
            release_pool_if_idle();
        }
    }
}

/// Return the memory MLX pools for reuse to the system, as soon as no MLX
/// operation is running.
///
/// Never waits for the lock. When MLX is idle the pool is cleared before this
/// returns. When an operation is in flight, on this thread or another, the
/// pool is cleared as that operation releases the lock, after the work it
/// left in flight has let go of the arrays it read, so dropping a model never
/// blocks behind someone else's generation.
pub fn release_pooled_memory() {
    POOL_RELEASE_REQUESTED.store(true, Ordering::SeqCst);
    release_pool_if_idle();
}

/// Clear the pool if that was requested and no MLX operation holds the lock.
fn release_pool_if_idle() {
    if !POOL_RELEASE_REQUESTED.load(Ordering::SeqCst) {
        return;
    }
    // Inside this thread's own operation: its outermost guard clears on the
    // way out. During thread teardown the count is gone; nothing is held.
    if HELD.try_with(Cell::get).unwrap_or(0) > 0 {
        return;
    }
    let _lock = match MLX_WORK.try_lock() {
        Ok(lock) => lock,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        // The operation in flight clears the pool when it lets go.
        Err(TryLockError::WouldBlock) => return,
    };
    clear_pool_if_requested();
}

/// Wait for the work an operation left in flight, and take the error it left
/// on the streams, if any. The caller holds [`MLX_WORK`]. See the
/// [module docs](self).
fn settle() {
    // SAFETY: the caller holds the lock, as synchronizing requires.
    if let Err(error) = unsafe { crate::ffi::synchronize_shared_streams() } {
        tracing::warn!(%error, "mlx.operation.left_failed_work");
    }
}

/// Clear the pool if that was requested. The caller holds [`MLX_WORK`], and
/// MLX is idle: every operation settles before it releases the lock, so the
/// arrays a dropped model's last evaluation read have reached the pool.
fn clear_pool_if_requested() {
    if POOL_RELEASE_REQUESTED.swap(false, Ordering::SeqCst) {
        // Nothing is pooled before MLX's first use (a load that failed
        // validation, say), and clearing would start the Metal runtime.
        if !crate::resources::metallib_ready() {
            return;
        }
        // SAFETY: the caller holds the lock; live arrays are unaffected.
        if let Err(error) = unsafe { crate::ffi::clear_cache() } {
            tracing::warn!(%error, "mlx.release_pooled_memory.failed");
        }
    }
}

/// Returns MLX's pooled memory to the system when dropped; see
/// [`release_pooled_memory`].
///
/// Make it the **last** field of a type that owns a model's MLX arrays.
/// Fields drop in declaration order, so the arrays are freed into the pool
/// first and the pool is cleared after.
#[derive(Debug, Default)]
pub struct ReleasePooledMemory {
    _private: (),
}

impl Drop for ReleasePooledMemory {
    fn drop(&mut self) {
        release_pooled_memory();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn work_on_different_threads_never_overlaps() {
        let inside = Arc::new(AtomicUsize::new(0));
        let overlaps = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let inside = inside.clone();
                let overlaps = overlaps.clone();
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let _mlx = exclusive();
                        if inside.fetch_add(1, Ordering::SeqCst) != 0 {
                            overlaps.fetch_add(1, Ordering::SeqCst);
                        }
                        std::thread::yield_now();
                        inside.fetch_sub(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("worker");
        }
        assert_eq!(overlaps.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn nesting_on_one_thread_does_not_deadlock_and_releases_at_the_outermost_guard() {
        let outer = exclusive();
        {
            let _inner = exclusive();
            let _innermost = exclusive();
        }
        // Still held: another thread cannot enter until `outer` drops.
        let entered = Arc::new(AtomicUsize::new(0));
        let waiter = {
            let entered = entered.clone();
            std::thread::spawn(move || {
                let _mlx = exclusive();
                entered.store(1, Ordering::SeqCst);
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(entered.load(Ordering::SeqCst), 0);
        drop(outer);
        waiter.join().expect("waiter");
        assert_eq!(entered.load(Ordering::SeqCst), 1);
    }

    /// An array whose evaluation fails on MLX's own thread, not the
    /// caller's: the inverse of a singular matrix, which the CPU stream's
    /// worker computes. It stands in for work the GPU rejects, which fails on
    /// Metal's thread and reaches the caller through the same mechanism.
    fn fails_on_the_stream_thread() -> crate::MlxArray {
        use mlx_c_sys::bindings as sys;

        let stream = crate::MlxStream::default_cpu().expect("cpu stream");
        let singular = crate::MlxArray::from_slice_f32(&[0.0; 4], &[2, 2]).expect("array");
        // SAFETY: both handles are live for the call, and `inverse` is a
        // fresh handle that mlx-c fills in.
        unsafe {
            let mut inverse = sys::mlx_array_new();
            let rc = sys::mlx_linalg_inv(&mut inverse, singular.as_raw(), stream.as_raw());
            assert_eq!(
                rc, 0,
                "building the graph does not fail; evaluating it does"
            );
            crate::MlxArray::from_raw(inverse)
        }
    }

    fn evaluates_on_the_cpu_stream() -> Vec<f32> {
        let stream = crate::MlxStream::default_cpu().expect("cpu stream");
        let ones = crate::MlxArray::from_slice_f32(&[1.0, 2.0], &[2]).expect("array");
        crate::ops::add(&ones, &ones, Some(&stream))
            .expect("add")
            .to_vec_f32()
            .expect("an evaluation after a failed one")
    }

    /// Work that fails on MLX's thread comes back as an error from the call
    /// that waited for it (before MLX v0.32 it ended the process), and the
    /// failure is spent: the next evaluation runs.
    #[test]
    fn work_that_fails_on_mlx_s_thread_is_an_error_and_is_then_spent() {
        let error = fails_on_the_stream_thread()
            .eval()
            .expect_err("a singular matrix has no inverse");
        assert!(error.to_string().contains("Inverse"), "{error}");

        assert_eq!(evaluates_on_the_cpu_stream(), [2.0, 4.0]);
    }

    /// An operation schedules work and ends without waiting for it, as a
    /// generation does with the token it never needed. If that work fails,
    /// MLX leaves the error on the stream for whoever touches it next. The
    /// guard takes it as the operation ends, so the next operation, on this
    /// thread or another, is not the one to fail.
    #[test]
    fn work_an_operation_did_not_wait_for_cannot_fail_the_next_operation() {
        {
            let _operation = exclusive();
            let failing = fails_on_the_stream_thread();
            crate::async_eval(&[&failing]).expect("scheduling does not fail");
        }

        assert_eq!(evaluates_on_the_cpu_stream(), [2.0, 4.0]);
        let elsewhere = std::thread::spawn(evaluates_on_the_cpu_stream)
            .join()
            .expect("an evaluation on another thread");
        assert_eq!(elsewhere, [2.0, 4.0]);
    }

    #[test]
    fn a_panic_under_the_lock_does_not_lock_everyone_else_out() {
        let panicked = std::thread::spawn(|| {
            let _mlx = exclusive();
            panic!("a streaming callback panicked");
        })
        .join();
        assert!(panicked.is_err());

        let _mlx = exclusive();
    }
}

//! Process-wide serialization of MLX work.
//!
//! MLX evaluates graphs on per-device default streams that every thread in
//! the process shares. Building and evaluating graphs from two threads at
//! once corrupts that shared state: two models generating at the same time
//! end in a segfault inside MLX, whichever architectures they are. One model
//! used from many threads in turn is fine.
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
            // Outermost guard: the operation is over. Clear the pool if a
            // model was dropped while it ran.
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
/// returns, after any evaluation still completing in the background has
/// let go of the arrays it read. When an operation is in flight, on this
/// thread or another, the pool is cleared as that operation releases the
/// lock, so dropping a model never blocks behind someone else's generation.
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

/// Clear the pool if that was requested. The caller holds [`MLX_WORK`].
fn clear_pool_if_requested() {
    if POOL_RELEASE_REQUESTED.swap(false, Ordering::SeqCst) {
        // Nothing is pooled before MLX's first use (a load that failed
        // validation, say), and clearing would start the Metal runtime.
        if !crate::resources::metallib_ready() {
            return;
        }
        // A model dropped right after its last evaluation is still held by
        // that evaluation until it completes; wait, so its memory reaches
        // the pool before the pool is cleared, not after.
        // SAFETY: the caller holds the lock, as both calls require; live
        // arrays are unaffected.
        let released = unsafe {
            crate::ffi::synchronize_default_streams().and_then(|()| crate::ffi::clear_cache())
        };
        if let Err(error) = released {
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

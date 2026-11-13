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
//! A static is the right owner here even though statics are otherwise avoided
//! for correctness-relevant state: the state being protected is MLX's own
//! process-global runtime, and `mlx-c-sys`'s `links` key guarantees one copy
//! of that runtime per process.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError};

static MLX_WORK: Mutex<()> = Mutex::new(());

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

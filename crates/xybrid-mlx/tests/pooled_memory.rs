//! MLX frees a dropped array's memory into a pool it keeps for reuse.
//! `release_pooled_memory` hands that pool back to the system without ever
//! waiting behind an MLX operation in flight.
//!
//! These tests read process-wide allocator counters, so they live in their
//! own test binary and run one at a time.

#![cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Mutex, MutexGuard, PoisonError};

use xybrid_mlx::{
    active_memory, async_eval, cache_memory, clear_cache, exclusive, ops, release_pooled_memory,
    MlxArray, MlxStream, ReleasePooledMemory,
};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn pooled() -> usize {
    cache_memory().expect("cache memory")
}

/// Leave one array's worth of memory in the pool, the way a dropped model
/// leaves its weights. Returns what the pool held before.
fn leave_memory_in_the_pool() -> usize {
    clear_cache().expect("clear cache");
    let before = pooled();
    {
        let array = MlxArray::from_slice_f32(&vec![1.0f32; 1 << 20], &[1 << 20]).expect("array");
        array.eval().expect("eval");
    }
    assert!(
        pooled() > before,
        "a dropped array should leave memory in the pool"
    );
    before
}

#[test]
fn pooled_memory_is_released_at_once_when_mlx_is_idle() {
    let _serial = one_at_a_time();
    let before = leave_memory_in_the_pool();

    release_pooled_memory();

    assert!(pooled() <= before);
}

#[test]
fn a_release_during_an_operation_neither_waits_nor_clears_under_it() {
    let _serial = one_at_a_time();
    let before = leave_memory_in_the_pool();

    let operation = exclusive();
    // Another thread drops a model while this one generates.
    std::thread::spawn(release_pooled_memory)
        .join()
        .expect("a release never waits for the lock");
    assert!(
        pooled() > before,
        "the pool is left alone while an operation runs"
    );

    // The operation clears it on the way out.
    drop(operation);
    assert!(pooled() <= before);
}

#[test]
fn a_release_inside_an_operation_waits_for_its_outermost_guard() {
    let _serial = one_at_a_time();
    let before = leave_memory_in_the_pool();

    let operation = exclusive();
    {
        // A model dropped by the thread that is generating, in a nested call.
        let _nested = exclusive();
        release_pooled_memory();
    }
    assert!(
        pooled() > before,
        "the pool is left alone while an operation runs"
    );

    drop(operation);
    assert!(pooled() <= before);
}

/// A model dropped while its last evaluation is still in flight: the
/// evaluation keeps the arrays it reads alive until it completes, on MLX's
/// own threads. The release waits for it, so the memory is not freed into
/// the pool just after the pool was cleared.
fn assert_release_waits_for_work_in_flight(stream: &MlxStream) {
    clear_cache().expect("clear cache");
    let (active_before, pooled_before) = (active_memory().expect("active memory"), pooled());

    for round in 0..20 {
        {
            let weights =
                MlxArray::from_slice_f32(&vec![1.0f32; 1 << 22], &[1 << 22]).expect("array");
            let mut output = ops::add(&weights, &weights, Some(stream)).expect("add");
            for _ in 0..8 {
                output = ops::mul(&output, &weights, Some(stream)).expect("mul");
            }
            async_eval(&[&output]).expect("async eval");
            // Dropped with the evaluation still running.
        }
        release_pooled_memory();

        assert!(
            active_memory().expect("active memory") <= active_before,
            "round {round}: arrays are still held after the release"
        );
        assert!(
            pooled() <= pooled_before,
            "round {round}: memory reached the pool after it was cleared"
        );
    }
}

#[test]
fn a_release_waits_for_gpu_work_that_still_holds_the_arrays() {
    let _serial = one_at_a_time();
    assert_release_waits_for_work_in_flight(&MlxStream::default_gpu().expect("gpu stream"));
}

#[test]
fn a_release_waits_for_cpu_work_that_still_holds_the_arrays() {
    let _serial = one_at_a_time();
    assert_release_waits_for_work_in_flight(&MlxStream::default_cpu().expect("cpu stream"));
}

/// A model whose load fails validation is dropped before MLX was ever used.
/// Its release must not start MLX: run alone, this test's process never
/// touches Metal. (Run with the others, MLX is up and the call is a no-op
/// clear; both are fine.)
#[test]
fn a_release_before_any_mlx_use_does_not_start_mlx() {
    let _serial = one_at_a_time();

    drop(ReleasePooledMemory::default());
    let operation = exclusive();
    release_pooled_memory();
    drop(operation);
}

#[test]
fn the_drop_guard_releases_pooled_memory() {
    let _serial = one_at_a_time();
    let before = leave_memory_in_the_pool();

    drop(ReleasePooledMemory::default());

    assert!(pooled() <= before);
}

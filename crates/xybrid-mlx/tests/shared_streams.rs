//! xybrid loads a model on one thread and runs it on others. MLX's own
//! streams are tied to the thread that created them; `MlxStream` hands out
//! process-wide ones instead, so an array built on one thread evaluates on
//! any other.

#![cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]

use xybrid_mlx::{ops, Device, MlxArray, MlxStream};

/// A graph built on one thread and evaluated on another, on each device.
/// With a stream per thread, MLX refuses the evaluation: "There is no
/// Stream(gpu, N) in current thread."
#[test]
fn a_graph_built_on_one_thread_evaluates_on_another() {
    for device in [Device::Gpu, Device::Cpu] {
        let lazy = std::thread::spawn(move || {
            let stream = MlxStream::new(device).expect("stream");
            let ones = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0], &[3]).expect("array");
            // Not evaluated here: only the graph crosses the thread boundary.
            ops::add(&ones, &ones, Some(&stream)).expect("add")
        })
        .join()
        .expect("builder thread");

        let evaluated = std::thread::spawn(move || lazy.to_vec_f32().expect("evaluate"))
            .join()
            .expect("evaluating thread");
        assert_eq!(evaluated, [2.0, 4.0, 6.0], "{device:?}");
    }
}

/// A stream handle made on one thread works on another: weights are
/// materialized with the loading thread's handle, and a later run on another
/// thread builds on the same stream.
#[test]
fn a_stream_handle_moves_between_threads() {
    let stream = MlxStream::default_gpu().expect("gpu stream");
    let sum = std::thread::spawn(move || {
        let ones = MlxArray::from_slice_f32(&[1.0, 1.0], &[2]).expect("array");
        ops::add(&ones, &ones, Some(&stream))
            .expect("add")
            .to_vec_f32()
            .expect("evaluate")
    })
    .join()
    .expect("worker thread");
    assert_eq!(sum, [2.0, 2.0]);
}

/// Many threads evaluating at once share the two streams, one at a time; none
/// of them needs, or leaves behind, a stream of its own.
#[test]
fn many_threads_evaluate_on_the_shared_streams() {
    let workers: Vec<_> = (0..16)
        .map(|worker| {
            std::thread::spawn(move || {
                let stream = MlxStream::default_gpu().expect("gpu stream");
                let value = worker as f32;
                let array = MlxArray::from_slice_f32(&[value; 64], &[64]).expect("array");
                let doubled = ops::add(&array, &array, Some(&stream)).expect("add");
                doubled.to_vec_f32().expect("evaluate")
            })
        })
        .collect();
    for (worker, handle) in workers.into_iter().enumerate() {
        let doubled = handle.join().expect("worker thread");
        assert!(doubled.iter().all(|v| *v == 2.0 * worker as f32));
    }
}

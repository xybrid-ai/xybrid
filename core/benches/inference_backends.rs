//! Benchmark suite for comparing ONNX Runtime execution providers.
//!
//! This benchmark compares inference performance across different backends:
//! - CPU (default, always available)
//! - CoreML with Neural Engine (macOS/iOS, requires `coreml-ep` feature)
//! - CoreML with GPU (macOS/iOS, requires `coreml-ep` feature)
//!
//! # Running Benchmarks
//!
//! ```bash
//! # CPU only
//! cargo bench -p xybrid-core
//!
//! # With CoreML (macOS/iOS)
//! cargo bench -p xybrid-core --features coreml-ep
//!
//! # View HTML report
//! open target/criterion/report/index.html
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use ndarray::{ArrayD, IxDyn};
use xybrid_core::runtime_adapter::onnx::{ExecutionProviderKind, ONNXSession};

/// Model configuration for benchmarking
struct ModelConfig {
    /// Path to the ONNX model file
    path: &'static str,
    /// Human-readable name for reports
    name: &'static str,
    /// Input tensor shape
    input_shape: &'static [usize],
    /// Input tensor name (from model metadata)
    input_name: &'static str,
}

/// Available models for benchmarking
const BENCHMARK_MODELS: &[ModelConfig] = &[
    ModelConfig {
        path: "test_models/mnist-12.onnx",
        name: "mnist",
        input_shape: &[1, 1, 28, 28],
        input_name: "Input3", // MNIST-12 uses this input name
    },
    // Add more models as needed:
    // ModelConfig {
    //     path: "test_models/wav2vec2-base-960h/model.onnx",
    //     name: "wav2vec2",
    //     input_shape: &[1, 16000],
    //     input_name: "input",
    // },
];

/// Find model path from multiple possible locations
fn find_model_path(model_config: &ModelConfig) -> Option<PathBuf> {
    let possible_paths = vec![
        PathBuf::from(model_config.path),
        PathBuf::from(format!("../{}", model_config.path)),
        PathBuf::from(format!("../../{}", model_config.path)),
        PathBuf::from(format!("repos/xybrid/core/{}", model_config.path)),
    ];

    possible_paths.into_iter().find(|p| p.exists())
}

/// Create test input tensor for a model
fn create_test_input(config: &ModelConfig) -> HashMap<String, ArrayD<f32>> {
    let size: usize = config.input_shape.iter().product();
    let data = vec![0.5f32; size]; // Use 0.5 as a neutral value

    let tensor = ArrayD::from_shape_vec(IxDyn(config.input_shape), data)
        .expect("Failed to create input tensor");

    let mut inputs = HashMap::new();
    inputs.insert(config.input_name.to_string(), tensor);
    inputs
}

/// Benchmark CPU execution provider
fn benchmark_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/cpu");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    for model_config in BENCHMARK_MODELS {
        let model_path = match find_model_path(model_config) {
            Some(p) => p,
            None => {
                eprintln!(
                    "Skipping {} benchmark: model not found at {}",
                    model_config.name, model_config.path
                );
                continue;
            }
        };

        // Create session with CPU provider
        let session = match ONNXSession::with_provider(
            model_path.to_str().unwrap(),
            ExecutionProviderKind::Cpu,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "Skipping {} benchmark: failed to create session: {}",
                    model_config.name, e
                );
                continue;
            }
        };

        let input = create_test_input(model_config);

        group.bench_with_input(
            BenchmarkId::new(model_config.name, "cpu"),
            &input,
            |b, input| {
                b.iter(|| {
                    let result = session.run(black_box(input.clone()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CoreML execution provider with Neural Engine
#[cfg(feature = "coreml-ep")]
fn benchmark_coreml_ane(c: &mut Criterion) {
    use xybrid_core::runtime_adapter::onnx::{CoreMLComputeUnits, CoreMLConfig};

    let mut group = c.benchmark_group("inference/coreml-ane");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    for model_config in BENCHMARK_MODELS {
        let model_path = match find_model_path(model_config) {
            Some(p) => p,
            None => {
                eprintln!(
                    "Skipping {} CoreML benchmark: model not found",
                    model_config.name
                );
                continue;
            }
        };

        // Create session with CoreML Neural Engine provider
        let session = match ONNXSession::with_provider(
            model_path.to_str().unwrap(),
            ExecutionProviderKind::CoreML(CoreMLConfig {
                compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
                use_subgraphs: true,
                require_static_shapes: false,
            }),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "Skipping {} CoreML benchmark: failed to create session: {}",
                    model_config.name, e
                );
                continue;
            }
        };

        let input = create_test_input(model_config);

        group.bench_with_input(
            BenchmarkId::new(model_config.name, "coreml-ane"),
            &input,
            |b, input| {
                b.iter(|| {
                    let result = session.run(black_box(input.clone()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CoreML execution provider with GPU
#[cfg(feature = "coreml-ep")]
fn benchmark_coreml_gpu(c: &mut Criterion) {
    use xybrid_core::runtime_adapter::onnx::{CoreMLComputeUnits, CoreMLConfig};

    let mut group = c.benchmark_group("inference/coreml-gpu");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    for model_config in BENCHMARK_MODELS {
        let model_path = match find_model_path(model_config) {
            Some(p) => p,
            None => {
                continue;
            }
        };

        // Create session with CoreML GPU provider
        let session = match ONNXSession::with_provider(
            model_path.to_str().unwrap(),
            ExecutionProviderKind::CoreML(CoreMLConfig {
                compute_units: CoreMLComputeUnits::CpuAndGpu,
                use_subgraphs: true,
                require_static_shapes: false,
            }),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "Skipping {} CoreML GPU benchmark: {}",
                    model_config.name, e
                );
                continue;
            }
        };

        let input = create_test_input(model_config);

        group.bench_with_input(
            BenchmarkId::new(model_config.name, "coreml-gpu"),
            &input,
            |b, input| {
                b.iter(|| {
                    let result = session.run(black_box(input.clone()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// Conditional benchmark groups based on features
#[cfg(not(feature = "coreml-ep"))]
criterion_group!(benches, benchmark_cpu);

#[cfg(feature = "coreml-ep")]
criterion_group!(benches, benchmark_cpu, benchmark_coreml_ane, benchmark_coreml_gpu);

criterion_main!(benches);

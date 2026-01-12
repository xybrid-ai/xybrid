//! Benchmark suite for comparing ONNX Runtime execution providers.
//!
//! This benchmark compares inference performance across different backends:
//! - CPU (default, always available)
//! - CoreML with Neural Engine (macOS/iOS, requires `coreml-ep` feature)
//! - CoreML with GPU (macOS/iOS, requires `coreml-ep` feature)
//!
//! # Setup
//!
//! Models are stored in the integration-tests fixtures directory. Download them first:
//!
//! ```bash
//! cd repos/xybrid/integration-tests
//! ./download.sh mnist
//! ./download.sh kokoro-82m
//! ```
//!
//! # Running Benchmarks
//!
//! ```bash
//! # CPU only (run from workspace root)
//! cargo bench -p xybrid-core
//!
//! # With CoreML (macOS/iOS)
//! cargo bench -p xybrid-core --features coreml-ep
//!
//! # View HTML report
//! open target/criterion/report/index.html
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ort::value::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use xybrid_core::runtime_adapter::onnx::{ExecutionProviderKind, ONNXSession};

// ============================================================================
// Model Configurations
// ============================================================================

/// Input type for different model architectures
#[derive(Clone)]
enum ModelInputs {
    /// Single f32 tensor input (e.g., MNIST, image classification)
    SingleFloat {
        name: &'static str,
        shape: &'static [usize],
    },
    /// TTS model with tokens (i64), style (f32), and speed (f32)
    Tts {
        tokens_name: &'static str,
        style_name: &'static str,
        speed_name: &'static str,
        /// Number of phoneme tokens to use (simulates sentence length)
        num_tokens: usize,
        /// Voice embedding dimension (typically 256)
        style_dim: usize,
    },
}

/// Model configuration for benchmarking
struct ModelConfig {
    /// Model directory name in integration-tests/fixtures/models/
    model_dir: &'static str,
    /// Model filename within the directory
    model_file: &'static str,
    /// Human-readable name for reports
    name: &'static str,
    /// Input configuration
    inputs: ModelInputs,
}

/// Available models for benchmarking
///
/// These correspond to models in repos/xybrid/integration-tests/fixtures/models/
/// Run `./download.sh <model_name>` to fetch them before benchmarking.
const BENCHMARK_MODELS: &[ModelConfig] = &[
    ModelConfig {
        model_dir: "mnist",
        model_file: "model.onnx",
        name: "mnist",
        inputs: ModelInputs::SingleFloat {
            name: "Input3",
            shape: &[1, 1, 28, 28],
        },
    },
    // MobileNetV2 - Vision model that should benefit from ANE
    // Input: 224x224 RGB image, normalized
    ModelConfig {
        model_dir: "mobilenet",
        model_file: "mobilenetv2-12.onnx",
        name: "mobilenet-v2",
        inputs: ModelInputs::SingleFloat {
            name: "input",
            shape: &[1, 3, 224, 224],
        },
    },
    ModelConfig {
        model_dir: "kokoro-82m",
        model_file: "kokoro-v1.0.fp16.onnx",
        name: "kokoro-82m",
        inputs: ModelInputs::Tts {
            tokens_name: "tokens",
            style_name: "style",
            speed_name: "speed",
            num_tokens: 50, // ~10 word sentence
            style_dim: 256,
        },
    },
];

// ============================================================================
// Helper Functions
// ============================================================================

/// Find model path from multiple possible locations
///
/// Searches for models in the integration-tests fixtures directory,
/// handling different working directory scenarios (workspace root vs core package).
fn find_model_path(model_config: &ModelConfig) -> Option<PathBuf> {
    let relative_path = format!(
        "integration-tests/fixtures/models/{}/{}",
        model_config.model_dir, model_config.model_file
    );

    let possible_paths = vec![
        // From workspace root (cargo bench -p xybrid-core)
        PathBuf::from(format!("repos/xybrid/{}", relative_path)),
        // From xybrid repo root
        PathBuf::from(&relative_path),
        // From core package directory
        PathBuf::from(format!("../{}", relative_path)),
        // From repos/xybrid/core
        PathBuf::from(format!("../../{}", relative_path)),
    ];

    possible_paths.into_iter().find(|p| p.exists())
}

/// Create test input tensors for a model (f32 version for simple models)
fn create_float_inputs(config: &ModelConfig) -> Option<HashMap<String, ArrayD<f32>>> {
    match &config.inputs {
        ModelInputs::SingleFloat { name, shape } => {
            let size: usize = shape.iter().product();
            let data = vec![0.5f32; size];
            let tensor =
                ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create input tensor");

            let mut inputs = HashMap::new();
            inputs.insert(name.to_string(), tensor);
            Some(inputs)
        }
        ModelInputs::Tts { .. } => None, // TTS needs special handling
    }
}

/// TTS input data (raw arrays that can be cloned)
#[derive(Clone)]
struct TtsInputData {
    tokens_name: String,
    style_name: String,
    speed_name: String,
    tokens: Array2<i64>,
    style: Array2<f32>,
    speed: Array1<f32>,
}

/// Create TTS input data (raw arrays that can be converted to Values)
fn create_tts_input_data(config: &ModelConfig) -> Option<TtsInputData> {
    match &config.inputs {
        ModelInputs::Tts {
            tokens_name,
            style_name,
            speed_name,
            num_tokens,
            style_dim,
        } => {
            // Tokens: [1, num_tokens] of i64 phoneme IDs
            // Use typical phoneme IDs (1-100 range)
            let token_data: Vec<i64> = (0..*num_tokens).map(|i| ((i % 80) + 1) as i64).collect();
            let tokens = Array2::from_shape_vec((1, *num_tokens), token_data)
                .expect("Failed to create tokens array");

            // Style: [1, style_dim] of f32 voice embedding
            // Use normalized values (typical voice embeddings are in -1 to 1 range)
            let style_data: Vec<f32> = (0..*style_dim)
                .map(|i| ((i as f32 / *style_dim as f32) * 2.0 - 1.0) * 0.5)
                .collect();
            let style = Array2::from_shape_vec((1, *style_dim), style_data)
                .expect("Failed to create style array");

            // Speed: [1] of f32 (1.0 = normal speed)
            let speed = Array1::from_vec(vec![1.0f32]);

            Some(TtsInputData {
                tokens_name: tokens_name.to_string(),
                style_name: style_name.to_string(),
                speed_name: speed_name.to_string(),
                tokens,
                style,
                speed,
            })
        }
        ModelInputs::SingleFloat { .. } => None,
    }
}

/// Convert TTS input data to ort Values (must be done fresh each run)
fn tts_data_to_values(data: &TtsInputData) -> HashMap<String, Value> {
    let mut inputs = HashMap::new();

    let tokens_value =
        Value::from_array(data.tokens.clone()).expect("Failed to convert tokens to Value");
    inputs.insert(data.tokens_name.clone(), tokens_value.into_dyn());

    let style_value =
        Value::from_array(data.style.clone()).expect("Failed to convert style to Value");
    inputs.insert(data.style_name.clone(), style_value.into_dyn());

    let speed_value =
        Value::from_array(data.speed.clone()).expect("Failed to convert speed to Value");
    inputs.insert(data.speed_name.clone(), speed_value.into_dyn());

    inputs
}

/// Run session with appropriate input type
fn run_session_with_inputs(
    session: &ONNXSession,
    config: &ModelConfig,
    float_inputs: &Option<HashMap<String, ArrayD<f32>>>,
    tts_data: &Option<TtsInputData>,
) -> Result<(), Box<dyn std::error::Error>> {
    match &config.inputs {
        ModelInputs::SingleFloat { .. } => {
            if let Some(inputs) = float_inputs {
                session.run(inputs.clone())?;
            }
        }
        ModelInputs::Tts { .. } => {
            if let Some(data) = tts_data {
                let inputs = tts_data_to_values(data);
                session.run_with_values(inputs)?;
            }
        }
    }
    Ok(())
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark CPU execution provider
fn benchmark_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/cpu");
    group.sample_size(10); // Reduced for TTS (slower)
    group.measurement_time(Duration::from_secs(30));

    for model_config in BENCHMARK_MODELS {
        let model_path = match find_model_path(model_config) {
            Some(p) => p,
            None => {
                eprintln!(
                    "Skipping {} benchmark: model not found. Run: cd repos/xybrid/integration-tests && ./download.sh {}",
                    model_config.name, model_config.model_dir
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

        // Prepare inputs based on model type
        let float_inputs = create_float_inputs(model_config);
        let tts_data = create_tts_input_data(model_config);

        group.bench_function(BenchmarkId::new(model_config.name, "cpu"), |b| {
            b.iter(|| {
                let _ = run_session_with_inputs(
                    &session,
                    model_config,
                    &float_inputs,
                    black_box(&tts_data),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark CoreML execution provider with Neural Engine
#[cfg(feature = "coreml-ep")]
fn benchmark_coreml_ane(c: &mut Criterion) {
    use xybrid_core::runtime_adapter::onnx::{CoreMLComputeUnits, CoreMLConfig};

    let mut group = c.benchmark_group("inference/coreml-ane");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

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

        let float_inputs = create_float_inputs(model_config);
        let tts_data = create_tts_input_data(model_config);

        group.bench_function(BenchmarkId::new(model_config.name, "coreml-ane"), |b| {
            b.iter(|| {
                let _ = run_session_with_inputs(
                    &session,
                    model_config,
                    &float_inputs,
                    black_box(&tts_data),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark CoreML execution provider with GPU
#[cfg(feature = "coreml-ep")]
fn benchmark_coreml_gpu(c: &mut Criterion) {
    use xybrid_core::runtime_adapter::onnx::{CoreMLComputeUnits, CoreMLConfig};

    let mut group = c.benchmark_group("inference/coreml-gpu");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

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

        let float_inputs = create_float_inputs(model_config);
        let tts_data = create_tts_input_data(model_config);

        group.bench_function(BenchmarkId::new(model_config.name, "coreml-gpu"), |b| {
            b.iter(|| {
                let _ = run_session_with_inputs(
                    &session,
                    model_config,
                    &float_inputs,
                    black_box(&tts_data),
                );
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

// Conditional benchmark groups based on features
#[cfg(not(feature = "coreml-ep"))]
criterion_group!(benches, benchmark_cpu);

#[cfg(feature = "coreml-ep")]
criterion_group!(benches, benchmark_cpu, benchmark_coreml_ane, benchmark_coreml_gpu);

criterion_main!(benches);

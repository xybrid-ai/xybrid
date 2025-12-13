# ONNX Runtime Adapter

ONNX Runtime inference backend via the [ort](https://crates.io/crates/ort) crate.

## Overview

The ONNX adapter is the default runtime backend for xybrid. It provides hardware-accelerated inference across all platforms using Microsoft's ONNX Runtime.

**Key features:**
- **Cross-platform**: Works on macOS, Linux, Windows, iOS, Android
- **Hardware acceleration**: Metal (Apple), CUDA (NVIDIA), NNAPI (Android)
- **Mature ecosystem**: Supports models from PyTorch, TensorFlow, scikit-learn
- **Dynamic shapes**: Handles variable batch sizes and sequence lengths

## Module Structure

```
onnx/
├── mod.rs           # Module exports
├── adapter.rs       # OnnxRuntimeAdapter (RuntimeAdapter trait impl)
├── backend.rs       # OnnxBackend (InferenceBackend trait impl)
├── session.rs       # ONNXSession wrapper (ort crate integration)
├── mobile.rs        # ONNXMobileRuntimeAdapter (Android NNAPI)
└── README.md        # This file
```

## Key Types

### OnnxRuntimeAdapter

High-level adapter implementing `RuntimeAdapter` trait:

```rust
pub struct OnnxRuntimeAdapter {
    models: HashMap<String, ModelMetadata>,
    sessions: HashMap<String, ONNXSession>,
    current_model: Option<String>,
}

impl RuntimeAdapter for OnnxRuntimeAdapter {
    fn name(&self) -> &str { "onnx" }
    fn supported_formats(&self) -> Vec<&'static str> { vec!["onnx", "onnx.gz"] }
    fn load_model(&mut self, path: &str) -> AdapterResult<()>;
    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope>;
}
```

**Use when:** Working with `Envelope` types, orchestration, high-level APIs.

### OnnxBackend

Low-level backend implementing `InferenceBackend` trait:

```rust
pub struct OnnxBackend {
    session: Option<ONNXSession>,
}

impl InferenceBackend for OnnxBackend {
    fn runtime_type(&self) -> RuntimeType { RuntimeType::Onnx }
    fn load_model(&mut self, model_path: &Path, config_path: Option<&Path>) -> BackendResult<()>;
    fn run_inference(&self, inputs: HashMap<String, ArrayD<f32>>) -> BackendResult<HashMap<String, ArrayD<f32>>>;
}
```

**Use when:** Working with tensors directly, custom execution pipelines.

### ONNXSession

Low-level wrapper around `ort::Session`:

```rust
pub struct ONNXSession {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
}
```

**Use when:** Direct ONNX Runtime access needed.

## Hardware Acceleration

| Platform | Backend | Auto-enabled |
|----------|---------|--------------|
| macOS/iOS | Metal | Yes |
| Android | NNAPI | Via `ONNXMobileRuntimeAdapter` |
| Linux/Windows | CUDA | If available |
| All | CPU | Fallback |

Metal is automatically enabled on Apple platforms:

```rust
#[cfg(any(target_os = "macos", target_os = "ios"))]
let use_metal = true;
```

## Usage

### Basic Inference

```rust
use xybrid_core::runtime_adapter::{RuntimeAdapter, OnnxRuntimeAdapter};
use xybrid_core::ir::{Envelope, EnvelopeKind};

let mut adapter = OnnxRuntimeAdapter::new();
adapter.load_model("model.onnx")?;

let input = Envelope::new(EnvelopeKind::Audio(audio_bytes));
let output = adapter.execute(&input)?;
```

### Multi-Model Management

```rust
use xybrid_core::runtime_adapter::RuntimeAdapterExt;

let mut adapter = OnnxRuntimeAdapter::new();

// Load multiple models
adapter.load_model("asr.onnx")?;
adapter.load_model("tts.onnx")?;

// Check loaded models
println!("{:?}", adapter.list_loaded_models());

// Infer on specific model
let output = adapter.infer("asr", &audio_input)?;

// Unload when done
adapter.unload_model("asr")?;
```

### Using OnnxBackend (Tensor-Level)

```rust
use xybrid_core::runtime_adapter::{InferenceBackend, OnnxBackend};
use ndarray::ArrayD;
use std::collections::HashMap;

let mut backend = OnnxBackend::new();
backend.load_model(Path::new("model.onnx"), None)?;

// Prepare input tensors
let mut inputs = HashMap::new();
inputs.insert("input".to_string(), ArrayD::zeros(vec![1, 3, 224, 224]));

// Run inference
let outputs = backend.run_inference(inputs)?;
```

### Android NNAPI

```rust
#[cfg(target_os = "android")]
use xybrid_core::runtime_adapter::ONNXMobileRuntimeAdapter;

let mut adapter = ONNXMobileRuntimeAdapter::new();
adapter.load_model("model.onnx")?;  // Uses NNAPI automatically
```

## Execution Template

ONNX models use `SimpleMode` execution template in `model_metadata.json`:

```json
{
  "execution_template": {
    "type": "SimpleMode",
    "model_file": "model.onnx"
  }
}
```

For multi-model execution:

```json
{
  "execution_template": {
    "type": "MultiModel",
    "encoder_file": "encoder.onnx",
    "decoder_file": "decoder.onnx"
  }
}
```

See [BUNDLE.md](../../../BUNDLE.md) for complete metadata schema.

## Model Compatibility

ONNX Runtime supports models exported from:

| Framework | Export Method |
|-----------|---------------|
| PyTorch | `torch.onnx.export()` |
| TensorFlow | `tf2onnx` converter |
| scikit-learn | `skl2onnx` converter |
| HuggingFace | `optimum` library |

### Quantization Support

| Format | Support |
|--------|---------|
| FP32 | Full |
| FP16 | Full |
| INT8 | Full (via ONNX quantization) |
| INT4 | Experimental |

## Comparison with Candle

| Feature | ONNX | Candle |
|---------|------|--------|
| Model format | `.onnx` | `.safetensors` |
| Dependencies | C++ (ort) | Pure Rust |
| Cross-compile | Requires ort libs | Easy |
| Model support | Broad (any ONNX) | Transformers focus |
| Performance | Highly optimized | Good, improving |

**Choose ONNX when:**
- You have an existing ONNX model
- You need maximum compatibility
- Hardware acceleration is critical

**Choose Candle when:**
- You want pure Rust (no C++ deps)
- You're using transformer models (Whisper, LLaMA)
- Cross-compilation simplicity matters

## Related Documentation

| Document | Description |
|----------|-------------|
| [../README.md](../README.md) | Runtime adapter overview |
| [../candle/README.md](../candle/README.md) | Candle backend (alternative) |
| [BUNDLE.md](../../../BUNDLE.md) | Bundle specification |
| [ort crate](https://docs.rs/ort) | ONNX Runtime Rust bindings |

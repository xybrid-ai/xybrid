# Runtime Adapter

Unified interface for model inference across different ML backends.

## Overview

The runtime adapter module abstracts over different inference runtimes (ONNX, CoreML, Candle), providing a consistent API for the orchestrator to execute models without knowing the underlying implementation.

**Key benefits:**
- **Runtime agnostic** - Same interface for ONNX, CoreML, Candle
- **Thread-safe** - All adapters implement `Send + Sync`
- **Feature-gated** - Optional backends via Cargo features

## Module Structure

```
runtime_adapter/
├── mod.rs                 # Module exports, RuntimeAdapter trait
├── inference_backend.rs   # InferenceBackend trait (tensor-level)
├── metadata_driven.rs     # MetadataDrivenAdapter (bundle execution)
├── tensor_utils.rs        # Tensor conversion utilities
├── mel_spectrogram.rs     # Audio feature extraction
│
├── onnx/                  # ONNX Runtime backend
│   ├── README.md
│   ├── adapter.rs         # OnnxRuntimeAdapter
│   ├── backend.rs         # OnnxBackend
│   ├── session.rs         # ONNXSession wrapper
│   └── mobile.rs          # Android NNAPI support
│
├── candle/                # Candle backend (pure Rust)
│   ├── README.md
│   ├── adapter.rs         # CandleRuntimeAdapter
│   ├── backend.rs         # CandleBackend
│   ├── device.rs          # Device selection
│   ├── model.rs           # CandleModel trait
│   └── whisper.rs         # Whisper ASR model
│
└── coreml/                # CoreML backend (Apple)
    ├── adapter.rs         # CoreMLRuntimeAdapter
    └── mod.rs
```

## Two-Level Architecture

### High-Level: RuntimeAdapter

Works with `Envelope` types (Audio, Text, Embedding):

```rust
pub trait RuntimeAdapter: Send + Sync {
    fn name(&self) -> &str;
    fn supported_formats(&self) -> Vec<&'static str>;
    fn load_model(&mut self, path: &str) -> AdapterResult<()>;
    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope>;
}
```

**Use when:** Building orchestration, pipelines, or high-level APIs.

### Low-Level: InferenceBackend

Works directly with tensors (`HashMap<String, ArrayD<f32>>`):

```rust
pub trait InferenceBackend: Send + Sync {
    fn runtime_type(&self) -> RuntimeType;
    fn load_model(&mut self, model_path: &Path, config_path: Option<&Path>) -> BackendResult<()>;
    fn run_inference(&self, inputs: HashMap<String, ArrayD<f32>>) -> BackendResult<HashMap<String, ArrayD<f32>>>;
}
```

**Use when:** Building custom execution pipelines or `TemplateExecutor` integration.

## Available Backends

| Backend | Feature Flag | Platforms | Model Format |
|---------|--------------|-----------|--------------|
| **ONNX** | (default) | All | `.onnx` |
| **Candle** | `candle` | All | `.safetensors` |
| **CoreML** | (auto) | macOS/iOS | `.mlpackage` |

## Feature Flags

```toml
[features]
default = ["onnx"]
onnx = ["dep:ort"]
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers"]
candle-metal = ["candle", "candle-core/metal"]
candle-cuda = ["candle", "candle-core/cuda"]
```

## Usage

### Basic ONNX Inference

```rust
use xybrid_core::runtime_adapter::{RuntimeAdapter, OnnxRuntimeAdapter};
use xybrid_core::ir::{Envelope, EnvelopeKind};

let mut adapter = OnnxRuntimeAdapter::new();
adapter.load_model("model.onnx")?;

let input = Envelope::new(EnvelopeKind::Audio(audio_bytes));
let output = adapter.execute(&input)?;
```

### Using InferenceBackend

```rust
use xybrid_core::runtime_adapter::{InferenceBackend, OnnxBackend};
use std::collections::HashMap;

let mut backend = OnnxBackend::new();
backend.load_model(Path::new("model.onnx"), None)?;

let mut inputs = HashMap::new();
inputs.insert("input".to_string(), tensor_data);

let outputs = backend.run_inference(inputs)?;
```

### Metadata-Driven Execution

For bundle-based execution, use `TemplateExecutor` instead:

```rust
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::execution::ModelMetadata;

let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
let mut executor = TemplateExecutor::with_base_path("path/to/model");
let output = executor.execute(&metadata, &input)?;
```

## Runtime Selection

The runtime is determined by `model_metadata.json`:

```json
{
  "execution_template": {
    "type": "SimpleMode",        // ONNX
    "model_file": "model.onnx"
  }
}
```

```json
{
  "execution_template": {
    "type": "CandleModel",       // Candle
    "model_file": "model.safetensors",
    "model_type": "whisper"
  }
}
```

## Related Documentation

| Document | Description |
|----------|-------------|
| [onnx/README.md](onnx/README.md) | ONNX Runtime backend details |
| [candle/README.md](candle/README.md) | Candle backend and model support |
| [BUNDLE.md](../../BUNDLE.md) | Bundle specification and execution templates |

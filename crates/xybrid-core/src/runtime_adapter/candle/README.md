# Candle Runtime Adapter

Pure Rust ML inference via [Candle](https://github.com/huggingface/candle) framework.

## Overview

The Candle adapter provides an alternative to ONNX Runtime for models that benefit from pure Rust execution. Key advantages:

- **No C++ dependencies**: Pure Rust, easier cross-compilation
- **Native transformer support**: Whisper, LLaMA, BERT implementations
- **Hardware acceleration**: Metal (macOS), CUDA (Linux/Windows)
- **SafeTensors format**: HuggingFace-compatible weights

## Module Structure

```
candle/
├── mod.rs           # Module exports
├── adapter.rs       # CandleRuntimeAdapter (RuntimeAdapter trait impl)
├── backend.rs       # CandleBackend (InferenceBackend trait impl)
├── device.rs        # Device selection (CPU/Metal/CUDA)
├── model.rs         # CandleModel trait + CandleModelType enum
├── whisper.rs       # WhisperModel implementation
└── README.md        # This file
```

## Feature Flags

```toml
# In Cargo.toml
[features]
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers"]
candle-metal = ["candle", "candle-core/metal"]
candle-cuda = ["candle", "candle-core/cuda"]
```

Build with: `cargo build --features candle`

## Architecture

### CandleModel Trait

All Candle models implement this trait for uniform handling:

```rust
pub trait CandleModel: Send {
    fn model_type(&self) -> CandleModelType;
    fn device(&self) -> &Device;
    fn run(&mut self, inputs: HashMap<String, Tensor>) -> ModelResult<HashMap<String, Tensor>>;
    fn input_names(&self) -> Vec<&str>;
    fn output_names(&self) -> Vec<&str>;
}
```

### CandleModelType Enum

Routes execution to appropriate model implementation:

```rust
pub enum CandleModelType {
    Whisper,  // ASR - implemented
    LLaMA,    // LLM - planned
    Bert,     // Embeddings - planned
    Generic,  // Fallback
}
```

### Factory Function

```rust
pub fn load_candle_model(
    model_type: CandleModelType,
    model_path: &Path,
    device: &Device,
) -> ModelResult<Box<dyn CandleModel>>
```

## Adding New Models

### Step 1: Add Model Type

Update `model.rs`:

```rust
pub enum CandleModelType {
    Whisper,
    LLaMA,    // Add new variant
    // ...
}

impl CandleModelType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "whisper" => Self::Whisper,
            "llama" | "llama2" | "llama3" => Self::LLaMA,
            // ...
        }
    }
}
```

### Step 2: Create Model Module

Create `llama.rs` with:

```rust
pub struct LlamaModel {
    model: candle_transformers::models::llama::Llama,
    tokenizer: tokenizers::Tokenizer,
    config: Config,
    device: Device,
}

impl LlamaModel {
    pub fn load(model_dir: &Path, device: &Device) -> anyhow::Result<Self> { ... }
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> anyhow::Result<String> { ... }
}
```

### Step 3: Create Trait Wrapper

In `model.rs`:

```rust
pub struct LlamaModelWrapper {
    pub model: super::llama::LlamaModel,
}

impl CandleModel for LlamaModelWrapper {
    fn model_type(&self) -> CandleModelType { CandleModelType::LLaMA }
    fn device(&self) -> &Device { self.model.device() }
    fn run(&mut self, inputs: HashMap<String, Tensor>) -> ModelResult<HashMap<String, Tensor>> {
        // Implementation
    }
    fn input_names(&self) -> Vec<&str> { vec!["prompt"] }
    fn output_names(&self) -> Vec<&str> { vec!["text"] }
}
```

### Step 4: Update Factory

In `model.rs`, add to `load_candle_model()`:

```rust
CandleModelType::LLaMA => {
    let model = LlamaModel::load(model_path, device)?;
    Ok(Box::new(LlamaModelWrapper { model }))
}
```

### Step 5: Export Module

In `mod.rs`:

```rust
mod llama;
pub use llama::{LlamaConfig, LlamaModel};
```

### Step 6: Update TemplateExecutor

In `template_executor.rs`, add case to `execute_candle_model()`:

```rust
"llama" => {
    // Handle LLaMA-specific execution
}
```

## Execution Template

Candle models use `CandleModel` execution template in `model_metadata.json`:

```json
{
  "execution_template": {
    "type": "CandleModel",
    "model_file": "model.safetensors",
    "config_file": "config.json",
    "tokenizer_file": "tokenizer.json",
    "model_type": "whisper"
  }
}
```

See [BUNDLE.md](../../../BUNDLE.md) for complete metadata schema.

## Device Selection

```rust
use xybrid_core::runtime_adapter::candle::{select_device, DeviceSelection};

// Auto-select best available device
let device = select_device(DeviceSelection::Auto)?;

// Force specific device
let device = select_device(DeviceSelection::Cpu)?;
let device = select_device(DeviceSelection::Metal)?;  // macOS only
let device = select_device(DeviceSelection::Cuda(0))?; // GPU index
```

## Currently Supported Models (Xybrid)

| Model | Type | Status | Notes |
|-------|------|--------|-------|
| Whisper | ASR | ✅ Complete | tiny/base/small/medium/large |
| LLaMA | LLM | ⏳ Planned | Abstraction ready |
| BERT | Embeddings | ⏳ Planned | Abstraction ready |

## Candle Framework Model Support

The [Candle framework](https://github.com/huggingface/candle) supports many more models that can be integrated into xybrid. Below is the full list of models available in `candle-transformers`:

### Core Models (High Priority for Xybrid)

| Model | Type | Use Case |
|-------|------|----------|
| **Whisper** | ASR | Speech recognition ✅ Implemented |
| **YOLO** | Vision | Pose estimation, object detection |
| **LLaMA v1/v2/v3** | LLM | Text generation (includes SOLAR-10.7B) |
| **T5** | LLM | Text generation, translation |
| **Segment Anything (SAM)** | Vision | Image segmentation |
| **BLIP** | Vision | Image captioning |

### Text Generation Models

| Model | Parameters | Notes |
|-------|------------|-------|
| **Phi-1/1.5/2/3** | 1.3b-3.8b | Performance on par with 7b models |
| **Gemma v1/v2** | 2b-9b | Google DeepMind |
| **RecurrentGemma** | 2b-7b | Griffin-based, attention + RNN |
| **Mistral 7B** | 7b | Better than 13b models (Sep 2023) |
| **Mixtral 8x7B** | 8x7b (MoE) | Sparse mixture of experts |
| **Falcon** | Various | General LLM |
| **Qwen 1.5** | Various | Bilingual (English/Chinese) |
| **Yi-6B/Yi-34B** | 6b-34b | Bilingual (English/Chinese) |
| **GLM4** | Various | Multilingual multimodal (THUDM) |
| **Mamba** | Various | State space model (inference only) |
| **RWKV v5/v6** | Various | RNN with transformer-level performance |

### Code Generation Models

| Model | Parameters | Notes |
|-------|------------|-------|
| **StarCoder/StarCoder2** | Various | Code generation |
| **Codegeex4** | Various | Code completion, function calling |
| **Replit-code-v1.5** | 3.3b | Code completion |

### Specialized Models

| Model | Parameters | Notes |
|-------|------------|-------|
| **StableLM-3B-4E1T** | 3b | Pre-trained on 1T tokens |
| **StableLM-2** | 1.6b | Trained on 2T tokens + code variants |
| **Quantized LLaMA** | Various | Same quantization as llama.cpp |

### Adding Candle Models to Xybrid

To add any of these models:

1. Check if `candle-transformers` has the model implementation
2. Follow the [Adding New Models](#adding-new-models) steps above
3. Create appropriate `model_metadata.json` with `CandleModel` execution template

Example for YOLO:
```json
{
  "execution_template": {
    "type": "CandleModel",
    "model_file": "yolov8n.safetensors",
    "model_type": "yolo"
  }
}
```

📖 See [candle-transformers models](https://github.com/huggingface/candle/tree/main/candle-transformers/src/models) for implementation details.

## Usage Example

```rust
use xybrid_core::runtime_adapter::candle::{
    select_device, DeviceSelection, WhisperConfig, WhisperModel,
};

let device = select_device(DeviceSelection::Auto)?;
let mut model = WhisperModel::load(&model_path, &device)?;
let text = model.transcribe_pcm(&audio_samples)?;
```

## Related Documentation

| Document | Description |
|----------|-------------|
| [../README.md](../README.md) | Runtime adapter overview |
| [../onnx/README.md](../onnx/README.md) | ONNX backend (alternative) |
| [BUNDLE.md](../../../BUNDLE.md) | Bundle specification with Candle section |
| [CANDLE_INTEGRATION_PLAN.md](../../../../docs/archive/CANDLE_INTEGRATION_PLAN.md) | Integration history (archived) |

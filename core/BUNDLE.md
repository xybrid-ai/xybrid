# Xybrid Bundle Specification

This document defines the structure and metadata format for xybrid model bundles (`.xyb` files).

## Overview

Xybrid uses a metadata-driven execution system that separates model files from execution logic. This allows:

- **Portable bundles**: Same model can run on multiple platforms with appropriate runtime
- **Declarative execution**: No custom code needed for new models
- **Standardized pipeline**: Preprocessing → Model Execution → Postprocessing

## Bundle Structure

A `.xyb` bundle is a gzip-compressed tar archive with the following structure:

```
my-model.xyb
├── manifest.json           # Bundle manifest (required)
├── model_metadata.json     # Execution metadata (required)
├── model.onnx              # Model file(s) (required)
├── vocab.txt               # Supporting files (varies by model)
└── voices.bin              # Additional resources (varies by model)
```

## Manifest (`manifest.json`)

The manifest describes the bundle for registry and download purposes:

```json
{
  "model_id": "kokoro-82m",
  "version": "1.0",
  "created_at": "2024-12-04T10:30:00Z",
  "platform": "macos-arm64",
  "model_type": "onnx",
  "has_metadata": true,
  "files": [
    "kokoro-v1.0.fp16.onnx",
    "tokens.txt",
    "voices.bin",
    "model_metadata.json"
  ],
  "checksum": "sha256:abc123..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Unique model identifier |
| `version` | string | Semantic version |
| `created_at` | string | ISO 8601 timestamp |
| `platform` | string | Target platform (see [Platforms](#platforms)) |
| `model_type` | string | Runtime type: `onnx`, `coreml`, `tflite` |
| `has_metadata` | bool | Whether bundle includes `model_metadata.json` |
| `files` | array | List of files in bundle |
| `checksum` | string | Bundle integrity hash |

## Model Metadata (`model_metadata.json`)

The model metadata defines how to execute the model. This is the core of xybrid's metadata-driven execution.

### Schema

```json
{
  "model_id": "kokoro-82m",
  "version": "1.0",
  "description": "Kokoro-82M TTS model",
  "execution_template": { ... },
  "preprocessing": [ ... ],
  "postprocessing": [ ... ],
  "files": [ ... ],
  "metadata": { ... }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Unique model identifier |
| `version` | string | Yes | Semantic version |
| `description` | string | No | Human-readable description |
| `execution_template` | object | Yes | How to run the model |
| `preprocessing` | array | No | Input transformations |
| `postprocessing` | array | No | Output transformations |
| `files` | array | Yes | List of required files |
| `metadata` | object | No | Additional model info (task, license, etc.) |

### Execution Templates

Defines how the model should be executed.

#### SimpleMode

For single-model inference (classifiers, embedders, encoders, TTS):

```json
{
  "execution_template": {
    "type": "SimpleMode",
    "model_file": "model.onnx"
  }
}
```

#### CandleModel

For models using the Candle runtime (pure Rust, SafeTensors format):

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

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_file` | string | Yes | SafeTensors weights file |
| `config_file` | string | No | HuggingFace config.json |
| `tokenizer_file` | string | No | HuggingFace tokenizer.json |
| `model_type` | string | Yes | Model type: `whisper`, `llama`, `bert` |

**Supported model types:**
- `whisper` - ASR (speech recognition)
- `llama` - LLM (text generation) - planned
- `bert` - Embeddings - planned

**Requirements:**
- Build with `--features candle`
- Models must be in SafeTensors format (HuggingFace compatible)

See [Candle README](src/runtime_adapter/candle/README.md) for implementation details.

#### Pipeline

For multi-stage models (encoder-decoder, autoregressive):

```json
{
  "execution_template": {
    "type": "Pipeline",
    "stages": [
      {
        "name": "encoder",
        "model_file": "encoder.onnx",
        "execution_mode": { "type": "SingleShot" },
        "inputs": ["mel_spectrogram"],
        "outputs": ["cross_k", "cross_v"]
      },
      {
        "name": "decoder",
        "model_file": "decoder.onnx",
        "execution_mode": {
          "type": "Autoregressive",
          "max_tokens": 448,
          "start_token_id": 50258,
          "end_token_id": 50256
        },
        "inputs": ["tokens", "kv_cache"],
        "outputs": ["logits"]
      }
    ],
    "config": {
      "kv_cache_shape": [4, 1, 448, 384]
    }
  }
}
```

#### Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `SingleShot` | Run once | Encoders, embedders |
| `Autoregressive` | Loop until end token | Language models, decoders |
| `IterativeRefinement` | Refine over N steps | Diffusion models |

### Preprocessing Steps

Applied to input data before model execution.

| Type | Description | Key Parameters |
|------|-------------|----------------|
| `MelSpectrogram` | Audio → Mel spectrogram | `n_mels`, `sample_rate`, `fft_size`, `hop_length` |
| `AudioDecode` | Audio bytes → PCM samples | `sample_rate`, `channels` |
| `Phonemize` | Text → Phoneme tokens | `tokens_file`, `backend`, `language` |
| `Tokenize` | Text → Token IDs | `vocab_file`, `tokenizer_type`, `max_length` |
| `Normalize` | Normalize tensor values | `mean`, `std` |
| `Resize` | Resize image | `width`, `height`, `interpolation` |
| `CenterCrop` | Center crop image | `width`, `height` |
| `Reshape` | Reshape tensor | `shape` |

#### MelSpectrogram Step

The `MelSpectrogram` preprocessing step supports explicit mel scale selection and presets.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | string | null | Preset name ("whisper", "whisper-large") |
| `n_mels` | int | 80 | Number of mel frequency bins |
| `sample_rate` | int | 16000 | Audio sample rate (Hz) |
| `fft_size` | int | 400 | FFT window size |
| `hop_length` | int | 160 | Hop length between frames |
| `mel_scale` | string | "slaney" | Mel scale: "slaney" or "htk" |
| `max_frames` | int | 3000 | Maximum output frames (30s @ 100fps) |

**Mel Scales:**

| Scale | Description | Use Case |
|-------|-------------|----------|
| `slaney` | Piecewise linear/log (librosa default) | Whisper, transformers.js |
| `htk` | Logarithmic formula | Older implementations |

**Presets:**

Using presets simplifies configuration for common models:

```json
{ "type": "MelSpectrogram", "preset": "whisper" }
```

This is equivalent to:
```json
{
  "type": "MelSpectrogram",
  "n_mels": 80,
  "sample_rate": 16000,
  "fft_size": 400,
  "hop_length": 160,
  "mel_scale": "slaney",
  "max_frames": 3000
}
```

**Slaney mel scale formula:**
- For freq < 1000 Hz: `mel = 3 × freq / 200`
- For freq >= 1000 Hz: `mel = 15 + 27 × ln(freq / 1000) / ln(6.4)`

**HTK mel scale formula:**
- `mel = 2595 × log10(1 + freq / 700)`

**Whisper-specific features** (when using Slaney scale):
- Reflect padding (200 samples each side)
- Slaney filter normalization: `2 / (upper_freq - lower_freq)`
- Output normalization: `(max(log_spec, max - 8) + 4) / 4`

#### Phonemize Step

For TTS models requiring phoneme input:

```json
{
  "type": "Phonemize",
  "tokens_file": "tokens.txt",
  "backend": "EspeakNG",
  "language": "en-us",
  "add_padding": true,
  "normalize_text": true
}
```

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `CmuDictionary` | CMU dict (ARPABET → IPA) | `dict_file` parameter |
| `EspeakNG` | espeak-ng direct IPA | System espeak-ng, `--features espeak` |

### Postprocessing Steps

Applied to model output before returning results.

| Type | Description | Key Parameters |
|------|-------------|----------------|
| `BPEDecode` | Token IDs → Text | `vocab_file` |
| `CTCDecode` | CTC output → Text | `vocab_file`, `blank_index` |
| `TTSAudioEncode` | Float waveform → Audio bytes | `sample_rate`, `apply_postprocessing` |
| `Argmax` | Get max index | `dim` |
| `Softmax` | Get probabilities | `dim` |
| `TopK` | Get top K predictions | `k`, `dim` |
| `MeanPool` | Pool embeddings | `dim` |
| `Denormalize` | Reverse normalization | `mean`, `std` |

## Platforms

Supported platform identifiers:

| Platform | Description |
|----------|-------------|
| `macos-arm64` | macOS Apple Silicon |
| `macos-x86_64` | macOS Intel |
| `ios-aarch64` | iOS ARM64 |
| `android-arm64` | Android ARM64 |
| `android-x86_64` | Android x86_64 (emulator) |
| `linux-x86_64` | Linux x86_64 |
| `windows-x86_64` | Windows x86_64 |
| `wasm-wgpu` | WebAssembly with WebGPU |

## Model Types

Supported runtime types:

| Type | Extension | Description |
|------|-----------|-------------|
| `onnx` | `.onnx` | ONNX Runtime (cross-platform) |
| `candle` | `.safetensors` | Candle (pure Rust, feature-gated) |
| `coreml` | `.mlmodel`, `.mlpackage` | Apple CoreML (iOS/macOS) |
| `tflite` | `.tflite` | TensorFlow Lite (mobile) |

## Registry Integration

Bundles are registered in `registry/bundles/bundles.json`:

```json
{
  "bundles": {
    "kokoro-82m": {
      "model_id": "kokoro-82m",
      "version": "1.0",
      "targets": [
        {
          "platform": "macos-arm64",
          "model_file": "kokoro-v1.0.fp16.onnx",
          "model_type": "onnx",
          "source_dir": "test_models/kokoro-82m",
          "fallback": "error"
        }
      ],
      "description": "Kokoro-82M TTS model"
    }
  }
}
```

### Target Configuration

| Field | Description |
|-------|-------------|
| `platform` | Target platform identifier |
| `model_file` | Primary model file name |
| `model_type` | Runtime type |
| `source_dir` | Path to model directory (contains `model_metadata.json`) |
| `source_path` | Alternative: direct path to model file |
| `fallback` | Error handling: `error` or `placeholder` |

## Complete Examples

### TTS Model (Kokoro-82M)

```json
{
  "model_id": "kokoro-82m",
  "version": "1.0",
  "description": "Kokoro-82M - High-quality TTS (82M params, 24 voices)",
  "execution_template": {
    "type": "SimpleMode",
    "model_file": "kokoro-v1.0.fp16.onnx"
  },
  "preprocessing": [
    {
      "type": "Phonemize",
      "tokens_file": "tokens.txt",
      "backend": "EspeakNG",
      "language": "en-us",
      "add_padding": true,
      "normalize_text": true
    }
  ],
  "postprocessing": [
    {
      "type": "TTSAudioEncode",
      "sample_rate": 24000,
      "apply_postprocessing": true
    }
  ],
  "files": [
    "kokoro-v1.0.fp16.onnx",
    "tokens.txt",
    "voices.bin"
  ],
  "metadata": {
    "task": "text-to-speech",
    "language": "en",
    "sample_rate": 24000,
    "license": "Apache-2.0"
  }
}
```

### ASR Model (Wav2Vec2)

```json
{
  "model_id": "wav2vec2-base-960h",
  "version": "1.0",
  "description": "Wav2Vec2 Base ASR with CTC decoding",
  "execution_template": {
    "type": "SimpleMode",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    {
      "type": "AudioDecode",
      "sample_rate": 16000,
      "channels": 1
    }
  ],
  "postprocessing": [
    {
      "type": "CTCDecode",
      "vocab_file": "vocab.json",
      "blank_index": 0
    }
  ],
  "files": [
    "model.onnx",
    "vocab.json"
  ],
  "metadata": {
    "task": "automatic-speech-recognition",
    "language": "en",
    "sample_rate": 16000
  }
}
```

### Image Classifier

```json
{
  "model_id": "resnet50",
  "version": "1.0",
  "description": "ResNet50 ImageNet classifier",
  "execution_template": {
    "type": "SimpleMode",
    "model_file": "resnet50.onnx"
  },
  "preprocessing": [
    {
      "type": "Resize",
      "width": 256,
      "height": 256
    },
    {
      "type": "CenterCrop",
      "width": 224,
      "height": 224
    },
    {
      "type": "Normalize",
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  ],
  "postprocessing": [
    {
      "type": "Softmax",
      "dim": 1
    },
    {
      "type": "TopK",
      "k": 5
    }
  ],
  "files": [
    "resnet50.onnx",
    "labels.txt"
  ],
  "metadata": {
    "task": "image-classification",
    "num_classes": 1000
  }
}
```

### Candle ASR Model (Whisper)

```json
{
  "model_id": "whisper-tiny-candle",
  "version": "1.0",
  "description": "Whisper Tiny ASR via Candle (pure Rust)",
  "execution_template": {
    "type": "CandleModel",
    "model_file": "model.safetensors",
    "config_file": "config.json",
    "tokenizer_file": "tokenizer.json",
    "model_type": "whisper"
  },
  "preprocessing": [
    {
      "type": "AudioDecode",
      "sample_rate": 16000,
      "channels": 1
    }
  ],
  "postprocessing": [],
  "files": [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "melfilters.bytes",
    "model_metadata.json"
  ],
  "metadata": {
    "task": "speech-recognition",
    "architecture": "WhisperForConditionalGeneration",
    "framework": "candle"
  }
}
```

**Notes:**
- Candle Whisper handles mel spectrogram internally (no `MelSpectrogram` preprocessing needed)
- Candle Whisper returns text directly (no `TokenDecode` postprocessing needed)
- Requires `melfilters.bytes` for audio preprocessing

## Quantization

ONNX models can be quantized to reduce size and improve performance:

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| FP32 | 100% | Baseline | Development/reference |
| FP16 | ~50% | Near-identical | **Default for deployment** |
| INT8 | ~25% | Minor degradation | Mobile/edge devices |

Quantized models use the same `model_metadata.json` - only the `model_file` field changes:

```json
// FP16 (recommended default)
"model_file": "kokoro-v1.0.fp16.onnx"

// INT8 (smallest size)
"model_file": "kokoro-v1.0.int8.onnx"
```

## Future Considerations

The following features are planned for future versions:

- **Quantization variants in registry**: Single model ID with multiple size/quality options
- **Conditional preprocessing**: Platform-specific preprocessing chains
- **Model composition**: Combining multiple models into a single bundle
- **Streaming execution**: For real-time applications
- **Full DAG execution**: Arbitrary computation graphs

## Reference

### Source Files

| Module | Path | Description |
|--------|------|-------------|
| Execution | `src/execution_template.rs` | `ModelMetadata`, `ExecutionTemplate`, `PreprocessingStep` |
| Template Executor | `src/template_executor.rs` | Executes model metadata |
| Preprocessing | `src/preprocessing/` | Preprocessing step wrappers |
| Mel Spectrogram (step) | `src/preprocessing/mel_spectrogram.rs` | WAV parsing, resampling, step interface |
| Mel Spectrogram (Whisper) | `src/audio/mel/whisper.rs` | Whisper-compatible implementation |
| Mel Spectrogram (legacy) | `src/runtime_adapter/mel_spectrogram.rs` | mel-spec library wrapper |

### Runtime Adapters

| Runtime | Path | Description |
|---------|------|-------------|
| ONNX | `src/runtime_adapter/onnx/` | ONNX Runtime adapter (default) |
| Candle | `src/runtime_adapter/candle/` | Pure Rust adapter (feature-gated) |
| CoreML | `src/runtime_adapter/coreml/` | Apple CoreML adapter (iOS/macOS) |

### Registry

- Bundle index: [registry/bundles/bundles.json](../registry/bundles/bundles.json)
- Candle implementation: [src/runtime_adapter/candle/README.md](src/runtime_adapter/candle/README.md)

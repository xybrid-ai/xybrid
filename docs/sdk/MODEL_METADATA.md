# model_metadata.json Reference

> Every model directory must contain a `model_metadata.json` file that tells xybrid how to execute the model.
>
> **JSON Schema**: [`model_metadata.schema.json`](model_metadata.schema.json) — use with your editor for validation and autocomplete.

## Top-Level Structure

```json
{
  "model_id": "string (required)",
  "version": "string (required)",
  "description": "string (optional)",
  "execution_template": { "type": "..." },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["model.onnx"],
  "metadata": {},
  "voices": null,
  "max_chunk_chars": null,
  "trim_trailing_samples": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | yes | Unique model identifier (e.g., `"kokoro-82m"`) |
| `version` | string | yes | Model version (e.g., `"1.0"`) |
| `description` | string | no | Human-readable description |
| `execution_template` | object | yes | Defines the model format and runtime — see [Execution Templates](#execution-templates) |
| `preprocessing` | array | no | Steps applied to input before model execution — see [Preprocessing Steps](#preprocessing-steps) |
| `postprocessing` | array | no | Steps applied to model output — see [Postprocessing Steps](#postprocessing-steps) |
| `files` | string[] | yes | All files in the model bundle (used for cache validation) |
| `metadata` | object | no | Arbitrary key-value metadata (task, architecture, license, etc.) |
| `voices` | object | no | Voice configuration for TTS models — see [Voice Configuration](#voice-configuration) |
| `max_chunk_chars` | integer | no | Maximum characters per TTS chunk (default: 350) |
| `trim_trailing_samples` | integer | no | Audio samples to trim per TTS chunk (default: 0) |

---

## Execution Templates

The `execution_template` field defines what model format to load and which runtime to use. Each type is identified by the `"type"` discriminator.

### `Onnx`

ONNX model execution via ONNX Runtime. The most common format.

```json
{
  "type": "Onnx",
  "model_file": "model.onnx"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_file` | string | yes | Path to `.onnx` file (relative to model directory) |

### `Gguf`

GGUF model execution for local LLMs via llama.cpp.

```json
{
  "type": "Gguf",
  "model_file": "model-Q4_K_M.gguf",
  "chat_template": null,
  "context_length": 4096
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_file` | string | yes | — | Path to `.gguf` file |
| `chat_template` | string | no | `null` | Path to chat template JSON file |
| `context_length` | integer | no | `4096` | Maximum context length in tokens |

### `SafeTensors`

SafeTensors model execution via Candle (pure Rust). Used for Whisper models.

```json
{
  "type": "SafeTensors",
  "model_file": "model.safetensors",
  "architecture": "whisper",
  "config_file": "config.json",
  "tokenizer_file": "tokenizer.json"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_file` | string | yes | Path to `.safetensors` file |
| `architecture` | string | no | Model architecture for routing (e.g., `"whisper"`) |
| `config_file` | string | no | Path to model configuration JSON |
| `tokenizer_file` | string | no | Path to tokenizer JSON |

### `CoreMl`

CoreML model execution (Apple platforms only).

```json
{
  "type": "CoreMl",
  "model_file": "model.mlmodel"
}
```

### `TfLite`

TensorFlow Lite model execution (mobile).

```json
{
  "type": "TfLite",
  "model_file": "model.tflite"
}
```

### `ModelGraph`

Multi-model DAG execution (pipeline of multiple models in a single bundle).

```json
{
  "type": "ModelGraph",
  "stages": [
    {
      "name": "encoder",
      "model_file": "encoder.onnx",
      "execution_mode": { "type": "SingleShot" },
      "inputs": ["audio"],
      "outputs": ["encoder_output"],
      "config": {}
    },
    {
      "name": "decoder",
      "model_file": "decoder.onnx",
      "execution_mode": {
        "type": "Autoregressive",
        "max_tokens": 256,
        "start_token_id": 1,
        "end_token_id": 2,
        "repetition_penalty": 1.1
      },
      "inputs": ["encoder_output"],
      "outputs": ["text"],
      "config": {}
    }
  ],
  "config": {}
}
```

#### Execution Modes

Each stage in a `ModelGraph` has an `execution_mode`:

| Mode | Description | Use Case |
|------|-------------|----------|
| `SingleShot` | Run the model once (default) | Encoders, classifiers |
| `Autoregressive` | Loop until end token | Text generation, decoding |
| `WhisperDecoder` | Whisper-specific with KV cache | Whisper ASR |
| `IterativeRefinement` | Multiple passes with schedule | Diffusion models |

---

## Preprocessing Steps

Preprocessing transforms input data before model execution. Each step is identified by `"type"`.

### `AudioDecode`

Converts raw WAV bytes to PCM float32 samples.

```json
{ "type": "AudioDecode", "sample_rate": 16000, "channels": 1 }
```

| Field | Type | Description |
|-------|------|-------------|
| `sample_rate` | integer | Target sample rate in Hz |
| `channels` | integer | Number of channels (1 = mono, 2 = stereo) |

**Input**: `Envelope::Audio(wav_bytes)` → **Output**: Float32 PCM samples

### `Phonemize`

Converts text to phoneme token IDs for TTS models.

```json
{
  "type": "Phonemize",
  "tokens_file": "tokens.txt",
  "backend": "MisakiDictionary",
  "dict_file": null,
  "language": null,
  "add_padding": true,
  "normalize_text": false,
  "silence_tokens": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tokens_file` | string | — | Path to token-to-ID mapping file |
| `backend` | string | `"MisakiDictionary"` | Phonemizer backend (see below) |
| `dict_file` | string | `null` | Dictionary file path (CmuDictionary only) |
| `language` | string | `null` | Language code for EspeakNG (e.g., `"en-us"`) |
| `add_padding` | boolean | `true` | Add padding tokens at start/end |
| `normalize_text` | boolean | `false` | Apply text cleanup before phonemization |
| `silence_tokens` | integer | `null` | Silence tokens to prepend (smooths plosives) |

**Phonemizer backends**:

| Backend | Dependencies | Notes |
|---------|-------------|-------|
| `MisakiDictionary` | None (pure Rust) | **Recommended.** Bundled dictionaries + rule-based G2P fallback |
| `CmuDictionary` | None (pure Rust) | Legacy ARPABET-based, English only |
| `EspeakNG` | `espeak-ng` system install | Multi-language support |
| `OpenPhonemizer` | ONNX model (~59MB) | Dictionary + neural G2P fallback |

**Input**: `Envelope::Text(string)` → **Output**: Token IDs (i64)

### `Tokenize`

Tokenizes text using a vocabulary file.

```json
{ "type": "Tokenize", "vocab_file": "vocab.json", "tokenizer_type": "BPE", "max_length": 512 }
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vocab_file` | string | — | Path to vocabulary file |
| `tokenizer_type` | string | — | `"BPE"`, `"WordPiece"`, or `"SentencePiece"` |
| `max_length` | integer | `null` | Maximum sequence length (truncate if exceeded) |

**Input**: `Envelope::Text(string)` → **Output**: Token IDs

### `MelSpectrogram`

Converts audio to mel spectrogram for ASR models.

```json
{
  "type": "MelSpectrogram",
  "preset": "whisper",
  "n_mels": 80,
  "sample_rate": 16000,
  "fft_size": 400,
  "hop_length": 160,
  "mel_scale": "slaney",
  "max_frames": 3000
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `preset` | string | `null` | Use preset config: `"whisper"`, `"whisper-large"` |
| `n_mels` | integer | `80` | Number of mel frequency bins |
| `sample_rate` | integer | `16000` | Audio sample rate in Hz |
| `fft_size` | integer | `400` | FFT window size |
| `hop_length` | integer | `160` | Hop length between frames |
| `mel_scale` | string | `"slaney"` | Mel scale type: `"slaney"` or `"htk"` |
| `max_frames` | integer | `3000` | Maximum output frames (3000 = 30s at 100fps) |

### `Normalize`

Normalizes tensor values per channel: `(x - mean) / std`.

```json
{ "type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] }
```

### `Resize`

Resizes image to target dimensions.

```json
{ "type": "Resize", "width": 224, "height": 224, "interpolation": "Bilinear" }
```

Interpolation methods: `"Nearest"`, `"Bilinear"` (default), `"Bicubic"`.

### `CenterCrop`

Center-crops image to target dimensions.

```json
{ "type": "CenterCrop", "width": 224, "height": 224 }
```

### `Reshape`

Reshapes tensor to target dimensions.

```json
{ "type": "Reshape", "shape": [1, 1, 28, 28] }
```

---

## Postprocessing Steps

Postprocessing transforms model output into the final result.

### `CTCDecode`

CTC (Connectionist Temporal Classification) decoding for ASR models.

```json
{ "type": "CTCDecode", "vocab_file": "vocab.json", "blank_index": 0 }
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vocab_file` | string | — | Path to vocabulary file |
| `blank_index` | integer | `0` | Blank token index |

**Input**: Logits tensor → **Output**: `Envelope::Text(decoded_string)`

### `TTSAudioEncode`

Converts TTS float32 waveform output to PCM audio bytes.

```json
{ "type": "TTSAudioEncode", "sample_rate": 24000, "apply_postprocessing": true, "trim_trailing_silence": false }
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_rate` | integer | — | Output sample rate in Hz |
| `apply_postprocessing` | boolean | `true` | Apply normalization and silence trimming |
| `trim_trailing_silence` | boolean | `false` | Trim trailing near-silence from waveform |

**Input**: Float32 waveform → **Output**: `Envelope::Audio(pcm_bytes)`

### `WhisperDecode`

Decodes Whisper token IDs to text using a HuggingFace tokenizer.

```json
{ "type": "WhisperDecode", "tokenizer_file": "tokenizer.json" }
```

### `BPEDecode`

Decodes BPE tokens to text.

```json
{ "type": "BPEDecode", "vocab_file": "vocab.json" }
```

### `Argmax`

Returns the index of the maximum value (class prediction).

```json
{ "type": "Argmax", "dim": null }
```

### `Softmax`

Applies softmax to get probabilities.

```json
{ "type": "Softmax", "dim": 1 }
```

### `TopK`

Returns top-K predictions with scores.

```json
{ "type": "TopK", "k": 5, "dim": null }
```

### `Threshold`

Applies threshold to convert probabilities to binary predictions.

```json
{ "type": "Threshold", "threshold": 0.5, "return_indices": false }
```

### `TemperatureSample`

Applies temperature sampling for token generation.

```json
{ "type": "TemperatureSample", "temperature": 0.7, "top_k": 40, "top_p": 0.9 }
```

### `Denormalize`

Inverse of `Normalize`: `x * std + mean`.

```json
{ "type": "Denormalize", "mean": [0.5], "std": [0.5] }
```

### `MeanPool`

Mean pooling over token embeddings to produce a sentence embedding.

```json
{ "type": "MeanPool", "dim": 1 }
```

---

## Voice Configuration

TTS models can include a `voices` section to define available voices.

```json
{
  "voices": {
    "format": "embedded",
    "file": "voices.bin",
    "loader": "binary_f32_256",
    "default": "af_heart",
    "selection_strategy": "FixedIndex",
    "catalog": [
      {
        "id": "af_heart",
        "name": "Heart",
        "gender": "female",
        "language": "en-US",
        "style": "neutral",
        "index": 0
      }
    ]
  }
}
```

### Voice Formats

| Format | Description | Example Models |
|--------|-------------|---------------|
| `embedded` | All voices in a single binary file | Kokoro, KittenTTS |
| `per_model` | Each voice is a separate model file | Piper |
| `cloning` | Voice cloning from reference audio | (future) |

### Voice Loaders (for `embedded` format)

| Loader | Description |
|--------|-------------|
| `binary_f32_256` | Contiguous f32 arrays, 256 dims (1024 bytes/voice) |
| `numpy_npz` | NumPy .npz archive |
| `json_base64` | JSON with base64-encoded embeddings (future) |

### Voice Selection Strategies

| Strategy | Description |
|----------|-------------|
| `FixedIndex` | Select by catalog index (default) |
| `TokenLength` | Select from voicepack by phoneme token count (Kokoro) |

---

## Metadata Conventions

The `metadata` field is freeform, but these keys are conventional:

| Key | Type | Description |
|-----|------|-------------|
| `task` | string | Model task: `"text-to-speech"`, `"speech-recognition"`, `"text-generation"`, `"image_classification"`, `"embedding"` |
| `architecture` | string | Model architecture (e.g., `"qwen35"`, `"whisper"`) |
| `backend` | string | Inference backend (e.g., `"llamacpp"`, `"onnxruntime"`) |
| `parameters` | integer | Parameter count |
| `license` | string | Model license (e.g., `"Apache-2.0"`) |
| `sample_rate` | integer | Audio sample rate (audio models) |
| `family` | string | Model family or organization |
| `source` | string | Source URL (e.g., HuggingFace repo) |
| `ram_minimum_mb` | integer | Minimum RAM in MB |
| `ram_recommended_mb` | integer | Recommended RAM in MB |

---

## Complete Examples

### ONNX Classification (MNIST)

```json
{
  "model_id": "mnist-digit-recognition",
  "version": "12",
  "description": "MNIST handwritten digit recognition. Input: [1,1,28,28] grayscale. Output: [1,10] probabilities.",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    { "type": "Reshape", "shape": [1, 1, 28, 28] },
    { "type": "Normalize", "mean": [0.0], "std": [255.0] }
  ],
  "postprocessing": [
    { "type": "Softmax", "dim": 1 }
  ],
  "files": ["model.onnx"],
  "metadata": {
    "task": "image_classification",
    "num_classes": 10
  }
}
```

### ONNX TTS with Phonemize (Kokoro)

```json
{
  "model_id": "kokoro-82m",
  "version": "1.0",
  "description": "Kokoro 82M - High-quality TTS with 24 voices",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    {
      "type": "Phonemize",
      "tokens_file": "tokens.txt",
      "backend": "MisakiDictionary",
      "add_padding": true,
      "normalize_text": true
    }
  ],
  "postprocessing": [
    { "type": "TTSAudioEncode", "sample_rate": 24000, "apply_postprocessing": true }
  ],
  "files": ["model.onnx", "voices.bin", "tokens.txt", "misaki/us_gold.json"],
  "voices": {
    "format": "embedded",
    "file": "voices.bin",
    "loader": "binary_f32_256",
    "default": "af_heart",
    "selection_strategy": "TokenLength",
    "catalog": [
      { "id": "af_heart", "name": "Heart", "gender": "female", "language": "en-US", "index": 0 }
    ]
  },
  "metadata": {
    "task": "text-to-speech",
    "parameters": 82000000,
    "sample_rate": 24000
  }
}
```

### GGUF LLM (Qwen)

```json
{
  "model_id": "qwen3.5-0.8b",
  "version": "1.0",
  "description": "Qwen 3.5 0.8B - Lightweight LLM for on-device text generation",
  "execution_template": {
    "type": "Gguf",
    "model_file": "Qwen3.5-0.8B-Q4_K_M.gguf",
    "context_length": 4096
  },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["Qwen3.5-0.8B-Q4_K_M.gguf"],
  "metadata": {
    "task": "text-generation",
    "architecture": "qwen35",
    "backend": "llamacpp",
    "parameters": 800000000,
    "license": "Apache-2.0",
    "ram_minimum_mb": 512,
    "ram_recommended_mb": 1024
  }
}
```

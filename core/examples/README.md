# xybrid-core Examples

Reference implementations demonstrating correct usage patterns for the xybrid runtime.

## What These Are

These are **examples**, not tests. They demonstrate:
- How to use the `TemplateExecutor` API correctly
- How to structure inference pipelines for different model types
- Real-world usage patterns for ASR, TTS, embeddings, and LLMs

## Running Examples

Examples require models to be downloaded first:

```bash
# Download models (from repo root)
./integration-tests/download.sh wav2vec2-base-960h  # For ASR examples
./integration-tests/download.sh kitten-tts kokoro-82m  # For TTS examples
./integration-tests/download.sh all-minilm distilbert  # For embedding examples

# Run an example
cargo run --example wav2vec2_transcription -p xybrid-core path/to/audio.wav
cargo run --example tts_kokoro_misaki -p xybrid-core
```

## Examples by Category

### Speech Recognition (ASR)

| Example | Model | Description |
|---------|-------|-------------|
| `wav2vec2_transcription` | wav2vec2-base-960h | ONNX-based ASR with CTC decoding |
| `asr_whisper` | whisper-tiny | Whisper ASR via ONNX |
| `candle_whisper` | whisper-tiny | Whisper ASR via Candle (SafeTensors) |
| `candle_whisper_bundle` | whisper-tiny | Bundle format for Candle models |
| `streaming_asr` | - | Demonstrates streaming ASR patterns |

### Text-to-Speech (TTS)

| Example | Model | Description |
|---------|-------|-------------|
| `tts_kitten` | kitten-tts | Lightweight TTS with CMU phonemizer |
| `tts_kokoro` | kokoro-82m | High-quality TTS with espeak-ng |
| `tts_kokoro_misaki` | kokoro-82m | TTS with Misaki dictionary (no espeak) |

### Text Embeddings

| Example | Model | Description |
|---------|-------|-------------|
| `sentence_embedding` | all-minilm | Generate sentence embeddings |
| `sentence_similarity` | distilbert | Compare semantic similarity |

### Image Classification

| Example | Model | Description |
|---------|-------|-------------|
| `mnist_simple_mode` | mnist | Digit classification demo |
| `mobilenetv2_imagenet` | mobilenet | Lightweight ImageNet classifier |
| `resnet50_imagenet` | resnet50 | Full ImageNet classifier |

### Local LLMs

| Example | Model | Description |
|---------|-------|-------------|
| `local_llm_qwen` | qwen2.5-0.5b | Local GGUF LLM inference |
| `local_llm_gemma3` | gemma3 | Gemma 3 LLM |
| `llama_cpp_test` | - | llama.cpp integration |

### Utilities & Demos

| Example | Description |
|---------|-------------|
| `device_metrics` | System metrics collection |
| `metadata_validation` | Validate model_metadata.json files |
| `voice_assistant_demo` | Full ASR → LLM → TTS pipeline |
| `authority_demo` | Permission/authority patterns |
| `cloud_llm_demo` | Cloud LLM integration |
| `vad_demo` | Voice Activity Detection |

## Integration Tests

For automated testing, see `integration-tests/tests/`:

```bash
# Run all integration tests
cargo test -p integration-tests

# Run specific test suite
cargo test -p integration-tests --test tts_integration
cargo test -p integration-tests --test asr_integration
```

## Key Pattern

All examples follow the same pattern:

```rust
use xybrid_core::testing::model_fixtures;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Get model path
    let model_dir = model_fixtures::require_model("model-name");

    // 2. Load metadata
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("model_metadata.json"))?
    )?;

    // 3. Create executor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // 4. Create input envelope
    let input = Envelope {
        kind: EnvelopeKind::Text("Hello world".into()),
        metadata: HashMap::new(),
    };

    // 5. Execute
    let output = executor.execute(&metadata, &input)?;

    Ok(())
}
```

# Xybrid Integration Tests

Integration tests for xybrid-core that require model files.

## Directory Structure

```
integration-tests/
├── fixtures/
│   ├── input/              # Small test inputs (checked in)
│   │   ├── sample.wav      # Short audio sample
│   │   └── sample.txt      # Text samples
│   ├── models/             # Model files (downloaded, gitignored)
│   │   ├── models.json     # Model manifest
│   │   ├── wav2vec2-base-960h/
│   │   ├── kitten-tts/
│   │   ├── kokoro-82m/
│   │   └── whisper-tiny/
│   └── pipelines/          # Pipeline YAML configs (checked in)
│       └── tts_pipeline.yaml
├── tests/                  # Integration test files
├── src/
│   ├── lib.rs
│   └── fixtures.rs         # Fixture path helpers
├── download.sh             # Model download script
└── Cargo.toml
```

## Quick Start

```bash
# From the xybrid repo root:

# 1. Download all test models
./integration-tests/download.sh

# 2. Download specific model
./integration-tests/download.sh kitten-tts

# 3. List available models
./integration-tests/download.sh --list

# 4. Check which models are present
./integration-tests/download.sh --check

# 5. Run integration tests
cargo test -p integration-tests
```

## Running Tests

### Unit tests (no models needed)
```bash
cargo test -p xybrid-core
```

### Integration tests (models required)
```bash
# Download models first
./integration-tests/download.sh

# Run all integration tests
cargo test -p integration-tests

# Run specific test file
cargo test -p integration-tests --test tts_integration
cargo test -p integration-tests --test local_llm --features local-llm

# Run specific test function
cargo test -p integration-tests --test tts_integration test_phonemize_text
```

## Available Tests

| Test File | Description | Required Models |
|-----------|-------------|-----------------|
| `integration_check.rs` | Verifies downloaded models exist | wav2vec2, kitten-tts |
| `asr_integration.rs` | ASR pipeline: audio decode, inference, CTC decode | wav2vec2, whisper-tiny |
| `tts_integration.rs` | TTS pipeline: phonemizer, tokens, voice embeddings | kitten-tts, kokoro-82m |
| `local_llm.rs` | Local LLM inference (GGUF) | qwen2.5-0.5b-instruct |

### ASR Integration Tests

```bash
# Download required models
./integration-tests/download.sh wav2vec2-base-960h whisper-tiny

# Run ASR tests
cargo test -p integration-tests --test asr_integration
```

Tests included:
- `test_wav2vec2_model_metadata` - Validate ASR model metadata
- `test_wav2vec2_model_files_exist` - Check required files present
- `test_wav2vec2_vocab_loading` - Load and validate vocab.json
- `test_wav2vec2_inference_with_audio` - End-to-end ASR inference
- `test_whisper_model_available` - Check whisper-tiny availability

### TTS Integration Tests

```bash
# Download required models
./integration-tests/download.sh kitten-tts kokoro-82m

# Run TTS tests
cargo test -p integration-tests --test tts_integration
```

Tests included:
- `test_load_tokens_file` - Load and validate tokens.txt
- `test_phonemize_text` - CMU dictionary phonemization
- `test_phonemes_to_token_ids` - Token ID conversion
- `test_load_voice_embeddings_bin` - voices.bin parsing
- `test_model_metadata_loading` - model_metadata.json validation
- `test_voice_embedding_loader_npz` - NPZ voice file support (kokoro)

## Model Sources

Models are hosted on HuggingFace under the `xybrid-ai` organization:
- https://huggingface.co/xybrid-ai/wav2vec2-base-960h
- https://huggingface.co/xybrid-ai/kitten-tts
- https://huggingface.co/xybrid-ai/kokoro-82m
- https://huggingface.co/xybrid-ai/whisper-tiny

## CI Integration

Integration tests are run separately from unit tests in CI:

```yaml
# .github/workflows/integration.yml
jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download models
        run: ./integration-tests/download.sh
      - name: Run integration tests
        run: cargo test -p integration-tests
```

## Migration from core/tests/

Tests that require model fixtures are being migrated from `core/tests/` to this crate.
The legacy `test_models/` directory is deprecated in favor of `integration-tests/fixtures/models/`.

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `core/tests/tts_integration.rs` | `integration-tests/tests/tts_integration.rs` | Migrated |
| `core/tests/system_validation.rs` | (stays in core - uses mocks) | No change |
| `core/tests/pipeline_yaml_integration.rs` | (stays in core - uses mocks) | No change |

## Adding New Models

1. Add model entry to `fixtures/models/models.json`
2. Update `download.sh` if needed
3. Create test in `tests/`
4. Update this README

## Fixtures

### Small fixtures (checked in)
- `fixtures/input/` - Sample audio, text files (<100KB each)
- `fixtures/pipelines/` - Pipeline YAML configurations

### Large fixtures (gitignored)
- `fixtures/models/*/` - Downloaded model directories

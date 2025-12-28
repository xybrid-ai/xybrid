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

# Run specific test
cargo test -p integration-tests test_tts_inference
```

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

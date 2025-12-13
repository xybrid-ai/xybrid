# Xybrid

**Hybrid cloud-edge ML inference orchestrator** - Run ML models on-device or in the cloud with intelligent routing.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Xybrid enables running ML models (ASR, TTS, LLMs, embeddings) either locally on-device or via cloud APIs, with automatic routing based on device capabilities, battery status, and connectivity.

## Packages

| Crate | Description |
|-------|-------------|
| `xybrid-core` | Core runtime library - model execution, preprocessing, postprocessing |
| `xybrid-cli` | Command-line interface for running models and managing bundles |
| `xybrid-sdk` | High-level SDK with caching, pipelines, and telemetry |
| `xybrid-macros` | Procedural macros for the SDK |
| `registry` | Bundle creation and registry server tools |

## Quick Start

```bash
# Install the CLI
cargo install --path cli

# Run a model
xybrid run --config pipeline.yaml

# List available bundles
xybrid list
```

## Model Execution

All model execution goes through `model_metadata.json` which defines preprocessing, inference, and postprocessing:

```rust
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};

let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string("model_metadata.json")?)?;
let mut executor = TemplateExecutor::with_base_path("./model_dir");
let input = Envelope { kind: EnvelopeKind::Audio(audio_bytes), metadata: HashMap::new() };
let output = executor.execute(&metadata, &input)?;
```

## Supported Model Types

- **ASR**: Wav2Vec2, Whisper (ONNX and Candle backends)
- **TTS**: KittenTTS, Kokoro
- **Embeddings**: Sentence transformers, DistilBERT
- **Vision**: ResNet, MobileNet, MNIST

## Documentation

See the [docs/](docs/) directory for detailed documentation.

## Related Projects

- [xybrid-flutter](https://github.com/xybrid-ai/xybrid-flutter) - Flutter plugin for mobile apps

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

### Open Core Model

Xybrid uses an **open-core licensing model**:

- **Open Source (Apache-2.0)**: Core runtime, SDK, CLI, and self-hosted registry
- **Commercial**: Cloud services, enterprise features, and managed offerings

This means you can freely use, modify, and distribute the open source components for any purpose, including commercial applications.

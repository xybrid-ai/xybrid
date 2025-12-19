<p align="center">
  <img src="./docs/logo.jpg" alt="Xybrid Logo" width="200"/>
</p>

<h1 align="center">Xybrid</h1>

<p align="center">
  <strong>Open Source Framework for Hybrid Cloud-Edge ML Inference</strong>
</p>

<p align="center">
  Run ML models on-device or in the cloud with intelligent routing—no infrastructure complexity.
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/xybrid-ai/xybrid/actions"><img src="https://img.shields.io/github/actions/workflow/status/xybrid-ai/xybrid/ci.yml?branch=main" alt="Build Status"></a>
  <a href="https://crates.io/crates/xybrid-core"><img src="https://img.shields.io/crates/v/xybrid-core.svg" alt="Crates.io"></a>
</p>

---

## Why Xybrid?

Building ML-powered apps shouldn't require choosing between **on-device privacy** and **cloud scalability**. Xybrid gives you both:

- **🔒 Privacy-first**: Run models locally on user devices—no data leaves the device
- **☁️ Cloud fallback**: Seamlessly route to cloud APIs when device resources are limited
- **🔋 Intelligent routing**: Automatic decisions based on battery, connectivity, and device capabilities
- **📦 One codebase**: Same API whether running on-device or in the cloud

## What You Can Build

| Use Case | Models | Example |
|----------|--------|---------|
| **Voice Assistants** | ASR + LLM + TTS | On-device speech recognition with cloud LLM fallback |
| **Real-time Transcription** | Whisper, Wav2Vec2 | Offline-capable meeting transcription |
| **Text-to-Speech** | Kokoro, KittenTTS | Natural voice synthesis that works offline |
| **Embeddings** | Sentence Transformers | Local semantic search without API costs |
| **Vision** | ResNet, MobileNet | On-device image classification |

## How It Works

Xybrid abstracts ML inference into three components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your App                                │
├─────────────────────────────────────────────────────────────────┤
│                      Xybrid SDK                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Pipeline   │  │   Cache     │  │   Intelligent Router    │  │
│  │  Builder    │  │  Manager    │  │  (battery/connectivity) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Xybrid Core                                │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │   On-Device (ONNX)  │    │      Cloud APIs                 │ │
│  │   • Whisper         │    │      • OpenAI                   │ │
│  │   • Wav2Vec2        │    │      • Anthropic                │ │
│  │   • Kokoro TTS      │    │      • ElevenLabs               │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Define once, run anywhere:**

```rust
use xybrid_sdk::prelude::*;

// Define your pipeline
let pipeline = Pipeline::builder()
    .add_stage("transcribe", asr_model("whisper-tiny"))
    .add_stage("respond", llm_model("claude-3-haiku"))
    .add_stage("speak", tts_model("kokoro-82m"))
    .build()?;

// Run it—Xybrid handles routing automatically
let audio_response = pipeline.run(audio_input).await?;
```

## Getting Started

### Install the CLI

```bash
# macOS (Apple Silicon)
curl -fsSL https://xybrid.dev/install.sh | sh

# From source
cargo install xybrid-cli
```

### Run Your First Model

```bash
# Download a model bundle
xybrid pull whisper-tiny

# Transcribe audio
xybrid run whisper-tiny --input recording.wav
```

### Use in Your App

```toml
# Cargo.toml
[dependencies]
xybrid-core = "0.1"
```

```rust
use xybrid_core::prelude::*;

let result = XybridRuntime::new()
    .load_model("whisper-tiny")?
    .transcribe(&audio_bytes)?;

println!("Transcription: {}", result.text);
```

## Platform Support

| Platform | Runtime | Status |
|----------|---------|--------|
| macOS (ARM) | ONNX Runtime | ✅ Stable |
| macOS (x86) | ONNX Runtime | ✅ Stable |
| Linux (x86) | ONNX Runtime | ✅ Stable |
| iOS | Core ML + ONNX | 🚧 In Progress |
| Android | NNAPI + ONNX | 🚧 In Progress |
| Windows | ONNX Runtime | 🚧 In Progress |

## Packages

| Crate | Description |
|-------|-------------|
| [`xybrid-core`](./core) | Core runtime—model execution, preprocessing, postprocessing |
| [`xybrid-cli`](./cli) | Command-line interface for running models |
| [`xybrid-sdk`](./sdk) | High-level SDK with pipelines, caching, and telemetry |
| [`registry`](./registry) | Bundle creation and registry server |

## Comparison

| Tool | What it does | How Xybrid differs |
|------|--------------|-------------------|
| **ONNX Runtime** | Low-level inference engine | Xybrid adds preprocessing, routing, and cloud fallback |
| **MLX** | Apple-optimized ML | Xybrid is cross-platform with cloud hybrid support |
| **Triton Server** | Cloud inference serving | Xybrid focuses on edge-first with cloud fallback |
| **TensorFlow Lite** | Mobile ML runtime | Xybrid provides unified API across edge and cloud |

## Community

- 📖 [Documentation](https://docs.xybrid.dev)
- 💬 [Discord](https://discord.gg/xybrid)
- 🐛 [Issues](https://github.com/xybrid-ai/xybrid/issues)
- 🗺️ [Roadmap](./ROADMAP.md)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

```bash
# Clone and build
git clone https://github.com/xybrid-ai/xybrid.git
cd xybrid
cargo build

# Run tests
cargo test
```

## License

Apache License 2.0 - see [LICENSE](./LICENSE) for details.

**Open Core Model**: The core runtime, SDK, CLI, and self-hosted registry are fully open source. Commercial cloud services and enterprise features are available separately.

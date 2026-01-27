# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-27

First production release of xybrid - a hybrid cloud-edge ML inference orchestrator.

### Added

#### CLI
- `xybrid models list` - List models from registry
- `xybrid models search <query>` - Search models
- `xybrid models info <id>` - Show model details
- `xybrid plan <pipeline.yaml>` - Show execution plan
- `xybrid fetch --model <id>` - Download model with progress
- `xybrid fetch <pipeline.yaml>` - Pre-download pipeline models
- `xybrid cache list` - Show cached models
- `xybrid cache status` - Cache statistics
- `xybrid cache clear` - Clear cache
- `xybrid run <pipeline.yaml>` - Execute pipeline
- `xybrid run --model <id>` - Direct model execution from registry
- `xybrid run --voice <index>` - TTS voice selection
- `xybrid run --output <file>` - Save output (WAV/text/JSON)
- `xybrid run --trace` - Execute with tracing

#### Core Runtime
- ONNX Runtime execution with preprocessing/postprocessing
- Whisper ASR with Metal acceleration (macOS/iOS)
- Metadata-driven model execution
- Policy-based orchestration with offline-first routing
- CoreML/ANE acceleration for Apple devices

#### LLM Inference
- Local LLM execution for GGUF models
- Desktop: CPU, Metal (macOS), CUDA (Linux/Windows)
- Android: Optimized for ARM devices
- Runtime backend selection via model metadata

#### SDK
- `PipelineRef::from_yaml()` - Instant YAML parsing
- `Pipeline::load_models()` - Model preloading with progress
- `Pipeline::run()` - Execute inference
- `RegistryClient` - Model discovery, resolution, and caching
- Telemetry with batching

#### Preprocessing
- `AudioDecode` - WAV bytes to float samples
- `Phonemize` - Text to phoneme tokens
- `Tokenize` - Text tokenization

#### Postprocessing
- `CTCDecode` - Logits to text transcription
- `TTSAudioEncode` - Waveform to PCM audio bytes
- `ArgMax` - Classification output

### Models Supported
- **Kokoro-82M** (TTS) - 24 voices
- **KittenTTS-nano** (TTS) - Lightweight
- **Whisper-tiny** (ASR) - Real-time capable
- **Wav2Vec2-base-960h** (ASR) - English
- **all-MiniLM-L6-v2** (Embeddings) - 384-dim vectors
- **MobileNetV2** (Vision) - 6.8x ANE speedup
- **Qwen 2.5 0.5B** (LLM) - On-device chat

### Platform Support

| Platform | ASR/TTS/Vision | LLM | Hardware Acceleration |
|----------|----------------|-----|----------------------|
| macOS arm64 | ✅ | ✅ | CoreML ANE, Metal GPU |
| macOS x86_64 | ✅ | ✅ | CoreML GPU |
| Linux x86_64 | ✅ | ✅ | CUDA |
| Windows x86_64 | ✅ | ✅ | CUDA |
| Android arm64 | ✅ | ✅ | CPU (NNAPI planned) |
| iOS arm64 | ✅ | Planned | CoreML ANE, Metal GPU |

## [Unreleased]

### Planned
- Android NNAPI execution provider
- MLX runtime for Apple Silicon
- Voice cloning support
- Streaming TTS

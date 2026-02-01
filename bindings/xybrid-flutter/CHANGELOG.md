## 0.0.20

### Platform Configuration

* **iOS**: Configured for Candle-only (Metal acceleration for Whisper ASR)
  - iOS 13.0+ deployment target
  - Metal, MetalPerformanceShaders, Accelerate frameworks linked
  - i386 architecture excluded
  - ONNX Runtime requires manual xcframework bundling (see PLAN-IOS-ONNX.md)

* **Android**: Full ONNX Runtime support via load-dynamic
  - 64-bit only: arm64-v8a, x86_64 (32-bit excluded)
  - Bundled libonnxruntime.so from Microsoft AAR v1.23.2
  - minSdkVersion 21, compileSdkVersion 34
  - Rust builds target API 28+ (required for aws-lc-sys getentropy)
  - FP16 target feature enabled for gemm-f16/Candle compatibility
  - TLS via rustls (no OpenSSL dependency)

* **macOS**: ONNX Runtime + Candle (Metal acceleration)
  - download-binaries for ONNX Runtime
  - candle-metal for Whisper ASR

* **Linux/Windows**: ONNX Runtime via download-binaries

### Breaking Changes

* `xybrid_core::llm` module renamed to `xybrid_core::cloud`
* `PipelineLoader` renamed to `PipelineRef`
* `XybridPipeline` renamed to `Pipeline`
* Direct TTS API removed (use pipeline execution instead)

### Model Support

| Model | macOS | iOS | Android | Linux/Windows |
|-------|-------|-----|---------|---------------|
| Whisper (Candle) | ✅ Metal | ✅ Metal | ❌ | ✅ CPU |
| Wav2Vec2 (ONNX) | ✅ | ❌ | ✅ | ✅ |
| Kokoro-82M (ONNX) | ✅ | ❌ | ✅ | ✅ |
| KittenTTS (ONNX) | ✅ | ❌ | ✅ | ✅ |

## 0.0.1

* Initial release with pipeline execution, audio processing, telemetry, and device capabilities.

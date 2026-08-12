# xybrid

Hybrid cloud-edge AI inference SDK for Rust.

This is the umbrella crate. It re-exports [`xybrid-sdk`](https://crates.io/crates/xybrid-sdk) so applications can depend on a single name:

```toml
[dependencies]
xybrid = "0.2"
```

## Features

Feature flags forward 1:1 to `xybrid-sdk`. Pick the preset that matches your target platform:

- `platform-macos` — CoreML + Metal + llama.cpp (vision) + whisper.cpp ASR
- `platform-ios` — CoreML + Metal + llama.cpp (vision) + whisper.cpp ASR
- `platform-android` — Dynamic ORT + llama.cpp (vision) + whisper.cpp ASR
- `platform-desktop` — CPU ORT + llama.cpp (vision) + whisper.cpp ASR (Linux/Windows)

For finer-grained control, enable individual features such as `ort-download`, `candle`, `llm-llamacpp`, `huggingface`, etc. On Linux, set `XYBRID_LLAMA_CPP_VULKAN=1` when building `platform-desktop` to opt into Vulkan acceleration.

## License

Apache-2.0

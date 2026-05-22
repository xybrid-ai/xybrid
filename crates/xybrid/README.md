# xybrid

Hybrid cloud-edge AI inference SDK for Rust.

This is the umbrella crate. It re-exports [`xybrid-sdk`](https://crates.io/crates/xybrid-sdk) so applications can depend on a single name:

```toml
[dependencies]
xybrid = "0.1"
```

## Features

Feature flags forward 1:1 to `xybrid-sdk`. Pick the preset that matches your target platform:

- `platform-macos` — CoreML + Metal + llama.cpp
- `platform-ios` — CoreML + Metal + llama.cpp
- `platform-android` — Dynamic ORT + Candle + llama.cpp
- `platform-desktop` — CPU ORT + llama.cpp (Linux/Windows)

For finer-grained control, enable individual features such as `ort-download`, `candle`, `llm-llamacpp`, `huggingface`, etc.

## License

Apache-2.0

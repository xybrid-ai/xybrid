# Feature 矩阵

本文档为 xybrid crate 层级中的所有 feature 标志、平台预设及有效组合提供全面参考。

## 目录

1. [xybrid-core Feature 标志](#xybrid-core-feature-标志)
2. [xybrid-sdk Feature 标志](#xybrid-sdk-feature-标志)
3. [xybrid-cli Feature 标志](#xybrid-cli-feature-标志)
4. [平台预设](#平台预设)
5. [受 Feature 控制的类型与模块](#受-feature-控制的类型与模块)
6. [无效的 Feature 组合](#无效的-feature-组合)
7. [发布门禁](#发布门禁)
8. [ORT 加载策略](#ort-加载策略)
9. [xtask 命令](#xtask-命令)
10. [构建架构](#构建架构)

---

## xybrid-core Feature 标志

| Feature | 说明 | 启用 |
|---------|-------------|---------|
| **default** | 默认特性 | `ort-download`（通过 `llm-llamacpp` 或平台预设启用 llama.cpp） |
| **ort-download** | 下载预编译的 ONNX Runtime 二进制文件 | `ort/download-binaries`、`ort/tls-native` |
| **ort-dynamic** | 在运行时加载 ONNX Runtime 的 .so | `ort/load-dynamic` |
| **ort-coreml** | Apple Neural Engine 加速 | `ort/coreml` |
| **candle** | 纯 Rust ML 框架 — SafeTensors Whisper 路径。**仅可显式启用**，自 whisper.cpp 取代其 ASR 角色后已从所有平台预设中移除 | `candle-core`、`candle-nn`、`candle-transformers`、`safetensors`、`byteorder`、`num-traits` |
| **asr-whispercpp** | whisper.cpp 语音识别，运行在 llama.cpp 已链接的 ggml 上。需要 `llm-llamacpp` —— ggml 正是由它提供 | `xybrid-whisper`、`xybrid-whisper-sys` |
| **candle-hub** | Candle + HuggingFace Hub 下载支持 | `candle`、`hf-hub`（需要 OpenSSL — **不适用于 Android**） |
| **candle-metal** | 带 Metal 显卡加速的 Candle | `candle`、`candle-core/metal`、`candle-nn/metal` |
| **candle-cuda** | 带 CUDA 显卡加速的 Candle | `candle`、`candle-core/cuda` |
| **llm-mistral** | mistral.rs LLM 后端（CPU） | `mistralrs` |
| **llm-mistral-metal** | 带 Metal 加速的 mistral.rs | `llm-mistral`、`mistralrs/metal` |
| **llm-mistral-cuda** | 带 CUDA 加速的 mistral.rs | `llm-mistral`、`mistralrs/cuda` |
| **vision** | 图像 Envelope 原语与图像预处理 | *（无额外依赖；使用始终存在的 `image` crate）* |
| **llm-llamacpp** | llama.cpp 后端（cmake 构建 + 链接） | `llama-cpp-sys/bindings`、`xybrid-llama/bindings` |
| **llm-llamacpp-vision** | 支持 `mmproj` / `mtmd` 的 llama.cpp VLM 路径 | `llm-llamacpp`、`vision`、`llama-cpp-sys/vision`、`xybrid-llama/vision` |

### 说明

- 启用 **`llm-llamacpp`** 会激活 `llama-cpp-sys/bindings`（llama.cpp 的 cmake
  构建 + `wrapper.cpp` shim）和 `xybrid-llama/bindings`（安全的 RAII 包装）。
  它**默认不启用** — 需要 cmake、C++ 工具链以及一份 llama.cpp 源码克隆。
  `xybrid-sdk` 上的全部四个 `platform-*` 预设都依赖它。不启用该特性的构建
  不会暴露 llama.cpp 后端类型。
- 三层 crate 结构：
  `llama-cpp-sys`（原始 FFI + cmake 构建）→ `xybrid-llama`（安全包装、
  类型化错误）→ `xybrid-core::runtime_adapter::llama_cpp`（轻量适配器）。
- 单独启用 `vision` 可获得图像 Envelope 与图像预处理。本地 llama.cpp
  VLM 生成需要 `llm-llamacpp-vision`，它将 `vision` 与 llama.cpp 后端组合，
  并链接随附的 `mtmd` 辅助库。

---

## xybrid-sdk Feature 标志

| Feature | 说明 | 转发到 xybrid-core |
|---------|-------------|-------------------------|
| **default** | 无默认特性 | *（无）* |
| **platform-android** | Android 预设 | `ort-dynamic`、`llm-llamacpp-vision`、`asr-whispercpp` |
| **platform-ios** | iOS 预设 | `ort-download`、`ort-coreml`、`llm-llamacpp-vision`、`asr-whispercpp` |
| **platform-macos** | macOS 预设 | `ort-download`、`ort-coreml`、`llm-llamacpp-vision`、`asr-whispercpp` |
| **platform-desktop** | 桌面（Linux/Windows）预设 | `ort-download`、`llm-llamacpp-vision`、`asr-whispercpp` |
| **ort-download** | 转发到 core | `xybrid-core/ort-download` |
| **ort-dynamic** | 转发到 core | `xybrid-core/ort-dynamic` |
| **ort-coreml** | 转发到 core | `xybrid-core/ort-coreml` |
| **candle** | 转发到 core | `xybrid-core/candle` |
| **candle-hub** | 转发到 core | `xybrid-core/candle-hub` |
| **candle-metal** | 转发到 core | `xybrid-core/candle-metal` |
| **candle-cuda** | 转发到 core | `xybrid-core/candle-cuda` |
| **llm-mistral** | 转发到 core | `xybrid-core/llm-mistral` |
| **llm-mistral-metal** | 转发到 core | `xybrid-core/llm-mistral-metal` |
| **llm-mistral-cuda** | 转发到 core | `xybrid-core/llm-mistral-cuda` |
| **llm-llamacpp** | 转发到 core | `xybrid-core/llm-llamacpp` |
| **vision** | 转发到 core | `xybrid-core/vision` |
| **llm-llamacpp-vision** | 转发到 core 的 VLM 路径 | `xybrid-core/llm-llamacpp-vision`、`llm-llamacpp`、`vision` |

---

## xybrid-cli Feature 标志

| Feature | 说明 | 启用 |
|---------|-------------|---------|
| **default** | CLI 默认支持带图像的输入，使得 `xybrid run --input-image` 在 `cargo install xybrid-cli` 的构建中无需额外标志即可工作 | `vision` |
| **huggingface** | 为 `xybrid run --huggingface` 提供直接的 HuggingFace 加载 | `xybrid-sdk/huggingface` |
| **onnx-inspect** | 为 `xybrid init` 提供 ONNX 元数据检查 | `xybrid-sdk/onnx-inspect` |
| **vision** | 为 VLM 对话提供 `xybrid run --input-image` 和 REPL `/image` 的 Envelope 构建 | `xybrid-core/vision`、`xybrid-sdk/vision` |
| **llm-llamacpp-vision** | llama.cpp VLM 运行时以及 CLI 图像输入支持 | `llm-llamacpp`、`vision`、`xybrid-sdk/llm-llamacpp-vision` |
| **platform-android** | Android 发布预设 — 转发到 `xybrid-sdk/platform-android` | `ort-dynamic`、`llm-llamacpp-vision`、`asr-whispercpp`、`llm-llamacpp`、`huggingface` |
| **platform-ios** | iOS 发布预设 — 转发到 `xybrid-sdk/platform-ios` | `ort-download`、`ort-coreml`、`llm-llamacpp-vision`、`asr-whispercpp`、`llm-llamacpp`、`huggingface` |
| **platform-macos** | macOS 发布预设 — 转发到 `xybrid-sdk/platform-macos` | `ort-download`、`ort-coreml`、`llm-llamacpp-vision`、`asr-whispercpp`、`llm-llamacpp`、`huggingface` |
| **platform-desktop** | Linux/Windows 发布预设 — 转发到 `xybrid-sdk/platform-desktop` | `ort-download`、`llm-llamacpp-vision`、`asr-whispercpp`、`llm-llamacpp`、`huggingface` |

---

## 平台预设

平台预设是平台专属特性组合的**单一事实来源**。它们定义在 `xybrid-sdk/Cargo.toml` 中，并通过 crate 层级向下转发。

四个平台预设均已包含视觉语言 llama.cpp 路径（`llm-llamacpp-vision`）与 whisper.cpp 语音识别（`asr-whispercpp`，在 Android `.so` 代理上约 0.2 MiB，已 strip）。VLM 与 ASR 开箱即用，无需额外组合。

Candle **不在**任何预设中。在同一 Android 代理上它约占 1.3 MiB（Apple 形态因额外链接 `candle-metal` 约占 1.9 MiB），即其替代方案的 6.5–9.5 倍；在 Pixel 8 上首个部分结果需 9871 ms，而 whisper.cpp 为 2724 ms。`candle*` 特性仍然保留且可编译，需要 SafeTensors 路径的用户可显式启用 —— 只是不再默认开启。

| 预设 | 目标平台 | 启用的 Core 特性 | VLM 默认 | ASR 默认 | 理由 |
|--------|-----------------|----------------------|-------------|-------------|-----------|
| **platform-android** | Android（所有 ABI） | `ort-dynamic`、`llm-llamacpp-vision`、`asr-whispercpp` | 开启 | whisper.cpp | 用于 AAR 分发的动态 ORT 加载；Whisper ASR 由 whisper.cpp 承担，复用 llama.cpp 已链接的 ggml（strip 后约 0.2 MiB）；llama.cpp 具备运行时 SIMD 检测；mistral.rs 在不具备 ARMv8.2-A FP16 的设备上会导致 SIGILL |
| **platform-ios** | iOS（arm64、模拟器） | `ort-download`、`ort-coreml`、`llm-llamacpp-vision`、`asr-whispercpp` | 开启 | whisper.cpp | 静态 ORT 链接；用于 ANE 加速的 CoreML；经由 ggml 使用 Metal |
| **platform-macos** | macOS（arm64、x86_64） | `ort-download`、`ort-coreml`、`llm-llamacpp-vision`、`asr-whispercpp` | 开启 | whisper.cpp | 与 iOS 相同 — 统一的 Apple 平台特性 |
| **platform-desktop** | Linux、Windows | `ort-download`、`llm-llamacpp-vision`、`asr-whispercpp` | 开启 | whisper.cpp | 静态 ORT 链接；LLM 推理用 llama.cpp，ASR 用 whisper.cpp（所有平台统一） |

> **注意**：CLI（`xybrid-cli`）会为其所有平台预设添加 `huggingface`，使 `xybrid run --huggingface` 在发布构建中可用。SDK/FFI 预设默认不包含 `huggingface` — 如有需要请单独添加。

VLM 构建示例：

```bash
cargo build -p xybrid-cli --features platform-macos,llm-llamacpp-vision
cargo check -p xybrid-sdk --features platform-desktop,llm-llamacpp-vision
```

### 为什么 llm-mistral 不用于 Android

mistral.rs 在 ARM 上以 `+fp16` 目标特性编译，这需要 ARMv8.2-A FP16 扩展。许多 Android 设备（包括流行的 Samsung 和 Pixel 设备）不具备这些扩展，会在运行时导致 **SIGILL**（非法指令）崩溃。

llama.cpp 通过 ggml 使用**运行时 SIMD 检测**，因此对所有 Android 设备都是安全的。

---

## 受 Feature 控制的类型与模块

以下类型和模块会根据 feature 标志进行条件编译：

### runtime_adapter/mod.rs

| 模块 | 条件 | 说明 |
|--------|-----------|-------------|
| `coreml` | `target_os = "macos" OR target_os = "ios" OR test` | CoreML 运行时适配器 |
| `candle` | `feature = "candle"` | Candle（纯 Rust）运行时适配器 |
| `llm` | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | LLM 共享类型与适配器 |
| `mistral` | `feature = "llm-mistral"` | MistralBackend 实现 |
| `llama_cpp` | `feature = "llm-llamacpp"` | LlamaCppBackend 实现 |

### execution/executor.rs

| 项 | 条件 | 说明 |
|------|-----------|-------------|
| `LlmRuntimeAdapter` 导入 | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | LLM 适配器导入 |
| `llm_adapter_cache` 字段 | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | TemplateExecutor 中缓存的 LLM 适配器 |
| `ExecutionTemplate::Gguf` 处理 | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | GGUF 模型执行路径 |
| `execute_streaming()` 完整实现 | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | 带回调的流式 |
| `execute_streaming()` 桩 | `NOT (llm-mistral OR llm-llamacpp)` | 回退到常规执行 |
| `execute_streaming_with_context()` | 同上 | 带对话上下文的流式 |
| `execute_llm()` | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | 内部 LLM 执行 |
| `execute_llm_streaming()` | 同上 | 内部流式执行 |

### runtime_adapter/mod.rs 中的重新导出

| 导出 | 条件 |
|--------|-----------|
| `ONNXMobileRuntimeAdapter` | `target_os = "android" OR test` |
| `CoreMLRuntimeAdapter` | `target_os = "macos" OR target_os = "ios" OR test` |
| `CandleBackend`、`CandleRuntimeAdapter` | `feature = "candle"` |
| `ChatMessage`、`GenerationConfig`、`GenerationOutput`、`LlmBackend`、`LlmConfig`、`LlmResult`、`LlmRuntimeAdapter` | `feature = "llm-mistral" OR feature = "llm-llamacpp"` |
| `MistralBackend` | `feature = "llm-mistral"` |
| `LlamaCppBackend` | `feature = "llm-llamacpp"` |
| `llama_log_get_verbosity`、`llama_log_set_verbosity` | `feature = "llm-llamacpp"` |

---

## 无效的 Feature 组合

以下 feature 组合无效，应该会产生编译错误：

| 组合 | 原因 | 推荐替代方案 |
|-------------|--------|------------------------|
| 在 `target_os = "android"` 上使用 `llm-mistral` | 在不具备 ARMv8.2-A FP16 的设备上发生 SIGILL 崩溃 | 改用 `llm-llamacpp` 或平台预设 |
| 同时启用 `ort-download` 与 `ort-dynamic` | 互斥的 ORT 加载策略 | 根据平台二选一 |
| 在非 Apple 目标上使用 `candle-metal` | Metal 仅限 Apple | 使用 `candle`（CPU）或 `candle-cuda` |
| 在 Apple 目标上使用 `candle-cuda` | Apple 上不提供 CUDA | 使用 `candle-metal` |
| 在非 Apple 目标上使用 `ort-coreml` | CoreML 仅限 Apple | 使用 `ort-download` |
| `cargo … --all-features` | 因目标而异：在每个受支持的目标三元组上，`--all-features` 都会触发上面至少一行（ORT 加载模式冲突是普遍的；Candle Metal/CUDA + ORT CoreML 这几行会在其受支持目标的相反目标上触发）。它还会启用仅作标记的 `llm-mistral*` 特性，而其背后的 crate 目前已从 workspace 中注释掉，因此无论目标如何，构建都会因缺少 `mistralrs` 导入而失败。 | 使用下方的[发布门禁](#发布门禁)；切勿将 `--all-features` 用作 CI 门禁。 |

**注意**：上表中列出的逐行 `compile_error!` 守卫已在 [`crates/xybrid-core/src/lib.rs`](../crates/xybrid-core/src/lib.rs) 中**实现**。每个冲突都会触发一个带修复提示信息的类型化编译错误 — 参见针对 Android 上 `llm-mistral`、`ort-download` 与 `ort-dynamic`、非 Apple 上 `candle-metal`、Apple 上 `candle-cuda`、以及非 Apple 上 `ort-coreml` 的 `compile_error!` 代码块。`--all-features` 这一行通过这些逐行守卫外加仅作标记的 `llm-mistral*` 构建中断来强制约束。

---

## 发布门禁

以下是 CI 为发布把关时必须运行的规范 feature 组合。任何要求 `cargo … --all-features -- -D warnings` 的验收标准都是错误的（参见上文[无效的 Feature 组合](#无效的-feature-组合)） — 将审阅者指向此处。

### 全工作区 clippy

| 门禁 | 命令 | 覆盖范围 |
|------|---------|--------|
| 默认特性的工作区 clippy | `cargo clippy --workspace -- -D warnings` | 默认的 `ort-download` 形态；在不启用其他任何特性时，随附的 crate 也能干净编译。 |
| Vision 总括的工作区 clippy | `cargo clippy --workspace --features llm-llamacpp-vision --tests --examples -- -D warnings` | 经由 llama.cpp `mtmd` 的完整 VLM 路径，包括以 `llm-llamacpp-vision` 为门禁的 vision 测试/示例。 |
| **禁止使用 `--all-features`。** | — | 参见上文的冲突表。 |

### 平台预设矩阵

在每个目标主机上运行（或在 CI 矩阵作业中运行）。每一行都与发布工作流实际构建的内容相匹配 — 即发布的构件，按 CI 的方式构建。在本地机器上与此不一致（例如对宿主三元组执行 clippy 而非交叉编译）会漏掉真实的平台相关 bug。

| 平台 | 构建主机 | 规范门禁 |
|---------|-----------|---------|
| macOS arm64 / x86_64 | macOS | `cargo clippy --workspace --features platform-macos -- -D warnings` + `cargo test --workspace --features platform-macos` |
| iOS arm64 + 模拟器 | macOS | `bazel build --config=ios //bindings/apple:XybridFFI`（rules_apple xcframework，设备 + 模拟器切片）。CI 变体参见 [`.github/workflows/build-apple.yml`](../.github/workflows/build-apple.yml)。 |
| Android arm64-v8a / armeabi-v7a / x86_64 | 任意（Bazel 自带 NDK） | `bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar`（功能完整的 3-ABI AAR）。CI 变体参见 [`.github/workflows/build-android.yml`](../.github/workflows/build-android.yml)。 |
| 桌面 Linux x86_64 | Linux | `cargo clippy --workspace --features platform-desktop -- -D warnings` + `cargo test --workspace --features platform-desktop` |
| 桌面 Windows x86_64 | Windows | 与 Linux 桌面相同 |

对于 iOS 或 Android 上的视觉语言 CI 门禁，上述规范 xtask 命令必须在 `xybrid-uniffi` 上组合 `llm-llamacpp-vision` 特性。build-apple/build-android 工作流已经接受这种组合 — 不要自创新的本地 clippy 调用；使用 CI 所用的方式。

### 格式与 diff 门禁

以下检查在每个主机上运行，且不产生平台专属构件：

```bash
cargo fmt --all --check
git diff --check          # 无空白字符错误
```

### 在 Apple Silicon 开发机上快速验证

前三条是提交 PR 前的规范本地检查：

```bash
cargo fmt --all --check
cargo clippy --workspace --features llm-llamacpp-vision --tests --examples -- -D warnings
cargo test --workspace --features llm-llamacpp-vision
```

这套检查在 2026-05-23 于 `codex/vision-models-support` 分支上通过（clippy 耗时 2 分 20 秒，远低于提交 PR 前健全性检查的时间预算）。复现这套检查是推送前的最低要求。

---

## ORT 加载策略

ONNX Runtime 的加载方式因平台而异：

| 平台 | 策略 | Feature | 环境变量 | 说明 |
|----------|----------|---------|---------------------|-------|
| 桌面（Linux/Windows） | 下载预编译 | `ort-download` | - | 在构建时下载 ORT 二进制文件 |
| macOS | 下载预编译 | `ort-download` | - | 在构建时下载 ORT 二进制文件 |
| iOS | XCFramework | `ort-download` | `ORT_IOS_XCFWK_LOCATION` | 必须指向 `onnxruntime.xcframework` |
| Android | 动态加载 | `ort-dynamic` | - | 在运行时从 AAR 加载 `libonnxruntime.so` |

### iOS XCFramework 设置

对于 iOS 构建，你必须设置 `ORT_IOS_XCFWK_LOCATION` 指向预编译的 ONNX Runtime iOS XCFramework：

```bash
# 方式 1：从 VOICEVOX 下载
# https://github.com/VOICEVOX/onnxruntime-builder/releases

# 方式 2：从 HuggingFace 下载
# https://huggingface.co/csukuangfj/ios-onnxruntime

# 方式 3：从源码构建
# https://onnxruntime.ai/docs/build/ios.html

export ORT_IOS_XCFWK_LOCATION=/path/to/onnxruntime.xcframework
```

---

## xtask 命令

`xtask` crate 提供构建自动化命令。运行 `cargo xtask --help` 查看完整文档。

| 命令 | 用途 | 平台 | 示例 |
|---------|---------|----------|---------|
| `setup-test-env` | 为集成测试下载模型 | 任意 | `cargo xtask setup-test-env` |
| `build-flutter` | 构建 Flutter 原生库 | 视情况而定 | `cargo xtask build-flutter --platform macos` |

### xtask 与 Feature 预设的映射

| xtask 命令 | 使用的平台预设 | 构建的目标 |
|---------------|---------------------|---------------|
| `build-flutter --platform ios` | `platform-ios` | aarch64-apple-ios、aarch64-apple-ios-sim |
| `build-flutter --platform android` | `platform-android` | aarch64-linux-android、armv7-linux-androideabi、x86_64-linux-android |
| `build-flutter --platform macos` | `platform-macos` | aarch64-apple-darwin、x86_64-apple-darwin |
| `build-flutter --platform linux` | `platform-desktop` | x86_64-unknown-linux-gnu |
| `build-flutter --platform windows` | `platform-desktop` | x86_64-pc-windows-msvc |

这些自动的 xtask 映射使用上面的纯文本平台预设。VLM 构建必须在该构建路径所用的 Cargo feature 集合中显式添加 `llm-llamacpp-vision`。

---

## 构建架构

Xybrid 使用**两层构建架构**：

### 第 1 层：xtask（编排）

**位置**：`xtask/src/main.rs`

**职责**：
- 交叉编译目标选择
- 多目标构建（例如所有 Android ABI）
- 平台专属工具（lipo、xcodebuild、cargo-ndk）
- 打包与分发（zip、tar.gz）
- CI/CD 集成

**不负责**：
- 原生依赖编译
- 链接器配置
- CMake 调用

### 第 2 层：llama-cpp-sys build.rs（编译）

**位置**：`crates/llama-cpp-sys/build.rs`

**职责**：
- 通过 CMake 编译随附的 llama.cpp
- 为 CMake 工具链检测 Android NDK
- 平台专属链接（Metal、Accelerate 等）
- 设置 `cargo:rustc-link-lib` 和 `cargo:rustc-link-search`

**触发条件**：
- 通过 `xybrid-core/llm-llamacpp` 触达的 `llama-cpp-sys/bindings` 特性
- 编译 llm-llamacpp 时的 Cargo 构建流程

## 快速参考

### 最小构建（无 LLM）

```bash
cargo check -p xybrid-core --no-default-features --features ort-download
```

### macOS 开发

```bash
cargo build -p xybrid-core --features "ort-download,ort-coreml,llm-llamacpp"
```

### macOS 视觉语言开发

```bash
cargo build -p xybrid-core --features "ort-download,ort-coreml,llm-llamacpp-vision"
```

### Android 构建

```bash
# Bazel 自带固定版本的 NDK — 无需机器配置
bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar
```

### 完整特性检查

```bash
# 仅 macOS（包含 Metal 特性）
cargo check -p xybrid-core --features "ort-download,ort-coreml,candle-metal,llm-llamacpp"
```

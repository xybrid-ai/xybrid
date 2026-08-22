<div align="center">
<p align="center">
  <a href="./README.md">English</a> · <a href="./README.zh-CN.md">简体中文</a> · <a href="./README.ja-JP.md">日本語</a>
</p>


<p align="center">
  <img src="./docs/logo.jpg" alt="Xybrid Logo" width="180"/>
</p>

<h1 align="center">Xybrid</h1>

<p align="center">
  <strong>在应用与游戏中原生运行 LLM、ASR 与 TTS。</strong><br/>
  <a href="#flutter">Flutter</a> · <a href="#swift">Swift</a> · <a href="#kotlin">Kotlin</a> · <a href="#unity">Unity</a> · <a href="#rust">Rust</a><br/>
  隐私优先，离线可用，无需云端。
</p>

<p align="center">

[![Docs][docs-shield]][docs-url]
[![Website][website-shield]][website-url]
[![Follow on X][twitter-shield]][twitter-url]
[![Discord][discord-shield]][discord-url]

</p>

<p align="center">

[![Build][build-shield]][build-url]
[![Release][release-shield]][release-url]
[![License][license-shield]][license-url]
[![OpenSSF Scorecard][scorecard-shield]][scorecard-url]
[![OpenSSF Best Practices][bestpractices-shield]][bestpractices-url]
<br>
[![crates.io][crates-shield]][crates-url]
[![pub.dev][pubdev-shield]][pubdev-url]
[![Maven Central][maven-shield]][maven-url]
[![Swift Package Manager][spm-shield]][spm-url]
<br>
[![Ask DeepWiki][deepwiki-shield]][deepwiki-url]
[![Stars][stars-shield]][stars-url]
[![Visitors][visitors-shield]][visitors-url]

</p>

<!-- Primary — for-the-badge -->
[docs-shield]: https://img.shields.io/badge/Docs-docs.xybrid.dev-1F6FEB?style=for-the-badge&logo=readthedocs&logoColor=white
[docs-url]: https://docs.xybrid.dev/
[website-shield]: https://img.shields.io/badge/Website-xybrid.ai-4285F4?style=for-the-badge
[website-url]: https://www.xybrid.ai/
[discord-shield]: https://img.shields.io/badge/Join_Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white
[discord-url]: https://discord.gg/YhFHHkhbad

<!-- Project health — flat-square -->
[build-shield]: https://img.shields.io/github/actions/workflow/status/xybrid-ai/xybrid/ci.yml?branch=master&style=flat-square
[build-url]: https://github.com/xybrid-ai/xybrid/actions
[release-shield]: https://img.shields.io/github/v/release/xybrid-ai/xybrid?style=flat-square&sort=semver
[release-url]: https://github.com/xybrid-ai/xybrid/releases
[license-shield]: https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square
[license-url]: https://opensource.org/licenses/Apache-2.0
[scorecard-shield]: https://api.scorecard.dev/projects/github.com/xybrid-ai/xybrid/badge
[scorecard-url]: https://scorecard.dev/viewer/?uri=github.com/xybrid-ai/xybrid
[bestpractices-shield]: https://www.bestpractices.dev/projects/13041/badge
[bestpractices-url]: https://www.bestpractices.dev/projects/13041

<!-- Packages — flat-square -->
[crates-shield]: https://img.shields.io/crates/v/xybrid?style=flat-square&label=crates.io&logo=rust
[crates-url]: https://crates.io/crates/xybrid
[pubdev-shield]: https://img.shields.io/pub/v/xybrid_flutter?style=flat-square&label=pub.dev
[pubdev-url]: https://pub.dev/packages/xybrid_flutter
[maven-shield]: https://img.shields.io/maven-central/v/ai.xybrid/xybrid-kotlin?style=flat-square&label=Maven%20Central
[maven-url]: https://central.sonatype.com/artifact/ai.xybrid/xybrid-kotlin
[spm-shield]: https://img.shields.io/badge/Swift_Package_Manager-compatible-F05138?style=flat-square&logo=swift&logoColor=white
[spm-url]: https://github.com/xybrid-ai/xybrid

<!-- Community — flat-square -->
[deepwiki-shield]: https://deepwiki.com/badge.svg
[deepwiki-url]: https://deepwiki.com/xybrid-ai/xybrid
[stars-shield]: https://img.shields.io/github/stars/xybrid-ai/xybrid?style=flat-square
[stars-url]: https://github.com/xybrid-ai/xybrid/stargazers
[twitter-shield]: https://img.shields.io/badge/Follow-%40xybrid__ai-000000?style=for-the-badge&logo=x&logoColor=white
[twitter-url]: https://x.com/xybrid_ai
[visitors-shield]: https://visitor-badge.laobi.icu/badge?page_id=xybrid-ai.xybrid
[visitors-url]: https://github.com/xybrid-ai/xybrid
</div>

<p align="center">
  <img src="docs/demo-desktop.gif" alt="Desktop demo" width="540">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/demo-android.gif" alt="Android demo" width="150">
</p>



## 从这里开始

| 目标 | 路径 |
|------|------|
| 最快上手（2 分钟） | [安装 CLI →](#快速开始) |
| 开发移动端或桌面应用 | [Flutter SDK →](bindings/flutter/) |
| 为游戏添加 AI NPC | [Unity SDK →](bindings/unity/)，体验 [3D 酒馆示例](https://github.com/xybrid-ai/xybrid-unity-tavern) |
| Android 原生开发 | [Kotlin SDK →](bindings/kotlin/) |
| Rust / 嵌入式 | [核心 crate →](crates/) |
---

<p align="center">
  <img src="docs/game-demo.gif" alt="游戏演示" width="540">
</p>

## SDK

Xybrid 是一个 **Rust 驱动的运行时**，为所有主流平台提供原生绑定：

| SDK | 平台 | 安装 | 状态 | 示例 |
|-----|------|------|------|------|
| **[Flutter](bindings/flutter/)** | iOS, Android, macOS, Linux, Windows | [pub.dev](https://pub.dev/packages/xybrid_flutter) | 可用 | [README](examples/flutter/README.md) |
| **[Unity](bindings/unity/)** | macOS, Windows, Linux, iOS, Android | [见下方](#快速开始) | 可用 | [Unity 3D AI 酒馆](https://github.com/xybrid-ai/xybrid-unity-tavern) |
| **[Swift](bindings/apple/)** | iOS, macOS | Swift Package Manager | 即将推出 | [README](examples/ios/README.md) |
| **[Kotlin](bindings/kotlin/)** | Android | Maven Central | 可用 | [README](examples/android/README.md) |
| **[CLI](https://github.com/xybrid-ai/xybrid/releases)** | macOS, Linux, Windows | `curl -sSL .../install.sh \| sh` | 可用 | — |
| **[Rust](crates/)** | 全平台 | [crates.io](https://crates.io/crates/xybrid) | 可用 | — |

所有 SDK 封装同一个 Rust 核心——跨平台行为和模型支持完全一致。

---

## 快速开始

选择你喜欢的语言，安装并运行模型。每个语言下都包含安装片段和最小示例。

### CLI

**安装：**

```bash
# macOS / Linux
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
```

**运行模型：**

```sh
xybrid run --model kokoro-82m --input-text "国破山河在，城春草木深" -o output.wav
```

### Flutter

**安装** 在 `pubspec.yaml`：

```yaml
dependencies:
  xybrid_flutter: ^0.5.0
```

**运行模型：**

```dart
final model = await Xybrid.model('kokoro-82m').load();
final result = await model.run(XybridEnvelope.text('国破山河在，城春草木深'));
// 输出 → 24kHz WAV 音频
```

### Kotlin

**安装** 在 `build.gradle.kts`：

```gradle
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.5.0")
}
```

**运行模型：**

```kotlin
val model = Xybrid.model("kokoro-82m").load()
val result = model.runAsync(Envelope.text("国破山河在，城春草木深"))
// 输出 → 24kHz WAV 音频
```

### Swift

**安装** 在 `Package.swift`：

```swift
dependencies: [
    .package(url: "https://github.com/xybrid-ai/xybrid.git", from: "0.5.0")
]
```

**运行模型：**

```swift
let model = try await Xybrid.model("kokoro-82m").load()
let result = try await model.runAsync(envelope: Envelope.text("国破山河在，城春草木深"))
// 输出 → 24kHz WAV 音频
```

### Unity

**安装** 通过 [OpenUPM](https://openupm.com/packages/ai.xybrid.sdk/)（`openupm add ai.xybrid.sdk`），或直接从 git 子目录安装：

```sh
https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity
```

**运行模型：**

```csharp
var model = XybridClient.LoadModel("kokoro-82m");
var result = model.Run(Envelope.Text("国破山河在，城春草木深"));
// 输出 → 24kHz WAV 音频
```

### Rust

**安装** 在 `Cargo.toml`：

```toml
[dependencies]
xybrid = "0.5.0"
```

**运行模型：**

```rust
let model = Xybrid::model("kokoro-82m").load()?;
let result = model.run(&Envelope::text("国破山河在，城春草木深"))?;
// 输出 → 24kHz WAV 音频
```

完整安装选项、硬件加速与 CLI 参考请参阅 [安装指南](docs/INSTALLATION.zh.md)。各平台的详细设置请参阅对应 SDK 的 README：[Flutter](bindings/flutter/) · [Unity](bindings/unity/) · [Swift](bindings/apple/) · [Kotlin](bindings/kotlin/) · [Rust](crates/)。

<details>
<summary><h3>多模型推理流水线 — MMP（实验性）</h3></summary>

将多个模型链接成一条多模型推理流水线（MMP）——用 3 行 YAML 搭建语音助手：

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny-ggml  # 语音 → 文本
  - model: qwen2.5-0.5b       # 用 LLM 处理
  - model: kokoro-82m         # 文本 → 语音
```

**CLI:**
```sh
xybrid run --config voice-assistant.yaml --input-audio question.wav -o response.wav
```

**Flutter:**
```dart
final pipeline = Xybrid.pipeline(yaml: yamlString);
final result = await pipeline.run(XybridEnvelope.audio(bytes: audioBytes, sampleRate: 16000));
```

**Kotlin:**
```kotlin
// 多模型流水线（MMP）支持即将推出，当前请使用单模型加载
```

**Swift:**
```swift
// 多模型流水线（MMP）支持即将推出，当前请使用单模型加载
```

**Unity (C#):**
```csharp
// 多模型流水线（MMP）支持即将推出，当前请使用单模型加载
```

**Rust:**
```rust
let pipeline = Xybrid::pipeline(&yaml_string).load()?;
pipeline.load_models()?;
let result = pipeline.run(&Envelope::audio(audio_bytes))?;
```

</details>

---

## 支持的模型

所有模型完全在设备端运行。无需云端，无需 API 密钥。使用 `xybrid models list` 查看完整的模型注册表。

### 从这些模型开始

| 模型 | 类型 | 参数量 | 推荐理由 |
|------|------|--------|----------|
| **SmolLM2 360M** | LLM | 360M | 最佳质量/体积比，适合任何设备 |
| **Kokoro 82M** | TTS | 82M | 高质量语音合成，24 种声音，速度快 |
| **Whisper Tiny** | ASR | 39M | 多语言转录，准确率高 |

### 语音转文本

| 模型 | 参数量 | 格式 | 简介 |
|------|--------|------|------|
| Whisper Tiny（`whisper-tiny-ggml`） | 39M | GGML Q5_1 | 多语言转录，基于 whisper.cpp — 所有平台预设均已内置 |
| Whisper Tiny（`whisper-tiny`） | 39M | SafeTensors | 相同权重，运行于 Candle — 需要启用 `candle` 特性重新构建 |
| Wav2Vec2 Base | 95M | ONNX | 英语 ASR，CTC 解码 |

### 文本转语音

| 模型 | 参数量 | 格式 | 简介 |
|------|--------|------|------|
| Kokoro 82M | 82M | ONNX | 高质量，24 种自然声音 |
| KittenTTS Nano | 15M | ONNX | 超轻量级，8 种声音 |

### 语言模型

| 模型 | 参数量 | 格式 | 简介 |
|------|--------|------|------|
| Gemma 3 1B | 1B | GGUF Q4_K_M | Google 为移动端优化的模型 |
| LFM2.5 230M | 230M | GGUF Q4_K_M | 最小的采用 Liquid AI 混合卷积+注意力架构的 LLM，适合边缘设备 |
| LFM2.5 350M | 354M | GGUF Q4_K_M | Liquid AI 混合卷积+注意力架构，9 种语言，工具调用 |
| Llama 3.2 1B | 1B | GGUF Q4_K_M | Meta 的通用模型，128K 上下文 |
| Qwen 2.5 0.5B | 500M | GGUF Q4_K_M | 紧凑的本地聊天模型 |
| Qwen 3.5 0.8B | 800M | GGUF Q4_K_M | 最新 Qwen，支持推理（思考模式） |
| Qwen 3.5 2B | 2B | GGUF Q4_K_M | 更大的 Qwen 3.5，扩展推理能力 |
| SmolLM2 360M | 360M | GGUF Q4_K_M | 最佳的微型模型，优秀的质量/体积比 |

### 视觉语言模型（VLM）

| 模型 | 参数量 | 格式 | 简介 |
|------|--------|------|------|
| LFM2-VL 450M | 450M | GGUF Q4_0 + mmproj | Liquid AI 的紧凑型 VLM（SigLIP2 视觉）— 图像+文本输入，通过 llama.cpp mtmd 运行 |

### 即将推出

| 模型 | 类型 | 参数量 | 优先级 | 状态 |
|------|------|--------|--------|------|
| Phi-4 Mini | LLM | 3.8B | P2 | 规格就绪（首个多量化：Q4, Q8, FP16） |
| Qwen3 0.6B | LLM | 600M | P2 | 计划中 |
| Trinity Nano | LLM (MoE) | 6B（1B 活跃） | P2 | 计划中 |
| LFM2-VL 700M | Vision+LLM | 700M | P2 | 计划中 |
| Nomic Embed Text v1.5 | 嵌入 | 137M | P1 | 受阻（需要 Tokenize/MeanPool 步骤） |
| Whisper Tiny CoreML | ASR | 39M | P2 | 计划中 |
| Qwen3-TTS 0.6B | TTS | 600M | P2 | 受阻（需要自定义 SafeTensors 运行时） |
| Chatterbox Turbo | TTS | 350M | P3 | 受阻（需要 ModelGraph 模板） |

<details>
<summary><h3>自定义模型（实验性）</h3></summary>

> **注意**：自定义模型支持为实验性功能。`model_metadata.json` schema 已稳定，但 AI 辅助工具（`/xybrid-init`）仍在积极开发中，可能尚不支持所有模型类型。

Xybrid 支持**任意** ONNX、GGUF 或 SafeTensors 模型，只需提供一个 `model_metadata.json` 告诉 xybrid 如何运行它。

**使用 AI 助手**（Claude Code、Codex 等）：

```sh
# 将 xybrid skills 安装到你的项目中
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/tools/scripts/install-skills.sh | sh

# 从 HuggingFace 模型生成 model_metadata.json
claude /xybrid-init hexgrad/Kokoro-82M-v1.0-ONNX
```

Skills 与 agent 无关，位于 [`agents/skills/`](agents/skills/)。安装脚本会为 Claude Code（`.claude/skills`）和 Codex（`.codex/skills`）创建符号链接。

**手动创建** — 在模型目录中新建 `model_metadata.json`：

```json
{
  "model_id": "my-model",
  "version": "1.0",
  "execution_template": { "type": "Onnx", "model_file": "model.onnx" },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["model.onnx"],
  "metadata": { "task": "text-generation" }
}
```

完整 schema 见 [model metadata 文档](docs/sdk/API_REFERENCE.md)，或参考 [`integration-tests/fixtures/models/`](integration-tests/fixtures/models/) 中的现有示例。

</details>

---

## 功能

| 能力 | iOS | Android | macOS | Linux | Windows |
|------|-----|---------|-------|-------|---------|
| 语音转文本 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 文本转语音 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 语言模型 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 视觉模型 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 嵌入模型 | 🔜 | 🔜 | 🔜 | 🔜 | 🔜 |
| 多模型流水线（MMP） | ✅ | ✅ | ✅ | ✅ | ✅ |
| 模型下载与缓存 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 硬件加速 | Metal, ANE | CPU | Metal, ANE | CPU，可选 Vulkan | CPU |

**SDK MMP 支持：** Flutter ✅ · Rust ✅ · Kotlin 🔜 · Swift 🔜 · Unity 🔜

---

## 为什么选择 Xybrid？

- **隐私优先** — 所有推理在设备端运行。你的数据永远不会离开你的设备。
- **离线可用** — 初次模型下载后无需互联网。
- **跨平台** — iOS、Android、macOS、Linux 和 Windows 使用统一的 API。
- **多模型流水线（MMP）** — 在单次调用中链接多个模型（ASR → LLM → TTS）。
- **硬件加速** — Apple 平台支持 Apple Neural Engine 与 Metal；Linux 的 llama.cpp 构建可选启用 Vulkan。Android 与 Windows 目前使用 CPU，预编译的 Linux 二进制也仅为 CPU（Vulkan 需在构建时启用 — 参见[安装说明](docs/INSTALLATION.md#platform-features)）。

### 与其他方案对比

| | Xybrid | Ollama | llama.cpp | ONNX Runtime |
|---|---|---|---|---|
| 移动端（iOS/Android） | ✅ | ❌ | ❌ | ✅ |
| 游戏引擎（Unity） | ✅ | ❌ | ❌ | ❌ |
| 多模型流水线（MMP） | ✅ | ❌ | ❌ | ❌ |
| ASR + TTS + LLM 统一 SDK | ✅ | ❌ | ❌ | ❌ |
| 进程内运行（无需服务器） | ✅ | ❌ | ✅ | ✅ |
| 无需云端 | ✅ | ✅ | ✅ | ✅ |

---

## 社区

- [文档](https://docs.xybrid.dev)
- [Discord](https://discord.gg/YhFHHkhbad)
- [X (Twitter)](https://x.com/xybrid_ai)
- [GitHub Issues](https://github.com/xybrid-ai/xybrid/issues)

## 贡献

我们欢迎贡献！请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解开发环境设置、提交 PR 和添加新模型的指南。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xybrid-ai/xybrid&type=date&legend=bottom-right)](https://www.star-history.com/#xybrid-ai/xybrid&type=date&legend=bottom-right)

## 许可证

Apache License 2.0 — 详见 [LICENSE](./LICENSE)。

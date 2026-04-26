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
  Rust 核心 · iOS · Android · Flutter · Unity<br/>
  隐私优先，离线可用，无需云端。
</p>

<p align="center">
  <a href="https://docs.xybrid.dev">文档</a> ·
  <a href="#sdk">SDK</a> ·
  <a href="https://www.xybrid.ai/models">模型</a> ·
  <a href="https://discord.gg/YhFHHkhbad">加入 Discord</a> ·
  <a href="https://x.com/xybrid_ai">关注 X</a> ·
  <a href="https://github.com/xybrid-ai/xybrid/issues">问题反馈</a>
</p>

<p align="center">

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
<br>
[![License][license-shield]][license-url]
[![Build][build-shield]][build-url]
[![OpenSSF Scorecard][scorecard-shield]][scorecard-url]
[![Stars][stars-shield]][stars-url]
[![Release][release-shield]][release-url]
[![Release Date][release-date-shield]][release-url]
<br>
[![pub.dev][pubdev-shield]][pubdev-url]
[![Maven Central][maven-shield]][maven-url]
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=xybrid-ai.xybrid)](https://github.com/xybrid-ai/xybrid)

</p>

[website-shield]: https://img.shields.io/badge/xybrid.ai-4285F4?style=flat
[website-url]: https://www.xybrid.ai/
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FYhFHHkhbad%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=5865F2&suffix=%20members
[discord-url]: https://discord.gg/YhFHHkhbad
[twitter-shield]: https://img.shields.io/twitter/follow/xybrid_ai
[twitter-url]: https://x.com/xybrid_ai
[license-shield]: https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat
[license-url]: https://opensource.org/licenses/Apache-2.0
[build-shield]: https://img.shields.io/github/actions/workflow/status/xybrid-ai/xybrid/ci.yml?branch=master&style=flat
[build-url]: https://github.com/xybrid-ai/xybrid/actions
[scorecard-shield]: https://api.scorecard.dev/projects/github.com/xybrid-ai/xybrid/badge
[scorecard-url]: https://scorecard.dev/viewer/?uri=github.com/xybrid-ai/xybrid
[stars-shield]: https://img.shields.io/github/stars/xybrid-ai/xybrid?style=flat
[stars-url]: https://github.com/xybrid-ai/xybrid/stargazers
[release-shield]: https://img.shields.io/github/v/release/xybrid-ai/xybrid?style=flat&sort=semver
[release-url]: https://github.com/xybrid-ai/xybrid/releases
[release-date-shield]: https://img.shields.io/github/release-date/xybrid-ai/xybrid?style=flat
[pubdev-shield]: https://img.shields.io/pub/v/xybrid_flutter?style=flat&label=pub.dev
[pubdev-url]: https://pub.dev/packages/xybrid_flutter
[maven-shield]: https://img.shields.io/maven-central/v/ai.xybrid/xybrid-kotlin?style=flat&label=Maven%20Central
[maven-url]: https://central.sonatype.com/artifact/ai.xybrid/xybrid-kotlin
</div>

<p align="center">
  <img src="docs/demo-desktop.gif" alt="Desktop demo" width="540">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/demo-android.gif" alt="Android demo" width="150">
</p>



## 从这里开始

| 目标 | 路径 |
|------|------|
| 最快上手（2 分钟） | [安装 CLI →](#安装) |
| 构建移动端或桌面应用 | [Flutter SDK →](bindings/flutter/) |
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
| **[Unity](bindings/unity/)** | macOS, Windows, Linux, iOS, Android | [见下方](#安装) | 可用 | [Unity 3D AI 酒馆](https://github.com/xybrid-ai/xybrid-unity-tavern) |
| **[Swift](bindings/apple/)** | iOS, macOS | Swift Package Manager | 即将推出 | [README](examples/ios/README.md) |
| **[Kotlin](bindings/kotlin/)** | Android | Maven Central | 可用 | [README](examples/android/README.md) |
| **[CLI](https://github.com/xybrid-ai/xybrid/releases)** | macOS, Linux, Windows | `curl -sSL .../install.sh \| sh` | 可用 | — |
| **[Rust](crates/)** | 全平台 | `xybrid-core` / `xybrid-sdk` | 可用 | — |

所有 SDK 封装同一个 Rust 核心——跨平台行为和模型支持完全一致。

### 安装

**Unity** → Package Manager：

```sh
https://github.com/xybrid-ai/xybrid.git#upm
```

**Flutter** → `pubspec.yaml`：

```yaml
dependencies:
  xybrid_flutter: ^0.1.0
```

**Swift (iOS / macOS)**：

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/xybrid-ai/xybrid.git", from: "0.1.0")
]
```

**Kotlin (Android)**：

```gradle
// build.gradle.kts
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.1.0-beta12")
}
```

**CLI**

```bash
# macOS / Linux
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
```

完整安装选项、硬件加速与 CLI 参考请参阅 [Installation Guide](docs/INSTALLATION.md)。

---

## 快速开始

各平台的详细设置请参阅对应 SDK 的 README：[Flutter](bindings/flutter/) · [Unity](bindings/unity/) · [Swift](bindings/apple/) · [Kotlin](bindings/kotlin/) · [Rust](crates/)

### 单一模型

通过 CLI 一行命令运行模型，或在任何 SDK 中用三行代码搞定：

**CLI:**
```sh
xybrid run kokoro-82m --input "国破山河在，城春草木深" -o output.wav
```

**Flutter:**
```dart
final model = await Xybrid.model('kokoro-82m').load();
final result = await model.run(XybridEnvelope.text('国破山河在，城春草木深'));
// 输出 → 24kHz WAV 音频
```

**Kotlin:**
```kotlin
val model = XybridModelLoader.fromRegistry("kokoro-82m").load()
val result = model.run(Envelope.text("国破山河在，城春草木深"))
// 输出 → 24kHz WAV 音频
```

**Swift:**
```swift
let model = try ModelLoader.fromRegistry(modelId: "kokoro-82m").load()
let result = try model.run(envelope: Envelope.text("国破山河在，城春草木深"))
// 输出 → 24kHz WAV 音频
```

**Unity (C#):**
```csharp
var model = XybridClient.LoadModel("kokoro-82m");
var result = model.Run(Envelope.Text("国破山河在，城春草木深"));
// 输出 → 24kHz WAV 音频
```

**Rust:**
```rust
let model = Xybrid::model("kokoro-82m").load()?;
let result = model.run(&Envelope::text("国破山河在，城春草木深"))?;
// 输出 → 24kHz WAV 音频
```

### 流水线

将模型链接在一起——用 3 行 YAML 搭建语音助手：

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny    # 语音 → 文本
  - model: qwen2.5-0.5b    # 用 LLM 处理
  - model: kokoro-82m      # 文本 → 语音
```

**CLI:**
```sh
xybrid run voice-assistant.yaml --input question.wav -o response.wav
```

**Flutter:**
```dart
final pipeline = Xybrid.pipeline(yaml: yamlString);
final result = await pipeline.run(XybridEnvelope.audio(bytes: audioBytes, sampleRate: 16000));
```

**Kotlin:**
```kotlin
// 流水线支持即将推出，当前请使用单模型加载
```

**Swift:**
```swift
// 流水线支持即将推出，当前请使用单模型加载
```

**Unity (C#):**
```csharp
// 流水线支持即将推出，当前请使用单模型加载
```

**Rust:**
```rust
let pipeline = Xybrid::pipeline(&yaml_string).load()?;
pipeline.load_models()?;
let result = pipeline.run(&Envelope::audio(audio_bytes))?;
```
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
| Whisper Tiny | 39M | SafeTensors | 多语言转录（Candle 运行时） |
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
| LFM2.5 350M | 354M | GGUF Q4_K_M | Liquid AI 混合卷积+注意力架构，9 种语言，工具调用 |
| Llama 3.2 1B | 1B | GGUF Q4_K_M | Meta 的通用模型，128K 上下文 |
| Qwen 2.5 0.5B | 500M | GGUF Q4_K_M | 紧凑的本地聊天模型 |
| Qwen 3.5 0.8B | 800M | GGUF Q4_K_M | 最新 Qwen，支持推理（思考模式） |
| Qwen 3.5 2B | 2B | GGUF Q4_K_M | 更大的 Qwen 3.5，扩展推理能力 |
| SmolLM2 360M | 360M | GGUF Q4_K_M | 最佳的微型模型，优秀的质量/体积比 |

### 即将推出

| 模型 | 类型 | 参数量 | 优先级 | 状态 |
|------|------|--------|--------|------|
| Phi-4 Mini | LLM | 3.8B | P2 | 规格就绪（首个多量化：Q4, Q8, FP16） |
| Qwen3 0.6B | LLM | 600M | P2 | 计划中 |
| Trinity Nano | LLM (MoE) | 6B（1B 活跃） | P2 | 计划中 |
| LFM2-VL 700M | Vision+LLM | 700M | P2 | 计划中 |
| Nomic Embed Text v1.5 | 嵌入 | 137M | P1 | 受阻（需要 Tokenize/MeanPool 步骤） |
| LFM2-VL 450M | 视觉 | 450M | P2 | 计划中 |
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
| 视觉模型 | 🔜 | 🔜 | 🔜 | 🔜 | 🔜 |
| 嵌入模型 | 🔜 | 🔜 | 🔜 | 🔜 | 🔜 |
| 流水线编排 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 模型下载与缓存 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 硬件加速 | Metal, ANE | CPU | Metal, ANE | CUDA | CUDA |

**SDK 流水线支持：** Flutter ✅ · Rust ✅ · Kotlin 🔜 · Swift 🔜 · Unity 🔜

---

## 为什么选择 Xybrid？

- **隐私优先** — 所有推理在设备端运行。你的数据永远不会离开你的设备。
- **离线可用** — 初次模型下载后无需互联网。
- **跨平台** — iOS、Android、macOS、Linux 和 Windows 使用统一的 API。
- **流水线编排** — 在单次调用中链接多个模型（ASR → LLM → TTS）。
- **自动优化** — 在 Apple Neural Engine、Metal 和 CUDA 上进行硬件加速。

### 与其他方案对比

| | Xybrid | Ollama | llama.cpp | ONNX Runtime |
|---|---|---|---|---|
| 移动端（iOS/Android） | ✅ | ❌ | ❌ | ✅ |
| 游戏引擎（Unity） | ✅ | ❌ | ❌ | ❌ |
| 多阶段流水线 | ✅ | ❌ | ❌ | ❌ |
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

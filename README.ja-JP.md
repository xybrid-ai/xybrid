<div align="center">
<p align="center">
  <a href="./README.md">English</a> · <a href="./README.zh-CN.md">简体中文</a> · <a href="./README.ja-JP.md">日本語</a>
</p>


<p align="center">
  <img src="./docs/logo.jpg" alt="Xybrid Logo" width="180"/>
</p>

<h1 align="center">Xybrid</h1>

<p align="center">
  <strong>LLM、ASR、TTSをアプリやゲームでネイティブに実行。</strong><br/>
  Rustコア · iOS · Android · Flutter · Unity<br/>
  プライベート、オフライン、クラウド不要。
</p>

<p align="center">
  <a href="https://docs.xybrid.dev">ドキュメント</a> ·
  <a href="#sdk">SDK</a> ·
  <a href="https://www.xybrid.ai/models">モデル</a> ·
  <a href="https://discord.gg/YhFHHkhbad">Discordに参加</a> ·
  <a href="https://x.com/xybrid_ai">Xでフォロー</a> ·
  <a href="https://github.com/xybrid-ai/xybrid/issues">Issues</a>
</p>

<p align="center">

[![Website][website-shield]][website-url]
[![Docs][docs-shield]][docs-url]
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
[![Swift Package Manager][spm-shield]][spm-url]
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=xybrid-ai.xybrid)](https://github.com/xybrid-ai/xybrid)

</p>

[website-shield]: https://img.shields.io/badge/xybrid.ai-4285F4?style=flat
[website-url]: https://www.xybrid.ai/
[docs-shield]: https://img.shields.io/badge/docs-xybrid.dev-1F6FEB?style=flat&logo=readthedocs&logoColor=white
[docs-url]: https://docs.xybrid.dev/
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
[spm-shield]: https://img.shields.io/badge/Swift_Package_Manager-compatible-F05138?style=flat&logo=swift&logoColor=white
[spm-url]: https://github.com/xybrid-ai/xybrid
</div>

<p align="center">
  <img src="docs/demo-desktop.gif" alt="デスクトップデモ" width="540">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/demo-android.gif" alt="Androidデモ" width="150">
</p>



## はじめに

| 目的 | パス |
|------|------|
| 最速デモ（2分） | [CLIをインストール →](#クイックスタート) |
| モバイルまたはデスクトップアプリを構築 | [Flutter SDK →](bindings/flutter/) |
| ゲームにAI NPCを追加 | [Unity SDK →](bindings/unity/) および [3D酒場デモ](https://github.com/xybrid-ai/xybrid-unity-tavern) を試す |
| Androidネイティブ | [Kotlin SDK →](bindings/kotlin/) |
| Rust / 組み込み | [Coreクレート →](crates/) |
---

<p align="center">
  <img src="docs/game-demo.gif" alt="ゲームデモ" width="540">
</p>

## SDK

Xybridは**Rustベースのランタイム**であり、すべての主要プラットフォーム向けのネイティブバインディングを提供します。

| SDK | プラットフォーム | インストール | ステータス | サンプル |
|-----|-----------|---------|--------|--------|
| **[Flutter](bindings/flutter/)** | iOS, Android, macOS, Linux, Windows | [pub.dev](https://pub.dev/packages/xybrid_flutter) | 利用可能 | [README](examples/flutter/README.md) |
| **[Unity](bindings/unity/)** | macOS, Windows, Linux, iOS, Android | [下記参照](#クイックスタート) | 利用可能 | [Unity 3D AI酒場](https://github.com/xybrid-ai/xybrid-unity-tavern) |
| **[Swift](bindings/apple/)** | iOS, macOS | Swift Package Manager | 近日公開 | [README](examples/ios/README.md) |
| **[Kotlin](bindings/kotlin/)** | Android | Maven Central | 利用可能 | [README](examples/android/README.md) |
| **[CLI](https://github.com/xybrid-ai/xybrid/releases)** | macOS, Linux, Windows | `curl -sSL .../install.sh \| sh` | 利用可能 | — |
| **[Rust](crates/)** | すべて | `xybrid-core` / `xybrid-sdk` | 利用可能 | — |

すべてのSDKは同じRustコアをラップしており、すべてのプラットフォームで同一のモデルサポートと動作を提供します。

---

## クイックスタート

お好みの言語でインストールしてモデルを実行できます。各セクションにはインストール手順と最小限のサンプルが含まれています。

<details>
<summary><b>CLI</b> — macOS、Linux、Windows</summary>

**インストール:**

```bash
# macOS / Linux
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
```

**モデルを実行:**

```sh
xybrid run kokoro-82m --input "Hello world" -o output.wav
```

</details>

<details>
<summary><b>Flutter</b> — iOS、Android、macOS、Linux、Windows</summary>

**インストール** `pubspec.yaml`:

```yaml
dependencies:
  xybrid_flutter: ^0.1.0
```

**モデルを実行:**

```dart
final model = await Xybrid.model('kokoro-82m').load();
final result = await model.run(XybridEnvelope.text('Hello world'));
// result → 24kHz WAVオーディオ
```

</details>

<details>
<summary><b>Kotlin</b> — Android</summary>

**インストール** `build.gradle.kts`:

```gradle
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.1.0-beta12")
}
```

**モデルを実行:**

```kotlin
val model = XybridModelLoader.fromRegistry("kokoro-82m").load()
val result = model.run(Envelope.text("Hello world"))
// result → 24kHz WAVオーディオ
```

</details>

<details>
<summary><b>Swift</b> — iOS、macOS</summary>

**インストール** `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/xybrid-ai/xybrid.git", from: "0.1.0")
]
```

**モデルを実行:**

```swift
let model = try ModelLoader.fromRegistry(modelId: "kokoro-82m").load()
let result = try model.run(envelope: Envelope.text("Hello world"))
// result → 24kHz WAVオーディオ
```

</details>

<details>
<summary><b>Unity (C#)</b> — macOS、Windows、Linux、iOS、Android</summary>

**インストール** Unity Package Managerを使用:

```sh
https://github.com/xybrid-ai/xybrid.git#upm
```

**モデルを実行:**

```csharp
var model = XybridClient.LoadModel("kokoro-82m");
var result = model.Run(Envelope.Text("Hello world"));
// result → 24kHz WAVオーディオ
```

</details>

<details>
<summary><b>Rust</b> — すべてのプラットフォーム</summary>

**インストール** `Cargo.toml`:

```toml
[dependencies]
xybrid-sdk = "0.1"
```

**モデルを実行:**

```rust
let model = Xybrid::model("kokoro-82m").load()?;
let result = model.run(&Envelope::text("Hello world"))?;
// result → 24kHz WAVオーディオ
```

</details>

すべてのオプション、ハードウェアアクセラレーション、CLIリファレンスについては、完全な[インストールガイド](docs/INSTALLATION.md)を参照してください。プラットフォーム固有のセットアップについては、各SDKのREADMEを参照してください: [Flutter](bindings/flutter/) · [Unity](bindings/unity/) · [Swift](bindings/apple/) · [Kotlin](bindings/kotlin/) · [Rust](crates/)。

<details>
<summary><h3>パイプライン（実験的機能）</h3></summary>

モデルを連鎖させて、3行のYAMLで音声アシスタントを構築:

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny    # 音声 → テキスト
  - model: qwen2.5-0.5b    # LLMで処理
  - model: kokoro-82m      # テキスト → 音声
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
// パイプラインサポートは近日対応予定 — 現在は単一モデルの読み込みを使用してください
```

**Swift:**
```swift
// パイプラインサポートは近日対応予定 — 現在は単一モデルの読み込みを使用してください
```

**Unity (C#):**
```csharp
// パイプラインサポートは近日対応予定 — 現在は単一モデルの読み込みを使用してください
```

**Rust:**
```rust
let pipeline = Xybrid::pipeline(&yaml_string).load()?;
pipeline.load_models()?;
let result = pipeline.run(&Envelope::audio(audio_bytes))?;
```

</details>

---

## 対応モデル

すべてのモデルは完全にオンデバイスで実行されます。クラウド不要、APIキー不要です。完全なレジストリは `xybrid models list` で閲覧できます。

### まずはこれから

| モデル | タイプ | パラメータ数 | おすすめの理由 |
|-------|------|--------|----------------|
| **SmolLM2 360M** | LLM | 360M | あらゆるデバイスで最高の品質対サイズ比 |
| **Kokoro 82M** | TTS | 82M | 高品質な音声、24種類の声、高速 |
| **Whisper Tiny** | ASR | 39M | 正確な多言語文字起こし |

### 音声認識（Speech-to-Text）

| モデル | パラメータ数 | フォーマット | 説明 |
|-------|--------|--------|-------------|
| Whisper Tiny | 39M | SafeTensors | 多言語文字起こし（Candleランタイム） |
| Wav2Vec2 Base | 95M | ONNX | CTC復号による英語ASR |

### テキスト読み上げ（Text-to-Speech）

| モデル | パラメータ数 | フォーマット | 説明 |
|-------|--------|--------|-------------|
| Kokoro 82M | 82M | ONNX | 高品質、24種類の自然な声 |
| KittenTTS Nano | 15M | ONNX | 超軽量、8種類の声 |

### 言語モデル

| モデル | パラメータ数 | フォーマット | 説明 |
|-------|--------|--------|-------------|
| Gemma 3 1B | 1B | GGUF Q4_K_M | Googleのモバイル最適化LLM |
| LFM2.5 350M | 354M | GGUF Q4_K_M | Liquid AIのハイブリッドconv+attention、9言語、ツール呼び出し対応 |
| Llama 3.2 1B | 1B | GGUF Q4_K_M | Metaの汎用LLM、128Kコンテキスト |
| Qwen 2.5 0.5B | 500M | GGUF Q4_K_M | コンパクトなオンデバイスチャット |
| Qwen 3.5 0.8B | 800M | GGUF Q4_K_M | 推論機能付き最新Qwen（思考モード） |
| Qwen 3.5 2B | 2B | GGUF Q4_K_M | 拡張推論機能付き大型Qwen 3.5 |
| SmolLM2 360M | 360M | GGUF Q4_K_M | 最高の小型LLM、優れた品質/サイズ比 |

### 近日公開

| モデル | タイプ | パラメータ数 | 優先度 | ステータス |
|-------|------|--------|----------|--------|
| Phi-4 Mini | LLM | 3.8B | P2 | 仕様準備完了（初のマルチ量子化: Q4, Q8, FP16） |
| Qwen3 0.6B | LLM | 600M | P2 | 計画中 |
| Trinity Nano | LLM (MoE) | 6B (1Bアクティブ) | P2 | 計画中 |
| LFM2-VL 700M | Vision+LLM | 700M | P2 | 計画中 |
| Nomic Embed Text v1.5 | Embeddings | 137M | P1 | ブロック中（Tokenize/MeanPoolステップが必要） |
| LFM2-VL 450M | Vision | 450M | P2 | 計画中 |
| Whisper Tiny CoreML | ASR | 39M | P2 | 計画中 |
| Qwen3-TTS 0.6B | TTS | 600M | P2 | ブロック中（カスタムSafeTensorsランタイムが必要） |
| Chatterbox Turbo | TTS | 350M | P3 | ブロック中（ModelGraphテンプレートが必要） |

<details>
<summary><h3>独自モデルの使用（実験的機能）</h3></summary>

> **注意**: BYMサポートは実験的な機能です。`model_metadata.json`のスキーマは安定していますが、AI支援ツール（`/xybrid-init`）は現在活発に開発中であり、すべてのモデルタイプに対応しているとは限りません。

Xybridは**任意の**ONNX、GGUF、またはSafeTensorsモデルで動作します。必要なのは、xybridにモデルの実行方法を伝える`model_metadata.json`だけです。

**AIアシスタントを使用する場合**（Claude Code、Codexなど）:

```sh
# プロジェクトにxybridスキルをインストール
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/tools/scripts/install-skills.sh | sh

# HuggingFaceモデルからmodel_metadata.jsonを生成
claude /xybrid-init hexgrad/Kokoro-82M-v1.0-ONNX
```

スキルはエージェント非依存で、[`agents/skills/`](agents/skills/)に配置されます。インストーラーはClaude Code（`.claude/skills`）とCodex（`.codex/skills`）にシンボリックリンクを作成します。

**手動の場合** — モデルディレクトリに`model_metadata.json`を作成:

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

完全なスキーマについては[モデルメタデータドキュメント](docs/sdk/API_REFERENCE.md)を参照するか、[`integration-tests/fixtures/models/`](integration-tests/fixtures/models/)の既存の例をご覧ください。

</details>

---

## 機能

| 機能 | iOS | Android | macOS | Linux | Windows |
|------------|-----|---------|-------|-------|---------|
| 音声認識 | ✅ | ✅ | ✅ | ✅ | ✅ |
| テキスト読み上げ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 言語モデル | ✅ | ✅ | ✅ | ✅ | ✅ |
| 画像認識モデル | 🔜 | 🔜 | 🔜 | 🔜 | 🔜 |
| 埋め込み | 🔜 | 🔜 | 🔜 | 🔜 | 🔜 |
| パイプラインオーケストレーション | ✅ | ✅ | ✅ | ✅ | ✅ |
| モデルのダウンロードとキャッシュ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ハードウェアアクセラレーション | Metal, ANE | CPU | Metal, ANE | CUDA | CUDA |

**SDKパイプラインサポート:** Flutter ✅ · Rust ✅ · Kotlin 🔜 · Swift 🔜 · Unity 🔜

---

## なぜXybridなのか？

- **プライバシー最優先** — すべての推論はオンデバイスで実行。データがデバイスから出ることはありません。
- **オフライン対応** — 初回のモデルダウンロード後はインターネット不要。
- **クロスプラットフォーム** — iOS、Android、macOS、Linux、Windowsで統一されたAPI。
- **パイプラインオーケストレーション** — モデルを連鎖（ASR → LLM → TTS）して1回の呼び出しで実行。
- **自動最適化** — Apple Neural Engine、Metal、CUDAによるハードウェアアクセラレーション。

### 比較

| | Xybrid | Ollama | llama.cpp | ONNX Runtime |
|---|---|---|---|---|
| モバイル（iOS/Android） | ✅ | ❌ | ❌ | ✅ |
| ゲームエンジン（Unity） | ✅ | ❌ | ❌ | ❌ |
| マルチステージパイプライン | ✅ | ❌ | ❌ | ❌ |
| ASR + TTS + LLMを1つのSDKで | ✅ | ❌ | ❌ | ❌ |
| インプロセス実行（サーバー不要） | ✅ | ❌ | ✅ | ✅ |
| クラウド不要 | ✅ | ✅ | ✅ | ✅ |

---

## コミュニティ

- [ドキュメント](https://docs.xybrid.dev)
- [Discord](https://discord.gg/YhFHHkhbad)
- [X (Twitter)](https://x.com/xybrid_ai)
- [GitHub Issues](https://github.com/xybrid-ai/xybrid/issues)

## コントリビューション

コントリビューションを歓迎します！開発環境のセットアップ、プルリクエストの提出、新しいモデルの追加に関するガイドラインについては、[CONTRIBUTING.md](./CONTRIBUTING.md)を参照してください。

## スター履歴

[![Star History Chart](https://api.star-history.com/svg?repos=xybrid-ai/xybrid&type=date&legend=bottom-right)](https://www.star-history.com/#xybrid-ai/xybrid&type=date&legend=bottom-right)

## ライセンス

Apache License 2.0 — 詳細は[LICENSE](./LICENSE)を参照してください。

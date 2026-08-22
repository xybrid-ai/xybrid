# 安装

## 快速安装（推荐）

安装脚本会检测你的操作系统与架构，下载最新的发布版二进制文件，并将其添加到 PATH。

**macOS / Linux：**

```bash
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh
```

**Windows（PowerShell）：**

```powershell
irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
```

## 下载二进制文件

[Releases](https://github.com/xybrid-ai/xybrid/releases) 页面提供以下平台的预编译二进制文件：

| 平台 | 架构 | 二进制文件 |
|----------|-------------|--------|
| macOS | Apple Silicon（M1+） | `xybrid-v*-macos-arm64` |
| Linux | x86_64 | `xybrid-v*-linux-x86_64` |
| Windows | x86_64 | `xybrid-v*-windows-x86_64.exe` |

下载适用于你平台的二进制文件，赋予可执行权限，并将其移动到 PATH 中：

```bash
# 示例：macOS Apple Silicon
chmod +x xybrid-v*-macos-arm64
sudo mv xybrid-v*-macos-arm64 /usr/local/bin/xybrid
```

## 从源码安装

需要 [Rust 工具链](https://rustup.rs/)（1.75+）。

```bash
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli
```

### 平台特性

默认情况下，`cargo install` 构建时不包含硬件加速。为获得最佳性能，请添加平台特性：

```bash
# macOS — Metal GPU + Apple Neural Engine + llama.cpp
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli --features platform-macos

# Linux / Windows — ONNX + llama.cpp
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli --features platform-desktop
```

安装脚本提供的预编译二进制已经为每个平台包含了正确的特性。

<details>
<summary>所有可用的 Feature 标志</summary>

| Feature | 说明 |
|---------|-------------|
| **平台预设** | |
| `platform-macos` | ONNX 下载 + CoreML + Metal + 支持视觉的 llama.cpp + whisper.cpp 语音识别 |
| `platform-ios` | ONNX 下载 + CoreML + Metal + 支持视觉的 llama.cpp + whisper.cpp 语音识别 |
| `platform-android` | ONNX 动态加载 + 支持视觉的 llama.cpp + whisper.cpp 语音识别 |
| `platform-desktop` | ONNX 下载 + 支持视觉的 llama.cpp + whisper.cpp 语音识别 |
| **独立标志** | |
| `ort-download` | 下载预编译的 ONNX Runtime 二进制文件 |
| `ort-dynamic` | 在运行时加载 ONNX Runtime 的 .so |
| `ort-coreml` | Apple Neural Engine 加速 |
| `candle` | Candle ML 后端（Whisper 模型） |
| `candle-metal` | Candle 的 Metal 显卡加速 |
| `candle-cuda` | Candle 的 CUDA 显卡加速 |
| `llm-llamacpp` | llama.cpp 语言模型运行时后端（推荐） |
| `vision` | 图像 Envelope 与预处理支持 |
| `llm-llamacpp-vision` | llama.cpp 视觉语言支持（`mmproj` / `mtmd`） |
| `llm-mistral` | mistral.rs 语言模型后端（备选） |

</details>

### 从本地克隆构建

```bash
git clone https://github.com/xybrid-ai/xybrid.git
cd xybrid
cargo build --release -p xybrid-cli --features platform-macos
# 二进制文件位于 target/release/xybrid
```

## 验证安装

```bash
xybrid --help
```

预期输出：

```
Xybrid CLI - Run hybrid cloud-edge AI inference pipelines

Usage: xybrid [OPTIONS] <COMMAND>

Commands:
  init     Generate model_metadata.json by inspecting model files
  models   Manage models from the registry
  run      Run a pipeline, model, or GGUF file
  repl     Interactive REPL mode
  fetch    Pre-download models from the registry
  cache    Manage the local model cache
  ...
```

查看可用模型：

```bash
xybrid models list
```

## 快速开始

### 文字转语音 (TTS)

```bash
xybrid run --model kokoro-82m --input-text "Hello world" --output hello.wav
```

### 语音识别 (STT)

```bash
xybrid run --model whisper-tiny-ggml --input-audio recording.wav
```

`whisper-tiny-ggml` 是运行在 whisper.cpp 上的 GGML 模型包，所有平台预设都已内置。
旧的 `whisper-tiny` 对应 SafeTensors 模型包，需要使用 `--features candle` 重新构建才能运行。

### 与LLM对话

```bash
# 交互式对话（消息之间保持模型加载）
xybrid repl --model smollm2-360m --stream

# 单次推理
xybrid run --model smollm2-360m --input-text "What is the capital of France?"
```

### 视觉模型输入

视觉语言模型需要启用 `vision` 或 `llm-llamacpp-vision` 编译的版本，
以及包含视觉编码器构件的模型Bundle。仅使用平台预设时仅支持纯文本输入；
如需本地 VLM 生成，请将预设与 `llm-llamacpp-vision` 组合。

```bash
cargo build --release -p xybrid-cli --features platform-macos,llm-llamacpp-vision
```

```bash
# 单次视觉对话
xybrid run --model lfm2-vl-450m \
  --input-text "Describe this image" \
  --input-image photo.jpg

# 交互式视觉对话
xybrid repl --model lfm2-vl-450m --stream
/image photo.jpg
What is in this image?
```

### 运行HuggingFace上任意的GGUF

无需注册表条目 — 直接指向 HuggingFace 仓库：

```bash
xybrid run --huggingface "unsloth/SmolLM2-360M-Instruct-GGUF:Q4_K_M" \
  --input-text "Hello!"

# 交互式对话
xybrid repl --huggingface "unsloth/SmolLM2-360M-Instruct-GGUF:Q4_K_M" --stream
```

### 运行本地 GGUF 文件

```bash
xybrid run --model-file ./my-model.gguf --input-text "Hello!"
```

### 流水线

使用 YAML 文件将多个模型串联起来：

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny-ggml
  - model: smollm2-360m
  - model: kokoro-82m
```

```bash
xybrid run --config voice-assistant.yaml --input-audio question.wav --output response.wav
```

## CLI 参考

### 命令

| 命令 | 说明 |
|---------|-------------|
| `xybrid run` | 使用模型、流水线、Bundle 或 GGUF 文件运行推理 |
| `xybrid repl` | 交互式 REPL — 保持模型加载以快速重复推理 |
| `xybrid init` | 通过检查模型目录生成 `model_metadata.json` |
| `xybrid models list` | 列出注册表中的所有模型 |
| `xybrid models search <query>` | 按名称、任务或描述搜索模型 |
| `xybrid models info <id>` | 显示指定模型的详细信息 |
| `xybrid models voices <id>` | 列出 TTS 模型的可用语音 |
| `xybrid fetch --model <id>` | 从注册表预下载模型 |
| `xybrid fetch --huggingface <repo>` | 从 HuggingFace 预下载模型 |
| `xybrid cache list` | 列出已缓存的模型 |
| `xybrid cache status` | 显示缓存大小与统计信息 |
| `xybrid cache clear [id]` | 清除已缓存的模型（全部或指定） |
| `xybrid prepare <file>` | 校验流水线 YAML |
| `xybrid plan <file>` | 显示流水线的执行计划 |
| `xybrid bundle <model>` | 拉取模型并创建 `.xyb` Bundle |
| `xybrid pack <name>` | 将本地模型构件打包为 `.xyb` Bundle |
| `xybrid trace` | 查看并分析历史会话的遥测数据 |

### 全局标志

| 标志 | 说明 |
|------|-------------|
| `-v`、`-vv` | 提升输出详细程度 |
| `-q`、`--quiet` | 抑制输出，仅显示错误 |
| `--api-key` | 用于遥测的平台 API Key（或使用 `XYBRID_API_KEY` 环境变量） |

### `run` 输入源

`run` 命令接受多种输入源（互斥）：

| 标志 | 说明 | 示例 |
|------|-------------|---------|
| `--model <id>` | 注册表模型 | `--model kokoro-82m` |
| `--config <file>` | 流水线 YAML | `--config pipeline.yaml` |
| `--pipeline <name>` | 内置流水线 | `--pipeline hiiipe` |
| `--bundle <file>` | `.xyb` Bundle | `--bundle model.xyb` |
| `--directory <dir>` | 本地模型目录 | `--directory ./my-model/` |
| `--huggingface <repo>` | HuggingFace 仓库 | `--huggingface "org/model:Q4_K_M"` |
| `--model-file <path>` | 本地 GGUF 文件 | `--model-file model.gguf` |

### `run` 选项

| 标志 | 说明 |
|------|-------------|
| `--input-text <text>` | 文本输入（用于 TTS、LLM） |
| `--input-audio <file>` | 音频输入 WAV 文件（用于 ASR） |
| `--input-image <file>` | 视觉语言模型的图像输入；可重复 |
| `--voice <id>` | TTS 语音 ID（如 `af_bella`） |
| `--output <file>` | 输出文件（音频为 .wav，文本为 .txt） |
| `--target <format>` | 目标格式（onnx、coreml、tflite） |
| `--dry-run` | 仅校验，不执行 |
| `--trace` | 启用执行追踪 |
| `--trace-export <file>` | 将追踪数据导出为 JSON（Chrome trace 格式） |

### `repl` 选项

| 标志 | 说明 |
|------|-------------|
| `--model <id>` | 要加载的注册表模型 |
| `--huggingface <repo>` | 要加载的 HuggingFace 模型 |
| `--model-file <path>` | 要加载的本地 GGUF 文件 |
| `--stream` | 生成时流式输出词元（token）（LLM） |
| `--system <prompt>` | 对话的系统提示词 |
| `--voice <id>` | TTS 语音 ID |

## 模型缓存

下载的模型缓存在 `~/.xybrid/cache/`。管理方式：

```bash
xybrid cache status    # 显示缓存大小
xybrid cache list      # 列出已缓存的模型
xybrid cache clear     # 清除全部
xybrid cache clear kokoro-82m  # 清除指定模型
```

## 卸载

**通过脚本安装：**

```bash
rm $(which xybrid)
# （可选）删除缓存
rm -rf ~/.xybrid
```

**通过 cargo 安装：**

```bash
cargo uninstall xybrid-cli
rm -rf ~/.xybrid
```

**Windows：**

```powershell
Remove-Item "$env:USERPROFILE\.xybrid" -Recurse -Force
```

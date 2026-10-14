# model_metadata.json 参考

> 每个模型目录都必须包含一个 `model_metadata.json` 文件，用于告诉 xybrid 如何执行该模型。
>
> **JSON Schema**：[`model_metadata.schema.json`](model_metadata.schema.json) — 配合编辑器使用，可获得验证和自动补全。

## 顶层结构

```json
{
  "model_id": "string (required)",
  "version": "string (required)",
  "description": "string (optional)",
  "execution_template": { "type": "..." },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["model.onnx"],
  "metadata": {},
  "voices": null,
  "max_chunk_chars": null,
  "trim_trailing_samples": null
}
```

| 字段 | 类型 | 必填 | 说明 |
|-------|------|----------|-------------|
| `model_id` | string | 是 | 唯一的模型标识符（如 `"kokoro-82m"`） |
| `version` | string | 是 | 模型版本（如 `"1.0"`） |
| `description` | string | 否 | 供人阅读的描述 |
| `execution_template` | object | 是 | 定义模型格式和运行时 — 参见[执行模板](#执行模板) |
| `preprocessing` | array | 否 | 在模型执行前应用于输入的步骤 — 参见[预处理步骤](#预处理步骤) |
| `postprocessing` | array | 否 | 应用于模型输出的步骤 — 参见[后处理步骤](#后处理步骤) |
| `files` | string[] | 是 | 模型 Bundle 中的所有文件（用于缓存校验） |
| `metadata` | object | 否 | 任意的键值元数据（任务、架构、许可证等） |
| `voices` | object | 否 | TTS 模型的音色配置 — 参见[音色配置](#音色配置) |
| `max_chunk_chars` | integer | 否 | 每个 TTS 分块的最大字符数（默认 350） |
| `trim_trailing_samples` | integer | 否 | 每个 TTS 分块要裁剪的音频采样点数（默认 0） |

---

## 执行模板

`execution_template` 字段定义要加载哪种模型格式以及使用哪个运行时。每种类型都由 `"type"` 判别字段标识。

### `Onnx`

通过 ONNX Runtime 执行 ONNX 模型。这是最常见的格式。

```json
{
  "type": "Onnx",
  "model_file": "model.onnx"
}
```

| 字段 | 类型 | 必填 | 说明 |
|-------|------|----------|-------------|
| `model_file` | string | 是 | `.onnx` 文件的路径（相对于模型目录） |

### `Gguf`

通过 llama.cpp 执行 GGUF 模型，用于本地 LLM。

```json
{
  "type": "Gguf",
  "model_file": "model-Q4_K_M.gguf",
  "chat_template": null,
  "context_length": 4096
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|-------|------|----------|---------|-------------|
| `model_file` | string | 是 | — | `.gguf` 文件的路径 |
| `chat_template` | string | 否 | `null` | 聊天模板 JSON 文件的路径 |
| `context_length` | integer | 否 | `4096` | 最大上下文长度（以 token 计） |

### `SafeTensors`

通过 Candle（纯 Rust）执行 SafeTensors 模型。用于 Whisper 模型。

```json
{
  "type": "SafeTensors",
  "model_file": "model.safetensors",
  "architecture": "whisper",
  "config_file": "config.json",
  "tokenizer_file": "tokenizer.json"
}
```

| 字段 | 类型 | 必填 | 说明 |
|-------|------|----------|-------------|
| `model_file` | string | 是 | `.safetensors` 文件的路径 |
| `architecture` | string | 否 | 用于路由的模型架构（如 `"whisper"`） |
| `config_file` | string | 否 | 模型配置 JSON 的路径 |
| `tokenizer_file` | string | 否 | 分词器 JSON 的路径 |

### `CoreMl`

执行 CoreML 模型（仅限 Apple 平台）。

```json
{
  "type": "CoreMl",
  "model_file": "model.mlmodel"
}
```

### `TfLite`

执行 TensorFlow Lite 模型（移动端）。

```json
{
  "type": "TfLite",
  "model_file": "model.tflite"
}
```

### `LiteRtLm`

旧版 LiteRT-LM 元数据，为保持序列化兼容性而保留。当前的浏览器 SDK（`@xybrid/web`）使用 `Gguf` 并拒绝此模板；原生运行时会以明确的错误拒绝此模板。

```json
{
  "type": "LiteRtLm",
  "model_file": "model.litertlm",
  "context_length": 2048
}
```

| 字段 | 类型 | 必填 | 说明 |
|-------|------|----------|-------------|
| `model_file` | string | 是 | `.litertlm` 文件的路径（相对于模型目录） |
| `context_length` | integer | 否 | 旧版上下文长度（以 token 计）；未设置时省略。若存在，必须至少为 1 |

### `ModelGraph`

多模型 DAG 执行（单个 Bundle 内由多个模型组成的流水线）。

```json
{
  "type": "ModelGraph",
  "stages": [
    {
      "name": "encoder",
      "model_file": "encoder.onnx",
      "execution_mode": { "type": "SingleShot" },
      "inputs": ["audio"],
      "outputs": ["encoder_output"],
      "config": {}
    },
    {
      "name": "decoder",
      "model_file": "decoder.onnx",
      "execution_mode": {
        "type": "Autoregressive",
        "max_tokens": 256,
        "start_token_id": 1,
        "end_token_id": 2,
        "repetition_penalty": 1.1
      },
      "inputs": ["encoder_output"],
      "outputs": ["text"],
      "config": {}
    }
  ],
  "config": {}
}
```

#### 执行模式

`ModelGraph` 中的每个阶段都有一个 `execution_mode`：

| 模式 | 说明 | 适用场景 |
|------|-------------|----------|
| `SingleShot` | 运行一次模型（默认） | 编码器、分类器 |
| `Autoregressive` | 循环运行，直到出现结束 token | 文本生成、解码 |
| `WhisperDecoder` | Whisper 专用，带 KV 缓存 | Whisper ASR |
| `IterativeRefinement` | 按调度计划执行多遍 | 扩散 (Diffusion) 模型 |

---

## 预处理步骤

预处理在模型执行前对输入数据进行转换。每个步骤由 `"type"` 标识。

### `AudioDecode`

将原始 WAV 字节转换为 PCM float32 采样点。

```json
{ "type": "AudioDecode", "sample_rate": 16000, "channels": 1 }
```

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `sample_rate` | integer | 目标采样率（Hz） |
| `channels` | integer | 声道数（1 = 单声道，2 = 立体声） |

**输入**：`Envelope::Audio(wav_bytes)` → **输出**：Float32 PCM 采样点

### `Phonemize`

将文本转换为音素 token ID，供 TTS 模型使用。

```json
{
  "type": "Phonemize",
  "tokens_file": "tokens.txt",
  "backend": "MisakiDictionary",
  "dict_file": null,
  "language": null,
  "add_padding": true,
  "normalize_text": false,
  "silence_tokens": null
}
```

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `tokens_file` | string | — | token 到 ID 映射文件的路径 |
| `backend` | string | `"MisakiDictionary"` | 音素化后端（见下文） |
| `dict_file` | string | `null` | 词典文件路径（仅限 CmuDictionary） |
| `language` | string | `null` | EspeakNG 的语言代码（如 `"en-us"`） |
| `add_padding` | boolean | `true` | 在开头/结尾添加填充 token |
| `normalize_text` | boolean | `false` | 在音素化之前进行文本清理 |
| `silence_tokens` | integer | `null` | 要在开头添加的静音 token 数量（使爆破音更平滑） |

**音素化后端**：

| 后端 | 依赖 | 说明 |
|---------|-------------|-------|
| `MisakiDictionary` | 无（纯 Rust） | **推荐**。随 Bundle 打包的词典 + 基于规则的 G2P 回退 |
| `CmuDictionary` | 无（纯 Rust） | 旧版，基于 ARPABET，仅支持英语 |
| `EspeakNG` | 系统中安装的 `espeak-ng` | 多语言支持 |
| `OpenPhonemizer` | ONNX 模型（~59MB） | 词典 + 神经网络 G2P 回退 |

**输入**：`Envelope::Text(string)` → **输出**：token ID（i64）

### `Tokenize`

使用词汇表文件对文本进行分词。

```json
{ "type": "Tokenize", "vocab_file": "vocab.json", "tokenizer_type": "BPE", "max_length": 512 }
```

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `vocab_file` | string | — | 词汇表文件的路径 |
| `tokenizer_type` | string | — | `"BPE"`、`"WordPiece"` 或 `"SentencePiece"` |
| `max_length` | integer | `null` | 最大序列长度（超出时截断） |

**输入**：`Envelope::Text(string)` → **输出**：token ID

### `MelSpectrogram`

将音频转换为梅尔频谱图，供 ASR 模型使用。

```json
{
  "type": "MelSpectrogram",
  "preset": "whisper",
  "n_mels": 80,
  "sample_rate": 16000,
  "fft_size": 400,
  "hop_length": 160,
  "mel_scale": "slaney",
  "max_frames": 3000
}
```

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `preset` | string | `null` | 使用预设配置：`"whisper"`、`"whisper-large"` |
| `n_mels` | integer | `80` | 梅尔频带数量 |
| `sample_rate` | integer | `16000` | 音频采样率（Hz） |
| `fft_size` | integer | `400` | FFT 窗口大小 |
| `hop_length` | integer | `160` | 相邻帧之间的步长 |
| `mel_scale` | string | `"slaney"` | 梅尔刻度类型：`"slaney"` 或 `"htk"` |
| `max_frames` | integer | `3000` | 最大输出帧数（3000 = 100fps 下的 30 秒） |

### `Normalize`

按通道对张量值进行归一化：`(x - mean) / std`。

```json
{ "type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] }
```

### `Resize`

将图像缩放到目标尺寸。

```json
{ "type": "Resize", "width": 224, "height": 224, "interpolation": "Bilinear" }
```

插值方法：`"Nearest"`、`"Bilinear"`（默认）、`"Bicubic"`。

### `CenterCrop`

将图像中心裁剪到目标尺寸。

```json
{ "type": "CenterCrop", "width": 224, "height": 224 }
```

### `Reshape`

将张量重塑为目标维度。

```json
{ "type": "Reshape", "shape": [1, 1, 28, 28] }
```

---

## 后处理步骤

后处理将模型输出转换为最终结果。

### `CTCDecode`

用于 ASR 模型的 CTC（Connectionist Temporal Classification）解码。

```json
{ "type": "CTCDecode", "vocab_file": "vocab.json", "blank_index": 0 }
```

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `vocab_file` | string | — | 词汇表文件的路径 |
| `blank_index` | integer | `0` | 空白 token 的索引 |

**输入**：logits 张量 → **输出**：`Envelope::Text(decoded_string)`

### `TTSAudioEncode`

将 TTS 输出的 float32 波形转换为 PCM 音频字节。

```json
{ "type": "TTSAudioEncode", "sample_rate": 24000, "apply_postprocessing": true, "trim_trailing_silence": false }
```

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `sample_rate` | integer | — | 输出采样率（Hz） |
| `apply_postprocessing` | boolean | `true` | 应用归一化和静音裁剪 |
| `trim_trailing_silence` | boolean | `false` | 裁剪波形末尾接近静音的部分 |

**输入**：Float32 波形 → **输出**：`Envelope::Audio(pcm_bytes)`

波形是模型的唯一输出；如果模型有多个输出，则是其中名为 `waveform`（优先）或 `audio` 的那个。例如，KittenTTS 0.8 还会返回每个音素的 `duration`。如果 TTS 模型有多个输出，却没有这两个名称中的任何一个，就会运行失败。

### `WhisperDecode`

使用 HuggingFace 分词器将 Whisper token ID 解码为文本。

```json
{ "type": "WhisperDecode", "tokenizer_file": "tokenizer.json" }
```

### `BPEDecode`

将 BPE token 解码为文本。

```json
{ "type": "BPEDecode", "vocab_file": "vocab.json" }
```

### `Argmax`

返回最大值的索引（类别预测）。

```json
{ "type": "Argmax", "dim": null }
```

### `Softmax`

应用 softmax 以得到概率。

```json
{ "type": "Softmax", "dim": 1 }
```

### `TopK`

返回 top-K 预测及其分数。

```json
{ "type": "TopK", "k": 5, "dim": null }
```

### `Threshold`

应用阈值，将概率转换为二值预测。

```json
{ "type": "Threshold", "threshold": 0.5, "return_indices": false }
```

### `TemperatureSample`

在生成 token 时应用温度采样。

```json
{ "type": "TemperatureSample", "temperature": 0.7, "top_k": 40, "top_p": 0.9 }
```

### `Denormalize`

`Normalize` 的逆操作：`x * std + mean`。

```json
{ "type": "Denormalize", "mean": [0.5], "std": [0.5] }
```

### `MeanPool`

对 token 嵌入做均值池化，生成句子嵌入。

```json
{ "type": "MeanPool", "dim": 1 }
```

---

## 音色配置

TTS 模型可以包含一个 `voices` 部分，用于定义可用的音色。

```json
{
  "voices": {
    "format": "embedded",
    "file": "voices.bin",
    "loader": "binary_f32_256",
    "default": "af_heart",
    "selection_strategy": "FixedIndex",
    "catalog": [
      {
        "id": "af_heart",
        "name": "Heart",
        "gender": "female",
        "language": "en-US",
        "style": "neutral",
        "index": 0
      }
    ]
  }
}
```

### 音色格式

| 格式 | 说明 | 示例模型 |
|--------|-------------|---------------|
| `embedded` | 所有音色都在单个二进制文件中 | Kokoro、KittenTTS |
| `per_model` | 每个音色是一个单独的模型文件 | Piper |
| `cloning` | 基于参考音频的声音克隆 | （规划中） |

### 音色加载器（用于 `embedded` 格式）

| 加载器 | 说明 |
|--------|-------------|
| `binary_f32_256` | 连续的 f32 数组，256 维（每个音色 1024 字节） |
| `numpy_npz` | NumPy .npz 归档文件 |
| `json_base64` | 包含 base64 编码嵌入的 JSON（规划中） |

### 音色选择策略

| 策略 | 说明 |
|----------|-------------|
| `FixedIndex` | 按 catalog 索引选择（默认） |
| `TokenLength` | 按音素 token 数量从音色包中选择（Kokoro） |

---

## 元数据约定

`metadata` 字段格式自由，但以下键是约定俗成的：

| 键 | 类型 | 说明 |
|-----|------|-------------|
| `task` | string | 模型任务：`"text-to-speech"`、`"speech-recognition"`、`"text-generation"`、`"image_classification"`、`"embedding"` |
| `architecture` | string | 模型架构（如 `"qwen35"`、`"whisper"`） |
| `backend` | string | 推理后端（如 `"llamacpp"`、`"onnxruntime"`） |
| `parameters` | integer | 参数量 |
| `license` | string | 模型许可证（如 `"Apache-2.0"`） |
| `sample_rate` | integer | 音频采样率（音频模型） |
| `family` | string | 模型家族或组织 |
| `source` | string | 来源 URL（如 HuggingFace 仓库） |
| `ram_minimum_mb` | integer | 最低内存（MB） |
| `ram_recommended_mb` | integer | 推荐内存（MB） |

---

## 完整示例

### ONNX 分类（MNIST）

```json
{
  "model_id": "mnist-digit-recognition",
  "version": "12",
  "description": "MNIST handwritten digit recognition. Input: [1,1,28,28] grayscale. Output: [1,10] probabilities.",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    { "type": "Reshape", "shape": [1, 1, 28, 28] },
    { "type": "Normalize", "mean": [0.0], "std": [255.0] }
  ],
  "postprocessing": [
    { "type": "Softmax", "dim": 1 }
  ],
  "files": ["model.onnx"],
  "metadata": {
    "task": "image_classification",
    "num_classes": 10
  }
}
```

### 带 Phonemize 的 ONNX TTS（Kokoro）

```json
{
  "model_id": "kokoro-82m",
  "version": "1.0",
  "description": "Kokoro 82M - High-quality TTS with 24 voices",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    {
      "type": "Phonemize",
      "tokens_file": "tokens.txt",
      "backend": "MisakiDictionary",
      "add_padding": true,
      "normalize_text": true
    }
  ],
  "postprocessing": [
    { "type": "TTSAudioEncode", "sample_rate": 24000, "apply_postprocessing": true }
  ],
  "files": ["model.onnx", "voices.bin", "tokens.txt", "misaki/us_gold.json"],
  "voices": {
    "format": "embedded",
    "file": "voices.bin",
    "loader": "binary_f32_256",
    "default": "af_heart",
    "selection_strategy": "TokenLength",
    "catalog": [
      { "id": "af_heart", "name": "Heart", "gender": "female", "language": "en-US", "index": 0 }
    ]
  },
  "metadata": {
    "task": "text-to-speech",
    "parameters": 82000000,
    "sample_rate": 24000
  }
}
```

### GGUF LLM（Qwen）

```json
{
  "model_id": "qwen3.5-0.8b",
  "version": "1.0",
  "description": "Qwen 3.5 0.8B - Lightweight LLM for on-device text generation",
  "execution_template": {
    "type": "Gguf",
    "model_file": "Qwen3.5-0.8B-Q4_K_M.gguf",
    "context_length": 4096
  },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["Qwen3.5-0.8B-Q4_K_M.gguf"],
  "metadata": {
    "task": "text-generation",
    "architecture": "qwen35",
    "backend": "llamacpp",
    "parameters": 800000000,
    "license": "Apache-2.0",
    "ram_minimum_mb": 512,
    "ram_recommended_mb": 1024
  }
}
```

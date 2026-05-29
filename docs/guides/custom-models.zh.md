# 自定义模型指南

> 使用 `fromDirectory()` 在 xybrid 中加载你自己的 ONNX、GGUF 或 SafeTensors 模型。

本指南介绍如何加载不在 xybrid 注册表中的模型。完成后，你将拥有一个可用的 `model_metadata.json`，并能通过 `fromDirectory()` 运行推理。

## 概览

工作流程如下：

1. **准备**一个包含模型文件的目录
2. **编写** `model_metadata.json`，描述输入、输出和处理步骤
3. **使用** `fromDirectory()` 在任意 SDK（Rust、Flutter、Kotlin、Swift、Unity）中加载
4. **通过**标准 `model.run()` API 运行推理

## 第一步：准备模型目录

创建一个包含模型及其支持文件的目录：

```
my-custom-model/
├── model_metadata.json    # 必填 — 告诉 xybrid 如何运行模型
├── model.onnx             # 模型文件（ONNX、GGUF 等）
├── tokenizer.json         # 可选 — 若模型需要分词
└── vocab.txt              # 可选 — 若前/后处理需要
```

`model_metadata.json` 中引用的所有文件必须位于此目录中。

## 第二步：编写 model_metadata.json

此文件告诉 xybrid 模型格式、预处理步骤（在模型推理前对输入的处理）和后处理步骤（对模型输出的处理）。

### 最简示例：ONNX 分类模型

对于 ONNX 图像分类模型（例如来自 [ONNX Model Zoo](https://github.com/onnx/models) 的 MobileNet）：

```json
{
  "model_id": "my-mobilenet",
  "version": "1.0",
  "description": "MobileNetV2 image classification",
  "execution_template": {
    "type": "Onnx",
    "model_file": "mobilenetv2.onnx"
  },
  "preprocessing": [
    { "type": "Reshape", "shape": [1, 3, 224, 224] },
    { "type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] }
  ],
  "postprocessing": [
    { "type": "Softmax", "dim": 1 },
    { "type": "TopK", "k": 5 }
  ],
  "files": ["mobilenetv2.onnx"],
  "metadata": {
    "task": "image_classification"
  }
}
```

### ONNX Token 分类（NER / PII 检测）

对于经过命名实体识别微调的 DistilBERT 等模型：

```json
{
  "model_id": "my-ner-model",
  "version": "1.0",
  "description": "NER token classification",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    {
      "type": "Tokenize",
      "vocab_file": "tokenizer.json",
      "tokenizer_type": "WordPiece",
      "max_length": 512
    }
  ],
  "postprocessing": [
    { "type": "ArgMax" }
  ],
  "files": ["model.onnx", "tokenizer.json"],
  "metadata": {
    "task": "token_classification",
    "architecture": "DistilBERT"
  }
}
```

### GGUF LLM（文本生成）

对于来自 HuggingFace 的量化 LLM（如 Qwen、Llama、Mistral）：

```json
{
  "model_id": "my-llm",
  "version": "1.0",
  "description": "Qwen 3.5 0.8B — lightweight text generation",
  "execution_template": {
    "type": "Gguf",
    "model_file": "model-Q4_K_M.gguf",
    "context_length": 4096
  },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["model-Q4_K_M.gguf"],
  "metadata": {
    "task": "text-generation",
    "architecture": "qwen35",
    "backend": "llamacpp"
  }
}
```

GGUF 模型通过 llama.cpp 内部处理分词，无需预处理步骤。

### 句子嵌入

对于Sentence Transformer模型（如 all-MiniLM-L6-v2）：

```json
{
  "model_id": "my-embeddings",
  "version": "1.0",
  "description": "Sentence embeddings (384 dimensions)",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    {
      "type": "Tokenize",
      "vocab_file": "tokenizer.json",
      "tokenizer_type": "WordPiece",
      "max_length": 512
    }
  ],
  "postprocessing": [
    { "type": "MeanPool", "dim": 1 }
  ],
  "files": ["model.onnx", "tokenizer.json"],
  "metadata": {
    "task": "sentence_embedding",
    "architecture": "BertModel"
  }
}
```

完整字段参考见 [MODEL_METADATA.md](../sdk/MODEL_METADATA.md)。如需编辑器自动补全和验证，请使用 [JSON Schema](../sdk/model_metadata.schema.json)。

## 第三步：使用 fromDirectory() 加载

### Rust

```rust
use xybrid_sdk::ModelLoader;
use xybrid_core::ir::{Envelope, EnvelopeKind};

let loader = ModelLoader::from_directory("/path/to/my-custom-model")?;
let model = loader.load()?;

let input = Envelope {
    kind: EnvelopeKind::Text("Hello, world!".into()),
    metadata: Default::default(),
};
let output = model.run(&input)?;
```

### Flutter (Dart)

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

final loader = XybridModelLoader.fromDirectory('/path/to/my-custom-model');
final model = await loader.load();

final result = await model.run(
  envelope: XybridEnvelope.text(text: 'Hello, world!'),
);
```

### Kotlin (Android)

```kotlin
import ai.xybrid.ModelLoader

val loader = ModelLoader.fromDirectory("/path/to/my-custom-model")
val model = loader.load()

val result = model.run(XybridEnvelope.text("Hello, world!"))
```

### Swift (iOS / macOS)

```swift
import Xybrid

let loader = try ModelLoader.fromDirectory(path: "/path/to/my-custom-model")
let model = try loader.load()

let result = try model.run(envelope: .text("Hello, world!"))
```

### Unity (C#)

```csharp
using Xybrid;

var loader = ModelLoader.FromDirectory("/path/to/my-custom-model");
var model = loader.Load();

var result = model.Run(Envelope.Text("Hello, world!"));
```

## 各平台使用模式

### Flutter — 打包资源

将模型目录打包到 Flutter 应用的 assets 中：

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/my-model/model_metadata.json
    - assets/models/my-model/model.onnx
```

运行时，将 assets 复制到可写目录（assets 是只读的）：

```dart
import 'dart:io';
import 'package:path_provider/path_provider.dart';

Future<String> prepareModel() async {
  final appDir = await getApplicationSupportDirectory();
  final modelDir = Directory('${appDir.path}/models/my-model');

  if (!await modelDir.exists()) {
    await modelDir.create(recursive: true);
    // 将每个文件从 assets 复制到目录
    for (final file in ['model_metadata.json', 'model.onnx']) {
      final data = await rootBundle.load('assets/models/my-model/$file');
      await File('${modelDir.path}/$file')
          .writeAsBytes(data.buffer.asUint8List());
    }
  }

  return modelDir.path;
}

// 然后加载
final path = await prepareModel();
final loader = XybridModelLoader.fromDirectory(path);
```

### Android — 应用本地存储

将模型放在 `assets/` 中或下载到内部存储：

```kotlin
// 从 assets 复制到内部存储
val modelDir = File(context.filesDir, "models/my-model")
if (!modelDir.exists()) {
    modelDir.mkdirs()
    listOf("model_metadata.json", "model.onnx").forEach { filename ->
        context.assets.open("models/my-model/$filename").use { input ->
            File(modelDir, filename).outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }
}

val loader = ModelLoader.fromDirectory(modelDir.absolutePath)
```

### iOS / macOS — 应用包

将模型文件作为文件夹引用添加到 Xcode 项目：

```swift
// 添加到 Xcode 项目的模型位于应用包中
guard let modelPath = Bundle.main.path(forResource: "my-model", ofType: nil) else {
    fatalError("Model directory not found in bundle")
}

let loader = try ModelLoader.fromDirectory(path: modelPath)
```

### Unity — StreamingAssets

将模型文件放在 `Assets/StreamingAssets/` 中：

```
Assets/
└── StreamingAssets/
    └── models/
        └── my-model/
            ├── model_metadata.json
            └── model.onnx
```

```csharp
// StreamingAssets 路径因平台而异
var modelPath = Path.Combine(Application.streamingAssetsPath, "models", "my-model");
var loader = ModelLoader.FromDirectory(modelPath);
```

### Tauri — 应用资源

在 Tauri 应用中，使用资源目录：

```rust
use tauri::Manager;

let app_handle = app.handle();
let resource_dir = app_handle.path().resource_dir()
    .expect("failed to get resource dir");
let model_path = resource_dir.join("models").join("my-model");

let loader = ModelLoader::from_directory(model_path.to_str().unwrap())?;
```

在 `tauri.conf.json` 中添加模型目录：

```json
{
  "bundle": {
    "resources": ["models/my-model/*"]
  }
}
```

## 实战示例：加载 HuggingFace ONNX 模型

以加载 [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 为例。

### 1. 下载模型文件

```bash
# 创建模型目录
mkdir -p my-minilm/

# 从 HuggingFace 下载（ONNX 格式）
curl -L "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" \
  -o my-minilm/model.onnx

# 下载分词器
curl -L "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json" \
  -o my-minilm/tokenizer.json
```

### 2. 编写 model_metadata.json

```bash
cat > my-minilm/model_metadata.json << 'EOF'
{
  "model_id": "my-minilm",
  "version": "1.0",
  "description": "all-MiniLM-L6-v2 sentence embeddings",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    {
      "type": "Tokenize",
      "vocab_file": "tokenizer.json",
      "tokenizer_type": "WordPiece",
      "max_length": 256
    }
  ],
  "postprocessing": [
    { "type": "MeanPool", "dim": 1 }
  ],
  "files": ["model.onnx", "tokenizer.json"],
  "metadata": {
    "task": "sentence_embedding",
    "hidden_size": 384
  }
}
EOF
```

### 3. 运行推理

```rust
use xybrid_sdk::ModelLoader;
use xybrid_core::ir::{Envelope, EnvelopeKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loader = ModelLoader::from_directory("my-minilm")?;
    let model = loader.load()?;

    let input = Envelope {
        kind: EnvelopeKind::Text("The quick brown fox jumps.".into()),
        metadata: Default::default(),
    };

    let output = model.run(&input)?;

    // output.kind 为 EnvelopeKind::Embedding(Vec<f32>)，共 384 维
    if let EnvelopeKind::Embedding(embedding) = &output.kind {
        println!("Embedding dimensions: {}", embedding.len());
        println!("First 5 values: {:?}", &embedding[..5]);
    }

    Ok(())
}
```

## 故障排查

### "DirectoryNotFound" 错误

传入 `fromDirectory()` 的路径在磁盘上不存在。

- 仔细检查路径是否为绝对路径（而非相对路径）
- 在移动端，确认模型已复制到可写目录——应用 assets 通常是只读的

### "MetadataNotFound" 错误

目录存在，但不包含 `model_metadata.json` 文件。

- 确保文件名完全为 `model_metadata.json`（区分大小写）
- 检查该文件是否随模型一起复制（尤其是打包资源时）

### "MetadataInvalid" 错误

`model_metadata.json` 存在，但包含无效 JSON 或不符合预期的 schema。

- 验证 JSON 语法（尾随逗号、缺少引号等）
- 使用 [JSON Schema](../sdk/model_metadata.schema.json) 进行编辑器验证——大多数编辑器支持 `$schema` 引用：
  ```json
  {
    "$schema": "../docs/sdk/model_metadata.schema.json",
    "model_id": "...",
    ...
  }
  ```
- 确保 `execution_template.type` 为以下之一：`Onnx`、`SimpleMode`、`Gguf`、`SafeTensors`、`CoreMl`、`TfLite`、`ModelGraph`

### 形状不匹配 / 输出错误

模型可以运行，但输出为乱码或报形状错误。

- **预处理不匹配**：`preprocessing` 步骤必须将输入转换为模型期望的精确格式。请查阅模型文档以了解期望的输入张量形状和数据类型。
- **后处理不匹配**：`postprocessing` 步骤必须与模型的输出格式匹配。例如，对分类模型使用 `CTCDecode` 会失败。
- **模型文件错误**：确认 `execution_template.model_file` 指向目录中正确的文件。

### 运行时找不到文件

模型加载成功，但尝试读取词汇表或分词器文件时失败。

- 预处理/后处理中引用的每个文件（如 `vocab_file`、`tokens_file`、`tokenizer.json`）都必须在模型目录中
- 每个文件都必须列在 `files` 数组中

### GGUF 模型无法加载

- 确保 xybrid 在构建时包含 `llm-llamacpp` feature（所有平台预设均已包含）
- GGUF 模型需要在执行模板中使用 `"type": "Gguf"`，而非 `"Onnx"`
- 在执行模板中设置 `context_length` 以控制内存用量

## 快速参考：任务 → 配置

使用此表查找适合你模型类型的配置。选择与你的任务匹配的行，然后复制对应的预处理和后处理步骤。

| 任务 | 格式 | 执行模板 | 预处理 | 后处理 |
| ---- | ------ | ------------------ | ------------- | -------------- |
| **图像分类** | ONNX | `Onnx` | `Reshape` → `Normalize` | `Softmax` 或 `ArgMax` |
| **语音识别（CTC）** | ONNX | `Onnx` | `AudioDecode` | `CTCDecode` |
| **语音识别（Whisper）** | SafeTensors | `SafeTensors` | _(内置)_ | _(内置)_ |
| **文字转语音** | ONNX | `Onnx` | `Phonemize` | `TTSAudioEncode` |
| **文本生成（LLM）** | GGUF | `Gguf` | _(无)_ | _(无)_ |
| **句子嵌入** | ONNX | `Onnx` | `Tokenize` | `MeanPool` |
| **文本分类 / NER** | ONNX | `Onnx` | `Tokenize` | `ArgMax` |
| **目标检测 / 视觉** | ONNX | `Onnx` | `Resize` → `Normalize` | `TopK` 或 `Threshold` |

**关键规则：**

- **GGUF 模型**内部处理分词——预处理和后处理始终留空
- **SafeTensors（Whisper）**内置解码——设置 `architecture: "whisper"`，xybrid 会处理其余部分
- **ONNX 模型**需要显式的预处理/后处理，以匹配模型期望的输入/输出格式

## 使用 AI 助手配置模型执行

如果你有 ONNX 或 GGUF 模型，但不确定需要哪些预处理/后处理步骤，可以使用 AI Agent（Claude、Cursor、Copilot 等）来解决。Agent可以检查你的模型文件、阅读 HuggingFace 模型卡，并为你生成可用的 `model_metadata.json`。

### AI Agent需要哪些信息

向 AI Agent提供以下信息：

| 信息 | 用途 | 获取方式 |
| ----------- | --------------- | ------------- |
| **模型文件格式** | 确定 `execution_template.type` | 检查文件扩展名：`.onnx`、`.gguf`、`.safetensors` |
| **任务类型** | 确定预处理/后处理步骤 | 查看 HuggingFace 模型卡的 `pipeline_tag` |
| **输入张量名称和形状** | 正确预处理所需 | ONNX：`python -c "import onnx; m = onnx.load('model.onnx'); print([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])"` |
| **输出张量名称和形状** | 正确后处理所需 | ONNX：`python -c "import onnx; m = onnx.load('model.onnx'); print([(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])"` |
| **支持文件** | 分词器、词汇表、声音嵌入 | 查看 HuggingFace 仓库文件列表 |
| **HuggingFace 模型卡** | 架构、期望输入、使用示例 | HuggingFace 仓库中的 `README.md` |

对于 GGUF 模型，可以跳过张量检查——llama.cpp 内部处理分词和解码。助手主要需要模型的架构和上下文长度，这些信息已嵌入 GGUF 元数据中。

### 可复用的提示词模板

将以下提示词复制粘贴到你的 AI Agent中，填写方括号中的占位符：

````
我有一个用于 [任务描述，如"命名实体识别"或"文字转语音"] 的 [ONNX / GGUF / SafeTensors] 模型。

模型来源：[HuggingFace 仓库 URL 或描述]
模型文件：[文件名，如 "model.onnx" 或 "model-Q4_K_M.gguf"]

以下是模型的输入张量：
[粘贴上方 ONNX 检查命令的输出，GGUF 模型填 "N/A"]

以下是模型的输出张量：
[粘贴上方 ONNX 检查命令的输出，GGUF 模型填 "N/A"]

模型目录中的支持文件：
[列出文件，如 "tokenizer.json, vocab.txt, config.json"]

请为 xybrid ML 推理框架生成一个 model_metadata.json 文件。

规则：
- execution_template.type 必须为以下之一：Onnx、Gguf、SafeTensors、CoreMl、TfLite、ModelGraph
- 可用的预处理步骤：AudioDecode、MelSpectrogram、Phonemize、Tokenize、Normalize、Resize、CenterCrop、Reshape
- 可用的后处理步骤：CTCDecode、TTSAudioEncode、WhisperDecode、BPEDecode、ArgMax、Softmax、TopK、Threshold、TemperatureSample、MeanPool、Denormalize
- 对于 GGUF 模型：预处理和后处理留为空数组（llama.cpp 内部处理分词）
- 对于 SafeTensors Whisper 模型：预处理和后处理留空，将 architecture 设为 "whisper"
- 所有引用的文件（vocab_file、tokens_file、tokenizer_file 等）必须列在 "files" 数组中
- metadata 对象应在适用时包含 "task"、"architecture" 和 "backend"

任务与步骤对应关系：
- 语音识别（CTC）：AudioDecode → 模型 → CTCDecode
- 语音识别（Whisper）：SafeTensors 模板，architecture 为 "whisper"（无需步骤）
- 文字转语音：Phonemize（backend: MisakiDictionary）→ 模型 → TTSAudioEncode
- 图像分类：Reshape + Normalize → 模型 → Softmax 或 ArgMax
- 句子嵌入：Tokenize → 模型 → MeanPool
- 文本分类 / NER：Tokenize → 模型 → ArgMax
- 文本生成（GGUF）：Gguf 模板，无步骤

完整字段参考：https://github.com/xybrid-ai/xybrid/blob/main/docs/sdk/MODEL_METADATA.md
JSON Schema：https://github.com/xybrid-ai/xybrid/blob/main/docs/sdk/model_metadata.schema.json
````

### 实战示例：配置 BERT NER 模型

假设你下载了 ONNX 格式的 [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)，但不确定如何配置。

**第一步：检查模型**

```bash
python -c "
import onnx
m = onnx.load('model.onnx')
print('Inputs:')
for i in m.graph.input:
    dims = [d.dim_value or 'dynamic' for d in i.type.tensor_type.shape.dim]
    print(f'  {i.name}: {dims}')
print('Outputs:')
for o in m.graph.output:
    dims = [d.dim_value or 'dynamic' for d in o.type.tensor_type.shape.dim]
    print(f'  {o.name}: {dims}')
"
```

输出：
```
Inputs:
  input_ids: ['dynamic', 'dynamic']
  attention_mask: ['dynamic', 'dynamic']
  token_type_ids: ['dynamic', 'dynamic']
Outputs:
  logits: ['dynamic', 'dynamic', 9]
```

**第二步：将提示词发送给 AI Agent**

```
我有一个用于命名实体识别（NER）的 ONNX 模型。

模型来源：https://huggingface.co/dslim/bert-base-NER
模型文件：model.onnx

输入张量：
  input_ids: [dynamic, dynamic]
  attention_mask: [dynamic, dynamic]
  token_type_ids: [dynamic, dynamic]

输出张量：
  logits: [dynamic, dynamic, 9]

支持文件：tokenizer.json, vocab.txt

请为 xybrid ML 推理框架生成一个 model_metadata.json 文件。
[...粘贴上方完整提示词模板...]
```

**第三步：AI Agent生成配置**

```json
{
  "model_id": "bert-base-ner",
  "version": "1.0",
  "description": "BERT base model fine-tuned for Named Entity Recognition (9 entity classes)",
  "execution_template": {
    "type": "Onnx",
    "model_file": "model.onnx"
  },
  "preprocessing": [
    {
      "type": "Tokenize",
      "vocab_file": "tokenizer.json",
      "tokenizer_type": "WordPiece",
      "max_length": 512
    }
  ],
  "postprocessing": [
    { "type": "ArgMax" }
  ],
  "files": ["model.onnx", "tokenizer.json"],
  "metadata": {
    "task": "token_classification",
    "architecture": "BertForTokenClassification",
    "num_labels": 9
  }
}
```

**第四步：验证并测试**

```bash
# 验证 JSON Schema（可选，需安装 ajv-cli）
npx ajv-cli validate -s docs/sdk/model_metadata.schema.json -d my-model/model_metadata.json

# 使用 xybrid 测试
cargo run --example custom_model -- --model-dir ./my-model --input "John works at Google in New York"
```

AI Agent的输出是尽力而为的起点。如果推理结果不符合预期，请查阅[故障排查](#故障排查)部分并调整预处理/后处理步骤。

### 获得更好结果的技巧

- **提供 HuggingFace 模型卡 URL**——助手可以读取它以了解模型用途、期望输入和训练细节。
- **直接粘贴张量检查输出**——不要概括，精确的张量名称和形状对于映射到正确的预处理步骤至关重要。
- **明确说明任务类型**（如"token 分类"、"文字转语音"、"语音识别"）——这有助于助手选择正确的预处理/后处理流程。
- **对于 GGUF 模型**，提供量化类型（Q4_K_M、Q8_0 等）和模型架构（Llama、Qwen、Mistral）——这些决定了上下文长度和后端设置。
- **如果第一次尝试不成功**，将错误信息分享给助手并请其修正配置。常见问题包括分词器类型错误、缺少 attention mask 处理或输出后处理不正确。

## 下一步

- **[MODEL_METADATA.md](../sdk/MODEL_METADATA.md)** — 每种执行模板类型、预处理步骤和后处理步骤的完整字段参考
- **[JSON Schema](../sdk/model_metadata.schema.json)** — 在编辑器中使用，获得自动补全和验证
- **[API Reference](../sdk/API_REFERENCE.md)** — 所有平台的完整 SDK API 文档

# Custom Models Guide

> Load your own ONNX, GGUF, or SafeTensors models with xybrid using `fromDirectory()`.

This guide walks through loading a model that isn't in the xybrid registry. By the end, you'll have a working `model_metadata.json` and be running inference via `fromDirectory()`.

## Overview

The workflow is:

1. **Prepare** a directory with your model file(s)
2. **Write** a `model_metadata.json` describing inputs, outputs, and processing steps
3. **Load** with `fromDirectory()` in any SDK (Rust, Flutter, Kotlin, Swift, Unity)
4. **Run** inference via the standard `model.run()` API

## Step 1: Prepare Your Model Directory

Create a directory containing your model and any supporting files:

```
my-custom-model/
├── model_metadata.json    # Required — tells xybrid how to run the model
├── model.onnx             # Your model file (ONNX, GGUF, etc.)
├── tokenizer.json         # Optional — if the model needs tokenization
└── vocab.txt              # Optional — if needed by pre/postprocessing
```

All files referenced in `model_metadata.json` must be in this directory.

## Step 2: Write model_metadata.json

This file tells xybrid the model format, preprocessing steps (applied to input before the model), and postprocessing steps (applied to model output).

### Minimal Example: ONNX Classification

For an ONNX image classification model (e.g., MobileNet from the [ONNX Model Zoo](https://github.com/onnx/models)):

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

### ONNX Token Classification (NER / PII Detection)

For a model like DistilBERT fine-tuned for Named Entity Recognition:

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

### GGUF LLM (Text Generation)

For a quantized LLM from HuggingFace (e.g., Qwen, Llama, Mistral):

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

GGUF models handle tokenization internally via llama.cpp — no preprocessing steps needed.

### Sentence Embeddings

For a sentence transformer model (e.g., all-MiniLM-L6-v2):

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

For the full field reference, see [MODEL_METADATA.md](../sdk/MODEL_METADATA.md). For editor autocomplete and validation, use the [JSON Schema](../sdk/model_metadata.schema.json).

## Step 3: Load with fromDirectory()

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

## Platform-Specific Patterns

### Flutter — Bundled Assets

Bundle the model directory in your Flutter app's assets:

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/my-model/model_metadata.json
    - assets/models/my-model/model.onnx
```

At runtime, copy assets to a writable directory (assets are read-only):

```dart
import 'dart:io';
import 'package:path_provider/path_provider.dart';

Future<String> prepareModel() async {
  final appDir = await getApplicationSupportDirectory();
  final modelDir = Directory('${appDir.path}/models/my-model');

  if (!await modelDir.exists()) {
    await modelDir.create(recursive: true);
    // Copy each file from assets to the directory
    for (final file in ['model_metadata.json', 'model.onnx']) {
      final data = await rootBundle.load('assets/models/my-model/$file');
      await File('${modelDir.path}/$file')
          .writeAsBytes(data.buffer.asUint8List());
    }
  }

  return modelDir.path;
}

// Then load
final path = await prepareModel();
final loader = XybridModelLoader.fromDirectory(path);
```

### Android — App-Local Storage

Place models in `assets/` or download to internal storage:

```kotlin
// Copy from assets to internal storage
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

### iOS / macOS — App Bundle

Add model files to your Xcode project as a folder reference:

```swift
// Models added to the Xcode project are in the app bundle
guard let modelPath = Bundle.main.path(forResource: "my-model", ofType: nil) else {
    fatalError("Model directory not found in bundle")
}

let loader = try ModelLoader.fromDirectory(path: modelPath)
```

### Unity — StreamingAssets

Place model files in `Assets/StreamingAssets/`:

```
Assets/
└── StreamingAssets/
    └── models/
        └── my-model/
            ├── model_metadata.json
            └── model.onnx
```

```csharp
// StreamingAssets path varies by platform
var modelPath = Path.Combine(Application.streamingAssetsPath, "models", "my-model");
var loader = ModelLoader.FromDirectory(modelPath);
```

### Tauri — App Resources

In a Tauri app, use the resource directory:

```rust
use tauri::Manager;

let app_handle = app.handle();
let resource_dir = app_handle.path().resource_dir()
    .expect("failed to get resource dir");
let model_path = resource_dir.join("models").join("my-model");

let loader = ModelLoader::from_directory(model_path.to_str().unwrap())?;
```

Add the model directory to `tauri.conf.json`:

```json
{
  "bundle": {
    "resources": ["models/my-model/*"]
  }
}
```

## Worked Example: Loading a HuggingFace ONNX Model

Let's walk through loading [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as a custom model.

### 1. Download the model files

```bash
# Create the model directory
mkdir -p my-minilm/

# Download from HuggingFace (ONNX format)
curl -L "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" \
  -o my-minilm/model.onnx

# Download the tokenizer
curl -L "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json" \
  -o my-minilm/tokenizer.json
```

### 2. Write model_metadata.json

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

### 3. Run inference

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

    // output.kind is EnvelopeKind::Embedding(Vec<f32>) with 384 dimensions
    if let EnvelopeKind::Embedding(embedding) = &output.kind {
        println!("Embedding dimensions: {}", embedding.len());
        println!("First 5 values: {:?}", &embedding[..5]);
    }

    Ok(())
}
```

## Troubleshooting

### "DirectoryNotFound" error

The path passed to `fromDirectory()` does not exist on disk.

- Double-check the path is absolute (not relative)
- On mobile, verify the model was copied to a writable directory — app assets are typically read-only

### "MetadataNotFound" error

The directory exists but contains no `model_metadata.json` file.

- Ensure the file is named exactly `model_metadata.json` (case-sensitive)
- Check the file was copied along with the model (especially for bundled assets)

### "MetadataInvalid" error

The `model_metadata.json` file exists but contains invalid JSON or doesn't match the expected schema.

- Validate your JSON syntax (trailing commas, missing quotes)
- Use the [JSON Schema](../sdk/model_metadata.schema.json) for editor validation — most editors support `$schema` references:
  ```json
  {
    "$schema": "../docs/sdk/model_metadata.schema.json",
    "model_id": "...",
    ...
  }
  ```
- Ensure `execution_template.type` is one of: `Onnx`, `SimpleMode`, `Gguf`, `SafeTensors`, `CoreMl`, `TfLite`, `ModelGraph`

### Shape mismatch / wrong output

The model runs but produces garbage output or a shape error.

- **Preprocessing mismatch**: The `preprocessing` steps must transform your input into the exact format the model expects. Check the model's documentation for expected input tensor shapes and data types.
- **Postprocessing mismatch**: The `postprocessing` steps must match the model's output format. For example, using `CTCDecode` on a classification model will fail.
- **Wrong model file**: Verify `execution_template.model_file` points to the correct file in your directory.

### Missing files at runtime

The model loads but fails when trying to read a vocabulary or tokenizer file.

- Every file referenced in preprocessing/postprocessing (e.g., `vocab_file`, `tokens_file`, `tokenizer.json`) must be in the model directory
- Every file must also be listed in the `files` array

### GGUF model not loading

- Ensure xybrid was built with the `llm-llamacpp` feature (included in all platform presets)
- GGUF models require `"type": "Gguf"` in the execution template — not `"Onnx"`
- Set `context_length` in the execution template to control memory usage

## Quick Reference: Task → Configuration

Use this table to find the right configuration for your model type. Pick the row matching your task, then copy the corresponding preprocessing and postprocessing steps.

| Task | Format | Execution Template | Preprocessing | Postprocessing |
| ---- | ------ | ------------------ | ------------- | -------------- |
| **Image classification** | ONNX | `Onnx` | `Reshape` → `Normalize` | `Softmax` or `ArgMax` |
| **Speech recognition (CTC)** | ONNX | `Onnx` | `AudioDecode` | `CTCDecode` |
| **Speech recognition (Whisper)** | SafeTensors | `SafeTensors` | _(built-in)_ | _(built-in)_ |
| **Text-to-speech** | ONNX | `Onnx` | `Phonemize` | `TTSAudioEncode` |
| **Text generation (LLM)** | GGUF | `Gguf` | _(none)_ | _(none)_ |
| **Sentence embeddings** | ONNX | `Onnx` | `Tokenize` | `MeanPool` |
| **Text classification / NER** | ONNX | `Onnx` | `Tokenize` | `ArgMax` |
| **Object detection / Vision** | ONNX | `Onnx` | `Resize` → `Normalize` | `TopK` or `Threshold` |

**Key rules:**

- **GGUF models** handle tokenization internally — always leave preprocessing/postprocessing empty
- **SafeTensors (Whisper)** have built-in decoding — set `architecture: "whisper"` and xybrid handles the rest
- **ONNX models** need explicit preprocessing/postprocessing matching the model's expected input/output

## Using AI Agents to Configure Model Execution

If you have an ONNX or GGUF model but aren't sure what preprocessing/postprocessing steps it needs, you can use an AI coding agent (Claude, Cursor, Copilot, etc.) to figure it out. The agent can inspect your model file, read the HuggingFace model card, and generate a working `model_metadata.json` for you.

### What the AI Agent Needs

Give your AI agent these pieces of information:

| Information | Why It's Needed | How to Get It |
| ----------- | --------------- | ------------- |
| **Model file format** | Determines `execution_template.type` | Check file extension: `.onnx`, `.gguf`, `.safetensors` |
| **Task type** | Determines pre/postprocessing steps | Check the HuggingFace model card `pipeline_tag` |
| **Input tensor names & shapes** | Needed for correct preprocessing | ONNX: `python -c "import onnx; m = onnx.load('model.onnx'); print([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])"` |
| **Output tensor names & shapes** | Needed for correct postprocessing | ONNX: `python -c "import onnx; m = onnx.load('model.onnx'); print([(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])"` |
| **Supporting files** | Tokenizer, vocab, voice embeddings | Check HuggingFace repo file listing |
| **HuggingFace model card** | Architecture, expected inputs, usage examples | The `README.md` in the HuggingFace repo |

For GGUF models, you can skip tensor inspection — llama.cpp handles tokenization and decoding internally. The agent mainly needs the model's architecture and context length, which are embedded in the GGUF metadata.

### Reusable Prompt Template

Copy and paste this prompt into your AI coding agent, filling in the bracketed placeholders:

````
I have a [ONNX / GGUF / SafeTensors] model for [task description, e.g., "named entity recognition" or "text-to-speech"].

Model source: [HuggingFace repo URL or description]
Model file: [filename, e.g., "model.onnx" or "model-Q4_K_M.gguf"]

Here are the model's input tensors:
[Paste output from the ONNX inspection command above, or "N/A for GGUF"]

Here are the model's output tensors:
[Paste output from the ONNX inspection command above, or "N/A for GGUF"]

Supporting files in the model directory:
[List files, e.g., "tokenizer.json, vocab.txt, config.json"]

Generate a model_metadata.json file for the xybrid ML inference framework.

Rules:
- execution_template.type must be one of: Onnx, Gguf, SafeTensors, CoreMl, TfLite, ModelGraph
- Available preprocessing steps: AudioDecode, MelSpectrogram, Phonemize, Tokenize, Normalize, Resize, CenterCrop, Reshape
- Available postprocessing steps: CTCDecode, TTSAudioEncode, WhisperDecode, BPEDecode, ArgMax, Softmax, TopK, Threshold, TemperatureSample, MeanPool, Denormalize
- For GGUF models: leave preprocessing and postprocessing as empty arrays (llama.cpp handles tokenization internally)
- For SafeTensors Whisper models: leave preprocessing and postprocessing empty, set architecture to "whisper"
- All referenced files (vocab_file, tokens_file, tokenizer_file, etc.) must be listed in the "files" array
- The metadata object should include "task", "architecture", and "backend" where applicable

Task-to-step mapping:
- Speech recognition (CTC): AudioDecode → model → CTCDecode
- Speech recognition (Whisper): SafeTensors template with architecture "whisper" (no steps needed)
- Text-to-speech: Phonemize (backend: MisakiDictionary) → model → TTSAudioEncode
- Image classification: Reshape + Normalize → model → Softmax or ArgMax
- Sentence embeddings: Tokenize → model → MeanPool
- Text classification / NER: Tokenize → model → ArgMax
- Text generation (GGUF): Gguf template, no steps

See the full field reference: https://github.com/xybrid-ai/xybrid/blob/main/docs/sdk/MODEL_METADATA.md
See the JSON Schema: https://github.com/xybrid-ai/xybrid/blob/main/docs/sdk/model_metadata.schema.json
````

### Worked Example: Configuring a BERT NER Model

Here's what a real interaction looks like. Suppose you downloaded [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) in ONNX format and aren't sure how to configure it.

**Step 1: Inspect the model**

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

Output:
```
Inputs:
  input_ids: ['dynamic', 'dynamic']
  attention_mask: ['dynamic', 'dynamic']
  token_type_ids: ['dynamic', 'dynamic']
Outputs:
  logits: ['dynamic', 'dynamic', 9]
```

**Step 2: Send the prompt to your AI agent**

```
I have an ONNX model for named entity recognition (NER).

Model source: https://huggingface.co/dslim/bert-base-NER
Model file: model.onnx

Here are the model's input tensors:
  input_ids: [dynamic, dynamic]
  attention_mask: [dynamic, dynamic]
  token_type_ids: [dynamic, dynamic]

Here are the model's output tensors:
  logits: [dynamic, dynamic, 9]

Supporting files in the model directory:
tokenizer.json, vocab.txt

Generate a model_metadata.json file for the xybrid ML inference framework.
[...rest of the prompt template above...]
```

**Step 3: AI agent generates the config**

The agent produces:

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

**Step 4: Validate and test**

```bash
# Validate against the JSON Schema (optional, if you have ajv-cli)
npx ajv-cli validate -s docs/sdk/model_metadata.schema.json -d my-model/model_metadata.json

# Test with xybrid
cargo run --example custom_model -- --model-dir ./my-model --input "John works at Google in New York"
```

The AI agent's output is a best-effort starting point. If inference produces unexpected results, check the [Troubleshooting](#troubleshooting) section and refine the preprocessing/postprocessing steps.

### Tips for Better Results

- **Include the HuggingFace model card URL** — agents can read it to understand the model's purpose, expected inputs, and training details.
- **Paste the tensor inspection output directly** — don't summarize it. The exact tensor names and shapes are critical for mapping to the right preprocessing steps.
- **Mention the task type explicitly** (e.g., "token classification", "text-to-speech", "speech recognition") — this helps the agent pick the right pre/postprocessing pipeline.
- **For GGUF models**, include the quantization type (Q4_K_M, Q8_0, etc.) and the model architecture (Llama, Qwen, Mistral) — these determine context length and backend settings.
- **If the first attempt doesn't work**, share the error message with the agent and ask it to fix the config. Common issues are wrong tokenizer type, missing attention mask handling, or incorrect output postprocessing.

## What's Next

- **[MODEL_METADATA.md](../sdk/MODEL_METADATA.md)** — Full field reference for every execution template type, preprocessing step, and postprocessing step
- **[JSON Schema](../sdk/model_metadata.schema.json)** — Use in your editor for autocomplete and validation
- **[API Reference](../sdk/API_REFERENCE.md)** — Complete SDK API documentation for all platforms

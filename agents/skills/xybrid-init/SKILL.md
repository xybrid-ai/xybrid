Generate a `model_metadata.json` for any ML model so it works with xybrid.

The user may provide a HuggingFace repo, a local directory, or nothing (in which case ask).

**Input**: $ARGUMENTS (optional — HuggingFace repo ID, URL, or local path)

---

## Step 1: Determine Source

If `$ARGUMENTS` is provided, use it. Otherwise ask:

> What model do you want to set up?
> - Paste a HuggingFace repo (e.g. `hexgrad/Kokoro-82M-v1.0-ONNX`)
> - Or a local directory path (e.g. `./my-model/`)

Detect the source type:
- Contains `huggingface.co/` or matches `org/repo` pattern → **HuggingFace**
- Is a local path that exists → **Local directory**
- Otherwise → ask the user to clarify

---

## Step 2: Gather Context

### If HuggingFace:

Fetch these three resources (use WebFetch for each):

1. **Model card**: `https://huggingface.co/{repo}/raw/main/README.md`
2. **File listing**: `https://huggingface.co/api/models/{repo}` (look at the `siblings` array for file names and sizes)
3. **Config file** (if exists): `https://huggingface.co/{repo}/raw/main/config.json`

Also check for these files (fetch if they exist in the file listing):
- `tokenizer_config.json`
- `tokenizer.json`
- `generation_config.json`

### If Local directory:

1. List all files in the directory
2. Read any `README.md`, `config.json`, `tokenizer_config.json` if present
3. If Python is available and there's an `.onnx` file, inspect it:
   ```bash
   python3 -c "import onnx; m = onnx.load('MODEL.onnx'); print('Inputs:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]); print('Outputs:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])"
   ```

---

## Step 3: Analyze and Generate

Using ALL the gathered context (model card, file list, config, ONNX inputs/outputs), generate a valid `model_metadata.json`.

### Decision Tree

Use the model card description, file extensions, and config to determine the model type:

**File-based detection:**
- `.gguf` file → **LLM** (Gguf template)
- `.safetensors` + whisper/candle architecture → **ASR** (SafeTensors template)
- `.onnx` file → continue to task detection below
- `.mlmodel` / `.mlpackage` → **CoreML** template
- `.tflite` → **TfLite** template

**Task detection (from model card + config):**
- Model card mentions "text-to-speech", "TTS", "speech synthesis" → **TTS**
- Model card mentions "speech recognition", "ASR", "transcription" → **ASR**
- Model card mentions "embedding", "sentence transformer", "similarity" → **Embedding**
- Model card mentions "classification", "image", "vision", "ImageNet" → **Vision**
- Model card mentions "language model", "text generation", "chat", "instruct" → **LLM**
- Model card mentions "object detection", "YOLO", "segmentation" → **Vision**

### Schema Reference

The `model_metadata.json` must conform to this exact schema:

```json
{
  "model_id": "string (required) — kebab-case identifier",
  "version": "string (required) — model version",
  "description": "string (optional) — human-readable description",
  "execution_template": { "type": "...", ... },
  "preprocessing": [ ... ],
  "postprocessing": [ ... ],
  "files": [ "list of all required files" ],
  "metadata": { "task": "...", ... },
  "voices": { "... (TTS only)" }
}
```

### Execution Templates

Choose ONE:

```json
// ONNX model
{ "type": "Onnx", "model_file": "model.onnx" }

// SafeTensors (Candle runtime — currently only Whisper)
{ "type": "SafeTensors", "model_file": "model.safetensors", "architecture": "whisper", "config_file": "config.json", "tokenizer_file": "tokenizer.json" }

// GGUF (local LLMs via llama.cpp)
{ "type": "Gguf", "model_file": "model.gguf", "context_length": 4096 }

// CoreML (Apple platforms)
{ "type": "CoreMl", "model_file": "model.mlpackage" }

// TFLite (mobile)
{ "type": "TfLite", "model_file": "model.tflite" }
```

### Preprocessing Steps

Choose the appropriate chain based on task:

**TTS (text-to-speech):**
```json
[{ "type": "Phonemize", "tokens_file": "tokens.txt", "backend": "MisakiDictionary", "add_padding": true, "normalize_text": true }]
```
Backends: `MisakiDictionary` (default, pure Rust), `EspeakNG` (multi-language, needs system install), `CmuDictionary` (legacy), `OpenPhonemizer` (hybrid dictionary + neural)

**ASR (speech recognition) with ONNX:**
```json
[{ "type": "AudioDecode", "sample_rate": 16000, "channels": 1 }]
```

**ASR with Whisper SafeTensors:** empty `[]` (Candle handles internally)

**Text embedding / NLP:**
```json
[{ "type": "Tokenize", "vocab_file": "tokenizer.json", "tokenizer_type": "WordPiece", "max_length": 512 }]
```
Tokenizer types: `WordPiece` (BERT), `BPE` (GPT), `SentencePiece` (T5)

**Image classification / vision:**
```json
[
  { "type": "Resize", "width": 224, "height": 224 },
  { "type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] }
]
```
Use ImageNet normalization values unless model card specifies otherwise.

**LLM (GGUF):** empty `[]` (llama.cpp handles internally)

### Postprocessing Steps

**TTS:**
```json
[{ "type": "TTSAudioEncode", "sample_rate": 24000, "apply_postprocessing": true }]
```

**ASR (CTC-based, e.g. Wav2Vec2):**
```json
[{ "type": "CTCDecode", "vocab_file": "vocab.json", "blank_index": 0 }]
```

**ASR (Whisper SafeTensors):** empty `[]`

**Text embedding:**
```json
[{ "type": "MeanPool", "dim": 1 }]
```

**Image classification:**
```json
[{ "type": "Softmax", "dim": 1 }]
```
Or `{ "type": "Argmax" }` if you just need the class index.

**LLM:** empty `[]`

### Voice Config (TTS only)

If the model has voice embeddings (e.g. `voices.bin`):
```json
{
  "voices": {
    "format": "embedded",
    "file": "voices.bin",
    "loader": "binary_f32_256",
    "default": "voice_id",
    "selection_strategy": "TokenLength",
    "catalog": [
      { "id": "voice_id", "name": "Display Name", "index": 0, "gender": "female", "language": "en-US", "style": "neutral" }
    ]
  }
}
```

---

## Step 4: Validate

Before presenting the result, verify:

1. **All files in `files` array exist** (in the HF repo or local directory)
2. **`model_file` matches** an actual file name
3. **Preprocessing/postprocessing steps match** the model task
4. **Step types are valid** — only use the types listed above
5. **The JSON is valid** — parseable, no trailing commas

---

## Step 5: Real Examples for Reference

### TTS (Kokoro 82M)
```json
{
  "model_id": "kokoro-82m",
  "version": "1.0",
  "description": "Kokoro 82M - High-quality TTS with 24 voices",
  "execution_template": { "type": "Onnx", "model_file": "model.onnx" },
  "preprocessing": [{ "type": "Phonemize", "tokens_file": "tokens.txt", "backend": "MisakiDictionary", "add_padding": true, "normalize_text": true }],
  "postprocessing": [{ "type": "TTSAudioEncode", "sample_rate": 24000, "apply_postprocessing": true }],
  "files": ["model.onnx", "voices.bin", "tokens.txt", "misaki/us_gold.json"],
  "metadata": { "task": "text-to-speech", "sample_rate": 24000, "parameters": 82000000, "family": "hexgrad", "license": "Apache-2.0" }
}
```

### LLM (Qwen 3.5 0.8B)
```json
{
  "model_id": "qwen3.5-0.8b",
  "version": "1.0",
  "description": "Qwen 3.5 0.8B - Lightweight multilingual LLM",
  "execution_template": { "type": "Gguf", "model_file": "Qwen3.5-0.8B-Q4_K_M.gguf", "context_length": 4096 },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["Qwen3.5-0.8B-Q4_K_M.gguf"],
  "metadata": { "task": "text-generation", "architecture": "qwen35", "backend": "llamacpp", "parameters": 800000000, "license": "Apache-2.0" }
}
```

### Text Embedding (all-MiniLM)
```json
{
  "model_id": "all-minilm",
  "version": "L6-v2",
  "execution_template": { "type": "Onnx", "model_file": "model.onnx" },
  "preprocessing": [{ "type": "Tokenize", "vocab_file": "tokenizer.json", "tokenizer_type": "WordPiece", "max_length": 512 }],
  "postprocessing": [{ "type": "MeanPool", "dim": 1 }],
  "files": ["model.onnx", "vocab.txt", "tokenizer.json", "config.json"],
  "metadata": { "task": "text-embedding", "embedding_dim": 384 }
}
```

### ASR (Whisper Tiny — SafeTensors)
```json
{
  "model_id": "whisper-tiny",
  "version": "1.0",
  "description": "Whisper Tiny - Fast multilingual ASR (Candle runtime)",
  "execution_template": { "type": "SafeTensors", "model_file": "model.safetensors", "config_file": "config.json", "tokenizer_file": "tokenizer.json" },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["model.safetensors", "config.json", "tokenizer.json", "melfilters.bytes"],
  "metadata": { "task": "speech-recognition", "runtime": "candle", "sample_rate": 16000, "parameters": 39000000 }
}
```

### Image Classification (MNIST)
```json
{
  "model_id": "mnist-digit-recognition",
  "version": "12",
  "description": "MNIST handwritten digit recognition",
  "execution_template": { "type": "Onnx", "model_file": "model.onnx" },
  "preprocessing": [
    { "type": "Reshape", "shape": [1, 1, 28, 28] },
    { "type": "Normalize", "mean": [0.0], "std": [255.0] }
  ],
  "postprocessing": [{ "type": "Softmax", "dim": 1 }],
  "files": ["model.onnx"],
  "metadata": { "task": "image_classification", "input_shape": [1, 1, 28, 28], "num_classes": 10 }
}
```

---

## Step 6: Present and Save

Show the generated `model_metadata.json` to the user with a brief explanation of the key decisions made (e.g., "Using MisakiDictionary phonemizer because the model card says it's a Kokoro-based TTS model").

Then ask:

> Save to `{directory}/model_metadata.json`?

If the user confirms, write the file.

---

## Step 7: Download Model Files (if HuggingFace)

If the source is HuggingFace and the model files aren't local yet, offer to download:

> Download model files (~{size})? This will fetch:
> - `model.onnx` (150 MB)
> - `voices.bin` (24 KB)
> - ...

If the user confirms, download each file listed in the `files` array:
```bash
curl -L "https://huggingface.co/{repo}/resolve/main/{file}" -o "{directory}/{file}"
```

For files in subdirectories (e.g. `misaki/us_gold.json`), create the subdirectory first.

---

## Step 8: Next Steps

After saving, print:

```
Your model is ready. Next steps:

# Test it works (use --input-audio <file>.wav for ASR models)
xybrid run --model {model_id} --input-text "test input"

# Or from Rust
cargo run --example your_test -p xybrid-core

# Or use /test-model to validate end-to-end
```

If `/test-model` is available (the user has xybrid cloned), suggest running it.

Test a model end-to-end using the xybrid execution system.

The user should specify which model to test (e.g., "test kokoro-82m" or "test the TTS model").

**Input**: $ARGUMENTS (optional — model name or path to model directory)

---

## Step 1: Locate the Model

If `$ARGUMENTS` is provided, use it to find the model. Otherwise ask which model to test.

Search for the model in these locations (in order):
1. Direct path if `$ARGUMENTS` is a directory path
2. `integration-tests/fixtures/models/{name}/`
3. `~/.xybrid/cache/{name}/`
4. Current directory

Verify that `model_metadata.json` exists in the model directory.

---

## Step 2: Validate model_metadata.json

Read `model_metadata.json` and check:

1. **All files listed in `files` array exist** in the model directory
2. **`model_file` in `execution_template`** points to an actual file
3. **JSON is valid** and has required fields (`model_id`, `version`, `execution_template`, `files`)
4. **Preprocessing/postprocessing types are valid** enum values

If any check fails, report the specific issue and suggest how to fix it.

---

## Step 3: Determine Test Strategy

Based on the `execution_template.type` and `metadata.task`:

| Task | Input | Expected Output | Feature Flags |
|------|-------|-----------------|---------------|
| `text-to-speech` | `Envelope::Text("Hello world")` | Audio bytes (length > 0) | default |
| `speech-recognition` (Onnx) | `Envelope::Audio(wav_bytes)` | Text transcription | default |
| `speech-recognition` (SafeTensors) | `Envelope::Audio(wav_bytes)` | Text transcription | `candle,candle-metal` |
| `text-generation` (Gguf) | `Envelope::Text("Hello")` | Text response | `llm-llamacpp` |
| `text-embedding` | `Envelope::Text("test sentence")` | Embedding vector (f32) | default |
| `image_classification` | Raw image bytes | Class probabilities | default |

---

## Step 4: Find or Create Test Example

Check for an existing example in `crates/xybrid-core/examples/` that matches the model.

If no example exists, create a minimal one at `crates/xybrid-core/examples/{model_id}_test.rs`:

```rust
//! Test example for {model_id}
use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = PathBuf::from("integration-tests/fixtures/models/{model_id}");
    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create appropriate input based on model task
    let input = Envelope {
        kind: EnvelopeKind::Text("Hello world".into()), // adjust per task
        metadata: HashMap::new(),
    };

    let output = executor.execute(&metadata, &input)?;
    println!("Output: {:?}", output.kind);
    println!("Test passed!");
    Ok(())
}
```

Adjust the input type based on the model task (Text for TTS/LLM/embedding, Audio for ASR).

---

## Step 5: Run the Test

Run from the `repos/xybrid/` directory (or the repo root if that's where Cargo.toml is):

```bash
cargo run --example {example_name} -p xybrid-core --features {features}
```

Add feature flags based on the model type (see table in Step 3).

---

## Step 6: Validate Output

Check the output based on model type:

- **TTS**: Output should be `EnvelopeKind::Audio(bytes)` with `bytes.len() > 0`. Optionally save to `output.wav` for manual listening.
- **ASR**: Output should be `EnvelopeKind::Text(transcription)` with non-empty text.
- **LLM**: Output should be `EnvelopeKind::Text(response)` with non-empty text.
- **Embedding**: Output should be `EnvelopeKind::Embedding(vec)` with expected dimensionality.
- **Classification**: Output should contain class probabilities or indices.

---

## Step 7: Report Results

Print a summary:

```
Model: {model_id}
Task:  {task}
Input: {input_type}
Output: {output_summary}
Status: PASS / FAIL

{If FAIL: specific error message and suggestion}
```

## Common Issues

- **"model_metadata.json not found"**: Check the model directory path
- **"file not found"**: A file listed in `files` array doesn't exist — download it or fix the path
- **"preprocessing failed"**: Wrong preprocessing step for the model type
- **"ONNX error"**: Model file may be corrupted or wrong format
- **"feature not enabled"**: Add the required feature flag (e.g., `--features candle` for SafeTensors models)
- **"llm backend not available"**: Add `--features llm-llamacpp` for GGUF models

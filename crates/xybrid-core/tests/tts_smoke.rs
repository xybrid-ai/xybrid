//! TTS model smoke tests — verify all TTS fixtures execute through the generic path.
//!
//! This test iterates over all model directories in integration-tests/fixtures/models/,
//! identifies TTS models by their `metadata.task == "text-to-speech"` field, and runs
//! each through `TemplateExecutor` to verify they produce non-empty audio output.
//!
//! Marked `#[ignore]` because it requires downloaded ONNX model files.
//!
//! ```bash
//! # Download TTS models first
//! ./integration-tests/download.sh kitten-tts-nano-0.2 kokoro-82m
//!
//! # Run smoke tests
//! cargo test -p xybrid-core --test tts_smoke -- --ignored
//! ```

use std::collections::HashMap;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

/// Discover all TTS model directories that have both model_metadata.json and
/// the ONNX model file present.
fn discover_tts_models() -> Vec<(String, std::path::PathBuf)> {
    let Some(models_dir) = model_fixtures::models_dir() else {
        return vec![];
    };

    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return vec![];
    };

    let mut tts_models = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let metadata_path = path.join("model_metadata.json");
        if !metadata_path.exists() {
            continue;
        }

        // Parse metadata to check task type
        let Ok(content) = std::fs::read_to_string(&metadata_path) else {
            continue;
        };
        let Ok(metadata) = serde_json::from_str::<ModelMetadata>(&content) else {
            continue;
        };

        // Check if this is a TTS model
        let is_tts = metadata
            .metadata
            .get("task")
            .and_then(|v| v.as_str())
            .map(|t| t == "text-to-speech")
            .unwrap_or(false);

        if !is_tts {
            continue;
        }

        // Check if any model file from the files list exists (skip if not downloaded)
        let has_model_file = metadata.files.iter().any(|f| {
            let ext = f.rsplit('.').next().unwrap_or("");
            matches!(ext, "onnx" | "safetensors" | "gguf") && path.join(f).exists()
        });

        if !has_model_file {
            eprintln!(
                "Skipping {}: model file not downloaded",
                entry.file_name().to_string_lossy(),
            );
            continue;
        }

        let name = entry.file_name().to_string_lossy().to_string();
        tts_models.push((name, path));
    }

    tts_models
}

#[test]
#[ignore]
fn test_all_tts_models_produce_audio() {
    let tts_models = discover_tts_models();

    if tts_models.is_empty() {
        eprintln!("No TTS models found with ONNX files. Download with:");
        eprintln!("  ./integration-tests/download.sh kitten-tts-nano-0.2 kokoro-82m");
        return;
    }

    println!("Found {} TTS model(s) to test", tts_models.len());

    let mut passed = 0;
    let mut failed = Vec::new();

    for (name, model_dir) in &tts_models {
        println!("\n--- Testing: {} ---", name);

        // Load metadata
        let metadata_path = model_dir.join("model_metadata.json");
        let content = std::fs::read_to_string(&metadata_path)
            .unwrap_or_else(|e| panic!("Failed to read metadata for {}: {}", name, e));
        let metadata: ModelMetadata = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse metadata for {}: {}", name, e));

        println!("  model_id: {}", metadata.model_id);
        println!("  preprocessing: {} step(s)", metadata.preprocessing.len());
        println!(
            "  postprocessing: {} step(s)",
            metadata.postprocessing.len()
        );

        // Create executor
        let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

        // Create text input
        let input = Envelope {
            kind: EnvelopeKind::Text("Hello world".to_string()),
            metadata: HashMap::new(),
        };

        // Execute
        match executor.execute(&metadata, &input) {
            Ok(output) => match &output.kind {
                EnvelopeKind::Audio(bytes) => {
                    assert!(
                        !bytes.is_empty(),
                        "Model {} produced empty audio output",
                        name
                    );
                    println!("  output: Audio ({} bytes)", bytes.len());
                    passed += 1;
                }
                other => {
                    let msg = format!(
                        "Model {} produced {:?} instead of Audio",
                        name,
                        std::mem::discriminant(other)
                    );
                    eprintln!("  FAIL: {}", msg);
                    failed.push(msg);
                }
            },
            Err(e) => {
                let msg = format!("Model {} execution failed: {}", name, e);
                eprintln!("  FAIL: {}", msg);
                failed.push(msg);
            }
        }
    }

    println!("\n=== Results: {}/{} passed ===", passed, tts_models.len());

    if !failed.is_empty() {
        panic!(
            "TTS smoke test failures:\n{}",
            failed
                .iter()
                .map(|f| format!("  - {}", f))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

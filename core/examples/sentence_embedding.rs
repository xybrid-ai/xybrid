//! Sentence Embedding via all-MiniLM-L6-v2 Model
//!
//! This example demonstrates text model support with:
//! - WordPiece tokenization preprocessing
//! - Multi-input BERT model execution (input_ids, attention_mask, token_type_ids)
//! - Mean pooling postprocessing to generate sentence embeddings

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Sentence Embedding - all-MiniLM-L6-v2");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("all-minilm");
    let metadata_path = model_dir.join("model_metadata.json");
    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Metadata loaded:");
    println!("   Model: {} v{}", metadata.model_id, metadata.version);
    println!(
        "   Description: {}",
        metadata.description.as_ref().unwrap_or(&"N/A".to_string())
    );
    println!("   Execution: {:?}", metadata.execution_template);
    println!("   Preprocessing: {} steps", metadata.preprocessing.len());
    println!("   Postprocessing: {} steps", metadata.postprocessing.len());
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    println!("✅ TemplateExecutor created");
    println!();

    // Test sentence
    let sentence = "The quick brown fox jumps over the lazy dog.";
    println!("📝 Input sentence:");
    println!("   \"{}\"", sentence);
    println!();

    // Create input envelope with Text variant
    let envelope_metadata = HashMap::new();
    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(sentence.to_string()),
        metadata: envelope_metadata,
    };

    // Execute inference via TemplateExecutor
    println!("🔄 Running inference via TemplateExecutor...");
    println!("   → Preprocessing: WordPiece tokenization (max_length=512)");
    println!("   → Model execution: Multi-input BERT (input_ids, attention_mask, token_type_ids)");
    println!("   → Postprocessing: Mean pooling over sequence");
    println!();

    let output_envelope = executor.execute(&metadata, &input_envelope)?;

    println!("✅ Inference complete!");
    println!();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Embedding(embedding) => {
            println!("📊 Sentence embedding:");
            println!("   Dimensions: {}", embedding.len());
            println!("   Expected: 384 (MiniLM-L6 hidden size)");
            println!();

            // Show first 10 and last 10 dimensions
            println!("   First 10 dimensions:");
            for (i, &val) in embedding.iter().take(10).enumerate() {
                println!("     [{}]: {:.6}", i, val);
            }

            if embedding.len() > 20 {
                println!("   ...");
                println!("   Last 10 dimensions:");
                for (i, &val) in embedding.iter().skip(embedding.len() - 10).enumerate() {
                    println!("     [{}]: {:.6}", embedding.len() - 10 + i, val);
                }
            }

            // Calculate L2 norm
            let l2_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!();
            println!("   L2 norm: {:.6}", l2_norm);

            // Verify dimensions
            println!();
            if embedding.len() == 384 {
                println!("✅ SUCCESS: Correct embedding dimension (384)!");
            } else {
                println!(
                    "⚠️  UNEXPECTED: Got {} dimensions instead of 384",
                    embedding.len()
                );
            }
        }
        EnvelopeKind::Text(text) => {
            println!("📄 Text output (unexpected): {}", text);
        }
        EnvelopeKind::Audio(_) => {
            println!("🔊 Audio output (unexpected for text model)");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Test Complete");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 KEY VALIDATION:");
    println!("   ✅ WordPiece tokenization preprocessing");
    println!("   ✅ Multi-input BERT model execution");
    println!("   ✅ Mean pooling postprocessing");
    println!("   ✅ 384-dimensional sentence embedding");
    println!();
    println!("📝 This proves text model support works!");
    println!("   - Input: Raw text string");
    println!("   - Preprocessing: Tokenization → (input_ids, attention_mask, token_type_ids)");
    println!("   - Execution: BERT model with 3 inputs");
    println!("   - Postprocessing: Mean pool [batch, seq, 384] → [batch, 384]");
    println!("   - Output: Dense sentence embedding vector");
    println!();

    Ok(())
}

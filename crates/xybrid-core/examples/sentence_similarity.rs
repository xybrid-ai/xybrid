//! Sentence Similarity using all-MiniLM-L6-v2
//!
//! This example demonstrates:
//! - Encoding two sentences into embeddings
//! - Computing cosine similarity between embeddings
//! - Comparing similar vs dissimilar sentence pairs

use std::collections::HashMap;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();

    dot_product / (norm_a * norm_b)
}

/// Encode a sentence into an embedding vector
fn encode_sentence(
    executor: &mut TemplateExecutor,
    metadata: &ModelMetadata,
    sentence: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let envelope_metadata = HashMap::new();
    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(sentence.to_string()),
        metadata: envelope_metadata,
    };

    let output_envelope = executor.execute(metadata, &input_envelope, None)?;

    match &output_envelope.kind {
        EnvelopeKind::DecisionRequest(_) | EnvelopeKind::Decisions(_) => {
            Err("Unexpected decision output".into())
        }
        EnvelopeKind::Embedding(embedding) => Ok(embedding.clone()),
        _ => Err("Unexpected output type".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Sentence Similarity - all-MiniLM-L6-v2");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("all-minilm");
    let metadata_path = model_dir.join("model_metadata.json");
    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Model: {} v{}", metadata.model_id, metadata.version);
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Test sentence pairs
    let pairs = [
        (
            "The cat sits on the mat.",
            "A feline rests on the rug.",
            "Similar meaning",
        ),
        (
            "I love machine learning.",
            "Machine learning is fascinating to me.",
            "Similar meaning",
        ),
        (
            "The weather is sunny today.",
            "Quantum physics is complex.",
            "Different topics",
        ),
        (
            "Python is a programming language.",
            "The python is a type of snake.",
            "Different meanings (homonym)",
        ),
    ];

    for (i, (sentence1, sentence2, description)) in pairs.iter().enumerate() {
        println!("═══════════════════════════════════════════════════════");
        println!("  Pair {} - {}", i + 1, description);
        println!("═══════════════════════════════════════════════════════");
        println!();
        println!("📝 Sentence 1:");
        println!("   \"{}\"", sentence1);
        println!();
        println!("📝 Sentence 2:");
        println!("   \"{}\"", sentence2);
        println!();

        // Encode both sentences
        println!("🔄 Encoding sentences...");
        let embedding1 = encode_sentence(&mut executor, &metadata, sentence1)?;
        let embedding2 = encode_sentence(&mut executor, &metadata, sentence2)?;
        println!("   ✅ Sentence 1: {} dimensions", embedding1.len());
        println!("   ✅ Sentence 2: {} dimensions", embedding2.len());
        println!();

        // Compute cosine similarity
        let similarity = cosine_similarity(&embedding1, &embedding2);
        println!(
            "📊 Cosine Similarity: {:.4} ({:.1}%)",
            similarity,
            similarity * 100.0
        );
        println!();

        // Interpretation
        if similarity > 0.8 {
            println!("   💚 Very similar (> 0.8)");
        } else if similarity > 0.6 {
            println!("   💛 Moderately similar (0.6 - 0.8)");
        } else if similarity > 0.4 {
            println!("   🧡 Somewhat similar (0.4 - 0.6)");
        } else {
            println!("   ❤️  Dissimilar (< 0.4)");
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 KEY ACHIEVEMENTS:");
    println!("   ✅ Text model support (WordPiece tokenization)");
    println!("   ✅ Multi-input BERT execution");
    println!("   ✅ Mean pooling for sentence embeddings");
    println!("   ✅ Cosine similarity computation");
    println!("   ✅ Semantic similarity detection");
    println!();
    println!("📝 Use Cases:");
    println!("   • Semantic search");
    println!("   • Document clustering");
    println!("   • Duplicate detection");
    println!("   • Question answering");
    println!("   • Recommendation systems");
    println!();

    Ok(())
}

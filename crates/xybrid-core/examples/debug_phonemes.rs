//! Debug phonemizer output — prints phonemes and token IDs for given text.
//!
//! Usage:
//!   cargo run --example debug_phonemes "Your text here"

use std::collections::HashMap;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::phonemizer::load_tokens_map;
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello world".to_string());

    println!("Input text ({} chars): {:?}", text.len(), text);
    println!();

    let model_dir = model_fixtures::require_model("kitten-tts-nano-0.2");
    let tokens_path = model_dir.join("tokens.txt");
    let metadata_path = model_dir.join("model_metadata.json");

    // Load and display tokens map
    let tokens_content = std::fs::read_to_string(&tokens_path)?;
    let tokens_map = load_tokens_map(&tokens_content);
    let _reverse_map: HashMap<i64, char> = tokens_map.iter().map(|(&c, &id)| (id, c)).collect();

    // Load metadata
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    println!(
        "Model: {} (max_chunk_chars: {:?}, trim_trailing_samples: {:?})",
        metadata.model_id, metadata.max_chunk_chars, metadata.trim_trailing_samples
    );
    println!();

    // Create executor (we won't actually run inference, just check phonemization)
    // Since we can't access preprocessing directly, let's do manual phonemization
    // using the public phonemizer API

    // Manual phonemization path: load misaki dictionaries and phonemize
    let misaki_dir = model_dir.join("misaki");
    if !misaki_dir.exists() {
        println!(
            "ERROR: misaki/ directory not found at {}",
            misaki_dir.display()
        );
        return Ok(());
    }

    // Use the public TemplateExecutor to run the full pipeline and see the output
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    let input = Envelope {
        kind: EnvelopeKind::Text(text.clone()),
        metadata: HashMap::new(),
    };

    println!("Running full TTS pipeline...");
    let output = executor.execute(&metadata, &input, None)?;

    match &output.kind {
        EnvelopeKind::Audio(bytes) => {
            let sample_rate = 24000;
            let num_samples = bytes.len() / 2;
            let duration = num_samples as f32 / sample_rate as f32;
            println!(
                "Output: {} bytes, {:.2}s at {}Hz",
                bytes.len(),
                duration,
                sample_rate
            );
        }
        other => println!("Unexpected output: {:?}", std::mem::discriminant(other)),
    }

    Ok(())
}

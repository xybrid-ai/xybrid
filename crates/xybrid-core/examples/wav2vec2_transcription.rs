//! Wav2Vec2 Speech Recognition Example
//!
//! This example demonstrates:
//! - Loading a WAV audio file
//! - Using Wav2Vec2 ONNX model for ASR (Automatic Speech Recognition)
//! - CTC decoding to convert model outputs to text
//!
//! Model: facebook/wav2vec2-base-960h (ONNX format)
//! Expected input: 16kHz mono WAV audio

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Wav2Vec2 Speech Recognition - ASR Demo");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("wav2vec2-base-960h");
    let metadata_path = model_dir.join("model_metadata.json");
    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Model: {} v{}", metadata.model_id, metadata.version);
    println!("   Architecture: Wav2Vec2ForCTC");
    println!("   Sample Rate: 16kHz");
    println!("   Vocabulary: 32 characters");
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Test with a sample audio file (user should provide this)
    let audio_path = std::env::args().nth(1).unwrap_or_else(|| {
        println!("⚠️  No audio file provided!");
        println!();
        println!("Usage: cargo run --example wav2vec2_transcription <audio.wav>");
        println!();
        println!("Example:");
        println!("  cargo run --example wav2vec2_transcription test_audio.wav");
        println!();
        println!("Note: Audio must be WAV format (any sample rate, will be resampled to 16kHz)");
        std::process::exit(1);
    });

    println!("🎤 Loading audio from: {}", audio_path);

    // Read audio file
    let audio_bytes = std::fs::read(&audio_path)?;
    println!("   File size: {} bytes", audio_bytes.len());
    println!();

    // Create input envelope
    let envelope_metadata = HashMap::new();
    let input_envelope = Envelope {
        kind: EnvelopeKind::Audio(audio_bytes),
        metadata: envelope_metadata,
    };

    println!("🔄 Running ASR pipeline...");
    println!("   1. AudioDecode: WAV → PCM samples (16kHz mono)");
    println!("   2. Wav2Vec2: Audio waveform → CTC logits");
    println!("   3. CTCDecode: Logits → Text transcription");
    println!();

    // Execute inference
    let output_envelope = executor.execute(&metadata, &input_envelope)?;

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Text(transcription) => {
            println!("═══════════════════════════════════════════════════════");
            println!("  Transcription Result");
            println!("═══════════════════════════════════════════════════════");
            println!();
            println!("📝 \"{}\"", transcription);
            println!();
        }
        _ => {
            println!("❌ Unexpected output type: {:?}", output_envelope.kind);
            return Err("Expected text output".into());
        }
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  Pipeline Summary");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("✅ Audio model support working!");
    println!();
    println!("🎯 Key Features:");
    println!("   • Metadata-driven ASR (no model-specific code)");
    println!("   • WAV file decoding with auto-resampling");
    println!("   • CTC decoding for character-level transcription");
    println!("   • Ready for mobile deployment");
    println!();
    println!("📊 Processing Steps Used:");
    println!("   Preprocessing:  AudioDecode");
    println!("   Postprocessing: CTCDecode");
    println!();

    Ok(())
}

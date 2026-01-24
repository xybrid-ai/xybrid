//! TTS using Xybrid Execution System
//!
//! This example validates that KittenTTS works through the standard
//! xybrid execution pipeline (TemplateExecutor + model_metadata.json).
//!
//! This is the correct way to use TTS - same pattern as wav2vec2_transcription.rs
//!
//! Prerequisites:
//! - Download model: ./integration-tests/download.sh kitten-tts
//! - model_metadata.json with Phonemize preprocessing
//! - cmudict.dict in the model directory
//!
//! Usage:
//!   cargo run --example tts_xybrid
//!   cargo run --example tts_xybrid "Hello, how are you today?"

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  TTS using Xybrid Execution System");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Get input text from command line or use default
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello world".to_string());

    println!("📝 Input text: \"{}\"", text);
    println!();

    // Load metadata (same pattern as wav2vec2_transcription.rs)
    let model_dir = model_fixtures::require_model("kitten-tts");
    let metadata_path = model_dir.join("model_metadata.json");

    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Model: {} v{}", metadata.model_id, metadata.version);
    println!("   Task: {:?}", metadata.metadata.get("task"));
    println!("   Preprocessing: {} step(s)", metadata.preprocessing.len());
    for (i, step) in metadata.preprocessing.iter().enumerate() {
        println!("     {}. {:?}", i + 1, step);
    }
    println!();

    // Create TemplateExecutor (same pattern as wav2vec2)
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create input envelope with TEXT (not audio like ASR)
    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(text.clone()),
        metadata: HashMap::new(),
    };

    println!("🔄 Running TTS pipeline...");
    println!("   1. Phonemize: Text → IPA phonemes → Token IDs");
    println!("   2. KittenTTS ONNX: Token IDs + Voice → Waveform");
    println!();

    // Execute inference
    let output_envelope = executor.execute(&metadata, &input_envelope)?;

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Audio(audio_bytes) => {
            println!("═══════════════════════════════════════════════════════");
            println!("  TTS Result");
            println!("═══════════════════════════════════════════════════════");
            println!();
            println!("✅ Generated {} audio bytes", audio_bytes.len());

            // Assuming 24kHz, 16-bit mono PCM
            let sample_rate = 24000;
            let bytes_per_sample = 2; // 16-bit
            let num_samples = audio_bytes.len() / bytes_per_sample;
            let duration_secs = num_samples as f32 / sample_rate as f32;

            println!("   Duration: {:.2}s at {}Hz", duration_secs, sample_rate);
            println!();

            // Save to WAV file
            let output_path = PathBuf::from("tts_xybrid_output.wav");
            save_wav(&output_path, audio_bytes, sample_rate)?;
            println!("💾 Saved to: {}", output_path.display());
            println!();
            println!("🎵 Play the output:");
            println!("   afplay tts_xybrid_output.wav   # macOS");
            println!("   aplay tts_xybrid_output.wav    # Linux");
        }
        EnvelopeKind::Text(text) => {
            println!("❌ Unexpected text output: {}", text);
            return Err("Expected audio output, got text".into());
        }
        EnvelopeKind::Embedding(emb) => {
            println!("❌ Unexpected embedding output: {} dimensions", emb.len());
            return Err("Expected audio output, got embedding".into());
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Pipeline Summary");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("✅ TTS working through Xybrid execution system!");
    println!();
    println!("🎯 This validates:");
    println!("   • model_metadata.json is correct");
    println!("   • Phonemize preprocessing works");
    println!("   • TemplateExecutor handles TTS models");
    println!("   • Ready for registry/Flutter integration");
    println!();

    Ok(())
}

/// Save raw audio bytes as WAV file
fn save_wav(
    path: &PathBuf,
    audio_bytes: &[u8],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Assuming input is already 16-bit PCM samples
    let data_size = audio_bytes.len() as u32;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?; // PCM
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    file.write_all(audio_bytes)?;

    Ok(())
}

//! Kokoro-82M TTS using Misaki Dictionary (no espeak-ng required)
//!
//! This example validates that Kokoro-82M works with the dictionary-based
//! phonemizer, making it suitable for mobile/embedded deployment.
//!
//! Prerequisites:
//! - Download model: ./integration-tests/download.sh kokoro-82m
//! - model_metadata.json with MisakiDictionary backend
//! - misaki/ directory with us_gold.json and us_silver.json
//!
//! Usage:
//!   cargo run --example tts_kokoro_misaki
//!   cargo run --example tts_kokoro_misaki "Hello, how are you today?"

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Kokoro-82M TTS with Misaki Dictionary");
    println!("  (No espeak-ng dependency)");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Get input text from command line or use default
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello world, this is Kokoro speaking with Misaki.".to_string());

    println!("📝 Input text: \"{}\"", text);
    println!();

    // Load metadata with Misaki backend (default for kokoro-82m)
    let model_dir = model_fixtures::require_model("kokoro-82m");
    let metadata_path = model_dir.join("model_metadata.json");

    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Model: {} v{}", metadata.model_id, metadata.version);
    if let Some(desc) = &metadata.description {
        println!("   Description: {}", desc);
    }
    println!("   Backend: MisakiDictionary (dictionary-based, no system deps)");
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create input envelope
    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(text.clone()),
        metadata: HashMap::new(),
    };

    println!("🔄 Running Kokoro TTS pipeline...");
    println!("   1. Phonemize: Text → IPA phonemes (Misaki dict) → Token IDs");
    println!("   2. Kokoro ONNX: Token IDs + Voice → Waveform");
    println!();

    // Execute inference
    let output_envelope = executor.execute(&metadata, &input_envelope)?;

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Audio(audio_bytes) => {
            println!("═══════════════════════════════════════════════════════");
            println!("  Kokoro TTS Result (Misaki)");
            println!("═══════════════════════════════════════════════════════");
            println!();
            println!("✅ Generated {} audio bytes", audio_bytes.len());

            let sample_rate = 24000;
            let bytes_per_sample = 2;
            let num_samples = audio_bytes.len() / bytes_per_sample;
            let duration_secs = num_samples as f32 / sample_rate as f32;

            println!("   Duration: {:.2}s at {}Hz", duration_secs, sample_rate);
            println!();

            // Save to WAV file
            let output_path = PathBuf::from("tts_kokoro_misaki_output.wav");
            save_wav(&output_path, audio_bytes, sample_rate)?;
            println!("💾 Saved to: {}", output_path.display());
            println!();
            println!("🎵 Play the output:");
            println!("   afplay tts_kokoro_misaki_output.wav   # macOS");
        }
        _ => {
            return Err("Expected audio output".into());
        }
    }

    println!();
    println!("✅ Kokoro-82M TTS working with Misaki dictionary!");
    println!("   • No espeak-ng required");
    println!("   • Ready for mobile/embedded deployment");
    println!();

    Ok(())
}

fn save_wav(
    path: &PathBuf,
    audio_bytes: &[u8],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let data_size = audio_bytes.len() as u32;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    file.write_all(audio_bytes)?;

    Ok(())
}

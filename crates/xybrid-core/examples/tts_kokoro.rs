//! Kokoro-82M TTS using Xybrid Execution System
//!
//! This example demonstrates the voice selection feature with Kokoro-82M,
//! which has 24 voices available (male/female, US/British accents).
//!
//! Prerequisites:
//! - Download model: ./integration-tests/download.sh kokoro-82m
//! - model_metadata.json with voice catalog and MisakiDictionary phonemization
//! - tokens.txt vocabulary file
//! - model.onnx model file
//! - voices.bin voice embeddings
//! - misaki/ dictionary files (us_gold.json)
//!
//! Usage:
//!   cargo run -p xybrid-core --example tts_kokoro
//!   cargo run -p xybrid-core --example tts_kokoro --voice af_bella
//!   cargo run -p xybrid-core --example tts_kokoro --voice am_adam "Hello from Adam!"
//!   cargo run -p xybrid-core --example tts_kokoro --list-voices

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

/// Simple CLI argument parser for the example
struct Args {
    text: String,
    voice_id: Option<String>,
    list_voices: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut text = "Hello world, this is Kokoro speaking with the new voice system.".to_string();
    let mut voice_id = None;
    let mut list_voices = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--voice" | "-v" => {
                if i + 1 < args.len() {
                    voice_id = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--list-voices" | "-l" => {
                list_voices = true;
            }
            "--help" | "-h" => {
                println!("Usage: tts_kokoro [OPTIONS] [TEXT]");
                println!();
                println!("Options:");
                println!("  --voice, -v <ID>    Select voice by ID (e.g., af_bella, am_adam)");
                println!("  --list-voices, -l   List all available voices");
                println!("  --help, -h          Show this help");
                println!();
                println!("Examples:");
                println!("  cargo run -p xybrid-core --example tts_kokoro");
                println!("  cargo run -p xybrid-core --example tts_kokoro --voice af_bella");
                println!("  cargo run -p xybrid-core --example tts_kokoro -v am_adam \"Hello from Adam!\"");
                println!("  cargo run -p xybrid-core --example tts_kokoro --list-voices");
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') => {
                text = arg.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    Args {
        text,
        voice_id,
        list_voices,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    println!("═══════════════════════════════════════════════════════");
    println!("  Kokoro-82M TTS - Voice Selection Demo");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("kokoro-82m");
    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("📋 Model: {} v{}", metadata.model_id, metadata.version);
    if let Some(desc) = &metadata.description {
        println!("   {}", desc);
    }
    println!();

    // Handle --list-voices
    if args.list_voices {
        print_voice_catalog(&metadata);
        return Ok(());
    }

    // Show available voices summary
    let voices = metadata.list_voices();
    if !voices.is_empty() {
        println!("🎤 Available Voices ({} total):", voices.len());
        println!();

        // Group by language
        let us_voices: Vec<_> = voices
            .iter()
            .filter(|v| v.language.as_deref() == Some("en-US"))
            .collect();
        let gb_voices: Vec<_> = voices
            .iter()
            .filter(|v| v.language.as_deref() == Some("en-GB"))
            .collect();

        println!("   🇺🇸 American English ({}):", us_voices.len());
        print_voice_group(&us_voices, 6);

        println!("   🇬🇧 British English ({}):", gb_voices.len());
        print_voice_group(&gb_voices, 6);

        println!();
    }

    // Determine which voice to use
    let selected_voice = if let Some(ref vid) = args.voice_id {
        match metadata.get_voice(vid) {
            Some(v) => v.clone(),
            None => {
                eprintln!("❌ Voice '{}' not found.", vid);
                eprintln!();
                eprintln!("Available voice IDs:");
                for v in voices.iter() {
                    eprintln!("  - {} ({})", v.id, v.name);
                }
                std::process::exit(1);
            }
        }
    } else {
        metadata
            .default_voice()
            .ok_or("No default voice configured")?
            .clone()
    };

    println!(
        "🎙️  Selected Voice: {} ({})",
        selected_voice.name, selected_voice.id
    );
    if let Some(ref gender) = selected_voice.gender {
        print!("    Gender: {}", gender);
    }
    if let Some(ref lang) = selected_voice.language {
        print!(" | Language: {}", lang);
    }
    if let Some(ref style) = selected_voice.style {
        print!(" | Style: {}", style);
    }
    println!();
    println!();

    println!("📝 Text: \"{}\"", args.text);
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create input envelope with voice_id in metadata
    let mut envelope_metadata = HashMap::new();
    envelope_metadata.insert("voice_id".to_string(), selected_voice.id.clone());

    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(args.text.clone()),
        metadata: envelope_metadata,
    };

    println!("🔄 Running TTS pipeline...");
    let start = std::time::Instant::now();
    let output_envelope = executor.execute(&metadata, &input_envelope)?;
    let elapsed = start.elapsed();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Audio(audio_bytes) => {
            // Kokoro outputs 24kHz, 16-bit mono PCM
            let sample_rate = 24000;
            let bytes_per_sample = 2;
            let num_samples = audio_bytes.len() / bytes_per_sample;
            let duration_secs = num_samples as f32 / sample_rate as f32;

            println!();
            println!("═══════════════════════════════════════════════════════");
            println!("  Result");
            println!("═══════════════════════════════════════════════════════");
            println!();
            println!(
                "✅ Generated {:.2}s of audio ({} bytes)",
                duration_secs,
                audio_bytes.len()
            );
            println!("   Processing time: {:?}", elapsed);
            println!(
                "   Real-time factor: {:.1}x",
                duration_secs / elapsed.as_secs_f32()
            );
            println!();

            // Save to WAV file with voice name
            let output_filename = format!("tts_kokoro_{}.wav", selected_voice.id);
            let output_path = PathBuf::from(&output_filename);
            save_wav(&output_path, audio_bytes, sample_rate)?;

            println!("💾 Saved: {}", output_path.display());
            println!();
            println!(
                "🎵 Play: afplay {} (macOS) or aplay {} (Linux)",
                output_filename, output_filename
            );
        }
        _ => {
            return Err("Expected audio output".into());
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Try Other Voices");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("  Female voices:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice af_bella");
    println!(
        "    cargo run -p xybrid-core --example tts_kokoro -- --voice af_nicole  # whisper style"
    );
    println!();
    println!("  Male voices:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice am_adam");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice am_fenrir");
    println!();
    println!("  British accents:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice bf_emma");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice bm_george");
    println!();
    println!("  List all voices:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --list-voices");
    println!();

    Ok(())
}

/// Print a group of voices in a compact format
fn print_voice_group(voices: &[&&xybrid_core::execution::template::VoiceInfo], max_show: usize) {
    let females: Vec<_> = voices
        .iter()
        .filter(|v| v.gender.as_deref() == Some("female"))
        .collect();
    let males: Vec<_> = voices
        .iter()
        .filter(|v| v.gender.as_deref() == Some("male"))
        .collect();

    if !females.is_empty() {
        let names: Vec<_> = females
            .iter()
            .take(max_show)
            .map(|v| v.name.to_string())
            .collect();
        let suffix = if females.len() > max_show {
            format!(" +{}", females.len() - max_show)
        } else {
            String::new()
        };
        println!("      ♀ {}{}", names.join(", "), suffix);
    }
    if !males.is_empty() {
        let names: Vec<_> = males
            .iter()
            .take(max_show)
            .map(|v| v.name.to_string())
            .collect();
        let suffix = if males.len() > max_show {
            format!(" +{}", males.len() - max_show)
        } else {
            String::new()
        };
        println!("      ♂ {}{}", names.join(", "), suffix);
    }
}

/// Print the full voice catalog
fn print_voice_catalog(metadata: &ModelMetadata) {
    let voices = metadata.list_voices();
    if voices.is_empty() {
        println!("No voices available.");
        return;
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  Kokoro-82M Voice Catalog ({} voices)", voices.len());
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Group by language
    let us_voices: Vec<_> = voices
        .iter()
        .filter(|v| v.language.as_deref() == Some("en-US"))
        .collect();
    let gb_voices: Vec<_> = voices
        .iter()
        .filter(|v| v.language.as_deref() == Some("en-GB"))
        .collect();

    if !us_voices.is_empty() {
        println!("🇺🇸 American English ({} voices)", us_voices.len());
        println!("─────────────────────────────────────────────────────");
        println!("{:<15} {:<12} {:<8} Style", "ID", "Name", "Gender");
        println!("─────────────────────────────────────────────────────");
        for v in us_voices {
            println!(
                "{:<15} {:<12} {:<8} {}",
                v.id,
                v.name,
                v.gender.as_deref().unwrap_or("-"),
                v.style.as_deref().unwrap_or("neutral")
            );
        }
        println!();
    }

    if !gb_voices.is_empty() {
        println!("🇬🇧 British English ({} voices)", gb_voices.len());
        println!("─────────────────────────────────────────────────────");
        println!("{:<15} {:<12} {:<8} Style", "ID", "Name", "Gender");
        println!("─────────────────────────────────────────────────────");
        for v in gb_voices {
            println!(
                "{:<15} {:<12} {:<8} {}",
                v.id,
                v.name,
                v.gender.as_deref().unwrap_or("-"),
                v.style.as_deref().unwrap_or("neutral")
            );
        }
        println!();
    }

    if let Some(default) = metadata.default_voice() {
        println!("Default voice: {} ({})", default.name, default.id);
    }
    println!();
    println!("Usage: cargo run -p xybrid-core --example tts_kokoro -- --voice <ID>");
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

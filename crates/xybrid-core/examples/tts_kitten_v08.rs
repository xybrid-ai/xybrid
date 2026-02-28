//! KittenTTS Nano v0.8 TTS using Xybrid Execution System + OpenPhonemizer
//!
//! This example demonstrates KittenTTS Nano v0.8 with the OpenPhonemizer backend,
//! which requires no system dependencies (no espeak-ng needed).
//!
//! Prerequisites:
//! - Model files in fixtures/models/kitten-tts-nano-0.8/:
//!   model.onnx, voices.npz, tokens.txt, open-phonemizer.onnx, dictionary.json
//!
//! Usage:
//!   cargo run -p xybrid-core --example tts_kitten_v08
//!   cargo run -p xybrid-core --example tts_kitten_v08 -- --voice Bella
//!   cargo run -p xybrid-core --example tts_kitten_v08 -- --voice Leo "Hello from Leo!"
//!   cargo run -p xybrid-core --example tts_kitten_v08 -- --speed 1.0
//!   cargo run -p xybrid-core --example tts_kitten_v08 -- --list-voices

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

struct Args {
    text: String,
    voice_name: Option<String>,
    speed: Option<f32>,
    list_voices: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut text = "Hello world, this is KittenTTS speaking with OpenPhonemizer.".to_string();
    let mut voice_name = None;
    let mut speed = None;
    let mut list_voices = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--voice" | "-v" => {
                if i + 1 < args.len() {
                    voice_name = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--speed" => {
                if i + 1 < args.len() {
                    speed = args[i + 1].parse::<f32>().ok();
                    i += 1;
                }
            }
            "--list-voices" | "-l" => {
                list_voices = true;
            }
            "--help" | "-h" => {
                println!("Usage: tts_kitten_v08 [OPTIONS] [TEXT]");
                println!();
                println!("Options:");
                println!("  --voice, -v <NAME>  Select voice by name (e.g., Bella, Leo)");
                println!("  --speed <FLOAT>     Speech speed 0.5-2.0 (default: use speed prior)");
                println!("  --list-voices, -l   List all available voices");
                println!("  --help, -h          Show this help");
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
        voice_name,
        speed,
        list_voices,
    }
}

/// Look up speed prior from model metadata for the given voice ID.
fn get_speed_prior(metadata: &ModelMetadata, voice_id: &str) -> f32 {
    metadata
        .metadata
        .get("speed_priors")
        .and_then(|sp| sp.as_object())
        .and_then(|obj| obj.get(voice_id))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    println!("═══════════════════════════════════════════════════════");
    println!("  KittenTTS Nano v0.8 - OpenPhonemizer (no system deps)");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("kitten-tts-nano-0.8");
    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("Model: {} v{}", metadata.model_id, metadata.version);
    if let Some(desc) = &metadata.description {
        println!("  {}", desc);
    }
    println!();

    // Handle --list-voices
    if args.list_voices {
        let voices = metadata.list_voices();
        println!("Available voices ({}):", voices.len());
        println!("{:<20} {:<12} {:<8}", "ID", "Name", "Gender");
        println!("{}", "-".repeat(44));
        for v in &voices {
            println!(
                "{:<20} {:<12} {:<8}",
                v.id,
                v.name,
                v.gender.as_deref().unwrap_or("-")
            );
        }
        if let Some(default) = metadata.default_voice() {
            println!();
            println!("Default: {} ({})", default.name, default.id);
        }
        return Ok(());
    }

    // Resolve voice: accept friendly name (Bella) or canonical ID (expr-voice-2-f)
    let voices = metadata.list_voices();
    let selected_voice = if let Some(ref name) = args.voice_name {
        // Try by name first (case-insensitive), then by ID
        voices
            .iter()
            .find(|v| v.name.eq_ignore_ascii_case(name))
            .or_else(|| voices.iter().find(|v| v.id == *name))
            .ok_or_else(|| {
                format!(
                    "Voice '{}' not found. Available: {}",
                    name,
                    voices
                        .iter()
                        .map(|v| v.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?
    } else {
        metadata
            .default_voice()
            .ok_or("No default voice configured")?
    };

    let voice_id = &selected_voice.id;

    // Determine speed: user override > speed prior from metadata > 1.0
    let speed_prior = get_speed_prior(&metadata, voice_id);
    let effective_speed = args.speed.unwrap_or(speed_prior);

    println!("Voice: {} ({})", selected_voice.name, voice_id);
    println!(
        "Speed: {:.1}x{}",
        effective_speed,
        if args.speed.is_none() {
            " (speed prior)"
        } else {
            ""
        }
    );
    println!("Text: \"{}\"", args.text);
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create input envelope with voice_id and speed
    let mut envelope_metadata = HashMap::new();
    envelope_metadata.insert("voice_id".to_string(), voice_id.clone());
    envelope_metadata.insert("speed".to_string(), effective_speed.to_string());

    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(args.text.clone()),
        metadata: envelope_metadata,
    };

    println!("Running TTS pipeline (OpenPhonemizer + KittenTTS ONNX)...");
    let start = std::time::Instant::now();
    let output_envelope = executor.execute(&metadata, &input_envelope, None)?;
    let elapsed = start.elapsed();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Audio(audio_bytes) => {
            let sample_rate = 24000;
            let bytes_per_sample = 2;
            let num_samples = audio_bytes.len() / bytes_per_sample;
            let duration_secs = num_samples as f32 / sample_rate as f32;

            println!();
            println!("Result:");
            println!(
                "  Audio: {:.2}s ({} bytes)",
                duration_secs,
                audio_bytes.len()
            );
            println!("  Time: {:?}", elapsed);
            println!(
                "  RTF: {:.1}x realtime",
                duration_secs / elapsed.as_secs_f32()
            );
            println!();

            // Save to WAV
            let output_filename =
                format!("tts_kitten_v08_{}.wav", selected_voice.name.to_lowercase());
            let output_path = PathBuf::from(&output_filename);
            save_wav(&output_path, audio_bytes, sample_rate)?;

            println!("Saved: {}", output_path.display());
            println!("Play:  afplay {} (macOS)", output_filename);
        }
        _ => {
            return Err("Expected audio output".into());
        }
    }

    Ok(())
}

/// Save raw audio bytes as WAV file
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

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
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

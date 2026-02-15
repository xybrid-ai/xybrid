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
//!   cargo run -p xybrid-core --example tts_kokoro -- --voice af_bella
//!   cargo run -p xybrid-core --example tts_kokoro -- --voice am_adam "Hello from Adam!"
//!   cargo run -p xybrid-core --example tts_kokoro -- --silence-tokens 2
//!   cargo run -p xybrid-core --example tts_kokoro -- --speed 0.8
//!   cargo run -p xybrid-core --example tts_kokoro -- --voice "af_heart.5+am_adam.5"
//!   cargo run -p xybrid-core --example tts_kokoro -- --long-text
//!   cargo run -p xybrid-core --example tts_kokoro -- --cjk
//!   cargo run -p xybrid-core --example tts_kokoro -- --list-voices

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::template::PreprocessingStep;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

/// Simple CLI argument parser for the example
struct Args {
    text: String,
    voice_id: Option<String>,
    list_voices: bool,
    silence_tokens: Option<u8>,
    speed: Option<f32>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut text = "Hello world, this is Kokoro speaking with the new voice system.".to_string();
    let mut voice_id = None;
    let mut list_voices = false;
    let mut silence_tokens = None;
    let mut speed = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--voice" | "-v" => {
                if i + 1 < args.len() {
                    voice_id = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--silence-tokens" | "-s" => {
                if i + 1 < args.len() {
                    silence_tokens = args[i + 1].parse::<u8>().ok();
                    i += 1;
                }
            }
            "--speed" => {
                if i + 1 < args.len() {
                    speed = args[i + 1].parse::<f32>().ok();
                    i += 1;
                }
            }
            "--long-text" => {
                text = "The advances in artificial intelligence over the past decade have been nothing \
                    short of remarkable, transforming industries from healthcare to transportation. \
                    Machine learning algorithms now power everything from voice assistants to \
                    autonomous vehicles, and the pace of innovation shows no signs of slowing down. \
                    However, with these advances come significant challenges, because ethical \
                    considerations around privacy, bias, and job displacement must be carefully \
                    addressed if we are to build a future where technology serves all of humanity \
                    equally and justly."
                    .to_string();
            }
            "--cjk" => {
                text = "Hello、welcome to the demo。This tests CJK punctuation！Does it work？Yes，it maps correctly。Let's verify：semicolons too；and exclamation marks！".to_string();
            }
            "--list-voices" | "-l" => {
                list_voices = true;
            }
            "--help" | "-h" => {
                println!("Usage: tts_kokoro [OPTIONS] [TEXT]");
                println!();
                println!("Options:");
                println!(
                    "  --voice, -v <ID>      Select voice (e.g., af_bella, af_heart.5+am_adam.5)"
                );
                println!("  --silence-tokens, -s <N>  Prepend N silence tokens before speech (default: from model config)");
                println!("  --speed <FLOAT>       Speech speed 0.5-2.0 (default: 1.0)");
                println!("  --long-text           Use a built-in 400+ char paragraph (tests center-break + crossfade)");
                println!("  --cjk                 Use a built-in CJK-punctuated string (tests punctuation mapping)");
                println!("  --list-voices, -l     List all available voices");
                println!("  --help, -h            Show this help");
                println!();
                println!("Examples:");
                println!("  cargo run -p xybrid-core --example tts_kokoro");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- --voice af_bella");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- -v am_adam \"Hello from Adam!\"");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- --silence-tokens 2");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- --speed 0.8");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- --voice \"af_heart.5+am_adam.5\"");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- --long-text");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- --cjk");
                println!("  cargo run -p xybrid-core --example tts_kokoro -- --list-voices");
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
        silence_tokens,
        speed,
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
    let mut metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    // Override silence_tokens in Phonemize step if requested
    if let Some(st) = args.silence_tokens {
        for step in &mut metadata.preprocessing {
            if let PreprocessingStep::Phonemize { silence_tokens, .. } = step {
                *silence_tokens = Some(st);
            }
        }
    }

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
    // Compound voice IDs (e.g., "af_heart.5+am_adam.5") bypass catalog lookup
    let is_compound = args.voice_id.as_ref().is_some_and(|vid| vid.contains('+'));

    let voice_id_for_envelope = if is_compound {
        let vid = args.voice_id.as_ref().unwrap();
        println!("🎙️  Voice Mix: {}", vid);
        println!();
        vid.clone()
    } else {
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
        selected_voice.id.clone()
    };

    // Show active settings
    if args.silence_tokens.is_some() || args.speed.is_some() {
        println!("⚙️  Settings:");
        if let Some(st) = args.silence_tokens {
            println!("    Silence tokens: {}", st);
        }
        if let Some(spd) = args.speed {
            println!("    Speed: {:.1}x", spd);
        }
        println!();
    }

    println!("📝 Text: \"{}\"", args.text);
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create input envelope with voice_id in metadata
    let mut envelope_metadata = HashMap::new();
    envelope_metadata.insert("voice_id".to_string(), voice_id_for_envelope.clone());
    if let Some(spd) = args.speed {
        envelope_metadata.insert("speed".to_string(), spd.to_string());
    }

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
            let safe_name = voice_id_for_envelope.replace('+', "_mix_").replace('.', "");
            let output_filename = format!("tts_kokoro_{}.wav", safe_name);
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
    println!("  Try More Options");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("  Voices:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice af_bella");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice am_adam");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice bf_emma");
    println!();
    println!("  Voice mixing:");
    println!(
        "    cargo run -p xybrid-core --example tts_kokoro -- --voice \"af_heart.5+am_adam.5\""
    );
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --voice \"af_bella.3+af_nicole.3+af_heart.4\"");
    println!();
    println!("  Silence tokens (smooth plosive onsets):");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --silence-tokens 2");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- -s 0  # compare without");
    println!();
    println!("  Speed control:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --speed 0.8  # slower");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --speed 1.3  # faster");
    println!();
    println!("  Center-break chunking + crossfade (text >350 chars):");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --long-text");
    println!();
    println!("  CJK punctuation mapping:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- --cjk");
    println!();
    println!("  Combine options:");
    println!("    cargo run -p xybrid-core --example tts_kokoro -- -v af_bella -s 2 --speed 0.9 \"Hello world\"");
    println!(
        "    cargo run -p xybrid-core --example tts_kokoro -- --long-text --speed 0.8 -v am_adam"
    );
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

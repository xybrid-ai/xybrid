//! NeuTTS Codec TTS using Xybrid Execution System
//!
//! This example demonstrates NeuTTS (codec-based TTS) end-to-end via
//! TemplateExecutor + model_metadata.json. NeuTTS uses a GGUF LLM backbone
//! to generate discrete speech tokens, then decodes them via an ONNX codec.
//!
//! Prerequisites:
//! - NeuTTS model files in the fixture directory (GGUF model + NeuCodec decoder)
//! - Precomputed voice codes (.bin) and transcripts (.txt) in voices/ subdirectory
//! - espeak-ng installed for PhonemeRaw preprocessing
//!
//! Usage:
//!   cargo run -p xybrid-core --example neutts_tts --features llm-llamacpp
//!   cargo run -p xybrid-core --example neutts_tts --features llm-llamacpp -- --voice dave
//!   cargo run -p xybrid-core --example neutts_tts --features llm-llamacpp -- --model-dir path/to/neutts-nano-q4
//!   cargo run -p xybrid-core --example neutts_tts --features llm-llamacpp -- --text "Hello world" --output out.wav

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

struct Args {
    text: String,
    voice: Option<String>,
    model_dir: Option<String>,
    output: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut text = "Hello, this is a test of the NeuTTS codec speech synthesis system.".to_string();
    let mut voice = None;
    let mut model_dir = None;
    let mut output = "neutts_output.wav".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--text" | "-t" if i + 1 < args.len() => {
                text = args[i + 1].clone();
                i += 1;
            }
            "--voice" | "-v" if i + 1 < args.len() => {
                voice = Some(args[i + 1].clone());
                i += 1;
            }
            "--model-dir" | "-m" if i + 1 < args.len() => {
                model_dir = Some(args[i + 1].clone());
                i += 1;
            }
            "--output" | "-o" if i + 1 < args.len() => {
                output = args[i + 1].clone();
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: neutts_tts [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --text, -t <TEXT>         Text to synthesize (default: demo sentence)");
                println!("  --voice, -v <ID>          Voice ID (e.g., jo, dave)");
                println!("  --model-dir, -m <PATH>    Path to NeuTTS model directory");
                println!(
                    "  --output, -o <PATH>       Output WAV file (default: neutts_output.wav)"
                );
                println!("  --help, -h                Show this help");
                println!();
                println!("Examples:");
                println!("  cargo run -p xybrid-core --example neutts_tts --features llm-llamacpp");
                println!("  cargo run -p xybrid-core --example neutts_tts --features llm-llamacpp -- --voice dave");
                println!("  cargo run -p xybrid-core --example neutts_tts --features llm-llamacpp -- -t \"Good morning\" -o morning.wav");
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
        voice,
        model_dir,
        output,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    println!("═══════════════════════════════════════════════════════");
    println!("  NeuTTS Codec TTS - Xybrid Execution System");
    println!("═══════════════════════════════════════════════════════");
    println!();

    let model_dir = if let Some(ref dir) = args.model_dir {
        PathBuf::from(dir)
    } else {
        model_fixtures::require_model("neutts-nano-q4")
    };

    let metadata_path = model_dir.join("model_metadata.json");
    println!("Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("Model: {} v{}", metadata.model_id, metadata.version);
    if let Some(ref desc) = metadata.description {
        println!("  {}", desc);
    }
    println!();

    // Show available voices
    let voices = metadata.list_voices();
    if !voices.is_empty() {
        println!("Available voices:");
        for v in &voices {
            let marker = if metadata
                .voices
                .as_ref()
                .is_some_and(|vc| vc.default == v.id)
            {
                " (default)"
            } else {
                ""
            };
            println!("  - {} ({}){}", v.id, v.name, marker);
        }
        println!();
    }

    println!("Text: \"{}\"", args.text);

    let mut envelope_metadata = HashMap::new();
    if let Some(ref voice_id) = args.voice {
        println!("Voice: {}", voice_id);
        envelope_metadata.insert("voice_id".to_string(), voice_id.clone());
    }
    println!();

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(args.text.clone()),
        metadata: envelope_metadata,
    };

    println!("Running NeuTTS pipeline...");
    println!("  1. PhonemeRaw: Text -> IPA phonemes (raw)");
    println!("  2. Load voice reference codes + transcript");
    println!("  3. Build prompt with phonemes + reference codes");
    println!("  4. LLM generate speech tokens (llama.cpp)");
    println!("  5. CodecDecode: Speech tokens -> PCM audio (NeuCodec ONNX)");
    println!();

    let start = std::time::Instant::now();
    let output_envelope = executor.execute(&metadata, &input_envelope, None)?;
    let elapsed = start.elapsed();

    match &output_envelope.kind {
        EnvelopeKind::Audio(audio_bytes) => {
            let sample_rate = 24000;
            let num_samples = audio_bytes.len() / 2;
            let duration_secs = num_samples as f32 / sample_rate as f32;

            println!("═══════════════════════════════════════════════════════");
            println!("  Result");
            println!("═══════════════════════════════════════════════════════");
            println!();
            println!(
                "Generated {:.2}s of audio ({} bytes)",
                duration_secs,
                audio_bytes.len()
            );
            println!("Processing time: {:?}", elapsed);
            println!(
                "Real-time factor: {:.1}x",
                duration_secs / elapsed.as_secs_f32()
            );
            println!();

            let output_path = PathBuf::from(&args.output);
            save_wav(&output_path, audio_bytes, sample_rate)?;
            println!("Saved: {}", output_path.display());
            println!();
            println!(
                "Play: afplay {} (macOS) or aplay {} (Linux)",
                args.output, args.output
            );
        }
        _ => {
            return Err("Expected audio output from NeuTTS pipeline".into());
        }
    }

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

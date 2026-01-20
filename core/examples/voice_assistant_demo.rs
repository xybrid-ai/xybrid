//! Voice Assistant Demo - Full Pipeline
//!
//! This example demonstrates a complete voice assistant pipeline:
//! 1. Voice Input → ASR (Wav2Vec2) → Text transcription
//! 2. Text → LLM (via Gateway or Direct API) → Response
//! 3. Response → TTS (KittenTTS) → Audio output
//!
//! Prerequisites:
//! - Download models: ./integration-tests/download.sh wav2vec2-base-960h kitten-tts
//! - CMU dictionary at ~/.xybrid/cmudict.dict
//! - For gateway mode: xybrid-gateway running at http://localhost:3000
//! - For direct mode: OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
//!
//! Usage:
//!   cargo run --example voice_assistant_demo [audio.wav] [gateway|openai|anthropic]
//!
//! Examples:
//!   # Use gateway (recommended - run xybrid-gateway first)
//!   cargo run --example voice_assistant_demo sample.wav gateway
//!
//!   # Use direct OpenAI API
//!   cargo run --example voice_assistant_demo sample.wav openai
//!
//!   # Use direct Anthropic API
//!   cargo run --example voice_assistant_demo sample.wav anthropic

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use xybrid_core::cloud::{Cloud, CloudConfig, CompletionRequest};
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::testing::model_fixtures;

/// Latency tracking for each pipeline stage
struct LatencyTracker {
    asr_latency: Option<Duration>,
    llm_latency: Option<Duration>,
    tts_latency: Option<Duration>,
    total_start: Instant,
}

impl LatencyTracker {
    fn new() -> Self {
        Self {
            asr_latency: None,
            llm_latency: None,
            tts_latency: None,
            total_start: Instant::now(),
        }
    }

    fn print_summary(&self) {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│                      Latency Summary                            │");
        println!("├─────────────────────────────────────────────────────────────────┤");

        if let Some(asr) = self.asr_latency {
            println!(
                "│  ⏱️  Stage 1 (ASR):  {:>10.2}ms                               │",
                asr.as_secs_f64() * 1000.0
            );
        } else {
            println!(
                "│  ⏱️  Stage 1 (ASR):  {:>10}   (skipped)                      │",
                "-"
            );
        }

        if let Some(llm) = self.llm_latency {
            println!(
                "│  ⏱️  Stage 2 (LLM):  {:>10.2}ms                               │",
                llm.as_secs_f64() * 1000.0
            );
        } else {
            println!(
                "│  ⏱️  Stage 2 (LLM):  {:>10}   (simulated)                    │",
                "-"
            );
        }

        if let Some(tts) = self.tts_latency {
            println!(
                "│  ⏱️  Stage 3 (TTS):  {:>10.2}ms                               │",
                tts.as_secs_f64() * 1000.0
            );
        } else {
            println!(
                "│  ⏱️  Stage 3 (TTS):  {:>10}   (skipped)                      │",
                "-"
            );
        }

        let total = self.total_start.elapsed();
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!(
            "│  ⏱️  Total:          {:>10.2}ms                               │",
            total.as_secs_f64() * 1000.0
        );
        println!("└─────────────────────────────────────────────────────────────────┘");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut latency = LatencyTracker::new();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           Xybrid Voice Assistant Demo                         ║");
    println!("║     Voice → ASR → LLM → TTS → Speech                          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let audio_path = args.get(1).cloned();
    let backend_arg = args.get(2).map(|s| s.as_str()).unwrap_or("gateway");

    // Determine LLM backend mode
    let use_gateway = backend_arg == "gateway";
    let direct_provider = if !use_gateway { backend_arg } else { "openai" };

    // =========================================================================
    // Stage 1: ASR - Voice to Text
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Stage 1: ASR (Automatic Speech Recognition)                     │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let transcription = if let Some(ref path) = audio_path {
        // Use provided audio file
        println!("🎤 Input audio: {}", path);

        let asr_model_dir = match model_fixtures::model_path("wav2vec2-base-960h") {
            Some(dir) => dir,
            None => {
                eprintln!("❌ Wav2Vec2 model not found");
                eprintln!("   Run: ./integration-tests/download.sh wav2vec2-base-960h");
                eprintln!("   Using simulated transcription instead.");
                PathBuf::new()
            }
        };
        if asr_model_dir.as_os_str().is_empty() {
            "Hello, how are you today?".to_string()
        } else {
            // Load ASR model and transcribe
            let asr_start = Instant::now();

            let metadata_path = asr_model_dir.join("model_metadata.json");
            let metadata_content = std::fs::read_to_string(&metadata_path)?;
            let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

            let audio_bytes = std::fs::read(path)?;
            let input_envelope = Envelope {
                kind: EnvelopeKind::Audio(audio_bytes),
                metadata: HashMap::new(),
            };

            let mut executor = TemplateExecutor::with_base_path(asr_model_dir.to_str().unwrap());
            let output = executor.execute(&metadata, &input_envelope)?;

            latency.asr_latency = Some(asr_start.elapsed());
            println!(
                "   ⏱️  ASR latency: {:.2}ms",
                latency.asr_latency.unwrap().as_secs_f64() * 1000.0
            );

            match output.kind {
                EnvelopeKind::Text(text) => text,
                _ => "Hello, how are you today?".to_string(),
            }
        }
    } else {
        // No audio provided - use default text
        println!("💡 No audio file provided. Using sample text input.");
        println!("   Usage: cargo run --example voice_assistant_demo <audio.wav> [gateway|openai|anthropic]");
        println!();
        "Hello, I'm interested in learning about Rust programming.".to_string()
    };

    println!();
    println!("📝 Transcription: \"{}\"", transcription);
    println!();

    // =========================================================================
    // Stage 2: LLM - Generate Response
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Stage 2: LLM (Language Model Response)                          │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Create LLM config based on backend mode
    let (config, display_mode) = if use_gateway {
        // Gateway mode - connect to local xybrid-gateway
        let gateway_url = std::env::var("XYBRID_GATEWAY_URL")
            .unwrap_or_else(|_| "http://localhost:3000/v1".to_string());
        let api_key = std::env::var("XYBRID_API_KEY").unwrap_or_else(|_| "test-key".to_string());

        println!("🌐 Mode: Gateway");
        println!("   URL: {}", gateway_url);

        let config = CloudConfig::gateway()
            .with_gateway_url(&gateway_url)
            .with_api_key(&api_key)
            .with_default_model("gpt-4o-mini");
        (config, "gateway".to_string())
    } else {
        // Direct API mode
        let env_var = match direct_provider {
            "anthropic" => "ANTHROPIC_API_KEY",
            _ => "OPENAI_API_KEY",
        };

        println!("🔗 Mode: Direct API");
        println!("   Provider: {}", direct_provider);

        if std::env::var(env_var).is_err() {
            eprintln!("⚠️  {} not set - will use simulated response", env_var);
        }

        let config = CloudConfig::direct(direct_provider);
        (config, format!("direct/{}", direct_provider))
    };

    let llm_response = match Cloud::with_config(config) {
        Ok(client) => {
            let request = CompletionRequest::new(&transcription)
                .with_system(
                    "You are a helpful voice assistant. \
                     Keep your responses brief and conversational, \
                     suitable for spoken output (1-2 sentences max).",
                )
                .with_max_tokens(100)
                .with_temperature(0.7);

            println!("🔄 Sending to LLM ({})...", display_mode);

            let llm_start = Instant::now();
            match client.complete(request) {
                Ok(response) => {
                    latency.llm_latency = Some(llm_start.elapsed());
                    println!("✅ Response received from {}", response.model);
                    println!(
                        "   ⏱️  LLM latency: {:.2}ms",
                        latency.llm_latency.unwrap().as_secs_f64() * 1000.0
                    );
                    if let Some(usage) = &response.usage {
                        println!(
                            "   Tokens: {} in, {} out",
                            usage.prompt_tokens, usage.completion_tokens
                        );
                    }
                    response.text
                }
                Err(e) => {
                    eprintln!("⚠️  LLM error: {}", e);
                    "I'm sorry, I couldn't process that. Could you please try again?".to_string()
                }
            }
        }
        Err(e) => {
            eprintln!("⚠️  Failed to create LLM client: {}", e);
            "Hello! I'm your voice assistant. How can I help you today?".to_string()
        }
    };

    println!();
    println!("💬 LLM Response: \"{}\"", llm_response);
    println!();

    // =========================================================================
    // Stage 3: TTS - Text to Speech
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Stage 3: TTS (Text-to-Speech Synthesis)                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let tts_model_dir = match model_fixtures::model_path("kitten-tts") {
        Some(dir) => dir,
        None => {
            eprintln!("❌ KittenTTS model not found");
            eprintln!("   Run: ./integration-tests/download.sh kitten-tts");
            eprintln!("   Skipping TTS synthesis.");
            println!();
            print_summary(false);
            latency.print_summary();
            return Ok(());
        }
    };
    let tts_metadata_path = tts_model_dir.join("model_metadata.json");

    let tts_start = Instant::now();

    // Load TTS model metadata (same pattern as ASR)
    let tts_metadata_content = std::fs::read_to_string(&tts_metadata_path)?;
    let tts_metadata: ModelMetadata = serde_json::from_str(&tts_metadata_content)?;

    println!(
        "🔊 TTS Model: {} v{}",
        tts_metadata.model_id, tts_metadata.version
    );
    println!("📝 Input text: \"{}\"", llm_response);
    println!();
    println!("🔄 Running TTS via Xybrid execution...");

    // Create text envelope for TTS (similar to ASR's audio envelope)
    let tts_input = Envelope {
        kind: EnvelopeKind::Text(llm_response.clone()),
        metadata: HashMap::new(),
    };

    // Execute TTS through TemplateExecutor (same pattern as ASR)
    let mut tts_executor = TemplateExecutor::with_base_path(tts_model_dir.to_str().unwrap());
    let tts_output = tts_executor.execute(&tts_metadata, &tts_input)?;

    latency.tts_latency = Some(tts_start.elapsed());

    // Extract audio from output envelope
    let audio_bytes = match tts_output.kind {
        EnvelopeKind::Audio(bytes) => bytes,
        _ => {
            eprintln!("❌ Unexpected TTS output type");
            return Err("TTS did not produce audio output".into());
        }
    };

    // Calculate duration (16-bit PCM at 24kHz)
    let sample_rate = 24000;
    let num_samples = audio_bytes.len() / 2;
    let duration_secs = num_samples as f32 / sample_rate as f32;

    println!(
        "✅ Generated {:.2}s of audio ({} bytes)",
        duration_secs,
        audio_bytes.len()
    );
    println!(
        "   ⏱️  TTS latency: {:.2}ms",
        latency.tts_latency.unwrap().as_secs_f64() * 1000.0
    );

    // Save output WAV
    let output_path = PathBuf::from("temp/voice_assistant_output.wav");
    save_wav_bytes(&output_path, &audio_bytes, sample_rate)?;
    println!("💾 Saved to: {}", output_path.display());

    println!();
    print_summary(true);

    println!();
    latency.print_summary();

    println!();
    println!("🎵 Play the output:");
    println!("   afplay voice_assistant_output.wav   # macOS");
    println!("   aplay voice_assistant_output.wav    # Linux");
    println!();

    Ok(())
}

fn print_summary(tts_complete: bool) {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    Pipeline Summary                           ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  ✅ Stage 1: ASR (Wav2Vec2) - Speech Recognition              ║");
    println!("║  ✅ Stage 2: LLM (Gateway/Direct) - Response Generation       ║");
    if tts_complete {
        println!("║  ✅ Stage 3: TTS (KittenTTS) - Speech Synthesis               ║");
    } else {
        println!("║  ⏭️  Stage 3: TTS (KittenTTS) - Skipped (model not found)     ║");
    }
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  This demonstrates Xybrid's hybrid cloud-edge architecture:   ║");
    println!("║  • On-device: ASR + TTS (private, low-latency)                ║");
    println!("║  • Cloud: LLM via Gateway (multi-provider routing)            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

/// Save raw PCM audio bytes as WAV file (16-bit PCM, mono)
fn save_wav_bytes(
    path: &PathBuf,
    audio_bytes: &[u8],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Ensure output directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

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

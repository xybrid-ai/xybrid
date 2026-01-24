//! LLM → TTS Pipeline Example
//!
//! Demonstrates a 2-stage pipeline:
//! 1. Local LLM generates text response
//! 2. Local TTS converts the response to speech
//!
//! Both stages run fully on-device, no network required after model download.
//!
//! Run with:
//!   cargo run --example llm_to_tts_pipeline -p xybrid-sdk --features llm-mistral

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::{DownloadProgress, PipelineRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  LLM → TTS Pipeline (Fully On-Device)");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Define a 2-stage pipeline: LLM → TTS
    let yaml = r#"
name: llm-to-speech

stages:
  # Stage 1: Local LLM generates text
  - id: llm
    model: qwen2.5-0.5b-instruct
    target: device

  # Stage 2: Local TTS synthesizes speech
  - id: tts
    model: kokoro-82m
    target: device
"#;

    println!("📋 Pipeline: LLM → TTS");
    println!("   1. qwen2.5-0.5b-instruct (text generation)");
    println!("   2. kokoro-82m (speech synthesis)");
    println!();

    // Load pipeline
    let pipeline_ref = PipelineRef::from_yaml(yaml)?;
    let pipeline = pipeline_ref.load()?;

    println!(
        "📦 Total download: {:.2} MB",
        pipeline.download_size() as f64 / 1024.0 / 1024.0
    );

    // Show stage info
    for stage in pipeline.stages() {
        println!(
            "   {} - {} ({:?})",
            stage.id,
            stage.model_id.as_deref().unwrap_or("n/a"),
            stage.status
        );
    }
    println!();

    // Preload models
    println!("🔄 Preloading models...");
    pipeline.load_models_with_progress(|progress: DownloadProgress| {
        let pct = progress.percent;
        let bar_len = 20;
        let filled = (pct as usize * bar_len) / 100;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_len - filled);
        print!("\r   {} [{}] {}%", progress.model_id, bar, pct);
        if pct == 100 {
            println!(" ✓");
        }
        use std::io::Write;
        std::io::stdout().flush().ok();
    })?;
    println!("✅ All models loaded");
    println!();

    // Create input
    let prompt = "Tell me a very short joke in one sentence.";
    println!("💬 Input: {}", prompt);
    println!();

    let mut metadata = HashMap::new();
    metadata.insert(
        "system_prompt".to_string(),
        "You are a comedian. Give short, funny jokes.".to_string(),
    );
    metadata.insert("max_tokens".to_string(), "50".to_string()); // Keep short for faster TTS

    let input = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata,
    };

    // Run pipeline
    println!("🔄 Running pipeline...");
    let start = std::time::Instant::now();
    let result = pipeline.run(&input)?;
    let elapsed = start.elapsed();

    println!(
        "✅ Pipeline complete! ({:.2}s total)",
        elapsed.as_secs_f32()
    );
    println!();

    // Display stage timings
    println!("═══════════════════════════════════════════════════════");
    println!("📊 Stage Results:");
    println!("═══════════════════════════════════════════════════════");

    for (i, stage) in result.stages.iter().enumerate() {
        println!(
            "   {}. {} → {} ({}ms)",
            i + 1,
            stage.name,
            stage.target,
            stage.latency_ms
        );
    }
    println!();
    println!("   Total: {}ms", result.total_latency_ms);
    println!();

    // Check output type
    println!("🔊 Final Output:");
    match result.output_type {
        xybrid_sdk::result::OutputType::Audio => {
            if let EnvelopeKind::Audio(audio_bytes) = &result.output.kind {
                println!("   Type: Audio ({} bytes)", audio_bytes.len());

                // Save to file
                let output_path = PathBuf::from("output_speech.wav");
                std::fs::write(&output_path, audio_bytes)?;
                println!("   Saved to: {}", output_path.display());
            }
        }
        xybrid_sdk::result::OutputType::Text => {
            if let Some(text) = result.text() {
                println!("   Type: Text");
                println!("   Content: {}", text);
            }
        }
        _ => {
            println!("   Type: {:?}", result.output_type);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  ✅ Pipeline demonstration complete!");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("Key takeaways:");
    println!("  • Both LLM and TTS ran fully on-device");
    println!("  • No network calls after model download");
    println!("  • Pipeline chaining is automatic (text → audio)");

    Ok(())
}

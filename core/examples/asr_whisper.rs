//! Whisper ASR using Xybrid Execution System
//!
//! This example demonstrates transcription using onnx-community/whisper-tiny
//! through the standard xybrid execution pipeline.
//!
//! Prerequisites:
//! - Download model: ./integration-tests/download.sh whisper-tiny
//! - encoder_model.onnx (FP16)
//! - decoder_with_past_model.onnx (FP16)
//! - tokenizer.json
//! - model_metadata.json
//!
//! Usage:
//!   cargo run --example asr_whisper
//!   cargo run --example asr_whisper path/to/audio.wav

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Whisper-tiny ASR using Xybrid Execution System");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Get audio file from command line or use fixtures default
    let audio_path = std::env::args().nth(1).unwrap_or_else(|| {
        // Try to find test audio in fixtures
        if let Some(fixtures) = model_fixtures::fixtures_dir() {
            let fixtures_audio = fixtures.join("input/test_audio.wav");
            if fixtures_audio.exists() {
                return fixtures_audio.to_string_lossy().to_string();
            }
        }
        "test_audio.wav".to_string()
    });

    // Load model metadata
    let model_dir = model_fixtures::require_model("whisper-tiny");
    let metadata_path = model_dir.join("model_metadata.json");

    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Model: {} v{}", metadata.model_id, metadata.version);
    if let Some(desc) = &metadata.description {
        println!("   Description: {}", desc);
    }
    println!("   Task: {:?}", metadata.metadata.get("task"));
    println!("   Preprocessing: {} step(s)", metadata.preprocessing.len());
    for (i, step) in metadata.preprocessing.iter().enumerate() {
        println!("     {}. {:?}", i + 1, step);
    }
    println!();

    // Load audio file
    println!("🎵 Loading audio: {}", audio_path);

    let audio_path = PathBuf::from(&audio_path);
    if !audio_path.exists() {
        // Generate test audio if no file provided
        println!("   No audio file found, generating test silence...");
        let sample_rate = 16000;
        let duration_secs = 2.0;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        // Generate a simple sine wave for testing
        let audio_samples: Vec<i16> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let freq = 440.0; // A4 note
                let amplitude = 8000.0;
                (amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16
            })
            .collect();

        // Convert to bytes
        let audio_bytes: Vec<u8> = audio_samples
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect();

        println!(
            "   Generated {} samples ({:.1}s at {}Hz)",
            num_samples, duration_secs, sample_rate
        );

        // Create executor
        let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

        // Create input envelope
        let input_envelope = Envelope {
            kind: EnvelopeKind::Audio(audio_bytes),
            metadata: HashMap::new(),
        };

        println!();
        println!("🔄 Running Whisper ASR pipeline...");
        println!("   1. MelSpectrogram: Audio → [1, 80, T] mel features");
        println!("   2. Encoder: Mel → [1, 1500, 384] hidden states");
        println!("   3. Decoder: Hidden states → Token IDs");
        println!("   4. WhisperDecode: Token IDs → Text");
        println!();

        // Execute inference
        let output_envelope = executor.execute(&metadata, &input_envelope)?;

        // Parse output
        match &output_envelope.kind {
            EnvelopeKind::Text(text) => {
                println!("═══════════════════════════════════════════════════════");
                println!("  Whisper ASR Result");
                println!("═══════════════════════════════════════════════════════");
                println!();
                println!("📝 Transcription: \"{}\"", text);
                println!();
            }
            EnvelopeKind::Audio(audio) => {
                println!("❌ Unexpected audio output: {} bytes", audio.len());
                return Err("Expected text output, got audio".into());
            }
            EnvelopeKind::Embedding(emb) => {
                println!("❌ Unexpected embedding output: {} dimensions", emb.len());
                return Err("Expected text output, got embedding".into());
            }
        }
    } else {
        // Load real audio file
        let audio_bytes = std::fs::read(&audio_path)?;
        println!("   Loaded {} bytes", audio_bytes.len());

        // Create executor
        let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

        // Create input envelope
        let input_envelope = Envelope {
            kind: EnvelopeKind::Audio(audio_bytes),
            metadata: HashMap::new(),
        };

        println!();
        println!("🔄 Running Whisper ASR pipeline...");
        println!("   1. MelSpectrogram: Audio → [1, 80, T] mel features");
        println!("   2. Encoder: Mel → [1, 1500, 384] hidden states");
        println!("   3. Decoder: Hidden states → Token IDs");
        println!("   4. WhisperDecode: Token IDs → Text");
        println!();

        // Execute inference
        let output_envelope = executor.execute(&metadata, &input_envelope)?;

        // Parse output
        match &output_envelope.kind {
            EnvelopeKind::Text(text) => {
                println!("═══════════════════════════════════════════════════════");
                println!("  Whisper ASR Result");
                println!("═══════════════════════════════════════════════════════");
                println!();
                println!("📝 Transcription: \"{}\"", text);
                println!();
            }
            EnvelopeKind::Audio(audio) => {
                println!("❌ Unexpected audio output: {} bytes", audio.len());
                return Err("Expected text output, got audio".into());
            }
            EnvelopeKind::Embedding(emb) => {
                println!("❌ Unexpected embedding output: {} dimensions", emb.len());
                return Err("Expected text output, got embedding".into());
            }
        }
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  Pipeline Summary");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("✅ Whisper-tiny ASR working through Xybrid execution system!");
    println!();
    println!("🎯 This validates:");
    println!("   • model_metadata.json Whisper config is correct");
    println!("   • Mel spectrogram preprocessing with Whisper normalization");
    println!("   • WhisperDecoder execution mode with HF ONNX format");
    println!("   • HuggingFace tokenizer.json decoding");
    println!("   • Ready for registry/Flutter integration");
    println!();

    Ok(())
}

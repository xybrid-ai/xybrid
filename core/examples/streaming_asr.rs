//! Streaming ASR example using StreamSession.
//!
//! This example demonstrates real-time speech transcription using the
//! StreamSession API with chunked audio processing.
//!
//! The backend (ONNX/Candle) is automatically detected from `model_metadata.json`.
//!
//! # Usage
//!
//! ```bash
//! # Run with Whisper (Candle backend, auto-detected)
//! cargo run --example streaming_asr --features candle-metal -- \
//!     --model-dir whisper-tiny-candle \
//!     --audio-file test.wav
//!
//! # Run with Wav2Vec2 (ONNX backend, auto-detected)
//! cargo run --example streaming_asr -- \
//!     --model-dir wav2vec2-base-960h \
//!     --audio-file test.wav
//!
//! # Simulate streaming with 500ms chunks
//! cargo run --example streaming_asr --features candle-metal -- \
//!     --model-dir whisper-tiny-candle \
//!     --chunk-ms 500
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};
use xybrid_core::streaming::{StreamConfig, StreamSession};
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    // Default model - will be resolved via model_fixtures
    let mut model_name = "whisper-tiny-candle".to_string();
    let mut audio_file: Option<PathBuf> = None;
    let mut chunk_ms: u64 = 0; // 0 = process all at once (non-streaming demo)
    let mut enable_vad = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_name = args[i + 1].clone();
                i += 2;
            }
            "--audio-file" => {
                audio_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--chunk-ms" => {
                chunk_ms = args[i + 1].parse()?;
                i += 2;
            }
            "--vad" => {
                enable_vad = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: streaming_asr [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --model-dir NAME    Model name (e.g. whisper-tiny-candle, wav2vec2-base-960h)");
                println!("  --audio-file PATH   Audio file to transcribe");
                println!("  --chunk-ms MS       Chunk size in ms for streaming simulation (0 = all at once)");
                println!("  --vad               Enable VAD (Voice Activity Detection) for smart chunking");
                println!("  --help              Show this help");
                println!();
                println!("Backend is auto-detected from model_metadata.json:");
                println!("  - CandleModel type → Whisper (Candle)");
                println!("  - SimpleMode type  → Wav2Vec2 (ONNX)");
                return Ok(());
            }
            _ => i += 1,
        }
    }

    // Resolve model directory via model_fixtures
    let model_dir = model_fixtures::require_model(&model_name);

    // Use provided audio file or default to jfk.wav in model dir
    let audio_file = audio_file.unwrap_or_else(|| model_dir.join("jfk.wav"));

    println!("=== Streaming ASR Example ===");
    println!("Model: {:?}", model_dir);
    println!("Audio: {:?}", audio_file);
    if chunk_ms > 0 {
        println!("Chunk size: {}ms (simulated streaming)", chunk_ms);
    } else {
        println!("Mode: Single-shot (feed all audio at once)");
    }
    if enable_vad {
        println!("VAD: Enabled (smart chunking at speech boundaries)");
    }
    println!();

    // Load audio file
    let audio_bytes = std::fs::read(&audio_file)?;
    let samples = decode_wav(&audio_bytes)?;
    println!(
        "Loaded {} samples ({:.2}s @ 16kHz)",
        samples.len(),
        samples.len() as f64 / 16000.0
    );

    // Create stream session - backend auto-detected from model_metadata.json
    let config = if enable_vad {
        StreamConfig::with_vad()
    } else {
        StreamConfig::default()
    };
    let mut session = StreamSession::new(&model_dir, config)?;

    println!("Model ID: {}", session.model_id());
    if session.has_vad() {
        println!("VAD: Active");
    }

    // Set partial result callback
    session.on_partial(|result| {
        if result.is_stable {
            println!("[STABLE] {}", result.text);
        } else {
            println!("[PARTIAL chunk {}] {}", result.chunk_sequence, result.text);
        }
    });

    let start = Instant::now();

    if chunk_ms > 0 {
        // Simulate streaming by feeding chunks
        let chunk_samples = (16000 * chunk_ms / 1000) as usize;
        let mut offset = 0;

        println!("\n--- Streaming {} sample chunks ---", chunk_samples);

        while offset < samples.len() {
            let end = (offset + chunk_samples).min(samples.len());
            let chunk = &samples[offset..end];

            // Simulate real-time delay
            std::thread::sleep(Duration::from_millis(chunk_ms / 2)); // Half speed for demo

            session.feed(chunk)?;
            offset = end;

            // Print stats periodically
            let stats = session.stats();
            print!(
                "\r[{:.1}s] chunks: {}, buffer: {} samples   ",
                stats.audio_duration.as_secs_f32(),
                stats.chunks_processed,
                stats.samples_received - stats.samples_processed
            );
            use std::io::Write;
            std::io::stdout().flush()?;
        }
        println!();
    } else {
        // Feed all at once (simpler demo)
        println!("\n--- Feeding all audio at once ---");
        session.feed(&samples)?;
    }

    // Flush and get final result
    println!("\n--- Flushing ---");
    let final_text = session.flush()?;

    let elapsed = start.elapsed();

    println!("\n=== Results ===");
    println!("Final transcript: {}", final_text);
    println!("Processing time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Real-time factor: {:.2}x",
        (samples.len() as f64 / 16000.0) / elapsed.as_secs_f64()
    );

    // Print final stats
    let stats = session.stats();
    println!("\n--- Session Stats ---");
    println!("State: {:?}", stats.state);
    println!("Samples received: {}", stats.samples_received);
    println!("Samples processed: {}", stats.samples_processed);
    println!("Chunks processed: {}", stats.chunks_processed);
    println!("Audio duration: {:.2}s", stats.audio_duration.as_secs_f64());

    Ok(())
}

/// Decode WAV file to f32 samples (16kHz mono)
fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Simple WAV parser
    if bytes.len() < 44 {
        return Err("WAV file too short".into());
    }

    // Check RIFF header
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut num_channels = 0u16;
    let mut data_start = 0;
    let mut data_size = 0usize;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            num_channels = u16::from_le_bytes([bytes[pos + 10], bytes[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                bytes[pos + 12],
                bytes[pos + 13],
                bytes[pos + 14],
                bytes[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([bytes[pos + 22], bytes[pos + 23]]);
        } else if chunk_id == b"data" {
            data_start = pos + 8;
            data_size = chunk_size;
            break;
        }

        pos += 8 + chunk_size;
    }

    if data_start == 0 {
        return Err("No data chunk found".into());
    }

    println!(
        "WAV: {} Hz, {} ch, {} bit",
        sample_rate, num_channels, bits_per_sample
    );

    // Decode samples
    let mut samples = Vec::new();
    let bytes_per_sample = (bits_per_sample / 8) as usize;

    pos = data_start;
    while pos + bytes_per_sample <= data_start + data_size && pos + bytes_per_sample <= bytes.len()
    {
        let sample = if bits_per_sample == 16 {
            let s = i16::from_le_bytes([bytes[pos], bytes[pos + 1]]);
            s as f32 / 32768.0
        } else if bits_per_sample == 32 {
            let s = i32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
            s as f32 / 2147483648.0
        } else {
            return Err(format!("Unsupported bit depth: {}", bits_per_sample).into());
        };

        samples.push(sample);
        pos += bytes_per_sample * num_channels as usize;
    }

    // Resample to 16kHz if needed
    if sample_rate != 16000 {
        println!("Resampling from {} Hz to 16000 Hz...", sample_rate);
        samples = resample(&samples, sample_rate, 16000);
    }

    Ok(samples)
}

/// Simple linear resampling
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let sample = if src_idx + 1 < input.len() {
            input[src_idx] * (1.0 - frac as f32) + input[src_idx + 1] * frac as f32
        } else if src_idx < input.len() {
            input[src_idx]
        } else {
            0.0
        };

        output.push(sample);
    }

    output
}

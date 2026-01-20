//! Voice Activity Detection demo.
//!
//! This example demonstrates using Silero VAD to detect speech segments
//! in an audio file.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example vad_demo -- --audio-file path/to/audio.wav
//! ```

use std::path::PathBuf;
use std::time::Instant;
use xybrid_core::audio::vad::{VadConfig, VadSession};
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let mut audio_file: Option<PathBuf> = None;
    let mut model_name = "silero-vad".to_string();
    let mut threshold = 0.5f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--audio-file" => {
                audio_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--model-dir" => {
                model_name = args[i + 1].clone();
                i += 2;
            }
            "--threshold" => {
                threshold = args[i + 1].parse()?;
                i += 2;
            }
            "--help" | "-h" => {
                println!("Usage: vad_demo [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --audio-file PATH   Audio file to analyze (required)");
                println!("  --model-dir NAME    Silero VAD model name (default: silero-vad)");
                println!("  --threshold FLOAT   VAD threshold 0.0-1.0 (default: 0.5)");
                println!("  --help              Show this help");
                return Ok(());
            }
            _ => i += 1,
        }
    }

    // Resolve model directory
    let model_dir = model_fixtures::require_model(&model_name);

    // Require audio file or use default from whisper-tiny-candle
    let audio_file = audio_file.unwrap_or_else(|| {
        model_fixtures::model_path("whisper-tiny-candle")
            .map(|p| p.join("jfk.wav"))
            .unwrap_or_else(|| PathBuf::from("test.wav"))
    });

    println!("=== Voice Activity Detection Demo ===");
    println!("Audio: {:?}", audio_file);
    println!("Model: {:?}", model_dir);
    println!("Threshold: {}", threshold);
    println!();

    // Load audio file
    let audio_bytes = std::fs::read(&audio_file)?;
    let samples = decode_wav(&audio_bytes)?;
    let audio_duration_s = samples.len() as f64 / 16000.0;
    println!(
        "Loaded {} samples ({:.2}s @ 16kHz)",
        samples.len(),
        audio_duration_s
    );

    // Create VAD session
    let config = VadConfig {
        threshold,
        ..VadConfig::default()
    };
    let mut vad = VadSession::new(&model_dir, config)?;
    println!("VAD model loaded");
    println!();

    // Process audio and detect speech segments
    let start = Instant::now();

    println!("--- Frame-by-frame analysis ---");
    let frame_size = vad.config().sample_rate.frame_size();
    let frame_ms = vad.config().sample_rate.frame_duration_ms();
    let num_frames = samples.len() / frame_size;

    let mut speech_frames = 0;
    let mut total_prob = 0.0f32;

    for i in 0..num_frames {
        let frame_start = i * frame_size;
        let frame_end = frame_start + frame_size;
        let frame = &samples[frame_start..frame_end];

        let result = vad.process_frame(frame)?;

        // Print periodic updates
        if i % 10 == 0 || result.is_speech {
            let time_ms = i as f32 * frame_ms;
            let speech_indicator = if result.is_speech {
                "🗣️ SPEECH"
            } else {
                "   silence"
            };
            println!(
                "[{:6.0}ms] prob: {:.3} | {}",
                time_ms, result.probability, speech_indicator
            );
        }

        if result.is_speech {
            speech_frames += 1;
        }
        total_prob += result.probability;
    }

    // Flush any remaining segment
    if let Some(segment) = vad.flush() {
        println!(
            "\n[SEGMENT] {:.0}ms - {:.0}ms (avg prob: {:.3})",
            segment.start_ms, segment.end_ms, segment.avg_probability
        );
    }

    let elapsed = start.elapsed();

    println!();
    println!("=== Results ===");
    println!("Processing time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("Frames processed: {}", num_frames);
    println!(
        "Speech frames: {} ({:.1}%)",
        speech_frames,
        speech_frames as f32 / num_frames as f32 * 100.0
    );
    println!("Average probability: {:.3}", total_prob / num_frames as f32);
    println!(
        "Real-time factor: {:.1}x",
        audio_duration_s / elapsed.as_secs_f64()
    );

    // Process entire audio to get segments
    println!();
    println!("--- Segment detection ---");
    vad.reset();
    let segments = vad.process_audio(&samples)?;

    // Add final segment if still in speech
    let mut all_segments = segments;
    if let Some(final_seg) = vad.flush() {
        all_segments.push(final_seg);
    }

    if all_segments.is_empty() {
        println!("No speech segments detected");
    } else {
        println!("Detected {} speech segment(s):", all_segments.len());
        for (i, seg) in all_segments.iter().enumerate() {
            let duration_ms = seg.end_ms - seg.start_ms;
            println!(
                "  {}. {:.0}ms - {:.0}ms ({:.0}ms, avg prob: {:.3})",
                i + 1,
                seg.start_ms,
                seg.end_ms,
                duration_ms,
                seg.avg_probability
            );
        }
    }

    Ok(())
}

/// Decode WAV file to f32 samples (16kHz mono)
fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if bytes.len() < 44 {
        return Err("WAV file too short".into());
    }

    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    // Find fmt and data chunks
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
            let s =
                i32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
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

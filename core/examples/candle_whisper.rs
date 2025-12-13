//! Candle Whisper ASR Example
//!
//! This example demonstrates how to use the Candle backend for Whisper ASR
//! with actual audio transcription.
//!
//! # Prerequisites
//!
//! The model files should be in `test_models/whisper-tiny-candle/`:
//! - model.safetensors (model weights)
//! - config.json (model configuration)
//! - tokenizer.json (tokenizer)
//! - melfilters.bytes (mel filter bank)
//!
//! # Running
//!
//! ```bash
//! cargo run --example candle_whisper --features candle
//! ```

#[cfg(feature = "candle")]
use xybrid_core::runtime_adapter::candle::{
    DeviceSelection, WhisperConfig, WhisperModel, WhisperSize,
};

#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use xybrid_core::runtime_adapter::candle::select_device;

    println!("=== Candle Whisper ASR Example ===\n");

    let model_path = std::path::Path::new("test_models/whisper-tiny-candle");

    // 1. Check model directory exists
    if !model_path.exists() {
        println!("Model directory not found: {:?}", model_path);
        println!("\nPlease ensure the following files exist:");
        println!("  - test_models/whisper-tiny-candle/model.safetensors");
        println!("  - test_models/whisper-tiny-candle/config.json");
        println!("  - test_models/whisper-tiny-candle/tokenizer.json");
        println!("  - test_models/whisper-tiny-candle/melfilters.bytes");
        return Ok(());
    }

    // 2. Select device
    println!("1. Selecting device...");
    let device = select_device(DeviceSelection::Auto)?;
    println!("   Using device: {:?}", device);

    // 3. Load model
    println!("\n2. Loading Whisper model...");
    let config = WhisperConfig {
        model_size: WhisperSize::Tiny,
        language: Some("en".to_string()),
        ..Default::default()
    };

    let mut model = WhisperModel::load_with_config(model_path, &device, config)?;
    println!("   Model loaded successfully!");
    println!("   Config: {:?}", model.config());

    // 4. Load test audio file
    let audio_path = model_path.join("jfk.wav");
    if !audio_path.exists() {
        println!("\n3. No test audio file found at {:?}", audio_path);
        println!("   Skipping transcription test.");
        println!("\n=== Example complete (model load only) ===");
        return Ok(());
    }

    println!("\n3. Loading audio file: {:?}", audio_path);

    // Load and decode audio using hound
    let reader = hound::WavReader::open(&audio_path)?;
    let spec = reader.spec();
    println!("   Sample rate: {} Hz", spec.sample_rate);
    println!("   Channels: {}", spec.channels);
    println!("   Bits per sample: {}", spec.bits_per_sample);

    // Convert to f32 samples
    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect()
    } else {
        reader
            .into_samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 2147483648.0)
            .collect()
    };

    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).unwrap_or(&0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    println!("   Audio duration: {:.2}s", mono_samples.len() as f32 / spec.sample_rate as f32);

    // Resample to 16kHz if needed
    let pcm_16k = if spec.sample_rate != 16000 {
        println!("   Resampling from {} Hz to 16000 Hz...", spec.sample_rate);
        resample(&mono_samples, spec.sample_rate, 16000)
    } else {
        mono_samples
    };

    println!("   PCM samples: {}", pcm_16k.len());

    // 5. Run transcription
    println!("\n4. Running transcription...");
    let start = std::time::Instant::now();
    let text = model.transcribe_pcm(&pcm_16k)?;
    let elapsed = start.elapsed();

    println!("\n=== Transcription Result ===");
    println!("{}", text);
    println!("============================");
    println!("\nTime: {:.2}s", elapsed.as_secs_f32());

    println!("\n=== Example complete ===");
    Ok(())
}

#[cfg(feature = "candle")]
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = src_idx - idx0 as f64;

        let sample = samples[idx0] * (1.0 - frac as f32) + samples[idx1] * frac as f32;
        resampled.push(sample);
    }

    resampled
}

#[cfg(not(feature = "candle"))]
fn main() {
    println!("This example requires the 'candle' feature.");
    println!("Run with: cargo run --example candle_whisper --features candle");
}

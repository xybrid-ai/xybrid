//! KittenTTS Text-to-Speech Example
//!
//! This example demonstrates:
//! - Text-to-IPA phonemization using CMU dictionary
//! - Converting phonemes to token IDs
//! - Loading voice embeddings from voices.bin
//! - Running ONNX inference for speech synthesis
//! - Saving output as WAV file
//!
//! Model: KittenTTS Nano (15M params, 24kHz output)
//!
//! Prerequisites:
//! - CMU dictionary at ~/.xybrid/cmudict.dict
//! - Download model: ./integration-tests/download.sh kitten-tts

use std::path::PathBuf;
use xybrid_core::phonemizer::{load_tokens_map, Phonemizer};
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  KittenTTS Text-to-Speech Demo");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Get input text from command line or use default
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello world".to_string());

    let voice_id: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    println!("📝 Input text: \"{}\"", text);
    println!("🎤 Voice ID: {} (0-7 available)", voice_id);
    println!();

    // Check model directory
    let model_dir = model_fixtures::require_model("kitten-tts");

    // Check CMU dictionary
    let dict_path = dirs::home_dir()
        .map(|h| h.join(".xybrid/cmudict.dict"))
        .filter(|p| p.exists())
        .ok_or("CMU dictionary not found at ~/.xybrid/cmudict.dict")?;

    println!("📚 Using CMU dictionary at: {}", dict_path.display());

    // Step 1: Phonemize text
    println!();
    println!("Step 1: Phonemization");
    println!("─────────────────────");

    let phonemizer = Phonemizer::new(&dict_path)?;
    let phonemes = phonemizer.phonemize(&text);
    println!("   Input:  \"{}\"", text);
    println!("   Output: \"{}\"", phonemes);

    // Step 2: Convert to token IDs
    println!();
    println!("Step 2: Token Conversion");
    println!("────────────────────────");

    let tokens_path = model_dir.join("tokens.txt");
    let tokens_content = std::fs::read_to_string(&tokens_path)?;
    let tokens_map = load_tokens_map(&tokens_content);

    let token_ids = phonemizer.text_to_token_ids(&text, &tokens_map, true);
    println!("   Token IDs: {:?}", token_ids);
    println!("   Count: {} tokens", token_ids.len());

    // Step 3: Load voice embedding
    println!();
    println!("Step 3: Voice Embedding");
    println!("───────────────────────");

    let voices_path = model_dir.join("voices.bin");
    let voices_data = std::fs::read(&voices_path)?;

    // Each voice is 256 float32 values (1024 bytes)
    let embedding_dim = 256;
    let offset = voice_id * embedding_dim * 4;
    let voice_embedding: Vec<f32> = voices_data[offset..offset + embedding_dim * 4]
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let voice_names = [
        "expr-voice-2-m", "expr-voice-2-f",
        "expr-voice-3-m", "expr-voice-3-f",
        "expr-voice-4-m", "expr-voice-4-f",
        "expr-voice-5-m", "expr-voice-5-f",
    ];
    println!("   Voice: {} ({})", voice_id, voice_names.get(voice_id).unwrap_or(&"unknown"));
    let mean: f32 = voice_embedding.iter().sum::<f32>() / embedding_dim as f32;
    println!("   Embedding mean: {:.4}", mean);

    // Step 4: Load ONNX model and run inference
    println!();
    println!("Step 4: ONNX Inference");
    println!("──────────────────────");

    let model_path = model_dir.join("model.fp16.onnx");
    println!("   Loading: {}", model_path.display());

    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ndarray::Array;

    // Initialize ORT
    ort::init().commit()?;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&model_path)?;

    println!("   Model loaded successfully!");
    println!();
    println!("   Model inputs:");
    for input in session.inputs.iter() {
        println!("     - {} ({:?})", input.name, input.input_type);
    }
    println!();
    println!("   Model outputs:");
    for output in session.outputs.iter() {
        println!("     - {} ({:?})", output.name, output.output_type);
    }

    // Prepare inputs
    let seq_len = token_ids.len();
    let input_ids: Array<i64, _> = Array::from_shape_vec((1, seq_len), token_ids)?;
    let style: Array<f32, _> = Array::from_shape_vec((1, 256), voice_embedding)?;
    let speed: Array<f32, _> = Array::from_shape_vec((1,), vec![1.0f32])?;

    println!();
    println!("   Running inference...");

    // Create input tensors (ort 2.0 API uses owned arrays)
    let input_ids_tensor = ort::value::Tensor::from_array(input_ids)?;
    let style_tensor = ort::value::Tensor::from_array(style)?;
    let speed_tensor = ort::value::Tensor::from_array(speed)?;

    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "style" => style_tensor,
        "speed" => speed_tensor
    ])?;

    // Get output waveform
    let waveform = outputs[0].try_extract_tensor::<f32>()?;
    let (_, audio_data) = waveform;
    let raw_samples: Vec<f32> = audio_data.iter().cloned().collect();

    println!("   ✅ Inference complete!");
    println!("   Raw samples: {}", raw_samples.len());
    println!("   Raw duration: {:.2}s at 24kHz", raw_samples.len() as f32 / 24000.0);

    // Check raw audio statistics
    let max_val = raw_samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = raw_samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let rms: f32 = (raw_samples.iter().map(|x| x * x).sum::<f32>() / raw_samples.len() as f32).sqrt();
    println!("   Raw range: [{:.4}, {:.4}]", min_val, max_val);
    println!("   Raw RMS: {:.4}", rms);

    // Step 5: Audio Postprocessing
    println!();
    println!("Step 5: Audio Postprocessing");
    println!("────────────────────────────");

    use xybrid_core::phonemizer::postprocess_tts_audio;
    let audio_samples = postprocess_tts_audio(&raw_samples, 24000);

    println!("   ✅ Applied: high-pass filter → silence trim → loudness normalization");
    println!("   Final samples: {} ({:.2}s)", audio_samples.len(), audio_samples.len() as f32 / 24000.0);

    // Check processed audio statistics
    let max_val = audio_samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = audio_samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let rms: f32 = (audio_samples.iter().map(|x| x * x).sum::<f32>() / audio_samples.len() as f32).sqrt();
    println!("   Final range: [{:.4}, {:.4}]", min_val, max_val);
    println!("   Final RMS: {:.4}", rms);

    // Step 6: Save as WAV file
    println!();
    println!("Step 6: Save Audio");
    println!("──────────────────");

    let output_path = PathBuf::from("output.wav");
    save_wav(&output_path, &audio_samples, 24000)?;
    println!("   ✅ Saved to: {}", output_path.display());

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  TTS Pipeline Complete!");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎵 Play the output:");
    println!("   afplay output.wav   # macOS");
    println!("   aplay output.wav    # Linux");
    println!();

    Ok(())
}

/// Save audio samples as WAV file (16-bit PCM, mono)
fn save_wav(path: &PathBuf, samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let num_samples = samples.len() as u32;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = num_samples * num_channels as u32 * bits_per_sample as u32 / 8;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;  // chunk size
    file.write_all(&1u16.to_le_bytes())?;   // audio format (PCM)
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // Convert float32 to int16
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * 32767.0) as i16;
        file.write_all(&int_sample.to_le_bytes())?;
    }

    Ok(())
}

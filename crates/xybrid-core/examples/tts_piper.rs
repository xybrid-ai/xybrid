use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(
        args.next()
            .ok_or("usage: tts_piper MODEL_DIR [VOICE_ID] [TEXT] [OUTPUT]")?,
    );
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(
        model_dir.join("model_metadata.json"),
    )?)?;
    let voice_id = args.next().unwrap_or_else(|| {
        metadata
            .voices
            .as_ref()
            .map(|voices| voices.default.clone())
            .unwrap_or_default()
    });
    let text = args
        .next()
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".to_string());
    let output_path = PathBuf::from(args.next().unwrap_or_else(|| "piper.wav".to_string()));

    let mut envelope_metadata = HashMap::new();
    envelope_metadata.insert("voice_id".to_string(), voice_id);
    let envelope = Envelope::with_metadata(EnvelopeKind::Text(text), envelope_metadata);
    let mut executor = TemplateExecutor::with_base_path(
        model_dir
            .to_str()
            .ok_or("model directory must be valid UTF-8")?,
    );
    let output = executor.execute(&metadata, &envelope, None)?;
    let pcm = output.audio_bytes().ok_or("TTS did not return audio")?;
    write_wav(&output_path, pcm, 22_050)?;
    println!("wrote {}", output_path.display());
    Ok(())
}

fn write_wav(path: &std::path::Path, pcm: &[u8], sample_rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let data_len = u32::try_from(pcm.len()).unwrap_or(u32::MAX);
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&36u32.saturating_add(data_len).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    file.write_all(pcm)
}

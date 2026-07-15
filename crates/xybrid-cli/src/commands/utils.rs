//! Shared utility functions for CLI commands.

use std::path::Path;
use xybrid_core::ir::{Envelope, EnvelopeKind};

use crate::ui;

/// A thinking model that spent its whole token budget inside `<think>` ends
/// with an empty answer, finish_reason "length", and captured reasoning —
/// the one situation where "success with no output" needs a nudge.
pub(crate) fn thinking_budget_exhausted(
    text: &str,
    finish_reason: Option<&str>,
    reasoning: Option<&str>,
) -> bool {
    text.trim().is_empty()
        && finish_reason == Some("length")
        && reasoning.is_some_and(|r| !r.trim().is_empty())
}

pub(crate) const THINKING_BUDGET_HINT: &str =
    "the model spent the entire token budget thinking — rerun with a larger --max-tokens, or --show-reasoning to see the partial reasoning";

pub(crate) fn maybe_warn_thinking_budget(output: &Envelope) {
    let EnvelopeKind::Text(text) = &output.kind else {
        return;
    };

    if thinking_budget_exhausted(
        text,
        output.metadata.get("finish_reason").map(String::as_str),
        output.metadata.get("reasoning_content").map(String::as_str),
    ) {
        ui::warning(THINKING_BUDGET_HINT);
    }
}

/// Format model parameter count (e.g., "82M", "1.5B").
pub fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.0}K", params as f64 / 1_000.0)
    } else {
        format!("{}", params)
    }
}

/// Format byte size (e.g., "256 MB", "1.2 GB").
pub fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.0} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Display a stage name, stripping any "@target" suffix.
pub fn display_stage_name(name: &str) -> &str {
    name.split('@').next().unwrap_or(name)
}

/// Save raw PCM audio bytes as a WAV file with proper headers.
///
/// # Arguments
/// * `path` - Output file path
/// * `audio_bytes` - Raw 16-bit PCM audio samples (little-endian)
/// * `sample_rate` - Sample rate in Hz (e.g., 24000 for Kokoro TTS)
/// * `num_channels` - Number of audio channels (1 for mono, 2 for stereo)
///
/// # Example
/// ```rust,ignore
/// save_wav_file(Path::new("output.wav"), &audio_data, 24000, 1)?;
/// ```
pub fn save_wav_file(
    path: &Path,
    audio_bytes: &[u8],
    sample_rate: u32,
    num_channels: u16,
) -> anyhow::Result<()> {
    use std::io::Write;

    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = audio_bytes.len() as u32;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Subchunk1Size (16 for PCM)
    file.write_all(&1u16.to_le_bytes())?; // AudioFormat (1 = PCM)
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

#[cfg(test)]
mod tests {
    use super::thinking_budget_exhausted;

    #[test]
    fn thinking_budget_exhausted_requires_all_three_signals() {
        assert!(thinking_budget_exhausted(
            "",
            Some("length"),
            Some("deliberation")
        ));
        assert!(!thinking_budget_exhausted(
            "answer",
            Some("length"),
            Some("deliberation")
        ));
        assert!(!thinking_budget_exhausted(
            "",
            Some("stop"),
            Some("deliberation")
        ));
        assert!(!thinking_budget_exhausted("", None, Some("deliberation")));
        assert!(!thinking_budget_exhausted("", Some("length"), None));
        assert!(!thinking_budget_exhausted("", Some("length"), Some("")));
        assert!(!thinking_budget_exhausted("", Some("length"), Some("   ")));
    }
}

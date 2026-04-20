//! Slice the dumped waveform into WAV files so you can hear where the weirdness lives.
//!
//! Pairs with the `XYBRID_CODEC_TTS_DUMP` output from `neutts_tts`. Reads
//! `waveform.f32` (raw f32 LE samples @ 24 kHz) and writes WAVs for the
//! requested range. Non-destructive — the original dump is untouched.
//!
//! Usage:
//!   cargo run --example neutts_decode_tokens -p xybrid-core --features llm-llamacpp -- \
//!     --dump-dir /tmp/neutts-debug --last-ms 300 --output /tmp/tail.wav
//!
//! Flags: --first-ms MS | --last-ms MS | --range-ms A:B | --all (default)

use std::path::PathBuf;
use xybrid_core::audio::samples_to_wav;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dump_dir = PathBuf::from("/tmp/neutts-debug");
    let mut output = PathBuf::from("/tmp/neutts-slice.wav");
    let mut mode = Mode::All;
    let sample_rate: u32 = 24000;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dump-dir" => {
                dump_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output" | "-o" => {
                output = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--last-ms" => {
                mode = Mode::LastMs(args[i + 1].parse()?);
                i += 2;
            }
            "--first-ms" => {
                mode = Mode::FirstMs(args[i + 1].parse()?);
                i += 2;
            }
            "--range-ms" => {
                let (a, b) = args[i + 1]
                    .split_once(':')
                    .ok_or("range must be A:B in ms")?;
                mode = Mode::RangeMs(a.parse()?, b.parse()?);
                i += 2;
            }
            "--all" => {
                mode = Mode::All;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown arg: {}", other).into()),
        }
    }

    let bytes = std::fs::read(dump_dir.join("waveform.f32"))?;
    if bytes.len() % 4 != 0 {
        return Err("waveform.f32 size is not a multiple of 4".into());
    }
    let all: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let total_ms = (all.len() as f32 / sample_rate as f32 * 1000.0) as usize;
    println!(
        "Full waveform: {} samples ({} ms @ {} Hz)",
        all.len(),
        total_ms,
        sample_rate
    );

    let ms_to_samples = |ms: usize| (sample_rate as usize * ms) / 1000;

    let slice: &[f32] = match mode {
        Mode::All => &all,
        Mode::LastMs(ms) => {
            let n = ms_to_samples(ms);
            &all[all.len().saturating_sub(n)..]
        }
        Mode::FirstMs(ms) => {
            let n = ms_to_samples(ms);
            &all[..n.min(all.len())]
        }
        Mode::RangeMs(a, b) => {
            let start = ms_to_samples(a).min(all.len());
            let end = ms_to_samples(b).min(all.len());
            &all[start..end.max(start)]
        }
    };

    let wav = samples_to_wav(slice, sample_rate);
    std::fs::write(&output, wav)?;
    let duration = slice.len() as f32 / sample_rate as f32;
    println!(
        "Wrote {} ({:.2}s, {} samples)",
        output.display(),
        duration,
        slice.len()
    );
    println!("Play: afplay {}", output.display());
    Ok(())
}

enum Mode {
    All,
    FirstMs(usize),
    LastMs(usize),
    RangeMs(usize, usize),
}

fn print_help() {
    println!("Slice a neutts_tts waveform dump into WAVs.");
    println!();
    println!("Options:");
    println!("  --dump-dir <DIR>    Dump dir (default: /tmp/neutts-debug)");
    println!("  --output, -o PATH   Output WAV (default: /tmp/neutts-slice.wav)");
    println!("  --first-ms <MS>     Slice the first N ms");
    println!("  --last-ms  <MS>     Slice the last N ms");
    println!("  --range-ms A:B      Slice [A..B) in ms");
    println!("  --all               Re-encode the whole dump (default)");
}

//! Shared native audio delivery, cancellation and terminal-state translation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use xybrid_zzz_sys::{
    ChunkInfo, KittenCancellation, StreamOutcome, SynthesisStatus, ZzzError, SYNTHESIS_CHANNELS,
    SYNTHESIS_SAMPLE_RATE,
};

use crate::audio::f32_to_pcm16;
use crate::execution::{TtsAudioChunk, TtsStatus, TtsStreamResult};
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult};

pub(super) fn stream_session(
    input: &Envelope,
    cancelled: &(dyn Fn() -> bool + Sync),
    on_chunk: &mut dyn FnMut(TtsAudioChunk) -> bool,
    cancel: KittenCancellation,
    synthesize: impl FnOnce(
        &str,
        &mut dyn FnMut(&[f32], ChunkInfo) -> bool,
    ) -> Result<StreamOutcome, ZzzError>,
) -> AdapterResult<TtsStreamResult> {
    let EnvelopeKind::Text(text) = &input.kind else {
        return Err(AdapterError::InvalidInput(
            "zzz TTS requires text input".into(),
        ));
    };
    let mut summary = TtsStreamResult {
        status: TtsStatus::Cancelled,
        sample_rate: SYNTHESIS_SAMPLE_RATE,
        channels: SYNTHESIS_CHANNELS,
        samples: 0,
        chunks: 0,
        limited_chunks: 0,
    };
    if cancelled() {
        return Ok(summary);
    }
    let finished = AtomicBool::new(false);
    let mut delivered_end = 0;
    // The watcher owns no session/model lock. Repeated cancellation covers the
    // native idle/start race; scope joins it before the native owner can close.
    let outcome = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Acquire) {
                if cancelled() {
                    cancel.cancel();
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        struct FinishOnDrop<'a>(&'a AtomicBool);
        impl Drop for FinishOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let finish_guard = FinishOnDrop(&finished);
        let result = synthesize(text, &mut |samples, info| {
            if cancelled() {
                cancel.cancel();
                return false;
            }
            summary.sample_rate = info.sample_rate;
            summary.channels = info.channels;
            delivered_end =
                info.first_sample + samples.len() as u64 / u64::from(info.channels.max(1));
            // Copy before the borrowed native buffer expires; packets receive
            // no fades, normalization or text splitting inside Xybrid.
            let keep_going = on_chunk(TtsAudioChunk {
                pcm: f32_to_pcm16(samples),
                sample_rate: info.sample_rate,
                channels: info.channels,
                first_sample: info.first_sample,
            });
            if !keep_going || cancelled() {
                cancel.cancel();
                return false;
            }
            true
        });
        drop(finish_guard);
        result
    })
    .map_err(|error| AdapterError::RuntimeError(error.to_string()))?;
    // Native completion/limits outrank a cancellation arriving after return.
    summary.status = match outcome.status {
        SynthesisStatus::Completed => TtsStatus::Completed,
        SynthesisStatus::Cancelled => TtsStatus::Cancelled,
        SynthesisStatus::Limited => TtsStatus::Limited,
    };
    if outcome.result.sample_rate != 0 {
        summary.sample_rate = outcome.result.sample_rate;
    }
    if outcome.result.channels != 0 {
        summary.channels = outcome.result.channels;
    }
    summary.samples = outcome.result.samples.max(delivered_end);
    summary.chunks = outcome.result.chunks;
    summary.limited_chunks = outcome.result.limited_chunks;
    Ok(summary)
}

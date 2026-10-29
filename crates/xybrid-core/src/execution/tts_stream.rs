//! Owned TTS packets and terminal delivery counters.
//!
//! Packets contain raw PCM16 LE with per-channel offsets. Backends copy native
//! memory before returning from callbacks; consumers may retain these buffers.

use crate::ir::{Envelope, EnvelopeKind};

/// One owned packet, ready for a consumer's playback queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtsAudioChunk {
    pub pcm: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u32,
    pub first_sample: u64,
}

/// How synthesis ended. A limit delivers partial audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsStatus {
    Completed,
    Cancelled,
    Limited,
}

/// Counters for a stream. Audio is delivered separately through packets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtsStreamResult {
    pub status: TtsStatus,
    pub sample_rate: u32,
    pub channels: u32,
    pub samples: u64,
    pub chunks: u32,
    pub limited_chunks: u32,
}

impl TtsStreamResult {
    /// Build the terminal stream envelope, without duplicating delivered audio.
    pub fn into_envelope(self) -> Envelope {
        let mut envelope = Envelope::new(EnvelopeKind::Audio(Vec::new()));
        let status = match self.status {
            TtsStatus::Completed => "completed",
            TtsStatus::Cancelled => "cancelled",
            TtsStatus::Limited => "limited",
        };
        envelope.metadata.insert("tts_status".into(), status.into());
        for (key, value) in [
            ("sample_rate", self.sample_rate.to_string()),
            ("channels", self.channels.to_string()),
            ("samples", self.samples.to_string()),
            ("chunks", self.chunks.to_string()),
            ("zzz_limited_chunks", self.limited_chunks.to_string()),
        ] {
            envelope.metadata.insert(key.into(), value);
        }
        envelope
    }
}

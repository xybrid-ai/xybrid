//! Raw FFI bindings to the pinned zzz CPU engine (Kitten TTS 2 slice).
//!
//! `libzzz_embed.a` is a prebuilt, privately staged static archive — see
//! `build.rs`, `natives-manifest.json` and the crate README. This crate owns
//! the FFI boundary and a thin safe surface: the safe types here are what
//! downstream code (`xybrid-core`'s adapter, anything else that wants the
//! engine without xybrid's envelopes) may touch.
//!
//! The declarations below mirror `include/zzz_embed.h` (ABI version 1)
//! by hand rather than through bindgen: the surface is six functions and
//! three structs, the header notes it is experimental, and a generated
//! snapshot would add a bindgen/libclang dependency (and a Bazel story) for
//! zero value at this size. Keep the two in lockstep when the engine's ABI
//! moves; the compile-time struct-size/link-layout assertions below catch
//! accidental drift.
//!
//! # Activation
//!
//! The FFI entry points and the safe session surface live behind the
//! `bindings` cargo feature. A default build compiles public types only, so
//! `cargo check --workspace` never needs a staged slice. An enabled build
//! links the pinned slice for the target or fails (see `build.rs`).
//!
//! # Public surface
//!
//! - [`abi_version`] / [`supports_model`] — capability probes
//! - [`EmbedOptions`] / [`KittenOptions`] / [`EmbedError`] / [`EmbedResult`]
//!   — C-layout types with `Default`s mirroring the header's rules
//! - [`KittenSession`] — RAII handle over `zzz_embed_open_kitten`, with
//!   `synthesize` (audio collected from the C callback, borrowed only for
//!   the duration of the call, per the header's ownership contract)
//! - [`SynthesisOutcome`] — delivery payload with the capped/partial flags
//!
//! Zero `unsafe` on the public surface: every `unsafe` block sits in
//! [`mod@session`] with a `# Safety` comment, mirroring `xybrid-llama`'s and
//! `xybrid-whisper`'s discipline. `KittenSession` is deliberately not
//! `Sync`: the header documents one synthesis at a time per session, with no
//! re-entrant or concurrent calls.

use std::os::raw::c_void;

/// ABI version pinned in `zzz_embed.h`; the manifest's `apis.zzz_embed`
/// agrees with it. An engine slice declaring anything else is a different
/// ABI and must fail staging, not adapt at runtime.
///
/// Note there is no `zzz_embed` const in the C header to take the address
/// of; the version comes from `zzz_embed_abi_version` at runtime.
pub const ZZZ_EMBED_ABI_VERSION: u32 = 1;

pub const ZZZ_EMBED_KITTEN_ACCELERATE: u32 = 1;
pub const ZZZ_EMBED_KITTEN_SENTENCE_CHUNKS: u32 = 2;

pub const ZZZ_EMBED_OK: i32 = 0;
pub const ZZZ_EMBED_INVALID_ARGUMENT: i32 = -1;
pub const ZZZ_EMBED_MODEL_NOT_COMPILED: i32 = -2;
pub const ZZZ_EMBED_LOAD_FAILED: i32 = -3;
pub const ZZZ_EMBED_OUT_OF_MEMORY: i32 = -4;
pub const ZZZ_EMBED_CANCELLED: i32 = -5;
pub const ZZZ_EMBED_LIMIT_REACHED: i32 = -6;
pub const ZZZ_EMBED_SYNTHESIS_FAILED: i32 = -7;

/// Caller-owned error storage the engine resets on every open/synthesize.
/// Mirrors C `zzz_embed_error`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EmbedError {
    pub code: i32,
    pub message: [u8; 192],
}

impl EmbedError {
    /// Zeroed storage, as the C callers zero-initialize before use.
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            code: 0,
            message: [0; 192],
        }
    }

    /// The engine's message as a borrowed `&str`, empty on non-UTF-8 bytes.
    /// Empty unless a call failed (the header: reset on every call).
    #[must_use]
    pub fn message(&self) -> &str {
        let end = self
            .message
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(self.message.len());
        std::str::from_utf8(&self.message[..end]).unwrap_or("")
    }

    /// True when the last call returned the OK code.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.code == ZZZ_EMBED_OK
    }
}

impl Default for EmbedError {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl core::fmt::Debug for EmbedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EmbedError")
            .field("code", &self.code)
            .field("message", &self.message())
            .finish()
    }
}

/// Session options. Initialize from [`EmbedOptions::default`], then override
/// fields. Zero is not a default sentinel: seed 0 is valid, while zero
/// capacities or non-positive temperature are not.
///
/// Mirrors C `zzz_embed_options` field-for-field. `struct_size` must equal
/// `sizeof` for this PoC ABI; both sides stay in lockstep because the engine
/// and these declarations come from the same pinned release.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct EmbedOptions {
    pub struct_size: u32,
    pub max_frames_per_chunk: u32,
    pub max_chunk_tokens: u32,
    pub temperature: f32,
    pub seed: u64,
}

/// Audio payload for one completed synthesis call. Mirrors C
/// `zzz_embed_result`; `limited_chunks` counts chunks that hit the
/// `max_tokens` cap — the header's LIMIT_REACHED-with-partial-audio case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct EmbedResult {
    pub sample_rate: u32,
    pub channels: u32,
    pub chunks: u32,
    pub limited_chunks: u32,
    pub samples: u64,
}

/// Callback signature mirroring C `zzz_embed_audio_callback`.
///
/// Samples are borrowed only until callback return (copy to retain). Called
/// synchronously on the synthesis thread; returning nonzero cancels the
/// remaining delivery. `first_sample` counts per-channel samples from zero
/// for the utterance.
pub type AudioCallback = unsafe extern "C" fn(
    samples: *const f32,
    sample_count: usize,
    sample_rate: u32,
    channels: u32,
    first_sample: u64,
    user_data: *mut c_void,
) -> i32;

/// Kitten-only open options mirroring C `zzz_embed_kitten_options`.
///
/// All four asset paths are required and must describe the same speaker
/// pair (language-model voice JSON + S3 decoder voice JSON). Zero-
/// initialize, set `struct_size`, then all paths — the safe wrapper does
/// both. `threads: 0` selects the engine default of 4; explicit limits are
/// 1..=64 threads and 1..=1023 `max_tokens` (0 uses available LM-window room, with the split-chunk runaway guard).
/// The seed (including 0) seeds the waveform stage; LM generation is greedy.
/// Paths need only stay valid through `open`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct KittenOptions {
    pub struct_size: u32,
    pub threads: u32,
    pub max_tokens: u32,
    pub flags: u32,
    pub seed: u64,
    pub language_model_path: *const u8,
    pub decoder_model_path: *const u8,
    pub language_voice_path: *const u8,
    pub decoder_voice_path: *const u8,
}

impl KittenOptions {
    /// Zeroed storage with the correct `struct_size`, ready for path fields.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            threads: 0,
            max_tokens: 0,
            flags: 0,
            seed: 0,
            language_model_path: core::ptr::null(),
            decoder_model_path: core::ptr::null(),
            language_voice_path: core::ptr::null(),
            decoder_voice_path: core::ptr::null(),
        }
    }
}

// Compile-time ABI checks: the C structs are extern-managed memory engines
// write into. If these constants stop matching the header, fail the build
// here rather than at the engine's runtime struct_size checks.
const _: () = {
    // i32 code (4) + u8[192] message; the repr(C) alignment keeps it 196,
    // never padded to a wider multiple.
    assert!(core::mem::size_of::<EmbedError>() == 196);
    // 4*u32/f32 slots + one u64: u32 struct_size, u32 max_frames_per_chunk,
    // u32 max_chunk_tokens, f32 temperature, then an aligned u64 seed.
    assert!(core::mem::size_of::<EmbedOptions>() == 24);
    // Four u32 slots + an aligned u64 samples.
    assert!(core::mem::size_of::<EmbedResult>() == 24);
    // u32*4 + u64 + four pointers on a 64-bit host: 24 + 32.
    #[cfg(target_pointer_width = "64")]
    {
        assert!(core::mem::size_of::<KittenOptions>() == 56);
        assert!(core::mem::offset_of!(KittenOptions, flags) == 12);
        assert!(core::mem::offset_of!(KittenOptions, seed) == 16);
        assert!(core::mem::offset_of!(KittenOptions, language_model_path) == 24);
        assert!(core::mem::offset_of!(KittenOptions, decoder_voice_path) == 48);
    }
};

#[cfg(feature = "bindings")]
mod session;
#[cfg(feature = "bindings")]
pub use session::{
    abi_version, default_options, supports_model, ChunkInfo, KittenCancellation, KittenFlags,
    KittenSession, KittenSettings, StreamOutcome, SynthesisOutcome, SynthesisStatus, ZzzError,
    SYNTHESIS_CHANNELS, SYNTHESIS_SAMPLE_RATE,
};

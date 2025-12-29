//! Execution mode implementations.
//!
//! This module contains the different execution strategies for model inference:
//!
//! | Module | Mode |
//! |--------|------|
//! | [`single_shot`] | Single forward pass execution |
//! | [`autoregressive`] | Token-by-token generation with KV cache |
//! | [`whisper`] | Whisper-specific decoder with encoder KV cache |
//! | [`bert`] | BERT-style models with integer token inputs |
//! | [`tts`] | TTS models with phoneme IDs and voice embeddings |
//!
//! ## Execution Flow
//!
//! Pipeline stages are executed according to their `ExecutionMode`:
//!
//! ```text
//! Pipeline Stage → ExecutionMode Dispatcher
//!        ↓
//! ┌──────────────────────────────────────────┐
//! │ SingleShot: run_with_inputs() once       │
//! │ Autoregressive: token generation loop    │
//! │ WhisperDecoder: encoder + decoder loop   │
//! │ BERT: integer token inputs (int64)       │
//! │ TTS: phoneme IDs + voice embedding       │
//! └──────────────────────────────────────────┘
//!        ↓
//!    RawOutputs
//! ```

pub mod autoregressive;
pub mod bert;
pub mod single_shot;
pub mod tts;
pub mod whisper;

// Re-export commonly used helpers
pub use autoregressive::execute_autoregressive_stage;
pub use bert::execute_bert_inference;
pub use single_shot::execute_single_shot_stage;
pub use tts::execute_tts_inference;
pub use whisper::execute_whisper_decoder_stage;

/// Parse KV cache input name from HuggingFace format.
/// Format: past_key_values.{layer}.{decoder|encoder}.{key|value}
/// Returns: (layer_index, is_encoder, is_key)
pub fn parse_kv_cache_name(name: &str) -> Option<(usize, bool, bool)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 4 || parts[0] != "past_key_values" {
        return None;
    }

    let layer = parts[1].parse::<usize>().ok()?;
    let is_encoder = parts[2] == "encoder";
    let is_key = parts[3] == "key";

    Some((layer, is_encoder, is_key))
}

/// Parse present output name from HuggingFace format (full version).
/// Format: present.{layer}.{decoder|encoder}.{key|value}
/// Returns: (layer_index, is_encoder, is_key)
pub fn parse_present_name_full(name: &str) -> Option<(usize, bool, bool)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 4 || parts[0] != "present" {
        return None;
    }

    let layer = parts[1].parse::<usize>().ok()?;
    let is_encoder = parts[2] == "encoder";
    let is_key = parts[3] == "key";

    Some((layer, is_encoder, is_key))
}

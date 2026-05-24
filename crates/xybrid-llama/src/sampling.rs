//! Sampling parameters for generation.
//!
//! Data-only. The defaults match the pre-refactor `SamplingParams`
//! that lived inside `xybrid-core::runtime_adapter::llama_cpp`
//! byte-for-byte.

/// Sampling parameters consumed by [`crate::generate_with_stops`] and
/// [`crate::generate_streaming`].
#[derive(Clone)]
pub struct SamplingParams {
    /// Sampling temperature. `0.0` is greedy.
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Top-k sampling. `0` disables top-k filtering.
    pub top_k: usize,
    /// Repetition penalty. `1.0` disables; values > 1 penalise repeats.
    pub repeat_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

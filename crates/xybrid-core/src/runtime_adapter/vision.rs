//! Shared vision-backend contracts.
//!
//! `VisionEncoder` is the backend-agnostic embedding-style seam for runtimes
//! that expose image features as tensors before language-model decode, such as
//! future MLX, Candle, or ORT vision encoders. llama.cpp multimodal models use
//! the richer backend-owned `mtmd` chunk/helper path instead, because that path
//! owns prompt chunk ordering, image tokenization, M-RoPE state, and decode
//! bookkeeping that should not be flattened into this contract.

use ndarray::ArrayD;

use super::BackendResult;

/// Token IDs inserted into the text prompt as image placeholders.
pub type VisionTokenId = i32;

/// Embedding-style vision encoder output consumed by compatible VLM backends.
#[derive(Debug, Clone)]
pub struct VisionEmbeddings {
    /// Placeholder tokens that occupy the image span in the text prompt.
    pub placeholder_tokens: Vec<VisionTokenId>,
    /// Image embeddings aligned with `placeholder_tokens`.
    pub embeddings: ArrayD<f32>,
}

/// Backend-agnostic contract for runtimes with a separate vision encoder.
pub trait VisionEncoder: Send + Sync {
    /// Encode a preprocessed image tensor into language-model placeholder
    /// tokens and embeddings.
    fn encode(&mut self, image_tensor: ArrayD<f32>) -> BackendResult<VisionEmbeddings>;
}

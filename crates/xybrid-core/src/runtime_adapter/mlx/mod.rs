//! MLX runtime adapter (Apple Silicon).
//!
//! This module hosts the MLX LLM adapter — config loader, KV cache, and
//! [`InferenceBackend`](crate::runtime_adapter::InferenceBackend) /
//! [`LlmBackend`](crate::runtime_adapter::llm::LlmBackend) trait wiring.
//! Architecture-specific builders (Qwen 3.5, Gemma 4, LFM 3.5) and the actual
//! generation loop land in subsequent stories (US-011..US-014); this iteration
//! provides the structural shell.
//!
//! # Why only Apple
//!
//! MLX links against Metal and Accelerate. The xcframework we vendor at
//! `vendor/mlx-apple/mlx.xcframework` only has Apple slices. Enabling this
//! feature on another target is a build configuration error, caught here
//! rather than at link time with a confusing symbol mismatch.
//!
//! # Relationship to other LLM backends
//!
//! MLX is **mutually compatible** with `llm-llamacpp` — both can be enabled in
//! the same build. The runtime selector (US-016) picks MLX when the host is
//! Apple Silicon and the registry has an `mlx` variant; otherwise it falls
//! through to llama.cpp.

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
compile_error!(
    "feature `llm-mlx` can only be enabled on macOS or iOS targets. \
     The MLX runtime depends on Metal and Accelerate, which are Apple-only. \
     Use `llm-llamacpp` on non-Apple platforms."
);

pub mod arch;
pub mod chat_template;
pub mod embedding;
pub mod generate;
pub mod kv_cache;
pub mod model;
pub mod sampler;
pub mod tokenizer;

pub use chat_template::{ChatTemplate, ChatTemplateError, ChatTemplateResult, RenderOptions};
pub use embedding::{
    apply_pooling, l2_norm, l2_normalize, MlxEmbeddingAdapter, MlxEmbeddingConfig, Pooling,
};
pub use kv_cache::{KvCache, KvCachePage, KV_CACHE_PAGE_TOKENS};
pub use model::{
    MlxLlmAdapter, MlxLlmConfig, MlxLlmError, MlxLlmResult, ModelArchitecture, ModelConfig,
};
pub use sampler::{Sampler, SamplerError, SamplerResult};

// Re-export the canonical xybrid-mlx error/dtype/envelope surface used by
// the adapter so downstream consumers don't have to also depend on
// `xybrid-mlx` directly.
pub use xybrid_mlx::{
    EnvelopePayload, EnvelopeSource, MlxDtype, MlxError, MlxResult, OwnedEnvelopePayload,
};

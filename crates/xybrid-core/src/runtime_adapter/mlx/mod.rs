//! MLX runtime adapter and cross-platform non-linking selector tier.
//!
//! This module hosts the MLX SafeTensors adapters: the local LLM adapter,
//! embedding adapter, config loaders, tokenizer/chat-template support, KV
//! cache, architecture-specific builders (Qwen 3, Gemma 4, LFM 2/2.5/3.5,
//! BERT/nomic-BERT), and trait wiring into the execution stack.
//!
//! # Feature tiers
//!
//! `llm-mlx` compiles this module's config parsing, tokenizer, selector, and
//! explicit error surfaces without linking MLX. `llm-mlx-runtime` is the Apple
//! Silicon macOS runtime tier: it links `vendor/mlx-apple/mlx.xcframework`
//! through `xybrid-mlx/bindings`, which depends on Metal and Accelerate.
//! Enabling the runtime tier on another target, including current iOS builds,
//! is a build configuration error, caught here rather than at link time with a
//! confusing symbol mismatch.
//!
//! # Relationship to other LLM backends
//!
//! MLX is **mutually compatible** with `llm-llamacpp` — both can be enabled in
//! the same build. The runtime selector (US-016) picks MLX when the host is
//! Apple Silicon macOS and the registry advertises a SafeTensors variant for
//! a backend-selectable LLM or embedding task; otherwise it falls through to
//! llama.cpp or the registry default.

#[cfg(all(
    feature = "llm-mlx-runtime",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!(
    "feature `llm-mlx-runtime` is currently supported only on Apple Silicon macOS \
     (`aarch64-apple-darwin`). iOS remains non-linking only until upstream MLX ships \
     a Metal-enabled iOS slice. Use `llm-mlx` for selector/docs checks or \
     `llm-llamacpp` for non-macOS inference."
);

pub mod arch;
#[doc(hidden)]
pub mod bench_fixtures;
#[doc(hidden)]
pub mod bench_report;
pub mod chat_template;
pub mod embedding;
pub mod generate;
pub mod kv_cache;
pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod weights;

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

#[cfg(all(
    test,
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) static MLX_RUNTIME_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

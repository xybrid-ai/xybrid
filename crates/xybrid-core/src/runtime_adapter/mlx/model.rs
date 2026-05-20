//! `MlxLlmAdapter` — MLX LLM inference adapter.
//!
//! Reads `config.json`, `tokenizer.json`, and SafeTensors weights from a model
//! directory, dispatches on the upstream `model_type` field, validates the
//! architecture-specific weight manifest, and hands generation calls to the
//! MLX generate loop when `llm-mlx-runtime` is available. Qwen has the
//! most complete Apple-linked runtime path. Qwen, Gemma, and LFM include
//! resident generation state for incremental decode; LFM carries both
//! attention K/V state and short-conv recurrent state. Gemma and LFM have
//! synthetic runtime coverage plus env-gated real bundle smokes; their exact
//! BF16 benchmark rows remain informational until the MLX measurements justify
//! parity gating.
//!
//! Trait coverage:
//! - [`LlmBackend`](crate::runtime_adapter::llm::LlmBackend): `load` /
//!   `is_loaded` / `unload` / `name` / `supported_formats` are wired; the
//!   generation methods route through `generate.rs`.
//! - [`InferenceBackend`](crate::runtime_adapter::InferenceBackend):
//!   `runtime_type` returns [`RuntimeType::Mlx`], `load_model` mirrors the
//!   `LlmBackend::load` path, and the tensor-level methods report that this
//!   adapter operates above tensors (LLM token loop, not single-shot tensor
//!   inference).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::ArrayD;
use serde::Deserialize;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::runtime_adapter::inference_backend::{
    BackendError, BackendResult, InferenceBackend, RuntimeType,
};
use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, LlmBackend, LlmConfig, LlmResult,
    StreamingCallback, StreamingError,
};
use crate::runtime_adapter::AdapterError;

use super::chat_template::ChatTemplate;
use super::generate::{self, GenerateParams};
use super::kv_cache::KvCache;
use super::sampler::SamplerError;
use super::tokenizer as tokenizer_loader;
use super::weights::SafeTensorBundle;

/// Result alias for the MLX adapter's local error type.
pub type MlxLlmResult<T> = Result<T, MlxLlmError>;

/// Errors produced by the MLX LLM adapter.
///
/// Distinct from [`AdapterError`] / [`BackendError`] so callers can react
/// specifically to MLX-shaped failures (notably
/// [`MlxLlmError::UnsupportedArchitecture`], which the runtime selector
/// uses as the "fall back to llama.cpp" signal). `From` impls funnel into
/// the trait-required error types when the adapter is consumed via the
/// generic [`LlmBackend`] / [`InferenceBackend`] trait surface.
#[derive(Debug, Error)]
pub enum MlxLlmError {
    /// `config.json`'s `model_type` field doesn't match any architecture
    /// builder we ship. The variant carries the raw string so the
    /// runtime-selector log line stays useful when a new MLX-LM model
    /// family appears upstream.
    #[error(
        "unsupported MLX model architecture `{model_type}`. \
         Supported: Qwen 3 (`qwen3`), Gemma 4 (`gemma4`), LFM 2/2.5/3.5 (`lfm2`, `lfm`, `lfm3`). \
         Fall back to llama.cpp until the matching builder lands."
    )]
    UnsupportedArchitecture { model_type: String },

    /// The model architecture is known, but its SafeTensors bundle uses a
    /// quantized layout that the MLX adapter cannot execute yet.
    #[error(
        "unsupported MLX quantization for `{model_type}`: {bits}-bit/group={group_size}. \
         {reason}. Register a GGUF fallback variant or republish this MLX variant as dequantized SafeTensors."
    )]
    UnsupportedQuantization {
        model_type: String,
        bits: u32,
        group_size: u32,
        reason: &'static str,
    },

    /// A required file is missing from the model directory.
    #[error("missing file `{file}` in MLX model directory `{dir}`")]
    MissingFile { file: &'static str, dir: PathBuf },

    /// A sharded SafeTensors shard was found without the HuggingFace index
    /// file required to map tensor names to shard files.
    #[error(
        "MLX sharded SafeTensors bundles require `model.safetensors.index.json`: \
         found orphan shard `{marker}` in `{dir}`."
    )]
    OrphanShardedWeights { marker: PathBuf, dir: PathBuf },

    /// Generic IO failure while reading model files.
    #[error("IO error reading MLX model file: {0}")]
    Io(#[from] std::io::Error),

    /// `config.json` failed to parse.
    #[error("failed to parse MLX `config.json`: {0}")]
    ConfigParse(#[from] serde_json::Error),

    /// `tokenizer.json` failed to load.
    #[error("failed to load MLX `tokenizer.json`: {0}")]
    TokenizerLoad(String),

    /// Adapter wasn't loaded when a method that requires a loaded model
    /// was called.
    #[error("MLX adapter has no model loaded")]
    NotLoaded,

    /// A code path is implemented only in a stricter build/runtime tier.
    /// Carries a human-readable feature label and the gate the caller must
    /// satisfy so non-runtime builds surface an actionable error.
    #[error("MLX adapter `{feature}` is unavailable in this build (gate: {story})")]
    NotImplemented {
        feature: &'static str,
        story: &'static str,
    },

    /// Architecture config failed arch-specific validation (e.g. zero
    /// dimensions, contradictory GQA group sizes). Distinct from
    /// [`Self::ConfigParse`], which is the serde-level parse error.
    #[error("MLX model config failed validation: {0}")]
    ConfigInvalid(String),

    /// Streaming callback returned an error. Keep the original boxed
    /// error intact so cloud-fallback abort markers survive the
    /// backend-specific MLX layer.
    #[error("streaming callback error: {0}")]
    StreamingCallback(StreamingError),

    /// Weight tensor loading failed — either the safetensors file is
    /// malformed or a required tensor is missing / wrong dtype. Carries
    /// the weight path for the error message and a free-form reason.
    #[error("failed to load MLX weights from `{path}`: {reason}")]
    WeightLoad { path: PathBuf, reason: String },

    /// Sampler rejected the logits distribution (empty vocab, non-finite
    /// values, or every candidate filtered out by top_k/top_p/min_p).
    /// Wraps the underlying [`SamplerError`] so callers see the exact
    /// failure mode.
    #[error("MLX sampler failed: {0}")]
    Sampler(#[from] SamplerError),

    /// A call into `xybrid-mlx` returned an error — typically from a
    /// dispatch into the underlying mlx-c library. Carries the
    /// [`xybrid_mlx::MlxError`] verbatim.
    #[error("xybrid-mlx error: {0}")]
    Mlx(#[from] xybrid_mlx::MlxError),
}

impl From<MlxLlmError> for AdapterError {
    fn from(e: MlxLlmError) -> Self {
        match e {
            MlxLlmError::NotLoaded => AdapterError::ModelNotLoaded(e.to_string()),
            MlxLlmError::MissingFile { .. } | MlxLlmError::Io(_) => {
                AdapterError::ModelNotFound(e.to_string())
            }
            MlxLlmError::StreamingCallback(error) => {
                AdapterError::from_streaming_callback_error(error)
            }
            _ => AdapterError::RuntimeError(e.to_string()),
        }
    }
}

impl From<MlxLlmError> for BackendError {
    fn from(e: MlxLlmError) -> Self {
        match e {
            MlxLlmError::NotLoaded => BackendError::ModelNotLoaded,
            MlxLlmError::MissingFile { .. } | MlxLlmError::Io(_) => {
                BackendError::LoadFailed(e.to_string())
            }
            _ => BackendError::RuntimeError(e.to_string()),
        }
    }
}

/// Configuration for the MLX LLM adapter. Distinct from
/// [`LlmConfig`](crate::runtime_adapter::llm::LlmConfig) — that one is shared
/// across all LLM backends and is keyed by file path; MLX prefers a directory
/// + structured options.
#[derive(Debug, Clone)]
pub struct MlxLlmConfig {
    /// Maximum sequence length the KV cache is sized for. Bounded by the
    /// model's `max_position_embeddings` field but capped by the consumer
    /// to keep cache memory predictable.
    pub max_seq_len: usize,
    /// Default generation parameters used when `LlmBackend::generate` is
    /// called with no per-call override.
    pub default_generation: GenerationConfig,
}

impl Default for MlxLlmConfig {
    fn default() -> Self {
        // 4096 mirrors the cross-backend default in `LlmConfig`. Models with
        // larger context windows can override; the cache is paged so a
        // higher cap costs only descriptor memory until written.
        Self {
            max_seq_len: 4096,
            default_generation: GenerationConfig::default(),
        }
    }
}

impl MlxLlmConfig {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            max_seq_len,
            default_generation: GenerationConfig::default(),
        }
    }

    pub fn with_generation(mut self, gen: GenerationConfig) -> Self {
        self.default_generation = gen;
        self
    }
}

/// Subset of an MLX-LM `config.json` we need at the trait surface. Captured
/// as a public type because the architecture builders read the same fields.
/// Unknown fields are tolerated via serde's default behaviour — only
/// `model_type` is strictly required for the generic dispatch path.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Upstream architecture identifier. Examples: `"qwen2"`, `"gemma2"`,
    /// `"llama"`, `"lfm"`. Used by [`MlxLlmAdapter::load`] to pick a
    /// builder.
    pub model_type: String,

    /// Hidden size (model dimension). Optional at the generic trait surface;
    /// architecture-specific builders validate it when their forward path
    /// requires it.
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of transformer layers. Used by the KV cache to size its
    /// per-layer page vectors. Optional here for the same reason as
    /// `hidden_size`.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,

    /// Maximum positions the model itself was trained for (upper bound on
    /// `MlxLlmConfig::max_seq_len`).
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
}

/// Architecture variant resolved from `config.json`.
///
/// Marked `#[non_exhaustive]` so downstream matchers need a catch-all arm;
/// adding a new variant in a later story is a non-breaking change.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ModelArchitecture {
    /// Qwen 3 family — upstream `model_type = "qwen3"`. See
    /// [`super::arch::qwen35`].
    Qwen35,
    /// Gemma 4 family — upstream `model_type = "gemma4"`. See
    /// [`super::arch::gemma4`].
    Gemma4,
    /// Liquid Foundation Model 2 / 2.5 / 3.5 family — upstream
    /// `model_type = "lfm2"`, `"lfm"` or `"lfm3"`.
    /// See [`super::arch::lfm35`].
    Lfm35,
}

/// Internal state held while a model is loaded. Kept in
/// [`MlxLlmAdapter::loaded`] so trait methods can `&self`-borrow the
/// constituent pieces (tokenizer, cache) without exposing them publicly.
#[derive(Debug)]
struct LoadedState {
    model_dir: PathBuf,
    config: ModelConfig,
    tokenizer: Tokenizer,
    kv_cache: KvCache,
    arch: ModelArchitecture,
    #[allow(dead_code)]
    qwen3_config: Option<super::arch::qwen35::Qwen3Config>,
    #[allow(dead_code)]
    gemma4_config: Option<super::arch::gemma4::Gemma4Config>,
    #[allow(dead_code)]
    lfm35_config: Option<super::arch::lfm35::Lfm35Config>,
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    qwen3_runtime_weights: Option<std::sync::Mutex<super::arch::qwen35::runtime::Qwen35Weights>>,
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    gemma4_runtime_weights: Option<std::sync::Mutex<super::arch::gemma4::runtime::Gemma4Weights>>,
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    lfm35_runtime_weights: Option<std::sync::Mutex<super::arch::lfm35::runtime::Lfm35Weights>>,
    /// Loaded chat template (from `tokenizer_config.json`). Optional
    /// because base models without a `chat_template` field still load —
    /// they just can't be called via [`LlmBackend::generate`] (which
    /// requires a rendered prompt).
    chat_template: Option<ChatTemplate>,
}

/// MLX LLM adapter. The adapter is `Send + Sync` —
/// the underlying tokenizer is `Send + Sync`, the generic KV descriptor cache
/// is plain Rust pre-allocated state, and runtime MLX arrays live behind
/// mutexes so `&self` generation calls can reuse loaded handles without
/// breaking the trait's thread-safety contract.
#[derive(Debug)]
pub struct MlxLlmAdapter {
    config: MlxLlmConfig,
    loaded: Option<LoadedState>,
}

impl MlxLlmAdapter {
    /// Empty, unloaded adapter. The tensor-level [`InferenceBackend`] entry
    /// path goes through this constructor + `load_model`.
    pub fn new(config: MlxLlmConfig) -> Self {
        Self {
            config,
            loaded: None,
        }
    }

    /// One-shot load: read `{model_dir}/config.json`,
    /// `{model_dir}/tokenizer.json`, verify `{model_dir}/model.safetensors`
    /// exists, dispatch on the parsed `model_type`, and return a fully
    /// loaded adapter. Returns [`MlxLlmError::UnsupportedArchitecture`] for
    /// any architecture without a builder.
    pub fn load(model_dir: &Path, config: &MlxLlmConfig) -> MlxLlmResult<Self> {
        let mut adapter = Self::new(config.clone());
        adapter.load_in_place(model_dir)?;
        Ok(adapter)
    }

    fn load_in_place(&mut self, model_dir: &Path) -> MlxLlmResult<()> {
        // Required files. We enumerate them by name so missing-file errors
        // pinpoint *which* file the user needs to re-download — a generic
        // IO error against the directory is not actionable.
        let config_path = require_file(model_dir, "config.json")?;
        let tokenizer_path = require_file(model_dir, "tokenizer.json")?;
        let weights = SafeTensorBundle::from_model_dir(model_dir)?;

        let config_json = fs::read_to_string(&config_path)?;
        let model_cfg: ModelConfig = serde_json::from_str(&config_json)?;

        let tokenizer = tokenizer_loader::load_from_file(&tokenizer_path)
            .map_err(|e| MlxLlmError::TokenizerLoad(e.to_string()))?;

        let arch = dispatch_architecture(&model_cfg.model_type)?;

        // Per-arch weight / config validation. The Qwen 3 builder reads
        // the full config and the safetensors header before we cross the
        // Metal boundary — that way the runtime selector (US-016) gets a
        // clean UnsupportedArchitecture / WeightLoad error for quantized
        // or malformed bundles instead of a confusing post-link failure.
        let mut qwen3_config = None;
        let mut gemma4_config = None;
        let mut lfm35_config = None;
        let n_layers = match arch {
            ModelArchitecture::Qwen35 => {
                let qwen_cfg = super::arch::qwen35::load(model_dir, &weights)?;
                let n_layers = qwen_cfg.num_hidden_layers;
                qwen3_config = Some(qwen_cfg);
                n_layers
            }
            ModelArchitecture::Gemma4 => {
                let gemma_cfg = super::arch::gemma4::load(model_dir, &weights)?;
                let n_layers = gemma_cfg.num_hidden_layers;
                gemma4_config = Some(gemma_cfg);
                n_layers
            }
            ModelArchitecture::Lfm35 => {
                let lfm_cfg = super::arch::lfm35::load(model_dir, &weights)?;
                let n_layers = lfm_cfg.num_hidden_layers;
                lfm35_config = Some(lfm_cfg);
                n_layers
            }
        };

        let kv_cache = KvCache::new(n_layers, self.config.max_seq_len);
        #[cfg(all(
            feature = "llm-mlx-runtime",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let qwen3_runtime_weights =
            build_qwen3_runtime_weights(arch, qwen3_config.as_ref(), &weights)?;
        #[cfg(all(
            feature = "llm-mlx-runtime",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let gemma4_runtime_weights =
            build_gemma4_runtime_weights(arch, gemma4_config.as_ref(), &weights)?;
        #[cfg(all(
            feature = "llm-mlx-runtime",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let lfm35_runtime_weights =
            build_lfm35_runtime_weights(arch, lfm35_config.as_ref(), &weights)?;

        // Chat template is optional: base (non-chat) models still load. The
        // `generate` path requires it; `generate_raw` does not. Missing-template
        // errors are surfaced only when a chat API is actually called.
        let chat_template = match ChatTemplate::load_from_dir(model_dir) {
            Ok(t) => Some(t),
            Err(super::chat_template::ChatTemplateError::MissingTemplate)
                if matches!(arch, ModelArchitecture::Gemma4) =>
            {
                Some(ChatTemplate::gemma4_fallback())
            }
            Err(super::chat_template::ChatTemplateError::Io(_))
                if matches!(arch, ModelArchitecture::Gemma4) =>
            {
                Some(ChatTemplate::gemma4_fallback())
            }
            Err(super::chat_template::ChatTemplateError::MissingTemplate) => None,
            Err(super::chat_template::ChatTemplateError::Io(_)) => None, // tokenizer_config.json missing
            Err(e) => {
                return Err(MlxLlmError::ConfigInvalid(format!(
                    "failed to load chat template: {e}"
                )));
            }
        };

        self.loaded = Some(LoadedState {
            model_dir: model_dir.to_path_buf(),
            config: model_cfg,
            tokenizer,
            kv_cache,
            arch,
            qwen3_config,
            gemma4_config,
            lfm35_config,
            #[cfg(all(
                feature = "llm-mlx-runtime",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            qwen3_runtime_weights,
            #[cfg(all(
                feature = "llm-mlx-runtime",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            gemma4_runtime_weights,
            #[cfg(all(
                feature = "llm-mlx-runtime",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            lfm35_runtime_weights,
            chat_template,
        });
        Ok(())
    }

    /// Read-only view of the parsed `config.json`. `None` until a model is
    /// loaded.
    pub fn model_config(&self) -> Option<&ModelConfig> {
        self.loaded.as_ref().map(|s| &s.config)
    }

    /// Read-only view of the active KV cache.
    pub fn kv_cache(&self) -> Option<&KvCache> {
        self.loaded.as_ref().map(|s| &s.kv_cache)
    }

    /// Mutable access to the KV cache for the generate loop (US-014).
    pub fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        self.loaded.as_mut().map(|s| &mut s.kv_cache)
    }

    /// Active tokenizer. `None` until a model is loaded.
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.loaded.as_ref().map(|s| &s.tokenizer)
    }

    /// Architecture variant resolved at load time. Used by the generation and
    /// embedding entry points to pick an architecture-specific forward
    /// function.
    pub fn architecture(&self) -> Option<ModelArchitecture> {
        self.loaded.as_ref().map(|s| s.arch)
    }

    /// Parsed Qwen 3 config retained on the adapter after load. The MLX
    /// runtime generation path uses this cached config alongside cached
    /// weight handles instead of reparsing `config.json` per request.
    #[cfg(test)]
    pub(crate) fn qwen3_config(&self) -> Option<&super::arch::qwen35::Qwen3Config> {
        self.loaded.as_ref().and_then(|s| s.qwen3_config.as_ref())
    }

    /// Borrow the loaded Qwen runtime state for generation. The weight
    /// set is mutex-protected because [`LlmBackend`] exposes generation
    /// through `&self`; concurrent calls serialize on the shared MLX
    /// handles instead of rebuilding them.
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(crate) fn qwen3_runtime_state(
        &self,
    ) -> MlxLlmResult<(
        &super::arch::qwen35::Qwen3Config,
        std::sync::MutexGuard<'_, super::arch::qwen35::runtime::Qwen35Weights>,
    )> {
        let loaded = self.loaded.as_ref().ok_or(MlxLlmError::NotLoaded)?;
        let cfg = loaded.qwen3_config.as_ref().ok_or_else(|| {
            MlxLlmError::ConfigInvalid(
                "Qwen generation requested but no Qwen config is loaded".into(),
            )
        })?;
        let weights = loaded.qwen3_runtime_weights.as_ref().ok_or_else(|| {
            MlxLlmError::ConfigInvalid(
                "Qwen generation requested but runtime weights are not loaded".into(),
            )
        })?;
        let guard = weights.lock().map_err(|_| {
            MlxLlmError::ConfigInvalid("Qwen runtime weight cache lock is poisoned".into())
        })?;
        Ok((cfg, guard))
    }

    /// Borrow the loaded Gemma runtime state for generation. The weight
    /// set is mutex-protected for the same reason as Qwen: generation is
    /// exposed through `&self` and MLX handles are Send but not Sync.
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(crate) fn gemma4_runtime_state(
        &self,
    ) -> MlxLlmResult<(
        &super::arch::gemma4::Gemma4Config,
        std::sync::MutexGuard<'_, super::arch::gemma4::runtime::Gemma4Weights>,
    )> {
        let loaded = self.loaded.as_ref().ok_or(MlxLlmError::NotLoaded)?;
        let cfg = loaded.gemma4_config.as_ref().ok_or_else(|| {
            MlxLlmError::ConfigInvalid(
                "Gemma generation requested but no Gemma config is loaded".into(),
            )
        })?;
        let weights = loaded.gemma4_runtime_weights.as_ref().ok_or_else(|| {
            MlxLlmError::ConfigInvalid(
                "Gemma generation requested but runtime weights are not loaded".into(),
            )
        })?;
        let guard = weights.lock().map_err(|_| {
            MlxLlmError::ConfigInvalid("Gemma runtime weight cache lock is poisoned".into())
        })?;
        Ok((cfg, guard))
    }

    /// Borrow the loaded LFM runtime state for generation.
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(crate) fn lfm35_runtime_state(
        &self,
    ) -> MlxLlmResult<(
        &super::arch::lfm35::Lfm35Config,
        std::sync::MutexGuard<'_, super::arch::lfm35::runtime::Lfm35Weights>,
    )> {
        let loaded = self.loaded.as_ref().ok_or(MlxLlmError::NotLoaded)?;
        let cfg = loaded.lfm35_config.as_ref().ok_or_else(|| {
            MlxLlmError::ConfigInvalid(
                "LFM generation requested but no LFM config is loaded".into(),
            )
        })?;
        let weights = loaded.lfm35_runtime_weights.as_ref().ok_or_else(|| {
            MlxLlmError::ConfigInvalid(
                "LFM generation requested but runtime weights are not loaded".into(),
            )
        })?;
        let guard = weights.lock().map_err(|_| {
            MlxLlmError::ConfigInvalid("LFM runtime weight cache lock is poisoned".into())
        })?;
        Ok((cfg, guard))
    }

    /// Path the model was loaded from. Useful for error messages.
    pub fn model_dir(&self) -> Option<&Path> {
        self.loaded.as_ref().map(|s| s.model_dir.as_path())
    }

    /// MLX adapter config (max_seq_len, default generation params).
    pub fn config(&self) -> &MlxLlmConfig {
        &self.config
    }

    /// Chat template loaded from the bundle's `tokenizer_config.json`.
    /// `None` when no model is loaded OR when the bundle is a base (non-
    /// chat) model without a `chat_template` field.
    pub fn chat_template(&self) -> Option<&ChatTemplate> {
        self.loaded.as_ref().and_then(|s| s.chat_template.as_ref())
    }

    fn prompt_from_messages(&self, messages: &[ChatMessage]) -> MlxLlmResult<String> {
        if self.loaded.is_none() {
            return Err(MlxLlmError::NotLoaded);
        }
        if self.chat_template().is_some() {
            return generate::render_prompt(self, messages);
        }
        match messages {
            [ChatMessage {
                role: crate::ir::MessageRole::User,
                content,
            }] => Ok(content.clone()),
            _ => Err(MlxLlmError::ConfigInvalid(
                "MLX bundle has no chat template; only a single user prompt can use raw generation"
                    .into(),
            )),
        }
    }

    /// Short model identifier used in telemetry events. Derived from the
    /// bundle directory's trailing path segment; falls back to `"mlx"`
    /// when no model is loaded.
    pub(crate) fn model_id_or_default(&self) -> String {
        self.loaded
            .as_ref()
            .and_then(|s| {
                s.model_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "mlx".to_string())
    }
}

fn require_file(dir: &Path, file: &'static str) -> MlxLlmResult<PathBuf> {
    let path = dir.join(file);
    if !path.exists() {
        return Err(MlxLlmError::MissingFile {
            file,
            dir: dir.to_path_buf(),
        });
    }
    Ok(path)
}

/// Match the upstream HuggingFace `model_type` string to one of our shipped
/// architecture builders.
///
/// The dispatch table is intentionally narrow — adding a new architecture
/// is a deliberate engineering act (new builder, new tests, new fixtures)
/// not a string-match drive-by. Any unknown family routes to
/// [`MlxLlmError::UnsupportedArchitecture`] which the runtime selector
/// (US-016) converts into a "fall back to llama.cpp" decision.
fn dispatch_architecture(model_type: &str) -> MlxLlmResult<ModelArchitecture> {
    match model_type {
        // US-011: Qwen 3 family.
        "qwen3" => Ok(ModelArchitecture::Qwen35),
        // US-012: Gemma 4 family.
        "gemma4" => Ok(ModelArchitecture::Gemma4),
        // US-013: LFM 2 / 2.5 / 3.5 family. Upstream MLX bundles use
        // `"lfm2"` for current public LFM2 releases, `"lfm"` for some
        // LFM 2 / 3 bundles, and `"lfm3"` for LFM 3.5 specifically.
        "lfm2" | "lfm" | "lfm3" => Ok(ModelArchitecture::Lfm35),
        other => Err(MlxLlmError::UnsupportedArchitecture {
            model_type: other.to_string(),
        }),
    }
}

#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
fn build_qwen3_runtime_weights(
    arch: ModelArchitecture,
    qwen3_config: Option<&super::arch::qwen35::Qwen3Config>,
    weights: &SafeTensorBundle,
) -> MlxLlmResult<Option<std::sync::Mutex<super::arch::qwen35::runtime::Qwen35Weights>>> {
    match arch {
        ModelArchitecture::Qwen35 => {
            let cfg = qwen3_config.ok_or_else(|| {
                MlxLlmError::ConfigInvalid(
                    "Qwen runtime weights requested without a parsed Qwen config".into(),
                )
            })?;
            let weights = super::arch::qwen35::runtime::build(cfg, weights)?;
            Ok(Some(std::sync::Mutex::new(weights)))
        }
        ModelArchitecture::Gemma4 | ModelArchitecture::Lfm35 => Ok(None),
    }
}

#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
fn build_gemma4_runtime_weights(
    arch: ModelArchitecture,
    gemma4_config: Option<&super::arch::gemma4::Gemma4Config>,
    weights: &SafeTensorBundle,
) -> MlxLlmResult<Option<std::sync::Mutex<super::arch::gemma4::runtime::Gemma4Weights>>> {
    match arch {
        ModelArchitecture::Gemma4 => {
            let cfg = gemma4_config.ok_or_else(|| {
                MlxLlmError::ConfigInvalid(
                    "Gemma runtime weights requested without a parsed Gemma config".into(),
                )
            })?;
            let weights = super::arch::gemma4::runtime::build(cfg, weights)?;
            Ok(Some(std::sync::Mutex::new(weights)))
        }
        ModelArchitecture::Qwen35 | ModelArchitecture::Lfm35 => Ok(None),
    }
}

#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
fn build_lfm35_runtime_weights(
    arch: ModelArchitecture,
    lfm35_config: Option<&super::arch::lfm35::Lfm35Config>,
    weights: &SafeTensorBundle,
) -> MlxLlmResult<Option<std::sync::Mutex<super::arch::lfm35::runtime::Lfm35Weights>>> {
    match arch {
        ModelArchitecture::Lfm35 => {
            let cfg = lfm35_config.ok_or_else(|| {
                MlxLlmError::ConfigInvalid(
                    "LFM runtime weights requested without a parsed LFM config".into(),
                )
            })?;
            let weights = super::arch::lfm35::runtime::build(cfg, weights)?;
            Ok(Some(std::sync::Mutex::new(weights)))
        }
        ModelArchitecture::Qwen35 | ModelArchitecture::Gemma4 => Ok(None),
    }
}

// =============================================================================
// LlmBackend impl
// =============================================================================

impl LlmBackend for MlxLlmAdapter {
    fn name(&self) -> &str {
        "mlx"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        // MLX-LM bundles ship a directory of files. The "format" key in
        // model_metadata.json — see xybrid-pack — is `mlx` and the bundle
        // expands to `config.json` + `tokenizer.json` + `model.safetensors`
        // (with optional sharded safetensors). Listing both extensions
        // signals the bundle resolver that the adapter accepts a directory
        // path containing a safetensors file.
        vec!["mlx", "safetensors"]
    }

    fn load(&mut self, config: &LlmConfig) -> LlmResult<()> {
        // The shared LlmConfig keys by file path; the MLX adapter wants a
        // directory. Treat `model_path` as a path that's either the model
        // directory itself or a file inside it (in which case we use the
        // parent).
        let path = Path::new(&config.model_path);
        let model_dir: &Path = if path.is_dir() {
            path
        } else {
            path.parent().ok_or_else(|| {
                AdapterError::InvalidInput(format!(
                    "MLX adapter expects a model directory; got file path with no parent: {}",
                    config.model_path
                ))
            })?
        };
        self.load_in_place(model_dir).map_err(AdapterError::from)?;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn unload(&mut self) -> LlmResult<()> {
        self.loaded = None;
        Ok(())
    }

    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        if self.loaded.is_none() {
            return Err(MlxLlmError::NotLoaded.into());
        }
        let prompt = self
            .prompt_from_messages(messages)
            .map_err(AdapterError::from)?;
        generate::generate_tokens(self, &prompt, GenerateParams::new(config), None)
            .map_err(AdapterError::from)
    }

    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
        if self.loaded.is_none() {
            return Err(MlxLlmError::NotLoaded.into());
        }
        generate::generate_tokens(self, prompt, GenerateParams::new(config), None)
            .map_err(AdapterError::from)
    }

    fn generate_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        on_token: StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        if self.loaded.is_none() {
            return Err(MlxLlmError::NotLoaded.into());
        }
        let prompt = self
            .prompt_from_messages(messages)
            .map_err(AdapterError::from)?;
        generate::generate_tokens(self, &prompt, GenerateParams::new(config), Some(on_token))
            .map_err(AdapterError::from)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn context_length(&self) -> Option<usize> {
        Some(self.config.max_seq_len)
    }
}

// =============================================================================
// InferenceBackend impl
// =============================================================================

impl InferenceBackend for MlxLlmAdapter {
    fn runtime_type(&self) -> RuntimeType {
        RuntimeType::Mlx
    }

    fn load_model(&mut self, model_path: &Path, _config_path: Option<&Path>) -> BackendResult<()> {
        let model_dir: &Path = if model_path.is_dir() {
            model_path
        } else {
            model_path.parent().ok_or_else(|| {
                BackendError::LoadFailed(format!(
                    "MLX adapter expects a model directory; got file path with no parent: {}",
                    model_path.display()
                ))
            })?
        };
        self.load_in_place(model_dir).map_err(BackendError::from)?;
        Ok(())
    }

    fn run_inference(
        &self,
        _inputs: HashMap<String, ArrayD<f32>>,
    ) -> BackendResult<HashMap<String, ArrayD<f32>>> {
        // The MLX LLM adapter operates above the single-shot tensor
        // surface — generation requires the chat / sampling loop in
        // `LlmBackend::generate`. Surfacing a dedicated error here makes
        // mis-routing through `InferenceBackend` obvious instead of
        // returning empty results.
        Err(BackendError::RuntimeError(
            "MLX LLM adapter does not support tensor-level run_inference; \
             use LlmBackend::generate (US-014) instead"
                .to_string(),
        ))
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn input_names(&self) -> BackendResult<Vec<String>> {
        if self.loaded.is_none() {
            return Err(BackendError::ModelNotLoaded);
        }
        Ok(vec!["input_ids".to_string()])
    }

    fn output_names(&self) -> BackendResult<Vec<String>> {
        if self.loaded.is_none() {
            return Err(BackendError::ModelNotLoaded);
        }
        Ok(vec!["logits".to_string()])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Writes a minimal MLX-LM bundle into `dir` for unit-test loading.
    /// `model_type` controls the dispatch arm exercised by the test.
    ///
    /// The config carries the Qwen 3-required fields so the arch
    /// validator accepts it when `model_type = "qwen3"`; the
    /// `model.safetensors` file is a valid (but tensor-light) manifest.
    /// For non-qwen3 `model_type` values the dispatch fails before
    /// either is read, so the extra fields are harmless.
    fn write_dummy_bundle(dir: &Path, model_type: &str) {
        // Upstream LFM uses `norm_eps` for the RMSNorm epsilon, but the
        // `Lfm35Config` deserialiser accepts `rms_norm_eps` as an alias
        // so one base JSON blob stays valid across dispatch branches.
        let mut cfg = serde_json::json!({
            "model_type": model_type,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "vocab_size": 100,
            "max_position_embeddings": 1024,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": true,
            "head_dim": 4,
        });
        if matches!(model_type, "lfm2" | "lfm" | "lfm3") {
            cfg["layer_types"] = serde_json::json!(["conv", "full_attention"]);
            cfg["conv_l_cache"] = serde_json::json!(3);
        }
        let mut f = fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(cfg.to_string().as_bytes()).unwrap();

        // Use the qwen tokenizer fixture as a stand-in — any real
        // tokenizer.json works; the dispatch test errors out before the
        // tokenizer is exercised, but parsing it still has to succeed.
        let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        fs::copy(&tok_src, dir.join("tokenizer.json")).unwrap();

        // `model.safetensors`: empty for unknown model types (dispatch
        // errors first), full manifest for `qwen3` / `gemma4` (arch
        // validator enumerates keys). The "full" manifest lists every
        // key the forward pass reads — with a 4-byte placeholder
        // tensor per key, tied embeddings (so no lm_head), and 2 layers.
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let data = vec![0u8; 4];
        let keys: Vec<String> = match model_type {
            "qwen3" => {
                use super::super::arch::qwen35::{expected_weight_keys, Qwen3Config};
                let q_cfg = Qwen3Config {
                    model_type: model_type.into(),
                    hidden_size: 16,
                    num_hidden_layers: 2,
                    num_attention_heads: 4,
                    num_key_value_heads: Some(2),
                    intermediate_size: 32,
                    vocab_size: 100,
                    max_position_embeddings: 1024,
                    rope_theta: 1_000_000.0,
                    rms_norm_eps: 1e-6,
                    tie_word_embeddings: true,
                    head_dim: Some(4),
                    quantization: None,
                };
                expected_weight_keys(&q_cfg)
            }
            "gemma4" => {
                use super::super::arch::gemma4::{expected_weight_keys, Gemma4Config};
                let g_cfg = Gemma4Config {
                    model_type: model_type.into(),
                    hidden_size: 16,
                    num_hidden_layers: 2,
                    num_attention_heads: 4,
                    num_key_value_heads: Some(2),
                    intermediate_size: 32,
                    vocab_size: 100,
                    max_position_embeddings: 1024,
                    rope_theta: 1_000_000.0,
                    rope_local_base_freq: 10_000.0,
                    rms_norm_eps: 1e-6,
                    tie_word_embeddings: true,
                    head_dim: Some(4),
                    global_head_dim: None,
                    layer_types: None,
                    sliding_window: 4096,
                    sliding_window_pattern: 6,
                    query_pre_attn_scalar: None,
                    final_logit_softcapping: None,
                    hidden_size_per_layer_input: None,
                    vocab_size_per_layer_input: None,
                    num_kv_shared_layers: None,
                    full_attention_partial_rotary_factor: 0.25,
                    quantization: None,
                };
                expected_weight_keys(&g_cfg)
            }
            "lfm2" | "lfm" | "lfm3" => {
                use super::super::arch::lfm35::{expected_weight_keys, Lfm35Config};
                let l_cfg = Lfm35Config {
                    model_type: model_type.into(),
                    hidden_size: 16,
                    num_hidden_layers: 2,
                    num_attention_heads: 4,
                    num_key_value_heads: Some(2),
                    intermediate_size: 32,
                    vocab_size: 100,
                    max_position_embeddings: 1024,
                    rope_theta: 1_000_000.0,
                    norm_eps: 1e-5,
                    tie_word_embeddings: true,
                    head_dim: Some(4),
                    conv_l_cache: 3,
                    conv_bias: false,
                    // Matches the bundle config.json above — layer 0
                    // is conv, layer 1 is full attention.
                    layer_types: Some(vec!["conv".into(), "full_attention".into()]),
                    full_attn_idxs: None,
                    quantization: None,
                };
                expected_weight_keys(&l_cfg)
            }
            _ => Vec::new(),
        };

        if !keys.is_empty() {
            let tensors: Vec<(String, TensorView<'_>)> = keys
                .into_iter()
                .map(|k| (k, TensorView::new(Dtype::F32, vec![1], &data).unwrap()))
                .collect();
            safetensors::serialize_to_file(tensors, &None, &dir.join("model.safetensors")).unwrap();
        } else {
            fs::File::create(dir.join("model.safetensors"))
                .unwrap()
                .write_all(b"placeholder")
                .unwrap();
        }
    }

    fn convert_single_weights_to_indexed_shard(dir: &Path) {
        let single = dir.join("model.safetensors");
        let shard_name = "model-00001-of-00002.safetensors";
        let shard = dir.join(shard_name);
        fs::rename(&single, &shard).unwrap();

        let bytes = fs::read(&shard).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let weight_map: serde_json::Map<String, serde_json::Value> = meta
            .tensors()
            .into_keys()
            .map(|name| (name, serde_json::Value::String(shard_name.to_string())))
            .collect();
        let index = serde_json::json!({
            "metadata": {},
            "weight_map": weight_map,
        });
        fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
    }

    #[test]
    fn load_unknown_model_type_returns_unsupported_architecture() {
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "totally-not-a-real-arch");
        let cfg = MlxLlmConfig::default();
        let err = MlxLlmAdapter::load(tmp.path(), &cfg).expect_err("dispatch should error");
        match err {
            MlxLlmError::UnsupportedArchitecture { model_type } => {
                assert_eq!(model_type, "totally-not-a-real-arch");
            }
            other => panic!("expected UnsupportedArchitecture, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_accepts_qwen3() {
        // US-011: `model_type = "qwen3"` now resolves to
        // `ModelArchitecture::Qwen35` rather than erroring out.
        let arch = dispatch_architecture("qwen3").expect("qwen3 dispatch");
        assert!(matches!(arch, ModelArchitecture::Qwen35));
    }

    #[test]
    fn dispatch_accepts_gemma4() {
        // US-012: `model_type = "gemma4"` resolves to
        // `ModelArchitecture::Gemma4`.
        let arch = dispatch_architecture("gemma4").expect("gemma4 dispatch");
        assert!(matches!(arch, ModelArchitecture::Gemma4));
    }

    #[test]
    fn dispatch_accepts_lfm2_lfm_and_lfm3() {
        // US-013: LFM model_type strings drift across public MLX bundles,
        // but all three currently share the same hybrid builder topology.
        let z = dispatch_architecture("lfm2").expect("lfm2 dispatch");
        assert!(matches!(z, ModelArchitecture::Lfm35));
        let a = dispatch_architecture("lfm").expect("lfm dispatch");
        assert!(matches!(a, ModelArchitecture::Lfm35));
        let b = dispatch_architecture("lfm3").expect("lfm3 dispatch");
        assert!(matches!(b, ModelArchitecture::Lfm35));
    }

    #[test]
    fn load_gemma4_returns_loaded_adapter() {
        // End-to-end: bundle with `model_type = "gemma4"` loads without
        // the UnsupportedArchitecture error.
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "gemma4");
        let adapter = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load ok");
        assert!(LlmBackend::is_loaded(&adapter));
        assert!(matches!(
            adapter.architecture(),
            Some(ModelArchitecture::Gemma4)
        ));
    }

    #[test]
    fn load_lfm3_returns_loaded_adapter() {
        // US-013 end-to-end: bundle with `model_type = "lfm3"` loads
        // without the UnsupportedArchitecture error. The dummy bundle
        // ships a mixed conv+attention layer schedule (layer 0 conv,
        // layer 1 attention), so the hybrid weight-key schedule is
        // exercised on the happy path.
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "lfm3");
        let adapter = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load ok");
        assert!(LlmBackend::is_loaded(&adapter));
        assert!(matches!(
            adapter.architecture(),
            Some(ModelArchitecture::Lfm35)
        ));
    }

    #[test]
    fn load_lfm2_returns_loaded_adapter() {
        // Public mlx-community LFM2 BF16 bundles use `model_type = "lfm2"`;
        // they still share the same hybrid conv+attention key schedule.
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "lfm2");
        let adapter = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load ok");
        assert!(LlmBackend::is_loaded(&adapter));
        assert!(matches!(
            adapter.architecture(),
            Some(ModelArchitecture::Lfm35)
        ));
    }

    #[test]
    fn load_qwen3_returns_loaded_adapter() {
        // End-to-end: bundle with `model_type = "qwen3"` loads without the
        // UnsupportedArchitecture error. The dummy bundle has no real
        // safetensors payload, but the non-linking load path only verifies
        // file presence — weight validation is deferred to the arch
        // builder which is exercised in its own tests.
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "qwen3");
        let adapter = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load ok");
        assert!(LlmBackend::is_loaded(&adapter));
        assert!(matches!(
            adapter.architecture(),
            Some(ModelArchitecture::Qwen35)
        ));
    }

    #[test]
    fn load_qwen3_caches_arch_config_for_generation_state() {
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "qwen3");

        let adapter = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load ok");
        let qwen_cfg = adapter
            .qwen3_config()
            .expect("Qwen config should be cached on the loaded adapter");

        assert_eq!(qwen_cfg.model_type, "qwen3");
        assert_eq!(qwen_cfg.num_hidden_layers, 2);
        assert_eq!(qwen_cfg.hidden_size, 16);
    }

    #[test]
    fn load_missing_config_returns_missing_file() {
        let tmp = TempDir::new().unwrap();
        // Intentionally do NOT write any files.
        let err = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default())
            .expect_err("missing files should error");
        match err {
            MlxLlmError::MissingFile { file, .. } => {
                assert_eq!(file, "config.json");
            }
            other => panic!("expected MissingFile, got {other:?}"),
        }
    }

    #[test]
    fn load_sharded_safetensors_index_returns_loaded_adapter() {
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "qwen3");
        convert_single_weights_to_indexed_shard(tmp.path());

        let adapter =
            MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load sharded ok");
        assert!(LlmBackend::is_loaded(&adapter));
        assert!(matches!(
            adapter.architecture(),
            Some(ModelArchitecture::Qwen35)
        ));
    }

    #[test]
    fn load_sharded_safetensors_file_without_index_returns_targeted_error() {
        let tmp = TempDir::new().unwrap();
        write_dummy_bundle(tmp.path(), "qwen3");
        fs::remove_file(tmp.path().join("model.safetensors")).unwrap();
        fs::write(tmp.path().join("model-00001-of-00002.safetensors"), b"").unwrap();

        let err = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default())
            .expect_err("orphan sharded safetensors should be rejected explicitly");
        match err {
            MlxLlmError::OrphanShardedWeights { marker, dir } => {
                assert_eq!(dir, tmp.path());
                assert_eq!(
                    marker.file_name().and_then(|n| n.to_str()),
                    Some("model-00001-of-00002.safetensors")
                );
            }
            other => panic!("expected OrphanShardedWeights, got {other:?}"),
        }
    }

    #[test]
    fn unloaded_adapter_reports_no_state() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::default());
        assert!(!LlmBackend::is_loaded(&adapter));
        assert!(!InferenceBackend::is_loaded(&adapter));
        assert!(adapter.model_config().is_none());
        assert!(adapter.kv_cache().is_none());
        assert!(adapter.tokenizer().is_none());
        assert!(adapter.architecture().is_none());
        assert!(adapter.model_dir().is_none());
        assert_eq!(adapter.config().max_seq_len, 4096);
    }

    #[test]
    fn unloaded_adapter_errors_on_inference_metadata() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::default());
        assert!(matches!(
            adapter.input_names(),
            Err(BackendError::ModelNotLoaded)
        ));
        assert!(matches!(
            adapter.output_names(),
            Err(BackendError::ModelNotLoaded)
        ));
    }

    #[test]
    fn run_inference_rejects_tensor_level_dispatch() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::default());
        let err = adapter
            .run_inference(HashMap::new())
            .expect_err("tensor dispatch should error");
        match err {
            BackendError::RuntimeError(msg) => {
                assert!(msg.contains("LlmBackend::generate"), "got: {msg}");
            }
            other => panic!("expected RuntimeError, got {other:?}"),
        }
    }

    /// The adapter is held inside `Box<dyn LlmBackend>` (Send+Sync) by the
    /// runtime selector. Statically assert the bound holds for the
    /// concrete type so a future field addition that breaks `Send` is
    /// caught at compile time, not at the call site.
    #[test]
    fn adapter_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MlxLlmAdapter>();
    }

    #[test]
    fn supported_formats_lists_mlx_and_safetensors() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::default());
        let formats = LlmBackend::supported_formats(&adapter);
        assert!(formats.contains(&"mlx"));
        assert!(formats.contains(&"safetensors"));
    }

    #[test]
    fn name_is_mlx() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::default());
        assert_eq!(LlmBackend::name(&adapter), "mlx");
    }

    #[test]
    fn runtime_type_is_mlx() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::default());
        assert_eq!(adapter.runtime_type(), RuntimeType::Mlx);
    }

    #[test]
    fn context_length_reflects_config() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::new(8192));
        assert_eq!(adapter.context_length(), Some(8192));
    }
}

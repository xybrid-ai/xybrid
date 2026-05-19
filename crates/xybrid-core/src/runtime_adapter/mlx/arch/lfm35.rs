//! LFM 2 / 2.5 / 3.5 (Liquid Foundation Model) architecture builder.
//!
//! Ports the MLX-LM Python reference's LFM 2 / 3 / 3.5 family to Rust +
//! the safe tensor ops from `xybrid_mlx::ops`. LFM is a **hybrid**
//! decoder: each block picks between two operator types in an
//! interleaved schedule:
//!
//! 1. **Short-convolution block** (hyena-style). A small 1D causal
//!    convolution along the sequence axis, surrounded by a fused
//!    `in_proj` / `out_proj` pair. The conv kernel is short (default
//!    `conv_L_cache = 3`), which makes the op asymptotically linear in
//!    sequence length — LFM's headline feature over plain
//!    transformers.
//! 2. **Full-attention block** (standard transformer). Grouped-Query
//!    Attention with per-head RMSNorm on Q and K (the same trick Qwen 3
//!    uses), RoPE, and fused SDPA.
//!
//! The block type for layer `l` is read from the config:
//! - `layer_types[l] in {"conv", "full_attention"}` — explicit list, or
//! - `full_attn_idxs: [i32, ...]` — indices of attention layers
//!   (everything else is conv).
//!
//! The FFN is standard SwiGLU for both block types; the only structural
//! difference is the token-mixing operator that sits between
//! `operator_norm` and `ffn_norm`.
//!
//! The builder lands in two layers, mirroring Qwen 3 / Gemma 4:
//!
//! 1. **Non-linking validation** (always compiled under `llm-mlx`) — parses
//!    `config.json`, enumerates the expected safetensors weight-key
//!    schedule per block type, and validates the safetensors header
//!    before we commit to linking Metal.
//! 2. **Runtime** (gated on `llm-mlx-runtime` + Apple Silicon macOS) — weight
//!    materialisation, attention K/V append/read, and short-conv recurrent
//!    state for incremental decode.
//!
//! Reference (upstream): <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2.py>
//! (LFM 3 / 3.5 inherit LFM 2's hybrid conv+attention topology on the
//! MLX-LM side; the family ID is `"lfm2"`, `"lfm"` or `"lfm3"` in the config
//! header per this PRD).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};
use super::super::weights::SafeTensorBundle;

// =============================================================================
// Config
// =============================================================================

/// Per-layer operator kind. Derived from `layer_types` or
/// `full_attn_idxs` in `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Short-convolution block (hyena-style).
    Conv,
    /// Full attention block (GQA + per-head RMSNorm + RoPE + SDPA).
    FullAttention,
}

/// Full LFM 2 / 2.5 / 3.5 `config.json` subset the builder needs.
///
/// Fields mirror the HuggingFace config layout (`model_type = "lfm2"`,
/// `"lfm"`, or `"lfm3"`). Missing fields fall back to upstream defaults
/// where they exist.
#[derive(Debug, Clone, Deserialize)]
pub struct Lfm35Config {
    /// `"lfm2"`, `"lfm"` or `"lfm3"` — all route to this builder per the
    /// US-013 dispatcher.
    pub model_type: String,
    /// Model hidden / embedding dimension.
    pub hidden_size: usize,
    /// Number of transformer blocks (attention + conv combined).
    pub num_hidden_layers: usize,
    /// Number of query-heads per attention block.
    pub num_attention_heads: usize,
    /// Number of K/V heads per attention block. Defaults to
    /// [`Self::num_attention_heads`] when omitted.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// FFN intermediate (up-projection) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum position index the model was trained for. LFM defaults to
    /// a large window (128k).
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// RoPE base frequency. LFM 3 pins `1e6` like Qwen 3.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Epsilon added under the sqrt in RMSNorm. Upstream calls this
    /// `norm_eps` (not `rms_norm_eps`).
    #[serde(default = "default_norm_eps", alias = "rms_norm_eps")]
    pub norm_eps: f32,
    /// When `true`, the input embedding and the LM head share one
    /// weight tensor. LFM bundles nearly always set this.
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    /// Per-head dimension. Defaults to
    /// `hidden_size / num_attention_heads` when omitted.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Length of the causal conv cache (= kernel size) used by the
    /// conv blocks. Default `3` matches upstream.
    #[serde(default = "default_conv_l_cache", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    /// Whether the conv kernel has a bias term. Default `false`.
    #[serde(default)]
    pub conv_bias: bool,
    /// Explicit per-layer operator kind list. When present, takes
    /// precedence over [`Self::full_attn_idxs`]. Length must match
    /// [`Self::num_hidden_layers`].
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Alternative spelling: indices of the attention layers; every
    /// other index is a conv layer. Used when `layer_types` is absent.
    #[serde(default)]
    pub full_attn_idxs: Option<Vec<usize>>,
    /// Set when weights are quantized. MLX 4-bit bundles report
    /// `{ bits: 4, group_size: 64 }` here.
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

fn default_max_position_embeddings() -> usize {
    128_000
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_conv_l_cache() -> usize {
    3
}

/// Quantization metadata block from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    pub bits: u32,
    pub group_size: u32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

fn default_quant_mode() -> String {
    "affine".to_string()
}

impl Lfm35Config {
    /// Parse `{model_dir}/config.json` as an LFM family config.
    pub fn from_model_dir(model_dir: &Path) -> MlxLlmResult<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path)?;
        let mut value: serde_json::Value = serde_json::from_str(&raw)?;
        normalize_config_aliases(&mut value);
        let cfg: Lfm35Config = serde_json::from_value(value)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> MlxLlmResult<()> {
        if !matches!(self.model_type.as_str(), "lfm2" | "lfm" | "lfm3") {
            return Err(MlxLlmError::UnsupportedArchitecture {
                model_type: self.model_type.clone(),
            });
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
        {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm config has zero-valued dimensions (hidden={}, layers={}, heads={}, ffn={}, vocab={})",
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.intermediate_size,
                self.vocab_size,
            )));
        }
        if self.head_dim.is_none() && !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm config hidden_size={} is not a multiple of num_attention_heads={}",
                self.hidden_size, self.num_attention_heads,
            )));
        }
        let head_dim = self.head_dim();
        let kv_heads = self.kv_heads();
        if head_dim == 0 || kv_heads == 0 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm config has invalid derived dimensions (head_dim={head_dim}, kv_heads={kv_heads})"
            )));
        }
        if !self.num_attention_heads.is_multiple_of(kv_heads) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm config num_attention_heads={} is not a multiple of num_key_value_heads={kv_heads}",
                self.num_attention_heads,
            )));
        }
        super::validate_i32_dimensions(
            "lfm",
            &[
                ("hidden_size", self.hidden_size),
                ("num_hidden_layers", self.num_hidden_layers),
                ("num_attention_heads", self.num_attention_heads),
                ("num_key_value_heads", kv_heads),
                ("intermediate_size", self.intermediate_size),
                ("vocab_size", self.vocab_size),
                ("max_position_embeddings", self.max_position_embeddings),
                ("head_dim", head_dim),
                ("conv_l_cache", self.conv_l_cache),
            ],
        )?;
        super::validate_i32_product(
            "lfm",
            "attention_projection_width",
            &[self.num_attention_heads, head_dim],
        )?;
        super::validate_i32_product("lfm", "kv_projection_width", &[kv_heads, head_dim])?;
        super::validate_i32_product("lfm", "conv_projection_width", &[3, self.hidden_size])?;
        if self.conv_l_cache == 0 {
            return Err(MlxLlmError::ConfigInvalid(
                "lfm config has conv_l_cache = 0 (must be >= 1)".into(),
            ));
        }
        if self.conv_bias {
            return Err(MlxLlmError::ConfigInvalid(
                "lfm conv_bias=true is not supported by the MLX runtime path yet".into(),
            ));
        }
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers {
                return Err(MlxLlmError::ConfigInvalid(format!(
                    "lfm layer_types length {} != num_hidden_layers {}",
                    types.len(),
                    self.num_hidden_layers
                )));
            }
            for (i, t) in types.iter().enumerate() {
                if !matches!(t.as_str(), "conv" | "full_attention") {
                    return Err(MlxLlmError::ConfigInvalid(format!(
                        "lfm layer_types[{i}] = `{t}` (expected `conv` or `full_attention`)"
                    )));
                }
            }
        }
        if self.layer_types.is_none() && self.full_attn_idxs.is_none() {
            return Err(MlxLlmError::ConfigInvalid(
                "lfm config must provide either `layer_types` or `full_attn_idxs`".into(),
            ));
        }
        if let Some(idxs) = &self.full_attn_idxs {
            for &i in idxs {
                if i >= self.num_hidden_layers {
                    return Err(MlxLlmError::ConfigInvalid(format!(
                        "lfm full_attn_idxs[{i}] >= num_hidden_layers {}",
                        self.num_hidden_layers
                    )));
                }
            }
        }
        Ok(())
    }

    /// Effective per-head dimension, falling back to
    /// `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Effective K/V head count (GQA-aware).
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Resolve the operator kind for every layer. Prefers
    /// `layer_types` when present, falls back to `full_attn_idxs`
    /// (which gates attention layers by index — the rest are conv).
    pub fn layer_kinds(&self) -> Vec<LayerKind> {
        if let Some(types) = &self.layer_types {
            return types
                .iter()
                .map(|t| match t.as_str() {
                    "full_attention" => LayerKind::FullAttention,
                    _ => LayerKind::Conv,
                })
                .collect();
        }
        let attn: HashSet<usize> = self
            .full_attn_idxs
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();
        (0..self.num_hidden_layers)
            .map(|l| {
                if attn.contains(&l) {
                    LayerKind::FullAttention
                } else {
                    LayerKind::Conv
                }
            })
            .collect()
    }

    /// `true` when weights are stored in a quantized format.
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }
}

fn normalize_config_aliases(value: &mut serde_json::Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };

    if !obj.contains_key("intermediate_size") {
        if let Some(block_ff_dim) = obj.get("block_ff_dim").cloned() {
            obj.insert("intermediate_size".into(), block_ff_dim);
        }
    }
}

// =============================================================================
// Expected safetensors schedule
// =============================================================================

/// Safetensors key names the LFM family forward pass reads, in the order
/// the upstream MLX-LM reference emits them.
///
/// Per-layer block (variable, depending on [`LayerKind`]):
///
/// **Full-attention** (11 tensors):
/// - `operator_norm.weight`
/// - `self_attn.{q,k,v,out}_proj.weight` (4)
/// - `self_attn.{q,k}_layernorm.weight` — per-head RMSNorm (2)
/// - `ffn_norm.weight`
/// - `feed_forward.{w1,w2,w3}.weight` (3)
///
/// **Conv** (8 tensors):
/// - `operator_norm.weight`
/// - `conv.in_proj.weight`
/// - `conv.conv.weight`
/// - `conv.out_proj.weight`
/// - `ffn_norm.weight`
/// - `feed_forward.{w1,w2,w3}.weight` (3)
///
/// Shared across blocks (2 or 3):
/// - `model.embed_tokens.weight`
/// - `model.embedding_norm.weight`
/// - `lm_head.weight` — only when
///   [`Lfm35Config::tie_word_embeddings`] is `false`.
pub fn expected_weight_keys(cfg: &Lfm35Config) -> Vec<String> {
    let kinds = cfg.layer_kinds();
    let mut keys = Vec::with_capacity(2 + 10 * cfg.num_hidden_layers);
    push_quantized_weight_key(&mut keys, "model.embed_tokens.weight", cfg.is_quantized());
    for (l, kind) in kinds.iter().enumerate() {
        let base = format!("model.layers.{l}");
        keys.push(format!("{base}.operator_norm.weight"));
        match kind {
            LayerKind::FullAttention => {
                push_quantized_weight_key(
                    &mut keys,
                    &format!("{base}.self_attn.q_proj.weight"),
                    cfg.is_quantized(),
                );
                push_quantized_weight_key(
                    &mut keys,
                    &format!("{base}.self_attn.k_proj.weight"),
                    cfg.is_quantized(),
                );
                push_quantized_weight_key(
                    &mut keys,
                    &format!("{base}.self_attn.v_proj.weight"),
                    cfg.is_quantized(),
                );
                push_quantized_weight_key(
                    &mut keys,
                    &format!("{base}.self_attn.out_proj.weight"),
                    cfg.is_quantized(),
                );
                keys.push(format!("{base}.self_attn.q_layernorm.weight"));
                keys.push(format!("{base}.self_attn.k_layernorm.weight"));
            }
            LayerKind::Conv => {
                push_quantized_weight_key(
                    &mut keys,
                    &format!("{base}.conv.in_proj.weight"),
                    cfg.is_quantized(),
                );
                keys.push(format!("{base}.conv.conv.weight"));
                push_quantized_weight_key(
                    &mut keys,
                    &format!("{base}.conv.out_proj.weight"),
                    cfg.is_quantized(),
                );
            }
        }
        keys.push(format!("{base}.ffn_norm.weight"));
        push_quantized_weight_key(
            &mut keys,
            &format!("{base}.feed_forward.w1.weight"),
            cfg.is_quantized(),
        );
        push_quantized_weight_key(
            &mut keys,
            &format!("{base}.feed_forward.w2.weight"),
            cfg.is_quantized(),
        );
        push_quantized_weight_key(
            &mut keys,
            &format!("{base}.feed_forward.w3.weight"),
            cfg.is_quantized(),
        );
    }
    keys.push("model.embedding_norm.weight".to_string());
    if !cfg.tie_word_embeddings {
        push_quantized_weight_key(&mut keys, "lm_head.weight", cfg.is_quantized());
    }
    keys
}

fn push_quantized_weight_key(keys: &mut Vec<String>, weight_key: &str, quantized: bool) {
    keys.push(weight_key.to_string());
    if quantized {
        if let Some(base) = weight_key.strip_suffix(".weight") {
            keys.push(format!("{base}.scales"));
            keys.push(format!("{base}.biases"));
        }
    }
}

// =============================================================================
// Safetensors validation
// =============================================================================

/// Validate that `model.safetensors` contains every key the LFM family
/// runtime reads.
///
/// Mirrors [`super::qwen35::validate_safetensors`] — quantized bundles
/// bail here with a pointed error so the runtime selector (US-016) can
/// fall back to llama.cpp.
pub fn validate_safetensors(path: &Path, cfg: &Lfm35Config) -> MlxLlmResult<()> {
    let weights = SafeTensorBundle::from_single_file(path.to_path_buf());
    validate_safetensors_bundle(&weights, cfg)
}

pub fn validate_safetensors_bundle(
    weights: &SafeTensorBundle,
    cfg: &Lfm35Config,
) -> MlxLlmResult<()> {
    let names = weights.tensor_names()?;

    let expected = expected_weight_keys(cfg);
    let missing: Vec<String> = expected
        .iter()
        .filter(|k| !names.contains(k.as_str()))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(MlxLlmError::WeightLoad {
            path: weights.path_for_error(),
            reason: format!(
                "missing {} required tensor(s); first few = {:?}",
                missing.len(),
                missing.iter().take(5).collect::<Vec<_>>()
            ),
        });
    }
    Ok(())
}

/// Top-level entry: load and validate an LFM family bundle's config +
/// weight manifest. Returns the parsed config so the non-linking load path
/// in [`super::super::model::MlxLlmAdapter::load`] can cache it.
///
/// Does NOT touch MLX / Metal — that happens in `runtime::build` under
/// the `llm-mlx-runtime` feature.
pub fn load(model_dir: &Path, weights: &SafeTensorBundle) -> MlxLlmResult<Lfm35Config> {
    let cfg = Lfm35Config::from_model_dir(model_dir)?;
    validate_safetensors_bundle(weights, &cfg)?;
    Ok(cfg)
}

/// Resolve the legacy single-file safetensors path for callers that need
/// that exact file. Normal LFM validation/runtime loading goes through
/// [`SafeTensorBundle`] and accepts either `model.safetensors` or a
/// `model.safetensors.index.json` shard manifest.
pub fn resolve_weights_path(model_dir: &Path) -> MlxLlmResult<PathBuf> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(single);
    }
    Err(MlxLlmError::MissingFile {
        file: "model.safetensors",
        dir: model_dir.to_path_buf(),
    })
}

// =============================================================================
// Runtime (llm-mlx-runtime + Apple Silicon macOS only)
// =============================================================================

/// MLX-backed forward pass — only compiled when the xcframework can be
/// linked (`llm-mlx-runtime` feature on Apple Silicon macOS).
#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod runtime {
    use super::super::super::model::{MlxLlmError, MlxLlmResult};
    use super::super::super::weights::SafeTensorBundle;
    use super::super::linear::{load_dense_or_dequantized, LinearQuantization, LinearWeight};
    use super::super::{shape_dim_i32, shape_dim_usize, shape_product_i32, shape_product_usize};
    use super::{LayerKind, Lfm35Config};

    use xybrid_mlx::ops::{
        add, concat, conv1d, gather, mul, reshape, rms_norm, rope, scaled_dot_product_attention,
        silu, slice, slice_update, transpose,
    };
    use xybrid_mlx::{MlxArray, MlxDtype, MlxStream};

    const KV_CACHE_STEP: usize = 256;

    /// One full-attention block's weights. Layout mirrors Qwen 3 with
    /// the LFM-specific field names (`out_proj` / `{q,k}_layernorm`).
    #[derive(Debug)]
    pub struct AttentionLayer {
        pub operator_norm: MlxArray,
        pub q_proj: LinearWeight,
        pub k_proj: LinearWeight,
        pub v_proj: LinearWeight,
        pub out_proj: LinearWeight,
        pub q_layernorm: MlxArray,
        pub k_layernorm: MlxArray,
        pub ffn_norm: MlxArray,
        pub w1: LinearWeight,
        pub w2: LinearWeight,
        pub w3: LinearWeight,
    }

    /// One short-conv block's weights. `conv.conv` is the actual 1D
    /// causal kernel; `in_proj` / `out_proj` wrap it with dense
    /// projections.
    #[derive(Debug)]
    pub struct ConvLayer {
        pub operator_norm: MlxArray,
        pub conv_in_proj: LinearWeight,
        pub conv_kernel: MlxArray,
        pub conv_out_proj: LinearWeight,
        pub ffn_norm: MlxArray,
        pub w1: LinearWeight,
        pub w2: LinearWeight,
        pub w3: LinearWeight,
    }

    /// One LFM layer — either [`AttentionLayer`] or [`ConvLayer`].
    #[derive(Debug)]
    pub enum Lfm35Layer {
        Attention(AttentionLayer),
        Conv(ConvLayer),
    }

    /// Resident per-layer decode state for a single generation call.
    #[derive(Debug)]
    pub enum Lfm35LayerCache {
        Attention(AttentionCache),
        Conv(ConvCache),
    }

    impl Lfm35LayerCache {
        fn reset(&mut self) {
            match self {
                Self::Attention(cache) => cache.reset(),
                Self::Conv(cache) => cache.reset(),
            }
        }
    }

    /// K/V cache for a full-attention LFM layer.
    #[derive(Debug, Default)]
    pub struct AttentionCache {
        keys: Option<CachedAxis2>,
        values: Option<CachedAxis2>,
    }

    impl AttentionCache {
        fn reset(&mut self) {
            self.keys = None;
            self.values = None;
        }
    }

    #[derive(Debug)]
    struct CachedAxis2 {
        storage: MlxArray,
        len: usize,
    }

    /// Recurrent short-conv state. Stores the previous
    /// `conv_L_cache - 1` `B * x` activations in channel-last layout.
    #[derive(Debug, Default)]
    pub struct ConvCache {
        state: Option<MlxArray>,
    }

    impl ConvCache {
        fn reset(&mut self) {
            self.state = None;
        }
    }

    /// Full LFM family weight set resident in MLX memory.
    #[derive(Debug)]
    pub struct Lfm35Weights {
        pub embed_tokens: MlxArray,
        pub embed_tokens_linear: LinearWeight,
        pub layers: Vec<Lfm35Layer>,
        pub embedding_norm: MlxArray,
        /// `None` when [`Lfm35Config::tie_word_embeddings`] is set.
        pub lm_head: Option<LinearWeight>,
        layer_cache: Vec<Lfm35LayerCache>,
    }

    impl Lfm35Weights {
        /// Clear per-generation attention and conv state while keeping
        /// resident weights.
        pub fn reset_kv_cache(&mut self) {
            for cache in &mut self.layer_cache {
                cache.reset();
            }
        }
    }

    /// Build [`Lfm35Weights`] from `model.safetensors`.
    ///
    /// Supports F32, F16, and BF16 weights. Quantized bundles are
    /// rejected up-front in [`super::validate_safetensors`].
    pub fn build(cfg: &Lfm35Config, weights: &SafeTensorBundle) -> MlxLlmResult<Lfm35Weights> {
        let quant = quantization(cfg)?;
        let quant = quant.as_ref();

        let embed_tokens = load_dense_or_dequantized(weights, "model.embed_tokens.weight", quant)?;
        let embed_tokens_linear = if cfg.is_quantized() {
            LinearWeight::load(weights, "model.embed_tokens.weight", quant)?
        } else {
            LinearWeight::dense(embed_tokens.clone())?
        };
        let kinds = cfg.layer_kinds();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (l, kind) in kinds.iter().enumerate() {
            let base = format!("model.layers.{l}");
            let layer = match kind {
                LayerKind::FullAttention => Lfm35Layer::Attention(AttentionLayer {
                    operator_norm: load_tensor(weights, &format!("{base}.operator_norm.weight"))?,
                    q_proj: LinearWeight::load(
                        weights,
                        &format!("{base}.self_attn.q_proj.weight"),
                        quant,
                    )?,
                    k_proj: LinearWeight::load(
                        weights,
                        &format!("{base}.self_attn.k_proj.weight"),
                        quant,
                    )?,
                    v_proj: LinearWeight::load(
                        weights,
                        &format!("{base}.self_attn.v_proj.weight"),
                        quant,
                    )?,
                    out_proj: LinearWeight::load(
                        weights,
                        &format!("{base}.self_attn.out_proj.weight"),
                        quant,
                    )?,
                    q_layernorm: load_tensor(
                        weights,
                        &format!("{base}.self_attn.q_layernorm.weight"),
                    )?,
                    k_layernorm: load_tensor(
                        weights,
                        &format!("{base}.self_attn.k_layernorm.weight"),
                    )?,
                    ffn_norm: load_tensor(weights, &format!("{base}.ffn_norm.weight"))?,
                    w1: LinearWeight::load(
                        weights,
                        &format!("{base}.feed_forward.w1.weight"),
                        quant,
                    )?,
                    w2: LinearWeight::load(
                        weights,
                        &format!("{base}.feed_forward.w2.weight"),
                        quant,
                    )?,
                    w3: LinearWeight::load(
                        weights,
                        &format!("{base}.feed_forward.w3.weight"),
                        quant,
                    )?,
                }),
                LayerKind::Conv => Lfm35Layer::Conv(ConvLayer {
                    operator_norm: load_tensor(weights, &format!("{base}.operator_norm.weight"))?,
                    conv_in_proj: LinearWeight::load(
                        weights,
                        &format!("{base}.conv.in_proj.weight"),
                        quant,
                    )?,
                    conv_kernel: load_tensor(weights, &format!("{base}.conv.conv.weight"))?,
                    conv_out_proj: LinearWeight::load(
                        weights,
                        &format!("{base}.conv.out_proj.weight"),
                        quant,
                    )?,
                    ffn_norm: load_tensor(weights, &format!("{base}.ffn_norm.weight"))?,
                    w1: LinearWeight::load(
                        weights,
                        &format!("{base}.feed_forward.w1.weight"),
                        quant,
                    )?,
                    w2: LinearWeight::load(
                        weights,
                        &format!("{base}.feed_forward.w2.weight"),
                        quant,
                    )?,
                    w3: LinearWeight::load(
                        weights,
                        &format!("{base}.feed_forward.w3.weight"),
                        quant,
                    )?,
                }),
            };
            layers.push(layer);
        }
        let embedding_norm = load_tensor(weights, "model.embedding_norm.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(LinearWeight::load(weights, "lm_head.weight", quant)?)
        };

        Ok(Lfm35Weights {
            embed_tokens,
            embed_tokens_linear,
            layers,
            embedding_norm,
            lm_head,
            layer_cache: kinds
                .iter()
                .map(|kind| match kind {
                    LayerKind::FullAttention => {
                        Lfm35LayerCache::Attention(AttentionCache::default())
                    }
                    LayerKind::Conv => Lfm35LayerCache::Conv(ConvCache::default()),
                })
                .collect(),
        })
    }

    /// Read one tensor from a SafeTensors view into an [`MlxArray`].
    fn load_tensor(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<MlxArray> {
        weights.read_array(name)
    }

    fn quantization(cfg: &Lfm35Config) -> MlxLlmResult<Option<LinearQuantization>> {
        cfg.quantization
            .as_ref()
            .map(|q| LinearQuantization::new(q.bits, q.group_size, &q.mode))
            .transpose()
    }

    /// Forward pass through the whole stack.
    ///
    /// `input_ids` shape: `[batch, seq_len]` (i64/i32). Returns logits of
    /// shape `[batch, seq_len, vocab_size]` (f32). Prefill seeds attention
    /// K/V and short-conv state; decode forwards one token with a non-zero
    /// `position_offset` and reads the cached prefix/state.
    pub fn forward(
        cfg: &Lfm35Config,
        weights: &mut Lfm35Weights,
        input_ids: &MlxArray,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        // Embedding lookup: [B, T] -> [B, T, H].
        let mut hidden = gather(&weights.embed_tokens, input_ids, 0, stream)?;

        let head_dim = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.kv_heads();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let layers = &weights.layers;
        let layer_cache = &mut weights.layer_cache;
        for (layer, cache) in layers.iter().zip(layer_cache.iter_mut()) {
            hidden = decoder_block(
                cfg,
                layer,
                cache,
                &hidden,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                position_offset,
                stream,
            )?;
        }

        let final_norm = rms_norm(&hidden, Some(&weights.embedding_norm), cfg.norm_eps, stream)?;
        let lm_head = weights
            .lm_head
            .as_ref()
            .unwrap_or(&weights.embed_tokens_linear);
        lm_head.forward(&final_norm, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn decoder_block(
        cfg: &Lfm35Config,
        layer: &Lfm35Layer,
        cache: &mut Lfm35LayerCache,
        hidden: &MlxArray,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let mixed = match (layer, cache) {
            (Lfm35Layer::Attention(layer), Lfm35LayerCache::Attention(cache)) => {
                let normed = rms_norm(hidden, Some(&layer.operator_norm), cfg.norm_eps, stream)?;
                attention_block(
                    cfg,
                    layer,
                    cache,
                    &normed,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale,
                    position_offset,
                    stream,
                )?
            }
            (Lfm35Layer::Conv(layer), Lfm35LayerCache::Conv(cache)) => {
                let normed = rms_norm(hidden, Some(&layer.operator_norm), cfg.norm_eps, stream)?;
                conv_block(cfg, layer, cache, &normed, stream)?
            }
            _ => {
                return Err(MlxLlmError::ConfigInvalid(
                    "LFM layer/cache kind mismatch".into(),
                ));
            }
        };
        let hidden = add(hidden, &mixed, stream)?;

        let ffn_norm = match layer {
            Lfm35Layer::Attention(layer) => &layer.ffn_norm,
            Lfm35Layer::Conv(layer) => &layer.ffn_norm,
        };
        let normed = rms_norm(&hidden, Some(ffn_norm), cfg.norm_eps, stream)?;
        let ffn = feed_forward(layer, &normed, stream)?;
        add(&hidden, &ffn, stream).map_err(Into::into)
    }

    fn feed_forward(
        layer: &Lfm35Layer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let (w1, w2, w3) = match layer {
            Lfm35Layer::Attention(layer) => (&layer.w1, &layer.w2, &layer.w3),
            Lfm35Layer::Conv(layer) => (&layer.w1, &layer.w2, &layer.w3),
        };
        let gate = w1.forward(hidden, stream)?;
        let up = w3.forward(hidden, stream)?;
        let gated = mul(&silu(&gate, stream)?, &up, stream)?;
        w2.forward(&gated, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_block(
        cfg: &Lfm35Config,
        layer: &AttentionLayer,
        cache: &mut AttentionCache,
        hidden: &MlxArray,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let q = layer.q_proj.forward(hidden, stream)?;
        let k = layer.k_proj.forward(hidden, stream)?;
        let v = layer.v_proj.forward(hidden, stream)?;

        let q = split_heads(&q, n_heads, head_dim, stream)?;
        let k = split_heads(&k, n_kv_heads, head_dim, stream)?;
        let v = split_heads(&v, n_kv_heads, head_dim, stream)?;

        let q = rms_norm(&q, Some(&layer.q_layernorm), cfg.norm_eps, stream)?;
        let k = rms_norm(&k, Some(&layer.k_layernorm), cfg.norm_eps, stream)?;
        let head_dim_i32 = shape_dim_i32("lfm", "head_dim", head_dim)?;

        let q = rope(
            &q,
            head_dim_i32,
            false,
            Some(cfg.rope_theta),
            1.0,
            position_offset,
            None,
            stream,
        )?;
        let k = rope(
            &k,
            head_dim_i32,
            false,
            Some(cfg.rope_theta),
            1.0,
            position_offset,
            None,
            stream,
        )?;

        let had_cached_prefix = cache.keys.is_some();
        let k = append_cached_axis2(&mut cache.keys, &k, stream)?;
        let v = append_cached_axis2(&mut cache.values, &v, stream)?;
        let q_len = q.shape().get(2).copied().unwrap_or(1);
        let causal = !had_cached_prefix && q_len > 1;

        let attn = scaled_dot_product_attention(&q, &k, &v, scale, causal, None, stream)?;
        let attn = merge_heads(&attn, n_heads, head_dim, stream)?;
        layer.out_proj.forward(&attn, stream)
    }

    fn append_cached_axis2(
        slot: &mut Option<CachedAxis2>,
        new_slice: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = new_slice.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm K/V cache slice must be rank-4, got shape {shape:?}"
            )));
        }
        debug_assert_eq!(
            new_slice.dtype().ok(),
            Some(MlxDtype::F32),
            "lfm K/V cache storage is f32; quantized LFM projections currently produce f32"
        );
        let b = shape[0];
        let h = shape[1];
        let step_len = shape_dim_usize("lfm", "kv_cache_step_len", shape[2])?;
        let d = shape[3];
        let old_len = slot.as_ref().map_or(0, |cache| cache.len);
        let new_len = old_len
            .checked_add(step_len)
            .ok_or_else(|| MlxLlmError::ConfigInvalid("lfm K/V cache length overflow".into()))?;

        let mut storage = match slot.take() {
            Some(cache) if new_len <= cache_capacity(&cache.storage)? => cache.storage,
            Some(cache) => grow_cached_axis2(&cache, b, h, d, new_len, stream)?,
            None => allocate_cached_axis2(b, h, d, new_len)?,
        };

        storage = slice_update(
            &storage,
            new_slice,
            &[0, 0, shape_dim_i32("lfm", "kv_cache_old_len", old_len)?, 0],
            &[b, h, shape_dim_i32("lfm", "kv_cache_new_len", new_len)?, d],
            &[1, 1, 1, 1],
            stream,
        )?;

        let visible = slice(
            &storage,
            &[0, 0, 0, 0],
            &[
                b,
                h,
                shape_dim_i32("lfm", "kv_cache_visible_len", new_len)?,
                d,
            ],
            &[1, 1, 1, 1],
            stream,
        )?;
        *slot = Some(CachedAxis2 {
            storage,
            len: new_len,
        });
        Ok(visible)
    }

    fn cache_capacity(cache: &MlxArray) -> MlxLlmResult<usize> {
        let shape = cache.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm K/V cache storage must be rank-4, got shape {shape:?}"
            )));
        }
        shape_dim_usize("lfm", "kv_cache_capacity", shape[2])
    }

    fn grow_cached_axis2(
        cache: &CachedAxis2,
        b: i32,
        h: i32,
        d: i32,
        required_len: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let grown = allocate_cached_axis2(b, h, d, required_len)?;
        let visible = slice(
            &cache.storage,
            &[0, 0, 0, 0],
            &[
                b,
                h,
                shape_dim_i32("lfm", "kv_cache_copy_len", cache.len)?,
                d,
            ],
            &[1, 1, 1, 1],
            stream,
        )?;
        slice_update(
            &grown,
            &visible,
            &[0, 0, 0, 0],
            &[
                b,
                h,
                shape_dim_i32("lfm", "kv_cache_copy_len", cache.len)?,
                d,
            ],
            &[1, 1, 1, 1],
            stream,
        )
        .map_err(Into::into)
    }

    fn allocate_cached_axis2(
        b: i32,
        h: i32,
        d: i32,
        required_len: usize,
    ) -> MlxLlmResult<MlxArray> {
        let capacity = required_len
            .max(1)
            .div_ceil(KV_CACHE_STEP)
            .checked_mul(KV_CACHE_STEP)
            .ok_or_else(|| MlxLlmError::ConfigInvalid("lfm K/V cache capacity overflow".into()))?;
        let b_usize = shape_dim_usize("lfm", "kv_cache_batch", b)?;
        let h_usize = shape_dim_usize("lfm", "kv_cache_heads", h)?;
        let d_usize = shape_dim_usize("lfm", "kv_cache_head_dim", d)?;
        let zeros_len = shape_product_usize(
            "lfm",
            "kv_cache_element_count",
            &[b_usize, h_usize, capacity, d_usize],
        )?;
        let zeros = vec![0.0f32; zeros_len];
        MlxArray::from_slice_f32(
            &zeros,
            &[
                b,
                h,
                shape_dim_i32("lfm", "kv_cache_capacity", capacity)?,
                d,
            ],
        )
        .map_err(Into::into)
    }

    fn conv_block(
        cfg: &Lfm35Config,
        layer: &ConvLayer,
        cache: &mut ConvCache,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let projected = layer.conv_in_proj.forward(hidden, stream)?;
        let b = take_hidden_range(&projected, 0, cfg.hidden_size, stream)?;
        let c = take_hidden_range(&projected, cfg.hidden_size, cfg.hidden_size, stream)?;
        let x = take_hidden_range(&projected, cfg.hidden_size * 2, cfg.hidden_size, stream)?;
        let bx = mul(&b, &x, stream)?;

        let padded = conv_input_with_cache(cache, &bx, cfg.conv_l_cache, stream)?;
        let conv = conv1d(
            &padded,
            &layer.conv_kernel,
            1,
            0,
            1,
            shape_dim_i32("lfm", "hidden_size", cfg.hidden_size)?,
            stream,
        )?;
        let y = mul(&c, &conv, stream)?;
        layer.conv_out_proj.forward(&y, stream)
    }

    fn conv_input_with_cache(
        cache: &mut ConvCache,
        bx: &MlxArray,
        conv_l_cache: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let n_keep = conv_l_cache.saturating_sub(1);
        let with_state = match cache.state.take() {
            Some(state) => concat(&[&state, bx], 1, stream)?,
            None => left_pad_sequence(bx, n_keep, stream)?,
        };

        cache.state = if n_keep == 0 {
            None
        } else {
            Some(take_last_sequence_tokens(&with_state, n_keep, stream)?)
        };

        Ok(with_state)
    }

    fn take_hidden_range(
        x: &MlxArray,
        start: usize,
        len: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let end = start.checked_add(len).ok_or_else(|| {
            MlxLlmError::ConfigInvalid("lfm hidden range end overflows usize".into())
        })?;
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm hidden range source must be rank-3, got shape {shape:?}"
            )));
        }
        let width = shape_dim_usize("lfm", "hidden_size", shape[2])?;
        if end > width {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm hidden range {start}..{end} exceeds width {width}"
            )));
        }
        slice(
            x,
            &[0, 0, shape_dim_i32("lfm", "hidden_range_start", start)?],
            &[
                shape[0],
                shape[1],
                shape_dim_i32("lfm", "hidden_range_end", end)?,
            ],
            &[1, 1, 1],
            stream,
        )
        .map_err(Into::into)
    }

    fn left_pad_sequence(
        x: &MlxArray,
        pad: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        if pad == 0 {
            return Ok(x.clone());
        }
        let shape = x.shape();
        let b = shape[0];
        let h = shape[2];
        let b_usize = shape_dim_usize("lfm", "batch", b)?;
        let h_usize = shape_dim_usize("lfm", "hidden_size", h)?;
        let zeros_len =
            shape_product_usize("lfm", "left_pad_element_count", &[b_usize, pad, h_usize])?;
        let zeros = vec![0.0f32; zeros_len];
        let pad_i32 = shape_dim_i32("lfm", "left_pad_len", pad)?;
        let pad_arr = MlxArray::from_slice_f32(&zeros, &[b, pad_i32, h])?;
        concat(&[&pad_arr, x], 1, stream).map_err(Into::into)
    }

    fn take_last_sequence_tokens(
        x: &MlxArray,
        len: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "lfm sequence slice source must be rank-3, got shape {shape:?}"
            )));
        }
        let seq_len = shape_dim_usize("lfm", "sequence_len", shape[1])?;
        let start = seq_len.saturating_sub(len);
        slice(
            x,
            &[0, shape_dim_i32("lfm", "sequence_start", start)?, 0],
            &[shape[0], shape[1], shape[2]],
            &[1, 1, 1],
            stream,
        )
        .map_err(Into::into)
    }

    /// Reshape `[B, T, heads * head_dim]` → `[B, heads, T, head_dim]`.
    fn split_heads(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = x.shape();
        let b = shape[0];
        let t = shape[1];
        let heads_i32 = shape_dim_i32("lfm", "num_attention_heads", heads)?;
        let head_dim_i32 = shape_dim_i32("lfm", "head_dim", head_dim)?;
        let reshaped = reshape(x, &[b, t, heads_i32, head_dim_i32], stream)?;
        transpose(&reshaped, &[0, 2, 1, 3], stream).map_err(Into::into)
    }

    /// Reshape `[B, heads, T, head_dim]` → `[B, T, heads * head_dim]`.
    fn merge_heads(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = x.shape();
        let b = shape[0];
        let t = shape[2];
        let transposed = transpose(x, &[0, 2, 1, 3], stream)?;
        let width = shape_product_i32("lfm", "merged_attention_width", &[heads, head_dim])?;
        reshape(&transposed, &[b, t, width], stream).map_err(Into::into)
    }

    #[cfg(test)]
    mod runtime_tests {
        use super::*;

        #[test]
        fn take_hidden_range_slices_contiguous_width() {
            let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
            let x = MlxArray::from_slice_f32(&data, &[1, 2, 6]).unwrap();

            let got = take_hidden_range(&x, 2, 3, None).unwrap();

            assert_eq!(got.shape(), vec![1, 2, 3]);
            assert_eq!(
                got.to_vec_f32().unwrap(),
                vec![2.0, 3.0, 4.0, 8.0, 9.0, 10.0]
            );
        }

        #[test]
        fn take_last_sequence_tokens_slices_tail() {
            let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
            let x = MlxArray::from_slice_f32(&data, &[1, 4, 3]).unwrap();

            let got = take_last_sequence_tokens(&x, 2, None).unwrap();

            assert_eq!(got.shape(), vec![1, 2, 3]);
            assert_eq!(
                got.to_vec_f32().unwrap(),
                vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
            );
        }

        #[test]
        fn append_cached_axis2_returns_visible_prefix() {
            let mut slot = None;
            let first = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
            let got = append_cached_axis2(&mut slot, &first, None).unwrap();

            assert_eq!(got.shape(), vec![1, 1, 2, 2]);
            assert_eq!(got.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(slot.as_ref().unwrap().len, 2);
            assert_eq!(
                cache_capacity(&slot.as_ref().unwrap().storage).unwrap(),
                256
            );

            let second = MlxArray::from_slice_f32(&[5.0, 6.0], &[1, 1, 1, 2]).unwrap();
            let got = append_cached_axis2(&mut slot, &second, None).unwrap();

            assert_eq!(got.shape(), vec![1, 1, 3, 2]);
            assert_eq!(
                got.to_vec_f32().unwrap(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            );
            assert_eq!(slot.as_ref().unwrap().len, 3);
            assert_eq!(
                cache_capacity(&slot.as_ref().unwrap().storage).unwrap(),
                256
            );
        }
    }
}

// =============================================================================
// Tests (non-linking — no bindings required)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config(n_layers: usize, tied: bool) -> Lfm35Config {
        Lfm35Config {
            model_type: "lfm3".into(),
            hidden_size: 16,
            num_hidden_layers: n_layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            intermediate_size: 32,
            vocab_size: 100,
            max_position_embeddings: 128_000,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
            tie_word_embeddings: tied,
            head_dim: Some(4),
            conv_l_cache: 3,
            conv_bias: false,
            // Alternate conv / attention — layer 0 is conv, layer 1 is
            // attention, so both branches of expected_weight_keys fire
            // in the same test.
            layer_types: Some(
                (0..n_layers)
                    .map(|l| {
                        if l.is_multiple_of(2) {
                            "conv".to_string()
                        } else {
                            "full_attention".to_string()
                        }
                    })
                    .collect(),
            ),
            full_attn_idxs: None,
            quantization: None,
        }
    }

    #[test]
    fn expected_keys_match_per_layer_schedule() {
        let cfg = dummy_config(2, false);
        let keys = expected_weight_keys(&cfg);
        // 1 (embed) + 1 conv layer * 8 + 1 attention layer * 11 + 2
        // (embedding_norm + lm_head) = 22.
        assert_eq!(keys.len(), 22);
        assert_eq!(keys.first().unwrap(), "model.embed_tokens.weight");
        // Conv layer 0 emits conv-specific keys.
        assert!(keys.iter().any(|k| k == "model.layers.0.conv.conv.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.conv.in_proj.weight"));
        // Attention layer 1 emits attention-specific keys.
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.1.self_attn.q_proj.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.1.self_attn.q_layernorm.weight"));
        // Both layers share operator_norm + ffn_norm + feed_forward.*.
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.operator_norm.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.1.feed_forward.w1.weight"));
        assert_eq!(keys.last().unwrap(), "lm_head.weight");
    }

    #[test]
    fn tied_embeddings_skip_lm_head() {
        let cfg = dummy_config(1, true);
        let keys = expected_weight_keys(&cfg);
        assert!(!keys.iter().any(|k| k == "lm_head.weight"));
        assert!(keys.iter().any(|k| k == "model.embedding_norm.weight"));
    }

    #[test]
    fn head_dim_falls_back_to_hidden_over_heads() {
        let mut cfg = dummy_config(1, false);
        cfg.hidden_size = 128;
        cfg.num_attention_heads = 8;
        cfg.head_dim = None;
        assert_eq!(cfg.head_dim(), 16);
    }

    #[test]
    fn kv_heads_default_to_num_attention_heads() {
        let mut cfg = dummy_config(1, false);
        cfg.num_key_value_heads = None;
        assert_eq!(cfg.kv_heads(), cfg.num_attention_heads);
    }

    #[test]
    fn layer_kinds_from_layer_types() {
        let cfg = dummy_config(4, true);
        let kinds = cfg.layer_kinds();
        // Pattern from `dummy_config`: even layers conv, odd layers attn.
        assert_eq!(
            kinds,
            vec![
                LayerKind::Conv,
                LayerKind::FullAttention,
                LayerKind::Conv,
                LayerKind::FullAttention,
            ]
        );
    }

    #[test]
    fn layer_kinds_from_full_attn_idxs() {
        let mut cfg = dummy_config(4, true);
        cfg.layer_types = None;
        cfg.full_attn_idxs = Some(vec![1, 3]);
        let kinds = cfg.layer_kinds();
        assert_eq!(
            kinds,
            vec![
                LayerKind::Conv,
                LayerKind::FullAttention,
                LayerKind::Conv,
                LayerKind::FullAttention,
            ]
        );
    }

    #[test]
    fn model_type_lfm_is_accepted() {
        let mut cfg = dummy_config(1, true);
        cfg.model_type = "lfm".into();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn model_type_lfm2_is_accepted() {
        let mut cfg = dummy_config(1, true);
        cfg.model_type = "lfm2".into();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn model_type_lfm3_is_accepted() {
        let cfg = dummy_config(1, true);
        assert_eq!(cfg.model_type, "lfm3");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn non_lfm_model_type_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.model_type = "llama".into();
        assert!(matches!(
            cfg.validate(),
            Err(MlxLlmError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn zero_dimensions_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.hidden_size = 0;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("zero-valued")),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn hidden_not_divisible_by_heads_rejected_when_head_dim_absent() {
        let mut cfg = dummy_config(1, true);
        cfg.head_dim = None;
        cfg.hidden_size = 17;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("multiple of")),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn oversized_dimensions_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.conv_l_cache = i32::MAX as usize + 1;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
                assert!(msg.contains("conv_l_cache"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn oversized_conv_projection_width_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.hidden_size = i32::MAX as usize / 3 + 1;
        cfg.num_attention_heads = 1;
        cfg.num_key_value_heads = Some(1);
        cfg.head_dim = Some(cfg.hidden_size);
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
                assert!(msg.contains("conv_projection_width"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn zero_conv_l_cache_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.conv_l_cache = 0;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("conv_l_cache"), "got: {msg}"),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn conv_bias_rejected_until_bias_wiring_lands() {
        let mut cfg = dummy_config(1, true);
        cfg.conv_bias = true;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("conv_bias"), "got: {msg}"),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn layer_types_length_mismatch_rejected() {
        let mut cfg = dummy_config(2, true);
        cfg.layer_types = Some(vec!["conv".into()]);
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("layer_types length"), "got: {msg}")
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn unknown_layer_type_rejected() {
        let mut cfg = dummy_config(2, true);
        cfg.layer_types = Some(vec!["conv".into(), "bogus".into()]);
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("bogus"), "got: {msg}"),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn missing_layer_schedule_rejected() {
        let mut cfg = dummy_config(2, true);
        cfg.layer_types = None;
        cfg.full_attn_idxs = None;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("layer_types"), "got: {msg}");
                assert!(msg.contains("full_attn_idxs"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_full_attn_idx_rejected() {
        let mut cfg = dummy_config(2, true);
        cfg.layer_types = None;
        cfg.full_attn_idxs = Some(vec![99]);
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("full_attn_idxs"), "got: {msg}")
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn quantized_manifest_requires_scales_and_biases() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let mut cfg = dummy_config(1, true);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
            mode: "affine".into(),
        });
        let tmp = tempfile::TempDir::new().unwrap();
        let mut keys = expected_weight_keys(&cfg);
        keys.retain(|k| !k.ends_with(".scales") && !k.ends_with(".biases"));
        let data = vec![0u8; 4];
        let tensors: Vec<(String, TensorView<'_>)> = keys
            .iter()
            .map(|k| {
                (
                    k.clone(),
                    TensorView::new(Dtype::F32, vec![1], &data).unwrap(),
                )
            })
            .collect();
        let path = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, &None, &path).unwrap();

        let err = validate_safetensors(&path, &cfg).unwrap_err();
        match err {
            MlxLlmError::WeightLoad { reason, .. } => {
                assert!(reason.contains("missing"), "got: {reason}");
                assert!(reason.contains(".scales"), "got: {reason}");
            }
            other => panic!("expected WeightLoad, got {other:?}"),
        }
    }

    #[test]
    fn resolve_weights_missing_returns_missing_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let err = resolve_weights_path(tmp.path()).unwrap_err();
        match err {
            MlxLlmError::MissingFile { file, .. } => assert_eq!(file, "model.safetensors"),
            other => panic!("expected MissingFile, got {other:?}"),
        }
    }

    #[test]
    fn validate_safetensors_reports_missing_keys() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = dummy_config(1, true);
        let data = vec![0u8; 4];
        let view = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let path = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file([("bogus".to_string(), view)], &None, &path).unwrap();

        let err = validate_safetensors(&path, &cfg).unwrap_err();
        match err {
            MlxLlmError::WeightLoad { reason, .. } => {
                assert!(reason.contains("missing"), "got: {reason}");
                assert!(reason.contains("required tensor"), "got: {reason}");
            }
            other => panic!("expected WeightLoad, got {other:?}"),
        }
    }

    #[test]
    fn validate_safetensors_accepts_full_manifest() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        // Mix of conv + attention layers so both per-layer schedules
        // are exercised by the round-trip.
        let cfg = dummy_config(2, true);
        let keys = expected_weight_keys(&cfg);
        let data = vec![0u8; 4];
        let tensors: Vec<(String, TensorView<'_>)> = keys
            .iter()
            .map(|k| {
                (
                    k.clone(),
                    TensorView::new(Dtype::F32, vec![1], &data).unwrap(),
                )
            })
            .collect();
        let path = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, &None, &path).unwrap();

        validate_safetensors(&path, &cfg).expect("full manifest should validate");
    }

    #[test]
    fn lfm35_quantized_manifest_accepts_projection_triplets() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        let mut cfg = dummy_config(2, true);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
            mode: "affine".into(),
        });
        let keys = expected_weight_keys(&cfg);
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.conv.in_proj.scales"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.1.self_attn.q_proj.biases"));
        assert!(keys.iter().any(|k| k == "model.embed_tokens.scales"));
        let data = vec![0u8; 4];
        let tensors: Vec<(String, TensorView<'_>)> = keys
            .iter()
            .map(|k| {
                (
                    k.clone(),
                    TensorView::new(Dtype::F32, vec![1], &data).unwrap(),
                )
            })
            .collect();
        let path = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, &None, &path).unwrap();

        validate_safetensors(&path, &cfg).expect("quantized manifest should validate");
    }

    #[test]
    fn norm_eps_alias_rms_norm_eps_parses() {
        // Upstream LFM uses `norm_eps`; consumers who paste Gemma / Qwen
        // configs sometimes write `rms_norm_eps`. The alias keeps both
        // working without touching downstream metadata.
        let raw = serde_json::json!({
            "model_type": "lfm3",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "vocab_size": 100,
            "rms_norm_eps": 2.5e-6,
            "layer_types": ["conv"]
        });
        let cfg: Lfm35Config = serde_json::from_str(&raw.to_string()).unwrap();
        assert!((cfg.norm_eps - 2.5e-6).abs() < f32::EPSILON);
    }

    #[test]
    fn model_dir_config_accepts_block_ff_dim_fallback() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::json!({
                "model_type": "lfm2",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "block_ff_dim": 48,
                "vocab_size": 100,
                "layer_types": ["conv"]
            })
            .to_string(),
        )
        .unwrap();

        let cfg = Lfm35Config::from_model_dir(tmp.path()).unwrap();
        assert_eq!(cfg.intermediate_size, 48);
    }

    #[test]
    fn model_dir_config_accepts_public_lfm25_duplicate_ffn_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::json!({
                "model_type": "lfm2",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "block_ff_dim": 48,
                "intermediate_size": 64,
                "vocab_size": 100,
                "layer_types": ["conv"]
            })
            .to_string(),
        )
        .unwrap();

        let cfg = Lfm35Config::from_model_dir(tmp.path()).unwrap();
        assert_eq!(cfg.intermediate_size, 64);
    }
}

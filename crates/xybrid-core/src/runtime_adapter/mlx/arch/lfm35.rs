//! LFM 3.5 (Liquid Foundation Model) architecture builder.
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
//! The builder lands in two layers, mirroring Qwen 3.5 / Gemma 4:
//!
//! 1. **Skeleton** (always compiled under `llm-mlx`) — parses
//!    `config.json`, enumerates the expected safetensors weight-key
//!    schedule per block type, and validates the safetensors header
//!    before we commit to linking Metal.
//! 2. **Runtime** (gated on `llm-mlx-runtime` + Apple target) — weight
//!    materialisation and the staged forward pass. The 1D causal conv
//!    primitive (`mlx_conv1d`) and the SwiGLU FFN activation (`silu`)
//!    are not yet wrapped in `xybrid_mlx::ops`, so the forward pass
//!    exercises the embedding + first-layer `operator_norm` and then
//!    bails with [`MlxLlmError::NotImplemented`] at the operator
//!    boundary — matching the deferral pattern Qwen 3.5 / Gemma 4 use
//!    for their activation gaps.
//!
//! Reference (upstream): <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2.py>
//! (LFM 3 / 3.5 inherit LFM 2's hybrid conv+attention topology on the
//! MLX-LM side; the family ID is `"lfm"` or `"lfm3"` in the config
//! header per this PRD).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::SafeTensors;
use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};

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

/// Full LFM 3.5 `config.json` subset the builder needs.
///
/// Fields mirror the HuggingFace config layout (`model_type = "lfm"` or
/// `"lfm3"`). Missing fields fall back to upstream defaults where they
/// exist.
#[derive(Debug, Clone, Deserialize)]
pub struct Lfm35Config {
    /// `"lfm"` or `"lfm3"` — both route to this builder per the
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
    #[serde(default = "default_conv_l_cache")]
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
}

impl Lfm35Config {
    /// Parse `{model_dir}/config.json` as an LFM 3.5 config.
    pub fn from_model_dir(model_dir: &Path) -> MlxLlmResult<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path)?;
        let cfg: Lfm35Config = serde_json::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> MlxLlmResult<()> {
        if !matches!(self.model_type.as_str(), "lfm" | "lfm3") {
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
        if self.conv_l_cache == 0 {
            return Err(MlxLlmError::ConfigInvalid(
                "lfm config has conv_l_cache = 0 (must be >= 1)".into(),
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

// =============================================================================
// Expected safetensors schedule
// =============================================================================

/// Safetensors key names the LFM 3.5 forward pass reads, in the order
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
    keys.push("model.embed_tokens.weight".to_string());
    for (l, kind) in kinds.iter().enumerate() {
        let base = format!("model.layers.{l}");
        keys.push(format!("{base}.operator_norm.weight"));
        match kind {
            LayerKind::FullAttention => {
                keys.push(format!("{base}.self_attn.q_proj.weight"));
                keys.push(format!("{base}.self_attn.k_proj.weight"));
                keys.push(format!("{base}.self_attn.v_proj.weight"));
                keys.push(format!("{base}.self_attn.out_proj.weight"));
                keys.push(format!("{base}.self_attn.q_layernorm.weight"));
                keys.push(format!("{base}.self_attn.k_layernorm.weight"));
            }
            LayerKind::Conv => {
                keys.push(format!("{base}.conv.in_proj.weight"));
                keys.push(format!("{base}.conv.conv.weight"));
                keys.push(format!("{base}.conv.out_proj.weight"));
            }
        }
        keys.push(format!("{base}.ffn_norm.weight"));
        keys.push(format!("{base}.feed_forward.w1.weight"));
        keys.push(format!("{base}.feed_forward.w2.weight"));
        keys.push(format!("{base}.feed_forward.w3.weight"));
    }
    keys.push("model.embedding_norm.weight".to_string());
    if !cfg.tie_word_embeddings {
        keys.push("lm_head.weight".to_string());
    }
    keys
}

// =============================================================================
// Safetensors validation
// =============================================================================

/// Validate that `model.safetensors` contains every key LFM 3.5's
/// forward pass reads.
///
/// Mirrors [`super::qwen35::validate_safetensors`] — quantized bundles
/// bail here with a pointed error so the runtime selector (US-016) can
/// fall back to llama.cpp.
pub fn validate_safetensors(path: &Path, cfg: &Lfm35Config) -> MlxLlmResult<()> {
    if cfg.is_quantized() {
        let bits = cfg.quantization.as_ref().map(|q| q.bits).unwrap_or(0);
        let gs = cfg.quantization.as_ref().map(|q| q.group_size).unwrap_or(0);
        return Err(MlxLlmError::UnsupportedArchitecture {
            model_type: format!(
                "{} (quantized {bits}-bit/group={gs} — mlx_fast_quantized_matmul lands in US-014)",
                cfg.model_type
            ),
        });
    }

    let bytes = fs::read(path).map_err(|e| MlxLlmError::WeightLoad {
        path: path.to_path_buf(),
        reason: format!("read safetensors: {e}"),
    })?;
    let (_, meta) = SafeTensors::read_metadata(&bytes).map_err(|e| MlxLlmError::WeightLoad {
        path: path.to_path_buf(),
        reason: format!("invalid safetensors header: {e}"),
    })?;
    let names: HashSet<String> = meta.tensors().into_keys().collect();

    let expected = expected_weight_keys(cfg);
    let missing: Vec<String> = expected
        .iter()
        .filter(|k| !names.contains(k.as_str()))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(MlxLlmError::WeightLoad {
            path: path.to_path_buf(),
            reason: format!(
                "missing {} required tensor(s); first few = {:?}",
                missing.len(),
                missing.iter().take(5).collect::<Vec<_>>()
            ),
        });
    }
    Ok(())
}

/// Top-level entry: load and validate an LFM 3.5 bundle's config +
/// weight manifest. Returns the parsed config so the skeleton load path
/// in [`super::super::model::MlxLlmAdapter::load`] can cache it.
///
/// Does NOT touch MLX / Metal — that happens in `runtime::build` under
/// the `llm-mlx-runtime` feature.
pub fn load(model_dir: &Path, weights_path: &Path) -> MlxLlmResult<Lfm35Config> {
    let cfg = Lfm35Config::from_model_dir(model_dir)?;
    validate_safetensors(weights_path, &cfg)?;
    Ok(cfg)
}

/// Resolve the safetensors weight file for a bundle. Sharded bundles
/// land with US-015 — this helper accepts the single-file layout only.
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
// Runtime (llm-mlx-runtime + Apple only)
// =============================================================================

/// MLX-backed forward pass — only compiled when the xcframework can be
/// linked (`llm-mlx-runtime` feature on an Apple target).
#[cfg(all(
    feature = "llm-mlx-runtime",
    any(target_os = "macos", target_os = "ios")
))]
pub mod runtime {
    use super::super::super::model::{MlxLlmError, MlxLlmResult};
    use super::{LayerKind, Lfm35Config};

    use std::path::Path;

    use safetensors::{Dtype as StDtype, SafeTensors};
    use xybrid_mlx::ops::rms_norm;
    use xybrid_mlx::{MlxArray, MlxStream};

    /// One full-attention block's weights. Layout mirrors Qwen 3 with
    /// the LFM-specific field names (`out_proj` / `{q,k}_layernorm`).
    #[derive(Debug)]
    pub struct AttentionLayer {
        pub operator_norm: MlxArray,
        pub q_proj: MlxArray,
        pub k_proj: MlxArray,
        pub v_proj: MlxArray,
        pub out_proj: MlxArray,
        pub q_layernorm: MlxArray,
        pub k_layernorm: MlxArray,
        pub ffn_norm: MlxArray,
        pub w1: MlxArray,
        pub w2: MlxArray,
        pub w3: MlxArray,
    }

    /// One short-conv block's weights. `conv.conv` is the actual 1D
    /// causal kernel; `in_proj` / `out_proj` wrap it with dense
    /// projections.
    #[derive(Debug)]
    pub struct ConvLayer {
        pub operator_norm: MlxArray,
        pub conv_in_proj: MlxArray,
        pub conv_kernel: MlxArray,
        pub conv_out_proj: MlxArray,
        pub ffn_norm: MlxArray,
        pub w1: MlxArray,
        pub w2: MlxArray,
        pub w3: MlxArray,
    }

    /// One LFM layer — either [`AttentionLayer`] or [`ConvLayer`].
    #[derive(Debug)]
    pub enum Lfm35Layer {
        Attention(AttentionLayer),
        Conv(ConvLayer),
    }

    /// Full LFM 3.5 weight set resident in MLX memory.
    #[derive(Debug)]
    pub struct Lfm35Weights {
        pub embed_tokens: MlxArray,
        pub layers: Vec<Lfm35Layer>,
        pub embedding_norm: MlxArray,
        /// `None` when [`Lfm35Config::tie_word_embeddings`] is set.
        pub lm_head: Option<MlxArray>,
    }

    /// Build [`Lfm35Weights`] from `model.safetensors`.
    ///
    /// Supports F32, F16, and BF16 weights. Quantized bundles are
    /// rejected up-front in [`super::validate_safetensors`].
    pub fn build(cfg: &Lfm35Config, weights_path: &Path) -> MlxLlmResult<Lfm35Weights> {
        let bytes = std::fs::read(weights_path).map_err(|e| MlxLlmError::WeightLoad {
            path: weights_path.to_path_buf(),
            reason: format!("read safetensors: {e}"),
        })?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| MlxLlmError::WeightLoad {
            path: weights_path.to_path_buf(),
            reason: format!("parse safetensors: {e}"),
        })?;

        let embed_tokens = load_tensor(&st, "model.embed_tokens.weight")?;
        let kinds = cfg.layer_kinds();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (l, kind) in kinds.iter().enumerate() {
            let base = format!("model.layers.{l}");
            let layer = match kind {
                LayerKind::FullAttention => Lfm35Layer::Attention(AttentionLayer {
                    operator_norm: load_tensor(&st, &format!("{base}.operator_norm.weight"))?,
                    q_proj: load_tensor(&st, &format!("{base}.self_attn.q_proj.weight"))?,
                    k_proj: load_tensor(&st, &format!("{base}.self_attn.k_proj.weight"))?,
                    v_proj: load_tensor(&st, &format!("{base}.self_attn.v_proj.weight"))?,
                    out_proj: load_tensor(&st, &format!("{base}.self_attn.out_proj.weight"))?,
                    q_layernorm: load_tensor(&st, &format!("{base}.self_attn.q_layernorm.weight"))?,
                    k_layernorm: load_tensor(&st, &format!("{base}.self_attn.k_layernorm.weight"))?,
                    ffn_norm: load_tensor(&st, &format!("{base}.ffn_norm.weight"))?,
                    w1: load_tensor(&st, &format!("{base}.feed_forward.w1.weight"))?,
                    w2: load_tensor(&st, &format!("{base}.feed_forward.w2.weight"))?,
                    w3: load_tensor(&st, &format!("{base}.feed_forward.w3.weight"))?,
                }),
                LayerKind::Conv => Lfm35Layer::Conv(ConvLayer {
                    operator_norm: load_tensor(&st, &format!("{base}.operator_norm.weight"))?,
                    conv_in_proj: load_tensor(&st, &format!("{base}.conv.in_proj.weight"))?,
                    conv_kernel: load_tensor(&st, &format!("{base}.conv.conv.weight"))?,
                    conv_out_proj: load_tensor(&st, &format!("{base}.conv.out_proj.weight"))?,
                    ffn_norm: load_tensor(&st, &format!("{base}.ffn_norm.weight"))?,
                    w1: load_tensor(&st, &format!("{base}.feed_forward.w1.weight"))?,
                    w2: load_tensor(&st, &format!("{base}.feed_forward.w2.weight"))?,
                    w3: load_tensor(&st, &format!("{base}.feed_forward.w3.weight"))?,
                }),
            };
            layers.push(layer);
        }
        let embedding_norm = load_tensor(&st, "model.embedding_norm.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_tensor(&st, "lm_head.weight")?)
        };

        Ok(Lfm35Weights {
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
        })
    }

    /// Read one tensor from a SafeTensors view into an [`MlxArray`].
    fn load_tensor(st: &SafeTensors<'_>, name: &str) -> MlxLlmResult<MlxArray> {
        let (floats, shape_i32) = read_as_f32(st, name)?;
        MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
    }

    /// Read one tensor and promote to `Vec<f32>`. F16 / BF16 tensors
    /// are promoted via the same half_to_f32 / bf16 helpers as the
    /// Qwen 3.5 / Gemma 4 loaders — duplication here is intentional to
    /// keep each arch builder self-contained.
    fn read_as_f32(st: &SafeTensors<'_>, name: &str) -> MlxLlmResult<(Vec<f32>, Vec<i32>)> {
        let view = st.tensor(name).map_err(|e| MlxLlmError::WeightLoad {
            path: std::path::PathBuf::from(name),
            reason: format!("tensor missing: {e}"),
        })?;
        let shape_i32: Vec<i32> = view
            .shape()
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(i32::MAX))
            .collect();
        let data = view.data();

        let floats: Vec<f32> = match view.dtype() {
            StDtype::F32 => {
                debug_assert!(data.len().is_multiple_of(4));
                debug_assert_eq!(data.as_ptr().align_offset(align_of::<f32>()), 0);
                // SAFETY: alignment and length divisibility asserted above.
                let slice: &[f32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len() / 4)
                };
                slice.to_vec()
            }
            StDtype::F16 => data
                .chunks_exact(2)
                .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            StDtype::BF16 => data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
                    f32::from_bits(bits)
                })
                .collect(),
            other => {
                return Err(MlxLlmError::WeightLoad {
                    path: std::path::PathBuf::from(name),
                    reason: format!("unsupported tensor dtype {other:?}"),
                });
            }
        };
        Ok((floats, shape_i32))
    }

    /// IEEE754 half-precision → f32. Handles subnormals + Inf/NaN.
    fn half_to_f32(bits: u16) -> f32 {
        let sign = u32::from(bits >> 15) << 31;
        let exp = u32::from((bits >> 10) & 0x1f);
        let mant = u32::from(bits & 0x3ff);
        let out = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut m = mant;
                let mut e: i32 = -14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                sign | (((e + 127) as u32) << 23) | (m << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (mant << 13)
        } else {
            sign | ((exp + 112) << 23) | (mant << 13)
        };
        f32::from_bits(out)
    }

    /// Forward pass through the whole stack.
    ///
    /// **Deferred**: the short 1D causal conv primitive (`mlx_conv1d`)
    /// and the SwiGLU FFN activation (`silu`) are not yet wrapped in
    /// `xybrid_mlx::ops`. Both land alongside US-014's generate loop.
    /// For US-013 the forward pass wires the embedding lookup + the
    /// first layer's `operator_norm` (to exercise the weight-loader
    /// path) and bails with [`MlxLlmError::NotImplemented`] at the
    /// operator (conv / attention) boundary.
    pub fn forward(
        cfg: &Lfm35Config,
        weights: &Lfm35Weights,
        input_ids: &MlxArray,
        _position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        // Embedding lookup: [B, T] -> [B, T, H].
        let hidden = xybrid_mlx::ops::gather(&weights.embed_tokens, input_ids, 0, stream)?;

        // Exercise the first layer's operator_norm so any corruption
        // in the weight loader surfaces here rather than sitting
        // dormant until US-014 runs. The norm is shape-preserving.
        if let Some(layer0) = weights.layers.first() {
            let norm = match layer0 {
                super::runtime::Lfm35Layer::Attention(l) => &l.operator_norm,
                super::runtime::Lfm35Layer::Conv(l) => &l.operator_norm,
            };
            let _ = rms_norm(&hidden, Some(norm), cfg.norm_eps, stream)?;
        }

        // Full conv / attention stack + SwiGLU FFN land in US-014.
        Err(MlxLlmError::NotImplemented {
            feature: "LFM 3.5 hybrid conv+attention stack + SwiGLU FFN (needs mlx_conv1d + silu)",
            story: "US-014",
        })
    }

    // =========================================================================
    // Runtime tests (apple + runtime feature)
    // =========================================================================

    #[cfg(test)]
    mod tests {
        #[test]
        fn half_to_f32_one() {
            assert_eq!(super::half_to_f32(0x3c00), 1.0_f32);
        }

        #[test]
        fn bf16_round_trip_one() {
            let bytes = [0x80_u8, 0x3f];
            let v: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
                    f32::from_bits(bits)
                })
                .collect();
            assert_eq!(v.len(), 1);
            assert!((v[0] - 1.0_f32).abs() < f32::EPSILON);
        }
    }
}

// =============================================================================
// Tests (skeleton — no bindings required)
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
    fn zero_conv_l_cache_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.conv_l_cache = 0;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("conv_l_cache"), "got: {msg}"),
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
    fn quantized_rejected_by_safetensors_check() {
        let mut cfg = dummy_config(1, true);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
        });
        assert!(cfg.is_quantized());
        let err = validate_safetensors(Path::new("/nonexistent"), &cfg).unwrap_err();
        match err {
            MlxLlmError::UnsupportedArchitecture { model_type } => {
                assert!(model_type.contains("quantized"), "got: {model_type}");
                assert!(model_type.contains("4-bit"), "got: {model_type}");
            }
            other => panic!("expected UnsupportedArchitecture, got {other:?}"),
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
}

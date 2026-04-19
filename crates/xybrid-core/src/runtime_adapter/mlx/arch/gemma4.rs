//! Gemma 4 architecture builder.
//!
//! Ports the MLX-LM Python reference's Gemma 4 architecture to Rust + the
//! safe tensor ops from `xybrid_mlx::ops`. Gemma 4 is a decoder-only
//! transformer that inherits the Gemma 3 layout with several distinctive
//! tricks:
//!
//! 1. **RMSNorm `(1 + weight)` scale**: every RMSNorm applies
//!    `x * (1 + weight) * rsqrt(mean(x^2) + eps)` rather than the LLaMA /
//!    Qwen `x * weight * rsqrt(...)`. The `+1.0` is baked into the scale
//!    weight at load time (host-side fixup in
//!    `runtime::load_rmsnorm_plus_one`) so the MLX `mlx_fast_rms_norm`
//!    kernel can be reused as-is — no additional primitive needed.
//! 2. **Gated-GeLU FFN**: `down(gelu(gate_proj(x)) * up_proj(x))`. Uses
//!    GeLU-tanh approximation instead of SiLU. The `gelu` primitive is
//!    not yet wrapped in `xybrid_mlx::ops` (wrapper lands with US-014's
//!    generate loop + sampling primitives), so the forward pass stages
//!    the attention half and returns [`MlxLlmError::NotImplemented`] at
//!    the FFN boundary — matching the deferral pattern Qwen 3.5 uses for
//!    its SwiGLU path.
//! 3. **Sliding-window attention**: Gemma 4 alternates between full
//!    global attention and a local sliding-window pattern at a
//!    configurable cadence (`sliding_window_pattern`, default every 6th
//!    layer is global). The fused `mlx_fast_scaled_dot_product_attention`
//!    accepts an explicit mask so the sliding-window path is implemented
//!    as a precomputed boolean mask handed to SDPA. Mask construction
//!    lands with the generate loop in US-014 (it needs the running token
//!    count).
//! 4. **Pre- and post-feedforward layernorms**: Gemma wraps the FFN
//!    residual with an additional RMSNorm on both the input and the
//!    output of the FFN block. The weight schedule reflects this.
//! 5. **Per-head RMSNorm on Q and K** (`q_norm` / `k_norm`): applied
//!    before RoPE, same shape as Qwen 3.5's but with the `(1 + weight)`
//!    Gemma scaling.
//! 6. **Query pre-attention scalar**: Gemma 4 scales the query projection
//!    by an explicit `query_pre_attn_scalar` (usually `head_dim ** -0.5`
//!    collapsed into the SDPA `scale` argument).
//!
//! The builder lands in two layers, mirroring Qwen 3.5:
//!
//! 1. **Skeleton** (always compiled under `llm-mlx`) — parses
//!    `config.json`, enumerates the expected safetensors weight-key
//!    schedule, and validates the safetensors header before we commit to
//!    linking Metal.
//! 2. **Runtime** (gated on `llm-mlx-runtime` + Apple target) — weight
//!    materialisation and the staged forward pass. The missing activation
//!    primitive (`gelu`) and the sliding-window mask construction land
//!    alongside the generate loop in US-014.
//!
//! Reference (upstream): <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3.py>
//! (Gemma 4 currently mirrors Gemma 3's text-only topology on the MLX-LM
//! side; the family ID is `"gemma4"` in the config header).

use std::fs;
use std::path::{Path, PathBuf};

use safetensors::SafeTensors;
use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};

// =============================================================================
// Config
// =============================================================================

/// Full Gemma 4 `config.json` subset the builder needs.
///
/// Fields mirror the HuggingFace config layout (`model_type = "gemma4"`).
/// Missing fields fall back to upstream defaults where they exist.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    /// Always `"gemma4"` for the Gemma 4 family.
    pub model_type: String,
    /// Model hidden / embedding dimension.
    pub hidden_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of query-heads per block.
    pub num_attention_heads: usize,
    /// Number of K/V heads per block. Defaults to
    /// [`Self::num_attention_heads`] when omitted.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// FFN intermediate (up-projection) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum position index the model was trained for.
    pub max_position_embeddings: usize,
    /// RoPE base frequency. Gemma 3/4 uses 1e6 for full (global)
    /// attention and an independent `rope_local_base_freq` for the
    /// sliding-window layers.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Separate RoPE base for the local (sliding-window) layers.
    /// Defaults to 10_000.0 (matches MLX-LM's Gemma 3 default) when
    /// omitted.
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f32,
    /// Epsilon added under the sqrt in RMSNorm.
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    /// When `true`, the input embedding and the LM head share one weight
    /// tensor. Gemma bundles nearly always set this.
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    /// Per-head dimension. Gemma 4 typically stores this explicitly
    /// because `hidden_size` is not divisible by `num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Window size (in tokens) for the sliding-window attention layers.
    /// Gemma 3 defaults to 4096; unused for the global layers.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    /// Cadence at which a layer uses full (global) attention. The
    /// default is `6` — meaning layers `0, 6, 12, ...` are global and
    /// the rest use the sliding-window path.
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    /// Multiplier applied to the query projection before the attention
    /// dot product. Defaults to `1.0 / sqrt(head_dim)` when omitted.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    /// Set when weights are quantized. MLX 4-bit bundles report
    /// `{ bits: 4, group_size: 64 }` here.
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_rope_local_base_freq() -> f32 {
    10_000.0
}

fn default_rms_eps() -> f32 {
    1e-6
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_sliding_window() -> usize {
    4096
}

fn default_sliding_window_pattern() -> usize {
    6
}

/// Quantization metadata block from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    pub bits: u32,
    pub group_size: u32,
}

impl Gemma4Config {
    /// Parse `{model_dir}/config.json` as a Gemma 4 config.
    pub fn from_model_dir(model_dir: &Path) -> MlxLlmResult<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path)?;
        let cfg: Gemma4Config = serde_json::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> MlxLlmResult<()> {
        if self.model_type != "gemma4" {
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
                "gemma4 config has zero-valued dimensions (hidden={}, layers={}, heads={}, ffn={}, vocab={})",
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.intermediate_size,
                self.vocab_size,
            )));
        }
        if self.sliding_window_pattern == 0 {
            return Err(MlxLlmError::ConfigInvalid(
                "gemma4 config has sliding_window_pattern = 0 (must be >= 1)".into(),
            ));
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

    /// Effective query pre-attention scalar. Defaults to
    /// `1.0 / sqrt(head_dim)` when the config doesn't pin one.
    pub fn query_scale(&self) -> f32 {
        self.query_pre_attn_scalar
            .unwrap_or_else(|| 1.0 / (self.head_dim() as f32).sqrt())
    }

    /// `true` when layer `l` uses full (global) attention, `false` when
    /// it uses the sliding-window path.
    pub fn layer_is_global(&self, l: usize) -> bool {
        l.is_multiple_of(self.sliding_window_pattern)
    }

    /// `true` when weights are stored in a quantized format.
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }
}

// =============================================================================
// Expected safetensors schedule
// =============================================================================

/// Safetensors key names the Gemma 4 forward pass reads, in the order
/// the upstream MLX-LM reference emits them.
///
/// Per-layer block (for layer `l`), 13 tensors:
/// - `input_layernorm.weight`
/// - `self_attn.{q,k,v,o}_proj.weight` (4)
/// - `self_attn.{q,k}_norm.weight` — per-head RMSNorm before RoPE (2)
/// - `post_attention_layernorm.weight`
/// - `pre_feedforward_layernorm.weight`
/// - `post_feedforward_layernorm.weight`
/// - `mlp.{gate,up,down}_proj.weight` (3)
///
/// Shared across blocks (2 or 3):
/// - `model.embed_tokens.weight`
/// - `model.norm.weight`
/// - `lm_head.weight` — only when
///   [`Gemma4Config::tie_word_embeddings`] is `false`.
pub fn expected_weight_keys(cfg: &Gemma4Config) -> Vec<String> {
    let mut keys = Vec::with_capacity(3 + 13 * cfg.num_hidden_layers);
    keys.push("model.embed_tokens.weight".to_string());
    for l in 0..cfg.num_hidden_layers {
        let base = format!("model.layers.{l}");
        keys.push(format!("{base}.input_layernorm.weight"));
        keys.push(format!("{base}.self_attn.q_proj.weight"));
        keys.push(format!("{base}.self_attn.k_proj.weight"));
        keys.push(format!("{base}.self_attn.v_proj.weight"));
        keys.push(format!("{base}.self_attn.o_proj.weight"));
        keys.push(format!("{base}.self_attn.q_norm.weight"));
        keys.push(format!("{base}.self_attn.k_norm.weight"));
        keys.push(format!("{base}.post_attention_layernorm.weight"));
        keys.push(format!("{base}.pre_feedforward_layernorm.weight"));
        keys.push(format!("{base}.post_feedforward_layernorm.weight"));
        keys.push(format!("{base}.mlp.gate_proj.weight"));
        keys.push(format!("{base}.mlp.up_proj.weight"));
        keys.push(format!("{base}.mlp.down_proj.weight"));
    }
    keys.push("model.norm.weight".to_string());
    if !cfg.tie_word_embeddings {
        keys.push("lm_head.weight".to_string());
    }
    keys
}

// =============================================================================
// Safetensors validation
// =============================================================================

/// Validate that `model.safetensors` contains every key Gemma 4's
/// forward pass reads.
///
/// Mirrors [`super::qwen35::validate_safetensors`] — quantized bundles
/// bail here with a pointed error so the runtime selector (US-016) can
/// fall back to llama.cpp.
pub fn validate_safetensors(path: &Path, cfg: &Gemma4Config) -> MlxLlmResult<()> {
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
    let names: std::collections::HashSet<String> = meta.tensors().into_keys().collect();

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

/// Top-level entry: load and validate a Gemma 4 bundle's config +
/// weight manifest. Returns the parsed config so the skeleton load path
/// in [`super::super::model::MlxLlmAdapter::load`] can cache it.
///
/// Does NOT touch MLX / Metal — that happens in `runtime::build` under
/// the `llm-mlx-runtime` feature.
pub fn load(model_dir: &Path, weights_path: &Path) -> MlxLlmResult<Gemma4Config> {
    let cfg = Gemma4Config::from_model_dir(model_dir)?;
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
    use super::Gemma4Config;

    use std::path::Path;

    use safetensors::{Dtype as StDtype, SafeTensors};
    use xybrid_mlx::ops::{matmul, reshape, rms_norm, rope, transpose};
    use xybrid_mlx::{MlxArray, MlxStream};

    /// One transformer block's weights, materialised as [`MlxArray`]s.
    /// RMSNorm scale weights have already been transformed into
    /// `weight + 1.0` (see [`load_rmsnorm_plus_one`]) so the MLX kernel
    /// sees a single scale factor.
    #[derive(Debug)]
    pub struct Gemma4Layer {
        pub input_layernorm: MlxArray,
        pub q_proj: MlxArray,
        pub k_proj: MlxArray,
        pub v_proj: MlxArray,
        pub o_proj: MlxArray,
        pub q_norm: MlxArray,
        pub k_norm: MlxArray,
        pub post_attention_layernorm: MlxArray,
        pub pre_feedforward_layernorm: MlxArray,
        pub post_feedforward_layernorm: MlxArray,
        pub mlp_gate_proj: MlxArray,
        pub mlp_up_proj: MlxArray,
        pub mlp_down_proj: MlxArray,
    }

    /// Full Gemma 4 weight set resident in MLX memory.
    #[derive(Debug)]
    pub struct Gemma4Weights {
        pub embed_tokens: MlxArray,
        pub layers: Vec<Gemma4Layer>,
        pub norm: MlxArray,
        /// `None` when [`Gemma4Config::tie_word_embeddings`] is set.
        pub lm_head: Option<MlxArray>,
    }

    /// Build [`Gemma4Weights`] from `model.safetensors`.
    ///
    /// Supports F32, F16, and BF16 weights. Quantized bundles are
    /// rejected up-front in [`super::validate_safetensors`].
    pub fn build(cfg: &Gemma4Config, weights_path: &Path) -> MlxLlmResult<Gemma4Weights> {
        let bytes = std::fs::read(weights_path).map_err(|e| MlxLlmError::WeightLoad {
            path: weights_path.to_path_buf(),
            reason: format!("read safetensors: {e}"),
        })?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| MlxLlmError::WeightLoad {
            path: weights_path.to_path_buf(),
            reason: format!("parse safetensors: {e}"),
        })?;

        let embed_tokens = load_tensor(&st, "model.embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let base = format!("model.layers.{l}");
            layers.push(Gemma4Layer {
                input_layernorm: load_rmsnorm_plus_one(
                    &st,
                    &format!("{base}.input_layernorm.weight"),
                )?,
                q_proj: load_tensor(&st, &format!("{base}.self_attn.q_proj.weight"))?,
                k_proj: load_tensor(&st, &format!("{base}.self_attn.k_proj.weight"))?,
                v_proj: load_tensor(&st, &format!("{base}.self_attn.v_proj.weight"))?,
                o_proj: load_tensor(&st, &format!("{base}.self_attn.o_proj.weight"))?,
                q_norm: load_rmsnorm_plus_one(&st, &format!("{base}.self_attn.q_norm.weight"))?,
                k_norm: load_rmsnorm_plus_one(&st, &format!("{base}.self_attn.k_norm.weight"))?,
                post_attention_layernorm: load_rmsnorm_plus_one(
                    &st,
                    &format!("{base}.post_attention_layernorm.weight"),
                )?,
                pre_feedforward_layernorm: load_rmsnorm_plus_one(
                    &st,
                    &format!("{base}.pre_feedforward_layernorm.weight"),
                )?,
                post_feedforward_layernorm: load_rmsnorm_plus_one(
                    &st,
                    &format!("{base}.post_feedforward_layernorm.weight"),
                )?,
                mlp_gate_proj: load_tensor(&st, &format!("{base}.mlp.gate_proj.weight"))?,
                mlp_up_proj: load_tensor(&st, &format!("{base}.mlp.up_proj.weight"))?,
                mlp_down_proj: load_tensor(&st, &format!("{base}.mlp.down_proj.weight"))?,
            });
        }
        let norm = load_rmsnorm_plus_one(&st, "model.norm.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_tensor(&st, "lm_head.weight")?)
        };

        Ok(Gemma4Weights {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Read one tensor from a SafeTensors view into an [`MlxArray`].
    fn load_tensor(st: &SafeTensors<'_>, name: &str) -> MlxLlmResult<MlxArray> {
        let (floats, shape_i32) = read_as_f32(st, name)?;
        MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
    }

    /// Gemma RMSNorm weights are used as `(1 + weight)`. Bake the
    /// offset in at load time so the MLX kernel sees a plain scale
    /// factor and no extra op is needed on the hot path.
    ///
    /// This is the single biggest deviation from Qwen 3.5 — both archs
    /// store `weight` in safetensors, but Gemma's forward pass uses it
    /// differently. Host-side fixup keeps the runtime identical.
    fn load_rmsnorm_plus_one(st: &SafeTensors<'_>, name: &str) -> MlxLlmResult<MlxArray> {
        let (mut floats, shape_i32) = read_as_f32(st, name)?;
        for v in &mut floats {
            *v += 1.0;
        }
        MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
    }

    /// Read one tensor and promote to `Vec<f32>`. F16 / BF16 tensors
    /// are promoted via the same half_to_f32 / bf16 helpers as the
    /// Qwen 3.5 loader — duplication here is intentional to keep each
    /// arch builder self-contained.
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
    /// **Deferred**: the Gated-GeLU FFN needs a `gelu` primitive and
    /// the sliding-window mask construction needs the running token
    /// count from the generate loop. Both land in US-014. For US-012
    /// the forward pass wires the embedding lookup + input layernorm
    /// (to exercise the `(1 + weight)` RMSNorm path) and bails with
    /// [`MlxLlmError::NotImplemented`] at the attention boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        cfg: &Gemma4Config,
        weights: &Gemma4Weights,
        input_ids: &MlxArray,
        _position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        // Embedding lookup: [B, T] -> [B, T, H]. Gemma scales the
        // embedding output by sqrt(hidden_size) immediately after
        // lookup — this is the "input embedding scaling" trick from
        // the original Gemma paper. We defer the scale multiply to
        // US-014 (needs a scalar-mul wrapper); the lookup itself is
        // still useful to wire so the weight path is exercised.
        let hidden = xybrid_mlx::ops::gather(&weights.embed_tokens, input_ids, 0, stream)?;

        // Exercise the `(1 + weight)` RMSNorm on the first layer so
        // the host-side fixup path is reached before we bail — any
        // corruption in `load_rmsnorm_plus_one` surfaces here rather
        // than sitting dormant until US-014 runs.
        if let Some(layer0) = weights.layers.first() {
            let _ = rms_norm(
                &hidden,
                Some(&layer0.input_layernorm),
                cfg.rms_norm_eps,
                stream,
            )?;
        }

        // Full attention block + gated-GeLU FFN + sliding-window mask
        // land in US-014. Returning here keeps the partial path from
        // producing incorrect logits for callers that accidentally
        // reach this code before US-014 ships.
        Err(MlxLlmError::NotImplemented {
            feature: "Gemma 4 attention block + gated-GeLU FFN + sliding-window mask",
            story: "US-014",
        })
    }

    /// Reshape `[B, T, heads * head_dim]` → `[B, heads, T, head_dim]`.
    /// Exported at `pub(crate)` for US-014 to reuse.
    #[allow(dead_code)]
    pub(crate) fn split_heads(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = x.shape();
        let b = shape[0];
        let t = shape[1];
        let reshaped = reshape(x, &[b, t, heads as i32, head_dim as i32], stream)?;
        transpose(&reshaped, &[0, 2, 1, 3], stream).map_err(Into::into)
    }

    /// Reshape `[B, heads, T, head_dim]` → `[B, T, heads * head_dim]`.
    #[allow(dead_code)]
    pub(crate) fn merge_heads(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = x.shape();
        let b = shape[0];
        let t = shape[2];
        let transposed = transpose(x, &[0, 2, 1, 3], stream)?;
        reshape(&transposed, &[b, t, (heads * head_dim) as i32], stream).map_err(Into::into)
    }

    /// Project, split into heads, apply per-head RMSNorm, then RoPE —
    /// helper that US-014 will wire into the full attention block.
    /// The RoPE base differs between global and sliding-window layers.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn project_and_rope(
        proj: &MlxArray,
        head_norm: &MlxArray,
        input: &MlxArray,
        heads: usize,
        head_dim: usize,
        rope_base: f32,
        rms_eps: f32,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let projected = matmul(input, &transpose(proj, &[1, 0], stream)?, stream)?;
        let heads_out = split_heads(&projected, heads, head_dim, stream)?;
        let normed = rms_norm(&heads_out, Some(head_norm), rms_eps, stream)?;
        rope(
            &normed,
            head_dim as i32,
            false,
            Some(rope_base),
            1.0,
            position_offset,
            None,
            stream,
        )
        .map_err(Into::into)
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

    fn dummy_config(n_layers: usize, tied: bool) -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4".into(),
            hidden_size: 16,
            num_hidden_layers: n_layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            intermediate_size: 32,
            vocab_size: 100,
            max_position_embeddings: 1024,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: tied,
            head_dim: Some(4),
            sliding_window: 4096,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: None,
            quantization: None,
        }
    }

    #[test]
    fn expected_keys_match_per_layer_schedule() {
        let cfg = dummy_config(2, false);
        let keys = expected_weight_keys(&cfg);
        // 1 (embed) + 2 layers * 13 (per-layer) + 2 (norm + lm_head) = 29.
        assert_eq!(keys.len(), 29);
        assert_eq!(keys.first().unwrap(), "model.embed_tokens.weight");
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.pre_feedforward_layernorm.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.1.post_feedforward_layernorm.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.self_attn.q_norm.weight"));
        assert_eq!(keys.last().unwrap(), "lm_head.weight");
    }

    #[test]
    fn tied_embeddings_skip_lm_head() {
        let cfg = dummy_config(1, true);
        let keys = expected_weight_keys(&cfg);
        assert!(!keys.iter().any(|k| k == "lm_head.weight"));
        assert!(keys.iter().any(|k| k == "model.norm.weight"));
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
    fn query_scale_defaults_to_inv_sqrt_head_dim() {
        let cfg = dummy_config(1, true);
        // head_dim = 4, so scale = 1/2 = 0.5.
        assert!((cfg.query_scale() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn query_scale_respects_explicit_pre_attn_scalar() {
        let mut cfg = dummy_config(1, true);
        cfg.query_pre_attn_scalar = Some(0.125);
        assert!((cfg.query_scale() - 0.125).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_pattern_marks_global_layers() {
        let cfg = dummy_config(12, true); // default pattern = 6.
        assert!(cfg.layer_is_global(0));
        assert!(!cfg.layer_is_global(1));
        assert!(!cfg.layer_is_global(5));
        assert!(cfg.layer_is_global(6));
        assert!(cfg.layer_is_global(12));
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
    fn non_gemma4_model_type_rejected() {
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
    fn zero_sliding_window_pattern_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.sliding_window_pattern = 0;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("sliding_window_pattern"), "got: {msg}")
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
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
        let cfg = dummy_config(1, true);
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
}

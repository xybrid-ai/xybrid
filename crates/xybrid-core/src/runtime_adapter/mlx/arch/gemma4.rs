//! Gemma 4 architecture builder.
//!
//! Ports the MLX-LM Python reference's Gemma 4 architecture to Rust + the
//! safe tensor ops from `xybrid_mlx::ops`. Gemma 4 is a decoder-only
//! transformer that inherits the Gemma 3 layout with several distinctive
//! tricks:
//!
//! 1. **RMSNorm scale**: MLX-converted Gemma 4 weights store the RMSNorm scale directly, so
//!    runtime loading passes those tensors to `mlx_fast_rms_norm` as-is.
//! 2. **Gated-GeLU FFN**: `down(gelu(gate_proj(x)) * up_proj(x))`. Uses
//!    GeLU-tanh approximation instead of SiLU. `xybrid_mlx::ops` exposes
//!    both exact BERT-style `gelu` and Gemma-style `gelu_tanh`; Gemma
//!    uses the tanh approximation to match MLX-LM.
//! 3. **Sliding-window attention**: Gemma 4 alternates between full
//!    global attention and local sliding-window layers. Some configs
//!    expose that through `sliding_window_pattern`; current public Gemma 4
//!    bundles also encode it in per-layer projection/head dimensions. The
//!    fused `mlx_fast_scaled_dot_product_attention` accepts an explicit mask
//!    so the sliding-window path is implemented as a precomputed additive
//!    mask handed to SDPA.
//! 4. **Pre- and post-feedforward layernorms**: Gemma wraps the FFN
//!    residual with an additional RMSNorm on both the input and the
//!    output of the FFN block. The weight schedule reflects this.
//! 5. **Per-head RMSNorm on Q, K, and V**: Q/K use learned scale tensors
//!    before RoPE; V uses an unscaled RMSNorm before attention.
//! 6. **Unscaled attention scores**: Gemma 4's MLX-LM path passes
//!    `scale = 1.0` to SDPA after the per-head normalisation.
//!
//! The builder lands in two layers, mirroring Qwen 3:
//!
//! 1. **Non-linking validation** (always compiled under `llm-mlx`) — parses
//!    `config.json`, enumerates the expected safetensors weight-key
//!    schedule, and validates the safetensors header before we commit to
//!    linking Metal.
//! 2. **Runtime** (gated on `llm-mlx-runtime` + Apple Silicon macOS) — weight
//!    materialisation, Gemma-specific FFN wiring, sliding-window mask
//!    construction, and resident K/V append/read for incremental decode.
//!
//! Reference (upstream): <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py>.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};
use super::super::weights::SafeTensorBundle;

const LANGUAGE_MODEL_PREFIX: &str = "language_model.";
const GEMMA4_LAYER_FULL_ATTENTION: &str = "full_attention";
const GEMMA4_LAYER_SLIDING_ATTENTION: &str = "sliding_attention";

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
    /// Wider head dimension for full-attention layers in Gemma 4 ISWA.
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    /// Ordered layer schedule from `config.json` (`sliding_attention` /
    /// `full_attention`).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Window size (in tokens) for the sliding-window attention layers.
    /// Gemma 3 defaults to 4096; unused for the global layers.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    /// Cadence at which a layer uses full (global) attention. The
    /// default is `6` — meaning layers `5, 11, 17, ...` are global when
    /// `layer_types` is absent, and the rest use the sliding-window path.
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    /// Scalar whose inverse square root is used as the attention scale.
    /// Upstream Gemma stores `head_dim` here and computes
    /// `query_pre_attn_scalar ** -0.5`.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    /// Optional tanh softcap applied to the final logits.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    /// Gemma 4 per-layer input embedding width.
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    /// Vocabulary size used by `embed_tokens_per_layer`.
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    /// Number of tail layers that reuse K/V states from earlier layers.
    #[serde(default)]
    pub num_kv_shared_layers: Option<usize>,
    /// Full-attention proportional RoPE rotary fraction.
    #[serde(default = "default_full_attention_partial_rotary_factor")]
    pub full_attention_partial_rotary_factor: f32,
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

fn default_full_attention_partial_rotary_factor() -> f32 {
    0.25
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

impl Gemma4Config {
    /// Parse `{model_dir}/config.json` as a Gemma 4 config.
    pub fn from_model_dir(model_dir: &Path) -> MlxLlmResult<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path)?;
        let cfg = Self::parse_json(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn parse_json(raw: &str) -> MlxLlmResult<Self> {
        match serde_json::from_str::<Gemma4Config>(raw) {
            Ok(cfg) => Ok(cfg),
            Err(direct_err) => {
                let root: serde_json::Value = serde_json::from_str(raw)?;
                let Some(text_config) = root.get("text_config") else {
                    return Err(MlxLlmError::ConfigParse(direct_err));
                };
                let mut text_config = text_config.clone();
                if let Some(obj) = text_config.as_object_mut() {
                    if let Some(model_type) = root.get("model_type").and_then(|v| v.as_str()) {
                        obj.insert(
                            "model_type".to_string(),
                            serde_json::Value::String(model_type.to_string()),
                        );
                    }
                    if !obj.contains_key("quantization") {
                        if let Some(quantization) = root
                            .get("quantization")
                            .or_else(|| root.get("quantization_config"))
                        {
                            obj.insert("quantization".to_string(), quantization.clone());
                        }
                    }
                    if !obj.contains_key("full_attention_partial_rotary_factor") {
                        if let Some(value) = obj
                            .get("rope_parameters")
                            .and_then(|v| v.get("full_attention"))
                            .and_then(|v| v.get("partial_rotary_factor"))
                            .cloned()
                        {
                            obj.insert("full_attention_partial_rotary_factor".to_string(), value);
                        }
                    }
                }
                serde_json::from_value(text_config).map_err(MlxLlmError::ConfigParse)
            }
        }
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
        if self.head_dim.is_none() && !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 config hidden_size={} is not a multiple of num_attention_heads={}",
                self.hidden_size, self.num_attention_heads,
            )));
        }
        let head_dim = self.head_dim();
        let kv_heads = self.kv_heads();
        if head_dim == 0 || kv_heads == 0 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 config has invalid derived dimensions (head_dim={head_dim}, kv_heads={kv_heads})"
            )));
        }
        if !self.num_attention_heads.is_multiple_of(kv_heads) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 config num_attention_heads={} is not a multiple of num_key_value_heads={kv_heads}",
                self.num_attention_heads,
            )));
        }
        super::validate_i32_dimensions(
            "gemma4",
            &[
                ("hidden_size", self.hidden_size),
                ("num_hidden_layers", self.num_hidden_layers),
                ("num_attention_heads", self.num_attention_heads),
                ("num_key_value_heads", kv_heads),
                ("intermediate_size", self.intermediate_size),
                ("vocab_size", self.vocab_size),
                ("max_position_embeddings", self.max_position_embeddings),
                ("head_dim", head_dim),
                ("sliding_window", self.sliding_window),
            ],
        )?;
        if let Some(global_head_dim) = self.global_head_dim {
            super::validate_i32_dimensions("gemma4", &[("global_head_dim", global_head_dim)])?;
        }
        if let Some(hidden_size_per_layer_input) = self.hidden_size_per_layer_input {
            super::validate_i32_dimensions(
                "gemma4",
                &[("hidden_size_per_layer_input", hidden_size_per_layer_input)],
            )?;
            super::validate_i32_product(
                "gemma4",
                "per_layer_input_width",
                &[self.num_hidden_layers, hidden_size_per_layer_input],
            )?;
        }
        if let Some(vocab_size_per_layer_input) = self.vocab_size_per_layer_input {
            super::validate_i32_dimensions(
                "gemma4",
                &[("vocab_size_per_layer_input", vocab_size_per_layer_input)],
            )?;
        }
        super::validate_i32_product(
            "gemma4",
            "attention_projection_width",
            &[self.num_attention_heads, head_dim],
        )?;
        super::validate_i32_product("gemma4", "kv_projection_width", &[kv_heads, head_dim])?;
        if self.sliding_window == 0 {
            return Err(MlxLlmError::ConfigInvalid(
                "gemma4 config has sliding_window = 0 (must be >= 1)".into(),
            ));
        }
        if self.sliding_window_pattern == 0 {
            return Err(MlxLlmError::ConfigInvalid(
                "gemma4 config has sliding_window_pattern = 0 (must be >= 1)".into(),
            ));
        }
        if let Some(layer_types) = self.layer_types.as_ref() {
            if layer_types.len() != self.num_hidden_layers {
                return Err(MlxLlmError::ConfigInvalid(format!(
                    "gemma4 config layer_types has {} entries but num_hidden_layers={}",
                    layer_types.len(),
                    self.num_hidden_layers,
                )));
            }
            for (idx, layer_type) in layer_types.iter().enumerate() {
                if layer_type != GEMMA4_LAYER_FULL_ATTENTION
                    && layer_type != GEMMA4_LAYER_SLIDING_ATTENTION
                {
                    return Err(MlxLlmError::ConfigInvalid(format!(
                        "gemma4 config layer_types[{idx}] has unsupported value `{layer_type}` \
                         (expected `{GEMMA4_LAYER_FULL_ATTENTION}` or \
                         `{GEMMA4_LAYER_SLIDING_ATTENTION}`)"
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

    /// Effective attention scale. Defaults to `1.0 / sqrt(head_dim)`
    /// when the config doesn't pin `query_pre_attn_scalar`.
    pub fn query_scale(&self) -> f32 {
        self.query_pre_attn_scalar
            .unwrap_or(self.head_dim() as f32)
            .powf(-0.5)
    }

    /// `true` when layer `l` uses full (global) attention, `false` when
    /// it uses the sliding-window path.
    pub fn layer_is_global(&self, l: usize) -> bool {
        if let Some(layer_types) = self.layer_types.as_ref() {
            return layer_types
                .get(l)
                .is_some_and(|layer_type| layer_type == GEMMA4_LAYER_FULL_ATTENTION);
        }
        l % self.sliding_window_pattern == self.sliding_window_pattern - 1
    }

    /// Effective attention head dimension for a specific layer.
    pub fn layer_head_dim(&self, l: usize) -> usize {
        if self.layer_is_global(l) {
            self.global_head_dim.unwrap_or_else(|| self.head_dim())
        } else {
            self.head_dim()
        }
    }

    /// Effective RoPE width for a specific layer.
    pub fn layer_rope_dims(&self, l: usize) -> usize {
        let head_dim = self.layer_head_dim(l);
        if self.layer_is_global(l) {
            let dims = (head_dim as f32 * self.full_attention_partial_rotary_factor).round();
            let dims = dims.max(2.0) as usize;
            dims - (dims % 2)
        } else {
            head_dim
        }
    }

    pub fn has_per_layer_inputs(&self) -> bool {
        self.hidden_size_per_layer_input.unwrap_or(0) > 0
    }

    pub fn hidden_size_per_layer_input(&self) -> Option<usize> {
        self.hidden_size_per_layer_input.filter(|v| *v > 0)
    }

    pub fn kv_shared_source_layer(&self, layer_idx: usize) -> Option<usize> {
        let shared_layers = self.num_kv_shared_layers.unwrap_or(0);
        if shared_layers == 0 || shared_layers >= self.num_hidden_layers {
            return None;
        }
        let first_shared = self.num_hidden_layers - shared_layers;
        if layer_idx < first_shared {
            return None;
        }

        let target_is_global = self.layer_is_global(layer_idx);
        (0..first_shared)
            .rev()
            .find(|idx| self.layer_is_global(*idx) == target_is_global)
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
    push_quantized_weight_key(&mut keys, "model.embed_tokens.weight", cfg.is_quantized());
    if cfg.has_per_layer_inputs() {
        push_quantized_weight_key(
            &mut keys,
            "model.embed_tokens_per_layer.weight",
            cfg.is_quantized(),
        );
        keys.push("model.per_layer_model_projection.weight".to_string());
        keys.push("model.per_layer_projection_norm.weight".to_string());
    }
    for l in 0..cfg.num_hidden_layers {
        let base = format!("model.layers.{l}");
        keys.push(format!("{base}.input_layernorm.weight"));
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
            &format!("{base}.self_attn.o_proj.weight"),
            cfg.is_quantized(),
        );
        keys.push(format!("{base}.self_attn.q_norm.weight"));
        keys.push(format!("{base}.self_attn.k_norm.weight"));
        keys.push(format!("{base}.post_attention_layernorm.weight"));
        keys.push(format!("{base}.pre_feedforward_layernorm.weight"));
        keys.push(format!("{base}.post_feedforward_layernorm.weight"));
        push_quantized_weight_key(
            &mut keys,
            &format!("{base}.mlp.gate_proj.weight"),
            cfg.is_quantized(),
        );
        push_quantized_weight_key(
            &mut keys,
            &format!("{base}.mlp.up_proj.weight"),
            cfg.is_quantized(),
        );
        push_quantized_weight_key(
            &mut keys,
            &format!("{base}.mlp.down_proj.weight"),
            cfg.is_quantized(),
        );
        if cfg.has_per_layer_inputs() {
            push_quantized_weight_key(
                &mut keys,
                &format!("{base}.per_layer_input_gate.weight"),
                cfg.is_quantized(),
            );
            push_quantized_weight_key(
                &mut keys,
                &format!("{base}.per_layer_projection.weight"),
                cfg.is_quantized(),
            );
            keys.push(format!("{base}.post_per_layer_input_norm.weight"));
        }
    }
    keys.push("model.norm.weight".to_string());
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

fn prefixed_weight_key(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}{key}")
    }
}

fn detect_weight_prefix(
    names: &std::collections::HashSet<String>,
    expected: &[String],
) -> &'static str {
    for prefix in ["", LANGUAGE_MODEL_PREFIX] {
        if expected.iter().all(|key| {
            let candidate = prefixed_weight_key(prefix, key);
            names.contains(candidate.as_str())
        }) {
            return prefix;
        }
    }
    ""
}

// =============================================================================
// Safetensors validation
// =============================================================================

/// Validate that `model.safetensors` contains every key Gemma 4's
/// forward pass reads.
///
/// Mirrors [`super::qwen35::validate_safetensors`] — quantized bundles
/// bail here with a pointed error so the runtime selector can fall back.
pub fn validate_safetensors(path: &Path, cfg: &Gemma4Config) -> MlxLlmResult<()> {
    let weights = SafeTensorBundle::from_single_file(path.to_path_buf());
    validate_safetensors_bundle(&weights, cfg)
}

pub fn validate_safetensors_bundle(
    weights: &SafeTensorBundle,
    cfg: &Gemma4Config,
) -> MlxLlmResult<()> {
    let names = weights.tensor_names()?;

    let expected = expected_weight_keys(cfg);
    let prefix = detect_weight_prefix(&names, &expected);
    let missing: Vec<String> = expected
        .iter()
        .map(|k| prefixed_weight_key(prefix, k))
        .filter(|k| !names.contains(k.as_str()))
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

/// Top-level entry: load and validate a Gemma 4 bundle's config +
/// weight manifest. Returns the parsed config so the non-linking load path
/// in [`super::super::model::MlxLlmAdapter::load`] can cache it.
///
/// Does NOT touch MLX / Metal — that happens in `runtime::build` under
/// the `llm-mlx-runtime` feature.
pub fn load(model_dir: &Path, weights: &SafeTensorBundle) -> MlxLlmResult<Gemma4Config> {
    let cfg = Gemma4Config::from_model_dir(model_dir)?;
    validate_safetensors_bundle(weights, &cfg)?;
    Ok(cfg)
}

/// Resolve the legacy single-file safetensors path for callers that need
/// that exact file. Normal Gemma validation/runtime loading goes through
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
    use super::super::{
        checked_i32_add, shape_dim_i32, shape_dim_usize, shape_product_i32, shape_product_usize,
    };
    use super::Gemma4Config;

    use xybrid_mlx::ops::{
        add, cast, concat, div, gather, gelu_tanh, mul, reshape, rms_norm, rope,
        scaled_dot_product_attention, slice, slice_update, tanh, transpose, zeros,
    };
    use xybrid_mlx::{MlxArray, MlxDtype, MlxStream};

    const GEMMA4_SLIDING_CACHE_BUFFER: usize = 256;

    /// One transformer block's weights.
    ///
    /// Projection/FFN weights are materialised as [`MlxArray`]s.
    /// RMSNorm scale weights are loaded exactly as stored in the MLX
    /// SafeTensors bundle.
    #[derive(Debug)]
    pub struct Gemma4Layer {
        pub input_layernorm: MlxArray,
        pub q_proj: LinearWeight,
        pub k_proj: LinearWeight,
        pub v_proj: LinearWeight,
        pub o_proj: LinearWeight,
        pub q_norm: MlxArray,
        pub k_norm: MlxArray,
        pub post_attention_layernorm: MlxArray,
        pub pre_feedforward_layernorm: MlxArray,
        pub post_feedforward_layernorm: MlxArray,
        pub mlp_gate_proj: LinearWeight,
        pub mlp_up_proj: LinearWeight,
        pub mlp_down_proj: LinearWeight,
        pub layer_scalar: Option<MlxArray>,
        pub per_layer_input_gate: Option<LinearWeight>,
        pub per_layer_projection: Option<LinearWeight>,
        pub post_per_layer_input_norm: Option<MlxArray>,
    }

    /// Resident K/V tensors for one Gemma decoder layer during a generation
    /// call. Reset before prefill; appended one token at a time during decode.
    #[derive(Debug, Clone, Copy, Default)]
    enum Gemma4CacheKind {
        #[default]
        Global,
        Sliding {
            window: usize,
        },
    }

    #[derive(Debug, Default)]
    pub struct Gemma4LayerCache {
        kind: Gemma4CacheKind,
        keys: Option<CachedAxis2>,
        values: Option<CachedAxis2>,
    }

    #[derive(Debug)]
    struct CachedAxis2 {
        storage: MlxArray,
        len: usize,
    }

    impl Gemma4LayerCache {
        fn new(kind: Gemma4CacheKind) -> Self {
            Self {
                kind,
                keys: None,
                values: None,
            }
        }

        fn is_sliding(&self) -> bool {
            matches!(self.kind, Gemma4CacheKind::Sliding { .. })
        }

        fn window(&self) -> Option<usize> {
            match self.kind {
                Gemma4CacheKind::Global => None,
                Gemma4CacheKind::Sliding { window } => Some(window),
            }
        }

        fn reset(&mut self) {
            self.keys = None;
            self.values = None;
        }

        fn cloned_kv(
            &self,
            layer_idx: usize,
            stream: Option<&MlxStream>,
        ) -> MlxLlmResult<(MlxArray, MlxArray)> {
            let keys = self.keys.as_ref().ok_or_else(|| {
                MlxLlmError::ConfigInvalid(format!(
                    "gemma4 shared K/V source layer {layer_idx} has no cached keys"
                ))
            })?;
            let values = self.values.as_ref().ok_or_else(|| {
                MlxLlmError::ConfigInvalid(format!(
                    "gemma4 shared K/V source layer {layer_idx} has no cached values"
                ))
            })?;
            Ok((
                cache_visible_axis2(keys, self.window(), stream)?,
                cache_visible_axis2(values, self.window(), stream)?,
            ))
        }
    }

    #[derive(Debug)]
    pub struct Gemma4PerLayerWeights {
        layer_inputs: Vec<Gemma4LayerInputWeights>,
        token_scale: MlxArray,
        projection_scale: MlxArray,
        input_scale: MlxArray,
    }

    #[derive(Debug)]
    struct Gemma4LayerInputWeights {
        embed_tokens: MlxArray,
        model_projection: LinearWeight,
        projection_norm: MlxArray,
    }

    /// Full Gemma 4 weight set resident in MLX memory.
    #[derive(Debug)]
    pub struct Gemma4Weights {
        pub embed_tokens: MlxArray,
        pub embed_tokens_linear: LinearWeight,
        pub per_layer: Option<Gemma4PerLayerWeights>,
        pub layers: Vec<Gemma4Layer>,
        pub norm: MlxArray,
        /// `None` when [`Gemma4Config::tie_word_embeddings`] is set.
        pub lm_head: Option<LinearWeight>,
        embed_scale: MlxArray,
        final_logit_softcap: Option<MlxArray>,
        layer_cache: Vec<Gemma4LayerCache>,
    }

    impl Gemma4Weights {
        /// Clear per-generation K/V state while keeping resident weights.
        pub fn reset_kv_cache(&mut self) {
            for cache in &mut self.layer_cache {
                cache.reset();
            }
        }
    }

    /// Build [`Gemma4Weights`] from a SafeTensors bundle.
    ///
    /// Supports F32, F16, and BF16 weights. Quantized bundles are
    /// rejected up-front in [`super::validate_safetensors`].
    pub fn build(cfg: &Gemma4Config, weights: &SafeTensorBundle) -> MlxLlmResult<Gemma4Weights> {
        let names = weights.tensor_names()?;
        let expected = super::expected_weight_keys(cfg);
        let prefix = super::detect_weight_prefix(&names, &expected);
        let key = |name: &str| super::prefixed_weight_key(prefix, name);

        let quant = quantization(cfg)?;
        let quant = quant.as_ref();

        let embed_tokens =
            load_dense_or_dequantized(weights, &key("model.embed_tokens.weight"), quant)?;
        let embed_tokens_linear = if cfg.is_quantized() {
            LinearWeight::load(weights, &key("model.embed_tokens.weight"), quant)?
        } else {
            LinearWeight::dense(embed_tokens.clone())?
        };
        let per_layer = if cfg.has_per_layer_inputs() {
            Some(build_per_layer_weights(
                cfg,
                load_dense_or_dequantized(
                    weights,
                    &key("model.embed_tokens_per_layer.weight"),
                    quant,
                )?,
                LinearWeight::load(
                    weights,
                    &key("model.per_layer_model_projection.weight"),
                    quant,
                )?,
                load_rmsnorm(weights, &key("model.per_layer_projection_norm.weight"))?,
                None,
            )?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let base = format!("model.layers.{l}");
            let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
                if cfg.has_per_layer_inputs() {
                    (
                        Some(LinearWeight::load(
                            weights,
                            &key(&format!("{base}.per_layer_input_gate.weight")),
                            quant,
                        )?),
                        Some(LinearWeight::load(
                            weights,
                            &key(&format!("{base}.per_layer_projection.weight")),
                            quant,
                        )?),
                        Some(load_rmsnorm(
                            weights,
                            &key(&format!("{base}.post_per_layer_input_norm.weight")),
                        )?),
                    )
                } else {
                    (None, None, None)
                };
            layers.push(Gemma4Layer {
                input_layernorm: load_rmsnorm(
                    weights,
                    &key(&format!("{base}.input_layernorm.weight")),
                )?,
                q_proj: LinearWeight::load(
                    weights,
                    &key(&format!("{base}.self_attn.q_proj.weight")),
                    quant,
                )?,
                k_proj: LinearWeight::load(
                    weights,
                    &key(&format!("{base}.self_attn.k_proj.weight")),
                    quant,
                )?,
                v_proj: LinearWeight::load(
                    weights,
                    &key(&format!("{base}.self_attn.v_proj.weight")),
                    quant,
                )?,
                o_proj: LinearWeight::load(
                    weights,
                    &key(&format!("{base}.self_attn.o_proj.weight")),
                    quant,
                )?,
                q_norm: load_rmsnorm(weights, &key(&format!("{base}.self_attn.q_norm.weight")))?,
                k_norm: load_rmsnorm(weights, &key(&format!("{base}.self_attn.k_norm.weight")))?,
                post_attention_layernorm: load_rmsnorm(
                    weights,
                    &key(&format!("{base}.post_attention_layernorm.weight")),
                )?,
                pre_feedforward_layernorm: load_rmsnorm(
                    weights,
                    &key(&format!("{base}.pre_feedforward_layernorm.weight")),
                )?,
                post_feedforward_layernorm: load_rmsnorm(
                    weights,
                    &key(&format!("{base}.post_feedforward_layernorm.weight")),
                )?,
                mlp_gate_proj: LinearWeight::load(
                    weights,
                    &key(&format!("{base}.mlp.gate_proj.weight")),
                    quant,
                )?,
                mlp_up_proj: LinearWeight::load(
                    weights,
                    &key(&format!("{base}.mlp.up_proj.weight")),
                    quant,
                )?,
                mlp_down_proj: LinearWeight::load(
                    weights,
                    &key(&format!("{base}.mlp.down_proj.weight")),
                    quant,
                )?,
                layer_scalar: load_optional_array(weights, &key(&format!("{base}.layer_scalar")))?,
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
            });
        }
        let norm = load_rmsnorm(weights, &key("model.norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(LinearWeight::load(weights, &key("lm_head.weight"), quant)?)
        };

        Ok(Gemma4Weights {
            embed_tokens,
            embed_tokens_linear,
            per_layer,
            layers,
            norm,
            lm_head,
            embed_scale: MlxArray::from_slice_f32(&[(cfg.hidden_size as f32).sqrt()], &[1])?,
            final_logit_softcap: cfg
                .final_logit_softcapping
                .map(|cap| MlxArray::from_slice_f32(&[cap], &[1]))
                .transpose()?,
            layer_cache: (0..cfg.num_hidden_layers)
                .map(|layer_idx| {
                    let kind = if cfg.layer_is_global(layer_idx) {
                        Gemma4CacheKind::Global
                    } else {
                        Gemma4CacheKind::Sliding {
                            window: cfg.sliding_window,
                        }
                    };
                    Gemma4LayerCache::new(kind)
                })
                .collect(),
        })
    }

    fn quantization(cfg: &Gemma4Config) -> MlxLlmResult<Option<LinearQuantization>> {
        cfg.quantization
            .as_ref()
            .map(|q| LinearQuantization::new(q.bits, q.group_size, &q.mode))
            .transpose()
    }

    /// Load an RMSNorm scale tensor as stored in MLX SafeTensors.
    ///
    /// Raw HuggingFace Gemma modules apply `(1 + weight)`, but MLX
    /// converted bundles store the effective scale directly. Adding here
    /// corrupts the public MLX 4-bit Gemma 4 bundles.
    fn load_rmsnorm(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<MlxArray> {
        let (floats, shape_i32) = load_rmsnorm_values_and_shape(weights, name)?;
        MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
    }

    fn load_optional_array(
        weights: &SafeTensorBundle,
        name: &str,
    ) -> MlxLlmResult<Option<MlxArray>> {
        if weights.has_tensor(name)? {
            Ok(Some(weights.read_array(name)?))
        } else {
            Ok(None)
        }
    }

    fn load_rmsnorm_values_and_shape(
        weights: &SafeTensorBundle,
        name: &str,
    ) -> MlxLlmResult<(Vec<f32>, Vec<i32>)> {
        weights.read_as_f32(name)
    }

    fn build_per_layer_weights(
        cfg: &Gemma4Config,
        embed_tokens_per_layer: MlxArray,
        per_layer_model_projection: LinearWeight,
        per_layer_projection_norm: MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<Gemma4PerLayerWeights> {
        let hidden_per_layer = cfg.hidden_size_per_layer_input().ok_or_else(|| {
            MlxLlmError::ConfigInvalid("gemma4 missing hidden_size_per_layer_input".into())
        })?;
        let token_scale = MlxArray::from_slice_f32(&[(hidden_per_layer as f32).sqrt()], &[1])?;
        let projection_scale =
            MlxArray::from_slice_f32(&[(cfg.hidden_size as f32).sqrt().recip()], &[1])?;
        let input_scale = MlxArray::from_slice_f32(&[2.0_f32.sqrt().recip()], &[1])?;
        let layer_inputs = split_per_layer_input_weights(
            cfg,
            &embed_tokens_per_layer,
            &per_layer_model_projection,
            &per_layer_projection_norm,
            stream,
        )?;
        Ok(Gemma4PerLayerWeights {
            layer_inputs,
            token_scale,
            projection_scale,
            input_scale,
        })
    }

    fn split_per_layer_input_weights(
        cfg: &Gemma4Config,
        embed_tokens_per_layer: &MlxArray,
        per_layer_model_projection: &LinearWeight,
        per_layer_projection_norm: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<Vec<Gemma4LayerInputWeights>> {
        let hidden_per_layer = cfg.hidden_size_per_layer_input().ok_or_else(|| {
            MlxLlmError::ConfigInvalid("gemma4 missing hidden_size_per_layer_input".into())
        })?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let start = layer_idx.checked_mul(hidden_per_layer).ok_or_else(|| {
                MlxLlmError::ConfigInvalid("gemma4 per-layer slice start overflows usize".into())
            })?;
            let embed_tokens = slice_dim_range(
                embed_tokens_per_layer,
                1,
                start,
                hidden_per_layer,
                "gemma4 embed_tokens_per_layer",
                stream,
            )?
            .contiguous_on_stream(stream)?;
            embed_tokens.eval()?;
            let model_projection =
                per_layer_model_projection.slice_output_rows(start, hidden_per_layer, stream)?;
            let projection_norm = slice_per_layer_projection_norm(
                per_layer_projection_norm,
                layer_idx,
                cfg.num_hidden_layers,
                hidden_per_layer,
                stream,
            )?;
            layers.push(Gemma4LayerInputWeights {
                embed_tokens,
                model_projection,
                projection_norm,
            });
        }
        Ok(layers)
    }

    fn slice_per_layer_projection_norm(
        norm: &MlxArray,
        layer_idx: usize,
        num_layers: usize,
        hidden_per_layer: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = norm.shape();
        match shape.as_slice() {
            [width] => {
                let width = shape_dim_usize("gemma4", "per_layer_projection_norm", *width)?;
                if width != hidden_per_layer {
                    return Err(MlxLlmError::ConfigInvalid(format!(
                        "gemma4 per_layer_projection_norm width is {width}, expected {hidden_per_layer}"
                    )));
                }
                Ok(norm.clone())
            }
            [layers, width] => {
                let layers =
                    shape_dim_usize("gemma4", "per_layer_projection_norm_layers", *layers)?;
                let width = shape_dim_usize("gemma4", "per_layer_projection_norm_width", *width)?;
                if layers != num_layers || width != hidden_per_layer {
                    return Err(MlxLlmError::ConfigInvalid(format!(
                        "gemma4 per_layer_projection_norm shape is {shape:?}, expected [{num_layers}, {hidden_per_layer}]"
                    )));
                }
                let sliced = slice_dim_range(
                    norm,
                    0,
                    layer_idx,
                    1,
                    "gemma4 per_layer_projection_norm",
                    stream,
                )?;
                let reshaped = reshape(
                    &sliced,
                    &[shape_dim_i32(
                        "gemma4",
                        "hidden_size_per_layer_input",
                        hidden_per_layer,
                    )?],
                    stream,
                )?;
                reshaped.eval()?;
                Ok(reshaped)
            }
            _ => Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 per_layer_projection_norm must be rank-1 or rank-2, got shape {shape:?}"
            ))),
        }
    }

    fn slice_dim_range(
        array: &MlxArray,
        axis: usize,
        start: usize,
        width: usize,
        label: &str,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = array.shape();
        if axis >= shape.len() {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "{label} slice axis {axis} exceeds rank {}",
                shape.len()
            )));
        }
        let dim = shape_dim_usize("gemma4", label, shape[axis])?;
        let stop = start.checked_add(width).ok_or_else(|| {
            MlxLlmError::ConfigInvalid(format!("{label} slice stop overflows usize"))
        })?;
        if width == 0 || stop > dim {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "{label} slice [{start}, {stop}) exceeds dimension {dim}"
            )));
        }
        let mut starts = vec![0; shape.len()];
        let mut stops = shape;
        let strides = vec![1; starts.len()];
        starts[axis] = shape_dim_i32("gemma4", "slice_start", start)?;
        stops[axis] = shape_dim_i32("gemma4", "slice_stop", stop)?;
        slice(array, &starts, &stops, &strides, stream).map_err(Into::into)
    }

    /// Forward pass through the whole stack.
    ///
    /// `input_ids` shape: `[batch, seq_len]` (i64/i32). Returns logits of
    /// shape `[batch, seq_len, vocab_size]` (f32). Prefill forwards the
    /// whole prompt and seeds per-layer K/V caches; decode forwards one
    /// token at a non-zero `position_offset` and reads the cached prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        cfg: &Gemma4Config,
        weights: &mut Gemma4Weights,
        input_ids: &MlxArray,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.kv_heads();

        // Embedding lookup: [B, T] -> [B, T, H]. Gemma scales the
        // embedding output by sqrt(hidden_size) immediately after
        // lookup.
        let mut hidden = gather(&weights.embed_tokens, input_ids, 0, stream)?;
        hidden = mul(&hidden, &weights.embed_scale, stream)?;
        let input_embeds = hidden.clone();
        let sliding_mask = prebuild_sliding_window_mask(cfg, input_ids, position_offset)?;

        let layers = &weights.layers;
        let layer_cache = &mut weights.layer_cache;
        let mut shared_sources = vec![false; layers.len()];
        for target_layer in 0..cfg.num_hidden_layers {
            if let Some(source_layer) = cfg.kv_shared_source_layer(target_layer) {
                shared_sources[source_layer] = true;
            }
        }
        let mut call_local_kv: Vec<Option<(MlxArray, MlxArray)>> = vec![None; layers.len()];
        for (layer_idx, layer) in layers.iter().enumerate() {
            let shared_kv = if let Some(source_layer) = cfg.kv_shared_source_layer(layer_idx) {
                if let Some((keys, values)) = call_local_kv[source_layer].as_ref() {
                    Some((keys.clone(), values.clone()))
                } else {
                    Some(layer_cache[source_layer].cloned_kv(source_layer, stream)?)
                }
            } else {
                None
            };
            let per_layer_input = if let Some(per_layer) = weights.per_layer.as_ref() {
                Some(build_per_layer_input(
                    cfg,
                    per_layer,
                    layer_idx,
                    input_ids,
                    &input_embeds,
                    stream,
                )?)
            } else {
                None
            };
            let cache = &mut layer_cache[layer_idx];
            let (next_hidden, own_kv) = transformer_block(
                cfg,
                layer,
                cache,
                &hidden,
                layer_idx,
                n_heads,
                n_kv_heads,
                per_layer_input.as_ref(),
                shared_kv.as_ref(),
                sliding_mask.as_ref(),
                position_offset,
                stream,
            )?;
            hidden = next_hidden;
            if shared_sources[layer_idx] {
                call_local_kv[layer_idx] = own_kv;
            }
        }

        let final_norm = rms_norm(&hidden, Some(&weights.norm), cfg.rms_norm_eps, stream)?;
        let lm_head = weights
            .lm_head
            .as_ref()
            .unwrap_or(&weights.embed_tokens_linear);
        let mut logits = lm_head.forward(&final_norm, stream)?;
        if let Some(cap) = weights.final_logit_softcap.as_ref() {
            logits = mul(&tanh(&div(&logits, cap, stream)?, stream)?, cap, stream)?;
        }
        Ok(logits)
    }

    /// One Gemma transformer block:
    /// attention(pre-norm(x)) → post-attention norm → residual →
    /// MLP(pre-FFN norm(h)) → post-FFN norm → residual.
    #[allow(clippy::too_many_arguments)]
    fn transformer_block(
        cfg: &Gemma4Config,
        layer: &Gemma4Layer,
        cache: &mut Gemma4LayerCache,
        hidden: &MlxArray,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        per_layer_input: Option<&MlxArray>,
        shared_kv: Option<&(MlxArray, MlxArray)>,
        sliding_mask: Option<&MlxArray>,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<(MlxArray, Option<(MlxArray, MlxArray)>)> {
        let attn_in = rms_norm(
            hidden,
            Some(&layer.input_layernorm),
            cfg.rms_norm_eps,
            stream,
        )?;
        let (attn, own_kv) = attention_block(
            cfg,
            layer,
            cache,
            &attn_in,
            layer_idx,
            n_heads,
            n_kv_heads,
            shared_kv,
            sliding_mask,
            position_offset,
            stream,
        )?;
        let attn = rms_norm(
            &attn,
            Some(&layer.post_attention_layernorm),
            cfg.rms_norm_eps,
            stream,
        )?;
        let hidden = add(hidden, &attn, stream)?;

        let mlp_in = rms_norm(
            &hidden,
            Some(&layer.pre_feedforward_layernorm),
            cfg.rms_norm_eps,
            stream,
        )?;
        let mlp = mlp_block(layer, &mlp_in, stream)?;
        let mlp = rms_norm(
            &mlp,
            Some(&layer.post_feedforward_layernorm),
            cfg.rms_norm_eps,
            stream,
        )?;
        let mut hidden = add(&hidden, &mlp, stream)?;

        if let Some(per_layer_input) = per_layer_input {
            hidden = per_layer_block(cfg, layer, &hidden, per_layer_input, stream)?;
        }
        if let Some(layer_scalar) = layer.layer_scalar.as_ref() {
            hidden = mul(&hidden, layer_scalar, stream)?;
        }
        Ok((hidden, own_kv))
    }

    /// Gated-GeLU FFN: `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))`.
    fn mlp_block(
        layer: &Gemma4Layer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let gate = layer.mlp_gate_proj.forward(hidden, stream)?;
        let up = layer.mlp_up_proj.forward(hidden, stream)?;
        let gated = mul(&gelu_tanh(&gate, stream)?, &up, stream)?;
        layer.mlp_down_proj.forward(&gated, stream)
    }

    fn build_per_layer_input(
        cfg: &Gemma4Config,
        weights: &Gemma4PerLayerWeights,
        layer_idx: usize,
        input_ids: &MlxArray,
        inputs_embeds: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let layer_weights = weights.layer_inputs.get(layer_idx).ok_or_else(|| {
            MlxLlmError::ConfigInvalid(format!(
                "gemma4 missing direct per-layer input weights for layer {layer_idx}"
            ))
        })?;
        let mut token_inputs = gather(&layer_weights.embed_tokens, input_ids, 0, stream)?;
        token_inputs = mul(&token_inputs, &weights.token_scale, stream)?;

        let mut projected = layer_weights
            .model_projection
            .forward(inputs_embeds, stream)?;
        projected = mul(&projected, &weights.projection_scale, stream)?;
        projected = rms_norm(
            &projected,
            Some(&layer_weights.projection_norm),
            cfg.rms_norm_eps,
            stream,
        )?;

        let combined = add(&projected, &token_inputs, stream)?;
        mul(&combined, &weights.input_scale, stream).map_err(Into::into)
    }

    #[cfg(test)]
    fn build_per_layer_inputs_reference(
        cfg: &Gemma4Config,
        embed_tokens_per_layer: &MlxArray,
        per_layer_model_projection: &LinearWeight,
        per_layer_projection_norm: &MlxArray,
        scales: &Gemma4PerLayerWeights,
        input_ids: &MlxArray,
        inputs_embeds: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let hidden_per_layer = cfg.hidden_size_per_layer_input().ok_or_else(|| {
            MlxLlmError::ConfigInvalid("gemma4 missing hidden_size_per_layer_input".into())
        })?;
        let shape = input_ids.shape();
        let b = shape[0];
        let t = shape[1];
        let hpl_i32 = shape_dim_i32("gemma4", "hidden_size_per_layer_input", hidden_per_layer)?;
        let layers_i32 = shape_dim_i32("gemma4", "num_hidden_layers", cfg.num_hidden_layers)?;

        let mut token_inputs = gather(embed_tokens_per_layer, input_ids, 0, stream)?;
        token_inputs = mul(&token_inputs, &scales.token_scale, stream)?;
        token_inputs = reshape(&token_inputs, &[b, t, layers_i32, hpl_i32], stream)?;

        let mut projected = per_layer_model_projection.forward(inputs_embeds, stream)?;
        projected = mul(&projected, &scales.projection_scale, stream)?;
        projected = reshape(&projected, &[b, t, layers_i32, hpl_i32], stream)?;
        projected = rms_norm(
            &projected,
            Some(per_layer_projection_norm),
            cfg.rms_norm_eps,
            stream,
        )?;

        let combined = add(&projected, &token_inputs, stream)?;
        mul(&combined, &scales.input_scale, stream).map_err(Into::into)
    }

    fn extract_per_layer_input(
        per_layer_inputs: &MlxArray,
        layer_idx: usize,
        hidden_per_layer: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = per_layer_inputs.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 per-layer inputs must be rank-4, got shape {shape:?}"
            )));
        }
        let b_i32 = shape[0];
        let t_i32 = shape[1];
        let layers = shape_dim_usize("gemma4", "num_hidden_layers", shape[2])?;
        if layer_idx >= layers {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 per-layer input index {layer_idx} >= layer count {layers}"
            )));
        }
        let actual_hpl = shape_dim_usize("gemma4", "hidden_size_per_layer_input", shape[3])?;
        if actual_hpl != hidden_per_layer {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 per-layer input width is {actual_hpl}, expected {hidden_per_layer}"
            )));
        }
        let layer_idx_i32 = shape_dim_i32("gemma4", "per_layer_input_index", layer_idx)?;
        let layer_stop = checked_i32_add("gemma4", "per_layer_input_stop", layer_idx_i32, 1)?;
        let selected = slice(
            per_layer_inputs,
            &[0, 0, layer_idx_i32, 0],
            &[b_i32, t_i32, layer_stop, shape[3]],
            &[1, 1, 1, 1],
            stream,
        )?;
        let selected = reshape(&selected, &[b_i32, t_i32, shape[3]], stream)?;
        let selected = selected.contiguous_on_stream(stream)?;
        // Without this materialization, the Gemma per-layer slice graph can
        // crash inside the native MLX runtime when consumed by the next block.
        selected.eval()?;
        Ok(selected)
    }

    fn per_layer_block(
        cfg: &Gemma4Config,
        layer: &Gemma4Layer,
        hidden: &MlxArray,
        per_layer_input: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let gate = layer
            .per_layer_input_gate
            .as_ref()
            .ok_or_else(|| {
                MlxLlmError::ConfigInvalid("gemma4 missing per_layer_input_gate".into())
            })?
            .forward(hidden, stream)?;
        let mixed = mul(&gelu_tanh(&gate, stream)?, per_layer_input, stream)?;
        let projected = layer
            .per_layer_projection
            .as_ref()
            .ok_or_else(|| {
                MlxLlmError::ConfigInvalid("gemma4 missing per_layer_projection".into())
            })?
            .forward(&mixed, stream)?;
        let normed = rms_norm(
            &projected,
            layer.post_per_layer_input_norm.as_ref(),
            cfg.rms_norm_eps,
            stream,
        )?;
        add(hidden, &normed, stream).map_err(Into::into)
    }

    /// The attention half of one transformer block. Global layers use the
    /// fused causal mask. Sliding-window layers use an explicit additive
    /// mask equivalent to MLX-LM's `create_causal_mask(...,
    /// window_size=sliding_window)`.
    #[allow(clippy::too_many_arguments)]
    fn attention_block(
        cfg: &Gemma4Config,
        layer: &Gemma4Layer,
        cache: &mut Gemma4LayerCache,
        hidden: &MlxArray,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        shared_kv: Option<&(MlxArray, MlxArray)>,
        sliding_mask: Option<&MlxArray>,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<(MlxArray, Option<(MlxArray, MlxArray)>)> {
        let layer_head_dim = head_norm_width(&layer.q_norm)?;
        let k_head_dim = head_norm_width(&layer.k_norm)?;
        if k_head_dim != layer_head_dim {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 q/k norm dimensions differ (q={}, k={})",
                layer_head_dim, k_head_dim
            )));
        }
        let layer_is_global = cfg.layer_is_global(layer_idx);
        let layer_scale = 1.0;

        let hidden = cast(hidden, MlxDtype::F32, stream)?;
        let q = layer.q_proj.forward(&hidden, stream)?;
        let q = device_split_heads_with_rms_norm(
            &q,
            n_heads,
            layer_head_dim,
            Some(&layer.q_norm),
            cfg.rms_norm_eps,
            stream,
        )?;

        let rope_base = if layer_is_global {
            cfg.rope_theta
        } else {
            cfg.rope_local_base_freq
        };
        let rope_dims_i32 = shape_dim_i32("gemma4", "rope_dims", cfg.layer_rope_dims(layer_idx))?;
        let q = rope(
            &q,
            rope_dims_i32,
            false,
            Some(rope_base),
            1.0,
            position_offset,
            None,
            stream,
        )?;

        let (k, v, own_kv) = if let Some((keys, values)) = shared_kv {
            (keys.clone(), values.clone(), None)
        } else {
            let k = layer.k_proj.forward(&hidden, stream)?;
            let v = layer.v_proj.forward(&hidden, stream)?;
            let k = device_split_heads_with_rms_norm(
                &k,
                n_kv_heads,
                layer_head_dim,
                Some(&layer.k_norm),
                cfg.rms_norm_eps,
                stream,
            )?;
            let v = device_split_heads_with_rms_norm(
                &v,
                n_kv_heads,
                layer_head_dim,
                None,
                cfg.rms_norm_eps,
                stream,
            )?;
            let k = rope(
                &k,
                rope_dims_i32,
                false,
                Some(rope_base),
                1.0,
                position_offset,
                None,
                stream,
            )?;
            let max_cache_len = cache.window();
            let k = append_cached_axis2(&mut cache.keys, &k, max_cache_len, stream)?;
            let v = append_cached_axis2(&mut cache.values, &v, max_cache_len, stream)?;
            let own_kv = Some((k.clone(), v.clone()));
            (k, v, own_kv)
        };
        let q_len = q.shape().get(2).copied().unwrap_or(1);

        let attn = if layer_is_global {
            let causal = position_offset == 0 && q_len > 1;
            scaled_dot_product_attention(&q, &k, &v, layer_scale, causal, None, stream)?
        } else if !sliding_attention_needs_mask(cache, q_len) {
            scaled_dot_product_attention(&q, &k, &v, layer_scale, false, None, stream)?
        } else if sliding_window_is_plain_causal(&q, &k, cfg.sliding_window)? {
            scaled_dot_product_attention(&q, &k, &v, layer_scale, true, None, stream)?
        } else if let Some(mask) = sliding_mask {
            if sliding_mask_matches_attention(mask, &q, &k)? {
                scaled_dot_product_attention(&q, &k, &v, layer_scale, false, Some(mask), stream)?
            } else {
                let mask = sliding_window_mask(&q, &k, cfg.sliding_window, position_offset)?;
                scaled_dot_product_attention(&q, &k, &v, layer_scale, false, Some(&mask), stream)?
            }
        } else {
            let mask = sliding_window_mask(&q, &k, cfg.sliding_window, position_offset)?;
            scaled_dot_product_attention(&q, &k, &v, layer_scale, false, Some(&mask), stream)?
        };
        let attn = merge_heads(&attn, n_heads, layer_head_dim, stream)?;
        let out = layer.o_proj.forward(&attn, stream)?;
        Ok((out, own_kv))
    }

    fn head_norm_width(weight: &MlxArray) -> MlxLlmResult<usize> {
        let shape = weight.shape();
        if shape.len() != 1 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 head norm must be rank-1, got shape {shape:?}"
            )));
        }
        shape_dim_usize("gemma4", "head_dim", shape[0])
    }

    fn device_split_heads_with_rms_norm(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        weight: Option<&MlxArray>,
        eps: f32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        if let Some(weight) = weight {
            let actual = head_norm_width(weight)?;
            if actual != head_dim {
                return Err(MlxLlmError::ConfigInvalid(format!(
                    "gemma4 head norm has {actual} values, expected head_dim={head_dim}"
                )));
            }
        }
        let split = split_heads(x, heads, head_dim, stream)?;
        rms_norm(&split, weight, eps, stream).map_err(Into::into)
    }

    #[cfg(test)]
    fn host_split_heads_with_rms_norm(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        weight: Option<&[f32]>,
        eps: f32,
    ) -> MlxLlmResult<MlxArray> {
        if weight.is_some_and(|weight| weight.len() != head_dim) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 head norm has {} values, expected head_dim={head_dim}",
                weight.map_or(0, <[f32]>::len)
            )));
        }
        let shape = x.shape();
        let data = x.to_vec_f32()?;
        let b = shape[0];
        let t = shape[1];
        let bsz = shape_dim_usize("gemma4", "batch", b)?;
        let seq = shape_dim_usize("gemma4", "sequence_len", t)?;
        let width = shape_product_usize("gemma4", "projection_width", &[heads, head_dim])?;
        let actual_width_i32 = shape.get(2).copied().ok_or_else(|| {
            MlxLlmError::ConfigInvalid("gemma4 projection is missing width dimension".into())
        })?;
        let actual_width = shape_dim_usize("gemma4", "projection_width", actual_width_i32)?;
        if actual_width != width {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 projection width is {}, expected heads * head_dim = {width}",
                actual_width_i32
            )));
        }

        let mut out = vec![0.0_f32; data.len()];
        for b in 0..bsz {
            for t in 0..seq {
                let row = ((b * seq) + t) * width;
                for h in 0..heads {
                    let head = row + h * head_dim;
                    let mut mean_square = 0.0_f32;
                    for d in 0..head_dim {
                        let v = data[head + d];
                        mean_square += v * v;
                    }
                    mean_square /= head_dim as f32;
                    let inv_rms = (mean_square + eps).sqrt().recip();
                    for d in 0..head_dim {
                        let src = head + d;
                        let dst = (((b * heads + h) * seq + t) * head_dim) + d;
                        let scale = weight.map_or(1.0, |weight| weight[d]);
                        out[dst] = data[src] * inv_rms * scale;
                    }
                }
            }
        }

        MlxArray::from_slice_f32(
            &out,
            &[
                b,
                shape_dim_i32("gemma4", "num_attention_heads", heads)?,
                t,
                shape_dim_i32("gemma4", "head_dim", head_dim)?,
            ],
        )
        .map_err(Into::into)
    }

    fn append_cached_axis2(
        slot: &mut Option<CachedAxis2>,
        new_slice: &MlxArray,
        max_len: Option<usize>,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = new_slice.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 cache tensors must be rank-4, got shape {shape:?}"
            )));
        }
        let incoming = shape_dim_usize("gemma4", "cache_update_len", shape[2])?;
        let dtype = new_slice.dtype()?;

        if let Some(max_len) = max_len {
            return append_sliding_cached_axis2(slot, new_slice, max_len, incoming, dtype, stream);
        }

        append_global_cached_axis2(slot, new_slice, incoming, dtype, stream)
    }

    fn append_global_cached_axis2(
        slot: &mut Option<CachedAxis2>,
        new_slice: &MlxArray,
        incoming: usize,
        dtype: MlxDtype,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = new_slice.shape();
        let old_len = slot.as_ref().map_or(0, |cache| cache.len);
        let new_len = old_len
            .checked_add(incoming)
            .ok_or_else(|| MlxLlmError::ConfigInvalid("gemma4 K/V cache length overflow".into()))?;

        let mut storage = match slot.take() {
            Some(cache) if new_len <= cache_capacity_axis2(&cache.storage)? => cache.storage,
            Some(cache) => {
                grow_cached_axis2(&cache, shape[0], shape[1], shape[3], new_len, dtype, stream)?
            }
            None => allocate_cached_axis2(shape[0], shape[1], shape[3], new_len, dtype, stream)?,
        };

        storage = slice_update(
            &storage,
            new_slice,
            &[0, 0, shape_dim_i32("gemma4", "cache_old_len", old_len)?, 0],
            &[
                shape[0],
                shape[1],
                shape_dim_i32("gemma4", "cache_new_len", new_len)?,
                shape[3],
            ],
            &[1, 1, 1, 1],
            stream,
        )?;

        let cached = CachedAxis2 {
            storage,
            len: new_len,
        };
        let visible = cache_visible_axis2(&cached, None, stream)?;
        *slot = Some(cached);
        Ok(visible)
    }

    fn append_sliding_cached_axis2(
        slot: &mut Option<CachedAxis2>,
        new_slice: &MlxArray,
        window: usize,
        incoming: usize,
        dtype: MlxDtype,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        if window == 0 {
            return Err(MlxLlmError::ConfigInvalid(
                "gemma4 sliding K/V cache window must be >= 1".into(),
            ));
        }

        match slot.take() {
            Some(cache) if cache.len >= window && incoming == 1 => {
                let (storage, len) = append_buffered_sliding_axis2(
                    cache,
                    new_slice,
                    window,
                    sliding_cache_capacity(window),
                    dtype,
                    stream,
                )?;
                let cached = CachedAxis2 { storage, len };
                let visible = cache_visible_axis2(&cached, Some(window), stream)?;
                *slot = Some(cached);
                Ok(visible)
            }
            Some(cache) => {
                let prefix = cache_visible_axis2(&cache, Some(window), stream)?;
                let combined = concat(&[&prefix, new_slice], 2, stream)?;
                let visible = take_last_axis2(&combined, Some(window), stream)?;
                let stored = materialize_sliding_cached_axis2(&visible, window, dtype, stream)?;
                *slot = Some(stored);
                Ok(visible)
            }
            None => {
                let visible = take_last_axis2(new_slice, Some(window), stream)?;
                let cached = materialize_sliding_cached_axis2(&visible, window, dtype, stream)?;
                *slot = Some(cached);
                Ok(new_slice.clone())
            }
        }
    }

    fn append_buffered_sliding_axis2(
        cache: CachedAxis2,
        new_slice: &MlxArray,
        window: usize,
        target_capacity: usize,
        dtype: MlxDtype,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<(MlxArray, usize)> {
        let shape = new_slice.shape();
        let current_capacity = cache_capacity_axis2(&cache.storage)?;
        let (mut storage, old_len) = if cache.len < current_capacity {
            (cache.storage, cache.len)
        } else {
            let tail = cache_visible_axis2(&cache, Some(window), stream)?;
            let mut storage = allocate_cached_axis2(
                shape[0],
                shape[1],
                shape[3],
                target_capacity,
                dtype,
                stream,
            )?;
            storage = slice_update(
                &storage,
                &tail,
                &[0, 0, 0, 0],
                &[
                    shape[0],
                    shape[1],
                    shape_dim_i32("gemma4", "cache_window", window)?,
                    shape[3],
                ],
                &[1, 1, 1, 1],
                stream,
            )?;
            (storage, window)
        };
        let new_len = old_len.checked_add(1).ok_or_else(|| {
            MlxLlmError::ConfigInvalid("gemma4 sliding K/V cache length overflow".into())
        })?;
        storage = slice_update(
            &storage,
            new_slice,
            &[0, 0, shape_dim_i32("gemma4", "cache_old_len", old_len)?, 0],
            &[
                shape[0],
                shape[1],
                shape_dim_i32("gemma4", "cache_new_len", new_len)?,
                shape[3],
            ],
            &[1, 1, 1, 1],
            stream,
        )?;
        Ok((storage, new_len))
    }

    fn materialize_sliding_cached_axis2(
        visible: &MlxArray,
        window: usize,
        dtype: MlxDtype,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<CachedAxis2> {
        let len = shape_dim_usize("gemma4", "sliding_cache_visible_len", visible.shape()[2])?;
        if len < window {
            return materialize_cached_axis2(visible, stream);
        }
        materialize_cached_axis2_with_capacity(
            visible,
            sliding_cache_capacity(window),
            dtype,
            stream,
        )
    }

    fn sliding_cache_capacity(window: usize) -> usize {
        window + GEMMA4_SLIDING_CACHE_BUFFER.min(window).max(1)
    }

    fn materialize_cached_axis2(
        visible: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<CachedAxis2> {
        let visible = visible.contiguous_on_stream(stream)?;
        let len = shape_dim_usize("gemma4", "cache_visible_len", visible.shape()[2])?;
        Ok(CachedAxis2 {
            storage: visible,
            len,
        })
    }

    fn materialize_cached_axis2_with_capacity(
        visible: &MlxArray,
        capacity: usize,
        dtype: MlxDtype,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<CachedAxis2> {
        let shape = visible.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 cache tensors must be rank-4, got shape {shape:?}"
            )));
        }
        let len = shape_dim_usize("gemma4", "cache_visible_len", shape[2])?;
        if len > capacity {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 cache len {len} exceeds requested capacity {capacity}"
            )));
        }
        let mut storage =
            allocate_cached_axis2(shape[0], shape[1], shape[3], capacity, dtype, stream)?;
        storage = slice_update(
            &storage,
            visible,
            &[0, 0, 0, 0],
            &[
                shape[0],
                shape[1],
                shape_dim_i32("gemma4", "cache_visible_len", len)?,
                shape[3],
            ],
            &[1, 1, 1, 1],
            stream,
        )?;
        Ok(CachedAxis2 { storage, len })
    }

    fn grow_cached_axis2(
        cache: &CachedAxis2,
        b: i32,
        h: i32,
        d: i32,
        required_len: usize,
        dtype: MlxDtype,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let current_capacity = cache_capacity_axis2(&cache.storage)?;
        let capacity = next_cache_capacity(current_capacity, required_len)?;
        let mut storage = allocate_cached_axis2(b, h, d, capacity, dtype, stream)?;
        let visible = cache_visible_axis2(cache, None, stream)?;
        storage = slice_update(
            &storage,
            &visible,
            &[0, 0, 0, 0],
            &[b, h, shape_dim_i32("gemma4", "cache_len", cache.len)?, d],
            &[1, 1, 1, 1],
            stream,
        )?;
        Ok(storage)
    }

    fn allocate_cached_axis2(
        b: i32,
        h: i32,
        d: i32,
        capacity: usize,
        dtype: MlxDtype,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let capacity_i32 = shape_dim_i32("gemma4", "cache_capacity", capacity)?;
        shape_product_usize(
            "gemma4",
            "cache_allocation_len",
            &[
                shape_dim_usize("gemma4", "cache_batch", b)?,
                shape_dim_usize("gemma4", "cache_heads", h)?,
                capacity,
                shape_dim_usize("gemma4", "cache_head_dim", d)?,
            ],
        )?;
        zeros(&[b, h, capacity_i32, d], dtype, stream).map_err(Into::into)
    }

    fn next_cache_capacity(current_capacity: usize, required_len: usize) -> MlxLlmResult<usize> {
        let step = GEMMA4_SLIDING_CACHE_BUFFER;
        let required_len = required_len.max(1);
        let step_capacity = required_len
            .div_ceil(step)
            .checked_mul(step)
            .ok_or_else(|| MlxLlmError::ConfigInvalid("gemma4 cache capacity overflow".into()))?;
        let doubled_capacity = current_capacity.saturating_mul(2);
        Ok(step_capacity.max(doubled_capacity).max(required_len))
    }

    fn cache_capacity_axis2(storage: &MlxArray) -> MlxLlmResult<usize> {
        let shape = storage.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 cache storage must be rank-4, got shape {shape:?}"
            )));
        }
        shape_dim_usize("gemma4", "cache_capacity", shape[2])
    }

    fn cache_visible_axis2(
        cache: &CachedAxis2,
        max_len: Option<usize>,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = cache.storage.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 cache storage must be rank-4, got shape {shape:?}"
            )));
        }
        let visible_len = max_len.map_or(cache.len, |max_len| cache.len.min(max_len));
        let start = cache.len - visible_len;
        slice(
            &cache.storage,
            &[
                0,
                0,
                shape_dim_i32("gemma4", "cache_visible_start", start)?,
                0,
            ],
            &[
                shape[0],
                shape[1],
                shape_dim_i32("gemma4", "cache_visible_stop", cache.len)?,
                shape[3],
            ],
            &[1, 1, 1, 1],
            stream,
        )
        .map_err(Into::into)
    }

    fn take_last_axis2(
        array: &MlxArray,
        max_len: Option<usize>,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let Some(max_len) = max_len else {
            return Ok(array.clone());
        };
        let shape = array.shape();
        if shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 cache tensors must be rank-4, got shape {shape:?}"
            )));
        }
        let len = shape_dim_usize("gemma4", "cache_axis2", shape[2])?;
        if len <= max_len {
            return Ok(array.clone());
        }
        let start = shape_dim_i32("gemma4", "cache_start", len - max_len)?;
        slice(
            array,
            &[0, 0, start, 0],
            &[shape[0], shape[1], shape[2], shape[3]],
            &[1, 1, 1, 1],
            stream,
        )
        .map_err(Into::into)
    }

    fn sliding_attention_needs_mask(cache: &Gemma4LayerCache, q_len: i32) -> bool {
        !(cache.is_sliding() && q_len == 1)
    }

    fn prebuild_sliding_window_mask(
        cfg: &Gemma4Config,
        input_ids: &MlxArray,
        position_offset: i32,
    ) -> MlxLlmResult<Option<MlxArray>> {
        let shape = input_ids.shape();
        let Some(&q_len_i32) = shape.get(1) else {
            return Ok(None);
        };
        let q_len = shape_dim_usize("gemma4", "query_len", q_len_i32)?;
        if q_len <= 1 || q_len <= cfg.sliding_window {
            return Ok(None);
        }
        let has_sliding_layers = (0..cfg.num_hidden_layers).any(|idx| !cfg.layer_is_global(idx));
        if !has_sliding_layers {
            return Ok(None);
        }
        sliding_window_mask_from_lengths(q_len_i32, q_len_i32, cfg.sliding_window, position_offset)
            .map(Some)
    }

    fn sliding_window_is_plain_causal(
        queries: &MlxArray,
        keys: &MlxArray,
        window_size: usize,
    ) -> MlxLlmResult<bool> {
        let q_len = shape_dim_usize("gemma4", "query_len", queries.shape()[2])?;
        let k_len = shape_dim_usize("gemma4", "key_len", keys.shape()[2])?;
        Ok(q_len == k_len && q_len <= window_size)
    }

    fn sliding_mask_matches_attention(
        mask: &MlxArray,
        queries: &MlxArray,
        keys: &MlxArray,
    ) -> MlxLlmResult<bool> {
        let mask_shape = mask.shape();
        let query_shape = queries.shape();
        let key_shape = keys.shape();
        if mask_shape.len() != 4 || query_shape.len() != 4 || key_shape.len() != 4 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 sliding attention expects rank-4 mask/query/key, got mask={mask_shape:?} q={query_shape:?} k={key_shape:?}"
            )));
        }
        Ok(mask_shape[2] == query_shape[2] && mask_shape[3] == key_shape[2])
    }

    fn sliding_window_mask(
        queries: &MlxArray,
        keys: &MlxArray,
        window_size: usize,
        position_offset: i32,
    ) -> MlxLlmResult<MlxArray> {
        let q_len_i32 = queries.shape()[2];
        let k_len_i32 = keys.shape()[2];
        sliding_window_mask_from_lengths(q_len_i32, k_len_i32, window_size, position_offset)
    }

    fn sliding_window_mask_from_lengths(
        q_len_i32: i32,
        k_len_i32: i32,
        window_size: usize,
        position_offset: i32,
    ) -> MlxLlmResult<MlxArray> {
        let q_len = shape_dim_usize("gemma4", "query_len", q_len_i32)?;
        let k_len = shape_dim_usize("gemma4", "key_len", k_len_i32)?;
        let mask_len = shape_product_usize("gemma4", "sliding_window_mask_len", &[q_len, k_len])?;
        let window_size_i32 = shape_dim_i32("gemma4", "sliding_window", window_size)?;
        let visible_end =
            checked_i32_add("gemma4", "visible_sequence_end", position_offset, q_len_i32)?;
        let key_position_offset = visible_end.checked_sub(k_len_i32).ok_or_else(|| {
            MlxLlmError::ConfigInvalid(format!(
                "gemma4 key length {k_len_i32} exceeds visible sequence end {visible_end}"
            ))
        })?;
        let mut mask = Vec::with_capacity(mask_len);
        for qi in 0..q_len {
            let qi_i32 = shape_dim_i32("gemma4", "query_index", qi)?;
            let q_pos = checked_i32_add("gemma4", "query_position", position_offset, qi_i32)?;
            for kj in 0..k_len {
                let kj_i32 = shape_dim_i32("gemma4", "key_index", kj)?;
                let k_pos = checked_i32_add("gemma4", "key_position", key_position_offset, kj_i32)?;
                let window_end =
                    checked_i32_add("gemma4", "sliding_window_end", k_pos, window_size_i32)?;
                let allowed = k_pos <= q_pos && q_pos < window_end;
                mask.push(if allowed { 0.0 } else { -1.0e9 });
            }
        }
        MlxArray::from_slice_f32(&mask, &[1, 1, q_len_i32, k_len_i32]).map_err(Into::into)
    }

    /// Reshape `[B, T, heads * head_dim]` into `[B, heads, T, head_dim]`.
    pub(crate) fn split_heads(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let reshaped = reshape_heads(x, heads, head_dim, stream)?;
        transpose(&reshaped, &[0, 2, 1, 3], stream).map_err(Into::into)
    }

    fn reshape_heads(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = x.shape();
        let b = shape[0];
        let t = shape[1];
        let heads_i32 = shape_dim_i32("gemma4", "num_attention_heads", heads)?;
        let head_dim_i32 = shape_dim_i32("gemma4", "head_dim", head_dim)?;
        reshape(x, &[b, t, heads_i32, head_dim_i32], stream).map_err(Into::into)
    }

    /// Reshape `[B, heads, T, head_dim]` into `[B, T, heads * head_dim]`.
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
        let width = shape_product_i32("gemma4", "merged_attention_width", &[heads, head_dim])?;
        reshape(&transposed, &[b, t, width], stream).map_err(Into::into)
    }

    #[cfg(test)]
    mod runtime_tests {
        use super::*;

        fn assert_close(got: &[f32], expected: &[f32], tol: f32, ctx: &str) {
            assert_eq!(
                got.len(),
                expected.len(),
                "{ctx}: length mismatch got={} expected={}",
                got.len(),
                expected.len()
            );
            for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                let diff = (g - e).abs();
                assert!(
                    diff <= tol,
                    "{ctx}: element {idx} differs got={g} expected={e} diff={diff}"
                );
            }
        }

        #[test]
        fn device_split_heads_rms_norm_matches_host_reference() {
            let x = MlxArray::from_slice_f32(
                &[
                    1.0, 2.0, 3.0, 4.0, //
                    5.0, 6.0, 7.0, 8.0,
                ],
                &[1, 2, 4],
            )
            .unwrap();
            let host = host_split_heads_with_rms_norm(&x, 2, 2, Some(&[1.0, 0.5]), 1.0e-5).unwrap();
            let weight = MlxArray::from_slice_f32(&[1.0, 0.5], &[2]).unwrap();

            let device =
                device_split_heads_with_rms_norm(&x, 2, 2, Some(&weight), 1.0e-5, None).unwrap();

            assert_eq!(device.shape(), host.shape());
            assert_close(
                &device.to_vec_f32().unwrap(),
                &host.to_vec_f32().unwrap(),
                1.0e-5,
                "device split+rms_norm",
            );
        }

        #[test]
        fn device_split_heads_rms_norm_without_weight_matches_host_reference() {
            let x = MlxArray::from_slice_f32(
                &[
                    1.0, 2.0, 3.0, 4.0, //
                    5.0, 6.0, 7.0, 8.0,
                ],
                &[1, 2, 4],
            )
            .unwrap();
            let host = host_split_heads_with_rms_norm(&x, 2, 2, None, 1.0e-5).unwrap();

            let device = device_split_heads_with_rms_norm(&x, 2, 2, None, 1.0e-5, None).unwrap();

            assert_eq!(device.shape(), host.shape());
            assert_close(
                &device.to_vec_f32().unwrap(),
                &host.to_vec_f32().unwrap(),
                1.0e-5,
                "device split+rms_norm without weight",
            );
        }

        #[test]
        fn extract_per_layer_input_uses_device_slice() {
            let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
            let per_layer = MlxArray::from_slice_f32(&data, &[1, 2, 3, 4]).unwrap();

            let got = extract_per_layer_input(&per_layer, 1, 4, None).unwrap();

            assert_eq!(got.shape(), vec![1, 2, 4]);
            assert_close(
                &got.to_vec_f32().unwrap(),
                &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0],
                1.0e-6,
                "per-layer input slice",
            );
        }

        #[test]
        fn direct_per_layer_input_matches_combined_reference() {
            let cfg = runtime_dummy_config_with_per_layer_inputs();
            let embed_tokens_per_layer = MlxArray::from_slice_f32(
                &[
                    1.0, 2.0, 3.0, 4.0, //
                    5.0, 6.0, 7.0, 8.0, //
                    9.0, 10.0, 11.0, 12.0,
                ],
                &[3, 4],
            )
            .unwrap();
            let embed_tokens_per_layer_ref = embed_tokens_per_layer.clone();
            let projection_weight_data = [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ];
            let per_layer_model_projection = LinearWeight::dense(
                MlxArray::from_slice_f32(&projection_weight_data, &[4, 4]).unwrap(),
            )
            .unwrap();
            let per_layer_model_projection_ref = LinearWeight::dense(
                MlxArray::from_slice_f32(&projection_weight_data, &[4, 4]).unwrap(),
            )
            .unwrap();
            let per_layer_projection_norm = MlxArray::from_slice_f32(&[1.0, 1.0], &[2]).unwrap();
            let per_layer_projection_norm_ref = per_layer_projection_norm.clone();
            let weights = build_per_layer_weights(
                &cfg,
                embed_tokens_per_layer,
                per_layer_model_projection,
                per_layer_projection_norm,
                None,
            )
            .unwrap();
            let input_ids = MlxArray::from_slice_i32(&[0, 1], &[1, 2]).unwrap();
            let inputs_embeds = MlxArray::from_slice_f32(
                &[
                    0.5, 1.0, 1.5, 2.0, //
                    2.5, 3.0, 3.5, 4.0,
                ],
                &[1, 2, 4],
            )
            .unwrap();

            let combined = build_per_layer_inputs_reference(
                &cfg,
                &embed_tokens_per_layer_ref,
                &per_layer_model_projection_ref,
                &per_layer_projection_norm_ref,
                &weights,
                &input_ids,
                &inputs_embeds,
                None,
            )
            .unwrap();
            let extracted = extract_per_layer_input(&combined, 1, 2, None).unwrap();
            let direct =
                build_per_layer_input(&cfg, &weights, 1, &input_ids, &inputs_embeds, None).unwrap();

            assert_eq!(direct.shape(), extracted.shape());
            assert_close(
                &direct.to_vec_f32().unwrap(),
                &extracted.to_vec_f32().unwrap(),
                1.0e-5,
                "direct per-layer input",
            );
        }

        fn runtime_dummy_config_with_per_layer_inputs() -> Gemma4Config {
            Gemma4Config {
                model_type: "gemma4".into(),
                hidden_size: 4,
                num_hidden_layers: 2,
                num_attention_heads: 2,
                num_key_value_heads: Some(1),
                intermediate_size: 8,
                vocab_size: 16,
                max_position_embeddings: 128,
                rope_theta: 1_000_000.0,
                rope_local_base_freq: 10_000.0,
                rms_norm_eps: 1.0e-6,
                tie_word_embeddings: true,
                head_dim: Some(2),
                global_head_dim: None,
                layer_types: None,
                sliding_window: 16,
                sliding_window_pattern: 2,
                query_pre_attn_scalar: None,
                final_logit_softcapping: None,
                hidden_size_per_layer_input: Some(2),
                vocab_size_per_layer_input: Some(3),
                num_kv_shared_layers: None,
                full_attention_partial_rotary_factor: 0.25,
                quantization: None,
            }
        }

        #[test]
        fn sliding_cache_prefill_returns_full_attention_and_stores_tail() {
            let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
            let prefill = MlxArray::from_slice_f32(&data, &[1, 1, 5, 2]).unwrap();
            let mut cache = Gemma4LayerCache::new(Gemma4CacheKind::Sliding { window: 3 });

            let max_cache_len = cache.window();
            let visible =
                append_cached_axis2(&mut cache.keys, &prefill, max_cache_len, None).unwrap();

            assert_eq!(visible.shape(), vec![1, 1, 5, 2]);
            assert_close(
                &visible.to_vec_f32().unwrap(),
                &data,
                1.0e-6,
                "prefill visible cache",
            );
            let stored = cache.keys.as_ref().expect("stored sliding cache tail");
            assert_eq!(stored.len, 3);
            assert_eq!(stored.storage.shape(), vec![1, 1, 6, 2]);
            let stored_visible = cache_visible_axis2(stored, Some(3), None).unwrap();
            assert_close(
                &stored_visible.to_vec_f32().unwrap(),
                &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                1.0e-6,
                "stored sliding cache tail",
            );
        }

        #[test]
        fn cloned_sliding_kv_exposes_stored_tail_only() {
            let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
            let prefill = MlxArray::from_slice_f32(&data, &[1, 1, 5, 2]).unwrap();
            let mut cache = Gemma4LayerCache::new(Gemma4CacheKind::Sliding { window: 3 });

            let max_cache_len = cache.window();
            let _ = append_cached_axis2(&mut cache.keys, &prefill, max_cache_len, None).unwrap();
            let _ = append_cached_axis2(&mut cache.values, &prefill, max_cache_len, None).unwrap();
            let (keys, values) = cache.cloned_kv(0, None).unwrap();

            assert_eq!(keys.shape(), vec![1, 1, 3, 2]);
            assert_eq!(values.shape(), vec![1, 1, 3, 2]);
            assert_close(
                &keys.to_vec_f32().unwrap(),
                &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                1.0e-6,
                "cloned sliding key tail",
            );
        }

        #[test]
        fn sliding_cache_decode_keeps_only_window() {
            let seed =
                MlxArray::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 3, 2]).unwrap();
            let update = MlxArray::from_slice_f32(&[6.0, 7.0], &[1, 1, 1, 2]).unwrap();
            let mut cache = Gemma4LayerCache::new(Gemma4CacheKind::Sliding { window: 3 });
            cache.keys = Some(materialize_cached_axis2(&seed, None).unwrap());

            let max_cache_len = cache.window();
            let visible =
                append_cached_axis2(&mut cache.keys, &update, max_cache_len, None).unwrap();

            assert_eq!(visible.shape(), vec![1, 1, 3, 2]);
            assert_close(
                &visible.to_vec_f32().unwrap(),
                &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                1.0e-6,
                "decode visible cache",
            );
            let stored = cache.keys.as_ref().expect("stored sliding decode cache");
            assert_eq!(stored.len, 4);
            assert_eq!(stored.storage.shape(), vec![1, 1, 6, 2]);
            let stored_visible = cache_visible_axis2(stored, Some(3), None).unwrap();
            assert_close(
                &stored_visible.to_vec_f32().unwrap(),
                &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                1.0e-6,
                "stored sliding visible cache",
            );
        }

        #[test]
        fn sliding_decode_with_bounded_cache_needs_no_mask() {
            let sliding = Gemma4LayerCache::new(Gemma4CacheKind::Sliding { window: 512 });
            let global = Gemma4LayerCache::new(Gemma4CacheKind::Global);

            assert!(!sliding_attention_needs_mask(&sliding, 1));
            assert!(sliding_attention_needs_mask(&sliding, 2));
            assert!(sliding_attention_needs_mask(&global, 1));
        }

        #[test]
        fn global_cache_capacity_grows_geometrically() {
            assert_eq!(
                next_cache_capacity(0, 17).unwrap(),
                GEMMA4_SLIDING_CACHE_BUFFER
            );
            assert_eq!(next_cache_capacity(256, 257).unwrap(), 512);
            assert_eq!(next_cache_capacity(512, 513).unwrap(), 1024);
            assert_eq!(next_cache_capacity(1024, 1800).unwrap(), 2048);
        }

        #[test]
        fn global_cache_capacity_rejects_overflow() {
            assert!(next_cache_capacity(0, usize::MAX).is_err());
        }

        #[test]
        fn short_sliding_prefill_can_use_causal_attention() {
            let q = MlxArray::from_slice_f32(&[0.0; 6], &[1, 1, 3, 2]).unwrap();
            let same_len_k = MlxArray::from_slice_f32(&[0.0; 6], &[1, 1, 3, 2]).unwrap();
            let longer_k = MlxArray::from_slice_f32(&[0.0; 8], &[1, 1, 4, 2]).unwrap();

            assert!(sliding_window_is_plain_causal(&q, &same_len_k, 3).unwrap());
            assert!(!sliding_window_is_plain_causal(&q, &same_len_k, 2).unwrap());
            assert!(!sliding_window_is_plain_causal(&q, &longer_k, 4).unwrap());
        }

        #[test]
        fn prebuilt_sliding_mask_must_match_visible_key_length() {
            let q = MlxArray::from_slice_f32(&[0.0; 40], &[1, 2, 5, 4]).unwrap();
            let tail_k = MlxArray::from_slice_f32(&[0.0; 12], &[1, 1, 3, 4]).unwrap();
            let prebuilt = sliding_window_mask_from_lengths(5, 5, 3, 0).unwrap();

            assert_eq!(prebuilt.shape(), vec![1, 1, 5, 5]);
            assert!(!sliding_mask_matches_attention(&prebuilt, &q, &tail_k).unwrap());

            let fallback = sliding_window_mask(&q, &tail_k, 3, 0).unwrap();
            assert_eq!(fallback.shape(), vec![1, 1, 5, 3]);
            assert!(sliding_mask_matches_attention(&fallback, &q, &tail_k).unwrap());
            assert_close(
                &fallback.to_vec_f32().unwrap(),
                &[
                    -1.0e9, -1.0e9, -1.0e9, //
                    -1.0e9, -1.0e9, -1.0e9, //
                    0.0, -1.0e9, -1.0e9, //
                    0.0, 0.0, -1.0e9, //
                    0.0, 0.0, 0.0,
                ],
                1.0e-3,
                "tail sliding mask",
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
        }
    }

    #[test]
    fn nested_text_config_parses_with_root_model_type() {
        let raw = serde_json::json!({
            "model_type": "gemma4",
            "text_config": {
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 100,
                "max_position_embeddings": 1024,
                "head_dim": 4
            }
        });
        let cfg = Gemma4Config::parse_json(&raw.to_string()).expect("parse nested config");
        assert_eq!(cfg.model_type, "gemma4");
        assert_eq!(cfg.hidden_size, 16);
        assert_eq!(cfg.intermediate_size, 32);
        assert_eq!(cfg.kv_heads(), 2);
    }

    #[test]
    fn nested_text_config_preserves_root_quantization() {
        let raw = serde_json::json!({
            "model_type": "gemma4",
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine"
            },
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 100,
                "max_position_embeddings": 1024,
                "head_dim": 4
            }
        });
        let cfg = Gemma4Config::parse_json(&raw.to_string()).expect("parse nested config");

        assert_eq!(cfg.model_type, "gemma4");
        let quant = cfg
            .quantization
            .as_ref()
            .expect("root quantization should be preserved");
        assert_eq!(quant.bits, 4);
        assert_eq!(quant.group_size, 64);
        assert!(cfg.is_quantized());
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
        cfg.query_pre_attn_scalar = Some(64.0);
        assert!((cfg.query_scale() - 0.125).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_pattern_marks_global_layers() {
        let cfg = dummy_config(12, true); // default pattern = 6.
        assert!(!cfg.layer_is_global(0));
        assert!(!cfg.layer_is_global(1));
        assert!(cfg.layer_is_global(5));
        assert!(!cfg.layer_is_global(6));
        assert!(cfg.layer_is_global(11));
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
        cfg.sliding_window = i32::MAX as usize + 1;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
                assert!(msg.contains("sliding_window"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn oversized_projection_width_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = Some(1);
        cfg.head_dim = Some(i32::MAX as usize / 2 + 1);
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
                assert!(msg.contains("attention_projection_width"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn zero_sliding_window_rejected() {
        let mut cfg = dummy_config(1, true);
        cfg.sliding_window = 0;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("sliding_window"), "got: {msg}")
            }
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
    fn layer_types_length_must_match_layer_count() {
        let mut cfg = dummy_config(3, true);
        cfg.layer_types = Some(vec!["sliding_attention".into(), "full_attention".into()]);

        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("layer_types"), "got: {msg}");
                assert!(msg.contains("2 entries"), "got: {msg}");
                assert!(msg.contains("num_hidden_layers=3"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn layer_types_reject_unknown_values() {
        let mut cfg = dummy_config(2, true);
        cfg.layer_types = Some(vec!["sliding_attention".into(), "global_attention".into()]);

        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("layer_types[1]"), "got: {msg}");
                assert!(msg.contains("global_attention"), "got: {msg}");
                assert!(msg.contains("full_attention"), "got: {msg}");
                assert!(msg.contains("sliding_attention"), "got: {msg}");
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

    #[test]
    fn validate_safetensors_accepts_language_model_prefixed_manifest() {
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
                    format!("{LANGUAGE_MODEL_PREFIX}{k}"),
                    TensorView::new(Dtype::F32, vec![1], &data).unwrap(),
                )
            })
            .collect();
        let path = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, &None, &path).unwrap();

        validate_safetensors(&path, &cfg).expect("prefixed manifest should validate");
    }

    #[test]
    fn gemma4_quantized_manifest_accepts_language_model_prefixed_triplets() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        let mut cfg = dummy_config(1, true);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
            mode: "affine".into(),
        });
        let keys = expected_weight_keys(&cfg);
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.self_attn.q_proj.scales"));
        assert!(keys.iter().any(|k| k == "model.embed_tokens.biases"));
        let data = vec![0u8; 4];
        let tensors: Vec<(String, TensorView<'_>)> = keys
            .iter()
            .map(|k| {
                (
                    format!("{LANGUAGE_MODEL_PREFIX}{k}"),
                    TensorView::new(Dtype::F32, vec![1], &data).unwrap(),
                )
            })
            .collect();
        let path = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, &None, &path).unwrap();

        validate_safetensors(&path, &cfg).expect("quantized prefixed manifest should validate");
    }
}

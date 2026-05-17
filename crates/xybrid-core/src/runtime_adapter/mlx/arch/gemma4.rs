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
//! 5. **Per-head RMSNorm on Q and K** (`q_norm` / `k_norm`): applied
//!    before RoPE, same shape as Qwen 3's but with the `(1 + weight)`
//!    Gemma scaling.
//! 6. **Query pre-attention scalar**: Gemma 4 scales the query projection
//!    by an explicit `query_pre_attn_scalar` (usually `head_dim ** -0.5`
//!    collapsed into the SDPA `scale` argument).
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
//! Reference (upstream): <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3.py>
//! (Gemma 4 currently mirrors Gemma 3's text-only topology on the MLX-LM
//! side; the family ID is `"gemma4"` in the config header).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};
use super::super::weights::SafeTensorBundle;

const LANGUAGE_MODEL_PREFIX: &str = "language_model.";

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
    /// Scalar whose inverse square root is used as the attention scale.
    /// Upstream Gemma stores `head_dim` here and computes
    /// `query_pre_attn_scalar ** -0.5`.
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
        l % self.sliding_window_pattern == self.sliding_window_pattern - 1
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
/// bail here with a pointed error so the runtime selector (US-016) can
/// fall back to llama.cpp.
pub fn validate_safetensors(path: &Path, cfg: &Gemma4Config) -> MlxLlmResult<()> {
    let weights = SafeTensorBundle::from_single_file(path.to_path_buf());
    validate_safetensors_bundle(&weights, cfg)
}

pub fn validate_safetensors_bundle(
    weights: &SafeTensorBundle,
    cfg: &Gemma4Config,
) -> MlxLlmResult<()> {
    if cfg.is_quantized() {
        let bits = cfg.quantization.as_ref().map(|q| q.bits).unwrap_or(0);
        let gs = cfg.quantization.as_ref().map(|q| q.group_size).unwrap_or(0);
        return Err(MlxLlmError::UnsupportedQuantization {
            model_type: cfg.model_type.clone(),
            bits,
            group_size: gs,
            reason: "mlx_fast_quantized_matmul is not wired for Gemma yet",
        });
    }

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
    use super::super::{
        checked_i32_add, shape_dim_i32, shape_dim_usize, shape_product_i32, shape_product_usize,
    };
    use super::Gemma4Config;

    use xybrid_mlx::ops::{
        add, cast, concat, gather, gelu_tanh, matmul, mul, reshape, rms_norm, rope,
        scaled_dot_product_attention, transpose,
    };
    use xybrid_mlx::{MlxArray, MlxDtype, MlxStream};

    /// One transformer block's weights.
    ///
    /// Projection/FFN weights are materialised as [`MlxArray`]s. Q/K
    /// per-head norm vectors stay host-side because the public Gemma 4
    /// BF16 bundle mixes 256-wide local layers with 512-wide global layers,
    /// and the current MLX fused RMSNorm/reshape path is not reliable for
    /// that head-prep graph.
    ///
    /// RMSNorm scale weights have already been transformed into
    /// `weight + 1.0` (see [`load_rmsnorm_plus_one`]) so the MLX kernel
    /// sees a single scale factor where it is used.
    #[derive(Debug)]
    pub struct Gemma4Layer {
        pub input_layernorm: MlxArray,
        pub q_proj: MlxArray,
        pub k_proj: MlxArray,
        pub v_proj: MlxArray,
        pub o_proj: MlxArray,
        pub q_norm: Vec<f32>,
        pub k_norm: Vec<f32>,
        pub post_attention_layernorm: MlxArray,
        pub pre_feedforward_layernorm: MlxArray,
        pub post_feedforward_layernorm: MlxArray,
        pub mlp_gate_proj: MlxArray,
        pub mlp_up_proj: MlxArray,
        pub mlp_down_proj: MlxArray,
    }

    /// Resident K/V tensors for one Gemma decoder layer during a generation
    /// call. Reset before prefill; appended one token at a time during decode.
    #[derive(Debug, Default)]
    pub struct Gemma4LayerCache {
        keys: Option<MlxArray>,
        values: Option<MlxArray>,
    }

    impl Gemma4LayerCache {
        fn reset(&mut self) {
            self.keys = None;
            self.values = None;
        }
    }

    /// Full Gemma 4 weight set resident in MLX memory.
    #[derive(Debug)]
    pub struct Gemma4Weights {
        pub embed_tokens: MlxArray,
        pub layers: Vec<Gemma4Layer>,
        pub norm: MlxArray,
        /// `None` when [`Gemma4Config::tie_word_embeddings`] is set.
        pub lm_head: Option<MlxArray>,
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

        let embed_tokens = load_tensor(weights, &key("model.embed_tokens.weight"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let base = format!("model.layers.{l}");
            layers.push(Gemma4Layer {
                input_layernorm: load_rmsnorm_plus_one(
                    weights,
                    &key(&format!("{base}.input_layernorm.weight")),
                )?,
                q_proj: load_tensor_f32(weights, &key(&format!("{base}.self_attn.q_proj.weight")))?,
                k_proj: load_tensor_f32(weights, &key(&format!("{base}.self_attn.k_proj.weight")))?,
                v_proj: load_tensor_f32(weights, &key(&format!("{base}.self_attn.v_proj.weight")))?,
                o_proj: load_tensor(weights, &key(&format!("{base}.self_attn.o_proj.weight")))?,
                q_norm: load_rmsnorm_plus_one_values(
                    weights,
                    &key(&format!("{base}.self_attn.q_norm.weight")),
                )?,
                k_norm: load_rmsnorm_plus_one_values(
                    weights,
                    &key(&format!("{base}.self_attn.k_norm.weight")),
                )?,
                post_attention_layernorm: load_rmsnorm_plus_one(
                    weights,
                    &key(&format!("{base}.post_attention_layernorm.weight")),
                )?,
                pre_feedforward_layernorm: load_rmsnorm_plus_one(
                    weights,
                    &key(&format!("{base}.pre_feedforward_layernorm.weight")),
                )?,
                post_feedforward_layernorm: load_rmsnorm_plus_one(
                    weights,
                    &key(&format!("{base}.post_feedforward_layernorm.weight")),
                )?,
                mlp_gate_proj: load_tensor(weights, &key(&format!("{base}.mlp.gate_proj.weight")))?,
                mlp_up_proj: load_tensor(weights, &key(&format!("{base}.mlp.up_proj.weight")))?,
                mlp_down_proj: load_tensor(weights, &key(&format!("{base}.mlp.down_proj.weight")))?,
            });
        }
        let norm = load_rmsnorm_plus_one(weights, &key("model.norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_tensor(weights, &key("lm_head.weight"))?)
        };

        Ok(Gemma4Weights {
            embed_tokens,
            layers,
            norm,
            lm_head,
            layer_cache: std::iter::repeat_with(Gemma4LayerCache::default)
                .take(cfg.num_hidden_layers)
                .collect(),
        })
    }

    /// Read one tensor from a SafeTensors view into an [`MlxArray`].
    fn load_tensor(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<MlxArray> {
        weights.read_array(name)
    }

    fn load_tensor_f32(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<MlxArray> {
        let (floats, shape_i32) = weights.read_as_f32(name)?;
        MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
    }

    /// Gemma RMSNorm weights are used as `(1 + weight)`. Bake the
    /// offset in at load time so the MLX kernel sees a plain scale
    /// factor and no extra op is needed on the hot path.
    ///
    /// This is the single biggest deviation from Qwen 3 — both archs
    /// store `weight` in safetensors, but Gemma's forward pass uses it
    /// differently. Host-side fixup keeps the runtime identical.
    fn load_rmsnorm_plus_one(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<MlxArray> {
        let (floats, shape_i32) = load_rmsnorm_plus_one_values_and_shape(weights, name)?;
        MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
    }

    fn load_rmsnorm_plus_one_values(
        weights: &SafeTensorBundle,
        name: &str,
    ) -> MlxLlmResult<Vec<f32>> {
        let (floats, _shape_i32) = load_rmsnorm_plus_one_values_and_shape(weights, name)?;
        Ok(floats)
    }

    fn load_rmsnorm_plus_one_values_and_shape(
        weights: &SafeTensorBundle,
        name: &str,
    ) -> MlxLlmResult<(Vec<f32>, Vec<i32>)> {
        let (mut floats, shape_i32) = weights.read_as_f32(name)?;
        for v in &mut floats {
            *v += 1.0;
        }
        Ok((floats, shape_i32))
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
        let head_dim = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.kv_heads();
        let scale = cfg.query_scale();

        // Embedding lookup: [B, T] -> [B, T, H]. Gemma scales the
        // embedding output by sqrt(hidden_size) immediately after
        // lookup.
        let mut hidden = gather(&weights.embed_tokens, input_ids, 0, stream)?;
        let embed_scale = MlxArray::from_slice_f32(&[(cfg.hidden_size as f32).sqrt()], &[1])?;
        hidden = mul(&hidden, &embed_scale, stream)?;

        let layers = &weights.layers;
        let layer_cache = &mut weights.layer_cache;
        for (layer_idx, (layer, cache)) in layers.iter().zip(layer_cache.iter_mut()).enumerate() {
            hidden = transformer_block(
                cfg,
                layer,
                cache,
                &hidden,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                position_offset,
                stream,
            )?;
        }

        let final_norm = rms_norm(&hidden, Some(&weights.norm), cfg.rms_norm_eps, stream)?;
        let lm_w = weights.lm_head.as_ref().unwrap_or(&weights.embed_tokens);
        let logits = matmul(&final_norm, &transpose(lm_w, &[1, 0], stream)?, stream)?;
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
        head_dim: usize,
        scale: f32,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let attn_in = rms_norm(
            hidden,
            Some(&layer.input_layernorm),
            cfg.rms_norm_eps,
            stream,
        )?;
        let attn = attention_block(
            cfg,
            layer,
            cache,
            &attn_in,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
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
        add(&hidden, &mlp, stream).map_err(Into::into)
    }

    /// Gated-GeLU FFN: `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))`.
    fn mlp_block(
        layer: &Gemma4Layer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let gate = matmul(
            hidden,
            &transpose(&layer.mlp_gate_proj, &[1, 0], stream)?,
            stream,
        )?;
        let up = matmul(
            hidden,
            &transpose(&layer.mlp_up_proj, &[1, 0], stream)?,
            stream,
        )?;
        let gated = mul(&gelu_tanh(&gate, stream)?, &up, stream)?;
        matmul(
            &gated,
            &transpose(&layer.mlp_down_proj, &[1, 0], stream)?,
            stream,
        )
        .map_err(Into::into)
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
        _layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        _head_dim: usize,
        _scale: f32,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let layer_head_dim = layer.q_norm.len();
        if layer.k_norm.len() != layer_head_dim {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 q/k norm dimensions differ (q={}, k={})",
                layer.q_norm.len(),
                layer.k_norm.len()
            )));
        }
        let layer_is_global = layer_head_dim != cfg.head_dim();
        let layer_scale = (layer_head_dim as f32).powf(-0.5);

        let hidden = cast(hidden, MlxDtype::F32, stream)?;
        let q = matmul(&hidden, &transpose(&layer.q_proj, &[1, 0], stream)?, stream)?;
        let k = matmul(&hidden, &transpose(&layer.k_proj, &[1, 0], stream)?, stream)?;
        let v = matmul(&hidden, &transpose(&layer.v_proj, &[1, 0], stream)?, stream)?;
        let q = host_split_heads_with_rms_norm(
            &q,
            n_heads,
            layer_head_dim,
            &layer.q_norm,
            cfg.rms_norm_eps,
        )?;
        let k = host_split_heads_with_rms_norm(
            &k,
            n_kv_heads,
            layer_head_dim,
            &layer.k_norm,
            cfg.rms_norm_eps,
        )?;
        let v = host_split_heads(&v, n_kv_heads, layer_head_dim)?;

        let rope_base = if layer_is_global {
            cfg.rope_theta
        } else {
            cfg.rope_local_base_freq
        };
        let layer_head_dim_i32 = shape_dim_i32("gemma4", "layer_head_dim", layer_head_dim)?;
        let q = rope(
            &q,
            layer_head_dim_i32,
            false,
            Some(rope_base),
            1.0,
            position_offset,
            None,
            stream,
        )?;
        let k = rope(
            &k,
            layer_head_dim_i32,
            false,
            Some(rope_base),
            1.0,
            position_offset,
            None,
            stream,
        )?;

        let had_cached_prefix = cache.keys.is_some();
        let k = append_cached_axis2(&mut cache.keys, &k, stream)?;
        let v = append_cached_axis2(&mut cache.values, &v, stream)?;
        let q_len = q.shape().get(2).copied().unwrap_or(1);

        let attn = if layer_is_global {
            let causal = !had_cached_prefix && q_len > 1;
            scaled_dot_product_attention(&q, &k, &v, layer_scale, causal, None, stream)?
        } else {
            let mask = sliding_window_mask(&q, &k, cfg.sliding_window, position_offset)?;
            scaled_dot_product_attention(&q, &k, &v, layer_scale, false, Some(&mask), stream)?
        };
        let attn = merge_heads(&attn, n_heads, layer_head_dim, stream)?;
        matmul(&attn, &transpose(&layer.o_proj, &[1, 0], stream)?, stream).map_err(Into::into)
    }

    fn host_split_heads_with_rms_norm(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        weight: &[f32],
        eps: f32,
    ) -> MlxLlmResult<MlxArray> {
        if weight.len() != head_dim {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "gemma4 head norm has {} values, expected head_dim={head_dim}",
                weight.len()
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
                        out[dst] = data[src] * inv_rms * weight[d];
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

    fn host_split_heads(x: &MlxArray, heads: usize, head_dim: usize) -> MlxLlmResult<MlxArray> {
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
                    for d in 0..head_dim {
                        let dst = (((b * heads + h) * seq + t) * head_dim) + d;
                        out[dst] = data[head + d];
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
        slot: &mut Option<MlxArray>,
        new_slice: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        match slot.take() {
            Some(existing) => {
                let combined = concat(&[&existing, new_slice], 2, stream)?;
                *slot = Some(combined.clone());
                Ok(combined)
            }
            None => {
                let cached = new_slice.clone();
                *slot = Some(cached.clone());
                Ok(cached)
            }
        }
    }

    fn sliding_window_mask(
        queries: &MlxArray,
        keys: &MlxArray,
        window_size: usize,
        position_offset: i32,
    ) -> MlxLlmResult<MlxArray> {
        let q_len_i32 = queries.shape()[2];
        let k_len_i32 = keys.shape()[2];
        let q_len = shape_dim_usize("gemma4", "query_len", q_len_i32)?;
        let k_len = shape_dim_usize("gemma4", "key_len", k_len_i32)?;
        let mask_len = shape_product_usize("gemma4", "sliding_window_mask_len", &[q_len, k_len])?;
        let window_size_i32 = shape_dim_i32("gemma4", "sliding_window", window_size)?;
        let mut mask = Vec::with_capacity(mask_len);
        for qi in 0..q_len {
            let qi_i32 = shape_dim_i32("gemma4", "query_index", qi)?;
            let q_pos = checked_i32_add("gemma4", "query_position", position_offset, qi_i32)?;
            for kj in 0..k_len {
                let kj_i32 = shape_dim_i32("gemma4", "key_index", kj)?;
                let k_pos = if k_len == q_len {
                    checked_i32_add("gemma4", "key_position", position_offset, kj_i32)?
                } else {
                    kj_i32
                };
                let window_end =
                    checked_i32_add("gemma4", "sliding_window_end", k_pos, window_size_i32)?;
                let allowed = k_pos <= q_pos && q_pos < window_end;
                mask.push(if allowed { 0.0 } else { -1.0e9 });
            }
        }
        MlxArray::from_slice_f32(&mask, &[1, 1, q_len_i32, k_len_i32]).map_err(Into::into)
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
        let width = shape_product_i32("gemma4", "merged_attention_width", &[heads, head_dim])?;
        reshape(&transposed, &[b, t, width], stream).map_err(Into::into)
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
        let head_dim_i32 = shape_dim_i32("gemma4", "head_dim", head_dim)?;
        rope(
            &normed,
            head_dim_i32,
            false,
            Some(rope_base),
            1.0,
            position_offset,
            None,
            stream,
        )
        .map_err(Into::into)
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
            sliding_window: 4096,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: None,
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
    fn quantized_rejected_by_safetensors_check() {
        let mut cfg = dummy_config(1, true);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
        });
        assert!(cfg.is_quantized());
        let err = validate_safetensors(Path::new("/nonexistent"), &cfg).unwrap_err();
        match &err {
            MlxLlmError::UnsupportedQuantization {
                model_type,
                bits,
                group_size,
                ..
            } => {
                assert_eq!(model_type, "gemma4");
                assert_eq!(*bits, 4);
                assert_eq!(*group_size, 64);
            }
            other => panic!("expected UnsupportedQuantization, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("unsupported MLX quantization"), "got: {msg}");
        assert!(msg.contains("gemma4"), "got: {msg}");
        assert!(msg.contains("4-bit/group=64"), "got: {msg}");
        assert!(msg.contains("GGUF fallback"), "got: {msg}");
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
}

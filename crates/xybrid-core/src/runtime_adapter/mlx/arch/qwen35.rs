//! Qwen 3 architecture builder.
//!
//! Ports the MLX-LM Python reference's `qwen3` architecture to Rust + the
//! safe tensor ops from `xybrid_mlx::ops`. The builder lands in two
//! layers:
//!
//! 1. **Non-linking validation** (always compiled under `llm-mlx`) — parses
//!    `config.json`, enumerates the expected safetensors weight-key
//!    schedule, and validates the safetensors header before we commit to
//!    linking Metal. Lets the runtime selector fall back to llama.cpp
//!    cleanly when a bundle is malformed.
//! 2. **Runtime** (gated on `llm-mlx-runtime` + Apple Silicon macOS) — the
//!    actual per-layer `runtime::Qwen35Weights` + forward pass, built on
//!    `xybrid_mlx::MlxArray` + the core tensor ops. Prefill writes K/V
//!    tensors into resident per-layer cache slots, and decode forwards only
//!    the newest token with an absolute RoPE offset.
//!
//! Reference (upstream): <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py>.
//! Qwen 3 is a LLaMA-style decoder with Grouped-Query Attention, SwiGLU
//! FFN, rotary position embeddings, and — distinctive to Qwen 3 — a
//! per-head RMSNorm on Q and K before RoPE (the `q_norm` / `k_norm`
//! weights).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};
use super::super::weights::SafeTensorBundle;

// =============================================================================
// Config
// =============================================================================

/// Full Qwen 3 `config.json` subset the builder needs.
///
/// Fields mirror the HuggingFace config layout (`model_type = "qwen3"`).
/// Missing fields fall back to upstream defaults — Qwen 3 pins
/// `rope_theta = 1_000_000` and `rms_norm_eps = 1e-6`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    /// Always `"qwen3"` for the Qwen 3 family.
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
    /// RoPE base frequency. Qwen 3 uses 1e6 (not the LLaMA default of 1e4).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Epsilon added under the sqrt in RMSNorm.
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    /// When `true`, the input embedding and the LM head share one weight
    /// tensor. Small Qwen 3 variants (0.6B) set this.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Per-head dimension. Defaults to `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Set when weights are quantized. MLX 4-bit bundles report
    /// `{ bits: 4, group_size: 64 }` here.
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_rms_eps() -> f32 {
    1e-6
}

fn default_quant_mode() -> String {
    "affine".into()
}

/// Quantization metadata block from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    pub bits: u32,
    pub group_size: u32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

impl Qwen3Config {
    /// Parse `{model_dir}/config.json` as a Qwen 3 config.
    pub fn from_model_dir(model_dir: &Path) -> MlxLlmResult<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path)?;
        let cfg = Self::parse_json(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn parse_json(raw: &str) -> MlxLlmResult<Self> {
        let mut root: serde_json::Value = serde_json::from_str(raw)?;
        if let Some(obj) = root.as_object_mut() {
            if !obj.contains_key("quantization") {
                if let Some(quantization) = obj.get("quantization_config").cloned() {
                    obj.insert("quantization".to_string(), quantization);
                }
            }
        }
        serde_json::from_value(root).map_err(MlxLlmError::ConfigParse)
    }

    fn validate(&self) -> MlxLlmResult<()> {
        if self.model_type != "qwen3" {
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
                "qwen3 config has zero-valued dimensions (hidden={}, layers={}, heads={}, ffn={}, vocab={})",
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.intermediate_size,
                self.vocab_size,
            )));
        }
        if self.head_dim.is_none() && !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "qwen3 config hidden_size={} is not a multiple of num_attention_heads={}",
                self.hidden_size, self.num_attention_heads,
            )));
        }
        let head_dim = self.head_dim();
        let kv_heads = self.kv_heads();
        if head_dim == 0 || kv_heads == 0 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "qwen3 config has invalid derived dimensions (head_dim={head_dim}, kv_heads={kv_heads})"
            )));
        }
        if !self.num_attention_heads.is_multiple_of(kv_heads) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "qwen3 config num_attention_heads={} is not a multiple of num_key_value_heads={kv_heads}",
                self.num_attention_heads,
            )));
        }
        super::validate_i32_dimensions(
            "qwen3",
            &[
                ("hidden_size", self.hidden_size),
                ("num_hidden_layers", self.num_hidden_layers),
                ("num_attention_heads", self.num_attention_heads),
                ("num_key_value_heads", kv_heads),
                ("intermediate_size", self.intermediate_size),
                ("vocab_size", self.vocab_size),
                ("max_position_embeddings", self.max_position_embeddings),
                ("head_dim", head_dim),
            ],
        )?;
        super::validate_i32_product(
            "qwen3",
            "attention_projection_width",
            &[self.num_attention_heads, head_dim],
        )?;
        super::validate_i32_product("qwen3", "kv_projection_width", &[kv_heads, head_dim])?;
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

    /// `true` when weights are stored in a quantized format.
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }
}

// =============================================================================
// Expected safetensors schedule
// =============================================================================

/// Safetensors key names the Qwen 3 forward pass reads, in the order the
/// upstream MLX-LM reference emits them.
///
/// Per-layer block (for layer `l`), 11 tensors:
/// - `input_layernorm.weight`
/// - `self_attn.{q,k,v,o}_proj.weight` (4)
/// - `self_attn.{q,k}_norm.weight` — Qwen 3-specific per-head RMSNorm (2)
/// - `post_attention_layernorm.weight`
/// - `mlp.{gate,up,down}_proj.weight` (3)
///
/// Shared across blocks (2 or 3):
/// - `model.embed_tokens.weight`
/// - `model.norm.weight`
/// - `lm_head.weight` — unless [`Qwen3Config::tie_word_embeddings`] is set.
pub fn expected_weight_keys(cfg: &Qwen3Config) -> Vec<String> {
    let mut keys = Vec::with_capacity(3 + 11 * cfg.num_hidden_layers);
    push_quantized_weight_key(&mut keys, "model.embed_tokens.weight", cfg.is_quantized());
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

// =============================================================================
// Safetensors validation
// =============================================================================

/// Validate that `model.safetensors` contains every key Qwen 3's forward
/// pass reads.
///
/// Used by [`super::super::model::MlxLlmAdapter::load`] to fail fast before
/// we cross the Metal boundary. The safetensors crate only parses the
/// header (a small JSON prefix) so this is cheap — no mmap of the full
/// weight tensor region.
///
/// Quantized MLX-LM bundles store additional `*.scales` / `*.biases`
/// tensors alongside packed U32 weights; these siblings are required here
/// so runtime loading can dispatch projections through quantized matmul.
pub fn validate_safetensors(path: &Path, cfg: &Qwen3Config) -> MlxLlmResult<()> {
    let weights = SafeTensorBundle::from_single_file(path.to_path_buf());
    validate_safetensors_bundle(&weights, cfg)
}

pub fn validate_safetensors_bundle(
    weights: &SafeTensorBundle,
    cfg: &Qwen3Config,
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

/// Top-level entry: load and validate a Qwen 3 bundle's config + weight
/// manifest. Returns the parsed config so the non-linking load path in
/// [`super::super::model::MlxLlmAdapter::load`] can cache it.
///
/// Does NOT touch MLX / Metal — that happens in `runtime::build` under
/// the `llm-mlx-runtime` feature.
pub fn load(model_dir: &Path, weights: &SafeTensorBundle) -> MlxLlmResult<Qwen3Config> {
    let cfg = Qwen3Config::from_model_dir(model_dir)?;
    validate_safetensors_bundle(weights, &cfg)?;
    Ok(cfg)
}

/// Resolve the legacy single-file safetensors path for callers that need
/// that exact file. Normal Qwen validation/runtime loading goes through
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
///
/// The metadata/validation path above runs on every platform that enables `llm-mlx`
/// (including the Linux CI runner that cross-compiles the feature for
/// `cargo check`). The runtime submodule is the only place that touches
/// Metal-backed allocations.
#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod runtime {
    use super::super::super::model::MlxLlmResult;
    use super::super::super::weights::SafeTensorBundle;
    use super::super::linear::{load_dense_or_dequantized, LinearQuantization, LinearWeight};
    use super::super::{shape_dim_i32, shape_product_i32};
    use super::Qwen3Config;

    use xybrid_mlx::ops::{
        add, concat, gather, matmul, mul, reshape, rms_norm, rope, scaled_dot_product_attention,
        silu, transpose,
    };
    use xybrid_mlx::{MlxArray, MlxStream};

    /// One transformer block's weights, materialised as [`MlxArray`]s in
    /// Metal unified memory.
    #[derive(Debug)]
    pub struct Qwen35Layer {
        pub input_layernorm: MlxArray,
        pub q_proj: LinearWeight,
        pub k_proj: LinearWeight,
        pub v_proj: LinearWeight,
        pub o_proj: LinearWeight,
        pub q_norm: MlxArray,
        pub k_norm: MlxArray,
        pub post_attention_layernorm: MlxArray,
        pub mlp_gate_proj: LinearWeight,
        pub mlp_up_proj: LinearWeight,
        pub mlp_down_proj: LinearWeight,
    }

    /// Resident K/V tensors for one decoder layer during a single generation
    /// call. The generate loop resets these before prefill, then each
    /// incremental decode step appends one `[B, kv_heads, 1, head_dim]` slice.
    #[derive(Debug, Default)]
    pub struct Qwen35LayerCache {
        keys: Option<MlxArray>,
        values: Option<MlxArray>,
    }

    impl Qwen35LayerCache {
        fn reset(&mut self) {
            self.keys = None;
            self.values = None;
        }
    }

    /// Full Qwen 3 weight set resident in MLX memory.
    #[derive(Debug)]
    pub struct Qwen35Weights {
        pub embed_tokens: MlxArray,
        pub layers: Vec<Qwen35Layer>,
        pub norm: MlxArray,
        /// `None` when [`Qwen3Config::tie_word_embeddings`] is set — the
        /// forward pass re-uses `embed_tokens` for the LM head.
        pub lm_head: Option<MlxArray>,
        layer_cache: Vec<Qwen35LayerCache>,
    }

    impl Qwen35Weights {
        /// Clear per-generation K/V state while keeping resident weights.
        pub fn reset_kv_cache(&mut self) {
            for cache in &mut self.layer_cache {
                cache.reset();
            }
        }
    }

    /// Build [`Qwen35Weights`] by reading `model.safetensors` and
    /// materialising each tensor as an [`MlxArray`].
    ///
    /// Supports dense F32/F16/BF16 weights and MLX affine quantized U32
    /// projection weights. Embeddings and LM head are dequantized at load
    /// time because `gather` and final logits still use dense-only ops.
    pub fn build(cfg: &Qwen3Config, weights: &SafeTensorBundle) -> MlxLlmResult<Qwen35Weights> {
        let quant = quantization(cfg)?;
        let quant = quant.as_ref();

        let embed_tokens = load_dense_or_dequantized(weights, "model.embed_tokens.weight", quant)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let base = format!("model.layers.{l}");
            layers.push(Qwen35Layer {
                input_layernorm: load_tensor(weights, &format!("{base}.input_layernorm.weight"))?,
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
                o_proj: LinearWeight::load(
                    weights,
                    &format!("{base}.self_attn.o_proj.weight"),
                    quant,
                )?,
                q_norm: load_tensor(weights, &format!("{base}.self_attn.q_norm.weight"))?,
                k_norm: load_tensor(weights, &format!("{base}.self_attn.k_norm.weight"))?,
                post_attention_layernorm: load_tensor(
                    weights,
                    &format!("{base}.post_attention_layernorm.weight"),
                )?,
                mlp_gate_proj: LinearWeight::load(
                    weights,
                    &format!("{base}.mlp.gate_proj.weight"),
                    quant,
                )?,
                mlp_up_proj: LinearWeight::load(
                    weights,
                    &format!("{base}.mlp.up_proj.weight"),
                    quant,
                )?,
                mlp_down_proj: LinearWeight::load(
                    weights,
                    &format!("{base}.mlp.down_proj.weight"),
                    quant,
                )?,
            });
        }
        let norm = load_tensor(weights, "model.norm.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_dense_or_dequantized(weights, "lm_head.weight", quant)?)
        };

        Ok(Qwen35Weights {
            embed_tokens,
            layers,
            norm,
            lm_head,
            layer_cache: std::iter::repeat_with(Qwen35LayerCache::default)
                .take(cfg.num_hidden_layers)
                .collect(),
        })
    }

    /// Read one tensor from a SafeTensors view into an [`MlxArray`].
    ///
    fn load_tensor(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<MlxArray> {
        weights.read_array(name)
    }

    fn quantization(cfg: &Qwen3Config) -> MlxLlmResult<Option<LinearQuantization>> {
        cfg.quantization
            .as_ref()
            .map(|q| LinearQuantization::new(q.bits, q.group_size, &q.mode))
            .transpose()
    }

    /// Forward pass through the whole stack.
    ///
    /// `input_ids` shape: `[batch, seq_len]` (i32). Returns logits of
    /// shape `[batch, seq_len, vocab_size]` (f32).
    ///
    /// The generate loop calls this once for full-prompt prefill, then with
    /// only the newest token for incremental decode. K/V tensors are appended
    /// into `weights.layer_cache` per layer and read back by later decode
    /// calls, so Qwen no longer recomputes the whole token history.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        cfg: &Qwen3Config,
        weights: &mut Qwen35Weights,
        input_ids: &MlxArray,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let head_dim = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.kv_heads();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Embedding lookup: [B, T] -> [B, T, H].
        let mut hidden = gather(&weights.embed_tokens, input_ids, 0, stream)?;

        let layers = &weights.layers;
        let layer_cache = &mut weights.layer_cache;
        for (layer, cache) in layers.iter().zip(layer_cache.iter_mut()) {
            hidden = attention_block(
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
            hidden = mlp_block(cfg, layer, &hidden, stream)?;
        }

        // Final norm + LM head.
        let final_norm = rms_norm(&hidden, Some(&weights.norm), cfg.rms_norm_eps, stream)?;
        let lm_w = weights.lm_head.as_ref().unwrap_or(&weights.embed_tokens);
        let logits = matmul(&final_norm, &transpose(lm_w, &[1, 0], stream)?, stream)?;
        Ok(logits)
    }

    /// SwiGLU FFN: `down_proj(silu(gate_proj(x)) * up_proj(x))` + residual.
    ///
    /// The pre-MLP RMSNorm (post_attention_layernorm in HF naming) is
    /// applied first. Weight layout follows the HF convention
    /// `[out_dim, in_dim]`, so each projection transposes the weight
    /// before the matmul — matching the MLX-LM Python reference exactly.
    fn mlp_block(
        cfg: &Qwen3Config,
        layer: &Qwen35Layer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let normed = rms_norm(
            hidden,
            Some(&layer.post_attention_layernorm),
            cfg.rms_norm_eps,
            stream,
        )?;
        let gate = layer.mlp_gate_proj.forward(&normed, stream)?;
        let up = layer.mlp_up_proj.forward(&normed, stream)?;
        let gated = mul(&silu(&gate, stream)?, &up, stream)?;
        let out = layer.mlp_down_proj.forward(&gated, stream)?;
        add(hidden, &out, stream).map_err(Into::into)
    }

    /// The attention half of one transformer block: pre-norm → QKV
    /// projections → per-head RMSNorm (Qwen 3 specific) → RoPE → fused
    /// SDPA → output projection → residual.
    ///
    /// The fused attention call receives GQA-shaped Q/K/V tensors; MLX
    /// handles the query-head to key/value-head grouping internally.
    #[allow(clippy::too_many_arguments)]
    fn attention_block(
        cfg: &Qwen3Config,
        layer: &Qwen35Layer,
        cache: &mut Qwen35LayerCache,
        hidden: &MlxArray,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        position_offset: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let normed = rms_norm(
            hidden,
            Some(&layer.input_layernorm),
            cfg.rms_norm_eps,
            stream,
        )?;

        let q = layer.q_proj.forward(&normed, stream)?;
        let k = layer.k_proj.forward(&normed, stream)?;
        let v = layer.v_proj.forward(&normed, stream)?;

        let q = split_heads(&q, n_heads, head_dim, stream)?;
        let k = split_heads(&k, n_kv_heads, head_dim, stream)?;
        let v = split_heads(&v, n_kv_heads, head_dim, stream)?;

        // Qwen 3-specific per-head RMSNorm on Q and K (applied before RoPE).
        let q = rms_norm(&q, Some(&layer.q_norm), cfg.rms_norm_eps, stream)?;
        let k = rms_norm(&k, Some(&layer.k_norm), cfg.rms_norm_eps, stream)?;
        let head_dim_i32 = shape_dim_i32("qwen3", "head_dim", head_dim)?;

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

        // Prefill has T>1 and needs the causal mask. Incremental decode has
        // q_len=1 and cached K/V only contains the prefix plus this token, so
        // no future positions exist to mask.
        let causal = !had_cached_prefix && q.shape().get(2).copied().unwrap_or(1) > 1;
        let attn = scaled_dot_product_attention(&q, &k, &v, scale, causal, None, stream)?;
        let attn = merge_heads(&attn, n_heads, head_dim, stream)?;
        let attn = layer.o_proj.forward(&attn, stream)?;

        add(hidden, &attn, stream).map_err(Into::into)
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
        let heads_i32 = shape_dim_i32("qwen3", "num_attention_heads", heads)?;
        let head_dim_i32 = shape_dim_i32("qwen3", "head_dim", head_dim)?;
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
        let width = shape_product_i32("qwen3", "merged_attention_width", &[heads, head_dim])?;
        reshape(&transposed, &[b, t, width], stream).map_err(Into::into)
    }
}

// =============================================================================
// Tests (non-linking — no bindings required)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config(n_layers: usize, tied: bool) -> Qwen3Config {
        Qwen3Config {
            model_type: "qwen3".into(),
            hidden_size: 16,
            num_hidden_layers: n_layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            intermediate_size: 32,
            vocab_size: 100,
            max_position_embeddings: 1024,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: tied,
            head_dim: Some(4),
            quantization: None,
        }
    }

    #[test]
    fn expected_keys_match_per_layer_schedule() {
        let cfg = dummy_config(2, false);
        let keys = expected_weight_keys(&cfg);
        // 1 (embed) + 2 layers * 11 (per-layer) + 2 (norm + lm_head) = 25.
        assert_eq!(keys.len(), 25);
        assert_eq!(keys.first().unwrap(), "model.embed_tokens.weight");
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.self_attn.q_norm.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.1.self_attn.k_norm.weight"));
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
        let cfg = Qwen3Config {
            model_type: "qwen3".into(),
            hidden_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 8,
            num_key_value_heads: Some(4),
            intermediate_size: 256,
            vocab_size: 100,
            max_position_embeddings: 1024,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            head_dim: None,
            quantization: None,
        };
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.kv_heads(), 4);
    }

    #[test]
    fn kv_heads_default_to_num_attention_heads() {
        let mut cfg = dummy_config(1, false);
        cfg.num_key_value_heads = None;
        assert_eq!(cfg.kv_heads(), cfg.num_attention_heads);
    }

    #[test]
    fn duplicate_quantization_config_alias_parses_once() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::json!({
                "model_type": "qwen3",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 100,
                "max_position_embeddings": 1024,
                "head_dim": 4,
                "quantization": {
                    "bits": 4,
                    "group_size": 64
                },
                "quantization_config": {
                    "bits": 4,
                    "group_size": 64
                }
            })
            .to_string(),
        )
        .unwrap();

        let cfg = Qwen3Config::from_model_dir(tmp.path()).expect("parse duplicated quant config");
        let quant = cfg.quantization.expect("quantization should parse");
        assert_eq!(quant.bits, 4);
        assert_eq!(quant.group_size, 64);
        assert_eq!(quant.mode, "affine");
    }

    #[test]
    fn qwen35_quantized_manifest_accepts_projection_triplets() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        let mut cfg = dummy_config(1, false);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
            mode: "affine".into(),
        });
        let keys = expected_weight_keys(&cfg);
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.self_attn.q_proj.scales"));
        assert!(keys
            .iter()
            .any(|k| k == "model.layers.0.self_attn.o_proj.biases"));
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
    fn non_qwen3_model_type_rejected() {
        let mut cfg = dummy_config(1, false);
        cfg.model_type = "llama".into();
        assert!(matches!(
            cfg.validate(),
            Err(MlxLlmError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn zero_dimensions_rejected() {
        let mut cfg = dummy_config(1, false);
        cfg.hidden_size = 0;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("zero-valued")),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn hidden_not_divisible_by_heads_rejected() {
        let mut cfg = dummy_config(1, false);
        cfg.head_dim = None;
        cfg.hidden_size = 17;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("multiple of")),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn oversized_dimensions_rejected() {
        let mut cfg = dummy_config(1, false);
        cfg.max_position_embeddings = i32::MAX as usize + 1;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
                assert!(msg.contains("max_position_embeddings"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn oversized_projection_width_rejected() {
        let mut cfg = dummy_config(1, false);
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
        // Serialize a nearly-empty safetensors file (one bogus tensor).
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
        let cfg = dummy_config(1, true); // tied: no lm_head
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

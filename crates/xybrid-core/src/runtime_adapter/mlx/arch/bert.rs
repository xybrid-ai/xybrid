//! BERT-family encoder architecture builder for MLX embeddings.
//!
//! BERT differs from the decoder-only LLM architectures (Qwen 3, Gemma 4,
//! LFM 3.5) in three important ways:
//!
//! 1. **Encoder-only** — no causal mask, no autoregressive loop. The
//!    forward pass is one bidirectional self-attention pass over the
//!    entire sequence and the output is a per-token hidden state matrix
//!    `[seq_len, hidden_dim]` that downstream pooling collapses into a
//!    single embedding vector.
//! 2. **Token-type / segment embeddings** — canonical BERT adds three
//!    learned embeddings (word + position + token-type) before the first
//!    encoder block. Models that swap learned position embeddings for
//!    RoPE (e.g. `nomic_bert`) drop the position-embedding tensor and
//!    apply rotary embeddings inside attention instead.
//! 3. **GeLU FFN** — canonical BERT uses GeLU (not SiLU). nomic-embed
//!    swaps the FFN for SwiGLU using a Wqkv-fused QKV projection.
//!
//! This module handles config parsing, weight-key enumeration,
//! safetensors header validation, and the Apple MLX runtime for
//! canonical BERT (`model_type = "bert"`) and the nomic variant
//! (`model_type = "nomic_bert"`). The nomic path follows the current
//! `nomic-ai/nomic-embed-text-v1.5` SafeTensors layout, either as one
//! `model.safetensors` file or as HuggingFace indexed shards:
//! no-prefix `embeddings.*`, `emb_ln.*`, and
//! `encoder.layers.*.{attn,mlp,norm1,norm2}` keys.
//!
//! Reference (upstream):
//! - <https://github.com/ml-explore/mlx-examples/tree/main/bert> (canonical)
//! - <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5> (nomic variant)

use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};
use super::super::weights::SafeTensorBundle;

// =============================================================================
// Config
// =============================================================================

/// Subset of an MLX-style BERT `config.json` the builder needs.
///
/// Accepted `model_type` values: `"bert"` (canonical) and `"nomic_bert"`
/// (nomic-embed variant). Unknown families are rejected at parse time
/// via the internal `validate()` invariant check.
#[derive(Debug, Clone)]
pub struct BertConfig {
    /// `"bert"` or `"nomic_bert"`.
    pub model_type: String,
    /// Hidden / embedding dimension.
    pub hidden_size: usize,
    /// Number of encoder blocks.
    pub num_hidden_layers: usize,
    /// Number of attention heads. BERT does not use Grouped-Query
    /// Attention, so K/V head count equals `num_attention_heads`.
    pub num_attention_heads: usize,
    /// FFN intermediate (up-projection) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum position index — the size of the learned position
    /// embedding table for canonical BERT, the upper RoPE bound for
    /// nomic-embed.
    pub max_position_embeddings: usize,
    /// Token-type vocab size (segment IDs). Canonical BERT uses 2; nomic
    /// variants typically also expose this even when they don't use
    /// segment embeddings actively.
    pub type_vocab_size: usize,
    /// LayerNorm epsilon. Canonical BERT uses 1e-12, nomic uses 1e-12.
    pub layer_norm_eps: f32,
    /// `true` when this variant uses rotary position embeddings instead
    /// of learned absolute position embeddings. Nomic configs are
    /// treated as rotary even when the current native Transformers
    /// config omits this legacy flag.
    pub use_rotary_embeddings: bool,
    /// `true` when the FFN uses a SwiGLU gating pair instead of the
    /// canonical single-linear + GeLU. nomic-embed-text-v1.5 sets this.
    pub use_swiglu: bool,
    /// RoPE base frequency. Older nomic configs call this
    /// `rotary_emb_base`; current Transformers docs expose it through
    /// `rope_parameters.rope_theta`. The real v1.5 weights use 1000.
    pub rope_theta: f32,
    /// Optional native Transformers RoPE config block.
    pub rope_parameters: Option<RopeParameters>,
    /// Nomic native configs expose `hidden_act = "silu"`; older
    /// trust-remote-code configs expose `activation_function = "swiglu"`.
    pub hidden_act: Option<String>,
    /// Set when weights are quantized.
    pub quantization: Option<QuantConfig>,
}

#[derive(Debug, Deserialize)]
struct BertConfigRaw {
    model_type: String,
    hidden_size: Option<usize>,
    n_embd: Option<usize>,
    num_hidden_layers: Option<usize>,
    n_layer: Option<usize>,
    num_attention_heads: Option<usize>,
    n_head: Option<usize>,
    intermediate_size: Option<usize>,
    n_inner: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    n_positions: Option<usize>,
    type_vocab_size: Option<usize>,
    layer_norm_eps: Option<f32>,
    layer_norm_epsilon: Option<f32>,
    use_rotary_embeddings: Option<bool>,
    use_swiglu: Option<bool>,
    rope_theta: Option<f32>,
    rotary_emb_base: Option<f32>,
    rope_parameters: Option<RopeParameters>,
    hidden_act: Option<String>,
    activation_function: Option<String>,
    quantization: Option<QuantConfig>,
}

impl<'de> Deserialize<'de> for BertConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = BertConfigRaw::deserialize(deserializer)?;
        Self::from_raw(raw).map_err(serde::de::Error::custom)
    }
}

fn default_type_vocab_size() -> usize {
    2
}

fn default_layer_norm_eps() -> f32 {
    1e-12
}

fn default_rope_theta() -> f32 {
    1000.0
}

fn pick<T>(primary: Option<T>, legacy: Option<T>, field: &str) -> Result<T, String> {
    primary
        .or(legacy)
        .ok_or_else(|| format!("missing required field {field}"))
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: Option<f32>,
}

/// Quantization metadata block from `config.json`. Mirrors the LLM
/// arch builders.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    pub bits: u32,
    pub group_size: u32,
}

impl BertConfig {
    fn from_raw(raw: BertConfigRaw) -> Result<Self, String> {
        Ok(Self {
            model_type: raw.model_type,
            hidden_size: pick(raw.hidden_size, raw.n_embd, "hidden_size/n_embd")?,
            num_hidden_layers: pick(
                raw.num_hidden_layers,
                raw.n_layer,
                "num_hidden_layers/n_layer",
            )?,
            num_attention_heads: pick(
                raw.num_attention_heads,
                raw.n_head,
                "num_attention_heads/n_head",
            )?,
            intermediate_size: pick(
                raw.intermediate_size,
                raw.n_inner,
                "intermediate_size/n_inner",
            )?,
            vocab_size: raw
                .vocab_size
                .ok_or_else(|| "missing required field vocab_size".to_string())?,
            max_position_embeddings: pick(
                raw.max_position_embeddings,
                raw.n_positions,
                "max_position_embeddings/n_positions",
            )?,
            type_vocab_size: raw.type_vocab_size.unwrap_or_else(default_type_vocab_size),
            layer_norm_eps: raw
                .layer_norm_eps
                .or(raw.layer_norm_epsilon)
                .unwrap_or_else(default_layer_norm_eps),
            use_rotary_embeddings: raw.use_rotary_embeddings.unwrap_or(false),
            use_swiglu: raw.use_swiglu.unwrap_or(false),
            rope_theta: raw
                .rope_theta
                .or(raw.rotary_emb_base)
                .unwrap_or_else(default_rope_theta),
            rope_parameters: raw.rope_parameters,
            hidden_act: raw.hidden_act.or(raw.activation_function),
            quantization: raw.quantization,
        })
    }

    /// Parse `{model_dir}/config.json` as a BERT config.
    pub fn from_model_dir(model_dir: &Path) -> MlxLlmResult<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path)?;
        let cfg: BertConfig = serde_json::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> MlxLlmResult<()> {
        if !matches!(self.model_type.as_str(), "bert" | "nomic_bert") {
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
                "bert config has zero-valued dimensions (hidden={}, layers={}, heads={}, ffn={}, vocab={})",
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.intermediate_size,
                self.vocab_size,
            )));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "bert config hidden_size={} is not a multiple of num_attention_heads={}",
                self.hidden_size, self.num_attention_heads,
            )));
        }
        super::validate_i32_dimensions(
            "bert",
            &[
                ("hidden_size", self.hidden_size),
                ("num_hidden_layers", self.num_hidden_layers),
                ("num_attention_heads", self.num_attention_heads),
                ("intermediate_size", self.intermediate_size),
                ("vocab_size", self.vocab_size),
                ("max_position_embeddings", self.max_position_embeddings),
                ("type_vocab_size", self.type_vocab_size),
                ("head_dim", self.head_dim()),
            ],
        )?;
        if self.is_nomic() {
            super::validate_i32_product("bert", "qkv_projection_width", &[3, self.hidden_size])?;
        }
        Ok(())
    }

    /// Per-head dimension. BERT does not store a `head_dim` override;
    /// `hidden_size / num_attention_heads` is canonical.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `true` when weights are stored in a quantized format.
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }

    /// `true` for the nomic-embed family, which uses a fused `Wqkv`
    /// projection plus SwiGLU FFN.
    pub fn is_nomic(&self) -> bool {
        self.model_type == "nomic_bert"
    }

    /// `true` when the model should use RoPE instead of learned absolute
    /// position embeddings.
    pub fn uses_rotary_embeddings(&self) -> bool {
        self.use_rotary_embeddings || self.is_nomic()
    }

    /// RoPE base frequency used by the runtime path.
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.rope_theta)
            .unwrap_or(self.rope_theta)
    }
}

// =============================================================================
// Expected safetensors schedule
// =============================================================================

/// Safetensors key names the BERT forward pass reads.
///
/// **Canonical BERT** (`model_type = "bert"`): per layer (16 tensors)
/// - `encoder.layer.{l}.attention.self.{query,key,value}.{weight,bias}` (6)
/// - `encoder.layer.{l}.attention.output.dense.{weight,bias}` (2)
/// - `encoder.layer.{l}.attention.output.LayerNorm.{weight,bias}` (2)
/// - `encoder.layer.{l}.intermediate.dense.{weight,bias}` (2)
/// - `encoder.layer.{l}.output.dense.{weight,bias}` (2)
/// - `encoder.layer.{l}.output.LayerNorm.{weight,bias}` (2)
///
/// **nomic_bert** (`model_type = "nomic_bert"`): per layer (9 tensors)
/// - `encoder.layers.{l}.attn.Wqkv.weight` (fused QKV, no bias)
/// - `encoder.layers.{l}.attn.out_proj.weight` (no bias)
/// - `encoder.layers.{l}.norm1.{weight,bias}` (post-attention LayerNorm)
/// - `encoder.layers.{l}.mlp.{fc11,fc12,fc2}.weight` (SwiGLU gate / up
///   / down, no bias)
/// - `encoder.layers.{l}.norm2.{weight,bias}` (post-MLP LayerNorm)
///
/// Shared across blocks:
/// - Canonical: `embeddings.word_embeddings.weight`,
///   `embeddings.position_embeddings.weight`,
///   `embeddings.token_type_embeddings.weight`,
///   `embeddings.LayerNorm.{weight,bias}`
/// - nomic: `embeddings.word_embeddings.weight`, optional
///   `embeddings.token_type_embeddings.weight` (gated on
///   `type_vocab_size > 0`), and `emb_ln.{weight,bias}`; the
///   position-embedding tensor is omitted because RoPE is applied inside
///   attention.
pub fn expected_weight_keys(cfg: &BertConfig) -> Vec<String> {
    if cfg.is_nomic() {
        nomic_expected_weight_keys(cfg)
    } else {
        canonical_expected_weight_keys(cfg)
    }
}

fn canonical_expected_weight_keys(cfg: &BertConfig) -> Vec<String> {
    let mut keys = Vec::with_capacity(5 + 16 * cfg.num_hidden_layers);
    keys.push("embeddings.word_embeddings.weight".to_string());
    if !cfg.uses_rotary_embeddings() {
        keys.push("embeddings.position_embeddings.weight".to_string());
    }
    keys.push("embeddings.token_type_embeddings.weight".to_string());
    keys.push("embeddings.LayerNorm.weight".to_string());
    keys.push("embeddings.LayerNorm.bias".to_string());
    for l in 0..cfg.num_hidden_layers {
        let base = format!("encoder.layer.{l}");
        keys.push(format!("{base}.attention.self.query.weight"));
        keys.push(format!("{base}.attention.self.query.bias"));
        keys.push(format!("{base}.attention.self.key.weight"));
        keys.push(format!("{base}.attention.self.key.bias"));
        keys.push(format!("{base}.attention.self.value.weight"));
        keys.push(format!("{base}.attention.self.value.bias"));
        keys.push(format!("{base}.attention.output.dense.weight"));
        keys.push(format!("{base}.attention.output.dense.bias"));
        keys.push(format!("{base}.attention.output.LayerNorm.weight"));
        keys.push(format!("{base}.attention.output.LayerNorm.bias"));
        keys.push(format!("{base}.intermediate.dense.weight"));
        keys.push(format!("{base}.intermediate.dense.bias"));
        keys.push(format!("{base}.output.dense.weight"));
        keys.push(format!("{base}.output.dense.bias"));
        keys.push(format!("{base}.output.LayerNorm.weight"));
        keys.push(format!("{base}.output.LayerNorm.bias"));
    }
    keys
}

fn nomic_expected_weight_keys(cfg: &BertConfig) -> Vec<String> {
    let mut keys = Vec::with_capacity(4 + 9 * cfg.num_hidden_layers);
    keys.push("embeddings.word_embeddings.weight".to_string());
    if cfg.type_vocab_size > 0 {
        keys.push("embeddings.token_type_embeddings.weight".to_string());
    }
    keys.push("emb_ln.weight".to_string());
    keys.push("emb_ln.bias".to_string());
    for l in 0..cfg.num_hidden_layers {
        let base = format!("encoder.layers.{l}");
        keys.push(format!("{base}.attn.Wqkv.weight"));
        keys.push(format!("{base}.attn.out_proj.weight"));
        keys.push(format!("{base}.norm1.weight"));
        keys.push(format!("{base}.norm1.bias"));
        keys.push(format!("{base}.mlp.fc11.weight"));
        keys.push(format!("{base}.mlp.fc12.weight"));
        keys.push(format!("{base}.mlp.fc2.weight"));
        keys.push(format!("{base}.norm2.weight"));
        keys.push(format!("{base}.norm2.bias"));
    }
    keys
}

// =============================================================================
// Safetensors validation
// =============================================================================

/// Validate that a SafeTensors bundle contains every key the encoder
/// reads. Mirrors the LLM arch builders' header-only validation pattern.
pub fn validate_safetensors(path: &Path, cfg: &BertConfig) -> MlxLlmResult<()> {
    let weights = SafeTensorBundle::from_single_file(path.to_path_buf());
    validate_safetensors_bundle(&weights, cfg)
}

pub fn validate_safetensors_bundle(
    weights: &SafeTensorBundle,
    cfg: &BertConfig,
) -> MlxLlmResult<()> {
    if cfg.is_quantized() {
        let bits = cfg.quantization.as_ref().map(|q| q.bits).unwrap_or(0);
        let gs = cfg.quantization.as_ref().map(|q| q.group_size).unwrap_or(0);
        return Err(MlxLlmError::UnsupportedQuantization {
            model_type: cfg.model_type.clone(),
            bits,
            group_size: gs,
            reason: "mlx_fast_quantized_matmul is not wired for BERT embeddings yet",
        });
    }

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

/// Top-level entry: load and validate a BERT bundle's config + weight
/// manifest. Returns the parsed config so the embedding adapter's load
/// path can cache it.
pub fn load(model_dir: &Path, weights: &SafeTensorBundle) -> MlxLlmResult<BertConfig> {
    let cfg = BertConfig::from_model_dir(model_dir)?;
    validate_safetensors_bundle(weights, &cfg)?;
    Ok(cfg)
}

/// Resolve the SafeTensors weight bundle for a BERT directory. Accepts both
/// single-file `model.safetensors` and HuggingFace `model.safetensors.index.json`
/// shard manifests.
pub fn resolve_weights_bundle(model_dir: &Path) -> MlxLlmResult<SafeTensorBundle> {
    SafeTensorBundle::from_model_dir(model_dir)
}

fn f32_le_bytes_to_vec(data: &[u8]) -> Result<Vec<f32>, String> {
    if !data.len().is_multiple_of(4) {
        return Err(format!(
            "f32 tensor byte length {} is not divisible by 4",
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn safetensor_shape_to_i32(shape: &[usize], tensor_name: &str) -> MlxLlmResult<Vec<i32>> {
    let mut out = Vec::with_capacity(shape.len());
    for (axis, &dim) in shape.iter().enumerate() {
        let dim = i32::try_from(dim).map_err(|_| MlxLlmError::WeightLoad {
            path: std::path::PathBuf::from(tensor_name),
            reason: format!(
                "tensor `{tensor_name}` dimension {axis}={dim} exceeds MLX i32 shape limit"
            ),
        })?;
        out.push(dim);
    }
    Ok(out)
}

// =============================================================================
// Runtime (llm-mlx-runtime + Apple Silicon macOS only)
// =============================================================================

/// MLX-backed BERT encoder forward pass — only compiled when the
/// xcframework can be linked on Apple Silicon macOS.
///
/// The runtime materialises canonical BERT and current nomic-bert
/// SafeTensors into typed weight structs, executes the encoder stack
/// through MLX, and returns hidden states for the embedding adapter to pool.
#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod runtime {
    use super::super::super::model::{MlxLlmError, MlxLlmResult};
    use super::super::super::weights::SafeTensorBundle;
    use super::super::{shape_dim_i32, shape_product_i32};
    use super::BertConfig;

    use xybrid_mlx::ops::{
        add, gather, gelu, layer_norm, matmul, mul, reshape, rope, scaled_dot_product_attention,
        silu, transpose,
    };
    use xybrid_mlx::{MlxArray, MlxStream};

    /// One canonical BERT encoder block's weights.
    #[derive(Debug)]
    pub struct CanonicalBertLayer {
        pub query_weight: MlxArray,
        pub query_bias: MlxArray,
        pub key_weight: MlxArray,
        pub key_bias: MlxArray,
        pub value_weight: MlxArray,
        pub value_bias: MlxArray,
        pub attention_output_weight: MlxArray,
        pub attention_output_bias: MlxArray,
        pub attention_layernorm_weight: MlxArray,
        pub attention_layernorm_bias: MlxArray,
        pub intermediate_weight: MlxArray,
        pub intermediate_bias: MlxArray,
        pub output_weight: MlxArray,
        pub output_bias: MlxArray,
        pub output_layernorm_weight: MlxArray,
        pub output_layernorm_bias: MlxArray,
    }

    /// One current nomic-bert encoder block's weights.
    #[derive(Debug)]
    pub struct NomicBertLayer {
        pub wqkv_weight: MlxArray,
        pub out_proj_weight: MlxArray,
        pub norm1_weight: MlxArray,
        pub norm1_bias: MlxArray,
        pub fc11_weight: MlxArray,
        pub fc12_weight: MlxArray,
        pub fc2_weight: MlxArray,
        pub norm2_weight: MlxArray,
        pub norm2_bias: MlxArray,
    }

    /// Materialised BERT weights resident in MLX memory. Currently just
    /// the embedding tensors plus canonical and nomic encoder layers.
    #[derive(Debug)]
    pub struct BertWeights {
        pub word_embeddings: MlxArray,
        pub position_embeddings: Option<MlxArray>,
        pub token_type_embeddings: Option<MlxArray>,
        pub embeddings_layernorm: Option<(MlxArray, MlxArray)>,
        pub canonical_layers: Vec<CanonicalBertLayer>,
        pub nomic_layers: Vec<NomicBertLayer>,
    }

    /// Materialise BERT tensors into MLX memory. Canonical BERT loads the
    /// full encoder layer set; nomic_bert loads the current no-prefix
    /// SafeTensors schedule used by `nomic-ai/nomic-embed-text-v1.5`.
    pub fn build(cfg: &BertConfig, weights: &SafeTensorBundle) -> MlxLlmResult<BertWeights> {
        let nomic = cfg.is_nomic();
        let word_embeddings = load_tensor(weights, "embeddings.word_embeddings.weight")?;

        let position_embeddings = if cfg.uses_rotary_embeddings() {
            None
        } else {
            Some(load_tensor(
                weights,
                "embeddings.position_embeddings.weight",
            )?)
        };

        let token_type_embeddings = if nomic && cfg.type_vocab_size == 0 {
            None
        } else {
            Some(load_tensor(
                weights,
                "embeddings.token_type_embeddings.weight",
            )?)
        };

        let embeddings_layernorm = if nomic {
            Some((
                load_tensor(weights, "emb_ln.weight")?,
                load_tensor(weights, "emb_ln.bias")?,
            ))
        } else {
            Some((
                load_tensor(weights, "embeddings.LayerNorm.weight")?,
                load_tensor(weights, "embeddings.LayerNorm.bias")?,
            ))
        };

        let canonical_layers = if nomic {
            Vec::new()
        } else {
            let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
            for l in 0..cfg.num_hidden_layers {
                let base = format!("encoder.layer.{l}");
                layers.push(CanonicalBertLayer {
                    query_weight: load_tensor(
                        weights,
                        &format!("{base}.attention.self.query.weight"),
                    )?,
                    query_bias: load_tensor(weights, &format!("{base}.attention.self.query.bias"))?,
                    key_weight: load_tensor(weights, &format!("{base}.attention.self.key.weight"))?,
                    key_bias: load_tensor(weights, &format!("{base}.attention.self.key.bias"))?,
                    value_weight: load_tensor(
                        weights,
                        &format!("{base}.attention.self.value.weight"),
                    )?,
                    value_bias: load_tensor(weights, &format!("{base}.attention.self.value.bias"))?,
                    attention_output_weight: load_tensor(
                        weights,
                        &format!("{base}.attention.output.dense.weight"),
                    )?,
                    attention_output_bias: load_tensor(
                        weights,
                        &format!("{base}.attention.output.dense.bias"),
                    )?,
                    attention_layernorm_weight: load_tensor(
                        weights,
                        &format!("{base}.attention.output.LayerNorm.weight"),
                    )?,
                    attention_layernorm_bias: load_tensor(
                        weights,
                        &format!("{base}.attention.output.LayerNorm.bias"),
                    )?,
                    intermediate_weight: load_tensor(
                        weights,
                        &format!("{base}.intermediate.dense.weight"),
                    )?,
                    intermediate_bias: load_tensor(
                        weights,
                        &format!("{base}.intermediate.dense.bias"),
                    )?,
                    output_weight: load_tensor(weights, &format!("{base}.output.dense.weight"))?,
                    output_bias: load_tensor(weights, &format!("{base}.output.dense.bias"))?,
                    output_layernorm_weight: load_tensor(
                        weights,
                        &format!("{base}.output.LayerNorm.weight"),
                    )?,
                    output_layernorm_bias: load_tensor(
                        weights,
                        &format!("{base}.output.LayerNorm.bias"),
                    )?,
                });
            }
            layers
        };

        let nomic_layers = if nomic {
            let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
            for l in 0..cfg.num_hidden_layers {
                let base = format!("encoder.layers.{l}");
                layers.push(NomicBertLayer {
                    wqkv_weight: load_tensor(weights, &format!("{base}.attn.Wqkv.weight"))?,
                    out_proj_weight: load_tensor(weights, &format!("{base}.attn.out_proj.weight"))?,
                    norm1_weight: load_tensor(weights, &format!("{base}.norm1.weight"))?,
                    norm1_bias: load_tensor(weights, &format!("{base}.norm1.bias"))?,
                    fc11_weight: load_tensor(weights, &format!("{base}.mlp.fc11.weight"))?,
                    fc12_weight: load_tensor(weights, &format!("{base}.mlp.fc12.weight"))?,
                    fc2_weight: load_tensor(weights, &format!("{base}.mlp.fc2.weight"))?,
                    norm2_weight: load_tensor(weights, &format!("{base}.norm2.weight"))?,
                    norm2_bias: load_tensor(weights, &format!("{base}.norm2.bias"))?,
                });
            }
            layers
        } else {
            Vec::new()
        };

        Ok(BertWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_layernorm,
            canonical_layers,
            nomic_layers,
        })
    }

    /// BERT encoder forward pass. Canonical BERT and current nomic-bert
    /// execute their full encoder stacks and return `[1, seq_len, hidden]`.
    pub fn forward(
        cfg: &BertConfig,
        weights: &BertWeights,
        token_ids: &[i32],
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let mut hidden = embedding_prelude(cfg, weights, token_ids, stream)?;
        if cfg.is_nomic() {
            if weights.nomic_layers.len() != cfg.num_hidden_layers {
                return Err(MlxLlmError::ConfigInvalid(format!(
                    "nomic BERT loaded {} layer(s), expected {}",
                    weights.nomic_layers.len(),
                    cfg.num_hidden_layers
                )));
            }
            for layer in &weights.nomic_layers {
                hidden = nomic_layer_forward(cfg, layer, &hidden, stream)?;
            }
            return Ok(hidden);
        }
        if weights.canonical_layers.len() != cfg.num_hidden_layers {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "canonical BERT loaded {} layer(s), expected {}",
                weights.canonical_layers.len(),
                cfg.num_hidden_layers
            )));
        }
        for layer in &weights.canonical_layers {
            hidden = canonical_layer_forward(cfg, layer, &hidden, stream)?;
        }
        Ok(hidden)
    }

    /// Token, position, and segment embedding prelude for BERT-family
    /// encoders. Returns hidden states with shape `[1, seq_len, hidden]`.
    ///
    /// Canonical BERT applies learned word + position + token-type
    /// embeddings followed by embeddings LayerNorm. nomic_bert skips
    /// learned position embeddings, applies token-type embeddings when
    /// present, then applies `emb_ln`.
    pub(crate) fn embedding_prelude(
        cfg: &BertConfig,
        weights: &BertWeights,
        token_ids: &[i32],
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let seq_len_i32 = i32::try_from(token_ids.len()).map_err(|_| {
            MlxLlmError::ConfigInvalid(format!(
                "bert input sequence length {} exceeds i32::MAX",
                token_ids.len()
            ))
        })?;
        let input_ids = MlxArray::from_slice_i32(token_ids, &[1, seq_len_i32])?;
        let mut hidden = gather(&weights.word_embeddings, &input_ids, 0, stream)?;

        if !cfg.uses_rotary_embeddings() {
            if token_ids.len() > cfg.max_position_embeddings {
                return Err(MlxLlmError::ConfigInvalid(format!(
                    "bert input sequence length {} exceeds max_position_embeddings={}",
                    token_ids.len(),
                    cfg.max_position_embeddings
                )));
            }
            let position_embeddings = weights.position_embeddings.as_ref().ok_or_else(|| {
                MlxLlmError::ConfigInvalid(
                    "canonical BERT embedding prelude requires position embeddings".into(),
                )
            })?;
            let positions: Vec<i32> = (0..seq_len_i32).collect();
            let position_ids = MlxArray::from_slice_i32(&positions, &[1, seq_len_i32])?;
            let position_hidden = gather(position_embeddings, &position_ids, 0, stream)?;
            hidden = add(&hidden, &position_hidden, stream)?;
        }

        if let Some(token_type_embeddings) = &weights.token_type_embeddings {
            let token_types = vec![0_i32; token_ids.len()];
            let token_type_ids = MlxArray::from_slice_i32(&token_types, &[1, seq_len_i32])?;
            let token_type_hidden = gather(token_type_embeddings, &token_type_ids, 0, stream)?;
            hidden = add(&hidden, &token_type_hidden, stream)?;
        } else if !cfg.is_nomic() {
            return Err(MlxLlmError::ConfigInvalid(
                "canonical BERT embedding prelude requires token-type embeddings".into(),
            ));
        }

        if let Some((weight, bias)) = &weights.embeddings_layernorm {
            hidden = layer_norm(&hidden, weight, bias, cfg.layer_norm_eps, stream)?;
        } else {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "{} embedding prelude requires embeddings LayerNorm",
                cfg.model_type
            )));
        }

        Ok(hidden)
    }

    fn canonical_layer_forward(
        cfg: &BertConfig,
        layer: &CanonicalBertLayer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let attention = canonical_self_attention(cfg, layer, hidden, stream)?;
        let attention_hidden = layer_norm(
            &add(hidden, &attention, stream)?,
            &layer.attention_layernorm_weight,
            &layer.attention_layernorm_bias,
            cfg.layer_norm_eps,
            stream,
        )?;

        let intermediate = matmul(
            &attention_hidden,
            &transpose(&layer.intermediate_weight, &[1, 0], stream)?,
            stream,
        )?;
        let intermediate = add(&intermediate, &layer.intermediate_bias, stream)?;
        let intermediate = gelu(&intermediate, stream)?;

        let output = matmul(
            &intermediate,
            &transpose(&layer.output_weight, &[1, 0], stream)?,
            stream,
        )?;
        let output = add(&output, &layer.output_bias, stream)?;
        layer_norm(
            &add(&attention_hidden, &output, stream)?,
            &layer.output_layernorm_weight,
            &layer.output_layernorm_bias,
            cfg.layer_norm_eps,
            stream,
        )
        .map_err(Into::into)
    }

    fn canonical_self_attention(
        cfg: &BertConfig,
        layer: &CanonicalBertLayer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let head_dim = cfg.head_dim();
        let q = linear(hidden, &layer.query_weight, &layer.query_bias, stream)?;
        let k = linear(hidden, &layer.key_weight, &layer.key_bias, stream)?;
        let v = linear(hidden, &layer.value_weight, &layer.value_bias, stream)?;
        let q = split_heads(&q, cfg.num_attention_heads, head_dim, stream)?;
        let k = split_heads(&k, cfg.num_attention_heads, head_dim, stream)?;
        let v = split_heads(&v, cfg.num_attention_heads, head_dim, stream)?;

        let attention = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (head_dim as f32).sqrt(),
            false,
            None,
            stream,
        )?;
        let merged = merge_heads(&attention, cfg.num_attention_heads, head_dim, stream)?;
        linear(
            &merged,
            &layer.attention_output_weight,
            &layer.attention_output_bias,
            stream,
        )
    }

    fn nomic_layer_forward(
        cfg: &BertConfig,
        layer: &NomicBertLayer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let attention = nomic_self_attention(cfg, layer, hidden, stream)?;
        let attention_hidden = layer_norm(
            &add(hidden, &attention, stream)?,
            &layer.norm1_weight,
            &layer.norm1_bias,
            cfg.layer_norm_eps,
            stream,
        )?;

        let y = matmul(
            &attention_hidden,
            &transpose(&layer.fc11_weight, &[1, 0], stream)?,
            stream,
        )?;
        let gate = matmul(
            &attention_hidden,
            &transpose(&layer.fc12_weight, &[1, 0], stream)?,
            stream,
        )?;
        let gated = mul(&y, &silu(&gate, stream)?, stream)?;
        let mlp = matmul(
            &gated,
            &transpose(&layer.fc2_weight, &[1, 0], stream)?,
            stream,
        )?;

        layer_norm(
            &add(&attention_hidden, &mlp, stream)?,
            &layer.norm2_weight,
            &layer.norm2_bias,
            cfg.layer_norm_eps,
            stream,
        )
        .map_err(Into::into)
    }

    fn nomic_self_attention(
        cfg: &BertConfig,
        layer: &NomicBertLayer,
        hidden: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let head_dim = cfg.head_dim();
        let qkv = matmul(
            hidden,
            &transpose(&layer.wqkv_weight, &[1, 0], stream)?,
            stream,
        )?;
        let shape = qkv.shape();
        let b = shape[0];
        let t = shape[1];
        let heads_i32 = shape_dim_i32("bert", "num_attention_heads", cfg.num_attention_heads)?;
        let head_dim_i32 = shape_dim_i32("bert", "head_dim", head_dim)?;
        let qkv = reshape(&qkv, &[b, t, 3, heads_i32, head_dim_i32], stream)?;
        let q = gather_qkv(&qkv, 0, stream)?;
        let k = gather_qkv(&qkv, 1, stream)?;
        let v = gather_qkv(&qkv, 2, stream)?;

        let q = rope(
            &q,
            head_dim_i32,
            false,
            Some(cfg.rope_theta()),
            1.0,
            0,
            None,
            stream,
        )?;
        let k = rope(
            &k,
            head_dim_i32,
            false,
            Some(cfg.rope_theta()),
            1.0,
            0,
            None,
            stream,
        )?;

        let attention = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (head_dim as f32).sqrt(),
            false,
            None,
            stream,
        )?;
        let merged = merge_heads(&attention, cfg.num_attention_heads, head_dim, stream)?;
        matmul(
            &merged,
            &transpose(&layer.out_proj_weight, &[1, 0], stream)?,
            stream,
        )
        .map_err(Into::into)
    }

    fn gather_qkv(
        qkv: &MlxArray,
        index: i32,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = qkv.shape();
        let b = shape[0];
        let t = shape[1];
        let h = shape[3];
        let d = shape[4];
        let index = MlxArray::from_slice_i32(&[index], &[1])?;
        let selected = gather(qkv, &index, 2, stream)?;
        let selected = reshape(&selected, &[b, t, h, d], stream)?;
        transpose(&selected, &[0, 2, 1, 3], stream).map_err(Into::into)
    }

    fn linear(
        input: &MlxArray,
        weight: &MlxArray,
        bias: &MlxArray,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let projected = matmul(input, &transpose(weight, &[1, 0], stream)?, stream)?;
        add(&projected, bias, stream).map_err(Into::into)
    }

    fn split_heads(
        x: &MlxArray,
        heads: usize,
        head_dim: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<MlxArray> {
        let shape = x.shape();
        let b = shape[0];
        let t = shape[1];
        let heads_i32 = shape_dim_i32("bert", "num_attention_heads", heads)?;
        let head_dim_i32 = shape_dim_i32("bert", "head_dim", head_dim)?;
        let reshaped = reshape(x, &[b, t, heads_i32, head_dim_i32], stream)?;
        transpose(&reshaped, &[0, 2, 1, 3], stream).map_err(Into::into)
    }

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
        let width = shape_product_i32("bert", "merged_attention_width", &[heads, head_dim])?;
        reshape(&transposed, &[b, t, width], stream).map_err(Into::into)
    }

    /// Read one tensor from a SafeTensors bundle into an [`MlxArray`].
    fn load_tensor(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<MlxArray> {
        let (floats, shape_i32) = weights.read_as_f32(name)?;
        MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn test_cfg() -> BertConfig {
            BertConfig {
                model_type: "bert".into(),
                hidden_size: 2,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                intermediate_size: 4,
                vocab_size: 4,
                max_position_embeddings: 4,
                type_vocab_size: 2,
                layer_norm_eps: 0.0,
                use_rotary_embeddings: false,
                use_swiglu: false,
                rope_theta: 1000.0,
                rope_parameters: None,
                hidden_act: None,
                quantization: None,
            }
        }

        fn nomic_cfg() -> BertConfig {
            BertConfig {
                model_type: "nomic_bert".into(),
                hidden_size: 2,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                intermediate_size: 4,
                vocab_size: 4,
                max_position_embeddings: 4,
                type_vocab_size: 2,
                layer_norm_eps: 0.0,
                use_rotary_embeddings: true,
                use_swiglu: true,
                rope_theta: 1000.0,
                rope_parameters: None,
                hidden_act: Some("silu".into()),
                quantization: None,
            }
        }

        #[test]
        fn embedding_prelude_applies_canonical_sum_and_layernorm() {
            let _guard = crate::runtime_adapter::mlx::MLX_RUNTIME_TEST_LOCK
                .lock()
                .unwrap();
            let weights = BertWeights {
                word_embeddings: MlxArray::from_slice_f32(
                    &[0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                    &[4, 2],
                )
                .unwrap(),
                position_embeddings: Some(
                    MlxArray::from_slice_f32(
                        &[100.0, 1000.0, 200.0, 2000.0, 300.0, 3000.0, 400.0, 4000.0],
                        &[4, 2],
                    )
                    .unwrap(),
                ),
                token_type_embeddings: Some(
                    MlxArray::from_slice_f32(&[7.0, 70.0, 8.0, 80.0], &[2, 2]).unwrap(),
                ),
                embeddings_layernorm: Some((
                    MlxArray::from_slice_f32(&[2.0, 3.0], &[2]).unwrap(),
                    MlxArray::from_slice_f32(&[0.5, -0.5], &[2]).unwrap(),
                )),
                canonical_layers: Vec::new(),
                nomic_layers: Vec::new(),
            };

            let out = embedding_prelude(&test_cfg(), &weights, &[1, 2], None).unwrap();
            assert_eq!(out.shape(), vec![1, 2, 2]);
            let got = out.to_vec_f32().unwrap();
            let expected = [-1.5_f32, 2.5, -1.5, 2.5];
            for (idx, (g, e)) in got.iter().zip(expected).enumerate() {
                let diff = (g - e).abs();
                assert!(
                    diff <= 1.0e-5,
                    "element {idx} differs: got {g}, expected {e}, diff {diff}"
                );
            }
        }

        #[test]
        fn canonical_forward_runs_one_encoder_layer() {
            let _guard = crate::runtime_adapter::mlx::MLX_RUNTIME_TEST_LOCK
                .lock()
                .unwrap();
            let weights = BertWeights {
                word_embeddings: MlxArray::from_slice_f32(
                    &[0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                    &[4, 2],
                )
                .unwrap(),
                position_embeddings: Some(
                    MlxArray::from_slice_f32(
                        &[100.0, 1000.0, 200.0, 2000.0, 300.0, 3000.0, 400.0, 4000.0],
                        &[4, 2],
                    )
                    .unwrap(),
                ),
                token_type_embeddings: Some(
                    MlxArray::from_slice_f32(&[7.0, 70.0, 8.0, 80.0], &[2, 2]).unwrap(),
                ),
                embeddings_layernorm: Some((
                    MlxArray::from_slice_f32(&[2.0, 3.0], &[2]).unwrap(),
                    MlxArray::from_slice_f32(&[0.5, -0.5], &[2]).unwrap(),
                )),
                canonical_layers: vec![zero_projection_layer()],
                nomic_layers: Vec::new(),
            };

            let out = forward(&test_cfg(), &weights, &[1, 2], None).unwrap();
            assert_eq!(out.shape(), vec![1, 2, 2]);
            let got = out.to_vec_f32().unwrap();
            let expected = [-1.0_f32, 1.0, -1.0, 1.0];
            for (idx, (g, e)) in got.iter().zip(expected).enumerate() {
                let diff = (g - e).abs();
                assert!(
                    diff <= 1.0e-5,
                    "element {idx} differs: got {g}, expected {e}, diff {diff}"
                );
            }
        }

        #[test]
        fn nomic_forward_runs_one_encoder_layer() {
            let _guard = crate::runtime_adapter::mlx::MLX_RUNTIME_TEST_LOCK
                .lock()
                .unwrap();
            let weights = BertWeights {
                word_embeddings: MlxArray::from_slice_f32(
                    &[0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                    &[4, 2],
                )
                .unwrap(),
                position_embeddings: None,
                token_type_embeddings: Some(
                    MlxArray::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
                ),
                embeddings_layernorm: Some((
                    MlxArray::from_slice_f32(&[1.0; 2], &[2]).unwrap(),
                    MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                )),
                canonical_layers: Vec::new(),
                nomic_layers: vec![zero_projection_nomic_layer()],
            };

            let out = forward(&nomic_cfg(), &weights, &[1, 2], None).unwrap();
            assert_eq!(out.shape(), vec![1, 2, 2]);
            let got = out.to_vec_f32().unwrap();
            let expected = [-1.0_f32, 1.0, -1.0, 1.0];
            for (idx, (g, e)) in got.iter().zip(expected).enumerate() {
                let diff = (g - e).abs();
                assert!(
                    diff <= 1.0e-5,
                    "element {idx} differs: got {g}, expected {e}, diff {diff}"
                );
            }
        }

        fn zero_projection_layer() -> CanonicalBertLayer {
            CanonicalBertLayer {
                query_weight: MlxArray::from_slice_f32(&[0.0; 4], &[2, 2]).unwrap(),
                query_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                key_weight: MlxArray::from_slice_f32(&[0.0; 4], &[2, 2]).unwrap(),
                key_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                value_weight: MlxArray::from_slice_f32(&[0.0; 4], &[2, 2]).unwrap(),
                value_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                attention_output_weight: MlxArray::from_slice_f32(&[0.0; 4], &[2, 2]).unwrap(),
                attention_output_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                attention_layernorm_weight: MlxArray::from_slice_f32(&[1.0; 2], &[2]).unwrap(),
                attention_layernorm_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                intermediate_weight: MlxArray::from_slice_f32(&[0.0; 8], &[4, 2]).unwrap(),
                intermediate_bias: MlxArray::from_slice_f32(&[0.0; 4], &[4]).unwrap(),
                output_weight: MlxArray::from_slice_f32(&[0.0; 8], &[2, 4]).unwrap(),
                output_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                output_layernorm_weight: MlxArray::from_slice_f32(&[1.0; 2], &[2]).unwrap(),
                output_layernorm_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
            }
        }

        fn zero_projection_nomic_layer() -> NomicBertLayer {
            NomicBertLayer {
                wqkv_weight: MlxArray::from_slice_f32(&[0.0; 12], &[6, 2]).unwrap(),
                out_proj_weight: MlxArray::from_slice_f32(&[0.0; 4], &[2, 2]).unwrap(),
                norm1_weight: MlxArray::from_slice_f32(&[1.0; 2], &[2]).unwrap(),
                norm1_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
                fc11_weight: MlxArray::from_slice_f32(&[0.0; 8], &[4, 2]).unwrap(),
                fc12_weight: MlxArray::from_slice_f32(&[0.0; 8], &[4, 2]).unwrap(),
                fc2_weight: MlxArray::from_slice_f32(&[0.0; 8], &[2, 4]).unwrap(),
                norm2_weight: MlxArray::from_slice_f32(&[1.0; 2], &[2]).unwrap(),
                norm2_bias: MlxArray::from_slice_f32(&[0.0; 2], &[2]).unwrap(),
            }
        }
    }
}

// =============================================================================
// Tests (no bindings required)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_canonical_config(n_layers: usize) -> BertConfig {
        BertConfig {
            model_type: "bert".into(),
            hidden_size: 16,
            num_hidden_layers: n_layers,
            num_attention_heads: 4,
            intermediate_size: 32,
            vocab_size: 100,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            use_rotary_embeddings: false,
            use_swiglu: false,
            rope_theta: 1000.0,
            rope_parameters: None,
            hidden_act: None,
            quantization: None,
        }
    }

    fn dummy_nomic_config(n_layers: usize) -> BertConfig {
        BertConfig {
            model_type: "nomic_bert".into(),
            hidden_size: 16,
            num_hidden_layers: n_layers,
            num_attention_heads: 4,
            intermediate_size: 32,
            vocab_size: 100,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            use_rotary_embeddings: true,
            use_swiglu: true,
            rope_theta: 1000.0,
            rope_parameters: None,
            hidden_act: Some("silu".into()),
            quantization: None,
        }
    }

    #[test]
    fn canonical_expected_keys_match_per_layer_schedule() {
        let cfg = dummy_canonical_config(2);
        let keys = expected_weight_keys(&cfg);
        // 5 (embeddings: word + position + type + LN.weight + LN.bias)
        // + 2 layers * 16 (per-layer) = 37.
        assert_eq!(keys.len(), 37);
        assert_eq!(keys.first().unwrap(), "embeddings.word_embeddings.weight");
        assert!(keys
            .iter()
            .any(|k| k == "embeddings.position_embeddings.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "encoder.layer.0.attention.self.query.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "encoder.layer.1.output.LayerNorm.bias"));
    }

    #[test]
    fn canonical_with_rotary_skips_position_embeddings() {
        let mut cfg = dummy_canonical_config(1);
        cfg.use_rotary_embeddings = true;
        let keys = expected_weight_keys(&cfg);
        assert!(!keys
            .iter()
            .any(|k| k == "embeddings.position_embeddings.weight"));
    }

    #[test]
    fn nomic_expected_keys_match_per_layer_schedule() {
        let cfg = dummy_nomic_config(2);
        let keys = expected_weight_keys(&cfg);
        // 4 (embeddings: word + token_type + emb_ln weight/bias)
        // + 2 layers * 9 = 22.
        assert_eq!(keys.len(), 22);
        assert_eq!(keys.first().unwrap(), "embeddings.word_embeddings.weight");
        assert!(keys.iter().any(|k| k == "emb_ln.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "encoder.layers.0.attn.Wqkv.weight"));
        assert!(keys.iter().any(|k| k == "encoder.layers.0.mlp.fc11.weight"));
        assert!(keys.iter().any(|k| k == "encoder.layers.1.mlp.fc2.weight"));
    }

    #[test]
    fn nomic_skips_token_type_embeddings_when_vocab_zero() {
        let mut cfg = dummy_nomic_config(1);
        cfg.type_vocab_size = 0;
        let keys = expected_weight_keys(&cfg);
        assert!(!keys
            .iter()
            .any(|k| k == "embeddings.token_type_embeddings.weight"));
    }

    #[test]
    fn head_dim_is_hidden_over_heads() {
        let cfg = BertConfig {
            model_type: "bert".into(),
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            vocab_size: 30522,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            use_rotary_embeddings: false,
            use_swiglu: false,
            rope_theta: 1000.0,
            rope_parameters: None,
            hidden_act: None,
            quantization: None,
        };
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn nomic_native_transformers_config_parses_defaults() {
        let raw = serde_json::json!({
            "model_type": "nomic_bert",
            "vocab_size": 30528,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "silu",
            "max_position_embeddings": 8192,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12
        });
        let cfg: BertConfig = serde_json::from_value(raw).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.is_nomic());
        assert!(cfg.uses_rotary_embeddings());
        assert_eq!(cfg.rope_theta(), 1000.0);
        assert_eq!(cfg.hidden_act.as_deref(), Some("silu"));
    }

    #[test]
    fn nomic_remote_code_config_aliases_parse() {
        let raw = serde_json::json!({
            "model_type": "nomic_bert",
            "vocab_size": 30528,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "activation_function": "swiglu",
            "n_positions": 8192,
            "type_vocab_size": 2,
            "layer_norm_epsilon": 1e-12,
            "rotary_emb_base": 1000.0
        });
        let cfg: BertConfig = serde_json::from_value(raw).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.max_position_embeddings, 8192);
        assert_eq!(cfg.hidden_act.as_deref(), Some("swiglu"));
        assert_eq!(cfg.rope_theta(), 1000.0);
    }

    #[test]
    fn unsupported_model_type_rejected() {
        let mut cfg = dummy_canonical_config(1);
        cfg.model_type = "roberta".into();
        assert!(matches!(
            cfg.validate(),
            Err(MlxLlmError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn zero_dimensions_rejected() {
        let mut cfg = dummy_canonical_config(1);
        cfg.vocab_size = 0;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("zero-valued")),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn hidden_not_divisible_by_heads_rejected() {
        let mut cfg = dummy_canonical_config(1);
        cfg.hidden_size = 17; // not a multiple of 4 heads
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("multiple of")),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn oversized_dimensions_rejected() {
        let mut cfg = dummy_canonical_config(1);
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
    fn oversized_qkv_projection_width_rejected() {
        let mut cfg = dummy_nomic_config(1);
        cfg.hidden_size = i32::MAX as usize / 3 + 1;
        cfg.num_attention_heads = 1;
        match cfg.validate().unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
                assert!(msg.contains("qkv_projection_width"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn quantized_rejected_by_safetensors_check() {
        let mut cfg = dummy_canonical_config(1);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
        });
        let err = validate_safetensors(Path::new("/nonexistent"), &cfg).unwrap_err();
        match &err {
            MlxLlmError::UnsupportedQuantization {
                model_type,
                bits,
                group_size,
                ..
            } => {
                assert_eq!(model_type, "bert");
                assert_eq!(*bits, 4);
                assert_eq!(*group_size, 64);
            }
            other => panic!("expected UnsupportedQuantization, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("unsupported MLX quantization"), "got: {msg}");
        assert!(msg.contains("bert"), "got: {msg}");
        assert!(msg.contains("4-bit/group=64"), "got: {msg}");
        assert!(msg.contains("GGUF fallback"), "got: {msg}");
    }

    #[test]
    fn resolve_weights_missing_returns_missing_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let err = resolve_weights_bundle(tmp.path()).unwrap_err();
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
        let cfg = dummy_canonical_config(1);
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
    fn validate_safetensors_accepts_full_canonical_manifest() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = dummy_canonical_config(1);
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

        validate_safetensors(&path, &cfg).expect("full canonical manifest should validate");
    }

    #[test]
    fn validate_safetensors_accepts_full_nomic_manifest() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = dummy_nomic_config(2);
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

        validate_safetensors(&path, &cfg).expect("full nomic manifest should validate");
    }

    #[test]
    fn validate_safetensors_accepts_indexed_nomic_manifest() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = dummy_nomic_config(2);
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
        let single = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, &None, &single).unwrap();

        let shard_name = "model-00001-of-00001.safetensors";
        let shard = tmp.path().join(shard_name);
        std::fs::rename(&single, &shard).unwrap();
        let bytes = std::fs::read(&shard).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let weight_map = meta
            .tensors()
            .into_keys()
            .map(|name| (name, serde_json::Value::String(shard_name.to_string())))
            .collect::<serde_json::Map<_, _>>();
        let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            index.to_string(),
        )
        .unwrap();

        let bundle = resolve_weights_bundle(tmp.path()).expect("resolve indexed bundle");
        validate_safetensors_bundle(&bundle, &cfg).expect("indexed nomic manifest should validate");
    }

    #[test]
    fn is_nomic_predicate() {
        assert!(!dummy_canonical_config(1).is_nomic());
        assert!(dummy_nomic_config(1).is_nomic());
    }

    #[test]
    fn f32_decoder_handles_unaligned_slices() {
        let mut bytes = vec![0xAA];
        bytes.extend_from_slice(&1.25f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.5f32).to_le_bytes());

        let decoded = f32_le_bytes_to_vec(&bytes[1..]).unwrap();

        assert_eq!(decoded, vec![1.25, -2.5]);
    }

    #[test]
    fn f32_decoder_rejects_truncated_bytes() {
        let err = f32_le_bytes_to_vec(&[0, 1, 2]).unwrap_err();

        assert!(err.contains("not divisible by 4"), "{err}");
    }

    #[test]
    fn safetensor_shape_rejects_dims_over_mlx_i32_limit() {
        let err = safetensor_shape_to_i32(
            &[2, i32::MAX as usize + 1],
            "embeddings.word_embeddings.weight",
        )
        .unwrap_err();

        match err {
            MlxLlmError::WeightLoad { reason, .. } => {
                assert!(reason.contains("exceeds MLX i32 shape limit"), "{reason}");
                assert!(reason.contains("dimension 1"), "{reason}");
            }
            other => panic!("expected WeightLoad, got {other:?}"),
        }
    }
}

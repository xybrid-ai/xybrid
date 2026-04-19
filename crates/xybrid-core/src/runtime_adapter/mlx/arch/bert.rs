//! BERT-family encoder architecture builder for MLX embeddings.
//!
//! BERT differs from the decoder-only LLM architectures (Qwen 3.5, Gemma 4,
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
//! This module ships the skeleton (config parsing, weight-key
//! enumeration, safetensors header validation) for canonical BERT
//! (`model_type = "bert"`) and the nomic variant (`model_type =
//! "nomic_bert"`). The forward-pass runtime is staged but bails at the
//! encoder boundary with [`MlxLlmError::NotImplemented`] — the GeLU
//! primitive isn't yet wrapped in `xybrid_mlx::ops` and the nomic
//! Wqkv-fused layout needs a dedicated load path. Both are tracked as
//! follow-ups; the embedding integration test against
//! `mlx-community/nomic-embed-text-v1.5` is deferred for the same reason
//! the LLM arch builders deferred their generate-loop integration tests
//! (needs the macOS xcframework runner staged).
//!
//! Reference (upstream):
//! - <https://github.com/ml-explore/mlx-examples/tree/main/bert> (canonical)
//! - <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5> (nomic variant)

use std::fs;
use std::path::{Path, PathBuf};

use safetensors::SafeTensors;
use serde::Deserialize;

use super::super::model::{MlxLlmError, MlxLlmResult};

// =============================================================================
// Config
// =============================================================================

/// Subset of an MLX-style BERT `config.json` the builder needs.
///
/// Accepted `model_type` values: `"bert"` (canonical) and `"nomic_bert"`
/// (nomic-embed variant). Unknown families are rejected at parse time
/// via the internal `validate()` invariant check.
#[derive(Debug, Clone, Deserialize)]
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
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    /// LayerNorm epsilon. Canonical BERT uses 1e-12, nomic uses 1e-12.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// `true` when this variant uses rotary position embeddings instead
    /// of learned absolute position embeddings. The skeleton inspects
    /// this when emitting the expected weight-key schedule (presence of
    /// `embeddings.position_embeddings.weight`).
    #[serde(default)]
    pub use_rotary_embeddings: bool,
    /// `true` when the FFN uses a SwiGLU gating pair instead of the
    /// canonical single-linear + GeLU. nomic-embed-text-v1.5 sets this.
    #[serde(default)]
    pub use_swiglu: bool,
    /// Set when weights are quantized.
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

fn default_type_vocab_size() -> usize {
    2
}

fn default_layer_norm_eps() -> f32 {
    1e-12
}

/// Quantization metadata block from `config.json`. Mirrors the LLM
/// arch builders.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    pub bits: u32,
    pub group_size: u32,
}

impl BertConfig {
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
/// **nomic_bert** (`model_type = "nomic_bert"`): per layer (10 tensors)
/// - `bert.encoder.layer.{l}.attention.self.Wqkv.{weight,bias}` (2; fused
///   QKV)
/// - `bert.encoder.layer.{l}.attention.output.dense.weight` (1; nomic
///   omits the bias)
/// - `bert.encoder.layer.{l}.attention.output.LayerNorm.{weight,bias}` (2)
/// - `bert.encoder.layer.{l}.mlp.{fc11,fc12,fc2}.weight` (3; SwiGLU
///   gate / up / down — no bias)
/// - `bert.encoder.layer.{l}.mlp.LayerNorm.{weight,bias}` (2)
///
/// Shared across blocks:
/// - Canonical: `embeddings.word_embeddings.weight`,
///   `embeddings.position_embeddings.weight`,
///   `embeddings.token_type_embeddings.weight`,
///   `embeddings.LayerNorm.{weight,bias}`
/// - nomic: `bert.embeddings.word_embeddings.weight` plus optional
///   `bert.embeddings.token_type_embeddings.weight` (gated on
///   `type_vocab_size > 1`); the position-embedding tensor is omitted
///   because RoPE is applied inside attention.
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
    if !cfg.use_rotary_embeddings {
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
    let mut keys = Vec::with_capacity(2 + 10 * cfg.num_hidden_layers);
    keys.push("bert.embeddings.word_embeddings.weight".to_string());
    if cfg.type_vocab_size > 1 {
        keys.push("bert.embeddings.token_type_embeddings.weight".to_string());
    }
    for l in 0..cfg.num_hidden_layers {
        let base = format!("bert.encoder.layer.{l}");
        keys.push(format!("{base}.attention.self.Wqkv.weight"));
        keys.push(format!("{base}.attention.self.Wqkv.bias"));
        keys.push(format!("{base}.attention.output.dense.weight"));
        keys.push(format!("{base}.attention.output.LayerNorm.weight"));
        keys.push(format!("{base}.attention.output.LayerNorm.bias"));
        keys.push(format!("{base}.mlp.fc11.weight"));
        keys.push(format!("{base}.mlp.fc12.weight"));
        keys.push(format!("{base}.mlp.fc2.weight"));
        keys.push(format!("{base}.mlp.LayerNorm.weight"));
        keys.push(format!("{base}.mlp.LayerNorm.bias"));
    }
    keys
}

// =============================================================================
// Safetensors validation
// =============================================================================

/// Validate that `model.safetensors` contains every key the encoder
/// reads. Mirrors the LLM arch builders' header-only validation pattern.
pub fn validate_safetensors(path: &Path, cfg: &BertConfig) -> MlxLlmResult<()> {
    if cfg.is_quantized() {
        let bits = cfg.quantization.as_ref().map(|q| q.bits).unwrap_or(0);
        let gs = cfg.quantization.as_ref().map(|q| q.group_size).unwrap_or(0);
        return Err(MlxLlmError::UnsupportedArchitecture {
            model_type: format!(
                "{} (quantized {bits}-bit/group={gs} — mlx_fast_quantized_matmul not yet wired for BERT)",
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

/// Top-level entry: load and validate a BERT bundle's config + weight
/// manifest. Returns the parsed config so the embedding adapter's load
/// path can cache it.
pub fn load(model_dir: &Path, weights_path: &Path) -> MlxLlmResult<BertConfig> {
    let cfg = BertConfig::from_model_dir(model_dir)?;
    validate_safetensors(weights_path, &cfg)?;
    Ok(cfg)
}

/// Resolve the safetensors weight file for a bundle. Single-file layout
/// only — sharded `model-{i}-of-{n}.safetensors` support lands as a
/// follow-up (the same helper would apply to every arch builder).
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

/// MLX-backed BERT encoder forward pass — only compiled when the
/// xcframework can be linked.
///
/// The runtime stages weight materialisation behind a typed
/// [`runtime::BertWeights`] surface but defers the full forward pass at
/// the encoder boundary because the GeLU activation primitive isn't
/// yet wrapped in `xybrid_mlx::ops` (the existing wrappers cover
/// `silu / sigmoid / exp / reciprocal` from US-014; `gelu` is a follow-
/// up). Once `gelu` lands, the deferred section becomes a straight port
/// of the canonical BERT encoder with mean / cls / last_token pooling
/// applied by the embedding adapter on the output hidden state.
#[cfg(all(
    feature = "llm-mlx-runtime",
    any(target_os = "macos", target_os = "ios")
))]
pub mod runtime {
    use super::super::super::model::{MlxLlmError, MlxLlmResult};
    use super::BertConfig;

    use std::path::Path;

    use safetensors::{Dtype as StDtype, SafeTensors};
    use xybrid_mlx::MlxArray;

    /// Materialised BERT weights resident in MLX memory. Currently just
    /// the embedding tensors — the per-layer encoder weights land
    /// alongside the GeLU primitive that the forward pass needs.
    #[derive(Debug)]
    pub struct BertWeights {
        pub word_embeddings: MlxArray,
        pub position_embeddings: Option<MlxArray>,
        pub token_type_embeddings: Option<MlxArray>,
        pub embeddings_layernorm: Option<(MlxArray, MlxArray)>,
    }

    /// Stage one of the BERT load: materialise embedding tensors only.
    /// Returns weights ready for the embedding-lookup half of the
    /// forward pass; the encoder half lands with the GeLU primitive.
    pub fn build(cfg: &BertConfig, weights_path: &Path) -> MlxLlmResult<BertWeights> {
        let bytes = std::fs::read(weights_path).map_err(|e| MlxLlmError::WeightLoad {
            path: weights_path.to_path_buf(),
            reason: format!("read safetensors: {e}"),
        })?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| MlxLlmError::WeightLoad {
            path: weights_path.to_path_buf(),
            reason: format!("parse safetensors: {e}"),
        })?;

        let nomic = cfg.is_nomic();
        let word_key = if nomic {
            "bert.embeddings.word_embeddings.weight"
        } else {
            "embeddings.word_embeddings.weight"
        };
        let word_embeddings = load_tensor(&st, word_key)?;

        let position_embeddings = if cfg.use_rotary_embeddings || nomic {
            None
        } else {
            Some(load_tensor(&st, "embeddings.position_embeddings.weight")?)
        };

        let type_key = if nomic {
            "bert.embeddings.token_type_embeddings.weight"
        } else {
            "embeddings.token_type_embeddings.weight"
        };
        let token_type_embeddings = if nomic && cfg.type_vocab_size <= 1 {
            None
        } else {
            Some(load_tensor(&st, type_key)?)
        };

        let embeddings_layernorm = if nomic {
            // nomic_bert applies LayerNorm inside the first encoder block,
            // not on the embeddings sum.
            None
        } else {
            Some((
                load_tensor(&st, "embeddings.LayerNorm.weight")?,
                load_tensor(&st, "embeddings.LayerNorm.bias")?,
            ))
        };

        Ok(BertWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_layernorm,
        })
    }

    /// BERT encoder forward pass — staged. Materialises weights and
    /// returns [`MlxLlmError::NotImplemented`] at the first encoder
    /// block boundary because the GeLU primitive isn't yet wrapped in
    /// `xybrid_mlx::ops`.
    pub fn forward(_cfg: &BertConfig, _weights: &BertWeights) -> MlxLlmResult<MlxArray> {
        Err(MlxLlmError::NotImplemented {
            feature: "BERT encoder forward pass (needs xybrid_mlx::ops::gelu)",
            story: "US-015 follow-up",
        })
    }

    /// Read one tensor from a SafeTensors view into an [`MlxArray`].
    /// Mirrors the loader in the LLM arch builders; the duplication is
    /// intentional (cfg-gated runtime modules stay self-contained per
    /// the codebase pattern documented in `progress.txt`).
    fn load_tensor(st: &SafeTensors<'_>, name: &str) -> MlxLlmResult<MlxArray> {
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

        match view.dtype() {
            StDtype::F32 => {
                let floats: &[f32] = cast_to_f32(data);
                MlxArray::from_slice_f32(floats, &shape_i32).map_err(Into::into)
            }
            StDtype::F16 => {
                let floats: Vec<f32> = f16_slice_to_f32(data);
                MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
            }
            StDtype::BF16 => {
                let floats: Vec<f32> = bf16_slice_to_f32(data);
                MlxArray::from_slice_f32(&floats, &shape_i32).map_err(Into::into)
            }
            other => Err(MlxLlmError::WeightLoad {
                path: std::path::PathBuf::from(name),
                reason: format!("unsupported tensor dtype {other:?}"),
            }),
        }
    }

    fn cast_to_f32(bytes: &[u8]) -> &[f32] {
        debug_assert!(bytes.len().is_multiple_of(4));
        debug_assert_eq!(bytes.as_ptr().align_offset(align_of::<f32>()), 0);
        // SAFETY: alignment and length divisibility asserted above.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
    }

    fn f16_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    fn bf16_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
                f32::from_bits(bits)
            })
            .collect()
    }

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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn half_to_f32_zero_and_one() {
            assert_eq!(super::half_to_f32(0x0000), 0.0_f32);
            assert_eq!(super::half_to_f32(0x3c00), 1.0_f32);
            assert_eq!(super::half_to_f32(0xbc00), -1.0_f32);
        }
    }
}

// =============================================================================
// Tests (skeleton — no bindings required)
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
        // 2 (embeddings: word + token_type) + 2 layers * 10 = 22.
        assert_eq!(keys.len(), 22);
        assert_eq!(
            keys.first().unwrap(),
            "bert.embeddings.word_embeddings.weight"
        );
        assert!(keys
            .iter()
            .any(|k| k == "bert.encoder.layer.0.attention.self.Wqkv.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "bert.encoder.layer.0.mlp.fc11.weight"));
        assert!(keys
            .iter()
            .any(|k| k == "bert.encoder.layer.1.mlp.fc2.weight"));
    }

    #[test]
    fn nomic_skips_token_type_embeddings_when_vocab_one() {
        let mut cfg = dummy_nomic_config(1);
        cfg.type_vocab_size = 1;
        let keys = expected_weight_keys(&cfg);
        assert!(!keys
            .iter()
            .any(|k| k == "bert.embeddings.token_type_embeddings.weight"));
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
            quantization: None,
        };
        assert_eq!(cfg.head_dim(), 64);
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
    fn quantized_rejected_by_safetensors_check() {
        let mut cfg = dummy_canonical_config(1);
        cfg.quantization = Some(QuantConfig {
            bits: 4,
            group_size: 64,
        });
        let err = validate_safetensors(Path::new("/nonexistent"), &cfg).unwrap_err();
        match err {
            MlxLlmError::UnsupportedArchitecture { model_type } => {
                assert!(model_type.contains("quantized"), "got: {model_type}");
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
    fn is_nomic_predicate() {
        assert!(!dummy_canonical_config(1).is_nomic());
        assert!(dummy_nomic_config(1).is_nomic());
    }
}

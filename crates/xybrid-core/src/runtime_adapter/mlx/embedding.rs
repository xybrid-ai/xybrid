//! `MlxEmbeddingAdapter` — produces dense text embeddings via MLX.
//!
//! Loads an MLX-LM-style bundle (`config.json + tokenizer.json +
//! model.safetensors`), runs the BERT-family encoder forward pass (see
//! [`super::arch::bert`]), applies one of three pooling strategies over
//! the per-token hidden states, optionally L2-normalises the result, and
//! emits the embedding as
//! [`EnvelopeKind::Embedding`](crate::ir::EnvelopeKind) so it flows
//! through pipelines transparently alongside the ONNX embedding path.
//!
//! The adapter is the embedding counterpart to
//! [`super::model::MlxLlmAdapter`] for generative LLMs. It shares the
//! same bundle layout, the same tokenizer loader, and the same
//! cross-platform non-linking vs. Apple-runtime split so non-Apple builds
//! still type-check and registry resolution can keep the default
//! embedding artifact, typically ONNX, when MLX is unavailable.
//!
//! # Pooling
//!
//! | Strategy | Description | Use case |
//! |----------|-------------|----------|
//! | [`Pooling::Mean`] | Mean over non-padding tokens | sentence-transformers default |
//! | [`Pooling::Cls`] | First-token (`[CLS]`) hidden state | classic BERT pooler |
//! | [`Pooling::LastToken`] | Last non-padding token | causal-style pooling |
//!
//! # L2 normalisation
//!
//! When [`MlxEmbeddingConfig::normalize`] is `true` the output is
//! divided by its L2 norm. Matches the sentence-transformers default
//! and is required for cosine-similarity scoring without further
//! preprocessing.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::ir::{Envelope, EnvelopeKind};

use super::arch::bert::{self, BertConfig};
use super::model::{MlxLlmError, MlxLlmResult};
use super::tokenizer as tokenizer_loader;

// =============================================================================
// Pooling strategy
// =============================================================================

/// Strategy for collapsing a `[seq_len, hidden_dim]` hidden-state matrix
/// into a single embedding vector.
///
/// Default is [`Pooling::Mean`] — matches the sentence-transformers
/// convention and is what `nomic-embed-text-v1.5` is trained against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// Mean over non-padding token positions.
    #[default]
    Mean,
    /// First-token (`[CLS]`) hidden state.
    Cls,
    /// Last non-padding token hidden state.
    LastToken,
}

impl Pooling {
    /// Parse a free-form metadata value (`"mean"`, `"cls"`,
    /// `"last_token"`) into a [`Pooling`] variant.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "mean" => Ok(Pooling::Mean),
            "cls" => Ok(Pooling::Cls),
            "last_token" => Ok(Pooling::LastToken),
            other => Err(format!(
                "unknown pooling strategy `{other}` (expected: mean, cls, last_token)"
            )),
        }
    }
}

// =============================================================================
// Config
// =============================================================================

/// Configuration for the MLX embedding adapter.
///
/// Mirrors [`super::model::MlxLlmConfig`] in shape but carries
/// embedding-specific knobs: pooling strategy, L2-normalisation flag,
/// and a soft cap on input sequence length so a runaway tokenisation
/// can't blow up the encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxEmbeddingConfig {
    /// Maximum input sequence length (in tokens) the encoder will
    /// process. Inputs longer than this are truncated; bounded by the
    /// model's `max_position_embeddings`.
    pub max_seq_len: usize,
    /// Pooling strategy used to collapse per-token hidden states into a
    /// single embedding vector. See [`Pooling`].
    pub pooling: Pooling,
    /// When `true`, apply L2-normalisation to the pooled embedding.
    /// Matches the sentence-transformers convention.
    pub normalize: bool,
}

impl Default for MlxEmbeddingConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 512,
            pooling: Pooling::Mean,
            normalize: false,
        }
    }
}

impl MlxEmbeddingConfig {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            max_seq_len,
            pooling: Pooling::Mean,
            normalize: false,
        }
    }

    pub fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = pooling;
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Build a config by reading `pooling` and `normalize` keys from a
    /// generic metadata map (typically the `metadata` field of a
    /// pipeline stage YAML or `model_metadata.json`). Unknown values
    /// for `pooling` produce a [`MlxLlmError::ConfigInvalid`].
    pub fn from_metadata(metadata: &HashMap<String, String>) -> MlxLlmResult<Self> {
        let mut cfg = Self::default();
        if let Some(p) = metadata.get("pooling") {
            cfg.pooling = Pooling::from_str(p).map_err(MlxLlmError::ConfigInvalid)?;
        }
        if let Some(n) = metadata.get("normalize") {
            cfg.normalize = n.eq_ignore_ascii_case("true") || n == "1";
        }
        if let Some(m) = metadata.get("max_seq_len") {
            cfg.max_seq_len = m
                .parse::<usize>()
                .map_err(|e| MlxLlmError::ConfigInvalid(format!("bad max_seq_len `{m}`: {e}")))?;
        }
        validate_embedding_config(&cfg)?;
        Ok(cfg)
    }
}

fn validate_embedding_config(config: &MlxEmbeddingConfig) -> MlxLlmResult<()> {
    if config.max_seq_len == 0 {
        return Err(MlxLlmError::ConfigInvalid(
            "MLX embedding max_seq_len must be >= 1".into(),
        ));
    }
    if config.max_seq_len > i32::MAX as usize {
        return Err(MlxLlmError::ConfigInvalid(format!(
            "MLX embedding max_seq_len={} exceeds MLX i32 shape limit",
            config.max_seq_len
        )));
    }
    Ok(())
}

// =============================================================================
// Adapter
// =============================================================================

/// State held by a loaded [`MlxEmbeddingAdapter`].
#[derive(Debug)]
struct LoadedState {
    model_dir: PathBuf,
    bert_config: BertConfig,
    tokenizer: Tokenizer,
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    runtime_weights: std::sync::Mutex<super::arch::bert::runtime::BertWeights>,
}

/// MLX-backed embedding adapter. The Apple runtime path keeps canonical BERT
/// and current nomic-bert SafeTensors weights resident in loaded adapter state
/// behind a mutex, so `&self` embedding calls can reuse MLX handles without
/// breaking the adapter's thread-safety contract. The cross-platform pooling
/// and normalisation helpers are unit-tested here so adapter coverage can stay
/// independent from MLX linkage on non-Apple builds.
#[derive(Debug)]
pub struct MlxEmbeddingAdapter {
    config: MlxEmbeddingConfig,
    loaded: Option<LoadedState>,
}

impl MlxEmbeddingAdapter {
    /// Empty, unloaded adapter.
    pub fn new(config: MlxEmbeddingConfig) -> Self {
        Self {
            config,
            loaded: None,
        }
    }

    /// One-shot load from a model directory. Reads `config.json`,
    /// `tokenizer.json`, and SafeTensors weights from `model_dir` and
    /// validates the BERT-family weight schedule.
    pub fn load(model_dir: &Path, config: &MlxEmbeddingConfig) -> MlxLlmResult<Self> {
        validate_embedding_config(config)?;
        let mut adapter = Self::new(config.clone());
        adapter.load_in_place(model_dir)?;
        Ok(adapter)
    }

    fn load_in_place(&mut self, model_dir: &Path) -> MlxLlmResult<()> {
        // Required files. Mirrors the LLM adapter's file-presence check
        // so users get pointed errors instead of generic IO failures.
        let config_path = require_file(model_dir, "config.json")?;
        let tokenizer_path = require_file(model_dir, "tokenizer.json")?;
        let weights = bert::resolve_weights_bundle(model_dir)?;

        // We re-read config.json through BertConfig::from_model_dir
        // (which also runs validate()), but verify the file exists
        // first so the missing-file error surface stays consistent.
        let _ = fs::metadata(&config_path)?;
        let bert_cfg = bert::BertConfig::from_model_dir(model_dir)?;
        bert::validate_safetensors_bundle(&weights, &bert_cfg)?;

        let tokenizer = tokenizer_loader::load_from_file(&tokenizer_path)
            .map_err(|e| MlxLlmError::TokenizerLoad(e.to_string()))?;
        #[cfg(all(
            feature = "llm-mlx-runtime",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let runtime_weights = super::arch::bert::runtime::build(&bert_cfg, &weights)?;

        self.loaded = Some(LoadedState {
            model_dir: model_dir.to_path_buf(),
            bert_config: bert_cfg,
            tokenizer,
            #[cfg(all(
                feature = "llm-mlx-runtime",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            runtime_weights: std::sync::Mutex::new(runtime_weights),
        });
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    pub fn unload(&mut self) {
        self.loaded = None;
    }

    pub fn bert_config(&self) -> Option<&BertConfig> {
        self.loaded.as_ref().map(|s| &s.bert_config)
    }

    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.loaded.as_ref().map(|s| &s.tokenizer)
    }

    pub fn config(&self) -> &MlxEmbeddingConfig {
        &self.config
    }

    pub fn model_dir(&self) -> Option<&Path> {
        self.loaded.as_ref().map(|s| s.model_dir.as_path())
    }

    /// Embed a single text input and return an [`EnvelopeKind::Embedding`]
    /// envelope. The Apple-runtime submodule owns the real MLX dispatch;
    /// non-Apple / non-runtime builds surface [`MlxLlmError::NotImplemented`]
    /// as a pointed build-gate error.
    pub fn embed(&self, text: &str) -> MlxLlmResult<Envelope> {
        let state = self.loaded.as_ref().ok_or(MlxLlmError::NotLoaded)?;

        let encoding = state
            .tokenizer
            .encode(text, true)
            .map_err(|e| MlxLlmError::TokenizerLoad(e.to_string()))?;

        validate_embedding_config(&self.config)?;

        let raw_token_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let truncated_len = raw_token_ids.len().min(self.config.max_seq_len);
        if truncated_len == 0 {
            return Err(MlxLlmError::ConfigInvalid(
                "MLX embedding input produced no tokens".into(),
            ));
        }
        let truncation_metadata = embedding_truncation_metadata(raw_token_ids.len(), truncated_len);
        if !truncation_metadata.is_empty() {
            log::warn!(
                target: "xybrid_core",
                "MLX embedding input truncated from {} to {} tokens (max_seq_len={}); \
                 the embedding covers only the clamped prefix",
                raw_token_ids.len(),
                truncated_len,
                self.config.max_seq_len
            );
        }
        if attention_mask.len() < truncated_len {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "tokenizer attention mask length {} is shorter than token length {}",
                attention_mask.len(),
                truncated_len
            )));
        }
        let token_ids =
            token_ids_to_mlx_i32(raw_token_ids, truncated_len, state.bert_config.vocab_size)?;
        let attention_mask = &attention_mask[..truncated_len];

        // The encoder forward pass takes no attention mask: self-attention
        // would attend over `[PAD]` positions and contaminate every content
        // token's hidden state BEFORE the mask-aware pooling runs. Single-text
        // encoding never pads, so a padding token here means the bundle's
        // tokenizer.json configures fixed/right padding — fail loudly instead
        // of silently returning corrupted embeddings.
        if attention_mask.contains(&0) {
            return Err(MlxLlmError::ConfigInvalid(
                "tokenizer produced padding tokens, but the MLX BERT encoder does not \
                 support padded input (the forward pass takes no attention mask); \
                 disable fixed padding in the bundle's tokenizer.json"
                    .into(),
            ));
        }

        let hidden_dim = state.bert_config.hidden_size;
        let hidden = run_encoder(state, &token_ids, attention_mask, hidden_dim)?;

        let mut pooled = apply_pooling(
            &hidden,
            truncated_len,
            hidden_dim,
            attention_mask,
            self.config.pooling,
        );
        if self.config.normalize {
            l2_normalize(&mut pooled);
        }

        let mut envelope = Envelope::new(EnvelopeKind::Embedding(pooled));
        envelope.metadata.extend(truncation_metadata);
        Ok(envelope)
    }
}

/// Truncation markers for an embedding run: empty when the input fit within
/// `max_seq_len`; otherwise `truncated=true` plus the original and clamped
/// token counts so callers can detect that the embedding only covers a
/// prefix of the input.
fn embedding_truncation_metadata(
    original_token_count: usize,
    clamped_token_count: usize,
) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    if clamped_token_count < original_token_count {
        metadata.insert("truncated".to_string(), "true".to_string());
        metadata.insert(
            "original_token_count".to_string(),
            original_token_count.to_string(),
        );
        metadata.insert(
            "clamped_token_count".to_string(),
            clamped_token_count.to_string(),
        );
    }
    metadata
}

fn token_ids_to_mlx_i32(
    raw_token_ids: &[u32],
    len: usize,
    vocab_size: usize,
) -> MlxLlmResult<Vec<i32>> {
    if len > raw_token_ids.len() {
        return Err(MlxLlmError::ConfigInvalid(format!(
            "requested {len} token ids from tokenizer output of length {}",
            raw_token_ids.len()
        )));
    }

    let mut token_ids = Vec::with_capacity(len);
    for (idx, &id) in raw_token_ids[..len].iter().enumerate() {
        let id_i32 = i32::try_from(id).map_err(|_| {
            MlxLlmError::ConfigInvalid(format!(
                "MLX embedding token id {id} at position {idx} exceeds MLX i32 token-id limit"
            ))
        })?;
        let id_usize = usize::try_from(id).map_err(|_| {
            MlxLlmError::ConfigInvalid(format!(
                "MLX embedding token id {id} at position {idx} exceeds this platform's usize limit"
            ))
        })?;
        if id_usize >= vocab_size {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "MLX embedding token id {id} at position {idx} exceeds vocab_size={vocab_size}"
            )));
        }
        token_ids.push(id_i32);
    }
    Ok(token_ids)
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

// =============================================================================
// Pooling helpers (cross-platform, unit-tested)
// =============================================================================

/// Apply the configured pooling strategy to a flat
/// `[seq_len * hidden_dim]` hidden-state buffer and return a single
/// `[hidden_dim]` embedding vector.
///
/// The buffer is laid out row-major over `(seq_idx, hidden_idx)` —
/// `hidden[s * hidden_dim + h]` is the hidden value at position `s` in
/// dimension `h`. The `attention_mask` slice (length `seq_len`)
/// distinguishes content tokens (`1`) from padding (`0`); padding
/// tokens are excluded from [`Pooling::Mean`] and [`Pooling::LastToken`].
///
/// # Edge cases
///
/// - When `seq_len == 0`: returns an all-zero vector of length
///   `hidden_dim`.
/// - When [`Pooling::Mean`] is selected but the attention mask sums to
///   zero (no content tokens): falls back to a plain mean over all
///   positions.
/// - When [`Pooling::LastToken`] is selected but no content token is
///   present: returns the embedding at the last position.
pub fn apply_pooling(
    hidden: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    attention_mask: &[u32],
    pooling: Pooling,
) -> Vec<f32> {
    if seq_len == 0 || hidden_dim == 0 {
        return vec![0.0; hidden_dim];
    }
    debug_assert_eq!(
        hidden.len(),
        seq_len * hidden_dim,
        "hidden must be flat [seq_len * hidden_dim]"
    );
    debug_assert_eq!(
        attention_mask.len(),
        seq_len,
        "attention_mask must have length seq_len"
    );

    match pooling {
        Pooling::Cls => hidden[0..hidden_dim].to_vec(),
        Pooling::Mean => mean_pool(hidden, seq_len, hidden_dim, attention_mask),
        Pooling::LastToken => last_token_pool(hidden, seq_len, hidden_dim, attention_mask),
    }
}

fn mean_pool(
    hidden: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    attention_mask: &[u32],
) -> Vec<f32> {
    let mut sum = vec![0.0_f32; hidden_dim];
    let mut count: usize = 0;
    for s in 0..seq_len {
        if attention_mask[s] == 0 {
            continue;
        }
        let row = &hidden[s * hidden_dim..(s + 1) * hidden_dim];
        for (acc, &v) in sum.iter_mut().zip(row) {
            *acc += v;
        }
        count += 1;
    }
    if count == 0 {
        // No content tokens — fall back to a plain average over all
        // positions so the caller still gets a non-degenerate vector.
        for s in 0..seq_len {
            let row = &hidden[s * hidden_dim..(s + 1) * hidden_dim];
            for (acc, &v) in sum.iter_mut().zip(row) {
                *acc += v;
            }
        }
        let denom = seq_len as f32;
        for v in &mut sum {
            *v /= denom;
        }
    } else {
        let denom = count as f32;
        for v in &mut sum {
            *v /= denom;
        }
    }
    sum
}

fn last_token_pool(
    hidden: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    attention_mask: &[u32],
) -> Vec<f32> {
    // Walk backwards for the first content token; fall back to the
    // very last position when no mask bit is set.
    let last = (0..seq_len)
        .rev()
        .find(|&s| attention_mask[s] != 0)
        .unwrap_or(seq_len - 1);
    hidden[last * hidden_dim..(last + 1) * hidden_dim].to_vec()
}

/// In-place L2 normalisation. A zero vector is left untouched (avoids
/// division by zero) — callers that need to detect the degenerate case
/// should inspect [`l2_norm`] directly.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm == 0.0 || !norm.is_finite() {
        return;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Euclidean (L2) norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// =============================================================================
// Encoder dispatch (cross-platform / Apple-runtime split)
// =============================================================================

/// Run the BERT encoder forward pass and return a flat
/// `[seq_len * hidden_dim]` hidden-state buffer.
///
/// Cross-platform / non-runtime builds surface [`MlxLlmError::NotImplemented`]
/// pointing at the `llm-mlx-runtime` feature gate. The Apple-runtime submodule
/// below owns the real MLX dispatch.
#[cfg(not(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
)))]
fn run_encoder(
    _state: &LoadedState,
    _token_ids: &[i32],
    _attention_mask: &[u32],
    _hidden_dim: usize,
) -> MlxLlmResult<Vec<f32>> {
    Err(MlxLlmError::NotImplemented {
        feature:
            "MLX BERT encoder forward pass (build with --features llm-mlx-runtime on Apple Silicon macOS, \
             then materialize mlx.xcframework via tools/scripts/fetch-mlx-xcframework.sh \
             or tools/scripts/build-local-mlx-xcframework.sh)",
        story: "llm-mlx-runtime",
    })
}

#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
fn run_encoder(
    state: &LoadedState,
    token_ids: &[i32],
    _attention_mask: &[u32],
    hidden_dim: usize,
) -> MlxLlmResult<Vec<f32>> {
    let weights = state.runtime_weights.lock().map_err(|_| {
        MlxLlmError::ConfigInvalid("BERT runtime weight cache lock is poisoned".into())
    })?;
    let hidden =
        super::arch::bert::runtime::forward(&state.bert_config, &weights, token_ids, None)?;
    let data = hidden.to_vec_f32()?;
    if data.len() != token_ids.len() * hidden_dim {
        return Err(MlxLlmError::ConfigInvalid(format!(
            "BERT encoder returned {} values, expected {} (seq_len={} * hidden_dim={})",
            data.len(),
            token_ids.len() * hidden_dim,
            token_ids.len(),
            hidden_dim
        )));
    }
    Ok(data)
}

// =============================================================================
// Tests (cross-platform — pooling and normalisation helpers)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    // -----------------------------------------------------------------
    // Pooling / normalisation helpers
    // -----------------------------------------------------------------

    fn flat_hidden(seq_len: usize, hidden_dim: usize) -> Vec<f32> {
        // [s * hidden_dim + h] = (s + 1) * (h + 1) — distinct values
        // per position so pooling differences are visible.
        let mut out = Vec::with_capacity(seq_len * hidden_dim);
        for s in 0..seq_len {
            for h in 0..hidden_dim {
                out.push(((s + 1) * (h + 1)) as f32);
            }
        }
        out
    }

    #[test]
    fn pooling_default_is_mean() {
        assert_eq!(Pooling::default(), Pooling::Mean);
    }

    #[test]
    fn pooling_from_str_round_trip() {
        assert_eq!(Pooling::from_str("mean").unwrap(), Pooling::Mean);
        assert_eq!(Pooling::from_str("cls").unwrap(), Pooling::Cls);
        assert_eq!(Pooling::from_str("last_token").unwrap(), Pooling::LastToken);
        assert!(Pooling::from_str("median").is_err());
    }

    #[test]
    fn cls_pooling_returns_first_row() {
        let hidden = flat_hidden(3, 4);
        let mask = vec![1, 1, 1];
        let v = apply_pooling(&hidden, 3, 4, &mask, Pooling::Cls);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mean_pooling_skips_padding() {
        // 3 tokens × 2 dims; second is padding.
        let hidden = vec![1.0, 2.0, 99.0, 99.0, 5.0, 6.0];
        let mask = vec![1, 0, 1];
        let v = apply_pooling(&hidden, 3, 2, &mask, Pooling::Mean);
        // Mean of [(1,2), (5,6)] = (3.0, 4.0).
        assert_eq!(v, vec![3.0, 4.0]);
    }

    #[test]
    fn mean_pooling_falls_back_when_mask_all_zero() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0, 0];
        let v = apply_pooling(&hidden, 2, 2, &mask, Pooling::Mean);
        // Plain average over both positions = (2.0, 3.0).
        assert_eq!(v, vec![2.0, 3.0]);
    }

    #[test]
    fn last_token_pooling_picks_last_content_position() {
        // 4 tokens × 2 dims; positions 0,1,2 are content, 3 is padding.
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 99.0];
        let mask = vec![1, 1, 1, 0];
        let v = apply_pooling(&hidden, 4, 2, &mask, Pooling::LastToken);
        // Last content token is at index 2 = (5, 6).
        assert_eq!(v, vec![5.0, 6.0]);
    }

    #[test]
    fn last_token_pooling_falls_back_to_seq_end_when_all_padding() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0, 0];
        let v = apply_pooling(&hidden, 2, 2, &mask, Pooling::LastToken);
        // No content token — fall back to the very last position.
        assert_eq!(v, vec![3.0, 4.0]);
    }

    #[test]
    fn pooling_handles_zero_seq_len() {
        let v = apply_pooling(&[], 0, 4, &[], Pooling::Mean);
        assert_eq!(v, vec![0.0; 4]);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0, 4.0]; // L2 norm = 5.0.
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_unchanged() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_norm_nonfinite_left_alone() {
        let mut v = vec![f32::NAN, 1.0];
        l2_normalize(&mut v);
        assert!(v[0].is_nan(), "NaN input should leave the vector untouched");
        assert_eq!(v[1], 1.0);
    }

    // -----------------------------------------------------------------
    // Config
    // -----------------------------------------------------------------

    #[test]
    fn default_config_picks_mean_no_normalize() {
        let cfg = MlxEmbeddingConfig::default();
        assert_eq!(cfg.pooling, Pooling::Mean);
        assert!(!cfg.normalize);
        assert_eq!(cfg.max_seq_len, 512);
    }

    #[test]
    fn config_from_metadata_parses_pooling_and_normalize() {
        let mut md = HashMap::new();
        md.insert("pooling".into(), "cls".into());
        md.insert("normalize".into(), "true".into());
        md.insert("max_seq_len".into(), "256".into());
        let cfg = MlxEmbeddingConfig::from_metadata(&md).unwrap();
        assert_eq!(cfg.pooling, Pooling::Cls);
        assert!(cfg.normalize);
        assert_eq!(cfg.max_seq_len, 256);
    }

    #[test]
    fn config_from_metadata_normalize_one_means_true() {
        let mut md = HashMap::new();
        md.insert("normalize".into(), "1".into());
        let cfg = MlxEmbeddingConfig::from_metadata(&md).unwrap();
        assert!(cfg.normalize);
    }

    #[test]
    fn config_from_metadata_rejects_bad_pooling() {
        let mut md = HashMap::new();
        md.insert("pooling".into(), "median".into());
        let err = MlxEmbeddingConfig::from_metadata(&md).unwrap_err();
        match err {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("pooling")),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn config_from_metadata_rejects_zero_max_seq_len() {
        let mut md = HashMap::new();
        md.insert("max_seq_len".into(), "0".into());
        let err = MlxEmbeddingConfig::from_metadata(&md).unwrap_err();
        match err {
            MlxLlmError::ConfigInvalid(msg) => assert!(msg.contains("max_seq_len"), "got: {msg}"),
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn token_ids_to_mlx_i32_rejects_ids_over_i32_limit() {
        let too_large = u32::try_from(i32::MAX).unwrap() + 1;
        let err = token_ids_to_mlx_i32(&[too_large], 1, usize::MAX).unwrap_err();
        match err {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("token id"), "got: {msg}");
                assert!(msg.contains("i32 token-id limit"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn token_ids_to_mlx_i32_rejects_ids_outside_vocab() {
        let err = token_ids_to_mlx_i32(&[5], 1, 5).unwrap_err();
        match err {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("token id 5"), "got: {msg}");
                assert!(msg.contains("vocab_size=5"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn token_ids_to_mlx_i32_validates_only_truncated_prefix() {
        let too_large = u32::try_from(i32::MAX).unwrap() + 1;
        let ids = token_ids_to_mlx_i32(&[1, too_large], 1, 2).unwrap();

        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn truncation_metadata_records_original_and_clamped_counts() {
        let metadata = embedding_truncation_metadata(17, 5);

        assert_eq!(metadata.get("truncated").map(String::as_str), Some("true"));
        assert_eq!(
            metadata.get("original_token_count").map(String::as_str),
            Some("17")
        );
        assert_eq!(
            metadata.get("clamped_token_count").map(String::as_str),
            Some("5")
        );
    }

    #[test]
    fn truncation_metadata_is_empty_when_input_fits() {
        let metadata = embedding_truncation_metadata(5, 5);

        assert!(metadata.is_empty(), "unexpected metadata: {metadata:?}");
    }

    // -----------------------------------------------------------------
    // Adapter load
    // -----------------------------------------------------------------

    /// Build a minimal canonical-BERT bundle in `dir` with a full
    /// safetensors manifest (4-byte placeholder per tensor — the
    /// header parses but the weights are degenerate). The tokenizer is
    /// the qwen fixture from US-009 — any valid tokenizer.json works
    /// because the load path only verifies presence + parseability.
    fn write_dummy_bert_bundle(dir: &Path, model_type: &str) {
        let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        fs::copy(&tok_src, dir.join("tokenizer.json")).unwrap();
        let tokenizer = Tokenizer::from_file(&tok_src).unwrap();

        let cfg = serde_json::json!({
            "model_type": model_type,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "vocab_size": tokenizer.get_vocab_size(true),
            "max_position_embeddings": 128,
            "type_vocab_size": 2,
            "layer_norm_eps": 1.0e-12,
            "use_rotary_embeddings": model_type == "nomic_bert",
            "use_swiglu": model_type == "nomic_bert"
        });
        let mut f = fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(cfg.to_string().as_bytes()).unwrap();

        // Build a full safetensors manifest matching the BERT key
        // schedule for the chosen variant.
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let bert_cfg = bert::BertConfig::from_model_dir(dir).unwrap();
        let keys = bert::expected_weight_keys(&bert_cfg);
        let data = vec![0u8; 4];
        let tensors: Vec<(String, TensorView<'_>)> = keys
            .into_iter()
            .map(|k| (k, TensorView::new(Dtype::F32, vec![1], &data).unwrap()))
            .collect();
        safetensors::serialize_to_file(tensors, &None, &dir.join("model.safetensors")).unwrap();
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn write_runtime_canonical_bert_bundle(dir: &Path, prompt: &str) {
        let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        fs::copy(&tok_src, dir.join("tokenizer.json")).unwrap();
        let tokenizer = Tokenizer::from_file(&tok_src).unwrap();
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let token_ids = encoding.get_ids();
        let seq_len = token_ids.len().max(1);
        let vocab_size = token_ids
            .iter()
            .copied()
            .max()
            .map(|id| id as usize + 1)
            .unwrap_or(1);

        let cfg = serde_json::json!({
            "model_type": "bert",
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 4,
            "vocab_size": vocab_size,
            "max_position_embeddings": seq_len,
            "type_vocab_size": 2,
            "layer_norm_eps": 0.0,
            "use_rotary_embeddings": false,
            "use_swiglu": false
        });
        let mut f = fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(cfg.to_string().as_bytes()).unwrap();

        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_f32_tensor(
            &mut tensors,
            "embeddings.word_embeddings.weight",
            vec![vocab_size, 2],
            vec![0.0; vocab_size * 2],
        );
        let mut position_values = Vec::with_capacity(seq_len * 2);
        for pos in 0..seq_len {
            let base = pos as f32 + 1.0;
            position_values.push(base);
            position_values.push(base * 10.0);
        }
        push_f32_tensor(
            &mut tensors,
            "embeddings.position_embeddings.weight",
            vec![seq_len, 2],
            position_values,
        );
        push_f32_tensor(
            &mut tensors,
            "embeddings.token_type_embeddings.weight",
            vec![2, 2],
            vec![0.0; 4],
        );
        push_f32_tensor(
            &mut tensors,
            "embeddings.LayerNorm.weight",
            vec![2],
            vec![1.0; 2],
        );
        push_f32_tensor(
            &mut tensors,
            "embeddings.LayerNorm.bias",
            vec![2],
            vec![0.0; 2],
        );

        let base = "encoder.layer.0";
        for name in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ] {
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.weight"),
                vec![2, 2],
                vec![0.0; 4],
            );
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.bias"),
                vec![2],
                vec![0.0; 2],
            );
        }
        for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.weight"),
                vec![2],
                vec![1.0; 2],
            );
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.bias"),
                vec![2],
                vec![0.0; 2],
            );
        }
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.intermediate.dense.weight"),
            vec![4, 2],
            vec![0.0; 8],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.intermediate.dense.bias"),
            vec![4],
            vec![0.0; 4],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.output.dense.weight"),
            vec![2, 4],
            vec![0.0; 8],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.output.dense.bias"),
            vec![2],
            vec![0.0; 2],
        );

        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors")).unwrap();
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn write_runtime_nomic_bert_bundle(dir: &Path, prompt: &str) {
        let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        fs::copy(&tok_src, dir.join("tokenizer.json")).unwrap();
        let tokenizer = Tokenizer::from_file(&tok_src).unwrap();
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let token_ids = encoding.get_ids();
        let seq_len = token_ids.len().max(1);
        let vocab_size = token_ids
            .iter()
            .copied()
            .max()
            .map(|id| id as usize + 1)
            .unwrap_or(1);

        let cfg = serde_json::json!({
            "model_type": "nomic_bert",
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 4,
            "vocab_size": vocab_size,
            "max_position_embeddings": seq_len,
            "type_vocab_size": 2,
            "layer_norm_eps": 0.0,
            "hidden_act": "silu",
            "use_rotary_embeddings": true,
            "use_swiglu": true,
            "rope_theta": 1000.0
        });
        let mut f = fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(cfg.to_string().as_bytes()).unwrap();

        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut word_values = Vec::with_capacity(vocab_size * 2);
        for _ in 0..vocab_size {
            word_values.push(1.0);
            word_values.push(10.0);
        }
        push_f32_tensor(
            &mut tensors,
            "embeddings.word_embeddings.weight",
            vec![vocab_size, 2],
            word_values,
        );
        push_f32_tensor(
            &mut tensors,
            "embeddings.token_type_embeddings.weight",
            vec![2, 2],
            vec![0.0; 4],
        );
        push_f32_tensor(&mut tensors, "emb_ln.weight", vec![2], vec![1.0; 2]);
        push_f32_tensor(&mut tensors, "emb_ln.bias", vec![2], vec![0.0; 2]);

        let base = "encoder.layers.0";
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.attn.Wqkv.weight"),
            vec![6, 2],
            vec![0.0; 12],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.attn.out_proj.weight"),
            vec![2, 2],
            vec![0.0; 4],
        );
        for name in ["norm1", "norm2"] {
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.weight"),
                vec![2],
                vec![1.0; 2],
            );
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.bias"),
                vec![2],
                vec![0.0; 2],
            );
        }
        for name in ["fc11", "fc12"] {
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.mlp.{name}.weight"),
                vec![4, 2],
                vec![0.0; 8],
            );
        }
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.mlp.fc2.weight"),
            vec![2, 4],
            vec![0.0; 8],
        );

        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors")).unwrap();
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn push_f32_tensor(
        tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        name: &str,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        tensors.push((name.to_string(), shape, bytes));
    }

    #[test]
    fn unloaded_adapter_reports_no_state() {
        let adapter = MlxEmbeddingAdapter::new(MlxEmbeddingConfig::default());
        assert!(!adapter.is_loaded());
        assert!(adapter.bert_config().is_none());
        assert!(adapter.tokenizer().is_none());
        assert!(adapter.model_dir().is_none());
    }

    #[test]
    fn embed_on_unloaded_returns_not_loaded() {
        let adapter = MlxEmbeddingAdapter::new(MlxEmbeddingConfig::default());
        let err = adapter.embed("hello").unwrap_err();
        assert!(matches!(err, MlxLlmError::NotLoaded));
    }

    #[test]
    fn load_canonical_bert_bundle() {
        let tmp = TempDir::new().unwrap();
        write_dummy_bert_bundle(tmp.path(), "bert");
        let adapter = MlxEmbeddingAdapter::load(tmp.path(), &MlxEmbeddingConfig::default())
            .expect("canonical bert load");
        assert!(adapter.is_loaded());
        assert_eq!(adapter.bert_config().unwrap().model_type, "bert");
        assert_eq!(adapter.bert_config().unwrap().hidden_size, 16);
    }

    #[cfg(not(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    #[test]
    fn embed_on_non_runtime_build_returns_actionable_error() {
        let tmp = TempDir::new().unwrap();
        write_dummy_bert_bundle(tmp.path(), "bert");
        let adapter =
            MlxEmbeddingAdapter::load(tmp.path(), &MlxEmbeddingConfig::default()).unwrap();
        let err = adapter.embed("hello").unwrap_err();
        let display = err.to_string();
        match err {
            MlxLlmError::NotImplemented { feature, story } => {
                assert!(
                    feature.contains("llm-mlx-runtime"),
                    "feature hint: {feature}"
                );
                assert!(
                    feature.contains("fetch-mlx-xcframework"),
                    "fetch hint: {feature}"
                );
                assert!(
                    feature.contains("build-local-mlx-xcframework"),
                    "source-build hint: {feature}"
                );
                assert_eq!(story, "llm-mlx-runtime");
                assert!(
                    display.contains("unavailable in this build"),
                    "display: {display}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn embed_canonical_bert_runtime_bundle_returns_embedding() {
        let _guard = crate::runtime_adapter::mlx::MLX_RUNTIME_TEST_LOCK
            .lock()
            .unwrap();
        let tmp = TempDir::new().unwrap();
        let prompt = "Hello";
        write_runtime_canonical_bert_bundle(tmp.path(), prompt);
        let adapter = MlxEmbeddingAdapter::load(
            tmp.path(),
            &MlxEmbeddingConfig {
                pooling: Pooling::Mean,
                normalize: false,
                max_seq_len: 16,
            },
        )
        .expect("canonical runtime bert load");
        fs::remove_file(tmp.path().join("model.safetensors"))
            .expect("resident weights should not need the source file after load");

        let out = adapter.embed(prompt).expect("canonical runtime embed");
        match out.kind {
            EnvelopeKind::Embedding(values) => {
                assert_eq!(values.len(), 2);
                assert!((values[0] + 1.0).abs() < 1.0e-5, "got {values:?}");
                assert!((values[1] - 1.0).abs() < 1.0e-5, "got {values:?}");
            }
            other => panic!("expected embedding, got {other:?}"),
        }
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn template_executor_routes_canonical_bert_runtime_bundle_to_mlx_embedding() {
        let _guard = crate::runtime_adapter::mlx::MLX_RUNTIME_TEST_LOCK
            .lock()
            .unwrap();
        let tmp = TempDir::new().unwrap();
        let prompt = "Hello";
        write_runtime_canonical_bert_bundle(tmp.path(), prompt);

        let mut metadata = crate::execution::ModelMetadata::safetensors(
            "synthetic-bert-embed",
            "1.0",
            "model.safetensors",
            "bert",
        );
        metadata.backend = Some("auto".to_string());
        metadata
            .metadata
            .insert("task".to_string(), serde_json::json!("text-embedding"));
        metadata
            .metadata
            .insert("pooling".to_string(), serde_json::json!("mean"));
        metadata
            .metadata
            .insert("normalize".to_string(), serde_json::json!(false));
        metadata
            .metadata
            .insert("max_seq_len".to_string(), serde_json::json!(16));

        let mut executor =
            crate::execution::TemplateExecutor::with_base_path(&tmp.path().to_string_lossy());
        let out = executor
            .execute(
                &metadata,
                &crate::ir::Envelope::new(crate::ir::EnvelopeKind::Text(prompt.to_string())),
                None,
            )
            .expect("TemplateExecutor should route BERT metadata through MLX embedding");

        match out.kind {
            crate::ir::EnvelopeKind::Embedding(values) => {
                assert_eq!(values.len(), 2);
                assert!((values[0] + 1.0).abs() < 1.0e-5, "got {values:?}");
                assert!((values[1] - 1.0).abs() < 1.0e-5, "got {values:?}");
            }
            other => panic!("expected embedding, got {other:?}"),
        }
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn embed_nomic_bert_runtime_bundle_returns_embedding() {
        let _guard = crate::runtime_adapter::mlx::MLX_RUNTIME_TEST_LOCK
            .lock()
            .unwrap();
        let tmp = TempDir::new().unwrap();
        let prompt = "Hello";
        write_runtime_nomic_bert_bundle(tmp.path(), prompt);
        let adapter = MlxEmbeddingAdapter::load(
            tmp.path(),
            &MlxEmbeddingConfig {
                pooling: Pooling::Mean,
                normalize: false,
                max_seq_len: 16,
            },
        )
        .expect("nomic runtime bert load");
        fs::remove_file(tmp.path().join("model.safetensors"))
            .expect("resident weights should not need the source file after load");

        let out = adapter.embed(prompt).expect("nomic runtime embed");
        match out.kind {
            EnvelopeKind::Embedding(values) => {
                assert_eq!(values.len(), 2);
                assert!((values[0] + 1.0).abs() < 1.0e-5, "got {values:?}");
                assert!((values[1] - 1.0).abs() < 1.0e-5, "got {values:?}");
            }
            other => panic!("expected embedding, got {other:?}"),
        }
    }

    #[test]
    fn load_nomic_bert_bundle() {
        let tmp = TempDir::new().unwrap();
        write_dummy_bert_bundle(tmp.path(), "nomic_bert");
        let adapter = MlxEmbeddingAdapter::load(tmp.path(), &MlxEmbeddingConfig::default())
            .expect("nomic bert load");
        assert!(adapter.is_loaded());
        assert!(adapter.bert_config().unwrap().is_nomic());
    }

    #[test]
    fn load_missing_config_returns_missing_file() {
        let tmp = TempDir::new().unwrap();
        let err = MlxEmbeddingAdapter::load(tmp.path(), &MlxEmbeddingConfig::default())
            .expect_err("missing files should error");
        match err {
            MlxLlmError::MissingFile { file, .. } => assert_eq!(file, "config.json"),
            other => panic!("expected MissingFile, got {other:?}"),
        }
    }

    #[test]
    fn load_unsupported_model_type_rejects() {
        let tmp = TempDir::new().unwrap();
        let cfg = serde_json::json!({
            "model_type": "roberta",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "vocab_size": 100,
            "max_position_embeddings": 128
        });
        std::fs::File::create(tmp.path().join("config.json"))
            .unwrap()
            .write_all(cfg.to_string().as_bytes())
            .unwrap();
        let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        std::fs::copy(&tok_src, tmp.path().join("tokenizer.json")).unwrap();
        std::fs::File::create(tmp.path().join("model.safetensors"))
            .unwrap()
            .write_all(b"placeholder")
            .unwrap();

        let err =
            MlxEmbeddingAdapter::load(tmp.path(), &MlxEmbeddingConfig::default()).unwrap_err();
        assert!(matches!(err, MlxLlmError::UnsupportedArchitecture { .. }));
    }

    /// Adapter must be Send + Sync — pipelines hold it inside an
    /// `Arc<dyn ...>` and the runtime selector dispatches across
    /// threads.
    #[test]
    fn adapter_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MlxEmbeddingAdapter>();
    }
}

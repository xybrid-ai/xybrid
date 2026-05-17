//! HuggingFace tokenizer loader for MLX-community models.
//!
//! MLX-LM bundles ship a `tokenizer.json` file in the standard HuggingFace
//! `tokenizers` crate format (BPE, Unigram, or WordPiece configured inside
//! the file itself). This module provides a thin loader that reads that file
//! from a model directory and returns a ready-to-use [`Tokenizer`].
//!
//! See US-009 in `tasks/prd-mlx-backend.md`.

use std::path::Path;

use anyhow::{Context, Result};
pub use tokenizers::Tokenizer;

/// File name expected inside the model directory.
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Load a HuggingFace tokenizer from `{model_dir}/tokenizer.json`.
///
/// # Errors
///
/// Returns an error if the file does not exist or if the tokenizer cannot be
/// parsed (malformed JSON, unsupported tokenizer configuration, etc.).
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// # #[cfg(feature = "llm-mlx")]
/// # fn example() -> anyhow::Result<()> {
/// let tok = xybrid_core::runtime_adapter::mlx::tokenizer::load(
///     Path::new("/tmp/mlx-community-qwen")
/// )?;
/// let enc = tok.encode("Hello, world!", false).map_err(|e| anyhow::anyhow!(e.to_string()))?;
/// assert!(!enc.get_ids().is_empty());
/// # Ok(())
/// # }
/// ```
pub fn load(model_dir: &Path) -> Result<Tokenizer> {
    let path = model_dir.join(TOKENIZER_FILE);
    load_from_file(&path)
}

/// Load a tokenizer from an explicit file path.
///
/// Exists so tests can point at fixtures without a full model bundle.
pub fn load_from_file(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .with_context(|| format!("failed to load tokenizer from {}", path.display()))
}

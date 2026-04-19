//! Per-architecture transformer builders for the MLX LLM adapter.
//!
//! Each submodule ports one upstream MLX-LM architecture to Rust + xybrid-mlx
//! ops. The adapter in [`super::model`] dispatches on `config.json`'s
//! `model_type` field and hands off to the matching builder's `build` /
//! `forward` functions.
//!
//! Builders landing:
//! - [`qwen35`] — Qwen 3 / Qwen 3.5 family (US-011, `model_type = "qwen3"`).
//! - [`gemma4`] — Gemma 4 family (US-012, `model_type = "gemma4"`).
//! - [`lfm35`] — Liquid Foundation Model 3.5 family (US-013,
//!   `model_type = "lfm"` or `"lfm3"`).
//! - [`bert`] — BERT-family encoders for embeddings (US-015,
//!   `model_type = "bert"` or `"nomic_bert"`).

pub mod bert;
pub mod gemma4;
pub mod lfm35;
pub mod qwen35;

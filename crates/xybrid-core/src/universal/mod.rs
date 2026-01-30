//! Universal Architecture System
//!
//! This module provides configuration structures for universal ML architectures.
//! Models are defined declaratively in JSON and built dynamically at runtime.
//!
//! Currently supports configuration for:
//! - Transformers (BERT, GPT, encoder-decoder models)
//! - Audio encoders (for speech models)
//!
//! Note: Actual model execution is handled via ONNX Runtime or Candle (planned).

pub mod config;

pub use config::{
    AttentionType, AudioEncoderConfig, ConvConfig, EncoderDecoderConfig, PositionEncodingType,
    TransformerConfig, WhisperConfig,
};

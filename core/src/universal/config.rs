//! Configuration structures for universal architectures
//!
//! These configs are serializable to/from JSON, allowing models to be
//! defined declaratively.

use serde::{Deserialize, Serialize};

/// Configuration for a universal transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Number of transformer layers (can be 1, 6, 12, 96, anything!)
    pub num_layers: usize,

    /// Hidden dimension (model width)
    pub hidden_dim: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Feedforward intermediate dimension (usually 4x hidden_dim)
    pub feedforward_dim: usize,

    /// Dropout probability
    #[serde(default = "default_dropout")]
    pub dropout: f64,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Attention type
    #[serde(default)]
    pub attention_type: AttentionType,

    /// Position encoding type
    #[serde(default)]
    pub position_encoding: PositionEncodingType,

    /// Layer normalization epsilon
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AttentionType {
    /// Self-attention (BERT, GPT encoder)
    #[default]
    Self_,

    /// Cross-attention (for encoder-decoder models)
    Cross,

    /// Both self and cross (decoder blocks)
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEncodingType {
    /// Learned positional embeddings (BERT, GPT)
    #[default]
    Learned,

    /// Sinusoidal positional encoding (original Transformer)
    Sinusoidal,

    /// Rotary position encoding (RoPE - LLaMA, GPT-NeoX)
    Rotary,

    /// Relative position bias (T5)
    Relative,
}

/// Configuration for encoder-decoder transformer (like Whisper, T5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderDecoderConfig {
    pub encoder: TransformerConfig,
    pub decoder: TransformerConfig,
}

/// Configuration for Whisper model specifically
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Audio encoder configuration
    pub audio_encoder: AudioEncoderConfig,

    /// Text decoder configuration
    pub text_decoder: TransformerConfig,

    /// Vocabulary size
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEncoderConfig {
    /// Number of mel frequency bins
    pub n_mels: usize,

    /// Transformer config for audio processing
    pub transformer: TransformerConfig,

    /// Audio-specific preprocessing convolutions
    #[serde(default)]
    pub conv_layers: Vec<ConvConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    #[serde(default = "default_stride")]
    pub stride: usize,
    #[serde(default = "default_padding")]
    pub padding: usize,
}

// Default value functions
fn default_dropout() -> f64 {
    0.1
}

fn default_layer_norm_eps() -> f64 {
    1e-5
}

fn default_stride() -> usize {
    1
}

fn default_padding() -> usize {
    0
}

impl TransformerConfig {
    /// Create a BERT-base configuration
    pub fn bert_base() -> Self {
        Self {
            num_layers: 12,
            hidden_dim: 768,
            num_heads: 12,
            feedforward_dim: 3072,
            dropout: 0.1,
            max_seq_len: 512,
            attention_type: AttentionType::Self_,
            position_encoding: PositionEncodingType::Learned,
            layer_norm_eps: 1e-12,
        }
    }

    /// Create a GPT-2 configuration
    pub fn gpt2() -> Self {
        Self {
            num_layers: 12,
            hidden_dim: 768,
            num_heads: 12,
            feedforward_dim: 3072,
            dropout: 0.1,
            max_seq_len: 1024,
            attention_type: AttentionType::Self_,
            position_encoding: PositionEncodingType::Learned,
            layer_norm_eps: 1e-5,
        }
    }

    /// Create a Whisper tiny encoder configuration
    pub fn whisper_tiny_encoder() -> Self {
        Self {
            num_layers: 4,
            hidden_dim: 384,
            num_heads: 6,
            feedforward_dim: 1536,
            dropout: 0.0,
            max_seq_len: 1500,
            attention_type: AttentionType::Self_,
            position_encoding: PositionEncodingType::Learned,
            layer_norm_eps: 1e-5,
        }
    }

    /// Create a Whisper tiny decoder configuration
    pub fn whisper_tiny_decoder() -> Self {
        Self {
            num_layers: 4,
            hidden_dim: 384,
            num_heads: 6,
            feedforward_dim: 1536,
            dropout: 0.0,
            max_seq_len: 448,
            attention_type: AttentionType::Both,
            position_encoding: PositionEncodingType::Learned,
            layer_norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serialization() {
        let config = TransformerConfig::bert_base();
        let json = serde_json::to_string_pretty(&config).unwrap();
        println!("BERT config:\n{}", json);

        let deserialized: TransformerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.num_layers, deserialized.num_layers);
    }

    #[test]
    fn test_whisper_config() {
        let encoder = TransformerConfig::whisper_tiny_encoder();
        let decoder = TransformerConfig::whisper_tiny_decoder();

        assert_eq!(encoder.num_layers, 4);
        assert_eq!(encoder.hidden_dim, 384);
        assert_eq!(decoder.num_layers, 4);
        assert_eq!(decoder.hidden_dim, 384);
    }
}

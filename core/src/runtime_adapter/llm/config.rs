//! Configuration types for local LLM inference.

use serde::{Deserialize, Serialize};

/// Configuration for loading a local LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Path to the model file (GGUF format).
    pub model_path: String,

    /// Path to chat template file (optional).
    /// If not provided, uses model's built-in template or default.
    pub chat_template: Option<String>,

    /// Maximum context length (tokens).
    /// Default: 4096
    #[serde(default = "default_context_length")]
    pub context_length: usize,

    /// Number of GPU layers to offload.
    /// 0 = CPU only, -1 = all layers on GPU.
    /// Default: 0 (CPU only)
    #[serde(default)]
    pub gpu_layers: i32,

    /// Enable paged attention for memory efficiency.
    /// Default: true
    #[serde(default = "default_paged_attention")]
    pub paged_attention: bool,

    /// Enable logging during inference.
    /// Default: false
    #[serde(default)]
    pub logging: bool,
}

fn default_context_length() -> usize {
    4096
}

fn default_paged_attention() -> bool {
    true
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            chat_template: None,
            context_length: default_context_length(),
            gpu_layers: 0,
            paged_attention: default_paged_attention(),
            logging: false,
        }
    }
}

impl LlmConfig {
    /// Create a new config with the model path.
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the chat template path.
    pub fn with_chat_template(mut self, path: impl Into<String>) -> Self {
        self.chat_template = Some(path.into());
        self
    }

    /// Set the context length.
    pub fn with_context_length(mut self, length: usize) -> Self {
        self.context_length = length;
        self
    }

    /// Set GPU layers to offload.
    pub fn with_gpu_layers(mut self, layers: i32) -> Self {
        self.gpu_layers = layers;
        self
    }

    /// Enable or disable paged attention.
    pub fn with_paged_attention(mut self, enabled: bool) -> Self {
        self.paged_attention = enabled;
        self
    }

    /// Enable or disable logging.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.logging = enabled;
        self
    }
}

/// Generation parameters for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate.
    /// Default: 256
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy, higher = more random).
    /// Default: 0.7
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    /// Default: 0.9
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled).
    /// Default: 40
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Repetition penalty (1.0 = disabled).
    /// Default: 1.1
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

fn default_top_k() -> usize {
    40
}

fn default_repetition_penalty() -> f32 {
    1.1
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            repetition_penalty: default_repetition_penalty(),
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationConfig {
    /// Create config for greedy decoding (deterministic).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        }
    }

    /// Create config for creative generation.
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Add stop sequence.
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }
}

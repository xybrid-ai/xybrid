//! Model information and provider definitions.

use serde::{Deserialize, Serialize};

/// Supported model providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    /// OpenAI (GPT models)
    OpenAI,
    /// Anthropic (Claude models)
    Anthropic,
    /// Groq (fast inference)
    Groq,
    /// Together.ai
    Together,
    /// Fireworks.ai
    Fireworks,
    /// Local ONNX model
    Local,
}

impl std::fmt::Display for ModelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelProvider::OpenAI => write!(f, "openai"),
            ModelProvider::Anthropic => write!(f, "anthropic"),
            ModelProvider::Groq => write!(f, "groq"),
            ModelProvider::Together => write!(f, "together"),
            ModelProvider::Fireworks => write!(f, "fireworks"),
            ModelProvider::Local => write!(f, "local"),
        }
    }
}

impl std::str::FromStr for ModelProvider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(ModelProvider::OpenAI),
            "anthropic" => Ok(ModelProvider::Anthropic),
            "groq" => Ok(ModelProvider::Groq),
            "together" => Ok(ModelProvider::Together),
            "fireworks" => Ok(ModelProvider::Fireworks),
            "local" => Ok(ModelProvider::Local),
            _ => Err(format!("Unknown provider: {}", s)),
        }
    }
}

impl ModelProvider {
    /// Get the API base URL for this provider.
    pub fn base_url(&self) -> &'static str {
        match self {
            ModelProvider::OpenAI => "https://api.openai.com/v1",
            ModelProvider::Anthropic => "https://api.anthropic.com/v1",
            ModelProvider::Groq => "https://api.groq.com/openai/v1",
            ModelProvider::Together => "https://api.together.xyz/v1",
            ModelProvider::Fireworks => "https://api.fireworks.ai/inference/v1",
            ModelProvider::Local => "",
        }
    }

    /// Get the environment variable name for the API key.
    pub fn api_key_env_var(&self) -> &'static str {
        match self {
            ModelProvider::OpenAI => "OPENAI_API_KEY",
            ModelProvider::Anthropic => "ANTHROPIC_API_KEY",
            ModelProvider::Groq => "GROQ_API_KEY",
            ModelProvider::Together => "TOGETHER_API_KEY",
            ModelProvider::Fireworks => "FIREWORKS_API_KEY",
            ModelProvider::Local => "",
        }
    }
}

/// Information about a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID (e.g., "gpt-4o-mini", "claude-3-sonnet-20240229")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Provider
    pub provider: ModelProvider,

    /// Context window size (tokens)
    pub context_window: u32,

    /// Maximum output tokens
    pub max_output_tokens: u32,

    /// Input cost per 1M tokens (USD)
    pub input_cost_per_million: f64,

    /// Output cost per 1M tokens (USD)
    pub output_cost_per_million: f64,

    /// Whether the model supports function calling
    pub supports_tools: bool,

    /// Whether the model supports vision (image input)
    pub supports_vision: bool,

    /// Whether the model supports JSON mode
    pub supports_json_mode: bool,
}

impl ModelInfo {
    /// Get predefined model information for common models.
    pub fn for_model(model_id: &str) -> Option<Self> {
        match model_id {
            // OpenAI models
            "gpt-4o" => Some(Self {
                id: "gpt-4o".to_string(),
                name: "GPT-4o".to_string(),
                provider: ModelProvider::OpenAI,
                context_window: 128000,
                max_output_tokens: 4096,
                input_cost_per_million: 5.0,
                output_cost_per_million: 15.0,
                supports_tools: true,
                supports_vision: true,
                supports_json_mode: true,
            }),
            "gpt-4o-mini" => Some(Self {
                id: "gpt-4o-mini".to_string(),
                name: "GPT-4o Mini".to_string(),
                provider: ModelProvider::OpenAI,
                context_window: 128000,
                max_output_tokens: 16384,
                input_cost_per_million: 0.15,
                output_cost_per_million: 0.6,
                supports_tools: true,
                supports_vision: true,
                supports_json_mode: true,
            }),
            "gpt-4-turbo" => Some(Self {
                id: "gpt-4-turbo".to_string(),
                name: "GPT-4 Turbo".to_string(),
                provider: ModelProvider::OpenAI,
                context_window: 128000,
                max_output_tokens: 4096,
                input_cost_per_million: 10.0,
                output_cost_per_million: 30.0,
                supports_tools: true,
                supports_vision: true,
                supports_json_mode: true,
            }),

            // Anthropic models
            "claude-3-5-sonnet-20241022" | "claude-3-5-sonnet" => Some(Self {
                id: "claude-3-5-sonnet-20241022".to_string(),
                name: "Claude 3.5 Sonnet".to_string(),
                provider: ModelProvider::Anthropic,
                context_window: 200000,
                max_output_tokens: 8192,
                input_cost_per_million: 3.0,
                output_cost_per_million: 15.0,
                supports_tools: true,
                supports_vision: true,
                supports_json_mode: false,
            }),
            "claude-3-opus-20240229" | "claude-3-opus" => Some(Self {
                id: "claude-3-opus-20240229".to_string(),
                name: "Claude 3 Opus".to_string(),
                provider: ModelProvider::Anthropic,
                context_window: 200000,
                max_output_tokens: 4096,
                input_cost_per_million: 15.0,
                output_cost_per_million: 75.0,
                supports_tools: true,
                supports_vision: true,
                supports_json_mode: false,
            }),
            "claude-3-haiku-20240307" | "claude-3-haiku" => Some(Self {
                id: "claude-3-haiku-20240307".to_string(),
                name: "Claude 3 Haiku".to_string(),
                provider: ModelProvider::Anthropic,
                context_window: 200000,
                max_output_tokens: 4096,
                input_cost_per_million: 0.25,
                output_cost_per_million: 1.25,
                supports_tools: true,
                supports_vision: true,
                supports_json_mode: false,
            }),

            // Groq models
            "llama-3.1-70b-versatile" => Some(Self {
                id: "llama-3.1-70b-versatile".to_string(),
                name: "Llama 3.1 70B".to_string(),
                provider: ModelProvider::Groq,
                context_window: 131072,
                max_output_tokens: 8192,
                input_cost_per_million: 0.59,
                output_cost_per_million: 0.79,
                supports_tools: true,
                supports_vision: false,
                supports_json_mode: true,
            }),
            "llama-3.1-8b-instant" => Some(Self {
                id: "llama-3.1-8b-instant".to_string(),
                name: "Llama 3.1 8B".to_string(),
                provider: ModelProvider::Groq,
                context_window: 131072,
                max_output_tokens: 8192,
                input_cost_per_million: 0.05,
                output_cost_per_million: 0.08,
                supports_tools: true,
                supports_vision: false,
                supports_json_mode: true,
            }),
            "mixtral-8x7b-32768" => Some(Self {
                id: "mixtral-8x7b-32768".to_string(),
                name: "Mixtral 8x7B".to_string(),
                provider: ModelProvider::Groq,
                context_window: 32768,
                max_output_tokens: 8192,
                input_cost_per_million: 0.24,
                output_cost_per_million: 0.24,
                supports_tools: true,
                supports_vision: false,
                supports_json_mode: true,
            }),

            _ => None,
        }
    }

    /// Estimate cost for a request.
    pub fn estimate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        let input_cost = (prompt_tokens as f64 / 1_000_000.0) * self.input_cost_per_million;
        let output_cost = (completion_tokens as f64 / 1_000_000.0) * self.output_cost_per_million;
        input_cost + output_cost
    }
}

/// List of available models (OpenAI /v1/models format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelListEntry>,
}

/// Entry in the models list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListEntry {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_str() {
        assert_eq!("openai".parse::<ModelProvider>().unwrap(), ModelProvider::OpenAI);
        assert_eq!("anthropic".parse::<ModelProvider>().unwrap(), ModelProvider::Anthropic);
        assert_eq!("groq".parse::<ModelProvider>().unwrap(), ModelProvider::Groq);
    }

    #[test]
    fn test_model_info() {
        let info = ModelInfo::for_model("gpt-4o-mini").unwrap();
        assert_eq!(info.provider, ModelProvider::OpenAI);
        assert!(info.supports_tools);
        assert!(info.supports_vision);
    }

    #[test]
    fn test_cost_estimation() {
        let info = ModelInfo::for_model("gpt-4o-mini").unwrap();
        // 1000 input + 100 output tokens
        let cost = info.estimate_cost(1000, 100);
        // $0.15/M input + $0.6/M output
        // = 0.15 * 0.001 + 0.6 * 0.0001
        // = 0.00015 + 0.00006
        // = 0.00021
        assert!((cost - 0.00021).abs() < 0.00001);
    }
}

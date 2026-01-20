//! Pipeline DSL Configuration
//!
//! This module defines the unified pipeline configuration schema used by both
//! the CLI and SDK. The DSL is intentionally minimal - only `stages` is required.
//!
//! # Minimal Pipeline
//!
//! ```yaml
//! stages:
//!   - kokoro-82m
//! ```
//!
//! # Full Pipeline
//!
//! ```yaml
//! name: voice-assistant
//! registry: "https://api.xybrid.dev"
//!
//! stages:
//!   - whisper-tiny
//!
//!   - model: gpt-4o-mini
//!     target: cloud
//!     provider: openai
//!     system_prompt: "Be concise."
//!
//!   - kokoro-82m
//! ```
//!
//! # Design Decisions
//!
//! - **No `input`/`output` fields**: Inferred from model_metadata.json preprocessing/postprocessing
//! - **No `metrics` field**: Auto-detected at runtime via LocalDeviceAdapter
//! - **No `availability` field**: Auto-checked via cache status
//!
//! This keeps pipelines simple and portable - they don't need to know about
//! the device they'll run on.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Pipeline configuration loaded from YAML.
///
/// Only `stages` is required. Everything else is optional or inferred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Optional pipeline name/identifier for display and logging
    #[serde(default)]
    pub name: Option<String>,

    /// Registry URL for model resolution.
    /// Defaults to `https://api.xybrid.dev` if not specified.
    #[serde(default)]
    pub registry: Option<String>,

    /// Pipeline stages (REQUIRED).
    /// At least one stage must be defined.
    #[serde(deserialize_with = "deserialize_stages")]
    pub stages: Vec<StageConfig>,
}

impl PipelineConfig {
    /// Parse a pipeline configuration from YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// Get the effective registry URL (defaults to api.xybrid.dev)
    pub fn registry_url(&self) -> &str {
        self.registry.as_deref().unwrap_or("https://api.xybrid.dev")
    }

    /// Get stage configurations
    pub fn stages(&self) -> &[StageConfig] {
        &self.stages
    }

    /// Get number of stages
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get stage names for display
    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.stage_id()).collect()
    }
}

// ============================================================================
// Stage Configuration
// ============================================================================

/// Stage configuration supporting multiple formats.
///
/// Stages can be defined as:
/// - Simple string: `"model-id"` or `"model-id@version"`
/// - Full object: `{ model: "model-id", target: "device", ... }`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StageConfig {
    /// Simple format: just a model ID or model@version
    Simple(String),
    /// Full object format with all options
    Object(StageObjectConfig),
}

impl StageConfig {
    /// Get the model ID (without version) for this stage.
    pub fn model_id(&self) -> String {
        match self {
            StageConfig::Simple(s) => {
                // Parse "model@version" -> "model"
                s.split('@').next().unwrap_or(s).to_string()
            }
            StageConfig::Object(obj) => obj
                .model
                .clone()
                .unwrap_or_else(|| obj.id.clone().unwrap_or_else(|| "unknown".to_string())),
        }
    }

    /// Get the stage ID (for display and logging).
    /// Uses explicit `id` if set, otherwise falls back to model ID.
    pub fn stage_id(&self) -> String {
        match self {
            StageConfig::Simple(_) => self.model_id(),
            StageConfig::Object(obj) => obj.id.clone().unwrap_or_else(|| self.model_id()),
        }
    }

    /// Get the model version if specified.
    pub fn version(&self) -> Option<String> {
        match self {
            StageConfig::Simple(s) => {
                // Parse "model@version" -> Some("version")
                s.split('@').nth(1).map(|v| v.to_string())
            }
            StageConfig::Object(obj) => obj.version.clone(),
        }
    }

    /// Get the execution target (device, cloud, auto).
    pub fn target(&self) -> Option<&str> {
        match self {
            StageConfig::Simple(_) => None, // Default to auto
            StageConfig::Object(obj) => obj.target.as_deref(),
        }
    }

    /// Get the cloud provider (openai, anthropic, google, etc.)
    pub fn provider(&self) -> Option<&str> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object(obj) => obj.provider.as_deref(),
        }
    }

    /// Get stage options (system_prompt, max_tokens, temperature, etc.)
    pub fn options(&self) -> HashMap<String, serde_json::Value> {
        match self {
            StageConfig::Simple(_) => HashMap::new(),
            StageConfig::Object(obj) => obj.options.clone(),
        }
    }

    /// Check if this is a cloud/integration stage.
    pub fn is_cloud_stage(&self) -> bool {
        matches!(self.target(), Some("cloud") | Some("integration")) || self.provider().is_some()
    }

    /// Check if this is a device stage.
    pub fn is_device_stage(&self) -> bool {
        matches!(self.target(), Some("device"))
    }

    /// Get the execution provider override (cpu, coreml, coreml-ane, coreml-gpu).
    pub fn execution_provider(&self) -> Option<&str> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object(obj) => obj.execution_provider.as_deref(),
        }
    }

    /// Convert to StageObjectConfig for uniform handling.
    pub fn to_object(&self) -> StageObjectConfig {
        match self {
            StageConfig::Simple(s) => {
                let (model, version) = if s.contains('@') {
                    let parts: Vec<&str> = s.split('@').collect();
                    (
                        Some(parts[0].to_string()),
                        parts.get(1).map(|v| v.to_string()),
                    )
                } else {
                    (Some(s.clone()), None)
                };
                StageObjectConfig {
                    id: None,
                    model,
                    version,
                    target: None,
                    provider: None,
                    execution_provider: None,
                    options: HashMap::new(),
                }
            }
            StageConfig::Object(obj) => obj.clone(),
        }
    }
}

/// Full stage configuration object.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageObjectConfig {
    /// Stage identifier (optional, defaults to model name)
    #[serde(default)]
    pub id: Option<String>,

    /// Model ID from registry
    #[serde(default)]
    pub model: Option<String>,

    /// Model version (e.g., "1.0", "latest")
    #[serde(default)]
    pub version: Option<String>,

    /// Execution target: "device", "cloud", "auto"
    #[serde(default)]
    pub target: Option<String>,

    /// Cloud provider: "openai", "anthropic", "google", "elevenlabs"
    #[serde(default)]
    pub provider: Option<String>,

    /// ONNX Runtime execution provider override.
    /// If not set, auto-selection will be used based on model hints.
    /// Valid values: "cpu", "coreml", "coreml-ane", "coreml-gpu"
    #[serde(default)]
    pub execution_provider: Option<String>,

    /// Stage-specific options (flattened for convenience)
    /// Common options: system_prompt, max_tokens, temperature
    #[serde(default, flatten)]
    pub options: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Custom Deserializers
// ============================================================================

/// Custom deserializer to handle both simple strings and full objects.
fn deserialize_stages<'de, D>(deserializer: D) -> Result<Vec<StageConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    let raw: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;
    let mut stages = Vec::new();

    for (i, item) in raw.into_iter().enumerate() {
        match item {
            serde_json::Value::String(s) => {
                stages.push(StageConfig::Simple(s));
            }
            serde_json::Value::Object(_) => {
                let config: StageObjectConfig = serde_json::from_value(item)
                    .map_err(|e| D::Error::custom(format!("Invalid stage {}: {}", i, e)))?;
                stages.push(StageConfig::Object(config));
            }
            _ => {
                return Err(D::Error::custom(format!(
                    "Stage {} must be a string or object",
                    i
                )));
            }
        }
    }

    if stages.is_empty() {
        return Err(D::Error::custom("Pipeline must have at least one stage"));
    }

    Ok(stages)
}

// ============================================================================
// Execution Target Enum
// ============================================================================

/// Execution target for a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionTarget {
    /// Execute on-device (edge inference)
    Device,
    /// Execute in the cloud (Xybrid API or third-party)
    Cloud,
    /// Let the runtime decide based on device capabilities
    #[default]
    Auto,
}

impl ExecutionTarget {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "device" | "local" | "edge" => ExecutionTarget::Device,
            "cloud" | "server" | "integration" => ExecutionTarget::Cloud,
            _ => ExecutionTarget::Auto,
        }
    }
}

impl std::fmt::Display for ExecutionTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionTarget::Device => write!(f, "device"),
            ExecutionTarget::Cloud => write!(f, "cloud"),
            ExecutionTarget::Auto => write!(f, "auto"),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_pipeline() {
        let yaml = r#"
stages:
  - kokoro-82m
"#;
        let config = PipelineConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.stage_count(), 1);
        assert_eq!(config.stages[0].model_id(), "kokoro-82m");
        assert_eq!(config.registry_url(), "https://api.xybrid.dev");
    }

    #[test]
    fn test_pipeline_with_version() {
        let yaml = r#"
stages:
  - whisper-tiny@1.0
"#;
        let config = PipelineConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.stages[0].model_id(), "whisper-tiny");
        assert_eq!(config.stages[0].version(), Some("1.0".to_string()));
    }

    #[test]
    fn test_mixed_stages() {
        let yaml = r#"
name: voice-assistant
stages:
  - whisper-tiny
  - model: gpt-4o-mini
    target: cloud
    provider: openai
    system_prompt: "Be concise."
  - kokoro-82m
"#;
        let config = PipelineConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.name, Some("voice-assistant".to_string()));
        assert_eq!(config.stage_count(), 3);

        // First stage: simple
        assert_eq!(config.stages[0].model_id(), "whisper-tiny");
        assert!(!config.stages[0].is_cloud_stage());

        // Second stage: cloud
        assert_eq!(config.stages[1].model_id(), "gpt-4o-mini");
        assert!(config.stages[1].is_cloud_stage());
        assert_eq!(config.stages[1].provider(), Some("openai"));

        // Third stage: simple
        assert_eq!(config.stages[2].model_id(), "kokoro-82m");
    }

    #[test]
    fn test_custom_registry() {
        let yaml = r#"
registry: "http://localhost:8080"
stages:
  - test-model
"#;
        let config = PipelineConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.registry_url(), "http://localhost:8080");
    }

    #[test]
    fn test_stage_with_id() {
        let yaml = r#"
stages:
  - id: asr
    model: whisper-tiny
"#;
        let config = PipelineConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.stages[0].stage_id(), "asr");
        assert_eq!(config.stages[0].model_id(), "whisper-tiny");
    }

    #[test]
    fn test_empty_stages_fails() {
        let yaml = r#"
stages: []
"#;
        let result = PipelineConfig::from_yaml(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_object() {
        let simple = StageConfig::Simple("model@1.0".to_string());
        let obj = simple.to_object();
        assert_eq!(obj.model, Some("model".to_string()));
        assert_eq!(obj.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_execution_provider_override() {
        let yaml = r#"
stages:
  - model: mobilenet-v2
    execution_provider: coreml-ane
"#;
        let config = PipelineConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.stages[0].execution_provider(), Some("coreml-ane"));
    }

    #[test]
    fn test_execution_provider_default_none() {
        let yaml = r#"
stages:
  - mobilenet-v2
"#;
        let config = PipelineConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.stages[0].execution_provider(), None);
    }
}

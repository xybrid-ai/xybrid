//! Authority types - Request/Response structures for orchestration decisions.
//!
//! These types define the interface between the Orchestrator and the OrchestrationAuthority.
//! All decisions are wrapped in `AuthorityDecision<T>` to provide explainability.

use crate::context::DeviceMetrics;
use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::ExecutionTarget;
use std::time::{SystemTime, UNIX_EPOCH};

/// Context for target resolution decisions.
///
/// Contains all the information needed to decide WHERE a stage should execute.
#[derive(Debug, Clone)]
pub struct StageContext {
    /// Stage identifier.
    pub stage_id: String,
    /// Model identifier to execute.
    pub model_id: String,
    /// Input envelope kind (Audio, Text, Embedding).
    pub input_kind: EnvelopeKind,
    /// Current device metrics.
    pub metrics: DeviceMetrics,
    /// Explicit target from pipeline YAML (if specified).
    pub explicit_target: Option<ExecutionTarget>,
}

/// Request for model selection.
///
/// Used when the authority needs to select which model variant to use.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// Model identifier (e.g., "whisper-tiny", "kokoro-82m").
    pub model_id: String,
    /// Task type (e.g., "asr", "tts", "llm", "embedding").
    pub task: String,
    /// Constraints for model selection.
    pub constraints: ModelConstraints,
}

/// Constraints for model selection.
#[derive(Debug, Clone, Default)]
pub struct ModelConstraints {
    /// Maximum model size in MB.
    pub max_size_mb: Option<u64>,
    /// Required accuracy threshold (0.0-1.0).
    pub required_accuracy: Option<f32>,
    /// Prefer quantized models for smaller size/faster inference.
    pub prefer_quantized: bool,
}

/// Request for policy evaluation.
///
/// Used to determine if a request should proceed.
#[derive(Debug, Clone)]
pub struct PolicyRequest {
    /// Stage identifier.
    pub stage_id: String,
    /// Input envelope to evaluate.
    pub envelope: Envelope,
    /// Current device metrics.
    pub metrics: DeviceMetrics,
}

/// Every decision is explainable.
///
/// This wrapper ensures all authority decisions include:
/// - The actual result
/// - A human-readable reason
/// - The source of the decision
/// - A confidence score
/// - A timestamp
#[derive(Debug, Clone)]
pub struct AuthorityDecision<T> {
    /// The decision result.
    pub result: T,
    /// Human-readable explanation.
    pub reason: String,
    /// Where this decision came from.
    pub source: DecisionSource,
    /// Confidence in this decision (0.0-1.0).
    pub confidence: f32,
    /// When this decision was made (ms since UNIX epoch).
    pub timestamp_ms: u64,
}

impl<T> AuthorityDecision<T> {
    /// Create a new authority decision.
    pub fn new(
        result: T,
        reason: impl Into<String>,
        source: DecisionSource,
        confidence: f32,
    ) -> Self {
        Self {
            result,
            reason: reason.into(),
            source,
            confidence,
            timestamp_ms: now_ms(),
        }
    }

    /// Create a local decision with full confidence.
    pub fn local(result: T, reason: impl Into<String>) -> Self {
        Self::new(result, reason, DecisionSource::Local, 1.0)
    }

    /// Create a default/fallback decision.
    pub fn default_fallback(result: T, reason: impl Into<String>) -> Self {
        Self::new(result, reason, DecisionSource::Default, 0.5)
    }
}

/// Source of an authority decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionSource {
    /// Computed locally (deterministic, high confidence).
    Local,
    /// Received from backend API.
    Remote,
    /// Retrieved from cache.
    Cached,
    /// Fallback default (used when remote fails or is unavailable).
    Default,
}

impl std::fmt::Display for DecisionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionSource::Local => write!(f, "local"),
            DecisionSource::Remote => write!(f, "remote"),
            DecisionSource::Cached => write!(f, "cached"),
            DecisionSource::Default => write!(f, "default"),
        }
    }
}

/// Target resolution result.
///
/// Indicates where a stage should execute.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedTarget {
    /// Execute on-device (local inference).
    Device,
    /// Execute in the cloud via a specific provider.
    Cloud { provider: String },
    /// Execute on a custom server endpoint.
    Server { endpoint: String },
}

impl ResolvedTarget {
    /// Convert to a string for logging.
    pub fn as_str(&self) -> &str {
        match self {
            ResolvedTarget::Device => "device",
            ResolvedTarget::Cloud { .. } => "cloud",
            ResolvedTarget::Server { .. } => "server",
        }
    }
}

impl std::fmt::Display for ResolvedTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedTarget::Device => write!(f, "device"),
            ResolvedTarget::Cloud { provider } => write!(f, "cloud:{}", provider),
            ResolvedTarget::Server { endpoint } => write!(f, "server:{}", endpoint),
        }
    }
}

/// Model selection result.
#[derive(Debug, Clone)]
pub struct ModelSelection {
    /// Selected model identifier.
    pub model_id: String,
    /// Model variant (e.g., "q4_k_m" for quantized).
    pub variant: Option<String>,
    /// Source of the model.
    pub source: ModelSource,
}

/// Source of a model.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    /// Model is available locally at the given path.
    Local { path: String },
    /// Model should be fetched from the registry.
    Registry { url: String },
    /// Model is available via cloud inference.
    Cloud { provider: String },
}

impl ModelSource {
    /// Check if this is a local source.
    pub fn is_local(&self) -> bool {
        matches!(self, ModelSource::Local { .. })
    }
}

/// Policy evaluation outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyOutcome {
    /// Request is allowed to proceed.
    Allow,
    /// Request is denied with a reason.
    Deny { reason: String },
    /// Request is allowed but requires transforms.
    Transform { transforms: Vec<String> },
}

impl PolicyOutcome {
    /// Check if this outcome allows the request.
    pub fn is_allowed(&self) -> bool {
        !matches!(self, PolicyOutcome::Deny { .. })
    }
}

/// Execution outcome for feedback/learning.
#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    /// Stage that was executed.
    pub stage_id: String,
    /// Where it was executed.
    pub target: ResolvedTarget,
    /// How long it took (ms).
    pub latency_ms: u64,
    /// Whether execution succeeded.
    pub success: bool,
    /// Optional error message if failed.
    pub error: Option<String>,
}

/// Get current timestamp in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authority_decision_creation() {
        let decision: AuthorityDecision<PolicyOutcome> =
            AuthorityDecision::local(PolicyOutcome::Allow, "test reason");

        assert_eq!(decision.source, DecisionSource::Local);
        assert_eq!(decision.confidence, 1.0);
        assert!(decision.result.is_allowed());
    }

    #[test]
    fn test_resolved_target_display() {
        assert_eq!(ResolvedTarget::Device.to_string(), "device");
        assert_eq!(
            ResolvedTarget::Cloud {
                provider: "openai".to_string()
            }
            .to_string(),
            "cloud:openai"
        );
        assert_eq!(
            ResolvedTarget::Server {
                endpoint: "http://localhost:8000".to_string()
            }
            .to_string(),
            "server:http://localhost:8000"
        );
    }

    #[test]
    fn test_policy_outcome_is_allowed() {
        assert!(PolicyOutcome::Allow.is_allowed());
        assert!(PolicyOutcome::Transform { transforms: vec![] }.is_allowed());
        assert!(!PolicyOutcome::Deny {
            reason: "test".to_string()
        }
        .is_allowed());
    }
}

//! Orchestration Authority - The decision-making interface for hybrid orchestration.
//!
//! This module defines the `OrchestrationAuthority` trait, the critical interface between
//! the open-source execution plane and the (future) closed control plane.
//!
//! ## Design Philosophy
//!
//! - **Clean open-core boundary**: The trait is open, the intelligence is protected
//! - **No phone-home required**: `LocalAuthority` works completely offline
//! - **Transparent decisions**: Every decision has a reason
//! - **Zero breaking changes**: Wrap existing engines, don't replace
//!
//! ## Implementations
//!
//! - [`LocalAuthority`]: Default offline implementation using device metrics and heuristics.
//!   Wraps the existing `PolicyEngine` and `RoutingEngine`.
//!
//! - [`RemoteAuthority`]: (Future) Delegates to xybrid backend for smarter decisions
//!   based on fleet-wide data and learned patterns. Falls back to `LocalAuthority`.
//!
//! ## Decision Timing
//!
//! | Method | When Evaluated | Notes |
//! |--------|----------------|-------|
//! | `apply_policy()` | Per-request | Security critical, always runs |
//! | `resolve_target()` | Per-stage | Can react to changing conditions |
//! | `select_model()` | Per-pipeline-load | Stable for session |

mod local;
mod remote;
pub mod types;

pub use local::LocalAuthority;
pub use remote::RemoteAuthority;
pub use types::*;

/// The orchestration authority decides WHERE and HOW to execute.
///
/// This trait defines the boundary between execution (open) and control (your choice).
///
/// ## Default: LocalAuthority
///
/// Works completely offline. Uses device metrics and simple heuristics.
/// You can inspect the source - no magic, no phone-home.
///
/// ```rust,ignore
/// use xybrid_core::orchestrator::authority::{LocalAuthority, OrchestrationAuthority};
///
/// let authority = LocalAuthority::new();
/// println!("Using authority: {}", authority.name());
/// ```
///
/// ## Optional: RemoteAuthority
///
/// Delegates to xybrid backend for smarter decisions based on fleet data.
/// Returns the same explainable decisions, just with more intelligence.
///
/// ```rust,ignore
/// use xybrid_core::orchestrator::authority::{RemoteAuthority, OrchestrationAuthority};
///
/// let authority = RemoteAuthority::new("https://api.xybrid.dev");
/// // Falls back to local if network unavailable
/// ```
pub trait OrchestrationAuthority: Send + Sync {
    /// Apply policy to determine if a request should proceed.
    ///
    /// Called: **Per-request** (security critical, always runs).
    ///
    /// # Arguments
    ///
    /// * `request` - The policy request containing stage info, envelope, and metrics.
    ///
    /// # Returns
    ///
    /// An `AuthorityDecision` containing the policy outcome with explanation.
    fn apply_policy(&self, request: &PolicyRequest) -> AuthorityDecision<PolicyOutcome>;

    /// Resolve where a stage should execute.
    ///
    /// Called: **Per-stage** (can react to changing conditions).
    ///
    /// # Arguments
    ///
    /// * `context` - The stage context including model info, metrics, and explicit target.
    ///
    /// # Returns
    ///
    /// An `AuthorityDecision` containing the resolved target with explanation.
    fn resolve_target(&self, context: &StageContext) -> AuthorityDecision<ResolvedTarget>;

    /// Select which model variant to use.
    ///
    /// Called: **Per-pipeline-load** (stable for session).
    ///
    /// # Arguments
    ///
    /// * `request` - The model request with constraints.
    ///
    /// # Returns
    ///
    /// An `AuthorityDecision` containing the selected model with explanation.
    fn select_model(&self, request: &ModelRequest) -> AuthorityDecision<ModelSelection>;

    /// Record execution outcome for learning (optional).
    ///
    /// `LocalAuthority`: no-op.
    /// `RemoteAuthority`: sends to backend for fleet-wide learning.
    fn record_outcome(&self, _outcome: &ExecutionOutcome) {
        // Default: no-op
    }

    /// Invalidate any cached decisions (optional).
    ///
    /// Call this when conditions change significantly (e.g., network status change).
    fn invalidate_cache(&self) {
        // Default: no-op
    }

    /// Get the authority name for logging.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DeviceMetrics;
    use crate::ir::{Envelope, EnvelopeKind};

    fn default_metrics() -> DeviceMetrics {
        DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        }
    }

    fn text_envelope(text: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    }

    #[test]
    fn test_local_authority_name() {
        let authority = LocalAuthority::new();
        assert_eq!(authority.name(), "local");
    }

    #[test]
    fn test_remote_authority_name() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        assert_eq!(authority.name(), "remote");
    }

    #[test]
    fn test_local_authority_allows_by_default() {
        let authority = LocalAuthority::new();
        let request = PolicyRequest {
            stage_id: "test".to_string(),
            envelope: text_envelope("hello"),
            metrics: default_metrics(),
        };

        let decision = authority.apply_policy(&request);
        assert!(decision.result.is_allowed());
        assert_eq!(decision.source, DecisionSource::Local);
    }

    #[test]
    fn test_local_authority_respects_explicit_target() {
        let authority = LocalAuthority::new();
        let context = StageContext {
            stage_id: "test".to_string(),
            model_id: "whisper-tiny".to_string(),
            input_kind: EnvelopeKind::Audio(vec![]),
            metrics: default_metrics(),
            explicit_target: Some(crate::pipeline::ExecutionTarget::Device),
        };

        let decision = authority.resolve_target(&context);
        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(decision.reason.to_lowercase().contains("explicit"));
    }

    #[test]
    fn test_remote_authority_falls_back_to_local() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let request = PolicyRequest {
            stage_id: "test".to_string(),
            envelope: text_envelope("hello"),
            metrics: default_metrics(),
        };

        let decision = authority.apply_policy(&request);
        assert!(decision.result.is_allowed());
        assert_eq!(decision.source, DecisionSource::Default); // Fallback
    }
}

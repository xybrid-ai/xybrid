//! Remote Orchestration Authority - Delegates to xybrid backend.
//!
//! This authority calls the xybrid backend for smarter decisions based on
//! fleet-wide data and learned patterns.
//!
//! ## For v0.1.0
//!
//! This is a **stub implementation** that falls back to `LocalAuthority`.
//! The actual backend endpoints will be implemented in a future version.
//!
//! ## Future Capabilities
//!
//! - **Fleet-wide learning**: Decisions informed by similar devices' experiences
//! - **A/B testing**: Experiment with routing strategies
//! - **Cost optimization**: Balance cost vs latency across cloud providers
//! - **Anomaly detection**: Identify and avoid failing execution targets
//!
//! ## Fallback Behavior
//!
//! When the backend is unavailable (offline, network error, timeout), decisions
//! fall back to `LocalAuthority` with `DecisionSource::Default`. This ensures
//! xybrid always works, even without connectivity.

use super::local::LocalAuthority;
use super::types::*;
use super::OrchestrationAuthority;

/// Remote orchestration authority - delegates to xybrid backend.
///
/// This authority calls the xybrid backend for smarter decisions
/// based on fleet-wide data and learned patterns.
///
/// ## Stub Implementation
///
/// For v0.1.0, this falls back to `LocalAuthority` for all decisions.
/// The backend integration will be implemented in a future version.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::orchestrator::authority::{RemoteAuthority, OrchestrationAuthority};
///
/// let authority = RemoteAuthority::new("https://api.xybrid.dev");
/// // Falls back to local if network unavailable
/// let decision = authority.apply_policy(&request);
/// ```
pub struct RemoteAuthority {
    /// Backend endpoint URL.
    #[allow(dead_code)]
    endpoint: String,
    /// Fallback to local authority when remote is unavailable.
    fallback: LocalAuthority,
    // Future: HTTP client, caches, circuit breaker, etc.
}

impl RemoteAuthority {
    /// Create a new RemoteAuthority with the given backend endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The xybrid backend URL (e.g., "https://api.xybrid.dev")
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            fallback: LocalAuthority::new(),
        }
    }

    /// Create a RemoteAuthority with a custom fallback authority.
    pub fn with_fallback(endpoint: &str, fallback: LocalAuthority) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            fallback,
        }
    }

    /// Get the backend endpoint URL.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

impl OrchestrationAuthority for RemoteAuthority {
    fn apply_policy(&self, request: &PolicyRequest) -> AuthorityDecision<PolicyOutcome> {
        // TODO: Call backend endpoint
        // POST /v1/authority/policy
        // Body: { stage_id, envelope_kind, metrics }
        // Response: { outcome, reason, confidence }

        // For now, fall back to local
        let mut decision = self.fallback.apply_policy(request);
        decision.source = DecisionSource::Default;
        decision.reason = format!(
            "Fallback to local (remote not implemented): {}",
            decision.reason
        );
        decision
    }

    fn resolve_target(&self, context: &StageContext) -> AuthorityDecision<ResolvedTarget> {
        // TODO: Call backend endpoint with caching (TTL 30s)
        // POST /v1/authority/target
        // Body: { stage_id, model_id, input_kind, metrics, explicit_target }
        // Response: { target, reason, confidence, cache_ttl_s }

        // For now, fall back to local
        let mut decision = self.fallback.resolve_target(context);
        decision.source = DecisionSource::Default;
        decision.reason = format!(
            "Fallback to local (remote not implemented): {}",
            decision.reason
        );
        decision
    }

    fn select_model(&self, request: &ModelRequest) -> AuthorityDecision<ModelSelection> {
        // TODO: Call backend endpoint
        // POST /v1/authority/model
        // Body: { model_id, task, constraints }
        // Response: { model_id, variant, source, reason, confidence }

        // For now, fall back to local
        let mut decision = self.fallback.select_model(request);
        decision.source = DecisionSource::Default;
        decision.reason = format!(
            "Fallback to local (remote not implemented): {}",
            decision.reason
        );
        decision
    }

    fn record_outcome(&self, outcome: &ExecutionOutcome) {
        // TODO: Send to backend for fleet-wide learning
        // POST /v1/authority/outcome
        // Body: { stage_id, target, latency_ms, success, error }

        // For now, no-op (local authority doesn't learn)
        let _ = outcome;
    }

    fn invalidate_cache(&self) {
        // TODO: Clear local cache of remote decisions
        // For now, no-op
    }

    fn name(&self) -> &str {
        "remote"
    }
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
    fn test_remote_authority_name() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        assert_eq!(authority.name(), "remote");
    }

    #[test]
    fn test_remote_authority_endpoint() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        assert_eq!(authority.endpoint(), "https://api.xybrid.dev");
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
        // Should allow (same as local)
        assert!(decision.result.is_allowed());
        // But source should indicate fallback
        assert_eq!(decision.source, DecisionSource::Default);
        // And reason should indicate fallback
        assert!(decision.reason.contains("Fallback to local"));
    }

    #[test]
    fn test_remote_authority_target_resolution_fallback() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let context = StageContext {
            stage_id: "test".to_string(),
            model_id: "test-model".to_string(),
            input_kind: EnvelopeKind::Text("test".to_string()),
            metrics: default_metrics(),
            explicit_target: None,
        };

        let decision = authority.resolve_target(&context);
        assert_eq!(decision.source, DecisionSource::Default);
        assert!(decision.reason.contains("Fallback"));
    }

    #[test]
    fn test_remote_authority_model_selection_fallback() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let request = ModelRequest {
            model_id: "test-model".to_string(),
            task: "test".to_string(),
            constraints: ModelConstraints::default(),
        };

        let decision = authority.select_model(&request);
        assert_eq!(decision.source, DecisionSource::Default);
        assert!(decision.reason.contains("Fallback"));
    }

    #[test]
    fn test_remote_authority_record_outcome_noop() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let outcome = ExecutionOutcome {
            stage_id: "test".to_string(),
            target: ResolvedTarget::Device,
            latency_ms: 100,
            success: true,
            error: None,
        };

        // Should not panic
        authority.record_outcome(&outcome);
    }
}

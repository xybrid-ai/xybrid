//! Local Orchestration Authority - Fully functional offline implementation.
//!
//! This is the default authority that ships with xybrid. It uses device metrics
//! and heuristics to make decisions locally. No network calls, no phone-home,
//! completely transparent.
//!
//! ## How It Works
//!
//! `LocalAuthority` wraps the existing `PolicyEngine` and `RoutingEngine`:
//!
//! - **Policy evaluation**: Delegates to `DefaultPolicyEngine`
//! - **Target resolution**: Delegates to `DefaultRoutingEngine`, respects explicit targets
//! - **Model selection**: Checks local availability, falls back to registry
//!
//! ## Decision Quality
//!
//! Local decisions are deterministic and have high confidence (1.0) because they
//! use only local information. For smarter decisions based on fleet data, use
//! `RemoteAuthority`.

use super::types::*;
use super::OrchestrationAuthority;
use crate::ir::Envelope;
use crate::orchestrator::policy_engine::{DefaultPolicyEngine, PolicyEngine};
use crate::orchestrator::routing_engine::{
    DefaultRoutingEngine, LocalAvailability, RouteTarget, RoutingEngine,
};
use crate::pipeline::ExecutionTarget;
use std::path::Path;
use std::sync::Mutex;

/// Local orchestration authority - fully functional offline.
///
/// This is the default authority that ships with xybrid.
/// It uses device metrics and heuristics to make decisions locally.
/// No network calls, no phone-home, completely transparent.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::orchestrator::authority::{LocalAuthority, OrchestrationAuthority, PolicyRequest};
///
/// let authority = LocalAuthority::new();
/// let decision = authority.apply_policy(&request);
/// println!("Decision: {:?} ({})", decision.result, decision.reason);
/// ```
pub struct LocalAuthority {
    policy_engine: DefaultPolicyEngine,
    /// Wrapped in Mutex for interior mutability (RoutingEngine::decide requires &mut self).
    routing_engine: Mutex<DefaultRoutingEngine>,
}

impl LocalAuthority {
    /// Create a new LocalAuthority with default policy and routing engines.
    pub fn new() -> Self {
        Self {
            policy_engine: DefaultPolicyEngine::with_default_policy(),
            routing_engine: Mutex::new(DefaultRoutingEngine::new()),
        }
    }

    /// Create a LocalAuthority with a custom policy engine.
    pub fn with_policy_engine(policy_engine: DefaultPolicyEngine) -> Self {
        Self {
            policy_engine,
            routing_engine: Mutex::new(DefaultRoutingEngine::new()),
        }
    }

    /// Check if a model exists locally in the cache.
    fn check_model_exists(model_id: &str) -> bool {
        // Check common locations for local models
        let cache_dir = dirs::home_dir()
            .map(|h| h.join(".xybrid").join("cache").join(model_id))
            .unwrap_or_default();

        if cache_dir.exists() {
            return true;
        }

        // Also check for test_models directory (for development)
        let test_models = Path::new("test_models").join(model_id);
        test_models.exists()
    }

    /// Find the local path for a model if available.
    fn find_local_model(model_id: &str) -> Option<String> {
        // Check cache directory first
        let cache_dir = dirs::home_dir()
            .map(|h| h.join(".xybrid").join("cache").join(model_id));

        if let Some(ref path) = cache_dir {
            if path.exists() {
                return path.to_str().map(|s| s.to_string());
            }
        }

        // Check test_models directory
        let test_models = Path::new("test_models").join(model_id);
        if test_models.exists() {
            return test_models.to_str().map(|s| s.to_string());
        }

        None
    }
}

impl Default for LocalAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl OrchestrationAuthority for LocalAuthority {
    fn apply_policy(&self, request: &PolicyRequest) -> AuthorityDecision<PolicyOutcome> {
        let result = self.policy_engine.evaluate(
            &request.stage_id,
            &request.envelope,
            &request.metrics,
        );

        let outcome = if result.allowed {
            if result.transforms_applied.is_empty() {
                PolicyOutcome::Allow
            } else {
                PolicyOutcome::Transform {
                    transforms: result.transforms_applied.clone(),
                }
            }
        } else {
            PolicyOutcome::Deny {
                reason: result.reason.clone().unwrap_or_else(|| "Policy denied".to_string()),
            }
        };

        let reason = result
            .reason
            .unwrap_or_else(|| "Local policy evaluation".to_string());

        AuthorityDecision {
            result: outcome,
            reason,
            source: DecisionSource::Local,
            confidence: 1.0, // Local decisions are deterministic
            timestamp_ms: now_ms(),
        }
    }

    fn resolve_target(&self, context: &StageContext) -> AuthorityDecision<ResolvedTarget> {
        // If explicit target specified in pipeline, use it
        if let Some(explicit) = &context.explicit_target {
            let target = match explicit {
                ExecutionTarget::Device => ResolvedTarget::Device,
                ExecutionTarget::Server => ResolvedTarget::Server {
                    endpoint: "https://api.xybrid.dev".to_string(),
                },
                ExecutionTarget::Cloud => ResolvedTarget::Cloud {
                    provider: "xybrid".to_string(),
                },
                ExecutionTarget::Auto => {
                    // Fall through to routing engine
                    return self.resolve_with_routing_engine(context);
                }
            };

            return AuthorityDecision {
                result: target,
                reason: format!("Explicit target from pipeline YAML: {:?}", explicit),
                source: DecisionSource::Local,
                confidence: 1.0,
                timestamp_ms: now_ms(),
            };
        }

        // Otherwise, use routing engine heuristics
        self.resolve_with_routing_engine(context)
    }

    fn select_model(&self, request: &ModelRequest) -> AuthorityDecision<ModelSelection> {
        // Check if model is available locally
        let local_path = Self::find_local_model(&request.model_id);

        let source = if let Some(path) = local_path {
            ModelSource::Local { path }
        } else {
            ModelSource::Registry {
                url: format!("https://api.xybrid.dev/v1/models/{}", request.model_id),
            }
        };

        let is_local = source.is_local();

        AuthorityDecision {
            result: ModelSelection {
                model_id: request.model_id.clone(),
                variant: None,
                source,
            },
            reason: if is_local {
                format!("Model '{}' found locally", request.model_id)
            } else {
                format!("Model '{}' not found locally, will fetch from registry", request.model_id)
            },
            source: DecisionSource::Local,
            confidence: 1.0,
            timestamp_ms: now_ms(),
        }
    }

    fn name(&self) -> &str {
        "local"
    }
}

impl LocalAuthority {
    /// Internal: resolve target using the routing engine.
    fn resolve_with_routing_engine(&self, context: &StageContext) -> AuthorityDecision<ResolvedTarget> {
        let availability = LocalAvailability {
            local_model_exists: Self::check_model_exists(&context.model_id),
        };

        // Create a minimal envelope for policy check
        let envelope = Envelope::new(context.input_kind.clone());

        let policy_result = self.policy_engine.evaluate(
            &context.stage_id,
            &envelope,
            &context.metrics,
        );

        // Use the stored routing engine (locked for interior mutability)
        let decision = {
            let mut routing_engine = self.routing_engine.lock().unwrap();
            routing_engine.decide(
                &context.stage_id,
                &context.metrics,
                &policy_result,
                &availability,
            )
        };

        let target = match decision.target {
            RouteTarget::Local => ResolvedTarget::Device,
            RouteTarget::Cloud => ResolvedTarget::Cloud {
                provider: "xybrid".to_string(),
            },
            RouteTarget::Fallback(id) => ResolvedTarget::Server {
                endpoint: format!("fallback:{}", id),
            },
        };

        AuthorityDecision {
            result: target,
            reason: decision.reason,
            source: DecisionSource::Local,
            confidence: 0.8, // Heuristic-based, slightly lower confidence
            timestamp_ms: decision.timestamp_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DeviceMetrics;
    use crate::ir::EnvelopeKind;

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
    fn test_local_authority_default_allows() {
        let authority = LocalAuthority::new();
        let request = PolicyRequest {
            stage_id: "test".to_string(),
            envelope: text_envelope("hello"),
            metrics: default_metrics(),
        };

        let decision = authority.apply_policy(&request);
        assert!(decision.result.is_allowed());
        assert_eq!(decision.source, DecisionSource::Local);
        assert_eq!(decision.confidence, 1.0);
    }

    #[test]
    fn test_local_authority_high_rtt_denies() {
        let authority = LocalAuthority::new();
        let request = PolicyRequest {
            stage_id: "test".to_string(),
            envelope: text_envelope("hello"),
            metrics: DeviceMetrics {
                network_rtt: 350, // Above 300ms threshold
                battery: 50,
                temperature: 25.0,
            },
        };

        let decision = authority.apply_policy(&request);
        // Policy should deny due to high RTT
        assert!(!decision.result.is_allowed());
    }

    #[test]
    fn test_local_authority_explicit_device_target() {
        let authority = LocalAuthority::new();
        let context = StageContext {
            stage_id: "test".to_string(),
            model_id: "test-model".to_string(),
            input_kind: EnvelopeKind::Text("test".to_string()),
            metrics: default_metrics(),
            explicit_target: Some(ExecutionTarget::Device),
        };

        let decision = authority.resolve_target(&context);
        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(decision.reason.contains("Explicit"));
    }

    #[test]
    fn test_local_authority_explicit_cloud_target() {
        let authority = LocalAuthority::new();
        let context = StageContext {
            stage_id: "test".to_string(),
            model_id: "test-model".to_string(),
            input_kind: EnvelopeKind::Text("test".to_string()),
            metrics: default_metrics(),
            explicit_target: Some(ExecutionTarget::Cloud),
        };

        let decision = authority.resolve_target(&context);
        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
    }

    #[test]
    fn test_local_authority_model_selection_not_found() {
        let authority = LocalAuthority::new();
        let request = ModelRequest {
            model_id: "nonexistent-model-xyz".to_string(),
            task: "test".to_string(),
            constraints: ModelConstraints::default(),
        };

        let decision = authority.select_model(&request);
        assert!(matches!(decision.result.source, ModelSource::Registry { .. }));
        assert!(decision.reason.contains("not found locally"));
    }

    #[test]
    fn test_local_authority_name() {
        let authority = LocalAuthority::new();
        assert_eq!(authority.name(), "local");
    }
}

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
//! | `resolve_stage()` | Per-stage | One policy evaluation + one snapshot; policy restricts the target |
//! | `resolve_target()` | Per-stage | Target only; can react to changing conditions |
//! | `load_policies()` | On configuration | Atomic swap of the active bundle |
//! | `select_model()` | Per-pipeline-load | Stable for session |

mod local;
mod remote;
pub mod types;

#[cfg(any(test, feature = "dev-tools"))]
pub mod test_seams;

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
/// ```no_run
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
/// ```no_run
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

    /// Resolve target and return feedback context for outcome learning.
    ///
    /// Existing authorities can keep implementing only `resolve_target`; richer
    /// authorities override this to attach signal buckets or effective model ids.
    fn resolve_target_with_feedback(&self, context: &StageContext) -> TargetResolution {
        TargetResolution::new(self.resolve_target(context), context.model_id.clone(), None)
    }

    /// Load (replace) the active policy bundle.
    ///
    /// Authorities that do not own a policy engine keep this default, which
    /// returns an error so a caller never believes a bundle took effect when
    /// it did not. Implementations validate the whole bundle before swapping
    /// it in; a failed load leaves the previous policy active.
    fn load_policies(&self, _bundle: &[u8]) -> Result<(), String> {
        Err("policy loading is not supported by this authority".to_string())
    }

    /// Decide policy and target for one stage in a single call.
    ///
    /// Called: **Per-stage**, with the actual input envelope.
    ///
    /// The default evaluates `apply_policy` once and, when the outcome
    /// forbids leaving the device, returns a device target without consulting
    /// target resolution at all — so no remote advice is requested for a
    /// denied input. Otherwise it delegates to `resolve_target_with_feedback`.
    /// The result is always restricted by the policy outcome; orchestrators
    /// still call [`StageResolution::enforce`] immediately before dispatch as
    /// defense in depth against custom implementations.
    fn resolve_stage(&self, request: &PolicyRequest, context: &StageContext) -> StageResolution {
        let policy = self.apply_policy(request);
        let target = if let Some(reason) = policy_local_constraint_reason(&policy.result) {
            TargetResolution::new(
                AuthorityDecision::new(
                    ResolvedTarget::Device,
                    reason,
                    policy.source.clone(),
                    policy.confidence,
                ),
                context.model_id.clone(),
                None,
            )
        } else {
            self.resolve_target_with_feedback(context)
        };
        StageResolution::new(policy, target)
    }

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
    use crate::device::ResourceMonitor;
    use crate::ir::{Envelope, EnvelopeKind};

    fn default_metrics() -> DeviceMetrics {
        DeviceMetrics::default()
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
            resource_monitor: ResourceMonitor::global(),
            explicit_target: Some(crate::pipeline::ExecutionTarget::Device),
            local_availability: None,
            device_class: None,
            device_class_schema_version: None,
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

    /// Minimal custom authority: implements only the required methods so the
    /// trait defaults are what gets exercised.
    struct StaticAuthority {
        policy: PolicyOutcome,
        target_calls: std::sync::atomic::AtomicUsize,
    }

    impl OrchestrationAuthority for StaticAuthority {
        fn apply_policy(&self, _request: &PolicyRequest) -> AuthorityDecision<PolicyOutcome> {
            AuthorityDecision::local(self.policy.clone(), "static policy")
        }

        fn resolve_target(&self, _context: &StageContext) -> AuthorityDecision<ResolvedTarget> {
            self.target_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            AuthorityDecision::local(
                ResolvedTarget::Cloud {
                    provider: "xybrid".to_string(),
                },
                "static cloud",
            )
        }

        fn select_model(&self, request: &ModelRequest) -> AuthorityDecision<ModelSelection> {
            AuthorityDecision::local(
                ModelSelection {
                    model_id: request.model_id.clone(),
                    variant: None,
                    source: ModelSource::Cloud {
                        provider: "xybrid".to_string(),
                    },
                },
                "static model",
            )
        }

        fn name(&self) -> &str {
            "static"
        }
    }

    fn static_context() -> StageContext {
        StageContext {
            stage_id: "test".to_string(),
            model_id: "m".to_string(),
            input_kind: EnvelopeKind::Text("hello".to_string()),
            metrics: default_metrics(),
            resource_monitor: ResourceMonitor::global(),
            explicit_target: None,
            local_availability: Some(crate::orchestrator::routing_engine::LocalAvailability::new(
                true,
            )),
            device_class: None,
            device_class_schema_version: None,
        }
    }

    fn static_request() -> PolicyRequest {
        PolicyRequest {
            stage_id: "test".to_string(),
            envelope: text_envelope("hello"),
            metrics: default_metrics(),
        }
    }

    #[test]
    fn default_load_policies_is_unsupported() {
        let authority = StaticAuthority {
            policy: PolicyOutcome::Allow,
            target_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let err = authority
            .load_policies(b"deny_cloud_if:\n  - \"true\"\n")
            .expect_err("custom authority without an engine must refuse");
        assert!(err.contains("not supported"), "{err}");
    }

    #[test]
    fn default_resolve_stage_skips_target_resolution_when_policy_requires_local() {
        for policy in [
            PolicyOutcome::Deny {
                reason: "no cloud".to_string(),
            },
            PolicyOutcome::Transform {
                transforms: vec!["scrub".to_string()],
            },
        ] {
            let authority = StaticAuthority {
                policy,
                target_calls: std::sync::atomic::AtomicUsize::new(0),
            };
            let resolution = authority.resolve_stage(&static_request(), &static_context());
            assert_eq!(resolution.target.decision.result, ResolvedTarget::Device);
            assert!(resolution.policy_requires_local());
            assert!(
                resolution.target.decision.reason.starts_with("policy_"),
                "{}",
                resolution.target.decision.reason
            );
            assert_eq!(
                authority
                    .target_calls
                    .load(std::sync::atomic::Ordering::SeqCst),
                0,
                "target resolution (and any remote advice) must not run for a local-only policy"
            );
        }
    }

    #[test]
    fn default_resolve_stage_delegates_when_policy_allows() {
        let authority = StaticAuthority {
            policy: PolicyOutcome::Allow,
            target_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let resolution = authority.resolve_stage(&static_request(), &static_context());
        assert!(matches!(
            resolution.target.decision.result,
            ResolvedTarget::Cloud { .. }
        ));
        assert_eq!(
            authority
                .target_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn stage_resolution_enforce_overrides_non_device_targets() {
        let mut resolution = StageResolution {
            policy: AuthorityDecision::local(
                PolicyOutcome::Transform {
                    transforms: vec!["scrub".to_string()],
                },
                "needs scrubbing",
            ),
            target: TargetResolution::new(
                AuthorityDecision::new(
                    ResolvedTarget::Cloud {
                        provider: "xybrid".to_string(),
                    },
                    "inconsistent custom authority",
                    DecisionSource::Remote,
                    0.9,
                ),
                "m",
                None,
            ),
        };

        assert!(resolution.enforce());
        assert_eq!(resolution.target.decision.result, ResolvedTarget::Device);
        let reason = resolution.target.decision.reason.clone();
        assert!(
            reason.starts_with("policy_transform_unsupported: scrub"),
            "{reason}"
        );
        assert!(
            reason.contains("overrode cloud:xybrid decision"),
            "{reason}"
        );
        assert!(reason.contains("inconsistent custom authority"), "{reason}");

        // Idempotent.
        assert!(!resolution.enforce());
        assert_eq!(resolution.target.decision.reason, reason);

        // Allow never overrides.
        let mut allowed = StageResolution {
            policy: AuthorityDecision::local(PolicyOutcome::Allow, "ok"),
            target: TargetResolution::new(
                AuthorityDecision::local(
                    ResolvedTarget::Cloud {
                        provider: "xybrid".to_string(),
                    },
                    "cloud",
                ),
                "m",
                None,
            ),
        };
        assert!(!allowed.enforce());
        assert!(matches!(
            allowed.target.decision.result,
            ResolvedTarget::Cloud { .. }
        ));
    }
}

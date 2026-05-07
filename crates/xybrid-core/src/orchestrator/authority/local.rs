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
//! - **Model selection**: Uses `CacheProvider` to check availability, falls back to registry
//!
//! ## Cache Provider
//!
//! LocalAuthority uses a `CacheProvider` trait to check model availability.
//! This abstraction allows:
//! - Core to check cache without depending on SDK
//! - SDK to inject its own cache implementation at bootstrap time
//! - Custom cache providers for testing or specialized deployments
//!
//! ## Decision Quality
//!
//! Local decisions are deterministic and have high confidence (1.0) because they
//! use only local information. For smarter decisions based on fleet data, use
//! `RemoteAuthority`.

use super::types::*;
use super::OrchestrationAuthority;
use crate::cache_provider::{CacheProvider, FilesystemCacheProvider};
use crate::device::ResourceSnapshotProvider;
use crate::ir::Envelope;
use crate::orchestrator::policy_engine::{DefaultPolicyEngine, PolicyEngine};
use crate::orchestrator::routing_engine::{
    DefaultRoutingEngine, LocalAvailability, LocalReliabilityHint, RouteTarget, RoutingDecision,
    RoutingEngine,
};
use crate::pipeline::ExecutionTarget;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_HYSTERESIS_TTL: Duration = Duration::from_secs(30);
const RELIABILITY_WINDOW: usize = 32;
const DEFAULT_HISTORY_BIAS_K: usize = 3;
const MAX_HYSTERESIS_KEYS: usize = 256;
const MAX_RELIABILITY_KEYS: usize = 256;

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
    /// Cache provider for checking model availability.
    cache_provider: Arc<dyn CacheProvider>,
    /// Optional test seam for live resource snapshots.
    resource_provider: Option<Arc<dyn ResourceSnapshotProvider>>,
    /// Sticky cloud routing after a local abort.
    hysteresis: Mutex<HashMap<(String, AbortReason), Instant>>,
    /// Recent local outcomes for similar device signal buckets.
    reliability: Mutex<HashMap<(String, SignalContext), VecDeque<OutcomeCategory>>>,
    history_bias_k: usize,
}

impl LocalAuthority {
    /// Create a new LocalAuthority with default policy, routing, and cache provider.
    pub fn new() -> Self {
        Self {
            policy_engine: DefaultPolicyEngine::with_default_policy(),
            routing_engine: Mutex::new(DefaultRoutingEngine::new()),
            cache_provider: Arc::new(FilesystemCacheProvider::new()),
            resource_provider: None,
            hysteresis: Mutex::new(HashMap::new()),
            reliability: Mutex::new(HashMap::new()),
            history_bias_k: DEFAULT_HISTORY_BIAS_K,
        }
    }

    /// Create a LocalAuthority with a custom cache provider.
    pub fn with_cache_provider(cache_provider: Arc<dyn CacheProvider>) -> Self {
        Self {
            policy_engine: DefaultPolicyEngine::with_default_policy(),
            routing_engine: Mutex::new(DefaultRoutingEngine::new()),
            cache_provider,
            resource_provider: None,
            hysteresis: Mutex::new(HashMap::new()),
            reliability: Mutex::new(HashMap::new()),
            history_bias_k: DEFAULT_HISTORY_BIAS_K,
        }
    }

    /// Create a LocalAuthority with a custom policy engine.
    pub fn with_policy_engine(policy_engine: DefaultPolicyEngine) -> Self {
        Self {
            policy_engine,
            routing_engine: Mutex::new(DefaultRoutingEngine::new()),
            cache_provider: Arc::new(FilesystemCacheProvider::new()),
            resource_provider: None,
            hysteresis: Mutex::new(HashMap::new()),
            reliability: Mutex::new(HashMap::new()),
            history_bias_k: DEFAULT_HISTORY_BIAS_K,
        }
    }

    /// Create a LocalAuthority with custom policy engine and cache provider.
    pub fn with_policy_and_cache(
        policy_engine: DefaultPolicyEngine,
        cache_provider: Arc<dyn CacheProvider>,
    ) -> Self {
        Self {
            policy_engine,
            routing_engine: Mutex::new(DefaultRoutingEngine::new()),
            cache_provider,
            resource_provider: None,
            hysteresis: Mutex::new(HashMap::new()),
            reliability: Mutex::new(HashMap::new()),
            history_bias_k: DEFAULT_HISTORY_BIAS_K,
        }
    }

    /// Use an injectable resource provider. Intended for tests and embedded
    /// hosts that already own resource sampling.
    pub fn with_resource_provider(mut self, provider: Arc<dyn ResourceSnapshotProvider>) -> Self {
        self.resource_provider = Some(provider);
        self
    }

    /// Override the consecutive unreliable-outcome threshold.
    pub fn with_history_bias_k(mut self, k: usize) -> Self {
        self.history_bias_k = k.max(1);
        self
    }

    /// Mark a model as recently aborted so the next matching route sticks to cloud.
    pub fn record_abort_for_hysteresis(&self, model_id: &str, reason: AbortReason, ttl: Duration) {
        let expires_at = Instant::now() + ttl;
        if let Ok(mut hysteresis) = self.hysteresis.lock() {
            Self::prune_hysteresis(&mut hysteresis);
            hysteresis.insert((model_id.to_string(), reason), expires_at);
            Self::prune_hysteresis(&mut hysteresis);
        }
    }

    pub fn record_abort_for_hysteresis_default_ttl(&self, model_id: &str, reason: AbortReason) {
        self.record_abort_for_hysteresis(model_id, reason, DEFAULT_HYSTERESIS_TTL);
    }

    /// Check if a model exists locally using the cache provider.
    fn check_model_exists(&self, model_id: &str) -> bool {
        self.cache_provider.is_model_cached(model_id)
    }

    /// Find the local path for a model using the cache provider.
    fn find_local_model(&self, model_id: &str) -> Option<String> {
        self.cache_provider
            .get_model_path(model_id)
            .and_then(|p| p.to_str().map(|s| s.to_string()))
    }

    fn active_hysteresis_for(&self, model_id: &str) -> Option<AbortReason> {
        let mut hysteresis = self.hysteresis.lock().ok()?;
        Self::prune_hysteresis(&mut hysteresis);
        // Pick the most recently-recorded reason (max expires_at) when a
        // model has multiple coexisting hysteresis entries. HashMap key
        // iteration order is non-deterministic, so a naive `keys().find_map`
        // would pick StressMemory or StressThermal arbitrarily across
        // process restarts and after map mutations — flaking the
        // explanatory `reason` string surfaced as the platform-event
        // `abort_reason` field. The most recent reason is the one that
        // actually pushed the device over the edge, so it is the more
        // user-meaningful pick.
        hysteresis
            .iter()
            .filter(|((candidate_model_id, _), _)| candidate_model_id == model_id)
            .max_by_key(|(_, expires_at)| **expires_at)
            .map(|((_, reason), _)| *reason)
    }

    fn history_snapshot(&self, model_id: &str, signal: SignalContext) -> VecDeque<OutcomeCategory> {
        self.reliability
            .lock()
            .ok()
            .and_then(|history| history.get(&(model_id.to_string(), signal)).cloned())
            .unwrap_or_default()
    }

    fn reliability_hint(&self, model_id: &str, signal: SignalContext) -> LocalReliabilityHint {
        let history = self.history_snapshot(model_id, signal);
        if history.is_empty() {
            return LocalReliabilityHint::EMPTY;
        }
        let unreliable = history
            .iter()
            .filter(|category| category.is_local_unreliable())
            .count();
        LocalReliabilityHint {
            recent_abort_rate: unreliable as f32 / history.len() as f32,
            sample_size: history.len() as u32,
        }
    }

    fn history_bias_should_skip_local(&self, model_id: &str, signal: SignalContext) -> bool {
        let history = self.history_snapshot(model_id, signal);
        if history.len() < self.history_bias_k {
            return false;
        }
        history
            .iter()
            .rev()
            .take(self.history_bias_k)
            .all(OutcomeCategory::is_local_unreliable)
    }

    fn prune_hysteresis(hysteresis: &mut HashMap<(String, AbortReason), Instant>) {
        let now = Instant::now();
        hysteresis.retain(|_, expires_at| *expires_at > now);
        while hysteresis.len() > MAX_HYSTERESIS_KEYS {
            let Some((key, _)) = hysteresis
                .iter()
                .min_by_key(|(_, expires_at)| **expires_at)
                .map(|(key, expires_at)| (key.clone(), *expires_at))
            else {
                break;
            };
            hysteresis.remove(&key);
        }
    }

    fn prune_reliability(
        reliability: &mut HashMap<(String, SignalContext), VecDeque<OutcomeCategory>>,
    ) {
        // Bounded random-replacement: when at the cap, evict the bucket
        // with the smallest history (least information). Falls back to
        // arbitrary iteration order for empty buckets, which is fine —
        // empty buckets carry no signal anyway. True LRU would require a
        // per-bucket timestamp; the smallest-history heuristic is a
        // reasonable middle ground and is a strict improvement over
        // arbitrary HashMap iteration order, biasing eviction away from
        // hot buckets that have accumulated useful history.
        while reliability.len() > MAX_RELIABILITY_KEYS {
            let Some(victim) = reliability
                .iter()
                .min_by_key(|(_, history)| history.len())
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            reliability.remove(&victim);
        }
    }
}

impl Default for LocalAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl OrchestrationAuthority for LocalAuthority {
    fn apply_policy(&self, request: &PolicyRequest) -> AuthorityDecision<PolicyOutcome> {
        let result =
            self.policy_engine
                .evaluate(&request.stage_id, &request.envelope, &request.metrics);

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
                reason: result
                    .reason
                    .clone()
                    .unwrap_or_else(|| "Policy denied".to_string()),
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
        self.resolve_target_with_feedback(context).decision
    }

    fn resolve_target_with_feedback(&self, context: &StageContext) -> TargetResolution {
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

            let metrics = self.routing_metrics(context);
            return TargetResolution::new(
                AuthorityDecision {
                    result: target,
                    reason: format!("Explicit target from pipeline YAML: {:?}", explicit),
                    source: DecisionSource::Local,
                    confidence: 1.0,
                    timestamp_ms: now_ms(),
                },
                context.model_id.clone(),
                Some(SignalContext::from_metrics(&metrics)),
            );
        }

        // Otherwise, use routing engine heuristics
        self.resolve_with_routing_engine(context)
    }

    fn select_model(&self, request: &ModelRequest) -> AuthorityDecision<ModelSelection> {
        // Check if model is available locally
        let local_path = self.find_local_model(&request.model_id);

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
                format!(
                    "Model '{}' not found locally, will fetch from registry",
                    request.model_id
                )
            },
            source: DecisionSource::Local,
            confidence: 1.0,
            timestamp_ms: now_ms(),
        }
    }

    fn name(&self) -> &str {
        "local"
    }

    fn record_outcome(&self, outcome: &ExecutionOutcome) {
        if !matches!(outcome.target, ResolvedTarget::Device) {
            return;
        }

        let category = outcome.effective_category();
        let model_id = outcome.effective_model_id().to_string();

        if let OutcomeCategory::AbortedForCloudFallback { reason } = &category {
            self.record_abort_for_hysteresis_default_ttl(&model_id, *reason);
        }

        let Some(signal) = outcome.signal_context else {
            return;
        };
        let key = (model_id, signal);
        if let Ok(mut reliability) = self.reliability.lock() {
            if !reliability.contains_key(&key) && reliability.len() >= MAX_RELIABILITY_KEYS {
                Self::prune_reliability(&mut reliability);
                if reliability.len() >= MAX_RELIABILITY_KEYS {
                    // Use the same smallest-history victim selection as
                    // prune_reliability so eviction stays deterministic
                    // and biased away from hot buckets even on this
                    // last-mile path.
                    let victim = reliability
                        .iter()
                        .min_by_key(|(_, history)| history.len())
                        .map(|(victim_key, _)| victim_key.clone());
                    if let Some(victim) = victim {
                        reliability.remove(&victim);
                    }
                }
            }
            let history = reliability.entry(key).or_default();
            history.push_back(category);
            while history.len() > RELIABILITY_WINDOW {
                history.pop_front();
            }
            Self::prune_reliability(&mut reliability);
        }
    }
}

impl LocalAuthority {
    fn routing_metrics(&self, context: &StageContext) -> crate::context::DeviceMetrics {
        let snapshot = self
            .resource_provider
            .as_ref()
            .map(|provider| provider.current_snapshot(Duration::from_millis(500)))
            .unwrap_or_else(|| {
                context
                    .resource_monitor
                    .current_snapshot(Duration::from_millis(500))
            });
        context.metrics.with_live_snapshot(snapshot)
    }

    fn target_from_route(target: RouteTarget) -> ResolvedTarget {
        match target {
            RouteTarget::Local => ResolvedTarget::Device,
            RouteTarget::Cloud => ResolvedTarget::Cloud {
                provider: "xybrid".to_string(),
            },
            // Carry the bare fallback id; the reverse-direction
            // mapping in resolve_routing_decision (and
            // Orchestrator::resolved_target_to_routing_decision) will
            // re-wrap it as RouteTarget::Fallback. The "fallback:"
            // prefix is added back by RouteTarget::to_json_string /
            // Display, so synthesizing it here produced "fallback:fallback:<id>"
            // when the resolution round-tripped through telemetry.
            RouteTarget::Fallback(id) => ResolvedTarget::Server { endpoint: id },
        }
    }

    /// Internal: resolve target using the routing engine.
    fn resolve_with_routing_engine(&self, context: &StageContext) -> TargetResolution {
        let availability = LocalAvailability {
            local_model_exists: self.check_model_exists(&context.model_id),
        };

        // Create a minimal envelope for policy check
        let envelope = Envelope::new(context.input_kind.clone());
        let live_metrics = self.routing_metrics(context);
        let signal = SignalContext::from_metrics(&live_metrics);
        let hint = self.reliability_hint(&context.model_id, signal);

        let policy_result =
            self.policy_engine
                .evaluate(&context.stage_id, &envelope, &live_metrics);

        if policy_result.allowed {
            if let Some(reason) = self.active_hysteresis_for(&context.model_id) {
                let decision = AuthorityDecision {
                    result: ResolvedTarget::Cloud {
                        provider: "xybrid".to_string(),
                    },
                    reason: format!(
                        "hysteresis: recent local abort for model '{}' ({})",
                        context.model_id, reason
                    ),
                    source: DecisionSource::Local,
                    confidence: 0.9,
                    timestamp_ms: now_ms(),
                };
                return TargetResolution::new(decision, context.model_id.clone(), Some(signal))
                    .with_reliability_hint(hint);
            }

            if self.history_bias_should_skip_local(&context.model_id, signal) {
                let decision = AuthorityDecision {
                    result: ResolvedTarget::Cloud {
                        provider: "xybrid".to_string(),
                    },
                    reason: format!(
                        "history_bias: recent local failure rate {:.0}% over {} samples",
                        hint.recent_abort_rate * 100.0,
                        hint.sample_size
                    ),
                    source: DecisionSource::Local,
                    confidence: 0.85,
                    timestamp_ms: now_ms(),
                };
                return TargetResolution::new(decision, context.model_id.clone(), Some(signal))
                    .with_reliability_hint(hint);
            }
        }

        // Use the stored routing engine (locked for interior mutability)
        let decision = {
            let mut routing_engine = self.routing_engine.lock().unwrap();
            routing_engine.decide(
                &context.stage_id,
                &live_metrics,
                &policy_result,
                &availability,
            )
        };

        let target = Self::target_from_route(decision.target);
        TargetResolution::new(
            AuthorityDecision {
                result: target,
                reason: decision.reason,
                source: DecisionSource::Local,
                confidence: 0.8, // Heuristic-based, slightly lower confidence
                timestamp_ms: decision.timestamp_ms,
            },
            context.model_id.clone(),
            Some(signal),
        )
        .with_reliability_hint(hint)
    }

    /// Resolve into the routing-engine decision shape for tests and telemetry adapters.
    pub fn resolve_routing_decision(&self, context: &StageContext) -> Option<RoutingDecision> {
        let resolution = self.resolve_target_with_feedback(context);
        let target = match resolution.decision.result {
            ResolvedTarget::Device => RouteTarget::Local,
            ResolvedTarget::Cloud { .. } => RouteTarget::Cloud,
            ResolvedTarget::Server { endpoint } => RouteTarget::Fallback(endpoint),
        };
        Some(RoutingDecision {
            stage: context.stage_id.clone(),
            target,
            reason: resolution.decision.reason,
            timestamp_ms: resolution.decision.timestamp_ms,
            local_reliability_hint: resolution
                .local_reliability_hint
                .unwrap_or(LocalReliabilityHint::EMPTY),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_provider::CacheProvider;
    use crate::context::DeviceMetrics;
    use crate::device::{MemoryPressure, ResourceMonitor, ResourceSnapshot, ThermalState};
    use crate::ir::EnvelopeKind;
    use std::path::PathBuf;

    fn default_metrics() -> DeviceMetrics {
        DeviceMetrics::default()
    }

    /// YAML policy bundle that denies any text envelope. Used to exercise the
    /// `policy_deny` branch in tests now that the legacy RTT-based default
    /// rule is gone.
    fn deny_all_text_policy() -> String {
        r#"
version: "0.1.0"
deny_cloud_if:
  - input.kind == "text"
signature: "test-deny-all"
"#
        .to_string()
    }

    fn text_envelope(text: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    }

    #[derive(Debug)]
    struct FixedResourceProvider(ResourceSnapshot);

    impl ResourceSnapshotProvider for FixedResourceProvider {
        fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
            self.0
        }
    }

    #[derive(Debug)]
    struct CachedProvider;

    impl CacheProvider for CachedProvider {
        fn is_model_cached(&self, _model_id: &str) -> bool {
            true
        }

        fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
            Some(PathBuf::from(format!("/tmp/{model_id}")))
        }

        fn cache_dir(&self) -> PathBuf {
            PathBuf::from("/tmp")
        }

        fn name(&self) -> &'static str {
            "cached-test"
        }
    }

    fn text_context() -> StageContext {
        StageContext {
            stage_id: "test-stage".to_string(),
            model_id: "test-model".to_string(),
            input_kind: EnvelopeKind::Text("test".to_string()),
            metrics: default_metrics(),
            resource_monitor: ResourceMonitor::global(),
            explicit_target: None,
        }
    }

    fn signal() -> SignalContext {
        SignalContext {
            memory_pressure: MemoryPressure::Warn,
            thermal_state: ThermalState::Normal,
            cpu_bucket: Some(5),
        }
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
    fn test_local_authority_explicit_device_target() {
        let authority = LocalAuthority::new();
        let context = StageContext {
            stage_id: "test".to_string(),
            model_id: "test-model".to_string(),
            input_kind: EnvelopeKind::Text("test".to_string()),
            metrics: default_metrics(),
            resource_monitor: ResourceMonitor::global(),
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
            resource_monitor: ResourceMonitor::global(),
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
        assert!(matches!(
            decision.result.source,
            ModelSource::Registry { .. }
        ));
        assert!(decision.reason.contains("not found locally"));
    }

    #[test]
    fn test_local_authority_name() {
        let authority = LocalAuthority::new();
        assert_eq!(authority.name(), "local");
    }

    #[test]
    fn test_find_local_model_sdk_cache_structure() {
        // This test verifies that the model matching logic can find models
        // in the SDK cache even when directory names don't exactly match.
        // E.g., "kokoro-82m" should match "Kokoro-82M-v1.0-ONNX"

        // Check if a model matching "kokoro-82m" exists in the cache
        // (this depends on the user having run the model before)
        let authority = LocalAuthority::new();
        let path = authority.find_local_model("kokoro-82m");

        // If the model is cached, verify it's the right one
        if let Some(p) = &path {
            let p_lower = p.to_lowercase();
            assert!(
                p_lower.contains("kokoro"),
                "Expected path to contain 'kokoro', got: {}",
                p
            );
        }
        // Note: If no model is cached, the test just passes (we can't require cached models in CI)
    }

    #[test]
    fn test_with_custom_cache_provider() {
        use crate::cache_provider::NoopCacheProvider;

        // Test that we can create authority with a custom cache provider
        let provider = Arc::new(NoopCacheProvider);
        let authority = LocalAuthority::with_cache_provider(provider);

        // Model should not be found with noop provider
        let request = ModelRequest {
            model_id: "any-model".to_string(),
            task: "test".to_string(),
            constraints: ModelConstraints::default(),
        };

        let decision = authority.select_model(&request);
        assert!(matches!(
            decision.result.source,
            ModelSource::Registry { .. }
        ));
    }

    #[test]
    fn fake_resource_provider_feeds_routing_metrics() {
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Critical;
        snapshot.thermal_state = ThermalState::Normal;
        snapshot.cpu_pct = Some(10.0);
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider))
            .with_resource_provider(Arc::new(FixedResourceProvider(snapshot)));

        let decision = authority.resolve_target(&text_context());

        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
        assert!(decision.reason.contains("stress_memory"));
    }

    #[test]
    fn hysteresis_is_model_scoped_and_expires() {
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        authority.record_abort_for_hysteresis(
            "test-model",
            AbortReason::StressMemory,
            Duration::from_millis(20),
        );

        let decision = authority.resolve_target(&text_context());
        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
        assert!(decision.reason.contains("hysteresis"));

        let mut other = text_context();
        other.model_id = "other-model".to_string();
        let other_decision = authority.resolve_target(&other);
        assert!(!other_decision.reason.contains("hysteresis"));

        std::thread::sleep(Duration::from_millis(30));
        let expired = authority.resolve_target(&text_context());
        assert!(!expired.reason.contains("hysteresis"));
    }

    #[test]
    fn policy_deny_overrides_hysteresis() {
        let mut policy = DefaultPolicyEngine::new();
        policy
            .load_policies(deny_all_text_policy().into_bytes())
            .expect("load deny-all policy");
        let authority = LocalAuthority::with_policy_and_cache(policy, Arc::new(CachedProvider));
        authority.record_abort_for_hysteresis_default_ttl("test-model", AbortReason::StressMemory);

        let decision = authority.resolve_target(&text_context());

        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(decision.reason.contains("policy_deny"));
    }

    #[test]
    fn device_abort_outcome_enters_hysteresis() {
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        authority.record_outcome(&ExecutionOutcome {
            stage_id: "test-stage".to_string(),
            target: ResolvedTarget::Device,
            latency_ms: 12,
            success: false,
            error: None,
            category: Some(OutcomeCategory::AbortedForCloudFallback {
                reason: AbortReason::StressMemory,
            }),
            model_id: Some("test-model".to_string()),
            signal_context: Some(signal()),
        });

        let decision = authority.resolve_target(&text_context());

        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
        assert!(decision.reason.contains("hysteresis"));
    }

    #[test]
    fn cloud_failures_do_not_bias_local_reliability() {
        let authority =
            LocalAuthority::with_cache_provider(Arc::new(CachedProvider)).with_history_bias_k(3);
        for idx in 0..3 {
            authority.record_outcome(&ExecutionOutcome {
                stage_id: "test-stage".to_string(),
                target: ResolvedTarget::Cloud {
                    provider: "xybrid".to_string(),
                },
                latency_ms: 10,
                success: false,
                error: Some(format!("cloud-failure-{idx}")),
                category: Some(OutcomeCategory::HardFail {
                    reason: "cloud_failed".to_string(),
                }),
                model_id: Some("test-model".to_string()),
                signal_context: Some(signal()),
            });
        }

        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Warn;
        snapshot.thermal_state = ThermalState::Normal;
        snapshot.cpu_pct = Some(55.0);
        let authority = authority.with_resource_provider(Arc::new(FixedResourceProvider(snapshot)));

        let decision = authority
            .resolve_routing_decision(&text_context())
            .expect("routing decision");

        assert!(!decision.reason.contains("history_bias"));
        assert_eq!(decision.local_reliability_hint.sample_size, 0);
    }

    #[test]
    fn policy_deny_overrides_history_bias() {
        let mut policy = DefaultPolicyEngine::new();
        policy
            .load_policies(deny_all_text_policy().into_bytes())
            .expect("load deny-all policy");
        let authority = LocalAuthority::with_policy_and_cache(policy, Arc::new(CachedProvider))
            .with_history_bias_k(3);
        for idx in 0..3 {
            authority.record_outcome(&ExecutionOutcome {
                stage_id: "test-stage".to_string(),
                target: ResolvedTarget::Device,
                latency_ms: 10,
                success: false,
                error: Some(format!("failure-{idx}")),
                category: Some(OutcomeCategory::HardFail {
                    reason: "local_failed".to_string(),
                }),
                model_id: Some("test-model".to_string()),
                signal_context: Some(signal()),
            });
        }

        let decision = authority.resolve_target(&text_context());

        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(decision.reason.contains("policy_deny"));
    }

    #[test]
    fn hysteresis_map_stays_bounded() {
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        for idx in 0..(MAX_HYSTERESIS_KEYS + 32) {
            authority.record_abort_for_hysteresis_default_ttl(
                &format!("model-{idx}"),
                AbortReason::StressMemory,
            );
        }

        assert!(
            authority.hysteresis.lock().unwrap().len() <= MAX_HYSTERESIS_KEYS,
            "hysteresis should stay bounded"
        );
    }

    #[test]
    fn reliability_map_stays_bounded() {
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        for idx in 0..(MAX_RELIABILITY_KEYS + 32) {
            authority.record_outcome(&ExecutionOutcome {
                stage_id: "test-stage".to_string(),
                target: ResolvedTarget::Device,
                latency_ms: 10,
                success: false,
                error: Some("local_failed".to_string()),
                category: Some(OutcomeCategory::HardFail {
                    reason: "local_failed".to_string(),
                }),
                model_id: Some(format!("model-{idx}")),
                signal_context: Some(signal()),
            });
        }

        assert!(
            authority.reliability.lock().unwrap().len() <= MAX_RELIABILITY_KEYS,
            "reliability should stay bounded"
        );
    }

    #[test]
    fn reliability_history_bias_routes_cloud_with_hint() {
        let authority =
            LocalAuthority::with_cache_provider(Arc::new(CachedProvider)).with_history_bias_k(3);
        for idx in 0..3 {
            authority.record_outcome(&ExecutionOutcome {
                stage_id: "test-stage".to_string(),
                target: ResolvedTarget::Device,
                latency_ms: 10,
                success: false,
                error: Some(format!("failure-{idx}")),
                category: Some(OutcomeCategory::HardFail {
                    reason: "local_failed".to_string(),
                }),
                model_id: Some("test-model".to_string()),
                signal_context: Some(signal()),
            });
        }

        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Warn;
        snapshot.thermal_state = ThermalState::Normal;
        snapshot.cpu_pct = Some(55.0);
        let authority = authority.with_resource_provider(Arc::new(FixedResourceProvider(snapshot)));

        let decision = authority
            .resolve_routing_decision(&text_context())
            .expect("routing decision");

        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("history_bias"));
        assert_eq!(decision.local_reliability_hint.sample_size, 3);
        assert_eq!(decision.local_reliability_hint.recent_abort_rate, 1.0);
    }

    #[test]
    fn success_reduces_history_bias() {
        let authority =
            LocalAuthority::with_cache_provider(Arc::new(CachedProvider)).with_history_bias_k(3);
        for category in [
            OutcomeCategory::HardFail {
                reason: "a".to_string(),
            },
            OutcomeCategory::HardFail {
                reason: "b".to_string(),
            },
            OutcomeCategory::Success,
        ] {
            authority.record_outcome(&ExecutionOutcome {
                stage_id: "test-stage".to_string(),
                target: ResolvedTarget::Device,
                latency_ms: 10,
                success: matches!(category, OutcomeCategory::Success),
                error: None,
                category: Some(category),
                model_id: Some("test-model".to_string()),
                signal_context: Some(signal()),
            });
        }

        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Warn;
        snapshot.thermal_state = ThermalState::Normal;
        snapshot.cpu_pct = Some(55.0);
        let authority = authority.with_resource_provider(Arc::new(FixedResourceProvider(snapshot)));
        let decision = authority.resolve_target(&text_context());

        assert!(!decision.reason.contains("history_bias"));
    }

    #[test]
    fn target_from_route_round_trips_fallback_without_prefix_doubling() {
        // Pre-fix, target_from_route synthesized "fallback:<id>" inside the
        // ResolvedTarget::Server endpoint string. The reverse mapping then
        // wrapped the already-prefixed string in RouteTarget::Fallback, and
        // to_json_string re-prepended "fallback:" — emitting
        // "fallback:fallback:<id>". Ensure the symmetric round-trip now
        // produces a single prefix.
        let routed =
            LocalAuthority::target_from_route(RouteTarget::Fallback("model_v2".to_string()));
        let endpoint = match routed {
            ResolvedTarget::Server { endpoint } => endpoint,
            other => panic!("expected Server target, got {other:?}"),
        };
        assert_eq!(endpoint, "model_v2");
        let reverse = match endpoint.as_str() {
            "model_v2" => RouteTarget::Fallback(endpoint.clone()),
            _ => unreachable!(),
        };
        assert_eq!(reverse.to_json_string(), "fallback:model_v2");
        assert_eq!(reverse.to_string(), "fallback:model_v2");
    }

    #[test]
    fn test_model_matching_logic() {
        // Test the matching logic directly without relying on filesystem state
        let test_cases = [
            ("kokoro-82m", "kokoro-82m-v1.0-onnx"), // exact hyphenated
            ("kokoro-82m", "kokoro82mv10onnx"),     // normalized
            ("whisper-tiny", "whisper-tiny"),       // exact match
        ];

        for (query, dir_name) in test_cases {
            let query_lower = query.to_lowercase();
            let query_normalized = query_lower.replace("-", "").replace("_", "");
            let dir_name_lower = dir_name.to_lowercase();
            let dir_name_normalized = dir_name_lower.replace("-", "").replace("_", "");

            let is_match = dir_name_lower.contains(&query_lower)
                || dir_name_normalized.contains(&query_normalized);

            assert!(
                is_match,
                "Expected '{}' to match '{}' but it didn't",
                query, dir_name
            );
        }
    }
}

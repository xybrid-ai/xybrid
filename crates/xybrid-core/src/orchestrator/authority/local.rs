//! Local Orchestration Authority - Fully functional offline implementation.
//!
//! This is the default authority that ships with xybrid. It uses device metrics
//! and heuristics to make decisions locally. No network calls, no phone-home,
//! completely transparent.
//!
//! ## How It Works
//!
//! `LocalAuthority` owns the one policy engine and wraps the routing ladder:
//!
//! - **One prepared decision**: every stage decision takes exactly one live
//!   resource snapshot and evaluates the policy exactly once, against the
//!   actual input envelope. The same prepared values drive the policy event,
//!   the routing decision, and the feedback signal bucket.
//! - **Policy is an invariant, not a hint**: when the policy denies cloud (or
//!   requires a transform that cannot be applied) the target is the device —
//!   ahead of explicit `cloud`/`server` targets, model availability,
//!   hysteresis, and reliability history. A stage with no usable local leg
//!   then fails locally rather than leaking to cloud.
//! - **Target resolution**: explicit target → local availability → policy
//!   cloud preference → hysteresis → history bias → the routing ladder
//!   (device stress, default local).
//! - **Model selection**: Uses `CacheProvider` to check availability, falls
//!   back to registry.
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
use crate::context::DeviceMetrics;
use crate::device::{ResourceMonitor, ResourceSnapshot, ResourceSnapshotProvider};
use crate::ir::Envelope;
use crate::orchestrator::policy_engine::{
    DefaultPolicyEngine, PolicyBundle, PolicyEngine, PolicyResult,
};
use crate::orchestrator::routing_engine::{
    DefaultRoutingEngine, LocalAvailability, LocalReliabilityHint, RouteTarget, RoutingDecision,
    RoutingEngine,
};
use crate::pipeline::ExecutionTarget;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

const DEFAULT_HYSTERESIS_TTL: Duration = Duration::from_secs(30);
const RELIABILITY_WINDOW: usize = 32;
const DEFAULT_HISTORY_BIAS_K: usize = 3;
const MAX_HYSTERESIS_KEYS: usize = 256;
const MAX_RELIABILITY_KEYS: usize = 256;
/// Maximum age of a cached resource snapshot before it is refreshed.
const RESOURCE_SNAPSHOT_MAX_AGE: Duration = Duration::from_millis(500);

/// Local orchestration authority - fully functional offline.
///
/// This is the default authority that ships with xybrid.
/// It uses device metrics and heuristics to make decisions locally.
/// No network calls, no phone-home, completely transparent.
///
/// # Example
///
/// ```no_run
/// # fn _example() {
/// use xybrid_core::orchestrator::authority::{LocalAuthority, OrchestrationAuthority, PolicyRequest};
///
/// # let request: PolicyRequest = unimplemented!();
/// let authority = LocalAuthority::new();
/// let decision = authority.apply_policy(&request);
/// println!("Decision: {:?} ({})", decision.result, decision.reason);
/// # }
/// ```
pub struct LocalAuthority {
    /// The one policy engine. Reads on the per-stage hot path, writes only
    /// on `load_policies`; a poisoned lock is recovered rather than panicked.
    policy_engine: RwLock<DefaultPolicyEngine>,
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

/// Everything one stage decision needs, computed exactly once: the live
/// metrics, the signal bucket derived from them, and the policy result for
/// the actual input. Shared between `LocalAuthority` and `RemoteAuthority` so
/// the remote path never re-evaluates or re-samples.
pub(super) struct PreparedStage {
    pub(super) metrics: DeviceMetrics,
    pub(super) signal: SignalContext,
    pub(super) policy: PolicyResult,
}

impl LocalAuthority {
    fn build(policy_engine: DefaultPolicyEngine, cache_provider: Arc<dyn CacheProvider>) -> Self {
        // Prewarm the static-capability cache. First call to
        // `detect_capabilities()` can take ~1s on macOS/iOS because
        // `MLAllComputeDevices` lazy-loads Core ML. Doing it here keeps
        // that cost out of latency-sensitive routing paths (e.g. the
        // hysteresis check measured in tens of ms).
        crate::device::capabilities::prewarm();
        Self {
            policy_engine: RwLock::new(policy_engine),
            routing_engine: Mutex::new(DefaultRoutingEngine::new()),
            cache_provider,
            resource_provider: None,
            hysteresis: Mutex::new(HashMap::new()),
            reliability: Mutex::new(HashMap::new()),
            history_bias_k: DEFAULT_HISTORY_BIAS_K,
        }
    }

    /// Create a new LocalAuthority with default policy, routing, and cache provider.
    pub fn new() -> Self {
        Self::build(
            DefaultPolicyEngine::with_default_policy(),
            Arc::new(FilesystemCacheProvider::new()),
        )
    }

    /// Create a LocalAuthority with a custom cache provider.
    pub fn with_cache_provider(cache_provider: Arc<dyn CacheProvider>) -> Self {
        Self::build(DefaultPolicyEngine::with_default_policy(), cache_provider)
    }

    /// Create a LocalAuthority with a custom policy engine.
    pub fn with_policy_engine(policy_engine: DefaultPolicyEngine) -> Self {
        Self::build(policy_engine, Arc::new(FilesystemCacheProvider::new()))
    }

    /// Create a LocalAuthority with custom policy engine and cache provider.
    pub fn with_policy_and_cache(
        policy_engine: DefaultPolicyEngine,
        cache_provider: Arc<dyn CacheProvider>,
    ) -> Self {
        Self::build(policy_engine, cache_provider)
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

    /// Inspect the active policy bundle (e.g. to report its version and rule
    /// count). The closure runs under the read lock; keep it short.
    pub fn with_policy_bundle<R>(&self, f: impl FnOnce(Option<&PolicyBundle>) -> R) -> R {
        f(self.policy_read().bundle())
    }

    fn policy_read(&self) -> RwLockReadGuard<'_, DefaultPolicyEngine> {
        self.policy_engine
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn policy_write(&self) -> RwLockWriteGuard<'_, DefaultPolicyEngine> {
        self.policy_engine
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Check if a model exists locally using the cache provider.
    pub(super) fn check_model_exists(&self, model_id: &str) -> bool {
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

    pub(super) fn reliability_hint(
        &self,
        model_id: &str,
        signal: SignalContext,
    ) -> LocalReliabilityHint {
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

    fn live_snapshot(&self, monitor: &ResourceMonitor) -> ResourceSnapshot {
        self.resource_provider
            .as_ref()
            .map(|provider| provider.current_snapshot(RESOURCE_SNAPSHOT_MAX_AGE))
            .unwrap_or_else(|| monitor.current_snapshot(RESOURCE_SNAPSHOT_MAX_AGE))
    }

    /// Take one live snapshot and evaluate the policy once against the actual
    /// input. Every decision path — standalone policy, target-only, and the
    /// combined stage decision — goes through here.
    pub(super) fn prepare_stage(
        &self,
        stage_id: &str,
        envelope: &Envelope,
        base_metrics: &DeviceMetrics,
        monitor: &ResourceMonitor,
    ) -> PreparedStage {
        let metrics = base_metrics.with_live_snapshot(self.live_snapshot(monitor));
        let signal = SignalContext::from_metrics(&metrics);
        let policy = self.policy_read().evaluate(stage_id, envelope, &metrics);
        PreparedStage {
            metrics,
            signal,
            policy,
        }
    }

    /// Convert an engine result into the authority's policy decision.
    pub(super) fn policy_decision(policy: &PolicyResult) -> AuthorityDecision<PolicyOutcome> {
        let outcome = if !policy.allowed {
            PolicyOutcome::Deny {
                reason: policy
                    .reason
                    .clone()
                    .unwrap_or_else(|| "Policy denied".to_string()),
            }
        } else if !policy.transforms_applied.is_empty() {
            PolicyOutcome::Transform {
                transforms: policy.transforms_applied.clone(),
            }
        } else {
            PolicyOutcome::Allow
        };
        AuthorityDecision {
            result: outcome,
            reason: policy
                .reason
                .clone()
                .unwrap_or_else(|| "Local policy evaluation".to_string()),
            source: DecisionSource::Local,
            confidence: 1.0, // Local decisions are deterministic
            timestamp_ms: now_ms(),
        }
    }

    fn cloud_target() -> ResolvedTarget {
        ResolvedTarget::Cloud {
            provider: "xybrid".to_string(),
        }
    }

    fn target_from_route(target: RouteTarget) -> ResolvedTarget {
        match target {
            RouteTarget::Local => ResolvedTarget::Device,
            RouteTarget::Cloud => Self::cloud_target(),
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

    /// Explicit pipeline target, if one was declared. Consulted only after
    /// the policy has permitted leaving the device.
    fn explicit_target(context: &StageContext) -> Option<(ResolvedTarget, String)> {
        let explicit = context.explicit_target.as_ref()?;
        let target = match explicit {
            ExecutionTarget::Device => ResolvedTarget::Device,
            ExecutionTarget::Server => ResolvedTarget::Server {
                endpoint: "https://api.xybrid.dev".to_string(),
            },
            ExecutionTarget::Cloud => Self::cloud_target(),
            ExecutionTarget::Auto => return None,
        };
        Some((
            target,
            format!("Explicit target from pipeline YAML: {:?}", explicit),
        ))
    }

    /// Resolve the target from an already-prepared decision.
    ///
    /// Precedence: policy local-only → explicit target → local availability →
    /// policy cloud preference → hysteresis → history bias → routing ladder
    /// (device stress, default local). The policy is not re-evaluated and no
    /// second snapshot is taken.
    pub(super) fn resolve_prepared(
        &self,
        context: &StageContext,
        prepared: PreparedStage,
    ) -> TargetResolution {
        let PreparedStage {
            metrics,
            signal,
            policy,
        } = prepared;
        let model_id = context.model_id.clone();
        let hint = self.reliability_hint(&model_id, signal);
        let finish = |decision: AuthorityDecision<ResolvedTarget>| {
            TargetResolution::new(decision, model_id.clone(), Some(signal))
                .with_reliability_hint(hint)
        };

        // 1. Policy forbids leaving the device. This outranks explicit
        //    targets, availability, hysteresis and history.
        if policy.requires_local() {
            let reason = if !policy.allowed {
                format!(
                    "policy_deny: {}",
                    policy
                        .reason
                        .as_deref()
                        .unwrap_or("policy denied cloud execution")
                )
            } else {
                format!(
                    "policy_transform_unsupported: {}",
                    policy.transforms_applied.join(",")
                )
            };
            return finish(AuthorityDecision::local(ResolvedTarget::Device, reason));
        }

        // 2. A permitted explicit target wins over every preference.
        if let Some((target, reason)) = Self::explicit_target(context) {
            return finish(AuthorityDecision::local(target, reason));
        }

        // 3. No local leg: cloud is the only usable target.
        let availability = context
            .local_availability
            .clone()
            .unwrap_or_else(|| LocalAvailability::new(self.check_model_exists(&model_id)));
        if !availability.local_model_exists {
            return finish(AuthorityDecision::new(
                Self::cloud_target(),
                "model_unavailable: local model not found",
                DecisionSource::Local,
                0.8,
            ));
        }

        // 4. The policy asked for cloud and the local leg exists.
        if policy.prefers_cloud() {
            return finish(AuthorityDecision::new(
                Self::cloud_target(),
                format!(
                    "policy_route_cloud: {}",
                    policy
                        .reason
                        .as_deref()
                        .unwrap_or("policy prefers cloud execution")
                ),
                DecisionSource::Local,
                0.9,
            ));
        }

        // 5. Sticky cloud after a recent local abort.
        if let Some(reason) = self.active_hysteresis_for(&model_id) {
            return finish(AuthorityDecision::new(
                Self::cloud_target(),
                format!(
                    "hysteresis: recent local abort for model '{}' ({})",
                    model_id, reason
                ),
                DecisionSource::Local,
                0.9,
            ));
        }

        // 6. Recent local failures under this signal bucket.
        if self.history_bias_should_skip_local(&model_id, signal) {
            return finish(AuthorityDecision::new(
                Self::cloud_target(),
                format!(
                    "history_bias: recent local failure rate {:.0}% over {} samples",
                    hint.recent_abort_rate * 100.0,
                    hint.sample_size
                ),
                DecisionSource::Local,
                0.85,
            ));
        }

        // 7. Device stress heuristics, default local. Recover from a poisoned
        //    lock rather than panicking: this is the per-stage routing hot
        //    path, so a single panic elsewhere must not turn every subsequent
        //    routing decision into a crash.
        let decision = {
            let mut routing_engine = self
                .routing_engine
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            routing_engine.decide(&context.stage_id, &metrics, &policy, &availability)
        };

        finish(AuthorityDecision {
            result: Self::target_from_route(decision.target),
            reason: decision.reason,
            source: DecisionSource::Local,
            confidence: 0.8, // Heuristic-based, slightly lower confidence
            timestamp_ms: decision.timestamp_ms,
        })
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

impl Default for LocalAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl OrchestrationAuthority for LocalAuthority {
    fn load_policies(&self, bundle: &[u8]) -> Result<(), String> {
        // Parse and compile the candidate before taking the write lock, and
        // swap only on success: an invalid reload leaves the active policy
        // untouched, and in-flight decisions finish on the engine they read.
        let mut candidate = DefaultPolicyEngine::new();
        candidate.load_policies(bundle.to_vec())?;
        *self.policy_write() = candidate;
        Ok(())
    }

    fn apply_policy(&self, request: &PolicyRequest) -> AuthorityDecision<PolicyOutcome> {
        // No stage context here, so the process-wide monitor supplies the
        // live snapshot (the injected provider still wins when present).
        let monitor = ResourceMonitor::global();
        let prepared = self.prepare_stage(
            &request.stage_id,
            &request.envelope,
            &request.metrics,
            &monitor,
        );
        Self::policy_decision(&prepared.policy)
    }

    fn resolve_target(&self, context: &StageContext) -> AuthorityDecision<ResolvedTarget> {
        self.resolve_target_with_feedback(context).decision
    }

    fn resolve_target_with_feedback(&self, context: &StageContext) -> TargetResolution {
        // Target-only callers have no request envelope; the input kind
        // carries the payload, so text rules still see the real text.
        let envelope = Envelope::new(context.input_kind.clone());
        let prepared = self.prepare_stage(
            &context.stage_id,
            &envelope,
            &context.metrics,
            &context.resource_monitor,
        );
        self.resolve_prepared(context, prepared)
    }

    fn resolve_stage(&self, request: &PolicyRequest, context: &StageContext) -> StageResolution {
        let prepared = self.prepare_stage(
            &request.stage_id,
            &request.envelope,
            &context.metrics,
            &context.resource_monitor,
        );
        let policy = Self::policy_decision(&prepared.policy);
        let target = self.resolve_prepared(context, prepared);
        StageResolution::new(policy, target)
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
            local_availability: None,
            device_class: None,
            device_class_schema_version: None,
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
            local_availability: None,
            device_class: None,
            device_class_schema_version: None,
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
            local_availability: None,
            device_class: None,
            device_class_schema_version: None,
        };

        let decision = authority.resolve_target(&context);
        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
    }

    #[test]
    fn caller_local_availability_overrides_cache_provider() {
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        let mut context = text_context();
        context.local_availability = Some(LocalAvailability::new(false));

        let decision = authority.resolve_target(&context);

        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
        assert!(decision.reason.contains("model_unavailable"));
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
    fn reliability_window_evicts_oldest_after_32_entries() {
        // Pin the exact retained FIFO sequence: writing failure-0..32 must
        // leave failure-1..32 in oldest-to-newest order. Asserting length
        // alone (or just the absence of failure-0) would silently accept
        // reversed eviction, duplicate retention, or stable-non-FIFO bugs.
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        for idx in 0..33 {
            authority.record_outcome(&ExecutionOutcome {
                stage_id: "test-stage".to_string(),
                target: ResolvedTarget::Device,
                latency_ms: 10,
                success: false,
                error: Some(format!("failure-{idx}")),
                category: Some(OutcomeCategory::HardFail {
                    reason: format!("failure-{idx}"),
                }),
                model_id: Some("test-model".to_string()),
                signal_context: Some(signal()),
            });
        }

        let history = authority.history_snapshot("test-model", signal());

        assert_eq!(history.len(), RELIABILITY_WINDOW);
        let actual_reasons: Vec<String> = history
            .iter()
            .map(|c| match c {
                OutcomeCategory::HardFail { reason } => reason.clone(),
                other => panic!("expected HardFail, got {other:?}"),
            })
            .collect();
        let expected_reasons: Vec<String> = (1..=RELIABILITY_WINDOW)
            .map(|i| format!("failure-{i}"))
            .collect();
        assert_eq!(
            actual_reasons, expected_reasons,
            "history must contain failure-1..failure-{RELIABILITY_WINDOW} in FIFO order (oldest first)"
        );
    }

    // Helper: record one HardFail under `(model, sig)` and return the
    // resulting snapshot. Keeps the per-dimension isolation tests compact.
    fn record_hard_fail_and_snapshot(
        authority: &LocalAuthority,
        model: &str,
        sig: SignalContext,
        reason: &str,
    ) -> std::collections::VecDeque<OutcomeCategory> {
        authority.record_outcome(&ExecutionOutcome {
            stage_id: "test-stage".to_string(),
            target: ResolvedTarget::Device,
            latency_ms: 10,
            success: false,
            error: Some(reason.to_string()),
            category: Some(OutcomeCategory::HardFail {
                reason: reason.to_string(),
            }),
            model_id: Some(model.to_string()),
            signal_context: Some(sig),
        });
        authority.history_snapshot(model, sig)
    }

    #[test]
    fn reliability_history_is_scoped_by_memory_pressure() {
        // Memory-pressure isolation: same (model, thermal, cpu_bucket) but
        // different memory_pressure must keep histories separate.
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        let mut warn_signal = signal();
        warn_signal.memory_pressure = MemoryPressure::Warn;
        let mut critical_signal = signal();
        critical_signal.memory_pressure = MemoryPressure::Critical;

        let warn_history =
            record_hard_fail_and_snapshot(&authority, "test-model", warn_signal, "warn-failure");
        let critical_history = record_hard_fail_and_snapshot(
            &authority,
            "test-model",
            critical_signal,
            "critical-failure",
        );

        assert_eq!(warn_history.len(), 1);
        assert_eq!(critical_history.len(), 1);
        assert_ne!(warn_history, critical_history);
    }

    #[test]
    fn reliability_history_is_scoped_by_thermal_state() {
        // Thermal-state isolation: same (model, memory_pressure, cpu_bucket)
        // but different thermal_state must keep histories separate. Without
        // this, a `Normal` device's hot history would leak into a `Hot`
        // device's bias decision.
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        let mut normal_signal = signal();
        normal_signal.thermal_state = ThermalState::Normal;
        let mut hot_signal = signal();
        hot_signal.thermal_state = ThermalState::Hot;

        let normal_history =
            record_hard_fail_and_snapshot(&authority, "test-model", normal_signal, "normal-fail");
        let hot_history =
            record_hard_fail_and_snapshot(&authority, "test-model", hot_signal, "hot-fail");

        assert_eq!(normal_history.len(), 1);
        assert_eq!(hot_history.len(), 1);
        assert_ne!(normal_history, hot_history);
    }

    #[test]
    fn reliability_history_is_scoped_by_cpu_bucket() {
        // cpu_bucket isolation: same (model, memory, thermal) but different
        // quantized CPU bucket must keep histories separate. The bucket is
        // the only continuous dimension in SignalContext, so a coarse
        // quantization regression would manifest here.
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));
        let mut low_cpu_signal = signal();
        low_cpu_signal.cpu_bucket = Some(2);
        let mut high_cpu_signal = signal();
        high_cpu_signal.cpu_bucket = Some(9);

        let low_history =
            record_hard_fail_and_snapshot(&authority, "test-model", low_cpu_signal, "low-cpu-fail");
        let high_history = record_hard_fail_and_snapshot(
            &authority,
            "test-model",
            high_cpu_signal,
            "high-cpu-fail",
        );

        assert_eq!(low_history.len(), 1);
        assert_eq!(high_history.len(), 1);
        assert_ne!(low_history, high_history);
    }

    #[test]
    fn reliability_history_is_scoped_by_model_id() {
        // model_id isolation: identical SignalContext but different model
        // IDs must keep histories separate. If the key ever collapsed to
        // SignalContext alone, this would conflate per-model reliability.
        let authority = LocalAuthority::with_cache_provider(Arc::new(CachedProvider));

        let history_a = record_hard_fail_and_snapshot(&authority, "model-a", signal(), "a-fail");
        let history_b = record_hard_fail_and_snapshot(&authority, "model-b", signal(), "b-fail");

        assert_eq!(history_a.len(), 1);
        assert_eq!(history_b.len(), 1);
        assert_ne!(history_a, history_b);
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

    // ── Phase 3: one evaluated decision, policy as an invariant ────────────

    /// Counts how many snapshots a decision takes.
    #[derive(Debug)]
    struct CountingProvider {
        snapshot: ResourceSnapshot,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl ResourceSnapshotProvider for CountingProvider {
        fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.snapshot
        }
    }

    fn text_request(text: &str) -> PolicyRequest {
        PolicyRequest {
            stage_id: "test-stage".to_string(),
            envelope: text_envelope(text),
            metrics: default_metrics(),
        }
    }

    fn cached_authority() -> LocalAuthority {
        LocalAuthority::with_cache_provider(Arc::new(CachedProvider))
    }

    #[test]
    fn load_policies_via_trait_denies_cloud() {
        let authority = cached_authority();
        authority
            .load_policies(deny_all_text_policy().as_bytes())
            .expect("deny policy loads");

        let decision = authority.resolve_target(&text_context());
        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(
            decision.reason.starts_with("policy_deny"),
            "{}",
            decision.reason
        );

        let resolution = authority.resolve_stage(&text_request("hello"), &text_context());
        assert!(matches!(
            resolution.policy.result,
            PolicyOutcome::Deny { .. }
        ));
        assert_eq!(resolution.target.decision.result, ResolvedTarget::Device);
        assert!(resolution.policy_requires_local());
    }

    #[test]
    fn load_policies_via_trait_prefers_cloud() {
        let authority = cached_authority();
        authority
            .load_policies(b"route_cloud_if:\n  - \"true\"\n")
            .expect("route_cloud policy loads");

        let decision = authority.resolve_target(&text_context());
        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
        assert!(
            decision.reason.starts_with("policy_route_cloud"),
            "{}",
            decision.reason
        );
        let resolution = authority.resolve_stage(&text_request("hello"), &text_context());
        assert_eq!(resolution.policy.result, PolicyOutcome::Allow);
        assert!(matches!(
            resolution.target.decision.result,
            ResolvedTarget::Cloud { .. }
        ));
    }

    #[test]
    fn load_policies_rejects_invalid_rule_and_keeps_previous() {
        let authority = cached_authority();
        authority
            .load_policies(deny_all_text_policy().as_bytes())
            .expect("deny policy loads");

        let err = authority
            .load_policies(b"deny_cloud_if:\n  - 'metrics.network_rtt > 300'\n")
            .expect_err("unknown operand must be rejected");
        assert!(err.contains("unknown operand"), "{err}");

        let decision = authority.resolve_target(&text_context());
        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(
            decision.reason.starts_with("policy_deny"),
            "{}",
            decision.reason
        );
        authority.with_policy_bundle(|bundle| {
            assert_eq!(bundle.unwrap().signature, "test-deny-all");
        });
    }

    #[test]
    fn apply_policy_overlays_live_metrics() {
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Critical;
        let authority =
            cached_authority().with_resource_provider(Arc::new(FixedResourceProvider(snapshot)));
        authority
            .load_policies(b"deny_cloud_if:\n  - 'metrics.memory_pressure == \"critical\"'\n")
            .expect("metrics policy loads");

        // The request carries default (unknown) metrics; the live overlay is
        // what the rule must see.
        let decision = authority.apply_policy(&text_request("hello"));
        assert!(
            matches!(decision.result, PolicyOutcome::Deny { .. }),
            "{decision:?}"
        );
    }

    #[test]
    fn policy_deny_overrides_explicit_cloud_and_server_targets() {
        let authority = cached_authority();
        authority
            .load_policies(deny_all_text_policy().as_bytes())
            .expect("deny policy loads");

        for explicit in [ExecutionTarget::Cloud, ExecutionTarget::Server] {
            let mut context = text_context();
            context.explicit_target = Some(explicit.clone());
            let resolution = authority.resolve_stage(&text_request("hello"), &context);
            assert_eq!(
                resolution.target.decision.result,
                ResolvedTarget::Device,
                "explicit {explicit:?} must not beat a policy denial"
            );
            assert!(
                resolution.target.decision.reason.starts_with("policy_deny"),
                "{}",
                resolution.target.decision.reason
            );
        }
    }

    #[test]
    fn policy_deny_beats_missing_local_model() {
        let authority = cached_authority();
        authority
            .load_policies(deny_all_text_policy().as_bytes())
            .expect("deny policy loads");
        let mut context = text_context();
        context.local_availability = Some(LocalAvailability::new(false));

        let decision = authority.resolve_target(&context);

        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(
            decision.reason.starts_with("policy_deny"),
            "{}",
            decision.reason
        );
        assert!(!decision.reason.contains("model_unavailable"));
    }

    #[test]
    fn policy_transform_overrides_hysteresis_history_and_explicit_cloud() {
        let authority = cached_authority().with_history_bias_k(1);
        authority
            .load_policies(
                b"rules:\n  - id: scrub\n    expression: 'input.kind == \"text\"'\n    action: redact\n",
            )
            .expect("redact policy loads");
        authority.record_abort_for_hysteresis_default_ttl("test-model", AbortReason::StressMemory);
        authority.record_outcome(&ExecutionOutcome {
            stage_id: "test-stage".to_string(),
            target: ResolvedTarget::Device,
            latency_ms: 10,
            success: false,
            error: Some("boom".to_string()),
            category: Some(OutcomeCategory::HardFail {
                reason: "boom".to_string(),
            }),
            model_id: Some("test-model".to_string()),
            signal_context: Some(signal()),
        });
        let mut context = text_context();
        context.explicit_target = Some(ExecutionTarget::Cloud);

        let resolution = authority.resolve_stage(&text_request("hello"), &context);

        assert!(matches!(
            resolution.policy.result,
            PolicyOutcome::Transform { .. }
        ));
        assert_eq!(resolution.target.decision.result, ResolvedTarget::Device);
        assert!(
            resolution
                .target
                .decision
                .reason
                .starts_with("policy_transform_unsupported"),
            "{}",
            resolution.target.decision.reason
        );
    }

    #[test]
    fn explicit_device_beats_policy_route_cloud() {
        let authority = cached_authority();
        authority
            .load_policies(b"route_cloud_if:\n  - \"true\"\n")
            .expect("route_cloud policy loads");
        let mut context = text_context();
        context.explicit_target = Some(ExecutionTarget::Device);

        let decision = authority.resolve_target(&context);

        assert_eq!(decision.result, ResolvedTarget::Device);
        assert!(decision.reason.contains("Explicit"), "{}", decision.reason);
    }

    #[test]
    fn resolve_stage_takes_one_snapshot_and_keeps_policy_and_target_consistent() {
        let provider = Arc::new(CountingProvider {
            snapshot: ResourceSnapshot::unknown(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let authority = cached_authority().with_resource_provider(provider.clone());
        authority
            .load_policies(deny_all_text_policy().as_bytes())
            .expect("deny policy loads");

        let resolution = authority.resolve_stage(&text_request("hello"), &text_context());

        assert_eq!(
            provider.calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a combined decision must sample the device exactly once"
        );
        assert!(matches!(
            resolution.policy.result,
            PolicyOutcome::Deny { .. }
        ));
        assert_eq!(resolution.target.decision.result, ResolvedTarget::Device);
        assert!(resolution.target.signal_context.is_some());
    }

    #[test]
    fn resolve_stage_evaluates_the_full_text_payload() {
        let authority = cached_authority();
        authority
            .load_policies(b"deny_cloud_if:\n  - 'input.text contains \"secret\"'\n")
            .expect("text policy loads");

        let denied = authority.resolve_stage(&text_request("my secret plan"), &text_context());
        assert!(matches!(denied.policy.result, PolicyOutcome::Deny { .. }));
        assert_eq!(denied.target.decision.result, ResolvedTarget::Device);

        let allowed = authority.resolve_stage(&text_request("public notes"), &text_context());
        assert_eq!(allowed.policy.result, PolicyOutcome::Allow);
    }
}

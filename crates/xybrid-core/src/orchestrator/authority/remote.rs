//! Remote Orchestration Authority - Delegates to xybrid backend.
//!
//! This authority calls the xybrid backend for smarter decisions based on
//! fleet-wide data and learned patterns.
//!
//! ## For v0.1.0
//!
//! Target routing advice is implemented via `GET /v1/routing/advice` with a
//! short TTL cache. Policy and model-selection endpoints still fall back to
//! `LocalAuthority` until the platform exposes those APIs.
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
use crate::context::DeviceMetrics;
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;
use url::Url;

const TARGET_ADVICE_CACHE_CAPACITY: usize = 256;

/// Remote orchestration authority - delegates to xybrid backend.
///
/// This authority calls the xybrid backend for smarter decisions
/// based on fleet-wide data and learned patterns.
///
/// ## Backend Integration
///
/// Target resolution calls the platform routing advice endpoint and falls back
/// to `LocalAuthority` when the backend is unavailable.
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
    endpoint: String,
    /// Optional platform API key for protected advice endpoints.
    api_key: Option<String>,
    /// Fallback to local authority when remote is unavailable.
    fallback: LocalAuthority,
    /// Successful target advice cache. Keeps remote routing resilient when the
    /// same hot path resolves repeatedly during a short session.
    target_cache: Mutex<TargetAdviceCache>,
}

#[derive(Debug, Clone)]
struct CachedTargetAdvice {
    decision: AuthorityDecision<ResolvedTarget>,
    expires_at_ms: u64,
}

#[derive(Debug, Default)]
struct TargetAdviceCache {
    entries: HashMap<String, CachedTargetAdvice>,
    lru: VecDeque<String>,
}

impl TargetAdviceCache {
    fn insert(&mut self, key: String, advice: CachedTargetAdvice, now_ms: u64) {
        self.entries.insert(key.clone(), advice);
        self.promote(&key);
        self.evict_expired(now_ms);
        self.evict_over_capacity();
    }

    fn get_fresh(&mut self, key: &str, now_ms: u64) -> Option<AuthorityDecision<ResolvedTarget>> {
        let cached = self.entries.get(key).cloned()?;
        if cached.expires_at_ms <= now_ms {
            self.remove(key);
            return None;
        }

        self.promote(key);
        let mut decision = cached.decision;
        decision.source = DecisionSource::Cached;
        decision.timestamp_ms = now_ms;
        Some(decision)
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    fn promote(&mut self, key: &str) {
        self.lru.retain(|candidate| candidate != key);
        self.lru.push_back(key.to_string());
    }

    fn remove(&mut self, key: &str) {
        self.entries.remove(key);
        self.lru.retain(|candidate| candidate != key);
    }

    fn evict_expired(&mut self, now_ms: u64) {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, advice)| advice.expires_at_ms <= now_ms)
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            self.remove(&key);
        }
    }

    fn evict_over_capacity(&mut self) {
        while self.entries.len() > TARGET_ADVICE_CACHE_CAPACITY {
            let Some(key) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&key);
        }
    }
}

#[derive(Debug, Deserialize)]
struct TargetAdviceResponse {
    target: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    ttl_ms: Option<u64>,
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
            api_key: None,
            fallback: LocalAuthority::new(),
            target_cache: Mutex::new(TargetAdviceCache::default()),
        }
    }

    /// Create a RemoteAuthority with a custom fallback authority.
    pub fn with_fallback(endpoint: &str, fallback: LocalAuthority) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            api_key: None,
            fallback,
            target_cache: Mutex::new(TargetAdviceCache::default()),
        }
    }

    /// Configure an API key for protected platform endpoints.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Get the backend endpoint URL.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn target_cache_guard(&self) -> MutexGuard<'_, TargetAdviceCache> {
        self.target_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn battery_cache_bucket(battery_pct: u8) -> String {
        let bucket = (battery_pct.min(100) / 10) * 10;
        bucket.to_string()
    }

    fn cpu_cache_bucket(cpu_pct: Option<f32>) -> String {
        let Some(cpu_pct) = cpu_pct else {
            return "unknown".to_string();
        };
        if !cpu_pct.is_finite() {
            return "unknown".to_string();
        }
        let bucket = ((cpu_pct.clamp(0.0, 100.0) / 10.0).floor() * 10.0) as u8;
        bucket.to_string()
    }

    fn target_cache_key(context: &StageContext, metrics: &DeviceMetrics) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            context.stage_id,
            context.model_id,
            context.input_kind.as_str(),
            context.device_class.as_deref().unwrap_or("unknown-device"),
            if context.device_class.is_some() {
                context.device_class_schema_version.unwrap_or(1).to_string()
            } else {
                "unknown-schema".to_string()
            },
            Self::battery_cache_bucket(metrics.capabilities.battery_level),
            metrics.capabilities.thermal_state.as_str(),
            metrics.resource.memory_pressure.as_str(),
            Self::cpu_cache_bucket(metrics.resource.cpu_pct),
            context
                .explicit_target
                .as_ref()
                .map(|target| target.to_string())
                .unwrap_or_else(|| "auto".to_string())
        )
    }

    fn target_advice_url(&self, context: &StageContext, metrics: &DeviceMetrics) -> Option<String> {
        let mut url = Url::parse(&self.endpoint)
            .ok()?
            .join("/v1/routing/advice")
            .ok()?;
        {
            let mut qp = url.query_pairs_mut();
            qp.append_pair("stage_id", &context.stage_id);
            qp.append_pair("model_id", &context.model_id);
            qp.append_pair("input_kind", context.input_kind.as_str());
            if let Some(device_class) = context.device_class.as_deref() {
                qp.append_pair("device_class", device_class);
                let schema_version = context.device_class_schema_version.unwrap_or(1);
                qp.append_pair("device_class_schema_version", &schema_version.to_string());
            }
            qp.append_pair(
                "battery_pct",
                &metrics.capabilities.battery_level.to_string(),
            );
            qp.append_pair("thermal_state", metrics.capabilities.thermal_state.as_str());
            qp.append_pair("memory_pressure", metrics.resource.memory_pressure.as_str());
            if let Some(cpu_pct) = metrics.resource.cpu_pct {
                qp.append_pair("cpu_pct", &format!("{cpu_pct:.2}"));
            }
            if let Some(explicit_target) = &context.explicit_target {
                qp.append_pair("explicit_target", &explicit_target.to_string());
            }
        }
        Some(url.to_string())
    }

    fn fetch_target_advice(
        &self,
        context: &StageContext,
        metrics: &DeviceMetrics,
    ) -> Option<AuthorityDecision<ResolvedTarget>> {
        let url = self.target_advice_url(context, metrics)?;
        let mut request = ureq::get(&url).timeout(Duration::from_millis(750));
        let auth_header = self
            .api_key
            .as_ref()
            .map(|api_key| format!("Bearer {api_key}"));
        if let Some(auth_header) = &auth_header {
            request = request.set("Authorization", auth_header);
        }
        let response = request.call().ok()?;

        if response.status() != 200 {
            return None;
        }

        let advice: TargetAdviceResponse = response.into_json().ok()?;
        let result = match advice.target.as_str() {
            "device" | "local" => ResolvedTarget::Device,
            "cloud" => ResolvedTarget::Cloud {
                provider: advice.provider.unwrap_or_else(|| "xybrid".to_string()),
            },
            "server" => ResolvedTarget::Server {
                endpoint: advice.endpoint?,
            },
            _ => return None,
        };

        let confidence = advice.confidence.unwrap_or(0.7).clamp(0.0, 1.0);
        let reason = advice
            .reason
            .unwrap_or_else(|| "Remote routing advice".to_string());
        let decision = AuthorityDecision::new(result, reason, DecisionSource::Remote, confidence);
        let ttl_ms = advice.ttl_ms.unwrap_or(30_000);
        let key = Self::target_cache_key(context, metrics);
        let expires_at_ms = decision.timestamp_ms.saturating_add(ttl_ms);

        if ttl_ms > 0 {
            self.target_cache_guard().insert(
                key,
                CachedTargetAdvice {
                    decision: decision.clone(),
                    expires_at_ms,
                },
                decision.timestamp_ms,
            );
        }

        Some(decision)
    }

    fn cached_target_advice(
        &self,
        context: &StageContext,
        metrics: &DeviceMetrics,
    ) -> Option<AuthorityDecision<ResolvedTarget>> {
        let key = Self::target_cache_key(context, metrics);
        let now = now_ms();
        self.target_cache_guard().get_fresh(&key, now)
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
        self.resolve_target_with_feedback(context).decision
    }

    fn resolve_target_with_feedback(&self, context: &StageContext) -> TargetResolution {
        // Mirror LocalAuthority's live overlay so the SignalContext attached
        // to a TargetResolution reflects the same real-time resource state
        // the routing decision is implicitly conditioned on. Without this
        // overlay, ExecutionOutcome.signal_context is bucketed under stale
        // pre-run device metrics, and the embedded LocalAuthority's
        // reliability history grows under buckets that no live request ever
        // queries — silently disabling the history-bias circuit breaker.
        let snapshot = context
            .resource_monitor
            .current_snapshot(Duration::from_millis(500));
        let live_metrics = context.metrics.with_live_snapshot(snapshot);
        let signal = Some(SignalContext::from_metrics(&live_metrics));
        let live_context = StageContext {
            metrics: live_metrics.clone(),
            ..context.clone()
        };

        if let Some(decision) = self.cached_target_advice(&live_context, &live_metrics) {
            return TargetResolution::new(decision, context.model_id.clone(), signal);
        }

        if let Some(decision) = self.fetch_target_advice(&live_context, &live_metrics) {
            return TargetResolution::new(decision, context.model_id.clone(), signal);
        }

        let mut resolution = self.fallback.resolve_target_with_feedback(&live_context);
        resolution.decision.source = DecisionSource::Default;
        resolution.decision.reason = format!(
            "Fallback to local (remote unavailable): {}",
            resolution.decision.reason
        );
        resolution
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

        self.fallback.record_outcome(outcome);
    }

    fn invalidate_cache(&self) {
        self.target_cache_guard().clear();
    }

    fn name(&self) -> &str {
        "remote"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DeviceMetrics;
    use crate::device::ResourceMonitor;
    use crate::ir::{Envelope, EnvelopeKind};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    fn default_metrics() -> DeviceMetrics {
        DeviceMetrics::default()
    }

    fn text_envelope(text: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    }

    fn default_context(endpoint_stage: &str) -> StageContext {
        StageContext {
            stage_id: endpoint_stage.to_string(),
            model_id: "test-model".to_string(),
            input_kind: EnvelopeKind::Text("test".to_string()),
            metrics: default_metrics(),
            resource_monitor: ResourceMonitor::global(),
            explicit_target: None,
            device_class: None,
            device_class_schema_version: None,
        }
    }

    fn context_with_device_class(endpoint_stage: &str, device_class: &str) -> StageContext {
        StageContext {
            device_class: Some(device_class.to_string()),
            device_class_schema_version: Some(1),
            ..default_context(endpoint_stage)
        }
    }

    fn spawn_advice_server(body: &'static str, max_requests: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind advice server");
        let addr = listener.local_addr().expect("local addr");
        thread::spawn(move || {
            for _ in 0..max_requests {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut buf = [0_u8; 2048];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}", addr)
    }

    fn spawn_header_capture_advice_server(body: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind advice server");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0_u8; 2048];
            let read = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..read]).to_string();
            let _ = tx.send(request);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
        });
        (format!("http://{}", addr), rx)
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
        let context = default_context("test");

        let decision = authority.resolve_target(&context);
        assert_eq!(decision.source, DecisionSource::Default);
        assert!(decision.reason.contains("Fallback"));
    }

    #[test]
    fn test_remote_authority_uses_backend_routing_advice() {
        let endpoint = spawn_advice_server(
            r#"{"target":"cloud","provider":"openai","reason":"fleet prefers cloud","confidence":0.91,"ttl_ms":30000}"#,
            1,
        );
        let authority = RemoteAuthority::new(&endpoint);
        let decision = authority.resolve_target(&default_context("test"));

        assert_eq!(decision.source, DecisionSource::Remote);
        assert_eq!(
            decision.result,
            ResolvedTarget::Cloud {
                provider: "openai".to_string()
            }
        );
        assert_eq!(decision.reason, "fleet prefers cloud");
        assert!((decision.confidence - 0.91).abs() < f32::EPSILON);
    }

    #[test]
    fn test_remote_authority_caches_routing_advice() {
        let endpoint = spawn_advice_server(
            r#"{"target":"device","reason":"warm local path","confidence":0.8,"ttl_ms":30000}"#,
            1,
        );
        let authority = RemoteAuthority::new(&endpoint);
        let context = default_context("cached");

        let first = authority.resolve_target(&context);
        let second = authority.resolve_target(&context);

        assert_eq!(first.source, DecisionSource::Remote);
        assert_eq!(second.source, DecisionSource::Cached);
        assert_eq!(second.result, ResolvedTarget::Device);
    }

    #[test]
    fn test_remote_authority_sends_authorization_header_when_configured() {
        let (endpoint, request_rx) = spawn_header_capture_advice_server(
            r#"{"target":"cloud","provider":"xybrid","reason":"authorized advice","confidence":0.8,"ttl_ms":0}"#,
        );
        let authority = RemoteAuthority::new(&endpoint).with_api_key("sk_test_routing");

        let decision = authority.resolve_target(&default_context("auth"));
        let request = request_rx.recv().expect("captured request");
        let request_lower = request.to_ascii_lowercase();

        assert_eq!(decision.source, DecisionSource::Remote);
        assert!(
            request_lower.contains("authorization: bearer sk_test_routing"),
            "request should carry bearer auth header, got: {request}"
        );
    }

    #[test]
    fn target_advice_url_includes_device_class_and_schema_when_available() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let context = context_with_device_class("classed", "iphone-15-pro");
        let url = authority
            .target_advice_url(&context, &context.metrics)
            .expect("routing advice url");

        assert!(url.contains("device_class=iphone-15-pro"), "{url}");
        assert!(url.contains("device_class_schema_version=1"), "{url}");
    }

    #[test]
    fn target_advice_url_omits_device_class_when_unavailable() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let context = default_context("unclassed");
        let url = authority
            .target_advice_url(&context, &context.metrics)
            .expect("routing advice url");

        assert!(!url.contains("device_class="), "{url}");
        assert!(!url.contains("device_class_schema_version="), "{url}");
    }

    #[test]
    fn target_advice_url_uses_live_metrics_argument() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let context = default_context("live");
        let mut live_metrics = context.metrics.clone();
        live_metrics.capabilities.battery_level = 7;
        live_metrics.capabilities.thermal_state = crate::device::ThermalState::Hot;
        live_metrics.resource.memory_pressure = crate::device::MemoryPressure::Critical;
        live_metrics.resource.cpu_pct = Some(98.4);

        let url = authority
            .target_advice_url(&context, &live_metrics)
            .expect("routing advice url");

        assert!(url.contains("battery_pct=7"), "{url}");
        assert!(url.contains("thermal_state=hot"), "{url}");
        assert!(url.contains("memory_pressure=critical"), "{url}");
        assert!(url.contains("cpu_pct=98.40"), "{url}");
    }

    #[test]
    fn target_cache_key_differs_by_device_class_and_schema() {
        let a = context_with_device_class("cached-class", "iphone-15-pro");
        let b = context_with_device_class("cached-class", "iphone-15");
        let mut c = context_with_device_class("cached-class", "iphone-15-pro");
        c.device_class_schema_version = Some(2);

        let key_a = RemoteAuthority::target_cache_key(&a, &a.metrics);
        let key_b = RemoteAuthority::target_cache_key(&b, &b.metrics);
        let key_c = RemoteAuthority::target_cache_key(&c, &c.metrics);

        assert_ne!(key_a, key_b);
        assert_ne!(key_a, key_c);
    }

    #[test]
    fn target_cache_key_differs_by_live_resource_fields() {
        let context = context_with_device_class("cached-resource", "iphone-15-pro");
        let mut baseline = context.metrics.clone();
        baseline.capabilities.battery_level = 80;
        baseline.capabilities.thermal_state = crate::device::ThermalState::Normal;
        baseline.resource.memory_pressure = crate::device::MemoryPressure::Normal;
        baseline.resource.cpu_pct = Some(20.0);

        let mut low_battery = baseline.clone();
        low_battery.capabilities.battery_level = 7;

        let mut hot_thermal = baseline.clone();
        hot_thermal.capabilities.thermal_state = crate::device::ThermalState::Hot;

        let mut critical_memory = baseline.clone();
        critical_memory.resource.memory_pressure = crate::device::MemoryPressure::Critical;

        let mut high_cpu = baseline.clone();
        high_cpu.resource.cpu_pct = Some(98.4);

        let key = RemoteAuthority::target_cache_key(&context, &baseline);

        assert_ne!(
            key,
            RemoteAuthority::target_cache_key(&context, &low_battery)
        );
        assert_ne!(
            key,
            RemoteAuthority::target_cache_key(&context, &hot_thermal)
        );
        assert_ne!(
            key,
            RemoteAuthority::target_cache_key(&context, &critical_memory)
        );
        assert_ne!(key, RemoteAuthority::target_cache_key(&context, &high_cpu));
    }

    #[test]
    fn target_cache_key_buckets_high_cardinality_percentages() {
        let context = context_with_device_class("bucketed-resource", "iphone-15-pro");
        let mut baseline = context.metrics.clone();
        baseline.capabilities.battery_level = 83;
        baseline.resource.cpu_pct = Some(24.2);

        let mut same_bucket = baseline.clone();
        same_bucket.capabilities.battery_level = 89;
        same_bucket.resource.cpu_pct = Some(29.9);

        let mut different_bucket = baseline.clone();
        different_bucket.capabilities.battery_level = 72;
        different_bucket.resource.cpu_pct = Some(31.0);

        assert_eq!(
            RemoteAuthority::target_cache_key(&context, &baseline),
            RemoteAuthority::target_cache_key(&context, &same_bucket)
        );
        assert_ne!(
            RemoteAuthority::target_cache_key(&context, &baseline),
            RemoteAuthority::target_cache_key(&context, &different_bucket)
        );
    }

    #[test]
    fn target_advice_cache_evicts_least_recent_entries() {
        let mut cache = TargetAdviceCache::default();
        for index in 0..(TARGET_ADVICE_CACHE_CAPACITY + 8) {
            cache.insert(
                format!("key-{index}"),
                CachedTargetAdvice {
                    decision: AuthorityDecision::new(
                        ResolvedTarget::Device,
                        "cached".to_string(),
                        DecisionSource::Remote,
                        0.8,
                    ),
                    expires_at_ms: u64::MAX,
                },
                1,
            );
        }

        assert_eq!(cache.entries.len(), TARGET_ADVICE_CACHE_CAPACITY);
        assert!(!cache.entries.contains_key("key-0"));
        assert!(cache
            .entries
            .contains_key(&format!("key-{}", TARGET_ADVICE_CACHE_CAPACITY + 7)));
    }

    #[test]
    fn remote_authority_recovers_poisoned_target_cache_lock() {
        let authority = RemoteAuthority::new("https://api.xybrid.dev");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = authority.target_cache.lock().expect("cache lock");
            panic!("poison target cache");
        }));
        assert!(result.is_err());

        authority.invalidate_cache();
        assert!(authority.target_cache_guard().entries.is_empty());
    }

    #[test]
    fn target_cache_key_differs_by_explicit_target() {
        let auto_context = context_with_device_class("cached-explicit", "iphone-15-pro");
        let mut cloud_context = auto_context.clone();
        cloud_context.explicit_target = Some(crate::pipeline::ExecutionTarget::Cloud);

        let auto_key = RemoteAuthority::target_cache_key(&auto_context, &auto_context.metrics);
        let cloud_key = RemoteAuthority::target_cache_key(&cloud_context, &cloud_context.metrics);

        assert_ne!(auto_key, cloud_key);
    }

    #[test]
    fn test_remote_authority_invalidate_cache_clears_cached_advice() {
        let endpoint = spawn_advice_server(
            r#"{"target":"device","reason":"warm local path","confidence":0.8,"ttl_ms":30000}"#,
            1,
        );
        let authority = RemoteAuthority::new(&endpoint);
        let context = default_context("invalidate");

        let first = authority.resolve_target(&context);
        authority.invalidate_cache();
        let second = authority.resolve_target(&context);

        assert_eq!(first.source, DecisionSource::Remote);
        assert_eq!(second.source, DecisionSource::Default);
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
            category: None,
            model_id: None,
            signal_context: None,
        };

        // Should not panic
        authority.record_outcome(&outcome);
    }
}

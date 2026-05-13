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
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
use url::Url;

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
    target_cache: Mutex<HashMap<String, CachedTargetAdvice>>,
}

#[derive(Debug, Clone)]
struct CachedTargetAdvice {
    decision: AuthorityDecision<ResolvedTarget>,
    expires_at_ms: u64,
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
            target_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Create a RemoteAuthority with a custom fallback authority.
    pub fn with_fallback(endpoint: &str, fallback: LocalAuthority) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            api_key: None,
            fallback,
            target_cache: Mutex::new(HashMap::new()),
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

    fn target_cache_key(context: &StageContext) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}",
            context.stage_id,
            context.model_id,
            context.input_kind.as_str(),
            context.metrics.capabilities.battery_level,
            context.metrics.capabilities.thermal_state.as_str(),
            context.metrics.resource.memory_pressure.as_str(),
            context
                .metrics
                .resource
                .cpu_pct
                .map(|cpu| format!("{cpu:.0}"))
                .unwrap_or_else(|| "unknown".to_string())
        )
    }

    fn target_advice_url(&self, context: &StageContext) -> Option<String> {
        let mut url = Url::parse(&self.endpoint)
            .ok()?
            .join("/v1/routing/advice")
            .ok()?;
        {
            let mut qp = url.query_pairs_mut();
            qp.append_pair("stage_id", &context.stage_id);
            qp.append_pair("model_id", &context.model_id);
            qp.append_pair("input_kind", context.input_kind.as_str());
            qp.append_pair(
                "battery_pct",
                &context.metrics.capabilities.battery_level.to_string(),
            );
            qp.append_pair(
                "thermal_state",
                context.metrics.capabilities.thermal_state.as_str(),
            );
            qp.append_pair(
                "memory_pressure",
                context.metrics.resource.memory_pressure.as_str(),
            );
            if let Some(cpu_pct) = context.metrics.resource.cpu_pct {
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
    ) -> Option<AuthorityDecision<ResolvedTarget>> {
        let url = self.target_advice_url(context)?;
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
        let key = Self::target_cache_key(context);
        let expires_at_ms = decision.timestamp_ms.saturating_add(ttl_ms);

        if ttl_ms > 0 {
            if let Ok(mut cache) = self.target_cache.lock() {
                cache.insert(
                    key,
                    CachedTargetAdvice {
                        decision: decision.clone(),
                        expires_at_ms,
                    },
                );
            }
        }

        Some(decision)
    }

    fn cached_target_advice(
        &self,
        context: &StageContext,
    ) -> Option<AuthorityDecision<ResolvedTarget>> {
        let key = Self::target_cache_key(context);
        let now = now_ms();
        let mut cache = self.target_cache.lock().ok()?;
        let cached = cache.get(&key)?;
        if cached.expires_at_ms <= now {
            cache.remove(&key);
            return None;
        }

        let mut decision = cached.decision.clone();
        decision.source = DecisionSource::Cached;
        decision.timestamp_ms = now;
        Some(decision)
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

        if let Some(decision) = self.cached_target_advice(context) {
            return TargetResolution::new(decision, context.model_id.clone(), signal);
        }

        if let Some(decision) = self.fetch_target_advice(context) {
            return TargetResolution::new(decision, context.model_id.clone(), signal);
        }

        let mut resolution = self.fallback.resolve_target_with_feedback(context);
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
        if let Ok(mut cache) = self.target_cache.lock() {
            cache.clear();
        }
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

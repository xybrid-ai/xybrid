//! Domain telemetry events.
//!
//! In-process subscribers (Flutter's event stream, host apps, the CLI) receive
//! [`TelemetryEvent`]s describing model completions, pipeline stages, routing
//! decisions, cloud fallback, and downloads. This stream never leaves the
//! process: export to an observability backend is handled by
//! [`crate::observability`] (OpenTelemetry), which is configured separately and
//! explicitly.
//!
//! # Orchestrator Bridge Context
//!
//! Producers attach context at orchestrator event publish time; bridge threads
//! must not assume task-local or thread-local state is available when they
//! drain events later.

use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
pub use xybrid_core::device::DeviceProfile;
use xybrid_core::device::{ResourceMonitor, ResourceTelemetryMode, ResourceUsageSummary};
use xybrid_core::event_bus::{EventContext, OrchestratorEvent};
use xybrid_core::execution::listener::{self as execution_listener, ExecutionEvent};
use xybrid_core::orchestrator::routing_engine::LocalReliabilityHint;

/// Telemetry event type (simplified for FFI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Event type name
    pub event_type: String,
    /// Stage name (if applicable)
    pub stage_name: Option<String>,
    /// Target (local/cloud/fallback)
    pub target: Option<String>,
    /// Latency in milliseconds (if applicable)
    pub latency_ms: Option<u32>,
    /// Error message (if applicable)
    pub error: Option<String>,
    /// Additional event data as JSON string
    pub data: Option<String>,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
}

/// Global telemetry event channel for streaming
pub type TelemetrySender = mpsc::Sender<TelemetryEvent>;

static TELEMETRY_SENDERS: Mutex<Vec<TelemetrySender>> = Mutex::new(Vec::new());

const CONTEXT_PIPELINE_ID_KEY: &str = "__xybrid_pipeline_id";
const CONTEXT_TRACE_ID_KEY: &str = "__xybrid_trace_id";

/// Read the `XYBRID_RESOURCE_TELEMETRY` env var, if set. Recognized values:
/// `off`, `boundary`, `summary`, `summary:<ms>`, `debug_local`,
/// `debug_local:<ms>`. Unknown values fall back to `None` so the caller can
/// use their configured default.
fn resource_mode_from_env() -> Option<ResourceTelemetryMode> {
    let raw = std::env::var("XYBRID_RESOURCE_TELEMETRY").ok()?;
    parse_resource_telemetry_mode(&raw)
}

fn parse_resource_telemetry_mode(raw: &str) -> Option<ResourceTelemetryMode> {
    let lower = raw.trim().to_ascii_lowercase();
    let (head, interval) = match lower.split_once(':') {
        Some((h, t)) => (h, t.parse::<u32>().ok()),
        None => (lower.as_str(), None),
    };
    let default_interval = ResourceTelemetryMode::DEFAULT_SUMMARY_INTERVAL_MS;
    let mode = match head {
        "off" => ResourceTelemetryMode::Off,
        "boundary" => ResourceTelemetryMode::Boundary,
        "summary" => ResourceTelemetryMode::Summary {
            interval_ms: interval.unwrap_or(default_interval),
        },
        "debug_local" | "debuglocal" | "debug-local" => ResourceTelemetryMode::DebugLocal {
            interval_ms: interval.unwrap_or(default_interval),
        },
        _ => return None,
    };
    Some(mode.normalized())
}

// -- Process-wide resource-telemetry mode ----------------------------------
//
// SDK operations that select no observability session (or a borrowed
// observer) sample device resources with this mode; an operation pinned to a
// configured session uses that session's mode instead. Written by
// `XybridInit::resource_telemetry` and by installing a default session (its
// configured mode); reads are cheap.
static RESOURCE_TELEMETRY_MODE: RwLock<ResourceTelemetryMode> =
    RwLock::new(ResourceTelemetryMode::Off);

pub(crate) fn set_resource_telemetry_mode(mode: ResourceTelemetryMode) {
    if let Ok(mut guard) = RESOURCE_TELEMETRY_MODE.write() {
        *guard = mode;
    }
}

/// Apply the `XYBRID_RESOURCE_TELEMETRY` env override on top of a configured
/// mode. The env var wins unconditionally when set (and parses to a known
/// variant) so operators can disable resource telemetry at runtime without a
/// rebuild.
fn resolve_resource_telemetry_mode(configured: ResourceTelemetryMode) -> ResourceTelemetryMode {
    match resource_mode_from_env() {
        Some(env_mode) => env_mode,
        None => configured,
    }
}

/// Resolve a configured mode against the env override and pre-warm the global
/// `ResourceMonitor` when it samples, so the first run doesn't pay the cold
/// read.
pub(crate) fn prepare_resource_telemetry(mode: ResourceTelemetryMode) -> ResourceTelemetryMode {
    let mode = resolve_resource_telemetry_mode(mode);
    if !mode.is_off() {
        ResourceMonitor::global().prewarm();
    }
    mode
}

/// Publish `mode` (after the env override) as the process-wide mode. `Off`
/// still stores the mode so a later rehydration can read a consistent value;
/// the monitor just sits idle until something flips it on.
pub(crate) fn activate_resource_telemetry(mode: ResourceTelemetryMode) {
    set_resource_telemetry_mode(prepare_resource_telemetry(mode));
}

/// Process-wide resource-telemetry mode: the one SDK operations that select
/// no observability session sample with. `Off` until set by
/// `XybridInit::resource_telemetry` or by installing a default session
/// configured with another mode. An operation pinned to a session
/// (`ObservationOptions::with_session`) uses that session's mode instead.
pub fn resource_telemetry_mode() -> ResourceTelemetryMode {
    match RESOURCE_TELEMETRY_MODE.read() {
        Ok(g) => *g,
        Err(poisoned) => *poisoned.into_inner(),
    }
}

/// Begin a resource-telemetry scope. Returns a guard that produces the
/// summary on `finish()`. Callers who want the summary attached to their
/// outgoing `TelemetryEvent` can use [`attach_resource_summary`] at emit time.
///
/// On a thread running an SDK operation (for example in a streaming callback,
/// or under a [`CapturedScope`](crate::observability::CapturedScope)) it
/// samples with that operation's mode, which follows the session it is pinned
/// to. Elsewhere it uses the process-wide [`resource_telemetry_mode`].
pub fn begin_resource_run() -> xybrid_core::device::RunGuard {
    ResourceMonitor::global().begin_run(crate::observability::current_resource_telemetry())
}

/// Mutate a `TelemetryEvent.data` JSON string to add `resource_summary`.
/// If `data` already parses as an object, the field is inserted alongside
/// existing keys; otherwise the event is left untouched so we never clobber
/// a publisher's bespoke payload. Safe to call with `None` summary (no-op).
pub fn attach_resource_summary(event: &mut TelemetryEvent, summary: Option<ResourceUsageSummary>) {
    let Some(summary) = summary else {
        return;
    };
    let Ok(summary_json) = serde_json::to_value(&summary) else {
        return;
    };
    let mut parsed: serde_json::Value = match event.data.as_deref() {
        Some(s) if !s.is_empty() => match serde_json::from_str(s) {
            Ok(v) => v,
            Err(_) => return,
        },
        _ => serde_json::json!({}),
    };
    if let Some(obj) = parsed.as_object_mut() {
        obj.insert("resource_summary".to_string(), summary_json);
        event.data = serde_json::to_string(&parsed).ok();
    }
}

/// Attach the resource summary from `guard`, publish the event, and return
/// the summary (so the run can also record it on its span). Run-site
/// shorthand for callers that always want the two steps bundled:
///
/// ```text
/// let guard = begin_resource_run();
/// // ... run inference ...
/// publish_with_resource_summary(event, guard);
/// ```
///
/// Safe to call with a disabled guard; `attach_resource_summary` becomes a
/// no-op and the publish path is unchanged.
pub fn publish_with_resource_summary(
    mut event: TelemetryEvent,
    guard: xybrid_core::device::RunGuard,
) -> Option<ResourceUsageSummary> {
    let summary = guard.finish();
    attach_resource_summary(&mut event, summary.clone());
    publish_telemetry_event(event);
    summary
}

/// Build a `LocalAborted` telemetry event for a resource-driven cloud-fallback
/// abort.
///
/// The event carries the abort reason, the local leg's latency and partial
/// token count, and a `correlation_id` shared with the paired `CloudRetry`
/// event so analytics joins on the two halves of one logical inference.
/// `correlation_id`, `outcome_category`, and `abort_reason` ride in
/// `event.data` for in-process subscribers. Observability exports are
/// recorded separately on OpenTelemetry spans.
pub fn local_aborted_event(
    correlation_id: &str,
    model_id: &str,
    abort_reason: xybrid_core::abort::AbortReason,
    latency_ms: u32,
    tokens_emitted: u32,
) -> TelemetryEvent {
    local_aborted_event_with_details(
        correlation_id,
        model_id,
        abort_reason,
        latency_ms,
        tokens_emitted,
        None,
        None,
    )
}

/// Build a `LocalAborted` event with resource and reliability context.
///
/// This is the cloud-fallback telemetry shape used by customer-ready routing:
/// analytics can join the local/cloud legs by `correlation_id`, inspect the
/// live resource summary that caused the abort, and feed the local reliability
/// hint back into fleet-aware routing decisions.
pub fn local_aborted_event_with_details(
    correlation_id: &str,
    model_id: &str,
    abort_reason: xybrid_core::abort::AbortReason,
    latency_ms: u32,
    tokens_emitted: u32,
    resource_summary: Option<ResourceUsageSummary>,
    local_reliability_hint: Option<LocalReliabilityHint>,
) -> TelemetryEvent {
    let mut data = serde_json::json!({
        "model_id": model_id,
        "correlation_id": correlation_id,
        "outcome_category": {
            "kind": "aborted_for_cloud_fallback",
            "reason": abort_reason.as_str(),
        },
        "abort_reason": abort_reason.as_str(),
        "tokens_emitted": tokens_emitted,
    });

    if let Some(obj) = data.as_object_mut() {
        if let Some(summary) =
            resource_summary.and_then(|summary| serde_json::to_value(summary).ok())
        {
            obj.insert("resource_summary".to_string(), summary);
        }
        if let Some(hint) = local_reliability_hint {
            obj.insert(
                "local_reliability_hint".to_string(),
                serde_json::json!({
                    "recent_abort_rate": hint.recent_abort_rate,
                    "sample_size": hint.sample_size,
                }),
            );
        }
    }

    TelemetryEvent {
        event_type: "LocalAborted".to_string(),
        stage_name: Some(model_id.to_string()),
        target: Some("local".to_string()),
        latency_ms: Some(latency_ms),
        error: None,
        data: Some(data.to_string()),
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    }
}

/// Convenience wrapper: build and publish a `LocalAborted` event.
pub fn publish_local_aborted(
    correlation_id: &str,
    model_id: &str,
    abort_reason: xybrid_core::abort::AbortReason,
    latency_ms: u32,
    tokens_emitted: u32,
) {
    publish_telemetry_event(local_aborted_event(
        correlation_id,
        model_id,
        abort_reason,
        latency_ms,
        tokens_emitted,
    ));
}

pub(crate) fn publish_local_aborted_with_details(
    correlation_id: &str,
    model_id: &str,
    abort_reason: xybrid_core::abort::AbortReason,
    latency_ms: u32,
    tokens_emitted: u32,
    resource_summary: Option<ResourceUsageSummary>,
    local_reliability_hint: Option<LocalReliabilityHint>,
) {
    publish_telemetry_event(local_aborted_event_with_details(
        correlation_id,
        model_id,
        abort_reason,
        latency_ms,
        tokens_emitted,
        resource_summary,
        local_reliability_hint,
    ));
}

pub(crate) fn redact_error_for_telemetry(message: &str) -> String {
    let redacted = [
        "Authorization: Bearer ",
        "authorization: Bearer ",
        "Bearer ",
        "api_key=",
        "api-key=",
        "x-api-key: ",
        "X-API-Key: ",
    ]
    .iter()
    .fold(message.to_string(), |current, marker| {
        redact_value_after_marker(&current, marker)
    });

    redact_secret_like_tokens(&redacted)
}

fn redact_value_after_marker(input: &str, marker: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(index) = rest.find(marker) {
        let marker_end = index + marker.len();
        output.push_str(&rest[..marker_end]);
        rest = &rest[marker_end..];

        let value_len = rest
            .find(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | ')' | ']'))
            .unwrap_or(rest.len());
        output.push_str("[REDACTED]");
        rest = &rest[value_len..];
    }

    output.push_str(rest);
    output
}

fn redact_secret_like_tokens(input: &str) -> String {
    input
        .split_whitespace()
        .map(redact_secret_like_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn redact_secret_like_token(token: &str) -> String {
    let trimmed =
        token.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')));
    if trimmed.starts_with("sk_")
        || trimmed.starts_with("sk-")
        || trimmed.starts_with("hf_")
        || trimmed.starts_with("ghp_")
        || trimmed.starts_with("gho_")
        || trimmed.starts_with("xoxb-")
        || trimmed.starts_with("xoxp-")
        // Xybrid's own key prefixes — the secret most likely to appear in a
        // *xybrid* error / gateway-auth failure string, and previously the one
        // provider key this redactor missed. Matched specifically (`xy_live_`
        // / `xy_test_`, the only two formats used across the SDK docs and
        // bindings) rather than a bare `xy_`, which would over-redact common
        // identifiers like `xy_coords` / `xy_axis` and corrupt legitimate
        // error messages.
        || trimmed.starts_with("xy_live_")
        || trimmed.starts_with("xy_test_")
    {
        token.replacen(trimmed, "[REDACTED]", 1)
    } else {
        token.to_string()
    }
}

/// Build a terminal event for a local abort whose cloud retry was blocked by
/// live policy. This is the one fallback path that remains a hard failure.
pub fn cloud_denied_by_policy_event(
    correlation_id: &str,
    model_id: &str,
    abort_reason: xybrid_core::abort::AbortReason,
    policy_reason: &str,
    latency_ms: u32,
) -> TelemetryEvent {
    let policy_reason = redact_error_for_telemetry(policy_reason);
    TelemetryEvent {
        event_type: "LocalFailed".to_string(),
        stage_name: Some(model_id.to_string()),
        target: Some("local".to_string()),
        latency_ms: Some(latency_ms),
        error: Some(format!("cloud_denied_by_policy: {}", policy_reason)),
        data: Some(
            serde_json::json!({
                "model_id": model_id,
                "correlation_id": correlation_id,
                "outcome_category": {
                    "kind": "hard_fail",
                    "reason": "cloud_denied_by_policy",
                },
                "terminal_state_tag": {
                    "kind": "cloud_denied_by_policy",
                    "abort_reason": abort_reason.as_str(),
                },
                "abort_reason": abort_reason.as_str(),
                "policy_reason": policy_reason,
                "status": "error",
            })
            .to_string(),
        ),
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    }
}

pub fn publish_cloud_denied_by_policy(
    correlation_id: &str,
    model_id: &str,
    abort_reason: xybrid_core::abort::AbortReason,
    policy_reason: &str,
    latency_ms: u32,
) {
    publish_telemetry_event(cloud_denied_by_policy_event(
        correlation_id,
        model_id,
        abort_reason,
        policy_reason,
        latency_ms,
    ));
}

/// Build a `CloudRetry` telemetry event for the cloud leg of a fallback run.
///
/// Emits with the shared `correlation_id` and the cloud provider name when
/// known. Pair with [`local_aborted_event`] (same `correlation_id`) to
/// reconstruct one logical inference across two execution targets.
///
/// `error` is `None` when the cloud leg streamed successfully; `Some(msg)`
/// when it failed before producing a final result. The failure branch sets
/// `event.error` and tags the payload with `status = "error"` so traces
/// list views and dashboards can distinguish a cloud-attempt-that-failed
/// from a cloud-attempt-that-succeeded.
pub fn cloud_retry_event(
    correlation_id: &str,
    model_id: &str,
    provider: Option<&str>,
    latency_ms: u32,
    tokens_emitted: u32,
    error: Option<&str>,
) -> TelemetryEvent {
    let provider_value = provider.unwrap_or("xybrid");
    let redacted_error = error.map(redact_error_for_telemetry);
    let (status, outcome_category) = if let Some(reason) = redacted_error.as_deref() {
        (
            "error",
            serde_json::json!({
                "kind": "hard_fail",
                "reason": reason,
            }),
        )
    } else {
        ("ok", serde_json::json!("cloud_success"))
    };
    TelemetryEvent {
        event_type: "CloudRetry".to_string(),
        stage_name: Some(model_id.to_string()),
        target: Some("cloud".to_string()),
        latency_ms: Some(latency_ms),
        error: redacted_error,
        data: Some(
            serde_json::json!({
                "model_id": model_id,
                "correlation_id": correlation_id,
                "provider": provider_value,
                "outcome_category": outcome_category,
                "tokens_emitted": tokens_emitted,
                "status": status,
            })
            .to_string(),
        ),
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    }
}

/// Convenience wrapper: build and publish a `CloudRetry` event.
pub fn publish_cloud_retry(
    correlation_id: &str,
    model_id: &str,
    provider: Option<&str>,
    latency_ms: u32,
    tokens_emitted: u32,
    error: Option<&str>,
) {
    publish_telemetry_event(cloud_retry_event(
        correlation_id,
        model_id,
        provider,
        latency_ms,
        tokens_emitted,
        error,
    ));
}

// ============================================================================
// First-inference dev nudge
// ============================================================================

/// Documentation for enabling observability export. Hard-coded rather than
/// wired through config because the nudge fires when no config was provided.
const OBSERVABILITY_DOCS_URL: &str = "https://docs.xybrid.dev/sdk/observability";

/// The console that issues project keys (the dashboard origin is retired).
const CONSOLE_URL: &str = "https://console.xybrid.dev";

/// Once-guard for [`maybe_emit_dev_nudge`].
static DEV_NUDGE: std::sync::Once = std::sync::Once::new();

/// Emit a one-shot info-level hint when the host app runs inference without
/// configuring observability export. Subsequent calls are no-ops.
///
/// Hooked into the run entry points so the nudge surfaces on first use rather
/// than at init time. Suppressed when `XYBRID_QUIET=1`.
pub(crate) fn maybe_emit_dev_nudge() {
    DEV_NUDGE.call_once(|| {
        let quiet = std::env::var("XYBRID_QUIET").ok();
        if !should_emit_dev_nudge(crate::observability::is_enabled(), quiet.as_deref()) {
            return;
        }
        log::info!(target: "xybrid_sdk", "{}", dev_nudge_text());
    });
}

/// Pure predicate that decides whether the dev-nudge should print.
fn should_emit_dev_nudge(observability_enabled: bool, quiet: Option<&str>) -> bool {
    quiet != Some("1") && !observability_enabled
}

/// The nudge: export is explicit, and a key (from `xybrid connect` or the
/// console) only authenticates until it is handed to the Xybrid preset.
fn dev_nudge_text() -> String {
    format!(
        "observability export is not configured; traces stay local. To send them to Xybrid, \
         get a project key with `xybrid connect` (or from {CONSOLE_URL}) and pass it to \
         ObservabilityConfig::xybrid; a key alone exports nothing. Any OpenTelemetry backend \
         works through ObservabilityConfig::otlp. See {OBSERVABILITY_DOCS_URL}"
    )
}

#[cfg(test)]
mod dev_nudge_tests {
    use super::{dev_nudge_text, should_emit_dev_nudge};

    #[test]
    fn the_nudge_points_at_connect_and_the_console_never_at_key_only_export() {
        let text = dev_nudge_text();
        assert!(text.contains("xybrid connect"), "{text}");
        assert!(text.contains("https://console.xybrid.dev"), "{text}");
        assert!(text.contains("ObservabilityConfig::xybrid"), "{text}");
        assert!(text.contains("a key alone exports nothing"), "{text}");
        assert!(!text.contains("dashboard.xybrid.dev"), "{text}");
    }

    #[test]
    fn emits_when_observability_is_not_configured() {
        assert!(should_emit_dev_nudge(false, None));
    }

    #[test]
    fn suppressed_when_quiet_flag_set() {
        assert!(!should_emit_dev_nudge(false, Some("1")));
    }

    #[test]
    fn suppressed_when_observability_is_configured() {
        assert!(!should_emit_dev_nudge(true, None));
    }

    #[test]
    fn quiet_other_value_not_suppressive() {
        assert!(should_emit_dev_nudge(false, Some("true")));
        assert!(should_emit_dev_nudge(false, Some("0")));
    }
}

/// Register a telemetry event sender
///
/// The first registration also subscribes to executor lifecycle events
/// (`ExecutionStarted` / `ExecutionCompleted` / `ExecutionFailed`), so they
/// are only produced when someone listens.
pub fn register_telemetry_sender(sender: TelemetrySender) {
    EXECUTION_LISTENER.call_once(register_execution_listener);
    if let Ok(mut senders) = TELEMETRY_SENDERS.lock() {
        senders.push(sender);
    }
}

static EXECUTION_LISTENER: std::sync::Once = std::sync::Once::new();

/// Forward `TemplateExecutor` lifecycle events to subscribers.
fn register_execution_listener() {
    execution_listener::set_execution_listener(|event| {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let telemetry_event = match event {
            ExecutionEvent::Started { model_id, method } => TelemetryEvent {
                event_type: "ExecutionStarted".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: None,
                error: None,
                data: Some(serde_json::json!({ "model": model_id }).to_string()),
                timestamp_ms,
            },
            ExecutionEvent::Completed {
                model_id,
                method,
                latency_ms,
            } => TelemetryEvent {
                event_type: "ExecutionCompleted".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: Some(latency_ms as u32),
                error: None,
                data: Some(serde_json::json!({ "model": model_id }).to_string()),
                timestamp_ms,
            },
            ExecutionEvent::Failed {
                model_id,
                method,
                latency_ms,
                error,
            } => TelemetryEvent {
                event_type: "ExecutionFailed".to_string(),
                stage_name: Some(model_id.clone()),
                target: Some("device".to_string()),
                latency_ms: Some(latency_ms as u32),
                error: Some(error),
                data: Some(
                    serde_json::json!({
                        "model": model_id,
                        "method": method,
                    })
                    .to_string(),
                ),
                timestamp_ms,
            },
        };
        publish_telemetry_event(telemetry_event);
    });
}

/// Convert OrchestratorEvent to TelemetryEvent
pub fn convert_orchestrator_event(event: &OrchestratorEvent) -> TelemetryEvent {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let context = event.context().clone();
    let telemetry_event = match event {
        OrchestratorEvent::PipelineStart { stages, .. } => TelemetryEvent {
            event_type: "PipelineStart".to_string(),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: Some(serde_json::json!({"stages": stages}).to_string()),
            timestamp_ms,
        },
        OrchestratorEvent::PipelineComplete {
            total_latency_ms, ..
        } => TelemetryEvent {
            event_type: "PipelineComplete".to_string(),
            stage_name: None,
            target: None,
            latency_ms: Some(*total_latency_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageStart { stage_name, .. } => TelemetryEvent {
            event_type: "StageStart".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageComplete {
            stage_name,
            target,
            latency_ms,
            ..
        } => TelemetryEvent {
            event_type: "StageComplete".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: Some(*latency_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageError {
            stage_name, error, ..
        } => TelemetryEvent {
            event_type: "StageError".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: Some(error.clone()),
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::RoutingDecided {
            stage_name,
            target,
            reason,
            recent_abort_rate,
            sample_size,
            ..
        } => TelemetryEvent {
            event_type: "RoutingDecided".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: None,
            // Embed the local reliability hint at the top level of event.data
            // so subscribers can inspect it alongside routing context.
            // Non-finite f32 values would serialize to JSON `null`; sanitize
            // them here so consumers always receive a numeric rate.
            data: Some(
                serde_json::json!({
                    "reason": reason,
                    "local_reliability_hint": {
                        "recent_abort_rate": if recent_abort_rate.is_finite() {
                            *recent_abort_rate
                        } else {
                            0.0_f32
                        },
                        "sample_size": sample_size,
                    },
                })
                .to_string(),
            ),
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionStarted {
            stage_name, target, ..
        } => TelemetryEvent {
            event_type: "ExecutionStarted".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionCompleted {
            stage_name,
            target,
            execution_time_ms,
            ..
        } => TelemetryEvent {
            event_type: "ExecutionCompleted".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: Some(*execution_time_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionFailed {
            stage_name,
            target,
            error,
            ..
        } => TelemetryEvent {
            event_type: "ExecutionFailed".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: Some(error.clone()),
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::PolicyEvaluated {
            stage_name,
            allowed,
            reason,
            ..
        } => TelemetryEvent {
            event_type: "PolicyEvaluated".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: if *allowed {
                None
            } else {
                reason.clone().or(Some("Policy violation".to_string()))
            },
            data: Some(
                serde_json::json!({
                    "allowed": allowed,
                    "reason": reason
                })
                .to_string(),
            ),
            timestamp_ms,
        },
        OrchestratorEvent::LocalAborted {
            stage_name,
            target,
            reason,
            ..
        } => TelemetryEvent {
            event_type: "LocalAborted".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: Some(reason.clone()),
            data: Some(serde_json::json!({ "reason": reason }).to_string()),
            timestamp_ms,
        },
        _ => TelemetryEvent {
            event_type: format!("{:?}", event),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: Some(format!("{:?}", event)),
            timestamp_ms,
        },
    };

    attach_event_context(telemetry_event, &context)
}

fn attach_event_context(event: TelemetryEvent, context: &EventContext) -> TelemetryEvent {
    if context.is_empty() {
        return event;
    }

    let mut data = match event.data.as_ref() {
        Some(raw) => match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(value) if value.is_object() => value,
            Ok(value) => serde_json::json!({ "value": value }),
            Err(_) => serde_json::json!({ "value": raw }),
        },
        None => serde_json::json!({}),
    };

    if let Some(obj) = data.as_object_mut() {
        if let Some(id) = context.pipeline_id {
            obj.entry(CONTEXT_PIPELINE_ID_KEY.to_string())
                .or_insert_with(|| serde_json::json!(id.to_string()));
        }
        if let Some(id) = context.trace_id {
            obj.entry(CONTEXT_TRACE_ID_KEY.to_string())
                .or_insert_with(|| serde_json::json!(id.to_string()));
        }
        if let Some(id) = context.correlation_id.as_ref() {
            obj.entry("correlation_id".to_string())
                .or_insert_with(|| serde_json::json!(id));
        }
        if let Some(id) = context.request_id.as_ref() {
            obj.entry("request_id".to_string())
                .or_insert_with(|| serde_json::json!(id));
        }
        if let Some(id) = context.model_id.as_ref() {
            obj.entry("model_id".to_string())
                .or_insert_with(|| serde_json::json!(id));
        }
        if let Some(id) = context.span_id.as_ref() {
            obj.entry("span_id".to_string())
                .or_insert_with(|| serde_json::json!(id));
        }
    }

    TelemetryEvent {
        data: Some(data.to_string()),
        ..event
    }
}

/// Build a `ModelDownload` event for a successful registry fetch.
///
/// Carries the cost-attribution fields agreed with the platform schema:
/// `model_id` (registry mask), `bytes_downloaded` (final on-disk size of
/// the model file or .xyb bundle), `source` (canonical download host
/// label — `r2` for Xybrid's R2 mirror, `huggingface` for direct HF
/// pulls; other hosts pass through as-is so a future provider doesn't
/// silently lose attribution), and `duration_ms` (wallclock time spent
/// inside the network download, excluding hash verification and cache
/// extraction so the field reflects bytes-on-the-wire latency).
///
/// The `event_type` is the literal string `"ModelDownload"`, with download
/// metadata encoded in `TelemetryEvent.data` for in-process subscribers.
///
/// Pure builder — does not publish, allows unit tests to inspect the
/// event shape without standing up the global exporter.
fn build_model_download_event(
    model_id: &str,
    bytes_downloaded: u64,
    source: &str,
    duration_ms: u32,
) -> TelemetryEvent {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let data = serde_json::json!({
        "model_id": model_id,
        "bytes_downloaded": bytes_downloaded,
        "source": source,
        "duration_ms": duration_ms,
    });
    TelemetryEvent {
        event_type: "ModelDownload".to_string(),
        stage_name: None,
        target: None,
        // Mirror duration onto the canonical `latency_ms` field so the
        // platform's existing latency column lights up for downloads
        // without a schema migration.
        latency_ms: Some(duration_ms),
        error: None,
        data: Some(data.to_string()),
        timestamp_ms: now_ms,
    }
}

/// Publish a `ModelDownload` event for a successful registry fetch.
///
/// Honors [`crate::telemetry_optout::is_telemetry_opted_out`]: when the
/// user has set `XYBRID_TELEMETRY_OPTOUT=1` no event is emitted, since
/// the same opt-out already gates the registry call telemetry header
/// and a model-download event would leak the same attribution surface
/// (which model the user pulled, how big it was).
///
/// Cache hits MUST NOT call this — the event represents a real network
/// transfer for cost accounting; emitting it on cache hits would
/// double-count and hide cache-effectiveness metrics.
pub fn publish_model_download(
    model_id: &str,
    bytes_downloaded: u64,
    source: &str,
    duration_ms: u32,
) {
    if crate::telemetry_optout::is_telemetry_opted_out() {
        return;
    }
    let event = build_model_download_event(model_id, bytes_downloaded, source, duration_ms);
    publish_telemetry_event(event);
}

/// Publish a telemetry event to all registered subscribers.
pub fn publish_telemetry_event(event: TelemetryEvent) {
    dispatch_telemetry_event(event);
}

fn dispatch_telemetry_event(event: TelemetryEvent) {
    dispatch_telemetry_event_with_optout(event, crate::telemetry_optout::is_telemetry_opted_out());
}

fn dispatch_telemetry_event_with_optout(event: TelemetryEvent, opted_out: bool) {
    if opted_out {
        return;
    }

    let Ok(senders) = TELEMETRY_SENDERS.lock() else {
        // Mutex is poisoned: skip rather than crash.
        return;
    };
    let mut dead_senders = Vec::new();

    for (idx, sender) in senders.iter().enumerate() {
        if sender.send(event.clone()).is_err() {
            dead_senders.push(idx);
        }
    }

    drop(senders);
    if !dead_senders.is_empty() {
        if let Ok(mut senders) = TELEMETRY_SENDERS.lock() {
            for idx in dead_senders.iter().rev() {
                senders.remove(*idx);
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("orchestrator event bridge is no longer running")]
    Stopped,
    #[error("orchestrator event bridge flush acknowledgement was dropped")]
    FlushAckDropped,
    #[error("orchestrator event bridge thread panicked")]
    ThreadPanicked,
}

enum BridgeCommand {
    Flush(mpsc::Sender<()>),
}

/// Handle for a scoped orchestrator-to-telemetry event bridge.
#[must_use = "BridgeHandle must be flushed or joined, otherwise queued orchestrator telemetry can be dropped"]
pub struct BridgeHandle {
    join_handle: thread::JoinHandle<()>,
    command_tx: mpsc::Sender<BridgeCommand>,
}

impl BridgeHandle {
    /// Block until all events queued before this call have been delivered to
    /// the SDK telemetry stream.
    pub fn flush(&self) -> Result<(), BridgeError> {
        let (ack_tx, ack_rx) = mpsc::channel();
        self.command_tx
            .send(BridgeCommand::Flush(ack_tx))
            .map_err(|_| BridgeError::Stopped)?;
        ack_rx.recv().map_err(|_| BridgeError::FlushAckDropped)
    }

    /// Compatibility alias for callers that used the previous scoped bridge
    /// drain API. New code should prefer [`Self::flush`] and handle errors.
    pub fn drain(&self) {
        let _ = self.flush();
    }

    /// Wait for the bridge thread to finish draining after the orchestrator's
    /// event bus has been dropped.
    pub fn join(self) -> Result<(), BridgeError> {
        let Self {
            join_handle,
            command_tx,
        } = self;
        drop(command_tx);
        join_handle.join().map_err(|_| BridgeError::ThreadPanicked)
    }
}

pub type OrchestratorEventBridge = BridgeHandle;

const BRIDGE_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Bridge orchestrator events to telemetry stream.
///
/// This function subscribes to orchestrator events and converts them to
/// telemetry events. The returned handle must be joined after the orchestrator
/// is dropped so queued events are drained before the caller tears down
/// telemetry subscribers or the runtime.
pub fn bridge_orchestrator_events(
    orchestrator: &xybrid_core::orchestrator::Orchestrator,
) -> BridgeHandle {
    let event_bus = orchestrator.event_bus();
    let subscription = event_bus.subscribe();
    let (command_tx, command_rx) = mpsc::channel();

    let join_handle = thread::spawn(move || bridge_loop(subscription, command_rx));

    BridgeHandle {
        join_handle,
        command_tx,
    }
}

/// Subscribe to orchestrator events and return a drainable bridge handle.
pub fn subscribe_orchestrator_events(
    orchestrator: &xybrid_core::orchestrator::Orchestrator,
) -> BridgeHandle {
    bridge_orchestrator_events(orchestrator)
}

fn bridge_loop(
    subscription: xybrid_core::event_bus::Subscription,
    command_rx: mpsc::Receiver<BridgeCommand>,
) {
    loop {
        drain_bridge_commands(&subscription, &command_rx);
        match subscription.recv_timeout(BRIDGE_POLL_INTERVAL) {
            Ok(event) => publish_orchestrator_event(event),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                drain_available_orchestrator_events(&subscription);
                drain_bridge_commands(&subscription, &command_rx);
                break;
            }
        }
    }
}

fn drain_bridge_commands(
    subscription: &xybrid_core::event_bus::Subscription,
    command_rx: &mpsc::Receiver<BridgeCommand>,
) {
    while let Ok(command) = command_rx.try_recv() {
        match command {
            BridgeCommand::Flush(ack_tx) => {
                drain_available_orchestrator_events(subscription);
                let _ = ack_tx.send(());
            }
        }
    }
}

fn drain_available_orchestrator_events(subscription: &xybrid_core::event_bus::Subscription) {
    loop {
        match subscription.try_recv() {
            Ok(event) => publish_orchestrator_event(event),
            Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => break,
        }
    }
}

/// Returns `true` when the orchestrator event is wire-noise that the SDK
/// already covers via its own publish sites, and should be dropped before
/// reaching the registered telemetry sender.
///
/// Single-stage local LLM calls were producing 6–9 dashboard rows per
/// turn because the orchestrator emits a full pipeline-execution event
/// chain (`PipelineStart`, `StageStart`, `ExecutionStarted`,
/// `ExecutionCompleted`, `StageComplete`, `PipelineComplete`) on top of
/// the SDK's own `ModelComplete` / `PipelineComplete` — every one of
/// these landed as its own row. The orchestrator-side events are useful
/// for in-process subscribers (other code listening on the event bus)
/// but they're noise at the wire boundary, where the user-facing
/// contract is one row per turn (the SDK's completion event).
///
/// Keep: `PolicyEvaluated`, `RoutingDecided` (routing-decision metadata
/// shown alongside the completion row), `StageError`, `ExecutionFailed`
/// (errors must always reach the wire).
///
/// Drop: pipeline-frame and per-stage success-path events. Errors on
/// these paths still surface via the `*Failed` / `*Error` variants
/// above, which the filter passes through.
fn orchestrator_event_is_wire_noise(event: &OrchestratorEvent) -> bool {
    matches!(
        event,
        OrchestratorEvent::PipelineStart { .. }
            | OrchestratorEvent::PipelineComplete { .. }
            | OrchestratorEvent::StageStart { .. }
            | OrchestratorEvent::StageComplete { .. }
            | OrchestratorEvent::ExecutionStarted { .. }
            | OrchestratorEvent::ExecutionCompleted { .. }
    )
}

fn publish_orchestrator_event(event: OrchestratorEvent) {
    if orchestrator_event_is_wire_noise(&event) {
        return;
    }
    let telemetry_event = convert_orchestrator_event(&event);
    publish_telemetry_event(telemetry_event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::MutexGuard;
    use uuid::Uuid;

    static TELEMETRY_SENDER_TEST_LOCK: Mutex<()> = Mutex::new(());

    // Serialize tests that replace the process-wide subscriber registry and
    // restore it even if a test panics. Bridge threads are joined before this
    // guard drops so queued events cannot escape into the next test.
    struct TelemetrySenderTestGuard {
        _lock: MutexGuard<'static, ()>,
        previous: Vec<TelemetrySender>,
    }

    impl TelemetrySenderTestGuard {
        fn acquire() -> Self {
            let lock = TELEMETRY_SENDER_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let previous = std::mem::take(
                &mut *TELEMETRY_SENDERS
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for TelemetrySenderTestGuard {
        fn drop(&mut self) {
            *TELEMETRY_SENDERS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                std::mem::take(&mut self.previous);
        }
    }

    fn publish_and_receive(event: TelemetryEvent) -> TelemetryEvent {
        let _guard = TelemetrySenderTestGuard::acquire();
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);
        publish_telemetry_event(event.clone());
        let received = receive_event(&rx, &event.event_type);
        assert_eq!(
            serde_json::to_value(&received).unwrap(),
            serde_json::to_value(&event).unwrap(),
            "subscribers must receive every event field unchanged",
        );
        received
    }

    fn receive_event(rx: &mpsc::Receiver<TelemetryEvent>, event_type: &str) -> TelemetryEvent {
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let received = rx
                .recv_timeout(remaining)
                .expect("the expected telemetry event must arrive before the deadline");
            // Other SDK unit tests can publish into the process-wide registry
            // without replacing it. Select this test's event before asserting
            // its payload so unrelated model activity cannot cause a failure.
            if received.event_type == event_type {
                return received;
            }
        }
    }

    #[test]
    fn local_aborted_and_cloud_retry_events_share_correlation_id() {
        let local = local_aborted_event(
            "run-abc-123",
            "qwen2.5-0.5b",
            xybrid_core::abort::AbortReason::StressMemory,
            180,
            4,
        );
        let cloud = cloud_retry_event("run-abc-123", "qwen2.5-0.5b", Some("openai"), 920, 38, None);

        assert_eq!(local.event_type, "LocalAborted");
        assert_eq!(local.target.as_deref(), Some("local"));
        assert_eq!(local.latency_ms, Some(180));

        assert_eq!(cloud.event_type, "CloudRetry");
        assert_eq!(cloud.target.as_deref(), Some("cloud"));
        assert_eq!(cloud.latency_ms, Some(920));

        let local_data: serde_json::Value =
            serde_json::from_str(local.data.as_ref().unwrap()).unwrap();
        let cloud_data: serde_json::Value =
            serde_json::from_str(cloud.data.as_ref().unwrap()).unwrap();

        assert_eq!(local_data["correlation_id"], "run-abc-123");
        assert_eq!(cloud_data["correlation_id"], "run-abc-123");
        assert_eq!(local_data["abort_reason"], "stress_memory");
        assert_eq!(
            local_data["outcome_category"]["kind"],
            "aborted_for_cloud_fallback"
        );
        assert_eq!(local_data["outcome_category"]["reason"], "stress_memory");
        assert_eq!(local_data["tokens_emitted"], 4);
        assert_eq!(cloud_data["provider"], "openai");
        assert_eq!(cloud_data["outcome_category"], "cloud_success");
        assert_eq!(cloud_data["tokens_emitted"], 38);

        let received_local = publish_and_receive(local);
        let received_cloud = publish_and_receive(cloud);
        let received_local_data: serde_json::Value =
            serde_json::from_str(received_local.data.as_deref().unwrap()).unwrap();
        let received_cloud_data: serde_json::Value =
            serde_json::from_str(received_cloud.data.as_deref().unwrap()).unwrap();
        assert_eq!(received_local_data, local_data);
        assert_eq!(received_cloud_data, cloud_data);
    }

    #[test]
    fn local_aborted_event_carries_resource_summary_and_reliability_hint() {
        let summary = ResourceUsageSummary {
            cpu_avg_pct: Some(61.0),
            cpu_peak_pct: Some(88.5),
            process_rss_peak_mb: Some(2048),
            available_mem_min_mb: Some(512),
            memory_pressure_peak: xybrid_core::device::MemoryPressure::Critical,
            thermal_state_peak: xybrid_core::device::ThermalState::Hot,
            thermal_measured: true,
            battery_pct_end: Some(51),
            sample_count: 3,
            sampling_mode: "debug_local".to_string(),
            sampling_interval_ms: Some(100),
        };
        let local = local_aborted_event_with_details(
            "run-rich-123",
            "qwen2.5-0.5b",
            xybrid_core::abort::AbortReason::StressMemory,
            180,
            4,
            Some(summary),
            Some(LocalReliabilityHint {
                recent_abort_rate: 0.5,
                sample_size: 2,
            }),
        );

        let data: serde_json::Value =
            serde_json::from_str(local.data.as_ref().expect("data present")).unwrap();
        assert_eq!(
            data["resource_summary"]["memory_pressure_peak"].as_str(),
            Some("critical")
        );
        assert_eq!(
            data["resource_summary"]["sampling_interval_ms"].as_i64(),
            Some(100)
        );
        assert_eq!(
            data["local_reliability_hint"]["recent_abort_rate"].as_f64(),
            Some(0.5)
        );
        assert_eq!(
            data["local_reliability_hint"]["sample_size"].as_i64(),
            Some(2)
        );

        let received = publish_and_receive(local);
        let received_data: serde_json::Value =
            serde_json::from_str(received.data.as_deref().unwrap()).unwrap();
        assert_eq!(received_data, data);
        assert_eq!(received_data["resource_summary"]["sample_count"], 3);
    }

    #[test]
    fn cloud_retry_event_defaults_provider_when_unspecified() {
        let cloud = cloud_retry_event("run-xyz", "model-x", None, 50, 10, None);
        let cloud_data: serde_json::Value =
            serde_json::from_str(cloud.data.as_ref().unwrap()).unwrap();
        assert_eq!(cloud_data["provider"], "xybrid");
        assert_eq!(cloud_data["outcome_category"], "cloud_success");
        assert_eq!(cloud_data["status"], "ok");
        assert!(cloud.error.is_none());
    }

    #[test]
    fn cloud_retry_event_marks_failure_branch() {
        let cloud = cloud_retry_event(
            "run-fail-1",
            "deepseek-chat",
            Some("deepseek"),
            842,
            0,
            Some("Gateway returned 502: Provider error"),
        );
        let cloud_data: serde_json::Value =
            serde_json::from_str(cloud.data.as_ref().unwrap()).unwrap();
        assert_eq!(cloud.event_type, "CloudRetry");
        assert_eq!(cloud.target.as_deref(), Some("cloud"));
        assert_eq!(cloud.latency_ms, Some(842));
        assert_eq!(
            cloud.error.as_deref(),
            Some("Gateway returned 502: Provider error")
        );
        assert_eq!(cloud_data["status"], "error");
        assert_eq!(cloud_data["outcome_category"]["kind"], "hard_fail");
        assert_eq!(
            cloud_data["outcome_category"]["reason"],
            "Gateway returned 502: Provider error"
        );
        assert_eq!(cloud_data["tokens_emitted"], 0);

        let received = publish_and_receive(cloud);
        let received_data: serde_json::Value =
            serde_json::from_str(received.data.as_deref().unwrap()).unwrap();
        assert_eq!(received_data, cloud_data);
        assert_eq!(
            received_data["outcome_category"],
            serde_json::json!({
                "kind": "hard_fail",
                "reason": "Gateway returned 502: Provider error"
            }),
        );
    }

    #[test]
    fn telemetry_error_redaction_removes_api_keys_and_bearer_tokens() {
        let redacted = redact_error_for_telemetry(
            "Gateway returned 401: Authorization: Bearer sk_test_abc123 and api_key=hf_secret_xyz",
        );

        assert!(redacted.contains("Authorization: Bearer [REDACTED]"));
        assert!(redacted.contains("api_key=[REDACTED]"));
        assert!(!redacted.contains("sk_test_abc123"));
        assert!(!redacted.contains("hf_secret_xyz"));
    }

    #[test]
    fn telemetry_error_redaction_removes_xybrid_own_key() {
        // The Xybrid key is the secret most likely to appear in a xybrid
        // gateway-auth error; the redactor must catch its `xy_live_` /
        // `xy_test_` prefixes just like every other provider's.
        let redacted = redact_error_for_telemetry(
            "Gateway auth failed for key xy_live_abc123def and xy_test_secret456",
        );
        assert!(
            !redacted.contains("xy_live_abc123def"),
            "redacted = {redacted}"
        );
        assert!(
            !redacted.contains("xy_test_secret456"),
            "redacted = {redacted}"
        );
        assert!(redacted.contains("[REDACTED]"));
    }

    #[test]
    fn telemetry_error_redaction_does_not_over_redact_xy_identifiers() {
        // The Xybrid-key match is intentionally specific (`xy_live_` /
        // `xy_test_`), not a bare `xy_` — common identifiers like
        // `xy_coords` must survive so error messages stay debuggable.
        let msg = "decode failed at xy_coords=(3,4), xy_axis=z, xy_plane unset";
        let redacted = redact_error_for_telemetry(msg);
        assert_eq!(
            redacted, msg,
            "non-key xy_ identifiers must not be redacted"
        );
    }

    #[test]
    fn attach_resource_summary_with_none_is_noop() {
        let original = TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: Some("{\"model_id\":\"x\"}".to_string()),
            timestamp_ms: 0,
        };
        let mut event = original.clone();
        attach_resource_summary(&mut event, None);
        assert_eq!(event.data, original.data);
    }

    #[test]
    fn env_resource_mode_parses_all_variants() {
        // Exercise the same parser as the env override without mutating the
        // environment while other SDK tests initialize observability.
        let cases: &[(&str, ResourceTelemetryMode)] = &[
            ("off", ResourceTelemetryMode::Off),
            ("boundary", ResourceTelemetryMode::Boundary),
            (
                "summary",
                ResourceTelemetryMode::Summary {
                    interval_ms: ResourceTelemetryMode::DEFAULT_SUMMARY_INTERVAL_MS,
                },
            ),
            (
                "summary:500",
                ResourceTelemetryMode::Summary { interval_ms: 500 },
            ),
            (
                "debug_local:250",
                ResourceTelemetryMode::DebugLocal { interval_ms: 250 },
            ),
        ];
        for (raw, expected) in cases {
            let parsed = parse_resource_telemetry_mode(raw).expect(raw);
            assert_eq!(&parsed, expected, "parsing `{}`", raw);
        }
        assert!(parse_resource_telemetry_mode("unknown").is_none());
        assert!(parse_resource_telemetry_mode("").is_none());
    }

    #[test]
    fn test_convert_stage_start_event() {
        let event = OrchestratorEvent::StageStart {
            stage_name: "asr".to_string(),
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageStart");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert!(telemetry.target.is_none());
        assert!(telemetry.latency_ms.is_none());
        assert!(telemetry.error.is_none());
        assert!(telemetry.timestamp_ms > 0);
    }

    #[test]
    fn test_convert_stage_complete_event() {
        let event = OrchestratorEvent::StageComplete {
            stage_name: "tts".to_string(),
            target: "local".to_string(),
            latency_ms: 150,
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageComplete");
        assert_eq!(telemetry.stage_name, Some("tts".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
        assert_eq!(telemetry.latency_ms, Some(150));
        assert!(telemetry.error.is_none());
    }

    #[test]
    fn test_convert_stage_error_event() {
        let event = OrchestratorEvent::StageError {
            stage_name: "asr".to_string(),
            error: "Model not found".to_string(),
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageError");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.error, Some("Model not found".to_string()));
    }

    #[test]
    fn test_convert_pipeline_start_event() {
        let event = OrchestratorEvent::PipelineStart {
            stages: vec!["asr".to_string(), "llm".to_string(), "tts".to_string()],
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PipelineStart");
        assert!(telemetry.stage_name.is_none());
        assert!(telemetry.data.is_some());
        let data = telemetry.data.unwrap();
        assert!(data.contains("asr"));
        assert!(data.contains("llm"));
        assert!(data.contains("tts"));
    }

    #[test]
    fn test_convert_pipeline_complete_event() {
        let event = OrchestratorEvent::PipelineComplete {
            total_latency_ms: 500,
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PipelineComplete");
        assert_eq!(telemetry.latency_ms, Some(500));
    }

    #[test]
    fn test_convert_routing_decided_event() {
        let event = OrchestratorEvent::RoutingDecided {
            stage_name: "asr".to_string(),
            target: "cloud".to_string(),
            reason: "network_optimal".to_string(),
            recent_abort_rate: 0.0,
            sample_size: 0,
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "RoutingDecided");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("cloud".to_string()));
        assert!(telemetry.data.is_some());
        let data = telemetry.data.unwrap();
        assert!(data.contains("network_optimal"));
    }

    #[test]
    fn routing_decided_event_carries_local_reliability_hint_end_to_end() {
        // Follow routing metadata through the orchestrator bridge and the
        // supported in-process subscriber stream.
        let event = OrchestratorEvent::RoutingDecided {
            stage_name: "stage-1".to_string(),
            target: "cloud".to_string(),
            reason: "history_bias".to_string(),
            recent_abort_rate: 0.75,
            sample_size: 4,
            context: Default::default(),
        };
        let telemetry_event = convert_orchestrator_event(&event);

        // The bridge embeds the hint in event.data for subscribers.
        let data_str = telemetry_event.data.as_ref().expect("data must be present");
        let parsed_data: serde_json::Value = serde_json::from_str(data_str).unwrap();
        assert_eq!(
            parsed_data["local_reliability_hint"]["recent_abort_rate"]
                .as_f64()
                .unwrap_or(-1.0),
            0.75
        );
        assert_eq!(
            parsed_data["local_reliability_hint"]["sample_size"].as_i64(),
            Some(4)
        );

        let received = publish_and_receive(telemetry_event);
        let received_data: serde_json::Value =
            serde_json::from_str(received.data.as_deref().unwrap()).unwrap();
        assert_eq!(received_data, parsed_data);
    }

    #[test]
    fn routing_decided_event_sanitizes_non_finite_recent_abort_rate() {
        // Defense-in-depth: the authority emits only finite rates, but
        // the bridge clamps NaN/Infinity to 0.0 so the platform's typed
        // Float32 column never receives a JSON null. The clamp lives in
        // the bridge so any future direct OrchestratorEvent caller (test,
        // example, FFI) cannot poison the telemetry stream.
        for bad_rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let event = OrchestratorEvent::RoutingDecided {
                stage_name: "stage-1".to_string(),
                target: "cloud".to_string(),
                reason: "history_bias".to_string(),
                recent_abort_rate: bad_rate,
                sample_size: 1,
                context: Default::default(),
            };
            let telemetry_event = convert_orchestrator_event(&event);
            let parsed_data: serde_json::Value =
                serde_json::from_str(telemetry_event.data.as_ref().unwrap()).unwrap();
            assert_eq!(
                parsed_data["local_reliability_hint"]["recent_abort_rate"].as_f64(),
                Some(0.0),
                "non-finite rate {bad_rate} must be sanitized to 0.0"
            );
        }
    }

    #[test]
    fn scoped_orchestrator_bridge_drains_queued_events_with_captured_context() {
        let _guard = TelemetrySenderTestGuard::acquire();
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let pipeline_id = Uuid::new_v4();
        let trace_id = Uuid::new_v4();
        let _event_context = xybrid_core::event_bus::EventContextGuard::install(
            xybrid_core::event_bus::EventContext::default()
                .with_pipeline_id(pipeline_id)
                .with_trace_id(trace_id),
        );
        let orchestrator = xybrid_core::orchestrator::Orchestrator::new();
        let bridge = bridge_orchestrator_events(&orchestrator);

        orchestrator
            .event_bus()
            .publish(OrchestratorEvent::RoutingDecided {
                stage_name: "scoped-bridge-context".to_string(),
                target: "cloud".to_string(),
                reason: "history_bias".to_string(),
                recent_abort_rate: 0.5,
                sample_size: 2,
                context: Default::default(),
            });
        bridge.drain();
        drop(orchestrator);
        bridge
            .join()
            .expect("bridge should stop before test cleanup");

        let mut received = None;
        for _ in 0..20 {
            match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(event)
                    if event.event_type == "RoutingDecided"
                        && event.stage_name.as_deref() == Some("scoped-bridge-context") =>
                {
                    received = Some(event);
                    break;
                }
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("telemetry receiver disconnected before RoutingDecided arrived")
                }
            }
        }
        let received = received.expect("drained bridge should publish queued orchestrator event");

        let data: serde_json::Value =
            serde_json::from_str(received.data.as_ref().expect("context-bearing data")).unwrap();
        assert_eq!(
            data[CONTEXT_PIPELINE_ID_KEY],
            serde_json::json!(pipeline_id)
        );
        assert_eq!(data[CONTEXT_TRACE_ID_KEY], serde_json::json!(trace_id));
        assert_eq!(
            data["local_reliability_hint"]["recent_abort_rate"].as_f64(),
            Some(0.5)
        );
        assert_eq!(
            data["local_reliability_hint"]["sample_size"].as_i64(),
            Some(2)
        );
    }

    #[test]
    fn test_convert_execution_started_event() {
        let event = OrchestratorEvent::ExecutionStarted {
            stage_name: "asr".to_string(),
            target: "local".to_string(),
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionStarted");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
    }

    #[test]
    fn test_convert_execution_completed_event() {
        let event = OrchestratorEvent::ExecutionCompleted {
            stage_name: "asr".to_string(),
            target: "local".to_string(),
            execution_time_ms: 75,
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionCompleted");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
        assert_eq!(telemetry.latency_ms, Some(75));
    }

    #[test]
    fn test_convert_execution_failed_event() {
        let event = OrchestratorEvent::ExecutionFailed {
            stage_name: "tts".to_string(),
            target: "cloud".to_string(),
            error: "Timeout".to_string(),
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionFailed");
        assert_eq!(telemetry.stage_name, Some("tts".to_string()));
        assert_eq!(telemetry.target, Some("cloud".to_string()));
        assert_eq!(telemetry.error, Some("Timeout".to_string()));
    }

    #[test]
    fn test_convert_policy_evaluated_allowed() {
        let event = OrchestratorEvent::PolicyEvaluated {
            stage_name: "asr".to_string(),
            allowed: true,
            reason: Some("All conditions met".to_string()),
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PolicyEvaluated");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert!(telemetry.error.is_none()); // No error when allowed
        assert!(telemetry.data.is_some());
    }

    #[test]
    fn test_convert_policy_evaluated_denied() {
        let event = OrchestratorEvent::PolicyEvaluated {
            stage_name: "llm".to_string(),
            allowed: false,
            reason: Some("Privacy policy violation".to_string()),
            context: Default::default(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PolicyEvaluated");
        assert_eq!(telemetry.stage_name, Some("llm".to_string()));
        assert_eq!(
            telemetry.error,
            Some("Privacy policy violation".to_string())
        );
    }

    fn routing_decided_event(stage_name: impl Into<String>) -> OrchestratorEvent {
        OrchestratorEvent::RoutingDecided {
            stage_name: stage_name.into(),
            target: "local".to_string(),
            reason: "test_route".to_string(),
            recent_abort_rate: 0.0,
            sample_size: 0,
            context: xybrid_core::event_bus::EventContext::default(),
        }
    }

    #[test]
    fn bridge_drains_all_events_on_orchestrator_drop() {
        let _guard = TelemetrySenderTestGuard::acquire();
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let orchestrator = xybrid_core::orchestrator::Orchestrator::new();
        let bridge = bridge_orchestrator_events(&orchestrator);
        for idx in 0..100 {
            orchestrator
                .event_bus()
                .publish(routing_decided_event(format!("bridge-drain-{idx:03}")));
        }

        drop(orchestrator);
        bridge.join().expect("bridge should drain cleanly");

        let events: Vec<_> = rx
            .try_iter()
            .filter(|event| {
                event.event_type == "RoutingDecided"
                    && event
                        .stage_name
                        .as_deref()
                        .is_some_and(|stage| stage.starts_with("bridge-drain-"))
            })
            .collect();
        assert_eq!(events.len(), 100);
        for idx in 0..100 {
            assert!(
                events
                    .iter()
                    .any(|event| event.stage_name.as_deref()
                        == Some(&format!("bridge-drain-{idx:03}"))),
                "missing routed event bridge-drain-{idx:03}; got {events:?}"
            );
        }
    }

    #[test]
    fn bridge_preserves_correlation_id_across_task_boundary() {
        let _guard = TelemetrySenderTestGuard::acquire();
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let publishers: Vec<_> = ["A", "B"]
            .into_iter()
            .map(|correlation_id| {
                thread::spawn(move || {
                    let _context = xybrid_core::event_bus::EventContextGuard::install(
                        xybrid_core::event_bus::EventContext::default()
                            .with_correlation_id(correlation_id),
                    );
                    let orchestrator = xybrid_core::orchestrator::Orchestrator::new();
                    let bridge = bridge_orchestrator_events(&orchestrator);
                    for idx in 0..10 {
                        orchestrator
                            .event_bus()
                            .publish(routing_decided_event(format!("{correlation_id}-{idx}")));
                        thread::yield_now();
                    }
                    drop(orchestrator);
                    bridge.join().expect("bridge should drain cleanly");
                })
            })
            .collect();

        for publisher in publishers {
            publisher.join().expect("publisher thread should not panic");
        }

        let events: Vec<_> = rx
            .try_iter()
            .filter(|event| {
                event.event_type == "RoutingDecided"
                    && event
                        .stage_name
                        .as_deref()
                        .is_some_and(|stage| stage.starts_with("A-") || stage.starts_with("B-"))
            })
            .collect();
        assert_eq!(events.len(), 20);
        for event in events {
            let stage = event.stage_name.as_deref().expect("stage name");
            let data: serde_json::Value =
                serde_json::from_str(event.data.as_deref().expect("event data")).unwrap();
            let expected = stage.split('-').next().unwrap();
            assert_eq!(
                data["correlation_id"].as_str(),
                Some(expected),
                "event {stage} carried wrong context: {data}"
            );
        }
    }

    #[test]
    fn flush_blocks_until_in_flight_events_delivered() {
        let _guard = TelemetrySenderTestGuard::acquire();
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let orchestrator = xybrid_core::orchestrator::Orchestrator::new();
        let bridge = bridge_orchestrator_events(&orchestrator);
        for idx in 0..5 {
            orchestrator
                .event_bus()
                .publish(routing_decided_event(format!("flush-{idx}")));
        }

        bridge.flush().expect("flush should complete");

        let mut delivered = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while delivered.len() < 5 && std::time::Instant::now() < deadline {
            let event = rx
                .recv_timeout(Duration::from_millis(50))
                .expect("flush should deliver event before returning");
            if event.event_type == "RoutingDecided"
                && event
                    .stage_name
                    .as_deref()
                    .is_some_and(|stage| stage.starts_with("flush-"))
            {
                delivered.push(event);
            }
        }
        assert_eq!(delivered.len(), 5);

        drop(orchestrator);
        bridge.join().expect("bridge should drain cleanly");
    }

    #[test]
    fn test_telemetry_event_serialization() {
        let event = TelemetryEvent {
            event_type: "StageStart".to_string(),
            stage_name: Some("asr".to_string()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms: 1234567890,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("StageStart"));
        assert!(json.contains("asr"));

        let deserialized: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_type, "StageStart");
        assert_eq!(deserialized.stage_name, Some("asr".to_string()));
    }

    #[test]
    fn test_register_and_publish() {
        let _guard = TelemetrySenderTestGuard::acquire();
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let event = TelemetryEvent {
            event_type: "TestEvent".to_string(),
            stage_name: Some("test".to_string()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms: 0,
        };

        publish_telemetry_event(event.clone());

        let received_event = receive_event(&rx, "TestEvent");
        assert_eq!(received_event.event_type, "TestEvent");
    }

    #[test]
    fn dispatch_skips_all_senders_when_telemetry_is_opted_out() {
        let _guard = TelemetrySenderTestGuard::acquire();
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let event = TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some("lfm2-vl-450m".to_string()),
            target: Some("local".to_string()),
            latency_ms: Some(42),
            error: None,
            data: Some(
                serde_json::json!({
                    "task": "vlm",
                    "image_preprocess_ms": 17
                })
                .to_string(),
            ),
            timestamp_ms: 0,
        };

        dispatch_telemetry_event_with_optout(event.clone(), true);

        assert!(
            !rx.try_iter().any(|received| {
                received.event_type == event.event_type && received.stage_name == event.stage_name
            }),
            "XYBRID_TELEMETRY_OPTOUT must suppress VLM inference events entirely"
        );
    }

    #[test]
    fn build_model_download_event_has_canonical_shape() {
        // ModelDownload is the cost-attribution event for a successful
        // registry fetch. The wire shape is fixed by the platform's
        // billing pipeline: top-level `event_type` literal
        // "ModelDownload", `latency_ms` mirrors `duration_ms`, and the
        // four cost fields land under `data` as a JSON object.
        let event = build_model_download_event("kokoro-82m", 1_234_567, "huggingface", 5_432);

        assert_eq!(event.event_type, "ModelDownload");
        assert_eq!(
            event.latency_ms,
            Some(5_432),
            "latency_ms must mirror duration_ms so the existing latency column lights up without a schema migration"
        );
        assert!(event.error.is_none());
        assert!(event.stage_name.is_none());
        assert!(event.target.is_none());
        assert!(event.timestamp_ms > 0, "timestamp must be populated");

        let data: serde_json::Value =
            serde_json::from_str(event.data.as_deref().expect("data present"))
                .expect("data is valid JSON");
        assert_eq!(data["model_id"].as_str(), Some("kokoro-82m"));
        assert_eq!(data["bytes_downloaded"].as_u64(), Some(1_234_567));
        assert_eq!(data["source"].as_str(), Some("huggingface"));
        assert_eq!(data["duration_ms"].as_u64(), Some(5_432));
    }

    #[test]
    fn build_model_download_event_carries_r2_source_label() {
        // Sanity that the closed-set source label is passed through as-is.
        // The classifier in `registry_client::classify_download_source`
        // is responsible for mapping URLs to labels; this test guards
        // the builder's pass-through contract.
        let event = build_model_download_event("test-model", 42, "r2", 1);
        let data: serde_json::Value = serde_json::from_str(event.data.as_deref().unwrap()).unwrap();
        assert_eq!(data["source"].as_str(), Some("r2"));
    }
}

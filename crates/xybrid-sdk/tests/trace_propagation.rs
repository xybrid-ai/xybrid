//! Trace context at the real outbound call sites.
//!
//! Xybrid-owned hops (the configured platform gateway, the model registry and
//! remote routing advice) carry the active context automatically. Third-party
//! provider origins (`backend: direct` over the gateway transport, and the
//! native direct clients) carry it only when the session allowlists that exact
//! origin. A gateway retry is a new execution attempt with its own
//! `traceparent`. Legacy `run_pipeline[_async]` runs (which report to the
//! default session) close their spans inside their workflow and observe stages.
//!
//! This is its own test binary because the platform gateway URL is
//! process-wide; every test here runs serially behind [`serial`].
use httpmock::prelude::*;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanProcessor};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::observability::{
    self, semconv, ObservabilityConfig, ObservabilitySession, ObservationOptions, Observer,
};
use xybrid_sdk::{PipelineRef, RunOptions};

fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    static CACHE: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let guard = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    // Registry clients open the SDK cache; keep it out of the user's home.
    let cache = CACHE.get_or_init(|| tempfile::tempdir().unwrap());
    xybrid_sdk::init_sdk_cache_dir(cache.path().join("models"));
    // A later test's mock server may reuse an earlier one's port.
    SEEN.lock().unwrap_or_else(PoisonError::into_inner).clear();
    guard
}

#[derive(Clone, Debug, Default)]
struct Memory(Arc<Mutex<Vec<SpanData>>>);

impl SpanProcessor for Memory {
    fn on_start(&self, _: &mut opentelemetry_sdk::trace::Span, _: &opentelemetry::Context) {}
    fn on_end(&self, span: SpanData) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(span);
    }
    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }
    fn shutdown_with_timeout(&self, _: Duration) -> OTelSdkResult {
        Ok(())
    }
}

impl Memory {
    fn spans(&self) -> Vec<SpanData> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// A session over an in-memory provider, allowlisting `origins`.
struct Traced {
    memory: Memory,
    session: ObservabilitySession,
    _provider: SdkTracerProvider,
}

impl Traced {
    fn new(origins: &[String]) -> Self {
        let memory = Memory::default();
        let provider = SdkTracerProvider::builder()
            .with_span_processor(memory.clone())
            .build();
        let mut config = ObservabilityConfig::with_observer(
            Observer::builder().tracer_provider(&provider).build(),
        );
        for origin in origins {
            config = config.propagate_context_to(origin).unwrap();
        }
        Self {
            memory,
            session: ObservabilitySession::new(config).unwrap(),
            _provider: provider,
        }
    }

    fn options(&self) -> RunOptions {
        RunOptions::default()
            .with_observation(ObservationOptions::default().with_session(&self.session))
    }
}

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.to_string())
}

fn traceparent_of(span: &SpanData) -> String {
    format!(
        "00-{}-{}-01",
        span.span_context.trace_id(),
        span.span_context.span_id()
    )
}

// ---------------------------------------------------------------------------
// Request capture: every request's `Host` and `traceparent`, in arrival order.
// ---------------------------------------------------------------------------

static SEEN: Mutex<Vec<(String, Option<String>)>> = Mutex::new(Vec::new());

fn header(request: &HttpMockRequest, name: &str) -> Option<String> {
    request.headers.as_ref().and_then(|headers| {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    })
}

/// Records the request; always matches.
fn record(request: &HttpMockRequest) -> bool {
    SEEN.lock().unwrap_or_else(PoisonError::into_inner).push((
        header(request, "host").unwrap_or_default(),
        header(request, "traceparent"),
    ));
    true
}

/// Records the request and matches only the first try of a request: the
/// first `traceparent` this host saw. A retry that re-sends the same context
/// fails again; one with its own context falls through to the next mock.
fn first_try(request: &HttpMockRequest) -> bool {
    let host = header(request, "host").unwrap_or_default();
    let traceparent = header(request, "traceparent");
    let mut seen = SEEN.lock().unwrap_or_else(PoisonError::into_inner);
    let first = seen
        .iter()
        .find(|(h, _)| *h == host)
        .map(|(_, tp)| tp.clone());
    seen.push((host, traceparent.clone()));
    first.is_none_or(|first| first == traceparent)
}

/// `traceparent` of every request `server` received, in order.
fn seen_by(server: &MockServer) -> Vec<Option<String>> {
    let host = server.address().to_string();
    SEEN.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .filter(|(h, _)| *h == host)
        .map(|(_, tp)| tp.clone())
        .collect()
}

const COMPLETION: &str =
    r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
const ANTHROPIC: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#;

fn recording(body: &'static str) -> MockServer {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.matches(record);
        then.status(200)
            .header("content-type", "application/json")
            .body(body);
    });
    server
}

fn run(stage_options: &str, options: &RunOptions) -> xybrid_sdk::pipeline::PipelineExecutionResult {
    let yaml = format!(
        "name: hops\nstages:\n  - id: draft\n    model: hop-model\n    target: cloud\n    timeout_ms: 20000\n{stage_options}"
    );
    PipelineRef::from_yaml(&yaml)
        .unwrap()
        .load()
        .unwrap()
        .run_with_options(&Envelope::new(EnvelopeKind::Text("hi".into())), options)
        .unwrap()
}

/// The attempt spans of the run's only stage inference, by index.
fn attempts(memory: &Memory) -> Vec<SpanData> {
    let spans = memory.spans();
    let stage = spans
        .iter()
        .find(|s| {
            attr(s, "xybrid.stage.name").as_deref() == Some("draft")
                && attr(s, semconv::attr::OPERATION_NAME).is_some()
        })
        .expect("stage inference");
    let mut attempts: Vec<SpanData> = spans
        .iter()
        .filter(|s| s.parent_span_id == stage.span_context.span_id())
        .filter(|s| attr(s, semconv::attr::ATTEMPT_INDEX).is_some())
        .cloned()
        .collect();
    attempts.sort_by_key(|s| attr(s, semconv::attr::ATTEMPT_INDEX));
    attempts
}

#[test]
fn the_xybrid_gateway_receives_the_live_attempt_context() {
    let _serial = serial();
    let platform = recording(COMPLETION);
    xybrid_sdk::set_gateway_url(platform.url("/v1"));
    let traced = Traced::new(&[]);

    run(
        "    provider: openai\n    api_key: platform-test-key\n",
        &traced.options(),
    );

    let attempts = attempts(&traced.memory);
    assert_eq!(attempts.len(), 1);
    assert_eq!(seen_by(&platform), [Some(traceparent_of(&attempts[0]))]);
}

#[test]
fn direct_provider_origins_receive_context_only_when_allowlisted() {
    let _serial = serial();
    // The platform gateway is elsewhere; these are third-party origins.
    let platform = recording(COMPLETION);
    xybrid_sdk::set_gateway_url(platform.url("/v1"));
    let openai = recording(COMPLETION);
    let anthropic = recording(ANTHROPIC);

    for allowlisted in [false, true] {
        let origins = if allowlisted {
            vec![openai.base_url(), anthropic.base_url()]
        } else {
            Vec::new()
        };
        let traced = Traced::new(&origins);
        // OpenAI-compatible `backend: direct` rides the gateway transport.
        run(
            &format!(
                "    provider: openai\n    backend: direct\n    gateway_url: \"{}\"\n    api_key: provider-test-key\n",
                openai.url("/v1")
            ),
            &traced.options(),
        );
        // Anthropic `backend: direct` uses the native cloud_llm client.
        run(
            &format!(
                "    provider: anthropic\n    backend: direct\n    gateway_url: \"{}\"\n    api_key: provider-test-key\n",
                anthropic.url("/v1")
            ),
            &traced.options(),
        );
        let openai_seen = seen_by(&openai);
        let anthropic_seen = seen_by(&anthropic);
        assert_eq!(
            openai_seen.last().unwrap().is_some(),
            allowlisted,
            "openai direct, allowlisted={allowlisted}: {openai_seen:?}"
        );
        assert_eq!(
            anthropic_seen.last().unwrap().is_some(),
            allowlisted,
            "anthropic direct, allowlisted={allowlisted}: {anthropic_seen:?}"
        );
    }
    assert!(seen_by(&platform).is_empty(), "no platform traffic");
}

#[test]
fn registry_and_routing_advice_are_xybrid_hops() {
    let _serial = serial();
    let registry = recording(r#"{"models":[]}"#);
    let advice = recording(
        r#"{"target":"cloud","provider":"xybrid","reason":"advice","confidence":0.8,"ttl_ms":0}"#,
    );
    let traced = Traced::new(&[]);
    let workflow = observability::start_workflow(
        "hops",
        &ObservationOptions::default().with_session(&traced.session),
    );
    let expected = workflow.context().unwrap().to_traceparent();
    {
        let _scope = workflow.enter();
        let models = xybrid_sdk::RegistryClient::with_url(registry.base_url())
            .unwrap()
            .list_models()
            .unwrap();
        assert!(models.is_empty());

        use xybrid_core::orchestrator::authority::{
            DecisionSource, OrchestrationAuthority, RemoteAuthority, StageContext,
        };
        let context = StageContext {
            stage_id: "draft".into(),
            model_id: "hop-model".into(),
            input_kind: EnvelopeKind::Text("hi".into()),
            metrics: xybrid_core::context::DeviceMetrics::default(),
            resource_monitor: xybrid_core::device::ResourceMonitor::global(),
            explicit_target: None,
            local_availability: Some(
                xybrid_core::orchestrator::routing_engine::LocalAvailability::new(true),
            ),
            device_class: None,
            device_class_schema_version: None,
        };
        let decision = RemoteAuthority::new(&advice.base_url()).resolve_target(&context);
        assert_eq!(decision.source, DecisionSource::Remote);
    }
    workflow.end();
    assert_eq!(seen_by(&registry), [Some(expected.clone())]);
    assert_eq!(seen_by(&advice), [Some(expected)]);
}

#[test]
fn a_gateway_retry_is_a_new_attempt_with_its_own_traceparent() {
    let _serial = serial();
    let platform = MockServer::start();
    platform.mock(|when, then| {
        when.matches(first_try);
        then.status(503)
            .header("content-type", "application/json")
            .body(r#"{"error":{"message":"busy"}}"#);
    });
    platform.mock(|_, then| {
        then.status(200)
            .header("content-type", "application/json")
            .body(COMPLETION);
    });
    xybrid_sdk::set_gateway_url(platform.url("/v1"));
    let traced = Traced::new(&[]);

    let result = run(
        "    provider: openai\n    api_key: platform-test-key\n",
        &traced.options(),
    );
    assert_eq!(result.text(), Some("ok"));

    let attempts = attempts(&traced.memory);
    let seen = seen_by(&platform);
    assert_eq!(attempts.len(), 2, "{attempts:?}");
    assert_eq!(seen.len(), 2);
    assert_ne!(seen[0], seen[1], "each try has its own parent span");
    assert_eq!(seen[0], Some(traceparent_of(&attempts[0])));
    assert_eq!(seen[1], Some(traceparent_of(&attempts[1])));
    assert_eq!(
        attr(&attempts[0], semconv::attr::ATTEMPT_REASON).as_deref(),
        Some("initial")
    );
    assert_eq!(
        attr(&attempts[0], semconv::attr::ERROR_TYPE).as_deref(),
        Some("gateway_error")
    );
    assert_eq!(
        attr(&attempts[1], semconv::attr::ATTEMPT_REASON).as_deref(),
        Some("retry")
    );
    assert_eq!(attr(&attempts[1], semconv::attr::ERROR_TYPE), None);
    // The failed try ends when it failed and the retry starts when it is
    // re-sent: core's backoff before the first retry (at least one second)
    // belongs to neither attempt.
    let backoff = attempts[1]
        .start_time
        .duration_since(attempts[0].end_time)
        .expect("the retry starts after the failed try ends");
    assert!(backoff >= Duration::from_millis(500), "{backoff:?}");
    let stage = traced
        .memory
        .spans()
        .into_iter()
        .find(|s| s.span_context.span_id() == attempts[0].parent_span_id)
        .unwrap();
    assert_eq!(
        attr(&stage, semconv::attr::ATTEMPT_COUNT).as_deref(),
        Some("2")
    );
}

/// The legacy `run_pipeline[_async]` entry points take no options, so they
/// report to the default session. Each run is one workflow whose core
/// `pipeline:` span closes inside it (never a `cancelled` span ending after
/// it), and each stage is a stage inference with its attempt.
///
/// A legacy config names stages only (no provider, model or target) and its
/// availability hints are ignored, so every stage routes to the cloud adapter,
/// which rejects it before any request ("Missing 'provider' in metadata"):
/// legacy runs cannot succeed, and the defect showed on every async run.
#[test]
fn legacy_pipeline_runs_close_their_pipeline_span_and_observe_stages() {
    let _serial = serial();
    let platform = recording(COMPLETION);
    xybrid_sdk::set_gateway_url(platform.url("/v1"));
    let traced = Traced::new(&[]);
    let previous = observability::install_default_session(traced.session.clone());
    let config = tempfile::tempdir().unwrap();
    let path = config.path().join("legacy.yaml");
    std::fs::write(
        &path,
        "name: legacy-hops\nstages:\n  - hop-model\ninput:\n  kind: Text\navailability: {}\n",
    )
    .unwrap();
    let path = path.to_str().unwrap();

    let sync_run = xybrid_sdk::run_pipeline(path);
    let async_run = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(xybrid_sdk::run_pipeline_async(path));
    match previous {
        Some(previous) => drop(observability::install_default_session(previous)),
        None => observability::shutdown_observability(Duration::ZERO).unwrap(),
    }
    for run in [&sync_run, &async_run] {
        let error = run.as_ref().map(|_| ()).unwrap_err().to_string();
        assert!(error.contains("Missing 'provider'"), "{error}");
    }

    let spans = traced.memory.spans();
    let workflows: Vec<&SpanData> = spans
        .iter()
        .filter(|s| s.name == "invoke_workflow legacy-hops")
        .collect();
    assert_eq!(workflows.len(), 2, "one workflow per run: {spans:?}");
    for workflow in workflows {
        let label = format!("workflow {}", workflow.span_context.trace_id());
        let in_run: Vec<&SpanData> = spans
            .iter()
            .filter(|s| s.span_context.trace_id() == workflow.span_context.trace_id())
            .collect();
        for span in &in_run {
            assert_ne!(
                attr(span, semconv::attr::ERROR_TYPE).as_deref(),
                Some("cancelled"),
                "{label}: {} was abandoned",
                span.name
            );
            assert!(
                span.end_time <= workflow.end_time,
                "{label}: {} ends after its workflow",
                span.name
            );
        }
        let pipeline = in_run
            .iter()
            .find(|s| s.name.starts_with("pipeline:"))
            .unwrap_or_else(|| panic!("{label}: pipeline span exported: {in_run:?}"));
        assert_eq!(
            pipeline.parent_span_id,
            workflow.span_context.span_id(),
            "{label}"
        );
        let stage = in_run
            .iter()
            .find(|s| {
                attr(s, "xybrid.stage.name").as_deref() == Some("hop-model")
                    && attr(s, semconv::attr::OPERATION_NAME).is_some()
            })
            .unwrap_or_else(|| panic!("{label}: stage inference exported: {in_run:?}"));
        assert_eq!(
            stage.parent_span_id,
            workflow.span_context.span_id(),
            "{label}"
        );
        assert_eq!(
            attr(stage, semconv::attr::ERROR_TYPE).as_deref(),
            Some("pipeline_error"),
            "{label}"
        );
        let attempts = in_run
            .iter()
            .filter(|s| {
                s.parent_span_id == stage.span_context.span_id()
                    && attr(s, semconv::attr::ATTEMPT_INDEX).is_some()
            })
            .count();
        assert_eq!(attempts, 1, "{label}: the stage's attempt is exported");
    }
    assert!(seen_by(&platform).is_empty(), "rejected before any request");
}

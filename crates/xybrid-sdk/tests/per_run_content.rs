//! Per-run content overrides against the process default: a per-run session or observer can lower content capture
//! below the default's policy, never raise it above it.
//!
//! Its own test binary: the default session is process-wide, and the
//! restrictive defaults installed here would cap the per-run sessions of the
//! tests running beside it (the tests here take turns through [`DEFAULT`]).
//! The run is a pipeline stage served by a loopback chat gateway, so the SDK
//! records the stage's input and output itself.

use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanProcessor};
use serde_json::Value;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::observability::{
    self, semconv, ContentCapture, ContentPolicy, ObservabilityConfig, ObservabilitySession,
    ObservationOptions, ObservationRef, Observer,
};
use xybrid_sdk::{PipelineRef, RunOptions};

const STAGE_INPUT: &str = "summarise the logbook";
const COMPLETION: &str = r#"{"choices":[{"message":{"role":"assistant","content":"calm seas all week"},"finish_reason":"stop"}]}"#;

/// Held by each test for its whole run: the process default is shared.
static DEFAULT: Mutex<()> = Mutex::new(());

/// One test's turn at the process default. Ending it uninstalls whatever
/// default the test left, a failing test included, so the next test starts
/// with none.
struct DefaultTurn {
    _lock: MutexGuard<'static, ()>,
}

impl DefaultTurn {
    fn take() -> Self {
        Self {
            _lock: DEFAULT.lock().unwrap_or_else(PoisonError::into_inner),
        }
    }
}

impl Drop for DefaultTurn {
    fn drop(&mut self) {
        let _ = observability::shutdown_observability(Duration::ZERO);
    }
}

/// A loopback chat gateway that answers every completion with [`COMPLETION`].
fn gateway() -> httpmock::MockServer {
    let gateway = httpmock::MockServer::start();
    gateway.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .body(COMPLETION);
    });
    gateway
}

#[derive(Clone, Debug, Default)]
struct Memory(Arc<Mutex<Vec<SpanData>>>);

impl SpanProcessor for Memory {
    fn on_start(&self, _: &mut opentelemetry_sdk::trace::Span, _: &opentelemetry::Context) {}
    fn on_end(&self, span: SpanData) {
        self.0.lock().unwrap().push(span);
    }
    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }
    fn shutdown_with_timeout(&self, _: Duration) -> OTelSdkResult {
        Ok(())
    }
}

/// A session over in-memory span and log providers whose observer records
/// content under `policy`.
struct Signals {
    session: ObservabilitySession,
    spans: Memory,
    logs: InMemoryLogExporter,
    _tracer: SdkTracerProvider,
    _logger: SdkLoggerProvider,
}

impl Signals {
    fn new(policy: ContentPolicy) -> Self {
        let spans = Memory::default();
        let tracer = SdkTracerProvider::builder()
            .with_span_processor(spans.clone())
            .build();
        let logs = InMemoryLogExporter::default();
        let logger = SdkLoggerProvider::builder()
            .with_simple_exporter(logs.clone())
            .build();
        let observer = Observer::builder()
            .tracer_provider(&tracer)
            .logger_provider(&logger)
            .content_policy(policy)
            .build();
        Self {
            session: ObservabilitySession::new(ObservabilityConfig::with_observer(observer))
                .unwrap(),
            spans,
            logs,
            _tracer: tracer,
            _logger: logger,
        }
    }

    /// Where the stage of `reference` recorded its input: on its span, on its
    /// details event.
    fn input_destinations(&self, reference: &ObservationRef) -> (bool, bool) {
        assert!(
            self.has_span(reference),
            "the stage reports to this session"
        );
        let spans = self.spans.0.lock().unwrap();
        let span = spans
            .iter()
            .find(|s| s.span_context.span_id().to_string() == reference.span_id)
            .expect("the span was just found");
        let on_span = span.attributes.iter().any(|kv| {
            kv.key.as_str() == semconv::attr::INPUT_MESSAGES
                && kv.value.as_str().contains(STAGE_INPUT)
        });
        let on_event = self.logs.get_emitted_logs().unwrap().iter().any(|log| {
            log.record.event_name() == Some(semconv::event::INFERENCE_DETAILS)
                && log
                    .record
                    .trace_context()
                    .is_some_and(|t| t.span_id.to_string() == reference.span_id)
                && format!("{:?}", log.record).contains(STAGE_INPUT)
        });
        (on_span, on_event)
    }

    /// Whether this session recorded the span of `reference`.
    fn has_span(&self, reference: &ObservationRef) -> bool {
        self.spans
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|s| s.span_context.span_id().to_string() == reference.span_id)
    }

    /// Whether any span or log record of this session carries the input.
    fn holds_input(&self) -> bool {
        let on_spans = self.spans.0.lock().unwrap().iter().any(|s| {
            s.attributes
                .iter()
                .any(|kv| kv.value.as_str().contains(STAGE_INPUT))
        });
        let on_logs = self
            .logs
            .get_emitted_logs()
            .unwrap()
            .iter()
            .any(|log| format!("{:?}", log.record).contains(STAGE_INPUT));
        on_spans || on_logs
    }
}

/// A pipeline run of [`STAGE_INPUT`] through one cloud chat stage served by
/// `gateway`, with `observation` as its per-run options. Returns the stage's
/// reference, if any.
fn run(gateway: &httpmock::MockServer, observation: ObservationOptions) -> Option<ObservationRef> {
    let yaml = format!(
        "name: per-run-content\nstages:\n  - id: draft\n    model: loopback-model\n    target: cloud\n    provider: openai\n    gateway_url: \"{}\"\n    api_key: local-test-only\n    timeout_ms: 20000\n",
        gateway.url("/v1")
    );
    let result = PipelineRef::from_yaml(&yaml)
        .unwrap()
        .load()
        .unwrap()
        .run_with_options(
            &Envelope::new(EnvelopeKind::Text(STAGE_INPUT.into())),
            &RunOptions::new().with_observation(observation),
        )
        .expect("the gateway answers");
    result.stages[0].observation_ref.clone()
}

/// [`run`], returning the stage's reference.
fn run_stage(gateway: &httpmock::MockServer, observation: ObservationOptions) -> ObservationRef {
    run(gateway, observation).expect("the stage is observed")
}

/// [`run`] inside an enclosing operation that observes nothing: its
/// observer is disabled.
fn run_inside_an_unobserved_operation(
    gateway: &httpmock::MockServer,
    observation: ObservationOptions,
) {
    let unobserved = ObservationOptions {
        observer: Some(Observer::disabled()),
        ..ObservationOptions::default()
    };
    let outer = observability::start_workflow("unobserved", &unobserved);
    let _entered = outer.enter();
    run(gateway, observation);
}

#[test]
fn a_per_run_session_cannot_raise_content_capture_above_the_default() {
    let _turn = DefaultTurn::take();
    let gateway = gateway();
    let everything =
        Signals::new(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent));
    let per_run = || ObservationOptions::default().with_session(&everything.session);

    // Control: with no default, the per-run session's own policy applies.
    let alone = run_stage(&gateway, per_run());
    assert_eq!(everything.input_destinations(&alone), (true, true));
    // So does it inside an operation that observes nothing, which suppresses
    // the stage's span but not its details event.
    let nested = Signals::new(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent));
    run_inside_an_unobserved_operation(
        &gateway,
        ObservationOptions::default().with_session(&nested.session),
    );
    assert!(nested.holds_input(), "the control records the input");

    // Under a metadata-only default it records no content at all.
    let metadata_only = Signals::new(ContentPolicy::metadata_only());
    observability::init_observability(ObservabilityConfig::with_observer(
        metadata_only.session.observer().clone(),
    ))
    .unwrap();
    // A run with no per-run session reports to the default.
    let defaulted = run_stage(&gateway, ObservationOptions::default());
    assert!(metadata_only.has_span(&defaulted) && !everything.has_span(&defaulted));
    let floored = run_stage(&gateway, per_run());
    assert_eq!(everything.input_destinations(&floored), (false, false));
    // Nor does a per-run observer.
    let observer = ObservationOptions {
        observer: Some(everything.session.observer().clone()),
        ..ObservationOptions::default()
    };
    let floored_observer = run_stage(&gateway, observer);
    assert_eq!(
        everything.input_destinations(&floored_observer),
        (false, false)
    );
    // Nor inside an operation that observes nothing: a disabled enclosing
    // session lifts no cap, so the default's still applies.
    let nested = Signals::new(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent));
    run_inside_an_unobserved_operation(
        &gateway,
        ObservationOptions::default().with_session(&nested.session),
    );
    assert!(!nested.holds_input(), "the metadata-only default caps it");

    // Under a span-only default it records only what both capture: the span.
    let span_only = Signals::new(ContentPolicy::default().with_capture(ContentCapture::SpanOnly));
    observability::install_default_session(span_only.session.clone());
    let intersected = run_stage(&gateway, per_run());
    assert_eq!(everything.input_destinations(&intersected), (true, false));

    // A default whose observer exports nothing caps nothing.
    observability::install_default_session(ObservabilitySession::borrowed(Observer::disabled()));
    let disabled_default = run_stage(&gateway, per_run());
    assert_eq!(
        everything.input_destinations(&disabled_default),
        (true, true)
    );

    // The runs went to the per-run session, never to the defaults.
    for reference in [&floored, &floored_observer, &intersected] {
        assert!(everything.has_span(reference));
        assert!(!metadata_only.has_span(reference) && !span_only.has_span(reference));
    }
    assert!(!span_only.holds_input());
    observability::shutdown_observability(Duration::ZERO).unwrap();
}

#[test]
fn a_per_run_observer_and_the_default_each_redact_a_value_once() {
    // The pipeline's stage re-caps the per-run observer by the pipeline's
    // session, whose policy that observer already shaped: its hook still
    // runs once per value, like the default's.
    let _turn = DefaultTurn::take();
    let gateway = gateway();
    let recording = |calls: &Arc<Mutex<Vec<String>>>| {
        let calls = calls.clone();
        move |kind: &str, value: Value| {
            calls.lock().unwrap().push(kind.to_owned());
            Some(value)
        }
    };
    let (default_calls, per_run_calls) = (Arc::default(), Arc::default());
    let default = Signals::new(
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_redactor(recording(&default_calls)),
    );
    observability::init_observability(ObservabilityConfig::with_observer(
        default.session.observer().clone(),
    ))
    .unwrap();
    let per_run = Signals::new(
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_redactor(recording(&per_run_calls)),
    );

    let stage = run_stage(
        &gateway,
        ObservationOptions {
            observer: Some(per_run.session.observer().clone()),
            ..ObservationOptions::default()
        },
    );
    assert_eq!(per_run.input_destinations(&stage), (true, true));
    let default_calls = default_calls.lock().unwrap().clone();
    assert!(
        default_calls
            .iter()
            .any(|kind| kind == semconv::attr::INPUT_MESSAGES),
        "{default_calls:?}"
    );
    assert_eq!(
        *per_run_calls.lock().unwrap(),
        default_calls,
        "each hook redacts each value once"
    );
    observability::shutdown_observability(Duration::ZERO).unwrap();
}

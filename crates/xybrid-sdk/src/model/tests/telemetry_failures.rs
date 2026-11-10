//! How SDK runs are observed when the network or telemetry around them fails
//! (reliability B1–B4).
//!
//! - A reactive local→cloud fallback is one logical request: a failed local
//!   attempt, then a remote `fallback` attempt whose context the gateway
//!   receives, with duration and usage counted once on the request.
//! - Model runs, streams and their token callbacks, and pipeline cloud-stage
//!   runs, never wait on an export the endpoint has not answered, nor on the
//!   end-of-run content hook; a failing hook fails closed and leaves the
//!   stream alone.
//! - Disposal (session shutdown, shutting the default session down, dropping
//!   a session, replacing the default session, unloading or dropping a model)
//!   never waits on that export, and leaves no export open behind it.
//!
//! The endpoint is a [`ParkedCollector`]: it reads each export and answers
//! none until the test releases it, after the call under test returned, and
//! records when a client closes an export it parked. A run or disposal that
//! waited on the export could never return, so every wait here is a hang
//! detector whose expiry is the failure, never a latency figure.

use super::*;
use crate::observability::semconv;
use crate::observability::test_support::{
    attr, default_session_lock, memory_session_with_signals_configured, parent_id, span_id,
    trace_id, DefaultSession, Point, Signals,
};
use crate::observability::{
    init_observability, shutdown_observability, start_workflow, ContentCapture, ContentPolicy,
    ObservabilityConfig, ObservabilitySession, ObservationOptions, Observer, Redactor,
};
use crate::run_options::{AbortPolicy, AbortSignal};
use opentelemetry_sdk::trace::SpanData;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::Condvar;
use xybrid_core::runtime_adapter::types::PartialToken;
use xybrid_observability::otlp::{
    BatchLimits, EnvironmentSource, OtlpConfig, Protocol, RetryLimits, SignalEndpoint,
};

/// Bound of every hang detector. Far above anything these calls take, far
/// below the parked export's own deadline ([`EXPORT_DEADLINE`]).
const HANG: Duration = Duration::from_secs(20);

/// The parked export's deadline: nothing but the test (or a shutdown) ends it.
const EXPORT_DEADLINE: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------------------
// A collector that parks every export until the test releases it.
// ---------------------------------------------------------------------------

/// What became of one request the collector read.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fate {
    /// Read and unanswered, its connection open.
    Parked,
    /// Answered after the test released the collector.
    Answered,
    /// Closed by the client while unanswered.
    Closed,
}

#[derive(Default)]
struct Shared {
    /// Path and fate of each request read so far, in arrival order.
    requests: Vec<(String, Fate)>,
    /// Whether the test released the collector.
    released: bool,
}

#[derive(Default)]
struct Parked {
    state: Mutex<Shared>,
    changed: Condvar,
    stop: AtomicBool,
}

impl Parked {
    /// Fate of the first request to `path`, once it was read.
    fn fate(state: &Shared, path: &str) -> Option<Fate> {
        state
            .requests
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, fate)| *fate)
    }

    /// Waits until the first request to `path` is no longer parked and
    /// returns what became of it. The bound only detects one left open.
    fn settled(&self, path: &str) -> Option<Fate> {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, HANG, |state| {
                matches!(Self::fate(state, path), None | Some(Fate::Parked))
            })
            .unwrap();
        Self::fate(&state, path)
    }

    /// Waits until the client closed the parked request to `path` unanswered.
    fn wait_closed(&self, path: &str) {
        let fate = self.settled(path);
        assert_eq!(fate, Some(Fate::Closed), "the export to {path} stayed open");
    }
}

/// An OTLP/HTTP JSON endpoint that reads each request in full and answers
/// none until [`ParkedCollector::release`]; from then on every request still
/// parked, or later, gets `200 {}`. It records when a client closes a parked
/// request. Dropping it releases it.
struct ParkedCollector {
    url: String,
    shared: Arc<Parked>,
    accept: Option<std::thread::JoinHandle<()>>,
}

impl ParkedCollector {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback collector");
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let shared = Arc::new(Parked::default());
        let accepting = shared.clone();
        let accept = std::thread::spawn(move || {
            while !accepting.stop.load(std::sync::atomic::Ordering::SeqCst) {
                match listener.accept() {
                    Ok((socket, _)) => {
                        let shared = accepting.clone();
                        std::thread::spawn(move || serve(&shared, socket));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => panic!("collector accept: {e}"),
                }
            }
        });
        Self {
            url,
            shared,
            accept: Some(accept),
        }
    }

    /// Traces, logs and metrics to this collector, without retries. Every
    /// deadline is [`EXPORT_DEADLINE`]; metrics are exported only on flush or
    /// shutdown.
    fn config(&self, batch: BatchLimits) -> OtlpConfig {
        let endpoint = |signal: &str| {
            Some(
                SignalEndpoint::new(format!("{}/v1/{signal}", self.url))
                    .with_protocol(Protocol::HttpJson)
                    .with_timeout(EXPORT_DEADLINE),
            )
        };
        OtlpConfig::default()
            .with_environment(EnvironmentSource::Disabled)
            .with_traces(endpoint("traces"))
            .with_logs(endpoint("logs"))
            .with_metrics(endpoint("metrics"))
            .with_batch(BatchLimits {
                export_timeout: EXPORT_DEADLINE,
                ..batch
            })
            .with_retry(RetryLimits {
                max_retries: 0,
                ..RetryLimits::default()
            })
            .with_metric_export_interval(Duration::from_secs(600))
    }

    /// Waits until a request to `path` is parked here.
    fn wait_parked(&self, path: &str) {
        let state = self.shared.state.lock().unwrap();
        let (state, wait) = self
            .shared
            .changed
            .wait_timeout_while(state, HANG, |state| Parked::fate(state, path).is_none())
            .unwrap();
        assert!(
            !wait.timed_out(),
            "no export to {path}: {:?}",
            state.requests
        );
    }

    /// Whether the first request to `path` is still parked: unanswered, its
    /// connection open.
    fn is_parked(&self, path: &str) -> bool {
        Parked::fate(&self.shared.state.lock().unwrap(), path) == Some(Fate::Parked)
    }

    /// Waits until the client closed the parked request to `path` unanswered.
    fn wait_closed(&self, path: &str) {
        self.shared.wait_closed(path);
    }

    /// A handle that outlives a move into a helper thread.
    fn watcher(&self) -> Arc<Parked> {
        self.shared.clone()
    }

    fn release(&self) {
        self.shared.state.lock().unwrap().released = true;
        self.shared.changed.notify_all();
    }
}

impl Drop for ParkedCollector {
    fn drop(&mut self) {
        self.release();
        self.shared
            .stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
    }
}

/// Reads one request, parks it until the collector is released or the client
/// closes it, and answers it if still open.
fn serve(shared: &Arc<Parked>, mut socket: TcpStream) {
    let _ = socket.set_nonblocking(false);
    let _ = socket.set_read_timeout(Some(HANG));
    let Some(path) = read_request(&mut socket) else {
        return;
    };
    let index = {
        let mut state = shared.state.lock().unwrap();
        state.requests.push((path, Fate::Parked));
        shared.changed.notify_all();
        state.requests.len() - 1
    };
    if let Ok(probe) = socket.try_clone() {
        let shared = shared.clone();
        std::thread::spawn(move || watch_close(&shared, probe, index));
    }
    // Bounded only so that a test failing before it releases the collector
    // leaves no thread behind.
    let state = shared.state.lock().unwrap();
    let (mut state, _) = shared
        .changed
        .wait_timeout_while(state, EXPORT_DEADLINE, |state| {
            !state.released && state.requests[index].1 == Fate::Parked
        })
        .unwrap();
    let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
    if state.released
        && state.requests[index].1 == Fate::Parked
        && socket.write_all(response).is_ok()
    {
        state.requests[index].1 = Fate::Answered;
        shared.changed.notify_all();
    }
}

/// Marks request `index` closed when its client closes the connection while
/// it is still parked.
fn watch_close(shared: &Parked, mut probe: TcpStream, index: usize) {
    let mut byte = [0_u8; 1];
    loop {
        match probe.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                // The read timeout only lets this thread notice the end.
                let answered = shared.state.lock().unwrap().requests[index].1 != Fate::Parked;
                if answered || shared.stop.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
            }
            Err(_) => break,
        }
    }
    let mut state = shared.state.lock().unwrap();
    if state.requests[index].1 == Fate::Parked {
        state.requests[index].1 = Fate::Closed;
        shared.changed.notify_all();
    }
}

/// The path of one complete HTTP request, or `None` if the client left first.
fn read_request(socket: &mut TcpStream) -> Option<String> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 16 * 1024];
    loop {
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&bytes[..end]).into_owned();
            let length = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.trim()
                        .eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                return head.split_whitespace().nth(1).map(str::to_owned);
            }
        }
        match socket.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(n) => bytes.extend_from_slice(&chunk[..n]),
        }
    }
}

/// One item per signal fills the queue, and one item is a full batch: the
/// first span is exported at once and parks, and while it is parked every
/// later span is dropped at admission (the queue counts in-flight items).
fn parking_batch() -> BatchLimits {
    BatchLimits {
        max_queue_size: 1,
        max_export_batch_size: 1,
        scheduled_delay: Duration::from_secs(60),
        ..BatchLimits::default()
    }
}

/// An OTLP session whose trace export is parked at `collector`.
fn parked_session(
    collector: &ParkedCollector,
    content: Option<ContentPolicy>,
) -> ObservabilitySession {
    let mut config = ObservabilityConfig::otlp(collector.config(parking_batch()));
    if let Some(policy) = content {
        config = config.content(policy);
    }
    let session = ObservabilitySession::new(config).expect("a valid OTLP session");
    start_workflow(
        "parked",
        &ObservationOptions::default().with_session(&session),
    )
    .end();
    collector.wait_parked("/v1/traces");
    session
}

/// Runs `work` on its own thread and returns its result. The bound only
/// detects a call that never returns; its expiry is the failure.
fn returns<T: Send + 'static>(what: &str, work: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    match rx.recv_timeout(HANG) {
        Ok(value) => value,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what} never returned: it waited on the parked export")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{what} panicked"),
    }
}

// ---------------------------------------------------------------------------
// Fakes.
// ---------------------------------------------------------------------------

/// Fake `"onnx"` runtime that answers every request with `text`.
struct TextRuntime(&'static str);

impl xybrid_core::runtime_adapter::ModelRuntime for TextRuntime {
    fn name(&self) -> &str {
        "onnx"
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["onnx"]
    }

    fn load(
        &mut self,
        _model_path: &std::path::Path,
    ) -> xybrid_core::runtime_adapter::AdapterResult<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(
        &mut self,
        _input: &xybrid_core::ir::Envelope,
    ) -> xybrid_core::runtime_adapter::AdapterResult<xybrid_core::ir::Envelope> {
        Ok(xybrid_core::ir::Envelope::new(
            xybrid_core::ir::EnvelopeKind::Text(self.0.to_string()),
        ))
    }
}

/// A cloud leg that streams `tokens`, one callback each.
struct TokensCloud(&'static [&'static str]);

impl xybrid_core::runtime_adapter::CloudStreaming for TokensCloud {
    fn execute_streaming(
        &self,
        _input: &xybrid_core::ir::Envelope,
        mut on_token: xybrid_core::runtime_adapter::types::StreamingCallback<'_>,
    ) -> xybrid_core::runtime_adapter::AdapterResult<xybrid_core::ir::Envelope> {
        let mut text = String::new();
        for (index, token) in self.0.iter().enumerate() {
            text.push_str(token);
            on_token(PartialToken {
                token: (*token).to_string(),
                token_id: None,
                index,
                cumulative_text: text.clone(),
                finish_reason: (index + 1 == self.0.len()).then(|| "stop".to_string()),
                tool_calls: Vec::new(),
                raw_text: None,
            })
            .map_err(|e| {
                xybrid_core::runtime_adapter::AdapterError::InferenceFailed(e.to_string())
            })?;
        }
        Ok(xybrid_core::ir::Envelope::new(
            xybrid_core::ir::EnvelopeKind::Text(text),
        ))
    }
}

const CLOUD_TOKENS: &[&str] = &["one ", "two ", "three ", "four ", "five"];

/// Options pinned to `session` under which critical memory pressure stops the
/// local leg before it runs, so a fallback run streams [`CLOUD_TOKENS`]. The
/// tests that use it assert no attempts: such a run records only its remote
/// attempt today, and whether it should also record a failed local one is
/// still open.
fn pressured(session: &ObservabilitySession) -> RunOptions {
    let mut critical = xybrid_core::device::ResourceSnapshot::unknown();
    critical.memory_pressure = xybrid_core::device::MemoryPressure::Critical;
    RunOptions::new()
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true),
        )
        .with_resource_provider(Arc::new(FixedUnitResourceProvider::new(critical)))
        .with_observation(ObservationOptions::default().with_session(session))
}

/// What happened during a streamed run, in order.
#[derive(Clone, Debug, PartialEq)]
enum Step {
    Redact(String),
    Token(String),
}

// ---------------------------------------------------------------------------
// B1: the reactive local→cloud fallback, observed.
// ---------------------------------------------------------------------------

/// `(host, traceparent, tracestate)` of one request the B1 gateway received.
type SeenRequest = (String, Option<String>, Option<String>);

/// Every request the B1 gateway received.
static GATEWAY_SEEN: Mutex<Vec<SeenRequest>> = Mutex::new(Vec::new());

fn header(request: &httpmock::prelude::HttpMockRequest, name: &str) -> Option<String> {
    request.headers.as_ref().and_then(|headers| {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    })
}

/// Records the request's trace headers; always matches.
fn seen_by_gateway(request: &httpmock::prelude::HttpMockRequest) -> bool {
    GATEWAY_SEEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push((
            header(request, "host").unwrap_or_default(),
            header(request, "traceparent"),
            header(request, "tracestate"),
        ));
    true
}

/// The gateway's OpenAI-compatible SSE answer: two tokens, then usage.
const GATEWAY_SSE: &str = concat!(
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"local-test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"local-test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"cloud\"},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"local-test-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n",
    "data: [DONE]\n\n",
);

fn traceparent_of(span: &SpanData) -> String {
    format!("00-{}-{}-01", trace_id(span), span_id(span))
}

/// Sum of every point of `name` (a counter's value, a histogram's count).
fn total(signals: &Signals, name: &str) -> u64 {
    signals.points(name).iter().map(|p| p.count).sum()
}

#[test]
fn a_reactive_fallback_is_one_request_with_a_failed_local_and_a_remote_attempt() {
    use opentelemetry::trace::{SpanKind, Status};

    const CALLER: &str = "00-5bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let gateway = httpmock::MockServer::start();
    gateway.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/chat/completions")
            .matches(seen_by_gateway);
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(GATEWAY_SSE);
    });
    // A loopback gateway is not the configured platform gateway: it receives
    // the run's context once the session allowlists its origin.
    let signals = memory_session_with_signals_configured(|config| {
        config
            .propagate_context_to(&gateway.base_url())
            .expect("a loopback HTTP origin")
    });
    let model = as_chat(test_model_with_runtime(Box::new(TextRuntime(
        "local draft",
    ))));
    let options = RunOptions::new()
        .with_abort_policy(AbortPolicy::default().with_cloud_fallback(true))
        .with_observation(
            ObservationOptions::from_traceparent(CALLER, Some("vendor=a"))
                .unwrap()
                .with_session(&signals.session),
        );
    // The local leg aborts for cloud fallback at its first token: the token
    // callback returns the typed `CloudFallbackAbort`, which is what the
    // resource gate wrapped around it returns mid-stream, so the local
    // attempt ends the same way. The gate itself is not used: it samples
    // resources at most every 100 ms, so tripping it mid-stream would need
    // the test to wait. (A gate that trips before the run records no local
    // attempt at all; that path is not pinned here.)
    let delivered = Arc::new(Mutex::new(Vec::new()));
    let seen = delivered.clone();
    let mut on_token = move |token: PartialToken| {
        let mut seen = seen.lock().unwrap();
        seen.push(token.token);
        if seen.len() == 1 {
            return Err(Box::new(xybrid_core::abort::CloudFallbackAbort::new(
                xybrid_core::abort::AbortReason::StressMemory,
            )) as Box<dyn std::error::Error + Send + Sync>);
        }
        Ok(())
    };
    let seams = Arc::new(AtomicUsize::new(0));
    let seam_count = seams.clone();
    let cloud =
        xybrid_core::runtime_adapter::CloudRuntimeAdapter::with_gateway(&gateway.url("/v1"));

    let result = model
        .run_streaming_with_fallback(
            &text_envelope("write me a haiku"),
            &options,
            &cloud,
            &mut on_token,
            &mut |_seam: SeamInfo| {
                seam_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            },
        )
        .expect("the cloud fallback answers");

    assert_eq!(result.text(), Some("hello cloud"));
    assert_eq!(seams.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        delivered.lock().unwrap().as_slice(),
        ["local draft", "hello ", "cloud"]
    );

    // One logical request under the caller, with both attempts beneath it.
    let caller = crate::observability::ObservationContext::from_traceparent(CALLER, None).unwrap();
    let request = signals.spans.named(&run_span_name(&model));
    assert_eq!(parent_id(&request), caller.span_id());
    assert_eq!(trace_id(&request), caller.trace_id());
    assert!(
        !matches!(request.status, Status::Error { .. }),
        "{:?}",
        request.status
    );
    assert_eq!(attr(&request, semconv::attr::ERROR_TYPE), None);
    assert_eq!(
        attr(&request, semconv::attr::ATTEMPT_COUNT).as_deref(),
        Some("2")
    );
    let local = signals.spans.named("attempt 0");
    let remote = signals.spans.named("attempt 1");
    for attempt in [&local, &remote] {
        assert_eq!(parent_id(attempt), span_id(&request), "{}", attempt.name);
    }
    let described = |span: &SpanData| {
        (
            attr(span, semconv::attr::ATTEMPT_REASON),
            attr(span, semconv::attr::EXECUTION_LOCATION),
            attr(span, semconv::attr::ERROR_TYPE),
        )
    };
    assert_eq!(
        described(&local),
        (
            Some("initial".into()),
            Some("local".into()),
            Some("aborted_for_cloud_fallback".into())
        )
    );
    assert!(
        matches!(local.status, Status::Error { .. }),
        "{:?}",
        local.status
    );
    assert_eq!(
        described(&remote),
        (Some("fallback".into()), Some("remote".into()), None)
    );
    assert_eq!(remote.span_kind, SpanKind::Client);
    assert!(
        !matches!(remote.status, Status::Error { .. }),
        "{:?}",
        remote.status
    );

    // The gateway request carried the remote attempt's context and the
    // caller's tracestate.
    let host = gateway.address().to_string();
    let requests: Vec<_> = GATEWAY_SEEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(h, _, _)| *h == host)
        .map(|(_, traceparent, tracestate)| (traceparent.clone(), tracestate.clone()))
        .collect();
    assert_eq!(
        requests,
        [(Some(traceparent_of(&remote)), Some("vendor=a".to_string()))]
    );

    // Duration and usage are counted once, on the request; attempts once each.
    let durations = signals.points(semconv::metric::INFERENCE_DURATION);
    assert_eq!(durations.len(), 1, "{durations:?}");
    assert_eq!(
        (durations[0].attr("error.type"), durations[0].count),
        (None, 1)
    );
    assert_eq!(total(&signals, semconv::metric::USAGE_INPUT_TOKENS), 7);
    assert_eq!(total(&signals, semconv::metric::USAGE_OUTPUT_TOKENS), 3);
    assert_eq!(total(&signals, semconv::metric::OPERATION_INPUT_TOKENS), 1);
    let mut attempts: Vec<_> = signals
        .points(semconv::metric::ATTEMPTS)
        .iter()
        .map(|p: &Point| {
            (
                p.attr(semconv::attr::ATTEMPT_REASON).map(str::to_owned),
                p.attr(semconv::attr::EXECUTION_LOCATION).map(str::to_owned),
                p.attr("error.type").map(str::to_owned),
                p.count,
            )
        })
        .collect();
    attempts.sort();
    assert_eq!(
        attempts,
        [
            (Some("fallback".into()), Some("remote".into()), None, 1),
            (
                Some("initial".into()),
                Some("local".into()),
                Some("aborted_for_cloud_fallback".into()),
                1
            ),
        ]
    );
}

// ---------------------------------------------------------------------------
// B2/B3: runs, streams and token callbacks never wait on export or on the
// end-of-run content hook.
// ---------------------------------------------------------------------------

/// A cloud stage's OpenAI-compatible JSON answer.
const STAGE_COMPLETION: &str = r#"{"choices":[{"message":{"role":"assistant","content":"stage output"},"finish_reason":"stop"}]}"#;

/// Pipeline cloud-stage runs per test: the first already meets a full queue.
const PIPELINE_RUNS: usize = 5;

#[test]
fn runs_streams_and_token_callbacks_complete_while_the_export_is_parked() {
    let collector = ParkedCollector::start();
    let session = parked_session(&collector, None);
    let model = as_chat(test_model_with_runtime(Box::new(TextRuntime(
        "local answer",
    ))));
    let gateway = httpmock::MockServer::start();
    gateway.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .body(STAGE_COMPLETION);
    });
    let pipeline = crate::pipeline::PipelineRef::from_yaml(&format!(
        "name: parked\nstages:\n  - id: draft\n    model: loopback-model\n    target: cloud\n    provider: openai\n    gateway_url: \"{}\"\n    api_key: local-test-only\n    timeout_ms: 20000\n",
        gateway.url("/v1")
    ))
    .unwrap()
    .load()
    .unwrap();

    let delivered = Arc::new(Mutex::new(Vec::new()));
    let worker = {
        let session = session.clone();
        let delivered = delivered.clone();
        move || {
            let options = RunOptions::new()
                .with_observation(ObservationOptions::default().with_session(&session));
            let stages: Vec<_> = (0..PIPELINE_RUNS)
                .map(|_| {
                    pipeline
                        .run_with_options(&text_envelope("stage"), &options)
                        .map(|result| result.text().map(str::to_owned))
                })
                .collect();
            let batch = model.run_with_options(&text_envelope("batch"), &options);
            let record = |token: PartialToken| {
                delivered.lock().unwrap().push(token.token);
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
            };
            let local = model.run_streaming_with_options(&text_envelope("local"), &options, record);
            let mut record = |token: PartialToken| {
                delivered.lock().unwrap().push(token.token);
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
            };
            let fallback = model.run_streaming_with_fallback(
                &text_envelope("fallback"),
                &pressured(&session),
                &TokensCloud(CLOUD_TOKENS),
                &mut record,
                &mut |_seam: SeamInfo| {},
            );
            // Disposal of the model touches no telemetry today; kept so that
            // a change that makes it wait on the session shows up here.
            let unloaded = model.unload();
            drop(model);
            (stages, batch, local, fallback, unloaded)
        }
    };
    let (stages, batch, local, fallback, unloaded) = returns("runs and streams", worker);

    // Every run finished, and every token reached its callback, while the
    // export they queued behind was still in flight and unanswered.
    assert!(
        collector.is_parked("/v1/traces"),
        "the parked export ended before the runs returned"
    );
    for stage in stages {
        assert_eq!(stage.unwrap().as_deref(), Some("stage output"));
    }
    assert_eq!(batch.unwrap().text(), Some("local answer"));
    assert_eq!(local.unwrap().text(), Some("local answer"));
    assert_eq!(
        fallback.unwrap().text(),
        Some(CLOUD_TOKENS.concat().as_str())
    );
    unloaded.unwrap();
    let mut expected = vec!["local answer"];
    expected.extend_from_slice(CLOUD_TOKENS);
    assert_eq!(delivered.lock().unwrap().as_slice(), expected.as_slice());
    // Their spans met a full queue and were dropped at admission, not waited
    // for: the parked span still holds the only slot.
    let traces = session.stats().expect("an owned pipeline").traces;
    assert_eq!(traces.pending, 1, "{traces:?}");
    assert!(traces.dropped > 0, "{traces:?}");

    collector.release();
    let _ = returns("shutdown", move || session.shutdown(HANG));
}

/// Holds the end-of-run output redaction until the test releases it.
struct OutputGate {
    /// Told when the output redaction starts.
    entered: Mutex<mpsc::Sender<()>>,
    /// Awaited before the output redaction returns.
    release: Mutex<mpsc::Receiver<()>>,
}

/// Logs each redaction as a [`Step`]. With a gate, blocks the end-of-run
/// output redaction until the test releases it; without one, fails every
/// value.
struct StepRedactor {
    steps: Arc<Mutex<Vec<Step>>>,
    gate: Option<OutputGate>,
}

impl Redactor for StepRedactor {
    fn redact(&self, kind: &str, value: serde_json::Value) -> Option<serde_json::Value> {
        self.steps
            .lock()
            .unwrap()
            .push(Step::Redact(kind.to_owned()));
        if let Some(gate) = &self.gate {
            if kind == semconv::attr::OUTPUT_MESSAGES {
                let _ = gate.entered.lock().unwrap().send(());
                // Bounded only so that a failed test leaves no thread behind.
                let _ = gate.release.lock().unwrap().recv_timeout(EXPORT_DEADLINE);
            }
        }
        Some(value)
    }

    fn try_redact(
        &self,
        kind: &str,
        value: serde_json::Value,
    ) -> Result<Option<serde_json::Value>, xybrid_observability::RedactionError> {
        if self.gate.is_none() {
            self.steps
                .lock()
                .unwrap()
                .push(Step::Redact(kind.to_owned()));
            return Err(xybrid_observability::RedactionError);
        }
        Ok(self.redact(kind, value))
    }
}

/// A streamed fallback run's token callback, logging each token as a [`Step`].
fn logging_tokens(
    steps: &Arc<Mutex<Vec<Step>>>,
) -> impl FnMut(PartialToken) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send {
    let steps = steps.clone();
    move |token: PartialToken| {
        steps.lock().unwrap().push(Step::Token(token.token));
        Ok(())
    }
}

#[test]
fn every_token_is_delivered_before_a_blocked_output_redaction_and_a_parked_export() {
    let (entered_tx, entered) = mpsc::channel();
    let (release, release_rx) = mpsc::channel();
    let steps = Arc::new(Mutex::new(Vec::new()));
    let policy = ContentPolicy::default()
        .with_capture(ContentCapture::SpanOnly)
        .with_redactor(StepRedactor {
            steps: steps.clone(),
            gate: Some(OutputGate {
                entered: Mutex::new(entered_tx),
                release: Mutex::new(release_rx),
            }),
        });
    let collector = ParkedCollector::start();
    let session = parked_session(&collector, Some(policy));
    let model = as_chat(test_loaded_model(true));
    let (done_tx, done) = mpsc::channel();
    {
        let session = session.clone();
        let mut on_token = logging_tokens(&steps);
        std::thread::spawn(move || {
            let result = model.run_streaming_with_fallback(
                &text_envelope("count to five"),
                &pressured(&session),
                &TokensCloud(CLOUD_TOKENS),
                &mut on_token,
                &mut |_seam: SeamInfo| {},
            );
            let _ = done_tx.send(result);
        });
    }

    // The run is inside the end-of-run output redaction: every token was
    // delivered before it.
    entered
        .recv_timeout(HANG)
        .expect("the run reaches its output redaction");
    let tokens = |steps: &[Step]| -> Vec<String> {
        steps
            .iter()
            .filter_map(|s| match s {
                Step::Token(t) => Some(t.clone()),
                Step::Redact(_) => None,
            })
            .collect()
    };
    assert_eq!(tokens(&steps.lock().unwrap()), CLOUD_TOKENS);
    release.send(()).unwrap();
    let result = done
        .recv_timeout(HANG)
        .expect("the run returns once released");
    assert_eq!(result.unwrap().text(), Some(CLOUD_TOKENS.concat().as_str()));
    assert!(
        collector.is_parked("/v1/traces"),
        "the parked export ended before the run returned"
    );

    // Content work brackets the stream: none of it runs between tokens.
    let steps = steps.lock().unwrap().clone();
    let first = steps
        .iter()
        .position(|s| matches!(s, Step::Token(_)))
        .unwrap();
    let last = steps
        .iter()
        .rposition(|s| matches!(s, Step::Token(_)))
        .unwrap();
    assert_eq!(last - first + 1, CLOUD_TOKENS.len(), "{steps:?}");
    assert!(
        steps[..first].contains(&Step::Redact(semconv::attr::INPUT_MESSAGES.into())),
        "{steps:?}"
    );
    assert_eq!(
        steps[last + 1..],
        [Step::Redact(semconv::attr::OUTPUT_MESSAGES.into())],
        "{steps:?}"
    );

    collector.release();
    let _ = returns("shutdown", move || session.shutdown(HANG));
}

#[test]
fn a_failing_redactor_fails_closed_and_leaves_the_stream_whole() {
    use opentelemetry_sdk::trace::SdkTracerProvider;

    let steps = Arc::new(Mutex::new(Vec::new()));
    let spans = crate::observability::test_support::Memory::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(spans.clone())
        .build();
    let observer = Observer::builder()
        .tracer_provider(&provider)
        .content_policy(
            ContentPolicy::default()
                .with_capture(ContentCapture::SpanOnly)
                .with_redactor(StepRedactor {
                    steps: steps.clone(),
                    gate: None,
                }),
        )
        .build();
    let session = ObservabilitySession::new(ObservabilityConfig::with_observer(observer)).unwrap();
    let model = as_chat(test_loaded_model(true));
    let mut on_token = logging_tokens(&steps);

    let result = model
        .run_streaming_with_fallback(
            &text_envelope("count to five"),
            &pressured(&session),
            &TokensCloud(CLOUD_TOKENS),
            &mut on_token,
            &mut |_seam: SeamInfo| {},
        )
        .expect("a failing redactor never fails the run");

    assert_eq!(result.text(), Some(CLOUD_TOKENS.concat().as_str()));
    let steps = steps.lock().unwrap().clone();
    let tokens: Vec<_> = steps
        .iter()
        .filter(|s| matches!(s, Step::Token(_)))
        .cloned()
        .collect();
    assert_eq!(
        tokens,
        CLOUD_TOKENS
            .iter()
            .map(|t| Step::Token((*t).to_string()))
            .collect::<Vec<_>>()
    );
    // Both values reached the hook, failed, and were left out and listed.
    for kind in [
        semconv::attr::INPUT_MESSAGES,
        semconv::attr::OUTPUT_MESSAGES,
    ] {
        assert!(
            steps.contains(&Step::Redact(kind.into())),
            "{kind}: {steps:?}"
        );
    }
    let request = spans.named(&run_span_name(&model));
    assert_eq!(attr(&request, semconv::attr::INPUT_MESSAGES), None);
    assert_eq!(attr(&request, semconv::attr::OUTPUT_MESSAGES), None);
    let truncated = attr(&request, semconv::attr::CONTENT_TRUNCATED_ATTRIBUTES).unwrap_or_default();
    for kind in [
        semconv::attr::INPUT_MESSAGES,
        semconv::attr::OUTPUT_MESSAGES,
    ] {
        assert!(truncated.contains(kind), "{kind}: {truncated}");
    }
}

// ---------------------------------------------------------------------------
// B4: disposal never waits on the parked export.
// ---------------------------------------------------------------------------

/// Wakes the thread that polls in [`poll_here`].
struct Unpark(std::thread::Thread);

impl std::task::Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Polls `future` on this thread until it is ready, or only once: `None`
/// when that poll left it pending.
fn poll_here<F: std::future::Future>(
    mut future: std::pin::Pin<&mut F>,
    once: bool,
) -> Option<F::Output> {
    let waker = std::task::Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return Some(output);
        }
        if once {
            return None;
        }
        std::thread::park();
    }
}

#[test]
fn session_shutdown_cuts_off_a_parked_export_at_its_deadline_and_returns_a_timeout() {
    let collector = ParkedCollector::start();
    let session = parked_session(&collector, None);

    // A shutdown call releases the transport only as it returns. Polled once
    // and left pending, this one has started; the parked export can then
    // close only because the shutdown's own deadline cut it off, not because
    // a fallback after it tore the transport down.
    let shutting = session.clone();
    let watched = collector.watcher();
    let result = returns("session shutdown", move || {
        let mut shutdown = std::pin::pin!(shutting.shutdown_async(Duration::from_millis(200)));
        // Ready at once only if this poll outlasted the whole drain.
        let early = poll_here(shutdown.as_mut(), true);
        watched.wait_closed("/v1/traces");
        early
            .or_else(|| poll_here(shutdown.as_mut(), false))
            .expect("polled until ready")
    });

    assert!(
        matches!(result, Err(SdkError::Timeout { timeout_ms: 200 })),
        "{result:?}"
    );
    // The result is remembered, and the cut-off export is counted failed.
    let again = session.shutdown(Duration::from_millis(200));
    assert!(
        matches!(again, Err(SdkError::Timeout { timeout_ms: 200 })),
        "{again:?}"
    );
    let stats = session.stats().expect("an owned pipeline");
    assert_eq!(
        (stats.traces.failed, stats.traces.pending),
        (1, 0),
        "{stats:?}"
    );
    for signal in [stats.traces, stats.logs, stats.metrics] {
        assert_eq!(signal.pending, 0, "{stats:?}");
        assert!(signal.is_consistent(), "{stats:?}");
    }
}

#[test]
fn shutting_the_default_session_down_returns_a_timeout_while_its_export_is_parked() {
    let _default = default_session_lock();
    let _restore = DefaultSession::install(&ObservabilitySession::borrowed(Observer::disabled()));
    let collector = ParkedCollector::start();
    init_observability(ObservabilityConfig::otlp(collector.config(parking_batch()))).unwrap();
    // A root workflow reports to the default session; its span parks.
    start_workflow("default", &ObservationOptions::default()).end();
    collector.wait_parked("/v1/traces");

    let result = returns("shutdown_observability", || {
        shutdown_observability(Duration::from_millis(200))
    });

    assert!(
        matches!(result, Err(SdkError::Timeout { timeout_ms: 200 })),
        "{result:?}"
    );
    // The session it shut down is no longer the default, and its parked
    // export did not outlive the call.
    assert!(crate::observability::default_session().is_none());
    collector.wait_closed("/v1/traces");
}

#[test]
fn dropping_the_last_session_handle_returns_while_its_export_is_parked() {
    let collector = ParkedCollector::start();
    let session = parked_session(&collector, None);

    returns("dropping the session", move || drop(session));

    // The pipeline went with it: nothing of it stays open.
    collector.wait_closed("/v1/traces");
}

#[test]
fn replacing_the_default_session_returns_while_its_export_is_parked() {
    let _default = default_session_lock();
    let _restore = DefaultSession::install(&ObservabilitySession::borrowed(Observer::disabled()));
    let collector = ParkedCollector::start();
    init_observability(ObservabilityConfig::otlp(collector.config(parking_batch()))).unwrap();
    // A root workflow reports to the default session; its span parks.
    start_workflow("replaced", &ObservationOptions::default()).end();
    collector.wait_parked("/v1/traces");

    // Re-initialising shuts the replaced helper-owned session down (bounded)
    // on the caller's thread.
    returns("re-initialising observability", || {
        init_observability(ObservabilityConfig::with_observer(Observer::disabled()))
    })
    .unwrap();

    // The replaced session's export did not outlive the call.
    collector.wait_closed("/v1/traces");
}

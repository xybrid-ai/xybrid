//! Outcomes of runs their caller ends, and of requests that end before they
//! run.
//!
//! - A caller that abandons a stream before its terminal event cancels the
//!   run: it stops at its next token and ends `cancelled` (span `ERROR`,
//!   `error.type` `cancelled`, one cancelled duration point, no exception
//!   log). The Rust `run_stream` drop and a token callback that returns
//!   [`AbortReason::UserCancelled`] (how the facade, Bolt and Flutter report
//!   a consumer that went away) both take that path.
//! - Only a stop on the caller's cancellation is `cancelled`. A run whose
//!   token was cancelled but that failed for its own reason (a batch run, or
//!   a stream whose policy does not observe `UserCancelled`) keeps its error.
//! - `chunk_count` is the number of tokens the caller received, whatever
//!   stopped the stream.
//! - A pre-run cancellation, a pre-run resource abort and a refused input
//!   are each one request: one span ended with the outcome and one duration
//!   point with its `error.type`, recorded without waiting for a busy model.
//!   On the fallback path a pre-run resource abort is the failed local
//!   attempt ahead of the remote one.
//!
//! Streams are paced: the fake runtime emits token `i` only after the test
//! sends its permit, so a drop or stop happens at a known token and the
//! runtime's callback count shows where the run stopped. Every wait is a hang
//! detector.
//!
//! An entry point that takes no options reports to the process default, so
//! its test installs one, under the default-session lock. A run pinned to no
//! session in a test beside it records there too, so such a test names its
//! model for itself and reads back only its own run: the span named after
//! that model, that model's duration points and that span's log records. It
//! makes a neighbour's failed run itself, beside its own.

use super::*;
use crate::observability::semconv;
use crate::observability::test_support::{
    attr, default_session_lock, memory_session_with_signals, DefaultSession, Signals,
};
use crate::run_options::{AbortPolicy, AbortReason, AbortSignal};
use opentelemetry_sdk::trace::SpanData;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use xybrid_core::runtime_adapter::types::PartialToken;
use xybrid_core::runtime_adapter::AdapterError;

/// Bound of every hang detector; nothing here waits on anything slow.
const HANG: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Fakes.
// ---------------------------------------------------------------------------

/// A token stream the test paces: token `i` is emitted once the test sends
/// its permit. Counts the callbacks it made, and stops at the first one that
/// fails, as a runtime does.
struct Paced {
    tokens: usize,
    permits: Mutex<mpsc::Receiver<()>>,
    callbacks: Arc<AtomicUsize>,
}

impl Paced {
    /// A stream of `tokens` tokens, its permit sender and its callback count.
    fn new(tokens: usize) -> (Self, mpsc::Sender<()>, Arc<AtomicUsize>) {
        let (permit, permits) = mpsc::channel();
        let callbacks = Arc::new(AtomicUsize::new(0));
        let paced = Self {
            tokens,
            permits: Mutex::new(permits),
            callbacks: callbacks.clone(),
        };
        (paced, permit, callbacks)
    }

    fn stream(
        &self,
        on_token: &mut dyn FnMut(PartialToken) -> Result<(), StreamingError>,
    ) -> xybrid_core::runtime_adapter::AdapterResult<Envelope> {
        let mut text = String::new();
        for index in 0..self.tokens {
            self.permits
                .lock()
                .unwrap()
                .recv_timeout(HANG)
                .expect("the test paces every token it lets the stream emit");
            self.callbacks.fetch_add(1, Ordering::SeqCst);
            let piece = format!("t{index} ");
            text.push_str(&piece);
            on_token(PartialToken {
                token: piece,
                token_id: None,
                index,
                cumulative_text: text.clone(),
                finish_reason: (index + 1 == self.tokens).then(|| "stop".to_string()),
                tool_calls: Vec::new(),
                raw_text: None,
            })
            .map_err(AdapterError::from_streaming_callback_error)?;
        }
        Ok(Envelope::new(EnvelopeKind::Text(text)))
    }
}

impl xybrid_core::runtime_adapter::CloudStreaming for Paced {
    fn execute_streaming(
        &self,
        _input: &Envelope,
        mut on_token: xybrid_core::runtime_adapter::types::StreamingCallback<'_>,
    ) -> xybrid_core::runtime_adapter::AdapterResult<Envelope> {
        self.stream(&mut on_token)
    }
}

/// Fake `"onnx"` runtime that blocks inside `execute()` until the test
/// releases it, then answers `outcome` (text, or a runtime failure).
struct Gated {
    entered: Mutex<Option<mpsc::Sender<()>>>,
    release: Mutex<mpsc::Receiver<()>>,
    outcome: Result<&'static str, &'static str>,
}

impl Gated {
    /// The runtime, a receiver that fires once it is inside `execute()`, and
    /// the sender that releases it.
    fn new(
        outcome: Result<&'static str, &'static str>,
    ) -> (Self, mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let runtime = Self {
            entered: Mutex::new(Some(entered_tx)),
            release: Mutex::new(release_rx),
            outcome,
        };
        (runtime, entered, release)
    }
}

impl xybrid_core::runtime_adapter::ModelRuntime for Gated {
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
        _input: &Envelope,
    ) -> xybrid_core::runtime_adapter::AdapterResult<Envelope> {
        if let Some(entered) = self.entered.lock().unwrap().take() {
            let _ = entered.send(());
        }
        self.release
            .lock()
            .unwrap()
            .recv_timeout(HANG)
            .expect("the test releases the runtime");
        match self.outcome {
            Ok(text) => Ok(Envelope::new(EnvelopeKind::Text(text.to_string()))),
            Err(message) => Err(AdapterError::RuntimeError(message.to_string())),
        }
    }
}

/// Fake `"onnx"` runtime answering `text` at once.
struct Answers(&'static str);

impl xybrid_core::runtime_adapter::ModelRuntime for Answers {
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
        _input: &Envelope,
    ) -> xybrid_core::runtime_adapter::AdapterResult<Envelope> {
        Ok(Envelope::new(EnvelopeKind::Text(self.0.to_string())))
    }
}

/// Fake `"onnx"` runtime that a request ending before it runs must never
/// reach.
struct NeverRuns;

impl xybrid_core::runtime_adapter::ModelRuntime for NeverRuns {
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
        _input: &Envelope,
    ) -> xybrid_core::runtime_adapter::AdapterResult<Envelope> {
        panic!("a request that ends before it runs never reaches the runtime")
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Options stopping on the caller's cancellation of `token`.
fn cancellable(token: &CancellationToken) -> RunOptions {
    RunOptions::new()
        .with_cancellation_token(token.clone())
        .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::UserCancelled))
}

/// Options under which critical memory pressure stops the local leg before
/// it runs, with cloud fallback allowed.
fn pressured() -> RunOptions {
    let mut critical = xybrid_core::device::ResourceSnapshot::unknown();
    critical.memory_pressure = xybrid_core::device::MemoryPressure::Critical;
    RunOptions::new()
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true),
        )
        .with_resource_provider(Arc::new(FixedUnitResourceProvider::new(critical)))
}

/// The caller's cancellation, as a consumer that went away reports it.
fn consumer_gone() -> StreamingError {
    Box::new(AbortReason::UserCancelled)
}

fn chunk_count(span: &SpanData) -> Option<String> {
    attr(span, semconv::attr::RESPONSE_CHUNK_COUNT)
}

/// `model` under `id`, a model id no other test uses: a test whose run
/// reports to the process default reads back only this model's records.
fn named(mut model: XybridModel, id: &str) -> XybridModel {
    model.model_id = id.to_string();
    model
}

/// The only `gen_ai.client.inference.duration` point of `model` has
/// `error.type` `expected` and counts one run, whatever other models' runs
/// recorded into the same session.
fn assert_one_duration_point_of(signals: &Signals, model: &XybridModel, expected: &str) {
    let points: Vec<_> = signals
        .points(semconv::metric::INFERENCE_DURATION)
        .into_iter()
        .filter(|p| p.attr(semconv::attr::REQUEST_MODEL) == Some(model.model_id.as_str()))
        .collect();
    assert_eq!(points.len(), 1, "{points:?}");
    assert_eq!(
        (points[0].attr(semconv::attr::ERROR_TYPE), points[0].count),
        (Some(expected), 1)
    );
}

/// No log record of `span` is an operation exception: a cancellation is not
/// one. Records of other runs in the same session are not read.
fn assert_no_exception_on(signals: &Signals, span: &SpanData) {
    let own = span.span_context.span_id();
    let events: Vec<_> = signals
        .logs
        .get_emitted_logs()
        .expect("logs")
        .iter()
        .filter(|l| l.record.trace_context().is_some_and(|t| t.span_id == own))
        .filter_map(|l| l.record.event_name().map(str::to_owned))
        .collect();
    assert!(
        !events
            .iter()
            .any(|e| e == semconv::event::OPERATION_EXCEPTION),
        "{}: {events:?}",
        span.name
    );
}

/// `span` ended with `error.type` `expected` and status `ERROR`.
fn assert_ended(span: &SpanData, expected: &str) {
    assert!(
        matches!(span.status, opentelemetry::trace::Status::Error { .. }),
        "{}: {:?}",
        span.name,
        span.status
    );
    assert_eq!(
        attr(span, semconv::attr::ERROR_TYPE).as_deref(),
        Some(expected),
        "{}",
        span.name
    );
}

/// `(reason, location, error.type, count)` of every attempts-counter point.
fn attempt_points(signals: &Signals) -> Vec<(String, String, Option<String>, u64)> {
    let mut points: Vec<_> = signals
        .points(semconv::metric::ATTEMPTS)
        .into_iter()
        .map(|p| {
            (
                p.attr(semconv::attr::ATTEMPT_REASON)
                    .unwrap_or_default()
                    .to_owned(),
                p.attr(semconv::attr::EXECUTION_LOCATION)
                    .unwrap_or_default()
                    .to_owned(),
                p.attr(semconv::attr::ERROR_TYPE).map(str::to_owned),
                p.count,
            )
        })
        .collect();
    points.sort();
    points
}

// ---------------------------------------------------------------------------
// Abandoning a stream is the caller's cancellation.
// ---------------------------------------------------------------------------

#[test]
fn dropping_a_run_stream_mid_stream_cancels_the_run_at_its_next_token() {
    use tokio_stream::StreamExt as _;
    // `run_stream` takes no options: it reports to the default session.
    let _default = default_session_lock();
    let signals = memory_session_with_signals();
    let _installed = DefaultSession::install(&signals.session);
    let model = named(
        as_chat(test_loaded_model(true)),
        "run-stream-dropped-mid-stream",
    );
    let span_name = run_span_name(&model);
    run_an_unpinned_neighbour_that_fails();
    let (paced, permit, callbacks) = Paced::new(5);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    // The runtime emits two tokens, the caller reads both, then drops the
    // stream before its terminal event.
    const DELIVERED: usize = 2;
    runtime.block_on(async {
        let mut stream = model.spawn_event_stream(
            text_envelope("tell me a long story"),
            None,
            move |model, _envelope, _config, _obs, on_token| {
                paced
                    .stream(on_token)
                    .map(|output| InferenceResult::new(output, &model.model_id, 0))
                    .map_err(streaming_execution_error)
            },
        );
        for index in 0..DELIVERED {
            permit.send(()).unwrap();
            match tokio::time::timeout(HANG, stream.next()).await {
                Ok(Some(StreamEvent::Token(token))) => assert_eq!(token.index, index),
                other => panic!("token {index}: {other:?}"),
            }
        }
        drop(stream);
        // The next token finds no receiver.
        permit.send(()).unwrap();
    });
    // Waits for the blocking run to return (it ends its span first).
    runtime.shutdown_timeout(HANG);

    // It stopped at that token: one callback past the two delivered.
    assert_eq!(callbacks.load(Ordering::SeqCst), DELIVERED + 1);
    let span = signals.spans.named(&span_name);
    assert_ended(&span, "cancelled");
    assert_eq!(chunk_count(&span).as_deref(), Some("2"));
    assert_one_duration_point_of(&signals, &model, "cancelled");
    assert_no_exception_on(&signals, &span);
}

#[test]
fn dropping_a_run_stream_before_its_token_cancels_the_run() {
    // Through `run_stream` itself: a local model emits its whole answer as
    // one token, which the run hands over after the caller dropped the
    // stream.
    let _default = default_session_lock();
    let signals = memory_session_with_signals();
    let _installed = DefaultSession::install(&signals.session);
    let (fake, entered, release) = Gated::new(Ok("the whole answer"));
    let model = named(
        as_chat(test_model_with_runtime(Box::new(fake))),
        "run-stream-dropped-before-its-token",
    );
    let span_name = run_span_name(&model);
    run_an_unpinned_neighbour_that_fails();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    {
        let _entered_runtime = runtime.enter();
        let stream = model.run_stream(text_envelope("hi"), None);
        entered
            .recv_timeout(HANG)
            .expect("the run enters the runtime");
        drop(stream);
    }
    release.send(()).unwrap();
    // Waits for the blocking run to return (it ends its span first).
    runtime.shutdown_timeout(HANG);

    let span = signals.spans.named(&span_name);
    assert_ended(&span, "cancelled");
    assert_eq!(chunk_count(&span), None);
    assert_one_duration_point_of(&signals, &model, "cancelled");
    assert_no_exception_on(&signals, &span);
}

#[test]
fn a_consumer_that_goes_away_mid_stream_cancels_the_run_at_that_token() {
    // The facade, Bolt and Flutter token callbacks return
    // `AbortReason::UserCancelled` once their consumer is gone. A remote
    // fallback leg streams several tokens without a real model: memory
    // pressure stops the local leg before it runs.
    const DELIVERED: usize = 2;
    for (consumer, expected) in [
        (consumer_gone as fn() -> StreamingError, "cancelled"),
        // A callback that fails for any other reason also stopped the
        // stream itself: the model did not fail, as in a streaming pipeline.
        (
            || -> StreamingError { "the consumer failed".into() },
            "cancelled",
        ),
    ] {
        let signals = memory_session_with_signals();
        let model = as_chat(test_model_with_runtime(Box::new(NeverRuns)));
        let span_name = run_span_name(&model);
        let (paced, permit, callbacks) = Paced::new(5);
        // Enough permits for one token past the delivered ones: a run that
        // did not stop there would wait for a sixth and trip the detector.
        for _ in 0..=DELIVERED {
            permit.send(()).unwrap();
        }
        let mut received = 0;
        let mut on_token = |_token: PartialToken| {
            if received == DELIVERED {
                return Err(consumer());
            }
            received += 1;
            Ok(())
        };

        model
            .run_streaming_with_fallback(
                &text_envelope("write me a haiku"),
                &observed(pressured(), &signals),
                &paced,
                &mut on_token,
                &mut |_seam: SeamInfo| {},
            )
            .expect_err("the consumer stopped the stream");

        assert_eq!(
            callbacks.load(Ordering::SeqCst),
            DELIVERED + 1,
            "{expected}"
        );
        let request = signals.spans.named(&span_name);
        assert_ended(&request, expected);
        assert_eq!(chunk_count(&request).as_deref(), Some("2"), "{expected}");
        assert_ended(&signals.spans.named("attempt 1"), expected);
        assert_one_duration_point(&signals, expected);
        assert_no_exception(&signals);
    }
}

#[test]
fn a_consumer_that_goes_away_cancels_runs_of_both_options_entry_points() {
    // A local run emits one token (the batch result) through the same gate
    // and delivery as a token stream.
    for (context, refusal) in [
        (false, consumer_gone as fn() -> StreamingError),
        (true, consumer_gone),
        // The error's type does not matter: the consumer stopped the stream.
        (false, || -> StreamingError { "the consumer failed".into() }),
        (true, || -> StreamingError { "the consumer failed".into() }),
    ] {
        let signals = memory_session_with_signals();
        let model = as_chat(test_model_with_runtime(Box::new(Answers("hello"))));
        let span_name = run_span_name(&model);
        let calls = AtomicUsize::new(0);
        let on_token = |_token: PartialToken| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(refusal())
        };
        let options = observed(RunOptions::new(), &signals);
        let input = text_envelope("hi");
        let error = if context {
            model.run_streaming_with_context_options(
                &input,
                &ConversationContext::new(),
                &options,
                on_token,
            )
        } else {
            model.run_streaming_with_options(&input, &options, on_token)
        }
        .expect_err("the consumer is gone");

        // What the caller receives is unchanged.
        assert!(
            matches!(&error, SdkError::InferenceError { message, .. } if message == "Streaming callback failed"),
            "{error:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let span = signals.spans.named(&span_name);
        assert_ended(&span, "cancelled");
        assert_eq!(chunk_count(&span), None, "context: {context}");
        assert_one_duration_point(&signals, "cancelled");
        assert_no_exception(&signals);
    }
}

// ---------------------------------------------------------------------------
// Precise cancellation and chunks counted after the gate.
// ---------------------------------------------------------------------------

#[test]
fn a_cancelled_run_that_fails_for_its_own_reason_keeps_its_error() {
    // The token is cancelled while the runtime runs, but nothing stops on
    // it: a batch run has no token boundary, and the streams' policy does not
    // observe `UserCancelled`. The runtime then fails on its own.
    #[derive(Clone, Copy)]
    enum Entry {
        Batch,
        Stream,
        ContextStream,
    }
    for entry in [Entry::Batch, Entry::Stream, Entry::ContextStream] {
        let signals = memory_session_with_signals();
        let (runtime, entered, release) = Gated::new(Err("decoder failed"));
        let model = as_chat(test_model_with_runtime(Box::new(runtime)));
        let span_name = run_span_name(&model);
        let token = CancellationToken::new();
        let options = observed(
            match entry {
                // The batch pre-run gate observes the token; the run does not.
                Entry::Batch => cancellable(&token),
                Entry::Stream | Entry::ContextStream => {
                    RunOptions::new().with_cancellation_token(token.clone())
                }
            },
            &signals,
        );
        let worker = {
            let model = model.clone();
            std::thread::spawn(move || {
                let input = text_envelope("hi");
                match entry {
                    Entry::Batch => model.run_with_options(&input, &options),
                    Entry::Stream => model.run_streaming_with_options(&input, &options, |_| Ok(())),
                    Entry::ContextStream => model.run_streaming_with_context_options(
                        &input,
                        &ConversationContext::new(),
                        &options,
                        |_| Ok(()),
                    ),
                }
            })
        };

        entered
            .recv_timeout(HANG)
            .expect("the run enters the runtime");
        token.cancel();
        release.send(()).unwrap();
        worker
            .join()
            .expect("no panic")
            .expect_err("the runtime fails");

        assert_ended(&signals.spans.named(&span_name), "inference_error");
        assert_one_duration_point(&signals, "inference_error");
    }
}

#[test]
fn a_token_stopped_by_the_abort_gate_is_not_a_delivered_chunk() {
    // The gate stops the stream at its first token: the caller receives
    // nothing, so no chunk is counted and no first chunk is timed.
    for context in [false, true] {
        let signals = memory_session_with_signals();
        let (runtime, entered, release) = Gated::new(Ok("partial output"));
        let model = as_chat(test_model_with_runtime(Box::new(runtime)));
        let span_name = run_span_name(&model);
        let token = CancellationToken::new();
        let options = observed(cancellable(&token), &signals);
        let delivered = Arc::new(AtomicUsize::new(0));
        let worker = {
            let model = model.clone();
            let delivered = delivered.clone();
            std::thread::spawn(move || {
                let input = text_envelope("hi");
                let on_token = move |_token: PartialToken| {
                    delivered.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                };
                if context {
                    model.run_streaming_with_context_options(
                        &input,
                        &ConversationContext::new(),
                        &options,
                        on_token,
                    )
                } else {
                    model.run_streaming_with_options(&input, &options, on_token)
                }
            })
        };

        entered
            .recv_timeout(HANG)
            .expect("the run enters the runtime");
        token.cancel();
        release.send(()).unwrap();
        worker
            .join()
            .expect("no panic")
            .expect_err("the gate stops the run");

        assert_eq!(delivered.load(Ordering::SeqCst), 0);
        let span = signals.spans.named(&span_name);
        assert_ended(&span, "cancelled");
        assert_eq!(chunk_count(&span), None, "context: {context}");
        assert_eq!(
            attr(&span, semconv::attr::RESPONSE_TIME_TO_FIRST_CHUNK),
            None,
            "context: {context}"
        );
        assert!(
            signals
                .points(semconv::metric::TIME_TO_FIRST_CHUNK)
                .is_empty(),
            "context: {context}"
        );
        assert_one_duration_point(&signals, "cancelled");
    }
}

// ---------------------------------------------------------------------------
// Requests that end before they run are counted.
// ---------------------------------------------------------------------------

/// A request through some entry point that ends before it runs.
struct Unstarted {
    name: &'static str,
    /// What the request ends with.
    expected: &'static str,
    /// A TTS model instead of a chat model.
    tts: bool,
    run: fn(&XybridModel, RunOptions) -> SdkResult<()>,
    options: fn() -> RunOptions,
}

fn pre_cancelled() -> RunOptions {
    let token = CancellationToken::new();
    token.cancel();
    cancellable(&token)
}

/// An input no chat or TTS model runs: a choice request.
fn refused_input() -> Envelope {
    Envelope::choice_request(
        text_envelope("pick one"),
        vec![Choice::new("a", "first"), Choice::new("b", "second")],
    )
}

fn ignore_tokens(_token: PartialToken) -> Result<(), StreamingError> {
    Ok(())
}

fn unstarted_requests() -> Vec<Unstarted> {
    vec![
        Unstarted {
            name: "run_with_options cancelled before it runs",
            expected: "cancelled",
            tts: false,
            options: pre_cancelled,
            run: |model, options| {
                model
                    .run_with_options(&text_envelope("hi"), &options)
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_with_options refused input",
            expected: "unsupported",
            tts: false,
            options: RunOptions::new,
            run: |model, options| model.run_with_options(&refused_input(), &options).map(drop),
        },
        Unstarted {
            name: "run_with_context_options cancelled before it runs",
            expected: "cancelled",
            tts: false,
            options: pre_cancelled,
            run: |model, options| {
                model
                    .run_with_context_options(
                        &text_envelope("hi"),
                        &ConversationContext::new(),
                        &options,
                    )
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_with_context_options refused history",
            expected: "unsupported",
            tts: false,
            options: RunOptions::new,
            run: |model, options| {
                let mut context = ConversationContext::new();
                context.push(refused_input());
                model
                    .run_with_context_options(&text_envelope("hi"), &context, &options)
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_streaming_with_options cancelled before it runs",
            expected: "cancelled",
            tts: false,
            options: pre_cancelled,
            run: |model, options| {
                model
                    .run_streaming_with_options(&text_envelope("hi"), &options, ignore_tokens)
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_streaming_with_options resource abort before it runs",
            expected: "aborted_for_cloud_fallback",
            tts: false,
            options: pressured,
            run: |model, options| {
                model
                    .run_streaming_with_options(&text_envelope("hi"), &options, ignore_tokens)
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_streaming_with_options refused input",
            expected: "unsupported",
            tts: false,
            options: RunOptions::new,
            run: |model, options| {
                model
                    .run_streaming_with_options(&refused_input(), &options, ignore_tokens)
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_streaming_with_context_options resource abort before it runs",
            expected: "aborted_for_cloud_fallback",
            tts: false,
            options: pressured,
            run: |model, options| {
                model
                    .run_streaming_with_context_options(
                        &text_envelope("hi"),
                        &ConversationContext::new(),
                        &options,
                        ignore_tokens,
                    )
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_streaming_with_context_options refused input",
            expected: "unsupported",
            tts: false,
            options: RunOptions::new,
            run: |model, options| {
                model
                    .run_streaming_with_context_options(
                        &refused_input(),
                        &ConversationContext::new(),
                        &options,
                        ignore_tokens,
                    )
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_streaming_with_fallback refused input",
            expected: "unsupported",
            tts: false,
            options: RunOptions::new,
            run: |model, options| {
                model
                    .run_streaming_with_fallback(
                        &refused_input(),
                        &options,
                        &FakeCloudAdapter::new("never sent"),
                        &mut ignore_tokens,
                        &mut |_seam: SeamInfo| {},
                    )
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_tts_streaming resource abort before synthesis",
            expected: "aborted_for_cloud_fallback",
            tts: true,
            options: pressured,
            run: |model, options| {
                model
                    .run_tts_streaming(&text_envelope("hello"), &options, |_| panic!("no audio"))
                    .map(drop)
            },
        },
        Unstarted {
            name: "run_tts_streaming refused input",
            expected: "unsupported",
            tts: true,
            options: RunOptions::new,
            run: |model, options| {
                model
                    .run_tts_streaming(&refused_input(), &options, |_| panic!("no audio"))
                    .map(drop)
            },
        },
    ]
}

fn tts_model() -> XybridModel {
    let mut metadata = ModelMetadata::onnx("tts-model", "1.0", "model.onnx");
    metadata
        .metadata
        .insert("task".into(), serde_json::json!("text-to-speech"));
    test_model_with_metadata(metadata)
}

#[test]
fn requests_that_end_before_they_run_are_one_request_with_their_outcome() {
    for request in unstarted_requests() {
        let signals = memory_session_with_signals();
        let model = if request.tts {
            tts_model()
        } else {
            as_chat(test_model_with_runtime(Box::new(NeverRuns)))
        };
        let span_name = run_span_name(&model);

        (request.run)(&model, observed((request.options)(), &signals)).expect_err(request.name);

        let spans = signals.spans.spans();
        assert_eq!(spans.len(), 1, "{}: {spans:?}", request.name);
        assert_eq!(spans[0].name, span_name, "{}", request.name);
        assert_ended(&spans[0], request.expected);
        let points = signals.points(semconv::metric::INFERENCE_DURATION);
        assert_eq!(points.len(), 1, "{}: {points:?}", request.name);
        assert_eq!(
            (points[0].attr(semconv::attr::ERROR_TYPE), points[0].count),
            (Some(request.expected), 1),
            "{}",
            request.name
        );
        if request.expected == "cancelled" {
            assert_no_exception(&signals);
        }
    }
}

#[test]
fn a_synthesis_cancelled_before_it_starts_is_one_cancelled_request() {
    let signals = memory_session_with_signals();
    let model = tts_model();
    let span_name = run_span_name(&model);

    let result = model
        .run_tts_streaming(
            &text_envelope("hello"),
            &observed(pre_cancelled(), &signals),
            |_| panic!("a cancelled request delivers no audio"),
        )
        .expect("a cancelled synthesis is a result, not an error");

    // The caller still receives a cancelled summary, now with the reference
    // of the request it recorded.
    assert_eq!(result.status, xybrid_core::execution::TtsStatus::Cancelled);
    assert_eq!(result.samples, 0);
    let reference = result.observation.expect("the request is recorded");
    let span = signals.spans.by_ref(&reference);
    assert_eq!(span.name, span_name);
    assert_ended(&span, "cancelled");
    assert_eq!(
        attr(&span, "xybrid.tts.status").as_deref(),
        Some("cancelled")
    );
    assert_one_duration_point(&signals, "cancelled");
    assert_no_exception(&signals);
}

#[test]
fn a_decision_request_that_ends_before_it_runs_is_one_request_with_its_outcome() {
    // A decision model ends on a cancelled token whatever the abort policy
    // observes, and refuses cloud fallback. Both return before the model is
    // read, and both are counted like any other request that never ran.
    let token = CancellationToken::new();
    token.cancel();
    let cases = [
        (
            "cancelled token",
            RunOptions::new().with_cancellation_token(token),
            "cancelled",
        ),
        (
            "cloud fallback",
            RunOptions::new().with_abort_policy(AbortPolicy::default().with_cloud_fallback(true)),
            "unsupported",
        ),
    ];
    for (name, options, expected) in cases {
        let signals = memory_session_with_signals();
        let model = test_model_with_metadata(liquid_metadata());
        let span_name = run_span_name(&model);

        model
            .run_with_options(&decision_warmup_request(), &observed(options, &signals))
            .expect_err(name);

        let spans = signals.spans.spans();
        assert_eq!(spans.len(), 1, "{name}: {spans:?}");
        assert_eq!(spans[0].name, span_name, "{name}");
        assert_ended(&spans[0], expected);
        // A decision is not generative inference: its duration is recorded
        // on the general operation metric.
        let points = signals.points(semconv::metric::OPERATION_DURATION);
        assert_eq!(points.len(), 1, "{name}: {points:?}");
        assert_eq!(
            (points[0].attr(semconv::attr::ERROR_TYPE), points[0].count),
            (Some(expected), 1),
            "{name}"
        );
        assert!(
            signals
                .points(semconv::metric::INFERENCE_DURATION)
                .is_empty(),
            "{name}"
        );
        if expected == "cancelled" {
            assert_no_exception(&signals);
        }
    }
}

#[test]
fn a_refused_run_stream_input_is_one_request() {
    use tokio_stream::StreamExt as _;
    let _default = default_session_lock();
    let signals = memory_session_with_signals();
    let _installed = DefaultSession::install(&signals.session);
    let model = named(
        as_chat(test_model_with_runtime(Box::new(NeverRuns))),
        "run-stream-refused-input",
    );
    let span_name = run_span_name(&model);
    run_an_unpinned_neighbour_that_fails();

    let events: Vec<_> = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(model.run_stream(refused_input(), None).collect());

    assert!(
        matches!(events.as_slice(), [StreamEvent::Error(_)]),
        "{events:?}"
    );
    assert_ended(&signals.spans.named(&span_name), "unsupported");
    assert_one_duration_point_of(&signals, &model, "unsupported");
}

/// An entry point, run by a request cancelled before it runs.
type CancelledRequest = fn(&XybridModel, &RunOptions) -> SdkResult<()>;

#[test]
fn requests_that_end_before_they_run_never_wait_for_a_busy_model() {
    // A run in flight holds the model's write lock. A request cancelled
    // before it runs returns without waiting for it, as it always did, and is
    // recorded under the model's subject as the load read it: a chat model's
    // request is a `chat` inference, counted with its other runs. The
    // reactive cloud fallback is no exception.
    let _residency = crate::model_registry::residency_test_lock();
    let dir = tempfile::tempdir().unwrap();
    let mut metadata = ModelMetadata::onnx("busy-model", "1.0", "model.onnx");
    metadata
        .metadata
        .insert("task".into(), serde_json::json!("text-generation"));
    std::fs::write(
        dir.path().join("model_metadata.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let model = ModelLoader::from_directory(dir.path())
        .unwrap()
        .load()
        .unwrap();
    let span_name = run_span_name(&model);
    assert_eq!(span_name, "chat busy-model");

    let entry_points: [(&str, CancelledRequest); 2] = [
        ("run_streaming_with_options", |model, options| {
            model
                .run_streaming_with_options(&text_envelope("hi"), options, ignore_tokens)
                .map(drop)
        }),
        ("run_streaming_with_fallback", |model, options| {
            let cloud = FakeCloudAdapter::new("never sent");
            let result = model
                .run_streaming_with_fallback(
                    &text_envelope("hi"),
                    options,
                    &cloud,
                    &mut ignore_tokens,
                    &mut |_seam: SeamInfo| {},
                )
                .map(drop);
            assert_eq!(cloud.call_count(), 0, "a cancelled request sends nothing");
            result
        }),
    ];
    for (entry, run) in entry_points {
        let signals = memory_session_with_signals();
        let held = model.handle.write().unwrap();
        let (send, recv) = mpsc::channel();
        let worker = {
            let model = model.clone();
            let options = observed(pre_cancelled(), &signals);
            std::thread::spawn(move || send.send(run(&model, &options)).unwrap())
        };
        let result = recv.recv_timeout(HANG).unwrap_or_else(|_| {
            panic!("{entry}: a request cancelled before it runs never waits for the model lock")
        });
        assert!(result.is_err(), "{entry}");
        drop(held);
        worker.join().unwrap();

        let span = signals.spans.named(&span_name);
        assert_ended(&span, "cancelled");
        assert_eq!(
            attr(&span, semconv::attr::OPERATION_NAME).as_deref(),
            Some("chat"),
            "{entry}"
        );
        // The chat inference duration metric, beside the model's other runs.
        assert_one_duration_point(&signals, "cancelled");
    }
}

/// A zzz synthesis through the batch entry point, whose token is cancelled
/// before it runs under a policy that does not stop on it, is recorded
/// without waiting for a run in flight either: the remembered subject says
/// the model is zzz while the lock is held.
#[cfg(feature = "tts-zzz")]
#[test]
fn a_cancelled_zzz_synthesis_never_waits_for_a_busy_model() {
    let signals = memory_session_with_signals();
    let mut metadata = ModelMetadata::onnx("busy-zzz", "1.0", "phonon.gguf");
    metadata
        .metadata
        .insert("task".into(), serde_json::json!("text-to-speech"));
    metadata.execution_template = serde_json::from_value(serde_json::json!({
        "type": "ZzzPhonon", "model_file": "phonon.gguf"
    }))
    .unwrap();
    let model = test_model_with_metadata(metadata);
    {
        // As the load does.
        let handle = model.handle.read().unwrap();
        model.remember_subject(&RunSubject::from_metadata(
            &model.model_id,
            &handle.metadata,
        ));
    }
    let span_name = run_span_name(&model);
    let token = CancellationToken::new();
    token.cancel();
    // The policy does not stop on the token: only the TTS path reads it.
    let options = observed(RunOptions::new().with_cancellation_token(token), &signals);

    let held = model.handle.write().unwrap();
    let (send, recv) = mpsc::channel();
    let worker = {
        let model = model.clone();
        std::thread::spawn(move || {
            send.send(
                model
                    .run_with_options(&text_envelope("hello"), &options)
                    .map(drop),
            )
            .unwrap();
        })
    };
    let result = recv
        .recv_timeout(HANG)
        .expect("a cancelled synthesis never waits for the model lock");
    assert!(
        result.is_ok(),
        "a cancelled synthesis is a result: {result:?}"
    );
    drop(held);
    worker.join().unwrap();

    assert_ended(&signals.spans.named(&span_name), "cancelled");
    assert_one_duration_point(&signals, "cancelled");
}

#[test]
fn a_resource_abort_before_the_local_leg_is_the_failed_local_attempt_of_a_fallback() {
    let signals = memory_session_with_signals();
    let model = as_chat(test_model_with_runtime(Box::new(NeverRuns)));
    let span_name = run_span_name(&model);
    let cloud = FakeCloudAdapter::new("hello from the cloud");
    let mut on_token = ignore_tokens;

    let result = model
        .run_streaming_with_fallback(
            &text_envelope("write me a haiku"),
            &observed(pressured(), &signals),
            &cloud,
            &mut on_token,
            &mut |_seam: SeamInfo| {},
        )
        .expect("the cloud fallback answers");

    assert_eq!(result.text(), Some("hello from the cloud"));
    assert_eq!(cloud.call_count(), 1);
    let request = signals.spans.named(&span_name);
    assert_eq!(attr(&request, semconv::attr::ERROR_TYPE), None);
    assert_eq!(
        attr(&request, semconv::attr::ATTEMPT_COUNT).as_deref(),
        Some("2")
    );
    let local = signals.spans.named("attempt 0");
    assert_ended(&local, "aborted_for_cloud_fallback");
    let describe = |span: &SpanData| {
        (
            attr(span, semconv::attr::ATTEMPT_REASON),
            attr(span, semconv::attr::EXECUTION_LOCATION),
        )
    };
    assert_eq!(
        describe(&local),
        (Some("initial".into()), Some("local".into()))
    );
    let remote = signals.spans.named("attempt 1");
    assert_eq!(
        describe(&remote),
        (Some("fallback".into()), Some("remote".into()))
    );
    assert_eq!(attr(&remote, semconv::attr::ERROR_TYPE), None);
    // One request, two attempts.
    let durations = signals.points(semconv::metric::INFERENCE_DURATION);
    assert_eq!(durations.len(), 1, "{durations:?}");
    assert_eq!(
        (
            durations[0].attr(semconv::attr::ERROR_TYPE),
            durations[0].count
        ),
        (None, 1)
    );
    assert_eq!(
        attempt_points(&signals),
        [
            ("fallback".into(), "remote".into(), None, 1),
            (
                "initial".into(),
                "local".into(),
                Some("aborted_for_cloud_fallback".into()),
                1
            ),
        ]
    );
}

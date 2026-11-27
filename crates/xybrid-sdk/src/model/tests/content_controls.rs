//! Content controls on automatic runs (reliability E1.1–E1.3).
//!
//! Each test runs a chat model over a fake runtime, observed by a session
//! whose own observer keeps spans, metric points and log records in memory
//! and records content under the policy the test sets. What the run records
//! comes from the SDK itself: the conversation's system prompt
//! (`gen_ai.system_instructions`), its history and the caller's input
//! (`gen_ai.input.messages`), and the runtime's answer
//! (`gen_ai.output.messages`).
//!
//! - Capture modes: that content is on the run's span exactly in `span_only`
//!   and `span_and_event`, and on the run's
//!   `gen_ai.client.inference.operation.details` log event exactly in
//!   `event_only` and `span_and_event`. `no_content` records it nowhere, and
//!   no other span or log record ever carries it. A packaged session applies
//!   the policy from `ObservabilityConfig::content` the same way on its OTLP
//!   export (`/v1/traces` and the details record on `/v1/logs`).
//! - Declarative rules: `omit_attribute` removes a governed key the run
//!   produced from both destinations without listing it as truncated, and
//!   `replace_literal` rewrites what remains.
//! - Fail closed: a redaction hook that fails or panics for one key, a value
//!   over the per-attribute budget and a value over the retained-operation
//!   budget each leave the run's result whole. The value is truncated with the
//!   marker or omitted, and listed in `xybrid.genai.content.truncated_attributes`.
//! - Request bound: with default budgets the run's span encodes under the
//!   480 KiB request bound in OTLP/JSON, and a raised budget is clamped so the
//!   span is exported with its content truncated rather than dropped.
//! - Per-run overrides: a per-run session or observer
//!   lowers capture below the session the run would otherwise use (the
//!   enclosing operation's, else the default) and never raises it: the run
//!   records content only where both policies capture. Every test here holds
//!   [`NoDefaultSession`], so no other test's default caps its per-run
//!   sessions. The default floor, which needs restrictive defaults, is pinned
//!   in its own process (`tests/per_run_content.rs`).

use super::*;
use crate::observability::semconv::{attr, event, TRUNCATION_MARKER};
use crate::observability::test_support::{
    memory_session_with_content, span_id, NoDefaultSession, Signals,
};
use crate::observability::{
    install_default_session, ContentCapture, ContentPolicy, ObservabilityConfig,
    ObservabilitySession, ObservationOptions, ObservationRef, RedactionRule, Redactor,
};
use opentelemetry_sdk::trace::SpanData;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use xybrid_core::ir::MessageRole;

/// The governed keys a chat run records, in the order it records them.
const GOVERNED: [&str; 3] = [
    attr::SYSTEM_INSTRUCTIONS,
    attr::INPUT_MESSAGES,
    attr::OUTPUT_MESSAGES,
];

const SYSTEM: &str = "answer like a pirate";
const EARLIER_QUESTION: &str = "where is the ship";
const EARLIER_ANSWER: &str = "in the harbour";
const QUESTION: &str = "when does it sail";
const ANSWER: &str = "at dawn, matey";

/// Fake `"onnx"` runtime that answers every request with its text.
struct Answer(String);

impl xybrid_core::runtime_adapter::ModelRuntime for Answer {
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
        Ok(Envelope::new(EnvelopeKind::Text(self.0.clone())))
    }
}

/// A chat model whose runtime answers `answer`.
fn chat_model(answer: &str) -> XybridModel {
    as_chat(test_model_with_runtime(Box::new(Answer(answer.to_owned()))))
}

fn message(role: MessageRole, text: &str) -> Envelope {
    Envelope::new(EnvelopeKind::Text(text.to_owned())).with_role(role)
}

const STAGE_INPUT: &str = "summarise the logbook";
const STAGE_OUTPUT: &str = "calm seas all week";
const COMPLETION: &str = r#"{"choices":[{"message":{"role":"assistant","content":"calm seas all week"},"finish_reason":"stop"}]}"#;

/// A loopback OpenAI-compatible gateway that answers every chat completion
/// with [`STAGE_OUTPUT`].
fn gateway() -> httpmock::MockServer {
    let server = httpmock::MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .body(COMPLETION);
    });
    server
}

/// A pipeline run of [`STAGE_INPUT`] through one cloud chat stage served by
/// `gateway`, observed by `signals`. Returns the stage's reference.
fn run_cloud_stage(gateway: &httpmock::MockServer, signals: &Signals) -> ObservationRef {
    let yaml = format!(
        "name: content-controls\nstages:\n  - id: draft\n    model: loopback-model\n    target: cloud\n    provider: openai\n    gateway_url: \"{}\"\n    api_key: local-test-only\n    timeout_ms: 20000\n",
        gateway.url("/v1")
    );
    let result = crate::pipeline::PipelineRef::from_yaml(&yaml)
        .unwrap()
        .load()
        .unwrap()
        .run_with_options(
            &Envelope::new(EnvelopeKind::Text(STAGE_INPUT.into())),
            &observed(RunOptions::new(), signals),
        )
        .expect("the gateway answers");
    assert_eq!(result.text(), Some(STAGE_OUTPUT));
    result.stages[0]
        .observation_ref
        .clone()
        .expect("the stage is observed")
}

/// A conversation with a system prompt and one earlier exchange.
fn conversation(system: &str) -> ConversationContext {
    let mut context = ConversationContext::new().with_system(message(MessageRole::System, system));
    context.push(message(MessageRole::User, EARLIER_QUESTION));
    context.push(message(MessageRole::Assistant, EARLIER_ANSWER));
    context
}

/// A batch run of `question` in `context`, observed by `signals`.
fn run_in_context(
    model: &XybridModel,
    question: &str,
    context: &ConversationContext,
    signals: &Signals,
) -> crate::result::InferenceResult {
    model
        .run_with_context_options(
            &message(MessageRole::User, question),
            context,
            &observed(RunOptions::new(), signals),
        )
        .expect("content controls never fail a run")
}

/// A streamed run of `question` in `context`, observed by `signals`.
fn stream_in_context(
    model: &XybridModel,
    question: &str,
    context: &ConversationContext,
    signals: &Signals,
) -> crate::result::InferenceResult {
    let mut streamed = String::new();
    let result = model
        .run_streaming_with_context_options(
            &message(MessageRole::User, question),
            context,
            &observed(RunOptions::new(), signals),
            |token| {
                streamed.push_str(&token.token);
                Ok(())
            },
        )
        .expect("content controls never fail a run");
    assert_eq!(
        Some(streamed.as_str()),
        result.text(),
        "the stream is whole"
    );
    result
}

/// What one observed run exported of its content.
#[derive(Debug)]
struct Exported {
    /// Governed content on the run's span, parsed from its JSON string.
    span: BTreeMap<String, Value>,
    /// `xybrid.genai.content.truncated_attributes` on the run's span.
    span_truncated: Vec<String>,
    /// Governed content on the run's details event, if it emitted one.
    event: Option<BTreeMap<String, Value>>,
    /// `xybrid.genai.content.truncated_attributes` on that event.
    event_truncated: Vec<String>,
}

impl Exported {
    /// The run of `reference`, as `signals` recorded it.
    fn of(signals: &Signals, reference: &ObservationRef) -> Self {
        let span = signals.spans.by_ref(reference);
        let details: Vec<BTreeMap<String, Value>> = signals
            .logs
            .get_emitted_logs()
            .expect("logs")
            .iter()
            .filter(|l| l.record.event_name() == Some(event::INFERENCE_DETAILS))
            .filter(|l| {
                l.record
                    .trace_context()
                    .is_some_and(|t| t.span_id.to_string() == reference.span_id)
            })
            .map(|l| {
                l.record
                    .attributes_iter()
                    .map(|(key, value)| (key.to_string(), any_to_json(value)))
                    .collect()
            })
            .collect();
        assert!(details.len() <= 1, "one details event per run: {details:?}");
        let event = details.into_iter().next();
        let event_truncated = event
            .as_ref()
            .and_then(|e| e.get(attr::CONTENT_TRUNCATED_ATTRIBUTES))
            .map(|v| {
                v.as_array()
                    .expect("truncated attributes are a list")
                    .iter()
                    .map(|k| k.as_str().expect("keys are strings").to_owned())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            span: governed_span_content(&span),
            span_truncated: span_truncated(&span),
            event: event.map(|e| {
                e.into_iter()
                    .filter(|(key, _)| GOVERNED.contains(&key.as_str()))
                    .collect()
            }),
            event_truncated,
        }
    }

    /// Governed content on the details event (empty without one).
    fn event_content(&self) -> BTreeMap<String, Value> {
        self.event.clone().unwrap_or_default()
    }
}

fn governed_span_content(span: &SpanData) -> BTreeMap<String, Value> {
    span.attributes
        .iter()
        .filter(|kv| GOVERNED.contains(&kv.key.as_str()))
        .map(|kv| {
            let json = kv.value.as_str();
            let value = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("{} is not JSON ({e}): {json}", kv.key));
            (kv.key.to_string(), value)
        })
        .collect()
}

fn span_truncated(span: &SpanData) -> Vec<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == attr::CONTENT_TRUNCATED_ATTRIBUTES)
        .map(|kv| match &kv.value {
            opentelemetry::Value::Array(opentelemetry::Array::String(keys)) => {
                keys.iter().map(|k| k.as_str().to_owned()).collect()
            }
            other => panic!("truncated attributes are a string array: {other:?}"),
        })
        .unwrap_or_default()
}

fn any_to_json(value: &opentelemetry::logs::AnyValue) -> Value {
    use opentelemetry::logs::AnyValue;
    match value {
        AnyValue::Int(v) => Value::from(*v),
        AnyValue::Double(v) => Value::from(*v),
        AnyValue::String(v) => Value::from(v.as_str()),
        AnyValue::Boolean(v) => Value::from(*v),
        AnyValue::Bytes(v) => Value::from(v.as_slice()),
        AnyValue::ListAny(items) => Value::Array(items.iter().map(any_to_json).collect()),
        AnyValue::Map(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.to_string(), any_to_json(v)))
                .collect(),
        ),
        other => panic!("unexpected log value {other:?}"),
    }
}

/// Every text part of a governed value, in order.
fn texts(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                if map.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = map.get("content").and_then(Value::as_str) {
                        out.push(text.to_owned());
                    }
                }
                map.values().for_each(|v| walk(v, out));
            }
            Value::Array(items) => items.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    walk(value, &mut out);
    out
}

/// Whether any attribute of any span the session ended mentions `needle`.
fn on_any_span(signals: &Signals, needle: &str) -> bool {
    signals.spans.spans().iter().any(|s| {
        s.attributes
            .iter()
            .any(|kv| kv.value.as_str().contains(needle))
    })
}

/// Whether any attribute of any log record the session emitted mentions
/// `needle`.
fn on_any_log(signals: &Signals, needle: &str) -> bool {
    signals
        .logs
        .get_emitted_logs()
        .expect("logs")
        .iter()
        .any(|l| {
            l.record
                .attributes_iter()
                .any(|(_, v)| any_to_json(v).to_string().contains(needle))
        })
}

/// Span ids of the spans the session ended whose attributes mention `needle`.
fn spans_mentioning(signals: &Signals, needle: &str) -> BTreeSet<String> {
    signals
        .spans
        .spans()
        .iter()
        .filter(|s| {
            s.attributes
                .iter()
                .any(|kv| kv.value.as_str().contains(needle))
        })
        .map(span_id)
        .collect()
}

/// The span id (`"none"` without a trace context) and event name of every
/// log record the session emitted whose attributes mention `needle`, sorted.
fn logs_mentioning(signals: &Signals, needle: &str) -> Vec<(String, Option<String>)> {
    let mut out: Vec<(String, Option<String>)> = signals
        .logs
        .get_emitted_logs()
        .expect("logs")
        .iter()
        .filter(|l| {
            l.record
                .attributes_iter()
                .any(|(_, v)| any_to_json(v).to_string().contains(needle))
        })
        .map(|l| {
            (
                l.record
                    .trace_context()
                    .map_or_else(|| "none".to_owned(), |t| t.span_id.to_string()),
                l.record.event_name().map(str::to_owned),
            )
        })
        .collect();
    out.sort();
    out
}

/// Whether anything the session exported mentions `needle`.
fn exported_anywhere(signals: &Signals, needle: &str) -> bool {
    on_any_span(signals, needle) || on_any_log(signals, needle)
}

/// The texts a run records under each governed key, in key order.
type Expected = [(&'static str, &'static [&'static str])];

/// What a context run of [`QUESTION`] records.
const CONVERSATION: &Expected = &[
    (
        attr::INPUT_MESSAGES,
        &[EARLIER_QUESTION, EARLIER_ANSWER, QUESTION],
    ),
    (attr::OUTPUT_MESSAGES, &[ANSWER]),
    // The system prompt is instructions, never a message.
    (attr::SYSTEM_INSTRUCTIONS, &[SYSTEM]),
];

/// What a pipeline cloud stage run of [`STAGE_INPUT`] records.
const STAGE: &Expected = &[
    (attr::INPUT_MESSAGES, &[STAGE_INPUT]),
    (attr::OUTPUT_MESSAGES, &[STAGE_OUTPUT]),
];

fn assert_content(content: &BTreeMap<String, Value>, expected: &Expected, what: &str) {
    assert_eq!(
        content.keys().map(String::as_str).collect::<Vec<_>>(),
        expected.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        "{what}: {content:?}"
    );
    for (key, texts_of_key) in expected {
        assert_eq!(texts(&content[*key]), *texts_of_key, "{what}: {key}");
    }
}

// ---------------------------------------------------------------------------
// E1.1: capture modes on automatic runs.
// ---------------------------------------------------------------------------

#[test]
fn each_capture_mode_puts_automatic_run_content_on_its_destinations() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    let gateway = gateway();
    for capture in [
        ContentCapture::NoContent,
        ContentCapture::SpanOnly,
        ContentCapture::EventOnly,
        ContentCapture::SpanAndEvent,
    ] {
        let on_spans = matches!(
            capture,
            ContentCapture::SpanOnly | ContentCapture::SpanAndEvent
        );
        let on_events = matches!(
            capture,
            ContentCapture::EventOnly | ContentCapture::SpanAndEvent
        );
        let signals = memory_session_with_content(ContentPolicy::default().with_capture(capture));
        let model = chat_model(ANSWER);
        let context = conversation(SYSTEM);
        let mut runs: Vec<(&str, ObservationRef, &Expected)> = Vec::new();
        for (kind, result) in [
            (
                "batch",
                run_in_context(&model, QUESTION, &context, &signals),
            ),
            (
                "streamed",
                stream_in_context(&model, QUESTION, &context, &signals),
            ),
        ] {
            assert_eq!(result.text(), Some(ANSWER), "{capture:?} {kind}");
            let reference = result.observation_ref().expect("the run is observed");
            runs.push((kind, reference.clone(), CONVERSATION));
        }
        runs.push(("pipeline stage", run_cloud_stage(&gateway, &signals), STAGE));

        for (kind, reference, expected) in &runs {
            let (kind, expected) = (*kind, *expected);
            let what = format!("{capture:?} {kind}");
            let exported = Exported::of(&signals, reference);
            if on_spans {
                assert_content(&exported.span, expected, &format!("{what} span"));
            } else {
                assert!(exported.span.is_empty(), "{what}: {:?}", exported.span);
            }
            if on_events {
                assert_content(
                    &exported.event_content(),
                    expected,
                    &format!("{what} event"),
                );
            } else {
                assert!(
                    exported.event.is_none(),
                    "{what}: no details event without event capture: {:?}",
                    exported.event
                );
            }
            if on_spans && on_events {
                assert_eq!(
                    Some(&exported.span),
                    exported.event.as_ref(),
                    "{what}: both destinations carry the same values"
                );
            }
            assert!(exported.span_truncated.is_empty(), "{what}");
            assert!(exported.event_truncated.is_empty(), "{what}");
        }
        // Nothing records content outside the run's content pipeline: each
        // text is on the spans of the runs that recorded it and on their
        // details events, and on no workflow, attempt or other span and no
        // other log record.
        for text in [
            SYSTEM,
            EARLIER_QUESTION,
            EARLIER_ANSWER,
            QUESTION,
            ANSWER,
            STAGE_INPUT,
            STAGE_OUTPUT,
        ] {
            let carriers: BTreeSet<String> = runs
                .iter()
                .filter(|(_, _, expected)| {
                    expected
                        .iter()
                        .any(|(_, texts_of_key)| texts_of_key.contains(&text))
                })
                .map(|(_, reference, _)| reference.span_id.clone())
                .collect();
            assert!(!carriers.is_empty(), "{text:?} is recorded by some run");
            assert_eq!(
                spans_mentioning(&signals, text),
                if on_spans {
                    carriers.clone()
                } else {
                    BTreeSet::new()
                },
                "{capture:?}: span ids of the spans carrying {text:?}"
            );
            let details: Vec<(String, Option<String>)> = if on_events {
                carriers
                    .iter()
                    .map(|id| (id.clone(), Some(event::INFERENCE_DETAILS.to_owned())))
                    .collect()
            } else {
                Vec::new()
            };
            assert_eq!(
                logs_mentioning(&signals, text),
                details,
                "{capture:?}: (span id, event name) of the log records carrying {text:?}"
            );
        }
    }
}

#[test]
fn a_packaged_session_exports_run_content_where_its_capture_mode_says() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    // E1.1 on an SDK-owned pipeline: the policy set with
    // `ObservabilityConfig::content` governs what an automatic run exports
    // over OTLP, on `/v1/traces` in span modes and on the details record on
    // `/v1/logs` in event modes. A default capture mode can match at most one
    // of the two cases.
    use httpmock::{Method::POST, Regex};
    use xybrid_observability::otlp::{EnvironmentSource, OtlpConfig, Protocol, SignalEndpoint};
    /// Hang detector for flush and shutdown; their expiry is the failure.
    const HANG: Duration = Duration::from_secs(20);
    const TEXTS: [&str; 5] = [SYSTEM, EARLIER_QUESTION, EARLIER_ANSWER, QUESTION, ANSWER];
    let any_text = || Regex::new(&TEXTS.join("|")).expect("the texts are literal patterns");

    let model = chat_model(ANSWER);
    let context = conversation(SYSTEM);
    for capture in [ContentCapture::SpanOnly, ContentCapture::EventOnly] {
        let collector = httpmock::MockServer::start();
        // httpmock answers a request with the first mock it matches, in
        // creation order: a request carrying all of the run's content, then
        // one carrying any of it, then any other request to the path.
        let mock = |path: &'static str, all: bool, any: bool| {
            collector.mock(|when, then| {
                let mut when = when.method(POST).path(path);
                if all {
                    for text in TEXTS {
                        when = when.body_contains(text);
                    }
                    if path == "/v1/logs" {
                        when = when.body_contains(event::INFERENCE_DETAILS);
                    }
                }
                if any {
                    when.body_matches(any_text());
                }
                then.status(200);
            })
        };
        let traces_whole = mock("/v1/traces", true, false);
        let traces_partial = mock("/v1/traces", false, true);
        let traces = mock("/v1/traces", false, false);
        let logs_whole = mock("/v1/logs", true, false);
        let logs_partial = mock("/v1/logs", false, true);
        let _logs = mock("/v1/logs", false, false);
        let endpoint = |signal: &str| {
            Some(
                SignalEndpoint::new(collector.url(format!("/v1/{signal}")))
                    .with_protocol(Protocol::HttpJson),
            )
        };
        let session = ObservabilitySession::new(
            ObservabilityConfig::otlp(
                OtlpConfig::default()
                    .with_environment(EnvironmentSource::Disabled)
                    .with_traces(endpoint("traces"))
                    .with_logs(endpoint("logs")),
            )
            .content(ContentPolicy::default().with_capture(capture))
            .device_attributes(false),
        )
        .expect("a valid packaged session");

        let result = model
            .run_with_context_options(
                &message(MessageRole::User, QUESTION),
                &context,
                &RunOptions::new()
                    .with_observation(ObservationOptions::default().with_session(&session)),
            )
            .expect("the run succeeds");
        assert_eq!(result.text(), Some(ANSWER), "{capture:?}");
        session
            .flush(HANG)
            .expect("the collector accepts every export");

        let on_spans = matches!(capture, ContentCapture::SpanOnly);
        assert!(
            traces_whole.hits() + traces.hits() > 0,
            "{capture:?}: the run's spans were exported"
        );
        assert_eq!(
            (traces_whole.hits() > 0, traces_partial.hits()),
            (on_spans, 0),
            "{capture:?}: run content on /v1/traces"
        );
        assert_eq!(
            (logs_whole.hits() > 0, logs_partial.hits()),
            (!on_spans, 0),
            "{capture:?}: run content on the /v1/logs details record"
        );
        session.shutdown(HANG).expect("the session shuts down");
    }
}

// ---------------------------------------------------------------------------
// E1.2: declarative rules on governed keys an automatic run produced.
// ---------------------------------------------------------------------------

#[test]
fn omit_rules_remove_governed_keys_an_automatic_run_produced() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    let replace = RedactionRule::replace_literal("secret", "[R]").unwrap();
    let policy = |omit: bool| {
        let mut rules = vec![replace.clone()];
        if omit {
            rules.push(RedactionRule::omit_attribute(attr::SYSTEM_INSTRUCTIONS).unwrap());
            rules.push(RedactionRule::omit_attribute(attr::OUTPUT_MESSAGES).unwrap());
        }
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_rules(rules)
    };
    let model = chat_model("the secret answer");
    let context = conversation("keep the secret system prompt");

    // Control: without the omit rules the run records all three keys, every
    // one rewritten by the replacement rule.
    let control = memory_session_with_content(policy(false));
    let result = run_in_context(&model, "tell me the secret", &context, &control);
    let exported = Exported::of(&control, result.observation_ref().unwrap());
    for content in [&exported.span, &exported.event_content()] {
        assert_eq!(content.len(), GOVERNED.len(), "{content:?}");
        for key in GOVERNED {
            let json = content[key].to_string();
            assert!(
                json.contains("[R]") && !json.contains("secret"),
                "{key}: {json}"
            );
        }
    }

    let signals = memory_session_with_content(policy(true));
    for result in [
        run_in_context(&model, "tell me the secret", &context, &signals),
        stream_in_context(&model, "tell me the secret", &context, &signals),
    ] {
        // Rules govern what is exported, never what the caller receives.
        assert_eq!(result.text(), Some("the secret answer"));
        let exported = Exported::of(&signals, result.observation_ref().unwrap());
        for (destination, content) in [
            ("span", exported.span.clone()),
            ("event", exported.event_content()),
        ] {
            assert_eq!(
                content.keys().map(String::as_str).collect::<Vec<_>>(),
                [attr::INPUT_MESSAGES],
                "{destination}: the omitted keys are gone, the input stays"
            );
            assert_eq!(
                texts(&content[attr::INPUT_MESSAGES]),
                [EARLIER_QUESTION, EARLIER_ANSWER, "tell me the [R]"],
                "{destination}"
            );
        }
        // An omission is intentional: it is not listed as truncated.
        assert!(exported.span_truncated.is_empty(), "{exported:?}");
        assert!(exported.event_truncated.is_empty(), "{exported:?}");
    }
    assert!(!exported_anywhere(&signals, "secret"));
    assert!(!exported_anywhere(&signals, "system prompt"));
    assert!(!exported_anywhere(&signals, "the [R] answer"));
}

// ---------------------------------------------------------------------------
// E1.3: content fails closed on automatic runs.
// ---------------------------------------------------------------------------

/// Redacts nothing, and fails for one key.
struct FailingFor(&'static str);

impl Redactor for FailingFor {
    fn redact(&self, _kind: &str, value: Value) -> Option<Value> {
        Some(value)
    }

    fn try_redact(
        &self,
        kind: &str,
        value: Value,
    ) -> Result<Option<Value>, xybrid_observability::RedactionError> {
        if kind == self.0 {
            Err(xybrid_observability::RedactionError)
        } else {
            Ok(Some(value))
        }
    }
}

#[test]
fn a_redaction_hook_that_fails_for_one_key_omits_and_lists_only_that_key() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    let signals = memory_session_with_content(
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_redactor(FailingFor(attr::OUTPUT_MESSAGES)),
    );
    let model = chat_model(ANSWER);
    let context = conversation(SYSTEM);
    for result in [
        run_in_context(&model, QUESTION, &context, &signals),
        stream_in_context(&model, QUESTION, &context, &signals),
    ] {
        assert_eq!(result.text(), Some(ANSWER), "the run is whole");
        let reference = result.observation_ref().unwrap();
        let span = signals.spans.by_ref(reference);
        assert_eq!(
            span.status,
            opentelemetry::trace::Status::Unset,
            "a failed hook is not a failed run"
        );
        let exported = Exported::of(&signals, reference);
        for (destination, content, truncated) in [
            ("span", exported.span.clone(), &exported.span_truncated),
            ("event", exported.event_content(), &exported.event_truncated),
        ] {
            assert_eq!(
                content.keys().map(String::as_str).collect::<Vec<_>>(),
                [attr::INPUT_MESSAGES, attr::SYSTEM_INSTRUCTIONS],
                "{destination}: only the failed key is left out"
            );
            assert_eq!(texts(&content[attr::SYSTEM_INSTRUCTIONS]), [SYSTEM]);
            assert_eq!(truncated, &[attr::OUTPUT_MESSAGES], "{destination}");
        }
    }
    assert!(!exported_anywhere(&signals, ANSWER));
    assert!(
        !signals
            .log_events()
            .iter()
            .any(|e| e == event::OPERATION_EXCEPTION),
        "{:?}",
        signals.log_events()
    );
}

#[test]
fn content_over_its_byte_budgets_is_truncated_or_omitted_and_listed() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    // Per attribute: a long input is cut to the budget with the marker and
    // stays valid JSON; the short output fits whole.
    const BUDGET: usize = 256;
    let long = format!("{QUESTION} {}", "x".repeat(4096));
    let signals = memory_session_with_content(
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_max_attribute_bytes(BUDGET),
    );
    let model = chat_model(ANSWER);
    let result = model
        .run_with_options(
            &message(MessageRole::User, &long),
            &observed(RunOptions::new(), &signals),
        )
        .unwrap();
    assert_eq!(result.text(), Some(ANSWER), "the run is whole");
    let exported = Exported::of(&signals, result.observation_ref().unwrap());
    let span = signals.spans.by_ref(result.observation_ref().unwrap());
    let input_json = span
        .attributes
        .iter()
        .find(|kv| kv.key.as_str() == attr::INPUT_MESSAGES)
        .map(|kv| kv.value.as_str().into_owned())
        .expect("the truncated input is kept");
    assert!(input_json.len() <= BUDGET, "{} bytes", input_json.len());
    for (destination, content, truncated) in [
        ("span", exported.span.clone(), &exported.span_truncated),
        ("event", exported.event_content(), &exported.event_truncated),
    ] {
        let input = texts(&content[attr::INPUT_MESSAGES]);
        assert_eq!(input.len(), 1, "{destination}: {input:?}");
        assert!(
            input[0].starts_with(QUESTION) && input[0].ends_with(TRUNCATION_MARKER),
            "{destination}: {input:?}"
        );
        assert!(input[0].len() < long.len(), "{destination}");
        assert_eq!(texts(&content[attr::OUTPUT_MESSAGES]), [ANSWER]);
        assert_eq!(truncated, &[attr::INPUT_MESSAGES], "{destination}");
    }

    // Per operation: the system instructions are recorded first and fit;
    // the large input would overrun what the run may retain, so it is
    // omitted and listed; the output recorded after it still fits.
    let big = format!("{QUESTION} {}", "y".repeat(4096));
    let signals = memory_session_with_content(
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_max_retained_operation_bytes(2048),
    );
    let result = run_in_context(&model, &big, &conversation(SYSTEM), &signals);
    assert_eq!(result.text(), Some(ANSWER), "the run is whole");
    let exported = Exported::of(&signals, result.observation_ref().unwrap());
    for (destination, content, truncated) in [
        ("span", exported.span.clone(), &exported.span_truncated),
        ("event", exported.event_content(), &exported.event_truncated),
    ] {
        assert_eq!(
            content.keys().map(String::as_str).collect::<Vec<_>>(),
            [attr::OUTPUT_MESSAGES, attr::SYSTEM_INSTRUCTIONS],
            "{destination}"
        );
        assert_eq!(texts(&content[attr::SYSTEM_INSTRUCTIONS]), [SYSTEM]);
        assert_eq!(texts(&content[attr::OUTPUT_MESSAGES]), [ANSWER]);
        assert_eq!(truncated, &[attr::INPUT_MESSAGES], "{destination}");
    }
    assert!(!exported_anywhere(&signals, "yyyy"));
}

// ---------------------------------------------------------------------------
// A redaction hook that panics fails closed on automatic runs.
// ---------------------------------------------------------------------------

/// Redacts nothing, and panics for one key.
struct PanicsFor(&'static str);

impl Redactor for PanicsFor {
    fn redact(&self, kind: &str, value: Value) -> Option<Value> {
        assert!(kind != self.0, "redaction hook panicked on {kind}");
        Some(value)
    }
}

#[test]
fn a_redaction_hook_that_panics_fails_closed_and_the_stream_is_whole() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    // The panic never reaches the run: every token is delivered, the run
    // succeeds, and the key the hook panicked on is omitted and listed on both
    // destinations, exactly as for a hook that returns an error. The session
    // stays usable: each later run calls the hook (and recovers) again.
    let signals = memory_session_with_content(
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_redactor(PanicsFor(attr::OUTPUT_MESSAGES)),
    );
    let model = chat_model(ANSWER);
    let context = conversation(SYSTEM);
    for round in 0..2 {
        let mut tokens: Vec<String> = Vec::new();
        let streamed = model
            .run_streaming_with_context_options(
                &message(MessageRole::User, QUESTION),
                &context,
                &observed(RunOptions::new(), &signals),
                |token| {
                    tokens.push(token.token.clone());
                    Ok(())
                },
            )
            .expect("a panicking hook never fails a run");
        assert!(!tokens.is_empty(), "round {round}: tokens were delivered");
        assert_eq!(
            tokens.concat(),
            ANSWER,
            "round {round}: every token arrived"
        );
        let batch = run_in_context(&model, QUESTION, &context, &signals);
        for result in [streamed, batch] {
            assert_eq!(
                result.text(),
                Some(ANSWER),
                "round {round}: the run is whole"
            );
            let reference = result.observation_ref().unwrap();
            let span = signals.spans.by_ref(reference);
            assert_eq!(
                span.status,
                opentelemetry::trace::Status::Unset,
                "round {round}: a panicking hook is not a failed run"
            );
            let exported = Exported::of(&signals, reference);
            for (destination, content, truncated) in [
                ("span", exported.span.clone(), &exported.span_truncated),
                ("event", exported.event_content(), &exported.event_truncated),
            ] {
                assert_eq!(
                    content.keys().map(String::as_str).collect::<Vec<_>>(),
                    [attr::INPUT_MESSAGES, attr::SYSTEM_INSTRUCTIONS],
                    "round {round}, {destination}: only the key it panicked on is left out"
                );
                assert_eq!(truncated, &[attr::OUTPUT_MESSAGES], "{destination}");
            }
        }
    }
    assert!(!exported_anywhere(&signals, ANSWER));
}

// ---------------------------------------------------------------------------
// Content budgets keep a run's span within one export request
// (the clamp itself: `a_raised_content_budget_is_clamped_so_the_run_span_is_exported`).
// ---------------------------------------------------------------------------

#[test]
fn every_governed_key_at_its_default_maximum_encodes_under_the_request_bound_in_json() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    // A guard on the defaults, not on the clamp (the next test pins that): at
    // the default 16 KiB attribute budget three governed keys stay far under
    // the default retained budget, so this passes with or without the clamp
    // and fails only if the effective default budgets let these keys outgrow
    // one request.
    // A system prompt, a question and an answer each far over the attribute
    // budget, and quote-heavy so that OTLP/JSON doubles them: the run's span
    // encodes under the 480 KiB request bound in OTLP/JSON, and every governed
    // key is cut to the budget and listed.
    use xybrid_observability::semconv::{
        DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES, DEFAULT_MAX_REQUEST_BODY_BYTES,
    };
    let quoted = |text: &str| format!("{text} {}", "\"".repeat(120_000));
    let signals = memory_session_with_content(
        ContentPolicy::default().with_capture(ContentCapture::SpanOnly),
    );
    let model = chat_model(&quoted(ANSWER));
    let result = run_in_context(
        &model,
        &quoted(QUESTION),
        &conversation(&quoted(SYSTEM)),
        &signals,
    );
    let span = signals.spans.by_ref(result.observation_ref().unwrap());
    let resource = opentelemetry_sdk::Resource::builder_empty()
        .with_service_name("svc")
        .build();
    let json = xybrid_observability::otlp::trace_json(vec![span.clone()], &resource).unwrap();
    assert!(
        json.len() < DEFAULT_MAX_REQUEST_BODY_BYTES,
        "{} bytes of OTLP/JSON",
        json.len()
    );
    for key in GOVERNED {
        let value = span
            .attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.as_str().into_owned())
            .unwrap_or_else(|| panic!("{key} is kept, truncated"));
        assert!(
            value.len() <= DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES,
            "{key}: {}",
            value.len()
        );
        assert!(value.contains(TRUNCATION_MARKER), "{key}");
    }
    assert_eq!(span_truncated(&span), GOVERNED);
}

#[test]
fn a_raised_content_budget_is_clamped_so_the_run_span_is_exported() {
    // No other test's default caps this test's per-run sessions.
    let _no_default = NoDefaultSession::hold();
    // A packaged session over OTLP/JSON with the attribute budget raised to
    // 400 KB: the answer, 300 KB of quotes, would encode to about 800 KB and
    // its span be dropped whole as an oversized item. The budget is clamped
    // to fit the request bound, so the answer is truncated, listed, and the
    // span exported.
    use httpmock::Method::POST;
    use xybrid_observability::otlp::{EnvironmentSource, OtlpConfig, Protocol, SignalEndpoint};
    /// Hang detector for flush and shutdown; their expiry is the failure.
    const HANG: Duration = Duration::from_secs(20);

    let collector = httpmock::MockServer::start();
    let truncated = collector.mock(|when, then| {
        when.method(POST)
            .path("/v1/traces")
            .body_contains(TRUNCATION_MARKER)
            .body_contains(attr::CONTENT_TRUNCATED_ATTRIBUTES);
        then.status(200);
    });
    let _traces = collector.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(200);
    });
    let session = ObservabilitySession::new(
        ObservabilityConfig::otlp(
            OtlpConfig::default()
                .with_environment(EnvironmentSource::Disabled)
                .with_traces(Some(
                    SignalEndpoint::new(collector.url("/v1/traces"))
                        .with_protocol(Protocol::HttpJson),
                )),
        )
        .content(
            ContentPolicy::default()
                .with_capture(ContentCapture::SpanOnly)
                .with_max_attribute_bytes(400_000)
                .with_max_retained_operation_bytes(1_000_000),
        )
        .device_attributes(false),
    )
    .expect("a valid packaged session");
    let answer = "\"".repeat(300_000);
    let result = chat_model(&answer)
        .run_with_options(
            &message(MessageRole::User, QUESTION),
            &RunOptions::new()
                .with_observation(ObservationOptions::default().with_session(&session)),
        )
        .expect("the run succeeds");
    assert_eq!(result.text(), Some(answer.as_str()));
    session.flush(HANG).expect("every span fits one request");
    let traces = session.stats().expect("a packaged session").traces;
    assert!(traces.accepted > 0, "{traces:?}");
    assert_eq!(
        (traces.exported, traces.dropped, traces.failed),
        (traces.accepted, 0, 0),
        "no span was dropped as oversized"
    );
    assert_eq!(
        truncated.hits(),
        1,
        "the run's span arrived with its answer truncated and listed"
    );
    session.shutdown(HANG).expect("the session shuts down");
}

// ---------------------------------------------------------------------------
// Per-run content overrides: a per-run session
// or observer can lower content capture below the session the run would
// otherwise use (the enclosing operation's, else the default), never raise it.
// ---------------------------------------------------------------------------

/// A context run of [`QUESTION`] with `observation` as its per-run options
/// (`None`: the default session).
fn run_with(model: &XybridModel, observation: Option<ObservationOptions>) -> ObservationRef {
    let options = match observation {
        Some(observation) => RunOptions::new().with_observation(observation),
        None => RunOptions::new(),
    };
    model
        .run_with_context_options(
            &message(MessageRole::User, QUESTION),
            &conversation(SYSTEM),
            &options,
        )
        .expect("content controls never fail a run")
        .observation_ref()
        .cloned()
        .expect("the run is observed")
}

fn with_session(signals: &Signals) -> Option<ObservationOptions> {
    Some(ObservationOptions::default().with_session(&signals.session))
}

/// Asserts what the run of `reference` exported to `signals`: its content on
/// the span and on the details event, each either the conversation or none.
fn assert_destinations(signals: &Signals, reference: &ObservationRef, span: bool, event: bool) {
    let exported = Exported::of(signals, reference);
    if span {
        assert_content(&exported.span, CONVERSATION, "span");
    } else {
        assert!(
            exported.span.is_empty(),
            "span content: {:?}",
            exported.span
        );
    }
    if event {
        assert_content(&exported.event_content(), CONVERSATION, "event");
    } else {
        assert!(
            exported.event.is_none(),
            "details event: {:?}",
            exported.event
        );
    }
}

#[test]
fn a_per_run_session_or_observer_lowers_content_capture_below_the_default() {
    let _no_default = NoDefaultSession::hold();
    let model = chat_model(ANSWER);
    let default = memory_session_with_content(
        ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent),
    );
    install_default_session(default.session.clone());
    let metadata_only = memory_session_with_content(ContentPolicy::metadata_only());

    // The default records content on both destinations.
    let plain = run_with(&model, None);
    assert_destinations(&default, &plain, true, true);

    // A metadata-only per-run session records none, on its own export.
    let lowered = run_with(&model, with_session(&metadata_only));
    assert_destinations(&metadata_only, &lowered, false, false);
    assert!(
        default
            .spans
            .spans()
            .iter()
            .all(|s| span_id(s) != lowered.span_id),
        "the run reports to its own session"
    );

    // The default's providers with capture capped for one run: the same
    // destination, metadata only.
    let capped = default
        .session
        .observer()
        .content_capped_by(&ContentPolicy::metadata_only());
    let same_destination = run_with(
        &model,
        Some(ObservationOptions {
            observer: Some(capped),
            ..ObservationOptions::default()
        }),
    );
    assert_destinations(&default, &same_destination, false, false);
    assert!(!on_any_span(&metadata_only, QUESTION) && !on_any_log(&metadata_only, QUESTION));
}

#[test]
fn a_per_run_session_cannot_raise_content_capture_above_its_enclosing_operation() {
    // The default floor installs restrictive defaults, which would cap other
    // tests' per-run sessions: it is pinned in its own test binary
    // (`tests/per_run_content.rs`). Here the floor is the enclosing
    // operation's session, which touches no process-wide state.
    let _no_default = NoDefaultSession::hold();
    let model = chat_model(ANSWER);
    let everything = memory_session_with_content(
        ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent),
    );

    // Control: with no other session, the per-run session's own policy applies.
    let alone = run_with(&model, with_session(&everything));
    assert_destinations(&everything, &alone, true, true);

    // Inside an operation pinned to a capturing-less session, the per-run
    // session records only what both capture.
    for (policy, span, event) in [
        (ContentPolicy::metadata_only(), false, false),
        (
            ContentPolicy::default().with_capture(ContentCapture::SpanOnly),
            true,
            false,
        ),
        (
            ContentPolicy::default().with_capture(ContentCapture::EventOnly),
            false,
            true,
        ),
    ] {
        let enclosing = memory_session_with_content(policy);
        let workflow = crate::observability::start_workflow(
            "floor",
            &ObservationOptions::default().with_session(&enclosing.session),
        );
        let nested = {
            let _scope = workflow.enter();
            run_with(&model, with_session(&everything))
        };
        workflow.end();
        assert_destinations(&everything, &nested, span, event);
        // The run still reports its metadata to the per-run session.
        assert_eq!(everything.spans.by_ref(&nested).name, run_span_name(&model));
        assert!(!on_any_span(&enclosing, QUESTION) && !on_any_log(&enclosing, QUESTION));
    }
}

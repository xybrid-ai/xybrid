//! Loopback proof of the public FFI execution and governance surfaces.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use xybrid_ffi_facade::*;

type Request = (String, Vec<(String, String)>, Vec<u8>);

struct Receiver {
    address: std::net::SocketAddr,
    stopped: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Request>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Receiver {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stopped = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let worker_stop = stopped.clone();
        let received = requests.clone();
        let worker = std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                if worker_stop.load(Ordering::SeqCst) {
                    break;
                }
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let path = first.split_whitespace().nth(1).unwrap().to_owned();
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    let (key, value) = line.trim().split_once(':').unwrap();
                    headers.push((key.to_ascii_lowercase(), value.trim().to_owned()));
                }
                let size = headers
                    .iter()
                    .find(|(key, _)| key == "content-length")
                    .map(|(_, value)| value.parse::<usize>().unwrap())
                    .unwrap_or(0);
                let mut body = vec![0; size];
                reader.read_exact(&mut body).unwrap();
                let gateway = path.contains("chat/completions");
                received.lock().unwrap().push((path, headers, body));
                let response = if gateway {
                    r#"{"choices":[{"message":{"role":"assistant","content":"secret answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#
                } else {
                    "{}"
                };
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).unwrap();
            }
        });
        Self {
            address,
            stopped,
            requests,
            worker: Some(worker),
        }
    }

    fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    fn spans(&self) -> Vec<Value> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(path, _, _)| path == "/v1/traces")
            .flat_map(|(_, _, body)| {
                let value: Value = serde_json::from_slice(body).unwrap();
                value["resourceSpans"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|resource| resource["scopeSpans"].as_array().unwrap().iter())
                    .flat_map(|scope| scope["spans"].as_array().unwrap().iter().cloned())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl Drop for Receiver {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.address);
        self.worker.take().unwrap().join().unwrap();
    }
}

fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["key"] == key)
        .map(|a| &a["value"])
}

/// Serializes the tests here: each installs the process-wide observability.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn execution_refs_content_rules_and_last_owner_cancellation_reach_otlp() {
    let _exclusive = exclusive();
    let receiver = Receiver::start();
    let config = ObservabilityConfigHandle::xybrid("local-test-key".into());
    config.set_traces_endpoint(format!("{}/v1/traces", receiver.url()));
    config.set_json_protocol(true);
    config.set_device_id("facade-test-device".into());
    config
        .add_literal_redaction("secret".into(), "[redacted]".into())
        .unwrap();
    config
        .omit_content_attribute("gen_ai.tool.definitions".into())
        .unwrap();
    config.allow_provider_context(receiver.url()).unwrap();
    assert!(config
        .allow_provider_context("https://provider.example/path".into())
        .is_err());
    config.init().unwrap();

    let parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let controls = ObservationControls {
        traceparent: Some(parent.into()),
        conversation_id: Some("facade-conversation".into()),
        ..Default::default()
    };
    let workflow = WorkflowObservation::start("facade-workflow".into(), controls.clone()).unwrap();
    let workflow_ref = workflow.observation_ref().unwrap();
    let child = ObservationControls {
        traceparent: Some(workflow.context().unwrap().traceparent),
        ..controls.clone()
    };
    let tool = ToolObservation::start(
        ToolCallInfo {
            name: "lookup".into(),
            arguments_json: Some(r#"{"q":"secret query"}"#.into()),
            ..Default::default()
        },
        child.clone(),
    )
    .unwrap();
    let cancelled_ref = tool.observation_ref().unwrap();
    let clone = tool.clone();
    drop(tool);
    assert!(!clone.is_finished());
    clone
        .record_result(r#"{"answer":"secret result"}"#.into())
        .unwrap();
    drop(clone); // final owner cancels, without flushing

    let dir = tempfile::tempdir().unwrap();
    let mut metadata =
        xybrid_core::execution::ModelMetadata::onnx("weightless-tts", "1.0", "missing.onnx");
    metadata
        .metadata
        .insert("task".into(), json!("text-to-speech"));
    std::fs::write(
        dir.path().join("model_metadata.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let model = ModelLoader::from_directory(dir.path().to_string_lossy().into())
        .unwrap()
        .with_observation(child.clone())
        .unwrap()
        .load()
        .unwrap();
    let load_ref = model.load_observation_ref().unwrap();
    // Blank text succeeds without model weights, exercising the real TTS worker terminal.
    let options = RunOptions {
        traceparent: child.traceparent.clone(),
        conversation_id: child.conversation_id.clone(),
        ..Default::default()
    };
    let stream = model
        .run_stream(Envelope::text(" ".into()), options.clone(), None)
        .unwrap();
    let StreamEvent::Complete(terminal) = stream.next().unwrap() else {
        panic!("TTS completes");
    };
    let tts_ref = terminal.observation.unwrap();
    assert!(stream.next().is_none());

    // Governed content: tool definitions are schema content recorded in every
    // capture mode, so only the omit rule can keep them off the export, while
    // other content on the same span still exports.
    let governed = InferenceObservation::start(
        InferenceRequest {
            operation: "chat".into(),
            provider: "openai".into(),
            model: Some("governed-model".into()),
            ..Default::default()
        },
        child.clone(),
    )
    .unwrap();
    let governed_ref = governed.observation_ref().unwrap();
    governed
        .record_system_instructions(r#"[{"type":"text","content":"governed instructions"}]"#.into())
        .unwrap();
    governed
        .record_tool_definitions(r#"[{"type":"function","name":"governed_tool"}]"#.into())
        .unwrap();
    governed.complete();

    // An origin the config never allowlisted: the pipeline still reaches it,
    // but the run's context does not.
    let unlisted = Receiver::start();
    let unlisted_yaml = format!("name: unlisted-pipeline\nstages:\n  - id: draft\n    model: loopback-model\n    target: cloud\n    provider: openai\n    gateway_url: \"{}/gateway/v1\"\n    api_key: local-test-key\n    timeout_ms: 5000\n", unlisted.url());
    let unlisted_result = Pipeline::from_yaml(unlisted_yaml)
        .unwrap()
        .run(Envelope::text("unlisted input".into()), options.clone())
        .unwrap();
    assert_eq!(unlisted_result.stages[0].text(), Some("secret answer"));

    let yaml = format!("name: facade-pipeline\nstages:\n  - id: draft\n    model: loopback-model\n    target: cloud\n    provider: openai\n    gateway_url: \"{}/gateway/v1\"\n    api_key: local-test-key\n    timeout_ms: 5000\n", receiver.url());
    let pipeline = Pipeline::from_yaml(yaml).unwrap();
    let result = pipeline
        .run(Envelope::text("secret input".into()), options)
        .unwrap();
    let pipeline_ref = result.observation.unwrap();
    let stage_ref = result.stages[0].observation.clone().unwrap();
    assert_eq!(result.stages[0].text(), Some("secret answer"));
    workflow.complete();
    workflow.cancel(); // first terminal wins
    observability_flush(5000).unwrap();
    let spans = receiver.spans();
    let by_ref = |reference: &ObservationRef| {
        spans
            .iter()
            .find(|s| s["spanId"] == reference.span_id)
            .unwrap()
    };
    let root = by_ref(&workflow_ref);
    assert_eq!(root["parentSpanId"], "00f067aa0ba902b7");
    assert!(attr(root, "error.type").is_none());
    for reference in [&load_ref, &tts_ref, &pipeline_ref, &cancelled_ref] {
        let span = by_ref(reference);
        assert_eq!(span["traceId"], workflow_ref.trace_id);
        assert_eq!(span["parentSpanId"], workflow_ref.span_id);
        assert_eq!(
            attr(span, "gen_ai.conversation.id").unwrap()["stringValue"],
            "facade-conversation"
        );
    }
    assert_eq!(
        attr(by_ref(&cancelled_ref), "error.type").unwrap()["stringValue"],
        "cancelled"
    );
    let stage = by_ref(&stage_ref);
    assert_eq!(stage["parentSpanId"], pipeline_ref.span_id);
    assert_eq!(
        attr(stage, "gen_ai.usage.input_tokens").unwrap()["intValue"],
        "2"
    );
    assert_eq!(
        attr(stage, "gen_ai.usage.output_tokens").unwrap()["intValue"],
        "3"
    );
    let attempt = spans
        .iter()
        .find(|s| {
            s["parentSpanId"] == stage_ref.span_id
                && attr(s, "xybrid.genai.attempt.reason")
                    .is_some_and(|value| value["stringValue"] == "initial")
        })
        .unwrap();
    assert_eq!(
        attr(attempt, "server.port").unwrap()["intValue"],
        receiver.address.port().to_string()
    );
    let governed_span = by_ref(&governed_ref);
    assert!(
        attr(governed_span, "gen_ai.system_instructions").unwrap()["stringValue"]
            .as_str()
            .unwrap()
            .contains("governed instructions")
    );
    assert!(
        attr(governed_span, "gen_ai.tool.definitions").is_none(),
        "the omit rule removes exactly its attribute"
    );
    {
        let unlisted_requests = unlisted.requests.lock().unwrap();
        let (_, unlisted_headers, _) = unlisted_requests
            .iter()
            .find(|(p, _, _)| p.contains("chat/completions"))
            .unwrap();
        assert!(
            unlisted_headers.iter().all(|(k, _)| k != "traceparent"),
            "a provider origin outside the allowlist receives no context: {unlisted_headers:?}"
        );
    }
    // The allowlisted origin receives the stage attempt's context.
    let requests = receiver.requests.lock().unwrap();
    let (_, headers, _) = requests
        .iter()
        .find(|(p, _, _)| p.contains("chat/completions"))
        .unwrap();
    assert_eq!(
        headers.iter().find(|(k, _)| k == "traceparent").unwrap().1,
        format!(
            "00-{}-{}-01",
            workflow_ref.trace_id,
            attempt["spanId"].as_str().unwrap()
        )
    );
    let exported = requests
        .iter()
        .filter(|(p, _, _)| p == "/v1/traces")
        .map(|(_, _, b)| String::from_utf8_lossy(b))
        .collect::<String>();
    assert!(!exported.contains("secret"));
    assert!(exported.contains("[redacted]"));
    assert!(!exported.contains("governed_tool"));
    assert!(exported.contains("facade-test-device"));
    drop(requests);
    observability_shutdown(5000).unwrap();
}

/// `set_max_content_bytes` bounds every content attribute an automatic run
/// exports: a pipeline cloud stage's long input reaches OTLP cut to the
/// budget with the truncation marker, still valid JSON, and listed as
/// truncated, while its short output exports whole and the caller's result
/// is untouched.
#[test]
fn max_content_bytes_truncates_automatic_run_content_on_the_export() {
    const BUDGET: usize = 256;
    // Bounds only a hang: loopback answers at once, and expiry is the failure.
    const HANG_MS: u32 = 20_000;
    let _exclusive = exclusive();
    let receiver = Receiver::start();
    let config = ObservabilityConfigHandle::xybrid("local-test-key".into());
    config.set_traces_endpoint(format!("{}/v1/traces", receiver.url()));
    config.set_json_protocol(true);
    config.set_max_content_bytes(BUDGET as u32);
    config.init().unwrap();

    let long_input = format!("budget input {}", "x".repeat(4096));
    let yaml = format!("name: budget-pipeline\nstages:\n  - id: draft\n    model: loopback-model\n    target: cloud\n    provider: openai\n    gateway_url: \"{}/gateway/v1\"\n    api_key: local-test-key\n    timeout_ms: 20000\n", receiver.url());
    let result = Pipeline::from_yaml(yaml)
        .unwrap()
        .run(Envelope::text(long_input), RunOptions::default())
        .unwrap();
    assert_eq!(result.stages[0].text(), Some("secret answer"));
    let stage_ref = result.stages[0].observation.clone().unwrap();
    observability_flush(HANG_MS).unwrap();

    let spans = receiver.spans();
    let stage = spans
        .iter()
        .find(|s| s["spanId"] == stage_ref.span_id)
        .unwrap();
    let string_attr = |key: &str| -> String {
        attr(stage, key).unwrap_or_else(|| panic!("{key} is exported"))["stringValue"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let input = string_attr("gen_ai.input.messages");
    assert!(input.len() <= BUDGET, "{} bytes", input.len());
    let messages: Value = serde_json::from_str(&input).expect("still valid JSON");
    let text = messages[0]["parts"][0]["content"].as_str().unwrap();
    assert!(
        text.starts_with("budget input ") && text.ends_with("…[truncated]"),
        "{text}"
    );
    let output = string_attr("gen_ai.output.messages");
    assert!(output.contains("secret answer"), "{output}");
    let truncated: Vec<&str> = attr(stage, "xybrid.genai.content.truncated_attributes")
        .expect("the truncated input is listed")["arrayValue"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["stringValue"].as_str().unwrap())
        .collect();
    assert_eq!(truncated, ["gen_ai.input.messages"]);
    {
        let requests = receiver.requests.lock().unwrap();
        let exported = requests
            .iter()
            .filter(|(p, _, _)| p == "/v1/traces")
            .map(|(_, _, b)| String::from_utf8_lossy(b).into_owned())
            .collect::<String>();
        assert!(
            !exported.contains(&"x".repeat(BUDGET)),
            "no export carries the input beyond the budget"
        );
    }
    observability_shutdown(HANG_MS).unwrap();
}

/// Bound of every hang detector in the stream-close test.
const HANG: Duration = Duration::from_secs(30);

/// What the paced gateway does next on its open completion stream.
enum Pace {
    /// Write one SSE content chunk.
    Token(&'static str),
    /// Wait for the client to close the stream, and report whether it did.
    ExpectClose,
}

/// A loopback Xybrid platform for speculative chat models. The registry
/// resolves any id to a GGUF chat model whose weights download is held open,
/// so the model answers from the gateway for the whole test. The gateway's
/// completion stream writes one SSE chunk per [`Pace`] the test sends.
struct PacedPlatform {
    address: std::net::SocketAddr,
    pace: std::sync::mpsc::Sender<Pace>,
    closed: std::sync::mpsc::Receiver<bool>,
    held: Arc<Mutex<Vec<TcpStream>>>,
    stopped: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl PacedPlatform {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (pace, paces) = std::sync::mpsc::channel();
        let paces = Arc::new(Mutex::new(paces));
        let (report, closed) = std::sync::mpsc::channel();
        let held = Arc::new(Mutex::new(Vec::new()));
        let stopped = Arc::new(AtomicBool::new(false));
        let worker = {
            let held = held.clone();
            let stopped = stopped.clone();
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stopped.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let (paces, report, held) = (paces.clone(), report.clone(), held.clone());
                    std::thread::spawn(move || {
                        serve_platform(stream, address, &paces, &report, &held)
                    });
                }
            })
        };
        Self {
            address,
            pace,
            closed,
            held,
            stopped,
            worker: Some(worker),
        }
    }

    fn url(&self) -> String {
        format!("http://{}", self.address)
    }
}

impl Drop for PacedPlatform {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.address);
        self.worker.take().unwrap().join().unwrap();
        // Ends the held downloads (they fail; nothing waits for them).
        self.held.lock().unwrap().clear();
    }
}

fn serve_platform(
    mut stream: TcpStream,
    address: std::net::SocketAddr,
    paces: &Mutex<std::sync::mpsc::Receiver<Pace>>,
    report: &std::sync::mpsc::Sender<bool>,
    held: &Mutex<Vec<TcpStream>>,
) {
    stream.set_read_timeout(Some(HANG)).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut first = String::new();
    if reader.read_line(&mut first).is_err() {
        return;
    }
    let Some(target) = first.split_whitespace().nth(1) else {
        return;
    };
    let path = target.split('?').next().unwrap_or_default().to_owned();
    let mut size = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        if line == "\r\n" {
            break;
        }
        if let Some((key, value)) = line.trim().split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                size = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0; size];
    if reader.read_exact(&mut body).is_err() {
        return;
    }

    if let Some(id) = path
        .strip_prefix("/v1/models/")
        .and_then(|rest| rest.strip_suffix("/resolve"))
    {
        let mut metadata = serde_json::to_value(xybrid_core::execution::ModelMetadata::onnx(
            id,
            "1",
            "model.onnx",
        ))
        .unwrap();
        metadata["execution_template"] = json!({"type": "Gguf", "model_file": "model.gguf"});
        metadata["files"] = json!(["model.gguf"]);
        let answer = json!({"mask": id, "platform": "universal", "resolved": {
            "hf_repo": format!("test/{id}"), "file": "model.gguf",
            "download_url": format!("http://{address}/weights/{id}.gguf"),
            "format": "gguf", "quantization": "none", "size_bytes": 4, "sha256": "",
            "passthrough": true, "model_metadata": metadata
        }})
        .to_string();
        let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}", answer.len());
    } else if path.starts_with("/weights/") {
        // Never answered while the test runs: the model keeps serving from
        // the gateway.
        held.lock().unwrap().push(stream);
    } else if path == "/v1/chat/completions" {
        if write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").is_err() {
            return;
        }
        let paces = paces.lock().unwrap();
        loop {
            match paces.recv_timeout(HANG) {
                Ok(Pace::Token(text)) => {
                    let chunk = json!({"id": "chatcmpl-paced", "object": "chat.completion.chunk",
                        "created": 1, "model": "paced",
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]});
                    if write!(stream, "data: {chunk}\n\n").is_err() {
                        return;
                    }
                }
                Ok(Pace::ExpectClose) => {
                    let mut byte = [0u8; 1];
                    let closed = match reader.read(&mut byte) {
                        Ok(read) => read == 0,
                        Err(error) => !matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ),
                    };
                    let _ = report.send(closed);
                    return;
                }
                // Hang detector, or the test is over: closing ends the run.
                Err(_) => return,
            }
        }
    } else {
        let _ = write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
    }
}

/// Points the process's registry, gateway and platform key at a
/// [`PacedPlatform`], and clears them again on drop (a failing test included),
/// so the other tests here keep their own configuration.
struct PlatformConfig;

impl PlatformConfig {
    fn point_at(platform: &PacedPlatform) -> Self {
        std::env::set_var("XYBRID_REGISTRY_URL", platform.url());
        set_api_key("paced-platform-key".into());
        xybrid_core::cloud::set_xybrid_gateway_url(Some(format!("{}/v1", platform.url())));
        Self
    }
}

impl Drop for PlatformConfig {
    fn drop(&mut self) {
        xybrid_core::cloud::set_xybrid_gateway_url(None);
        xybrid_core::cloud::set_xybrid_api_key(None);
        std::env::remove_var("XYBRID_REGISTRY_URL");
    }
}

/// Flushes until the span named `name` is exported. The run ends its span on
/// its worker just after the gateway sees the stream close, so this waits for
/// that worker; [`HANG`] bounds it.
fn exported_span(receiver: &Receiver, name: &str) -> Value {
    let deadline = std::time::Instant::now() + HANG;
    loop {
        observability_flush(HANG.as_millis() as u32).unwrap();
        if let Some(span) = receiver.spans().into_iter().find(|s| s["name"] == name) {
            return span;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{name} is never exported"
        );
        std::thread::yield_now();
    }
}

/// An abandoned stream is a cancellation, through the facade's own streaming
/// entry points: a caller that drops its streaming session before the terminal event (what
/// Bolt `stream_close` does) cancels the run. The run stops at the next token
/// and exports `error.type` `cancelled` with the two delivered chunks.
///
/// A speculative chat model streams from a loopback gateway the test paces,
/// so the drop lands between known tokens without a local model.
#[test]
fn dropping_a_token_stream_exports_a_cancelled_run_that_stopped_at_the_next_token() {
    let _exclusive = exclusive();
    let receiver = Receiver::start();
    let config = ObservabilityConfigHandle::xybrid("local-test-key".into());
    config.set_traces_endpoint(format!("{}/v1/traces", receiver.url()));
    config.set_json_protocol(true);
    config.init().unwrap();
    let platform = PacedPlatform::start();
    let cache = tempfile::tempdir().unwrap();
    init_sdk_cache_dir(cache.path().join("models").to_string_lossy().into_owned());
    let _platform_config = PlatformConfig::point_at(&platform);

    for (id, context) in [("paced-stream", false), ("paced-context-stream", true)] {
        let model = ModelLoader::from_registry_speculative(id.into())
            .load()
            .unwrap();
        assert!(model.is_cloud_serving(), "{id}: its download never lands");
        let input = Envelope::text("tell me a long story".into());
        let session = if context {
            model.run_stream_with_context(
                input,
                ConversationContextHandle::new(),
                RunOptions::default(),
                None,
            )
        } else {
            model.run_stream(input, RunOptions::default(), None)
        }
        .unwrap();
        // A token that never comes ends the gateway's stream at its hang
        // detector, which ends `next()` too.
        for (index, text) in ["t0 ", "t1 "].into_iter().enumerate() {
            platform.pace.send(Pace::Token(text)).unwrap();
            match session.next() {
                Some(StreamEvent::Token(token)) => {
                    assert_eq!((token.index, token.token.as_str()), (index as u64, text))
                }
                other => panic!("{id}: token {index}: {other:?}"),
            }
        }

        drop(session);
        // The run hands its next token to a session that is gone.
        platform.pace.send(Pace::Token("t2 ")).unwrap();
        platform.pace.send(Pace::ExpectClose).unwrap();
        // The gateway gives up waiting for the close after `HANG`.
        let closed = platform
            .closed
            .recv_timeout(2 * HANG)
            .expect("the gateway reports whether the stream closed");
        assert!(
            closed,
            "{id}: the run stops at that token and closes its stream"
        );

        let span = exported_span(&receiver, &format!("chat {id}"));
        assert_eq!(
            attr(&span, "error.type").map(|v| &v["stringValue"]),
            Some(&json!("cancelled")),
            "{id}"
        );
        assert_eq!(span["status"]["code"], 2, "{id}: {}", span["status"]);
        assert_eq!(
            attr(&span, "xybrid.genai.response.chunk_count").map(|v| &v["intValue"]),
            Some(&json!("2")),
            "{id}"
        );
    }

    observability_shutdown(HANG.as_millis() as u32).unwrap();
}

/// An OTLP collector that reads every request and never answers it, so each
/// export stays in flight until the client gives up. Reports the path of
/// every request it has read in full.
struct Parking {
    address: std::net::SocketAddr,
    requests: std::sync::mpsc::Receiver<String>,
    held: Arc<Mutex<Vec<TcpStream>>>,
    stopped: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Parking {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (report, requests) = std::sync::mpsc::channel();
        let held = Arc::new(Mutex::new(Vec::new()));
        let stopped = Arc::new(AtomicBool::new(false));
        let worker = {
            let (held, stopped) = (held.clone(), stopped.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stopped.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let (report, held) = (report.clone(), held.clone());
                    std::thread::spawn(move || {
                        if let Some(path) = read_request(&stream) {
                            // Parked: held open, never answered.
                            held.lock().unwrap().push(stream);
                            let _ = report.send(path);
                        }
                    });
                }
            })
        };
        Self {
            address,
            requests,
            held,
            stopped,
            worker: Some(worker),
        }
    }

    fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Returns once a request for each of `paths` has been read in full.
    fn wait_for(&self, paths: &[&str]) {
        let mut seen = std::collections::BTreeSet::new();
        while !paths.iter().all(|path| seen.contains(*path)) {
            let path = self
                .requests
                .recv_timeout(HANG)
                .unwrap_or_else(|_| panic!("only {seen:?} of {paths:?} were exported"));
            seen.insert(path);
        }
    }
}

impl Drop for Parking {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.address);
        self.worker.take().unwrap().join().unwrap();
        self.held.lock().unwrap().clear();
    }
}

/// Reads one HTTP request (head and body) from `stream`; its path, or `None`
/// when the connection closes first.
fn read_request(stream: &TcpStream) -> Option<String> {
    stream.set_read_timeout(Some(HANG)).ok()?;
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut first = String::new();
    if reader.read_line(&mut first).ok()? == 0 {
        return None;
    }
    let path = first.split_whitespace().nth(1)?.to_owned();
    let mut size = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        if line == "\r\n" {
            break;
        }
        if let Some((key, value)) = line.trim().split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                size = value.trim().parse().ok()?;
            }
        }
    }
    let mut body = vec![0; size];
    reader.read_exact(&mut body).ok()?;
    Some(path)
}

const SIGNAL_PATHS: [&str; 3] = ["/v1/traces", "/v1/logs", "/v1/metrics"];

/// Starts an OTLP/JSON session to `collector`, records one span, one log
/// record and one metric point, and returns once an export of each signal
/// is in flight (read by the collector, never answered).
fn exports_in_flight(collector: &Parking) -> std::thread::JoinHandle<Result<()>> {
    let config = ObservabilityConfigHandle::otlp(collector.url());
    config.set_json_protocol(true);
    config.init().unwrap();
    let workflow =
        WorkflowObservation::start("in-flight".into(), ObservationControls::default()).unwrap();
    let reference = workflow.observation_ref().unwrap();
    workflow.complete();
    drop(workflow);
    assert!(observability_record_feedback(&reference, Some(1.0), None, None).unwrap());
    // The flush sends every signal and then waits on the collector; the
    // shutdown below ends that wait.
    let flush = std::thread::spawn(|| observability_flush(HANG.as_millis() as u32));
    collector.wait_for(&SIGNAL_PATHS);
    flush
}

/// A binding caller reads the final
/// per-signal counters after `observability_shutdown`, including what the
/// shutdown cancelled in flight, until the next init.
#[test]
fn final_counters_stay_readable_after_a_shutdown_that_cancels_in_flight_exports() {
    let _exclusive = exclusive();
    let collector = Parking::start();
    let flush = exports_in_flight(&collector);
    let before = observability_stats();
    assert!(before.enabled);
    assert!(
        before.traces.pending > 0 && before.logs.pending > 0,
        "{before:?}"
    );

    assert!(
        observability_shutdown(200).is_err(),
        "the shutdown cancels the exports in flight at its deadline"
    );
    let last = observability_stats();
    assert!(!last.enabled, "export is off");
    for (signal, counters) in [
        ("traces", last.traces),
        ("logs", last.logs),
        ("metrics", last.metrics),
    ] {
        assert!(counters.accepted > 0, "{signal}: {counters:?}");
        assert_eq!(counters.pending, 0, "{signal}: settled by the shutdown");
        assert_eq!(counters.exported, 0, "{signal}: nothing was acknowledged");
        assert!(
            counters.failed + counters.dropped > 0,
            "{signal}: the losses are visible: {counters:?}"
        );
        assert_eq!(
            counters.accepted,
            counters.exported + counters.dropped + counters.failed,
            "{signal}: every accepted item is in one terminal bucket"
        );
    }
    assert!(
        flush.join().unwrap().is_err(),
        "the interrupted flush reports the loss"
    );
    // Repeating the shutdown (idempotent) and reading again change nothing.
    observability_shutdown(200).unwrap();
    assert_eq!(observability_stats(), last);

    // The next init starts the new session's counters.
    let receiver = Receiver::start();
    let config = ObservabilityConfigHandle::otlp(receiver.url());
    config.set_json_protocol(true);
    config.init().unwrap();
    let fresh = observability_stats();
    assert!(fresh.enabled);
    assert_eq!(
        (fresh.traces, fresh.logs),
        (SignalStats::default(), SignalStats::default())
    );
    observability_shutdown(HANG.as_millis() as u32).unwrap();
}

/// The re-init variant: an init that replaces a session with exports in
/// flight reports the new session's counters at once: final counters stay
/// readable only "until the next init". The replaced session retires (it drains once nothing
/// holds it); its counters are not carried over, so a caller that needs them
/// shuts down before the init. (Whether a replaced session's final counters
/// should be readable some other way is an open product question.)
#[test]
fn an_init_that_replaces_a_session_with_exports_in_flight_starts_fresh_counters() {
    let _exclusive = exclusive();
    let collector = Parking::start();
    let flush = exports_in_flight(&collector);

    let receiver = Receiver::start();
    let config = ObservabilityConfigHandle::otlp(receiver.url());
    config.set_json_protocol(true);
    config.init().unwrap();
    let fresh = observability_stats();
    assert!(fresh.enabled);
    assert_eq!(
        (fresh.traces, fresh.logs),
        (SignalStats::default(), SignalStats::default()),
        "the new session's counters, not the replaced one's"
    );
    // One span of the replacement's own, so its final counters cannot be
    // mistaken for none (which read as zeros) or for the replaced session's.
    WorkflowObservation::start("replacement".into(), ObservationControls::default())
        .unwrap()
        .complete();
    observability_shutdown(HANG.as_millis() as u32).unwrap();
    let last = observability_stats();
    assert!(!last.enabled);
    assert_eq!(
        (
            last.traces.accepted,
            last.traces.exported,
            last.traces.pending
        ),
        (1, 1, 0),
        "the replacement's final counters: {:?}",
        last.traces
    );
    assert_eq!(last.logs, SignalStats::default(), "it recorded no log");

    // Closing the parked connections fails the replaced session's exports,
    // which ends the flush that waits on them.
    drop(collector);
    assert!(flush.join().unwrap().is_err());
}

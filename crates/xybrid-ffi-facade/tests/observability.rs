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

#[test]
fn execution_refs_content_rules_and_last_owner_cancellation_reach_otlp() {
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
    assert!(exported.contains("facade-test-device"));
    drop(requests);
    observability_shutdown(5000).unwrap();
}

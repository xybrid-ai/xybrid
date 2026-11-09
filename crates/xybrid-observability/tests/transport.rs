//! Real HTTP wire, bounded lifecycle, and cancellation regressions for every signal.
#![cfg(feature = "otlp")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry::KeyValue;
use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use prost::Message;
use xybrid_observability::otlp::{
    BatchLimits, EnvironmentSource, ObservabilityPipeline, OtlpConfig, Protocol, RetryLimits,
    SignalEndpoint,
};
use xybrid_observability::{
    ContentCapture, ContentPolicy, InferenceRequest, Observer, StartOptions, ToolCallInfo, Usage,
};

#[derive(Clone, Debug)]
struct Request {
    path: String,
    body: Vec<u8>,
}
#[derive(Clone, Copy)]
enum Reply {
    Success,
    SlowSuccess,
    Partial,
    Retry,
    Hang,
    Lost,
}
struct Server {
    url: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn start(reply: Reply) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (captured, done) = (requests.clone(), stop.clone());
        let handle = thread::spawn(move || {
            let mut handlers = Vec::new();
            while !done.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((socket, _)) => {
                        let (captured, done) = (captured.clone(), done.clone());
                        handlers.push(thread::spawn(move || handle(socket, reply, captured, done)));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            }
            for handler in handlers {
                handler.join().unwrap();
            }
        });
        Self {
            url,
            requests,
            stop,
            thread: Some(handle),
        }
    }
    fn wait(&self, n: usize) {
        let until = Instant::now() + Duration::from_secs(3);
        while self.requests.lock().unwrap().len() < n {
            assert!(Instant::now() < until, "request never arrived");
            thread::sleep(Duration::from_millis(2));
        }
    }
    fn config(&self, protocol: Protocol) -> OtlpConfig {
        OtlpConfig::collector(&self.url)
            .with_environment(EnvironmentSource::Disabled)
            .with_traces(Some(
                SignalEndpoint::new(format!("{}/v1/traces", self.url)).with_protocol(protocol),
            ))
            .with_logs(Some(
                SignalEndpoint::new(format!("{}/v1/logs", self.url)).with_protocol(protocol),
            ))
            .with_metrics(None)
            .with_batch(BatchLimits {
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            })
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
    }
}
fn handle(
    mut socket: TcpStream,
    reply: Reply,
    captured: Arc<Mutex<Vec<Request>>>,
    done: Arc<AtomicBool>,
) {
    socket
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let mut bytes = Vec::new();
    let request = loop {
        if done.load(Ordering::Relaxed) {
            return;
        }
        if let Some(end) = bytes.windows(4).position(|p| p == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let length = headers
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            if bytes.len() >= end + 4 + length {
                break Request {
                    path: headers.split_whitespace().nth(1).unwrap().into(),
                    body: bytes[end + 4..end + 4 + length].to_vec(),
                };
            }
        }
        let mut buf = [0; 8192];
        match socket.read(&mut buf) {
            Ok(0) => return,
            Ok(n) => bytes.extend_from_slice(&buf[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return,
        }
    };
    captured.lock().unwrap().push(request);
    let (status, extra, body) = match reply {
        Reply::Success => (200, "", ""),
        Reply::SlowSuccess => {
            thread::sleep(Duration::from_millis(80));
            (200, "", "")
        }
        Reply::Partial => (200, "", r#"{"partialSuccess":{"rejectedSpans":"1"}}"#),
        Reply::Retry => (503, "Retry-After: 2\r\n", ""),
        Reply::Lost => return,
        Reply::Hang => {
            while !done.load(Ordering::Relaxed) {
                let mut buf = [0; 1];
                if matches!(socket.read(&mut buf), Ok(0)) {
                    return;
                }
            }
            return;
        }
    };
    let _ = write!(
        socket,
        "HTTP/1.1 {status} Reply\r\nConnection: close\r\n{extra}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
}
fn emit(observer: &Observer) {
    let op = observer.start_inference(
        InferenceRequest::new("chat", "fake"),
        StartOptions::default(),
    );
    op.set_attribute(KeyValue::new("gen_ai.request.temperature", 1.0));
    op.record_usage(&Usage::tokens(Some(2), Some(3)));
    op.end();
}

#[test]
fn wire_types_and_structured_nulls_survive_both_encodings() {
    for protocol in [Protocol::HttpProtobuf, Protocol::HttpJson] {
        let server = Server::start(Reply::Success);
        let pipeline = ObservabilityPipeline::build(server.config(protocol)).unwrap();
        let observer = pipeline
            .observer_with(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent));
        emit(&observer);
        let tool = observer.start_tool(ToolCallInfo::new("lookup"), StartOptions::default());
        tool.record_result(&serde_json::json!([null, ""]));
        tool.end();
        pipeline.shutdown(Duration::from_secs(2)).unwrap();
        let requests = server.requests.lock().unwrap();
        let traces = requests.iter().find(|r| r.path == "/v1/traces").unwrap();
        let logs = requests.iter().find(|r| r.path == "/v1/logs").unwrap();
        let traces = match protocol {
            Protocol::HttpProtobuf => {
                ExportTraceServiceRequest::decode(traces.body.as_slice()).unwrap()
            }
            _ => serde_json::from_slice(&traces.body).unwrap(),
        };
        let logs = match protocol {
            Protocol::HttpProtobuf => {
                ExportLogsServiceRequest::decode(logs.body.as_slice()).unwrap()
            }
            _ => serde_json::from_slice(&logs.body).unwrap(),
        };
        let spans = &traces.resource_spans[0].scope_spans[0].spans;
        let temp = spans
            .iter()
            .flat_map(|s| &s.attributes)
            .find(|a| a.key == "gen_ai.request.temperature")
            .unwrap();
        assert_eq!(
            temp.value.as_ref().unwrap().value,
            Some(opentelemetry_proto::tonic::common::v1::any_value::Value::DoubleValue(1.0))
        );
        let result = logs
            .resource_logs
            .iter()
            .flat_map(|r| &r.scope_logs)
            .flat_map(|s| &s.log_records)
            .flat_map(|l| &l.attributes)
            .find(|a| a.key == "gen_ai.tool.call.result")
            .unwrap();
        let opentelemetry_proto::tonic::common::v1::any_value::Value::ArrayValue(array) =
            result.value.as_ref().unwrap().value.as_ref().unwrap()
        else {
            panic!("expected array")
        };
        assert_eq!(array.values[0].value, None);
        assert_eq!(
            array.values[1].value,
            Some(
                opentelemetry_proto::tonic::common::v1::any_value::Value::StringValue(String::new())
            )
        );
    }
}

#[test]
fn shutdown_cancels_all_signals_and_remembers_its_result() {
    for signal in ["traces", "logs", "metrics"] {
        let server = Server::start(Reply::Hang);
        let config = OtlpConfig::default()
            .with_traces(
                (signal == "traces")
                    .then(|| SignalEndpoint::new(format!("{}/v1/traces", server.url))),
            )
            .with_logs(
                (signal == "logs").then(|| SignalEndpoint::new(format!("{}/v1/logs", server.url))),
            )
            .with_metrics(
                (signal == "metrics")
                    .then(|| SignalEndpoint::new(format!("{}/v1/metrics", server.url))),
            )
            .with_batch(BatchLimits {
                max_export_batch_size: 1,
                ..BatchLimits::default()
            });
        let pipeline = Arc::new(ObservabilityPipeline::build(config).unwrap());
        emit(
            &pipeline
                .observer_with(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent)),
        );
        let flush_pipeline = pipeline.clone();
        let flush = thread::spawn(move || flush_pipeline.force_flush(Duration::from_secs(3)));
        server.wait(1);
        let started = Instant::now();
        let first = pipeline.shutdown(Duration::from_millis(50));
        // R-CANCEL-1: the in-flight request was aborted at the deadline.
        assert!(
            first
                .as_ref()
                .is_err_and(xybrid_observability::Error::is_timeout),
            "{signal}: {first:?}"
        );
        let first = first.map_err(|e| e.to_string());
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(flush.join().unwrap().is_err());
        assert_eq!(
            pipeline
                .shutdown(Duration::from_secs(1))
                .map_err(|e| e.to_string()),
            first
        );
        assert_eq!(
            pipeline
                .force_flush(Duration::from_secs(1))
                .map_err(|e| e.to_string()),
            first
        );
        let stats = pipeline.stats();
        let stats = match signal {
            "traces" => stats.traces,
            "logs" => stats.logs,
            _ => stats.metrics,
        };
        assert!(stats.is_consistent());
        assert_eq!(stats.pending, 0);
        assert!(stats.failed > 0, "{signal}: {stats:?}");
    }
}

#[test]
fn shutdown_during_retry_after_stops_future_requests() {
    let server = Server::start(Reply::Retry);
    let pipeline = Arc::new(
        ObservabilityPipeline::build(server.config(Protocol::HttpJson).with_logs(None)).unwrap(),
    );
    emit(&pipeline.observer());
    let copy = pipeline.clone();
    let flush = thread::spawn(move || copy.force_flush(Duration::from_secs(3)));
    server.wait(1);
    let result = pipeline.shutdown(Duration::from_millis(30));
    // R-CANCEL-2: the backoff was cancelled at the deadline.
    assert!(
        result
            .as_ref()
            .is_err_and(xybrid_observability::Error::is_timeout),
        "{result:?}"
    );
    assert!(flush.join().unwrap().is_err());
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    assert_eq!(pipeline.stats().traces.failed, 1);
}

#[test]
fn partial_acknowledgements_and_lost_acks_count_items_once() {
    for reply in [Reply::Partial, Reply::Lost] {
        let server = Server::start(reply);
        let config = server
            .config(Protocol::HttpJson)
            .with_logs(None)
            .with_retry(RetryLimits {
                initial_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(1),
                ..RetryLimits::default()
            });
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        emit(&pipeline.observer());
        emit(&pipeline.observer());
        assert!(pipeline.force_flush(Duration::from_secs(2)).is_err());
        let requests = server.requests.lock().unwrap();
        let stats = pipeline.stats().traces;
        assert_eq!(stats.accepted, 2);
        assert!(stats.is_consistent());
        assert_eq!(stats.pending, 0);
        if matches!(reply, Reply::Partial) {
            assert_eq!(requests.len(), 1);
            assert_eq!(stats.exported, 1);
            assert_eq!(stats.failed, 1);
        } else {
            assert_eq!(requests.len(), 3);
            assert_eq!(requests[0].body, requests[2].body);
            assert_eq!(stats.failed, 2);
        }
    }
}

#[test]
fn dropping_an_export_future_still_settles_admission_and_releases_bytes() {
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanExporter};
    let memory = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(memory.clone())
        .build();
    let observer = Observer::builder().tracer_provider(&provider).build();
    emit(&observer);
    let spans = memory.get_finished_spans().unwrap();
    let server = Server::start(Reply::SlowSuccess);
    let mut exporters =
        xybrid_observability::otlp::exporters(&server.config(Protocol::HttpJson).with_logs(None))
            .unwrap();
    let exporter = exporters.traces.take().unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        assert!(
            tokio::time::timeout(Duration::from_millis(15), exporter.export(spans.clone()))
                .await
                .is_err()
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        while exporter.stats().pending > 0 {
            assert!(
                Instant::now() < deadline,
                "dropped future stranded an export"
            );
            tokio::task::yield_now().await;
        }
        assert_eq!(exporter.stats().exported, 1);
        exporter.export(spans).await.unwrap();
    });
    assert_eq!(exporter.stats().exported, 2);
    assert!(exporter.stats().is_consistent());
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();
}

#[test]
fn encoded_size_splitting_preserves_every_span_and_drops_only_singletons() {
    for protocol in [Protocol::HttpJson, Protocol::HttpProtobuf] {
        let server = Server::start(Reply::Success);
        let mut config = server.config(protocol).with_logs(None);
        config.max_request_body_bytes = 1800;
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        let observer = pipeline.observer();
        for i in 0..9 {
            let span = observer.start_span(
                format!("span-{i}"),
                vec![KeyValue::new("payload", "x".repeat(300))],
                StartOptions::default(),
            );
            span.end();
        }
        let oversized = observer.start_span(
            "oversized",
            vec![KeyValue::new("payload", "x".repeat(5000))],
            StartOptions::default(),
        );
        oversized.end();
        assert!(pipeline.force_flush(Duration::from_secs(2)).is_err());
        let requests = server.requests.lock().unwrap();
        assert!(requests.len() > 1);
        let mut names = Vec::new();
        for request in requests.iter() {
            assert!(request.body.len() <= 1800);
            let decoded: ExportTraceServiceRequest = match protocol {
                Protocol::HttpProtobuf => {
                    ExportTraceServiceRequest::decode(request.body.as_slice()).unwrap()
                }
                Protocol::HttpJson => serde_json::from_slice(&request.body).unwrap(),
            };
            names.extend(
                decoded
                    .resource_spans
                    .into_iter()
                    .flat_map(|r| r.scope_spans)
                    .flat_map(|s| s.spans)
                    .map(|s| s.name),
            );
        }
        assert_eq!(
            names,
            (0..9).map(|i| format!("span-{i}")).collect::<Vec<_>>()
        );
        let stats = pipeline.stats().traces;
        assert_eq!(stats.exported, 9);
        assert_eq!(stats.dropped, 1);
        assert!(stats.is_consistent());
    }
}

#[test]
fn concurrent_async_lifecycle_callers_have_independent_wait_bounds() {
    let server = Server::start(Reply::Hang);
    let pipeline = Arc::new(
        ObservabilityPipeline::build(server.config(Protocol::HttpJson).with_logs(None)).unwrap(),
    );
    emit(&pipeline.observer());
    let copy = pipeline.clone();
    let shutdown = thread::spawn(move || copy.shutdown(Duration::from_millis(400)));
    server.wait(1);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let started = Instant::now();
        assert!(pipeline
            .shutdown_async(Duration::from_millis(20))
            .await
            .is_err());
        assert!(started.elapsed() < Duration::from_millis(200));
        let started = Instant::now();
        assert!(pipeline
            .force_flush_async(Duration::from_millis(20))
            .await
            .is_err());
        assert!(started.elapsed() < Duration::from_millis(200));
    });
    assert!(shutdown.join().unwrap().is_err());
    assert_eq!(pipeline.stats().traces.pending, 0);
}

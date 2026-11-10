//! A scriptable OTLP/HTTP endpoint for the delivery, wire and lifecycle
//! regressions: one thread per connection, one request per connection
//! (`Connection: close`), every request and every client disconnect recorded.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use prost::Message;
use xybrid_observability::otlp::{
    BatchLimits, EnvironmentSource, OtlpConfig, Protocol, SignalEndpoint,
};

/// One request as the server read it.
#[derive(Clone, Debug)]
pub struct Received {
    /// Arrival order across all connections, from 0.
    pub index: usize,
    pub path: String,
    /// Lower-case names, in arrival order.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub at: Instant,
}

impl Received {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn is_json(&self) -> bool {
        self.header("content-type") == Some("application/json")
    }

    pub fn traces(&self) -> ExportTraceServiceRequest {
        if self.is_json() {
            serde_json::from_slice(&self.body).expect("OTLP/JSON traces")
        } else {
            ExportTraceServiceRequest::decode(self.body.as_slice()).expect("OTLP traces")
        }
    }

    pub fn logs(&self) -> ExportLogsServiceRequest {
        if self.is_json() {
            serde_json::from_slice(&self.body).expect("OTLP/JSON logs")
        } else {
            ExportLogsServiceRequest::decode(self.body.as_slice()).expect("OTLP logs")
        }
    }

    pub fn metrics(&self) -> ExportMetricsServiceRequest {
        if self.is_json() {
            serde_json::from_slice(&self.body).expect("OTLP/JSON metrics")
        } else {
            ExportMetricsServiceRequest::decode(self.body.as_slice()).expect("OTLP metrics")
        }
    }

    /// Span names in body order.
    pub fn span_names(&self) -> Vec<String> {
        self.traces()
            .resource_spans
            .into_iter()
            .flat_map(|r| r.scope_spans)
            .flat_map(|s| s.spans)
            .map(|s| s.name)
            .collect()
    }

    /// Log records in the body.
    pub fn log_record_count(&self) -> u64 {
        self.logs()
            .resource_logs
            .into_iter()
            .flat_map(|r| r.scope_logs)
            .map(|s| s.log_records.len() as u64)
            .sum()
    }

    /// Metric data points in the body (the unit the metrics counters count).
    pub fn metric_point_count(&self) -> u64 {
        use opentelemetry_proto::tonic::metrics::v1::metric::Data;
        self.metrics()
            .resource_metrics
            .into_iter()
            .flat_map(|r| r.scope_metrics)
            .flat_map(|s| s.metrics)
            .filter_map(|m| m.data)
            .map(|data| match data {
                Data::Gauge(g) => g.data_points.len(),
                Data::Sum(s) => s.data_points.len(),
                Data::Histogram(h) => h.data_points.len(),
                Data::ExponentialHistogram(h) => h.data_points.len(),
                Data::Summary(s) => s.data_points.len(),
            } as u64)
            .sum()
    }

    /// The items in the body: spans, log records or metric data points, by
    /// the request's path.
    pub fn item_count(&self) -> u64 {
        match self.path.as_str() {
            "/v1/traces" => self.span_names().len() as u64,
            "/v1/logs" => self.log_record_count(),
            "/v1/metrics" => self.metric_point_count(),
            other => panic!("not an OTLP signal path: {other}"),
        }
    }
}

/// What the server does with one request.
#[derive(Clone, Debug)]
pub enum Action {
    /// Answers after `delay`; a client that disconnects meanwhile is recorded.
    Respond {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        delay: Duration,
    },
    /// Announces `total` body bytes, then writes `chunk` bytes every `every`.
    Trickle {
        status: u16,
        total: usize,
        chunk: usize,
        every: Duration,
    },
    /// Never answers; waits for the client to close the connection.
    Hang,
    /// Reads the request, then closes without answering (lost acknowledgement).
    Lost,
}

impl Action {
    pub fn status(status: u16) -> Self {
        Self::Respond {
            status,
            headers: Vec::new(),
            body: Vec::new(),
            delay: Duration::ZERO,
        }
    }

    pub fn ok() -> Self {
        Self::status(200)
    }

    pub fn after(mut self, wait: Duration) -> Self {
        if let Self::Respond { delay, .. } = &mut self {
            *delay = wait;
        }
        self
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        if let Self::Respond { headers, .. } = &mut self {
            headers.push((name.into(), value.into()));
        }
        self
    }

    pub fn body(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        if let Self::Respond { body, .. } = &mut self {
            *body = bytes.into();
        }
        self
    }
}

/// Something the server observed on a connection after reading its request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// The client closed the connection before the response was complete.
    Disconnected { index: usize, at: Instant },
    /// The full response was written.
    Responded { index: usize, at: Instant },
}

type Script = dyn Fn(&Received) -> Action + Send + Sync;

pub struct Server {
    pub url: String,
    received: Arc<Mutex<Vec<Received>>>,
    events: Arc<Mutex<Vec<Event>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Server {
    /// Starts a server that answers each request as `script` decides.
    pub fn start(script: impl Fn(&Received) -> Action + Send + Sync + 'static) -> Self {
        Self::start_on(TcpListener::bind("127.0.0.1:0").unwrap(), script)
    }

    /// [`Server::start`] on a listener the test bound itself (an endpoint
    /// that refused connections and then comes up on the same port).
    pub fn start_on(
        listener: TcpListener,
        script: impl Fn(&Received) -> Action + Send + Sync + 'static,
    ) -> Self {
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        let events = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let script: Arc<Script> = Arc::new(script);
        let counter = Arc::new(AtomicUsize::new(0));
        let shared = (received.clone(), events.clone(), stop.clone());
        let thread = thread::spawn(move || {
            let (received, events, stop) = shared;
            let mut handlers = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((socket, _)) => {
                        let conn = Connection {
                            received: received.clone(),
                            events: events.clone(),
                            stop: stop.clone(),
                            script: script.clone(),
                            counter: counter.clone(),
                        };
                        handlers.push(thread::spawn(move || conn.serve(socket)));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            }
            for handler in handlers {
                let _ = handler.join();
            }
        });
        Self {
            url,
            received,
            events,
            stop,
            thread: Some(thread),
        }
    }

    /// Answers every request with `action`.
    pub fn always(action: Action) -> Self {
        Self::start(move |_| action.clone())
    }

    /// Answers every request with 200.
    pub fn ok() -> Self {
        Self::always(Action::ok())
    }

    /// Answers request `i` with `actions[i]` (the last one repeats).
    pub fn sequence(actions: Vec<Action>) -> Self {
        Self::start(move |r| actions[r.index.min(actions.len() - 1)].clone())
    }

    pub fn endpoint(&self, signal: &str, protocol: Protocol) -> SignalEndpoint {
        SignalEndpoint::new(format!("{}/v1/{signal}", self.url)).with_protocol(protocol)
    }

    /// Traces only, no environment, a long scheduled delay (exports happen on
    /// flush, shutdown or a full batch).
    pub fn traces_config(&self, protocol: Protocol) -> OtlpConfig {
        OtlpConfig::default()
            .with_environment(EnvironmentSource::Disabled)
            .with_traces(Some(self.endpoint("traces", protocol)))
            .with_batch(BatchLimits {
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            })
            .with_metric_export_interval(Duration::from_secs(600))
    }

    /// All three signals (see [`Server::traces_config`]).
    pub fn all_config(&self, protocol: Protocol) -> OtlpConfig {
        self.traces_config(protocol)
            .with_logs(Some(self.endpoint("logs", protocol)))
            .with_metrics(Some(self.endpoint("metrics", protocol)))
    }

    pub fn requests(&self) -> Vec<Received> {
        self.received.lock().unwrap().clone()
    }

    pub fn requests_to(&self, path: &str) -> Vec<Received> {
        self.requests()
            .into_iter()
            .filter(|r| r.path == path)
            .collect()
    }

    pub fn count(&self) -> usize {
        self.received.lock().unwrap().len()
    }

    /// Waits until `n` requests arrived.
    pub fn wait_for(&self, n: usize, within: Duration) {
        let until = Instant::now() + within;
        while self.count() < n {
            assert!(
                Instant::now() < until,
                "expected {n} requests, got {}",
                self.count()
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Waits until `n` requests to `path` arrived.
    pub fn wait_for_path(&self, path: &str, n: usize, within: Duration) {
        let until = Instant::now() + within;
        while self.requests_to(path).len() < n {
            assert!(
                Instant::now() < until,
                "expected {n} requests to {path}, got {}",
                self.requests_to(path).len()
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    /// When the client dropped the connection of request `index`.
    pub fn disconnect_of(&self, index: usize) -> Option<Instant> {
        self.events().into_iter().find_map(|e| match e {
            Event::Disconnected { index: i, at } if i == index => Some(at),
            _ => None,
        })
    }

    /// When the response of request `index` was fully written.
    pub fn response_of(&self, index: usize) -> Option<Instant> {
        self.events().into_iter().find_map(|e| match e {
            Event::Responded { index: i, at } if i == index => Some(at),
            _ => None,
        })
    }

    /// Waits for the disconnect of request `index`.
    pub fn wait_disconnect(&self, index: usize, within: Duration) -> Instant {
        let until = Instant::now() + within;
        loop {
            if let Some(at) = self.disconnect_of(index) {
                return at;
            }
            assert!(
                Instant::now() < until,
                "the client never closed request {index}"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Connection {
    received: Arc<Mutex<Vec<Received>>>,
    events: Arc<Mutex<Vec<Event>>>,
    stop: Arc<AtomicBool>,
    script: Arc<Script>,
    counter: Arc<AtomicUsize>,
}

enum Read1 {
    Data,
    Closed,
    Idle,
}

impl Connection {
    fn read_some(socket: &mut TcpStream, buffer: &mut Vec<u8>) -> Read1 {
        let mut chunk = [0; 16 * 1024];
        match socket.read(&mut chunk) {
            Ok(0) => Read1::Closed,
            Ok(n) => {
                buffer.extend_from_slice(&chunk[..n]);
                Read1::Data
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                Read1::Idle
            }
            Err(_) => Read1::Closed,
        }
    }

    fn serve(self, mut socket: TcpStream) {
        // Accepted sockets inherit O_NONBLOCK on macOS/BSD (not on Linux), which
        // would make the read timeout below a busy WouldBlock instead of a wait.
        socket.set_nonblocking(false).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_millis(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let (path, headers, body) = loop {
            if self.stop.load(Ordering::Relaxed) {
                return;
            }
            if let Some(end) = bytes.windows(4).position(|p| p == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&bytes[..end]).into_owned();
                let mut lines = head.split("\r\n");
                let path = lines
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or_default()
                    .to_owned();
                let headers: Vec<(String, String)> = lines
                    .filter_map(|l| l.split_once(':'))
                    .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
                    .collect();
                let length = headers
                    .iter()
                    .find(|(k, _)| k == "content-length")
                    .and_then(|(_, v)| v.parse::<usize>().ok())
                    .unwrap_or(0);
                if bytes.len() >= end + 4 + length {
                    break (path, headers, bytes[end + 4..end + 4 + length].to_vec());
                }
            }
            if let Read1::Closed = Self::read_some(&mut socket, &mut bytes) {
                return;
            }
        };
        let request = Received {
            index: self.counter.fetch_add(1, Ordering::SeqCst),
            path,
            headers,
            body,
            at: Instant::now(),
        };
        let index = request.index;
        let action = (self.script)(&request);
        self.received.lock().unwrap().push(request);
        let event = |e: Event| self.events.lock().unwrap().push(e);
        // Waits up to `wait` while watching for the client to go away.
        let watch = |socket: &mut TcpStream, wait: Duration| -> bool {
            let until = Instant::now() + wait;
            let mut sink = Vec::new();
            while Instant::now() < until {
                if self.stop.load(Ordering::Relaxed) {
                    return false;
                }
                if let Read1::Closed = Self::read_some(socket, &mut sink) {
                    event(Event::Disconnected {
                        index,
                        at: Instant::now(),
                    });
                    return false;
                }
            }
            true
        };
        match action {
            Action::Lost => {}
            Action::Hang => {
                watch(&mut socket, Duration::from_secs(3600));
            }
            Action::Respond {
                status,
                headers,
                body,
                delay,
            } => {
                if !watch(&mut socket, delay) {
                    return;
                }
                let mut head = format!(
                    "HTTP/1.1 {status} Test\r\nConnection: close\r\nContent-Length: {}\r\n",
                    body.len()
                );
                for (k, v) in headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                let written = socket
                    .write_all(head.as_bytes())
                    .and_then(|()| socket.write_all(&body))
                    .and_then(|()| socket.flush());
                event(match written {
                    Ok(()) => Event::Responded {
                        index,
                        at: Instant::now(),
                    },
                    Err(_) => Event::Disconnected {
                        index,
                        at: Instant::now(),
                    },
                });
            }
            Action::Trickle {
                status,
                total,
                chunk,
                every,
            } => {
                let head = format!(
                    "HTTP/1.1 {status} Test\r\nConnection: close\r\nContent-Length: {total}\r\n\r\n"
                );
                if socket.write_all(head.as_bytes()).is_err() {
                    event(Event::Disconnected {
                        index,
                        at: Instant::now(),
                    });
                    return;
                }
                let mut sent = 0;
                while sent < total {
                    if !watch(&mut socket, every) {
                        return;
                    }
                    let n = chunk.min(total - sent);
                    if socket.write_all(&vec![b' '; n]).is_err() {
                        event(Event::Disconnected {
                            index,
                            at: Instant::now(),
                        });
                        return;
                    }
                    sent += n;
                }
                event(Event::Responded {
                    index,
                    at: Instant::now(),
                });
            }
        }
    }
}

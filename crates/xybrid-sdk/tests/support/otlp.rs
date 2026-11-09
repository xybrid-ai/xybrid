//! A local OTLP/HTTP JSON collector shared by SDK observability tests.
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;
use xybrid_sdk::observability::{BatchLimits, OtlpConfig, Protocol, SignalEndpoint};

pub fn start_mock_ingest() -> (String, mpsc::Receiver<String>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock ingest should bind");
    let addr = listener.local_addr().expect("mock ingest should have addr");
    let (tx, rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("mock ingest should accept");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];
        let mut header_end = None;
        let mut content_length = 0_usize;

        loop {
            let read = stream.read(&mut chunk).expect("mock ingest should read");
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);

            if header_end.is_none() {
                header_end = find_header_end(&buffer);
                if let Some(end) = header_end {
                    content_length = parse_content_length(&buffer[..end]);
                }
            }

            if let Some(end) = header_end {
                let body_start = end + 4;
                if buffer.len() >= body_start + content_length {
                    break;
                }
            }
        }

        let body = header_end
            .map(|end| {
                let body_start = end + 4;
                let body_end = (body_start + content_length).min(buffer.len());
                String::from_utf8_lossy(&buffer[body_start..body_end]).to_string()
            })
            .unwrap_or_default();
        tx.send(body).expect("mock ingest should send body");

        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
        stream
            .write_all(response)
            .expect("mock ingest should respond");
    });

    (format!("http://{addr}"), rx, handle)
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(headers: &[u8]) -> usize {
    String::from_utf8_lossy(headers)
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
}

pub fn trace_config(endpoint: &str) -> OtlpConfig {
    OtlpConfig::default()
        .with_traces(Some(
            SignalEndpoint::new(format!("{endpoint}/v1/traces"))
                .with_protocol(Protocol::HttpJson)
                .with_timeout(Duration::from_secs(5)),
        ))
        .with_batch(BatchLimits {
            scheduled_delay: Duration::from_secs(60),
            ..BatchLimits::default()
        })
}

pub fn spans(body: &serde_json::Value) -> Vec<&serde_json::Value> {
    body["resourceSpans"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r["scopeSpans"].as_array().unwrap())
        .flat_map(|s| s["spans"].as_array().unwrap())
        .collect()
}

pub fn attribute<'a>(
    attributes: &'a serde_json::Value,
    key: &str,
) -> Option<&'a serde_json::Value> {
    attributes
        .as_array()?
        .iter()
        .find(|attribute| attribute["key"] == key)
        .map(|attribute| &attribute["value"])
}

//! Model-free end-to-end: `xybrid run` with a native-direct Anthropic stage.
//!
//! A loopback Anthropic surface records every request. The child environment
//! carries a dummy `ANTHROPIC_API_KEY` and a dummy `XYBRID_API_KEY`, so the
//! tests prove that neither ambient credential follows a custom `gateway_url`:
//!
//! - no explicit `api_key` + custom destination → configuration error, zero
//!   requests;
//! - explicit `api_key` → exactly one `/messages` request carrying that key
//!   (and only that key), answered from a fake Anthropic response.
//!
//! Both cases are cloud-only stages, so this needs no model and runs in
//! ordinary CI.
#![cfg(unix)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const DUMMY_ANTHROPIC_KEY: &str = "dummy-anthropic-key";
const DUMMY_PLATFORM_KEY: &str = "dummy-platform-key";
const FAKE_REPLY: &str = "FAKE_ANTHROPIC_REPLY";
const CHILD_TIMEOUT: Duration = Duration::from_secs(120);

fn xybrid_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_xybrid"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Loopback Anthropic-compatible endpoint that records every request verbatim.
struct FakeAnthropic {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FakeAnthropic {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let requests = requests.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => Self::serve(stream, &requests),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            })
        };
        Self {
            url: format!("http://{addr}"),
            requests,
            stop,
            thread: Some(thread),
        }
    }

    fn serve(mut stream: TcpStream, requests: &Arc<Mutex<Vec<String>>>) {
        stream.set_nonblocking(false).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let read = match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            request.extend_from_slice(&buf[..read]);
            if let Some(header_end) = request
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|pos| pos + 4)
            {
                let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                while request.len() < header_end + content_length {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }
                break;
            }
        }
        requests
            .lock()
            .unwrap()
            .push(String::from_utf8_lossy(&request).into_owned());

        let body = format!(
            r#"{{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[{{"type":"text","text":"{FAKE_REPLY}"}}],"stop_reason":"end_turn","usage":{{"input_tokens":3,"output_tokens":2}}}}"#
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for FakeAnthropic {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn run_cli(home: &Path, registry_url: &str, args: &[&str]) -> Run {
    let mut child = Command::new(xybrid_bin())
        .current_dir(workspace_root())
        .env("HOME", home)
        // Dummy ambient credentials the native client must not leak.
        .env("ANTHROPIC_API_KEY", DUMMY_ANTHROPIC_KEY)
        .env("XYBRID_API_KEY", DUMMY_PLATFORM_KEY)
        // Point the registry at the same loopback recorder so an unexpected
        // registry call would show up as a captured request too.
        .env("XYBRID_REGISTRY_URL", registry_url)
        .env("NO_COLOR", "1")
        .env("XYBRID_TELEMETRY_OPTOUT", "1")
        .env_remove("XYBRID_GATEWAY_URL")
        .env_remove("XYBRID_PLATFORM_URL")
        .env_remove("HTTP_PROXY")
        .env_remove("HTTPS_PROXY")
        .env_remove("http_proxy")
        .env_remove("https_proxy")
        .env_remove("ALL_PROXY")
        .arg("run")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn xybrid");

    let mut stdout_pipe = child.stdout.take().unwrap();
    let mut stderr_pipe = child.stderr.take().unwrap();
    let stdout_reader = thread::spawn(move || {
        let mut out = String::new();
        let _ = stdout_pipe.read_to_string(&mut out);
        out
    });
    let stderr_reader = thread::spawn(move || {
        let mut out = String::new();
        let _ = stderr_pipe.read_to_string(&mut out);
        out
    });

    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            break status;
        }
        if started.elapsed() > CHILD_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait().expect("reap killed child");
            let stdout = stdout_reader.join().unwrap_or_default();
            let stderr = stderr_reader.join().unwrap_or_default();
            panic!("xybrid run exceeded {CHILD_TIMEOUT:?}\nstdout:\n{stdout}\nstderr:\n{stderr}");
        }
        thread::sleep(Duration::from_millis(50));
    };

    Run {
        status,
        stdout: stdout_reader.join().unwrap_or_default(),
        stderr: stderr_reader.join().unwrap_or_default(),
    }
}

fn write_pipeline(dir: &Path, fake: &FakeAnthropic, explicit_api_key: Option<&str>) -> PathBuf {
    let api_key_line = explicit_api_key
        .map(|key| format!("    api_key: \"{key}\"\n"))
        .unwrap_or_default();
    let yaml = format!(
        r#"name: anthropic-direct-credentials
stages:
  - id: llm
    model: claude-3-5-sonnet-20241022
    target: cloud
    provider: anthropic
    backend: direct
    gateway_url: "{url}"
    max_tokens: 16
    temperature: 0
{api_key_line}"#,
        url = fake.url,
    );
    let path = dir.join("anthropic-direct.yaml");
    std::fs::write(&path, yaml).unwrap();
    path
}

#[test]
fn custom_origin_without_explicit_key_fails_before_any_request() {
    let temp = tempfile::tempdir().unwrap();
    let fake = FakeAnthropic::start();
    let pipeline = write_pipeline(temp.path(), &fake, None);

    let run = run_cli(
        temp.path(),
        &fake.url,
        &["-c", pipeline.to_str().unwrap(), "--input-text", "hello"],
    );

    assert!(
        !run.status.success(),
        "a custom destination without an explicit key must fail\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("explicit 'api_key'") || combined.contains("Configuration error"),
        "error must identify the missing explicit key:\n{combined}"
    );
    assert!(
        fake.requests().is_empty(),
        "no credential may reach the endpoint: {:?}",
        fake.requests()
    );
}

#[test]
fn explicit_key_reaches_only_the_configured_custom_origin() {
    let temp = tempfile::tempdir().unwrap();
    let fake = FakeAnthropic::start();
    let pipeline = write_pipeline(temp.path(), &fake, Some(DUMMY_ANTHROPIC_KEY));

    let run = run_cli(
        temp.path(),
        &fake.url,
        &["-c", pipeline.to_str().unwrap(), "--input-text", "hello"],
    );

    assert!(
        run.status.success(),
        "explicit-key request must succeed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stdout.contains(FAKE_REPLY),
        "fake Anthropic reply must surface:\n{}",
        run.stdout
    );

    let requests = fake.requests();
    assert_eq!(
        requests.len(),
        1,
        "exactly one request (no registry call): {requests:?}"
    );
    let request = &requests[0];
    assert!(request.starts_with("POST /messages "), "{request}");
    let lower = request.to_ascii_lowercase();
    assert!(
        lower.contains(&format!("x-api-key: {DUMMY_ANTHROPIC_KEY}")),
        "explicit key must be the one presented: {request}"
    );
    assert!(
        !request.contains(DUMMY_PLATFORM_KEY),
        "platform key must never reach a provider: {request}"
    );
}

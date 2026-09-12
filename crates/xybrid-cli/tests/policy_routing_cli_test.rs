//! End-to-end: `xybrid run --policy` routes one hybrid LLM stage between a
//! real local GGUF model and a (fake) DeepSeek endpoint.
//!
//! The local leg is the real `functiongemma-270m-it` fixture (override with
//! `XYBRID_POLICY_E2E_MODEL_ID`), staged into a temporary `HOME` so the CLI
//! resolves it offline. The cloud leg is a loopback HTTP server that speaks
//! OpenAI chat-completions and records every request, so the assertions are
//! about what actually crossed the wire: the path, the bearer credential, the
//! model, the thinking mode and the shared generation options.
//!
//! No test here asserts what a *policy-free* run chooses: live device stress
//! can legitimately pick the cloud leg. Those cases are pinned under fixed
//! snapshots in the core and CLI unit tests.
#![cfg(all(feature = "llm-llamacpp", unix))]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use xybrid_core::testing::model_fixtures;

const DEFAULT_MODEL_ID: &str = "functiongemma-270m-it";
const PROMPT: &str = "Reply with the single word hello.";
const SYSTEM_PROMPT: &str = "You are a terse assistant for the policy routing test.";
const FAKE_REPLY: &str = "FAKE_DEEPSEEK_REPLY";
const DUMMY_DEEPSEEK_KEY: &str = "dummy-deepseek-key";
const DUMMY_PLATFORM_KEY: &str = "dummy-platform-key";
const CHILD_TIMEOUT: Duration = Duration::from_secs(600);

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

fn model_id() -> String {
    std::env::var("XYBRID_POLICY_E2E_MODEL_ID").unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string())
}

/// The fixture directory, or `None` to skip. When `XYBRID_REQUIRE_MODELS` is
/// set a missing model is a failure, never a silent skip.
fn fixture_dir_or_skip(model_id: &str) -> Option<PathBuf> {
    match model_fixtures::model_or_skip(model_id) {
        Some(dir) => Some(dir),
        None => {
            assert!(
                std::env::var_os("XYBRID_REQUIRE_MODELS").is_none(),
                "XYBRID_REQUIRE_MODELS is set but model '{model_id}' is not available"
            );
            None
        }
    }
}

fn link_or_copy(source: &Path, destination: &Path) {
    if std::fs::hard_link(source, destination).is_ok() {
        return;
    }
    if std::os::unix::fs::symlink(source, destination).is_ok() {
        return;
    }
    std::fs::copy(source, destination)
        .unwrap_or_else(|err| panic!("failed to materialize {}: {err}", source.display()));
}

/// Stage the fixture byte-for-byte into `<home>/.xybrid/cache/extracted/<id>/`,
/// which is what `RegistryClient::resolve_offline` checks (metadata plus
/// every file the metadata lists). Generation params are NOT rewritten.
fn seed_home(home: &Path, model_id: &str, fixture_dir: &Path) {
    let extracted = home
        .join(".xybrid")
        .join("cache")
        .join("extracted")
        .join(model_id);
    std::fs::create_dir_all(&extracted).unwrap();
    std::fs::create_dir_all(home.join(".xybrid").join("cache").join("models")).unwrap();

    let metadata_path = fixture_dir.join("model_metadata.json");
    std::fs::copy(&metadata_path, extracted.join("model_metadata.json")).unwrap();
    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).unwrap()).unwrap();
    for file in metadata["files"].as_array().expect("metadata.files") {
        let name = file.as_str().expect("file name");
        link_or_copy(&fixture_dir.join(name), &extracted.join(name));
    }
}

/// Loopback OpenAI-compatible endpoint that records every request verbatim.
struct FakeDeepSeek {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FakeDeepSeek {
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
            url: format!("http://{}/v1", addr),
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
            r#"{{"id":"chatcmpl-1","model":"deepseek-flash","choices":[{{"index":0,"message":{{"role":"assistant","content":"{FAKE_REPLY}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}}}"#
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

impl Drop for FakeDeepSeek {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Loopback registry that serves a passthrough resolve response and the real
/// GGUF bytes, so the cold-cache download/extraction path can be exercised
/// end to end without touching the network. Once `reject_unexpected_requests`
/// is set, every further request is answered with 503 (and still recorded).
struct FakeRegistry {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    reject: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FakeRegistry {
    fn start(model_id: &str, fixture_dir: &Path) -> Self {
        let metadata: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixture_dir.join("model_metadata.json")).unwrap(),
        )
        .unwrap();
        let model_file_name = metadata["files"]
            .as_array()
            .expect("metadata.files")
            .iter()
            .filter_map(|file| file.as_str())
            .find(|name| name.ends_with(".gguf"))
            .expect("fixture declares a GGUF")
            .to_string();
        let model_file = fixture_dir.join(&model_file_name);

        let sha256 = {
            use sha2::{Digest, Sha256};
            let mut file = std::fs::File::open(&model_file).unwrap();
            let mut hasher = Sha256::new();
            let mut buf = [0u8; 64 * 1024];
            loop {
                let read = file.read(&mut buf).unwrap();
                if read == 0 {
                    break;
                }
                hasher.update(&buf[..read]);
            }
            format!("{:x}", hasher.finalize())
        };
        let size_bytes = std::fs::metadata(&model_file).unwrap().len();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://{addr}");
        let resolve_body = serde_json::json!({
            "mask": model_id,
            "platform": "e2e",
            "resolved": {
                "hf_repo": "ggml-org/functiongemma-270m-it-GGUF",
                "file": model_file_name,
                "download_url": format!("{base}/files/{model_file_name}"),
                "format": "gguf",
                "quantization": "Q8_0",
                "size_bytes": size_bytes,
                "sha256": sha256,
                "passthrough": true,
                "model_metadata": metadata,
            }
        });

        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let reject = Arc::new(AtomicBool::new(false));
        let thread = {
            let requests = requests.clone();
            let stop = stop.clone();
            let reject = reject.clone();
            let resolve_body = resolve_body.clone();
            let model_file = model_file.clone();
            thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            Self::serve(stream, &requests, &reject, &resolve_body, &model_file)
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            })
        };

        Self {
            url: base,
            requests,
            stop,
            reject,
            thread: Some(thread),
        }
    }

    fn serve(
        mut stream: TcpStream,
        requests: &Arc<Mutex<Vec<String>>>,
        reject: &Arc<AtomicBool>,
        resolve_body: &serde_json::Value,
        model_file: &Path,
    ) {
        stream.set_nonblocking(false).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 8192];
        while request.windows(4).position(|w| w == b"\r\n\r\n").is_none() {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => request.extend_from_slice(&buf[..n]),
            }
        }
        let request_text = String::from_utf8_lossy(&request).into_owned();
        let path = request_text
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/")
            .to_string();
        requests.lock().unwrap().push(request_text);

        if reject.load(Ordering::SeqCst) {
            let _ = write_http_response(&mut stream, 503, "text/plain", b"unexpected request");
            return;
        }

        if path.contains("/v1/models/") && path.contains("/resolve") {
            let body = serde_json::to_vec(resolve_body).unwrap();
            let _ = write_http_response(&mut stream, 200, "application/json", &body);
            return;
        }

        if let Some(name) = path.strip_prefix("/files/") {
            let expected = model_file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if name == expected {
                match std::fs::File::open(model_file) {
                    Ok(mut file) => {
                        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n"
                        );
                        if stream.write_all(header.as_bytes()).is_ok() {
                            let _ = std::io::copy(&mut file, &mut stream);
                        }
                    }
                    Err(_) => {
                        let _ = write_http_response(&mut stream, 404, "text/plain", b"missing");
                    }
                }
                return;
            }
        }

        let _ = write_http_response(&mut stream, 404, "text/plain", b"not found");
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }

    fn reject_unexpected_requests(&self) {
        self.reject.store(true, Ordering::SeqCst);
    }
}

impl Drop for FakeRegistry {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = if status == 200 { "OK" } else { "Error" };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn run_cli(home: &Path, args: &[&str]) -> Run {
    run_cli_with_env(home, args, &[])
}

/// Spawn `xybrid run` with an isolated child environment. `env_overrides` are
/// applied after the removals, so an explicit loopback registry URL wins over
/// the "never leak the developer's env" cleanup.
fn run_cli_with_env(home: &Path, args: &[&str], env_overrides: &[(&str, &str)]) -> Run {
    let mut command = Command::new(xybrid_bin());
    command
        .current_dir(workspace_root())
        .env("HOME", home)
        .env("DEEPSEEK_API_KEY", DUMMY_DEEPSEEK_KEY)
        .env("XYBRID_API_KEY", DUMMY_PLATFORM_KEY)
        .env("NO_COLOR", "1")
        .env("XYBRID_TELEMETRY_OPTOUT", "1")
        // Never let a developer's configured platform/gateway/proxy leak in.
        .env_remove("XYBRID_GATEWAY_URL")
        .env_remove("XYBRID_PLATFORM_URL")
        .env_remove("XYBRID_REGISTRY_URL")
        .env_remove("HTTP_PROXY")
        .env_remove("HTTPS_PROXY")
        .env_remove("http_proxy")
        .env_remove("https_proxy")
        .env_remove("ALL_PROXY");
    for (key, value) in env_overrides {
        command.env(key, value);
    }

    let mut child = command
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

/// Value of a `ui::kv` line whose key is exactly `key`.
///
/// The format is `"  {:<16} {}"`: the key is padded to 16 columns, so a
/// real match is followed by at least two spaces. That also keeps `Policy`
/// from matching the `Policy file` line.
fn kv_line_value(line: &str, key: &str) -> Option<String> {
    let rest = line.trim_start().strip_prefix(key)?;
    rest.starts_with("  ").then(|| rest.trim().to_string())
}

/// Value of the first `ui::kv` line whose key is `key`.
fn kv_value(stdout: &str, key: &str) -> Option<String> {
    stdout.lines().find_map(|line| kv_line_value(line, key))
}

struct TestEnv {
    _temp: tempfile::TempDir,
    home: PathBuf,
    pipeline: PathBuf,
    local_only: PathBuf,
    prefer_cloud: PathBuf,
    two_stage: PathBuf,
    fake: FakeDeepSeek,
}

fn setup() -> Option<TestEnv> {
    let model_id = model_id();
    let fixture_dir = fixture_dir_or_skip(&model_id)?;
    let temp = tempfile::tempdir().expect("temp dir");
    let home = temp.path().join("home");
    seed_home(&home, &model_id, &fixture_dir);

    let fake = FakeDeepSeek::start();
    let stage = format!(
        r#"  - id: llm
    model: {model_id}
    target: auto
    provider: deepseek
    cloud_model: deepseek-flash
    gateway_url: "{url}"
    api_key: "$DEEPSEEK_API_KEY"
    thinking: disabled
    system_prompt: "{SYSTEM_PROMPT}"
    temperature: 0
    max_tokens: 16
"#,
        url = fake.url
    );
    let pipeline = temp.path().join("hybrid.yaml");
    std::fs::write(&pipeline, format!("name: policy-e2e\nstages:\n{stage}")).unwrap();
    let two_stage = temp.path().join("two-stage.yaml");
    std::fs::write(
        &two_stage,
        format!("name: policy-e2e-two\nstages:\n{stage}{stage}").replacen(
            "id: llm",
            "id: first",
            1,
        ),
    )
    .unwrap();
    let local_only = temp.path().join("local-only.yaml");
    std::fs::write(
        &local_only,
        "version: \"1.0.0\"\ndeny_cloud_if:\n  - \"true\"\n",
    )
    .unwrap();
    let prefer_cloud = temp.path().join("prefer-cloud.yaml");
    std::fs::write(
        &prefer_cloud,
        "version: \"1.0.0\"\nroute_cloud_if:\n  - \"true\"\n",
    )
    .unwrap();

    Some(TestEnv {
        _temp: temp,
        home,
        pipeline,
        local_only,
        prefer_cloud,
        two_stage,
        fake,
    })
}

fn assert_success(run: &Run) {
    assert!(
        run.status.success(),
        "xybrid run failed ({:?})\nstdout:\n{}\nstderr:\n{}",
        run.status.code(),
        run.stdout,
        run.stderr
    );
}

#[test]
fn local_only_policy_runs_the_gguf_model() {
    let Some(env) = setup() else { return };

    let run = run_cli(
        &env.home,
        &[
            "-c",
            env.pipeline.to_str().unwrap(),
            "--input-text",
            PROMPT,
            "--policy",
            env.local_only.to_str().unwrap(),
        ],
    );
    assert_success(&run);

    assert_eq!(
        kv_value(&run.stdout, "Routing").as_deref(),
        Some("local"),
        "{}",
        run.stdout
    );
    let reason = kv_value(&run.stdout, "Reason").unwrap_or_default();
    assert!(reason.contains("policy_deny"), "{reason}");
    assert_eq!(
        kv_value(&run.stdout, "Backend").as_deref(),
        Some("template-executor"),
        "{}",
        run.stdout
    );
    assert!(
        run.stdout.contains("Pipeline completed successfully"),
        "{}",
        run.stdout
    );
    assert!(
        !run.stdout.contains(FAKE_REPLY),
        "local run must not show the fake cloud reply"
    );
    assert!(
        env.fake.requests().is_empty(),
        "a denied stage must make zero cloud requests: {:?}",
        env.fake.requests()
    );
}

#[test]
fn prefer_cloud_policy_calls_the_fake_deepseek_endpoint() {
    let Some(env) = setup() else { return };

    let run = run_cli(
        &env.home,
        &[
            "-c",
            env.pipeline.to_str().unwrap(),
            "--input-text",
            PROMPT,
            "--policy",
            env.prefer_cloud.to_str().unwrap(),
        ],
    );
    assert_success(&run);

    assert_eq!(
        kv_value(&run.stdout, "Routing").as_deref(),
        Some("cloud"),
        "{}",
        run.stdout
    );
    let reason = kv_value(&run.stdout, "Reason").unwrap_or_default();
    assert!(reason.contains("policy_route_cloud"), "{reason}");
    assert_eq!(
        kv_value(&run.stdout, "Backend").as_deref(),
        Some("cloud:deepseek:gateway"),
        "{}",
        run.stdout
    );
    assert!(run.stdout.contains(FAKE_REPLY), "{}", run.stdout);

    let requests = env.fake.requests();
    assert_eq!(requests.len(), 1, "exactly one cloud request: {requests:?}");
    let request = &requests[0];
    assert!(
        request.starts_with("POST /v1/chat/completions "),
        "{request}"
    );
    let lower = request.to_ascii_lowercase();
    assert!(
        lower.contains(&format!("authorization: bearer {DUMMY_DEEPSEEK_KEY}")),
        "{request}"
    );
    assert!(
        !request.contains(DUMMY_PLATFORM_KEY),
        "platform key must never reach a provider: {request}"
    );
    let body_start = request.find("\r\n\r\n").unwrap() + 4;
    let body: serde_json::Value = serde_json::from_str(&request[body_start..]).unwrap();
    assert_eq!(body["model"], "deepseek-flash");
    assert_eq!(body["thinking"], serde_json::json!({"type": "disabled"}));
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], SYSTEM_PROMPT);
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], PROMPT);
}

#[test]
fn dry_run_loads_policy_without_inference() {
    let Some(env) = setup() else { return };

    let denied = run_cli(
        &env.home,
        &[
            "-c",
            env.pipeline.to_str().unwrap(),
            "--input-text",
            PROMPT,
            "--dry-run",
            "--policy",
            env.local_only.to_str().unwrap(),
        ],
    );
    assert_success(&denied);
    assert_eq!(
        kv_value(&denied.stdout, "Policy").as_deref(),
        Some("DENIED"),
        "{}",
        denied.stdout
    );
    let routing = kv_value(&denied.stdout, "Routing").unwrap_or_default();
    assert!(routing.starts_with("local (policy_deny"), "{routing}");
    assert!(
        !denied.stdout.contains("Backend"),
        "dry-run must not claim an executed backend"
    );

    let preferred = run_cli(
        &env.home,
        &[
            "-c",
            env.pipeline.to_str().unwrap(),
            "--input-text",
            PROMPT,
            "--dry-run",
            "--policy",
            env.prefer_cloud.to_str().unwrap(),
        ],
    );
    assert_success(&preferred);
    assert_eq!(
        kv_value(&preferred.stdout, "Policy").as_deref(),
        Some("ALLOWED"),
        "{}",
        preferred.stdout
    );
    let routing = kv_value(&preferred.stdout, "Routing").unwrap_or_default();
    assert!(
        routing.starts_with("cloud (policy_route_cloud"),
        "{routing}"
    );

    // Two stages: the second depends on upstream output and is not simulated.
    let two = run_cli(
        &env.home,
        &[
            "-c",
            env.two_stage.to_str().unwrap(),
            "--input-text",
            PROMPT,
            "--dry-run",
            "--policy",
            env.prefer_cloud.to_str().unwrap(),
        ],
    );
    assert_success(&two);
    let policies: Vec<String> = two
        .stdout
        .lines()
        .filter_map(|line| kv_line_value(line, "Policy"))
        .collect();
    assert_eq!(
        policies,
        vec!["ALLOWED".to_string(), "UNKNOWN".to_string()],
        "{}",
        two.stdout
    );
    assert!(
        two.stdout
            .contains("upstream output unavailable without execution"),
        "{}",
        two.stdout
    );

    assert!(
        env.fake.requests().is_empty(),
        "dry-run must perform no inference: {:?}",
        env.fake.requests()
    );
    assert!(!denied.stdout.contains(FAKE_REPLY) && !preferred.stdout.contains(FAKE_REPLY));
}

/// Cold cache: the hybrid local leg must download, extract (metadata included),
/// and run locally; the next run must reuse the extracted copy without
/// touching the registry.
#[test]
fn cold_cache_hybrid_downloads_extracts_and_then_runs_offline() {
    let model_id = model_id();
    let Some(fixture_dir) = fixture_dir_or_skip(&model_id) else {
        return;
    };

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();

    let registry = FakeRegistry::start(&model_id, &fixture_dir);
    let fake = FakeDeepSeek::start();

    let stage = format!(
        r#"  - id: llm
    model: {model_id}
    target: auto
    provider: deepseek
    cloud_model: deepseek-flash
    gateway_url: "{url}"
    api_key: "$DEEPSEEK_API_KEY"
    thinking: disabled
    system_prompt: "{SYSTEM_PROMPT}"
    temperature: 0
    max_tokens: 16
"#,
        url = fake.url
    );
    let pipeline = temp.path().join("hybrid-cold.yaml");
    std::fs::write(
        &pipeline,
        format!("name: policy-e2e-cold\nstages:\n{stage}"),
    )
    .unwrap();
    let local_only = temp.path().join("local-only.yaml");
    std::fs::write(
        &local_only,
        "version: \"1.0.0\"\ndeny_cloud_if:\n  - \"true\"\n",
    )
    .unwrap();

    let args = [
        "-c",
        pipeline.to_str().unwrap(),
        "--input-text",
        PROMPT,
        "--policy",
        local_only.to_str().unwrap(),
    ];
    let registry_env = [("XYBRID_REGISTRY_URL", registry.url.as_str())];

    // First run: cold cache -> registry resolve + passthrough download.
    let run = run_cli_with_env(&home, &args, &registry_env);
    assert_success(&run);
    assert_eq!(
        kv_value(&run.stdout, "Routing").as_deref(),
        Some("local"),
        "{}",
        run.stdout
    );
    assert_eq!(
        kv_value(&run.stdout, "Backend").as_deref(),
        Some("template-executor"),
        "{}",
        run.stdout
    );

    let extracted = home
        .join(".xybrid")
        .join("cache")
        .join("extracted")
        .join(&model_id);
    assert!(
        extracted.join("model_metadata.json").is_file(),
        "extraction must materialize model_metadata.json: {}",
        extracted.display()
    );
    let metadata: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(extracted.join("model_metadata.json")).unwrap(),
    )
    .unwrap();
    for file in metadata["files"].as_array().expect("metadata.files") {
        let name = file.as_str().unwrap();
        assert!(
            extracted.join(name).is_file(),
            "extraction must materialize {name}"
        );
    }

    let first_requests = registry.requests();
    assert!(
        first_requests
            .iter()
            .any(|request| request.contains("/v1/models/")),
        "cold run must resolve through the registry: {first_requests:?}"
    );
    assert!(
        first_requests
            .iter()
            .any(|request| request.contains("/files/")),
        "cold run must download the model file: {first_requests:?}"
    );
    assert!(
        fake.requests().is_empty(),
        "local-only policy must make no cloud request: {:?}",
        fake.requests()
    );

    // Second run: the extracted copy is reused; the registry rejects
    // everything from here on, so any request is a failure.
    registry.reject_unexpected_requests();
    let request_count = registry.requests().len();

    let offline = run_cli_with_env(&home, &args, &registry_env);
    assert_success(&offline);
    assert_eq!(
        kv_value(&offline.stdout, "Routing").as_deref(),
        Some("local"),
        "{}",
        offline.stdout
    );
    assert_eq!(
        kv_value(&offline.stdout, "Backend").as_deref(),
        Some("template-executor"),
        "{}",
        offline.stdout
    );
    assert_eq!(
        registry.requests().len(),
        request_count,
        "a warm extracted cache must not call the registry again: {:?}",
        registry.requests()
    );
    assert!(
        fake.requests().is_empty(),
        "offline run must still make no cloud request: {:?}",
        fake.requests()
    );
}

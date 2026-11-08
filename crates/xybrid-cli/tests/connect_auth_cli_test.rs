//! `xybrid connect` and `xybrid auth` against a loopback platform (httpmock).
//!
//! Every case runs the real binary in a sandbox: its own HOME, credentials
//! file (`XYBRID_CREDENTIALS_FILE`, which also holds the install ID and
//! notice dates), Git configuration and project repository, with CI markers,
//! opt-outs, proxies, keys and OTLP variables removed unless a case sets them.
//! The platform is a custom loopback host, so `--allow-custom-host` is passed
//! and the console comes from `XYBRID_CONSOLE_URL`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use chrono::{SecondsFormat, Utc};
use httpmock::prelude::*;
use httpmock::Mock;
use serde_json::{json, Value};

const KEY: &str = concat!("sk_live", "_ZZsecretTESTkey0123456789abcdefXY");
const SECOND_KEY: &str = concat!("sk_live", "_YYsecondTESTkey0123456789abcdefWX");
const PROJECT_KEY: &str = concat!("sk_live", "_XXprojectTESTkey0123456789abcdefV");
const CLAIM_TOKEN: &str = "ClaimTokenClaimTokenClaimTokenClaimToken123";
const SECOND_CLAIM_TOKEN: &str = "SecondClaimSecondClaimSecondClaimSecond456";
/// An independently computed proof-of-work vector (Python `hashlib`): the
/// smallest nonce giving this challenge 8 leading zero bits is 47.
const CHALLENGE: &str = "xybrid-pow-test-vector";
const CHALLENGE_NONCE: &str = "47";
const CONSOLE: &str = "http://localhost:5173";
/// A random v4 UUID seeded as the sandbox's install ID.
const INSTALL_ID: &str = "6f1c1a4e-0b7d-4c55-9a8e-1f2d3c4b5a69";

/// Environment a developer machine or CI runner may carry into the CLI.
const SCRUBBED: &[&str] = &[
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "JENKINS_URL",
    "TF_BUILD",
    "TEAMCITY_VERSION",
    "BITBUCKET_BUILD_NUMBER",
    "CODEBUILD_BUILD_ID",
    "TRAVIS",
    "DRONE",
    "XYBRID_API_KEY",
    "XYBRID_PLATFORM_URL",
    "XYBRID_GATEWAY_URL",
    "XYBRID_CONSOLE_URL",
    "XYBRID_TELEMETRY_OPTOUT",
    "XYBRID_QUIET",
    "XYBRID_DEVICE_ID",
    "DO_NOT_TRACK",
    "CLAUDECODE",
    "CODEX_SANDBOX",
    "GEMINI_CLI",
    "OTEL_SDK_DISABLED",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
];

fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Skip (pass) a case on a machine without Git: `connect` needs it to keep
/// keys out of commits, and these cases run inside a repository.
macro_rules! require_git {
    () => {
        if !git_available() {
            eprintln!("skipping: git is not installed");
            return;
        }
    };
}

fn iso(days: i64) -> String {
    (Utc::now() + chrono::Duration::days(days)).to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn named(slug: &str, name: &str) -> Value {
    json!({"slug": slug, "name": name})
}

/// A `POST /v1/agent/connect` answer, deadlines relative to now.
fn provision(api_key: &str, claim_token: Option<&str>, key_days: i64, replayed: bool) -> Value {
    let mut body = json!({
        "api_key": api_key,
        "key_prefix": &api_key[8..16],
        "scope": "client",
        "organization": named("my-app", "my-app"),
        "project": named("default", "Default"),
        "key_expires_at": iso(key_days),
        "claim_expires_at": iso(key_days + 7),
        "limits": {"gateway_daily_requests": 25, "max_projects": 5},
        "terms_url": "https://www.xybrid.ai/terms",
        "privacy_url": "https://www.xybrid.ai/privacy",
        "replayed": replayed,
    });
    if let Some(token) = claim_token {
        body["claim_token"] = json!(token);
    }
    body
}

/// A `GET /v1/agent/workspace` answer.
fn workspace(expired: bool, claimed: bool, verification: &str) -> Value {
    let key_days = if expired { -1 } else { 7 };
    json!({
        "organization": named("my-app", "my-app"),
        "project": named("default", "Default"),
        "scope": "client",
        "provisional": !claimed,
        "claimed": claimed,
        "expired": expired,
        "key_expires_at": iso(key_days),
        "claim_expires_at": iso(key_days + 7),
        "verification": {
            "state": verification,
            "traces_received": if verification == "unavailable" { Value::Null } else { json!(verification == "received") },
            "spans": if verification == "received" { json!(3) } else { json!(0) },
            "traces": if verification == "received" { json!(1) } else { json!(0) },
            "first_ingested_at": Value::Null,
            "last_ingested_at": Value::Null,
        }
    })
}

/// Only allowlisted, non-identifying fields reach `connect`.
fn sends_no_identifying_fields(request: &HttpMockRequest) -> bool {
    let Some(body) = request.body.as_ref() else {
        return false;
    };
    let Ok(Value::Object(body)) = serde_json::from_slice::<Value>(body) else {
        return false;
    };
    let top = [
        "source",
        "project_name",
        "install_id",
        "client_info",
        "challenge",
        "nonce",
    ];
    let info = ["cli_version", "os", "arch", "binding"];
    body.keys().all(|k| top.contains(&k.as_str()))
        && body
            .get("client_info")
            .and_then(Value::as_object)
            .is_some_and(|ci| ci.keys().all(|k| info.contains(&k.as_str())))
}

fn mock_challenge(server: &MockServer) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(GET).path("/v1/agent/challenge");
        then.status(200).json_body(json!({
            "challenge": CHALLENGE,
            "difficulty": 8,
            "expires_at": iso(0),
            "algorithm": "sha256-leading-zero-bits",
        }));
    })
}

fn mock_connect(server: &MockServer) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(POST)
            .path("/v1/agent/connect")
            .header_exists("Idempotency-Key")
            .json_body_partial(
                json!({"challenge": CHALLENGE, "nonce": CHALLENGE_NONCE, "install_id": INSTALL_ID})
                    .to_string(),
            )
            .matches(sends_no_identifying_fields);
        then.status(201)
            .header("Cache-Control", "no-store")
            .json_body(provision(KEY, Some(CLAIM_TOKEN), 7, false));
    })
}

fn mock_workspace<'a>(server: &'a MockServer, key: &str, body: Value) -> Mock<'a> {
    server.mock(|when, then| {
        when.method(GET)
            .path("/v1/agent/workspace")
            .header("Authorization", format!("Bearer {key}"));
        then.status(200).json_body(body);
    })
}

/// The finished process.
struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl Run {
    fn json(&self) -> Value {
        serde_json::from_str(&self.stdout)
            .unwrap_or_else(|err| panic!("stdout is not one JSON object ({err}):\n{self}"))
    }

    /// No raw key ever reaches the terminal or the JSON.
    fn assert_no_secret(&self, secret: &str) {
        assert!(!self.stdout.contains(secret), "key on stdout:\n{self}");
        assert!(!self.stderr.contains(secret), "key on stderr:\n{self}");
    }
}

impl std::fmt::Display for Run {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "exit {:?}\n--- stdout\n{}\n--- stderr\n{}",
            self.code, self.stdout, self.stderr
        )
    }
}

impl From<Output> for Run {
    fn from(output: Output) -> Self {
        Self {
            code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    }
}

/// An isolated user: home, private state, Git configuration and a project.
struct Sandbox {
    _dir: tempfile::TempDir,
    home: PathBuf,
    state: PathBuf,
    git_config: PathBuf,
    project: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let sandbox = Self::new_without_git();
        sandbox.git(&sandbox.project, &["init", "-q"]);
        sandbox
    }

    /// A sandbox whose project is a plain directory.
    fn new_without_git() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        let base = dir.path().canonicalize().expect("canonical temp dir");
        let home = base.join("home");
        let project = base.join("my-app");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        let git_config = base.join("gitconfig");
        std::fs::write(&git_config, "").unwrap();
        let sandbox = Self {
            _dir: dir,
            home,
            state: base.join("state"),
            git_config,
            project,
        };
        sandbox.seed_install_id();
        sandbox
    }

    /// A second repository sharing this sandbox's user state.
    fn another_project(&self, name: &str) -> PathBuf {
        let path = self.project.parent().unwrap().join(name);
        std::fs::create_dir_all(&path).unwrap();
        self.git(&path, &["init", "-q"]);
        path
    }

    fn seed_install_id(&self) {
        create_private_dir(&self.state);
        std::fs::write(self.state.join("install-id"), format!("{INSTALL_ID}\n")).unwrap();
    }

    fn credentials_path(&self) -> PathBuf {
        self.state.join("credentials.json")
    }

    fn credentials(&self) -> Value {
        let text = std::fs::read_to_string(self.credentials_path()).expect("credentials file");
        serde_json::from_str(&text).expect("credentials JSON")
    }

    fn git(&self, dir: &Path, args: &[&str]) -> Output {
        let output = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_CONFIG_GLOBAL", &self.git_config)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", self.home.join(".config"))
            .output()
            .expect("git runs");
        output
    }

    /// The CLI in `dir` with a scrubbed environment.
    fn command_in(&self, dir: &Path, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xybrid"));
        command.args(args).current_dir(dir);
        for name in SCRUBBED {
            command.env_remove(name);
        }
        command
            .env("HOME", &self.home)
            .env("USERPROFILE", &self.home)
            .env("XDG_CONFIG_HOME", self.home.join(".config"))
            .env("XYBRID_CREDENTIALS_FILE", self.credentials_path())
            .env("GIT_CONFIG_GLOBAL", &self.git_config)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("NO_COLOR", "1")
            .stdin(Stdio::null());
        command
    }

    fn command(&self, args: &[&str]) -> Command {
        self.command_in(&self.project, args)
    }

    fn run(&self, args: &[&str]) -> Run {
        self.command(args).output().expect("xybrid runs").into()
    }

    fn run_with(&self, args: &[&str], env: &[(&str, &str)]) -> Run {
        let mut command = self.command(args);
        command.envs(env.iter().copied());
        command.output().expect("xybrid runs").into()
    }
}

#[cfg(unix)]
fn create_private_dir(path: &Path) {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
        .unwrap();
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) {
    std::fs::create_dir_all(path).unwrap();
}

/// `xybrid --platform-url <server> <args…> --allow-custom-host`.
fn platform_args<'a>(url: &'a str, args: &[&'a str]) -> Vec<&'a str> {
    let mut all = vec!["--platform-url", url];
    all.extend_from_slice(args);
    all.push("--allow-custom-host");
    all
}

fn connect_args(url: &str) -> Vec<&str> {
    platform_args(
        url,
        &[
            "connect",
            "--agent",
            "claude-code",
            "--write-env",
            ".env",
            "--json",
        ],
    )
}

/// The steps an agent is told to take, in order.
fn next_step_kinds(report: &Value) -> Vec<String> {
    report["next_steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["kind"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn connect_creates_a_workspace_and_keeps_the_key_private() {
    require_git!();
    let sandbox = Sandbox::new();
    std::fs::write(
        sandbox.project.join(".env"),
        "# app\nDATABASE_URL=postgres://localhost/app\n",
    )
    .unwrap();
    std::fs::write(sandbox.project.join(".gitignore"), "node_modules\n").unwrap();
    #[cfg(unix)]
    let env_mode = {
        use std::os::unix::fs::PermissionsExt;
        let path = sandbox.project.join(".env");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777
    };
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let connect = mock_connect(&server);
    let url = server.base_url();

    let run = sandbox.run_with(&connect_args(&url), &[("XYBRID_CONSOLE_URL", CONSOLE)]);

    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(KEY);
    challenge.assert_hits(1);
    connect.assert_hits(1);
    let report = run.json();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["command"], "connect");
    assert_eq!(report["status"], "created");
    assert_eq!(report["key"]["source"], "created");
    assert_eq!(report["key"]["scope"], "client");
    let preview = report["key"]["preview"].as_str().unwrap();
    assert!(
        preview.starts_with("sk_live_") && preview.len() < 20,
        "{preview}"
    );
    assert_eq!(report["platform"]["custom_host"], true);
    assert_eq!(report["platform"]["api_origin"], url);
    assert_eq!(report["workspace"]["organization"]["name"], "my-app");
    assert_eq!(report["workspace"]["provisional"], true);
    assert!(report["workspace"]["key_expires_at"]
        .as_str()
        .unwrap()
        .ends_with('Z'));
    assert_eq!(report["env_file"]["path"], ".env");
    assert_eq!(report["env_file"]["variable"], "XYBRID_API_KEY");
    assert_eq!(report["env_file"]["written"], true);
    assert_eq!(report["env_file"]["gitignored"], true);
    assert_eq!(report["env_file"]["gitignore_updated"], true);
    assert_eq!(report["credentials"]["saved"], true);
    assert_eq!(report["credentials"]["recovery"], true);
    assert_eq!(
        report["claim_url"],
        format!("{CONSOLE}/claim/{CLAIM_TOKEN}"),
        "built locally from the console origin"
    );
    assert_eq!(report["verification"]["state"], "not_checked");
    assert_eq!(next_step_kinds(&report), ["wire_init", "verify", "relay"]);
    let tell = report["tell_the_user"].as_str().unwrap();
    assert!(
        tell.contains("Created a provisional Xybrid workspace"),
        "{tell}"
    );
    assert!(
        tell.contains(&format!("{CONSOLE}/claim/{CLAIM_TOKEN}")),
        "{tell}"
    );
    assert!(tell.contains("Trace export is not configured"), "{tell}");
    assert!(!tell.contains("verified"), "nothing was verified: {tell}");
    assert!(tell.contains("https://www.xybrid.ai/terms"), "{tell}");

    // The project file: unrelated lines kept, key appended under a comment
    // that names the claim command but never the token or link.
    let env = std::fs::read_to_string(sandbox.project.join(".env")).unwrap();
    assert!(
        env.starts_with("# app\nDATABASE_URL=postgres://localhost/app\n"),
        "{env}"
    );
    assert!(env.contains(&format!("\nXYBRID_API_KEY={KEY}\n")), "{env}");
    assert!(env.contains("xybrid auth claim"), "{env}");
    assert!(
        !env.contains(CLAIM_TOKEN) && !env.contains("/claim/"),
        "{env}"
    );

    // Ignored by Git, verified with Git itself.
    let gitignore = std::fs::read_to_string(sandbox.project.join(".gitignore")).unwrap();
    assert!(gitignore.starts_with("node_modules\n"), "{gitignore}");
    assert!(gitignore.contains("\n/.env\n"), "{gitignore}");
    let check = sandbox.git(&sandbox.project, &["check-ignore", "-q", ".env"]);
    assert!(check.status.success(), "git check-ignore .env");

    // Private recovery state: key and claim token, no leftover retry state.
    let credentials = sandbox.credentials();
    let entry = &credentials["credentials"][0];
    assert_eq!(entry["platform_origin"], url);
    assert_eq!(entry["api_key"], KEY);
    assert_eq!(entry["claim_token"], CLAIM_TOKEN);
    assert_eq!(entry["source"], "claude-code");
    assert_eq!(entry["env_file"]["path"], ".env");
    assert_eq!(credentials["pending"], json!([]));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&sandbox.credentials_path()), 0o600);
        assert_eq!(mode(&sandbox.state), 0o700);
        assert_eq!(
            mode(&sandbox.project.join(".env")),
            env_mode,
            "an existing file keeps its permissions"
        );
    }
}

#[test]
fn a_new_env_file_is_created_private_and_reused_on_the_next_connect() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let connect = mock_connect(&server);
    let status = mock_workspace(&server, KEY, workspace(false, false, "none"));
    let url = server.base_url();

    let first = sandbox.run(&connect_args(&url));
    assert_eq!(first.code, Some(0), "{first}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(sandbox.project.join(".env"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "a new key file is private");
    }
    // No console configured for this custom host: no link, honest copy.
    let report = first.json();
    assert_eq!(report["claim_url"], Value::Null);
    assert!(
        report["tell_the_user"]
            .as_str()
            .unwrap()
            .contains("xybrid auth claim"),
        "{first}"
    );

    // Same project again: the saved credential is checked and reused.
    let again = sandbox.run(&connect_args(&url));
    assert_eq!(again.code, Some(0), "{again}");
    again.assert_no_secret(KEY);
    let report = again.json();
    assert_eq!(report["status"], "existing");
    assert_eq!(report["key"]["source"], "credentials");
    assert_eq!(report["workspace"]["lifecycle_source"], "server");
    assert_eq!(report["verification"]["state"], "none");
    assert_eq!(report["env_file"]["already_configured"], true);
    assert_eq!(report["env_file"]["written"], false);
    challenge.assert_hits(1);
    connect.assert_hits(1);
    status.assert_hits(1);
    let env = std::fs::read_to_string(sandbox.project.join(".env")).unwrap();
    assert_eq!(env.matches("XYBRID_API_KEY=").count(), 1, "{env}");
}

#[test]
fn auth_status_claim_and_logout_use_the_saved_credentials() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    let status = mock_workspace(&server, KEY, workspace(false, false, "received"));
    let url = server.base_url();
    let console = [("XYBRID_CONSOLE_URL", CONSOLE)];
    assert_eq!(
        sandbox.run_with(&connect_args(&url), &console).code,
        Some(0)
    );

    let run = sandbox.run(&platform_args(&url, &["auth", "status", "--json"]));
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(KEY);
    assert!(
        !run.stdout.contains(CLAIM_TOKEN),
        "status omits the link by default"
    );
    let report = run.json();
    assert_eq!(report["status"], "connected");
    assert_eq!(report["key"]["source"], "credentials");
    assert_eq!(report["verification"]["state"], "received");
    assert_eq!(report["verification"]["traces_received"], true);
    assert_eq!(report["claim_url"], Value::Null);
    assert!(report["stored"]["key_expires_at"].is_string(), "{run}");

    let run = sandbox.run(&platform_args(
        &url,
        &["auth", "status", "--json", "--show-claim-url"],
    ));
    assert_eq!(
        run.json()["claim_url"],
        format!("{CONSOLE}/claim/{CLAIM_TOKEN}")
    );
    status.assert_hits(2);

    let run = sandbox.run(&platform_args(
        &url,
        &["auth", "claim", "--no-browser", "--json"],
    ));
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(KEY);
    let report = run.json();
    assert_eq!(report["status"], "claimable");
    assert_eq!(report["opened_browser"], false);
    assert_eq!(
        report["claim_url"],
        format!("{CONSOLE}/claim/{CLAIM_TOKEN}")
    );

    let run = sandbox.run(&platform_args(&url, &["auth", "logout", "--json"]));
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(KEY);
    let report = run.json();
    assert_eq!(report["status"], "logged_out");
    assert_eq!(report["removed"], true);
    assert_eq!(report["env_file"]["already_configured"], true);
    let tell = report["tell_the_user"].as_str().unwrap();
    assert!(tell.contains("did not revoke"), "{tell}");
    assert!(tell.contains(".env still sets XYBRID_API_KEY"), "{tell}");
    assert_eq!(sandbox.credentials()["credentials"], json!([]));

    let run = sandbox.run(&platform_args(&url, &["auth", "status", "--json"]));
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["status"], "not_connected");
    // `auth claim` asked the platform once (the workspace may have been
    // claimed in the console); nothing after logout did.
    status.assert_hits(3);
}

#[test]
fn an_existing_environment_key_is_never_replaced() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let url = server.base_url();

    let run = sandbox.run_with(&connect_args(&url), &[("XYBRID_API_KEY", SECOND_KEY)]);
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(SECOND_KEY);
    let report = run.json();
    assert_eq!(report["status"], "existing");
    assert_eq!(report["key"]["source"], "env");
    // Export may still need wiring: the agent is told how, and how to verify.
    assert_eq!(next_step_kinds(&report), ["wire_init", "verify", "relay"]);
    challenge.assert_hits(0);
    assert!(!sandbox.project.join(".env").exists());

    // A key already in the project file is kept as well, and --new refuses
    // to replace it.
    std::fs::write(
        sandbox.project.join(".env"),
        format!("XYBRID_API_KEY={SECOND_KEY}\n"),
    )
    .unwrap();
    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(SECOND_KEY);
    let report = run.json();
    assert_eq!(report["status"], "existing");
    assert_eq!(report["key"]["source"], "env_file");
    assert_eq!(next_step_kinds(&report), ["wire_init", "verify", "relay"]);
    let verify = report["next_steps"][1]["summary"].as_str().unwrap();
    assert!(
        verify.contains("did not save that key"),
        "the check needs that file's key: {verify}"
    );
    let mut new = connect_args(&url);
    new.push("--new");
    let run = sandbox.run(&new);
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "env_file_has_key");
    challenge.assert_hits(0);
    let env = std::fs::read_to_string(sandbox.project.join(".env")).unwrap();
    assert_eq!(env, format!("XYBRID_API_KEY={SECOND_KEY}\n"));
}

#[test]
fn closed_signup_exits_3_without_saving_anything() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = server.mock(|when, then| {
        when.method(GET).path("/v1/agent/challenge");
        then.status(403).json_body(json!({
            "error": "agent_signup_closed",
            "message": "Agent signup is closed. Create a workspace and an API key in the console instead.",
            "reason": "disabled",
            "console_url": "https://console.xybrid.dev",
            "min_cli_version": "999.0.0",
        }));
    });
    let url = server.base_url();

    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(3), "{run}");
    challenge.assert_hits(1);
    let report = run.json();
    assert_eq!(report["status"], "signup_closed");
    assert_eq!(report["error"]["code"], "agent_signup_closed");
    let tell = report["tell_the_user"].as_str().unwrap();
    assert!(
        tell.contains("999.0.0"),
        "older CLI is told to update: {tell}"
    );
    assert!(!sandbox.project.join(".env").exists());
    assert!(
        !sandbox.credentials_path().exists(),
        "nothing to recover: no credential and no retry state"
    );
}

#[test]
fn an_unreachable_platform_exits_4() {
    require_git!();
    let sandbox = Sandbox::new();
    let port = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let url = format!("http://127.0.0.1:{port}");

    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(4), "{run}");
    let report = run.json();
    assert_eq!(report["status"], "offline");
    assert!(report["tell_the_user"]
        .as_str()
        .unwrap()
        .contains("On-device inference works"));

    let run = sandbox.run_with(
        &platform_args(&url, &["auth", "status", "--json"]),
        &[("XYBRID_API_KEY", KEY)],
    );
    assert_eq!(run.code, Some(4), "{run}");
    run.assert_no_secret(KEY);
    assert_eq!(run.json()["status"], "offline");
}

#[test]
fn ci_and_privacy_opt_outs_skip_before_any_request_or_identity() {
    require_git!();
    let sandbox = Sandbox::new();
    std::fs::remove_file(sandbox.state.join("install-id")).unwrap();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let url = server.base_url();
    let mut allow_ci = connect_args(&url);
    allow_ci.push("--allow-ci");

    for (args, env, status) in [
        (connect_args(&url), vec![("CI", "true")], "ci_skipped"),
        (
            connect_args(&url),
            vec![("GITHUB_ACTIONS", "true")],
            "ci_skipped",
        ),
        (
            allow_ci.clone(),
            vec![("CI", "true"), ("XYBRID_TELEMETRY_OPTOUT", "1")],
            "opted_out",
        ),
        (allow_ci.clone(), vec![("DO_NOT_TRACK", "1")], "opted_out"),
        (
            allow_ci.clone(),
            vec![("OTEL_SDK_DISABLED", "true")],
            "opted_out",
        ),
    ] {
        let run = sandbox.run_with(&args, &env);
        assert_eq!(run.code, Some(5), "{env:?}: {run}");
        assert_eq!(run.json()["status"], status, "{env:?}");
    }
    challenge.assert_hits(0);
    assert!(
        !sandbox.state.join("install-id").exists(),
        "no install ID is created before preflight passes"
    );
    assert!(!sandbox.credentials_path().exists());
    assert!(!sandbox.project.join(".env").exists());
}

#[test]
fn a_tracked_env_file_is_refused_with_untracked_alternatives() {
    require_git!();
    let sandbox = Sandbox::new();
    std::fs::write(sandbox.project.join(".env"), "PUBLIC_FLAG=1\n").unwrap();
    assert!(sandbox
        .git(&sandbox.project, &["add", ".env"])
        .status
        .success());
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let url = server.base_url();

    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(6), "{run}");
    let report = run.json();
    assert_eq!(report["status"], "env_file_tracked");
    assert_eq!(report["env_file"]["tracked"], true);
    assert_eq!(report["env_file"]["suggestions"], json!([".env.local"]));
    assert!(report["tell_the_user"]
        .as_str()
        .unwrap()
        .contains("--write-env .env.local"));
    challenge.assert_hits(0);
    assert_eq!(
        std::fs::read_to_string(sandbox.project.join(".env")).unwrap(),
        "PUBLIC_FLAG=1\n"
    );

    // A file that is not an assignment file (e.g. a raw-key resource) is refused.
    std::fs::create_dir(sandbox.project.join("Assets")).unwrap();
    let run = sandbox.run(&platform_args(
        &url,
        &["connect", "--write-env", "Assets/key.txt", "--json"],
    ));
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "unsupported_env_file", "{run}");
    challenge.assert_hits(0);
}

#[test]
fn an_expired_workspace_exits_7_and_stays_claimable() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let connect = mock_connect(&server);
    let url = server.base_url();
    let console = [("XYBRID_CONSOLE_URL", CONSOLE)];
    assert_eq!(
        sandbox.run_with(&connect_args(&url), &console).code,
        Some(0)
    );

    let mut status = mock_workspace(&server, KEY, workspace(true, false, "none"));
    let run = sandbox.run_with(&connect_args(&url), &console);
    assert_eq!(run.code, Some(7), "{run}");
    run.assert_no_secret(KEY);
    let report = run.json();
    assert_eq!(report["status"], "workspace_expired");
    assert_eq!(report["workspace"]["expired"], true);
    assert_eq!(
        report["claim_url"],
        format!("{CONSOLE}/claim/{CLAIM_TOKEN}")
    );
    assert!(report["tell_the_user"]
        .as_str()
        .unwrap()
        .contains("xybrid connect --new"));
    // Expiry never provisions a replacement on its own.
    challenge.assert_hits(1);
    connect.assert_hits(1);

    let run = sandbox.run(&platform_args(&url, &["auth", "status", "--json"]));
    assert_eq!(run.code, Some(7), "{run}");
    assert_eq!(run.json()["status"], "workspace_expired");
    status.delete();

    // The A3 shape on API routes: 401 workspace_expired + X-Xybrid-Workspace.
    // It is expiry, not revocation: the credential stays.
    let expired = server.mock(|when, then| {
        when.method(GET).path("/v1/agent/workspace");
        then.status(401)
            .header("X-Xybrid-Workspace", "expired")
            .json_body(json!({
                "error": "workspace_expired",
                "message": "This key expired; claim the workspace with `xybrid auth claim`",
                "key_expires_at": iso(-1),
                "claim_expires_at": iso(6),
            }));
    });
    let run = sandbox.run(&platform_args(&url, &["auth", "status", "--json"]));
    assert_eq!(run.code, Some(7), "{run}");
    assert_eq!(run.json()["status"], "workspace_expired");
    expired.assert_hits(1);
    assert_eq!(sandbox.credentials()["credentials"][0]["api_key"], KEY);
}

#[test]
fn a_rejected_saved_key_is_set_aside_but_its_claim_link_recoverable() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();
    let console = [("XYBRID_CONSOLE_URL", CONSOLE)];
    assert_eq!(
        sandbox.run_with(&connect_args(&url), &console).code,
        Some(0)
    );
    server.mock(|when, then| {
        when.method(GET).path("/v1/agent/workspace");
        then.status(401).json_body(json!({
            "error": "invalid_api_key",
            "message": "Send a valid API key as `Authorization: Bearer sk_...`",
        }));
    });

    let run = sandbox.run(&platform_args(&url, &["auth", "status", "--json"]));
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "stored_credential_rejected");
    let credentials = sandbox.credentials();
    assert_eq!(credentials["credentials"], json!([]));
    assert_eq!(credentials["retired"][0]["credential"]["api_key"], KEY);

    let run = sandbox.run_with(
        &platform_args(&url, &["auth", "claim", "--no-browser", "--json"]),
        &console,
    );
    assert_eq!(run.code, Some(0), "{run}");
    assert_eq!(
        run.json()["claim_url"],
        format!("{CONSOLE}/claim/{CLAIM_TOKEN}")
    );

    // A rejected explicit key is an error, not a quarantine.
    let run = sandbox.run_with(
        &platform_args(&url, &["auth", "status", "--json"]),
        &[("XYBRID_API_KEY", SECOND_KEY)],
    );
    assert_eq!(run.code, Some(1), "{run}");
    run.assert_no_secret(SECOND_KEY);
    assert_eq!(run.json()["error"]["code"], "invalid_api_key");
}

/// Wait until `mock` was hit `hits` times (httpmock counts a request when it
/// arrives, before a configured delay).
fn wait_for_hits(mock: &Mock<'_>, hits: usize) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while mock.hits() < hits {
        assert!(Instant::now() < deadline, "the request never arrived");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_connect_killed_mid_request_retries_with_the_same_idempotency_key() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let mut hanging = server.mock(|when, then| {
        when.method(POST).path("/v1/agent/connect");
        then.status(201)
            .delay(Duration::from_secs(40))
            .json_body(provision(KEY, Some(CLAIM_TOKEN), 7, false));
    });
    let url = server.base_url();

    // The process dies after the POST left, before any answer.
    let mut child = sandbox
        .command(&connect_args(&url))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for_hits(&hanging, 1);
    child.kill().unwrap();
    child.wait().unwrap();
    hanging.delete();

    // The retry state was saved before the POST.
    let credentials = sandbox.credentials();
    assert_eq!(credentials["credentials"], json!([]));
    let pending = &credentials["pending"][0];
    assert_eq!(pending["kind"], "connect");
    let idempotency_key = pending["idempotency_key"].as_str().unwrap().to_string();
    assert!((32..=128).contains(&idempotency_key.len()));
    assert_eq!(pending["challenge"], CHALLENGE);
    assert_eq!(pending["nonce"], CHALLENGE_NONCE);

    // The server applied it; a retry with the same key gets the workspace
    // back (fresh credentials, `replayed`), without a new challenge.
    let retry = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/agent/connect")
            .header("Idempotency-Key", idempotency_key.as_str());
        then.status(200)
            .json_body(provision(SECOND_KEY, Some(SECOND_CLAIM_TOKEN), 7, true));
    });
    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(SECOND_KEY);
    let report = run.json();
    assert_eq!(report["status"], "created");
    assert_eq!(report["workspace"]["replayed"], true);
    retry.assert_hits(1);
    challenge.assert_hits(1);
    let credentials = sandbox.credentials();
    assert_eq!(credentials["pending"], json!([]));
    assert_eq!(credentials["credentials"][0]["api_key"], SECOND_KEY);
    assert_eq!(
        credentials["credentials"][0]["claim_token"],
        SECOND_CLAIM_TOKEN
    );
    let env = std::fs::read_to_string(sandbox.project.join(".env")).unwrap();
    assert!(env.contains(SECOND_KEY) && !env.contains(KEY), "{env}");
}

#[test]
fn a_lost_response_keeps_the_retry_state_and_reuses_it() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let mut failing = server.mock(|when, then| {
        when.method(POST).path("/v1/agent/connect");
        then.status(502).body("bad gateway");
    });
    let url = server.base_url();

    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["status"], "error");
    failing.assert_hits(1);
    failing.delete();
    let pending = sandbox.credentials()["pending"][0].clone();
    let idempotency_key = pending["idempotency_key"].as_str().unwrap().to_string();

    let retry = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/agent/connect")
            .header("Idempotency-Key", idempotency_key.as_str());
        then.status(200)
            .json_body(provision(KEY, Some(CLAIM_TOKEN), 7, true));
    });
    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(0), "{run}");
    retry.assert_hits(1);
    challenge.assert_hits(1);
}

#[test]
fn another_repository_joins_the_unclaimed_workspace() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let connect = mock_connect(&server);
    let projects = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/agent/projects")
            .header_exists("Idempotency-Key")
            .json_body_partial(
                json!({"claim_token": CLAIM_TOKEN, "project_name": "mobile"}).to_string(),
            );
        let mut body = provision(PROJECT_KEY, None, 7, false);
        body["project"] = named("mobile", "mobile");
        then.status(201).json_body(body);
    });
    let url = server.base_url();
    assert_eq!(sandbox.run(&connect_args(&url)).code, Some(0));

    let mobile = sandbox.another_project("mobile");
    let run: Run = sandbox
        .command_in(&mobile, &connect_args(&url))
        .output()
        .unwrap()
        .into();
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(PROJECT_KEY);
    let report = run.json();
    assert_eq!(report["status"], "created");
    assert_eq!(
        report["workspace"]["new_project_in_existing_workspace"],
        true
    );
    assert_eq!(report["workspace"]["project"]["name"], "mobile");
    projects.assert_hits(1);
    challenge.assert_hits(1);
    connect.assert_hits(1);
    let credentials = sandbox.credentials();
    assert_eq!(credentials["credentials"].as_array().unwrap().len(), 2);
    let env = std::fs::read_to_string(mobile.join(".env")).unwrap();
    assert!(env.contains(PROJECT_KEY), "{env}");
}

#[test]
fn unpinned_hosts_are_refused_without_a_request() {
    require_git!();
    let sandbox = Sandbox::new();
    for (url, extra, code) in [
        (
            "https://platform.example.test",
            false,
            "custom_host_not_allowed",
        ),
        (
            "https://api.xybrid.dev.evil.example",
            false,
            "custom_host_not_allowed",
        ),
        ("http://api.xybrid.dev", false, "custom_host_not_allowed"),
        (
            "http://platform.example.test",
            true,
            "insecure_platform_url",
        ),
        ("https://api.xybrid.dev/v1", false, "invalid_platform_url"),
    ] {
        for command in [
            vec!["connect", "--json"],
            vec!["auth", "status", "--json"],
            vec!["auth", "claim", "--json"],
        ] {
            let mut args = vec!["--platform-url", url];
            args.extend(command.iter().copied());
            if extra {
                args.push("--allow-custom-host");
            }
            let run = sandbox.run(&args);
            assert_eq!(run.code, Some(1), "{args:?}: {run}");
            assert_eq!(run.json()["error"]["code"], code, "{args:?}");
        }
    }
    // XYBRID_PLATFORM_URL is held to the same rule as the flag.
    let run = sandbox.run_with(
        &["connect", "--json"],
        &[("XYBRID_PLATFORM_URL", "https://platform.example.test")],
    );
    assert_eq!(
        run.json()["error"]["code"],
        "custom_host_not_allowed",
        "{run}"
    );
}

#[test]
fn redirects_are_never_followed_with_credentials() {
    require_git!();
    let sandbox = Sandbox::new();
    let elsewhere = MockServer::start();
    let stolen = elsewhere.mock(|when, then| {
        when.any_request();
        then.status(200).json_body(workspace(false, false, "none"));
    });
    let server = MockServer::start();
    server.mock(|when, then| {
        when.any_request();
        then.status(307)
            .header("Location", elsewhere.url("/v1/agent/workspace"));
    });
    let url = server.base_url();

    let run = sandbox.run_with(
        &platform_args(&url, &["auth", "status", "--json"]),
        &[("XYBRID_API_KEY", KEY)],
    );
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "redirect_refused");
    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "redirect_refused");
    stolen.assert_hits(0);
}

#[test]
fn connect_and_auth_never_start_otlp_export() {
    require_git!();
    let sandbox = Sandbox::new();
    let collector = MockServer::start();
    let exported = collector.mock(|when, then| {
        when.any_request();
        then.status(200);
    });
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    mock_workspace(&server, KEY, workspace(false, false, "none"));
    let url = server.base_url();
    let otlp = [
        ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.base_url()),
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json".to_string()),
    ];
    let otlp: Vec<(&str, &str)> = otlp.iter().map(|(k, v)| (*k, v.as_str())).collect();

    let mut observed = connect_args(&url);
    observed.insert(0, "--observability");
    for args in [
        connect_args(&url),
        observed,
        platform_args(&url, &["auth", "status", "--json"]),
        platform_args(&url, &["auth", "status"]),
        platform_args(&url, &["auth", "logout"]),
    ] {
        let run = sandbox.run_with(&args, &otlp);
        assert!(
            !run.stderr.contains("Observability enabled")
                && !run.stdout.contains("Observability enabled"),
            "{args:?}: {run}"
        );
        run.assert_no_secret(KEY);
    }
    exported.assert_hits(0);

    // Any other command does start the configured exporter, and says so on
    // stderr only.
    let run = sandbox.run_with(&["telemetry", "status"], &otlp);
    assert_eq!(run.code, Some(0), "{run}");
    assert!(run.stderr.contains("Observability enabled"), "{run}");
    assert!(!run.stdout.contains("Observability enabled"), "{run}");
    let quiet = sandbox.run_with(&["--quiet", "telemetry", "status"], &otlp);
    assert!(!quiet.stderr.contains("Observability enabled"), "{quiet}");
}

#[test]
fn observability_resolves_the_saved_key_and_requires_one() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let url = server.base_url();

    // No key anywhere: an actionable error, not a silent success.
    let run = sandbox.run(&["--observability", "telemetry", "status"]);
    assert_eq!(run.code, Some(1), "{run}");
    assert!(run.stderr.contains("xybrid connect"), "{run}");

    mock_challenge(&server);
    mock_connect(&server);
    assert_eq!(sandbox.run(&connect_args(&url)).code, Some(0));

    let traces = server.url("/v1/traces");
    let run = sandbox.run(&[
        "--platform-url",
        &url,
        "--observability",
        "--traces-endpoint",
        &traces,
        "telemetry",
        "status",
    ]);
    assert_eq!(run.code, Some(0), "the saved key is enough: {run}");
    assert!(run.stderr.contains("Observability enabled"), "{run}");
    run.assert_no_secret(KEY);

    // ...but it never leaves its platform.
    let run = sandbox.run(&[
        "--platform-url",
        &url,
        "--observability",
        "--traces-endpoint",
        "https://collector.example.test/v1/traces",
        "telemetry",
        "status",
    ]);
    assert_eq!(run.code, Some(1), "{run}");
    assert!(run.stderr.contains("Refusing"), "{run}");
    run.assert_no_secret(KEY);
}

#[test]
fn the_claim_reminder_is_daily_and_quiet_where_it_should_be() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();
    assert_eq!(sandbox.run(&connect_args(&url)).code, Some(0));
    let args = ["--platform-url", url.as_str(), "telemetry", "status"];

    for env in [vec![("XYBRID_QUIET", "1")], vec![("CI", "true")]] {
        let run = sandbox.run_with(&args, &env);
        assert!(!run.stderr.contains("xybrid auth claim"), "{env:?}: {run}");
    }
    let mut quiet = vec!["--quiet"];
    quiet.extend_from_slice(&args);
    assert!(!sandbox.run(&quiet).stderr.contains("xybrid auth claim"));

    let run = sandbox.run(&args);
    assert!(run.stderr.contains("xybrid auth claim"), "{run}");
    assert!(
        !run.stdout.contains("xybrid auth claim"),
        "stderr only: {run}"
    );
    run.assert_no_secret(KEY);
    assert!(!run.stderr.contains(CLAIM_TOKEN), "{run}");
    let again = sandbox.run(&args);
    assert!(
        !again.stderr.contains("xybrid auth claim"),
        "once a day: {again}"
    );
}

#[test]
fn json_mode_prints_no_hints_even_with_otlp_configured() {
    let sandbox = Sandbox::new_without_git();
    let empty = sandbox.project.join("no-model");
    std::fs::create_dir_all(&empty).unwrap();
    let empty = empty.to_str().unwrap();
    let otlp = [("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")];

    // `init --json` reports this failure as a JSON object on stderr; nothing
    // else may share either stream.
    let run = sandbox.run_with(&["init", "--json", "--yes", empty], &otlp);
    assert_eq!(run.code, Some(1), "{run}");
    assert!(run.stdout.trim().is_empty(), "{run}");
    let error: Value = serde_json::from_str(run.stderr.trim())
        .unwrap_or_else(|_| panic!("stderr is only the JSON error: {run}"));
    assert_eq!(error["status"], "error");

    let run = sandbox.run_with(&["init", "--yes", empty], &otlp);
    assert!(run.stderr.contains("Observability enabled"), "{run}");
    assert!(!run.stdout.contains("Observability enabled"), "{run}");

    // An OTLP endpoint the exporter rejects: in JSON mode its warning must not
    // land on either stream; otherwise it goes to stderr, never stdout.
    let broken = [("OTEL_EXPORTER_OTLP_ENDPOINT", "not a url")];
    let run = sandbox.run_with(&["init", "--json", "--yes", empty], &broken);
    assert_eq!(run.code, Some(1), "{run}");
    assert!(run.stdout.trim().is_empty(), "{run}");
    let error: Value = serde_json::from_str(run.stderr.trim())
        .unwrap_or_else(|_| panic!("stderr is only the JSON error: {run}"));
    assert_eq!(error["status"], "error");

    let run = sandbox.run_with(&["init", "--yes", empty], &broken);
    assert!(
        run.stderr.contains("Observability was not started"),
        "{run}"
    );
    assert!(!run.stdout.contains("Observability"), "{run}");
}

#[test]
fn outside_a_repository_the_file_is_honestly_not_gitignored() {
    let sandbox = Sandbox::new_without_git();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();

    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(0), "{run}");
    let report = run.json();
    assert_eq!(report["status"], "created");
    assert_eq!(report["env_file"]["written"], true);
    assert_eq!(report["env_file"]["gitignored"], false);
    assert_eq!(report["env_file"]["tracked"], false);
    assert!(
        report["tell_the_user"]
            .as_str()
            .unwrap()
            .contains("not in a Git repository"),
        "{run}"
    );
    assert!(!sandbox.project.join(".gitignore").exists());
}

#[test]
fn a_file_in_another_repository_is_refused_not_written() {
    require_git!();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let url = server.base_url();
    let write_env = |sandbox: &Sandbox, target: &str| -> Run {
        sandbox.run(&platform_args(
            &url,
            &["connect", "--write-env", target, "--json"],
        ))
    };

    // From a directory that is not a repository into an app's repository,
    // whose .env Git tracks.
    let sandbox = Sandbox::new_without_git();
    let app = sandbox.project.join("app");
    std::fs::create_dir_all(&app).unwrap();
    sandbox.git(&app, &["init", "-q"]);
    std::fs::write(app.join(".env"), "PUBLIC_FLAG=1\n").unwrap();
    assert!(sandbox.git(&app, &["add", ".env"]).status.success());
    let run = write_env(&sandbox, "app/.env");
    assert_eq!(run.code, Some(1), "{run}");
    let report = run.json();
    assert_eq!(report["error"]["code"], "env_file_outside_repository");
    let message = report["error"]["message"].as_str().unwrap();
    assert!(
        message.contains(&format!("Run `xybrid connect` from {}", app.display())),
        "{message}"
    );
    assert_eq!(
        std::fs::read_to_string(app.join(".env")).unwrap(),
        "PUBLIC_FLAG=1\n"
    );
    assert!(!app.join(".gitignore").exists());
    assert!(!sandbox.credentials_path().exists());

    // From one repository into an independent repository nested in it: the
    // outer repository's checks and .gitignore say nothing about that file.
    let sandbox = Sandbox::new();
    let nested = sandbox.project.join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    sandbox.git(&nested, &["init", "-q"]);
    std::fs::write(nested.join(".env"), "PUBLIC_FLAG=1\n").unwrap();
    assert!(sandbox.git(&nested, &["add", ".env"]).status.success());
    let run = write_env(&sandbox, "nested/.env");
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "env_file_outside_repository");
    assert_eq!(
        std::fs::read_to_string(nested.join(".env")).unwrap(),
        "PUBLIC_FLAG=1\n"
    );
    assert!(
        !sandbox.project.join(".gitignore").exists(),
        "the outer repository is left alone"
    );
    challenge.assert_hits(0);
}

#[test]
fn only_the_platforms_own_401_sets_a_saved_key_aside() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();
    assert_eq!(sandbox.run(&connect_args(&url)).code, Some(0));

    // A proxy or load balancer in front of the platform answers 401: that
    // says nothing about the key.
    let proxy = server.mock(|when, then| {
        when.method(GET).path("/v1/agent/workspace");
        then.status(401)
            .header("Content-Type", "text/html")
            .body("<html><body>401 Authorization Required</body></html>");
    });
    let run = sandbox.run(&platform_args(&url, &["auth", "status", "--json"]));
    assert_eq!(run.code, Some(1), "{run}");
    run.assert_no_secret(KEY);
    let report = run.json();
    assert_eq!(report["status"], "error");
    assert_ne!(report["error"]["code"], "stored_credential_rejected");
    assert_eq!(report["verification"]["state"], "unavailable");

    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(0), "{run}");
    let report = run.json();
    assert_eq!(report["status"], "existing");
    assert_eq!(report["verification"]["state"], "unavailable");
    proxy.assert_hits(2);
    challenge.assert_hits(1);
    let credentials = sandbox.credentials();
    assert_eq!(credentials["credentials"][0]["api_key"], KEY);
    assert_eq!(credentials["retired"], json!([]));
}

#[test]
fn a_claim_made_in_the_console_reaches_every_repository_of_the_workspace() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    server.mock(|when, then| {
        when.method(POST).path("/v1/agent/projects");
        let mut body = provision(PROJECT_KEY, None, 7, false);
        body["project"] = named("mobile", "mobile");
        then.status(201).json_body(body);
    });
    let url = server.base_url();
    let console = [("XYBRID_CONSOLE_URL", CONSOLE)];
    assert_eq!(
        sandbox.run_with(&connect_args(&url), &console).code,
        Some(0)
    );
    let mobile = sandbox.another_project("mobile");
    let run: Run = sandbox
        .command_in(&mobile, &connect_args(&url))
        .envs(console)
        .output()
        .unwrap()
        .into();
    assert_eq!(run.code, Some(0), "{run}");

    // The user claimed it in the console; only the platform knows so far.
    let claimed = mock_workspace(&server, KEY, workspace(false, true, "none"));
    let run = sandbox.run_with(
        &platform_args(&url, &["auth", "claim", "--no-browser", "--json"]),
        &console,
    );
    assert_eq!(run.code, Some(0), "{run}");
    let report = run.json();
    assert_eq!(report["status"], "claimed");
    assert_eq!(report["claim_url"], Value::Null);
    assert_eq!(report["workspace"]["lifecycle_source"], "server");
    claimed.assert_hits(1);
    let claimed_flags: Vec<bool> = sandbox.credentials()["credentials"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["claimed"].as_bool().unwrap())
        .collect();
    assert_eq!(claimed_flags, [true, true], "both repositories");

    // No claim reminder in either repository: not just not today, none due.
    for dir in [&sandbox.project, &mobile] {
        let run: Run = sandbox
            .command_in(dir, &["--platform-url", &url, "telemetry", "status"])
            .output()
            .unwrap()
            .into();
        assert!(!run.stderr.contains("xybrid auth claim"), "{run}");
    }
    assert!(!sandbox.state.join("notices.json").exists());
}

#[test]
fn the_claim_reminder_follows_the_saved_key_from_the_environment() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();
    assert_eq!(sandbox.run(&connect_args(&url)).code, Some(0));
    let args = ["--platform-url", url.as_str(), "telemetry", "status"];

    // Another key is in use: the saved workspace's deadlines are not its own.
    let run = sandbox.run_with(&args, &[("XYBRID_API_KEY", SECOND_KEY)]);
    assert!(!run.stderr.contains("xybrid auth claim"), "{run}");

    // The saved key exported from the project's .env (direnv, dotenv-cli).
    let run = sandbox.run_with(&args, &[("XYBRID_API_KEY", KEY)]);
    assert!(run.stderr.contains("xybrid auth claim"), "{run}");
    assert!(run.stderr.contains("when last checked"), "{run}");
    run.assert_no_secret(KEY);
}

/// The output of `id <flag>` as a number.
#[cfg(unix)]
fn id(flag: &str) -> u32 {
    let output = Command::new("id").arg(flag).output().expect("id runs");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("numeric id")
}

/// Without XYBRID_CREDENTIALS_FILE, in a ~/.xybrid an earlier `xybrid run`
/// already created with the umask (the SDK's model cache lives there).
#[cfg(unix)]
#[test]
fn the_default_store_works_in_an_existing_private_group_directory() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    mock_connect(&server);
    mock_workspace(&server, KEY, workspace(false, false, "none"));
    let url = server.base_url();
    let xybrid = sandbox.home.join(".xybrid");
    std::fs::create_dir_all(xybrid.join("cache").join("models")).unwrap();
    std::fs::write(xybrid.join("install-id"), format!("{INSTALL_ID}\n")).unwrap();
    let chmod = |mode: u32| {
        std::fs::set_permissions(&xybrid, std::fs::Permissions::from_mode(mode)).unwrap();
    };
    let connect = || -> Run {
        let mut command = sandbox.command(&connect_args(&url));
        command.env_remove("XYBRID_CREDENTIALS_FILE");
        command.output().unwrap().into()
    };
    let store = xybrid.join("credentials.json");

    // Writable by every account: refused before any request, with the fix.
    chmod(0o777);
    let run = connect();
    assert_eq!(run.code, Some(1), "{run}");
    let report = run.json();
    assert_eq!(report["error"]["code"], "credentials_unsafe");
    let message = report["error"]["message"].as_str().unwrap();
    assert!(
        message.contains(&format!("chmod go-w {}", xybrid.display())),
        "{message}"
    );
    challenge.assert_hits(0);

    // Group-writable, as a 002 umask leaves it: accepted when the group is
    // the user's own private group, else refused with the same fix.
    chmod(0o775);
    let gid = std::fs::metadata(&xybrid).unwrap().gid();
    let private_group = gid == id("-g") && id("-g") == id("-u");
    let run = connect();
    if private_group {
        assert_eq!(run.code, Some(0), "{run}");
        let report = run.json();
        assert_eq!(report["status"], "created");
        assert_eq!(report["credentials"]["path"], store.display().to_string());
        let mode = std::fs::metadata(&store).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        run.assert_no_secret(KEY);
    } else {
        assert_eq!(run.code, Some(1), "{run}");
        let report = run.json();
        assert_eq!(report["error"]["code"], "credentials_unsafe");
        let message = report["error"]["message"].as_str().unwrap();
        assert!(message.contains("chmod go-w"), "{message}");
        challenge.assert_hits(0);
    }

    // Not group-writable: always accepted (the saved key is reused if any).
    chmod(0o755);
    let run = connect();
    assert_eq!(run.code, Some(0), "{run}");
    let expected = if private_group { "existing" } else { "created" };
    assert_eq!(run.json()["status"], expected, "{run}");
    challenge.assert_hits(1);
    assert!(store.exists());
}

#[test]
fn another_repositorys_expired_workspace_is_reported_without_its_facts() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let connect = mock_connect(&server);
    let projects = server.mock(|when, then| {
        when.method(POST).path("/v1/agent/projects");
        then.status(403).json_body(json!({
            "error": "workspace_expired",
            "message": "This workspace's keys have expired. Claim it to keep using it.",
            "claim_expires_at": iso(6),
        }));
    });
    let url = server.base_url();
    let console = [("XYBRID_CONSOLE_URL", CONSOLE)];
    assert_eq!(
        sandbox.run_with(&connect_args(&url), &console).code,
        Some(0)
    );

    let mobile = sandbox.another_project("mobile");
    let run: Run = sandbox
        .command_in(&mobile, &connect_args(&url))
        .envs(console)
        .output()
        .unwrap()
        .into();
    assert_eq!(run.code, Some(7), "{run}");
    run.assert_no_secret(KEY);
    let report = run.json();
    assert_eq!(report["status"], "workspace_expired");
    assert_eq!(report["key"], Value::Null, "the other repository's key");
    assert_eq!(report["credentials"]["saved"], false);
    assert_eq!(report["credentials"]["recovery"], false);
    assert_eq!(report["workspace"]["project"], Value::Null);
    let tell = report["tell_the_user"].as_str().unwrap();
    assert!(
        tell.contains(&format!(
            "`xybrid auth claim` in {}",
            sandbox.project.display()
        )),
        "{tell}"
    );
    projects.assert_hits(1);
    challenge.assert_hits(1);
    connect.assert_hits(1);
    let roots: Vec<Value> = sandbox.credentials()["credentials"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["repository_root"].clone())
        .collect();
    assert_eq!(roots, [json!(sandbox.project.display().to_string())]);
}

#[test]
fn a_project_file_that_fails_after_provisioning_still_reports_the_workspace() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let slow = server.mock(|when, then| {
        when.method(POST).path("/v1/agent/connect");
        then.status(201)
            .delay(Duration::from_secs(2))
            .json_body(provision(KEY, Some(CLAIM_TOKEN), 7, false));
    });
    let url = server.base_url();

    let child = sandbox
        .command(&connect_args(&url))
        .env("XYBRID_CONSOLE_URL", CONSOLE)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    wait_for_hits(&slow, 1);
    // While the workspace is being created, .env becomes unwritable.
    std::fs::create_dir(sandbox.project.join(".env")).unwrap();
    let run: Run = child.wait_with_output().unwrap().into();
    assert_eq!(run.code, Some(1), "{run}");
    run.assert_no_secret(KEY);
    let report = run.json();
    assert_eq!(report["status"], "error");
    assert_eq!(report["error"]["code"], "env_file_unwritable");
    assert_eq!(report["credentials"]["saved"], true);
    let link = format!("{CONSOLE}/claim/{CLAIM_TOKEN}");
    assert_eq!(report["claim_url"], link);
    let tell = report["tell_the_user"].as_str().unwrap();
    assert!(
        tell.contains("Created the provisional Xybrid workspace"),
        "{tell}"
    );
    assert!(tell.contains(&link), "{tell}");
    assert!(tell.contains("reuses the saved key"), "{tell}");

    // Once fixed, the next run reuses the saved key: no second workspace.
    std::fs::remove_dir(sandbox.project.join(".env")).unwrap();
    let status = mock_workspace(&server, KEY, workspace(false, false, "none"));
    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(0), "{run}");
    let report = run.json();
    assert_eq!(report["status"], "existing");
    assert_eq!(report["env_file"]["written"], true);
    status.assert_hits(1);
    slow.assert_hits(1);
    challenge.assert_hits(1);
    let env = std::fs::read_to_string(sandbox.project.join(".env")).unwrap();
    assert!(env.contains(&format!("XYBRID_API_KEY={KEY}")), "{env}");
}

#[test]
fn a_credentials_store_inside_the_repository_is_kept_out_of_git() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();

    // Directly in the repository root: refused before anything is created.
    let at_root = sandbox.project.join("credentials.json");
    let run = sandbox.run_with(
        &connect_args(&url),
        &[("XYBRID_CREDENTIALS_FILE", at_root.to_str().unwrap())],
    );
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "credentials_in_repository");
    challenge.assert_hits(0);
    assert!(!at_root.exists());

    // In a directory of its own: that whole directory is ignored, since the
    // lock, install ID, notice dates and any corrupt copy land there too.
    let state = sandbox.project.join(".xybrid");
    create_private_dir(&state);
    std::fs::write(state.join("install-id"), format!("{INSTALL_ID}\n")).unwrap();
    let store = state.join("credentials.json");
    let run = sandbox.run_with(
        &connect_args(&url),
        &[("XYBRID_CREDENTIALS_FILE", store.to_str().unwrap())],
    );
    assert_eq!(run.code, Some(0), "{run}");
    assert!(store.exists());
    let gitignore = std::fs::read_to_string(sandbox.project.join(".gitignore")).unwrap();
    assert!(gitignore.lines().any(|l| l == "/.xybrid"), "{gitignore}");
    std::fs::write(
        state.join("credentials.json.corrupt-20260101T000000.000Z"),
        "{}",
    )
    .unwrap();
    let status = sandbox.git(
        &sandbox.project,
        &["status", "--porcelain", "--untracked-files=all"],
    );
    let listing = String::from_utf8_lossy(&status.stdout);
    assert!(!listing.contains(".xybrid/"), "{listing}");
    assert!(listing.contains(".gitignore"), "{listing}");
}

#[test]
fn after_the_claim_deadline_nothing_offers_a_claim() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();
    let console = [("XYBRID_CONSOLE_URL", CONSOLE)];
    assert_eq!(
        sandbox.run_with(&connect_args(&url), &console).code,
        Some(0)
    );
    // Expired, and the claim period is over too (purge pending).
    let mut body = workspace(true, false, "none");
    body["key_expires_at"] = json!(iso(-8));
    body["claim_expires_at"] = json!(iso(-1));
    mock_workspace(&server, KEY, body);

    for args in [
        connect_args(&url),
        platform_args(&url, &["auth", "status", "--json", "--show-claim-url"]),
    ] {
        let run = sandbox.run_with(&args, &console);
        assert_eq!(run.code, Some(1), "{args:?}: {run}");
        run.assert_no_secret(KEY);
        let report = run.json();
        assert_eq!(report["status"], "error", "{args:?}");
        assert_eq!(report["error"]["code"], "claim_expired", "{args:?}");
        assert_eq!(report["claim_url"], Value::Null, "{args:?}");
        let tell = report["tell_the_user"].as_str().unwrap();
        assert!(tell.contains("claim period ended"), "{tell}");
        assert!(!tell.contains("xybrid auth claim"), "{tell}");
        assert!(!tell.contains(CLAIM_TOKEN), "{tell}");
        assert!(tell.contains("xybrid connect --new"), "{tell}");
    }
    let run = sandbox.run(&platform_args(
        &url,
        &["auth", "claim", "--no-browser", "--json"],
    ));
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.json()["error"]["code"], "claim_expired");
}

/// `auth status` attaches saved details (the claim link, a refreshed
/// lifecycle) only to the key they belong to, however that key arrived.
#[test]
fn auth_status_uses_saved_details_only_for_their_own_key() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    mock_challenge(&server);
    mock_connect(&server);
    let url = server.base_url();
    let console = ("XYBRID_CONSOLE_URL", CONSOLE);
    assert_eq!(
        sandbox.run_with(&connect_args(&url), &[console]).code,
        Some(0)
    );
    let args = platform_args(&url, &["auth", "status", "--json", "--show-claim-url"]);
    let link = format!("{CONSOLE}/claim/{CLAIM_TOKEN}");

    // Another workspace's key: the saved claim link and deadlines are not its own.
    let other = mock_workspace(&server, SECOND_KEY, workspace(false, false, "none"));
    let run = sandbox.run_with(&args, &[console, ("XYBRID_API_KEY", SECOND_KEY)]);
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(SECOND_KEY);
    assert!(!run.stdout.contains(CLAIM_TOKEN), "{run}");
    let report = run.json();
    assert_eq!(report["key"]["source"], "env");
    assert_eq!(report["claim_url"], Value::Null);
    assert_eq!(report["stored"], Value::Null);
    other.assert_hits(1);

    // The saved key exported into the environment is that credential.
    let mut unclaimed = mock_workspace(&server, KEY, workspace(false, false, "none"));
    let run = sandbox.run_with(&args, &[console, ("XYBRID_API_KEY", KEY)]);
    assert_eq!(run.code, Some(0), "{run}");
    run.assert_no_secret(KEY);
    let report = run.json();
    assert_eq!(report["key"]["source"], "env");
    assert_eq!(report["claim_url"], link);
    assert!(report["stored"]["key_expires_at"].is_string(), "{run}");
    unclaimed.assert_hits(1);
    unclaimed.delete();

    // A claim made in the console is saved through it as well.
    let claimed = mock_workspace(&server, KEY, workspace(false, true, "none"));
    let run = sandbox.run_with(&args, &[console, ("XYBRID_API_KEY", KEY)]);
    assert_eq!(run.code, Some(0), "{run}");
    let report = run.json();
    assert_eq!(report["claim_url"], Value::Null, "claimed: {run}");
    claimed.assert_hits(1);
    assert_eq!(sandbox.credentials()["credentials"][0]["claimed"], true);
}

/// Several empty assignments of the variable leave no single line to fill
/// (the app reads one of them, depending on its loader): refused before
/// anything is created or changed.
#[test]
fn a_variable_assigned_on_several_empty_lines_is_refused_up_front() {
    require_git!();
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let challenge = mock_challenge(&server);
    let url = server.base_url();
    let text = "XYBRID_API_KEY=\nOTHER=1\nXYBRID_API_KEY=\n";
    std::fs::write(sandbox.project.join(".env"), text).unwrap();

    let run = sandbox.run(&connect_args(&url));
    assert_eq!(run.code, Some(1), "{run}");
    let report = run.json();
    assert_eq!(report["status"], "error");
    assert_eq!(report["error"]["code"], "env_file_ambiguous");
    let message = report["error"]["message"].as_str().unwrap();
    assert!(message.contains(".env"), "{message}");
    challenge.assert_hits(0);
    assert!(!sandbox.credentials_path().exists());
    assert!(!sandbox.project.join(".gitignore").exists());
    assert_eq!(
        std::fs::read_to_string(sandbox.project.join(".env")).unwrap(),
        text
    );
}

/// Write `contents` to a new file only this account can read.
fn write_private_file(path: &Path, contents: &str) {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .and_then(|mut file| file.write_all(contents.as_bytes()))
        .unwrap();
}

/// A store that can't be trusted or read keeps local commands working, but
/// they say on stderr that its saved key is ignored.
#[test]
fn an_unreadable_store_is_reported_not_silently_ignored() {
    let sandbox = Sandbox::new_without_git();
    write_private_file(&sandbox.credentials_path(), "{ not json");
    const WARNING: &str = "Ignoring saved Xybrid credentials";

    let run = sandbox.run(&["telemetry", "status"]);
    assert_eq!(run.code, Some(0), "{run}");
    assert_eq!(run.stderr.matches(WARNING).count(), 1, "{run}");
    assert!(!run.stdout.contains(WARNING), "stderr only: {run}");

    let run = sandbox.run(&["--quiet", "telemetry", "status"]);
    assert_eq!(run.code, Some(0), "{run}");
    assert!(!run.stderr.contains(WARNING), "{run}");

    // `--observability` fails without the key and says why, once.
    let run = sandbox.run(&["--observability", "telemetry", "status"]);
    assert_eq!(run.code, Some(1), "{run}");
    assert_eq!(run.stderr.matches(WARNING).count(), 1, "{run}");

    // JSON mode: stderr holds only the JSON error.
    let empty = sandbox.project.join("no-model");
    std::fs::create_dir_all(&empty).unwrap();
    let run = sandbox.run(&["init", "--json", "--yes", empty.to_str().unwrap()]);
    assert_eq!(run.code, Some(1), "{run}");
    let error: Value = serde_json::from_str(run.stderr.trim())
        .unwrap_or_else(|_| panic!("stderr is only the JSON error: {run}"));
    assert_eq!(error["status"], "error");
}

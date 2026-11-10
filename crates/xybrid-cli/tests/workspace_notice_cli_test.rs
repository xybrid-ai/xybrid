//! What `xybrid run` says about the Xybrid workspace behind its trace export.
//!
//! Xybrid's trace ingest answers an accepted export from a provisional key
//! with `X-Xybrid-Workspace: provisional; expires_at=…`, and refuses an
//! expired key with HTTP 401, `X-Xybrid-Workspace: expired` and a
//! `google.rpc.Status{code: 16, message: "workspace_expired"}` body. Each
//! case runs the real binary against a loopback `httpmock` ingest, in its own
//! HOME and credentials file, with CI markers, quiet switches, keys and OTLP
//! variables removed unless the case sets them.
//!
//! The run is `run --bundle <missing>`: it fails at once, offline, after one
//! workflow span, so the only export is the final drain at exit. Every
//! notice here is therefore first received during that drain.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use httpmock::prelude::*;
use httpmock::Mock;
use serde_json::{json, Value};

const KEY: &str = concat!("sk_live", "_WSnoticeTESTkey0123456789abcdefXY");
const OTHER_KEY: &str = concat!("sk_live", "_WSotherTESTkey0123456789abcdefWX");
const CLAIM_TOKEN: &str = "WorkspaceNoticeClaimTokenWorkspaceNotice12";
/// `Status { code: 16, message: "workspace_expired" }` in protobuf.
const EXPIRED_PROTOBUF: &[u8] = b"\x08\x10\x12\x11workspace_expired";

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
];

fn deadline(days: i64) -> DateTime<Utc> {
    let at = Utc::now() + chrono::Duration::days(days);
    DateTime::parse_from_rfc3339(&at.to_rfc3339_opts(SecondsFormat::Secs, true))
        .unwrap()
        .with_timezone(&Utc)
}

fn iso(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// A deadline as the CLI's notices print it (local date).
fn shown(at: DateTime<Utc>) -> String {
    at.with_timezone(&chrono::Local)
        .format("%a, %b %-d %Y")
        .to_string()
}

/// An ingest that accepts `key`'s traces as a provisional workspace's. The
/// answer is slowed down so it lands while the CLI is shutting down.
fn provisional_ingest<'a>(server: &'a MockServer, key: &str, until: DateTime<Utc>) -> Mock<'a> {
    server.mock(|when, then| {
        when.method(POST)
            .path("/v1/traces")
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/x-protobuf");
        then.status(200)
            .header(
                "X-Xybrid-Workspace",
                format!("provisional; expires_at={}", iso(until)),
            )
            .delay(Duration::from_millis(500));
    })
}

/// An ingest that refuses an expired workspace's key, as the platform does.
fn expired_ingest(server: &MockServer) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(401)
            .header("X-Xybrid-Workspace", "expired")
            .header("content-type", "application/x-protobuf")
            .body(EXPIRED_PROTOBUF)
            .delay(Duration::from_millis(300));
    })
}

/// The finished process.
struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl Run {
    fn assert_no_secret(&self) {
        for secret in [KEY, OTHER_KEY, CLAIM_TOKEN] {
            assert!(!self.stdout.contains(secret), "secret on stdout:\n{self}");
            assert!(!self.stderr.contains(secret), "secret on stderr:\n{self}");
        }
    }

    fn claim_nudges(&self) -> usize {
        self.stderr.matches("xybrid auth claim").count()
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

/// An isolated user: home, private state, and a project directory (not a
/// Git repository: its directory is the project root).
struct Sandbox {
    _dir: tempfile::TempDir,
    home: PathBuf,
    state: PathBuf,
    project: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        let base = dir.path().canonicalize().expect("canonical temp dir");
        let home = base.join("home");
        let project = base.join("my-app");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        let state = base.join("state");
        create_private_dir(&state);
        Self {
            _dir: dir,
            home,
            state,
            project,
        }
    }

    fn credentials_path(&self) -> PathBuf {
        self.state.join("credentials.json")
    }

    /// Save `connect`'s private credential for `origin` and this project.
    fn save_credential(&self, origin: &str, key_expires_at: DateTime<Utc>) {
        let store = json!({
            "version": 1,
            "credentials": [{
                "platform_origin": origin,
                "repository_root": self.project,
                "api_key": KEY,
                "scope": "client",
                "organization": {"slug": "my-app", "name": "my-app"},
                "project": {"slug": "default", "name": "Default"},
                "key_expires_at": iso(key_expires_at),
                "claim_expires_at": iso(key_expires_at + chrono::Duration::days(7)),
                "claim_token": CLAIM_TOKEN,
                "claimed": false,
                "created_at": iso(Utc::now()),
            }],
        });
        write_private(
            &self.credentials_path(),
            &serde_json::to_vec_pretty(&store).unwrap(),
        );
    }

    fn saved_key_deadline(&self) -> DateTime<Utc> {
        let text = std::fs::read_to_string(self.credentials_path()).unwrap();
        let store: Value = serde_json::from_str(&text).unwrap();
        let raw = store["credentials"][0]["key_expires_at"]
            .as_str()
            .expect("a saved key deadline")
            .to_string();
        DateTime::parse_from_rfc3339(&raw)
            .unwrap()
            .with_timezone(&Utc)
    }

    fn missing_bundle(&self) -> String {
        self.project
            .join("missing.xyb")
            .to_str()
            .unwrap()
            .to_string()
    }

    fn run_with(&self, args: &[&str], env: &[(&str, &str)]) -> Run {
        let mut command = Command::new(xybrid_bin());
        command.args(args).current_dir(&self.project);
        for name in SCRUBBED {
            command.env_remove(name);
        }
        command
            .env("HOME", &self.home)
            .env("USERPROFILE", &self.home)
            .env("XDG_CONFIG_HOME", self.home.join(".config"))
            .env("XYBRID_CREDENTIALS_FILE", self.credentials_path())
            .env("GIT_CEILING_DIRECTORIES", self.project.parent().unwrap())
            .env("NO_COLOR", "1")
            .envs(env.iter().copied())
            .stdin(Stdio::null());
        command.output().expect("xybrid runs").into()
    }

    /// `xybrid --api-key <key> --observability --traces-endpoint <ingest>
    /// [extra…] run --bundle <missing> --input-text hi`: an export through
    /// the Xybrid preset, on the SDK's default session.
    fn observed_run(&self, ingest: &str, key: &str, extra: &[&str], env: &[(&str, &str)]) -> Run {
        let bundle = self.missing_bundle();
        let mut args = vec![
            "--api-key",
            key,
            "--observability",
            "--traces-endpoint",
            ingest,
        ];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["run", "--bundle", &bundle, "--input-text", "hi"]);
        self.run_with(&args, env)
    }
}

/// The CLI binary as an absolute path (Bazel sets a runfiles-relative one).
fn xybrid_bin() -> &'static Path {
    static BIN: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    BIN.get_or_init(|| {
        let path = PathBuf::from(env!("CARGO_BIN_EXE_xybrid"));
        if path.is_absolute() {
            path
        } else {
            std::env::current_dir()
                .expect("the test's working directory")
                .join(path)
        }
    })
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

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .unwrap();
    file.write_all(bytes).unwrap();
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn a_provisional_notice_from_the_final_drain_is_reported_once_a_day() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let until = deadline(5);
    let ingest = provisional_ingest(&server, KEY, until);
    let traces = server.url("/v1/traces");

    let run = sandbox.observed_run(&traces, KEY, &[], &[]);
    assert_eq!(run.code, Some(1), "the bundle is missing: {run}");
    ingest.assert_hits(1);
    assert!(run.stderr.contains("unclaimed workspace"), "{run}");
    assert_eq!(run.claim_nudges(), 1, "{run}");
    assert!(
        !run.stdout.contains("xybrid auth claim"),
        "stderr only: {run}"
    );
    run.assert_no_secret();

    // The same day, the claim nudge is not repeated.
    let again = sandbox.observed_run(&traces, KEY, &[], &[]);
    ingest.assert_hits(2);
    assert!(!again.stderr.contains("unclaimed workspace"), "{again}");
    assert_eq!(again.claim_nudges(), 0, "{again}");
}

#[test]
fn a_local_trace_export_reports_the_notice_too() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let ingest = provisional_ingest(&server, KEY, deadline(5));
    let trace_file = sandbox.project.join("trace.json");
    let trace_file = trace_file.to_str().unwrap();
    let bundle = sandbox.missing_bundle();
    let traces = server.url("/v1/traces");
    // `--trace-export` builds the CLI's own pipeline (local capture plus
    // export) instead of the SDK's default session.
    let run = sandbox.run_with(
        &[
            "--api-key",
            KEY,
            "--observability",
            "--traces-endpoint",
            &traces,
            "run",
            "--bundle",
            &bundle,
            "--input-text",
            "hi",
            "--trace-export",
            trace_file,
        ],
        &[],
    );
    assert_eq!(run.code, Some(1), "{run}");
    ingest.assert_hits(1);
    assert!(Path::new(trace_file).exists(), "{run}");
    assert!(run.stderr.contains("unclaimed workspace"), "{run}");
    assert_eq!(run.claim_nudges(), 1, "{run}");
    run.assert_no_secret();
}

#[test]
fn an_expired_key_is_reported_on_every_run_and_points_at_claim() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let ingest = expired_ingest(&server);
    let traces = server.url("/v1/traces");
    for attempt in 1..=2 {
        let run = sandbox.observed_run(&traces, KEY, &[], &[]);
        assert_eq!(run.code, Some(1), "{run}");
        ingest.assert_hits(attempt);
        assert!(run.stderr.contains("key expired"), "{run}");
        assert!(run.stderr.contains("xybrid auth claim"), "{run}");
        assert!(
            run.stderr.contains("local inference is unaffected"),
            "{run}"
        );
        run.assert_no_secret();
    }
}

#[test]
fn quiet_ci_and_xybrid_quiet_suppress_the_notice_not_the_export() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let ingest = expired_ingest(&server);
    let traces = server.url("/v1/traces");
    /// Extra global flags, extra environment.
    type Case = (
        &'static [&'static str],
        &'static [(&'static str, &'static str)],
    );
    let cases: [Case; 3] = [
        (&["--quiet"], &[]),
        (&[], &[("XYBRID_QUIET", "1")]),
        (&[], &[("CI", "true")]),
    ];
    for (index, (extra, env)) in cases.into_iter().enumerate() {
        let run = sandbox.observed_run(&traces, KEY, extra, env);
        assert_eq!(run.code, Some(1), "{extra:?} {env:?}: {run}");
        ingest.assert_hits(index + 1);
        assert!(
            !run.stderr.contains("xybrid auth claim"),
            "{extra:?} {env:?}: {run}"
        );
        assert!(
            !run.stderr.contains("key expired"),
            "{extra:?} {env:?}: {run}"
        );
        run.assert_no_secret();
    }
}

#[test]
fn a_generic_401_or_a_collector_answer_is_never_an_expiry() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    // An ordinary invalid key: 401 and its status, no workspace marker.
    let invalid = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(401)
            .header("content-type", "application/x-protobuf")
            .body(b"\x08\x10\x12\x0fInvalid API key");
    });
    let traces = server.url("/v1/traces");
    let run = sandbox.observed_run(&traces, KEY, &[], &[]);
    invalid.assert_hits(1);
    assert!(!run.stderr.contains("key expired"), "{run}");
    assert!(!run.stderr.contains("xybrid auth claim"), "{run}");

    // A generic OTLP Collector (environment configuration, no Xybrid
    // preset) answering exactly like an expired Xybrid key.
    let collector = MockServer::start();
    let expired = expired_ingest(&collector);
    let base = collector.base_url();
    let bundle = sandbox.missing_bundle();
    let run = sandbox.run_with(
        &["run", "--bundle", &bundle, "--input-text", "hi"],
        &[("OTEL_EXPORTER_OTLP_ENDPOINT", base.as_str())],
    );
    assert_eq!(run.code, Some(1), "{run}");
    expired.assert_hits(1);
    assert!(!run.stderr.contains("key expired"), "{run}");
    assert!(!run.stderr.contains("xybrid auth claim"), "{run}");
}

#[test]
fn a_provisional_deadline_refreshes_only_the_saved_credential_it_came_from() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let origin = server.base_url();
    let saved = deadline(2);
    sandbox.save_credential(&origin, saved);
    let traces = server.url("/v1/traces");

    // Another key on the same platform and project: the saved deadline
    // stays. (Its notice is shown, and takes today's claim slot.)
    let other = provisional_ingest(&server, OTHER_KEY, deadline(6));
    let run = sandbox.observed_run(&traces, OTHER_KEY, &["--platform-url", &origin], &[]);
    other.assert_hits(1);
    assert_eq!(sandbox.saved_key_deadline(), saved, "{run}");
    assert_eq!(run.claim_nudges(), 1, "{run}");

    // The saved key itself (resolved from the store, no flag): the platform's
    // newer deadline replaces the saved one. Today's claim slot is used, so
    // neither the held reminder nor the notice repeats it.
    let refreshed = deadline(5);
    let own = provisional_ingest(&server, KEY, refreshed);
    let bundle = sandbox.missing_bundle();
    let run = sandbox.run_with(
        &[
            "--platform-url",
            &origin,
            "--observability",
            "--traces-endpoint",
            &traces,
            "run",
            "--bundle",
            &bundle,
            "--input-text",
            "hi",
        ],
        &[],
    );
    assert_eq!(run.code, Some(1), "{run}");
    own.assert_hits(1);
    assert_eq!(sandbox.saved_key_deadline(), refreshed, "{run}");
    assert_eq!(run.claim_nudges(), 0, "one claim nudge a day: {run}");
    run.assert_no_secret();
}

/// `xybrid --platform-url <origin> --observability --traces-endpoint <ingest>
/// run …` with the key the store holds for this project (no key flag).
fn saved_key_run(sandbox: &Sandbox, origin: &str, traces: &str) -> Run {
    let bundle = sandbox.missing_bundle();
    sandbox.run_with(
        &[
            "--platform-url",
            origin,
            "--observability",
            "--traces-endpoint",
            traces,
            "run",
            "--bundle",
            &bundle,
            "--input-text",
            "hi",
        ],
        &[],
    )
}

#[test]
fn a_connected_project_gets_one_claim_line_with_the_platforms_deadline() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let origin = server.base_url();
    let traces = server.url("/v1/traces");
    // `connect` saved one deadline; the platform now reports another.
    let saved = deadline(2);
    let reported = deadline(5);
    sandbox.save_credential(&origin, saved);
    let ingest = provisional_ingest(&server, KEY, reported);

    let run = saved_key_run(&sandbox, &origin, &traces);
    assert_eq!(run.code, Some(1), "{run}");
    ingest.assert_hits(1);
    // The reminder waited for ingest's answer (first received in the final
    // drain): one claim line, carrying the platform's deadline.
    assert_eq!(run.claim_nudges(), 1, "{run}");
    assert!(run.stderr.contains("unclaimed workspace"), "{run}");
    assert!(run.stderr.contains(&shown(reported)), "{run}");
    assert!(
        !run.stderr.contains(&shown(saved)),
        "not the stale date: {run}"
    );
    assert_eq!(sandbox.saved_key_deadline(), reported);
    run.assert_no_secret();

    let again = saved_key_run(&sandbox, &origin, &traces);
    ingest.assert_hits(2);
    assert_eq!(again.claim_nudges(), 0, "once a day: {again}");
}

#[test]
fn without_word_from_ingest_the_held_reminder_still_prints() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let origin = server.base_url();
    let traces = server.url("/v1/traces");
    let saved = deadline(2);
    sandbox.save_credential(&origin, saved);
    // Accepted without the workspace header: no notice from ingest.
    let ingest = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(200);
    });

    // A quiet run neither prints the held reminder nor uses up the day's.
    let quiet = sandbox.run_with(
        &[
            "--quiet",
            "--platform-url",
            &origin,
            "--observability",
            "--traces-endpoint",
            &traces,
            "run",
            "--bundle",
            &sandbox.missing_bundle(),
            "--input-text",
            "hi",
        ],
        &[],
    );
    ingest.assert_hits(1);
    assert_eq!(quiet.claim_nudges(), 0, "{quiet}");

    let run = saved_key_run(&sandbox, &origin, &traces);
    assert_eq!(run.code, Some(1), "{run}");
    ingest.assert_hits(2);
    assert_eq!(run.claim_nudges(), 1, "{run}");
    assert!(run.stderr.contains("when last checked"), "{run}");
    assert!(run.stderr.contains(&shown(saved)), "{run}");
    assert_eq!(sandbox.saved_key_deadline(), saved);
    run.assert_no_secret();
}

#[test]
fn an_expiry_after_accepted_batches_does_not_claim_the_whole_run_was_lost() {
    let sandbox = Sandbox::new();
    let server = MockServer::start();
    let ingest = expired_ingest(&server);
    let run = sandbox.observed_run(&server.url("/v1/traces"), KEY, &[], &[]);
    ingest.assert_hits(1);
    assert!(
        run.stderr.contains("sent after that were not stored"),
        "{run}"
    );
    assert!(!run.stderr.contains("this run's traces were not"), "{run}");
}

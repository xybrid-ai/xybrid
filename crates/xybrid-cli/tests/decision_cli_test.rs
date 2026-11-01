//! `xybrid run --decision` through the built CLI: the request is read,
//! parsed and checked before any model is opened, and this build refuses to
//! run it. Every run gets its own `HOME`, no API key and telemetry off; the
//! model paths name nothing, which proves no model is opened.

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const CANARY: &str = "canary-71fd0a";

const NOT_SUPPORTED: &str = "decision models are not supported by this build yet";

fn xybrid_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_xybrid"))
}

/// A valid request with the canary in its state, ids, keys and text.
fn request() -> String {
    format!(
        r#"{{"model": "d1-2B", "state": "{CANARY} charged twice",
            "questions": {{
              "{CANARY}-route": {{"type": "choice", "instructions": "{CANARY}?", "criteria": {{"{CANARY}-a": "{CANARY}", "b": null}}}},
              "angry": {{"type": "noul", "instructions": {{"{CANARY}": [1, 2.5]}}}},
              "urgency": {{"type": "score", "instructions": "x", "criteria": [null, "{CANARY}"]}}
            }}}}"#
    )
}

/// One isolated `xybrid run`: its own `HOME`, no API key, telemetry off and
/// plain output. `stdin` is piped in when given.
fn run<I, S>(home: &Path, args: I, stdin: Option<&str>) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(xybrid_bin());
    command
        .arg("run")
        .args(args)
        .env("HOME", home)
        .env("NO_COLOR", "1")
        .env_remove("CLICOLOR_FORCE")
        .env("XYBRID_TELEMETRY_OPTOUT", "1")
        .env_remove("XYBRID_API_KEY")
        .env_remove("XYBRID_GATEWAY_URL")
        .env_remove("XYBRID_PLATFORM_URL")
        .env_remove("RUST_BACKTRACE")
        .env_remove("RUST_LIB_BACKTRACE")
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn the xybrid CLI");
    if let Some(text) = stdin {
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(text.as_bytes())
            .expect("write the request to stdin");
    }
    child.wait_with_output().expect("wait for the xybrid CLI")
}

/// A scratch directory with `request.json` holding `body`.
fn with_request(body: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("request.json");
    std::fs::write(&path, body).unwrap();
    (dir, path)
}

#[track_caller]
fn assert_exit(output: &Output, code: i32, expected: &str) -> (String, String) {
    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    let err = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout:\n{out}\nstderr:\n{err}"
    );
    assert!(
        out.contains(expected) || err.contains(expected),
        "expected {expected:?} in\nstdout:\n{out}\nstderr:\n{err}"
    );
    assert!(
        !out.contains(CANARY) && !err.contains(CANARY),
        "request content leaked:\nstdout:\n{out}\nstderr:\n{err}"
    );
    (out, err)
}

#[test]
fn a_valid_request_is_checked_then_refused_before_the_model_opens() {
    let (dir, request) = with_request(&request());
    let model = dir.path().join("absent.gguf");
    let output = run(
        dir.path(),
        [
            OsStr::new("--model-file"),
            model.as_os_str(),
            OsStr::new("--decision"),
            request.as_os_str(),
        ],
        None,
    );
    let (out, _) = assert_exit(&output, 1, NOT_SUPPORTED);
    assert!(out.contains("3 (1 choice, 1 noul, 1 score)"), "{out}");
    assert!(!out.contains("Results"), "{out}");
}

#[test]
fn the_request_can_come_from_stdin() {
    let home = tempfile::tempdir().unwrap();
    let output = run(
        home.path(),
        ["--model", "d1-2b", "--decision", "-"],
        Some(&request()),
    );
    let (out, _) = assert_exit(&output, 1, NOT_SUPPORTED);
    assert!(out.contains("standard input"), "{out}");
}

#[test]
fn a_dry_run_checks_the_request_alone() {
    let (dir, request) = with_request(&request());
    let output = run(
        dir.path(),
        [
            OsStr::new("--directory"),
            dir.path().join("absent").as_os_str(),
            OsStr::new("--decision"),
            request.as_os_str(),
            OsStr::new("--dry-run"),
        ],
        None,
    );
    let (out, _) = assert_exit(&output, 0, "Decision request is valid");
    assert!(out.contains("the model was not checked"), "{out}");
}

#[test]
fn invalid_requests_are_refused_by_kind_without_content() {
    let cases = [
        (
            format!(
                r#"{{"state": "{CANARY}", "questions": {{"q": {{"type": "noul", "instructions": "x", "{CANARY}": true}}}}}}"#
            ),
            "decision request refused (unknown_field; positions count from 0)",
        ),
        (
            format!(r#"{{"state": "{CANARY}", "questions": {{"{CANARY}": "not an object"}}}}"#),
            "decision request refused (invalid_question; positions count from 0)",
        ),
        (
            format!(
                r#"{{"state": {{"{CANARY}": 18446744073709551615}}, "questions": {{"q": {{"type": "noul", "instructions": "x"}}}}}}"#
            ),
            "decision request refused (number_out_of_range; positions count from 0)",
        ),
        (
            format!(
                r#"{{"state": "{CANARY}", "files": [], "questions": {{"q": {{"type": "noul", "instructions": "x"}}}}}}"#
            ),
            "decision request refused (media_unsupported; positions count from 0)",
        ),
        (
            format!(
                r#"{{"state": "{CANARY}", "questions": {{"q": {{"type": "noul", "instructions": "x"}}}}"#
            ),
            "decision request refused (invalid_json; positions count from 0)",
        ),
    ];
    for (body, expected) in &cases {
        // Both a run and a dry run refuse it, from a file or stdin.
        let (dir, request) = with_request(body);
        for dry_run in [false, true] {
            let mut args = vec![
                OsStr::new("--model-file"),
                OsStr::new("absent.gguf"),
                OsStr::new("--decision"),
                request.as_os_str(),
            ];
            if dry_run {
                args.push(OsStr::new("--dry-run"));
            }
            let output = run(dir.path(), args, None);
            assert_exit(&output, 1, expected);
        }
        let output = run(
            dir.path(),
            ["--model-file", "absent.gguf", "--decision", "-"],
            Some(body),
        );
        assert_exit(&output, 1, expected);
    }
}

#[test]
fn a_structurally_invalid_request_is_refused_after_parsing() {
    // Seventeen questions: the transport reads them, validation refuses them.
    let questions: Vec<String> = (0..17)
        .map(|i| format!(r#""q{i}": {{"type": "noul", "instructions": "{CANARY}"}}"#))
        .collect();
    let body = format!(
        r#"{{"state": null, "questions": {{{}}}}}"#,
        questions.join(", ")
    );
    let (dir, request) = with_request(&body);
    let output = run(
        dir.path(),
        [
            OsStr::new("--model-file"),
            OsStr::new("absent.gguf"),
            OsStr::new("--decision"),
            request.as_os_str(),
        ],
        None,
    );
    assert_exit(&output, 1, "17 questions; at most 16 are accepted");
}

#[test]
fn a_decision_needs_a_model_and_a_readable_request() {
    let (dir, request) = with_request(&request());
    let output = run(
        dir.path(),
        [OsStr::new("--decision"), request.as_os_str()],
        None,
    );
    assert_exit(&output, 1, "--decision needs a model");

    let output = run(
        dir.path(),
        [
            OsStr::new("--model-file"),
            OsStr::new("absent.gguf"),
            OsStr::new("--decision"),
            dir.path().join("missing.json").as_os_str(),
        ],
        None,
    );
    assert_exit(&output, 1, "Failed to read the decision request");
}

#[test]
fn decision_is_a_usage_error_with_input_flags() {
    let (dir, request) = with_request(&request());
    let output = run(
        dir.path(),
        [
            OsStr::new("--model-file"),
            OsStr::new("absent.gguf"),
            OsStr::new("--decision"),
            request.as_os_str(),
            OsStr::new("--input-text"),
            OsStr::new("context"),
        ],
        None,
    );
    assert_exit(&output, 2, "cannot be used with");
}

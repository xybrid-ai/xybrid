//! Decision requests through the built CLI, including stdin, machine output
//! and parity with the Rust SDK. Model-required CI must stage the pinned Q4.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const REQUEST: &str = r#"{
  "state": {"ticket": "Please refund the duplicate charge.", "refunds": 0},
  "questions": {
    "zeta": {"type": "choice", "instructions": "Choose the customer request.",
      "criteria": {"refund": "Refund a charge", "cancel": "Cancel a subscription"}},
    "alpha": {"type": "noul", "instructions": "Is the customer requesting a refund?"},
    "Middle": {"type": "score", "instructions": "How urgent is this request?",
      "criteria": ["Can wait", "Today", "Immediate"]}
  }
}"#;

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xybrid"));
    command
        .arg("run")
        .env("XYBRID_TELEMETRY_OPTOUT", "1")
        // Exercise the telemetry-enabled CLI branch without exporting data.
        .env("XYBRID_API_KEY", "decision-cli-test")
        .env("XYBRID_QUIET", "1")
        .env("NO_COLOR", "1")
        .env_remove("CLICOLOR_FORCE")
        .env_remove("XYBRID_SPECULATIVE_CLOUD")
        .env_remove("RUST_BACKTRACE")
        .env_remove("RUST_LIB_BACKTRACE");
    command
}

fn stdin_run(model_dir: &Path, bytes: &[u8]) -> Output {
    let mut child = command()
        .arg("--directory")
        .arg(model_dir)
        .args(["--decision-request", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    child.wait_with_output().unwrap()
}

fn unloaded_model() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("model_metadata.json"),
        r#"{"model_id":"decision-test","version":"1","execution_template":{
          "type":"Decision","protocol":"lfm3-d1/text/v1","model_file":"absent.gguf"
        },"files":[]}"#,
    )
    .unwrap();
    dir
}

#[test]
fn decision_cli_rejects_conflicts_before_model_loading() {
    for flags in [
        vec!["--input-text", "text"],
        vec!["--max-tokens", "1"],
        vec!["--speculative-cloud"],
        vec!["--trace"],
        vec!["--target", "cloud"],
    ] {
        let output = command()
            .args([
                "--directory",
                "nonexistent-decision-cli-model",
                "--decision-request",
                "-",
            ])
            .args(flags)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
    }
}

#[test]
fn decision_cli_bounds_stdin_and_redacts_decoder_errors_before_model_loading() {
    let dir = unloaded_model();
    let oversized = vec![b' '; xybrid_core::execution::decision::wire::MAX_RAW_BYTES + 1];
    for bytes in [
        oversized.as_slice(),
        &b"private-decision-cli-canary\xff"[..],
        &b"{\"private-decision-cli-canary\": 1}"[..],
    ] {
        let output = stdin_run(dir.path(), bytes);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!error.contains("private-decision-cli-canary"));
        assert!(!error.contains("absent.gguf"));
        assert!(!error.contains("requires llm-llamacpp"));
    }
}

#[cfg(feature = "llm-llamacpp")]
fn staged_model() -> Option<PathBuf> {
    xybrid_core::testing::model_fixtures::staged_artifact("d1-2b-q4km", "d1-2B-Q4_K_M.gguf")
        .map(|path| path.parent().unwrap().to_path_buf())
}

#[cfg(not(feature = "llm-llamacpp"))]
fn staged_model() -> Option<PathBuf> {
    assert!(
        !xybrid_core::testing::model_fixtures::models_required(),
        "XYBRID_REQUIRE_MODELS requires the llm-llamacpp feature for CLI parity"
    );
    eprintln!("Skipping decision CLI parity: build with llm-llamacpp");
    None
}

fn assert_record(bytes: &[u8], expected: &serde_json::Value, input_tokens: u64) {
    let text = std::str::from_utf8(bytes).unwrap();
    assert_eq!(
        text.lines().count(),
        1,
        "stdout must contain only one JSON record"
    );
    let record: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(record["format"], "xybrid.decision.v1");
    assert_eq!(record["model_id"], "d1-2b-q4km");
    assert_eq!(record["tokens_in"], input_tokens);
    assert_eq!(record["tokens_out"], 0);
    assert_eq!(&record["decisions"], expected);
}

#[test]
fn decision_cli_file_stdin_and_machine_output_match_the_rust_sdk() {
    let Some(model_dir) = staged_model() else {
        return;
    };
    let model = xybrid_sdk::ModelLoader::from_directory(&model_dir)
        .unwrap()
        .with_speculative_cloud(false)
        .load()
        .unwrap();
    let request = xybrid_sdk::decode_decision_request(REQUEST).unwrap();
    let result = model
        .run(&xybrid_sdk::ir::Envelope::decision_request(request), None)
        .unwrap();
    let answers = result.decisions().unwrap().answers();
    assert_eq!(
        answers
            .iter()
            .map(|answer| answer.question_id())
            .collect::<Vec<_>>(),
        ["zeta", "alpha", "Middle"]
    );
    let expected = serde_json::to_value(result.decisions().unwrap()).unwrap();
    let input_tokens = result.tokens_in().unwrap();
    drop(model);

    let dir = tempfile::tempdir().unwrap();
    let request_path = dir.path().join("request.json");
    std::fs::write(&request_path, REQUEST).unwrap();
    let output = command()
        .arg("--directory")
        .arg(&model_dir)
        .arg("--decision-request")
        .arg(&request_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_record(&output.stdout, &expected, input_tokens);

    let output = stdin_run(&model_dir, REQUEST.as_bytes());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_record(&output.stdout, &expected, input_tokens);

    let output_path = dir.path().join("decisions.json");
    let output = command()
        .arg("--directory")
        .arg(&model_dir)
        .arg("--decision-request")
        .arg(&request_path)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert_record(
        &std::fs::read(output_path).unwrap(),
        &expected,
        input_tokens,
    );
}

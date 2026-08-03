#![cfg(feature = "llm-llamacpp")]

use std::path::PathBuf;
use std::process::Command;
use xybrid_core::testing::model_fixtures;

const MODEL_ID: &str = "qwen3-0.6b";
const ROOK_PROMPT: &str = "How many ways can you place 3 rooks on a 3x3 chessboard so that none attack each other? Think through every case carefully before answering.";

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

#[test]
fn run_with_tiny_budget_warns_about_thinking_exhaustion() {
    let Some(model_dir) = model_fixtures::model_or_skip(MODEL_ID) else {
        return;
    };

    let output = Command::new(xybrid_bin())
        .current_dir(workspace_root())
        .args(["run", "--directory"])
        .arg(model_dir)
        .args(["--input-text", ROOK_PROMPT, "--max-tokens", "32"])
        .output()
        .expect("run xybrid CLI with tiny reasoning budget");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "xybrid run failed with exit code {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        stdout,
        stderr
    );
    // Stable substrings from THINKING_BUDGET_HINT in commands/utils.rs.
    assert!(stdout.contains("token budget thinking"), "stdout: {stdout}");
    assert!(stdout.contains("--max-tokens"), "stdout: {stdout}");
}

#[test]
fn run_with_ample_budget_produces_answer() {
    let Some(model_dir) = model_fixtures::model_or_skip(MODEL_ID) else {
        return;
    };
    let temp_dir = tempfile::tempdir().expect("create CLI test temp dir");
    let output_path = temp_dir.path().join("answer.txt");

    let output = Command::new(xybrid_bin())
        .current_dir(workspace_root())
        .args(["run", "--directory"])
        .arg(model_dir)
        .args([
            "--input-text",
            "What is 2+2? Answer with just the number.",
            "--max-tokens",
            "2048",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("run xybrid CLI with ample reasoning budget");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "xybrid run failed with exit code {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        stdout,
        stderr
    );
    let answer = std::fs::read_to_string(&output_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", output_path.display()));
    assert!(
        !answer.trim().is_empty(),
        "CLI answer output must not be empty"
    );
    assert!(
        !stdout.contains("token budget thinking"),
        "ample budget unexpectedly triggered warning: {stdout}"
    );
}

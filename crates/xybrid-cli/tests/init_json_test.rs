//! Integration test for `xybrid init --json` output mode.
//!
//! Verifies that the --json flag produces valid JSON to stdout
//! with the expected structure: { status, path, task, confidence }.

use std::process::Command;

/// Get the path to the compiled xybrid binary.
fn xybrid_bin() -> std::path::PathBuf {
    // CARGO_BIN_EXE_xybrid is set by cargo for integration tests
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_xybrid"))
}

/// Get the workspace root (repos/xybrid/).
fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .unwrap()
        .parent() // repos/xybrid/
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_init_json_on_mnist_fixture() {
    // Copy the mnist model.onnx to a temp directory (without model_metadata.json)
    let tmp = tempfile::tempdir().expect("Failed to create temp dir");
    let mnist_dir = workspace_root().join("integration-tests/fixtures/models/mnist");
    std::fs::copy(mnist_dir.join("model.onnx"), tmp.path().join("model.onnx"))
        .expect("Failed to copy model.onnx");

    let output = Command::new(xybrid_bin())
        .args(["init", "--json", "--yes"])
        .arg(tmp.path().to_str().unwrap())
        .output()
        .expect("Failed to run xybrid init");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "xybrid init --json failed with exit code {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        stdout,
        stderr
    );

    // Parse stdout as JSON
    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|_| panic!("Invalid JSON in stdout: {}", stdout));

    // Verify required fields
    assert_eq!(json["status"], "ok", "Expected status 'ok', got: {}", json);
    assert!(
        json["path"].as_str().is_some(),
        "Expected 'path' field in JSON: {}",
        json
    );
    assert!(
        json["task"].as_str().is_some(),
        "Expected 'task' field in JSON: {}",
        json
    );
    assert!(
        json["confidence"].as_str().is_some(),
        "Expected 'confidence' field in JSON: {}",
        json
    );

    // Verify the path points to model_metadata.json in the temp dir
    let path = json["path"].as_str().unwrap();
    assert!(
        path.ends_with("model_metadata.json"),
        "Path should end with model_metadata.json: {}",
        path
    );
    assert!(
        std::path::Path::new(path).exists(),
        "model_metadata.json should exist at: {}",
        path
    );

    // Verify confidence is a valid value
    let confidence = json["confidence"].as_str().unwrap();
    assert!(
        ["high", "medium", "low"].contains(&confidence),
        "Confidence should be high/medium/low, got: {}",
        confidence
    );

    // Verify no human-readable output leaked into stdout (only JSON)
    assert!(
        !stdout.contains("Scanning"),
        "stdout should not contain human-readable output in --json mode"
    );
}

#[test]
fn test_init_json_error_on_nonexistent_dir() {
    let output = Command::new(xybrid_bin())
        .args(["init", "--json", "/nonexistent/path/that/does/not/exist"])
        .output()
        .expect("Failed to run xybrid init");

    // Should fail with non-zero exit code
    assert!(
        !output.status.success(),
        "Expected failure for nonexistent directory"
    );

    // stderr should contain valid JSON error (or at least the process should fail)
    // Note: canonicalize() failure may produce anyhow error before our JSON handler
    // so we just verify it's non-zero exit
    assert_ne!(output.status.code(), Some(0));
}

#[test]
fn test_init_json_implies_yes() {
    // --json should imply --yes (non-interactive), so it should not hang
    // on ambiguous models. We test this by running on a fixture without
    // --yes explicitly — if it hangs, the test will timeout.
    let tmp = tempfile::tempdir().expect("Failed to create temp dir");
    let mnist_dir = workspace_root().join("integration-tests/fixtures/models/mnist");
    std::fs::copy(mnist_dir.join("model.onnx"), tmp.path().join("model.onnx"))
        .expect("Failed to copy model.onnx");

    let output = Command::new(xybrid_bin())
        .args(["init", "--json"]) // no --yes, but --json implies it
        .arg(tmp.path().to_str().unwrap())
        .output()
        .expect("Failed to run xybrid init");

    assert!(
        output.status.success(),
        "xybrid init --json should succeed without --yes (implied)"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|_| panic!("Invalid JSON in stdout: {}", stdout));

    assert_eq!(json["status"], "ok");
}

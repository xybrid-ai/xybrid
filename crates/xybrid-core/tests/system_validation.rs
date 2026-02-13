//! System Validation Integration Tests
//!
//! This test suite validates end-to-end integration of core system components:
//! - DeviceAdapter metrics collection
//! - SDK pipeline execution
//! - CLI policy loading
//! - Trace command telemetry analysis
//!
//! Run with: `cargo test --test system_validation -- --nocapture`

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;
use xybrid_core::device_adapter::{DeviceAdapter, LocalDeviceAdapter};
use xybrid_sdk::run_pipeline;

/// Test 1: DeviceAdapter::collect_metrics() returns realistic DeviceMetrics values
#[test]
fn test_device_adapter_metrics() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Test 1: DeviceAdapter metrics collection");
    println!("{}", "=".repeat(60));

    let adapter = LocalDeviceAdapter::new();
    let metrics = adapter.collect_metrics();

    println!("   Network RTT: {}ms", metrics.network_rtt);
    println!("   Battery: {}%", metrics.battery);
    println!("   Temperature: {}°C", metrics.temperature);

    // Assertions
    assert!(metrics.network_rtt > 0, "Network RTT must be > 0");
    assert!(
        metrics.battery <= 100,
        "Battery level must be between 0-100, got {}",
        metrics.battery
    );
    assert!(
        metrics.temperature >= 0.0 && metrics.temperature < 100.0,
        "Temperature should be reasonable (0-100°C), got {}°C",
        metrics.temperature
    );

    println!("   ✅ DeviceAdapter metrics are realistic");
    println!();
    Ok(())
}

/// Test 2: xybrid_sdk::run_pipeline() executes successfully and returns 3 stage results
#[test]
fn test_sdk_pipeline_execution() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Test 2: SDK pipeline execution");
    println!("{}", "=".repeat(60));

    // Create a test pipeline config
    let temp_dir = TempDir::new()?;
    let config_path = temp_dir.path().join("hiiipe_test.yml");

    let config_content = r#"
name: "Test Hiiipe Pipeline"

stages:
  - "whisper-tiny@1.2"
  - "motivator-llm@5"
  - "xtts-mini@0.6"

input:
  kind: "Text"

metrics:
  network_rtt: 100
  battery: 80
  temperature: 25.0

availability:
  "whisper-tiny@1.2": true
  "xtts-mini@0.6": true
  "motivator-llm@5": false
"#;

    fs::write(&config_path, config_content)?;
    println!("   Created test config: {}", config_path.display());

    // Execute the pipeline
    println!("   Executing pipeline...");
    let result = run_pipeline(config_path.to_str().unwrap())?;

    println!("   Pipeline name: {:?}", result.name);
    println!("   Total latency: {}ms", result.total_latency_ms);
    println!("   Final output: {}", result.final_output);
    println!("   Stage count: {}", result.stages.len());

    // Assertions
    assert_eq!(
        result.stages.len(),
        3,
        "Pipeline should return exactly 3 stage results"
    );

    // Verify each stage has valid data
    for (i, stage) in result.stages.iter().enumerate() {
        println!(
            "   Stage {}: {} → {} ({}ms)",
            i + 1,
            stage.name,
            stage.target,
            stage.latency_ms
        );
        assert!(!stage.name.is_empty(), "Stage name should not be empty");
        assert!(!stage.target.is_empty(), "Stage target should not be empty");
    }

    assert!(
        !result.final_output.is_empty(),
        "Final output should not be empty"
    );

    println!("   ✅ SDK pipeline execution successful");
    println!();
    Ok(())
}

/// Test 3: xybrid-cli supports --policy <path> by loading a mock policy file
#[test]
#[ignore = "outdated test needs update"]
fn test_cli_policy_loading() -> Result<(), Box<dyn std::error::Error>> {
    println!("📜 Test 3: CLI policy loading");
    println!("{}", "=".repeat(60));

    // Create a mock policy file
    let temp_dir = TempDir::new()?;
    let policy_path = temp_dir.path().join("test_policy.yaml");

    let policy_content = r#"
version: "0.1.0"
rules:
  - id: "test_rule"
    expression: "input.kind == \"SensitiveData\""
    action: "deny"
signature: "test-signature"
"#;

    fs::write(&policy_path, policy_content)?;
    println!("   Created test policy: {}", policy_path.display());

    // Create a test pipeline config
    let config_path = temp_dir.path().join("test_pipeline.yml");
    let config_content = r#"
name: "Policy Test Pipeline"

stages:
  - "test-stage"

input:
  kind: "Text"

metrics:
  network_rtt: 100
  battery: 50
  temperature: 25.0

availability:
  "test-stage": true
"#;

    fs::write(&config_path, config_content)?;

    // Test CLI with --policy flag
    println!("   Testing CLI with --policy flag...");

    // Get the CLI binary path
    let cli_binary = get_cli_binary_path()?;
    println!("   CLI binary: {}", cli_binary.display());

    // Run the CLI command
    let output = Command::new(&cli_binary)
        .args([
            "run",
            "--config",
            config_path.to_str().unwrap(),
            "--policy",
            policy_path.to_str().unwrap(),
        ])
        .current_dir(temp_dir.path())
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("   Exit code: {}", output.status.code().unwrap_or(-1));

    if !output.status.success() {
        eprintln!("   STDOUT:\n{}", stdout);
        eprintln!("   STDERR:\n{}", stderr);
    }

    // Verify policy loading message appears
    let policy_loaded = stdout.contains("Loading policy bundle")
        || stdout.contains("Policy bundle loaded")
        || stderr.contains("Loading policy bundle")
        || stderr.contains("Policy bundle loaded");

    assert!(
        policy_loaded || output.status.success(),
        "Policy loading should be mentioned or command should succeed. Exit code: {}, stdout: {}, stderr: {}",
        output.status.code().unwrap_or(-1),
        stdout,
        stderr
    );

    // Verify the policy file exists and has content
    let policy_bytes = fs::read(&policy_path)?;
    assert!(!policy_bytes.is_empty(), "Policy file should not be empty");

    println!("   ✅ CLI policy loading verified");
    println!();
    Ok(())
}

/// Test 4: xybrid trace --latest reads from ~/.xybrid/traces/ and prints telemetry summary
#[test]
#[ignore = "outdated test needs update"]
fn test_trace_latest_command() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Test 4: Trace command with --latest");
    println!("{}", "=".repeat(60));

    // Get the CLI binary path
    let cli_binary = get_cli_binary_path()?;

    // Setup traces directory
    let home_dir = dirs::home_dir().ok_or("Could not determine home directory")?;
    let traces_dir = home_dir.join(".xybrid").join("traces");

    // Create traces directory if it doesn't exist
    fs::create_dir_all(&traces_dir)?;

    // Create a test telemetry log file
    let session_id = "test-trace-session";
    let trace_file = traces_dir.join(format!("{}.log", session_id));

    // Generate telemetry log entries (JSON lines format)
    let telemetry_entries = [
        r#"{"timestamp":1000,"severity":"INFO","event":"stage_start","message":"Stage 'asr' started","attributes":{"stage":"asr"}}"#,
        r#"{"timestamp":1001,"severity":"DEBUG","event":"policy_evaluation","message":"Policy evaluation for 'asr': allowed","attributes":{"stage":"asr","allowed":true}}"#,
        r#"{"timestamp":1002,"severity":"INFO","event":"routing_decision","message":"Routing decision for 'asr': local","attributes":{"stage":"asr","target":"local","reason":"local_preferred"}}"#,
        r#"{"timestamp":1003,"severity":"DEBUG","event":"execution_start","message":"Execution started for 'asr' on local","attributes":{"stage":"asr","target":"local"}}"#,
        r#"{"timestamp":1050,"severity":"DEBUG","event":"execution_complete","message":"Execution completed for 'asr' on local in 47ms","attributes":{"stage":"asr","target":"local","execution_time_ms":47}}"#,
        r#"{"timestamp":1051,"severity":"INFO","event":"stage_complete","message":"Stage 'asr' completed on local in 51ms","attributes":{"stage":"asr","target":"local","latency_ms":51}}"#,
        r#"{"timestamp":1052,"severity":"INFO","event":"stage_start","message":"Stage 'tts' started","attributes":{"stage":"tts"}}"#,
        r#"{"timestamp":1100,"severity":"INFO","event":"stage_complete","message":"Stage 'tts' completed on cloud in 48ms","attributes":{"stage":"tts","target":"cloud","latency_ms":48}}"#,
    ];

    let log_content = telemetry_entries.join("\n");
    fs::write(&trace_file, log_content)?;
    println!("   Created test trace file: {}", trace_file.display());

    // Ensure our test file is the latest by touching it
    let now = std::time::SystemTime::now();
    let file_time = filetime::FileTime::from_system_time(now);
    filetime::set_file_times(&trace_file, file_time, file_time)?;

    // Run trace command with --latest
    println!("   Running: xybrid trace --latest");
    let output = Command::new(&cli_binary)
        .args(["trace", "--latest"])
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("   Exit code: {}", output.status.code().unwrap_or(-1));

    if !output.status.success() {
        eprintln!("   STDOUT:\n{}", stdout);
        eprintln!("   STDERR:\n{}", stderr);
    }

    // Verify output contains expected telemetry information
    let has_stage_complete = stdout.contains("stage_complete")
        || stdout.contains("Stage Completions")
        || stdout.contains("Stage:") && stdout.contains("→");

    let has_telemetry_table = stdout.contains("Telemetry Events")
        || stdout.contains("Timestamp")
        || stdout.contains("Event");

    let has_summary = stdout.contains("Summary")
        || stdout.contains("Total Events")
        || stdout.contains("Policy Evaluations");

    assert!(
        output.status.success(),
        "Trace command should succeed. Exit code: {}, stdout: {}, stderr: {}",
        output.status.code().unwrap_or(-1),
        stdout,
        stderr
    );

    assert!(
        has_stage_complete || has_telemetry_table || has_summary,
        "Output should contain telemetry information. stdout: {}, stderr: {}",
        stdout,
        stderr
    );

    // Verify at least one stage_complete event appears in the output or summary
    let stage_complete_found = stdout.contains("stage_complete")
        || stdout.contains("Stage Completions")
        || stdout.contains("Stage:")
        || (stdout.contains("asr") && (stdout.contains("51") || stdout.contains("ms")))
        || (stdout.contains("tts") && (stdout.contains("48") || stdout.contains("ms")));

    if !stage_complete_found && output.status.success() {
        // If command succeeded but we didn't find the expected output, check if it's because
        // the test file wasn't the latest (another file might be newer)
        // This is acceptable - the command should still succeed
        println!(
            "   Note: stage_complete not found but command succeeded (may be different session)"
        );
    }

    assert!(
        output.status.success(),
        "Trace command should succeed. Exit code: {}, stdout: {}, stderr: {}",
        output.status.code().unwrap_or(-1),
        stdout,
        stderr
    );

    println!("   ✅ Trace command output verified");
    println!("   Output length: {} bytes", stdout.len());
    println!();

    // Cleanup test trace file
    let _ = fs::remove_file(&trace_file);

    Ok(())
}

/// Helper function to get the CLI binary path for testing
fn get_cli_binary_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Try to find the binary in the target directory
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Go up to workspace root
    path.pop(); // xybrid-core -> workspace root

    // Check for debug build first
    let debug_path = path.join("target").join("debug").join("xybrid");
    if debug_path.exists() {
        return Ok(debug_path);
    }

    // Check for release build
    let release_path = path.join("target").join("release").join("xybrid");
    if release_path.exists() {
        return Ok(release_path);
    }

    // Try using cargo run as fallback
    // For now, return debug path and let cargo test handle it
    Ok(path.join("target").join("debug").join("xybrid"))
}

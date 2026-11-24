//! Integration tests for `xybrid telemetry status`.
//!
//! Verifies the command reports the correct opt-in / opt-out state based on
//! the `XYBRID_TELEMETRY_OPTOUT` env var. Each case spawns a fresh subprocess
//! so the OnceLock cache in `is_telemetry_opted_out()` is process-isolated.

use std::process::Command;

fn xybrid_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_xybrid"))
}

#[test]
fn telemetry_status_reports_enabled_when_optout_unset() {
    let output = Command::new(xybrid_bin())
        .args(["telemetry", "status"])
        .env_remove("XYBRID_TELEMETRY_OPTOUT")
        .output()
        .expect("Failed to run xybrid telemetry status");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "telemetry status should exit 0 (got {:?}), stderr: {}",
        output.status.code(),
        stderr
    );
    assert_eq!(output.status.code(), Some(0));
    assert!(
        stdout.contains("enabled"),
        "expected stdout to contain 'enabled', got: {}",
        stdout
    );
    assert!(
        !stdout.contains("registry telemetry: disabled"),
        "expected stdout NOT to contain 'disabled', got: {}",
        stdout
    );
}

#[test]
fn telemetry_status_reports_disabled_when_optout_set() {
    let output = Command::new(xybrid_bin())
        .args(["telemetry", "status"])
        .env("XYBRID_TELEMETRY_OPTOUT", "1")
        .output()
        .expect("Failed to run xybrid telemetry status");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "telemetry status should exit 0 (got {:?}), stderr: {}",
        output.status.code(),
        stderr
    );
    assert_eq!(output.status.code(), Some(0));
    assert!(
        stdout.contains("disabled"),
        "expected stdout to contain 'disabled', got: {}",
        stdout
    );
    assert!(
        stdout.contains("XYBRID_TELEMETRY_OPTOUT=1"),
        "expected stdout to mention the env var, got: {}",
        stdout
    );
}

#[test]
fn telemetry_status_help_mentions_opt_out_env_var() {
    let output = Command::new(xybrid_bin())
        .args(["telemetry", "status", "--help"])
        .output()
        .expect("Failed to run xybrid telemetry status --help");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "help should exit 0");
    assert!(
        stdout.contains("XYBRID_TELEMETRY_OPTOUT"),
        "help should mention the opt-out env var, got: {}",
        stdout
    );
}

/// Accepts connections on a loopback port and counts them (no reply needed:
/// an export attempt is visible as a connection).
fn counting_listener() -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let count = std::sync::Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            if stream.is_ok() {
                seen.fetch_add(1, Ordering::SeqCst);
            }
        }
    });
    (format!("http://{address}/v1/traces"), count)
}

#[test]
fn telemetry_status_json_is_valid_json_and_starts_no_exporter() {
    let (endpoint, connections) = counting_listener();
    let output = Command::new(xybrid_bin())
        .args([
            "--api-key",
            "test-key",
            "--observability",
            "--traces-endpoint",
            &endpoint,
            "telemetry",
            "status",
            "--json",
        ])
        .env_remove("OTEL_SDK_DISABLED")
        .output()
        .expect("Failed to run xybrid telemetry status --json");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let status: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        panic!(
            "stdout must be one JSON document ({e}): {}",
            String::from_utf8_lossy(&output.stdout)
        )
    });
    assert_eq!(status["export_configured"], true);
    assert_eq!(status["redaction_rules"], 0);
    assert_eq!(status["resource_telemetry"], "off");
    assert!(
        !status.to_string().contains("test-key"),
        "credentials are never printed"
    );
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert_eq!(
        connections.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "status reports configuration; it exports nothing"
    );
}

#[test]
fn trace_export_respects_otel_sdk_disabled_and_writes_the_local_trace() {
    for disabled in [true, false] {
        let (endpoint, connections) = counting_listener();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.json");
        let mut command = Command::new(xybrid_bin());
        command
            .args([
                "--api-key",
                "test-key",
                "--observability",
                "--traces-endpoint",
                &endpoint,
                "run",
                "--bundle",
            ])
            .arg(dir.path().join("missing.xyb"))
            .args(["--input-text", "hi", "--trace-export"])
            .arg(&path);
        if disabled {
            command.env("OTEL_SDK_DISABLED", "true");
        } else {
            command.env_remove("OTEL_SDK_DISABLED");
        }
        let output = command.output().expect("Failed to run xybrid run");
        assert!(!output.status.success(), "the bundle is missing");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let trace: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("trace file written"))
                .expect("OTLP/JSON trace");
        let spans = trace["resourceSpans"][0]["scopeSpans"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|s| s["spans"].as_array().unwrap().clone())
            .collect::<Vec<_>>();
        assert!(spans
            .iter()
            .any(|s| s["name"] == "invoke_workflow xybrid-cli run"));
        let resource = trace["resourceSpans"][0]["resource"]["attributes"].to_string();
        assert!(resource.contains("telemetry.sdk.name"), "{resource}");
        // Shutdown completes before the process exits.
        let attempts = connections.load(std::sync::atomic::Ordering::SeqCst);
        if disabled {
            assert_eq!(attempts, 0, "OTEL_SDK_DISABLED must stop export: {stderr}");
            assert!(stderr.contains("OTEL_SDK_DISABLED"), "{stderr}");
            assert!(!stderr.contains("Observability enabled"), "{stderr}");
        } else {
            assert!(attempts > 0, "export was configured: {stderr}");
        }
        assert!(
            !String::from_utf8_lossy(&output.stdout).contains("Observability enabled"),
            "status lines go to stderr"
        );
    }
}

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
        !stdout.contains("disabled"),
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

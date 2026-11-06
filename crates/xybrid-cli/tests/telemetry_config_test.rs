//! Exercise telemetry identity configuration through isolated CLI processes.
//!
//! On Windows `dirs::home_dir` uses the OS profile API, so environment variables
//! cannot isolate the subprocess home. Storage logic is tested on all platforms
//! by the install_id unit tests; these process tests must not touch a real profile.

#![cfg(unix)]

use std::path::Path;
use std::process::{Command, Output};

fn cli(home: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xybrid"));
    cmd.args(["telemetry", "status"])
        .env("HOME", home)
        .env("XYBRID_API_KEY", "sk_test_placeholder")
        .env("XYBRID_INGEST_URL", "http://127.0.0.1:1")
        .env_remove("XYBRID_DEVICE_ID")
        .env_remove("XYBRID_TELEMETRY_OPTOUT");
    cmd
}

#[test]
fn ingest_environment_and_flag_override_the_default_independently_of_platform() {
    let home = tempfile::tempdir().unwrap();
    for (environment, flag, expected) in [
        (None, None, "https://ingest.xybrid.dev"),
        (Some("http://localhost:8081"), None, "http://localhost:8081"),
        (
            Some("http://localhost:8081"),
            Some("http://localhost:8082"),
            "http://localhost:8082",
        ),
    ] {
        let mut cmd = cli(home.path());
        // The first case keeps the production default, so opt out: the startup
        // hint still names the endpoint, and no event can reach it.
        cmd.env_remove("XYBRID_INGEST_URL")
            .env("XYBRID_PLATFORM_URL", "http://localhost:3001")
            .env("XYBRID_TELEMETRY_OPTOUT", "1")
            .env("NO_COLOR", "1");
        if let Some(value) = environment {
            cmd.env("XYBRID_INGEST_URL", value);
        }
        if let Some(value) = flag {
            cmd.args(["--ingest-url", value]);
        }
        let output = cmd.output().unwrap();
        // Startup displays the same parsed destination passed to the exporter.
        // No inference runs here, so no requests go to any configured endpoint.
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(&format!("Telemetry enabled ({expected})")),
            "{stdout}"
        );
        assert_success(output);
    }
}

fn assert_success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn default_identity_persists_across_cli_processes() {
    let home = tempfile::tempdir().unwrap();
    assert_success(cli(home.path()).output().unwrap());
    let path = home.path().join(".xybrid/install-id");
    let id = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        uuid::Uuid::parse_str(id.trim()).unwrap().get_version(),
        Some(uuid::Version::Random)
    );
    assert_success(cli(home.path()).output().unwrap());
    assert_eq!(std::fs::read_to_string(path).unwrap(), id);
}

#[test]
fn explicit_identity_opt_out_and_no_key_do_not_create_install_id() {
    let home = tempfile::tempdir().unwrap();
    assert_success(
        cli(home.path())
            .env("XYBRID_TELEMETRY_OPTOUT", "1")
            .output()
            .unwrap(),
    );
    assert_success(
        cli(home.path())
            .args(["--device-id", "test-device"])
            .output()
            .unwrap(),
    );
    assert_success(
        cli(home.path())
            .env("XYBRID_DEVICE_ID", "env-device")
            .output()
            .unwrap(),
    );
    assert_success(
        cli(home.path())
            .env_remove("XYBRID_API_KEY")
            .output()
            .unwrap(),
    );
    assert!(!home.path().join(".xybrid/install-id").exists());
}

#[test]
fn unavailable_identity_storage_does_not_break_commands() {
    let home = tempfile::tempdir().unwrap();
    // A file in place of the directory works across platforms and under root.
    std::fs::write(home.path().join(".xybrid"), "blocked").unwrap();
    assert_success(cli(home.path()).output().unwrap());
}

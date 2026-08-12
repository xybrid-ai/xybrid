#![cfg(unix)]

use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output};

const PAYLOAD_SHA256: &str = "4aec73f34f94387203f6b7b5b6977085006ccea54136b82960dc7d0d8dada0c1";
const MISMATCH_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const SUCCESSFUL_CURL: &str = r#"#!/usr/bin/env bash
set -euo pipefail
while (( $# > 0 )); do
    if [[ "$1" == "-o" ]]; then
        printf 'fixture model bytes' > "$2"
        exit 0
    fi
    shift
done
exit 1
"#;
const FAILING_CURL: &str = r#"#!/usr/bin/env bash
set -euo pipefail
while (( $# > 0 )); do
    if [[ "$1" == "-o" ]]; then
        printf 'partial model bytes' > "$2"
        exit 22
    fi
    shift
done
exit 22
"#;
const INVALID_RESPONSE_CURL: &str = r#"#!/usr/bin/env bash
set -euo pipefail
while (( $# > 0 )); do
    if [[ "$1" == "-o" ]]; then
        printf '404 error' > "$2"
        exit 0
    fi
    shift
done
exit 1
"#;

struct DownloadRun {
    _temp: tempfile::TempDir,
    model_dir: PathBuf,
    output: Output,
}

fn run_download(
    file_sha256: Option<&str>,
    metadata_sha256: Option<&str>,
    fake_curl_script: &str,
) -> Result<DownloadRun, Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let integration_dir = temp.path().join("integration-tests");
    let models_dir = integration_dir.join("fixtures/models");
    let fake_bin = temp.path().join("bin");
    fs::create_dir_all(&models_dir)?;
    fs::create_dir_all(&fake_bin)?;

    let downloader = integration_dir.join("download.sh");
    fs::write(&downloader, include_str!("../download.sh"))?;
    fs::set_permissions(&downloader, fs::Permissions::from_mode(0o755))?;

    let manifest = serde_json::json!({
        "models": {
            "checksum-fixture": {
                "description": "checksum fixture",
                "source": "url",
                "size_mb": 1,
                "files": [{
                    "url": "https://example.invalid/fixture.gguf",
                    "output": "fixture.gguf",
                    "sha256": file_sha256
                }],
                "model_metadata": {
                    "model_id": "checksum-fixture",
                    "version": "1.0",
                    "execution_template": {
                        "type": "Gguf",
                        "model_file": "fixture.gguf"
                    },
                    "files": ["fixture.gguf"],
                    "metadata": {
                        "sha256": metadata_sha256
                    }
                }
            }
        }
    });
    fs::write(
        models_dir.join("models.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    let fake_curl = fake_bin.join("curl");
    fs::write(&fake_curl, fake_curl_script)?;
    fs::set_permissions(&fake_curl, fs::Permissions::from_mode(0o755))?;

    let existing_path = env::var_os("PATH").unwrap_or_default();
    let path =
        env::join_paths(std::iter::once(fake_bin.clone()).chain(env::split_paths(&existing_path)))?;
    let output = Command::new("bash")
        .arg(&downloader)
        .arg("checksum-fixture")
        .env("PATH", path)
        .output()?;

    Ok(DownloadRun {
        _temp: temp,
        model_dir: models_dir.join("checksum-fixture"),
        output,
    })
}

#[test]
fn direct_url_download_rejects_sha256_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    // Given a manifest whose declared hashes do not match the downloaded bytes.
    // When the normal downloader fetches that model.
    let run = run_download(
        Some(MISMATCH_SHA256),
        Some(MISMATCH_SHA256),
        SUCCESSFUL_CURL,
    )?;

    // Then it rejects and removes the complete partial model directory.
    let stdout = String::from_utf8_lossy(&run.output.stdout);
    assert!(
        !run.output.status.success(),
        "download unexpectedly passed:\n{stdout}"
    );
    assert!(stdout.contains("SHA-256 mismatch"), "output was:\n{stdout}");
    assert!(!run.model_dir.exists());
    Ok(())
}

#[test]
fn direct_url_download_requires_declared_model_checksum_per_file(
) -> Result<(), Box<dyn std::error::Error>> {
    // Given model metadata that declares a checksum but a file entry that omits it.
    // When the normal downloader fetches that model.
    let run = run_download(None, Some(PAYLOAD_SHA256), SUCCESSFUL_CURL)?;

    // Then it rejects the incomplete manifest and leaves no partial model.
    let stdout = String::from_utf8_lossy(&run.output.stdout);
    assert!(!run.output.status.success(), "output was:\n{stdout}");
    assert!(stdout.contains("missing SHA-256"), "output was:\n{stdout}");
    assert!(!run.model_dir.exists());
    Ok(())
}

#[test]
fn direct_url_download_accepts_matching_sha256() -> Result<(), Box<dyn std::error::Error>> {
    // Given matching per-file and model metadata checksums.
    // When the normal downloader fetches those exact bytes.
    let run = run_download(Some(PAYLOAD_SHA256), Some(PAYLOAD_SHA256), SUCCESSFUL_CURL)?;

    // Then the model download succeeds and keeps the verified artifact.
    let stdout = String::from_utf8_lossy(&run.output.stdout);
    assert!(run.output.status.success(), "output was:\n{stdout}");
    assert!(run.model_dir.join("fixture.gguf").is_file());
    Ok(())
}

#[test]
fn direct_url_download_removes_partial_directory_on_transfer_failure(
) -> Result<(), Box<dyn std::error::Error>> {
    let run = run_download(Some(PAYLOAD_SHA256), Some(PAYLOAD_SHA256), FAILING_CURL)?;

    let stdout = String::from_utf8_lossy(&run.output.stdout);
    assert!(!run.output.status.success(), "output was:\n{stdout}");
    assert!(stdout.contains("download failed"), "output was:\n{stdout}");
    assert!(!run.model_dir.exists());
    Ok(())
}

#[test]
fn direct_url_download_removes_partial_directory_on_invalid_response(
) -> Result<(), Box<dyn std::error::Error>> {
    let run = run_download(
        Some(PAYLOAD_SHA256),
        Some(PAYLOAD_SHA256),
        INVALID_RESPONSE_CURL,
    )?;

    let stdout = String::from_utf8_lossy(&run.output.stdout);
    assert!(!run.output.status.success(), "output was:\n{stdout}");
    assert!(stdout.contains("invalid response"), "output was:\n{stdout}");
    assert!(!run.model_dir.exists());
    Ok(())
}

#[test]
fn direct_url_download_allows_legacy_file_without_sha256() -> Result<(), Box<dyn std::error::Error>>
{
    let run = run_download(None, None, SUCCESSFUL_CURL)?;

    let stdout = String::from_utf8_lossy(&run.output.stdout);
    assert!(run.output.status.success(), "output was:\n{stdout}");
    assert!(run.model_dir.join("fixture.gguf").is_file());
    Ok(())
}

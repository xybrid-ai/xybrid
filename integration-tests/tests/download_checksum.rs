#![cfg(unix)]

use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

#[test]
fn direct_url_download_rejects_sha256_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    // Given a direct-download manifest with a checksum that does not match the
    // bytes returned by curl.
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
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
                }],
                "model_metadata": {
                    "model_id": "checksum-fixture",
                    "version": "1.0",
                    "execution_template": {
                        "type": "Gguf",
                        "model_file": "fixture.gguf"
                    },
                    "files": ["fixture.gguf"]
                }
            }
        }
    });
    fs::write(
        models_dir.join("models.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    let fake_curl = fake_bin.join("curl");
    fs::write(
        &fake_curl,
        r#"#!/usr/bin/env bash
set -euo pipefail
while (( $# > 0 )); do
    if [[ "$1" == "-o" ]]; then
        printf 'fixture model bytes' > "$2"
        exit 0
    fi
    shift
done
exit 1
"#,
    )?;
    fs::set_permissions(&fake_curl, fs::Permissions::from_mode(0o755))?;

    let existing_path = env::var_os("PATH").unwrap_or_default();
    let path =
        env::join_paths(std::iter::once(fake_bin.clone()).chain(env::split_paths(&existing_path)))?;

    // When the normal downloader fetches that model.
    let output = Command::new("bash")
        .arg(&downloader)
        .arg("checksum-fixture")
        .env("PATH", path)
        .output()?;

    // Then it rejects and removes the untrusted artifact.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "download unexpectedly passed:\n{stdout}"
    );
    assert!(stdout.contains("SHA-256 mismatch"), "output was:\n{stdout}");
    assert!(!models_dir.join("checksum-fixture/fixture.gguf").exists());
    Ok(())
}

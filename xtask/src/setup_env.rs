use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use anyhow::{Context, Result, bail};

pub fn run(registry_url: Option<String>) -> Result<()> {
    let root = project_root();
    let download_script = root.join("integration-tests/download.sh");
    
    if !download_script.exists() {
        bail!("download.sh not found at {}", download_script.display());
    }

    println!("Executing download script: {}", download_script.display());

    let mut cmd = Command::new(&download_script);
    
    // Pass --all to download everything, as per bootstrap recipe
    cmd.arg("--all");
    
    // We could pass registry URL if the script supported overriding it via env var or arg,
    // but the current script hardcodes it or just uses what's there. 
    // The user's script doesn't seem to take a registry arg easily without modification,
    // but we are wrapping the existing script.
    if let Some(reg) = registry_url {
        println!("Note: Custom registry URL '{}' passed but download.sh might need update to use it.", reg);
        // cmd.env("REGISTRY_API", reg); // Uncomment if we modify script to use env var
    }

    cmd.stdout(Stdio::inherit())
       .stderr(Stdio::inherit());

    let status = cmd.status().context("Failed to execute download.sh")?;

    if !status.success() {
        bail!("download.sh failed with exit code: {}", status);
    }
    
    Ok(())
}

fn project_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf()
}

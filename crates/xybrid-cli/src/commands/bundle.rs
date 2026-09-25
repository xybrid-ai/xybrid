//! `xybrid bundle` command handler.

use anyhow::{Context, Result};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::{Path, PathBuf};
use xybrid_core::bundler::XyBundle;
use xybrid_sdk::registry_client::RegistryClient;

use super::utils::format_size;

/// Drive the bundle command's progress bar from an SDK download status.
///
/// Re-sizes the bar when the SDK reports a total: a multi-file model's real
/// total spans every artifact, which `resolved.size_bytes` does not cover.
fn apply_download_status(pb: &ProgressBar, status: &xybrid_sdk::DownloadStatus) {
    if let Some(total) = status.total_bytes {
        if pb.length() != Some(total) {
            pb.set_length(total);
        }
    }
    pb.set_position(status.downloaded_bytes);
}

/// Handle `xybrid bundle` command: fetch a registry model and produce a .xyb bundle.
pub(crate) fn handle_bundle_command(
    model_id: &str,
    output: Option<PathBuf>,
    platform: Option<&str>,
) -> Result<()> {
    println!("📦 Xybrid Bundle");
    println!("{}", "=".repeat(60));
    println!("Model: {}", model_id.cyan().bold());
    if let Some(p) = platform {
        println!("Platform: {}", p);
    } else {
        println!("Platform: auto-detect");
    }
    println!();

    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    println!("Resolving {}...", model_id);
    let resolved = client
        .resolve(model_id, platform)
        .context(format!("Failed to resolve model '{}'", model_id))?;

    println!("   Repository: {}", resolved.hf_repo);
    println!("   File: {}", resolved.file);
    println!(
        "   Size: {}",
        format_size(resolved.size_bytes).bright_cyan()
    );
    println!("   Format: {} ({})", resolved.format, resolved.quantization);
    println!(
        "   Type: {}",
        if resolved.passthrough {
            "passthrough"
        } else {
            "bundle (.xyb)"
        }
    );
    println!();

    let target = platform.unwrap_or("universal");

    if !resolved.passthrough {
        bundle_from_cache(&client, model_id, platform, &output, target, &resolved)?;
    } else {
        bundle_from_passthrough(&client, model_id, platform, &output, target, &resolved)?;
    }

    println!();
    println!("{}", "=".repeat(60));
    Ok(())
}

fn bundle_from_cache(
    client: &RegistryClient,
    model_id: &str,
    platform: Option<&str>,
    output: &Option<PathBuf>,
    target: &str,
    resolved: &xybrid_sdk::registry_client::ResolvedVariant,
) -> Result<()> {
    let cache_path = client.get_cache_path(resolved);

    if !cache_path.exists() {
        let pb = create_progress_bar(resolved.size_bytes, model_id);

        client
            .fetch(model_id, platform, |status| {
                apply_download_status(&pb, &status);
            })
            .context(format!("Failed to fetch model '{}'", model_id))?;

        pb.finish_with_message(format!("✅ Downloaded {}", model_id));
        println!();
    } else {
        println!("✅ Model already cached at {}", cache_path.display());
    }

    let cached_bundle = XyBundle::load(&cache_path)
        .with_context(|| format!("Failed to load cached bundle: {}", cache_path.display()))?;
    let version = cached_bundle.manifest().version.clone();

    let out_path = resolve_bundle_output(output, model_id, &version, target)?;

    fs::copy(&cache_path, &out_path).with_context(|| {
        format!(
            "Failed to copy bundle from {} to {}",
            cache_path.display(),
            out_path.display()
        )
    })?;

    let out_size = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!();
    println!("✅ Bundle written: {}", out_path.display());
    println!("   Size: {}", format_size(out_size));
    println!("   Files: {}", cached_bundle.manifest().files.len());

    Ok(())
}

fn bundle_from_passthrough(
    client: &RegistryClient,
    model_id: &str,
    platform: Option<&str>,
    output: &Option<PathBuf>,
    target: &str,
    resolved: &xybrid_sdk::registry_client::ResolvedVariant,
) -> Result<()> {
    let pb = create_progress_bar(resolved.size_bytes, model_id);

    let extract_dir = client
        .fetch_extracted(model_id, platform, |status| {
            apply_download_status(&pb, &status);
        })
        .context(format!("Failed to fetch model '{}'", model_id))?;

    pb.finish_with_message(format!("✅ Downloaded {}", model_id));
    println!();

    let version = read_model_version(&extract_dir).unwrap_or_else(|| "1.0".to_string());

    println!("Bundling into .xyb...");
    let mut bundle = XyBundle::new(model_id, &version, target);

    let mut files_to_add = Vec::new();
    visit_dir(&extract_dir, &mut files_to_add)?;

    if files_to_add.is_empty() {
        return Err(anyhow::anyhow!(
            "No files found in extracted directory: {}",
            extract_dir.display()
        ));
    }

    let mut added = 0usize;
    for file in &files_to_add {
        let rel_path = file
            .strip_prefix(&extract_dir)
            .unwrap_or(file.as_path())
            .to_string_lossy()
            .to_string();

        if rel_path.ends_with(".sha256") {
            continue;
        }

        bundle
            .add_file_with_relative_path(file, &rel_path)
            .with_context(|| format!("Failed to add file: {}", file.display()))?;

        let file_size = fs::metadata(file).map(|m| m.len()).unwrap_or(0);
        println!("  Added: {} ({})", rel_path, format_size(file_size));
        added += 1;
    }

    if added == 0 {
        return Err(anyhow::anyhow!(
            "No files added to bundle from {}",
            extract_dir.display()
        ));
    }

    let out_path = resolve_bundle_output(output, model_id, &version, target)?;

    bundle
        .write(&out_path)
        .with_context(|| format!("Failed to write bundle: {}", out_path.display()))?;

    let out_size = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!();
    println!("✅ Bundle written: {}", out_path.display());
    println!("   Size: {} (compressed)", format_size(out_size));
    println!("   Files: {}", added);

    Ok(())
}

fn create_progress_bar(size_bytes: u64, model_id: &str) -> ProgressBar {
    let pb = ProgressBar::new(size_bytes);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} Downloading {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("█▓▒░  ")
    );
    pb.set_message(model_id.to_string());
    pb
}

/// Resolve the output path for a bundle command.
fn resolve_bundle_output(
    output: &Option<PathBuf>,
    model_id: &str,
    version: &str,
    target: &str,
) -> Result<PathBuf> {
    let out_path = if let Some(out) = output {
        out.clone()
    } else {
        let mut dist_dir = std::env::current_dir().context("Failed to get current directory")?;
        dist_dir.push("dist");
        dist_dir.join(format!("{}-{}-{}.xyb", model_id, version, target))
    };

    if let Some(parent) = out_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
    }

    if out_path.exists() {
        println!("⚠️  Overwriting existing file: {}", out_path.display());
    }

    Ok(out_path)
}

/// Read the version from model_metadata.json in a directory.
pub(crate) fn read_model_version(dir: &Path) -> Option<String> {
    let metadata_path = dir.join("model_metadata.json");
    let content = fs::read_to_string(&metadata_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("version")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        fs::read_dir(dir).with_context(|| format!("Failed to read dir: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_dir(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

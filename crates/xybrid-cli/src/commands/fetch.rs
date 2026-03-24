//! `xybrid fetch` command handler.

use anyhow::{Context, Result};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::Path;

use super::utils::format_size;

/// Handle `xybrid fetch --model <id>` command.
pub(crate) fn handle_fetch_command(model_id: &str, platform: Option<&str>) -> Result<()> {
    println!("📥 Fetching model: {}", model_id.cyan().bold());
    if let Some(p) = platform {
        println!("   Platform: {}", p);
    } else {
        println!("   Platform: auto-detect");
    }
    println!("{}", "=".repeat(60));
    println!();

    let client = xybrid_sdk::registry_client::RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    let resolved = client
        .resolve(model_id, platform)
        .context(format!("Failed to resolve model '{}'", model_id))?;

    print_resolved_variant(&resolved);

    if client
        .is_cached(model_id, platform)
        .context("Failed to check cache status")?
    {
        println!("✅ Model is already cached and verified");
        let cache_path = client.get_cache_path(&resolved);
        println!("   Location: {}", cache_path.display());
        return Ok(());
    }

    let pb = create_download_progress_bar(resolved.size_bytes, model_id);

    let bundle_path = client
        .fetch(model_id, platform, |progress| {
            let bytes_done = (progress * resolved.size_bytes as f32) as u64;
            pb.set_position(bytes_done);
        })
        .context(format!("Failed to fetch model '{}'", model_id))?;

    pb.finish_with_message(format!("✅ Downloaded {}", model_id));
    println!();
    println!("✅ Model downloaded successfully!");
    println!("   Location: {}", bundle_path.display());
    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}

/// Handle `xybrid fetch --huggingface <repo>` command.
///
/// Downloads a model directly from HuggingFace Hub and auto-generates metadata.
pub(crate) fn handle_fetch_huggingface_command(repo: &str) -> Result<()> {
    println!("📥 Fetching from HuggingFace: {}", repo.cyan().bold());
    println!("{}", "=".repeat(60));
    println!();

    // Compute cache dir to check for auto-generated metadata after load
    let sanitized = repo.replace('/', "--");
    let cache_dir =
        dirs::home_dir().map(|h| h.join(".xybrid").join("cache").join("hf").join(&sanitized));

    let loader = xybrid_sdk::ModelLoader::from_huggingface(repo);
    let model = loader.load().context(format!(
        "Failed to load model from HuggingFace repo '{}'",
        repo
    ))?;

    println!("✅ Model downloaded successfully!");
    println!("   Model ID: {}", model.model_id().cyan());
    println!("   Version: {}", model.version());

    if let Some(ref dir) = cache_dir {
        println!("   Directory: {}", dir.display());

        let metadata_path = dir.join("model_metadata.json");
        if metadata_path.exists() {
            if let Ok(content) = fs::read_to_string(&metadata_path) {
                if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&content) {
                    if metadata.get("auto_generated").and_then(|v| v.as_bool()) == Some(true) {
                        println!();
                        println!(
                            "{}",
                            "⚠️  model_metadata.json was auto-generated. Review and adjust if needed:"
                                .yellow()
                        );
                        println!("   {}", metadata_path.display());
                    }
                }
            }
        }
    }

    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}

/// Handle `xybrid fetch <pipeline.yaml>` command.
///
/// Pre-downloads all models required by the pipeline.
pub(crate) fn handle_fetch_pipeline_command(
    config_path: &Path,
    platform: Option<&str>,
) -> Result<()> {
    println!();

    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Pipeline config not found: {}",
            config_path.display()
        ));
    }

    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = xybrid_core::pipeline_config::PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    let client = xybrid_sdk::registry_client::RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    let pipeline_name = config.name.as_deref().unwrap_or(
        config_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pipeline"),
    );
    println!("Fetching models for: {}", pipeline_name.cyan().bold());
    println!("{}", "━".repeat(60));
    println!();

    let models_to_fetch: Vec<String> = config
        .stages
        .iter()
        .filter(|stage| !stage.is_cloud_stage())
        .map(|stage| stage.model_id())
        .collect();

    if models_to_fetch.is_empty() {
        println!("ℹ️  No device models to fetch in this pipeline.");
        return Ok(());
    }

    let (success_count, skip_count, error_count) =
        fetch_models(&client, &models_to_fetch, platform)?;

    println!();
    println!("{}", "━".repeat(60));

    if error_count == 0 {
        println!(
            "✅ All models ready ({} downloaded, {} cached)",
            success_count, skip_count
        );
    } else {
        println!(
            "⚠️  Completed with errors: {} downloaded, {} cached, {} failed",
            success_count, skip_count, error_count
        );
    }

    Ok(())
}

fn print_resolved_variant(resolved: &xybrid_sdk::registry_client::ResolvedVariant) {
    println!("📦 Resolved variant:");
    println!("   Repository: {}", resolved.hf_repo);
    println!("   File: {}", resolved.file);
    println!(
        "   Size: {}",
        format_size(resolved.size_bytes).bright_cyan()
    );
    println!("   Format: {} ({})", resolved.format, resolved.quantization);
    println!();
}

fn create_download_progress_bar(size_bytes: u64, model_id: &str) -> ProgressBar {
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

fn fetch_models(
    client: &xybrid_sdk::registry_client::RegistryClient,
    models: &[String],
    platform: Option<&str>,
) -> Result<(usize, usize, usize)> {
    let mut success_count = 0;
    let mut skip_count = 0;
    let mut error_count = 0;

    for model_id in models {
        match client.is_cached(model_id, platform) {
            Ok(true) => {
                println!("{} {} (cached)", "✅".bright_green(), model_id.cyan());
                skip_count += 1;
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                println!(
                    "{} {} (cache check failed: {})",
                    "❌".bright_red(),
                    model_id,
                    e
                );
                error_count += 1;
                continue;
            }
        }

        match client.resolve(model_id, platform) {
            Ok(resolved) => {
                let pb = ProgressBar::new(resolved.size_bytes);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                        .unwrap()
                        .progress_chars("█▓▒░  ")
                );
                pb.set_message(model_id.clone());

                match client.fetch(model_id, platform, |progress| {
                    let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                    pb.set_position(bytes_done);
                }) {
                    Ok(_) => {
                        pb.finish_with_message(format!("{} ✓", model_id));
                        success_count += 1;
                    }
                    Err(e) => {
                        pb.abandon_with_message(format!("{} ✗ {}", model_id, e));
                        error_count += 1;
                    }
                }
            }
            Err(e) => {
                println!(
                    "{} {} (resolution failed: {})",
                    "❌".bright_red(),
                    model_id,
                    e
                );
                error_count += 1;
            }
        }
    }

    Ok((success_count, skip_count, error_count))
}

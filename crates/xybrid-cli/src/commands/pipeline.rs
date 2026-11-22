//! `xybrid prepare` and `xybrid plan` command handlers.

use anyhow::{Context, Result};
use colored::*;
use std::fs;
use std::path::Path;
use xybrid_core::pipeline_config::PipelineConfig;
use xybrid_sdk::registry_client::RegistryClient;

use super::utils::format_size;

/// Handle `xybrid prepare <pipeline.yaml>` command.
///
/// Parses and validates the pipeline configuration.
pub(crate) fn handle_prepare_command(config_path: &Path) -> Result<()> {
    println!("📋 Xybrid Pipeline Prepare");
    println!("{}", "=".repeat(60));
    println!();

    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Pipeline config not found: {}",
            config_path.display()
        ));
    }

    println!("📂 Loading: {}", config_path.display());
    println!();

    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    println!("✅ Pipeline configuration is valid");
    println!();

    if let Some(name) = &config.name {
        println!("  Name:     {}", name.cyan().bold());
    }

    println!("  Registry: {}", config.registry_url());
    println!("  Stages:   {}", config.stage_count());
    println!();

    print_stage_details(&config);

    println!();
    println!("{}", "=".repeat(60));
    println!("✅ Pipeline is ready for execution");
    println!();
    println!("Next steps:");
    println!(
        "  xybrid plan {}   # Show execution plan with model status",
        config_path.display()
    );
    println!(
        "  xybrid fetch {}  # Pre-download all models",
        config_path.display()
    );
    println!(
        "  xybrid run -c {} # Execute the pipeline",
        config_path.display()
    );

    Ok(())
}

fn print_stage_details(config: &PipelineConfig) {
    println!("📦 Stages:");
    for (i, stage) in config.stages.iter().enumerate() {
        println!("  {}. {}", i + 1, stage.stage_id().cyan().bold());
        println!("     Model:  {}", stage.model_id());
        if let Some(version) = stage.version() {
            println!("     Version: {}", version);
        }
        if let Some(target) = stage.target() {
            let target_colored = match target {
                "device" => target.bright_green(),
                "cloud" => target.bright_blue(),
                "integration" => target.bright_magenta(),
                _ => target.white(),
            };
            println!("     Target: {}", target_colored);
        }
        if let Some(provider) = stage.provider() {
            println!("     Provider: {}", provider);
        }
        if let Some(message) = stage.unapplied_execution_provider_warning() {
            crate::ui::warning(&message);
        }
    }
}

/// Handle `xybrid plan <pipeline.yaml>` command.
///
/// Shows execution plan with model resolution status.
pub(crate) fn handle_plan_command(config_path: &Path) -> Result<()> {
    println!();

    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Pipeline config not found: {}",
            config_path.display()
        ));
    }

    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    let pipeline_name = config.name.as_deref().unwrap_or(
        config_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pipeline"),
    );
    println!("Pipeline: {}", pipeline_name.cyan().bold());
    println!("{}", "━".repeat(60));
    println!();

    let mut total_download_bytes: u64 = 0;
    let mut requires_network = false;
    let mut all_cached = true;

    for (i, stage) in config.stages.iter().enumerate() {
        println!("Stage {}: {}", i + 1, stage.stage_id().cyan().bold());

        if stage.is_cloud_stage() {
            print_cloud_stage(stage);
            requires_network = true;
        } else {
            let (download_bytes, cached) = print_device_stage(stage, &client)?;
            total_download_bytes += download_bytes;
            if !cached {
                all_cached = false;
            }
        }

        println!();
    }

    print_plan_summary(
        total_download_bytes,
        all_cached,
        requires_network,
        config_path,
    );

    Ok(())
}

fn print_cloud_stage(stage: &xybrid_core::pipeline_config::StageConfig) {
    if let Some(provider) = stage.provider() {
        println!("  Target:   {} ({})", "cloud".bright_magenta(), provider);
    } else {
        println!("  Target:   {}", "cloud".bright_magenta());
    }
    println!("  Status:   {} Requires network", "🌐".bright_blue());
}

fn print_device_stage(
    stage: &xybrid_core::pipeline_config::StageConfig,
    client: &RegistryClient,
) -> Result<(u64, bool)> {
    let model_id = stage.model_id();
    let target = stage.target().unwrap_or("device");
    println!("  Model:    {}", model_id);

    match client.resolve(&model_id, None) {
        Ok(resolved) => {
            let size_str = format_size(resolved.size_bytes);
            println!(
                "  Variant:  {} ({}, {})",
                resolved.file,
                size_str.bright_black(),
                format!("{}/{}", resolved.format, resolved.quantization).bright_black()
            );

            let target_colored = match target {
                "device" => target.bright_green(),
                "cloud" => target.bright_blue(),
                _ => target.white(),
            };
            println!("  Target:   {}", target_colored);

            match client.is_cached(&model_id, None) {
                Ok(true) => {
                    println!("  Status:   {} Cached", "✅".bright_green());
                    Ok((0, true))
                }
                Ok(false) => {
                    println!(
                        "  Status:   {} Not cached ({} to download)",
                        "⬇️".bright_yellow(),
                        size_str.bright_cyan()
                    );
                    Ok((resolved.size_bytes, false))
                }
                Err(e) => {
                    println!(
                        "  Status:   {} Cache check failed: {}",
                        "❌".bright_red(),
                        e
                    );
                    Ok((0, false))
                }
            }
        }
        Err(e) => {
            println!("  Status:   {} Resolution failed: {}", "❌".bright_red(), e);
            Ok((0, false))
        }
    }
}

fn print_plan_summary(
    total_download_bytes: u64,
    all_cached: bool,
    requires_network: bool,
    config_path: &Path,
) {
    println!("{}", "━".repeat(60));

    if total_download_bytes > 0 {
        println!(
            "Total download: {}",
            format_size(total_download_bytes).bright_cyan()
        );
    } else if all_cached {
        println!(
            "Total download: {} (all models cached)",
            "0 bytes".bright_green()
        );
    }

    let offline_capable = !requires_network && all_cached;
    if offline_capable {
        println!("Offline capable: {}", "Yes".bright_green());
    } else if requires_network {
        println!(
            "Offline capable: {} (cloud stages require network)",
            "No".bright_yellow()
        );
    } else {
        println!(
            "Offline capable: {} (models need downloading)",
            "No".bright_yellow()
        );
    }

    println!();

    if total_download_bytes > 0 {
        println!(
            "Run `xybrid fetch {}` to pre-download models.",
            config_path.display()
        );
    }
}

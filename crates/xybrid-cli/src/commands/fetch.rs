//! `xybrid fetch` command handler.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use super::utils::format_size;
use crate::ui;

/// Handle `xybrid fetch --model <id>` command.
pub(crate) fn handle_fetch_command(model_id: &str, platform: Option<&str>) -> Result<()> {
    ui::header(&format!("Fetch · {}", model_id));

    if let Some(p) = platform {
        ui::kv("Platform", p);
    } else {
        ui::kv("Platform", "auto-detect");
    }

    let client = xybrid_sdk::registry_client::RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    let resolved = client
        .resolve(model_id, platform)
        .context(format!("Failed to resolve model '{}'", model_id))?;

    print_resolved_variant(&resolved);

    if let Some(cache_path) = cache_location(&client, model_id, platform, &resolved)
        .context("Failed to check cache status")?
    {
        ui::ok("Model is already cached and verified");
        ui::kv("Location", &cache_path.display().to_string());
        return Ok(());
    }

    let pb = ui::download_bar(resolved.size_bytes, model_id);

    let model_path = fetch_resolved_model(&client, model_id, platform, &resolved, |progress| {
        let bytes_done = (progress * resolved.size_bytes as f32) as u64;
        pb.set_position(bytes_done);
    })
    .context(format!("Failed to fetch model '{}'", model_id))?;

    pb.finish_and_clear();
    println!();
    ui::ok("Model downloaded successfully");
    ui::kv("Location", &model_path.display().to_string());
    println!();

    Ok(())
}

/// Handle `xybrid fetch --huggingface <repo>` command.
pub(crate) fn handle_fetch_huggingface_command(repo: &str) -> Result<()> {
    ui::header(&format!("Fetch · HuggingFace · {}", repo));

    let loader = xybrid_sdk::ModelLoader::from_huggingface_parsed(repo);
    let model = loader.load().context(format!(
        "Failed to load model from HuggingFace repo '{}'",
        repo
    ))?;

    ui::ok("Model downloaded successfully");
    ui::kv("Model ID", model.model_id());
    ui::kv("Version", model.version());

    let cache_repo = xybrid_sdk::ModelSource::parse_huggingface(repo);
    let repo_id = cache_repo.model_id().unwrap_or(repo);
    let cache_dir = xybrid_sdk::CacheManager::new()
        .ok()
        .and_then(|manager| manager.huggingface_cache_dir(repo_id));

    if let Some(ref dir) = cache_dir {
        ui::kv("Directory", &dir.display().to_string());

        let metadata_path = dir.join("model_metadata.json");
        if metadata_path.exists() {
            if let Ok(content) = fs::read_to_string(&metadata_path) {
                if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&content) {
                    if metadata.get("auto_generated").and_then(|v| v.as_bool()) == Some(true) {
                        println!();
                        ui::warning(
                            "model_metadata.json was auto-generated. Review and adjust if needed:",
                        );
                        ui::hint(&metadata_path.display().to_string());
                    }
                }
            }
        }
    }

    println!();

    Ok(())
}

/// Handle `xybrid fetch <pipeline.yaml>` command.
pub(crate) fn handle_fetch_pipeline_command(
    config_path: &Path,
    platform: Option<&str>,
) -> Result<()> {
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
    ui::header(&format!("Fetch Pipeline · {}", pipeline_name));

    let models_to_fetch: Vec<String> = config
        .stages
        .iter()
        .filter(|stage| !stage.is_cloud_stage())
        .map(|stage| stage.model_id())
        .collect();

    if models_to_fetch.is_empty() {
        ui::hint("No device models to fetch in this pipeline.");
        return Ok(());
    }

    println!();
    let (success_count, skip_count, error_count) =
        fetch_models(&client, &models_to_fetch, platform)?;

    println!();

    if error_count == 0 {
        ui::ok(&format!(
            "All models ready ({} downloaded, {} cached)",
            success_count, skip_count
        ));
    } else {
        ui::warning(&format!(
            "Completed with errors: {} downloaded, {} cached, {} failed",
            success_count, skip_count, error_count
        ));
    }

    println!();

    Ok(())
}

fn print_resolved_variant(resolved: &xybrid_sdk::registry_client::ResolvedVariant) {
    println!();
    ui::kv("Repository", &resolved.hf_repo);
    ui::kv("File", &resolved.file);
    ui::kv("Size", &format_size(resolved.size_bytes));
    ui::kv(
        "Format",
        &format!("{} ({})", resolved.format, resolved.quantization),
    );
    println!();
}

fn uses_extracted_model_path(resolved: &xybrid_sdk::registry_client::ResolvedVariant) -> bool {
    resolved.passthrough
}

fn cache_location(
    client: &xybrid_sdk::registry_client::RegistryClient,
    model_id: &str,
    platform: Option<&str>,
    resolved: &xybrid_sdk::registry_client::ResolvedVariant,
) -> Result<Option<std::path::PathBuf>> {
    if uses_extracted_model_path(resolved) {
        return Ok(client.resolve_offline(model_id));
    }

    if client
        .is_cached(model_id, platform)
        .context("Failed to check bundle cache status")?
    {
        return Ok(Some(client.get_cache_path(resolved)));
    }

    Ok(None)
}

fn fetch_resolved_model<F>(
    client: &xybrid_sdk::registry_client::RegistryClient,
    model_id: &str,
    platform: Option<&str>,
    resolved: &xybrid_sdk::registry_client::ResolvedVariant,
    progress_callback: F,
) -> Result<std::path::PathBuf>
where
    F: Fn(f32),
{
    if uses_extracted_model_path(resolved) {
        client
            .fetch_extracted(model_id, platform, progress_callback)
            .context(format!("Failed to fetch passthrough model '{}'", model_id))
    } else {
        client
            .fetch(model_id, platform, progress_callback)
            .context(format!("Failed to fetch model '{}'", model_id))
    }
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
        match client.resolve(model_id, platform) {
            Ok(resolved) => {
                match cache_location(client, model_id, platform, &resolved) {
                    Ok(Some(_)) => {
                        ui::ok(&format!("{} (cached)", model_id));
                        skip_count += 1;
                        continue;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        ui::err(&format!("{} (cache check failed: {})", model_id, e));
                        error_count += 1;
                        continue;
                    }
                }

                let pb = ui::download_bar(resolved.size_bytes, model_id);

                match fetch_resolved_model(client, model_id, platform, &resolved, |progress| {
                    let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                    pb.set_position(bytes_done);
                }) {
                    Ok(_) => {
                        pb.finish_and_clear();
                        ui::ok(model_id);
                        success_count += 1;
                    }
                    Err(e) => {
                        pb.abandon();
                        ui::err(&format!("{} ({})", model_id, e));
                        error_count += 1;
                    }
                }
            }
            Err(e) => {
                ui::err(&format!("{} (resolution failed: {})", model_id, e));
                error_count += 1;
            }
        }
    }

    Ok((success_count, skip_count, error_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved_variant(passthrough: bool) -> xybrid_sdk::registry_client::ResolvedVariant {
        xybrid_sdk::registry_client::ResolvedVariant {
            hf_repo: "prism-ml/Bonsai-27B-gguf".to_string(),
            file: "Bonsai-27B-Q1_0.gguf".to_string(),
            download_url: "https://example.test/Bonsai-27B-Q1_0.gguf".to_string(),
            format: "gguf".to_string(),
            quantization: "q1_0_g128".to_string(),
            size_bytes: 1,
            sha256: "a".repeat(64),
            artifacts: Vec::new(),
            passthrough,
            model_metadata: None,
        }
    }

    #[test]
    fn passthrough_variants_use_the_extracted_model_path() {
        assert!(uses_extracted_model_path(&resolved_variant(true)));
        assert!(!uses_extracted_model_path(&resolved_variant(false)));
    }
}

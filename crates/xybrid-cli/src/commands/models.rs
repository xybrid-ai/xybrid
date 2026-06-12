//! `xybrid models` command handlers.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use xybrid_core::bundler::XyBundle;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::runtime_adapter::BackendChoice;
use xybrid_sdk::model::SdkError;
use xybrid_sdk::registry_client::{
    registry_format_for_backend, ModelDetail, ModelSummary, RegistryClient,
};

use super::types::ModelsCommand;
use super::utils::{format_params, format_size};
use crate::ui;

/// Handle `xybrid models` subcommands.
pub(crate) fn handle_models_command(command: ModelsCommand) -> Result<()> {
    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    match command {
        ModelsCommand::List { backend } => list_models(&client, backend.as_deref()),
        ModelsCommand::Search { query } => search_models(&client, &query),
        ModelsCommand::Info { model_id } => show_model_info(&client, &model_id),
        ModelsCommand::Voices { model_id } => handle_voices_command(&client, &model_id),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackendModelFilter {
    backend: BackendChoice,
    format: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VariantDisplay {
    All {
        variants: Vec<String>,
    },
    Compatible {
        format: &'static str,
        variants: Vec<String>,
    },
    Unknown {
        variants: Vec<String>,
    },
}

impl VariantDisplay {
    fn is_empty_known_match(&self) -> bool {
        matches!(self, VariantDisplay::Compatible { variants, .. } if variants.is_empty())
    }

    fn line(&self) -> Option<String> {
        match self {
            VariantDisplay::All { variants } if variants.is_empty() => None,
            VariantDisplay::All { variants } => Some(format!("variants: {}", variants.join(", "))),
            VariantDisplay::Compatible { variants, .. } if variants.is_empty() => None,
            VariantDisplay::Compatible { format, variants } => Some(format!(
                "compatible variants ({}): {}",
                format,
                variants.join(", ")
            )),
            VariantDisplay::Unknown { variants } if variants.is_empty() => {
                Some("variants: format metadata unavailable; not filtered".to_string())
            }
            VariantDisplay::Unknown { variants } => Some(format!(
                "variants: {} (format metadata unavailable; not filtered)",
                variants.join(", ")
            )),
        }
    }
}

fn parse_models_backend_filter(raw: Option<&str>) -> Result<Option<BackendModelFilter>> {
    let Some(raw) = raw else {
        return Ok(None);
    };

    let Some(backend) = BackendChoice::parse(raw)
        .map_err(|err| anyhow::anyhow!("Invalid backend filter: {}", err))?
    else {
        return Ok(None);
    };

    Ok(Some(BackendModelFilter {
        backend,
        format: listing_format_for_backend(backend),
    }))
}

fn listing_format_for_backend(backend: BackendChoice) -> &'static str {
    registry_format_for_backend(backend).unwrap_or(match backend {
        BackendChoice::Mistral => "gguf",
        BackendChoice::Mlx => "safetensors",
        BackendChoice::LlamaCpp => "gguf",
    })
}

fn compatible_variant_display(
    summary: &ModelSummary,
    detail: Option<&ModelDetail>,
    filter: &BackendModelFilter,
) -> VariantDisplay {
    let Some(detail) = detail.filter(|detail| !detail.variants.is_empty()) else {
        return VariantDisplay::Unknown {
            variants: summary.variants.clone(),
        };
    };

    let mut variants: Vec<String> = detail
        .variants
        .iter()
        .filter(|(_, info)| info.format.eq_ignore_ascii_case(filter.format))
        .map(|(name, _)| name.clone())
        .collect();
    variants.sort();

    VariantDisplay::Compatible {
        format: filter.format,
        variants,
    }
}

fn list_models(client: &RegistryClient, backend: Option<&str>) -> Result<()> {
    ui::header("Model Registry");

    let backend_filter = parse_models_backend_filter(backend)?;
    if let Some(filter) = backend_filter {
        ui::kv("Backend", filter.backend.as_str());
        ui::hint(&format!(
            "Showing registry variants compatible with {} artifacts.",
            filter.format
        ));
        println!();
    }

    // If the registry is reachable, show the full catalog. If we're offline,
    // fall back to listing the models that are already cached locally so the
    // user still sees something useful instead of a bare error.
    let models = match client.list_models() {
        Ok(models) => models,
        Err(SdkError::Offline { message, .. }) => {
            ui::warning(&format!("Registry unreachable ({}).", message));
            ui::hint("Showing models available offline from local cache:");
            return list_offline_models(client);
        }
        Err(e) => return Err(anyhow::Error::from(e).context("Failed to list models from registry")),
    };

    if models.is_empty() {
        ui::hint("No models found in registry.");
        return Ok(());
    }

    let mut by_task: BTreeMap<String, Vec<(&ModelSummary, VariantDisplay)>> = BTreeMap::new();
    for model in &models {
        let display = if let Some(filter) = backend_filter {
            let detail = client.get_model(&model.id).ok();
            compatible_variant_display(model, detail.as_ref(), &filter)
        } else {
            VariantDisplay::All {
                variants: model.variants.clone(),
            }
        };

        if display.is_empty_known_match() {
            continue;
        }

        by_task
            .entry(model.task.clone())
            .or_default()
            .push((model, display));
    }

    let listed_count: usize = by_task.values().map(Vec::len).sum();
    if listed_count == 0 {
        if let Some(filter) = backend_filter {
            ui::hint(&format!(
                "No models found with {} variants for backend '{}'.",
                filter.format,
                filter.backend.as_str()
            ));
        } else {
            ui::hint("No models found in registry.");
        }
        return Ok(());
    }

    for (task, task_models) in &by_task {
        ui::section(&task.to_uppercase());
        println!();

        for (model, variant_display) in task_models {
            let params_str = format_params(model.parameters);
            let meta = format!("{} · {} params", model.family, params_str);
            ui::bullet(&model.id, &meta);
            ui::sub(&model.description);
            if let Some(line) = variant_display.line() {
                ui::sub(&line);
            }
        }
    }

    ui::footer(&format!("{} models available", listed_count));

    Ok(())
}

/// Render the local-cache model listing for offline use.
///
/// Called as a fallback when the registry is unreachable. Shows every model
/// that's been downloaded and extracted on this machine — these are the ones
/// the user can run right now without needing a network.
fn list_offline_models(client: &RegistryClient) -> Result<()> {
    let ids = client.list_offline_models();

    if ids.is_empty() {
        println!();
        ui::hint("No models are currently cached on this machine.");
        ui::hint("Connect to the network and run a model to download it.");
        return Ok(());
    }

    println!();
    ui::section("CACHED LOCALLY");
    println!();
    for id in &ids {
        ui::bullet(id, "ready to run offline");
    }
    ui::footer(&format!("{} models available offline", ids.len()));
    Ok(())
}

fn search_models(client: &RegistryClient, query: &str) -> Result<()> {
    ui::header(&format!("Search: {}", query));

    let models = client
        .list_models()
        .context("Failed to list models from registry")?;

    let query_lower = query.to_lowercase();
    let matches: Vec<_> = models
        .iter()
        .filter(|m| {
            m.id.to_lowercase().contains(&query_lower)
                || m.family.to_lowercase().contains(&query_lower)
                || m.task.to_lowercase().contains(&query_lower)
                || m.description.to_lowercase().contains(&query_lower)
        })
        .collect();

    if matches.is_empty() {
        ui::hint(&format!("No models found matching '{}'", query));
        return Ok(());
    }

    println!();
    for model in matches.iter() {
        let params_str = format_params(model.parameters);
        let meta = format!("{} · {} · {} params", model.task, model.family, params_str);
        ui::bullet(&model.id, &meta);
        ui::sub(&model.description);
    }

    ui::footer(&format!("{} models found", matches.len()));

    Ok(())
}

fn show_model_info(client: &RegistryClient, model_id: &str) -> Result<()> {
    ui::header("Model Details");

    let model = client
        .get_model(model_id)
        .context(format!("Failed to get model '{}'", model_id))?;

    ui::panel(&[
        format!("{}", ui::accent(&model.id)),
        format!("{}", ui::dim(&model.description)),
    ]);

    println!();
    ui::kv_accent("ID", &model.id);
    ui::kv("Family", &model.family);
    ui::kv("Task", &model.task);
    ui::kv("Parameters", &format_params(model.parameters));

    if let Some(default) = &model.default_variant {
        ui::kv("Default", default);
    }

    if !model.variants.is_empty() {
        ui::section("Variants");
        println!();

        let mut table = ui::Table::new(vec!["Name", "Platform", "Format", "Quantization", "Size"]);
        for (name, info) in &model.variants {
            table.row(vec![
                name,
                &info.platform,
                &info.format,
                &info.quantization,
                &format_size(info.size_bytes),
            ]);
        }
        table.print();
    }

    if model.task.to_lowercase().contains("tts")
        || model.task.to_lowercase().contains("text-to-speech")
    {
        println!();
        ui::hint(&format!(
            "TTS model — run 'xybrid models voices {}' to see voices",
            model_id
        ));
    }

    println!();

    Ok(())
}

/// Handle `xybrid models voices <model-id>` command.
fn handle_voices_command(client: &RegistryClient, model_id: &str) -> Result<()> {
    ui::header(&format!("Voices · {}", model_id));

    let model = client
        .get_model(model_id)
        .context(format!("Failed to get model '{}'", model_id))?;

    if !model.task.to_lowercase().contains("tts")
        && !model.task.to_lowercase().contains("text-to-speech")
    {
        ui::hint(&format!(
            "Model '{}' is not a TTS model (task: {})",
            model_id, model.task
        ));
        ui::hint("Voice selection is only available for text-to-speech models.");
        return Ok(());
    }

    let resolved = client
        .resolve(model_id, None)
        .context(format!("Failed to resolve model '{}'", model_id))?;

    let bundle_path = if client.is_cached(model_id, None).unwrap_or(false) {
        client.get_cache_path(&resolved)
    } else {
        let pb = ui::download_bar(resolved.size_bytes, "Downloading voice catalog...");

        let path = client.fetch(model_id, None, |progress| {
            let bytes_done = (progress * resolved.size_bytes as f32) as u64;
            pb.set_position(bytes_done);
        })?;

        pb.finish_and_clear();
        path
    };

    let mut metadata = load_metadata_from_bundle(&bundle_path)?;
    metadata = try_local_fixtures_fallback(metadata, model_id);

    if !metadata.has_voices() {
        print_no_voices_hint(model_id);
        return Ok(());
    }

    let voices = metadata.list_voices();
    ui::ok(&format!("Found {} voices for {}", voices.len(), model_id));
    println!();

    print_voices_by_language(&voices);

    if let Some(default) = metadata.default_voice() {
        ui::kv(
            "Default voice",
            &format!("{} ({})", default.name, default.id),
        );
    }

    println!();
    ui::hint(&format!(
        "Usage: xybrid run --model {} --input-text \"Hello\" --voice <voice-id>",
        model_id
    ));
    println!();

    Ok(())
}

fn load_metadata_from_bundle(bundle_path: &Path) -> Result<ModelMetadata> {
    if bundle_path.is_dir() {
        let metadata_path = bundle_path.join("model_metadata.json");
        if !metadata_path.exists() {
            anyhow::bail!(
                "model_metadata.json not found at {}",
                metadata_path.display()
            );
        }
        let content = fs::read_to_string(&metadata_path)?;
        return Ok(serde_json::from_str(&content)?);
    }

    if bundle_path.extension().is_some_and(|ext| ext == "xyb") {
        let bundle = XyBundle::load(bundle_path)?;
        let metadata_json = bundle.get_metadata_json()?.ok_or_else(|| {
            anyhow::anyhow!(
                "model_metadata.json not found in bundle at {}",
                bundle_path.display()
            )
        })?;
        return Ok(serde_json::from_str(&metadata_json)?);
    }

    let metadata_path = bundle_path.join("model_metadata.json");
    if !metadata_path.exists() {
        anyhow::bail!(
            "model_metadata.json not found at {}",
            metadata_path.display()
        );
    }
    let content = fs::read_to_string(&metadata_path)?;
    Ok(serde_json::from_str(&content)?)
}

fn try_local_fixtures_fallback(mut metadata: ModelMetadata, model_id: &str) -> ModelMetadata {
    if metadata.has_voices() {
        return metadata;
    }

    let fixtures_path = PathBuf::from("integration-tests/fixtures/models")
        .join(model_id)
        .join("model_metadata.json");

    if fixtures_path.exists() {
        if let Ok(content) = fs::read_to_string(&fixtures_path) {
            if let Ok(local_metadata) = serde_json::from_str::<ModelMetadata>(&content) {
                if local_metadata.has_voices() {
                    ui::hint("Using voice catalog from local fixtures");
                    ui::hint("(Registry bundle may need updating)");
                    println!();
                    metadata = local_metadata;
                }
            }
        }
    }

    metadata
}

fn print_no_voices_hint(model_id: &str) {
    ui::hint(&format!(
        "Model '{}' does not have a voice catalog.",
        model_id
    ));
    println!();
    ui::hint("This TTS model may use a single default voice, or the");
    ui::hint("registry bundle needs to be updated with voice info.");
    println!();
    ui::hint("For local development with Kokoro, run:");
    ui::hint("  ./integration-tests/download.sh kokoro-82m");
    ui::hint("  cargo run -p xybrid-core --example tts_kokoro -- --list-voices");
}

fn print_voices_by_language(voices: &[&xybrid_core::execution_template::VoiceInfo]) {
    let mut by_language: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for voice in voices {
        let lang = voice.language.as_deref().unwrap_or("unknown").to_string();
        by_language.entry(lang).or_default().push(voice);
    }

    for (language, lang_voices) in by_language {
        let flag = match language.as_str() {
            "en-US" => "🇺🇸",
            "en-GB" => "🇬🇧",
            "ja-JP" => "🇯🇵",
            "zh-CN" => "🇨🇳",
            "de-DE" => "🇩🇪",
            "fr-FR" => "🇫🇷",
            "es-ES" => "🇪🇸",
            _ => "🌐",
        };

        println!(
            "  {} {} {}",
            flag,
            ui::secondary(&language),
            ui::dim(&format!("({} voices)", lang_voices.len()))
        );

        let mut table = ui::Table::new(vec!["ID", "Name", "Gender", "Style"]);
        for voice in &lang_voices {
            let gender = voice.gender.as_deref().unwrap_or("-");
            let gender_display = match gender {
                "female" => "♀ female",
                "male" => "♂ male",
                other => other,
            };
            table.row(vec![
                &voice.id,
                &voice.name,
                gender_display,
                voice.style.as_deref().unwrap_or("neutral"),
            ]);
        }
        table.print();
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use xybrid_sdk::registry_client::{ModelDetail, ModelSummary, VariantInfo};

    fn summary() -> ModelSummary {
        ModelSummary {
            id: "qwen3-4b".to_string(),
            family: "qwen".to_string(),
            task: "text-generation".to_string(),
            parameters: 4_000_000_000,
            description: "Qwen".to_string(),
            variants: vec!["mlx-fp16".to_string(), "q4-k-m".to_string()],
        }
    }

    fn variant(format: &str) -> VariantInfo {
        VariantInfo {
            platform: "macos-arm64".to_string(),
            format: format.to_string(),
            quantization: "fp16".to_string(),
            size_bytes: 1,
            hf_repo: "xybrid/qwen".to_string(),
            file: "model".to_string(),
        }
    }

    #[test]
    fn backend_filter_keeps_matching_format_variants() {
        let detail = ModelDetail {
            id: "qwen3-4b".to_string(),
            family: "qwen".to_string(),
            task: "text-generation".to_string(),
            parameters: 4_000_000_000,
            description: "Qwen".to_string(),
            default_variant: None,
            variants: HashMap::from([
                ("mlx-fp16".to_string(), variant("safetensors")),
                ("q4-k-m".to_string(), variant("gguf")),
            ]),
        };
        let filter = parse_models_backend_filter(Some("mlx"))
            .expect("valid backend")
            .expect("explicit backend");

        let display = compatible_variant_display(&summary(), Some(&detail), &filter);

        assert_eq!(
            display,
            VariantDisplay::Compatible {
                format: "safetensors",
                variants: vec!["mlx-fp16".to_string()]
            }
        );
    }

    #[test]
    fn backend_filter_keeps_unknown_format_metadata_visible() {
        let filter = parse_models_backend_filter(Some("mlx"))
            .expect("valid backend")
            .expect("explicit backend");

        let display = compatible_variant_display(&summary(), None, &filter);

        assert_eq!(
            display,
            VariantDisplay::Unknown {
                variants: vec!["mlx-fp16".to_string(), "q4-k-m".to_string()]
            }
        );
    }

    #[test]
    fn mistral_models_filter_lists_gguf_variants() {
        let filter = parse_models_backend_filter(Some("mistral"))
            .expect("valid backend")
            .expect("explicit backend");

        assert_eq!(filter.format, "gguf");
    }
}

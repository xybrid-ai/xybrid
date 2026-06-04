//! `xybrid models` command handlers.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use xybrid_core::bundler::XyBundle;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_sdk::model::SdkError;
use xybrid_sdk::registry_client::RegistryClient;

use super::types::ModelsCommand;
use super::utils::{format_params, format_size};
use crate::ui;

/// Handle `xybrid models` subcommands.
pub(crate) fn handle_models_command(command: ModelsCommand) -> Result<()> {
    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    match command {
        ModelsCommand::List => list_models(&client),
        ModelsCommand::Search { query } => search_models(&client, &query),
        ModelsCommand::Info { model_id } => show_model_info(&client, &model_id),
        ModelsCommand::Voices { model_id } => handle_voices_command(&client, &model_id),
    }
}

fn list_models(client: &RegistryClient) -> Result<()> {
    ui::header("Model Registry");

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

    let mut by_task: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for model in &models {
        by_task.entry(model.task.clone()).or_default().push(model);
    }

    for (task, task_models) in by_task {
        ui::section(&task.to_uppercase());
        println!();

        for model in task_models {
            let params_str = format_params(model.parameters);
            let meta = format!("{} · {} params", model.family, params_str);
            ui::bullet(&model.id, &meta);
            ui::sub(&model.description);
            if !model.variants.is_empty() {
                ui::sub(&format!("variants: {}", model.variants.join(", ")));
            }
        }
    }

    ui::footer(&format!("{} models available", models.len()));

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

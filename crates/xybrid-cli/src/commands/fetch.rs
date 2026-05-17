//! `xybrid fetch` command handler.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use xybrid_core::runtime_adapter::SelectorCfg;
use xybrid_sdk::registry_client::{
    registry_format_preference_for_backend_override_with_registry_context, RegistryClient,
    RegistryFormatPreference,
};

use super::registry_format::registry_format_for_auto_local_backend;
use super::utils::format_size;
use crate::ui;

/// Handle `xybrid fetch --model <id>` command.
pub(crate) fn handle_fetch_command(
    model_id: &str,
    platform: Option<&str>,
    backend_override: Option<&str>,
) -> Result<()> {
    ui::header(&format!("Fetch · {}", model_id));

    if let Some(p) = platform {
        ui::kv("Platform", p);
    } else {
        ui::kv("Platform", "auto-detect");
    }
    if let Some(backend) = backend_override {
        ui::kv("Backend", backend);
    }

    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    let selector_cfg = SelectorCfg::current();
    let format =
        registry_format_for_fetch_backend(&client, model_id, backend_override, &selector_cfg)?;
    let resolved = if let Some(format) = format {
        client
            .resolve_with_format(model_id, platform, format)
            .context(format!("Failed to resolve model '{}'", model_id))?
    } else {
        client
            .resolve(model_id, platform)
            .context(format!("Failed to resolve model '{}'", model_id))?
    };

    print_resolved_variant(&resolved);

    if is_model_ready(&client, model_id, platform, format)? {
        ui::ok("Model is already cached and ready");
        let location = cache_location(&client, model_id, format, &resolved);
        ui::kv("Location", &location.display().to_string());
        return Ok(());
    }

    let pb = ui::download_bar(resolved.size_bytes, model_id);

    let model_path = if let Some(format) = format {
        client.fetch_extracted_with_format(model_id, platform, format, |progress| {
            let bytes_done = (progress * resolved.size_bytes as f32) as u64;
            pb.set_position(bytes_done);
        })
    } else {
        client.fetch(model_id, platform, |progress| {
            let bytes_done = (progress * resolved.size_bytes as f32) as u64;
            pb.set_position(bytes_done);
        })
    }
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

    let sanitized = repo.replace('/', "--");
    let cache_dir =
        dirs::home_dir().map(|h| h.join(".xybrid").join("cache").join("hf").join(&sanitized));

    let loader = xybrid_sdk::ModelLoader::from_huggingface_parsed(repo);
    let model = loader.load().context(format!(
        "Failed to load model from HuggingFace repo '{}'",
        repo
    ))?;

    ui::ok("Model downloaded successfully");
    ui::kv("Model ID", model.model_id());
    ui::kv("Version", model.version());

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
    backend_override: Option<&str>,
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

    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    let pipeline_name = config.name.as_deref().unwrap_or(
        config_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pipeline"),
    );
    ui::header(&format!("Fetch Pipeline · {}", pipeline_name));

    let models_to_fetch: Vec<ModelFetchRequest> = config
        .stages
        .iter()
        .filter(|stage| !stage.is_cloud_stage())
        .map(|stage| ModelFetchRequest {
            model_id: stage.model_id(),
            backend: backend_override
                .or_else(|| stage.backend())
                .map(std::string::ToString::to_string),
        })
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

#[derive(Debug, Clone)]
struct ModelFetchRequest {
    model_id: String,
    backend: Option<String>,
}

fn is_model_ready(
    client: &RegistryClient,
    model_id: &str,
    platform: Option<&str>,
    format: Option<&str>,
) -> Result<bool> {
    if let Some(format) = format {
        Ok(client
            .resolve_offline_with_format(model_id, format)
            .is_some())
    } else {
        client
            .is_cached(model_id, platform)
            .context("Failed to check cache status")
    }
}

fn cache_location(
    client: &RegistryClient,
    model_id: &str,
    format: Option<&str>,
    resolved: &xybrid_sdk::registry_client::ResolvedVariant,
) -> std::path::PathBuf {
    if let Some(format) = format {
        client
            .resolve_offline_with_format(model_id, format)
            .unwrap_or_else(|| client.extraction_dir_with_format(model_id, format))
    } else {
        client.get_cache_path(resolved)
    }
}

fn fetch_models(
    client: &RegistryClient,
    models: &[ModelFetchRequest],
    platform: Option<&str>,
) -> Result<(usize, usize, usize)> {
    fetch_models_with_selector_cfg(client, models, platform, &SelectorCfg::current())
}

fn fetch_models_with_selector_cfg(
    client: &RegistryClient,
    models: &[ModelFetchRequest],
    platform: Option<&str>,
    cfg: &SelectorCfg,
) -> Result<(usize, usize, usize)> {
    let mut success_count = 0;
    let mut skip_count = 0;
    let mut error_count = 0;

    for request in models {
        let model_id = &request.model_id;
        let format = match registry_format_for_fetch_backend(
            client,
            model_id,
            request.backend.as_deref(),
            cfg,
        ) {
            Ok(format) => format,
            Err(e) => {
                ui::err(&format!("{} (invalid backend: {})", model_id, e));
                error_count += 1;
                continue;
            }
        };

        match is_model_ready(client, model_id, platform, format) {
            Ok(true) => {
                ui::ok(&format!("{} (cached)", model_id));
                skip_count += 1;
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                ui::err(&format!("{} (cache check failed: {})", model_id, e));
                error_count += 1;
                continue;
            }
        }

        let resolved = if let Some(format) = format {
            client.resolve_with_format(model_id, platform, format)
        } else {
            client.resolve(model_id, platform)
        };

        match resolved {
            Ok(resolved) => {
                let pb = ui::download_bar(resolved.size_bytes, model_id);

                let fetch_result = if let Some(format) = format {
                    client.fetch_extracted_with_format(model_id, platform, format, |progress| {
                        let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                        pb.set_position(bytes_done);
                    })
                } else {
                    client.fetch(model_id, platform, |progress| {
                        let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                        pb.set_position(bytes_done);
                    })
                };

                match fetch_result {
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

fn registry_format_for_fetch_backend(
    client: &RegistryClient,
    model_id: &str,
    backend_override: Option<&str>,
    cfg: &SelectorCfg,
) -> Result<Option<&'static str>> {
    match registry_format_preference_for_backend_override_with_registry_context(
        client,
        model_id,
        backend_override,
        cfg,
    )? {
        RegistryFormatPreference::Auto => Ok(registry_format_for_auto_local_backend(
            client, model_id, cfg,
        )?),
        RegistryFormatPreference::ExplicitBackend { format } => Ok(format),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    fn apple_mlx_selector_cfg() -> SelectorCfg {
        SelectorCfg {
            target: "macos-aarch64".to_string(),
            host_is_apple_arm64: true,
            mlx_compiled: true,
            llamacpp_compiled: true,
            mistral_compiled: false,
            mlx_runtime_ok: true,
        }
    }

    fn linux_llamacpp_selector_cfg() -> SelectorCfg {
        SelectorCfg {
            target: "linux-x86_64".to_string(),
            host_is_apple_arm64: false,
            mlx_compiled: false,
            llamacpp_compiled: true,
            mistral_compiled: false,
            mlx_runtime_ok: false,
        }
    }

    #[test]
    fn fetch_backend_format_rejects_unavailable_explicit_mlx() {
        let client = RegistryClient::with_url("http://127.0.0.1:9").unwrap();

        let err = registry_format_for_fetch_backend(
            &client,
            "qwen3-4b",
            Some("mlx"),
            &linux_llamacpp_selector_cfg(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("MLX backend requested but not available"),
            "{err}"
        );
    }

    #[test]
    fn fetch_backend_format_rejects_llamacpp_for_embedding_task() {
        let server = MockServer::start();
        let model_id = format!("test-embed-{}", uuid::Uuid::new_v4());
        let detail_body = format!(
            r#"{{
                "id":"{model_id}",
                "family":"test",
                "task":"text-embedding",
                "parameters":1,
                "description":"d",
                "default_variant":null,
                "variants":{{
                    "llamacpp-q4":{{
                        "platform":"macos-arm64",
                        "format":"gguf",
                        "quantization":"q4",
                        "size_bytes":34,
                        "hf_repo":"xybrid-ai/{model_id}",
                        "file":"model.gguf"
                    }}
                }}
            }}"#
        );
        let detail_mock = server.mock(|when, then| {
            when.method(GET).path(format!("/v1/models/{model_id}"));
            then.status(200)
                .header("content-type", "application/json")
                .body(detail_body);
        });
        let client = RegistryClient::with_url(server.base_url()).unwrap();

        let err = registry_format_for_fetch_backend(
            &client,
            &model_id,
            Some("llamacpp"),
            &apple_mlx_selector_cfg(),
        )
        .unwrap_err();

        assert!(
            detail_mock.hits() > 0,
            "explicit embedding backend validation must inspect registry task metadata"
        );
        assert!(
            err.to_string()
                .contains("does not support registry embedding model"),
            "{err}"
        );
    }

    #[test]
    fn fetch_models_auto_fetches_embedding_selector_format_variant() {
        let server = MockServer::start();
        let model_id = format!("test-embed-{}", uuid::Uuid::new_v4());
        let download_url = format!("{}/model.safetensors", server.base_url());
        let detail_body = format!(
            r#"{{
                "id":"{model_id}",
                "family":"test",
                "task":"text-embedding",
                "parameters":1,
                "description":"d",
                "default_variant":null,
                "variants":{{
                    "llamacpp-q4":{{
                        "platform":"macos-arm64",
                        "format":"gguf",
                        "quantization":"q4",
                        "size_bytes":34,
                        "hf_repo":"xybrid-ai/{model_id}",
                        "file":"model.gguf"
                    }},
                    "mlx-fp16":{{
                        "platform":"macos-arm64",
                        "format":"safetensors",
                        "quantization":"fp16",
                        "size_bytes":12,
                        "hf_repo":"xybrid-ai/{model_id}",
                        "file":"model.safetensors"
                    }}
                }}
            }}"#
        );
        let resolve_body = format!(
            r#"{{
                "mask":"{model_id}",
                "platform":"macos-arm64",
                "resolved":{{
                    "hf_repo":"xybrid-ai/{model_id}",
                    "file":"model.safetensors",
                    "download_url":"{}",
                    "format":"safetensors",
                    "quantization":"fp16",
                    "size_bytes":12,
                    "sha256":"",
                    "passthrough":true,
                    "model_metadata":{{
                        "model_id":"{model_id}",
                        "version":"1.0",
                        "execution_template":{{
                            "type":"Safetensors",
                            "model_file":"model.safetensors",
                            "architecture":"nomic_bert"
                        }},
                        "preprocessing":[],
                        "postprocessing":[],
                        "files":["model.safetensors"],
                        "metadata":{{"task":"text-embedding"}}
                    }}
                }}
            }}"#,
            download_url
        );

        let detail_mock = server.mock(|when, then| {
            when.method(GET).path(format!("/v1/models/{model_id}"));
            then.status(200)
                .header("content-type", "application/json")
                .body(detail_body);
        });
        let resolve_mock = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/v1/models/{model_id}/resolve"))
                .query_param_exists("platform")
                .query_param("format", "safetensors");
            then.status(200)
                .header("content-type", "application/json")
                .body(resolve_body);
        });
        let download_mock = server.mock(|when, then| {
            when.method(GET).path("/model.safetensors");
            then.status(200).body("model-bytes");
        });

        let client = RegistryClient::with_url(server.base_url()).unwrap();
        let requests = vec![ModelFetchRequest {
            model_id,
            backend: None,
        }];

        let counts =
            fetch_models_with_selector_cfg(&client, &requests, None, &apple_mlx_selector_cfg())
                .unwrap();

        assert!(
            detail_mock.hits() > 0,
            "fetch auto resolution must inspect model variants"
        );
        assert!(
            resolve_mock.hits() > 0,
            "fetch auto resolution must send the selector-chosen format query"
        );
        download_mock.assert();
        assert_eq!(counts, (1, 0, 0));
    }

    #[test]
    fn fetch_models_auto_keeps_embedding_registry_default_without_safetensors() {
        let server = MockServer::start();
        let model_id = format!("test-embed-{}", uuid::Uuid::new_v4());
        let download_url = format!("{}/model.onnx", server.base_url());
        let detail_body = format!(
            r#"{{
                "id":"{model_id}",
                "family":"test",
                "task":"text-embedding",
                "parameters":1,
                "description":"d",
                "default_variant":null,
                "variants":{{
                    "llamacpp-q4":{{
                        "platform":"macos-arm64",
                        "format":"gguf",
                        "quantization":"q4",
                        "size_bytes":34,
                        "hf_repo":"xybrid-ai/{model_id}",
                        "file":"model.gguf"
                    }}
                }}
            }}"#
        );
        let resolve_body = format!(
            r#"{{
                "mask":"{model_id}",
                "platform":"macos-arm64",
                "resolved":{{
                    "hf_repo":"xybrid-ai/{model_id}",
                    "file":"model.onnx",
                    "download_url":"{}",
                    "format":"onnx",
                    "quantization":"fp32",
                    "size_bytes":12,
                    "sha256":"",
                    "passthrough":true,
                    "model_metadata":{{
                        "model_id":"{model_id}",
                        "version":"1.0",
                        "execution_template":{{
                            "type":"Onnx",
                            "model_file":"model.onnx"
                        }},
                        "preprocessing":[],
                        "postprocessing":[],
                        "files":["model.onnx"],
                        "metadata":{{"task":"text-embedding"}}
                    }}
                }}
            }}"#,
            download_url
        );

        let detail_mock = server.mock(|when, then| {
            when.method(GET).path(format!("/v1/models/{model_id}"));
            then.status(200)
                .header("content-type", "application/json")
                .body(detail_body);
        });
        let wrong_gguf_mock = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/v1/models/{model_id}/resolve"))
                .query_param("format", "gguf");
            then.status(500).body("unexpected gguf format request");
        });
        let resolve_mock = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/v1/models/{model_id}/resolve"))
                .query_param_exists("platform");
            then.status(200)
                .header("content-type", "application/json")
                .body(resolve_body);
        });
        let download_mock = server.mock(|when, then| {
            when.method(GET).path("/model.onnx");
            then.status(200).body("model-bytes");
        });

        let client = RegistryClient::with_url(server.base_url()).unwrap();
        let requests = vec![ModelFetchRequest {
            model_id,
            backend: None,
        }];

        let counts =
            fetch_models_with_selector_cfg(&client, &requests, None, &apple_mlx_selector_cfg())
                .unwrap();

        assert!(
            detail_mock.hits() > 0,
            "fetch auto resolution must inspect model variants"
        );
        assert_eq!(
            wrong_gguf_mock.hits(),
            0,
            "embedding fallback must not request a GGUF variant"
        );
        assert!(
            resolve_mock.hits() > 0,
            "embedding fallback should use the registry default resolution"
        );
        download_mock.assert();
        assert_eq!(counts, (1, 0, 0));
    }
}

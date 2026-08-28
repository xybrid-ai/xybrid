//! Registry artifact-format selection shared by CLI commands.

use xybrid_core::runtime_adapter::{select_with_cfg, RegistryView, SelectionParams, SelectorCfg};
use xybrid_sdk::registry_client::{
    registry_format_for_backend,
    registry_format_preference_for_backend_override_with_registry_context, RegistryClient,
    RegistryFormatPreference,
};
use xybrid_sdk::SdkError;

fn registry_backend_selectable_kind(task: &str) -> Option<&'static str> {
    match xybrid_core::execution::template::stage_kind_from_task(task) {
        Some(kind @ ("llm" | "embed")) => Some(kind),
        _ => None,
    }
}

fn registry_metadata_error_can_fall_back_to_default(err: &SdkError) -> bool {
    matches!(
        err,
        SdkError::Offline { .. }
            | SdkError::NetworkError { .. }
            | SdkError::Timeout { .. }
            | SdkError::CircuitOpen(_)
            // Non-transient registry answers must not brick a fully-cached
            // model either: a delisted id (404), auth misconfig (401/403) or
            // persistent 429 falls back to the offline format resolution, and
            // paths that genuinely need the registry fail later with the
            // specific error. Master ran these entirely offline-first.
            | SdkError::ModelNotFound(_)
            | SdkError::ConfigError(_)
            | SdkError::RateLimited { .. }
    )
}

fn offline_auto_registry_format(
    client: &RegistryClient,
    model_id: &str,
    cfg: &SelectorCfg,
) -> Option<&'static str> {
    if cfg.host_is_apple_arm64
        && cfg.mlx_compiled
        && cfg.mlx_runtime_ok
        && client
            .resolve_offline_with_format(model_id, "safetensors")
            .is_some()
    {
        return Some("safetensors");
    }

    if cfg.llamacpp_compiled
        && client
            .resolve_offline_with_format(model_id, "gguf")
            .is_some()
    {
        return Some("gguf");
    }

    None
}

/// Resolve the registry format preference for an automatic local backend load.
pub(crate) fn registry_format_for_auto_local_backend(
    client: &RegistryClient,
    model_id: &str,
    cfg: &SelectorCfg,
) -> Result<Option<&'static str>, SdkError> {
    let detail = match client.get_model(model_id) {
        Ok(detail) => detail,
        Err(err) if registry_metadata_error_can_fall_back_to_default(&err) => {
            return Ok(offline_auto_registry_format(client, model_id, cfg));
        }
        Err(err) => return Err(err),
    };

    let Some(stage_kind) = registry_backend_selectable_kind(&detail.task) else {
        return Ok(None);
    };

    let selected = select_with_cfg(&SelectionParams::new(model_id), &detail, cfg)
        .map_err(|err| SdkError::ConfigError(err.to_string()))?;

    let Some(format) = registry_format_for_backend(selected) else {
        return Ok(None);
    };

    if stage_kind == "embed" && format != "safetensors" {
        return Ok(None);
    }

    if detail.has_variant(model_id, format) {
        Ok(Some(format))
    } else {
        Ok(None)
    }
}

/// Resolve the artifact format for a CLI registry stage.
///
/// Automatic loads ask the selector to pick the best locally executable
/// backend. Explicit overrides request the format for that backend before
/// downloading, so `--backend mlx` fetches SafeTensors while `--backend
/// llamacpp` fetches GGUF for LLM tasks.
pub(crate) fn registry_format_for_stage_backend(
    client: &RegistryClient,
    model_id: &str,
    backend_override: Option<&str>,
    cfg: &SelectorCfg,
) -> Result<Option<&'static str>, SdkError> {
    match registry_format_preference_for_backend_override_with_registry_context(
        client,
        model_id,
        backend_override,
        cfg,
    )? {
        RegistryFormatPreference::Auto => {
            registry_format_for_auto_local_backend(client, model_id, cfg)
        }
        RegistryFormatPreference::ExplicitBackend { format } => Ok(format),
    }
}

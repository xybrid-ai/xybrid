//! Registry artifact-format selection shared by CLI commands.

use xybrid_core::runtime_adapter::SelectorCfg;
pub(crate) use xybrid_sdk::registry_client::registry_format_for_auto_local_backend;
use xybrid_sdk::registry_client::{
    registry_format_preference_for_backend_override_with_registry_context, RegistryClient,
    RegistryFormatPreference,
};
use xybrid_sdk::SdkError;

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

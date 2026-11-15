//! Registry artifact-format selection shared by CLI commands.

use xybrid_core::runtime_adapter::SelectorCfg;
use xybrid_sdk::registry_client::{
    parse_backend_override, registry_format_for_load, RegistryClient,
};
use xybrid_sdk::SdkError;

/// Resolve the artifact format a CLI command fetches for a registry model.
///
/// With no `--backend` (or `--backend auto`) that is the registry's default
/// variant. An explicit backend requests its own format before anything is
/// downloaded, so `--backend mlx` fetches SafeTensors and `--backend llamacpp`
/// fetches GGUF, and fails here when this build cannot run it.
pub(crate) fn registry_format_for_stage_backend(
    client: &RegistryClient,
    model_id: &str,
    backend_override: Option<&str>,
    cfg: &SelectorCfg,
) -> Result<Option<&'static str>, SdkError> {
    let backend = parse_backend_override(backend_override)?;
    registry_format_for_load(client, model_id, backend, cfg)
}

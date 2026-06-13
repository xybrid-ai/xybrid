use anyhow::Result;
use xybrid_core::context::StageDescriptor;
use xybrid_core::pipeline::ExecutionTarget;
use xybrid_core::pipeline_config::StageConfig;

use crate::ui;

pub(super) fn parse_repl_target(target: Option<&str>) -> Result<Option<ExecutionTarget>> {
    target
        .map(|value| {
            value
                .parse::<ExecutionTarget>()
                .map_err(|err| anyhow::anyhow!(err))
        })
        .transpose()
}

pub(super) fn target_allows_local(target: Option<&ExecutionTarget>) -> bool {
    match target {
        Some(target) => target.allows_local(),
        None => true,
    }
}

pub(super) fn stage_is_locally_available(stage: &StageDescriptor) -> bool {
    stage.is_locally_runnable()
}

pub(super) fn stage_config_allows_local_cache(
    stage_config: &StageConfig,
    target: Option<&ExecutionTarget>,
) -> bool {
    match target {
        Some(target) => target_allows_local(Some(target)),
        None => !stage_config.is_cloud_stage(),
    }
}

pub(super) fn parse_stage_config_target(stage_config: &StageConfig) -> Option<ExecutionTarget> {
    let value = stage_config.target()?;
    match value.parse::<ExecutionTarget>() {
        Ok(target) => Some(target),
        Err(err) => {
            ui::warning(&format!(
                "Ignoring invalid stage target '{}': {}",
                value, err
            ));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_core::pipeline_config::PipelineConfig;

    #[test]
    fn network_targets_do_not_allow_local_runtime() {
        assert!(!target_allows_local(Some(&ExecutionTarget::Cloud)));
        assert!(!target_allows_local(Some(&ExecutionTarget::Server)));
    }

    #[test]
    fn local_and_auto_targets_allow_local_runtime() {
        assert!(target_allows_local(None));
        assert!(target_allows_local(Some(&ExecutionTarget::Device)));
        assert!(target_allows_local(Some(&ExecutionTarget::Auto)));
    }

    #[test]
    fn network_target_with_bundle_path_is_not_local_available() {
        let stage = StageDescriptor::new("test-model")
            .with_bundle_path("/tmp/test-model")
            .with_target(ExecutionTarget::Cloud);

        assert!(!stage_is_locally_available(&stage));
    }

    #[test]
    fn server_config_stage_skips_local_cache() {
        let config = PipelineConfig::from_yaml(
            r#"
name: test
stages:
  - id: llm
    model: test-model
    target: server
"#,
        )
        .unwrap();

        assert!(!stage_config_allows_local_cache(
            &config.stages[0],
            Some(&ExecutionTarget::Server)
        ));
    }

    #[test]
    fn repl_target_accepts_local_execution_aliases() {
        assert_eq!(
            parse_repl_target(Some("local")).unwrap(),
            Some(ExecutionTarget::Device)
        );
        assert_eq!(
            parse_repl_target(Some("device")).unwrap(),
            Some(ExecutionTarget::Device)
        );
    }

    #[test]
    fn repl_target_rejects_model_format_values() {
        let err = parse_repl_target(Some("onnx")).unwrap_err();

        assert!(
            err.to_string().contains("Unknown execution target"),
            "{err}"
        );
    }
}

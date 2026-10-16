use std::path::PathBuf;

use xybrid_core::context::StageDescriptor;
use xybrid_sdk::model::{ModelLoader, XybridModel};
use xybrid_sdk::SdkResult;

use crate::commands::utils::refuse_choice_scorer;
use crate::ui;

use super::targeting::target_allows_local;

/// Load the model behind a stage that runs here, or `None` for a stage that
/// cannot: a target that excludes local execution, or no `bundle_path`
/// (remote-only / integration stages). Loading reads the metadata; weights
/// load on first use.
fn load_local_stage(stage: &StageDescriptor) -> Option<SdkResult<XybridModel>> {
    if !target_allows_local(stage.target.as_ref()) {
        return None;
    }
    let bundle_path = PathBuf::from(stage.bundle_path.as_ref()?);
    let loader = if bundle_path.extension().is_some_and(|ext| ext == "xyb") {
        ModelLoader::from_bundle(&bundle_path)
    } else {
        ModelLoader::from_directory(&bundle_path)
    };
    Some(loader.and_then(|loader| loader.load()))
}

/// Refuse a pipeline with a choice scorer among its local stages, before any
/// stage warms. A scorer answers only `xybrid run --choice`: warmup would
/// score a synthetic request and report it ready, and the pipeline would
/// refuse its stage on the first turn. A stage that fails to load is left to
/// [`warmup_models`], which reports it.
pub(super) fn refuse_choice_scorer_stages(stages: &[StageDescriptor]) -> anyhow::Result<()> {
    for model in stages.iter().filter_map(load_local_stage).flatten() {
        refuse_choice_scorer(&model, "the REPL runs text turns")?;
    }
    Ok(())
}

/// Warm up local stages by constructing an `XybridModel` per stage with
/// a populated `bundle_path` and calling `model.warmup()`.
///
/// Why not just call `orchestrator.execute_pipeline()` (which also runs
/// a forward pass per stage)? The orchestrator emits a full pipeline
/// execution event chain (`PipelineStart`, `StageStart`,
/// `PolicyEvaluated`, `RoutingDecided`, `ExecutionStarted`,
/// `ExecutionCompleted`, `StageComplete`, `PipelineComplete`). The
/// SDK→wire bridge filter drops most of those, but `PolicyEvaluated`
/// and `RoutingDecided` legitimately pass through (they're routing
/// metadata). On a real inference those events sit behind a
/// `ModelComplete` / `PipelineComplete` row on the dashboard; on a
/// warmup pass there's no completion event behind them, so they
/// render as phantom 0 ms rows.
///
/// `XybridModel::warmup` goes directly through the executor — no
/// orchestrator events fire — and publishes a single `ModelWarmup`
/// event per stage with measured latency. The Traces dashboard then
/// sees one labelled warmup row per local stage instead of two phantom
/// rows per stage.
///
/// Stages without a `bundle_path` (remote-only / integration stages)
/// can't warm locally and are skipped.
pub(super) fn warmup_models(stages: &[StageDescriptor]) {
    let sp = ui::spinner("Warming up models...");
    let mut warmed = 0_usize;
    let mut failed: Vec<(String, String)> = Vec::new();

    for stage in stages {
        let Some(loaded) = load_local_stage(stage) else {
            continue;
        };

        match loaded.and_then(|model| model.warmup()) {
            Ok(()) => warmed += 1,
            Err(e) => failed.push((stage.name.clone(), e.to_string())),
        }
    }

    sp.finish_and_clear();
    if failed.is_empty() {
        if warmed == 0 {
            // All stages were remote — nothing to warm; treat as silent OK.
            ui::ok("No local stages to warm. Ready for input!");
        } else {
            ui::ok("Models loaded and warm. Ready for input!");
        }
    } else {
        for (stage_name, err) in &failed {
            ui::warning(&format!(
                "Warmup failed for {} ({}), first query may be slow",
                stage_name, err
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::refuse_choice_scorer_stages;
    use std::path::Path;
    use xybrid_core::context::StageDescriptor;
    use xybrid_core::pipeline::ExecutionTarget;

    /// A stage whose directory holds only `model_id`'s metadata.
    fn stage(dir: &Path, model_id: &str, template: serde_json::Value) -> StageDescriptor {
        let model_dir = dir.join(model_id);
        std::fs::create_dir_all(&model_dir).unwrap();
        let metadata = serde_json::json!({
            "model_id": model_id,
            "version": "1.0",
            "execution_template": template,
            "files": [],
        });
        std::fs::write(model_dir.join("model_metadata.json"), metadata.to_string()).unwrap();
        StageDescriptor::new(model_id).with_bundle_path(model_dir.display().to_string())
    }

    fn chat(dir: &Path, model_id: &str) -> StageDescriptor {
        stage(
            dir,
            model_id,
            serde_json::json!({ "type": "Gguf", "model_file": "model.gguf" }),
        )
    }

    fn scorer(dir: &Path, model_id: &str) -> StageDescriptor {
        let field = |ids: &str, mask: &str| {
            serde_json::json!({
                "offset": 1, "max_len": 8, "overflow": "Truncate",
                "ids_input": ids, "mask_input": mask,
            })
        };
        stage(
            dir,
            model_id,
            serde_json::json!({
                "type": "ChoiceScorer",
                "scorer": {
                    "type": "OnnxByteOptionScorer",
                    "model_file": "scorer.onnx",
                    "context": field("context_ids", "context_mask"),
                    "choice": field("option_ids", "option_token_mask"),
                    "choice_mask_input": "option_mask",
                    "logits_output": "logits",
                    "max_choices": 4,
                    "fixed_choices": [],
                },
            }),
        )
    }

    #[test]
    fn a_local_scorer_stage_refuses_the_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let stages = [chat(dir.path(), "chat"), scorer(dir.path(), "forms")];
        let message = refuse_choice_scorer_stages(&stages)
            .expect_err("a scorer stage is refused")
            .to_string();
        assert!(
            message.starts_with("'forms' is a choice scorer; the REPL runs text turns."),
            "{message}"
        );
    }

    #[test]
    fn stages_that_do_not_run_here_or_cannot_load_pass() {
        let dir = tempfile::tempdir().unwrap();
        let stages = [
            chat(dir.path(), "chat"),
            // Not run here: a cloud target, and a stage with no bundle.
            scorer(dir.path(), "cloud-scorer").with_target(ExecutionTarget::Cloud),
            StageDescriptor::new("remote"),
            // Left to warmup, which reports a stage that fails to load.
            StageDescriptor::new("missing")
                .with_bundle_path(dir.path().join("missing").display().to_string()),
        ];
        refuse_choice_scorer_stages(&stages).expect("no local scorer");
    }
}

use std::path::PathBuf;

use xybrid_core::context::StageDescriptor;
use xybrid_sdk::model::ModelLoader;

use crate::ui;

use super::targeting::target_allows_local;

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
        if !target_allows_local(stage.target.as_ref()) {
            continue;
        }

        let Some(bundle_path_str) = stage.bundle_path.as_ref() else {
            // Remote / integration stage — nothing to warm locally.
            continue;
        };
        let bundle_path = PathBuf::from(bundle_path_str);

        let loader_result = if bundle_path.extension().is_some_and(|ext| ext == "xyb") {
            ModelLoader::from_bundle(&bundle_path)
        } else {
            ModelLoader::from_directory(&bundle_path)
        };

        let warmup_result = loader_result
            .and_then(|loader| loader.load())
            .and_then(|model| model.warmup());

        match warmup_result {
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

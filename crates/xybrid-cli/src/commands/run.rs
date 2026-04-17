//! `xybrid run` command handlers for pipeline, bundle, and model execution.

#![allow(clippy::too_many_arguments)]

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use xybrid_core::context::StageDescriptor;
use xybrid_core::device_adapter::{DeviceAdapter, LocalDeviceAdapter};
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::policy_engine::PolicyEngine;
use xybrid_core::orchestrator::routing_engine::{LocalAvailability, RoutingEngine};
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::pipeline_config::PipelineConfig;
use xybrid_core::target::{Platform, TargetResolver};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_sdk::registry_client::RegistryClient;

use super::utils::{display_stage_name, format_size, save_wav_file};
use crate::ui;

/// Run a pipeline from a configuration file.
pub(crate) fn run_pipeline(
    config_path: &PathBuf,
    dry_run: bool,
    policy_path: Option<&PathBuf>,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    target: Option<&str>,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    let _pipeline_span = if trace_enabled {
        Some(crate::tracing_viz::SpanGuard::new("pipeline_execution"))
    } else {
        None
    };

    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    let pipeline_name = config.name.as_deref().unwrap_or("unnamed");
    ui::header(&format!("Pipeline · {}", pipeline_name));

    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;
    let stages = resolve_pipeline_stages(&config, &client)?;
    let input = build_input_envelope(input_audio, input_text, voice)?;

    let device_adapter = LocalDeviceAdapter::new();
    let metrics = device_adapter.collect_metrics();
    let availability_fn = build_availability_fn(&stages);

    print_pipeline_config(&stages, &input, &metrics, target);

    if dry_run {
        return run_dry_run(&stages, &input, &metrics, &availability_fn);
    }

    execute_pipeline(
        &stages,
        &input,
        &metrics,
        &availability_fn,
        policy_path,
        output_path,
        trace_enabled,
        trace_export,
    )
}

fn resolve_pipeline_stages(
    config: &PipelineConfig,
    client: &RegistryClient,
) -> Result<Vec<StageDescriptor>> {
    let mut stages = Vec::new();

    for stage_config in &config.stages {
        let model_id = stage_config.model_id();
        let mut desc = StageDescriptor::new(&model_id);

        if stage_config.is_cloud_stage() {
            configure_cloud_stage(&mut desc, stage_config, &model_id);
        } else {
            resolve_device_stage(&mut desc, &model_id, client)?;
        }

        stages.push(desc);
    }

    Ok(stages)
}

fn configure_cloud_stage(
    desc: &mut StageDescriptor,
    stage_config: &xybrid_core::pipeline_config::StageConfig,
    model_id: &str,
) {
    if let Some(provider) = stage_config.provider() {
        desc.provider = Some(match provider {
            "openai" => xybrid_core::pipeline::IntegrationProvider::OpenAI,
            "anthropic" => xybrid_core::pipeline::IntegrationProvider::Anthropic,
            "google" => xybrid_core::pipeline::IntegrationProvider::Google,
            _ => xybrid_core::pipeline::IntegrationProvider::OpenAI,
        });
    }
    desc.target = Some(xybrid_core::pipeline::ExecutionTarget::Cloud);
    desc.model = Some(model_id.to_string());

    let opts = stage_config.options();
    if !opts.is_empty() {
        let mut stage_opts = xybrid_core::pipeline::StageOptions::new();
        for (key, value) in opts {
            stage_opts.values.insert(key, value);
        }
        desc.options = Some(stage_opts);
    }
}

fn resolve_device_stage(
    desc: &mut StageDescriptor,
    model_id: &str,
    client: &RegistryClient,
) -> Result<()> {
    // Offline-first: if the model is already extracted locally, point the stage
    // at its extraction directory and skip the registry. This prevents the
    // circuit breaker from tripping on a cached model when the registry is
    // unreachable.
    if let Some(extract_dir) = client.resolve_offline(model_id) {
        desc.bundle_path = Some(extract_dir.to_string_lossy().to_string());
        return Ok(());
    }

    // Not cached locally — must fetch from the registry.
    download_model(desc, model_id, client)
}

fn download_model(
    desc: &mut StageDescriptor,
    model_id: &str,
    client: &RegistryClient,
) -> Result<()> {
    match client.resolve(model_id, None) {
        Ok(resolved) => {
            let pb = ui::download_bar(resolved.size_bytes, model_id);

            match client.fetch(model_id, None, |progress| {
                let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                pb.set_position(bytes_done);
            }) {
                Ok(bundle_path) => {
                    pb.finish_and_clear();
                    ui::ok(&format!("{} downloaded", model_id));
                    desc.bundle_path = Some(bundle_path.to_string_lossy().to_string());
                }
                Err(e) => {
                    pb.abandon();
                    return Err(anyhow::anyhow!(
                        "Failed to download model '{}': {}",
                        model_id,
                        e
                    ));
                }
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to resolve model '{}': {}",
                model_id,
                e
            ));
        }
    }

    Ok(())
}

fn build_input_envelope(
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
) -> Result<Envelope> {
    let mut input = if let Some(audio_path) = input_audio {
        ui::kv("Input", &format!("audio ({})", audio_path.display()));
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        ui::kv("Size", &format!("{} bytes", audio_bytes.len()));
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else if let Some(text) = input_text {
        ui::kv("Input", &format!("\"{}\"", text));
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    } else {
        Envelope::new(EnvelopeKind::Text(String::new()))
    };

    if let Some(voice_id) = voice {
        ui::kv("Voice", voice_id);
        input
            .metadata
            .insert("voice_id".to_string(), voice_id.to_string());
    }

    Ok(input)
}

fn build_availability_fn(stages: &[StageDescriptor]) -> impl Fn(&str) -> LocalAvailability + '_ {
    let stage_bundle_paths: std::collections::HashMap<String, bool> = stages
        .iter()
        .map(|s| (s.name.clone(), s.bundle_path.is_some()))
        .collect();
    move |stage: &str| -> LocalAvailability {
        let available = stage_bundle_paths.get(stage).copied().unwrap_or(false);
        LocalAvailability::new(available)
    }
}

fn print_pipeline_config(
    stages: &[StageDescriptor],
    input: &Envelope,
    metrics: &xybrid_core::context::DeviceMetrics,
    target: Option<&str>,
) {
    ui::section("Configuration");
    println!();
    ui::kv("Stages", &stages.len().to_string());
    for (i, stage) in stages.iter().enumerate() {
        println!(
            "  {}  {}. {}",
            ui::dim(""),
            i + 1,
            ui::accent(display_stage_name(&stage.name))
        );
    }
    println!();
    ui::kv("Input", input.kind_str());

    ui::section("Device");
    println!();
    ui::kv("RTT", &format!("{}ms", metrics.network_rtt));
    ui::kv("Battery", &format!("{}%", metrics.battery));
    ui::kv(
        "Temperature",
        &format!("{:.1}\u{00B0}C", metrics.temperature),
    );

    let platform = Platform::detect();
    let resolved_target = TargetResolver::new()
        .with_requested(target)
        .with_platform(platform)
        .resolve();

    ui::section("Target");
    println!();
    ui::kv("Platform", &platform.to_string());
    ui::kv("Requested", target.unwrap_or("(auto)"));
    ui::kv("Resolved", &resolved_target.to_string());
    println!();
}

fn run_dry_run(
    stages: &[StageDescriptor],
    input: &Envelope,
    metrics: &xybrid_core::context::DeviceMetrics,
    availability_fn: &dyn Fn(&str) -> LocalAvailability,
) -> Result<()> {
    ui::section("Dry Run · Routing Simulation");
    println!();

    let mut routing_engine = xybrid_core::orchestrator::routing_engine::DefaultRoutingEngine::new();
    let policy_engine =
        xybrid_core::orchestrator::policy_engine::DefaultPolicyEngine::with_default_policy();

    let mut current_input = input.clone();

    for (i, stage) in stages.iter().enumerate() {
        println!(
            "  {} {}",
            ui::dim(&format!("Stage {}", i + 1)),
            ui::accent(display_stage_name(&stage.name))
        );

        let policy_result = policy_engine.evaluate(&stage.name, &current_input, metrics);
        let policy_status = if policy_result.allowed {
            format!("{}", ui::success("ALLOWED"))
        } else {
            format!("{}", ui::error("DENIED"))
        };
        ui::kv("  Policy", &policy_status);
        if let Some(ref reason) = policy_result.reason {
            ui::kv("  Reason", reason);
        }

        let availability = availability_fn(&stage.name);
        let routing_decision =
            routing_engine.decide(&stage.name, metrics, &policy_result, &availability);
        ui::kv(
            "  Routing",
            &format!("{} ({})", routing_decision.target, routing_decision.reason),
        );

        let new_kind = match &current_input.kind {
            EnvelopeKind::Audio(_) => EnvelopeKind::Text("transcribed".to_string()),
            EnvelopeKind::Text(t) => EnvelopeKind::Text(format!("{}-output", t)),
            EnvelopeKind::Embedding(_) => EnvelopeKind::Text("result".to_string()),
        };
        current_input = Envelope::new(new_kind);
        ui::kv("  Output", current_input.kind_str());
        println!();
    }

    ui::ok("Dry run completed — no execution performed");
    println!();
    Ok(())
}

fn execute_pipeline(
    stages: &[StageDescriptor],
    input: &Envelope,
    metrics: &xybrid_core::context::DeviceMetrics,
    availability_fn: &dyn Fn(&str) -> LocalAvailability,
    policy_path: Option<&PathBuf>,
    output_path: Option<&PathBuf>,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    let mut orchestrator = Orchestrator::new();
    xybrid_sdk::bridge_orchestrator_events(&orchestrator);

    if let Some(policy_file) = policy_path {
        ui::kv("Policy", &policy_file.display().to_string());
        let policy_bytes = fs::read(policy_file)
            .with_context(|| format!("Failed to read policy file: {}", policy_file.display()))?;

        orchestrator
            .load_policies(policy_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load policies: {}", e))?;

        ui::ok("Policy bundle loaded");
        println!();
    }

    let sp = ui::spinner("Executing pipeline...");

    match orchestrator.execute_pipeline(stages, input, metrics, availability_fn) {
        Ok(results) => {
            sp.finish_and_clear();
            print_pipeline_results(&results, output_path)?;
            print_trace_output(trace_enabled, trace_export)?;
            Ok(())
        }
        Err(e) => {
            sp.finish_and_clear();
            ui::err(&format!("Pipeline execution failed: {}", e));
            Err(anyhow::anyhow!("Pipeline execution failed: {}", e))
        }
    }
}

fn print_pipeline_results(
    results: &[xybrid_core::orchestrator::StageExecutionResult],
    output_path: Option<&PathBuf>,
) -> Result<()> {
    ui::section("Results");
    println!();

    for (i, result) in results.iter().enumerate() {
        println!(
            "  {} {}",
            ui::dim(&format!("Stage {}", i + 1)),
            ui::accent(display_stage_name(&result.stage))
        );
        ui::kv("  Routing", &result.routing_decision.target.to_string());
        ui::kv("  Reason", &result.routing_decision.reason);
        ui::kv("  Time", &format!("{}ms", result.latency_ms));
        ui::kv("  Output", result.output.kind_str());

        match &result.output.kind {
            EnvelopeKind::Text(text) => {
                if !text.is_empty() {
                    println!();
                    println!("    {}", text);
                }
            }
            EnvelopeKind::Audio(data) => {
                ui::kv("  Size", &format!("{} bytes", data.len()));
            }
            EnvelopeKind::Embedding(vec) => {
                ui::kv("  Dimensions", &format!("{} elements", vec.len()));
                if vec.len() <= 10 {
                    println!("    {:?}", vec);
                } else {
                    println!("    {:?} ...", &vec[..5]);
                }
            }
        }
        println!();
    }

    save_pipeline_output(results, output_path)?;

    ui::ok("Pipeline completed successfully");
    println!();

    Ok(())
}

fn save_pipeline_output(
    results: &[xybrid_core::orchestrator::StageExecutionResult],
    output_path: Option<&PathBuf>,
) -> Result<()> {
    if let Some(path) = output_path {
        if let Some(last_result) = results.last() {
            match &last_result.output.kind {
                EnvelopeKind::Text(text) => {
                    fs::write(path, text)
                        .with_context(|| format!("Failed to write output to {}", path.display()))?;
                    ui::ok(&format!("Output saved to {}", path.display()));
                }
                EnvelopeKind::Audio(data) => {
                    save_wav_file(path, data, 24000, 1)
                        .with_context(|| format!("Failed to write audio to {}", path.display()))?;
                    ui::ok(&format!("Audio saved to {}", path.display()));
                }
                EnvelopeKind::Embedding(vec) => {
                    let json = serde_json::to_string_pretty(vec)
                        .context("Failed to serialize embedding")?;
                    fs::write(path, json).with_context(|| {
                        format!("Failed to write embedding to {}", path.display())
                    })?;
                    ui::ok(&format!("Embedding saved to {}", path.display()));
                }
            }
        }
    } else if let Some(last_result) = results.last() {
        if matches!(last_result.output.kind, EnvelopeKind::Audio(_)) {
            ui::hint("Use --output <file.wav> to save the audio");
        }
    }

    Ok(())
}

fn print_trace_output(trace_enabled: bool, trace_export: Option<&PathBuf>) -> Result<()> {
    if trace_enabled {
        println!("{}", crate::tracing_viz::render_trace());

        if let Some(export_path) = trace_export {
            let json = crate::tracing_viz::GLOBAL_COLLECTOR
                .lock()
                .unwrap()
                .to_chrome_trace_json();
            fs::write(export_path, json)
                .with_context(|| format!("Failed to export trace to {}", export_path.display()))?;
            ui::ok(&format!("Trace exported to {}", export_path.display()));
        }
    }

    Ok(())
}

/// Run inference directly on a .xyb bundle file.
pub(crate) fn run_bundle(
    bundle_path: &Path,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    dry_run: bool,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    if trace_enabled {
        crate::tracing_viz::reset_collector();
    }
    let _bundle_span = if trace_enabled {
        Some(crate::tracing_viz::SpanGuard::new("bundle_execution"))
    } else {
        None
    };

    let trace_id = uuid::Uuid::new_v4();
    xybrid_sdk::set_telemetry_pipeline_context(None, Some(trace_id));

    ui::header("Run · Bundle");
    ui::kv("Bundle", &bundle_path.display().to_string());

    if !bundle_path.exists() {
        return Err(anyhow::anyhow!(
            "Bundle file not found: {}",
            bundle_path.display()
        ));
    }

    let sp = ui::spinner("Loading and extracting bundle...");
    let cache = xybrid_sdk::cache::CacheManager::new().context("Failed to create cache manager")?;
    let extract_dir = cache
        .ensure_extracted(bundle_path)
        .context("Failed to extract bundle")?;
    sp.finish_and_clear();

    let (metadata, input) =
        prepare_bundle_execution(&extract_dir, input_audio, input_text, voice, dry_run)?;

    emit_pipeline_start_event(&metadata, &bundle_path.display().to_string());

    if dry_run {
        ui::section("Dry Run");
        println!();
        ui::ok("Bundle is valid and ready for execution");
        ui::hint("Use without --dry-run to run inference");
        println!();
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(&extract_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
    print_llm_trace_block(&output, trace_enabled);
    emit_pipeline_complete_event(&metadata, &output, elapsed);

    if trace_enabled {
        drop(_bundle_span);
        print_trace_output(trace_enabled, trace_export)?;
    }

    Ok(())
}

/// Print the per-inference LLM telemetry block populated by the adapter's
/// streaming path. Keys match `runtime_adapter/llm.rs`'s envelope metadata
/// inserts. Non-LLM stages (no `ttft_ms`) emit nothing so ASR/TTS runs stay
/// quiet under `--trace`.
fn print_llm_trace_block(output: &Envelope, trace_enabled: bool) {
    if !trace_enabled || !output.metadata.contains_key("ttft_ms") {
        return;
    }
    let m = &output.metadata;
    println!();
    println!("LLM Trace");
    println!("{}", "-".repeat(60));
    let rows: &[(&str, &str, &str)] = &[
        ("ttft_ms", "TTFT", " ms"),
        ("decode_tps", "Decode TPS", " tok/s"),
        ("prefill_tps", "Prefill TPS", " tok/s"),
        ("tokens_per_second", "Wallclock TPS", " tok/s"),
        ("mean_itl_ms", "Mean ITL", " ms"),
        ("p95_itl_ms", "p95 ITL", " ms"),
        ("emitted_chunks", "Emitted chunks", ""),
        ("tokens_generated", "Tokens generated", ""),
        ("finish_reason", "Finish reason", ""),
    ];
    for (key, label, suffix) in rows {
        let present = m.contains_key(*key);
        let val = m.get(*key).map(|s| s.as_str()).unwrap_or("—");
        println!(
            "  {:<22}: {}{}",
            label,
            val,
            if present { suffix } else { "" }
        );
    }
    println!("{}", "-".repeat(60));
    println!();
}

/// Run inference on a model directly from the registry.
pub(crate) fn run_model(
    model_id: &str,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    platform: Option<&str>,
    dry_run: bool,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    if trace_enabled {
        crate::tracing_viz::reset_collector();
    }
    let _model_span = if trace_enabled {
        Some(crate::tracing_viz::SpanGuard::new("model_execution"))
    } else {
        None
    };

    let trace_id = uuid::Uuid::new_v4();
    xybrid_sdk::set_telemetry_pipeline_context(None, Some(trace_id));

    ui::header(&format!("Run · {}", model_id));

    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    let _fetch_span = if trace_enabled {
        Some(crate::tracing_viz::SpanGuard::new("registry_fetch"))
    } else {
        None
    };

    // Offline-first: if the model has already been downloaded and extracted,
    // skip the registry entirely. This keeps `xybrid run` working when the
    // machine is offline (or the registry is unreachable) for any model that
    // was previously fetched.
    let extract_dir = if let Some(cached_dir) = client.resolve_offline(model_id) {
        println!();
        ui::ok("Using locally cached model");
        ui::kv("Location", &cached_dir.display().to_string());
        println!();
        drop(_fetch_span);
        cached_dir
    } else {
        let resolved = client.resolve(model_id, platform).context(format!(
            "Failed to resolve model '{}' from registry",
            model_id
        ))?;

        println!();
        ui::kv("Repository", &resolved.hf_repo);
        ui::kv("File", &resolved.file);
        ui::kv("Size", &format_size(resolved.size_bytes));
        ui::kv(
            "Format",
            &format!("{} ({})", resolved.format, resolved.quantization),
        );

        if resolved.passthrough {
            // Passthrough models (e.g., GGUF): download raw file + write metadata directly
            let pb = ui::download_bar(resolved.size_bytes, model_id);
            let dir = client
                .fetch_extracted(model_id, platform, |progress| {
                    let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                    pb.set_position(bytes_done);
                })
                .context(format!(
                    "Failed to fetch passthrough model '{}' from registry",
                    model_id
                ))?;
            pb.finish_and_clear();
            ui::ok(&format!("Downloaded {}", model_id));
            ui::kv("Location", &dir.display().to_string());
            println!();
            drop(_fetch_span);
            dir
        } else {
            // Standard .xyb bundle flow
            let bundle_path = fetch_or_cache(&client, model_id, platform, &resolved)?;
            ui::kv("Location", &bundle_path.display().to_string());
            println!();
            drop(_fetch_span);

            let sp = ui::spinner("Loading and extracting bundle...");
            let cache =
                xybrid_sdk::cache::CacheManager::new().context("Failed to create cache manager")?;
            let dir = cache
                .ensure_extracted(&bundle_path)
                .context("Failed to extract bundle")?;
            sp.finish_and_clear();
            dir
        }
    };

    let (metadata, input) =
        prepare_bundle_execution(&extract_dir, input_audio, input_text, voice, dry_run)?;

    emit_pipeline_start_event(&metadata, "registry");

    if dry_run {
        ui::section("Dry Run");
        println!();
        ui::ok("Model is valid and ready for execution");
        ui::hint("Use without --dry-run to run inference");
        println!();
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(&extract_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
    print_llm_trace_block(&output, trace_enabled);
    emit_pipeline_complete_event(&metadata, &output, elapsed);

    if trace_enabled {
        drop(_model_span);
        print_trace_output(trace_enabled, trace_export)?;
    }

    Ok(())
}

/// Run inference from a local model directory.
pub(crate) fn run_directory(
    dir: &Path,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    dry_run: bool,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    if trace_enabled {
        crate::tracing_viz::reset_collector();
    }

    ui::header("Run · Local Directory");
    ui::kv("Directory", &dir.display().to_string());

    if !dir.exists() {
        return Err(anyhow::anyhow!("Directory not found: {}", dir.display()));
    }

    let (metadata, input) = prepare_bundle_execution(dir, input_audio, input_text, voice, dry_run)?;

    emit_pipeline_start_event(&metadata, "directory");

    if dry_run {
        ui::section("Dry Run");
        println!();
        ui::ok("Model is valid and ready for execution");
        ui::hint("Use without --dry-run to run inference");
        println!();
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
    print_llm_trace_block(&output, trace_enabled);
    emit_pipeline_complete_event(&metadata, &output, elapsed);

    if trace_enabled {
        print_trace_output(trace_enabled, trace_export)?;
    }

    Ok(())
}

/// Run inference from a HuggingFace model (downloads if needed, auto-generates metadata).
pub(crate) fn run_huggingface(
    repo: &str,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    dry_run: bool,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    if trace_enabled {
        crate::tracing_viz::reset_collector();
    }

    ui::header(&format!("Run · HuggingFace · {}", repo));

    let sp = ui::spinner("Loading from HuggingFace...");
    let loader = xybrid_sdk::ModelLoader::from_huggingface_parsed(repo);
    let model = loader.load().context(format!(
        "Failed to load model from HuggingFace repo '{}'",
        repo
    ))?;
    sp.finish_and_clear();

    ui::ok(&format!("Model loaded: {}", model.model_id()));
    println!();

    let sanitized = repo.replace('/', "--");
    let cache_dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?
        .join(".xybrid")
        .join("cache")
        .join("hf")
        .join(&sanitized);

    let (metadata, input) =
        prepare_bundle_execution(&cache_dir, input_audio, input_text, voice, dry_run)?;

    emit_pipeline_start_event(&metadata, "huggingface");

    if dry_run {
        ui::section("Dry Run");
        println!();
        ui::ok("Model is valid and ready for execution");
        ui::hint("Use without --dry-run to run inference");
        println!();
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(&cache_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
    print_llm_trace_block(&output, trace_enabled);
    emit_pipeline_complete_event(&metadata, &output, elapsed);

    if trace_enabled {
        print_trace_output(trace_enabled, trace_export)?;
    }

    Ok(())
}

/// Run inference on an arbitrary GGUF model file (auto-generates metadata).
pub(crate) fn run_model_file(
    gguf_path: &Path,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    dry_run: bool,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    if trace_enabled {
        crate::tracing_viz::reset_collector();
    }

    let gguf_path = gguf_path
        .canonicalize()
        .with_context(|| format!("GGUF file not found: {}", gguf_path.display()))?;

    ui::header("Run · GGUF File");
    ui::kv("File", &gguf_path.display().to_string());

    let metadata = xybrid_sdk::metadata_gen::generate_metadata_for_gguf_file(&gguf_path)
        .map_err(|e| anyhow::anyhow!("Failed to generate metadata for GGUF file: {}", e))?;

    println!();
    ui::kv("Model ID", &metadata.model_id);
    if let xybrid_core::execution::ExecutionTemplate::Gguf { context_length, .. } =
        &metadata.execution_template
    {
        ui::kv("Context", &context_length.to_string());
    }
    if let Some(arch) = metadata.metadata.get("architecture") {
        ui::kv("Architecture", &arch.to_string());
    }

    let parent_dir = gguf_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine parent directory of GGUF file"))?;

    let metadata_path = parent_dir.join("model_metadata.json");
    if !metadata_path.exists() {
        let json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, &json)?;
        println!();
        ui::warning("model_metadata.json was auto-generated. Review and adjust if needed.");
        ui::hint(&metadata_path.display().to_string());
    }
    println!();

    emit_pipeline_start_event(&metadata, "model-file");

    if dry_run {
        ui::section("Dry Run");
        println!();
        ui::ok("Model is valid and ready for execution");
        ui::hint("Use without --dry-run to run inference");
        println!();
        return Ok(());
    }

    let input = if let Some(audio_path) = input_audio {
        ui::kv("Input", &format!("audio ({})", audio_path.display()));
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else if let Some(text) = input_text {
        ui::kv("Input", &format!("\"{}\"", text));
        let mut envelope = Envelope::new(EnvelopeKind::Text(text.to_string()));
        if let Some(voice_id) = voice {
            ui::kv("Voice", voice_id);
            envelope
                .metadata
                .insert("voice_id".to_string(), voice_id.to_string());
        }
        envelope
    } else {
        return Err(anyhow::anyhow!(
            "No input provided. Use --input-audio <file> or --input-text <text>"
        ));
    };

    let (output, elapsed) = run_inference(parent_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
    print_llm_trace_block(&output, trace_enabled);
    emit_pipeline_complete_event(&metadata, &output, elapsed);

    if trace_enabled {
        print_trace_output(trace_enabled, trace_export)?;
    }

    Ok(())
}

fn fetch_or_cache(
    client: &RegistryClient,
    model_id: &str,
    platform: Option<&str>,
    resolved: &xybrid_sdk::registry_client::ResolvedVariant,
) -> Result<PathBuf> {
    if client
        .is_cached(model_id, platform)
        .context("Failed to check cache status")?
    {
        ui::ok("Model cached");
        Ok(client.get_cache_path(resolved))
    } else {
        let pb = ui::download_bar(resolved.size_bytes, model_id);

        let path = client
            .fetch(model_id, platform, |progress| {
                let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                pb.set_position(bytes_done);
            })
            .context(format!(
                "Failed to fetch model '{}' from registry",
                model_id
            ))?;

        pb.finish_and_clear();
        ui::ok(&format!("Downloaded {}", model_id));
        Ok(path)
    }
}

fn prepare_bundle_execution(
    extract_dir: &Path,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    voice: Option<&str>,
    dry_run: bool,
) -> Result<(ModelMetadata, Option<Envelope>)> {
    let metadata_path = extract_dir.join("model_metadata.json");
    let metadata_content =
        fs::read_to_string(&metadata_path).context("Failed to read model_metadata.json")?;
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_content).context("Failed to parse model_metadata.json")?;

    ui::section("Model");
    println!();
    ui::kv("ID", &metadata.model_id);
    ui::kv("Version", &metadata.version);
    if let Some(desc) = &metadata.description {
        ui::kv("Description", desc);
    }
    ui::kv(
        "Preprocessing",
        &format!("{} steps", metadata.preprocessing.len()),
    );
    ui::kv(
        "Postprocessing",
        &format!("{} steps", metadata.postprocessing.len()),
    );
    println!();

    if dry_run {
        return Ok((metadata, None));
    }

    let mut input = if let Some(audio_path) = input_audio {
        ui::kv("Input", &format!("audio ({})", audio_path.display()));
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        ui::kv("Size", &format!("{} bytes", audio_bytes.len()));
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else if let Some(text) = input_text {
        ui::kv("Input", &format!("\"{}\"", text));
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    } else {
        return Ok((metadata, None));
    };

    if let Some(voice_id) = voice {
        ui::kv("Voice", voice_id);
        input
            .metadata
            .insert("voice_id".to_string(), voice_id.to_string());
    }
    println!();

    Ok((metadata, Some(input)))
}

fn run_inference(
    extract_dir: &Path,
    metadata: &ModelMetadata,
    input: &Envelope,
    trace_enabled: bool,
) -> Result<(Envelope, std::time::Duration)> {
    let sp = ui::spinner("Running inference...");

    let base_path = extract_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid extraction path"))?;

    let mut executor = TemplateExecutor::with_base_path(base_path);

    let _inference_span = if trace_enabled {
        let span = crate::tracing_viz::SpanGuard::new(format!("inference:{}", metadata.model_id));
        crate::tracing_viz::add_metadata("model_id", &metadata.model_id);
        crate::tracing_viz::add_metadata("version", &metadata.version);
        Some(span)
    } else {
        None
    };

    let start_time = std::time::Instant::now();
    let output = executor
        .execute(metadata, input, None)
        .map_err(|e| anyhow::anyhow!("Inference failed: {:?}", e))?;
    let elapsed = start_time.elapsed();

    sp.finish_and_clear();

    Ok((output, elapsed))
}

fn print_inference_results(
    metadata: &ModelMetadata,
    output: &Envelope,
    elapsed: std::time::Duration,
    output_path: Option<&PathBuf>,
) -> Result<()> {
    ui::section("Results");
    println!();

    ui::kv(
        "Model",
        &format!("{} v{}", metadata.model_id, metadata.version),
    );
    ui::kv("Time", &format!("{:.2}s", elapsed.as_secs_f32()));
    ui::kv("Output", output.kind_str());

    match &output.kind {
        EnvelopeKind::Text(text) => {
            if !text.is_empty() {
                println!();
                println!("    {}", text);
            }
            if let Some(path) = output_path {
                fs::write(path, text)
                    .with_context(|| format!("Failed to write output to {}", path.display()))?;
                println!();
                ui::ok(&format!("Output saved to {}", path.display()));
            }
        }
        EnvelopeKind::Audio(data) => {
            ui::kv("Size", &format!("{} bytes", data.len()));
            if let Some(path) = output_path {
                save_wav_file(path, data, 24000, 1)
                    .with_context(|| format!("Failed to write audio to {}", path.display()))?;
                println!();
                ui::ok(&format!("Audio saved to {}", path.display()));
            } else {
                println!();
                ui::hint("Use --output <file.wav> to save the audio");
            }
        }
        EnvelopeKind::Embedding(vec) => {
            ui::kv("Dimensions", &format!("{} elements", vec.len()));
            if vec.len() <= 10 {
                println!("    {:?}", vec);
            } else {
                println!("    {:?} ...", &vec[..5]);
            }
            if let Some(path) = output_path {
                let json =
                    serde_json::to_string_pretty(vec).context("Failed to serialize embedding")?;
                fs::write(path, json)
                    .with_context(|| format!("Failed to write embedding to {}", path.display()))?;
                println!();
                ui::ok(&format!("Embedding saved to {}", path.display()));
            }
        }
    }

    println!();
    ui::ok("Inference completed successfully");
    println!();

    Ok(())
}

fn emit_pipeline_start_event(metadata: &ModelMetadata, source: &str) {
    xybrid_sdk::publish_telemetry_event(xybrid_sdk::TelemetryEvent {
        event_type: "PipelineStart".to_string(),
        stage_name: Some(metadata.model_id.clone()),
        target: Some("local".to_string()),
        latency_ms: None,
        error: None,
        data: Some(
            serde_json::json!({
                "model_id": metadata.model_id,
                "version": metadata.version,
                "source": source
            })
            .to_string(),
        ),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    });
}

fn emit_pipeline_complete_event(
    metadata: &ModelMetadata,
    output: &Envelope,
    elapsed: std::time::Duration,
) {
    xybrid_sdk::publish_telemetry_event(xybrid_sdk::TelemetryEvent {
        event_type: "PipelineComplete".to_string(),
        stage_name: Some(metadata.model_id.clone()),
        target: Some("local".to_string()),
        latency_ms: Some(elapsed.as_millis() as u32),
        error: None,
        data: Some(
            serde_json::json!({
                "model_id": metadata.model_id,
                "version": metadata.version,
                "output_type": output.kind_str()
            })
            .to_string(),
        ),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    });
}

//! `xybrid run` command handlers for pipeline, bundle, and model execution.

#![allow(clippy::too_many_arguments)]

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use xybrid_core::context::{DeviceMetrics, StageDescriptor};
use xybrid_core::execution_template::{
    is_mlx_embedding_safetensors_metadata, is_mlx_llm_safetensors_metadata, stage_kind_from_task,
    ExecutionTemplate, ModelMetadata,
};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::policy_engine::PolicyEngine;
use xybrid_core::orchestrator::routing_engine::{LocalAvailability, RoutingEngine};
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::pipeline_config::PipelineConfig;
use xybrid_core::runtime_adapter::{
    select_with_cfg, BackendChoice, RegistryView, SelectionParams, SelectorCfg,
};
use xybrid_core::target::{Platform, TargetResolver};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_sdk::registry_client::RegistryClient;

use super::registry_format::registry_format_for_stage_backend;
use super::utils::{display_stage_name, format_size, maybe_warn_thinking_budget, save_wav_file};
use crate::ui;

/// Run a pipeline from a configuration file.
pub(crate) fn run_pipeline(
    config_path: &PathBuf,
    dry_run: bool,
    policy_path: Option<&PathBuf>,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    input_images: &[PathBuf],
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    target: Option<&str>,
    backend_override: Option<&str>,
    trace_enabled: bool,
    show_reasoning: bool,
    max_tokens: Option<usize>,
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
    let stages = resolve_pipeline_stages(&config, &client, backend_override)?;
    let input = build_input_envelope(input_audio, input_text, input_images, voice, max_tokens)?;

    let metrics = DeviceMetrics::default();
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
        show_reasoning,
        trace_export,
    )
}

fn resolve_pipeline_stages(
    config: &PipelineConfig,
    client: &RegistryClient,
    backend_override: Option<&str>,
) -> Result<Vec<StageDescriptor>> {
    resolve_pipeline_stages_with_selector_cfg(
        config,
        client,
        backend_override,
        &SelectorCfg::current(),
    )
}

fn resolve_pipeline_stages_with_selector_cfg(
    config: &PipelineConfig,
    client: &RegistryClient,
    backend_override: Option<&str>,
    cfg: &SelectorCfg,
) -> Result<Vec<StageDescriptor>> {
    let mut stages = Vec::new();

    for stage_config in &config.stages {
        let model_id = stage_config.model_id();
        let mut desc = StageDescriptor::new(&model_id);
        let stage_backend = backend_override.or_else(|| stage_config.backend());

        if stage_config.is_cloud_stage() {
            configure_cloud_stage(&mut desc, stage_config, &model_id);
        } else {
            resolve_device_stage(&mut desc, &model_id, client, stage_backend, cfg)?;
            desc.backend = canonical_backend_override(stage_backend)?;
        }

        stages.push(desc);
    }

    Ok(stages)
}

fn canonical_backend_override(raw_backend: Option<&str>) -> Result<Option<String>> {
    let Some(raw_backend) = raw_backend else {
        return Ok(None);
    };

    let parsed = BackendChoice::parse(raw_backend)
        .map_err(|err| anyhow::anyhow!("Invalid backend override: {}", err))?;
    Ok(Some(
        parsed.map_or("auto", BackendChoice::as_str).to_string(),
    ))
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
    backend_override: Option<&str>,
    cfg: &SelectorCfg,
) -> Result<()> {
    let format = registry_format_for_stage_backend(client, model_id, backend_override, cfg)?;

    // Offline-first: if the model is already extracted locally, point the stage
    // at its extraction directory and skip the registry. This prevents the
    // circuit breaker from tripping on a cached model when the registry is
    // unreachable.
    let offline_dir = if let Some(format) = format {
        client.resolve_offline_with_format(model_id, format)
    } else {
        client.resolve_offline(model_id)
    };
    if let Some(extract_dir) = offline_dir {
        desc.bundle_path = Some(extract_dir.to_string_lossy().to_string());
        return Ok(());
    }

    // Not cached locally — must fetch from the registry.
    download_model(desc, model_id, client, format)
}

fn download_model(
    desc: &mut StageDescriptor,
    model_id: &str,
    client: &RegistryClient,
    format: Option<&str>,
) -> Result<()> {
    let resolved = if let Some(format) = format {
        client.resolve_with_format(model_id, None, format)
    } else {
        client.resolve(model_id, None)
    };

    match resolved {
        Ok(resolved) => {
            let pb = ui::download_bar(resolved.size_bytes, model_id);

            let fetch_result = if let Some(format) = format {
                client.fetch_extracted_with_format(model_id, None, format, |progress| {
                    let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                    pb.set_position(bytes_done);
                })
            } else {
                client.fetch(model_id, None, |progress| {
                    let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                    pb.set_position(bytes_done);
                })
            };

            match fetch_result {
                Ok(model_path) => {
                    pb.finish_and_clear();
                    ui::ok(&format!("{} downloaded", model_id));
                    desc.bundle_path = Some(model_path.to_string_lossy().to_string());
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
    input_images: &[PathBuf],
    voice: Option<&str>,
    max_tokens: Option<usize>,
) -> Result<Envelope> {
    if !input_images.is_empty() {
        if input_audio.is_some() {
            return Err(anyhow::anyhow!(
                "--input-image cannot be combined with --input-audio"
            ));
        }
        if voice.is_some() {
            return Err(anyhow::anyhow!(
                "--voice cannot be combined with --input-image"
            ));
        }

        let mut input = build_multimodal_input_envelope(input_text, input_images)?;
        if let Some(max_tokens) = max_tokens {
            input
                .metadata
                .insert("max_tokens".to_string(), max_tokens.to_string());
        }
        return Ok(input);
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
        Envelope::new(EnvelopeKind::Text(String::new()))
    };

    if let Some(voice_id) = voice {
        ui::kv("Voice", voice_id);
        input
            .metadata
            .insert("voice_id".to_string(), voice_id.to_string());
    }

    if let Some(max_tokens) = max_tokens {
        input
            .metadata
            .insert("max_tokens".to_string(), max_tokens.to_string());
    }

    Ok(input)
}

fn build_multimodal_input_envelope(
    input_text: Option<&str>,
    input_images: &[PathBuf],
) -> Result<Envelope> {
    let text = input_text.unwrap_or_default();
    if !text.is_empty() {
        ui::kv("Input", &format!("\"{}\"", text));
    }

    let mut images = Vec::with_capacity(input_images.len());
    for image_path in input_images {
        ui::kv("Input image", &image_path.display().to_string());
        let image_bytes = fs::read(image_path)
            .with_context(|| format!("Failed to read image file: {}", image_path.display()))?;
        let format = image_format_hint(image_path)?;
        ui::kv("Image size", &format!("{} bytes", image_bytes.len()));
        images.push(
            Envelope::image(image_bytes, format)
                .with_context(|| format!("Invalid image input: {}", image_path.display()))?,
        );
    }

    Envelope::user_message(text, images).context("Failed to build multimodal user input")
}

fn image_format_hint(path: &Path) -> Result<&str> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| anyhow::anyhow!("Image file has no extension: {}", path.display()))
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
    ui::kv(
        "Battery",
        &format!("{}%", metrics.capabilities.battery_level),
    );
    ui::kv(
        "Thermal",
        &format!("{:?}", metrics.capabilities.thermal_state),
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
            EnvelopeKind::TokenIds(_) => EnvelopeKind::Text("tokens-result".to_string()),
            EnvelopeKind::Image { .. } | EnvelopeKind::MultiPart(_) => {
                EnvelopeKind::Text("vision-output".to_string())
            }
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
    show_reasoning: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    let mut orchestrator = Orchestrator::new();

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

    let bridge = xybrid_sdk::bridge_orchestrator_events(&orchestrator);
    let execution_result = orchestrator.execute_pipeline(stages, input, metrics, availability_fn);
    drop(orchestrator);
    bridge
        .join()
        .map_err(|e| anyhow::anyhow!("Orchestrator event bridge failed: {}", e))?;

    match execution_result {
        Ok(results) => {
            sp.finish_and_clear();
            print_pipeline_results(&results, output_path, show_reasoning)?;
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
    show_reasoning: bool,
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
                // Chain-of-thought for this stage, opt-in via `--show-reasoning`.
                if show_reasoning {
                    if let Some(reasoning) = result
                        .output
                        .metadata
                        .get("reasoning_content")
                        .filter(|r| !r.is_empty())
                    {
                        println!();
                        ui::kv("  Reasoning", "");
                        println!("    {}", reasoning);
                    }
                }
                maybe_warn_thinking_budget(&result.output);
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
            EnvelopeKind::Image { .. } => {
                print_image_summary("  ", &result.output);
            }
            EnvelopeKind::MultiPart(parts) => {
                ui::kv("  Parts", &format!("{}", parts.len()));
            }
            EnvelopeKind::TokenIds(ids) => {
                ui::kv("  Tokens", &format!("{} ids", ids.len()));
                if ids.len() <= 16 {
                    println!("    {:?}", ids);
                } else {
                    println!("    {:?} ...", &ids[..8]);
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
                EnvelopeKind::Image { .. } => {
                    save_image_output(path, &last_result.output)?;
                }
                EnvelopeKind::MultiPart(_) => {
                    save_envelope_json(path, &last_result.output)?;
                }
                EnvelopeKind::TokenIds(ids) => {
                    let json = serde_json::to_string_pretty(ids)
                        .context("Failed to serialize token IDs")?;
                    fs::write(path, json).with_context(|| {
                        format!("Failed to write token IDs to {}", path.display())
                    })?;
                    ui::ok(&format!("Token IDs saved to {}", path.display()));
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
                .map_err(|_| anyhow!("Trace collector lock poisoned"))?
                .to_chrome_trace_json();
            fs::write(export_path, json)
                .with_context(|| format!("Failed to export trace to {}", export_path.display()))?;
            ui::ok(&format!("Trace exported to {}", export_path.display()));
        }
    }

    Ok(())
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn duration_millis_u32(duration: Duration) -> u32 {
    u32::try_from(duration.as_millis()).unwrap_or(u32::MAX)
}

fn unix_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(duration_millis_u64)
        .unwrap_or(0)
}

/// Run inference directly on a .xyb bundle file.
pub(crate) fn run_bundle(
    bundle_path: &Path,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    input_images: &[PathBuf],
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    backend_override: Option<&str>,
    dry_run: bool,
    trace_enabled: bool,
    show_reasoning: bool,
    max_tokens: Option<usize>,
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

    let (metadata, input) = prepare_bundle_execution(
        &extract_dir,
        input_audio,
        input_text,
        input_images,
        voice,
        dry_run,
        backend_override,
        max_tokens,
    )?;

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
        anyhow::anyhow!("No input provided. Use --input-audio <file>, --input-text <text>, or --input-image <file>")
    })?;

    let (output, elapsed) = run_inference(&extract_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path, show_reasoning)?;
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
    input_images: &[PathBuf],
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    platform: Option<&str>,
    backend_override: Option<&str>,
    dry_run: bool,
    trace_enabled: bool,
    show_reasoning: bool,
    max_tokens: Option<usize>,
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
    let selector_cfg = SelectorCfg::current();
    let format =
        registry_format_for_stage_backend(&client, model_id, backend_override, &selector_cfg)?;

    // Offline-first: if the model has already been downloaded and extracted,
    // skip the registry entirely. This keeps `xybrid run` working when the
    // machine is offline (or the registry is unreachable) for any model that
    // was previously fetched.
    let cached_dir = if let Some(format) = format {
        client.resolve_offline_with_format(model_id, format)
    } else {
        client.resolve_offline(model_id)
    };
    let extract_dir = if let Some(cached_dir) = cached_dir {
        println!();
        ui::ok("Using locally cached model");
        ui::kv("Location", &cached_dir.display().to_string());
        println!();
        drop(_fetch_span);
        cached_dir
    } else {
        let resolved = if let Some(format) = format {
            client.resolve_with_format(model_id, platform, format)
        } else {
            client.resolve(model_id, platform)
        }
        .context(format!(
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

        if let Some(format) = format {
            let pb = ui::download_bar(resolved.size_bytes, model_id);
            let dir = client
                .fetch_extracted_with_format(model_id, platform, format, |progress| {
                    let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                    pb.set_position(bytes_done);
                })
                .context(format!(
                    "Failed to fetch model '{}' from registry",
                    model_id
                ))?;
            pb.finish_and_clear();
            ui::ok(&format!("Downloaded {}", model_id));
            ui::kv("Location", &dir.display().to_string());
            println!();
            drop(_fetch_span);
            dir
        } else if resolved.passthrough {
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

    let (metadata, input) = prepare_bundle_execution(
        &extract_dir,
        input_audio,
        input_text,
        input_images,
        voice,
        dry_run,
        backend_override,
        max_tokens,
    )?;

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
        anyhow::anyhow!("No input provided. Use --input-audio <file>, --input-text <text>, or --input-image <file>")
    })?;

    let (output, elapsed) = run_inference(&extract_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path, show_reasoning)?;
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
    input_images: &[PathBuf],
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    backend_override: Option<&str>,
    dry_run: bool,
    trace_enabled: bool,
    show_reasoning: bool,
    max_tokens: Option<usize>,
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

    let (metadata, input) = prepare_bundle_execution(
        dir,
        input_audio,
        input_text,
        input_images,
        voice,
        dry_run,
        backend_override,
        max_tokens,
    )?;

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
        anyhow::anyhow!("No input provided. Use --input-audio <file>, --input-text <text>, or --input-image <file>")
    })?;

    let (output, elapsed) = run_inference(dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path, show_reasoning)?;
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
    input_images: &[PathBuf],
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    backend_override: Option<&str>,
    dry_run: bool,
    trace_enabled: bool,
    show_reasoning: bool,
    max_tokens: Option<usize>,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    if trace_enabled {
        crate::tracing_viz::reset_collector();
    }

    ui::header(&format!("Run · HuggingFace · {}", repo));

    let sp = ui::spinner("Loading from HuggingFace...");
    let mut loader = xybrid_sdk::ModelLoader::from_huggingface_parsed(repo);
    if let Some(raw_backend) = backend_override {
        if let Some(choice) = BackendChoice::parse(raw_backend)
            .map_err(|err| anyhow::anyhow!("Invalid backend override: {}", err))?
        {
            loader = loader.with_backend(choice);
        }
    }
    let model = loader.load().context(format!(
        "Failed to load model from HuggingFace repo '{}'",
        repo
    ))?;
    sp.finish_and_clear();

    ui::ok(&format!("Model loaded: {}", model.model_id()));
    println!();

    let cache_repo = xybrid_sdk::ModelSource::parse_huggingface(repo);
    let sanitized = cache_repo.model_id().unwrap_or(repo).replace('/', "--");
    let cache_dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?
        .join(".xybrid")
        .join("cache")
        .join("hf")
        .join(&sanitized);

    let (metadata, input) = prepare_bundle_execution(
        &cache_dir,
        input_audio,
        input_text,
        input_images,
        voice,
        dry_run,
        backend_override,
        max_tokens,
    )?;

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
        anyhow::anyhow!("No input provided. Use --input-audio <file>, --input-text <text>, or --input-image <file>")
    })?;

    let (output, elapsed) = run_inference(&cache_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path, show_reasoning)?;
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
    input_images: &[PathBuf],
    voice: Option<&str>,
    output_path: Option<&PathBuf>,
    dry_run: bool,
    backend_override: Option<&str>,
    trace_enabled: bool,
    show_reasoning: bool,
    max_tokens: Option<usize>,
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

    let mut metadata = xybrid_sdk::metadata_gen::generate_metadata_for_gguf_file(&gguf_path)
        .map_err(|e| anyhow::anyhow!("Failed to generate metadata for GGUF file: {}", e))?;
    apply_backend_override(&mut metadata, backend_override)?;

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

    let input = if input_audio.is_some() || input_text.is_some() || !input_images.is_empty() {
        build_input_envelope(input_audio, input_text, input_images, voice, max_tokens)?
    } else {
        return Err(anyhow::anyhow!(
            "No input provided. Use --input-audio <file>, --input-text <text>, or --input-image <file>"
        ));
    };

    let (output, elapsed) = run_inference(parent_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path, show_reasoning)?;
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
    input_images: &[PathBuf],
    voice: Option<&str>,
    dry_run: bool,
    backend_override: Option<&str>,
    max_tokens: Option<usize>,
) -> Result<(ModelMetadata, Option<Envelope>)> {
    let metadata_path = extract_dir.join("model_metadata.json");
    let metadata_content =
        fs::read_to_string(&metadata_path).context("Failed to read model_metadata.json")?;
    let mut metadata: ModelMetadata =
        serde_json::from_str(&metadata_content).context("Failed to parse model_metadata.json")?;
    apply_backend_override(&mut metadata, backend_override)?;

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

    if input_audio.is_none() && input_text.is_none() && input_images.is_empty() {
        return Ok((metadata, None));
    }

    let input = build_input_envelope(input_audio, input_text, input_images, voice, max_tokens)?;
    println!();

    Ok((metadata, Some(input)))
}

fn apply_backend_override(
    metadata: &mut ModelMetadata,
    backend_override: Option<&str>,
) -> Result<()> {
    apply_backend_override_with_selector_cfg(metadata, backend_override, &SelectorCfg::current())
}

fn apply_backend_override_with_selector_cfg(
    metadata: &mut ModelMetadata,
    backend_override: Option<&str>,
    cfg: &SelectorCfg,
) -> Result<()> {
    let Some(raw_backend) = backend_override else {
        return Ok(());
    };

    let parsed = BackendChoice::parse(raw_backend)
        .map_err(|err| anyhow::anyhow!("Invalid backend override: {}", err))?;
    let canonical = parsed.map_or("auto", BackendChoice::as_str);

    if let Some(choice) = parsed {
        validate_local_backend_override(metadata, choice, cfg)?;
    }

    metadata.backend = Some(canonical.to_string());
    Ok(())
}

fn validate_local_backend_override(
    metadata: &ModelMetadata,
    choice: BackendChoice,
    cfg: &SelectorCfg,
) -> Result<()> {
    if !metadata_uses_local_backend_selector(metadata) {
        return Ok(());
    }

    validate_local_backend_artifact_compatibility(metadata, choice)?;

    select_with_cfg(
        &SelectionParams::new(&metadata.model_id).with_explicit(Some(choice)),
        &NoRegistryVariants,
        cfg,
    )
    .map(|_| ())
    .map_err(|err| anyhow::anyhow!(err.to_string()))
}

fn metadata_uses_local_backend_selector(metadata: &ModelMetadata) -> bool {
    if matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. })
        || is_mlx_llm_safetensors_metadata(metadata)
        || is_mlx_embedding_safetensors_metadata(metadata)
    {
        return true;
    }

    metadata
        .metadata
        .get("task")
        .and_then(|v| v.as_str())
        .and_then(stage_kind_from_task)
        .is_some_and(|kind| matches!(kind, "llm" | "embed"))
}

fn validate_local_backend_artifact_compatibility(
    metadata: &ModelMetadata,
    choice: BackendChoice,
) -> Result<()> {
    match &metadata.execution_template {
        ExecutionTemplate::Gguf { .. } => match choice {
            BackendChoice::Mlx => Err(anyhow::anyhow!(
                "backend `mlx` does not support local GGUF model `{}`; use `auto`, `llamacpp`, or `mistral`",
                metadata.model_id
            )),
            BackendChoice::LlamaCpp | BackendChoice::Mistral => Ok(()),
        },
        ExecutionTemplate::SafeTensors { architecture, .. } => match choice {
            BackendChoice::Mlx => {
                if metadata_is_mlx_compatible_safetensors(metadata) {
                    Ok(())
                } else {
                    Err(anyhow::anyhow!(
                        "backend `mlx` does not support local SafeTensors architecture `{}` for model `{}`",
                        architecture.as_deref().unwrap_or("unknown"),
                        metadata.model_id
                    ))
                }
            }
            BackendChoice::LlamaCpp | BackendChoice::Mistral => Err(anyhow::anyhow!(
                "backend `{}` does not support local SafeTensors model `{}`; use `auto` or `mlx`",
                choice.as_str(),
                metadata.model_id
            )),
        },
        _ => Ok(()),
    }
}

fn metadata_is_mlx_compatible_safetensors(metadata: &ModelMetadata) -> bool {
    let mut forced = metadata.clone();
    forced.backend = Some("mlx".to_string());
    is_mlx_llm_safetensors_metadata(&forced) || is_mlx_embedding_safetensors_metadata(&forced)
}

struct NoRegistryVariants;

impl RegistryView for NoRegistryVariants {
    fn has_variant(&self, _model_id: &str, _format: &str) -> bool {
        false
    }
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
        .context("Inference failed")?;
    let elapsed = start_time.elapsed();

    sp.finish_and_clear();

    Ok((output, elapsed))
}

fn print_inference_results(
    metadata: &ModelMetadata,
    output: &Envelope,
    elapsed: std::time::Duration,
    output_path: Option<&PathBuf>,
    show_reasoning: bool,
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
            maybe_warn_thinking_budget(output);
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
        EnvelopeKind::Image { .. } => {
            print_image_summary("", output);
            if let Some(path) = output_path {
                save_image_output(path, output)?;
            }
        }
        EnvelopeKind::MultiPart(parts) => {
            ui::kv("Parts", &format!("{}", parts.len()));
            if let Some(path) = output_path {
                save_envelope_json(path, output)?;
            }
        }
        EnvelopeKind::TokenIds(ids) => {
            ui::kv("Tokens", &format!("{} ids", ids.len()));
            if ids.len() <= 16 {
                println!("    {:?}", ids);
            } else {
                println!("    {:?} ...", &ids[..8]);
            }
            if let Some(path) = output_path {
                let json =
                    serde_json::to_string_pretty(ids).context("Failed to serialize token IDs")?;
                fs::write(path, json)
                    .with_context(|| format!("Failed to write token IDs to {}", path.display()))?;
                println!();
                ui::ok(&format!("Token IDs saved to {}", path.display()));
            }
        }
    }

    // Chain-of-thought, opt-in via `--show-reasoning`. Thinking models emit
    // `<think>...</think>` blocks that are stripped from the answer text above;
    // the reasoning is carried on the envelope metadata under "reasoning_content".
    if show_reasoning {
        match output.metadata.get("reasoning_content") {
            Some(reasoning) if !reasoning.is_empty() => {
                println!();
                ui::section("Reasoning");
                println!();
                println!("    {}", reasoning);
            }
            _ => {
                println!();
                ui::hint("No reasoning emitted (model produced no <think> blocks)");
            }
        }
    }

    println!();
    ui::ok("Inference completed successfully");
    println!();

    Ok(())
}

fn print_image_summary(prefix: &str, output: &Envelope) {
    ui::kv(
        &format!("{prefix}Size"),
        &format!("{} bytes", output.payload_size()),
    );
    if let Some(dimensions) = output.image_dimensions() {
        ui::kv(
            &format!("{prefix}Dimensions"),
            &format!("{}x{}", dimensions.width, dimensions.height),
        );
    }
}

fn save_image_output(path: &PathBuf, output: &Envelope) -> Result<()> {
    let (bytes, _format) = output
        .as_image()
        .ok_or_else(|| anyhow::anyhow!("Image output is not encoded"))?;
    fs::write(path, bytes)
        .with_context(|| format!("Failed to write image to {}", path.display()))?;
    ui::ok(&format!("Image saved to {}", path.display()));
    Ok(())
}

fn save_envelope_json(path: &PathBuf, output: &Envelope) -> Result<()> {
    let json = serde_json::to_string_pretty(output).context("Failed to serialize envelope")?;
    fs::write(path, json)
        .with_context(|| format!("Failed to write envelope to {}", path.display()))?;
    ui::ok(&format!("Envelope saved to {}", path.display()));
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
        timestamp_ms: unix_timestamp_millis(),
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
        latency_ms: Some(duration_millis_u32(elapsed)),
        error: None,
        data: Some(
            serde_json::json!({
                "model_id": metadata.model_id,
                "version": metadata.version,
                "output_type": output.kind_str()
            })
            .to_string(),
        ),
        timestamp_ms: unix_timestamp_millis(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_image(width: u32, height: u32) -> Vec<u8> {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb([17, 34, 51]),
        ));
        let mut encoded = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, image::ImageFormat::Png)
            .expect("test image encodes");
        encoded.into_inner()
    }

    #[test]
    fn build_input_envelope_combines_text_and_image_parts() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("fixture.png");
        fs::write(&image_path, png_image(2, 3)).unwrap();

        let input =
            build_input_envelope(None, Some("describe this"), &[image_path], None, None).unwrap();
        let parts = input.as_multipart().expect("text+image input is multipart");

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].as_text(), Some("describe this"));
        assert!(parts[1].is_image());
        assert_eq!(
            parts[1].image_dimensions(),
            Some(xybrid_core::ir::ImageDimensions {
                width: 2,
                height: 3,
            })
        );
    }

    use httpmock::prelude::*;
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    use std::io::Write;
    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    use std::sync::{Mutex, MutexGuard, OnceLock};

    #[test]
    fn duration_millis_conversions_saturate() {
        assert_eq!(
            duration_millis_u64(std::time::Duration::from_millis(42)),
            42
        );
        assert_eq!(
            duration_millis_u64(std::time::Duration::from_secs(u64::MAX)),
            u64::MAX
        );
        assert_eq!(
            duration_millis_u32(std::time::Duration::from_millis(u64::from(u32::MAX) + 1)),
            u32::MAX
        );
    }

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

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn mlx_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("MLX CLI test lock poisoned")
    }

    #[test]
    fn canonical_backend_override_normalizes_case_aliases_and_auto() {
        assert_eq!(
            canonical_backend_override(Some("MLX")).unwrap().as_deref(),
            Some("mlx")
        );
        assert_eq!(
            canonical_backend_override(Some("llama.cpp"))
                .unwrap()
                .as_deref(),
            Some("llamacpp")
        );
        assert_eq!(
            canonical_backend_override(Some("AUTO")).unwrap().as_deref(),
            Some("auto")
        );
        assert_eq!(canonical_backend_override(None).unwrap(), None);
    }

    #[test]
    fn run_pipeline_resolution_rejects_unavailable_explicit_mlx() {
        let config = PipelineConfig::from_yaml(
            r#"
stages:
  - id: llm
    model: qwen3-4b
    target: device
"#,
        )
        .unwrap();
        let client = RegistryClient::with_url("http://127.0.0.1:9").unwrap();

        let err = resolve_pipeline_stages_with_selector_cfg(
            &config,
            &client,
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
    fn build_input_envelope_rejects_corrupt_image_with_redacted_error() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("corrupt.jpeg");
        fs::write(&image_path, [42_u8, 42, 42, 42]).unwrap();

        let err = build_input_envelope(None, Some("describe this"), &[image_path], None, None)
            .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("Invalid image input"));
        assert!(message.contains("invalid or corrupt jpeg image bytes"));
        assert!(!message.contains("[42"));
        assert!(!message.contains("42, 42"));
    }

    #[test]
    fn build_input_envelope_rejects_oversized_image_payload() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("huge.png");
        fs::write(
            &image_path,
            vec![0_u8; xybrid_core::ir::envelope::DEFAULT_MAX_ENCODED_IMAGE_BYTES + 1],
        )
        .unwrap();

        let err = build_input_envelope(None, Some("describe this"), &[image_path], None, None)
            .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("Invalid image input"));
        assert!(message.contains("Image payload too large"));
        assert!(!message.contains("[0"));
    }

    #[test]
    fn run_pipeline_resolution_rejects_llamacpp_for_embedding_task() {
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
        let config = PipelineConfig::from_yaml(&format!(
            r#"
stages:
  - id: embed
    model: {model_id}
    target: device
"#
        ))
        .unwrap();
        let client = RegistryClient::with_url(server.base_url()).unwrap();

        let err = resolve_pipeline_stages_with_selector_cfg(
            &config,
            &client,
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
    fn local_llm_backend_override_rejects_unavailable_explicit_mlx() {
        let mut metadata =
            ModelMetadata::safetensors("qwen3-4b", "1.0", "model.safetensors", "qwen3");

        let err = apply_backend_override_with_selector_cfg(
            &mut metadata,
            Some("mlx"),
            &linux_llamacpp_selector_cfg(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("MLX backend requested but not available"),
            "{err}"
        );
        assert_eq!(metadata.backend, None);
    }

    #[test]
    fn local_llm_backend_override_stamps_explicit_mlx_when_available() {
        let mut metadata =
            ModelMetadata::safetensors("qwen3-4b", "1.0", "model.safetensors", "qwen3");

        apply_backend_override_with_selector_cfg(
            &mut metadata,
            Some("mlx"),
            &apple_mlx_selector_cfg(),
        )
        .unwrap();

        assert_eq!(metadata.backend.as_deref(), Some("mlx"));
    }

    #[test]
    fn local_safetensors_backend_override_rejects_llamacpp_artifact_mismatch() {
        let mut metadata =
            ModelMetadata::safetensors("qwen3-4b", "1.0", "model.safetensors", "qwen3");

        let err = apply_backend_override_with_selector_cfg(
            &mut metadata,
            Some("llamacpp"),
            &apple_mlx_selector_cfg(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("does not support local SafeTensors model"),
            "{err}"
        );
        assert_eq!(metadata.backend, None);
    }

    #[test]
    fn local_gguf_backend_override_rejects_explicit_mlx_artifact_mismatch() {
        let mut metadata = ModelMetadata::onnx("qwen3-4b-gguf", "1.0", "model.gguf");
        metadata.execution_template = ExecutionTemplate::Gguf {
            model_file: "model.gguf".to_string(),
            chat_template: None,
            context_length: 4096,
            generation_params: None,
        };

        let err = apply_backend_override_with_selector_cfg(
            &mut metadata,
            Some("mlx"),
            &apple_mlx_selector_cfg(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("does not support local GGUF model"),
            "{err}"
        );
        assert_eq!(metadata.backend, None);
    }

    #[test]
    fn local_embedding_backend_override_rejects_unavailable_explicit_mlx() {
        let mut metadata =
            ModelMetadata::safetensors("nomic-embed", "1.0", "model.safetensors", "nomic_bert");
        metadata
            .metadata
            .insert("task".to_string(), serde_json::json!("text-embedding"));

        let err = apply_backend_override_with_selector_cfg(
            &mut metadata,
            Some("mlx"),
            &linux_llamacpp_selector_cfg(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("MLX backend requested but not available"),
            "{err}"
        );
        assert_eq!(metadata.backend, None);
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn run_directory_routes_synthetic_mlx_embedding_bundle() {
        let _guard = mlx_test_lock();
        let tmp = tempfile::tempdir().unwrap();
        let prompt = "Hello";
        write_synthetic_canonical_bert_bundle(tmp.path(), prompt);
        assert_run_directory_synthetic_mlx_embedding(tmp.path(), prompt);
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn run_directory_routes_synthetic_mlx_embedding_indexed_shards() {
        let _guard = mlx_test_lock();
        let tmp = tempfile::tempdir().unwrap();
        let prompt = "Hello";
        write_synthetic_canonical_bert_bundle(tmp.path(), prompt);
        convert_single_weights_to_indexed_shard(tmp.path());

        assert_run_directory_synthetic_mlx_embedding(tmp.path(), prompt);
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn assert_run_directory_synthetic_mlx_embedding(dir: &Path, prompt: &str) {
        let mut metadata =
            ModelMetadata::safetensors("synthetic-bert-embed", "1.0", "model.safetensors", "bert");
        metadata.backend = Some("auto".to_string());
        metadata
            .metadata
            .insert("task".to_string(), serde_json::json!("text-embedding"));
        metadata
            .metadata
            .insert("pooling".to_string(), serde_json::json!("mean"));
        metadata
            .metadata
            .insert("normalize".to_string(), serde_json::json!(false));
        metadata
            .metadata
            .insert("max_seq_len".to_string(), serde_json::json!(16));

        fs::write(
            dir.join("model_metadata.json"),
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .unwrap();

        let output_path = dir.join("embedding.json");
        run_directory(
            dir,
            None,
            Some(prompt),
            &[],
            None,
            Some(&output_path),
            Some("mlx"),
            false,
            false,
            false,
            None,
            None,
        )
        .expect("CLI local-directory path should run synthetic MLX embedding bundle");

        let values: Vec<f32> =
            serde_json::from_str(&fs::read_to_string(output_path).unwrap()).unwrap();
        assert_eq!(values.len(), 2);
        assert!((values[0] + 1.0).abs() < 1.0e-5, "got {values:?}");
        assert!((values[1] - 1.0).abs() < 1.0e-5, "got {values:?}");
    }

    #[test]
    fn resolve_pipeline_stages_fetches_backend_format_variant() {
        let server = MockServer::start();
        let model_id = format!("test-llm-{}", uuid::Uuid::new_v4());
        let download_url = format!("{}/model.safetensors", server.base_url());
        let detail_body = format!(
            r#"{{
                "id":"{model_id}",
                "family":"test",
                "task":"text-generation",
                "parameters":1,
                "description":"d",
                "default_variant":null,
                "variants":{{
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
                            "architecture":"qwen3"
                        }},
                        "preprocessing":[],
                        "postprocessing":[],
                        "files":["model.safetensors"]
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

        let config = PipelineConfig::from_yaml(&format!(
            r#"
stages:
  - id: llm
    model: {model_id}
    target: device
    backend: MLX
"#
        ))
        .unwrap();
        let client = RegistryClient::with_url(server.base_url()).unwrap();

        let stages = resolve_pipeline_stages_with_selector_cfg(
            &config,
            &client,
            None,
            &apple_mlx_selector_cfg(),
        )
        .unwrap();

        assert!(
            detail_mock.hits() > 0,
            "explicit backend resolution must inspect registry task metadata"
        );
        assert!(
            resolve_mock.hits() > 0,
            "run pipeline stage resolution must send the registry format query"
        );
        download_mock.assert();
        assert_eq!(stages[0].backend.as_deref(), Some("mlx"));
        assert!(stages[0]
            .bundle_path
            .as_deref()
            .unwrap_or_default()
            .contains("__safetensors"));
    }

    #[test]
    fn resolve_pipeline_stages_auto_fetches_selector_format_variant() {
        let server = MockServer::start();
        let model_id = format!("test-llm-{}", uuid::Uuid::new_v4());
        let download_url = format!("{}/model.safetensors", server.base_url());
        let detail_body = format!(
            r#"{{
                "id":"{model_id}",
                "family":"test",
                "task":"text-generation",
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
                            "architecture":"qwen3"
                        }},
                        "preprocessing":[],
                        "postprocessing":[],
                        "files":["model.safetensors"]
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

        let config = PipelineConfig::from_yaml(&format!(
            r#"
stages:
  - id: llm
    model: {model_id}
    target: device
"#
        ))
        .unwrap();
        let client = RegistryClient::with_url(server.base_url()).unwrap();

        let stages = resolve_pipeline_stages_with_selector_cfg(
            &config,
            &client,
            None,
            &apple_mlx_selector_cfg(),
        )
        .unwrap();

        assert!(
            detail_mock.hits() > 0,
            "run pipeline auto stage resolution must inspect model variants"
        );
        assert!(
            resolve_mock.hits() > 0,
            "run pipeline auto stage resolution must send the selector-chosen format query"
        );
        download_mock.assert();
        assert_eq!(stages[0].backend, None);
        assert!(stages[0]
            .bundle_path
            .as_deref()
            .unwrap_or_default()
            .contains("__safetensors"));
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn write_synthetic_canonical_bert_bundle(dir: &Path, prompt: &str) {
        let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../xybrid-core/tests/fixtures/qwen_tokenizer.json");
        fs::copy(&tok_src, dir.join("tokenizer.json")).expect("copy tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_src).expect("load tokenizer");
        let encoding = tokenizer.encode(prompt, true).expect("encode prompt");
        let token_ids = encoding.get_ids();
        let seq_len = token_ids.len().max(1);
        let vocab_size = token_ids
            .iter()
            .copied()
            .max()
            .map(|id| id as usize + 1)
            .unwrap_or(1);

        let cfg = serde_json::json!({
            "model_type": "bert",
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 4,
            "vocab_size": vocab_size,
            "max_position_embeddings": seq_len,
            "type_vocab_size": 2,
            "layer_norm_eps": 0.0,
            "use_rotary_embeddings": false,
            "use_swiglu": false
        });
        let mut f = fs::File::create(dir.join("config.json")).expect("create config");
        f.write_all(cfg.to_string().as_bytes())
            .expect("write config");

        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_f32_tensor(
            &mut tensors,
            "embeddings.word_embeddings.weight",
            vec![vocab_size, 2],
            vec![0.0; vocab_size * 2],
        );
        let mut position_values = Vec::with_capacity(seq_len * 2);
        for pos in 0..seq_len {
            let base = pos as f32 + 1.0;
            position_values.push(base);
            position_values.push(base * 10.0);
        }
        push_f32_tensor(
            &mut tensors,
            "embeddings.position_embeddings.weight",
            vec![seq_len, 2],
            position_values,
        );
        push_f32_tensor(
            &mut tensors,
            "embeddings.token_type_embeddings.weight",
            vec![2, 2],
            vec![0.0; 4],
        );
        push_f32_tensor(
            &mut tensors,
            "embeddings.LayerNorm.weight",
            vec![2],
            vec![1.0; 2],
        );
        push_f32_tensor(
            &mut tensors,
            "embeddings.LayerNorm.bias",
            vec![2],
            vec![0.0; 2],
        );

        let base = "encoder.layer.0";
        for name in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ] {
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.weight"),
                vec![2, 2],
                vec![0.0; 4],
            );
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.bias"),
                vec![2],
                vec![0.0; 2],
            );
        }
        for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.weight"),
                vec![2],
                vec![1.0; 2],
            );
            push_f32_tensor(
                &mut tensors,
                &format!("{base}.{name}.bias"),
                vec![2],
                vec![0.0; 2],
            );
        }
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.intermediate.dense.weight"),
            vec![4, 2],
            vec![0.0; 8],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.intermediate.dense.bias"),
            vec![4],
            vec![0.0; 4],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.output.dense.weight"),
            vec![2, 4],
            vec![0.0; 8],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.output.dense.bias"),
            vec![2],
            vec![0.0; 2],
        );

        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("tensor view"),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors"))
            .expect("write safetensors");
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn convert_single_weights_to_indexed_shard(dir: &Path) {
        let single = dir.join("model.safetensors");
        let shard_name = "model-00001-of-00001.safetensors";
        let shard = dir.join(shard_name);
        fs::rename(&single, &shard).expect("rename single weights to shard");

        let bytes = fs::read(&shard).expect("read shard");
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes)
            .expect("read shard safetensors metadata");
        let weight_map = meta
            .tensors()
            .into_keys()
            .map(|name| (name, serde_json::Value::String(shard_name.to_string())))
            .collect::<serde_json::Map<_, _>>();
        let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
        fs::write(dir.join("model.safetensors.index.json"), index.to_string())
            .expect("write shard index");
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn push_f32_tensor(
        tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        name: &str,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        tensors.push((name.to_string(), shape, bytes));
    }
}

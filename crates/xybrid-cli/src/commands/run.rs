//! `xybrid run` command handlers for pipeline, bundle, and model execution.

#![allow(clippy::too_many_arguments)]

use anyhow::{Context, Result};
use colored::*;
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

    println!("🚀 Xybrid Pipeline Runner");
    if let Some(name) = &config.name {
        println!("📋 Pipeline: {}\n", name);
    }

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
    let is_cached = client.is_cached(model_id, None).unwrap_or(false);

    if !is_cached {
        download_model(desc, model_id, client)?;
    } else {
        match client.resolve(model_id, None) {
            Ok(resolved) => {
                let cache_path = client.get_cache_path(&resolved);
                desc.bundle_path = Some(cache_path.to_string_lossy().to_string());
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to resolve model '{}': {}",
                    model_id,
                    e
                ));
            }
        }
    }

    Ok(())
}

fn download_model(
    desc: &mut StageDescriptor,
    model_id: &str,
    client: &RegistryClient,
) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};

    println!("📥 Downloading model: {}", model_id);

    match client.resolve(model_id, None) {
        Ok(resolved) => {
            let pb = ProgressBar::new(resolved.size_bytes);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                    )
                    .unwrap()
                    .progress_chars("█▓▒░  "),
            );
            pb.set_message(model_id.to_string());

            match client.fetch(model_id, None, |progress| {
                let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                pb.set_position(bytes_done);
            }) {
                Ok(bundle_path) => {
                    pb.finish_with_message(format!("{} ✓", model_id));
                    desc.bundle_path = Some(bundle_path.to_string_lossy().to_string());
                }
                Err(e) => {
                    pb.abandon_with_message(format!("{} ✗", model_id));
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
        println!("📂 Loading audio file: {}", audio_path.display());
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        println!("   Loaded {} bytes", audio_bytes.len());
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else if let Some(text) = input_text {
        println!("📝 Input text: \"{}\"", text);
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    } else {
        Envelope::new(EnvelopeKind::Text(String::new()))
    };

    if let Some(voice_id) = voice {
        println!("🎙️  Voice: {}", voice_id);
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
    println!("📊 Configuration:");
    println!("   Stages: {}", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        println!("      {}. {}", i + 1, display_stage_name(&stage.name));
    }
    println!();

    println!("📦 Input: {}", input.kind_str());

    println!("📊 Device Metrics (live):");
    println!("   Network RTT: {}ms", metrics.network_rtt);
    println!("   Battery: {}%", metrics.battery);
    println!("   Temperature: {:.1}°C", metrics.temperature);
    println!();

    let platform = Platform::detect();
    let resolved_target = TargetResolver::new()
        .with_requested(target)
        .with_platform(platform)
        .resolve();

    println!("🎯 Target Resolution:");
    println!("   Platform: {}", platform);
    println!("   Requested: {}", target.unwrap_or("(auto)"));
    println!("   Resolved: {}", resolved_target);
    println!();
}

fn run_dry_run(
    stages: &[StageDescriptor],
    input: &Envelope,
    metrics: &xybrid_core::context::DeviceMetrics,
    availability_fn: &dyn Fn(&str) -> LocalAvailability,
) -> Result<()> {
    println!("🔎 Dry Run: Routing Simulation");
    println!("{}", "=".repeat(60));
    println!();

    let mut routing_engine = xybrid_core::orchestrator::routing_engine::DefaultRoutingEngine::new();
    let policy_engine =
        xybrid_core::orchestrator::policy_engine::DefaultPolicyEngine::with_default_policy();

    let mut current_input = input.clone();

    for (i, stage) in stages.iter().enumerate() {
        println!("Stage {}: {}", i + 1, display_stage_name(&stage.name));

        let policy_result = policy_engine.evaluate(&stage.name, &current_input, metrics);
        println!(
            "   Policy: {}",
            if policy_result.allowed {
                "✓ ALLOWED"
            } else {
                "✗ DENIED"
            }
        );
        if let Some(ref reason) = policy_result.reason {
            println!("           {}", reason);
        }

        let availability = availability_fn(&stage.name);
        let routing_decision =
            routing_engine.decide(&stage.name, metrics, &policy_result, &availability);
        println!(
            "   Routing: {} ({})",
            routing_decision.target, routing_decision.reason
        );

        let new_kind = match &current_input.kind {
            EnvelopeKind::Audio(_) => EnvelopeKind::Text("transcribed".to_string()),
            EnvelopeKind::Text(t) => EnvelopeKind::Text(format!("{}-output", t)),
            EnvelopeKind::Embedding(_) => EnvelopeKind::Text("result".to_string()),
        };
        current_input = Envelope::new(new_kind);
        println!("   Output:  {}", current_input.kind_str());
        println!();
    }

    println!("{}", "=".repeat(60));
    println!("✅ Dry run completed - no execution performed");
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
        println!("📜 Loading policy bundle: {}", policy_file.display());
        let policy_bytes = fs::read(policy_file)
            .with_context(|| format!("Failed to read policy file: {}", policy_file.display()))?;

        orchestrator
            .load_policies(policy_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load policies: {}", e))?;

        println!("   ✓ Policy bundle loaded successfully");
        println!();
    }

    println!("⚙️  Executing pipeline...");
    println!("{}", "=".repeat(60));
    println!();

    match orchestrator.execute_pipeline(stages, input, metrics, availability_fn) {
        Ok(results) => {
            print_pipeline_results(&results, output_path)?;
            print_trace_output(trace_enabled, trace_export)?;
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Pipeline execution failed: {}", e);
            Err(anyhow::anyhow!("Pipeline execution failed: {}", e))
        }
    }
}

fn print_pipeline_results(
    results: &[xybrid_core::orchestrator::StageExecutionResult],
    output_path: Option<&PathBuf>,
) -> Result<()> {
    println!();
    println!("📊 Pipeline Results:");
    println!("{}", "=".repeat(60));

    for (i, result) in results.iter().enumerate() {
        println!("\nStage {}: {}", i + 1, display_stage_name(&result.stage));
        println!("  Routing: {}", result.routing_decision.target);
        println!("  Reason: {}", result.routing_decision.reason);
        println!("  Latency: {}ms", result.latency_ms);
        println!("  Output Type: {}", result.output.kind_str());

        match &result.output.kind {
            EnvelopeKind::Text(text) => {
                if !text.is_empty() {
                    println!("  Output Content:");
                    println!("    \"{}\"", text);
                }
            }
            EnvelopeKind::Audio(data) => {
                println!("  Output Size: {} bytes", data.len());
            }
            EnvelopeKind::Embedding(vec) => {
                println!("  Output Dimensions: {} elements", vec.len());
                if vec.len() <= 10 {
                    println!("  Values: {:?}", vec);
                } else {
                    println!("  First 5: {:?}", &vec[..5]);
                }
            }
        }
    }

    save_pipeline_output(results, output_path)?;

    println!();
    println!("{}", "=".repeat(60));
    println!("✨ Pipeline completed successfully!");

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
                    println!();
                    println!("💾 Output saved to: {}", path.display());
                }
                EnvelopeKind::Audio(data) => {
                    save_wav_file(path, data, 24000, 1)
                        .with_context(|| format!("Failed to write audio to {}", path.display()))?;
                    println!();
                    println!("💾 Audio saved to: {}", path.display());
                }
                EnvelopeKind::Embedding(vec) => {
                    let json = serde_json::to_string_pretty(vec)
                        .context("Failed to serialize embedding")?;
                    fs::write(path, json).with_context(|| {
                        format!("Failed to write embedding to {}", path.display())
                    })?;
                    println!();
                    println!("💾 Embedding saved to: {}", path.display());
                }
            }
        }
    } else if let Some(last_result) = results.last() {
        if matches!(last_result.output.kind, EnvelopeKind::Audio(_)) {
            println!();
            println!("💡 Tip: Use --output <file.wav> to save the audio");
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
            println!("💾 Trace exported to: {}", export_path.display());
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

    println!("🚀 Xybrid Bundle Runner");
    println!("📦 Bundle: {}\n", bundle_path.display());

    if !bundle_path.exists() {
        return Err(anyhow::anyhow!(
            "Bundle file not found: {}",
            bundle_path.display()
        ));
    }

    println!("📂 Loading and extracting bundle...");
    let cache = xybrid_sdk::cache::CacheManager::new().context("Failed to create cache manager")?;
    let extract_dir = cache
        .ensure_extracted(bundle_path)
        .context("Failed to extract bundle")?;

    let (metadata, input) =
        prepare_bundle_execution(&extract_dir, input_audio, input_text, voice, dry_run)?;

    emit_pipeline_start_event(&metadata, &bundle_path.display().to_string());

    if dry_run {
        println!("🔎 Dry Run: Bundle inspection only");
        println!("{}", "=".repeat(60));
        println!();
        println!("Bundle is valid and ready for execution.");
        println!("Use without --dry-run to run inference.");
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(&extract_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
    emit_pipeline_complete_event(&metadata, &output, elapsed);

    if trace_enabled {
        drop(_bundle_span);
        print_trace_output(trace_enabled, trace_export)?;
    }

    Ok(())
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

    println!("🚀 Xybrid Model Runner");
    println!("🔖 Model: {}\n", model_id.cyan().bold());

    let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

    let _fetch_span = if trace_enabled {
        Some(crate::tracing_viz::SpanGuard::new("registry_fetch"))
    } else {
        None
    };

    let resolved = client.resolve(model_id, platform).context(format!(
        "Failed to resolve model '{}' from registry",
        model_id
    ))?;

    println!("📦 Resolved variant:");
    println!("   Repository: {}", resolved.hf_repo);
    println!("   File: {}", resolved.file);
    println!(
        "   Size: {}",
        format_size(resolved.size_bytes).bright_cyan()
    );
    println!("   Format: {} ({})", resolved.format, resolved.quantization);
    println!();

    let bundle_path = fetch_or_cache(&client, model_id, platform, &resolved)?;
    println!("   Location: {}", bundle_path.display());
    println!();

    drop(_fetch_span);

    println!("📂 Loading and extracting bundle...");
    let cache = xybrid_sdk::cache::CacheManager::new().context("Failed to create cache manager")?;
    let extract_dir = cache
        .ensure_extracted(&bundle_path)
        .context("Failed to extract bundle")?;

    let (metadata, input) =
        prepare_bundle_execution(&extract_dir, input_audio, input_text, voice, dry_run)?;

    emit_pipeline_start_event(&metadata, "registry");

    if dry_run {
        println!("🔎 Dry Run: Model inspection only");
        println!("{}", "=".repeat(60));
        println!();
        println!("Model is valid and ready for execution.");
        println!("Use without --dry-run to run inference.");
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(&extract_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
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

    println!("🚀 Xybrid Model Runner (local directory)");
    println!(
        "📂 Directory: {}\n",
        dir.display().to_string().cyan().bold()
    );

    if !dir.exists() {
        return Err(anyhow::anyhow!("Directory not found: {}", dir.display()));
    }

    let (metadata, input) = prepare_bundle_execution(dir, input_audio, input_text, voice, dry_run)?;

    emit_pipeline_start_event(&metadata, "directory");

    if dry_run {
        println!("🔎 Dry Run: Model inspection only");
        println!("{}", "=".repeat(60));
        println!("\nModel is valid and ready for execution.");
        println!("Use without --dry-run to run inference.");
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
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

    println!("🚀 Xybrid Model Runner (HuggingFace)");
    println!("🤗 Repo: {}\n", repo.cyan().bold());

    println!("📥 Loading from HuggingFace (downloading if needed)...");
    let loader = xybrid_sdk::ModelLoader::from_huggingface(repo);
    let model = loader.load().context(format!(
        "Failed to load model from HuggingFace repo '{}'",
        repo
    ))?;

    println!("✅ Model loaded: {}", model.model_id().cyan());
    println!();

    // Resolve the cache directory for direct execution
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
        println!("🔎 Dry Run: Model inspection only");
        println!("{}", "=".repeat(60));
        println!("\nModel is valid and ready for execution.");
        println!("Use without --dry-run to run inference.");
        return Ok(());
    }

    let input = input.ok_or_else(|| {
        anyhow::anyhow!("No input provided. Use --input-audio <file> or --input-text <text>")
    })?;

    let (output, elapsed) = run_inference(&cache_dir, &metadata, &input, trace_enabled)?;

    print_inference_results(&metadata, &output, elapsed, output_path)?;
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
        println!("✅ Model is cached");
        Ok(client.get_cache_path(resolved))
    } else {
        println!("📥 Downloading model...");

        use indicatif::{ProgressBar, ProgressStyle};
        let pb = ProgressBar::new(resolved.size_bytes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} Downloading {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("█▓▒░  ")
        );
        pb.set_message(model_id.to_string());

        let path = client
            .fetch(model_id, platform, |progress| {
                let bytes_done = (progress * resolved.size_bytes as f32) as u64;
                pb.set_position(bytes_done);
            })
            .context(format!(
                "Failed to fetch model '{}' from registry",
                model_id
            ))?;

        pb.finish_with_message(format!("✅ Downloaded {}", model_id));
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

    println!("📋 Model Metadata:");
    println!("   ID: {}", metadata.model_id);
    println!("   Version: {}", metadata.version);
    if let Some(desc) = &metadata.description {
        println!("   Description: {}", desc);
    }
    println!("   Preprocessing: {} steps", metadata.preprocessing.len());
    println!("   Postprocessing: {} steps", metadata.postprocessing.len());
    println!();

    if dry_run {
        return Ok((metadata, None));
    }

    let mut input = if let Some(audio_path) = input_audio {
        println!("🎤 Loading audio file: {}", audio_path.display());
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        println!("   Loaded {} bytes", audio_bytes.len());
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else if let Some(text) = input_text {
        println!("📝 Input text: \"{}\"", text);
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    } else {
        return Ok((metadata, None));
    };

    if let Some(voice_id) = voice {
        println!("🎙️  Voice: {}", voice_id);
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
    println!("⚙️  Running inference...");
    println!("{}", "=".repeat(60));

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

    Ok((output, elapsed))
}

fn print_inference_results(
    metadata: &ModelMetadata,
    output: &Envelope,
    elapsed: std::time::Duration,
    output_path: Option<&PathBuf>,
) -> Result<()> {
    println!();
    println!("📊 Results:");
    println!("{}", "=".repeat(60));
    println!();
    println!("  Model: {} v{}", metadata.model_id, metadata.version);
    println!("  Latency: {:.2}ms", elapsed.as_millis());
    println!("  Output Type: {}", output.kind_str());

    match &output.kind {
        EnvelopeKind::Text(text) => {
            if !text.is_empty() {
                println!();
                println!("  Output:");
                println!("    \"{}\"", text);
            }
            if let Some(path) = output_path {
                fs::write(path, text)
                    .with_context(|| format!("Failed to write output to {}", path.display()))?;
                println!();
                println!("💾 Output saved to: {}", path.display());
            }
        }
        EnvelopeKind::Audio(data) => {
            println!("  Output Size: {} bytes", data.len());
            if let Some(path) = output_path {
                save_wav_file(path, data, 24000, 1)
                    .with_context(|| format!("Failed to write audio to {}", path.display()))?;
                println!();
                println!("💾 Audio saved to: {}", path.display());
            } else {
                println!();
                println!("💡 Tip: Use --output <file.wav> to save the audio");
            }
        }
        EnvelopeKind::Embedding(vec) => {
            println!("  Output Dimensions: {} elements", vec.len());
            if vec.len() <= 10 {
                println!("  Values: {:?}", vec);
            } else {
                println!("  First 5: {:?}", &vec[..5]);
            }
            if let Some(path) = output_path {
                let json =
                    serde_json::to_string_pretty(vec).context("Failed to serialize embedding")?;
                fs::write(path, json)
                    .with_context(|| format!("Failed to write embedding to {}", path.display()))?;
                println!();
                println!("💾 Embedding saved to: {}", path.display());
            }
        }
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("✨ Inference completed successfully!");

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

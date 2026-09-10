//! `xybrid repl` command handler - interactive REPL mode.

#![allow(clippy::too_many_arguments)]

mod agent_loop;
mod targeting;
mod tools;
mod warmup;

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use xybrid_core::context::{DeviceMetrics, StageDescriptor};
use xybrid_core::conversation::ConversationContext;
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
use xybrid_core::orchestrator::routing_engine::LocalAvailability;
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::pipeline::ExecutionTarget;
use xybrid_core::pipeline_config::PipelineConfig;
use xybrid_sdk::model::ModelLoader;
use xybrid_sdk::registry_client::RegistryClient;

use colored::Colorize;

use targeting::{
    parse_repl_target, parse_stage_config_target, stage_config_allows_local_cache,
    stage_is_locally_available, target_allows_local,
};
use warmup::warmup_models;

use super::utils::{maybe_warn_thinking_budget, thinking_budget_exhausted, THINKING_BUDGET_HINT};
use crate::ui;

/// Arguments for the interactive REPL, grouped to keep the entry point legible.
pub(crate) struct ReplArgs {
    pub config: Option<PathBuf>,
    pub model: Option<String>,
    pub model_file: Option<PathBuf>,
    pub huggingface: Option<String>,
    pub voice: Option<String>,
    pub target: Option<String>,
    pub stream: bool,
    pub show_reasoning: bool,
    pub max_tokens: Option<usize>,
    pub system_prompt: Option<String>,
    /// Serve from cloud while the registry model downloads, then switch local.
    pub speculative_cloud: bool,
    pub no_tools: bool,
    pub tools_file: Option<PathBuf>,
    pub verbose: u8,
}

/// Interactive REPL mode - keeps models loaded for fast repeated inference.
pub(crate) fn handle_repl_command(args: ReplArgs) -> Result<()> {
    let ReplArgs {
        config,
        model,
        model_file,
        huggingface,
        voice,
        target,
        stream,
        show_reasoning,
        max_tokens,
        system_prompt,
        speculative_cloud,
        no_tools,
        tools_file,
        verbose,
    } = args;
    use std::io::{self, Write};

    ui::brand_with_version(env!("CARGO_PKG_VERSION"));
    println!();
    ui::hint("Models loaded once and kept warm for fast inference");
    ui::hint("Type 'quit' or 'exit' to exit. Type 'help' for commands.");

    print_streaming_status(stream);
    let execution_target = parse_repl_target(target.as_deref())?;
    if let Some(target) = &execution_target {
        ui::kv("Target", target.as_str());
    }
    println!();

    // Speculative cloud only applies to a bare registry --model (not config /
    // HuggingFace / GGUF file). When it engages, the model serves from cloud
    // immediately and the weights download in the background.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    let want_speculative = speculative_cloud
        && model.is_some()
        && config.is_none()
        && huggingface.is_none()
        && model_file.is_none();
    // Without LLM features the loop has no model-driven path that could consume
    // the cloud-backed handle — fall through to the normal blocking download.
    #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
    let want_speculative = {
        if speculative_cloud {
            ui::warning(
                "--speculative-cloud requires LLM features (llm-llamacpp) — downloading, then running locally",
            );
        }
        false
    };

    // Holds the cloud-backed (or already-local) model produced by the
    // speculative path, installed into `loaded_model` below.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    let mut speculative_model: Option<xybrid_sdk::model::XybridModel> = None;

    // --huggingface: load from HuggingFace repo
    let stages = if want_speculative {
        let model_id = model.clone().expect("want_speculative implies a model id");
        // Serve the registry model itself: the gateway routes its id to xycloud
        // (the CPU cluster that runs the edge model) while it downloads locally.
        let loader = ModelLoader::from_registry(&model_id).with_speculative_cloud(true);

        if loader.will_speculate() {
            ui::ok(&format!(
                "Speculative cloud: serving '{}' via xycloud while it downloads in the background",
                model_id
            ));
        } else if xybrid_sdk::cache::CacheManager::new()
            .map(|c| c.is_extracted(&model_id))
            .unwrap_or(false)
        {
            ui::hint("Model already cached locally — running on device (no speculation needed)");
        } else {
            ui::hint(
                "Speculative cloud unavailable (no API key?) — downloading, then running locally",
            );
        }

        let model_obj = loader
            .load()
            .context("Failed to load speculative cloud model")?;
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        {
            speculative_model = Some(model_obj);
        }
        #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
        {
            drop(model_obj);
        }

        // No bundle_path: the weights aren't on disk yet, so warmup and the
        // local-load block skip this stage — the speculative model drives the
        // loop and transparently switches to local once the download lands.
        let mut stage = StageDescriptor::new(&model_id);
        stage.target = execution_target.clone();
        vec![stage]
    } else if let Some(ref repo) = huggingface {
        let sp = ui::spinner(&format!("Loading from HuggingFace: {}...", repo));
        let loader = ModelLoader::from_huggingface_parsed(repo);
        let _model = loader.load().context(format!(
            "Failed to load model from HuggingFace repo '{}'",
            repo
        ))?;

        let cache_repo = xybrid_sdk::ModelSource::parse_huggingface(repo);
        let repo_id = cache_repo.model_id().unwrap_or(repo);
        let cache_dir = xybrid_sdk::CacheManager::new()
            .context("Failed to open the model cache")?
            .huggingface_cache_dir(repo_id)
            .ok_or_else(|| {
                anyhow::anyhow!("HuggingFace model '{repo_id}' has no materialized cache directory")
            })?;

        sp.finish_and_clear();
        ui::ok("Model loaded from HuggingFace");

        let mut stage = StageDescriptor::new(_model.model_id());
        stage.bundle_path = Some(cache_dir.to_string_lossy().to_string());
        stage.target = execution_target.clone();
        vec![stage]
    } else if let Some(ref gguf_path) = model_file {
        // --model-file: load a bare GGUF file with auto-generated metadata
        let gguf_path = gguf_path
            .canonicalize()
            .with_context(|| format!("GGUF file not found: {}", gguf_path.display()))?;

        let metadata = xybrid_sdk::metadata_gen::generate_metadata_for_gguf_file(&gguf_path)
            .map_err(|e| anyhow::anyhow!("Failed to generate metadata for GGUF file: {}", e))?;

        let parent_dir = gguf_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine parent directory of GGUF file"))?;

        ui::kv("GGUF", &gguf_path.display().to_string());
        if verbose > 0 {
            ui::kv("Model ID", &metadata.model_id);
            if let xybrid_core::execution::ExecutionTemplate::Gguf { context_length, .. } =
                &metadata.execution_template
            {
                ui::kv("Context", &context_length.to_string());
            }
            if let Some(arch) = metadata.metadata.get("architecture") {
                ui::kv("Architecture", &arch.to_string());
            }
        }

        // Write metadata to parent dir so ModelLoader can find it
        let metadata_path = parent_dir.join("model_metadata.json");
        let needs_write = !metadata_path.exists();
        if needs_write {
            let json = serde_json::to_string_pretty(&metadata)?;
            fs::write(&metadata_path, &json)?;
            if verbose > 0 {
                ui::hint("Generated model_metadata.json");
            }
        }

        let mut stage = StageDescriptor::new(metadata.model_id.clone());
        stage.bundle_path = Some(parent_dir.to_string_lossy().to_string());
        stage.target = execution_target.clone();
        vec![stage]
    } else {
        let client = RegistryClient::from_env().context("Failed to initialize registry client")?;

        let (config_path, model_id) = if let Some(config) = config {
            (Some(config), None)
        } else if let Some(model) = model {
            (None, Some(model))
        } else {
            return Err(anyhow::anyhow!(
                "Either --config, --model, --model-file, or --huggingface must be specified"
            ));
        };

        let pipeline_config = if let Some(ref path) = config_path {
            let content = fs::read_to_string(path)
                .with_context(|| format!("Failed to read config: {}", path.display()))?;
            Some(PipelineConfig::from_yaml(&content)?)
        } else {
            None
        };

        load_stages(
            &client,
            &pipeline_config,
            &model_id,
            execution_target.as_ref(),
        )?
    };

    let mut conversation_context: Option<ConversationContext> = None;
    let mut loaded_model: Option<xybrid_sdk::model::XybridModel> = None;

    if stages.len() == 1 && stage_is_locally_available(&stages[0]) {
        let bundle_path = PathBuf::from(stages[0].bundle_path.as_ref().unwrap());
        let model_result = if bundle_path.extension().is_some_and(|ext| ext == "xyb") {
            ModelLoader::from_bundle(&bundle_path).and_then(|loader| loader.load())
        } else {
            ModelLoader::from_directory(&bundle_path).and_then(|loader| loader.load())
        };

        if let Ok(model) = model_result {
            if model.is_llm() {
                ui::ok("LLM detected — conversation context enabled");
                let mut ctx = ConversationContext::new();
                if let Some(ref prompt) = system_prompt {
                    ui::kv("System", prompt);
                    ctx = ctx.with_system(
                        Envelope::new(EnvelopeKind::Text(prompt.clone()))
                            .with_role(MessageRole::System),
                    );
                }
                conversation_context = Some(ctx);
                if verbose > 0 {
                    ui::hint("Use 'history' to view conversation, 'clear' to reset");
                }
            }
            loaded_model = Some(model);
        }
    }

    // Install the speculative model. Its placeholder handle isn't locally
    // available, so the block above skipped it. Speculation targets LLM/chat,
    // so enable conversation context up front (the placeholder can't report
    // `is_llm()` until the local weights land).
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    if loaded_model.is_none() {
        if let Some(model) = speculative_model.take() {
            let mut ctx = ConversationContext::new();
            if let Some(ref prompt) = system_prompt {
                ui::kv("System", prompt);
                ctx = ctx.with_system(
                    Envelope::new(EnvelopeKind::Text(prompt.clone()))
                        .with_role(MessageRole::System),
                );
            }
            conversation_context = Some(ctx);
            loaded_model = Some(model);
        }
    }

    // Tool calling: on by default when the bundle's metadata declares
    // support (`tool_calling: true`), off via --no-tools or `/tools off`.
    // The flag is advisory and per-model — most bundles do not declare it,
    // so this is effectively a per-model allowlist. `--tools-file` adds
    // user-defined tools (and counts as an explicit opt-in for models
    // whose metadata is silent).
    let user_tools = match &tools_file {
        Some(path) => tools::load_user_tools(path)
            .with_context(|| format!("Failed to load tools file: {}", path.display()))?,
        None => Vec::new(),
    };
    let mut tools_state = ToolsState::resolve(
        loaded_model.as_ref(),
        no_tools,
        user_tools,
        tools_file.is_some(),
    );
    if tools_state.active() {
        ui::ok("Tool calling: on");
        ui::hint("web_search / fetch_url reach the network; current_time stays local");
        ui::kv("Search", tools_state.toolbox.provider.label());
        if !tools_state.toolbox.user_tools.is_empty() {
            let names: Vec<&str> = tools_state
                .toolbox
                .user_tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect();
            ui::kv("User tools", &names.join(", "));
        }
        ui::hint("Disable with --no-tools or '/tools off'");
    }

    // Steering for the tool loop: the user's --system wins; otherwise a
    // default prompt that licenses tool use with a clear stop condition.
    let resolved_system = system_prompt.clone().or_else(|| {
        tools_state
            .active()
            .then(|| agent_loop::TOOL_SYSTEM_PROMPT.to_string())
    });
    if system_prompt.is_none() && tools_state.active() {
        conversation_context = conversation_context.map(|ctx| {
            ctx.with_system(
                Envelope::new(EnvelopeKind::Text(
                    agent_loop::TOOL_SYSTEM_PROMPT.to_string(),
                ))
                .with_role(MessageRole::System),
            )
        });
    }

    // Streaming for the LLM chat route is resolved once up front.
    let llm_stream = {
        let is_llm = loaded_model.as_ref().is_some_and(|m| m.is_llm());
        let supports = loaded_model
            .as_ref()
            .is_some_and(|m| m.supports_token_streaming());
        if stream && is_llm && !supports {
            ui::warning("Token streaming not available (LLM features not compiled) — using batch");
        } else if stream && show_reasoning {
            ui::hint("Token streaming disabled so reasoning can be shown before the answer");
        }
        stream && supports && !show_reasoning
    };

    let metrics = DeviceMetrics::default();

    let stage_bundle_paths: std::collections::HashMap<String, bool> = stages
        .iter()
        .map(|s| (s.name.clone(), stage_is_locally_available(s)))
        .collect();
    let availability_fn = move |stage: &str| -> LocalAvailability {
        LocalAvailability::new(stage_bundle_paths.get(stage).copied().unwrap_or(false))
    };

    let mut orchestrator = Orchestrator::new();
    let bridge = xybrid_sdk::bridge_orchestrator_events(&orchestrator);

    warmup_models(&stages);

    println!();
    ui::hint("Enter text and press Enter to run inference");
    ui::hint("Use '/image <path>' to attach an image to the next message");
    println!("  {}", "─".repeat(50).truecolor(60, 60, 70));

    let stdin = io::stdin();
    let mut pending_images = ReplPendingImages::default();
    loop {
        print!("\n  {} ", "❯".truecolor(120, 180, 255).bold());
        io::stdout().flush()?;

        let mut input_line = String::new();
        if stdin.read_line(&mut input_line)? == 0 {
            break;
        }

        let input_line = input_line.trim();

        let handled = handle_special_command(
            input_line,
            &mut conversation_context,
            &mut pending_images,
            &mut tools_state,
            verbose,
        );

        match handled {
            SpecialCommandResult::Quit => break,
            SpecialCommandResult::Continue => continue,
            SpecialCommandResult::NotSpecial => {}
        }

        // LLM chat route: a single locally-loaded GGUF model runs through
        // the agent loop — conversation context, built-in tool calling, and
        // token streaming in one path. Image turns and pipelines take the
        // general path below.
        if pending_images.is_empty() {
            if let Some(model) = loaded_model.as_ref().filter(|m| m.is_llm()) {
                let start = std::time::Instant::now();
                match agent_loop::run_query(
                    model,
                    conversation_context.as_ref(),
                    input_line,
                    tools_state.active().then_some(&tools_state.toolbox),
                    resolved_system.as_deref(),
                    llm_stream,
                    show_reasoning,
                    max_tokens,
                    verbose,
                ) {
                    Ok(outcome) => {
                        let elapsed = start.elapsed();
                        if outcome.already_printed {
                            println!();
                        } else {
                            println!();
                            println!("  {}", outcome.answer);
                        }
                        if thinking_budget_exhausted(
                            &outcome.answer,
                            outcome.finish_reason.as_deref(),
                            outcome.reasoning_present.then_some("present"),
                        ) {
                            ui::hint(THINKING_BUDGET_HINT);
                        }

                        // Push user + assistant AFTER the run — pushing the
                        // input before would double it in the prompt.
                        if let Some(ref mut ctx) = conversation_context {
                            ctx.push(
                                Envelope::new(EnvelopeKind::Text(input_line.to_string()))
                                    .with_role(MessageRole::User),
                            );
                            ctx.push(
                                Envelope::new(EnvelopeKind::Text(outcome.answer.clone()))
                                    .with_role(MessageRole::Assistant),
                            );
                            if verbose > 1 {
                                ui::hint(&format!(
                                    "Context updated (total: {} messages)",
                                    ctx.history().len()
                                ));
                            }
                        }

                        println!();
                        print_llm_chat_stats(&outcome, elapsed);
                    }
                    Err(e) => ui::err(&format!("{e:#}")),
                }
                continue;
            }
        }

        let input = match build_repl_input(
            input_line,
            voice.as_deref(),
            conversation_context.is_some(),
            &mut pending_images,
            max_tokens,
        ) {
            Ok(input) => input,
            Err(e) => {
                ui::err(&format!("{}", e));
                continue;
            }
        };

        let start = std::time::Instant::now();

        // Try streaming execution
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        let use_streaming = {
            let can_stream = stages.len() == 1 && {
                let locally_available = stage_is_locally_available(&stages[0]);
                // The speculative cloud-backed model has no local bundle, so the
                // orchestrator can't run its stage: the model must drive the turn
                // directly (tokens print as they arrive) even without --stream —
                // regardless of --show-reasoning. Local models stream only when
                // asked and when not showing reasoning (which needs the whole
                // answer up front).
                let speculative_drives = loaded_model.is_some() && !locally_available;
                speculative_drives || (stream && !show_reasoning && locally_available)
            };
            if stream && show_reasoning && !can_stream {
                ui::hint("Token streaming disabled so reasoning can be shown before the answer");
            } else if stream && !can_stream {
                ui::warning("Streaming conditions not met");
                if verbose > 0 {
                    ui::hint(&format!("stages.len() = {} (need 1)", stages.len()));
                    ui::hint(&format!(
                        "bundle_path = {:?}",
                        stages.first().map(|s| &s.bundle_path)
                    ));
                }
            }
            can_stream
        };

        #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
        let use_streaming = {
            if stream {
                ui::warning("Streaming requested but LLM features not enabled");
                ui::hint("Build with: --features llm-llamacpp (or llm-mistral)");
            }
            false
        };

        if use_streaming {
            #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
            {
                let did_stream = try_streaming_execution(
                    &stages,
                    &input,
                    &mut conversation_context,
                    &loaded_model,
                    max_tokens,
                    start,
                    verbose,
                );
                if did_stream {
                    continue;
                }
            }
        }

        // Non-streaming execution path (default)
        execute_batch(
            &mut orchestrator,
            &stages,
            &input,
            &metrics,
            &availability_fn,
            &mut conversation_context,
            start,
            show_reasoning,
            verbose,
        );
    }

    drop(orchestrator);
    bridge
        .join()
        .map_err(|e| anyhow::anyhow!("Orchestrator event bridge failed: {}", e))?;

    Ok(())
}

fn print_streaming_status(stream: bool) {
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    if stream {
        ui::ok("Token streaming: enabled");
    }
    #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
    if stream {
        ui::warning("Token streaming: not available (LLM features not compiled)");
    }
}

fn load_stages(
    client: &RegistryClient,
    pipeline_config: &Option<PipelineConfig>,
    model_id: &Option<String>,
    execution_target: Option<&ExecutionTarget>,
) -> Result<Vec<StageDescriptor>> {
    let mut stages = Vec::new();

    if let Some(ref config) = pipeline_config {
        let name = config.name.as_deref().unwrap_or("unnamed");
        ui::kv("Pipeline", name);
        for stage_config in &config.stages {
            let model_id = stage_config.model_id();
            let mut desc = StageDescriptor::new(&model_id);
            let configured_target = parse_stage_config_target(stage_config);
            desc.target = execution_target.cloned().or(configured_target);

            if stage_config_allows_local_cache(stage_config, desc.target.as_ref()) {
                ensure_model_cached(&mut desc, &model_id, client)?;
            }
            stages.push(desc);
        }
    } else if let Some(ref model_id) = model_id {
        ui::kv("Model", model_id);
        let mut desc = StageDescriptor::new(model_id);
        desc.target = execution_target.cloned();
        if target_allows_local(desc.target.as_ref()) {
            ensure_model_cached(&mut desc, model_id, client)?;
        }
        stages.push(desc);
    }

    Ok(stages)
}

fn ensure_model_cached(
    desc: &mut StageDescriptor,
    model_id: &str,
    client: &RegistryClient,
) -> Result<()> {
    let resolved = client.resolve(model_id, None)?;

    if !client.is_cached(model_id, None).unwrap_or(false) {
        let pb = ui::download_bar(resolved.size_bytes, model_id);
        let model_dir = client.fetch_extracted(model_id, None, |p| {
            pb.set_position((p * resolved.size_bytes as f32) as u64);
        })?;
        pb.finish_and_clear();
        ui::ok(&format!("{} downloaded", model_id));
        desc.bundle_path = Some(model_dir.to_string_lossy().to_string());
    } else if resolved.passthrough {
        // Passthrough models: extraction dir is managed by fetch_extracted (idempotent)
        let model_dir = client.fetch_extracted(model_id, None, |_| {})?;
        desc.bundle_path = Some(model_dir.to_string_lossy().to_string());
    } else {
        // Standard .xyb bundle: extract from cache
        let cache = xybrid_sdk::cache::CacheManager::new()?;
        let xyb_path = client.get_cache_path(&resolved);
        let model_dir = cache.ensure_extracted(&xyb_path)?;
        desc.bundle_path = Some(model_dir.to_string_lossy().to_string());
    }
    Ok(())
}

enum SpecialCommandResult {
    Quit,
    Continue,
    NotSpecial,
}

/// Session state for tool calling (built-ins + `--tools-file` user tools).
struct ToolsState {
    /// Tools may be offered to this model at all.
    available: bool,
    /// Session toggle (`--no-tools`, `/tools on|off`).
    enabled: bool,
    toolbox: tools::ToolBox,
}

impl ToolsState {
    fn resolve(
        model: Option<&xybrid_sdk::model::XybridModel>,
        no_tools: bool,
        user_tools: Vec<tools::UserTool>,
        explicit_tools_file: bool,
    ) -> Self {
        let is_llm = model.is_some_and(|m| m.is_llm());
        let declared = model.and_then(|m| m.supports_tool_calling());
        // The advisory metadata flag gates the default. An explicit
        // --tools-file is a user opt-in that overrides *silence* (a model
        // whose template cannot render tools still fails loudly at run
        // time) — but not an explicit `tool_calling: false`.
        let available = is_llm
            && match declared {
                Some(true) => true,
                Some(false) => false,
                None => explicit_tools_file,
            };
        if explicit_tools_file {
            if !is_llm {
                ui::warning("--tools-file ignored: no locally-loaded LLM in this session");
            } else if declared == Some(false) {
                ui::warning("The model's metadata declares tool_calling: false — tools stay off");
            } else if no_tools {
                ui::warning("--no-tools wins over --tools-file: tools are off");
            }
        }
        let (provider, warning) = tools::SearchProvider::from_env();
        if available && !no_tools {
            if let Some(warning) = warning {
                ui::warning(&warning);
            }
        }
        Self {
            available,
            enabled: available && !no_tools,
            toolbox: tools::ToolBox {
                provider,
                user_tools,
            },
        }
    }

    fn active(&self) -> bool {
        self.available && self.enabled
    }
}

/// Post-query stats line for the LLM chat route: token throughput when the
/// answer streamed, otherwise wall-clock (with the tool-call count when the
/// turn used tools).
fn print_llm_chat_stats(outcome: &agent_loop::QueryOutcome, elapsed: std::time::Duration) {
    if let Some(stats) = &outcome.stream_stats {
        let ttft_ms = stats.ttft.map(|d| d.as_millis()).unwrap_or(0);
        let decode_tok_s = stats.ttft.and_then(|ttft| {
            let decode_time = elapsed.saturating_sub(ttft).as_secs_f64();
            (stats.tokens >= 2 && decode_time > 0.001)
                .then(|| (stats.tokens - 1) as f64 / decode_time)
        });
        match decode_tok_s {
            Some(tok_s) => ui::hint(&format!(
                "{} tokens in {:.2}s ({:.1} tok/s, {}ms to first token)",
                stats.tokens,
                elapsed.as_secs_f64(),
                tok_s,
                ttft_ms
            )),
            None => ui::hint(&format!(
                "{} tokens in {:.2}s",
                stats.tokens,
                elapsed.as_secs_f64()
            )),
        }
    } else if outcome.tool_calls_run > 0 {
        ui::hint(&format!(
            "Inference time: {:.2}s ({} tool call{})",
            elapsed.as_secs_f32(),
            outcome.tool_calls_run,
            if outcome.tool_calls_run == 1 { "" } else { "s" }
        ));
    } else {
        ui::hint(&format!("Inference time: {:.2}s", elapsed.as_secs_f32()));
    }
}

fn print_reasoning(show_reasoning: bool, reasoning: Option<&str>) {
    if !show_reasoning {
        return;
    }

    match reasoning {
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

#[derive(Default)]
struct ReplPendingImages {
    paths: Vec<PathBuf>,
}

impl ReplPendingImages {
    fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    fn len(&self) -> usize {
        self.paths.len()
    }

    fn push(&mut self, path: PathBuf) {
        self.paths.push(path);
    }

    fn clear(&mut self) {
        self.paths.clear();
    }
}

fn handle_special_command(
    input: &str,
    conversation_context: &mut Option<ConversationContext>,
    pending_images: &mut ReplPendingImages,
    tools_state: &mut ToolsState,
    verbose: u8,
) -> SpecialCommandResult {
    if let Some(result) = handle_image_command(input, pending_images) {
        return result;
    }
    if let Some(result) = handle_tools_command(input, tools_state) {
        return result;
    }

    match input.to_lowercase().as_str() {
        "quit" | "exit" | "q" => {
            println!();
            ui::hint("Goodbye!");
            SpecialCommandResult::Quit
        }
        "help" | "?" => {
            println!();
            ui::hint("Commands:");
            println!("    {}  Exit REPL", ui::dim("quit, exit, q"));
            println!("    {}       Show this help", ui::dim("help, ?"));
            println!(
                "    {}   Attach image to next message",
                ui::dim("/image <path>")
            );
            println!(
                "    {} List / toggle built-in tools",
                ui::dim("/tools [on|off]")
            );
            if conversation_context.is_some() {
                println!("    {}      Show conversation history", ui::dim("history"));
                println!("    {}        Clear conversation history", ui::dim("clear"));
            }
            println!("    {}       Run inference", ui::dim("<text>"));
            SpecialCommandResult::Continue
        }
        "history" if conversation_context.is_some() => {
            let ctx = conversation_context.as_ref().unwrap();
            let history = ctx.history();
            if history.is_empty() {
                ui::hint("No conversation history yet.");
            } else {
                println!();
                ui::hint(&format!(
                    "Conversation history ({} messages):",
                    history.len()
                ));
                println!("  {}", "─".repeat(50).truecolor(60, 60, 70));
                for (i, envelope) in history.iter().enumerate() {
                    let role = envelope.role().map(|r| r.as_str()).unwrap_or("unknown");
                    let text = match &envelope.kind {
                        EnvelopeKind::Text(t) => t.as_str(),
                        _ => "[non-text]",
                    };
                    let display_text = if verbose == 0 && text.len() > 100 {
                        format!("{}...", &text[..100])
                    } else {
                        text.to_string()
                    };
                    let role_colored = match role {
                        "user" => role.to_uppercase().truecolor(120, 180, 255),
                        "assistant" => role.to_uppercase().truecolor(180, 140, 255),
                        "system" => role.to_uppercase().truecolor(120, 120, 130),
                        _ => role.to_uppercase().normal(),
                    };
                    println!("  [{}] {} {}", i + 1, role_colored, display_text);
                }
                println!("  {}", "─".repeat(50).truecolor(60, 60, 70));
            }
            SpecialCommandResult::Continue
        }
        "clear" if conversation_context.is_some() => {
            let ctx = conversation_context.as_mut().unwrap();
            ctx.clear();
            ui::ok("Conversation history cleared");
            SpecialCommandResult::Continue
        }
        "" => SpecialCommandResult::Continue,
        _ => SpecialCommandResult::NotSpecial,
    }
}

/// `/tools` — list the built-in tools; `/tools on|off` toggles them for the
/// session (on requires the model to declare support).
fn handle_tools_command(input: &str, tools_state: &mut ToolsState) -> Option<SpecialCommandResult> {
    let trimmed = input.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    if !parts
        .next()
        .unwrap_or_default()
        .eq_ignore_ascii_case("/tools")
    {
        return None;
    }

    match parts.next().map(str::trim).unwrap_or("") {
        "" => {
            println!();
            if tools_state.active() {
                ui::hint("Tool calling: on");
            } else if tools_state.available {
                ui::hint("Tool calling: off — enable with '/tools on'");
            } else {
                ui::hint(
                    "Tool calling: unavailable — the model's metadata does not \
                     declare `tool_calling: true`",
                );
            }
            println!(
                "    {}    Search the web via {}",
                ui::dim("web_search"),
                tools_state.toolbox.provider.label()
            );
            println!(
                "    {}     Fetch a public http(s) URL",
                ui::dim("fetch_url")
            );
            println!("    {}  Local date and time", ui::dim("current_time"));
            for tool in &tools_state.toolbox.user_tools {
                println!("    {}  {} (user)", ui::dim(&tool.name), tool.description);
            }
            ui::hint("Search provider: set XYBRID_SEARCH_PROVIDER=wikipedia|tavily|brave");
            ui::hint("Add your own tools with --tools-file <file>");
        }
        "on" => {
            if tools_state.available {
                tools_state.enabled = true;
                ui::ok("Tool calling enabled");
            } else {
                ui::err(
                    "This model does not declare tool support \
                     (metadata `tool_calling: true`)",
                );
            }
        }
        "off" => {
            tools_state.enabled = false;
            ui::ok("Tool calling disabled for this session");
        }
        other => {
            ui::err(&format!("Unknown option '{other}'. Usage: /tools [on|off]"));
        }
    }
    Some(SpecialCommandResult::Continue)
}

fn handle_image_command(
    input: &str,
    pending_images: &mut ReplPendingImages,
) -> Option<SpecialCommandResult> {
    let trimmed = input.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    if !command.eq_ignore_ascii_case("/image") {
        return None;
    }

    let Some(path) = parts.next().map(str::trim).filter(|path| !path.is_empty()) else {
        ui::err("Usage: /image <path>");
        return Some(SpecialCommandResult::Continue);
    };

    {
        let path = PathBuf::from(path);
        if !path.exists() {
            ui::err(&format!("Image not found: {}", path.display()));
            return Some(SpecialCommandResult::Continue);
        }

        pending_images.push(path);
        ui::ok(&format!(
            "Image attached to next message ({} pending)",
            pending_images.len()
        ));
    }

    Some(SpecialCommandResult::Continue)
}

fn build_repl_input(
    input_line: &str,
    voice: Option<&str>,
    conversation_context_enabled: bool,
    pending_images: &mut ReplPendingImages,
    max_tokens: Option<usize>,
) -> Result<Envelope> {
    if !pending_images.is_empty() {
        if voice.is_some() {
            return Err(anyhow::anyhow!(
                "--voice cannot be combined with /image attachments"
            ));
        }
        let mut input = build_repl_multimodal_input(input_line, pending_images)?;
        if let Some(max_tokens) = max_tokens {
            input
                .metadata
                .insert("max_tokens".to_string(), max_tokens.to_string());
        }
        return Ok(input);
    }

    let mut input = Envelope::new(EnvelopeKind::Text(input_line.to_string()));
    if conversation_context_enabled {
        input = input.with_role(MessageRole::User);
    }
    if let Some(voice_id) = voice {
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

fn build_repl_multimodal_input(
    input_line: &str,
    pending_images: &mut ReplPendingImages,
) -> Result<Envelope> {
    let images = read_repl_images(&pending_images.paths)?;
    let input = Envelope::user_message(input_line, images)
        .context("Failed to build multimodal REPL input")?;
    pending_images.clear();
    Ok(input)
}

fn read_repl_images(image_paths: &[PathBuf]) -> Result<Vec<Envelope>> {
    let mut images = Vec::with_capacity(image_paths.len());
    for image_path in image_paths {
        let image_bytes = fs::read(image_path)
            .with_context(|| format!("Failed to read image file: {}", image_path.display()))?;
        let format = image_format_hint(image_path)?;
        images.push(
            Envelope::image(image_bytes, format)
                .with_context(|| format!("Invalid image input: {}", image_path.display()))?,
        );
    }
    Ok(images)
}

fn image_format_hint(path: &Path) -> Result<&str> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| anyhow::anyhow!("Image file has no extension: {}", path.display()))
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn try_streaming_execution(
    stages: &[StageDescriptor],
    input: &Envelope,
    conversation_context: &mut Option<ConversationContext>,
    loaded_model: &Option<xybrid_sdk::model::XybridModel>,
    max_tokens: Option<usize>,
    start: std::time::Instant,
    verbose: u8,
) -> bool {
    // A pre-loaded model (including the speculative cloud-backed handle, which
    // has no on-disk bundle) drives the turn directly. Resolve `bundle_path`
    // only in the fall-back branch below, so a bundle-less stage never panics.
    if let Some(model) = loaded_model.as_ref() {
        if model.supports_token_streaming() {
            return execute_streaming(
                model,
                input,
                conversation_context,
                max_tokens,
                start,
                verbose,
            );
        } else {
            ui::warning("Streaming only supported for GGUF models, falling back to batch mode");
            return false;
        }
    }

    let bundle_path = match stages[0].bundle_path.as_ref() {
        Some(path) => PathBuf::from(path),
        None => {
            ui::warning("No local bundle for stage, falling back to batch mode");
            return false;
        }
    };

    // Fall back to loading the model if not pre-loaded
    let model_result = if bundle_path.extension().is_some_and(|ext| ext == "xyb") {
        ModelLoader::from_bundle(&bundle_path).and_then(|loader| loader.load())
    } else {
        ModelLoader::from_directory(&bundle_path).and_then(|loader| loader.load())
    };

    match model_result {
        Ok(model) => {
            if model.supports_token_streaming() {
                execute_streaming(
                    &model,
                    input,
                    conversation_context,
                    max_tokens,
                    start,
                    verbose,
                )
            } else {
                ui::warning("Streaming only supported for GGUF models, falling back to batch mode");
                false
            }
        }
        Err(e) => {
            ui::warning(&format!(
                "Failed to load model: {}, falling back to batch mode",
                e
            ));
            false
        }
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn execute_streaming(
    model: &xybrid_sdk::model::XybridModel,
    input: &Envelope,
    conversation_context: &mut Option<ConversationContext>,
    max_tokens: Option<usize>,
    start: std::time::Instant,
    verbose: u8,
) -> bool {
    use std::io;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    let accumulated_text = Arc::new(Mutex::new(String::new()));
    let text_clone = Arc::clone(&accumulated_text);
    let token_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let token_count_clone = Arc::clone(&token_count);
    let first_token_time = Arc::new(Mutex::new(None::<std::time::Instant>));
    let first_token_clone = Arc::clone(&first_token_time);
    let config = max_tokens.map(|max_tokens| {
        model
            .default_generation_config()
            .with_max_tokens(max_tokens)
    });

    let streaming_result = if let Some(ref ctx) = conversation_context {
        model.run_streaming_with_context(input, ctx, config.as_ref(), |token| {
            print!("{}", token.token);
            io::stdout().flush()?;
            let count = token_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count == 0 {
                if let Ok(mut ft) = first_token_clone.lock() {
                    *ft = Some(std::time::Instant::now());
                }
            }
            if let Ok(mut text) = text_clone.lock() {
                text.push_str(&token.token);
            }
            Ok(())
        })
    } else {
        model.run_streaming(input, config.as_ref(), |token| {
            print!("{}", token.token);
            io::stdout().flush()?;
            let count = token_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count == 0 {
                if let Ok(mut ft) = first_token_clone.lock() {
                    *ft = Some(std::time::Instant::now());
                }
            }
            if let Ok(mut text) = text_clone.lock() {
                text.push_str(&token.token);
            }
            Ok(())
        })
    };

    match streaming_result {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!();
            maybe_warn_thinking_budget(result.envelope());

            if let Some(ref mut ctx) = conversation_context {
                // Push the user turn only after the run: the streaming
                // context path appends the input itself, so pushing before
                // dispatch doubles it in the prompt.
                ctx.push(input.clone());
                if let Ok(text) = accumulated_text.lock() {
                    let assistant_response = Envelope::new(EnvelopeKind::Text(text.clone()))
                        .with_role(MessageRole::Assistant);
                    ctx.push(assistant_response);
                    if verbose > 1 {
                        ui::hint(&format!(
                            "Added assistant response to context (total: {} messages)",
                            ctx.history().len()
                        ));
                    }
                }
            }

            let tokens = token_count.load(std::sync::atomic::Ordering::Relaxed);
            let ttft = first_token_time
                .lock()
                .ok()
                .and_then(|ft| ft.map(|t| t.duration_since(start)));

            let decode_tok_s = ttft.and_then(|ttft_dur| {
                let decode_time = elapsed.saturating_sub(ttft_dur).as_secs_f64();
                if tokens >= 2 && decode_time > 0.001 {
                    Some((tokens - 1) as f64 / decode_time)
                } else {
                    None
                }
            });

            if let Some(tok_s) = decode_tok_s {
                let ttft_ms = ttft.map(|d| d.as_millis()).unwrap_or(0);
                println!();
                ui::hint(&format!(
                    "{} tokens in {:.2}s ({:.1} tok/s, {}ms to first token)",
                    tokens,
                    elapsed.as_secs_f64(),
                    tok_s,
                    ttft_ms
                ));
            } else {
                println!();
                ui::hint(&format!(
                    "{} tokens in {:.2}s",
                    tokens,
                    elapsed.as_secs_f64()
                ));
            }
            true
        }
        Err(e) => {
            ui::err(&format!("{}", e));
            true
        }
    }
}

fn execute_batch(
    orchestrator: &mut Orchestrator,
    stages: &[StageDescriptor],
    input: &Envelope,
    metrics: &xybrid_core::context::DeviceMetrics,
    availability_fn: &dyn Fn(&str) -> LocalAvailability,
    conversation_context: &mut Option<ConversationContext>,
    start: std::time::Instant,
    show_reasoning: bool,
    verbose: u8,
) {
    match orchestrator.execute_pipeline(stages, input, metrics, availability_fn) {
        Ok(results) => {
            let elapsed = start.elapsed();
            println!();

            // Record the user turn only now that the run succeeded (and
            // never before dispatch — context-aware paths add the input
            // themselves, so an early push doubles it in the prompt).
            if let Some(ref mut ctx) = conversation_context {
                ctx.push(input.clone());
            }

            for result in &results {
                match &result.output.kind {
                    EnvelopeKind::Text(text) => {
                        print_reasoning(
                            show_reasoning,
                            result
                                .output
                                .metadata
                                .get("reasoning_content")
                                .map(String::as_str),
                        );
                        println!("  {}", text);
                        maybe_warn_thinking_budget(&result.output);

                        if let Some(ref mut ctx) = conversation_context {
                            let assistant_response =
                                Envelope::new(EnvelopeKind::Text(text.clone()))
                                    .with_role(MessageRole::Assistant);
                            ctx.push(assistant_response);
                            if verbose > 1 {
                                ui::hint(&format!(
                                    "Added assistant response to context (total: {} messages)",
                                    ctx.history().len()
                                ));
                            }
                        }
                    }
                    EnvelopeKind::Audio(data) => {
                        ui::ok(&format!("Audio output: {} bytes", data.len()));
                        ui::hint("Use the 'run' command with --output to save audio");
                    }
                    EnvelopeKind::Embedding(vec) => {
                        ui::ok(&format!("Embedding: {} dimensions", vec.len()));
                    }
                    EnvelopeKind::Image { .. } => {
                        ui::ok(&format!(
                            "Image output: {} bytes",
                            result.output.payload_size()
                        ));
                    }
                    EnvelopeKind::MultiPart(parts) => {
                        ui::ok(&format!("Multi-part output: {} parts", parts.len()));
                    }
                }
            }

            println!();
            ui::hint(&format!("Inference time: {:.2}s", elapsed.as_secs_f32()));
        }
        Err(e) => {
            ui::err(&format!("{}", e));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tools_state(available: bool) -> ToolsState {
        ToolsState {
            available,
            enabled: available,
            toolbox: tools::ToolBox {
                provider: tools::SearchProvider::Wikipedia,
                user_tools: Vec::new(),
            },
        }
    }

    #[test]
    fn image_command_is_handled() {
        let mut conversation_context = None;
        let mut pending_images = ReplPendingImages::default();
        let mut tools_state = test_tools_state(false);

        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("fixture.png");
        fs::write(&image_path, png_image(2, 3)).unwrap();

        let result = handle_special_command(
            &format!("/image {}", image_path.display()),
            &mut conversation_context,
            &mut pending_images,
            &mut tools_state,
            0,
        );

        assert!(matches!(result, SpecialCommandResult::Continue));
        assert_eq!(pending_images.len(), 1);
        assert_eq!(pending_images.paths[0], image_path);
    }

    #[test]
    fn tools_command_toggles_session_state() {
        // Toggle works when the model declares support…
        let mut tools_state = test_tools_state(true);
        assert!(tools_state.active());

        assert!(matches!(
            handle_tools_command("/tools off", &mut tools_state),
            Some(SpecialCommandResult::Continue)
        ));
        assert!(!tools_state.active());

        handle_tools_command("/tools on", &mut tools_state);
        assert!(tools_state.active());

        // …and `on` is refused when it does not.
        let mut unavailable = test_tools_state(false);
        handle_tools_command("/tools on", &mut unavailable);
        assert!(!unavailable.active());

        // Non-/tools input passes through untouched.
        assert!(handle_tools_command("hello", &mut tools_state).is_none());
    }

    #[test]
    fn repl_multimodal_input_consumes_pending_image() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("fixture.png");
        fs::write(&image_path, png_image(2, 3)).unwrap();
        let mut pending_images = ReplPendingImages::default();
        pending_images.push(image_path);

        let input =
            build_repl_input("describe this", None, true, &mut pending_images, None).unwrap();
        let parts = input.as_multipart().expect("REPL input is multipart");

        assert!(pending_images.is_empty());
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
        assert_eq!(input.role(), Some(MessageRole::User));
    }

    #[test]
    fn repl_multimodal_input_rejects_corrupt_image_with_redacted_error() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("corrupt.jpeg");
        fs::write(&image_path, [42_u8, 42, 42, 42]).unwrap();
        let mut pending_images = ReplPendingImages::default();
        pending_images.push(image_path);

        let err =
            build_repl_input("describe this", None, true, &mut pending_images, None).unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("Invalid image input"));
        assert!(message.contains("invalid or corrupt jpeg image bytes"));
        assert!(!message.contains("[42"));
        assert!(!message.contains("42, 42"));
    }

    #[test]
    fn direct_model_network_target_skips_registry_cache_lookup() {
        for target in [ExecutionTarget::Cloud, ExecutionTarget::Server] {
            let client = RegistryClient::with_url("http://127.0.0.1:9").unwrap();

            let stages = load_stages(
                &client,
                &None,
                &Some("test-model".to_string()),
                Some(&target),
            )
            .unwrap();

            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].target.as_ref(), Some(&target));
            assert!(stages[0].bundle_path.is_none());
        }
    }

    #[test]
    fn invalid_yaml_target_is_ignored_without_hard_failure() {
        let config = PipelineConfig::from_yaml(
            r#"
name: test
stages:
  - id: llm
    model: test-model
    target: clod
    provider: openai
"#,
        )
        .unwrap();
        let client = RegistryClient::with_url("http://127.0.0.1:9").unwrap();

        let stages = load_stages(&client, &Some(config), &None, None).unwrap();

        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].target, None);
        assert!(stages[0].bundle_path.is_none());
    }

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
}

//! `xybrid repl` command handler - interactive REPL mode.

#![allow(clippy::too_many_arguments)]

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use xybrid_core::context::{DeviceMetrics, StageDescriptor};
use xybrid_core::conversation::ConversationContext;
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
use xybrid_core::orchestrator::routing_engine::LocalAvailability;
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::pipeline_config::PipelineConfig;
use xybrid_sdk::model::ModelLoader;
use xybrid_sdk::registry_client::RegistryClient;

use colored::Colorize;

use crate::ui;

/// Interactive REPL mode - keeps models loaded for fast repeated inference.
pub(crate) fn handle_repl_command(
    config: Option<PathBuf>,
    model: Option<String>,
    model_file: Option<PathBuf>,
    huggingface: Option<String>,
    voice: Option<String>,
    _target: Option<String>,
    stream: bool,
    system_prompt: Option<String>,
    verbose: u8,
) -> Result<()> {
    use std::io::{self, Write};

    ui::brand_with_version(env!("CARGO_PKG_VERSION"));
    println!();
    ui::hint("Models loaded once and kept warm for fast inference");
    ui::hint("Type 'quit' or 'exit' to exit. Type 'help' for commands.");

    print_streaming_status(stream);
    println!();

    // --huggingface: load from HuggingFace repo
    let stages = if let Some(ref repo) = huggingface {
        let sp = ui::spinner(&format!("Loading from HuggingFace: {}...", repo));
        let loader = ModelLoader::from_huggingface_parsed(repo);
        let _model = loader.load().context(format!(
            "Failed to load model from HuggingFace repo '{}'",
            repo
        ))?;

        let sanitized = repo.replace('/', "--");
        let cache_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?
            .join(".xybrid")
            .join("cache")
            .join("hf")
            .join(&sanitized);

        sp.finish_and_clear();
        ui::ok("Model loaded from HuggingFace");

        let mut stage = StageDescriptor::new(_model.model_id());
        stage.bundle_path = Some(cache_dir.to_string_lossy().to_string());
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

        load_stages(&client, &pipeline_config, &model_id)?
    };

    let mut conversation_context: Option<ConversationContext> = None;
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    let mut loaded_model: Option<xybrid_sdk::model::XybridModel> = None;

    if stages.len() == 1 && stages[0].bundle_path.is_some() {
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
            #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
            {
                loaded_model = Some(model);
            }
        }
    }

    let metrics = DeviceMetrics::default();

    let stage_bundle_paths: std::collections::HashMap<String, bool> = stages
        .iter()
        .map(|s| (s.name.clone(), s.bundle_path.is_some()))
        .collect();
    let availability_fn = move |stage: &str| -> LocalAvailability {
        LocalAvailability::new(stage_bundle_paths.get(stage).copied().unwrap_or(false))
    };

    let mut orchestrator = Orchestrator::new();
    xybrid_sdk::bridge_orchestrator_events(&orchestrator);

    warmup_models(&mut orchestrator, &stages, &metrics, &availability_fn);

    println!();
    ui::hint("Enter text and press Enter to run inference");
    println!("  {}", "─".repeat(50).truecolor(60, 60, 70));

    let stdin = io::stdin();
    loop {
        print!("\n  {} ", "❯".truecolor(120, 180, 255).bold());
        io::stdout().flush()?;

        let mut input_line = String::new();
        if stdin.read_line(&mut input_line)? == 0 {
            break;
        }

        let input_line = input_line.trim();

        let handled = handle_special_command(input_line, &mut conversation_context, verbose);

        match handled {
            SpecialCommandResult::Quit => break,
            SpecialCommandResult::Continue => continue,
            SpecialCommandResult::NotSpecial => {}
        }

        let mut input = Envelope::new(EnvelopeKind::Text(input_line.to_string()));
        if conversation_context.is_some() {
            input = input.with_role(MessageRole::User);
        }
        if let Some(ref voice_id) = voice {
            input
                .metadata
                .insert("voice_id".to_string(), voice_id.clone());
        }

        if let Some(ref mut ctx) = conversation_context {
            ctx.push(input.clone());
            if verbose > 1 {
                ui::hint(&format!(
                    "Added user message to context (total: {} messages)",
                    ctx.history().len()
                ));
            }
        }

        let start = std::time::Instant::now();

        // Try streaming execution
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        let use_streaming = {
            let can_stream = stream && stages.len() == 1 && stages[0].bundle_path.is_some();
            if stream && !can_stream {
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
            verbose,
        );
    }

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
) -> Result<Vec<StageDescriptor>> {
    let mut stages = Vec::new();

    if let Some(ref config) = pipeline_config {
        let name = config.name.as_deref().unwrap_or("unnamed");
        ui::kv("Pipeline", name);
        for stage_config in &config.stages {
            let model_id = stage_config.model_id();
            let mut desc = StageDescriptor::new(&model_id);

            if !stage_config.is_cloud_stage() {
                ensure_model_cached(&mut desc, &model_id, client)?;
            }
            stages.push(desc);
        }
    } else if let Some(ref model_id) = model_id {
        ui::kv("Model", model_id);
        let mut desc = StageDescriptor::new(model_id);
        ensure_model_cached(&mut desc, model_id, client)?;
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

fn warmup_models(
    orchestrator: &mut Orchestrator,
    stages: &[StageDescriptor],
    metrics: &xybrid_core::context::DeviceMetrics,
    availability_fn: &dyn Fn(&str) -> LocalAvailability,
) {
    let sp = ui::spinner("Warming up models...");
    let warmup_input = Envelope {
        kind: EnvelopeKind::Text("Hi".to_string()),
        metadata: std::collections::HashMap::new(),
    };
    match orchestrator.execute_pipeline(stages, &warmup_input, metrics, availability_fn) {
        Ok(_) => {
            sp.finish_and_clear();
            ui::ok("Models loaded and warm. Ready for input!");
        }
        Err(e) => {
            sp.finish_and_clear();
            ui::warning(&format!("Warmup failed ({}), first query may be slow", e));
        }
    }
}

enum SpecialCommandResult {
    Quit,
    Continue,
    NotSpecial,
}

fn handle_special_command(
    input: &str,
    conversation_context: &mut Option<ConversationContext>,
    verbose: u8,
) -> SpecialCommandResult {
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

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn try_streaming_execution(
    stages: &[StageDescriptor],
    input: &Envelope,
    conversation_context: &mut Option<ConversationContext>,
    loaded_model: &Option<xybrid_sdk::model::XybridModel>,
    start: std::time::Instant,
    verbose: u8,
) -> bool {
    let bundle_path_str = stages[0].bundle_path.as_ref().unwrap();
    let bundle_path = PathBuf::from(bundle_path_str);

    let model_for_streaming = loaded_model.as_ref();

    if let Some(model) = model_for_streaming {
        if model.supports_token_streaming() {
            return execute_streaming(model, input, conversation_context, start, verbose);
        } else {
            ui::warning("Streaming only supported for GGUF models, falling back to batch mode");
            return false;
        }
    }

    // Fall back to loading the model if not pre-loaded
    let model_result = if bundle_path.extension().is_some_and(|ext| ext == "xyb") {
        ModelLoader::from_bundle(&bundle_path).and_then(|loader| loader.load())
    } else {
        ModelLoader::from_directory(&bundle_path).and_then(|loader| loader.load())
    };

    match model_result {
        Ok(model) => {
            if model.supports_token_streaming() {
                execute_streaming(&model, input, conversation_context, start, verbose)
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

    let streaming_result = if let Some(ref ctx) = conversation_context {
        model.run_streaming_with_context(input, ctx, None, |token| {
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
        model.run_streaming(input, None, |token| {
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
        Ok(_result) => {
            let elapsed = start.elapsed();
            println!();

            if let Some(ref mut ctx) = conversation_context {
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
    verbose: u8,
) {
    match orchestrator.execute_pipeline(stages, input, metrics, availability_fn) {
        Ok(results) => {
            let elapsed = start.elapsed();
            println!();

            for result in &results {
                match &result.output.kind {
                    EnvelopeKind::Text(text) => {
                        println!("  {}", text);

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

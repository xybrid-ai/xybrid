//! `xybrid repl` command handler - interactive REPL mode.

#![allow(clippy::too_many_arguments)]

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use xybrid_core::context::StageDescriptor;
use xybrid_core::conversation::ConversationContext;
use xybrid_core::device_adapter::{DeviceAdapter, LocalDeviceAdapter};
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
use xybrid_core::orchestrator::routing_engine::LocalAvailability;
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::pipeline_config::PipelineConfig;
use xybrid_sdk::model::ModelLoader;
use xybrid_sdk::registry_client::RegistryClient;

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

    println!("🚀 Xybrid REPL Mode");
    println!("{}", "=".repeat(60));
    println!("Models will be loaded once and kept warm for fast inference.");
    println!("Type 'quit' or 'exit' to exit. Type 'help' for commands.");

    print_streaming_status(stream);
    println!();

    // --huggingface: load from HuggingFace repo
    let stages = if let Some(ref repo) = huggingface {
        println!("🤗 Loading from HuggingFace: {}", repo);
        let loader = ModelLoader::from_huggingface(repo);
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

        println!("✅ Model loaded from HuggingFace");

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

        println!("📦 Loading local GGUF: {}", gguf_path.display());
        if verbose > 0 {
            println!("   Model ID: {}", metadata.model_id);
            if let xybrid_core::execution::ExecutionTemplate::Gguf { context_length, .. } =
                &metadata.execution_template
            {
                println!("   Context length: {}", context_length);
            }
            if let Some(arch) = metadata.metadata.get("architecture") {
                println!("   Architecture: {}", arch);
            }
        }

        // Write metadata to parent dir so ModelLoader can find it
        let metadata_path = parent_dir.join("model_metadata.json");
        let needs_write = !metadata_path.exists();
        if needs_write {
            let json = serde_json::to_string_pretty(&metadata)?;
            fs::write(&metadata_path, &json)?;
            if verbose > 0 {
                println!("   Generated model_metadata.json");
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
                println!("💬 LLM detected - conversation context enabled");
                let mut ctx = ConversationContext::new();
                if let Some(ref prompt) = system_prompt {
                    println!("📋 System prompt: {}", prompt);
                    ctx = ctx.with_system(
                        Envelope::new(EnvelopeKind::Text(prompt.clone()))
                            .with_role(MessageRole::System),
                    );
                }
                conversation_context = Some(ctx);
                if verbose > 0 {
                    println!("   (Use 'history' to view conversation, 'clear' to reset)");
                }
            }
            #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
            {
                loaded_model = Some(model);
            }
        }
    }

    let device_adapter = LocalDeviceAdapter::new();
    let metrics = device_adapter.collect_metrics();

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

    println!("\nEnter text and press Enter to run inference.");
    println!("{}", "=".repeat(60));

    let stdin = io::stdin();
    loop {
        print!("\n> ");
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
                println!(
                    "📝 Added user message to context (total: {} messages)",
                    ctx.history().len()
                );
            }
        }

        let start = std::time::Instant::now();

        // Try streaming execution
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        let use_streaming = {
            let can_stream = stream && stages.len() == 1 && stages[0].bundle_path.is_some();
            if stream && !can_stream {
                eprintln!("⚠️  Streaming conditions not met:");
                eprintln!("   - stages.len() = {} (need 1)", stages.len());
                eprintln!(
                    "   - bundle_path = {:?}",
                    stages.first().map(|s| &s.bundle_path)
                );
            }
            can_stream
        };

        #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
        let use_streaming = {
            if stream {
                eprintln!("⚠️  Streaming requested but LLM features not enabled.");
                eprintln!("   Build with: --features llm-llamacpp (or llm-mistral)");
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
        println!("📡 Token streaming: ENABLED");
    }
    #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
    if stream {
        println!("⚠️  Token streaming: NOT AVAILABLE (LLM features not compiled)");
    }
}

fn load_stages(
    client: &RegistryClient,
    pipeline_config: &Option<PipelineConfig>,
    model_id: &Option<String>,
) -> Result<Vec<StageDescriptor>> {
    let mut stages = Vec::new();

    if let Some(ref config) = pipeline_config {
        println!(
            "📋 Pipeline: {}",
            config.name.as_deref().unwrap_or("unnamed")
        );
        for stage_config in &config.stages {
            let model_id = stage_config.model_id();
            let mut desc = StageDescriptor::new(&model_id);

            if !stage_config.is_cloud_stage() {
                ensure_model_cached(&mut desc, &model_id, client)?;
            }
            stages.push(desc);
        }
    } else if let Some(ref model_id) = model_id {
        println!("📦 Model: {}", model_id);
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
    if !client.is_cached(model_id, None).unwrap_or(false) {
        println!("📥 Downloading model: {}", model_id);
        use indicatif::{ProgressBar, ProgressStyle};
        let resolved = client.resolve(model_id, None)?;
        let pb = ProgressBar::new(resolved.size_bytes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes}")
                .unwrap(),
        );
        pb.set_message(model_id.to_string());
        let model_dir = client.fetch_extracted(model_id, None, |p| {
            pb.set_position((p * resolved.size_bytes as f32) as u64);
        })?;
        pb.finish_with_message(format!("{} ✓", model_id));
        desc.bundle_path = Some(model_dir.to_string_lossy().to_string());
    } else {
        let cache = xybrid_sdk::cache::CacheManager::new()?;
        let xyb_path = client.get_cache_path(&client.resolve(model_id, None)?);
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
    println!("\n⏳ Warming up models (this may take a moment)...");
    let warmup_input = Envelope {
        kind: EnvelopeKind::Text("Hi".to_string()),
        metadata: std::collections::HashMap::new(),
    };
    match orchestrator.execute_pipeline(stages, &warmup_input, metrics, availability_fn) {
        Ok(_) => println!("🔥 Models loaded and warm. Ready for input!"),
        Err(e) => println!("⚠️  Warmup failed ({}), first query may be slow", e),
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
            println!("👋 Goodbye!");
            SpecialCommandResult::Quit
        }
        "help" | "?" => {
            println!("Commands:");
            println!("  quit, exit, q  - Exit REPL");
            println!("  help, ?        - Show this help");
            if conversation_context.is_some() {
                println!("  history        - Show conversation history (LLM only)");
                println!("  clear          - Clear conversation history (LLM only)");
            }
            println!("  <text>         - Run inference with the given text");
            SpecialCommandResult::Continue
        }
        "history" if conversation_context.is_some() => {
            let ctx = conversation_context.as_ref().unwrap();
            let history = ctx.history();
            if history.is_empty() {
                println!("📜 No conversation history yet.");
            } else {
                println!("📜 Conversation history ({} messages):", history.len());
                println!("{}", "-".repeat(50));
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
                    println!("[{}] {}: {}", i + 1, role.to_uppercase(), display_text);
                }
                println!("{}", "-".repeat(50));
            }
            SpecialCommandResult::Continue
        }
        "clear" if conversation_context.is_some() => {
            let ctx = conversation_context.as_mut().unwrap();
            ctx.clear();
            println!("🗑️  Conversation history cleared.");
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
            eprintln!("⚠️  Streaming only supported for GGUF models, falling back to batch mode");
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
                eprintln!(
                    "⚠️  Streaming only supported for GGUF models, falling back to batch mode"
                );
                false
            }
        }
        Err(e) => {
            eprintln!(
                "⚠️  Failed to load model: {}, falling back to batch mode",
                e
            );
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

    let streaming_result = if let Some(ref ctx) = conversation_context {
        model.run_streaming_with_context(input, ctx, None, |token| {
            print!("{}", token.token);
            io::stdout().flush()?;
            if let Ok(mut text) = text_clone.lock() {
                text.push_str(&token.token);
            }
            Ok(())
        })
    } else {
        model.run_streaming(input, None, |token| {
            print!("{}", token.token);
            io::stdout().flush()?;
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

            if let Some(ref mut ctx) = conversation_context {
                if let Ok(text) = accumulated_text.lock() {
                    let assistant_response = Envelope::new(EnvelopeKind::Text(text.clone()))
                        .with_role(MessageRole::Assistant);
                    ctx.push(assistant_response);
                    if verbose > 1 {
                        println!(
                            "📝 Added assistant response to context (total: {} messages)",
                            ctx.history().len()
                        );
                    }
                }
            }

            println!(
                "\n⏱️  Inference time: {:.2}s ({}ms latency)",
                elapsed.as_secs_f32(),
                result.latency_ms()
            );
            true
        }
        Err(e) => {
            eprintln!("\n❌ Error: {}", e);
            true // Still handled, even on error
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
                        println!("{}", text);

                        if let Some(ref mut ctx) = conversation_context {
                            let assistant_response =
                                Envelope::new(EnvelopeKind::Text(text.clone()))
                                    .with_role(MessageRole::Assistant);
                            ctx.push(assistant_response);
                            if verbose > 1 {
                                println!(
                                    "📝 Added assistant response to context (total: {} messages)",
                                    ctx.history().len()
                                );
                            }
                        }
                    }
                    EnvelopeKind::Audio(data) => {
                        println!("🔊 Audio output: {} bytes", data.len());
                        println!("   Use the 'run' command with --output to save audio.");
                    }
                    EnvelopeKind::Embedding(vec) => {
                        println!("📊 Embedding: {} dimensions", vec.len());
                    }
                }
            }

            println!("\n⏱️  Inference time: {:.2}s", elapsed.as_secs_f32());
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }
}

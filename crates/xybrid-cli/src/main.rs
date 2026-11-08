//! Xybrid CLI - Command-line interface for running hybrid cloud-edge AI inference pipelines.
//!
//! This binary provides subcommands for managing and executing ML inference pipelines.
//!
//! ## Module Organization
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`commands`] | Command handlers organized by subcommand |
//! | [`tracing_viz`] | Trace visualization utilities |
//!
//! ## Commands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `run` | Execute a pipeline from config or bundle |
//! | `repl` | Interactive REPL mode |
//! | `models` | Manage models from the registry |
//! | `cache` | Manage the local model cache |
//! | `prepare` | Parse and validate a pipeline configuration |
//! | `plan` | Show execution plan for a pipeline |
//! | `fetch` | Pre-download models from the registry |
//! | `trace` | View and analyze telemetry sessions |
//! | `pack` | Create a model bundle |
//! | `bundle` | Fetch and bundle a registry model |

mod commands;
#[allow(dead_code)]
mod tracing_viz;
pub mod ui;

/// Serializes tests that install the process-wide default observability session.
#[cfg(test)]
pub(crate) static OBSERVABILITY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

use commands::{CacheCommand, ModelsCommand, TelemetryCommand};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;
use xybrid_core::ir::Choice;
use xybrid_sdk::observability::{ObservabilityConfig, OtlpConfig, XYBRID_TRACES_ENDPOINT};
use xybrid_sdk::SdkError;

/// Xybrid CLI - Hybrid Cloud-Edge AI Inference Pipeline Runner
#[derive(Parser)]
#[command(name = "xybrid")]
#[command(about = "Xybrid CLI - Run hybrid cloud-edge AI inference pipelines", long_about = None)]
struct Cli {
    /// Platform API key for cloud access (does not enable observability by itself)
    #[arg(long, global = true, env = "XYBRID_API_KEY")]
    api_key: Option<String>,

    /// Cloud gateway endpoint (default: https://api.xybrid.dev)
    #[arg(
        long,
        global = true,
        env = "XYBRID_PLATFORM_URL",
        default_value = "https://api.xybrid.dev"
    )]
    platform_url: String,

    /// Export traces to Xybrid; alternatively configure OTEL_EXPORTER_OTLP_* variables
    #[arg(long, global = true, requires = "api_key")]
    observability: bool,

    /// Full OTLP trace URL for --observability (default: https://ingest.xybrid.dev/v1/traces)
    #[arg(long, global = true, requires = "observability")]
    traces_endpoint: Option<String>,

    /// Device ID for telemetry attribution
    #[arg(long, global = true, env = "XYBRID_DEVICE_ID")]
    device_id: Option<String>,

    /// Increase verbosity (-v for verbose, -vv for very verbose with library logs)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Quiet mode - suppress most output, show errors only
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate model_metadata.json by inspecting model files in a directory
    Init {
        /// Path to the model directory to inspect
        #[arg(value_name = "DIRECTORY")]
        directory: String,

        /// Overwrite existing model_metadata.json
        #[arg(long)]
        force: bool,

        /// Non-interactive mode (accept defaults for ambiguous cases)
        #[arg(long)]
        yes: bool,

        /// Override detected task (e.g., "text-classification", "text-to-speech")
        #[arg(long, value_name = "TASK")]
        task: Option<String>,

        /// Output structured JSON instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Override auto-generated model ID
        #[arg(long, value_name = "ID")]
        model_id: Option<String>,
    },
    /// Manage models from the registry
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Parse and validate a pipeline configuration
    Prepare {
        /// Path to the pipeline configuration file (YAML)
        #[arg(value_name = "FILE")]
        config: PathBuf,
    },
    /// Show execution plan for a pipeline (models, targets, download status)
    Plan {
        /// Path to the pipeline configuration file (YAML)
        #[arg(value_name = "FILE")]
        config: PathBuf,
    },
    /// Pre-download models from the registry or HuggingFace
    Fetch {
        /// Path to pipeline configuration file (downloads all models)
        #[arg(value_name = "FILE", conflicts_with_all = ["model", "huggingface"])]
        config: Option<PathBuf>,

        /// Model ID to fetch from registry (e.g., "kokoro-82m")
        #[arg(short, long, value_name = "ID", conflicts_with = "huggingface")]
        model: Option<String>,

        /// HuggingFace repo to fetch (e.g., "LiquidAI/LFM2.5-350M-GGUF:Q4_K_M")
        /// Append :VARIANT to select a specific GGUF quantization (defaults to Q4_K_M)
        #[arg(long, value_name = "REPO[:VARIANT]", conflicts_with = "model")]
        huggingface: Option<String>,

        /// Target platform (auto-detected if not specified)
        #[arg(short, long, value_name = "PLATFORM")]
        platform: Option<String>,
    },
    /// Manage the local model cache
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
    /// Run a pipeline from a configuration file, predefined pipeline name, or model ID
    Run {
        /// Path to the pipeline configuration file (YAML)
        #[arg(short, long, value_name = "FILE", conflicts_with_all = ["pipeline", "bundle", "model", "directory", "huggingface", "model_file"])]
        config: Option<PathBuf>,

        /// Predefined pipeline name (e.g., "hiiipe")
        #[arg(short, long, value_name = "NAME", conflicts_with_all = ["config", "bundle", "model", "directory", "huggingface", "model_file"])]
        pipeline: Option<String>,

        /// Path to a .xyb bundle file for direct execution
        #[arg(short, long, value_name = "FILE", conflicts_with_all = ["config", "pipeline", "model", "directory", "huggingface", "model_file"])]
        bundle: Option<PathBuf>,

        /// Model ID to run directly from registry (e.g., "kokoro-82m")
        /// Downloads the model if not cached, then runs inference
        #[arg(short, long, value_name = "ID", conflicts_with_all = ["config", "pipeline", "bundle", "directory", "huggingface", "model_file"])]
        model: Option<String>,

        /// Path to a local model directory containing model_metadata.json
        #[arg(short, long, value_name = "DIR", conflicts_with_all = ["config", "pipeline", "bundle", "model", "huggingface", "model_file"])]
        directory: Option<PathBuf>,

        /// HuggingFace repo to run (e.g., "LiquidAI/LFM2.5-350M-GGUF:Q4_K_M")
        /// Append :VARIANT to select a specific GGUF quantization (defaults to Q4_K_M)
        #[arg(long, value_name = "REPO[:VARIANT]", conflicts_with_all = ["config", "pipeline", "bundle", "model", "directory", "model_file"])]
        huggingface: Option<String>,

        /// Path to a local GGUF model file (auto-generates metadata)
        #[arg(long, value_name = "PATH", conflicts_with_all = ["config", "pipeline", "bundle", "model", "directory", "huggingface"])]
        model_file: Option<PathBuf>,

        /// Dry run the pipeline without executing it
        #[arg(long, default_value = "false")]
        dry_run: bool,

        /// Path to policy bundle file (YAML or JSON) to load into orchestrator
        #[arg(long, value_name = "FILE")]
        policy: Option<PathBuf>,

        /// Path to input audio file (WAV format)
        #[arg(long, value_name = "FILE")]
        input_audio: Option<PathBuf>,

        /// Input text for text-based models
        #[arg(long, value_name = "TEXT")]
        input_text: Option<String>,

        /// Path to an input image for vision-language models (repeatable)
        #[arg(long = "input-image", value_name = "FILE")]
        input_images: Vec<PathBuf>,

        /// A candidate for a choice-scoring model to score against
        /// --input-text (repeatable). The id names the candidate in the
        /// result; the model reads the text. The model's own fixed choices
        /// are scored after these.
        #[arg(
            long = "choice",
            value_name = "ID=TEXT",
            value_parser = parse_choice,
            requires = "input_text",
            conflicts_with_all = [
                "config",
                "pipeline",
                "input_audio",
                "input_images",
                "voice",
                "max_tokens",
                "speculative_cloud",
            ]
        )]
        choices: Vec<Choice>,

        /// Voice ID for TTS models (e.g., "af_bella", "am_adam")
        #[arg(long, value_name = "VOICE")]
        voice: Option<String>,

        /// Output file path for saving results (audio: .wav, text: .txt,
        /// choice scores: .json)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Target format for model resolution (onnx, coreml, tflite)
        /// If not specified, auto-detects based on platform
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,

        /// Enable detailed execution tracing with flame graph output
        #[arg(long, default_value = "false")]
        trace: bool,

        /// Print the model's chain-of-thought reasoning (`<think>` blocks),
        /// which is stripped out of the answer text by default
        #[arg(long, default_value = "false")]
        show_reasoning: bool,

        /// Token budget for LLM generation, shared by thinking and answer
        /// (default 2048; auto-raised for known thinking models). For multi-stage
        /// pipelines this applies to the first stage.
        #[arg(long, value_name = "N", value_parser = parse_max_tokens)]
        max_tokens: Option<usize>,

        /// Write an OTLP/JSON artifact (includes governed content; use no_content for metadata only)
        #[arg(long, value_name = "FILE")]
        trace_export: Option<PathBuf>,

        /// Serve from the cloud while the registry model downloads, instead of
        /// blocking on the download. Requires --api-key (or XYBRID_API_KEY) and
        /// a registry --model. Chat models only: the registry is asked first,
        /// and any other model downloads, then runs locally.
        #[arg(long)]
        speculative_cloud: bool,
    },
    /// Interactive REPL mode - keeps models loaded for fast repeated inference
    Repl {
        /// Path to the pipeline configuration file (YAML)
        #[arg(short, long, value_name = "FILE", conflicts_with_all = ["model", "model_file", "huggingface"])]
        config: Option<PathBuf>,

        /// Model ID to run directly from registry (e.g., "qwen2.5-0.5b-instruct")
        #[arg(short, long, value_name = "ID", conflicts_with_all = ["config", "model_file", "huggingface"])]
        model: Option<String>,

        /// Path to a local GGUF model file (auto-generates metadata)
        #[arg(long, value_name = "PATH", conflicts_with_all = ["config", "model", "huggingface"])]
        model_file: Option<PathBuf>,

        /// HuggingFace repo to run (e.g., "LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        /// Append :VARIANT to select a specific GGUF quantization (defaults to Q4_K_M)
        #[arg(long, value_name = "REPO[:VARIANT]", conflicts_with_all = ["config", "model", "model_file"])]
        huggingface: Option<String>,

        /// Voice ID for TTS models (e.g., "af_bella", "am_adam")
        #[arg(long, value_name = "VOICE")]
        voice: Option<String>,

        /// Execution target for REPL inference (auto, local/device, cloud, server)
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,

        /// Stream tokens as they are generated (LLM models only)
        #[arg(long)]
        stream: bool,

        /// Print the model's chain-of-thought reasoning (`<think>` blocks),
        /// which is stripped out of the answer text by default
        #[arg(long, default_value = "false")]
        show_reasoning: bool,

        /// Token budget for LLM generation, shared by thinking and answer
        /// (default 2048; auto-raised for known thinking models). Applies to every REPL turn.
        #[arg(long, value_name = "N", value_parser = parse_max_tokens)]
        max_tokens: Option<usize>,

        /// System prompt to set the assistant's behavior
        #[arg(long, value_name = "PROMPT")]
        system: Option<String>,

        /// Serve from the cloud while the registry model downloads in the
        /// background, then switch to local once ready. Requires --api-key
        /// (or XYBRID_API_KEY) and a registry --model. Chat models only: the
        /// registry is asked first, and any other model downloads, then runs
        /// locally.
        #[arg(long)]
        speculative_cloud: bool,

        /// Disable built-in tool calling (web_search, fetch_url, current_time),
        /// which is otherwise on for models whose metadata declares support
        #[arg(long)]
        no_tools: bool,

        /// Add user-defined tools from a JSON or YAML file. Each entry maps a
        /// JSON-schema'd function to a command; the model's arguments arrive
        /// as JSON on the command's stdin, its stdout is the tool result
        #[arg(long, value_name = "FILE")]
        tools_file: Option<PathBuf>,
    },
    /// Analyze an OTLP/JSON trace artifact produced by --trace-export
    Trace {
        /// OTLP/JSON trace artifact
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Export the content-free summary to JSON
        #[arg(long, value_name = "FILE")]
        export: Option<PathBuf>,
    },
    /// Package model artifacts into a .xyb bundle
    Pack {
        /// Model name (expects artifacts under ./models/<name>/ unless --path is specified)
        #[arg(value_name = "NAME")]
        name: String,

        /// Version string (e.g., 1.0.0)
        ///
        /// Long-only: `-v` belongs to the global `--verbose` flag, and clap
        /// panics at startup on the collision.
        #[arg(long, value_name = "VERSION", default_value = "0.1.0")]
        version: String,

        /// Target format (onnx, coreml, tflite, generic)
        #[arg(short, long, value_name = "TARGET", default_value = "onnx")]
        target: String,

        /// Custom path to model directory (overrides default ./models/<name>/)
        #[arg(short, long, value_name = "PATH")]
        path: Option<PathBuf>,
    },
    /// Fetch a model from the registry and produce a .xyb bundle
    Bundle {
        /// Model ID (e.g., "kokoro-82m", "qwen2.5-0.5b-instruct")
        #[arg(value_name = "MODEL")]
        model: String,

        /// Output path for the .xyb bundle
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Target platform (auto-detected if not specified)
        #[arg(short, long, value_name = "PLATFORM")]
        platform: Option<String>,
    },
    /// Report or manage registry telemetry settings
    Telemetry {
        #[command(subcommand)]
        command: TelemetryCommand,
    },
}

/// `--max-tokens 0` would load the model and only then fail in the native
/// layer ("non-positive max_tokens"); reject it at parse time instead.
fn parse_max_tokens(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(0) => Err("must be at least 1".to_string()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

/// `--choice ID=TEXT`: split at the first `=`, so the text may hold more of
/// them. The text may be empty and nothing is trimmed. Repeated and reserved
/// ids are left to the scorer's own check, which knows the fixed choices.
fn parse_choice(s: &str) -> Result<Choice, String> {
    match s.split_once('=') {
        None => Err("expected ID=TEXT".to_string()),
        Some(("", _)) => Err("the id before '=' is empty".to_string()),
        Some((id, text)) => Ok(Choice::new(id, text)),
    }
}

fn main() -> Result<()> {
    // Registry requests and telemetry from the CLI say `cli`, not the `rust`
    // that Rust SDK apps report.
    xybrid_sdk::set_binding("cli");

    let cli = Cli::parse();

    configure_log_level(&cli);

    if let Some(key) = &cli.api_key {
        xybrid_sdk::set_api_key(key);
    }
    let (render_trace, trace_path) = match &cli.command {
        Commands::Run {
            trace,
            trace_export,
            ..
        } => (*trace, trace_export.clone()),
        _ => (false, None),
    };
    // OTEL_SDK_DISABLED disables export on every path; a requested local
    // trace (--trace / --trace-export) still renders and writes in-process.
    let sdk_disabled = otel_sdk_disabled();
    let local_capture = if render_trace || trace_path.is_some() {
        let requested = observability_config(&cli, OtlpConfig::from_env());
        if sdk_disabled && requested.is_some() && !cli.quiet {
            ui::status("OTEL_SDK_DISABLED is set: the trace stays local, nothing is exported");
        }
        Some(tracing_viz::LocalCapture::configure(
            requested.filter(|_| !sdk_disabled),
            cli.device_id.as_deref(),
        )?)
    } else {
        None
    };
    // `telemetry status` only reports configuration: it starts no exporter.
    let reports_only = matches!(cli.command, Commands::Telemetry { .. });
    let exporting = match &local_capture {
        Some(capture) => capture.exports(),
        None if reports_only => false,
        None => init_observability(&cli),
    };
    if exporting && !cli.quiet {
        ui::status("Observability enabled");
    }

    let root = local_capture.as_ref().map(tracing_viz::LocalCapture::begin);
    let scope = root.as_ref().map(|root| root.enter());
    let result = run_command(cli);
    drop(scope);
    if let Some(root) = &root {
        if result.is_ok() {
            root.end();
        } else {
            root.end_with_error(&xybrid_sdk::observability::ErrorInfo::new("command_error"));
        }
    }
    if let Some(capture) = &local_capture {
        capture.finish(render_trace, trace_path.as_deref())?;
        if let Err(error) = capture.shutdown() {
            ui::warning(&format!("Observability shutdown failed: {error}"));
        }
    }

    if exporting && local_capture.is_none() {
        if let Err(error) = xybrid_sdk::shutdown_observability(Duration::from_secs(5)) {
            ui::warning(&format!("Observability shutdown failed: {error}"));
        }
    }

    if let Err(ref err) = result {
        if let Some(msg) = find_model_not_found(err) {
            ui::err(msg);
            println!();
            ui::hint("Want this model added? Open a model request:");
            ui::hint("  https://github.com/xybrid-ai/xybrid/issues/new?template=model-request.yml");
            std::process::exit(1);
        }

        if find_offline_error(err).is_some() {
            render_offline_error(err);
            std::process::exit(1);
        }
    }

    result
}

/// Walk the anyhow error chain looking for SdkError::ModelNotFound.
fn find_model_not_found(err: &anyhow::Error) -> Option<&str> {
    for cause in err.chain() {
        if let Some(SdkError::ModelNotFound(msg)) = cause.downcast_ref::<SdkError>() {
            return Some(msg.as_str());
        }
    }
    None
}

/// Walk the anyhow error chain looking for SdkError::Offline.
///
/// Returns the underlying offline-error message (e.g., "Failed to resolve model
/// (DNS resolution failed)") so callers can tell whether the failure was a
/// local unreachability issue rather than a registry-side problem.
fn find_offline_error(err: &anyhow::Error) -> Option<&str> {
    for cause in err.chain() {
        if let Some(SdkError::Offline { message, .. }) = cause.downcast_ref::<SdkError>() {
            return Some(message.as_str());
        }
    }
    None
}

/// Render the user-facing offline error with actionable hints.
///
/// Walks the anyhow chain to reconstruct the top-level context (e.g., "Failed
/// to resolve model 'X' from registry") and pairs it with a friendly
/// explanation plus a list of models that ARE available for offline use. The
/// goal is to replace jargon like "Circuit breaker open" with something the
/// user can actually act on.
fn render_offline_error(err: &anyhow::Error) {
    // The top-level anyhow message is the context we want (e.g., the
    // model ID the user asked for). The Offline variant lives deeper in
    // the chain and carries the concrete cause (DNS / connect refused).
    ui::err(&format!("{}", err));
    println!();
    ui::hint("Can't reach the xybrid registry — looks like you're offline");
    ui::hint("or the registry host is unreachable. Check your network");
    ui::hint("connection and try again.");

    // Show what the user can run right now without a network. If we can't
    // load a registry client (shouldn't happen in practice) or the cache
    // is empty, fall back to a simpler hint.
    if let Ok(client) = xybrid_sdk::registry_client::RegistryClient::from_env() {
        let cached = client.list_offline_models();
        if !cached.is_empty() {
            println!();
            ui::hint(&format!("Models available offline ({}):", cached.len()));
            for id in &cached {
                ui::sub(id);
            }
        }
    }
}

/// Configure the global log level based on CLI verbosity flags.
fn configure_log_level(cli: &Cli) {
    use xybrid_core::telemetry::{set_global_log_level, LogLevel};

    let level = if cli.quiet {
        LogLevel::Quiet
    } else {
        match cli.verbose {
            0 => LogLevel::Normal,
            1 => LogLevel::Verbose,
            _ => LogLevel::VeryVerbose,
        }
    };

    set_global_log_level(level);
}

/// Select explicit trace export or standard OTLP environment configuration.
fn observability_config(cli: &Cli, environment: Option<OtlpConfig>) -> Option<OtlpConfig> {
    let mut config = if cli.observability {
        OtlpConfig::xybrid_traces_to(
            cli.traces_endpoint
                .as_deref()
                .unwrap_or(XYBRID_TRACES_ENDPOINT),
            cli.api_key.as_deref()?,
        )
    } else {
        environment?
    };
    if let Some(device_id) = &cli.device_id {
        config = config.with_resource_attribute(xybrid_sdk::observability::KeyValue::new(
            "device.id",
            device_id.clone(),
        ));
    }
    Some(config.with_service("xybrid-cli", Some(env!("CARGO_PKG_VERSION").into())))
}

/// Whether `OTEL_SDK_DISABLED=true` turns OpenTelemetry export off.
pub(crate) fn otel_sdk_disabled() -> bool {
    std::env::var("OTEL_SDK_DISABLED").is_ok_and(|value| value.trim().eq_ignore_ascii_case("true"))
}

/// Configure cloud authentication independently from optional observability.
fn init_observability(cli: &Cli) -> bool {
    if let Some(ref api_key) = cli.api_key {
        // Make the key resolvable by the cloud gateway too (speculative cloud,
        // reactive fallback). `--api-key` may
        // arrive by flag rather than the XYBRID_API_KEY env var, so set the
        // in-memory cell that CloudConfig::resolve_api_key reads.
        xybrid_sdk::set_api_key(api_key);
    }
    if otel_sdk_disabled() {
        return false;
    }
    let Some(config) = observability_config(cli, OtlpConfig::from_env()) else {
        return false;
    };
    match xybrid_sdk::init_observability(
        ObservabilityConfig::otlp(config)
            .resource_telemetry(commands::telemetry::CLI_RESOURCE_TELEMETRY),
    ) {
        Ok(()) => true,
        Err(error) => {
            ui::warning(&format!("Observability was not started: {error}"));
            false
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_command(cli: Cli) -> Result<()> {
    let verbose = cli.verbose;
    let telemetry_config = observability_config(&cli, OtlpConfig::from_env());
    let telemetry_explicit = cli.observability;
    // Point the SDK's cloud gateway at the requested endpoint, so cloud calls
    // (speculative cloud, reactive fallback) don't silently fall back to
    // the production default (`api.xybrid.dev`) and 401 with a staging key.
    // `cli.platform_url` already encodes clap's flag > env > default precedence,
    // so set it unconditionally. Use the in-memory setter rather than
    // `std::env::set_var`: exporter threads may already be running by now, and a
    // concurrent `setenv`/`getenv` is UB. XYBRID_GATEWAY_URL still wins if set.
    xybrid_sdk::set_platform_url(&cli.platform_url);
    match cli.command {
        Commands::Init {
            directory,
            force,
            yes,
            task,
            json,
            model_id,
        } => commands::init::handle_init_command(
            &directory,
            commands::init::InitFlags {
                force,
                yes,
                task,
                json,
                model_id,
            },
        ),
        Commands::Models { command } => commands::models::handle_models_command(command),
        Commands::Prepare { config } => commands::pipeline::handle_prepare_command(&config),
        Commands::Plan { config } => commands::pipeline::handle_plan_command(&config),
        Commands::Fetch {
            config,
            model,
            huggingface,
            platform,
        } => {
            if let Some(config_path) = config {
                commands::fetch::handle_fetch_pipeline_command(&config_path, platform.as_deref())
            } else if let Some(model_id) = model {
                commands::fetch::handle_fetch_command(&model_id, platform.as_deref())
            } else if let Some(repo) = huggingface {
                commands::fetch::handle_fetch_huggingface_command(&repo)
            } else {
                Err(anyhow::anyhow!(
                    "Either a pipeline config file, --model <id>, or --huggingface <repo> must be specified"
                ))
            }
        }
        Commands::Cache { command } => commands::cache::handle_cache_command(command),
        Commands::Run {
            config,
            pipeline,
            bundle,
            model,
            directory,
            huggingface,
            model_file,
            dry_run,
            policy,
            input_audio,
            input_text,
            input_images,
            choices,
            voice,
            output,
            target,
            trace,
            show_reasoning,
            max_tokens,
            // Captured and written by the command owner in `main`.
            trace_export: _,
            speculative_cloud,
        } => {
            if let Some(gguf_path) = model_file {
                return commands::run::run_model_file(
                    &gguf_path,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    &input_images,
                    &choices,
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    show_reasoning,
                    max_tokens,
                );
            }

            // A dry run must not hit the network, so let `--dry-run` fall
            // through to `run_model` (plan-only), which also honours
            // --trace/--trace-export that the speculative path doesn't emit.
            if speculative_cloud && !dry_run {
                if let Some(model_id) = model.as_deref() {
                    return commands::run::run_model_speculative(
                        model_id,
                        input_audio.as_ref(),
                        input_text.as_deref(),
                        &input_images,
                        voice.as_deref(),
                        output.as_ref(),
                        max_tokens,
                    );
                }
                return Err(anyhow::anyhow!(
                    "--speculative-cloud requires a registry --model"
                ));
            }

            if let Some(model_id) = model {
                return commands::run::run_model(
                    &model_id,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    &input_images,
                    &choices,
                    voice.as_deref(),
                    output.as_ref(),
                    target.as_deref(),
                    dry_run,
                    speculative_cloud,
                    trace,
                    show_reasoning,
                    max_tokens,
                );
            }

            if let Some(dir) = directory {
                return commands::run::run_directory(
                    &dir,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    &input_images,
                    &choices,
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    show_reasoning,
                    max_tokens,
                );
            }

            if let Some(repo) = huggingface {
                return commands::run::run_huggingface(
                    &repo,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    &input_images,
                    &choices,
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    show_reasoning,
                    max_tokens,
                );
            }

            if let Some(bundle_path) = bundle {
                return commands::run::run_bundle(
                    &bundle_path,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    &input_images,
                    &choices,
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    show_reasoning,
                    max_tokens,
                );
            }

            let config_path = resolve_config_path(config, pipeline)?;
            commands::run::run_pipeline(
                &config_path,
                dry_run,
                policy.as_ref(),
                input_audio.as_ref(),
                input_text.as_deref(),
                &input_images,
                voice.as_deref(),
                output.as_ref(),
                target.as_deref(),
                trace,
                show_reasoning,
                max_tokens,
            )
        }
        Commands::Repl {
            config,
            model,
            model_file,
            huggingface,
            voice,
            target,
            stream,
            show_reasoning,
            max_tokens,
            system,
            speculative_cloud,
            no_tools,
            tools_file,
        } => commands::repl::handle_repl_command(commands::repl::ReplArgs {
            config,
            model,
            model_file,
            huggingface,
            voice,
            target,
            stream,
            show_reasoning,
            max_tokens,
            system_prompt: system,
            speculative_cloud,
            no_tools,
            tools_file,
            verbose,
        }),
        Commands::Trace { file, export } => commands::trace::analyze(&file, export.as_deref()),
        Commands::Pack {
            name,
            version,
            target,
            path,
        } => commands::pack::pack_model(&name, &version, &target, path.as_deref()),
        Commands::Bundle {
            model,
            output,
            platform,
        } => commands::bundle::handle_bundle_command(&model, output, platform.as_deref()),
        Commands::Telemetry { command } => commands::telemetry::handle_telemetry_command(
            command,
            telemetry_config,
            telemetry_explicit,
        ),
    }
}

/// Resolve the pipeline config path from --config or --pipeline arguments.
fn resolve_config_path(config: Option<PathBuf>, pipeline: Option<String>) -> Result<PathBuf> {
    if let Some(path) = config {
        return Ok(path);
    }

    if let Some(pipeline_name) = pipeline {
        let mut base_dir = std::env::current_dir().context("Failed to get current directory")?;
        base_dir.push("xybrid-cli");
        base_dir.push("examples");

        let mut p = base_dir.clone();
        p.push(format!("{}.yml", pipeline_name));

        if !p.exists() {
            p = base_dir.clone();
            p.push(format!("{}.yaml", pipeline_name));
        }

        if !p.exists() {
            let mut root_dir =
                std::env::current_dir().context("Failed to get current directory")?;
            root_dir.push("examples");

            p = root_dir.clone();
            p.push(format!("{}.yml", pipeline_name));

            if !p.exists() {
                p = root_dir;
                p.push(format!("{}.yaml", pipeline_name));
            }
        }

        if !p.exists() {
            return Err(anyhow::anyhow!(
                "Pipeline '{}' not found. Looked in:\n  - xybrid-cli/examples/{}.yml\n  - xybrid-cli/examples/{}.yaml\n  - examples/{}.yml\n  - examples/{}.yaml",
                pipeline_name, pipeline_name, pipeline_name, pipeline_name, pipeline_name
            ));
        }

        return Ok(p);
    }

    Err(anyhow::anyhow!(
        "Either --config, --pipeline, --bundle, or --model must be specified"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_observability_args(args: &[&str]) -> Result<Cli, clap::Error> {
        use clap::{CommandFactory, FromArgMatches};

        // Keep authentication and device attribution tests independent of the
        // developer's environment without mutating process-wide variables.
        let matches = Cli::command()
            .mut_arg("api_key", |arg| arg.env(None::<&str>))
            .mut_arg("device_id", |arg| arg.env(None::<&str>))
            .try_get_matches_from(args)?;
        Cli::from_arg_matches(&matches)
    }

    #[test]
    fn cloud_authentication_alone_does_not_enable_export() {
        let cli = parse_observability_args(&["xybrid", "--api-key", "cloud-key", "models", "list"])
            .expect("cloud authentication is optional");
        assert!(observability_config(&cli, None).is_none());
    }

    #[test]
    fn explicit_trace_export_is_independent_of_cloud_gateway() {
        let cli = parse_observability_args(&[
            "xybrid",
            "--api-key",
            "trace-key",
            "--observability",
            "--platform-url",
            "https://gateway.example.test",
            "models",
            "list",
        ])
        .expect("explicit tracing should parse");
        let config =
            observability_config(&cli, Some(OtlpConfig::collector("http://localhost:4318")))
                .expect("explicit tracing should be configured");
        let traces = config.traces.expect("trace destination");
        assert_eq!(traces.endpoint, XYBRID_TRACES_ENDPOINT);
        assert_eq!(
            traces.headers,
            [("Authorization".into(), "Bearer trace-key".into())]
        );
        assert!(config.metrics.is_none());
        assert!(config.logs.is_none());
        assert!(
            config.resource.attributes.is_empty(),
            "no inferred device identity"
        );
        assert_eq!(config.resource.service_name.as_deref(), Some("xybrid-cli"));
    }

    #[test]
    fn explicit_trace_export_preserves_custom_endpoint_and_device() {
        let cli = parse_observability_args(&[
            "xybrid",
            "--api-key",
            "trace-key",
            "--observability",
            "--traces-endpoint",
            "https://collector.example.test/custom/traces",
            "--device-id",
            "test-device",
            "models",
            "list",
        ])
        .expect("custom tracing should parse");
        let config = observability_config(&cli, None).expect("trace configuration");
        assert_eq!(
            config.traces.unwrap().endpoint,
            "https://collector.example.test/custom/traces"
        );
        assert_eq!(
            config.resource.attributes,
            [xybrid_sdk::observability::KeyValue::new(
                "device.id",
                "test-device"
            )]
        );
    }

    #[test]
    fn standard_otlp_config_does_not_inherit_cloud_credentials() {
        for args in [
            vec!["xybrid", "models", "list"],
            vec!["xybrid", "--api-key", "cloud-key", "models", "list"],
        ] {
            let cli = parse_observability_args(&args).expect("cloud credentials are optional");
            let collector = OtlpConfig::collector("http://localhost:4318");
            let config = observability_config(&cli, Some(collector.clone())).expect("OTLP opt-in");
            assert_eq!(config.traces, collector.traces);
            assert_eq!(config.metrics, collector.metrics);
            assert_eq!(config.logs, collector.logs);
        }
    }

    #[test]
    fn trace_flags_require_explicit_authentication_and_opt_in() {
        for args in [
            vec!["xybrid", "--observability", "models", "list"],
            vec![
                "xybrid",
                "--api-key",
                "cloud-key",
                "--traces-endpoint",
                "http://localhost:4318/v1/traces",
                "models",
                "list",
            ],
        ] {
            let Err(error) = parse_observability_args(&args) else {
                panic!("missing tracing requirement should be rejected");
            };
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::MissingRequiredArgument
            );
        }
    }

    #[test]
    fn every_subcommand_parses_without_arg_collisions() {
        // clap only detects duplicate short flags (like `pack -v` vs the
        // global `-v/--verbose`) when the subcommand's arg set is built —
        // exercise each one so a collision fails here instead of panicking
        // at run time.
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn repl_accepts_show_reasoning_flag() {
        let parsed = Cli::try_parse_from([
            "xybrid",
            "repl",
            "--show-reasoning",
            "--model-file",
            "model.gguf",
        ]);

        let cli = parsed.unwrap_or_else(|err| panic!("repl should accept --show-reasoning: {err}"));
        let Commands::Repl { show_reasoning, .. } = cli.command else {
            panic!("expected repl command");
        };
        assert!(show_reasoning);
    }

    #[test]
    fn run_and_repl_accept_max_tokens_flag() {
        let run = Cli::try_parse_from([
            "xybrid",
            "run",
            "--max-tokens",
            "64",
            "--model-file",
            "model.gguf",
        ])
        .unwrap_or_else(|err| panic!("run should accept --max-tokens: {err}"));
        let Commands::Run { max_tokens, .. } = run.command else {
            panic!("expected run command");
        };
        assert_eq!(max_tokens, Some(64));

        let repl = Cli::try_parse_from([
            "xybrid",
            "repl",
            "--max-tokens",
            "64",
            "--model-file",
            "model.gguf",
        ])
        .unwrap_or_else(|err| panic!("repl should accept --max-tokens: {err}"));
        let Commands::Repl { max_tokens, .. } = repl.command else {
            panic!("expected repl command");
        };
        assert_eq!(max_tokens, Some(64));
    }

    #[test]
    fn run_collects_repeated_choices_in_order() {
        let cli = Cli::try_parse_from([
            "xybrid",
            "run",
            "--directory",
            "scorer",
            "--input-text",
            "FORM Intake",
            "--choice",
            "tel=fill Tel: (503) 555-0142",
            "--choice",
            "name=fill Name: Jane Doe",
        ])
        .unwrap_or_else(|err| panic!("run should accept --choice: {err}"));
        let Commands::Run { choices, .. } = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(
            choices,
            [
                Choice::new("tel", "fill Tel: (503) 555-0142"),
                Choice::new("name", "fill Name: Jane Doe"),
            ]
        );
    }

    #[test]
    fn parse_choice_splits_at_the_first_equals_sign() {
        assert_eq!(parse_choice("a=b"), Ok(Choice::new("a", "b")));
        assert_eq!(parse_choice("a=b=c"), Ok(Choice::new("a", "b=c")));
        assert_eq!(parse_choice("a="), Ok(Choice::new("a", "")));
        assert_eq!(parse_choice(" a = b "), Ok(Choice::new(" a ", " b ")));
        assert_eq!(parse_choice("=b"), Err("the id before '=' is empty".into()));
        assert_eq!(parse_choice("ab"), Err("expected ID=TEXT".into()));
    }

    #[test]
    fn malformed_choices_fail_at_parse_time() {
        for (value, expected) in [
            ("ab", "expected ID=TEXT"),
            ("=b", "the id before '=' is empty"),
        ] {
            let result = Cli::try_parse_from([
                "xybrid",
                "run",
                "--model",
                "m",
                "--input-text",
                "ctx",
                "--choice",
                value,
            ]);
            let Err(err) = result else {
                panic!("--choice {value} must be rejected before any model loads");
            };
            assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
            assert!(err.to_string().contains(expected), "{err}");
        }
    }

    #[test]
    fn choice_needs_input_text() {
        let result = Cli::try_parse_from(["xybrid", "run", "--model", "m", "--choice", "a=b"]);
        let Err(err) = result else {
            panic!("--choice without --input-text must be rejected");
        };
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::MissingRequiredArgument,
            "{err}"
        );
        assert!(err.to_string().contains("--input-text"), "{err}");
    }

    #[test]
    fn choice_conflicts_with_flags_a_scorer_cannot_use() {
        let cases: [&[&str]; 7] = [
            &["--config", "pipeline.yml"],
            &["--pipeline", "hiiipe"],
            &["--model", "m", "--input-audio", "speech.wav"],
            &["--model", "m", "--input-image", "page.png"],
            &["--model", "m", "--voice", "af_bella"],
            &["--model", "m", "--max-tokens", "8"],
            &["--model", "m", "--speculative-cloud"],
        ];
        for extra in cases {
            let mut args = vec!["xybrid", "run", "--input-text", "ctx", "--choice", "a=b"];
            args.extend_from_slice(extra);
            let Err(err) = Cli::try_parse_from(&args) else {
                panic!("{extra:?} must conflict with --choice");
            };
            assert_eq!(
                err.kind(),
                clap::error::ErrorKind::ArgumentConflict,
                "{extra:?}: {err}"
            );
        }
    }

    #[test]
    fn run_and_repl_reject_zero_max_tokens_at_parse_time() {
        for cmd in ["run", "repl"] {
            let result = Cli::try_parse_from([
                "xybrid",
                cmd,
                "--max-tokens",
                "0",
                "--model-file",
                "model.gguf",
            ]);
            let Err(err) = result else {
                panic!("--max-tokens 0 must be rejected before any model loads");
            };
            assert!(
                err.to_string().contains("must be at least 1"),
                "unexpected parse error: {err}"
            );
        }
    }
}

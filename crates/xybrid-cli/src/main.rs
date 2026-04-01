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

use commands::{CacheCommand, ModelsCommand};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;
use xybrid_core::target::Platform;
use xybrid_sdk::SdkError;

/// Xybrid CLI - Hybrid Cloud-Edge AI Inference Pipeline Runner
#[derive(Parser)]
#[command(name = "xybrid")]
#[command(about = "Xybrid CLI - Run hybrid cloud-edge AI inference pipelines", long_about = None)]
struct Cli {
    /// Platform API key for telemetry (can also be set via XYBRID_API_KEY env var)
    #[arg(long, global = true, env = "XYBRID_API_KEY")]
    api_key: Option<String>,

    /// Platform API endpoint for telemetry (default: https://api.xybrid.dev)
    #[arg(
        long,
        global = true,
        env = "XYBRID_PLATFORM_URL",
        default_value = "https://api.xybrid.dev"
    )]
    platform_url: String,

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

        /// HuggingFace repo to fetch (e.g., "DataikuNLP/kiji-pii-model-onnx")
        #[arg(long, value_name = "REPO", conflicts_with = "model")]
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

        /// HuggingFace repo to run (e.g., "DataikuNLP/kiji-pii-model-onnx")
        /// Downloads if not cached, auto-generates metadata if needed
        #[arg(long, value_name = "REPO", conflicts_with_all = ["config", "pipeline", "bundle", "model", "directory", "model_file"])]
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

        /// Voice ID for TTS models (e.g., "af_bella", "am_adam")
        #[arg(long, value_name = "VOICE")]
        voice: Option<String>,

        /// Output file path for saving results (audio: .wav, text: .txt)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Target format for model resolution (onnx, coreml, tflite)
        /// If not specified, auto-detects based on platform
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,

        /// Enable detailed execution tracing with flame graph output
        #[arg(long, default_value = "false")]
        trace: bool,

        /// Export trace to JSON file (Chrome trace format)
        #[arg(long, value_name = "FILE")]
        trace_export: Option<PathBuf>,
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

        /// HuggingFace repo to run (e.g., "prism-ml/Bonsai-8B-gguf")
        /// Downloads if not cached, auto-generates metadata if needed
        #[arg(long, value_name = "REPO", conflicts_with_all = ["config", "model", "model_file"])]
        huggingface: Option<String>,

        /// Voice ID for TTS models (e.g., "af_bella", "am_adam")
        #[arg(long, value_name = "VOICE")]
        voice: Option<String>,

        /// Target format for model resolution (onnx, coreml, tflite)
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,

        /// Stream tokens as they are generated (LLM models only)
        #[arg(long)]
        stream: bool,

        /// System prompt to set the assistant's behavior
        #[arg(long, value_name = "PROMPT")]
        system: Option<String>,
    },
    /// Trace and analyze telemetry logs from a session
    Trace {
        /// Session ID to load telemetry logs for
        #[arg(short, long, value_name = "ID", conflicts_with = "latest")]
        session: Option<String>,

        /// Load the most recent session
        #[arg(long)]
        latest: bool,

        /// Export trace summary to JSON file
        #[arg(long, value_name = "FILE")]
        export: Option<PathBuf>,
    },
    /// Package model artifacts into a .xyb bundle
    Pack {
        /// Model name (expects artifacts under ./models/<name>/ unless --path is specified)
        #[arg(value_name = "NAME")]
        name: String,

        /// Version string (e.g., 1.0.0)
        #[arg(short, long, value_name = "VERSION", default_value = "0.1.0")]
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    configure_log_level(&cli);

    let telemetry_enabled = init_telemetry(&cli);
    if telemetry_enabled && !cli.quiet {
        println!("📡 Telemetry enabled ({})", cli.platform_url);
    }

    let result = run_command(cli);

    if telemetry_enabled {
        xybrid_sdk::flush_platform_telemetry();
        xybrid_sdk::shutdown_platform_telemetry();
    }

    if let Err(ref err) = result {
        if let Some(msg) = find_model_not_found(err) {
            eprintln!("{} {}\n", "error:".bright_red().bold(), msg);
            eprintln!(
                "Want this model added? Open a model request:\n  https://github.com/xybrid-ai/xybrid/issues/new?template=model-request.yml"
            );
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

/// Initialize platform telemetry from CLI args.
fn init_telemetry(cli: &Cli) -> bool {
    if let Some(ref api_key) = cli.api_key {
        let platform = Platform::detect().to_string();

        let device_id = cli.device_id.clone().unwrap_or_else(|| {
            hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "cli-unknown".to_string())
        });

        let mut config = xybrid_sdk::TelemetryConfig::new(&cli.platform_url, api_key);
        config = config.with_device(&device_id, &platform);
        config = config.with_app_version(env!("CARGO_PKG_VERSION"));

        xybrid_sdk::init_platform_telemetry(config);
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn run_command(cli: Cli) -> Result<()> {
    let verbose = cli.verbose;
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
            voice,
            output,
            target,
            trace,
            trace_export,
        } => {
            if trace {
                tracing_viz::reset_collector();
            }

            if let Some(gguf_path) = model_file {
                return commands::run::run_model_file(
                    &gguf_path,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    trace_export.as_ref(),
                );
            }

            if let Some(model_id) = model {
                return commands::run::run_model(
                    &model_id,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    voice.as_deref(),
                    output.as_ref(),
                    target.as_deref(),
                    dry_run,
                    trace,
                    trace_export.as_ref(),
                );
            }

            if let Some(dir) = directory {
                return commands::run::run_directory(
                    &dir,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    trace_export.as_ref(),
                );
            }

            if let Some(repo) = huggingface {
                return commands::run::run_huggingface(
                    &repo,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    trace_export.as_ref(),
                );
            }

            if let Some(bundle_path) = bundle {
                return commands::run::run_bundle(
                    &bundle_path,
                    input_audio.as_ref(),
                    input_text.as_deref(),
                    voice.as_deref(),
                    output.as_ref(),
                    dry_run,
                    trace,
                    trace_export.as_ref(),
                );
            }

            let config_path = resolve_config_path(config, pipeline)?;
            commands::run::run_pipeline(
                &config_path,
                dry_run,
                policy.as_ref(),
                input_audio.as_ref(),
                input_text.as_deref(),
                voice.as_deref(),
                output.as_ref(),
                target.as_deref(),
                trace,
                trace_export.as_ref(),
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
            system,
        } => commands::repl::handle_repl_command(
            config,
            model,
            model_file,
            huggingface,
            voice,
            target,
            stream,
            system,
            verbose,
        ),
        Commands::Trace {
            session,
            latest,
            export,
        } => {
            let session_id = if latest {
                commands::trace::find_latest_session()?
            } else {
                session
            };
            if session_id.is_none() && latest {
                return Err(anyhow::anyhow!(
                    "No sessions found. Use --session <id> to specify a session."
                ));
            }
            commands::trace::trace_session(session_id, export.as_deref())
        }
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

//! Xybrid CLI - Command-line interface for running hybrid cloud-edge AI inference pipelines.
//!
//! This binary provides subcommands for managing and executing ML inference pipelines.
//!
//! ## Module Organization
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`commands`] | Command handlers organized by subcommand |
//! | [`platform`] | Xybrid platform credentials for `connect` / `auth` and the key resolver |
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
//! | `connect` | Provision a claimable Xybrid workspace and key for this project |
//! | `auth` | Inspect, recover or forget this project's Xybrid credentials |
//!
//! `connect` and `auth` are dispatched before any observability setup and
//! never export. Every other command resolves one platform key (`--api-key`,
//! then `XYBRID_API_KEY`, then the credentials `connect` saved for this host
//! and repository); finding a key configures cloud authentication only, and
//! trace export still needs an explicit `--observability`.

mod commands;
mod platform;
#[allow(dead_code)]
mod tracing_viz;
pub mod ui;

use commands::{CacheCommand, ModelsCommand, TelemetryCommand};

use anyhow::{Context, Result};
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, Subcommand};
use platform::host::{self, PlatformHost};
use platform::resolve::{self, KeySource, ResolvedKey};
use std::io::Write;
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
    /// Platform API key for cloud access (does not enable observability by
    /// itself). When neither this nor XYBRID_API_KEY is set, the key `xybrid
    /// connect` saved for this platform host and repository is used
    #[arg(long, global = true, env = "XYBRID_API_KEY")]
    api_key: Option<String>,

    /// Xybrid platform API origin: the cloud gateway, and the host `connect`
    /// and `auth` talk to (production and staging map to pinned console and
    /// ingest origins; anything else needs `--allow-custom-host` there)
    #[arg(
        long,
        global = true,
        env = "XYBRID_PLATFORM_URL",
        default_value = "https://api.xybrid.dev"
    )]
    platform_url: String,

    /// Export traces to Xybrid with the resolved key (--api-key,
    /// XYBRID_API_KEY or `xybrid connect`'s saved key); alternatively configure
    /// OTEL_EXPORTER_OTLP_* variables
    #[arg(long, global = true)]
    observability: bool,

    /// Full OTLP trace URL for --observability (default: https://ingest.xybrid.dev/v1/traces,
    /// or the platform's pinned ingest URL for a saved key)
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

        /// Export trace to JSON file (Chrome trace format)
        #[arg(long, value_name = "FILE")]
        trace_export: Option<PathBuf>,

        /// Serve from the cloud while the registry model downloads, instead of
        /// blocking on the download. Requires a key (--api-key, XYBRID_API_KEY
        /// or `xybrid connect`'s saved key) and a registry --model. Chat models only: the registry is asked first,
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
        /// background, then switch to local once ready. Requires a key
        /// (--api-key, XYBRID_API_KEY or `xybrid connect`'s saved key) and a
        /// registry --model. Chat models only: the
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
    /// Connect this project to a claimable Xybrid workspace and save its key
    ///
    /// Built for coding agents: no browser and no account. Provisions (or
    /// reuses) a workspace with a restricted key, writes the key to an
    /// untracked, gitignored project file when --write-env is given, and keeps
    /// the claim link in private credentials (`xybrid auth claim`). It never
    /// enables trace export and never prints the key. Exit codes: 0 created or
    /// existing, 3 signup closed, 4 offline, 5 CI or opted out, 6 project file
    /// tracked by Git, 7 workspace expired, 1 error.
    Connect(commands::connect::ConnectArgs),
    /// Inspect, recover or forget this project's Xybrid credentials
    Auth {
        #[command(subcommand)]
        command: commands::auth::AuthCommand,
    },
}

impl Commands {
    /// Whether this invocation prints machine-readable JSON on stdout, so
    /// every hint and notice must stay off it (and out of the way).
    fn json_output(&self) -> bool {
        match self {
            Commands::Init { json, .. } => *json,
            _ => false,
        }
    }

    /// Commands that run inference, where the no-export tip is useful.
    fn runs_inference(&self) -> bool {
        matches!(self, Commands::Run { .. } | Commands::Repl { .. })
    }
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

    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|err| err.exit());
    let explicit_key = explicit_key(&cli, &matches);

    configure_log_level(&cli);

    // `connect` and `auth` manage credentials. They run before, and instead
    // of, any observability or cloud setup, even with OTEL_EXPORTER_OTLP_*
    // configured: they neither export nor emit inference notices.
    if let Some(code) = run_platform_command(&cli, explicit_key.clone()) {
        let _ = std::io::stdout().flush();
        std::process::exit(code);
    }

    // Point the SDK's cloud gateway at the requested endpoint, so cloud calls
    // (speculative cloud, reactive fallback) don't silently fall back to
    // the production default (`api.xybrid.dev`) and 401 with a staging key.
    // `cli.platform_url` already encodes clap's flag > env > default precedence,
    // so set it unconditionally. Use the in-memory setter rather than
    // `std::env::set_var`: exporter threads may start below, and a concurrent
    // `setenv`/`getenv` is UB. XYBRID_GATEWAY_URL still wins if set.
    xybrid_sdk::set_platform_url(&cli.platform_url);

    let host = PlatformHost::parse(&cli.platform_url).ok();
    let project_root = std::env::current_dir()
        .ok()
        .and_then(|cwd| platform::git::project_root(&cwd).ok());
    let (key, store_warning) =
        resolve::resolve(explicit_key, host.as_ref(), project_root.as_deref());
    configure_cloud_auth(key.as_ref(), host.as_ref());

    let json = cli.command.json_output();
    let observability_enabled =
        init_observability(&cli, key.as_ref(), host.as_ref(), store_warning.as_deref())?;
    if observability_enabled && !cli.quiet && !json {
        // stderr: stdout belongs to the command's own output.
        ui::hint_stderr("Observability enabled");
    }
    platform::notices::maybe_claim_reminder(
        key.as_ref()
            .filter(|key| key.source == KeySource::Credentials)
            .and_then(|key| key.credential.as_ref()),
        cli.quiet,
        json,
    );
    if key.is_some() && !observability_enabled && cli.command.runs_inference() {
        platform::notices::maybe_export_tip(cli.quiet, json);
    }

    let result = run_command(cli);

    if observability_enabled {
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

/// `--api-key` / `XYBRID_API_KEY` (Clap merges them), with where it came from.
fn explicit_key(cli: &Cli, matches: &ArgMatches) -> Option<ResolvedKey> {
    let from_flag = matches.value_source("api_key") == Some(clap::parser::ValueSource::CommandLine);
    resolve::explicit(cli.api_key.as_deref(), from_flag)
}

/// Run `connect` / `auth`, returning their exit code, or `None` for any
/// other command.
fn run_platform_command(cli: &Cli, explicit_key: Option<ResolvedKey>) -> Option<i32> {
    let ctx = commands::connect::PlatformArgs {
        platform_url: &cli.platform_url,
        explicit_key,
        quiet: cli.quiet,
    };
    match &cli.command {
        Commands::Connect(args) => Some(commands::connect::run(&ctx, args)),
        Commands::Auth { command } => Some(commands::auth::run(&ctx, command)),
        _ => None,
    }
}

/// Make the resolved key available to the cloud gateway (speculative cloud,
/// reactive fallback) through the in-memory cell `CloudConfig` reads, since
/// `--api-key` may arrive by flag rather than `XYBRID_API_KEY`. A saved key
/// belongs to one platform origin, so it is only handed over when the gateway
/// is that origin (`XYBRID_GATEWAY_URL` can point elsewhere).
fn configure_cloud_auth(key: Option<&ResolvedKey>, host: Option<&PlatformHost>) {
    let Some(key) = key else {
        return;
    };
    let allowed = match key.source {
        KeySource::Flag | KeySource::Env => true,
        KeySource::Credentials => host.is_some_and(|host| {
            xybrid_core::cloud::same_origin(
                &xybrid_core::cloud::platform_gateway_url(),
                host.origin(),
            )
        }),
    };
    if allowed {
        xybrid_sdk::set_api_key(key.key.expose());
    }
}

/// Where `--observability` sends traces with `key`.
///
/// An explicit key (flag or environment) keeps the documented default and
/// any `--traces-endpoint`. A saved key only goes to its platform's pinned
/// ingest endpoint, or to a `--traces-endpoint` on that platform.
fn trace_endpoint(cli: &Cli, key: &ResolvedKey, host: Option<&PlatformHost>) -> Result<String> {
    match key.source {
        KeySource::Flag | KeySource::Env => Ok(cli
            .traces_endpoint
            .clone()
            .unwrap_or_else(|| XYBRID_TRACES_ENDPOINT.to_string())),
        KeySource::Credentials => {
            let host = host.context("a saved key always belongs to a parsed platform host")?;
            match cli.traces_endpoint.as_deref() {
                Some(endpoint) if host::trace_endpoint_allowed(host, endpoint) => {
                    Ok(endpoint.to_string())
                }
                Some(endpoint) => Err(anyhow::anyhow!(
                    "Refusing to send the key saved for {} to {endpoint}: a saved key only goes \
                     to its own platform's trace endpoint. Pass the key with --api-key to send \
                     it elsewhere on purpose.",
                    host.origin()
                )),
                None => host.traces_endpoint().map(str::to_string).ok_or_else(|| {
                    anyhow::anyhow!(
                        "The saved key belongs to the custom platform host {}; pass its OTLP \
                         trace URL with --traces-endpoint.",
                        host.origin()
                    )
                }),
            }
        }
    }
}

/// Select explicit trace export or standard OTLP environment configuration.
///
/// `--observability` needs a resolved key; its absence is an error, never a
/// silent no-op. Generic OTLP configuration never receives the Xybrid key.
fn observability_config(
    cli: &Cli,
    key: Option<&ResolvedKey>,
    host: Option<&PlatformHost>,
    environment: Option<OtlpConfig>,
) -> Result<Option<OtlpConfig>> {
    let mut config = if cli.observability {
        let key = key.context(
            "--observability needs a Xybrid key: pass --api-key, set XYBRID_API_KEY, or run \
             `xybrid connect` in this project first",
        )?;
        OtlpConfig::xybrid_traces_to(&trace_endpoint(cli, key, host)?, key.key.expose())
    } else {
        match environment {
            Some(config) => config,
            None => return Ok(None),
        }
    };
    if let Some(device_id) = &cli.device_id {
        config = config.with_resource_attribute(xybrid_sdk::observability::KeyValue::new(
            "device.id",
            device_id.clone(),
        ));
    }
    Ok(Some(config.with_service(
        "xybrid-cli",
        Some(env!("CARGO_PKG_VERSION").into()),
    )))
}

/// Start explicit or environment-configured export. Returns whether it runs.
fn init_observability(
    cli: &Cli,
    key: Option<&ResolvedKey>,
    host: Option<&PlatformHost>,
    store_warning: Option<&str>,
) -> Result<bool> {
    if platform::otel_sdk_disabled() {
        return Ok(false);
    }
    let config = observability_config(cli, key, host, OtlpConfig::from_env()).map_err(|error| {
        match store_warning {
            Some(warning) => error.context(warning.to_string()),
            None => error,
        }
    })?;
    let Some(config) = config else {
        return Ok(false);
    };
    match xybrid_sdk::init_observability(ObservabilityConfig::otlp(config)) {
        Ok(()) => Ok(true),
        Err(error) => {
            ui::warning(&format!("Observability was not started: {error}"));
            Ok(false)
        }
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
            input_images,
            choices,
            voice,
            output,
            target,
            trace,
            show_reasoning,
            max_tokens,
            trace_export,
            speculative_cloud,
        } => {
            if trace {
                tracing_viz::reset_collector();
            }

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
                    trace_export.as_ref(),
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
                    trace_export.as_ref(),
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
                    trace_export.as_ref(),
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
                    trace_export.as_ref(),
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
                &input_images,
                voice.as_deref(),
                output.as_ref(),
                target.as_deref(),
                trace,
                show_reasoning,
                max_tokens,
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
        Commands::Telemetry { command } => commands::telemetry::handle_telemetry_command(command),
        Commands::Connect(_) | Commands::Auth { .. } => {
            unreachable!("connect and auth are dispatched before observability setup")
        }
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

    fn parse_matches(args: &[&str]) -> Result<(Cli, ArgMatches), clap::Error> {
        // Keep authentication and device attribution tests independent of the
        // developer's environment without mutating process-wide variables.
        let matches = Cli::command()
            .mut_arg("api_key", |arg| arg.env(None::<&str>))
            .mut_arg("device_id", |arg| arg.env(None::<&str>))
            .mut_arg("platform_url", |arg| arg.env(None::<&str>))
            .try_get_matches_from(args)?;
        Ok((Cli::from_arg_matches(&matches)?, matches))
    }

    fn parse_observability_args(args: &[&str]) -> Result<Cli, clap::Error> {
        parse_matches(args).map(|(cli, _)| cli)
    }

    /// The explicit key the resolver sees for `cli` (flag only: env is off).
    fn flag_key(cli: &Cli) -> Option<ResolvedKey> {
        resolve::explicit(cli.api_key.as_deref(), true)
    }

    /// A key as if `xybrid connect` had saved it for `host`.
    fn saved_key(host: &PlatformHost) -> ResolvedKey {
        ResolvedKey {
            key: platform::credentials::Secret::new("sk_live_saved"),
            source: KeySource::Credentials,
            credential: Some(platform::credentials::Credential {
                platform_origin: host.origin().to_string(),
                repository_root: PathBuf::from("/repo"),
                api_key: platform::credentials::Secret::new("sk_live_saved"),
                scope: Some("client".into()),
                organization: None,
                project: None,
                key_expires_at: None,
                claim_expires_at: None,
                claim_token: None,
                source: None,
                console_origin: None,
                env_file: None,
                claimed: false,
                terms_url: None,
                privacy_url: None,
                created_at: chrono::Utc::now(),
            }),
        }
    }

    #[test]
    fn cloud_authentication_alone_does_not_enable_export() {
        let cli = parse_observability_args(&["xybrid", "--api-key", "cloud-key", "trace"])
            .expect("cloud authentication is optional");
        assert!(
            observability_config(&cli, flag_key(&cli).as_ref(), None, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn api_key_source_distinguishes_flag_from_environment() {
        for args in [
            vec!["xybrid", "--api-key", "k", "trace"],
            vec!["xybrid", "trace", "--api-key", "k"],
            vec!["xybrid", "auth", "status", "--api-key", "k"],
        ] {
            let (cli, matches) = parse_matches(&args).unwrap();
            assert_eq!(
                explicit_key(&cli, &matches).unwrap().source,
                KeySource::Flag,
                "{args:?}"
            );
        }
        let matches = Cli::command()
            .mut_arg("api_key", |arg| arg.env("XYBRID_TEST_ONLY_UNSET_VARIABLE"))
            .try_get_matches_from(["xybrid", "trace"])
            .unwrap();
        let cli = Cli::from_arg_matches(&matches).unwrap();
        assert!(explicit_key(&cli, &matches).is_none());
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
            "trace",
        ])
        .expect("explicit tracing should parse");
        let config = observability_config(
            &cli,
            flag_key(&cli).as_ref(),
            None,
            Some(OtlpConfig::collector("http://localhost:4318")),
        )
        .unwrap()
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
            "trace",
        ])
        .expect("custom tracing should parse");
        let config = observability_config(&cli, flag_key(&cli).as_ref(), None, None)
            .unwrap()
            .expect("trace configuration");
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
            vec!["xybrid", "trace"],
            vec!["xybrid", "--api-key", "cloud-key", "trace"],
        ] {
            let cli = parse_observability_args(&args).expect("cloud credentials are optional");
            let collector = OtlpConfig::collector("http://localhost:4318");
            let config =
                observability_config(&cli, flag_key(&cli).as_ref(), None, Some(collector.clone()))
                    .unwrap()
                    .expect("OTLP opt-in");
            assert_eq!(config.traces, collector.traces);
            assert_eq!(config.metrics, collector.metrics);
            assert_eq!(config.logs, collector.logs);
        }
    }

    #[test]
    fn trace_endpoint_requires_explicit_opt_in() {
        let Err(error) = parse_observability_args(&[
            "xybrid",
            "--api-key",
            "cloud-key",
            "--traces-endpoint",
            "http://localhost:4318/v1/traces",
            "trace",
        ]) else {
            panic!("--traces-endpoint without --observability should be rejected");
        };
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn observability_without_a_resolved_key_is_an_actionable_error() {
        // Parsing no longer requires --api-key: a saved key must work too.
        let cli = parse_observability_args(&["xybrid", "--observability", "trace"])
            .expect("the key is validated after resolution");
        let error = observability_config(&cli, None, None, None).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("xybrid connect"), "{message}");
        assert!(message.contains("--api-key"), "{message}");
    }

    #[test]
    fn a_saved_key_goes_only_to_its_platforms_trace_endpoint() {
        let prod = PlatformHost::parse("https://api.xybrid.dev").unwrap();
        let staging = PlatformHost::parse("https://staging.api.xybrid.dev").unwrap();
        let cli = parse_observability_args(&["xybrid", "--observability", "trace"]).unwrap();
        for (host, endpoint) in [
            (&prod, XYBRID_TRACES_ENDPOINT),
            (&staging, "https://staging.ingest.xybrid.dev/v1/traces"),
        ] {
            let config = observability_config(&cli, Some(&saved_key(host)), Some(host), None)
                .unwrap()
                .unwrap();
            let traces = config.traces.unwrap();
            assert_eq!(traces.endpoint, endpoint);
            assert_eq!(
                traces.headers,
                [("Authorization".into(), "Bearer sk_live_saved".into())]
            );
        }

        let elsewhere = parse_observability_args(&[
            "xybrid",
            "--observability",
            "--traces-endpoint",
            "https://collector.example.test/v1/traces",
            "trace",
        ])
        .unwrap();
        let error = observability_config(&elsewhere, Some(&saved_key(&prod)), Some(&prod), None)
            .unwrap_err();
        assert!(format!("{error:#}").contains("Refusing"), "{error:#}");

        let local = PlatformHost::parse("http://localhost:3001").unwrap();
        let error =
            observability_config(&cli, Some(&saved_key(&local)), Some(&local), None).unwrap_err();
        assert!(
            format!("{error:#}").contains("--traces-endpoint"),
            "{error:#}"
        );
        let local_cli = parse_observability_args(&[
            "xybrid",
            "--observability",
            "--traces-endpoint",
            "http://localhost:8081/v1/traces",
            "trace",
        ])
        .unwrap();
        let config = observability_config(&local_cli, Some(&saved_key(&local)), Some(&local), None)
            .unwrap()
            .unwrap();
        assert_eq!(
            config.traces.unwrap().endpoint,
            "http://localhost:8081/v1/traces"
        );
    }

    #[test]
    fn generic_otlp_configuration_never_receives_a_saved_key() {
        let prod = PlatformHost::parse("https://api.xybrid.dev").unwrap();
        let cli = parse_observability_args(&["xybrid", "trace"]).unwrap();
        let collector = OtlpConfig::collector("http://localhost:4318");
        let config = observability_config(
            &cli,
            Some(&saved_key(&prod)),
            Some(&prod),
            Some(collector.clone()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(config.traces, collector.traces);
    }

    #[test]
    fn platform_commands_parse() {
        for args in [
            vec![
                "xybrid",
                "connect",
                "--agent",
                "claude-code",
                "--write-env",
                ".env",
                "--json",
            ],
            vec![
                "xybrid",
                "connect",
                "--new",
                "--allow-ci",
                "--project-name",
                "my app",
            ],
            vec![
                "xybrid",
                "--platform-url",
                "http://localhost:3001",
                "connect",
                "--allow-custom-host",
                "--write-env",
                "app.env",
                "--env-name",
                "EXPO_PUBLIC_XYBRID_API_KEY",
            ],
            vec!["xybrid", "auth", "status", "--json", "--show-claim-url"],
            vec!["xybrid", "auth", "claim", "--no-browser"],
            vec!["xybrid", "auth", "logout"],
            vec!["xybrid", "auth", "signup", "--json"],
        ] {
            parse_observability_args(&args).unwrap_or_else(|err| panic!("{args:?}: {err}"));
        }
        let Err(error) = parse_observability_args(&["xybrid", "connect", "--env-name", "X"]) else {
            panic!("--env-name needs --write-env");
        };
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
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

//! Xybrid CLI - Command-line interface for running hybrid cloud-edge AI inference pipelines.
//!
//! This binary provides a `run` subcommand that loads pipeline configuration
//! and executes it using the xybrid-core orchestrator.
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
//! | `models` | Manage models from the registry |
//! | `cache` | Manage the local model cache |
//! | `prepare` | Parse and validate a pipeline configuration |
//! | `plan` | Show execution plan for a pipeline |
//! | `fetch` | Pre-download models from the registry |
//! | `trace` | View and analyze telemetry sessions |
//! | `pack` | Create a model bundle |
//! | `deploy` | Deploy a bundle to the registry |

#[macro_use]
extern crate lazy_static;

mod commands;
mod tracing_viz;

// Import utility functions from commands module
// Note: Some functions (format_timestamp, format_system_time, truncate) are kept
// locally due to chrono dependencies and slightly different implementations
use commands::{display_stage_name, format_params, format_size, dir_size_bytes};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use xybrid_core::bundler::XyBundle;
use xybrid_core::context::{DeviceMetrics, StageDescriptor};
use xybrid_core::device_adapter::{DeviceAdapter, LocalDeviceAdapter};
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::orchestrator::policy_engine::PolicyEngine;
use xybrid_core::orchestrator::routing_engine::{LocalAvailability, RoutingEngine};
use xybrid_core::pipeline_config::PipelineConfig;
use xybrid_core::target::{Platform, TargetResolver};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_sdk::registry_client::RegistryClient;

/// Xybrid CLI - Hybrid Cloud-Edge AI Inference Pipeline Runner
#[derive(Parser)]
#[command(name = "xybrid")]
#[command(about = "Xybrid CLI - Run hybrid cloud-edge AI inference pipelines", long_about = None)]
struct Cli {
    /// Platform API key for telemetry (can also be set via XYBRID_API_KEY env var)
    #[arg(long, global = true, env = "XYBRID_API_KEY")]
    api_key: Option<String>,

    /// Platform API endpoint for telemetry (default: https://api.xybrid.dev)
    #[arg(long, global = true, env = "XYBRID_PLATFORM_URL", default_value = "https://api.xybrid.dev")]
    platform_url: String,

    /// Device ID for telemetry attribution
    #[arg(long, global = true, env = "XYBRID_DEVICE_ID")]
    device_id: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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
    /// Pre-download models from the registry
    Fetch {
        /// Path to pipeline configuration file (downloads all models)
        #[arg(value_name = "FILE", conflicts_with = "model")]
        config: Option<PathBuf>,

        /// Model ID to fetch (e.g., "kokoro-82m")
        #[arg(short, long, value_name = "ID")]
        model: Option<String>,

        /// Target platform (auto-detected if not specified)
        #[arg(short, long, value_name = "PLATFORM")]
        platform: Option<String>,
    },
    /// Manage the local model cache
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
    /// Run a pipeline from a configuration file or predefined pipeline name
    Run {
        /// Path to the pipeline configuration file (YAML)
        #[arg(short, long, value_name = "FILE", conflicts_with_all = ["pipeline", "bundle"])]
        config: Option<PathBuf>,

        /// Predefined pipeline name (e.g., "hiiipe")
        #[arg(short, long, value_name = "NAME", conflicts_with_all = ["config", "bundle"])]
        pipeline: Option<String>,

        /// Path to a .xyb bundle file for direct execution
        #[arg(short, long, value_name = "FILE", conflicts_with_all = ["config", "pipeline"])]
        bundle: Option<PathBuf>,

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
}

/// Subcommands for `xybrid models`
#[derive(Subcommand)]
enum ModelsCommand {
    /// List all available models in the registry
    List,
    /// Search for models by name or task
    Search {
        /// Search query (matches model ID, family, task, or description)
        #[arg(value_name = "QUERY")]
        query: String,
    },
    /// Show details about a specific model
    Info {
        /// Model ID (e.g., "kokoro-82m")
        #[arg(value_name = "ID")]
        model_id: String,
    },
}

/// Subcommands for `xybrid cache`
#[derive(Subcommand)]
enum CacheCommand {
    /// List all cached models
    List,
    /// Show cache statistics
    Status,
    /// Clear cached models
    Clear {
        /// Model ID to clear (clears all if not specified)
        #[arg(value_name = "ID")]
        model_id: Option<String>,
    },
}

// PipelineConfig is now imported from xybrid_core::pipeline_config
// This unified schema makes `stages` the only required field.
// Metrics are auto-detected at runtime via LocalDeviceAdapter.
// Input/output types are inferred from model_metadata.json.

// display_stage_name is now in commands/utils.rs

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize platform telemetry if API key is provided
    let telemetry_enabled = init_telemetry(&cli);
    if telemetry_enabled {
        println!("📡 Telemetry enabled ({})", cli.platform_url);
    }

    // Run command and ensure telemetry is flushed on exit
    let result = run_command(cli);

    // Flush and shutdown telemetry
    if telemetry_enabled {
        xybrid_sdk::flush_platform_telemetry();
        xybrid_sdk::shutdown_platform_telemetry();
    }

    result
}

/// Initialize platform telemetry from CLI args
fn init_telemetry(cli: &Cli) -> bool {
    if let Some(ref api_key) = cli.api_key {
        // Detect platform
        let platform = Platform::detect().to_string();

        // Use hostname as device ID if not provided
        let device_id = cli.device_id.clone().unwrap_or_else(|| {
            hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "cli-unknown".to_string())
        });

        // Build telemetry config
        let mut config = xybrid_sdk::TelemetryConfig::new(&cli.platform_url, api_key);
        config = config.with_device(&device_id, &platform);
        config = config.with_app_version(env!("CARGO_PKG_VERSION"));

        xybrid_sdk::init_platform_telemetry(config);
        true
    } else {
        false
    }
}

fn run_command(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Models { command } => handle_models_command(command),
        Commands::Prepare { config } => handle_prepare_command(&config),
        Commands::Plan { config } => handle_plan_command(&config),
        Commands::Fetch { config, model, platform } => {
            if let Some(config_path) = config {
                handle_fetch_pipeline_command(&config_path, platform.as_deref())
            } else if let Some(model_id) = model {
                handle_fetch_command(&model_id, platform.as_deref())
            } else {
                Err(anyhow::anyhow!("Either a pipeline config file or --model <id> must be specified"))
            }
        },
        Commands::Cache { command } => handle_cache_command(command),
        Commands::Run {
            config,
            pipeline,
            bundle,
            dry_run,
            policy,
            input_audio,
            input_text,
            target,
            trace,
            trace_export,
        } => {
            // Reset trace collector for fresh trace
            if trace {
                tracing_viz::reset_collector();
            }

            // Handle direct bundle execution
            if let Some(bundle_path) = bundle {
                return run_bundle(&bundle_path, input_audio.as_ref(), input_text.as_deref(), dry_run, trace, trace_export.as_ref());
            }

            let config_path = if let Some(path) = config {
                path
            } else if let Some(pipeline_name) = pipeline {
                // Try to find the pipeline config in examples directory
                // Check for both .yml and .yaml extensions
                let mut base_dir =
                    std::env::current_dir().context("Failed to get current directory")?;

                // Try xybrid-cli/examples first
                base_dir.push("xybrid-cli");
                base_dir.push("examples");

                // Try .yml first (as specified), then .yaml
                let mut p = base_dir.clone();
                p.push(format!("{}.yml", pipeline_name));

                if !p.exists() {
                    p = base_dir.clone();
                    p.push(format!("{}.yaml", pipeline_name));
                }

                if !p.exists() {
                    // Try examples/ directory at root
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

                p
            } else {
                return Err(anyhow::anyhow!(
                    "Either --config, --pipeline, or --bundle must be specified"
                ));
            };
            run_pipeline(&config_path, dry_run, policy.as_ref(), input_audio.as_ref(), input_text.as_deref(), target.as_deref(), trace, trace_export.as_ref())
        }
        Commands::Trace {
            session,
            latest,
            export,
        } => {
            let session_id = if latest {
                find_latest_session()?
            } else {
                session
            };
            if session_id.is_none() && latest {
                return Err(anyhow::anyhow!(
                    "No sessions found. Use --session <id> to specify a session."
                ));
            }
            trace_session(session_id, export.as_deref())
        }
        Commands::Pack {
            name,
            version,
            target,
            path,
        } => pack_model(&name, &version, &target, path.as_deref()),
    }
}

/// Trace and analyze telemetry logs from a session.
fn trace_session(session: Option<String>, export_path: Option<&Path>) -> Result<()> {
    println!("📊 Xybrid Trace Analyzer");
    println!("{}", "=".repeat(60));
    println!();

    // Get traces directory path
    let traces_dir = get_traces_directory()?;

    if let Some(session_id) = &session {
        println!("🔍 Loading telemetry for session: {}", session_id);
        println!();

        let trace_file = traces_dir.join(format!("{}.log", session_id));

        if !trace_file.exists() {
            return Err(anyhow::anyhow!(
                "Session '{}' not found.\n  Looked in: {}",
                session_id,
                trace_file.display()
            ));
        }

        // Read and parse telemetry logs
        let entries = read_telemetry_log(&trace_file)?;

        if entries.is_empty() {
            println!("⚠️  No telemetry entries found in session.");
            return Ok(());
        }

        // Display telemetry in table format (with colors)
        display_telemetry_table(&entries);

        // Summary statistics (with colors)
        display_summary(&entries);

        // Export to JSON if requested
        if let Some(export_path) = export_path {
            export_trace_summary(&entries, export_path)?;
            println!("💾 Trace summary exported to: {}", export_path.display());
            println!();
        }
    } else {
        // List available sessions
        list_sessions(&traces_dir)?;
    }

    Ok(())
}

/// Package model artifacts from ./models/<name>/ (or custom path) into ./dist/<name>-<version>-<target>.xyb using XyBundle.
fn pack_model(name: &str, version: &str, target: &str, custom_path: Option<&Path>) -> Result<()> {
    println!("📦 Xybrid Packager");
    println!("{}", "=".repeat(60));
    println!("Model: {}", name);
    println!("Version: {}", version);
    println!("Target: {}", target);
    println!();

    // Resolve input directory: use custom path if provided, otherwise default to ./models/<name>
    let models_dir = if let Some(custom_path) = custom_path {
        custom_path.to_path_buf()
    } else {
        let mut dir = std::env::current_dir().context("Failed to get current directory")?;
        dir.push("models");
        dir.push(name);
        dir
    };

    if !models_dir.exists() || !models_dir.is_dir() {
        return Err(anyhow::anyhow!(
            "Model directory not found: {}",
            models_dir.display()
        ));
    }

    println!("📂 Source: {}", models_dir.display());

    let mut dist_dir = std::env::current_dir().context("Failed to get current directory")?;
    dist_dir.push("dist");
    if !dist_dir.exists() {
        fs::create_dir_all(&dist_dir).context("Failed to create dist directory")?;
    }

    // New naming convention: {name}-{version}-{target}.xyb
    let out_path = dist_dir.join(format!("{}-{}-{}.xyb", name, version, target));

    // Create bundle
    let mut bundle = XyBundle::new(name, version, target);

    // Check if model_metadata.json exists and prioritize adding it first
    let metadata_path = models_dir.join("model_metadata.json");
    let has_metadata = metadata_path.exists();

    if has_metadata {
        println!("🔍 Found model_metadata.json - including in bundle");
        // Note: metadata will be added automatically during file walk below
    }

    // Walk model directory (non-hidden files), add all regular files
    let mut added = 0usize;
    let mut duplicates: Vec<String> = Vec::new();

    fn visit_dir(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
        for entry in
            fs::read_dir(dir).with_context(|| format!("Failed to read dir: {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dir(&path, files)?;
            } else if path.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut files_to_add: Vec<std::path::PathBuf> = Vec::new();
    visit_dir(&models_dir, &mut files_to_add)?;

    // Track seen basenames to avoid collisions (bundler stores by filename)
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();

    for file in files_to_add {
        let fname = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if fname.is_empty() {
            continue;
        }
        if seen.contains(&fname) {
            duplicates.push(fname);
            continue;
        }
        seen.insert(fname);
        bundle
            .add_file(&file)
            .with_context(|| format!("Failed to add file: {}", file.display()))?;
        added += 1;
    }

    if !duplicates.is_empty() {
        println!("⚠️  Skipped duplicate filenames (consider flattening tree):");
        duplicates.sort();
        duplicates.dedup();
        for d in duplicates {
            println!("   - {}", d);
        }
        println!();
    }

    if added == 0 {
        return Err(anyhow::anyhow!(
            "No files found to add in {}",
            models_dir.display()
        ));
    }

    // Write output bundle
    bundle
        .write(&out_path)
        .with_context(|| format!("Failed to write bundle: {}", out_path.display()))?;

    // Log metadata
    println!("✅ Bundle created: {}", out_path.display());
    println!("   Model ID: {}", bundle.manifest().model_id);
    println!("   Version:  {}", bundle.manifest().version);
    println!("   Target:   {}", bundle.manifest().target);
    println!("   Files:    {}", bundle.manifest().files.len());
    println!("   Hash:     {}", bundle.manifest().hash);

    if bundle.manifest().has_metadata {
        println!("   Metadata: ✅ Included (metadata-driven execution enabled)");
    } else {
        println!("   Metadata: ⚠️  Not found (consider adding model_metadata.json)");
    }
    println!();

    Ok(())
}


/// Get the traces directory path (~/.xybrid/traces/).
fn get_traces_directory() -> Result<PathBuf> {
    let mut path =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
    path.push(".xybrid");
    path.push("traces");

    // Create directory if it doesn't exist
    if !path.exists() {
        fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create traces directory: {}", path.display()))?;
    }

    Ok(path)
}

/// Find the latest session ID from the traces directory.
fn find_latest_session() -> Result<Option<String>> {
    let traces_dir = get_traces_directory()?;

    if !traces_dir.exists() {
        return Ok(None);
    }

    let entries = fs::read_dir(&traces_dir)
        .with_context(|| format!("Failed to read traces directory: {}", traces_dir.display()))?;

    let mut sessions: Vec<(String, std::time::SystemTime)> = Vec::new();

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "log" {
                    if let Some(stem) = path.file_stem() {
                        if let Some(session_id) = stem.to_str() {
                            if let Ok(metadata) = entry.metadata() {
                                if let Ok(modified) = metadata.modified() {
                                    sessions.push((session_id.to_string(), modified));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if sessions.is_empty() {
        return Ok(None);
    }

    // Sort by modification time (newest first)
    sessions.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(sessions.first().map(|(id, _)| id.clone()))
}

/// Read and parse telemetry log entries from a file.
fn read_telemetry_log(file_path: &Path) -> Result<Vec<TelemetryLogEntry>> {
    let file = fs::File::open(file_path)
        .with_context(|| format!("Failed to open trace file: {}", file_path.display()))?;

    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("Failed to read line {} from trace file", line_num + 1))?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Parse JSON line
        // Skip routing decision JSON from routing engine (not in telemetry format)
        if line.trim().starts_with("{\"stage\":") && !line.contains("\"event\"") {
            // This is the routing engine's JSON output, skip it
            continue;
        }

        match serde_json::from_str::<Value>(&line) {
            Ok(json) => {
                // Only process entries that have the telemetry structure (with "event" field)
                if json.get("event").is_some() {
                    if let Ok(entry) = parse_telemetry_entry(&json) {
                        entries.push(entry);
                    }
                }
            }
            Err(_) => {
                // Silently skip non-JSON lines (like routing engine JSON)
            }
        }
    }

    // Sort by timestamp
    entries.sort_by_key(|e| e.timestamp);

    Ok(entries)
}

/// Parsed telemetry log entry.
#[derive(Debug, Clone)]
struct TelemetryLogEntry {
    timestamp: u64,
    severity: String,
    event: String,
    #[allow(dead_code)] // Reserved for future use in detailed views
    message: String,
    stage: Option<String>,
    target: Option<String>,
    latency_ms: Option<u32>,
    allowed: Option<bool>,
    reason: Option<String>,
}

/// Parse a JSON value into a TelemetryLogEntry.
fn parse_telemetry_entry(json: &Value) -> Result<TelemetryLogEntry> {
    let timestamp = json["timestamp"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("Missing or invalid timestamp"))?;

    let severity = json["severity"].as_str().unwrap_or("UNKNOWN").to_string();

    let event = json["event"].as_str().unwrap_or("unknown").to_string();

    let message = json["message"].as_str().unwrap_or("").to_string();

    // Extract attributes
    let attrs = json.get("attributes").and_then(|a| a.as_object());

    let stage = attrs
        .and_then(|a| a.get("stage"))
        .and_then(|s| s.as_str())
        .map(|s| s.to_string());

    let target = attrs
        .and_then(|a| a.get("target"))
        .and_then(|t| t.as_str())
        .map(|t| t.to_string());

    let latency_ms = attrs
        .and_then(|a| a.get("latency_ms"))
        .and_then(|l| l.as_u64())
        .map(|l| l as u32);

    let allowed = attrs
        .and_then(|a| a.get("allowed"))
        .and_then(|a| a.as_bool());

    let reason = attrs
        .and_then(|a| a.get("reason"))
        .and_then(|r| r.as_str())
        .map(|r| r.to_string());

    Ok(TelemetryLogEntry {
        timestamp,
        severity,
        event,
        message,
        stage,
        target,
        latency_ms,
        allowed,
        reason,
    })
}

/// Display telemetry entries in a table format with color codes.
fn display_telemetry_table(entries: &[TelemetryLogEntry]) {
    println!("{}", "📋 Telemetry Events:".bold().cyan());
    println!("{}", "=".repeat(100).bright_black());

    // Table header with colors
    println!(
        "{:<12} {:<8} {:<20} {:<25} {:<15} {:<20}",
        "Timestamp".bold(),
        "Severity".bold(),
        "Event".bold(),
        "Stage".bold(),
        "Target".bold(),
        "Details".bold()
    );
    println!("{}", "-".repeat(100).bright_black());

    // Calculate relative timestamps
    let first_timestamp = entries.first().map(|e| e.timestamp).unwrap_or(0);

    for entry in entries {
        let relative_timestamp = if entry.timestamp >= first_timestamp {
            entry.timestamp - first_timestamp
        } else {
            0
        };
        let timestamp_str = format_timestamp(relative_timestamp);

        // Color-coded severity
        let (severity_icon, severity_color) = match entry.severity.as_str() {
            "INFO" => ("ℹ️", entry.severity.as_str().bright_green()),
            "DEBUG" => ("🔍", entry.severity.as_str().bright_blue()),
            "WARN" => ("⚠️", entry.severity.as_str().bright_yellow()),
            "ERROR" => ("❌", entry.severity.as_str().bright_red()),
            _ => ("•", entry.severity.as_str().white()),
        };

        let stage = entry.stage.as_deref().unwrap_or("-");
        let target = entry.target.as_deref().unwrap_or("-");

        // Color-code target
        let target_colored = match target {
            "local" => target.bright_green(),
            "cloud" => target.bright_blue(),
            s if s.starts_with("fallback") => target.bright_yellow(),
            _ => target.white(),
        };

        // Build details column with colors
        let mut details = String::new();
        if let Some(latency) = entry.latency_ms {
            let latency_str = format!("{}ms", latency);
            // Color-code latency (green for fast, yellow for medium, red for slow)
            let latency_colored = if latency < 50 {
                latency_str.bright_green()
            } else if latency < 200 {
                latency_str.bright_yellow()
            } else {
                latency_str.bright_red()
            };
            details.push_str(&format!("{} ", latency_colored));
        }
        if let Some(allowed) = entry.allowed {
            let policy_str = format!("Policy: {}", if allowed { "✓" } else { "✗" });
            let policy_colored = if allowed {
                policy_str.bright_green()
            } else {
                policy_str.bright_red()
            };
            details.push_str(&format!("{} ", policy_colored));
        }
        if let Some(ref reason) = entry.reason {
            details.push_str(&format!("Reason: {}", truncate(reason, 30).bright_black()));
        }
        if details.is_empty() {
            details.push_str(&"-".bright_black().to_string());
        }

        // Truncate event for display and color-code
        let event_display = truncate(&entry.event, 20);
        let event_colored_display = match entry.event.as_str() {
            "stage_complete" => event_display.bright_green(),
            "stage_start" => event_display.bright_cyan(),
            "policy_evaluation" => event_display.bright_magenta(),
            "routing_decision" => event_display.bright_blue(),
            "execution_complete" => event_display.green(),
            "execution_start" => event_display.cyan(),
            _ => event_display.white(),
        };

        println!(
            "{:<12} {:<8} {:<20} {:<25} {:<15} {:<20}",
            timestamp_str.bright_black(),
            format!("{} {}", severity_icon, severity_color),
            event_colored_display,
            truncate(stage, 25).cyan(),
            target_colored,
            details
        );
    }

    println!("{}", "=".repeat(100).bright_black());
    println!();
}

/// Display summary statistics with color codes.
fn display_summary(entries: &[TelemetryLogEntry]) {
    println!("{}", "📊 Summary Statistics:".bold().cyan());
    println!("{}", "=".repeat(60).bright_black());
    println!();

    let stage_completions: Vec<_> = entries
        .iter()
        .filter(|e| e.event == "stage_complete")
        .collect();

    if !stage_completions.is_empty() {
        println!("{}", "Stage Completions:".bold());
        for entry in &stage_completions {
            let stage = entry.stage.as_deref().unwrap_or("unknown");
            let target = entry.target.as_deref().unwrap_or("unknown");
            let latency = entry
                .latency_ms
                .map(|l| format!("{}ms", l))
                .unwrap_or_else(|| "N/A".to_string());

            let target_colored = match target {
                "local" => target.bright_green(),
                "cloud" => target.bright_blue(),
                _ => target.white(),
            };

            let latency_colored = if let Some(lat) = entry.latency_ms {
                if lat < 50 {
                    latency.bright_green()
                } else if lat < 200 {
                    latency.bright_yellow()
                } else {
                    latency.bright_red()
                }
            } else {
                latency.white()
            };

            println!(
                "  {} {} → {} ({})",
                "•".bright_cyan(),
                stage.cyan(),
                target_colored,
                latency_colored
            );
        }
        println!();
    }

    // Policy evaluations
    let policy_evals: Vec<_> = entries
        .iter()
        .filter(|e| e.event == "policy_evaluation")
        .collect();

    if !policy_evals.is_empty() {
        let allowed_count = policy_evals
            .iter()
            .filter(|e| e.allowed == Some(true))
            .count();
        let denied_count = policy_evals.len() - allowed_count;
        println!("{}", "Policy Evaluations:".bold());
        println!(
            "  {} {}",
            "•".bright_cyan(),
            format!("Allowed: {}", allowed_count.to_string().bright_green())
        );
        println!(
            "  {} {}",
            "•".bright_cyan(),
            format!("Denied: {}", denied_count.to_string().bright_red())
        );
        println!();
    }

    // Routing decisions
    let routing_decisions: Vec<_> = entries
        .iter()
        .filter(|e| e.event == "routing_decision")
        .collect();

    if !routing_decisions.is_empty() {
        println!("{}", "Routing Decisions:".bold());
        for entry in &routing_decisions {
            let stage = entry.stage.as_deref().unwrap_or("unknown");
            let target = entry.target.as_deref().unwrap_or("unknown");
            let reason = entry.reason.as_deref().unwrap_or("N/A");

            let target_colored = match target {
                "local" => target.bright_green(),
                "cloud" => target.bright_blue(),
                _ => target.white(),
            };

            println!(
                "  {} {} → {} ({})",
                "•".bright_cyan(),
                stage.cyan(),
                target_colored,
                truncate(reason, 40).bright_black()
            );
        }
        println!();
    }

    println!(
        "{} {}",
        "Total Events:".bold(),
        entries.len().to_string().bright_cyan()
    );
}

/// Export trace summary to JSON file.
fn export_trace_summary(entries: &[TelemetryLogEntry], export_path: &Path) -> Result<()> {
    // Build summary statistics
    let stage_completions: Vec<_> = entries
        .iter()
        .filter(|e| e.event == "stage_complete")
        .map(|e| {
            json!({
                "stage": e.stage,
                "target": e.target,
                "latency_ms": e.latency_ms,
            })
        })
        .collect();

    let policy_evals: Vec<_> = entries
        .iter()
        .filter(|e| e.event == "policy_evaluation")
        .map(|e| {
            json!({
                "stage": e.stage,
                "allowed": e.allowed,
                "reason": e.reason,
            })
        })
        .collect();

    let routing_decisions: Vec<_> = entries
        .iter()
        .filter(|e| e.event == "routing_decision")
        .map(|e| {
            json!({
                "stage": e.stage,
                "target": e.target,
                "reason": e.reason,
            })
        })
        .collect();

    // Count by severity
    let severity_counts: HashMap<String, usize> =
        entries.iter().fold(HashMap::new(), |mut acc, e| {
            *acc.entry(e.severity.clone()).or_insert(0) += 1;
            acc
        });

    // Count by event type
    let event_counts: HashMap<String, usize> = entries.iter().fold(HashMap::new(), |mut acc, e| {
        *acc.entry(e.event.clone()).or_insert(0) += 1;
        acc
    });

    // Calculate total latency
    let total_latency: u32 = entries.iter().filter_map(|e| e.latency_ms).sum();

    // Build summary JSON
    let summary = json!({
        "session": {
            "total_events": entries.len(),
            "total_latency_ms": total_latency,
            "first_timestamp": entries.first().map(|e| e.timestamp),
            "last_timestamp": entries.last().map(|e| e.timestamp),
        },
        "statistics": {
            "severity_counts": severity_counts,
            "event_counts": event_counts,
            "stage_completions": stage_completions.len(),
            "policy_evaluations": policy_evals.len(),
            "routing_decisions": routing_decisions.len(),
        },
        "stage_completions": stage_completions,
        "policy_evaluations": policy_evals,
        "routing_decisions": routing_decisions,
        "events": entries.iter().map(|e| json!({
            "timestamp": e.timestamp,
            "severity": e.severity,
            "event": e.event,
            "stage": e.stage,
            "target": e.target,
            "latency_ms": e.latency_ms,
            "allowed": e.allowed,
            "reason": e.reason,
        })).collect::<Vec<_>>(),
    });

    // Write to file
    let json_str = serde_json::to_string_pretty(&summary)?;
    let mut file = fs::File::create(export_path)
        .with_context(|| format!("Failed to create export file: {}", export_path.display()))?;
    file.write_all(json_str.as_bytes())
        .with_context(|| format!("Failed to write export file: {}", export_path.display()))?;

    Ok(())
}

/// List available sessions in the traces directory.
fn list_sessions(traces_dir: &Path) -> Result<()> {
    if !traces_dir.exists() {
        println!("ℹ️  No traces directory found at: {}", traces_dir.display());
        println!("   Sessions will be created when pipelines are executed.");
        return Ok(());
    }

    let entries = fs::read_dir(traces_dir)
        .with_context(|| format!("Failed to read traces directory: {}", traces_dir.display()))?;

    let mut sessions: Vec<(String, std::time::SystemTime)> = Vec::new();

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "log" {
                    if let Some(stem) = path.file_stem() {
                        if let Some(session_id) = stem.to_str() {
                            if let Ok(metadata) = entry.metadata() {
                                if let Ok(modified) = metadata.modified() {
                                    sessions.push((session_id.to_string(), modified));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if sessions.is_empty() {
        println!("ℹ️  No sessions found.");
        println!("   Sessions will be created when pipelines are executed.");
        return Ok(());
    }

    // Sort by modification time (newest first)
    sessions.sort_by(|a, b| b.1.cmp(&a.1));

    println!("📋 Available Sessions:");
    println!("{}", "=".repeat(60));
    println!();

    for (session_id, modified) in sessions {
        let time_str = format_system_time(modified);
        println!("  • {} (last modified: {})", session_id, time_str);
    }

    println!();
    println!("Usage: xybrid trace --session <session-id>");

    Ok(())
}

/// Format timestamp (milliseconds since epoch) as readable time.
///
/// Converts to relative time since first event for easier reading.
fn format_timestamp(ts_ms: u64) -> String {
    // For relative display, show as seconds with milliseconds
    let total_secs = ts_ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    let millis = ts_ms % 1000;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
    } else if mins > 0 {
        format!("{:02}:{:02}.{:03}", mins, secs, millis)
    } else {
        format!("{}.{:03}s", secs, millis)
    }
}

/// Format system time as readable string.
fn format_system_time(time: std::time::SystemTime) -> String {
    use std::time::UNIX_EPOCH;

    if let Ok(duration) = time.duration_since(UNIX_EPOCH) {
        let secs = duration.as_secs();
        let datetime = chrono::DateTime::<chrono::Utc>::from_timestamp(secs as i64, 0);

        if let Some(dt) = datetime {
            dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
        } else {
            "unknown".to_string()
        }
    } else {
        "unknown".to_string()
    }
}

/// Truncate string to max length with ellipsis.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Run a pipeline from a configuration file.
fn run_pipeline(
    config_path: &PathBuf,
    dry_run: bool,
    policy_path: Option<&PathBuf>,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    target: Option<&str>,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    // Start pipeline trace span
    let _pipeline_span = if trace_enabled {
        Some(tracing_viz::SpanGuard::new("pipeline_execution"))
    } else {
        None
    };
    // Load and parse configuration file
    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    println!("🚀 Xybrid Pipeline Runner");
    if let Some(name) = &config.name {
        println!("📋 Pipeline: {}\n", name);
    }

    // Build stage descriptors from config
    let stages: Vec<StageDescriptor> = config
        .stages
        .iter()
        .map(|stage| StageDescriptor::new(stage.model_id()))
        .collect();

    // Create input envelope based on CLI args
    let input = if let Some(audio_path) = input_audio {
        println!("📂 Loading audio file: {}", audio_path.display());
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        println!("   Loaded {} bytes", audio_bytes.len());
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else if let Some(text) = input_text {
        println!("📝 Input text: \"{}\"", text);
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    } else {
        // Default to empty text envelope
        Envelope::new(EnvelopeKind::Text(String::new()))
    };

    // Collect device metrics at runtime (not from YAML)
    let device_adapter = LocalDeviceAdapter::new();
    let metrics = device_adapter.collect_metrics();

    // Create availability function using registry client cache check
    let registry_client = RegistryClient::from_env().ok();
    let availability_fn = move |stage: &str| -> LocalAvailability {
        if let Some(ref client) = registry_client {
            let is_cached = client.is_cached(stage, None).unwrap_or(false);
            LocalAvailability::new(is_cached)
        } else {
            // Fallback: assume not available locally
            LocalAvailability::new(false)
        }
    };

    // Display configuration
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

    // Target resolution
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

    if dry_run {
        println!("🔎 Dry Run: Routing Simulation");
        println!("{}", "=".repeat(60));
        println!();

        // Simulate routing decisions without executing
        let mut routing_engine = xybrid_core::orchestrator::routing_engine::DefaultRoutingEngine::new();
        let policy_engine = xybrid_core::orchestrator::policy_engine::DefaultPolicyEngine::with_default_policy();

        let mut current_input = input.clone();

        for (i, stage) in stages.iter().enumerate() {
            println!("Stage {}: {}", i + 1, display_stage_name(&stage.name));

            // Evaluate policy
            let policy_result = policy_engine.evaluate(&stage.name, &current_input, &metrics);
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

            // Get availability
            let availability = availability_fn(&stage.name);

            // Make routing decision
            let routing_decision =
                routing_engine.decide(&stage.name, &metrics, &policy_result, &availability);
            println!(
                "   Routing: {} ({})",
                routing_decision.target, routing_decision.reason
            );

            // Simulate output
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
        return Ok(());
    }

    // Create orchestrator for actual execution
    let mut orchestrator = Orchestrator::new();

    // Bridge orchestrator events to telemetry (sends events to platform if API key is configured)
    xybrid_sdk::bridge_orchestrator_events(&orchestrator);

    // Load policy bundle if provided
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

    // Execute the pipeline
    println!("⚙️  Executing pipeline...");
    println!("{}", "=".repeat(60));
    println!();

    match orchestrator.execute_pipeline(&stages, &input, &metrics, &availability_fn) {
        Ok(results) => {
            println!();
            println!("📊 Pipeline Results:");
            println!("{}", "=".repeat(60));

            for (i, result) in results.iter().enumerate() {
                println!("\nStage {}: {}", i + 1, display_stage_name(&result.stage));
                println!("  Routing: {}", result.routing_decision.target);
                println!("  Reason: {}", result.routing_decision.reason);
                println!("  Latency: {}ms", result.latency_ms);
                println!("  Output Type: {}", result.output.kind_str());

                // Display actual output content
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

            println!();
            println!("{}", "=".repeat(60));
            println!("✨ Pipeline completed successfully!");

            // Output trace visualization if enabled
            if trace_enabled {
                println!("{}", tracing_viz::render_trace());

                // Export trace if requested
                if let Some(export_path) = trace_export {
                    let json = tracing_viz::GLOBAL_COLLECTOR.lock().unwrap().to_chrome_trace_json();
                    fs::write(export_path, json)
                        .with_context(|| format!("Failed to export trace to {}", export_path.display()))?;
                    println!("💾 Trace exported to: {}", export_path.display());
                }
            }

            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Pipeline execution failed: {}", e);
            Err(anyhow::anyhow!("Pipeline execution failed: {}", e))
        }
    }
}

/// Run inference directly on a .xyb bundle file
fn run_bundle(
    bundle_path: &Path,
    input_audio: Option<&PathBuf>,
    input_text: Option<&str>,
    dry_run: bool,
    trace_enabled: bool,
    trace_export: Option<&PathBuf>,
) -> Result<()> {
    // Reset and start bundle trace span
    if trace_enabled {
        tracing_viz::reset_collector();
    }
    let _bundle_span = if trace_enabled {
        Some(tracing_viz::SpanGuard::new("bundle_execution"))
    } else {
        None
    };

    // Generate trace ID for this execution
    let trace_id = uuid::Uuid::new_v4();
    xybrid_sdk::set_telemetry_pipeline_context(None, Some(trace_id));

    println!("🚀 Xybrid Bundle Runner");
    println!("📦 Bundle: {}\n", bundle_path.display());

    // Validate bundle exists
    if !bundle_path.exists() {
        return Err(anyhow::anyhow!(
            "Bundle file not found: {}",
            bundle_path.display()
        ));
    }

    // Load and extract bundle
    println!("📂 Loading bundle...");
    let bundle = XyBundle::load(bundle_path)
        .with_context(|| format!("Failed to load bundle: {}", bundle_path.display()))?;

    let manifest = bundle.manifest();
    println!("   Model ID: {}", manifest.model_id);
    println!("   Version: {}", manifest.version);
    println!("   Files: {:?}", manifest.files);
    println!();

    // Create temp directory for extraction
    let temp_dir = tempfile::tempdir()
        .context("Failed to create temp directory for bundle extraction")?;
    let extract_dir = temp_dir.path();

    // Extract bundle contents
    println!("📦 Extracting bundle to temp directory...");
    bundle.extract_to(extract_dir)
        .context("Failed to extract bundle")?;

    // Try to load model_metadata.json
    let metadata_path = extract_dir.join("model_metadata.json");
    if !metadata_path.exists() {
        return Err(anyhow::anyhow!(
            "Bundle does not contain model_metadata.json. Cannot execute without metadata."
        ));
    }

    let metadata_content = fs::read_to_string(&metadata_path)
        .context("Failed to read model_metadata.json")?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)
        .context("Failed to parse model_metadata.json")?;

    println!("📋 Model Metadata:");
    println!("   ID: {}", metadata.model_id);
    println!("   Version: {}", metadata.version);
    if let Some(desc) = &metadata.description {
        println!("   Description: {}", desc);
    }
    println!("   Preprocessing: {} steps", metadata.preprocessing.len());
    println!("   Postprocessing: {} steps", metadata.postprocessing.len());
    println!();

    // Emit PipelineStart telemetry event
    xybrid_sdk::publish_telemetry_event(xybrid_sdk::TelemetryEvent {
        event_type: "PipelineStart".to_string(),
        stage_name: Some(metadata.model_id.clone()),
        target: Some("local".to_string()),
        latency_ms: None,
        error: None,
        data: Some(serde_json::json!({
            "model_id": metadata.model_id,
            "version": metadata.version,
            "bundle_path": bundle_path.display().to_string()
        }).to_string()),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    });

    if dry_run {
        println!("🔎 Dry Run: Bundle inspection only");
        println!("{}", "=".repeat(60));
        println!();
        println!("Bundle is valid and ready for execution.");
        println!("Use without --dry-run to run inference.");
        return Ok(());
    }

    // Create input envelope based on provided input
    let input = if let Some(audio_path) = input_audio {
        println!("🎤 Loading audio file: {}", audio_path.display());
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        println!("   Loaded {} bytes", audio_bytes.len());
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else if let Some(text) = input_text {
        println!("📝 Input text: \"{}\"", text);
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    } else {
        return Err(anyhow::anyhow!(
            "No input provided. Use --input-audio <file> or --input-text <text>"
        ));
    };
    println!();

    // Create TemplateExecutor with the extracted bundle directory
    println!("⚙️  Running inference...");
    println!("{}", "=".repeat(60));

    let base_path = extract_dir.to_str().ok_or_else(|| {
        anyhow::anyhow!("Invalid extraction path")
    })?;

    let mut executor = TemplateExecutor::with_base_path(base_path);

    // Trace inference execution
    let _inference_span = if trace_enabled {
        let span = tracing_viz::SpanGuard::new(format!("inference:{}", metadata.model_id));
        tracing_viz::add_metadata("model_id", &metadata.model_id);
        tracing_viz::add_metadata("version", &metadata.version);
        Some(span)
    } else {
        None
    };

    let start_time = std::time::Instant::now();
    let output = executor.execute(&metadata, &input)
        .map_err(|e| anyhow::anyhow!("Inference failed: {:?}", e))?;
    let elapsed = start_time.elapsed();

    println!();
    println!("📊 Results:");
    println!("{}", "=".repeat(60));
    println!();
    println!("  Model: {} v{}", metadata.model_id, metadata.version);
    println!("  Latency: {:.2}ms", elapsed.as_millis());
    println!("  Output Type: {}", output.kind_str());

    // Display output content
    match &output.kind {
        EnvelopeKind::Text(text) => {
            if !text.is_empty() {
                println!();
                println!("  Output:");
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

    println!();
    println!("{}", "=".repeat(60));
    println!("✨ Inference completed successfully!");

    // Emit PipelineComplete telemetry event
    xybrid_sdk::publish_telemetry_event(xybrid_sdk::TelemetryEvent {
        event_type: "PipelineComplete".to_string(),
        stage_name: Some(metadata.model_id.clone()),
        target: Some("local".to_string()),
        latency_ms: Some(elapsed.as_millis() as u32),
        error: None,
        data: Some(serde_json::json!({
            "model_id": metadata.model_id,
            "version": metadata.version,
            "output_type": output.kind_str()
        }).to_string()),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    });

    // End spans and output trace visualization if enabled
    if trace_enabled {
        // Explicitly end the inference span
        drop(_inference_span);
        // Explicitly end the bundle span
        drop(_bundle_span);

        println!("{}", tracing_viz::render_trace());

        // Export trace if requested
        if let Some(export_path) = trace_export {
            let json = tracing_viz::GLOBAL_COLLECTOR.lock().unwrap().to_chrome_trace_json();
            fs::write(export_path, json)
                .with_context(|| format!("Failed to export trace to {}", export_path.display()))?;
            println!("💾 Trace exported to: {}", export_path.display());
        }
    }

    Ok(())
}

// ============================================================================
// Models Command Handlers
// ============================================================================

/// Handle `xybrid models` subcommands
fn handle_models_command(command: ModelsCommand) -> Result<()> {
    let client = RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    match command {
        ModelsCommand::List => {
            println!("📦 Xybrid Model Registry");
            println!("{}", "=".repeat(60));
            println!();

            let models = client.list_models()
                .context("Failed to list models from registry")?;

            if models.is_empty() {
                println!("ℹ️  No models found in registry.");
                return Ok(());
            }

            // Group by task
            use std::collections::BTreeMap;
            let mut by_task: BTreeMap<String, Vec<_>> = BTreeMap::new();
            for model in &models {
                by_task.entry(model.task.clone()).or_default().push(model);
            }

            for (task, task_models) in by_task {
                println!("{}", format!("📁 {}", task.to_uppercase()).cyan().bold());
                println!();

                for model in task_models {
                    let params_str = format_params(model.parameters);
                    println!("  {} {}", "•".bright_cyan(), model.id.cyan().bold());
                    println!("    {} {} | {} params",
                        model.family.bright_black(),
                        "|".bright_black(),
                        params_str.bright_black()
                    );
                    println!("    {}", model.description.bright_black());
                    if !model.variants.is_empty() {
                        println!("    Variants: {}", model.variants.join(", ").bright_green());
                    }
                    println!();
                }
            }

            println!("{}", "=".repeat(60));
            println!("Total: {} models", models.len());

            Ok(())
        }
        ModelsCommand::Search { query } => {
            println!("🔍 Searching for: {}", query.cyan().bold());
            println!("{}", "=".repeat(60));
            println!();

            let models = client.list_models()
                .context("Failed to list models from registry")?;

            let query_lower = query.to_lowercase();
            let matches: Vec<_> = models.iter()
                .filter(|m| {
                    m.id.to_lowercase().contains(&query_lower) ||
                    m.family.to_lowercase().contains(&query_lower) ||
                    m.task.to_lowercase().contains(&query_lower) ||
                    m.description.to_lowercase().contains(&query_lower)
                })
                .collect();

            if matches.is_empty() {
                println!("ℹ️  No models found matching '{}'", query);
                return Ok(());
            }

            for model in matches.iter() {
                let params_str = format_params(model.parameters);
                println!("  {} {}", "•".bright_cyan(), model.id.cyan().bold());
                println!("    {} | {} | {} params",
                    model.task.bright_magenta(),
                    model.family.bright_black(),
                    params_str.bright_black()
                );
                println!("    {}", model.description.bright_black());
                println!();
            }

            println!("{}", "=".repeat(60));
            println!("Found: {} models", matches.len());

            Ok(())
        }
        ModelsCommand::Info { model_id } => {
            println!("📋 Model Details: {}", model_id.cyan().bold());
            println!("{}", "=".repeat(60));
            println!();

            let model = client.get_model(&model_id)
                .context(format!("Failed to get model '{}'", model_id))?;

            println!("  ID:          {}", model.id.cyan().bold());
            println!("  Family:      {}", model.family);
            println!("  Task:        {}", model.task.bright_magenta());
            println!("  Parameters:  {}", format_params(model.parameters));
            println!("  Description: {}", model.description);
            println!();

            if let Some(default) = &model.default_variant {
                println!("  Default Variant: {}", default.bright_green());
            }

            if !model.variants.is_empty() {
                println!();
                println!("  {} Variants:", "📦".bright_cyan());
                for (name, info) in &model.variants {
                    let size_str = format_size(info.size_bytes);
                    println!("    {} {} ({}, {})",
                        "•".bright_cyan(),
                        name.bright_green(),
                        info.platform,
                        size_str.bright_black()
                    );
                    println!("      Format: {} | Quantization: {}",
                        info.format.bright_blue(),
                        info.quantization.bright_yellow()
                    );
                }
            }

            println!();
            println!("{}", "=".repeat(60));

            Ok(())
        }
    }
}

/// Handle `xybrid fetch` command
fn handle_fetch_command(model_id: &str, platform: Option<&str>) -> Result<()> {
    println!("📥 Fetching model: {}", model_id.cyan().bold());
    if let Some(p) = platform {
        println!("   Platform: {}", p);
    } else {
        println!("   Platform: auto-detect");
    }
    println!("{}", "=".repeat(60));
    println!();

    let client = RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    // First resolve to show what we're downloading
    let resolved = client.resolve(model_id, platform)
        .context(format!("Failed to resolve model '{}'", model_id))?;

    println!("📦 Resolved variant:");
    println!("   Repository: {}", resolved.hf_repo);
    println!("   File: {}", resolved.file);
    println!("   Size: {}", format_size(resolved.size_bytes).bright_cyan());
    println!("   Format: {} ({})", resolved.format, resolved.quantization);
    println!();

    // Check if already cached
    if client.is_cached(model_id, platform)
        .context("Failed to check cache status")? {
        println!("✅ Model is already cached and verified");
        let cache_path = client.get_cache_path(&resolved);
        println!("   Location: {}", cache_path.display());
        return Ok(());
    }

    // Download with progress bar
    println!("⬇️  Downloading...");
    println!();

    // Use Cell for interior mutability in closure
    use std::cell::Cell;
    let last_percent = Cell::new(0u32);
    let bar_width = 40;

    let bundle_path = client.fetch(model_id, platform, |progress| {
        let percent = (progress * 100.0) as u32;
        let prev = last_percent.get();

        // Only update if percentage changed
        if percent != prev {
            last_percent.set(percent);

            // Calculate filled portion
            let filled = (bar_width * percent / 100) as usize;
            let empty = bar_width as usize - filled;

            // Build progress bar: [████████████░░░░░░░░] 45%
            let bar = format!(
                "\r   [{}{}] {:>3}%",
                "█".repeat(filled),
                "░".repeat(empty),
                percent
            );
            print!("{}", bar);
            std::io::stdout().flush().ok();
        }
    }).context(format!("Failed to fetch model '{}'", model_id))?;

    println!();
    println!();
    println!("✅ Model downloaded successfully!");
    println!("   Location: {}", bundle_path.display());
    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}

/// Handle `xybrid cache` subcommands
fn handle_cache_command(command: CacheCommand) -> Result<()> {
    let mut client = RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    match command {
        CacheCommand::List => {
            println!("📦 Xybrid Model Cache");
            println!("{}", "=".repeat(60));
            println!();

            let stats = client.cache_stats()
                .context("Failed to get cache stats")?;

            println!("📂 Cache directory: {}", stats.cache_path.display());
            println!();

            if stats.model_count == 0 {
                println!("ℹ️  Cache is empty.");
                println!("   Use 'xybrid fetch --model <id>' to download models.");
                return Ok(());
            }

            // List cached models
            if stats.cache_path.exists() {
                for entry in fs::read_dir(&stats.cache_path)? {
                    let entry = entry?;
                    if entry.path().is_dir() {
                        let model_name = entry.file_name();
                        let model_name = model_name.to_string_lossy();

                        // Get size of this model's cache
                        let model_size = dir_size_bytes(&entry.path()).unwrap_or(0);
                        let size_str = format_size(model_size);

                        println!("  {} {} ({})",
                            "•".bright_cyan(),
                            model_name.cyan().bold(),
                            size_str.bright_black()
                        );
                    }
                }
            }

            println!();
            println!("{}", "=".repeat(60));
            println!("Total: {} models, {}", stats.model_count, stats.total_size_human());

            Ok(())
        }
        CacheCommand::Status => {
            println!("📊 Xybrid Cache Status");
            println!("{}", "=".repeat(60));
            println!();

            let stats = client.cache_stats()
                .context("Failed to get cache stats")?;

            println!("  Cache Directory: {}", stats.cache_path.display());
            println!("  Cached Models:   {}", stats.model_count);
            println!("  Total Size:      {}", stats.total_size_human().bright_cyan());

            // Check if directory exists
            if !stats.cache_path.exists() {
                println!();
                println!("  ℹ️  Cache directory does not exist yet.");
                println!("     It will be created when you download your first model.");
            }

            println!();
            println!("{}", "=".repeat(60));

            Ok(())
        }
        CacheCommand::Clear { model_id } => {
            if let Some(id) = model_id {
                println!("🗑️  Clearing cache for: {}", id.cyan().bold());
                println!("{}", "=".repeat(60));
                println!();

                client.clear_cache(&id)
                    .context(format!("Failed to clear cache for '{}'", id))?;

                println!("✅ Cache cleared for model '{}'", id);
            } else {
                println!("🗑️  Clearing entire model cache");
                println!("{}", "=".repeat(60));
                println!();

                // Confirm dangerous operation
                println!("⚠️  This will delete ALL cached models.");
                println!("   Press Enter to continue or Ctrl+C to cancel...");

                let mut input = String::new();
                std::io::stdin().read_line(&mut input).ok();

                client.clear_all_cache()
                    .context("Failed to clear cache")?;

                println!("✅ All cached models cleared");
            }

            println!();
            println!("{}", "=".repeat(60));

            Ok(())
        }
    }
}

// ============================================================================
// Pipeline Commands (Prepare, Plan, Fetch)
// ============================================================================

/// Handle `xybrid prepare <pipeline.yaml>` command
/// Parses and validates the pipeline configuration.
fn handle_prepare_command(config_path: &Path) -> Result<()> {
    println!("📋 Xybrid Pipeline Prepare");
    println!("{}", "=".repeat(60));
    println!();

    // Check file exists
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Pipeline config not found: {}",
            config_path.display()
        ));
    }

    println!("📂 Loading: {}", config_path.display());
    println!();

    // Load and parse configuration
    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    // Display parsed configuration
    println!("✅ Pipeline configuration is valid");
    println!();

    if let Some(name) = &config.name {
        println!("  Name:     {}", name.cyan().bold());
    }

    println!("  Registry: {}", config.registry_url());
    println!("  Stages:   {}", config.stage_count());
    println!();

    // List stages with details
    println!("📦 Stages:");
    for (i, stage) in config.stages.iter().enumerate() {
        println!("  {}. {}", i + 1, stage.stage_id().cyan().bold());
        println!("     Model:  {}", stage.model_id());
        if let Some(version) = stage.version() {
            println!("     Version: {}", version);
        }
        if let Some(target) = stage.target() {
            let target_colored = match target {
                "device" => target.bright_green(),
                "cloud" => target.bright_blue(),
                "integration" => target.bright_magenta(),
                _ => target.white(),
            };
            println!("     Target: {}", target_colored);
        }
        if let Some(provider) = stage.provider() {
            println!("     Provider: {}", provider);
        }
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("✅ Pipeline is ready for execution");
    println!();
    println!("Next steps:");
    println!("  xybrid plan {}   # Show execution plan with model status", config_path.display());
    println!("  xybrid fetch {}  # Pre-download all models", config_path.display());
    println!("  xybrid run -c {} # Execute the pipeline", config_path.display());

    Ok(())
}

/// Handle `xybrid plan <pipeline.yaml>` command
/// Shows execution plan with model resolution status.
fn handle_plan_command(config_path: &Path) -> Result<()> {
    println!();

    // Check file exists
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Pipeline config not found: {}",
            config_path.display()
        ));
    }

    // Load and parse configuration
    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    // Initialize registry client
    let client = RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    // Display header
    let pipeline_name = config.name.as_deref().unwrap_or(
        config_path.file_stem().and_then(|s| s.to_str()).unwrap_or("pipeline")
    );
    println!("Pipeline: {}", pipeline_name.cyan().bold());
    println!("{}", "━".repeat(60));
    println!();

    let mut total_download_bytes: u64 = 0;
    let mut requires_network = false;
    let mut all_cached = true;

    // Process each stage
    for (i, stage) in config.stages.iter().enumerate() {
        println!("Stage {}: {}", i + 1, stage.stage_id().cyan().bold());

        // Check target type
        let target = stage.target().unwrap_or("device");

        if stage.is_cloud_stage() {
            // Integration target - requires network
            if let Some(provider) = stage.provider() {
                println!("  Target:   {} ({})", "cloud".bright_magenta(), provider);
            } else {
                println!("  Target:   {}", "cloud".bright_magenta());
            }
            println!("  Status:   {} Requires network", "🌐".bright_blue());
            requires_network = true;
        } else {
            // Device target with model - check registry
            let model_id = stage.model_id();
            println!("  Model:    {}", model_id);

            // Try to resolve the model
            match client.resolve(&model_id, None) {
                Ok(resolved) => {
                    let size_str = format_size(resolved.size_bytes);
                    println!("  Variant:  {} ({}, {})",
                        resolved.file,
                        size_str.bright_black(),
                        format!("{}/{}", resolved.format, resolved.quantization).bright_black()
                    );

                    let target_colored = match target {
                        "device" => target.bright_green(),
                        "cloud" => target.bright_blue(),
                        _ => target.white(),
                    };
                    println!("  Target:   {}", target_colored);

                    // Check cache status
                    match client.is_cached(&model_id, None) {
                        Ok(true) => {
                            println!("  Status:   {} Cached", "✅".bright_green());
                        }
                        Ok(false) => {
                            println!("  Status:   {} Not cached ({} to download)",
                                "⬇️".bright_yellow(),
                                size_str.bright_cyan()
                            );
                            total_download_bytes += resolved.size_bytes;
                            all_cached = false;
                        }
                        Err(e) => {
                            println!("  Status:   {} Cache check failed: {}", "❌".bright_red(), e);
                            all_cached = false;
                        }
                    }
                }
                Err(e) => {
                    println!("  Status:   {} Resolution failed: {}", "❌".bright_red(), e);
                    all_cached = false;
                }
            }
        }

        println!();
    }

    // Summary
    println!("{}", "━".repeat(60));

    if total_download_bytes > 0 {
        println!("Total download: {}", format_size(total_download_bytes).bright_cyan());
    } else if all_cached {
        println!("Total download: {} (all models cached)", "0 bytes".bright_green());
    }

    let offline_capable = !requires_network && all_cached;
    if offline_capable {
        println!("Offline capable: {}", "Yes".bright_green());
    } else if requires_network {
        println!("Offline capable: {} (cloud stages require network)", "No".bright_yellow());
    } else {
        println!("Offline capable: {} (models need downloading)", "No".bright_yellow());
    }

    println!();

    if total_download_bytes > 0 {
        println!("Run `xybrid fetch {}` to pre-download models.", config_path.display());
    }

    Ok(())
}

/// Handle `xybrid fetch <pipeline.yaml>` command
/// Pre-downloads all models required by the pipeline.
fn handle_fetch_pipeline_command(config_path: &Path, platform: Option<&str>) -> Result<()> {
    println!();

    // Check file exists
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Pipeline config not found: {}",
            config_path.display()
        ));
    }

    // Load and parse configuration
    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let config = PipelineConfig::from_yaml(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    // Initialize registry client
    let client = RegistryClient::from_env()
        .context("Failed to initialize registry client")?;

    // Display header
    let pipeline_name = config.name.as_deref().unwrap_or(
        config_path.file_stem().and_then(|s| s.to_str()).unwrap_or("pipeline")
    );
    println!("Fetching models for: {}", pipeline_name.cyan().bold());
    println!("{}", "━".repeat(60));
    println!();

    // Collect models to fetch (skip cloud/integration stages)
    let models_to_fetch: Vec<String> = config.stages.iter()
        .filter(|stage| !stage.is_cloud_stage())
        .map(|stage| stage.model_id())
        .collect();

    if models_to_fetch.is_empty() {
        println!("ℹ️  No device models to fetch in this pipeline.");
        return Ok(());
    }

    let mut success_count = 0;
    let mut skip_count = 0;
    let mut error_count = 0;

    for model_id in models_to_fetch {
        // Check if already cached
        match client.is_cached(&model_id, platform) {
            Ok(true) => {
                println!("{} {} (cached)", "✅".bright_green(), model_id.cyan());
                skip_count += 1;
                continue;
            }
            Ok(false) => {
                // Need to download
            }
            Err(e) => {
                println!("{} {} (cache check failed: {})", "❌".bright_red(), model_id, e);
                error_count += 1;
                continue;
            }
        }

        // Resolve and show info
        match client.resolve(&model_id, platform) {
            Ok(resolved) => {
                let size_str = format_size(resolved.size_bytes);
                print!("{} {} [{}] ", "⬇️".bright_yellow(), model_id.cyan(), size_str.bright_black());
                std::io::stdout().flush().ok();

                // Download with inline progress
                use std::cell::Cell;
                let last_percent = Cell::new(0u32);

                match client.fetch(&model_id, platform, |progress| {
                    let percent = (progress * 100.0) as u32;
                    let prev = last_percent.get();
                    if percent != prev && percent % 10 == 0 {
                        last_percent.set(percent);
                        print!("{}%", percent);
                        if percent < 100 { print!(".."); }
                        std::io::stdout().flush().ok();
                    }
                }) {
                    Ok(_) => {
                        println!(" ✓");
                        success_count += 1;
                    }
                    Err(e) => {
                        println!(" ✗ ({})", e);
                        error_count += 1;
                    }
                }
            }
            Err(e) => {
                println!("{} {} (resolution failed: {})", "❌".bright_red(), model_id, e);
                error_count += 1;
            }
        }
    }

    println!();
    println!("{}", "━".repeat(60));

    if error_count == 0 {
        println!("✅ All models ready ({} downloaded, {} cached)", success_count, skip_count);
    } else {
        println!("⚠️  Completed with errors: {} downloaded, {} cached, {} failed",
            success_count, skip_count, error_count);
    }

    Ok(())
}

// Utility functions (format_params, format_size, dir_size_bytes, etc.)
// are now defined in commands/utils.rs and imported at the top of this file.

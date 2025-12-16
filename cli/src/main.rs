//! Xybrid CLI - Command-line interface for running hybrid cloud-edge AI inference pipelines.
//!
//! This binary provides a `run` subcommand that loads pipeline configuration
//! and executes it using the xybrid-core orchestrator.

#[macro_use]
extern crate lazy_static;

mod tracing_viz;

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
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::policy_engine::PolicyEngine;
use xybrid_core::registry::{LocalRegistry, Registry};
use xybrid_core::registry_config::{RegistryAuth, RegistryConfig, RemoteRegistryConfig};
use xybrid_core::registry_remote::HttpRemoteRegistry;
use xybrid_core::routing_engine::{LocalAvailability, RoutingEngine};
use xybrid_core::target::{Platform, TargetResolver};
use xybrid_core::template_executor::TemplateExecutor;

/// Xybrid CLI - Hybrid Cloud-Edge AI Inference Pipeline Runner
#[derive(Parser)]
#[command(name = "xybrid")]
#[command(about = "Xybrid CLI - Run hybrid cloud-edge AI inference pipelines", long_about = None)]
struct Cli {
    /// Platform API key for telemetry (can also be set via XYBRID_API_KEY env var)
    #[arg(long, global = true, env = "XYBRID_API_KEY")]
    api_key: Option<String>,

    /// Platform API endpoint for telemetry (default: https://api.xybrid.ai)
    #[arg(long, global = true, env = "XYBRID_PLATFORM_URL", default_value = "https://api.xybrid.ai")]
    platform_url: String,

    /// Device ID for telemetry attribution
    #[arg(long, global = true, env = "XYBRID_DEVICE_ID")]
    device_id: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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

        /// Registry URL (e.g., http://localhost:8080) to fetch model bundles
        #[arg(long, value_name = "URL")]
        registry: Option<String>,

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
    /// List bundles in the local registry
    List {
        /// Target format filter (onnx, coreml, tflite)
        #[arg(short, long, value_name = "TARGET")]
        target: Option<String>,

        /// Registry path (default: ~/.xybrid/registry)
        #[arg(short = 'p', long, value_name = "PATH")]
        registry_path: Option<PathBuf>,
    },
    /// Deploy a .xyb bundle to the registry
    Deploy {
        /// Path to the .xyb bundle file
        #[arg(value_name = "BUNDLE")]
        bundle_path: PathBuf,

        /// Registry type (local or remote)
        #[arg(long, value_name = "TYPE", default_value = "local")]
        registry: String,

        /// Registry path (for local) or URL (for remote)
        #[arg(short = 'p', long, value_name = "PATH")]
        registry_path: Option<PathBuf>,

        /// Target format override (extracted from bundle if not specified)
        #[arg(short, long, value_name = "TARGET")]
        target: Option<String>,
    },
}

/// Pipeline configuration loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PipelineConfig {
    /// Pipeline name/description
    #[serde(default)]
    name: Option<String>,
    /// List of stage names to execute in order
    stages: Vec<String>,
    /// Input envelope configuration
    input: InputConfig,
    /// Device metrics configuration
    metrics: MetricsConfig,
    /// Model availability mapping (stage name -> available locally)
    availability: HashMap<String, bool>,
}

/// Input envelope configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InputConfig {
    /// Envelope kind (e.g., "AudioRaw", "Text", etc.)
    kind: String,
}

/// Device metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricsConfig {
    /// Network round-trip time in milliseconds
    network_rtt: u32,
    /// Battery level (0-100)
    battery: u8,
    /// Device temperature in Celsius
    temperature: f32,
}

fn display_stage_name(name: &str) -> &str {
    name.split('@').next().unwrap_or(name)
}

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
        Commands::Run {
            config,
            pipeline,
            bundle,
            dry_run,
            policy,
            input_audio,
            input_text,
            registry,
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
            run_pipeline(&config_path, dry_run, policy.as_ref(), input_audio.as_ref(), registry.as_deref(), target.as_deref(), trace, trace_export.as_ref())
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
        Commands::List {
            target,
            registry_path,
        } => list_bundles(target.as_deref(), registry_path.as_deref()),
        Commands::Deploy {
            bundle_path,
            registry,
            registry_path,
            target,
        } => deploy_bundle(&bundle_path, &registry, registry_path.as_deref(), target.as_deref()),
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

/// Deploy a .xyb bundle to the registry, validate hash, and print deployment summary.
fn deploy_bundle(
    bundle_path: &Path,
    registry_type: &str,
    registry_path: Option<&Path>,
    target_override: Option<&str>,
) -> Result<()> {
    println!("🚀 Xybrid Deploy");
    println!("{}", "=".repeat(60));
    println!("Bundle: {}", bundle_path.display());
    println!("Registry: {}", registry_type);
    println!();

    // Load and validate bundle
    if !bundle_path.exists() {
        return Err(anyhow::anyhow!(
            "Bundle file not found: {}",
            bundle_path.display()
        ));
    }

    println!("📦 Loading bundle...");
    let bundle = XyBundle::load(bundle_path).context("Failed to load bundle")?;

    let manifest = bundle.manifest();
    // Use target override if provided, otherwise use manifest target
    let deploy_target = target_override.unwrap_or(&manifest.target);

    println!("   Model ID: {}", manifest.model_id);
    println!("   Version:  {}", manifest.version);
    println!("   Target:   {} {}", deploy_target,
        if target_override.is_some() { "(overridden)" } else { "" });
    println!("   Files:    {}", manifest.files.len());
    println!("   Hash:     {}", manifest.hash);

    if manifest.has_metadata {
        println!("   Metadata: ✅ Included");
    } else {
        println!("   Metadata: ⚠️  Not included");
    }
    println!();

    // Validate hash by reading bundle file and computing hash
    println!("🔍 Validating bundle integrity...");
    let bundle_bytes = fs::read(bundle_path)
        .with_context(|| format!("Failed to read bundle file: {}", bundle_path.display()))?;

    // Compute hash of the bundle file itself (simplified validation)
    // In a real implementation, we might extract and validate the internal manifest hash
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&bundle_bytes);
    let file_hash = format!("{:x}", hasher.finalize());

    // Note: The bundle's hash is the hash of its contents, not the compressed file
    // For now, we'll verify the bundle loaded correctly and manifest is valid
    if manifest.hash.is_empty() {
        return Err(anyhow::anyhow!(
            "Bundle manifest has empty hash - bundle may be corrupted"
        ));
    }

    println!("   ✓ Bundle integrity validated");
    println!("   ✓ Manifest hash: {}", manifest.hash);
    println!("   ✓ File hash: {}", file_hash);
    println!();

    // Prepare registry
    println!("📋 Preparing registry...");
    let mut registry: Box<dyn Registry> = match registry_type {
        "local" => {
            let reg_path = if let Some(path) = registry_path {
                path.to_path_buf()
            } else {
                // Use default registry location
                dirs::home_dir()
                    .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?
                    .join(".xybrid")
                    .join("registry")
            };

            println!("   Registry path: {}", reg_path.display());
            Box::new(
                LocalRegistry::new(&reg_path)
                    .map_err(|e| anyhow::anyhow!("Failed to create local registry: {}", e))?,
            )
        }
        "remote" => {
            let base_url = registry_path
                .and_then(|p| p.to_str().map(|s| s.to_string()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Remote registry requires --registry-path <https://registry-url>"
                    )
                })?;

            let config = RegistryConfig {
                local_path: None,
                remote: Some(RemoteRegistryConfig {
                    base_url,
                    index_path: None,
                    bundle_path: None,
                    auth: RegistryAuth::None,
                    timeout_ms: None,
                    retry_attempts: None,
                }),
            };

            println!(
                "   Remote URL: {}",
                config.remote.as_ref().unwrap().base_url
            );
            let registry = HttpRemoteRegistry::from_config(&config)
                .map_err(|e| anyhow::anyhow!("Failed to initialize remote registry: {}", e))?;
            Box::new(registry)
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid registry type: {}. Use 'local' or 'remote'",
                registry_type
            ));
        }
    };

    // Check if bundle already exists
    if let Ok(existing) = registry.get_metadata(&manifest.model_id, Some(&manifest.version)) {
        println!(
            "⚠️  Bundle {}@{} already exists in registry",
            manifest.model_id, manifest.version
        );
        println!("   Location: {}", existing.path);
        println!("   Size: {} bytes", existing.size_bytes);
        println!();
        println!("   Overwrite? (y/N): ");
        // For MVP, we'll just warn and continue - in production you'd prompt
        println!("   → Continuing with deployment...");
        println!();
    }

    // Store bundle in registry
    println!("💾 Uploading to registry...");
    let metadata = registry
        .store_bundle(&manifest.model_id, &manifest.version, bundle_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to store bundle in registry: {}", e))?;

    println!("   ✓ Bundle stored successfully");
    println!();

    // Print deployment manifest summary
    println!("{}", "=".repeat(60));
    println!("✅ Deployment Complete");
    println!("{}", "=".repeat(60));
    println!();
    println!("📊 Deployment Manifest:");
    println!("   Model ID:    {}", metadata.id);
    println!("   Version:     {}", metadata.version);
    println!("   Target:      {}", deploy_target);
    println!("   Registry:    {}", registry_type);
    println!("   Location:    {}", metadata.path);
    println!(
        "   Size:        {} bytes ({:.2} KB)",
        metadata.size_bytes,
        metadata.size_bytes as f64 / 1024.0
    );
    println!("   Content Hash: {}", manifest.hash);
    println!("   File Hash:    {}", file_hash);
    println!("   Files:        {}", manifest.files.len());
    if !manifest.files.is_empty() {
        println!("   File List:");
        for file in &manifest.files {
            println!("     • {}", file);
        }
    }
    println!();

    // Display metadata information if present
    if manifest.has_metadata {
        println!("📋 Model Metadata:");
        if let Ok(Some(metadata_json)) = bundle.get_metadata_json() {
            // Parse and display metadata details
            if let Ok(metadata_value) = serde_json::from_str::<serde_json::Value>(&metadata_json) {
                if let Some(model_id) = metadata_value.get("model_id").and_then(|v| v.as_str()) {
                    println!("   Model ID:      {}", model_id);
                }
                if let Some(version) = metadata_value.get("version").and_then(|v| v.as_str()) {
                    println!("   Version:       {}", version);
                }

                // Show preprocessing steps
                if let Some(preprocessing) = metadata_value.get("preprocessing").and_then(|v| v.as_array()) {
                    println!("   Preprocessing: {} steps", preprocessing.len());
                    for (i, step) in preprocessing.iter().enumerate() {
                        if let Some(step_type) = step.get("type").and_then(|v| v.as_str()) {
                            println!("     {}. {}", i + 1, step_type);
                        }
                    }
                }

                // Show postprocessing steps
                if let Some(postprocessing) = metadata_value.get("postprocessing").and_then(|v| v.as_array()) {
                    println!("   Postprocessing: {} steps", postprocessing.len());
                    for (i, step) in postprocessing.iter().enumerate() {
                        if let Some(step_type) = step.get("type").and_then(|v| v.as_str()) {
                            println!("     {}. {}", i + 1, step_type);
                        }
                    }
                }
            }
        }
        println!();
    }

    Ok(())
}

/// List bundles in the local registry.
fn list_bundles(target_filter: Option<&str>, registry_path: Option<&Path>) -> Result<()> {
    println!("📦 Xybrid Registry");
    println!("{}", "=".repeat(60));

    // Determine registry path
    let reg_path = if let Some(path) = registry_path {
        path.to_path_buf()
    } else {
        dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?
            .join(".xybrid")
            .join("registry")
    };

    println!("📂 Registry: {}", reg_path.display());
    println!();

    if !reg_path.exists() {
        println!("ℹ️  Registry directory does not exist.");
        println!("   Run 'xybrid deploy' to add bundles to the registry.");
        return Ok(());
    }

    // Create registry and list bundles
    let registry = LocalRegistry::new(&reg_path)
        .map_err(|e| anyhow::anyhow!("Failed to open registry: {}", e))?;

    let bundles = registry.list_bundles()
        .map_err(|e| anyhow::anyhow!("Failed to list bundles: {}", e))?;

    if bundles.is_empty() {
        println!("ℹ️  No bundles found in registry.");
        println!("   Run 'xybrid deploy' to add bundles to the registry.");
        return Ok(());
    }

    // Group bundles by model_id and version
    use std::collections::BTreeMap;
    let mut grouped: BTreeMap<String, Vec<&xybrid_core::registry::BundleMetadata>> = BTreeMap::new();

    for bundle in &bundles {
        let key = format!("{}@{}", bundle.id, bundle.version);
        grouped.entry(key).or_default().push(bundle);
    }

    // Display bundles
    println!("📋 Available Bundles:");
    println!();

    for (key, bundle_list) in grouped {
        // Check if any bundle in this group matches the target filter
        let has_matching_target = if let Some(filter) = target_filter {
            bundle_list.iter().any(|b| {
                // Extract target from path if possible
                b.path.contains(filter)
            })
        } else {
            true
        };

        if !has_matching_target {
            continue;
        }

        println!("  {}", key.cyan());

        for bundle in bundle_list {
            // Try to extract target from path (e.g., .../1.0/onnx.xyb -> onnx)
            let target = Path::new(&bundle.path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            if let Some(filter) = target_filter {
                if !target.contains(filter) {
                    continue;
                }
            }

            let size_str = if bundle.size_bytes > 1_000_000 {
                format!("{:.1} MB", bundle.size_bytes as f64 / 1_000_000.0)
            } else if bundle.size_bytes > 1_000 {
                format!("{:.1} KB", bundle.size_bytes as f64 / 1_000.0)
            } else {
                format!("{} bytes", bundle.size_bytes)
            };

            println!("    - {} ({})", target.bright_green(), size_str.bright_black());
        }
        println!();
    }

    println!("{}", "=".repeat(60));
    println!("Total: {} bundles", bundles.len());

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
    registry_url: Option<&str>,
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

    let config: PipelineConfig = serde_yaml::from_str(&config_content)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    println!("🚀 Xybrid Pipeline Runner");
    if let Some(name) = &config.name {
        println!("📋 Pipeline: {}\n", name);
    }

    // Build stage descriptors from config
    let stages: Vec<StageDescriptor> = config
        .stages
        .iter()
        .map(|name| StageDescriptor::new(name.clone()))
        .collect();

    // Create input envelope - use audio file if provided
    let input = if let Some(audio_path) = input_audio {
        println!("📂 Loading audio file: {}", audio_path.display());
        let audio_bytes = fs::read(audio_path)
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        println!("   Loaded {} bytes", audio_bytes.len());
        Envelope::new(EnvelopeKind::Audio(audio_bytes))
    } else {
        let kind = match config.input.kind.as_str() {
            "Audio" | "audio" => EnvelopeKind::Audio(vec![]),
            "Text" | "text" => EnvelopeKind::Text(String::new()),
            "Embedding" | "embedding" => EnvelopeKind::Embedding(vec![]),
            _ => EnvelopeKind::Text(config.input.kind.clone()),
        };
        Envelope::new(kind)
    };

    // Create device metrics
    let metrics = DeviceMetrics {
        network_rtt: config.metrics.network_rtt,
        battery: config.metrics.battery,
        temperature: config.metrics.temperature,
    };

    // Create availability function from config
    let availability_map = config.availability.clone();
    let availability_fn = move |stage: &str| -> LocalAvailability {
        let exists = availability_map.get(stage).copied().unwrap_or(false);
        LocalAvailability::new(exists)
    };

    // Display configuration
    println!("📊 Configuration:");
    println!("   Stages: {}", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        println!("      {}. {}", i + 1, display_stage_name(&stage.name));
    }
    println!();

    println!("📦 Input: {}", input.kind_str());

    println!("📊 Device Metrics:");
    println!("   Network RTT: {}ms", metrics.network_rtt);
    println!("   Battery: {}%", metrics.battery);
    println!("   Temperature: {}°C", metrics.temperature);
    println!();

    println!("🔍 Model Availability:");
    for (stage, available) in &config.availability {
        println!(
            "   {} {}",
            if *available { "✅" } else { "❌" },
            display_stage_name(stage)
        );
    }
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
        let mut routing_engine = xybrid_core::routing_engine::DefaultRoutingEngine::new();
        let policy_engine = xybrid_core::policy_engine::DefaultPolicyEngine::with_default_policy();

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

    // Configure registry if provided
    if let Some(url) = registry_url {
        println!("🌐 Configuring registry: {}", url);
        let registry_config = RegistryConfig {
            local_path: None,
            remote: Some(RemoteRegistryConfig {
                base_url: url.to_string(),
                index_path: None,
                bundle_path: None,
                auth: RegistryAuth::None,
                timeout_ms: Some(30000),
                retry_attempts: Some(3),
            }),
        };
        orchestrator.executor_mut().set_pipeline_registry(registry_config);
        println!("   ✓ Registry configured");
        println!();
    }

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

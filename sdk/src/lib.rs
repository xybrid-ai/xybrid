//! Xybrid SDK - Developer-facing API for hybrid cloud-edge AI inference.
//!
//! This crate provides high-level abstractions for:
//! - Loading and running ML models (ASR, TTS, embeddings)
//! - Streaming inference for real-time applications
//! - Multi-stage pipelines with intelligent routing
//!
//! # Architecture
//!
//! The SDK follows a **Loader → Model → Run** pattern:
//!
//! ```text
//! ModelLoader::from_registry()  →  loader.load()  →  model.run(&envelope)
//!                                                 →  model.stream(config)
//! ```
//!
//! # Quick Start
//!
//! ## Batch Inference
//!
//! ```rust,no_run
//! use xybrid_sdk::{ModelLoader, Envelope};
//!
//! // Load model from registry
//! let loader = ModelLoader::from_registry("http://localhost:8080", "whisper-tiny", "1.0");
//! let model = loader.load()?;
//!
//! // Run inference
//! let result = model.run(&Envelope::audio(audio_bytes))?;
//! println!("Transcription: {}", result.unwrap_text());
//! ```
//!
//! ## Streaming ASR
//!
//! ```rust,no_run
//! use xybrid_sdk::{ModelLoader, StreamConfig};
//!
//! let model = ModelLoader::from_directory("test_models/whisper-tiny")?.load()?;
//! let stream = model.stream(StreamConfig::with_vad())?;
//!
//! // Feed audio chunks
//! stream.feed(&audio_samples)?;
//!
//! // Get final transcript
//! let result = stream.flush()?;
//! println!("Transcript: {}", result.text);
//! ```
//!
//! ## Pipelines
//!
//! ```rust,no_run
//! use xybrid_sdk::run_pipeline;
//!
//! let result = run_pipeline("examples/pipeline.yaml")?;
//! println!("Pipeline completed in {}ms", result.total_latency_ms);
//! for stage in &result.stages {
//!     println!("  {}: {}ms ({})", stage.name, stage.latency_ms, stage.target);
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use tokio::sync::mpsc;
use xybrid_core::context::{DeviceMetrics, StageDescriptor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::{Orchestrator, StageExecutionResult};
use xybrid_core::routing_engine::LocalAvailability;

// ============================================================================
// Module Declarations
// ============================================================================

pub mod cache;
pub mod model;
pub mod pipeline;
pub mod result;
pub mod source;
pub mod stream;
pub mod telemetry;

// ============================================================================
// Re-exports
// ============================================================================

// Re-export xybrid_core modules (selective to avoid conflicts)
pub use xybrid_core::bundler;
pub use xybrid_core::context;
pub use xybrid_core::execution_template;
pub use xybrid_core::ir;
pub use xybrid_core::orchestrator;
pub use xybrid_core::routing_engine;
pub use xybrid_core::template_executor;

// SDK types (new API)
pub use cache::{CacheManager, CacheStatus};
pub use model::{ModelLoader, SdkResult, StreamConfig, XybridModel};
pub use model::SdkError;
pub use pipeline::{PipelineExecutionResult, PipelineInputType, PipelineLoader, XybridPipeline};
pub use pipeline::StageTiming as PipelineStageTiming;  // Renamed to avoid conflict with legacy StageTiming
pub use result::{InferenceResult, OutputType};
pub use source::ModelSource;
pub use stream::{PartialResult, StreamState, StreamStats, TranscriptionResult, XybridStream};
pub use telemetry::{
    register_telemetry_sender, publish_telemetry_event, TelemetryEvent, TelemetrySender,
    // Platform telemetry exports
    TelemetryConfig, HttpTelemetryExporter,
    init_platform_telemetry, init_platform_telemetry_from_env,
    set_telemetry_pipeline_context, flush_platform_telemetry, shutdown_platform_telemetry,
    // Orchestrator event bridge
    bridge_orchestrator_events, convert_orchestrator_event,
};

/// Re-export OrchestratorEvent for event subscriptions
pub use xybrid_core::event_bus::OrchestratorEvent;

// ============================================================================
// SDK Configuration
// ============================================================================

/// Set the Xybrid API key for gateway authentication.
///
/// This sets the `XYBRID_API_KEY` environment variable which is used
/// by the LLM client when routing through the Xybrid Gateway.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::set_api_key;
///
/// // Set API key at startup
/// set_api_key("your-xybrid-api-key");
///
/// // Now pipelines will use this key for gateway requests
/// ```
///
/// # Note
///
/// For Flutter apps, you can also set this from Dart before running pipelines.
/// The key is stored in the process environment and persists for the app lifetime.
pub fn set_api_key(api_key: &str) {
    std::env::set_var("XYBRID_API_KEY", api_key);
}

/// Set a provider-specific API key for direct API calls.
///
/// Use this when running LLM stages with `backend: "direct"` in the pipeline.
///
/// # Supported Providers
///
/// - `"openai"` → Sets `OPENAI_API_KEY`
/// - `"anthropic"` → Sets `ANTHROPIC_API_KEY`
/// - `"google"` → Sets `GOOGLE_API_KEY`
/// - `"openrouter"` → Sets `OPENROUTER_API_KEY`
/// - `"elevenlabs"` → Sets `ELEVENLABS_API_KEY`
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::set_provider_api_key;
///
/// // For direct OpenAI API calls
/// set_provider_api_key("openai", "sk-...");
/// ```
pub fn set_provider_api_key(provider: &str, api_key: &str) {
    let env_var = match provider.to_lowercase().as_str() {
        "openai" => "OPENAI_API_KEY",
        "anthropic" | "claude" => "ANTHROPIC_API_KEY",
        "google" | "gemini" => "GOOGLE_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "elevenlabs" => "ELEVENLABS_API_KEY",
        _ => {
            // Custom provider - use uppercase with _API_KEY suffix
            let custom_var = format!("{}_API_KEY", provider.to_uppercase());
            std::env::set_var(&custom_var, api_key);
            return;
        }
    };
    std::env::set_var(env_var, api_key);
}

/// Get the currently configured Xybrid API key (if set).
///
/// Returns `None` if no API key is configured.
pub fn get_api_key() -> Option<String> {
    std::env::var("XYBRID_API_KEY").ok()
}

/// Check if the Xybrid API key is configured.
pub fn has_api_key() -> bool {
    std::env::var("XYBRID_API_KEY").is_ok()
}

/// Re-export common types for convenience
pub mod prelude {
    pub use xybrid_core::context::{DeviceMetrics, StageDescriptor};
    pub use xybrid_core::ir::{Envelope, EnvelopeKind};
    pub use xybrid_core::orchestrator::{Orchestrator, OrchestratorError, StageExecutionResult};
    pub use xybrid_core::routing_engine::{LocalAvailability, RouteTarget, RoutingDecision};
}

/// Async event stream for subscribing to orchestrator events.
pub struct EventStream {
    receiver: mpsc::Receiver<OrchestratorEvent>,
}

impl EventStream {
    /// Receive the next event asynchronously.
    pub async fn recv(&mut self) -> Option<OrchestratorEvent> {
        self.receiver.recv().await
    }

    /// Try to receive an event without blocking.
    pub fn try_recv(&mut self) -> Result<OrchestratorEvent, mpsc::error::TryRecvError> {
        self.receiver.try_recv()
    }
}

/// Create an async event stream from an orchestrator's event bus.
pub fn subscribe_events(orchestrator: &Orchestrator) -> EventStream {
    let (tx, rx) = mpsc::channel(100);
    let event_bus = orchestrator.event_bus();
    let subscription = event_bus.subscribe();

    // Bridge sync receiver to async channel using a dedicated thread
    // The subscription receiver is not Send, so we use a blocking thread
    std::thread::spawn(move || {
        loop {
            match subscription.recv() {
                Ok(event) => {
                    // Use blocking_send since we're in a blocking thread
                    if tx.blocking_send(event).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    EventStream { receiver: rx }
}

/// Hybrid routing macros module.
///
/// This module provides the `#[hybrid::route]` macro for annotating
/// inference functions with hybrid routing capabilities.
pub mod hybrid {
    /// Route decorator macro for hybrid inference stages.
    ///
    /// Use this macro to annotate functions that should be routed
    /// by the Xybrid orchestrator between local and cloud execution.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use xybrid_sdk::hybrid;
    ///
    /// #[hybrid::route]
    /// fn process_audio(input: String) -> String {
    ///     // Function will be executed via orchestrator
    ///     todo!()
    /// }
    /// ```
    pub use xybrid_macros::route;
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

/// Timing information for a single pipeline stage.
#[derive(Debug, Clone, Serialize)]
pub struct StageTiming {
    /// Stage name
    pub name: String,
    /// Stage latency in milliseconds
    pub latency_ms: u32,
    /// Routing target (local, cloud, or fallback)
    pub target: String,
}

/// Result of pipeline execution with timing information.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineResult {
    /// Pipeline name (if specified in config)
    pub name: Option<String>,
    /// Stage timing information
    pub stages: Vec<StageTiming>,
    /// Total pipeline latency in milliseconds
    pub total_latency_ms: u32,
    /// Final output envelope kind
    pub final_output: String,
}

/// Legacy error type for pipeline YAML execution (kept for backward compatibility).
/// For the new model API, use `model::SdkError`.
#[derive(Debug, thiserror::Error)]
pub enum PipelineConfigError {
    #[error("Failed to read config file: {0}")]
    ConfigReadError(String),
    #[error("Failed to parse YAML config: {0}")]
    ConfigParseError(String),
    #[error("Pipeline execution failed: {0}")]
    ExecutionError(String),
}

/// Run a pipeline from a configuration file.
///
/// This function loads a YAML configuration file, creates an orchestrator,
/// and executes the pipeline. Returns timing information for all stages.
///
/// # Arguments
///
/// * `config_path` - Path to the YAML configuration file
///
/// # Returns
///
/// A `PipelineResult` containing stage timings and total latency, or an error.
///
/// # Example
///
/// ```rust,no_run
/// use xybrid_sdk::run_pipeline;
///
/// match run_pipeline("examples/hiiipe.yaml") {
///     Ok(result) => {
///         println!("Pipeline completed in {}ms", result.total_latency_ms);
///         for stage in &result.stages {
///             println!("  {}: {}ms ({})", stage.name, stage.latency_ms, stage.target);
///         }
///     }
///     Err(e) => eprintln!("Pipeline failed: {}", e),
/// }
/// ```
pub fn run_pipeline(config_path: &str) -> Result<PipelineResult, PipelineConfigError> {
    // Load and parse configuration file
    let config_content = fs::read_to_string(config_path)
        .map_err(|e| PipelineConfigError::ConfigReadError(format!("{}: {}", config_path, e)))?;

    let config: PipelineConfig = serde_yaml::from_str(&config_content)
        .map_err(|e| PipelineConfigError::ConfigParseError(format!("{}: {}", config_path, e)))?;

    // Minimal logging
    if let Some(name) = &config.name {
        eprintln!("[xybrid-sdk] Running pipeline: {}", name);
    } else {
        eprintln!("[xybrid-sdk] Running pipeline from: {}", config_path);
    }

    // Build stage descriptors from config
    let stages: Vec<StageDescriptor> = config
        .stages
        .iter()
        .map(|name| StageDescriptor::new(name.clone()))
        .collect();

    eprintln!("[xybrid-sdk] Pipeline has {} stages", stages.len());

    // Create input envelope
    let kind = match config.input.kind.as_str() {
        "Audio" | "audio" => EnvelopeKind::Audio(vec![]),
        "Text" | "text" => EnvelopeKind::Text(String::new()),
        "Embedding" | "embedding" => EnvelopeKind::Embedding(vec![]),
        _ => EnvelopeKind::Text(config.input.kind.clone()),
    };
    let input = Envelope::new(kind);

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

    // Create orchestrator
    let mut orchestrator = Orchestrator::new();

    // Execute the pipeline
    let start_time = std::time::Instant::now();
    let results: Vec<StageExecutionResult> = orchestrator
        .execute_pipeline(&stages, &input, &metrics, &availability_fn)
        .map_err(|e| PipelineConfigError::ExecutionError(format!("{}", e)))?;
    let total_latency_ms = start_time.elapsed().as_millis() as u32;

    // Convert to SDK result format
    let stage_timings: Vec<StageTiming> = results
        .iter()
        .map(|result| StageTiming {
            name: result.stage.clone(),
            latency_ms: result.latency_ms,
            target: result.routing_decision.target.to_string(),
        })
        .collect();

    let final_output = results
        .last()
        .map(|r| r.output.kind_str().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!("[xybrid-sdk] Pipeline completed in {}ms", total_latency_ms);

    Ok(PipelineResult {
        name: config.name.clone(),
        stages: stage_timings,
        total_latency_ms,
        final_output,
    })
}

/// Run a pipeline from a configuration file asynchronously.
///
/// This function loads a YAML configuration file, creates an orchestrator,
/// and executes the pipeline asynchronously. Returns timing information for all stages.
///
/// # Arguments
///
/// * `config_path` - Path to the YAML configuration file
///
/// # Returns
///
/// A future that resolves to a `PipelineResult` containing stage timings and total latency, or an error.
///
/// # Example
///
/// ```rust,no_run
/// use xybrid_sdk::run_pipeline_async;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let result = run_pipeline_async("examples/hiiipe.yaml").await?;
/// println!("Pipeline completed in {}ms", result.total_latency_ms);
/// for stage in &result.stages {
///     println!("  {}: {}ms ({})", stage.name, stage.latency_ms, stage.target);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn run_pipeline_async(config_path: &str) -> Result<PipelineResult, PipelineConfigError> {
    // Load and parse configuration file
    let config_content = tokio::fs::read_to_string(config_path)
        .await
        .map_err(|e| PipelineConfigError::ConfigReadError(format!("{}: {}", config_path, e)))?;

    let config: PipelineConfig = serde_yaml::from_str(&config_content)
        .map_err(|e| PipelineConfigError::ConfigParseError(format!("{}: {}", config_path, e)))?;

    // Minimal logging
    if let Some(name) = &config.name {
        eprintln!("[xybrid-sdk] Running pipeline (async): {}", name);
    } else {
        eprintln!(
            "[xybrid-sdk] Running pipeline (async) from: {}",
            config_path
        );
    }

    // Build stage descriptors from config
    let stages: Vec<StageDescriptor> = config
        .stages
        .iter()
        .map(|name| StageDescriptor::new(name.clone()))
        .collect();

    eprintln!("[xybrid-sdk] Pipeline has {} stages", stages.len());

    // Create input envelope
    let kind = match config.input.kind.as_str() {
        "Audio" | "audio" => EnvelopeKind::Audio(vec![]),
        "Text" | "text" => EnvelopeKind::Text(String::new()),
        "Embedding" | "embedding" => EnvelopeKind::Embedding(vec![]),
        _ => EnvelopeKind::Text(config.input.kind.clone()),
    };
    let input = Envelope::new(kind);

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

    // Create orchestrator
    let mut orchestrator = Orchestrator::new();

    // Execute the pipeline asynchronously
    let start_time = std::time::Instant::now();
    let results: Vec<StageExecutionResult> = orchestrator
        .execute_pipeline_async(&stages, &input, &metrics, &availability_fn)
        .await
        .map_err(|e| PipelineConfigError::ExecutionError(format!("{}", e)))?;
    let total_latency_ms = start_time.elapsed().as_millis() as u32;

    // Convert to SDK result format
    let stage_timings: Vec<StageTiming> = results
        .iter()
        .map(|result| StageTiming {
            name: result.stage.clone(),
            latency_ms: result.latency_ms,
            target: result.routing_decision.target.to_string(),
        })
        .collect();

    let final_output = results
        .last()
        .map(|r| r.output.kind_str().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!("[xybrid-sdk] Pipeline completed in {}ms", total_latency_ms);

    Ok(PipelineResult {
        name: config.name.clone(),
        stages: stage_timings,
        total_latency_ms,
        final_output,
    })
}

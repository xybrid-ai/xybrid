// Allow dead code in alpha crate — many modules are scaffolded but not yet wired up.
#![allow(dead_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::ptr_arg,
    clippy::map_identity,
    clippy::while_let_loop
)]

//! Xybrid SDK - Developer-facing API for hybrid cloud-edge AI inference.
//!
//! The SDK is **local-first**: model load/run/stream and pipelines work fully
//! offline with no account. Authenticating with an API key adds an **additive
//! platform layer** on top of the same runtime — cloud routing and
//! control-plane sync. Observability (OpenTelemetry traces, metrics, and logs)
//! is configured separately and explicitly; see [`observability`]. This crate provides high-level abstractions for:
//! - Loading and running ML models (ASR, TTS, embeddings) — local by default
//! - Streaming inference for real-time applications
//! - Multi-stage pipelines with intelligent local→cloud routing
//! - Platform integration (auth, cloud routing) once a key is provided
//! - OpenTelemetry observability to Xybrid or any OTLP backend, once configured
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
//! ## Initialization
//!
//! Anonymous use works out of the box — local inference, no account needed.
//! Provide an API key to layer on the platform (cloud routing) without
//! changing the local path. Trace export is a separate, explicit choice:
//!
//! ```no_run
//! use xybrid_sdk::observability::ObservabilityConfig;
//!
//! // Anonymous — local inference, nothing exported
//! xybrid_sdk::init().run();
//!
//! // Cloud routing plus traces to the Xybrid dashboard
//! xybrid_sdk::init()
//!     .api_key("xy_live_...")
//!     .observability(ObservabilityConfig::xybrid("xy_live_..."))
//!     .run();
//! ```
//!
//! Get a free key at <https://dashboard.xybrid.dev>. The first inference
//! without observability configured emits a one-shot info log pointing at
//! the observability docs; set `XYBRID_QUIET=1` to suppress it.
//!
//! ## Batch Inference
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use xybrid_sdk::ModelLoader;
//! use xybrid_sdk::ir::{Envelope, EnvelopeKind};
//!
//! // Load model from registry
//! let loader = ModelLoader::from_registry("whisper-tiny");
//! let model = loader.load()?;
//!
//! // Run inference
//! let audio_bytes: Vec<u8> = vec![];
//! let envelope = Envelope::new(EnvelopeKind::Audio(audio_bytes));
//! let result = model.run(&envelope, None)?;
//! println!("Transcription: {}", result.unwrap_text());
//! # Ok(())
//! # }
//! ```
//!
//! ## Streaming ASR
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use xybrid_sdk::{ModelLoader, StreamConfig};
//!
//! let model = ModelLoader::from_directory("/path/to/whisper-model")?.load()?;
//! let stream = model.stream(StreamConfig::with_vad())?;
//!
//! // Feed audio chunks
//! let audio_samples: Vec<f32> = vec![];
//! stream.feed(&audio_samples)?;
//!
//! // Get final transcript
//! let result = stream.flush()?;
//! println!("Transcript: {}", result.text);
//! # Ok(())
//! # }
//! ```
//!
//! ## Pipelines
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use xybrid_sdk::run_pipeline;
//!
//! let result = run_pipeline("examples/pipeline.yaml")?;
//! println!("Pipeline completed in {}ms", result.total_latency_ms);
//! for stage in &result.stages {
//!     println!("  {}: {}ms ({})", stage.name, stage.latency_ms, stage.target);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Model Warmup (for LLM and other large models)
//!
//! Pre-load models at app startup for fast first inference:
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use xybrid_sdk::{ModelLoader, PipelineRef};
//! use xybrid_sdk::ir::{Envelope, EnvelopeKind};
//!
//! // Option 1: Warmup a single model
//! let loader = ModelLoader::from_registry("gemma-3-1b");
//! let model = loader.load()?;
//! model.warmup()?;  // Pre-loads model weights, compiles shaders
//! let envelope = Envelope::new(EnvelopeKind::Text("Hello".into()));
//! let result = model.run(&envelope, None)?;  // Fast!
//!
//! // Option 2: Warmup a pipeline
//! let yaml = "stages: []";
//! let pipeline = PipelineRef::from_yaml(yaml)?.load()?;
//! pipeline.load_models()?;  // Download models
//! pipeline.warmup()?;       // Pre-load into memory
//! let result = pipeline.run(&envelope)?;  // Fast!
//!
//! // Option 3: Async warmup for background loading
//! let model = loader.load()?;
//! tokio::spawn(async move {
//!     model.warmup_async().await
//! });
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use tokio::sync::mpsc;
use xybrid_core::context::{DeviceMetrics, StageDescriptor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::routing_engine::LocalAvailability;
use xybrid_core::orchestrator::{Orchestrator, StageExecutionResult};

// ============================================================================
// Module Declarations
// ============================================================================

pub mod benchmark;
pub mod cache;
pub mod device;
pub mod download;
pub mod llm;
pub mod metadata_gen;
pub mod model;
pub mod model_registry;
pub mod observability;
pub mod pipeline;
pub mod platform;
pub mod registry_client;
pub mod result;
pub mod run_options;
pub mod source;
pub mod stream;
pub mod streaming;
pub mod telemetry;
pub mod telemetry_optout;

// ============================================================================
// Re-exports
// ============================================================================

// Re-export xybrid_core modules (selective to avoid conflicts)
pub use xybrid_core::bundler;
pub use xybrid_core::cache_provider::CacheProvider;
pub use xybrid_core::context;
pub use xybrid_core::conversation::ConversationContext;
pub use xybrid_core::device::{
    clear_battery_level, clear_thermal_state, set_battery_level, set_thermal_state, DeviceProfile,
    MemoryPressure, ResourceMonitor, ResourceSnapshot, ResourceTelemetryMode, ResourceUsageSummary,
    RunGuard as ResourceRunGuard, ThermalState,
};
pub use xybrid_core::execution;
pub use xybrid_core::features;
/// Tool-calling request and response types re-exported for SDK users.
pub use xybrid_core::gateway::{FunctionCall, FunctionDefinition, Tool, ToolCall};
pub use xybrid_core::ir;
pub use xybrid_core::orchestrator;
pub use xybrid_core::orchestrator::routing_engine;

// Re-export voice types for TTS model discovery
pub use xybrid_core::execution::{VoiceConfig, VoiceInfo};

// Re-export streaming and generation types for LLM inference (always available for FFI/bindings)
pub use xybrid_core::runtime_adapter::types::{
    GenerationConfig, PartialToken, StreamingCallback, StreamingError,
};

// Re-export JSON-Schema → GBNF conversion for constrained decoding, so SDK
// consumers and binding crates can produce `GenerationConfig::grammar` values
// without depending on xybrid-core directly.
pub use xybrid_core::runtime_adapter::grammar::{
    json_schema_str_to_gbnf, json_schema_to_gbnf, GrammarError,
};

// Backwards compatibility re-exports
#[doc(hidden)]
pub use xybrid_core::execution as template_executor;
#[doc(hidden)]
pub use xybrid_core::execution::template as execution_template;

// SDK types (new API)
pub use benchmark::{compare_benchmarks, BenchmarkResult, ExecutionProviderInfo};
pub use cache::{CacheEntryInfo, CacheEntryLocation, CacheManager, CacheStatus, SdkCacheProvider};
pub use device::{device_id, Device};
pub use download::{DownloadState, DownloadStatus, ModelDownload};
pub use llm::{
    default_gateway_url, set_gateway_url, ChatMessage, CompletionRequest, CompletionResponse,
    LlmBackend, LlmClientConfig, MessageRole, TokenUsage,
};
pub use model::SdkError;
pub use model::{
    LoadState, ModelLoader, SdkResult, SeamInfo, StreamConfig, StreamEvent, StreamToken,
    XybridModel,
};
pub use model_registry::{release_memory, AutoReleasePolicy};
pub use platform::current_platform;
pub use registry_client::{CacheStats, ModelSummary, RegistryClient, ResolvedVariant};
pub use run_options::{AbortPolicy, AbortReason, AbortSignal, CancellationToken, RunOptions};
// Re-exported so callers can use the trait form of `SdkError::is_retryable`
// / `retry_after` (e.g. in generic retry helpers) without naming
// `xybrid_core`. The inherent methods on `SdkError` cover the common case
// without any import.
pub use xybrid_core::http::RetryableError;
// Pipeline API (PipelineRef → Pipeline)
pub use pipeline::{
    // Config types for FFI bindings (Flutter, Kotlin, Swift)
    AudioInputConfig,
    AudioSampleFormat,
    ConfigOutputType,
    DownloadProgress,
    // FFI result types for platform bindings (Flutter, Kotlin, Swift)
    FfiPipelineExecutionResult,
    FfiStageExecutionResult,
    InputConfig,
    InputType,
    Pipeline,
    PipelineExecutionResult,
    PipelineInputType,
    PipelineRef,
    PipelineSource,
    StageInfo,
    StageStatus,
    StageTarget,
    StageTiming as PipelineStageTiming, // Alias to avoid conflict with legacy StageTiming
    TextInputConfig,
    Xybrid,
};
pub use result::{
    ExecutionProvenance, InferenceMetrics, InferenceResult, OutputType, StageLatency,
};
// Choice scoring: the request a caller sends to a decision model and the
// scores it returns.
pub use source::ModelSource;
pub use stream::{PartialResult, StreamState, StreamStats, TranscriptionResult, XybridStream};
pub use xybrid_core::ir::{Choice, ChoiceRequest, ChoiceScore, ChoiceScores};
// Decision models: a state plus typed questions, and one answer per question.
pub use xybrid_core::ir::{
    AnswerInvariant, DecisionAnswer, DecisionContent, DecisionContentView, DecisionError,
    DecisionInstructions, DecisionLevel, DecisionMedia, DecisionOption, DecisionOrigin,
    DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionScore, DecisionState,
    DecisionUsage, DecisionValue, Decisions, JsonErrorKind, JsonKind, JsonNumber, JsonView,
    OrderedJson, ScoreStatistic,
};
// Per-phoneme timings a TTS result carries (`InferenceResult::phoneme_timings`).
pub use xybrid_core::ir::PhonemeTiming;
// FFI streaming types for platform bindings (Flutter, Kotlin, Swift)
pub use observability::{
    flush_observability, flush_observability_async, init_observability, shutdown_observability,
    shutdown_observability_async, CapturedContext, ObservabilityConfig, ObservabilitySession,
    ObservationContext, ObservationOptions, ObservationRef,
};
pub use streaming::{FfiPartialResult, FfiStreamState, FfiStreamStats, FfiStreamingConfig};
// In-process domain events (local subscribers; not an export path).
pub use telemetry::{
    bridge_orchestrator_events, convert_orchestrator_event, publish_telemetry_event,
    register_telemetry_sender, subscribe_orchestrator_events, BridgeError, BridgeHandle,
    OrchestratorEventBridge, TelemetryEvent, TelemetrySender,
};
pub use telemetry_optout::is_telemetry_opted_out;

/// Re-export OrchestratorEvent for event subscriptions
pub use xybrid_core::event_bus::OrchestratorEvent;

/// Re-export execution listener types for custom instrumentation
pub use xybrid_core::execution::listener::{
    clear_execution_listener, set_execution_listener, ExecutionEvent,
};

// ============================================================================
// SDK Configuration
// ============================================================================

use std::sync::OnceLock;

/// Global SDK configuration.
static SDK_CONFIG: OnceLock<SdkConfig> = OnceLock::new();

/// Process-global binding identifier set by platform bindings at init.
///
/// First-set-wins (backed by [`OnceLock`]); after the first
/// [`set_binding`] call, subsequent calls are silent no-ops. Read via
/// [`get_binding`], which returns [`DEFAULT_BINDING`] when unset.
static BINDING: OnceLock<&'static str> = OnceLock::new();

/// Binding reported until [`set_binding`] runs; see [`set_fallback_binding`].
static FALLBACK_BINDING: OnceLock<&'static str> = OnceLock::new();

/// Default binding identifier reported in the registry telemetry header.
///
/// Each platform binding (Flutter, Kotlin, React Native, Swift, Unity)
/// overrides this via [`set_binding`] so registry calls can be attributed
/// correctly.
pub const DEFAULT_BINDING: &str = "rust";

/// SDK crate version, stamped onto every telemetry event as `sdk_version` and
/// used in the `X-Xybrid-Client` registry header.
///
/// Sourced from `CARGO_PKG_VERSION` at compile time so the value tracks the
/// `xybrid-sdk` crate's own `Cargo.toml` without manual sync. The `xybrid-core`
/// version is exposed separately as [`xybrid_core::VERSION`]; the two can
/// diverge across releases.
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Register the binding identifier for this process.
///
/// Each platform binding (Flutter, Kotlin, React Native, Swift, Unity) calls
/// this once at SDK init. The first call wins — subsequent calls are silent
/// no-ops, which matches the lifecycle (a process is bound to exactly one
/// platform binding). A binding that wraps another SDK (React Native over
/// Swift or Kotlin) registers before the SDK it wraps.
///
/// `RegistryClient` default constructors (`new`, `default_client`,
/// `with_url`, `from_env`) read this value via [`get_binding`], so any
/// `RegistryClient` constructed after [`set_binding`] reports the configured
/// binding in the `X-Xybrid-Client` header without explicit threading.
pub fn set_binding(binding: &'static str) {
    let _ = BINDING.set(binding);
}

/// Set the binding reported until one is registered with [`set_binding`].
///
/// A native library that several SDKs load uses this for the SDK that usually
/// ships it, so an app that never registers is still attributed: `xybrid-bolt`
/// names `swift` on Apple platforms and `kotlin` on Android. Unlike
/// [`set_binding`] it never blocks a registration, so an SDK that registers
/// later (Unity, React Native, Python) reports its own name from then on. The
/// first call wins.
pub fn set_fallback_binding(binding: &'static str) {
    let _ = FALLBACK_BINDING.set(binding);
}

/// Read the process-global binding identifier.
///
/// Returns the value passed to the first [`set_binding`] call, else the one
/// passed to [`set_fallback_binding`], else [`DEFAULT_BINDING`].
pub fn get_binding() -> &'static str {
    BINDING
        .get()
        .or_else(|| FALLBACK_BINDING.get())
        .copied()
        .unwrap_or(DEFAULT_BINDING)
}

/// SDK configuration options.
#[derive(Debug, Clone, Default)]
pub struct SdkConfig {
    /// Custom cache directory (required on Android, optional elsewhere)
    pub cache_dir: Option<std::path::PathBuf>,
    /// Never read: no API takes an `SdkConfig`, so this reaches no request.
    /// Register the binding with [`set_binding`] or [`XybridInit::binding`].
    pub binding: Option<&'static str>,
}

impl SdkConfig {
    /// Store a binding that nothing reads; see [`SdkConfig::binding`].
    #[deprecated(
        note = "has no effect; use xybrid_sdk::set_binding or xybrid_sdk::init().binding()"
    )]
    pub fn with_binding(mut self, binding: &'static str) -> Self {
        self.binding = Some(binding);
        self
    }

    /// The stored binding, or [`DEFAULT_BINDING`]; not what requests report.
    #[deprecated(note = "has no effect; read xybrid_sdk::get_binding instead")]
    pub fn binding(&self) -> &'static str {
        self.binding.unwrap_or(DEFAULT_BINDING)
    }
}

/// Initialize the SDK with a custom cache directory.
///
/// **IMPORTANT**: On Android, this MUST be called before any model loading operations.
/// The cache directory should be obtained from Flutter's `path_provider` package
/// (e.g., `getApplicationDocumentsDirectory()`).
///
/// On other platforms (iOS, macOS, Linux, Windows), this is optional - the SDK
/// will use platform-specific default directories if not configured.
///
/// This function also sets environment variables (`HOME`, `HF_HOME`) to ensure
/// that third-party libraries (like mistralrs/hf-hub) can find cache directories
/// on platforms like Android where standard Unix paths don't exist.
///
/// The first call wins: later calls change neither the folder nor those
/// variables, so an SDK's default never overrides an app's earlier choice.
///
/// # Example (Flutter/Dart)
///
/// ```dart
/// import 'package:path_provider/path_provider.dart';
///
/// Future<void> initXybrid() async {
///   final appDir = await getApplicationDocumentsDirectory();
///   final cacheDir = '${appDir.path}/xybrid/models';
///   initSdkCacheDir(cacheDir);
/// }
/// ```
///
/// # Arguments
///
/// * `cache_dir` - Path to the directory where model bundles will be cached
pub fn init_sdk_cache_dir(cache_dir: impl Into<std::path::PathBuf>) {
    let cache_path = cache_dir.into();
    // `get_or_init` runs this once: a later call changes neither the folder nor
    // the environment that points third-party libraries at it.
    SDK_CONFIG.get_or_init(|| {
        export_cache_env(&cache_path);
        SdkConfig {
            cache_dir: Some(cache_path),
            ..SdkConfig::default()
        }
    });
}

/// Point third-party libraries (hf-hub, mistralrs, ...) at the cache folder.
fn export_cache_env(cache_path: &std::path::Path) {
    // Set environment variables for third-party libraries (hf-hub, mistralrs, etc.)
    // On Android, dirs::home_dir() and dirs::cache_dir() return None, causing panics.
    // Setting these env vars provides fallback paths for those libraries.
    if let Some(cache_str) = cache_path.to_str() {
        // Set HOME if not already set (used by dirs::home_dir() fallback)
        if std::env::var("HOME").is_err() {
            // Use parent of cache dir as HOME
            if let Some(parent) = cache_path.parent() {
                if let Some(parent_str) = parent.to_str() {
                    std::env::set_var("HOME", parent_str);
                }
            }
        }

        // Defensive HF_HOME fallback for HF-aware libraries (and a future
        // mistralrs dep). Takes priority over XDG_CACHE_HOME. Note: the SDK's
        // own HuggingFace loader overrides this via `with_cache_dir(...)`, so
        // SDK downloads land under `<cache_root>/hf-hub`, not here.
        let hf_cache = cache_path.join("huggingface");
        if let Some(hf_str) = hf_cache.to_str() {
            std::env::set_var("HF_HOME", hf_str);
        }

        // Hint offline mode to HF-aware libraries that honor it. Best-effort
        // guard, not a guarantee: the `huggingface` feature's loader downloads
        // deliberately, and not every hf-hub version reads this variable.
        std::env::set_var("HF_HUB_OFFLINE", "1");

        // Also set XDG_CACHE_HOME as a fallback for other XDG-compliant libraries
        if std::env::var("XDG_CACHE_HOME").is_err() {
            std::env::set_var("XDG_CACHE_HOME", cache_str);
        }
    }
}

/// Get the configured cache directory (if set).
pub fn get_sdk_cache_dir() -> Option<std::path::PathBuf> {
    SDK_CONFIG.get().and_then(|c| c.cache_dir.clone())
}

/// Check if the SDK cache directory has been configured.
pub fn is_sdk_cache_configured() -> bool {
    SDK_CONFIG
        .get()
        .and_then(|c| c.cache_dir.as_ref())
        .is_some()
}

/// Set the Xybrid API key for gateway authentication.
///
/// The key is stored in process memory (not the environment) and used by the
/// LLM client when routing through the Xybrid Gateway. Keeping it out of the
/// environment means it is not inherited by child processes the host app
/// spawns. A key set in the real `XYBRID_API_KEY` environment variable
/// (e.g. via the CLI, Flutter `--dart-define`, or iOS `ProcessInfo`) is still
/// honored as a fallback.
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
/// The key persists in memory for the app lifetime.
pub fn set_api_key(api_key: &str) {
    xybrid_core::cloud::set_xybrid_api_key(Some(api_key.to_string()));
}

/// Point the cloud gateway at a specific platform base URL, held in memory.
///
/// Mirrors [`set_api_key`]: the value is stored in a process-memory cell rather
/// than the environment, so it is safe to call after telemetry threads have
/// started (a concurrent `setenv`/`getenv` is UB). Consulted by the gateway
/// ahead of the `XYBRID_PLATFORM_URL` env var; a full gateway URL set via
/// [`set_gateway_url`] or `XYBRID_GATEWAY_URL` (explicit, `/v1`-suffixed)
/// still takes precedence. Pass a bare base URL — the `/v1` suffix is applied
/// internally.
pub fn set_platform_url(url: &str) {
    xybrid_core::cloud::set_xybrid_platform_url(Some(url.to_string()));
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

/// Enable or disable speculative cloud fallback globally.
///
/// When enabled, a [`ModelLoader`] whose model isn't downloaded yet is served
/// from the Xybrid gateway while the weights download in the background —
/// provided a cloud API key is resolvable and the registry reports a chat
/// model (see [`ModelLoader::will_speculate`]). This sets the process-global
/// default; individual loads override it with
/// [`ModelLoader::with_speculative_cloud`]. The flag persists in memory for the
/// app lifetime.
///
/// # Examples
/// ```
/// xybrid_sdk::set_speculative_cloud(true);
/// assert!(xybrid_sdk::is_speculative_cloud_enabled());
/// # xybrid_sdk::set_speculative_cloud(false);
/// ```
pub fn set_speculative_cloud(enabled: bool) {
    SPECULATIVE_CLOUD.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Whether speculative cloud fallback is enabled by the global default.
///
/// Reflects the most recent [`set_speculative_cloud`] call; defaults to
/// `false`. A per-load override via [`ModelLoader::with_speculative_cloud`]
/// takes precedence over this value for that load.
pub fn is_speculative_cloud_enabled() -> bool {
    SPECULATIVE_CLOUD.load(std::sync::atomic::Ordering::Relaxed)
}

/// Process-global speculative-cloud default. See [`set_speculative_cloud`].
static SPECULATIVE_CLOUD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable or disable automatic model release globally.
///
/// When enabled, loading a model while the device reports memory pressure
/// first releases least-recently-used idle models. A released model reloads
/// itself the next time it is used — no reload call is needed and no run
/// starts failing. Individual loads override this with
/// [`ModelLoader::with_auto_release`]. The flag persists in memory for the app
/// lifetime.
///
/// [`release_memory`] ignores this flag: calling it *is* the consent.
///
/// # Examples
/// ```
/// xybrid_sdk::set_auto_release(true);
/// assert!(xybrid_sdk::auto_release_policy().on_pressure);
/// # xybrid_sdk::set_auto_release(false);
/// ```
pub fn set_auto_release(enabled: bool) {
    AUTO_RELEASE.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// The process-global auto-release policy.
///
/// Reflects the most recent [`set_auto_release`] call; defaults to disabled.
/// A per-load override via [`ModelLoader::with_auto_release`] takes precedence
/// for that load.
pub fn auto_release_policy() -> AutoReleasePolicy {
    AutoReleasePolicy::from(AUTO_RELEASE.load(std::sync::atomic::Ordering::Relaxed))
}

/// Process-global auto-release default. See [`set_auto_release`].
static AUTO_RELEASE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Get the currently configured Xybrid API key (if set).
///
/// Returns the key set via [`set_api_key`] (held in memory), falling back to
/// the `XYBRID_API_KEY` environment variable. Returns `None` if neither is
/// configured.
pub fn get_api_key() -> Option<String> {
    xybrid_core::cloud::xybrid_api_key().or_else(|| std::env::var("XYBRID_API_KEY").ok())
}

/// Check if the Xybrid API key is configured.
///
/// Avoids cloning the key (unlike [`get_api_key`]) — checks the in-memory cell
/// then the `XYBRID_API_KEY` environment variable for presence only.
pub fn has_api_key() -> bool {
    xybrid_core::cloud::has_xybrid_api_key() || std::env::var("XYBRID_API_KEY").is_ok()
}

// ============================================================================
// XybridInit — one-stop builder for SDK initialization
// ============================================================================

/// Start configuring the SDK. Call `.run()` to apply.
///
/// See [`XybridInit`] for the full builder surface.
pub fn init() -> XybridInit {
    XybridInit::default()
}

/// Builder that bundles SDK initialization into a single call.
///
/// # Anonymous use
///
/// Omitting [`api_key`](Self::api_key) is supported: every inference path
/// still runs locally.
///
/// # Observability
///
/// Nothing is exported unless [`observability`](Self::observability) is set —
/// an API key alone does not start export. Without it, the first inference
/// logs a one-shot info-level hint; set `XYBRID_QUIET=1` to suppress it.
///
/// # Examples
///
/// Anonymous, default cache directory:
///
/// ```no_run
/// xybrid_sdk::init().run();
/// ```
///
/// Cloud routing, traces to the Xybrid dashboard, resource sampling on:
///
/// ```no_run
/// use xybrid_sdk::observability::ObservabilityConfig;
/// use xybrid_sdk::ResourceTelemetryMode;
///
/// xybrid_sdk::init()
///     .api_key("xy_live_...")
///     .observability(
///         ObservabilityConfig::xybrid("xy_live_...")
///             .resource_telemetry(ResourceTelemetryMode::Summary { interval_ms: 5000 }),
///     )
///     .run();
/// ```
///
/// Everything to an OpenTelemetry Collector:
///
/// ```no_run
/// use xybrid_sdk::observability::{ObservabilityConfig, OtlpConfig};
///
/// xybrid_sdk::init()
///     .observability(ObservabilityConfig::otlp(OtlpConfig::collector("http://localhost:4318")))
///     .run();
/// ```
#[derive(Debug, Clone, Default)]
#[must_use = "call .run() to apply the configuration"]
pub struct XybridInit {
    api_key: Option<String>,
    cache_dir: Option<std::path::PathBuf>,
    gateway_url: Option<String>,
    observability: Option<observability::ObservabilityConfig>,
    resource_telemetry: Option<xybrid_core::device::ResourceTelemetryMode>,
    binding: Option<&'static str>,
}

impl XybridInit {
    /// Set the Xybrid API key used for cloud routing and the registry. It
    /// does not enable observability export.
    ///
    /// Skip this call to run anonymously.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the model cache directory. Required on Android (where
    /// `dirs::cache_dir()` returns `None`); optional elsewhere.
    pub fn cache_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Override the LLM gateway URL for SDK LLM clients and for pipeline
    /// stages routed to the cloud. Pass the full URL including `/v1`.
    /// Defaults to the production gateway or to `XYBRID_GATEWAY_URL` /
    /// `XYBRID_PLATFORM_URL` if set.
    pub fn gateway_url(mut self, url: impl Into<String>) -> Self {
        self.gateway_url = Some(url.into());
        self
    }

    /// Enable observability export (traces, metrics, logs) with `config`.
    pub fn observability(mut self, config: observability::ObservabilityConfig) -> Self {
        self.observability = Some(config);
        self
    }

    /// Configure resource sampling (CPU, memory, thermal, battery) around
    /// runs. Defaults to `Off`. Overrides the mode set on the observability
    /// configuration.
    ///
    /// Without [`Self::observability`] it sets the process-wide mode, used by
    /// operations that select no observability session. An operation pinned
    /// to a session (`ObservationOptions::with_session`) keeps that session's
    /// mode.
    pub fn resource_telemetry(mut self, mode: xybrid_core::device::ResourceTelemetryMode) -> Self {
        self.resource_telemetry = Some(mode);
        self
    }

    /// Register the binding identifier for this process (e.g. `"flutter"`,
    /// `"kotlin"`, `"react-native"`, `"swift"`, `"unity"`). Bindings call
    /// this; host apps rarely need to.
    pub fn binding(mut self, binding: &'static str) -> Self {
        self.binding = Some(binding);
        self
    }

    /// Apply the configuration. Cache dir, gateway URL, API key, and
    /// observability are wired up in the order other code in the SDK expects
    /// them.
    ///
    /// Idempotent for the cache dir and binding (first-set-wins via
    /// `OnceLock`). A second observability configuration replaces the first.
    pub fn run(self) {
        if let Some(binding) = self.binding {
            set_binding(binding);
        }
        if let Some(dir) = self.cache_dir {
            init_sdk_cache_dir(dir);
        }
        if let Some(url) = self.gateway_url.as_deref() {
            set_gateway_url(url);
        }
        if let Some(key) = self.api_key.as_deref() {
            set_api_key(key);
        }
        match (self.observability, self.resource_telemetry) {
            (Some(mut config), mode) => {
                if let Some(mode) = mode {
                    config = config.resource_telemetry(mode);
                }
                if let Err(e) = observability::init_observability(config) {
                    log::warn!(target: "xybrid_sdk", "observability not started: {e}");
                }
            }
            (None, Some(mode)) => telemetry::activate_resource_telemetry(mode),
            (None, None) => {}
        }
    }
}

/// Re-export common types for convenience
pub mod prelude {
    pub use xybrid_core::context::{DeviceMetrics, StageDescriptor};
    pub use xybrid_core::ir::{Envelope, EnvelopeKind};
    pub use xybrid_core::orchestrator::routing_engine::{
        LocalAvailability, RouteTarget, RoutingDecision,
    };
    pub use xybrid_core::orchestrator::{Orchestrator, OrchestratorError, StageExecutionResult};
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
    /// ```no_run
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

/// Pipeline configuration loaded from YAML (legacy format).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyPipelineConfig {
    /// Pipeline name/description
    #[serde(default)]
    name: Option<String>,
    /// List of stage names to execute in order
    stages: Vec<String>,
    /// Input envelope configuration
    input: LegacyInputConfig,
    /// Legacy device-metrics block. Still parsed so existing YAMLs load,
    /// but the values are ignored — capabilities are detected at runtime
    /// and live resource signals come from `ResourceMonitor`.
    #[serde(default)]
    #[allow(dead_code)]
    metrics: Option<serde_yaml::Value>,
    /// Model availability mapping (stage name -> available locally)
    availability: HashMap<String, bool>,
}

/// Input envelope configuration (legacy format).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyInputConfig {
    /// Envelope kind (e.g., "AudioRaw", "Text", etc.)
    kind: String,
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
    /// Reference of the run's `invoke_workflow` operation, when traced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_ref: Option<ObservationRef>,
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
/// ```no_run
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

    let config: LegacyPipelineConfig = serde_yaml::from_str(&config_content)
        .map_err(|e| PipelineConfigError::ConfigParseError(format!("{}: {}", config_path, e)))?;

    // Log pipeline start
    if let Some(name) = &config.name {
        log::info!(target: "xybrid_sdk", "Running pipeline: {}", name);
    } else {
        log::info!(target: "xybrid_sdk", "Running pipeline from: {}", config_path);
    }

    // Build stage descriptors from config
    let stages: Vec<StageDescriptor> = config
        .stages
        .iter()
        .map(|name| StageDescriptor::new(name.clone()))
        .collect();

    log::debug!(target: "xybrid_sdk", "Pipeline has {} stages", stages.len());

    // Create input envelope
    let kind = match config.input.kind.as_str() {
        "Audio" | "audio" => EnvelopeKind::Audio(vec![]),
        "Text" | "text" => EnvelopeKind::Text(String::new()),
        "Embedding" | "embedding" => EnvelopeKind::Embedding(vec![]),
        _ => EnvelopeKind::Text(config.input.kind.clone()),
    };
    let input = Envelope::new(kind);

    // Create device metrics
    let metrics = DeviceMetrics::default();

    // Ignore legacy availability hints because these configs have no resolved
    // bundle paths. Without a bundle path or preloaded adapter, local execution
    // is not runnable.
    if !config.availability.is_empty() {
        log::warn!(
            target: "xybrid_sdk",
            "Legacy pipeline availability hints are ignored; use PipelineRef::load() and load_models() for local execution"
        );
    }
    let availability_fn =
        move |_stage: &str| -> LocalAvailability { LocalAvailability::new(false) };

    // Create orchestrator
    let mut orchestrator = Orchestrator::new();
    let orchestrator_bridge = telemetry::subscribe_orchestrator_events(&orchestrator);

    // Execute the pipeline as one observed workflow.
    let start_time = std::time::Instant::now();
    let workflow = observability::WorkflowScope::start(
        config.name.as_deref().unwrap_or("pipeline"),
        &RunOptions::default(),
    );
    let executed = execute_legacy_stages(
        &mut orchestrator,
        &stages,
        &input,
        &metrics,
        &availability_fn,
        workflow.context(),
    );
    let observation_ref = workflow.end(
        &executed
            .as_ref()
            .map(|_| ())
            .map_err(|e| SdkError::pipeline(e.to_string())),
    );
    let results: Vec<StageExecutionResult> = executed.map_err(|e| {
        orchestrator_bridge.drain();
        PipelineConfigError::ExecutionError(format!("{}", e))
    })?;
    orchestrator_bridge.drain();
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

    log::info!(target: "xybrid_sdk", "Pipeline completed in {}ms", total_latency_ms);

    Ok(PipelineResult {
        name: config.name.clone(),
        stages: stage_timings,
        total_latency_ms,
        final_output,
        observation_ref,
    })
}

/// Runs legacy stages through the routed stage observation path used by
/// [`PipelineRef`] runs: once a stage's route is known it becomes a stage
/// inference (with its execution attempt) under `parent`, so legacy runs get
/// the same stage operations, usage and failure classes.
fn execute_legacy_stages(
    orchestrator: &mut Orchestrator,
    stages: &[StageDescriptor],
    input: &Envelope,
    metrics: &DeviceMetrics,
    availability_fn: &dyn Fn(&str) -> LocalAvailability,
    parent: Option<ObservationContext>,
) -> xybrid_core::orchestrator::OrchestratorResult<Vec<StageExecutionResult>> {
    let options = RunOptions::default();
    orchestrator.execute_pipeline_with_stage(
        stages,
        input,
        metrics,
        availability_fn,
        |orchestrator, stage, input, metrics, availability| {
            let mut observation = None;
            let result = orchestrator.execute_stage_routed(
                stage,
                input,
                metrics,
                availability,
                &mut |route| {
                    observation = Some(observability::StageObservation::start(
                        stage,
                        input,
                        route,
                        &options,
                        parent.clone(),
                        observability::PipelinePurpose::Run,
                    ));
                },
            );
            if let Some(observation) = observation {
                observation.finish(&result);
            }
            result
        },
    )
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
/// ```no_run
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

    let config: LegacyPipelineConfig = serde_yaml::from_str(&config_content)
        .map_err(|e| PipelineConfigError::ConfigParseError(format!("{}: {}", config_path, e)))?;

    // Log pipeline start
    if let Some(name) = &config.name {
        log::info!(target: "xybrid_sdk", "Running pipeline (async): {}", name);
    } else {
        log::info!(target: "xybrid_sdk", "Running pipeline (async) from: {}", config_path);
    }

    // Build stage descriptors from config
    let stages: Vec<StageDescriptor> = config
        .stages
        .iter()
        .map(|name| StageDescriptor::new(name.clone()))
        .collect();

    log::debug!(target: "xybrid_sdk", "Pipeline has {} stages", stages.len());

    // Create input envelope
    let kind = match config.input.kind.as_str() {
        "Audio" | "audio" => EnvelopeKind::Audio(vec![]),
        "Text" | "text" => EnvelopeKind::Text(String::new()),
        "Embedding" | "embedding" => EnvelopeKind::Embedding(vec![]),
        _ => EnvelopeKind::Text(config.input.kind.clone()),
    };
    let input = Envelope::new(kind);

    // Create device metrics
    let metrics = DeviceMetrics::default();

    // Ignore legacy availability hints because these configs have no resolved
    // bundle paths. Without a bundle path or preloaded adapter, local execution
    // is not runnable.
    if !config.availability.is_empty() {
        log::warn!(
            target: "xybrid_sdk",
            "Legacy pipeline availability hints are ignored; use PipelineRef::load() and load_models() for local execution"
        );
    }
    let availability_fn =
        move |_stage: &str| -> LocalAvailability { LocalAvailability::new(false) };

    // Create orchestrator
    let mut orchestrator = Orchestrator::new();
    let orchestrator_bridge = telemetry::subscribe_orchestrator_events(&orchestrator);

    // Execute the pipeline asynchronously as one observed workflow; its
    // context is installed on every poll, never held across an await. The
    // stages run on the blocking pool through the same observed, routed path
    // as `run_pipeline`, so their stage spans (the `pipeline:` span included)
    // open and close inside one synchronous call rather than across polls.
    let start_time = std::time::Instant::now();
    let workflow = observability::AsyncWorkflow::start(
        config.name.as_deref().unwrap_or("pipeline"),
        &RunOptions::default(),
    );
    let parent = workflow.context();
    let executed = workflow
        .instrument(async move {
            let recorder = xybrid_core::tracing::current_recorder();
            tokio::task::spawn_blocking(move || {
                let _scope = recorder.map(xybrid_core::tracing::RecorderScope::install);
                let executed = execute_legacy_stages(
                    &mut orchestrator,
                    &stages,
                    &input,
                    &metrics,
                    &availability_fn,
                    parent,
                )
                .map_err(|e| e.to_string());
                // Keep the orchestrator (and its event bus) alive until the
                // bridge has drained below.
                (orchestrator, executed)
            })
            .await
        })
        .await;
    let (_orchestrator, executed) = match executed {
        Ok(done) => done,
        Err(join) => {
            let error = format!("pipeline task failed: {join}");
            workflow.end(&Err::<(), _>(&error));
            orchestrator_bridge.drain();
            return Err(PipelineConfigError::ExecutionError(error));
        }
    };
    let observation_ref = workflow.end(&executed);
    let results: Vec<StageExecutionResult> = executed.map_err(|e| {
        orchestrator_bridge.drain();
        PipelineConfigError::ExecutionError(e)
    })?;
    orchestrator_bridge.drain();
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

    log::info!(target: "xybrid_sdk", "Pipeline completed in {}ms", total_latency_ms);

    Ok(PipelineResult {
        name: config.name.clone(),
        stages: stage_timings,
        total_latency_ms,
        final_output,
        observation_ref,
    })
}

#[cfg(test)]
mod sdk_config_tests {
    use super::DEFAULT_BINDING;

    #[test]
    fn default_binding_is_rust() {
        assert_eq!(DEFAULT_BINDING, "rust");
    }

    #[test]
    fn sdk_version_is_stamped_by_the_build() {
        // rules_rust fills CARGO_PKG_VERSION with "0.0.0" unless the target
        // passes `version` (//bazel:cargo_version.bzl). Running under both
        // cargo and `bazel test`, this proves the version reaches the crate;
        // that EVERY target passes it is a `bazel query` check in bazel.yml.
        assert_ne!(crate::SDK_VERSION, "0.0.0");
    }
}

#[cfg(test)]
mod xybrid_init_tests {
    use super::{init, XybridInit};
    use xybrid_core::device::ResourceTelemetryMode;

    #[test]
    fn init_returns_default_anonymous_builder() {
        let builder = init();
        let default = XybridInit::default();
        // Default builder carries no configuration — the anonymous path.
        assert_eq!(format!("{:?}", builder), format!("{:?}", default));
    }

    #[test]
    fn api_key_setter_stores_value() {
        let builder = init().api_key("xy_test_123");
        let dbg = format!("{:?}", builder);
        assert!(dbg.contains("xy_test_123"), "debug = {}", dbg);
    }

    #[test]
    fn cache_dir_setter_stores_path() {
        let builder = init().cache_dir("/tmp/xybrid-test-cache");
        let dbg = format!("{:?}", builder);
        assert!(
            dbg.contains("xybrid-test-cache"),
            "cache_dir not stored, debug = {}",
            dbg
        );
    }

    #[test]
    fn observability_setter_stores_config() {
        let builder = init().observability(crate::observability::ObservabilityConfig::otlp(
            crate::observability::OtlpConfig::collector("http://192.168.1.78:4318"),
        ));
        let dbg = format!("{:?}", builder);
        assert!(dbg.contains("192.168.1.78"), "debug = {}", dbg);
    }

    #[test]
    fn api_key_alone_does_not_enable_observability() {
        let builder = init().api_key("xy_test_123");
        let dbg = format!("{:?}", builder);
        assert!(dbg.contains("observability: None"), "debug = {}", dbg);
    }

    #[test]
    fn gateway_url_setter_stores_value() {
        let builder = init().gateway_url("https://gateway.example/v1");
        let dbg = format!("{:?}", builder);
        assert!(dbg.contains("gateway.example"), "debug = {}", dbg);
    }

    #[test]
    fn resource_telemetry_setter_stores_mode() {
        let builder = init().resource_telemetry(ResourceTelemetryMode::Boundary);
        let dbg = format!("{:?}", builder);
        assert!(dbg.contains("Boundary"), "debug = {}", dbg);
    }

    #[test]
    fn builder_is_chainable() {
        let _builder = init()
            .api_key("xy_test")
            .cache_dir("/tmp/x")
            .gateway_url("https://gateway")
            .resource_telemetry(ResourceTelemetryMode::Off)
            .binding("rust");
    }
}

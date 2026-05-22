#![allow(unpredictable_function_pointer_comparisons)]

//! UniFFI bindings for xybrid-sdk.
//!
//! This crate exposes xybrid-sdk types and functions to Swift and Kotlin
//! via UniFFI code generation.

use std::collections::HashMap;
use std::sync::Arc;

uniffi::setup_scaffolding!();

use xybrid_ffi_facade as facade;
use xybrid_sdk::{
    ir::{Envelope as CoreEnvelope, EnvelopeKind as CoreEnvelopeKind},
    InferenceResult as CoreInferenceResult, ModelLoader as CoreModelLoader, SdkError,
    XybridModel as CoreXybridModel,
};

/// Initialize the SDK cache directory.
///
/// On Android, this MUST be called before any model loading operations.
/// The Kotlin SDK wrapper `Xybrid.init(context)` calls this automatically.
#[uniffi::export]
fn init_sdk_cache_dir(cache_dir: String) {
    facade::init_sdk_cache_dir(cache_dir);
}

/// Register the binding identifier for this process.
///
/// The xybrid-uniffi crate is shared by both Kotlin and Swift, so the
/// identity must be supplied by the platform-side wrapper at SDK init —
/// the Kotlin `Xybrid.init(...)` calls `setBinding("kotlin")`, and the
/// Swift `Xybrid.initialize()` calls `setBinding(binding: "swift")`.
///
/// Routes through [`facade::set_binding`], which collapses unknown values
/// to [`xybrid_sdk::DEFAULT_BINDING`] to bound cardinality on the registry
/// side — same defensive shape as `build_client_header`'s sanitizer.
///
/// First call wins (process-global `OnceLock` in xybrid-sdk); subsequent
/// calls are silent no-ops.
#[uniffi::export]
fn set_binding(binding: String) {
    facade::set_binding(binding);
}

// -- Platform-state push API --
//
// Mobile telemetry APIs (`UIDevice.batteryLevel` on iOS,
// `BatteryManager.ACTION_BATTERY_CHANGED` on Android,
// `PowerManager.OnThermalStatusChangedListener` on Android) are
// notification-based and live in the host runtime — there is no clean
// in-Rust path on those platforms. The host SDK wrappers
// (`Xybrid.init(context)` on Kotlin, `Xybrid.initialize()` on Swift)
// register the OS observers and forward each value through these FFI
// calls. The Rust side just stores into the same `RwLock<PlatformState>`
// the desktop pollers feed, so routing decisions are uniform across
// platforms.
//
// One-way push (host → Rust) is intentional: a callback-interface design
// would marshal a `Context` / `NotificationCenter` handle across the
// boundary and re-enter Rust on every change, which is much more surface
// for marginal benefit.

/// Thermal pressure state forwarded by the host.
///
/// Maps directly to [`xybrid_sdk::ThermalState`] — mirrored here as a
/// UniFFI-exposed enum so Swift gets `enum XybridThermalState` and
/// Kotlin gets `enum class XybridThermalState`. Variants are documented
/// with the same Celsius bands as the desktop pollers so host code can
/// quantize the OS signal consistently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
#[non_exhaustive]
pub enum XybridThermalState {
    /// Normal operating temperature (< 60 °C). No throttling expected.
    Normal,
    /// Warm — first throttling tier (60–70 °C).
    Warm,
    /// Hot — performance reduced (70–80 °C).
    Hot,
    /// Critical — heavy operations should pause (> 80 °C).
    Critical,
}

impl From<XybridThermalState> for facade::ThermalState {
    fn from(value: XybridThermalState) -> Self {
        match value {
            XybridThermalState::Normal => facade::ThermalState::Normal,
            XybridThermalState::Warm => facade::ThermalState::Warm,
            XybridThermalState::Hot => facade::ThermalState::Hot,
            XybridThermalState::Critical => facade::ThermalState::Critical,
        }
    }
}

/// Forward a battery charge percentage (0..=100) from the host.
///
/// Values above 100 are clamped by the underlying setter — pass through
/// whatever the OS observer reports without rounding host-side, so the
/// SDK has the freshest possible signal.
#[uniffi::export]
fn set_battery_level(percent: u8) {
    facade::set_battery_level(percent);
}

/// Mark the battery level as unknown.
///
/// Hosts call this on observer teardown or when the OS reports an
/// unknown / unavailable charge (e.g. desktop docks without battery
/// sensors). The routing engine treats `None` as "no signal" rather
/// than substituting an optimistic default.
#[uniffi::export]
fn clear_battery_level() {
    facade::clear_battery_level();
}

/// Forward a thermal pressure reading from the host.
#[uniffi::export]
fn set_thermal_state(state: XybridThermalState) {
    facade::set_thermal_state(state.into());
}

/// Mark the thermal state as unknown.
#[uniffi::export]
fn clear_thermal_state() {
    facade::clear_thermal_state();
}

/// Error type exposed via UniFFI to Swift/Kotlin consumers.
///
/// This enum represents all possible errors that can occur during
/// xybrid operations, allowing consumers to handle errors appropriately.
///
/// In Swift this becomes an `enum XybridError: Error` with associated values.
/// In Kotlin this becomes a `sealed class XybridException : Exception()`.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum XybridError {
    #[error("Model not found: {message}")]
    ModelNotFound { message: String },
    #[error("Directory not found: {message}")]
    DirectoryNotFound { message: String },
    #[error("model_metadata.json not found in directory: {message}")]
    MetadataNotFound { message: String },
    #[error("model_metadata.json is invalid: {message}")]
    MetadataInvalid { message: String },
    #[error("Failed to load model: {message}")]
    LoadError { message: String },
    #[error("Inference failed: {message}")]
    InferenceError { message: String },
    #[error("Streaming not supported by this model")]
    StreamingNotSupported,
    #[error("Model not loaded")]
    NotLoaded,
    #[error("Invalid configuration: {message}")]
    ConfigError { message: String },
    #[error("Network error: {message}")]
    NetworkError { message: String },
    #[error("IO error: {message}")]
    IoError { message: String },
    #[error("Cache error: {message}")]
    CacheError { message: String },
    #[error("Pipeline error: {message}")]
    PipelineError { message: String },
    #[error("Circuit breaker open: {message}")]
    CircuitOpen { message: String },
    #[error("Rate limited, retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },
    #[error("Request timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
}

impl From<facade::Error> for XybridError {
    /// Map the canonical facade error (one definition for the whole
    /// workspace, lives in [`xybrid_ffi_facade`]) into the UniFFI-exposed
    /// enum. The SDK→facade leg is owned by the facade crate; this leg
    /// only handles the uniffi-specific shape decisions:
    ///
    /// - `Offline` collapses into `NetworkError` — the Swift/Kotlin
    ///   public API committed to a fixed variant set in
    ///   `docs/sdk/api-surface.yaml`. Adding a new variant here would
    ///   break the generated sealed/enum hierarchies and needs a
    ///   spec-first contract update.
    /// - `AbortedForCloudFallback` collapses into `InferenceError` with
    ///   a formatted message — same backwards-compat reason.
    fn from(e: facade::Error) -> Self {
        match e {
            facade::Error::ModelNotFound { id } => XybridError::ModelNotFound { message: id },
            facade::Error::DirectoryNotFound { path } => {
                XybridError::DirectoryNotFound { message: path }
            }
            facade::Error::MetadataNotFound { path } => {
                XybridError::MetadataNotFound { message: path }
            }
            facade::Error::MetadataInvalid { message } => XybridError::MetadataInvalid { message },
            facade::Error::LoadError { message } => XybridError::LoadError { message },
            facade::Error::InferenceError { message } => XybridError::InferenceError { message },
            facade::Error::AbortedForCloudFallback { reason } => XybridError::InferenceError {
                message: format!("Aborted for cloud fallback: {reason}"),
            },
            facade::Error::StreamingNotSupported => XybridError::StreamingNotSupported,
            facade::Error::NotLoaded => XybridError::NotLoaded,
            facade::Error::ConfigError { message } => XybridError::ConfigError { message },
            facade::Error::NetworkError { message } | facade::Error::Offline { message } => {
                XybridError::NetworkError { message }
            }
            facade::Error::IoError { message } => XybridError::IoError { message },
            facade::Error::CacheError { message } => XybridError::CacheError { message },
            facade::Error::PipelineError { message } => XybridError::PipelineError { message },
            facade::Error::CircuitOpen { message } => XybridError::CircuitOpen { message },
            facade::Error::RateLimited { retry_after_secs } => {
                XybridError::RateLimited { retry_after_secs }
            }
            facade::Error::Timeout { timeout_ms } => XybridError::Timeout { timeout_ms },
        }
    }
}

impl From<SdkError> for XybridError {
    /// Routes through the facade's canonical `From<SdkError>` map so the
    /// SDK variant ↔ FFI variant pairing lives in exactly one place
    /// ([`xybrid_ffi_facade::Error::from`]) rather than being duplicated
    /// across every binding crate. Adding a new `SdkError` variant only
    /// requires updating the facade map and (if it surfaces a new
    /// user-visible category) the [`From<facade::Error>`] arm above.
    fn from(e: SdkError) -> Self {
        facade::Error::from(e).into()
    }
}

/// Generation parameters for LLM inference.
///
/// All fields are optional. When `None`, the model's default value is used.
///
/// In Kotlin: `XybridGenerationConfig(temperature = 0.3f, maxTokens = 512u)`
/// In Swift: `XybridGenerationConfig(temperature: 0.3, maxTokens: 512)`
#[derive(uniffi::Record, Clone)]
pub struct XybridGenerationConfig {
    /// Maximum tokens to generate. Default: 2048
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 = deterministic, higher = more random). Default: 0.7
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold. Default: 0.9
    pub top_p: Option<f32>,
    /// Min-p sampling threshold. Default: 0.05
    pub min_p: Option<f32>,
    /// Top-k sampling (0 = disabled). Default: 40
    pub top_k: Option<u32>,
    /// Repetition penalty (1.0 = disabled). Default: 1.1
    pub repetition_penalty: Option<f32>,
    /// Stop sequences. When `None` or empty, only EOS token stops generation.
    pub stop_sequences: Option<Vec<String>>,
}

impl XybridGenerationConfig {
    /// Re-shape into the facade POD; the facade owns the single canonical
    /// "option overrides → SDK defaults" mapping that this used to
    /// duplicate inline.
    fn to_facade(&self) -> facade::GenerationConfig {
        facade::GenerationConfig {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            stop_sequences: self.stop_sequences.clone().unwrap_or_default(),
        }
    }

    fn to_sdk(&self) -> xybrid_sdk::GenerationConfig {
        self.to_facade().to_sdk()
    }
}

/// Envelope type for passing data to xybrid models.
///
/// This enum represents the different types of input that can be passed
/// to xybrid models for inference. Each variant contains the data and
/// associated metadata needed for that input type.
#[derive(uniffi::Enum, Debug, Clone)]
pub enum XybridEnvelope {
    /// Audio input for ASR (speech-to-text) models.
    Audio {
        /// Raw audio bytes (typically PCM or WAV format).
        bytes: Vec<u8>,
        /// Sample rate in Hz (e.g., 16000, 44100).
        sample_rate: u32,
        /// Number of audio channels (1 = mono, 2 = stereo).
        channels: u32,
    },
    /// Text input for TTS (text-to-speech) or LLM models.
    Text {
        /// The text content to process.
        text: String,
        /// Optional voice ID for TTS models.
        voice_id: Option<String>,
        /// Optional speech speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed).
        speed: Option<f64>,
    },
    /// Embedding vector for similarity search or downstream models.
    Embedding {
        /// The embedding vector as f32 values.
        data: Vec<f32>,
    },
}

/// Per-stage latency entry for pipeline runs.
///
/// One entry per executed stage; `stage_id` matches the stage name in the
/// pipeline definition.
#[derive(uniffi::Record, Clone)]
pub struct XybridStageLatency {
    pub stage_id: String,
    pub latency_ms: u32,
}

impl From<&facade::StageLatency> for XybridStageLatency {
    fn from(s: &facade::StageLatency) -> Self {
        Self {
            stage_id: s.stage_id.clone(),
            latency_ms: s.latency_ms,
        }
    }
}

/// Typed inference metrics surfaced on every `XybridResult`.
///
/// LLM-specific fields (`ttft_ms`, `tokens_per_second`, `prefill_tps`,
/// `decode_tps`, `tokens_in`, `tokens_out`) are `None` for ASR/TTS/embedding
/// runs. `stage_latencies_ms` is empty for `model.run()` and populated for
/// `pipeline.run()`.
#[derive(uniffi::Record, Clone)]
pub struct XybridInferenceMetrics {
    pub total_ms: u32,
    pub ttft_ms: Option<u32>,
    pub tokens_per_second: Option<f32>,
    pub prefill_tps: Option<f32>,
    pub decode_tps: Option<f32>,
    pub tokens_out: Option<u32>,
    pub stage_latencies_ms: Vec<XybridStageLatency>,
}

impl From<&facade::InferenceMetrics> for XybridInferenceMetrics {
    fn from(m: &facade::InferenceMetrics) -> Self {
        Self {
            total_ms: m.total_ms,
            ttft_ms: m.ttft_ms,
            tokens_per_second: m.tokens_per_second,
            prefill_tps: m.prefill_tps,
            decode_tps: m.decode_tps,
            tokens_out: m.tokens_out,
            stage_latencies_ms: m.stage_latencies_ms.iter().map(Into::into).collect(),
        }
    }
}

/// Result type returned from xybrid model inference.
///
/// This struct contains the output from running inference on a model,
/// including success/failure status, output data, and timing information.
#[derive(uniffi::Record, Clone)]
pub struct XybridResult {
    pub success: bool,
    pub text: Option<String>,
    pub audio_bytes: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
    pub latency_ms: u32,
    pub metrics: XybridInferenceMetrics,
}

impl XybridResult {
    /// Build from the SDK type by routing through the facade — the
    /// payload-extraction logic (text / audio / embedding accessors,
    /// metrics conversion) is owned by `facade::InferenceResult` so we
    /// don't carry per-binding copies.
    pub(crate) fn from_inference_result(r: CoreInferenceResult) -> Self {
        let facade_result = facade::InferenceResult::from_sdk(r);
        let metrics = XybridInferenceMetrics::from(&facade_result.metrics);
        let (text, audio_bytes, embedding) = match facade_result.envelope.kind {
            facade::EnvelopeKind::Text { text } => (Some(text), None, None),
            facade::EnvelopeKind::Audio { bytes } => (None, Some(bytes), None),
            facade::EnvelopeKind::Embedding { values } => (None, None, Some(values)),
        };
        Self {
            success: true,
            text,
            audio_bytes,
            embedding,
            latency_ms: facade_result.latency_ms,
            metrics,
        }
    }
}

/// Voice metadata for TTS models.
///
/// Describes a single voice available in a TTS model's voice catalog.
/// Use `XybridModel.voices()` to list all available voices.
///
/// In Swift this becomes a `struct XybridVoiceInfo`.
/// In Kotlin this becomes a `data class XybridVoiceInfo`.
#[derive(uniffi::Record, Clone)]
pub struct XybridVoiceInfo {
    /// Unique voice identifier (e.g., "af_bella").
    pub id: String,
    /// Human-readable display name (e.g., "Bella").
    pub name: String,
    /// Gender: "male", "female", or "neutral".
    pub gender: Option<String>,
    /// BCP-47 language tag (e.g., "en-US", "en-GB").
    pub language: Option<String>,
    /// Style descriptor (e.g., "neutral", "cheerful").
    pub style: Option<String>,
}

impl From<facade::VoiceInfo> for XybridVoiceInfo {
    fn from(v: facade::VoiceInfo) -> Self {
        Self {
            id: v.id,
            name: v.name,
            gender: v.gender,
            language: v.language,
            style: v.style,
        }
    }
}

impl From<XybridEnvelope> for CoreEnvelope {
    fn from(envelope: XybridEnvelope) -> Self {
        match envelope {
            XybridEnvelope::Audio {
                bytes,
                sample_rate,
                channels,
            } => {
                let mut metadata = HashMap::new();
                metadata.insert("sample_rate".to_string(), sample_rate.to_string());
                metadata.insert("channels".to_string(), channels.to_string());
                CoreEnvelope::with_metadata(CoreEnvelopeKind::Audio(bytes.clone()), metadata)
            }
            XybridEnvelope::Text {
                text,
                voice_id,
                speed,
            } => {
                let mut metadata = HashMap::new();
                if let Some(voice) = voice_id {
                    metadata.insert("voice_id".to_string(), voice.clone());
                }
                if let Some(s) = speed {
                    metadata.insert("speed".to_string(), s.to_string());
                }
                CoreEnvelope::with_metadata(CoreEnvelopeKind::Text(text.clone()), metadata)
            }
            XybridEnvelope::Embedding { data } => {
                CoreEnvelope::new(CoreEnvelopeKind::Embedding(data.clone()))
            }
        }
    }
}

/// A loaded xybrid model ready for inference.
///
/// This object represents a model that has been loaded and is ready to run
/// inference. Use `XybridModelLoader` to obtain instances of this type.
#[derive(uniffi::Object)]
pub struct XybridModel {
    /// Internal model state.
    inner: CoreXybridModel,
}

#[uniffi::export(async_runtime = "tokio")]
impl XybridModel {
    /// Run inference on this model with the provided input envelope.
    ///
    /// Pass an optional `config` to control generation parameters (temperature, top-p, etc.).
    /// When `None`, the model's default parameters are used.
    pub async fn run(
        &self,
        envelope: XybridEnvelope,
        config: Option<XybridGenerationConfig>,
    ) -> Result<XybridResult, XybridError> {
        let sdk_config = config.as_ref().map(|c| c.to_sdk());
        let result = self
            .inner
            .run_async(&envelope.into(), sdk_config.as_ref())
            .await
            .map_err(XybridError::from)?;
        Ok(XybridResult::from_inference_result(result))
    }

    /// Get all available voices for this TTS model.
    ///
    /// Returns `None` for non-TTS models or models without voice configuration.
    pub fn voices(&self) -> Option<Vec<XybridVoiceInfo>> {
        self.inner.voices().map(|vs| {
            vs.into_iter()
                .map(facade::VoiceInfo::from_sdk)
                .map(XybridVoiceInfo::from)
                .collect()
        })
    }

    /// Get the default voice ID for this TTS model.
    ///
    /// Returns `None` for non-TTS models or models without voice configuration.
    pub fn default_voice_id(&self) -> Option<String> {
        self.inner.voice_config().map(|vc| vc.default)
    }

    /// Check if this model has voice support.
    pub fn has_voices(&self) -> bool {
        self.inner.has_voices()
    }

    /// Get a specific voice by ID.
    ///
    /// Returns `None` if the voice is not found or the model has no voice support.
    pub fn voice(&self, voice_id: String) -> Option<XybridVoiceInfo> {
        self.inner
            .voice(&voice_id)
            .map(facade::VoiceInfo::from_sdk)
            .map(XybridVoiceInfo::from)
    }
}

/// A model loader for loading xybrid models from registry, bundles, or directories.
///
/// Use the constructors to create a loader pointing to a model source,
/// then call `load()` to actually load the model for inference.
///
/// # Example (Swift)
///
/// ```swift
/// // Load from registry
/// let loader = XybridModelLoader.fromRegistry(modelId: "whisper-tiny")
/// let model = try await loader.load()
///
/// // Load from local bundle
/// let bundleLoader = XybridModelLoader.fromBundle(path: "/path/to/model.xyb")
/// let bundleModel = try await bundleLoader.load()
///
/// // Load from a directory with model_metadata.json
/// let dirLoader = try XybridModelLoader.fromDirectory(path: "/path/to/model/")
/// let dirModel = try await dirLoader.load()
/// ```
#[derive(uniffi::Object)]
pub struct XybridModelLoader {
    /// Internal loader state.
    inner: CoreModelLoader,
}

#[uniffi::export(async_runtime = "tokio")]
impl XybridModelLoader {
    /// Create a model loader that will load from the xybrid model registry.
    ///
    /// The model will be downloaded from the registry if not already cached.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The unique identifier of the model (e.g., "whisper-tiny", "kokoro-82m").
    ///
    /// # Returns
    ///
    /// A new `XybridModelLoader` instance configured to load from the registry.
    #[uniffi::constructor]
    pub fn from_registry(model_id: String) -> Arc<Self> {
        Arc::new(Self {
            inner: CoreModelLoader::from_registry(model_id.as_str()),
        })
    }

    /// Create a model loader that will load from a local bundle file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to the model bundle (.xyb file or directory).
    ///
    /// # Returns
    ///
    /// A new `XybridModelLoader` instance configured to load from the bundle.
    #[uniffi::constructor]
    pub fn from_bundle(path: String) -> Arc<Self> {
        Arc::new(Self {
            inner: CoreModelLoader::from_bundle(&path).unwrap(),
        })
    }

    /// Create a model loader that will load from a local directory containing
    /// model files and a `model_metadata.json`.
    ///
    /// The directory must contain a valid `model_metadata.json` file that
    /// describes the model's execution template, preprocessing, and
    /// postprocessing steps.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to the directory containing the model files.
    ///
    /// # Returns
    ///
    /// A new `XybridModelLoader` instance, or a `XybridError` if the
    /// directory does not exist, or the metadata file is missing or invalid.
    ///
    /// # Example (Swift)
    ///
    /// ```swift
    /// let loader = try XybridModelLoader.fromDirectory(path: "/path/to/model/")
    /// let model = try await loader.load()
    /// ```
    #[uniffi::constructor]
    pub fn from_directory(path: String) -> Result<Arc<Self>, XybridError> {
        let inner = CoreModelLoader::from_directory(&path)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Create a model loader that will download from a HuggingFace Hub repository.
    ///
    /// Downloads model files from HuggingFace and caches them locally.
    /// Model metadata is auto-generated if not present in the repository.
    ///
    /// Requires the `huggingface` feature flag.
    ///
    /// # Arguments
    ///
    /// * `repo` - The HuggingFace repository ID (e.g., "xybrid-ai/kokoro-82m").
    ///
    /// # Returns
    ///
    /// A new `XybridModelLoader` instance configured to download from HuggingFace.
    ///
    /// # Example (Swift)
    ///
    /// ```swift
    /// let loader = XybridModelLoader.fromHuggingface(repo: "xybrid-ai/kokoro-82m")
    /// let model = try await loader.load()
    /// ```
    #[uniffi::constructor]
    pub fn from_huggingface(repo: String) -> Arc<Self> {
        Arc::new(Self {
            inner: CoreModelLoader::from_huggingface(&repo),
        })
    }

    /// Load the model and prepare it for inference.
    ///
    /// This method downloads the model if needed (for registry sources),
    /// loads the model files, and initializes the runtime for inference.
    ///
    /// # Returns
    ///
    /// An `Arc<XybridModel>` ready for inference, or a `XybridError` if loading fails.
    ///
    /// # Example (Swift)
    ///
    /// ```swift
    /// let loader = XybridModelLoader.fromRegistry(modelId: "whisper-tiny")
    /// do {
    ///     let model = try loader.load()
    ///     // model is now ready for inference
    /// } catch {
    ///     print("Failed to load model: \(error)")
    /// }
    /// ```
    pub async fn load(&self) -> Result<Arc<XybridModel>, XybridError> {
        let model = self.inner.load_async().await?;
        Ok(Arc::new(XybridModel { inner: model }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Single combined integration test: the binding is process-global via
    // OnceLock, so splitting into multiple tests that call `set_binding`
    // would race on which one observes the first set. The Kotlin path is
    // the canonical wire-through; resolution of unknown / Swift platform
    // strings is covered by the facade's own test suite.
    #[test]
    fn set_binding_kotlin_registers_kotlin_binding() {
        // Kotlin wrapper calls this from Xybrid.init().
        set_binding("kotlin".to_string());

        // Process-global binding now resolves to "kotlin".
        assert_eq!(xybrid_sdk::get_binding(), "kotlin");

        // RegistryClient default constructors pick up the configured binding,
        // so the X-Xybrid-Client header on every metadata call from a Kotlin
        // app will report binding=kotlin.
        let client = xybrid_sdk::RegistryClient::default_client()
            .expect("default_client should succeed in tests");
        assert_eq!(client.binding(), "kotlin");

        // OnceLock first-set-wins: a later call (e.g. from the Swift wrapper
        // running in the same process, or a misbehaving consumer) cannot
        // overwrite the registered identity.
        set_binding("swift".to_string());
        assert_eq!(xybrid_sdk::get_binding(), "kotlin");

        // Unknown values must not propagate raw to the registry header. The
        // OnceLock is already set so behavior is unobservable here, but
        // the wire-through call still goes through the facade's closed
        // match — the `_ => DEFAULT_BINDING` branch protects a cold-start
        // process from header pollution and is exercised directly by the
        // facade's own test suite.
        set_binding("evil_unknown".to_string());
        assert_eq!(xybrid_sdk::get_binding(), "kotlin");
    }

    // Pure conversion tests for XybridThermalState. The push setters
    // themselves write into a process-global RwLock that other tests
    // (and other crates' integration tests) also touch — covering the
    // mapping at the conversion layer keeps these tests deterministic
    // regardless of test ordering.
    #[test]
    fn thermal_state_maps_to_facade_variants() {
        assert_eq!(
            facade::ThermalState::from(XybridThermalState::Normal),
            facade::ThermalState::Normal
        );
        assert_eq!(
            facade::ThermalState::from(XybridThermalState::Warm),
            facade::ThermalState::Warm
        );
        assert_eq!(
            facade::ThermalState::from(XybridThermalState::Hot),
            facade::ThermalState::Hot
        );
        assert_eq!(
            facade::ThermalState::from(XybridThermalState::Critical),
            facade::ThermalState::Critical
        );
    }

    // Confirm the SDK→XybridError leg routes through the facade. Spot-check
    // a couple of variants (full coverage lives in the facade's tests).
    #[test]
    fn xybrid_error_from_sdk_routes_via_facade() {
        let sdk_err = xybrid_sdk::SdkError::ModelNotFound("foo".to_string());
        match XybridError::from(sdk_err) {
            XybridError::ModelNotFound { message } => assert_eq!(message, "foo"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }

        // SdkError::Offline → XybridError::NetworkError is the documented
        // ABI-compat collapse; protect it explicitly so an accidental
        // facade re-shape doesn't quietly break the Swift/Kotlin sealed
        // hierarchy.
        let sdk_err = xybrid_sdk::SdkError::Offline("dns".to_string());
        match XybridError::from(sdk_err) {
            XybridError::NetworkError { message } => assert_eq!(message, "dns"),
            other => panic!("expected NetworkError, got {other:?}"),
        }
    }
}

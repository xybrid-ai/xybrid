#![allow(unpredictable_function_pointer_comparisons)]

//! UniFFI bindings for xybrid-sdk.
//!
//! This crate exposes xybrid-sdk types and functions to Swift and Kotlin
//! via UniFFI code generation.

use std::collections::HashMap;
use std::sync::Arc;

uniffi::setup_scaffolding!();

use xybrid_sdk::{
    ir::{Envelope as CoreEnvelope, EnvelopeKind as CoreEnvelopeKind},
    InferenceResult as CoreInferenceResult, ModelLoader as CoreModelLoader, SdkError,
    VoiceInfo as CoreVoiceInfo, XybridModel as CoreXybridModel,
};

/// Initialize the SDK cache directory.
///
/// On Android, this MUST be called before any model loading operations.
/// The Kotlin SDK wrapper `Xybrid.init(context)` calls this automatically.
#[uniffi::export]
fn init_sdk_cache_dir(cache_dir: String) {
    xybrid_sdk::init_sdk_cache_dir(cache_dir);
}

/// Register the binding identifier for this process.
///
/// The xybrid-uniffi crate is shared by both Kotlin and Swift, so the
/// identity must be supplied by the platform-side wrapper at SDK init —
/// the Kotlin `Xybrid.init(...)` calls `setBinding("kotlin")`, and the
/// Swift `Xybrid.initialize()` calls `setBinding(binding: "swift")`.
///
/// Only the known platform values are forwarded to `xybrid_sdk::set_binding`
/// (which requires a `&'static str`). Any other input collapses to
/// `xybrid_sdk::DEFAULT_BINDING` to bound cardinality on the registry side
/// — the same defensive shape used by `build_client_header`'s sanitizer.
///
/// First call wins (process-global `OnceLock` in xybrid-sdk); subsequent
/// calls are silent no-ops.
#[uniffi::export]
fn set_binding(binding: String) {
    xybrid_sdk::set_binding(resolve_binding(binding.as_str()));
}

/// Pure helper that maps a runtime binding string to a `&'static str`.
///
/// Factored out of `set_binding` so tests can exercise every accepted
/// platform without touching the process-global `OnceLock` in xybrid-sdk
/// (the OnceLock's first-set-wins semantics make per-platform integration
/// tests in the same process race-prone).
fn resolve_binding(binding: &str) -> &'static str {
    match binding {
        "kotlin" => "kotlin",
        "swift" => "swift",
        _ => xybrid_sdk::DEFAULT_BINDING,
    }
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

impl From<SdkError> for XybridError {
    fn from(e: SdkError) -> Self {
        match e {
            SdkError::ModelNotFound(s) => XybridError::ModelNotFound { message: s },
            SdkError::DirectoryNotFound(s) => XybridError::DirectoryNotFound { message: s },
            SdkError::MetadataNotFound(s) => XybridError::MetadataNotFound { message: s },
            SdkError::MetadataInvalid(s) => XybridError::MetadataInvalid { message: s },
            SdkError::LoadError(s) => XybridError::LoadError { message: s },
            SdkError::InferenceError(s) => XybridError::InferenceError { message: s },
            SdkError::StreamingNotSupported => XybridError::StreamingNotSupported,
            SdkError::NotLoaded => XybridError::NotLoaded,
            SdkError::ConfigError(s) => XybridError::ConfigError { message: s },
            SdkError::NetworkError(s) => XybridError::NetworkError { message: s },
            // `SdkError::Offline` is a Rust-side refinement of NetworkError
            // (see xybrid-sdk). We collapse it back to `NetworkError` at the
            // UniFFI boundary so the Swift/Kotlin public API stays stable —
            // adding a new variant here would be a breaking change to the
            // generated sealed/enum hierarchies and needs to go through the
            // spec-first API contract update in docs/sdk/api-surface.yaml.
            SdkError::Offline(s) => XybridError::NetworkError { message: s },
            SdkError::IoError(e) => XybridError::IoError {
                message: e.to_string(),
            },
            SdkError::CacheError(s) => XybridError::CacheError { message: s },
            SdkError::PipelineError(s) => XybridError::PipelineError { message: s },
            SdkError::CircuitOpen(s) => XybridError::CircuitOpen { message: s },
            SdkError::RateLimited { retry_after_secs } => {
                XybridError::RateLimited { retry_after_secs }
            }
            SdkError::Timeout { timeout_ms } => XybridError::Timeout { timeout_ms },
        }
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
    fn to_sdk(&self) -> xybrid_sdk::GenerationConfig {
        let mut config = xybrid_sdk::GenerationConfig::default();
        if let Some(v) = self.max_tokens {
            config.max_tokens = v as usize;
        }
        if let Some(v) = self.temperature {
            config.temperature = v;
        }
        if let Some(v) = self.top_p {
            config.top_p = v;
        }
        if let Some(v) = self.min_p {
            config.min_p = v;
        }
        if let Some(v) = self.top_k {
            config.top_k = v as usize;
        }
        if let Some(v) = self.repetition_penalty {
            config.repetition_penalty = v;
        }
        if let Some(ref v) = self.stop_sequences {
            config.stop_sequences = v.clone();
        }
        config
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
}

impl XybridResult {
    pub(crate) fn from_inference_result(r: &CoreInferenceResult) -> Self {
        Self {
            success: true,
            text: r.text().map(|s| s.to_string()),
            audio_bytes: r.audio_bytes().map(|b| b.to_vec()),
            embedding: r.embedding().map(|e| e.to_vec()),
            latency_ms: r.latency_ms(),
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

impl From<CoreVoiceInfo> for XybridVoiceInfo {
    fn from(v: CoreVoiceInfo) -> Self {
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
        Ok(XybridResult::from_inference_result(&result))
    }

    /// Get all available voices for this TTS model.
    ///
    /// Returns `None` for non-TTS models or models without voice configuration.
    pub fn voices(&self) -> Option<Vec<XybridVoiceInfo>> {
        self.inner
            .voices()
            .map(|vs| vs.into_iter().map(XybridVoiceInfo::from).collect())
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
        self.inner.voice(&voice_id).map(XybridVoiceInfo::from)
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

    // Pure-helper tests: exercise every accepted platform without touching
    // the process-global OnceLock in xybrid-sdk. This is the only way to
    // assert "swift" → "swift" mapping in the same test process where the
    // Kotlin integration test below has already locked the OnceLock.
    #[test]
    fn resolve_binding_kotlin_returns_kotlin() {
        assert_eq!(resolve_binding("kotlin"), "kotlin");
    }

    #[test]
    fn resolve_binding_swift_returns_swift() {
        assert_eq!(resolve_binding("swift"), "swift");
    }

    #[test]
    fn resolve_binding_unknown_returns_default() {
        assert_eq!(resolve_binding("evil_unknown"), xybrid_sdk::DEFAULT_BINDING);
        assert_eq!(resolve_binding(""), xybrid_sdk::DEFAULT_BINDING);
        assert_eq!(resolve_binding("KOTLIN"), xybrid_sdk::DEFAULT_BINDING);
        assert_eq!(resolve_binding("flutter"), xybrid_sdk::DEFAULT_BINDING);
    }

    // Single combined integration test: the binding is process-global via
    // OnceLock, so splitting into multiple tests that call `set_binding`
    // would race on which one observes the first set. The Kotlin path is
    // the canonical wire-through; the Swift path is verified at the pure
    // `resolve_binding` layer above.
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

        // Unknown values must not propagate raw to the registry header
        // (defensive sanitization parallel to build_client_header). The
        // OnceLock is already set, so behavior is unobservable here, but
        // the wire-through call still goes through `resolve_binding`'s
        // closed match — the `_ => DEFAULT_BINDING` branch is what
        // protects a cold-start process from header pollution and is
        // exercised directly by `resolve_binding_unknown_returns_default`.
        set_binding("evil_unknown".to_string());
        assert_eq!(xybrid_sdk::get_binding(), "kotlin");
    }
}

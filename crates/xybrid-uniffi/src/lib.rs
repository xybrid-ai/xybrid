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
    XybridModel as CoreXybridModel,
};

/// Initialize the SDK cache directory.
///
/// On Android, this MUST be called before any model loading operations.
/// The Kotlin SDK wrapper `Xybrid.init(context)` calls this automatically.
#[uniffi::export]
fn init_sdk_cache_dir(cache_dir: String) {
    xybrid_sdk::init_sdk_cache_dir(cache_dir);
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
            SdkError::LoadError(s) => XybridError::LoadError { message: s },
            SdkError::InferenceError(s) => XybridError::InferenceError { message: s },
            SdkError::StreamingNotSupported => XybridError::StreamingNotSupported,
            SdkError::NotLoaded => XybridError::NotLoaded,
            SdkError::ConfigError(s) => XybridError::ConfigError { message: s },
            SdkError::NetworkError(s) => XybridError::NetworkError { message: s },
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
    pub async fn run(&self, envelope: XybridEnvelope) -> Result<XybridResult, XybridError> {
        let result = self
            .inner
            .run_async(&envelope.into())
            .await
            .map_err(XybridError::from)?;
        Ok(XybridResult::from_inference_result(&result))
    }
}

/// A model loader for loading xybrid models from registry or bundles.
///
/// Use the constructors to create a loader pointing to a model source,
/// then call `load()` to actually load the model for inference.
///
/// # Example (Swift)
///
/// ```swift
/// // Load from registry
/// let loader = XybridModelLoader.fromRegistry(modelId: "whisper-tiny")
/// let model = try loader.load()
///
/// // Load from local bundle
/// let bundleLoader = XybridModelLoader.fromBundle(path: "/path/to/model.xyb")
/// let bundleModel = try bundleLoader.load()
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
            inner: CoreModelLoader::from_registry(&model_id.as_str()),
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

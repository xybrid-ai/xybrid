//! UniFFI bindings for xybrid-sdk.
//!
//! This crate exposes xybrid-sdk types and functions to Swift and Kotlin
//! via UniFFI code generation.

use std::fmt;
use std::sync::Arc;

uniffi::setup_scaffolding!();

// Re-export xybrid-sdk for internal use (used in later user stories)
#[allow(unused_imports)]
use xybrid_sdk as _sdk;

/// Error type exposed via UniFFI to Swift/Kotlin consumers.
///
/// This enum represents all possible errors that can occur during
/// xybrid operations, allowing consumers to handle errors appropriately.
#[derive(uniffi::Error, Debug, Clone)]
pub enum XybridError {
    /// The requested model was not found in the registry or cache.
    ModelNotFound { model_id: String },
    /// Inference execution failed.
    InferenceFailed { message: String },
    /// The input provided to the model was invalid.
    InvalidInput { message: String },
    /// An I/O error occurred (file read/write, network, etc.).
    IoError { message: String },
}

impl fmt::Display for XybridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XybridError::ModelNotFound { model_id } => {
                write!(f, "Model not found: {}", model_id)
            }
            XybridError::InferenceFailed { message } => {
                write!(f, "Inference failed: {}", message)
            }
            XybridError::InvalidInput { message } => {
                write!(f, "Invalid input: {}", message)
            }
            XybridError::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
        }
    }
}

impl std::error::Error for XybridError {}

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
#[derive(uniffi::Record, Debug, Clone)]
pub struct XybridResult {
    /// Whether the inference completed successfully.
    pub success: bool,
    /// Error message if inference failed (None if successful).
    pub error: Option<String>,
    /// The type of output produced (e.g., "text", "audio", "embedding").
    pub output_type: String,
    /// Text output for ASR or LLM models.
    pub text: Option<String>,
    /// Embedding output for embedding models.
    pub embedding: Option<Vec<f32>>,
    /// Raw audio bytes for TTS models.
    pub audio_bytes: Option<Vec<u8>>,
    /// Inference latency in milliseconds.
    pub latency_ms: u32,
}

/// Internal model state wrapper.
///
/// This holds the model identifier and will be extended to hold
/// actual model state (executor, session) in future stories.
struct ModelState {
    model_id: String,
    // TODO: Add TemplateExecutor or runtime adapter in future story
}

/// A loaded xybrid model ready for inference.
///
/// This object represents a model that has been loaded and is ready to run
/// inference. Use `XybridModelLoader` to obtain instances of this type.
#[derive(uniffi::Object)]
pub struct XybridModel {
    /// Internal model state.
    inner: ModelState,
}

#[uniffi::export]
impl XybridModel {
    /// Run inference on the model with the provided input envelope.
    ///
    /// # Arguments
    ///
    /// * `envelope` - The input data to process (audio, text, or embedding).
    ///
    /// # Returns
    ///
    /// A `XybridResult` containing the inference output on success, or
    /// a `XybridError` if inference fails.
    ///
    /// # Example (Swift)
    ///
    /// ```swift
    /// let envelope = XybridEnvelope.text(text: "Hello, world!", voiceId: nil, speed: nil)
    /// let result = try model.run(envelope: envelope)
    /// if result.success {
    ///     print(result.text ?? "No text output")
    /// }
    /// ```
    pub fn run(&self, envelope: XybridEnvelope) -> Result<XybridResult, XybridError> {
        // TODO: Implement actual inference using TemplateExecutor in future story
        // For now, return a placeholder result to satisfy the UniFFI build
        let start = std::time::Instant::now();

        // Determine output type based on input envelope
        let output_type = match &envelope {
            XybridEnvelope::Audio { .. } => "text".to_string(), // ASR -> text
            XybridEnvelope::Text { .. } => "audio".to_string(), // TTS -> audio
            XybridEnvelope::Embedding { .. } => "embedding".to_string(),
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        Ok(XybridResult {
            success: true,
            error: None,
            output_type,
            text: None,
            embedding: None,
            audio_bytes: None,
            latency_ms,
        })
    }

    /// Get the model ID of this loaded model.
    ///
    /// # Returns
    ///
    /// The unique identifier string for this model (e.g., "whisper-tiny", "kokoro-82m").
    pub fn model_id(&self) -> String {
        self.inner.model_id.clone()
    }
}

/// Source of the model to load.
#[derive(Debug, Clone)]
enum LoaderSource {
    /// Load from the xybrid model registry.
    Registry { model_id: String },
    /// Load from a local bundle file path.
    Bundle { path: String },
}

/// Internal loader state wrapper.
struct LoaderState {
    source: LoaderSource,
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
    inner: LoaderState,
}

#[uniffi::export]
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
            inner: LoaderState {
                source: LoaderSource::Registry { model_id },
            },
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
            inner: LoaderState {
                source: LoaderSource::Bundle { path },
            },
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
    pub fn load(&self) -> Result<Arc<XybridModel>, XybridError> {
        // TODO: Implement actual model loading using xybrid-sdk's RegistryClient
        // For now, create a placeholder model to satisfy the UniFFI build

        let model_id = match &self.inner.source {
            LoaderSource::Registry { model_id } => model_id.clone(),
            LoaderSource::Bundle { path } => {
                // Extract model ID from bundle path (e.g., "/path/to/whisper-tiny.xyb" -> "whisper-tiny")
                std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            }
        };

        Ok(Arc::new(XybridModel {
            inner: ModelState { model_id },
        }))
    }
}

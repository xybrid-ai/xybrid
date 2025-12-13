//! Model loading and execution for xybrid-sdk.
//!
//! This module provides:
//! - `ModelLoader`: Preparatory step for loading models (from registry, bundle, or directory)
//! - `XybridModel`: Loaded model ready for inference
//! - `ModelHandle`: Internal state management for the loaded model

use crate::result::{InferenceResult, OutputType};
use crate::source::{detect_platform, ModelSource};
use crate::stream::XybridStream;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tempfile::TempDir;
use xybrid_core::bundler::XyBundle;
use xybrid_core::execution_template::{ExecutionTemplate, ModelMetadata};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::streaming::{StreamConfig as CoreStreamConfig, VadStreamConfig as CoreVadConfig};
use xybrid_core::template_executor::TemplateExecutor;

/// SDK-level error type.
#[derive(Debug, thiserror::Error)]
pub enum SdkError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Inference failed: {0}")]
    InferenceError(String),
    #[error("Streaming not supported by this model")]
    StreamingNotSupported,
    #[error("Model not loaded")]
    NotLoaded,
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Pipeline error: {0}")]
    PipelineError(String),
}

/// Result type for SDK operations.
pub type SdkResult<T> = Result<T, SdkError>;

/// Configuration for streaming ASR sessions.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Enable VAD (Voice Activity Detection) for smart chunking
    pub enable_vad: bool,
    /// VAD threshold (0.0-1.0)
    pub vad_threshold: f32,
    /// Language hint for ASR
    pub language: Option<String>,
    /// Path to VAD model (uses default if None)
    pub vad_model_dir: Option<String>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            enable_vad: false,
            vad_threshold: 0.5,
            language: Some("en".to_string()),
            vad_model_dir: None,
        }
    }
}

impl StreamConfig {
    /// Create config with VAD enabled.
    pub fn with_vad() -> Self {
        Self {
            enable_vad: true,
            ..Default::default()
        }
    }

    /// Set language hint.
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set VAD threshold.
    pub fn vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold;
        self
    }
}

/// Internal handle holding the loaded model state.
struct ModelHandle {
    /// Template executor for running inference
    executor: TemplateExecutor,
    /// Model metadata
    metadata: ModelMetadata,
    /// Temporary directory for extracted bundles (kept alive while model is loaded)
    _temp_dir: Option<TempDir>,
    /// Model directory path
    model_dir: PathBuf,
    /// Whether model is currently loaded
    loaded: bool,
}

/// Represents a model that can be loaded.
///
/// Created by `Xybrid::model()`, must call `.load()` to use.
/// This is a preparatory step that doesn't download or load anything.
///
/// # Example
///
/// ```ignore
/// let loader = ModelLoader::from_registry("http://localhost:8080", "whisper-tiny", "1.0");
/// let model = loader.load()?;
/// let result = model.run(&envelope)?;
/// ```
#[derive(Debug, Clone)]
pub struct ModelLoader {
    source: ModelSource,
    model_id: Option<String>,
    version: Option<String>,
}

impl ModelLoader {
    /// Create loader from registry.
    pub fn from_registry(url: &str, model_id: &str, version: &str) -> Self {
        Self {
            source: ModelSource::registry(url, model_id, version),
            model_id: Some(model_id.to_string()),
            version: Some(version.to_string()),
        }
    }

    /// Create loader from registry with explicit platform.
    pub fn from_registry_with_platform(
        url: &str,
        model_id: &str,
        version: &str,
        platform: &str,
    ) -> Self {
        Self {
            source: ModelSource::registry_with_platform(url, model_id, version, platform),
            model_id: Some(model_id.to_string()),
            version: Some(version.to_string()),
        }
    }

    /// Create loader from local bundle file.
    pub fn from_bundle(path: impl Into<PathBuf>) -> SdkResult<Self> {
        let path: PathBuf = path.into();
        if !path.exists() {
            return Err(SdkError::ModelNotFound(format!(
                "Bundle not found: {:?}",
                path
            )));
        }
        Ok(Self {
            source: ModelSource::bundle(path),
            model_id: None,
            version: None,
        })
    }

    /// Create loader from local model directory.
    pub fn from_directory(path: impl Into<PathBuf>) -> SdkResult<Self> {
        let path: PathBuf = path.into();
        if !path.exists() {
            return Err(SdkError::ModelNotFound(format!(
                "Directory not found: {:?}",
                path
            )));
        }
        let metadata_path = path.join("model_metadata.json");
        if !metadata_path.exists() {
            return Err(SdkError::ConfigError(format!(
                "model_metadata.json not found in {:?}",
                path
            )));
        }
        Ok(Self {
            source: ModelSource::directory(path),
            model_id: None,
            version: None,
        })
    }

    /// Get the model ID (if known).
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Get the version (if known).
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// Get the source type.
    pub fn source_type(&self) -> &'static str {
        self.source.source_type()
    }

    /// Load the model into memory (synchronous).
    ///
    /// This will:
    /// - For registry: Download the bundle if not cached, extract it
    /// - For bundle: Extract the bundle to a temp directory
    /// - For directory: Load directly from the directory
    ///
    /// Returns a loaded `XybridModel` ready for inference.
    pub fn load(&self) -> SdkResult<XybridModel> {
        match &self.source {
            ModelSource::Registry {
                url,
                model_id,
                version,
                platform,
            } => self.load_from_registry(url, model_id, version, platform.as_deref()),
            ModelSource::Bundle { path } => self.load_from_bundle(path),
            ModelSource::Directory { path } => self.load_from_directory(path),
        }
    }

    /// Load the model asynchronously.
    pub async fn load_async(&self) -> SdkResult<XybridModel> {
        // For now, wrap the sync version. Real async would use tokio::fs and async HTTP.
        let loader = self.clone();
        tokio::task::spawn_blocking(move || loader.load())
            .await
            .map_err(|e| SdkError::LoadError(format!("Task join error: {}", e)))?
    }

    fn load_from_registry(
        &self,
        url: &str,
        model_id: &str,
        version: &str,
        platform: Option<&str>,
    ) -> SdkResult<XybridModel> {
        let platform = platform
            .map(String::from)
            .unwrap_or_else(detect_platform);

        // Build bundle URL
        let bundle_url = format!(
            "{}/bundles/{}/{}/{}/{}.xyb",
            url.trim_end_matches('/'),
            model_id,
            version,
            platform,
            model_id
        );

        // Download bundle to temp file
        let temp_dir =
            TempDir::new().map_err(|e| SdkError::IoError(e))?;
        let bundle_path = temp_dir.path().join(format!("{}.xyb", model_id));

        // Use blocking HTTP client
        let response = ureq::get(&bundle_url)
            .call()
            .map_err(|e| SdkError::NetworkError(format!("Failed to download bundle: {}", e)))?;

        if response.status() != 200 {
            return Err(SdkError::ModelNotFound(format!(
                "Bundle not found at registry: {} (status {})",
                bundle_url,
                response.status()
            )));
        }

        // Write bundle to temp file
        let mut file = std::fs::File::create(&bundle_path)?;
        std::io::copy(&mut response.into_reader(), &mut file)?;

        // Extract and load
        self.load_from_bundle_with_temp(&bundle_path, Some(temp_dir))
    }

    fn load_from_bundle(&self, path: &PathBuf) -> SdkResult<XybridModel> {
        self.load_from_bundle_with_temp(path, None)
    }

    fn load_from_bundle_with_temp(
        &self,
        path: &PathBuf,
        existing_temp: Option<TempDir>,
    ) -> SdkResult<XybridModel> {
        // Load the bundle
        let bundle = XyBundle::load(path)
            .map_err(|e| SdkError::LoadError(format!("Failed to load bundle: {}", e)))?;

        // Create temp directory for extraction (or use existing)
        let temp_dir = match existing_temp {
            Some(t) => t,
            None => TempDir::new()?,
        };
        let extract_dir = temp_dir.path().join("model");
        std::fs::create_dir_all(&extract_dir)?;

        // Extract bundle
        bundle
            .extract_to(&extract_dir)
            .map_err(|e| SdkError::LoadError(format!("Failed to extract bundle: {}", e)))?;

        // Get model ID from manifest
        let manifest = bundle.manifest();
        let model_id = manifest.model_id.clone();
        let version = manifest.version.clone();

        // Load from extracted directory
        let handle = Self::create_model_handle(&extract_dir, Some(temp_dir))?;

        // Determine if streaming is supported
        let supports_streaming = Self::check_streaming_support(&handle.metadata);

        // Determine output type
        let output_type = Self::infer_output_type(&handle.metadata);

        Ok(XybridModel {
            handle: Arc::new(RwLock::new(handle)),
            model_id,
            version,
            output_type,
            supports_streaming,
        })
    }

    fn load_from_directory(&self, path: &PathBuf) -> SdkResult<XybridModel> {
        let handle = Self::create_model_handle(path, None)?;

        let model_id = handle.metadata.model_id.clone();
        let version = handle.metadata.version.clone();
        let supports_streaming = Self::check_streaming_support(&handle.metadata);
        let output_type = Self::infer_output_type(&handle.metadata);

        Ok(XybridModel {
            handle: Arc::new(RwLock::new(handle)),
            model_id,
            version,
            output_type,
            supports_streaming,
        })
    }

    fn create_model_handle(
        model_dir: &PathBuf,
        temp_dir: Option<TempDir>,
    ) -> SdkResult<ModelHandle> {
        // Load metadata
        let metadata_path = model_dir.join("model_metadata.json");
        let metadata_str = std::fs::read_to_string(&metadata_path).map_err(|e| {
            SdkError::LoadError(format!("Failed to read model_metadata.json: {}", e))
        })?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| SdkError::LoadError(format!("Failed to parse metadata: {}", e)))?;

        // Create executor with base path
        let executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap_or("."));

        Ok(ModelHandle {
            executor,
            metadata,
            _temp_dir: temp_dir,
            model_dir: model_dir.clone(),
            loaded: true,
        })
    }

    fn check_streaming_support(metadata: &ModelMetadata) -> bool {
        // Check if this is an ASR model (supports streaming)
        // Look at metadata task or model type (metadata is HashMap<String, serde_json::Value>)
        if let Some(task) = metadata.metadata.get("task").and_then(|v| v.as_str()) {
            if task == "speech-recognition" || task == "asr" {
                return true;
            }
        }

        // Check execution template type
        match &metadata.execution_template {
            ExecutionTemplate::CandleModel { model_type, .. } => {
                model_type.as_deref() == Some("whisper")
            }
            ExecutionTemplate::SimpleMode { .. } => {
                // Check if preprocessing includes AudioDecode (likely ASR)
                metadata.preprocessing.iter().any(|step| {
                    matches!(
                        step,
                        xybrid_core::execution_template::PreprocessingStep::AudioDecode { .. }
                    )
                })
            }
            _ => false,
        }
    }

    fn infer_output_type(metadata: &ModelMetadata) -> OutputType {
        // Check metadata hints (metadata is HashMap<String, serde_json::Value>)
        if let Some(task) = metadata.metadata.get("task").and_then(|v| v.as_str()) {
            match task {
                "speech-recognition" | "asr" | "transcription" => return OutputType::Text,
                "text-to-speech" | "tts" | "speech-synthesis" => return OutputType::Audio,
                "embedding" | "feature-extraction" => return OutputType::Embedding,
                _ => {}
            }
        }

        // Check postprocessing steps for hints
        for step in &metadata.postprocessing {
            match step {
                xybrid_core::execution_template::PostprocessingStep::CTCDecode { .. }
                | xybrid_core::execution_template::PostprocessingStep::WhisperDecode { .. } => {
                    return OutputType::Text
                }
                xybrid_core::execution_template::PostprocessingStep::TTSAudioEncode { .. } => {
                    return OutputType::Audio
                }
                _ => {}
            }
        }

        OutputType::Unknown
    }
}

/// Represents a loaded model ready for inference.
///
/// Created by `ModelLoader::load()`. Provides both batch and streaming inference.
///
/// # Example
///
/// ```ignore
/// let model = loader.load()?;
///
/// // Batch inference
/// let result = model.run(&audio_envelope)?;
/// println!("Transcription: {}", result.unwrap_text());
///
/// // Streaming inference (if supported)
/// if model.supports_streaming() {
///     let stream = model.stream(StreamConfig::with_vad())?;
///     stream.feed(&samples)?;
///     let transcript = stream.flush()?;
/// }
///
/// // Cleanup
/// model.unload()?;
/// ```
pub struct XybridModel {
    handle: Arc<RwLock<ModelHandle>>,
    model_id: String,
    version: String,
    output_type: OutputType,
    supports_streaming: bool,
}

impl XybridModel {
    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the model version.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Check if the model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.handle
            .read()
            .map(|h| h.loaded)
            .unwrap_or(false)
    }

    /// Check if this model supports streaming.
    pub fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }

    /// Get the expected output type for this model.
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    /// Run batch inference with an Envelope.
    ///
    /// # Arguments
    ///
    /// * `envelope` - Input data wrapped in an Envelope
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the output with convenient accessors.
    pub fn run(&self, envelope: &Envelope) -> SdkResult<InferenceResult> {
        let start = Instant::now();

        let mut handle = self.handle.write().map_err(|_| {
            SdkError::InferenceError("Failed to acquire model lock".to_string())
        })?;

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to avoid borrow conflict with executor
        let metadata = handle.metadata.clone();
        let output = handle
            .executor
            .execute(&metadata, envelope)
            .map_err(|e| SdkError::InferenceError(format!("Execution failed: {}", e)))?;

        let latency_ms = start.elapsed().as_millis() as u32;

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run batch inference asynchronously.
    pub async fn run_async(&self, envelope: &Envelope) -> SdkResult<InferenceResult> {
        let handle = self.handle.clone();
        let model_id = self.model_id.clone();
        let envelope = envelope.clone();

        tokio::task::spawn_blocking(move || {
            let start = Instant::now();

            let mut guard = handle.write().map_err(|_| {
                SdkError::InferenceError("Failed to acquire model lock".to_string())
            })?;

            if !guard.loaded {
                return Err(SdkError::NotLoaded);
            }

            // Clone metadata to avoid borrow conflict with executor
            let metadata = guard.metadata.clone();
            let output = guard
                .executor
                .execute(&metadata, &envelope)
                .map_err(|e| SdkError::InferenceError(format!("Execution failed: {}", e)))?;

            let latency_ms = start.elapsed().as_millis() as u32;
            Ok(InferenceResult::new(output, &model_id, latency_ms))
        })
        .await
        .map_err(|e| SdkError::InferenceError(format!("Task join error: {}", e)))?
    }

    /// Create a streaming session for real-time ASR.
    ///
    /// Returns an error if `!supports_streaming()`.
    ///
    /// # Arguments
    ///
    /// * `config` - Streaming configuration (VAD, language, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stream = model.stream(StreamConfig::with_vad())?;
    ///
    /// // Feed audio chunks
    /// stream.feed(&audio_samples)?;
    ///
    /// // Get partial results
    /// if let Some(partial) = stream.partial_result() {
    ///     println!("Partial: {}", partial.text);
    /// }
    ///
    /// // Get final transcript
    /// let transcript = stream.flush()?;
    /// ```
    pub fn stream(&self, config: StreamConfig) -> SdkResult<XybridStream> {
        if !self.supports_streaming {
            return Err(SdkError::StreamingNotSupported);
        }

        let handle = self.handle.read().map_err(|_| {
            SdkError::InferenceError("Failed to acquire model lock".to_string())
        })?;

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Convert to core StreamConfig
        let core_config = CoreStreamConfig {
            vad: CoreVadConfig {
                enabled: config.enable_vad,
                model_dir: config.vad_model_dir,
                threshold: config.vad_threshold,
                ..Default::default()
            },
            language: config.language,
            ..Default::default()
        };

        XybridStream::new(&handle.model_dir, core_config, &self.model_id)
    }

    /// Unload the model from memory.
    ///
    /// The model can be reloaded by creating a new ModelLoader.
    pub fn unload(&self) -> SdkResult<()> {
        let mut handle = self.handle.write().map_err(|_| {
            SdkError::InferenceError("Failed to acquire model lock".to_string())
        })?;

        handle.loaded = false;
        // Clear the session cache (drop executor and recreate empty)
        handle.executor = TemplateExecutor::new();

        Ok(())
    }
}

// Make XybridModel cloneable (shares the handle)
impl Clone for XybridModel {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            model_id: self.model_id.clone(),
            version: self.version.clone(),
            output_type: self.output_type,
            supports_streaming: self.supports_streaming,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loader_from_registry() {
        let loader = ModelLoader::from_registry("http://localhost:8080", "whisper", "1.0");
        assert_eq!(loader.model_id(), Some("whisper"));
        assert_eq!(loader.version(), Some("1.0"));
        assert_eq!(loader.source_type(), "registry");
    }

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert!(!config.enable_vad);
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_stream_config_with_vad() {
        let config = StreamConfig::with_vad().language("fr").vad_threshold(0.7);
        assert!(config.enable_vad);
        assert_eq!(config.language, Some("fr".to_string()));
        assert_eq!(config.vad_threshold, 0.7);
    }
}

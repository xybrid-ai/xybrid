//! Candle Runtime Adapter implementation.
//!
//! This module provides a high-level `RuntimeAdapter` implementation
//! for Candle-based inference, working with xybrid's Envelope types.

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use std::collections::HashMap;
use std::path::Path;

use super::device::{select_device, DeviceSelection};
use super::whisper::{WhisperConfig, WhisperModel};
use candle_core::Device;

/// Candle Runtime Adapter.
///
/// Provides high-level inference capabilities using the Candle framework.
/// Currently supports Whisper ASR models, with more model types planned.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::runtime_adapter::candle::CandleRuntimeAdapter;
/// use xybrid_core::runtime_adapter::RuntimeAdapter;
/// use xybrid_core::ir::{Envelope, EnvelopeKind};
///
/// let mut adapter = CandleRuntimeAdapter::new()?;
/// adapter.load_model("/path/to/whisper-tiny")?;
///
/// let audio_bytes = std::fs::read("audio.wav")?;
/// let input = Envelope::new(EnvelopeKind::Audio(audio_bytes));
/// let output = adapter.execute(&input)?;
/// ```
pub struct CandleRuntimeAdapter {
    /// Loaded models (model_id -> WhisperModel)
    models: HashMap<String, WhisperModel>,
    /// Model metadata
    metadata: HashMap<String, ModelMetadata>,
    /// Currently active model
    current_model: Option<String>,
    /// Selected device
    device: Device,
    /// Whisper configuration
    whisper_config: WhisperConfig,
}

impl CandleRuntimeAdapter {
    /// Create a new Candle Runtime Adapter with automatic device selection.
    pub fn new() -> AdapterResult<Self> {
        Self::with_device(DeviceSelection::Auto)
    }

    /// Create a new adapter with specific device preference.
    pub fn with_device(preference: DeviceSelection) -> AdapterResult<Self> {
        let device = select_device(preference)
            .map_err(|e| AdapterError::RuntimeError(format!("Device selection failed: {}", e)))?;

        Ok(Self {
            models: HashMap::new(),
            metadata: HashMap::new(),
            current_model: None,
            device,
            whisper_config: WhisperConfig::default(),
        })
    }

    /// Set Whisper configuration for subsequently loaded models.
    pub fn set_whisper_config(&mut self, config: WhisperConfig) {
        self.whisper_config = config;
    }

    /// Get the current device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Extract model ID from path.
    fn extract_model_id(&self, path: &str) -> String {
        Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Process embedding envelope (pre-computed mel spectrogram) for Whisper inference.
    ///
    /// Expects the embedding to be a flattened mel spectrogram [1, n_mels, n_frames].
    /// The shape should be provided in metadata as "mel_shape": "[1, 80, N]".
    fn process_mel_input(
        &self,
        mel_data: &[f32],
        metadata: &HashMap<String, String>,
    ) -> AdapterResult<candle_core::Tensor> {
        // Try to get shape from metadata
        let n_mels = 80; // Whisper standard
        let n_frames = mel_data.len() / n_mels;

        // Validate shape
        if mel_data.len() % n_mels != 0 {
            return Err(AdapterError::InvalidInput(format!(
                "Mel spectrogram size {} is not divisible by n_mels {}",
                mel_data.len(),
                n_mels
            )));
        }

        candle_core::Tensor::from_vec(mel_data.to_vec(), (1, n_mels, n_frames), &self.device)
            .map_err(|e| AdapterError::RuntimeError(format!("Tensor creation failed: {}", e)))
    }
}

impl RuntimeAdapter for CandleRuntimeAdapter {
    fn name(&self) -> &str {
        "candle"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["safetensors", "bin", "pt"]
    }

    fn load_model(&mut self, path: &str) -> AdapterResult<()> {
        let model_path = Path::new(path);

        // Validate path exists
        if !model_path.exists() {
            return Err(AdapterError::ModelNotFound(format!(
                "Model directory not found: {}",
                path
            )));
        }

        let model_id = self.extract_model_id(path);

        // Check if already loaded - just log and continue
        if self.models.contains_key(&model_id) {
            log::warn!("Model '{}' is already loaded, skipping reload", model_id);
            return Ok(());
        }

        // Load Whisper model
        let model =
            WhisperModel::load_with_config(model_path, &self.device, self.whisper_config.clone())
                .map_err(|e| AdapterError::RuntimeError(format!("Failed to load model: {}", e)))?;

        // Create metadata
        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            version: "1.0.0".to_string(),
            runtime_type: "candle".to_string(),
            model_path: path.to_string(),
            input_schema: {
                let mut schema = HashMap::new();
                schema.insert("mel".to_string(), vec![1, 80, 3000]); // 30 seconds max
                schema
            },
            output_schema: {
                let mut schema = HashMap::new();
                schema.insert("text".to_string(), vec![1]);
                schema
            },
        };

        self.models.insert(model_id.clone(), model);
        self.metadata.insert(model_id.clone(), metadata);
        self.current_model = Some(model_id);

        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        let model_id = self.current_model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load_model() first.".to_string())
        })?;

        // Get mutable reference to model
        // Note: We need interior mutability for autoregressive decoding
        // This is a design limitation - consider RefCell or Mutex
        let model = self.models.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Model '{}' not found", model_id))
        })?;

        match &input.kind {
            EnvelopeKind::Embedding(mel_data) => {
                // Process pre-computed mel spectrogram
                let mel = self.process_mel_input(mel_data, &input.metadata)?;

                // Note: We can't call encode() because it needs &mut self but we have &self
                // This is a design limitation - RuntimeAdapter::execute takes &self
                // For now, return an error explaining the limitation
                // TODO: Use RefCell/Mutex for interior mutability

                Err(AdapterError::RuntimeError(
                    "Candle Whisper requires mutable model access for inference. \
                     Use CandleBackend::run_whisper() directly with mutable reference, \
                     or wait for interior mutability implementation."
                        .to_string(),
                ))
            }
            EnvelopeKind::Audio(_) => Err(AdapterError::InvalidInput(
                "Candle Whisper expects pre-computed mel spectrogram as Embedding. \
                     Use xybrid preprocessing pipeline to convert Audio to Embedding first."
                    .to_string(),
            )),
            EnvelopeKind::Text(_) => Err(AdapterError::InvalidInput(
                "Whisper expects Embedding (mel spectrogram) input, not Text".to_string(),
            )),
        }
    }
}

impl RuntimeAdapterExt for CandleRuntimeAdapter {
    fn is_loaded(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    fn get_metadata(&self, model_id: &str) -> AdapterResult<&ModelMetadata> {
        self.metadata.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Model '{}' is not loaded", model_id))
        })
    }

    fn infer(&self, model_id: &str, input: &Envelope) -> AdapterResult<Envelope> {
        if !self.is_loaded(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' is not loaded",
                model_id
            )));
        }

        // Temporarily set current model and execute
        // Note: This is a bit hacky - consider better design
        let original_current = self.current_model.clone();

        // We can't actually change current_model without &mut self
        // For now, just verify the model exists and delegate to execute
        if self.current_model.as_ref() == Some(&model_id.to_string()) {
            self.execute(input)
        } else {
            Err(AdapterError::RuntimeError(
                "Cannot switch models in infer() without mutable access. \
                 Use load_model() to set the active model."
                    .to_string(),
            ))
        }
    }

    fn unload_model(&mut self, model_id: &str) -> AdapterResult<()> {
        if !self.models.contains_key(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' is not loaded",
                model_id
            )));
        }

        self.models.remove(model_id);
        self.metadata.remove(model_id);

        if self.current_model.as_ref() == Some(&model_id.to_string()) {
            self.current_model = None;
        }

        Ok(())
    }

    fn list_loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = CandleRuntimeAdapter::new();
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_adapter_name() {
        let adapter = CandleRuntimeAdapter::new().unwrap();
        assert_eq!(adapter.name(), "candle");
    }

    #[test]
    fn test_supported_formats() {
        let adapter = CandleRuntimeAdapter::new().unwrap();
        let formats = adapter.supported_formats();
        assert!(formats.contains(&"safetensors"));
    }

    #[test]
    fn test_load_nonexistent_model() {
        let mut adapter = CandleRuntimeAdapter::new().unwrap();
        let result = adapter.load_model("/nonexistent/path");
        assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
    }

    #[test]
    fn test_execute_without_model() {
        let adapter = CandleRuntimeAdapter::new().unwrap();
        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));
        let result = adapter.execute(&input);
        assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
    }

    #[test]
    fn test_list_loaded_models_empty() {
        let adapter = CandleRuntimeAdapter::new().unwrap();
        assert!(adapter.list_loaded_models().is_empty());
    }
}

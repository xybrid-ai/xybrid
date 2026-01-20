//! Candle inference backend implementation.
//!
//! This module implements the `InferenceBackend` trait for Candle,
//! providing a low-level tensor-based interface for model execution.

use crate::runtime_adapter::{BackendError, BackendResult, InferenceBackend, RuntimeType};
use candle_core::{Device, Tensor};
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

use super::device::{select_device, DeviceSelection};
use super::whisper::WhisperModel;

/// Candle inference backend.
///
/// Provides tensor-level inference using Candle framework.
/// Currently supports Whisper models, with more model types planned.
pub struct CandleBackend {
    /// Selected compute device
    device: Device,
    /// Loaded Whisper model (if any)
    whisper_model: Option<WhisperModel>,
    /// Model path (for metadata)
    model_path: Option<String>,
}

impl CandleBackend {
    /// Create a new Candle backend with automatic device selection.
    pub fn new() -> BackendResult<Self> {
        Self::with_device(DeviceSelection::Auto)
    }

    /// Create a new Candle backend with specific device preference.
    pub fn with_device(preference: DeviceSelection) -> BackendResult<Self> {
        let device = select_device(preference)
            .map_err(|e| BackendError::RuntimeError(format!("Device selection failed: {}", e)))?;

        Ok(Self {
            device,
            whisper_model: None,
            model_path: None,
        })
    }

    /// Get the current device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if a Whisper model is loaded.
    pub fn has_whisper_model(&self) -> bool {
        self.whisper_model.is_some()
    }

    /// Run Whisper inference on mel spectrogram input.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram tensor [1, n_mels, n_frames]
    ///
    /// # Returns
    ///
    /// Transcribed text
    pub fn run_whisper(&mut self, mel: &Tensor) -> BackendResult<String> {
        let model = self
            .whisper_model
            .as_mut()
            .ok_or_else(|| BackendError::ModelNotLoaded)?;

        model
            .transcribe(mel)
            .map_err(|e| BackendError::InferenceFailed(format!("Whisper inference failed: {}", e)))
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create default CandleBackend")
    }
}

impl InferenceBackend for CandleBackend {
    fn runtime_type(&self) -> RuntimeType {
        RuntimeType::Candle
    }

    fn load_model(&mut self, model_path: &Path, _config_path: Option<&Path>) -> BackendResult<()> {
        let path_str = model_path.to_string_lossy();

        // Detect model type from path
        // Whisper models typically have "whisper" in the path or are in a whisper directory
        let is_whisper = path_str.contains("whisper")
            || model_path.join("config.json").exists()
            || model_path.join("model.safetensors").exists();

        if is_whisper {
            let model = WhisperModel::load(model_path, &self.device).map_err(|e| {
                BackendError::LoadFailed(format!("Failed to load Whisper model: {}", e))
            })?;
            self.whisper_model = Some(model);
            self.model_path = Some(path_str.to_string());
            Ok(())
        } else {
            Err(BackendError::LoadFailed(format!(
                "Unsupported model type at path: {}. Currently only Whisper models are supported.",
                path_str
            )))
        }
    }

    fn run_inference(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> BackendResult<HashMap<String, ArrayD<f32>>> {
        // For Whisper, we expect a "mel" input and return "text" as embedding
        // (The actual text decoding is done in run_whisper)

        let mel_input = inputs
            .get("mel")
            .or_else(|| inputs.get("input"))
            .ok_or_else(|| {
                BackendError::InvalidInput(
                    "Expected 'mel' or 'input' tensor for Whisper inference".to_string(),
                )
            })?;

        // Convert ndarray to Candle tensor
        let shape: Vec<usize> = mel_input.shape().to_vec();
        let data: Vec<f32> = mel_input.iter().copied().collect();

        let mel_tensor = Tensor::from_vec(data, shape.as_slice(), &self.device)
            .map_err(|e| BackendError::InvalidInput(format!("Failed to create tensor: {}", e)))?;

        // Note: This is a limitation - InferenceBackend::run_inference takes &self
        // For Whisper's encoder, we need &mut self due to KV cache
        // Return an error explaining the limitation

        // Note: mel_tensor is created but unused because we return an error
        // This is intentional - we validate the input before returning the error
        let _ = mel_tensor;

        Err(BackendError::InferenceFailed(
            "Candle Whisper requires mutable model access for inference. \
             Use CandleBackend::run_whisper() directly with mutable reference."
                .to_string(),
        ))
    }

    fn is_loaded(&self) -> bool {
        self.whisper_model.is_some()
    }

    fn input_names(&self) -> BackendResult<Vec<String>> {
        if self.whisper_model.is_some() {
            Ok(vec!["mel".to_string()])
        } else {
            Err(BackendError::ModelNotLoaded)
        }
    }

    fn output_names(&self) -> BackendResult<Vec<String>> {
        if self.whisper_model.is_some() {
            Ok(vec!["encoder_output".to_string(), "text".to_string()])
        } else {
            Err(BackendError::ModelNotLoaded)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = CandleBackend::new();
        assert!(backend.is_ok());
        let backend = backend.unwrap();
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_runtime_type() {
        let backend = CandleBackend::new().unwrap();
        assert_eq!(backend.runtime_type(), RuntimeType::Candle);
    }

    #[test]
    fn test_input_names_without_model() {
        let backend = CandleBackend::new().unwrap();
        assert!(matches!(
            backend.input_names(),
            Err(BackendError::ModelNotLoaded)
        ));
    }
}

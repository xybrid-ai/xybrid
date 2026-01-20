//! Abstract model trait for Candle models.
//!
//! This module defines a common interface for different Candle model types,
//! allowing the framework to work with various models (Whisper, LLaMA, BERT, etc.)
//! in a unified way.

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Model type identifier for routing to appropriate execution logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandleModelType {
    /// Whisper ASR model
    Whisper,
    /// LLaMA language model (future)
    LLaMA,
    /// BERT encoder model (future)
    Bert,
    /// Generic/unknown model type
    Generic,
}

impl CandleModelType {
    /// Parse model type from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "whisper" => Self::Whisper,
            "llama" | "llama2" | "llama3" => Self::LLaMA,
            "bert" => Self::Bert,
            _ => Self::Generic,
        }
    }
}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Error type for model operations
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Failed to load model: {0}")]
    LoadFailed(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Abstract trait for Candle-based models.
///
/// This trait provides a common interface for different model types,
/// allowing the runtime adapter to work with them uniformly.
///
/// # Implementing New Models
///
/// To add a new model type (e.g., LLaMA):
///
/// 1. Create a new module: `candle/llama.rs`
/// 2. Implement `CandleModel` for your model struct
/// 3. Add the variant to `CandleModelType`
/// 4. Update `load_candle_model()` to handle the new type
pub trait CandleModel: Send {
    /// Get the model type
    fn model_type(&self) -> CandleModelType;

    /// Get the device this model is loaded on
    fn device(&self) -> &Device;

    /// Run inference on the model.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input tensor names to tensors
    ///
    /// # Returns
    ///
    /// Map of output tensor names to tensors
    fn run(&mut self, inputs: HashMap<String, Tensor>) -> ModelResult<HashMap<String, Tensor>>;

    /// Get expected input tensor names
    fn input_names(&self) -> Vec<&str>;

    /// Get expected output tensor names
    fn output_names(&self) -> Vec<&str>;
}

/// Factory function to load a Candle model based on type.
///
/// # Arguments
///
/// * `model_type` - Type of model to load
/// * `model_path` - Path to model directory
/// * `device` - Device to load model on
///
/// # Returns
///
/// Boxed trait object implementing `CandleModel`
pub fn load_candle_model(
    model_type: CandleModelType,
    model_path: &Path,
    device: &Device,
) -> ModelResult<Box<dyn CandleModel>> {
    match model_type {
        CandleModelType::Whisper => {
            use super::whisper::{WhisperConfig, WhisperModel};
            let model =
                WhisperModel::load_with_config(model_path, device, WhisperConfig::default())
                    .map_err(|e| ModelError::LoadFailed(e.to_string()))?;
            Ok(Box::new(WhisperModelWrapper { model }))
        }
        CandleModelType::LLaMA => Err(ModelError::Unsupported(
            "LLaMA model support not yet implemented".to_string(),
        )),
        CandleModelType::Bert => Err(ModelError::Unsupported(
            "BERT model support not yet implemented".to_string(),
        )),
        CandleModelType::Generic => Err(ModelError::Unsupported(
            "Generic model loading not supported. Specify a model type.".to_string(),
        )),
    }
}

/// Wrapper to make WhisperModel implement CandleModel trait
pub struct WhisperModelWrapper {
    pub model: super::whisper::WhisperModel,
}

impl CandleModel for WhisperModelWrapper {
    fn model_type(&self) -> CandleModelType {
        CandleModelType::Whisper
    }

    fn device(&self) -> &Device {
        self.model.device()
    }

    fn run(&mut self, inputs: HashMap<String, Tensor>) -> ModelResult<HashMap<String, Tensor>> {
        // Whisper expects either "mel" tensor or "pcm" audio
        if let Some(mel) = inputs.get("mel") {
            let text = self
                .model
                .transcribe(mel)
                .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

            // Return text as a simple tensor (for compatibility with trait interface)
            // In practice, callers should use transcribe_pcm() directly for text output
            let text_bytes: Vec<f32> = text.bytes().map(|b| b as f32).collect();
            let text_tensor = Tensor::from_vec(text_bytes.clone(), text_bytes.len(), self.device())
                .map_err(ModelError::Candle)?;

            let mut outputs = HashMap::new();
            outputs.insert("text".to_string(), text_tensor);
            Ok(outputs)
        } else if let Some(_pcm) = inputs.get("pcm") {
            // PCM audio input - convert to mel and transcribe
            Err(ModelError::Unsupported(
                "Direct PCM input not supported via trait interface. \
                 Use WhisperModel::transcribe_pcm() directly."
                    .to_string(),
            ))
        } else {
            Err(ModelError::InvalidInput(
                "Whisper expects 'mel' tensor input".to_string(),
            ))
        }
    }

    fn input_names(&self) -> Vec<&str> {
        vec!["mel"]
    }

    fn output_names(&self) -> Vec<&str> {
        vec!["text"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_from_str() {
        assert_eq!(
            CandleModelType::from_str("whisper"),
            CandleModelType::Whisper
        );
        assert_eq!(
            CandleModelType::from_str("WHISPER"),
            CandleModelType::Whisper
        );
        assert_eq!(CandleModelType::from_str("llama"), CandleModelType::LLaMA);
        assert_eq!(CandleModelType::from_str("bert"), CandleModelType::Bert);
        assert_eq!(
            CandleModelType::from_str("unknown"),
            CandleModelType::Generic
        );
    }
}

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult, ModelRuntime};
use crate::audio::convert::decode_wav_audio;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use super::whisper::{WhisperModel, WhisperConfig, WhisperSize};
use super::device::{select_device, DeviceSelection};

/// Candle-based model runtime implementation.
///
/// Manages Candle models (currently Whisper) and executes inference.
pub struct CandleRuntime {
    /// Cache of loaded models (key: model directory path)
    models: HashMap<String, WhisperModel>,
    /// Active model key (most recently loaded or used)
    active_model: Option<String>,
}

impl CandleRuntime {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            active_model: None,
        }
    }

    fn resolve_model(&mut self, input: &Envelope) -> AdapterResult<&mut WhisperModel> {
        // If envelope specifies a model_id/path in metadata, try to use it
        // Otherwise use active_model
        
        let model_key = if let Some(id) = input.metadata.get("model_id") {
             // For now assuming model_id might map to our cache keys
             Some(id.clone())
        } else {
            self.active_model.clone()
        };

        let key = model_key.ok_or_else(|| {
             AdapterError::ModelNotLoaded("No model selected and no active model".to_string())
        })?;

        // If key matches a loaded model, return it
        // Note: The key in our cache is currently the full path string
        // We might need better logical ID mapping later
        
        // Find best match in cache (exact match or suffix match)
        // Since we don't have the full path from just an ID sometimes
        let match_key = self.models.keys().find(|k| k.ends_with(&key) || k == &&key).cloned();

        if let Some(real_key) = match_key {
            self.models.get_mut(&real_key).ok_or_else(|| {
                AdapterError::ModelNotLoaded(format!("Model not found in cache: {}", real_key))
            })
        } else {
             Err(AdapterError::ModelNotLoaded(format!("Model '{}' not loaded", key)))
        }
    }
}

impl ModelRuntime for CandleRuntime {
    fn name(&self) -> &str {
        "candle"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["safetensors"]
    }

    fn load(&mut self, model_path: &Path) -> AdapterResult<()> {
        let path_str = model_path.to_string_lossy().to_string();
        
        // If path is a file (e.g. model.safetensors), use parent dir
        let model_dir = if model_path.is_file() {
            model_path.parent().unwrap_or(model_path)
        } else {
            model_path
        };
        let dir_str = model_dir.to_string_lossy().to_string();

        if self.models.contains_key(&dir_str) {
            self.active_model = Some(dir_str);
            return Ok(());
        }

        // Load model
        // Determine configuration (can infer from path or default)
        // For now, default to Tiny/English like TemplateExecutor, or read config
        let device = select_device(DeviceSelection::Auto)
            .map_err(|e| AdapterError::RuntimeError(format!("Device selection failed: {}", e)))?;

        // Try to load
        let model = WhisperModel::load(model_dir, &device)
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to load Candle model: {}", e)))?;

        self.models.insert(dir_str.clone(), model);
        self.active_model = Some(dir_str);

        Ok(())
    }

    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        // Need mutable access to model (which is in self.models)
        // We can't borrow self.models mutably AND call methods on self easily if we aren't careful
        // But resolve_model takes &mut self and returns &mut WhisperModel.
        // It borrows self.models, so we can't use other parts of self.
        
        // Resolve model first
        // If we don't have a specific model targeting mechanism in Envelope yet, we rely on active_model
        // But active_model is stored in self.
        
        // To simplify, just get the active model:
        let key = self.active_model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No active model loaded".to_string())
        })?.clone();

        let model = self.models.get_mut(&key).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Active model '{}' missing from cache", key))
        })?;

        match &input.kind {
            EnvelopeKind::Audio(bytes) => {
                // Decode audio
                // Whisper expects 16kHz mono
                 let samples = decode_wav_audio(bytes, 16000, 1)
                     .map_err(|e| AdapterError::InvalidInput(format!("Audio decode failed: {}", e)))?;
                
                // Transcribe
                let text = model.transcribe_pcm(&samples)
                    .map_err(|e| AdapterError::InferenceFailed(format!("Transcription failed: {}", e)))?;
                    
                Ok(Envelope::new(EnvelopeKind::Text(text)))
            }
            EnvelopeKind::Embedding(mel) => {
                 // Assume Mel spectrogram input [1, n_mels, n_frames] flattened
                 // We need to reconstruct Tensor from Vec<f32>
                 // Use helper from pcm_to_mel (but that does conversion)
                 // Need transcribe() method on model.
                 let tensor = model.pcm_to_mel_tensor(&[]) // Hack/Stub? No, we need direct mel tensor creation
                     .map_err(|e| AdapterError::RuntimeError(format!("Failed to create tensor context: {}", e)))?;
                 
                 // Actually we can't easily create a tensor from generic slice without device context which is inside model.
                 // We should add a method to WhisperModel to transcribe_from_mel_slice.
                 // For now, fail or implement if critical. TemplateExecutor mostly does Audio -> Text.
                 Err(AdapterError::InvalidInput("Direct Mel spectrogram input not fully supported in CandleRuntime yet".to_string()))
            }
            _ => Err(AdapterError::InvalidInput("Candle runtime expects Audio input".to_string()))
        }
    }
}

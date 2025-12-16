//! Template Executor - Metadata-driven model execution engine.
//!
//! This module implements the core execution logic that interprets ModelMetadata
//! and runs inference without hard-coding model-specific logic.
//!
//! The TemplateExecutor handles:
//! - Preprocessing (mel spectrogram, tokenization, normalization)
//! - Model execution (single-shot, autoregressive loops, iterative refinement)
//! - Postprocessing (BPE decoding, argmax, sampling)
//! - Multi-stage pipelines (encoder → decoder, etc.)

use crate::execution_template::{
    ExecutionMode, ExecutionTemplate, MelScaleType, ModelMetadata, PipelineStage,
    PostprocessingStep, PreprocessingStep,
};
use crate::ir::{Envelope, EnvelopeKind};
use crate::tracing as trace;
// Unified mel spectrogram API
use crate::audio::mel::{compute_mel_spectrogram, MelConfig, MelScale, PaddingMode};
// Legacy imports for audio bytes handling
use crate::preprocessing::mel_spectrogram::audio_bytes_to_whisper_mel;
use crate::runtime_adapter::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{ArrayD, IxDyn};
use ort::value::Value;
use std::collections::HashMap;
use std::path::Path;

pub type ExecutorResult<T> = Result<T, AdapterError>;

/// Execute TTS inference with phoneme IDs, voice embedding, and speed
/// This is a standalone function to avoid borrow issues with the executor
fn execute_tts_inference(
    session: &ONNXSession,
    phoneme_ids: &[i64],
    voice_embedding: Vec<f32>,
) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
    use ndarray::{Array1, Array2};
    use ort::value::Value;

    // Get model input names
    let input_names = session.input_names();

    let batch_size = 1;
    let seq_len = phoneme_ids.len();
    let embedding_len = voice_embedding.len();

    // Build inputs - we need to create Value objects fresh for each input
    // because ort::Value doesn't implement Clone for dynamic types
    let mut value_inputs: HashMap<String, Value> = HashMap::new();

    for input_name in input_names.iter() {
        // Token/phoneme IDs input - KittenTTS uses "input_ids", Kokoro uses "tokens"
        if input_name.contains("input_ids") || input_name == "input_ids"
            || input_name.contains("tokens") || input_name == "tokens" {
            let arr = Array2::<i64>::from_shape_vec(
                (batch_size, seq_len),
                phoneme_ids.to_vec(),
            ).map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to create input_ids array: {}", e))
            })?;
            let val: Value = Value::from_array(arr)
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create input_ids value: {}", e))
                })?
                .into();
            value_inputs.insert(input_name.clone(), val);
        } else if input_name.contains("style") || input_name == "style" {
            let arr = Array2::<f32>::from_shape_vec(
                (1, embedding_len),
                voice_embedding.clone(),
            ).map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to create style array: {}", e))
            })?;
            let val: Value = Value::from_array(arr)
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create style value: {}", e))
                })?
                .into();
            value_inputs.insert(input_name.clone(), val);
        } else if input_name.contains("speed") || input_name == "speed" {
            let arr = Array1::<f32>::from_vec(vec![1.0]);
            let val: Value = Value::from_array(arr)
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create speed value: {}", e))
                })?
                .into();
            value_inputs.insert(input_name.clone(), val);
        }
    }

    // Verify we mapped all inputs
    if value_inputs.len() != input_names.len() {
        return Err(AdapterError::InvalidInput(format!(
            "TTS model input mismatch. Expected {} inputs ({:?}), mapped {}",
            input_names.len(),
            input_names,
            value_inputs.len()
        )));
    }

    // Run inference
    session.run_with_values(value_inputs)
}

/// Template-driven executor that interprets metadata and runs models
pub struct TemplateExecutor {
    /// Cache of loaded ONNX sessions (model_file -> session)
    session_cache: HashMap<String, ONNXSession>,

    /// Cache of loaded Candle Whisper models (model_file -> model)
    #[cfg(feature = "candle")]
    candle_whisper_cache: HashMap<String, crate::runtime_adapter::candle::WhisperModel>,

    /// Base path for resolving model files
    base_path: String,
}

impl TemplateExecutor {
    /// Create a new TemplateExecutor
    pub fn new() -> Self {
        Self {
            session_cache: HashMap::new(),
            #[cfg(feature = "candle")]
            candle_whisper_cache: HashMap::new(),
            base_path: String::new(),
        }
    }

    /// Create a new TemplateExecutor with a base path for model files
    pub fn with_base_path(base_path: impl Into<String>) -> Self {
        Self {
            session_cache: HashMap::new(),
            #[cfg(feature = "candle")]
            candle_whisper_cache: HashMap::new(),
            base_path: base_path.into(),
        }
    }

    /// Execute a model based on its metadata
    pub fn execute(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        // Start execution span
        let _exec_span = trace::SpanGuard::new(format!("execute:{}", metadata.model_id));
        trace::add_metadata("model_id", &metadata.model_id);
        trace::add_metadata("version", &metadata.version);

        // Step 1: Run preprocessing pipeline
        let preprocessed = self.run_preprocessing(metadata, input)?;

        // Step 2: Execute based on template type
        let raw_outputs = match &metadata.execution_template {
            ExecutionTemplate::SimpleMode { model_file } => {
                let _span = trace::SpanGuard::new("onnx_inference");
                trace::add_metadata("model_file", model_file);
                trace::add_metadata("mode", "simple");
                self.execute_simple_mode(model_file, preprocessed, metadata)?
            }
            ExecutionTemplate::CandleModel {
                model_file,
                config_file,
                tokenizer_file,
                model_type,
            } => {
                #[cfg(feature = "candle")]
                {
                    let _span = trace::SpanGuard::new("candle_inference");
                    trace::add_metadata("model_file", model_file);
                    trace::add_metadata("mode", "candle");
                    if let Some(mt) = model_type {
                        trace::add_metadata("model_type", mt);
                    }
                    self.execute_candle_model(
                        model_file,
                        config_file.as_deref(),
                        tokenizer_file.as_deref(),
                        model_type.as_deref(),
                        preprocessed,
                        metadata,
                    )?
                }
                #[cfg(not(feature = "candle"))]
                {
                    return Err(AdapterError::RuntimeError(
                        "Candle support not enabled. Rebuild with --features candle".to_string(),
                    ));
                }
            }
            ExecutionTemplate::Pipeline { stages, config } => {
                let _span = trace::SpanGuard::new("pipeline_inference");
                trace::add_metadata("stages", &stages.len().to_string());
                self.execute_pipeline(stages, config, preprocessed, metadata)?
            }
        };

        // Step 3: Run postprocessing pipeline
        let output = self.run_postprocessing(metadata, raw_outputs)?;

        Ok(output)
    }

    /// Run preprocessing steps from metadata
    fn run_preprocessing(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<PreprocessedData> {
        if metadata.preprocessing.is_empty() {
            return PreprocessedData::from_envelope(input);
        }

        let _preprocess_span = trace::SpanGuard::new("preprocessing");
        trace::add_metadata("steps", &metadata.preprocessing.len().to_string());

        let mut data = PreprocessedData::from_envelope(input)?;

        for step in &metadata.preprocessing {
            // Create span for each preprocessing step
            let step_name = format!("preprocessing:{}", step.step_name());
            let _step_span = trace::SpanGuard::new(&step_name);

            data = self.apply_preprocessing_step(step, data, metadata, input)?;
        }

        Ok(data)
    }

    /// Apply a single preprocessing step.
    fn apply_preprocessing_step(
        &mut self,
        step: &PreprocessingStep,
        data: PreprocessedData,
        _metadata: &ModelMetadata,
        input_envelope: &Envelope,
    ) -> ExecutorResult<PreprocessedData> {
        match step {
            PreprocessingStep::MelSpectrogram {
                preset,
                n_mels,
                sample_rate,
                fft_size,
                hop_length,
                mel_scale,
                max_frames,
            } => mel_spectrogram_step(
                data,
                preset.as_deref(),
                *n_mels,
                *sample_rate,
                *fft_size,
                *hop_length,
                *mel_scale,
                *max_frames,
            ),

            PreprocessingStep::Tokenize {
                vocab_file,
                tokenizer_type,
                max_length,
            } => tokenize_step(data, &self.resolve_file_path(vocab_file), tokenizer_type, *max_length),

            PreprocessingStep::Normalize { mean, std } => normalize_step(data, mean, std),

            PreprocessingStep::AudioDecode {
                sample_rate,
                channels,
            } => decode_audio_step(data, input_envelope, *sample_rate, *channels),

            PreprocessingStep::Reshape { shape } => reshape_step(data, shape),

            PreprocessingStep::CenterCrop { width, height } => center_crop_step(data, *width, *height),

            PreprocessingStep::Resize {
                width,
                height,
                interpolation,
            } => resize_step(data, *width, *height, interpolation),

            PreprocessingStep::Phonemize {
                tokens_file,
                backend,
                dict_file,
                language,
                add_padding,
                normalize_text,
            } => {
                let dict_path = dict_file.as_ref().map(|p| self.resolve_file_path(p));
                phonemize_step(
                    data,
                    &self.resolve_file_path(tokens_file),
                    backend,
                    dict_path.as_deref(),
                    language.as_deref(),
                    *add_padding,
                    *normalize_text,
                )
            }
        }
    }

    /// Execute SimpleMode: single model, run once
    fn execute_simple_mode(
        &mut self,
        model_file: &str,
        input: PreprocessedData,
        metadata: &ModelMetadata,
    ) -> ExecutorResult<RawOutputs> {
        // For TTS models (PhonemeIds), load voice embedding first to avoid borrow conflicts
        let tts_voice_embedding = if input.is_phoneme_ids() {
            Some(self.load_voice_embedding(metadata, 0)?)
        } else {
            None
        };

        let session = self.get_or_load_session(model_file)?;

        // Get input names from the ONNX model
        let input_names = session.input_names();
        if input_names.is_empty() {
            return Err(AdapterError::InvalidInput(
                "Model has no inputs".to_string(),
            ));
        }

        // Build input map based on input type
        let outputs = match &input {
            PreprocessedData::TokenIds {
                ids,
                attention_mask,
                token_type_ids,
                ..
            } => {
                // Multi-input case for BERT-style models
                // BERT models expect int64 tensors, so we use run_with_values()
                use ort::value::Value;
                use ndarray::Array2;

                let batch_size = 1;
                let seq_len = ids.len();

                // Convert token IDs to int64 tensor [batch_size, seq_len]
                let input_ids_data: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
                let input_ids_array = Array2::<i64>::from_shape_vec(
                    (batch_size, seq_len),
                    input_ids_data,
                )
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create input_ids array: {}", e))
                })?;
                let input_ids_value = Value::from_array(input_ids_array)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!("Failed to create input_ids value: {}", e))
                    })?;

                // Convert attention mask to int64 tensor [batch_size, seq_len]
                let attention_mask_data: Vec<i64> =
                    attention_mask.iter().map(|&mask| mask as i64).collect();
                let attention_mask_array = Array2::<i64>::from_shape_vec(
                    (batch_size, seq_len),
                    attention_mask_data,
                )
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create attention_mask array: {}", e))
                })?;
                let attention_mask_value = Value::from_array(attention_mask_array)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!("Failed to create attention_mask value: {}", e))
                    })?;

                // Convert token type IDs to int64 tensor [batch_size, seq_len]
                let token_type_ids_data: Vec<i64> =
                    token_type_ids.iter().map(|&type_id| type_id as i64).collect();
                let token_type_ids_array = Array2::<i64>::from_shape_vec(
                    (batch_size, seq_len),
                    token_type_ids_data,
                )
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create token_type_ids array: {}", e))
                })?;
                let token_type_ids_value = Value::from_array(token_type_ids_array)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!("Failed to create token_type_ids value: {}", e))
                    })?;

                // Map values to ONNX input names
                let mut value_inputs = HashMap::new();

                // Match ONNX input names to our values
                // BERT models typically have: input_ids, attention_mask, token_type_ids
                // Convert to dynamic Value type using .into()
                for input_name in input_names.iter() {
                    if input_name.contains("input_ids") || input_name == "input_ids" {
                        value_inputs.insert(input_name.clone(), input_ids_value.clone().into());
                    } else if input_name.contains("attention_mask") || input_name == "attention_mask"
                    {
                        value_inputs.insert(input_name.clone(), attention_mask_value.clone().into());
                    } else if input_name.contains("token_type_ids") || input_name == "token_type_ids"
                    {
                        value_inputs.insert(input_name.clone(), token_type_ids_value.clone().into());
                    }
                }

                // Verify we mapped all inputs
                if value_inputs.len() != input_names.len() {
                    return Err(AdapterError::InvalidInput(format!(
                        "Could not map all model inputs. Expected {} inputs, mapped {}. Input names: {:?}",
                        input_names.len(),
                        value_inputs.len(),
                        input_names
                    )));
                }

                // Run inference with mixed types using run_with_values
                session.run_with_values(value_inputs)?
            }
            PreprocessedData::PhonemeIds { ids, .. } => {
                // TTS model case - requires input_ids, style (voice embedding), and speed
                // Voice embedding was pre-loaded before getting session to avoid borrow issues
                let voice_embedding = tts_voice_embedding
                    .ok_or_else(|| AdapterError::InvalidInput("TTS voice embedding not loaded".to_string()))?;
                execute_tts_inference(session, ids, voice_embedding)?
            }
            _ => {
                // Single-input case (original behavior)
                let input_tensor = input.to_tensor()?;
                let mut input_map = HashMap::new();
                input_map.insert(input_names[0].clone(), input_tensor);
                session.run(input_map)?
            }
        };

        Ok(RawOutputs::TensorMap(outputs))
    }


    /// Load voice embedding from voices.bin file
    fn load_voice_embedding(
        &self,
        metadata: &ModelMetadata,
        voice_index: usize,
    ) -> ExecutorResult<Vec<f32>> {
        // Find voices.bin in the model files
        let voices_file = metadata
            .files
            .iter()
            .find(|f| f.contains("voices.bin"))
            .ok_or_else(|| {
                AdapterError::InvalidInput("TTS model missing voices.bin file".to_string())
            })?;

        let voices_path = self.resolve_file_path(voices_file);

        // Check file format (NPZ vs raw binary)
        let voices_bytes = std::fs::read(&voices_path).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to read voices.bin: {}", e))
        })?;

        // Get voice embedding dimension from metadata (default 256)
        let embedding_dim = metadata
            .metadata
            .get("voice_embedding_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;

        // Check if file is NPZ format (starts with PK magic bytes for ZIP)
        if voices_bytes.len() >= 2 && voices_bytes[0] == b'P' && voices_bytes[1] == b'K' {
            // NPZ format (Kokoro-style)
            self.load_voice_embedding_npz(&voices_path, metadata, voice_index, embedding_dim)
        } else {
            // Raw binary format (KittenTTS-style)
            self.load_voice_embedding_raw(&voices_bytes, voice_index, embedding_dim)
        }
    }

    /// Load voice embedding from NPZ file (Kokoro format)
    /// NPZ contains named arrays, each with shape (510, 1, 256)
    /// The embedding is selected based on token length
    fn load_voice_embedding_npz(
        &self,
        voices_path: &str,
        metadata: &ModelMetadata,
        voice_index: usize,
        embedding_dim: usize,
    ) -> ExecutorResult<Vec<f32>> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        let file = File::open(voices_path).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to open voices NPZ: {}", e))
        })?;

        let mut npz = NpzReader::new(file).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to read NPZ file: {}", e))
        })?;

        // Get list of voice names
        let voice_names = npz.names().map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to get NPZ names: {}", e))
        })?;

        // Get voice name by index or from metadata
        let voice_name = if let Some(names) = metadata.metadata.get("voice_names") {
            if let Some(names_array) = names.as_array() {
                if voice_index < names_array.len() {
                    names_array[voice_index]
                        .as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| {
                            AdapterError::InvalidInput("Invalid voice name in metadata".to_string())
                        })?
                } else {
                    return Err(AdapterError::InvalidInput(format!(
                        "Voice index {} out of range (max {})",
                        voice_index,
                        names_array.len() - 1
                    )));
                }
            } else {
                // Fall back to first voice in NPZ
                voice_names.first().cloned().ok_or_else(|| {
                    AdapterError::InvalidInput("No voices in NPZ file".to_string())
                })?
            }
        } else {
            // Fall back to first voice in NPZ
            voice_names.first().cloned().ok_or_else(|| {
                AdapterError::InvalidInput("No voices in NPZ file".to_string())
            })?
        };

        // Load the voice array - shape is (510, 1, 256)
        let voice_data: ndarray::Array3<f32> = npz.by_name(&voice_name).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to load voice '{}': {}", voice_name, e))
        })?;

        // Use a default token length for the embedding (e.g., middle of range)
        // In actual use, this should be based on the number of tokens
        // For now, use index 100 as a reasonable default
        let token_len_idx = 100.min(voice_data.shape()[0] - 1);

        // Extract embedding at token_len_idx, row 0
        let embedding: Vec<f32> = voice_data
            .slice(ndarray::s![token_len_idx, 0, ..])
            .iter()
            .copied()
            .collect();

        if embedding.len() != embedding_dim {
            return Err(AdapterError::InvalidInput(format!(
                "Voice embedding dimension mismatch: expected {}, got {}",
                embedding_dim,
                embedding.len()
            )));
        }

        Ok(embedding)
    }

    /// Load voice embedding from raw binary file (KittenTTS format)
    fn load_voice_embedding_raw(
        &self,
        voices_bytes: &[u8],
        voice_index: usize,
        embedding_dim: usize,
    ) -> ExecutorResult<Vec<f32>> {
        // Each voice embedding is embedding_dim * 4 bytes (f32)
        let voice_size = embedding_dim * 4;
        let num_voices = voices_bytes.len() / voice_size;

        if voice_index >= num_voices {
            return Err(AdapterError::InvalidInput(format!(
                "Voice index {} out of range (max {})",
                voice_index,
                num_voices - 1
            )));
        }

        // Extract voice embedding
        let start = voice_index * voice_size;
        let end = start + voice_size;
        let voice_bytes = &voices_bytes[start..end];

        // Convert bytes to f32 (little-endian)
        let voice_embedding: Vec<f32> = voice_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(bytes)
            })
            .collect();

        Ok(voice_embedding)
    }

    /// Execute CandleModel: Whisper ASR using Candle runtime
    #[cfg(feature = "candle")]
    fn execute_candle_model(
        &mut self,
        model_file: &str,
        _config_file: Option<&str>,
        _tokenizer_file: Option<&str>,
        model_type: Option<&str>,
        input: PreprocessedData,
        _metadata: &ModelMetadata,
    ) -> ExecutorResult<RawOutputs> {
        use crate::runtime_adapter::candle::{select_device, DeviceSelection, WhisperConfig, WhisperModel, WhisperSize};

        // Currently only Whisper is supported
        let model_type = model_type.unwrap_or("whisper");
        if model_type != "whisper" {
            return Err(AdapterError::RuntimeError(format!(
                "Unsupported Candle model type: {}. Currently only 'whisper' is supported.",
                model_type
            )));
        }

        // Extract PCM audio from preprocessed data
        let pcm_data = match &input {
            PreprocessedData::AudioSamples(samples) => samples.clone(),
            PreprocessedData::Tensor(tensor) => {
                // If tensor is 1D or flat, treat as PCM
                tensor.as_slice().map(|s| s.to_vec()).ok_or_else(|| {
                    AdapterError::InvalidInput("Cannot extract PCM from tensor".to_string())
                })?
            }
            _ => {
                return Err(AdapterError::InvalidInput(
                    "Candle Whisper expects AudioSamples (PCM f32 @ 16kHz) input. \
                     Add 'AudioDecode' preprocessing step.".to_string()
                ));
            }
        };

        // Get or load model
        let model_path = self.resolve_file_path(model_file);
        let model_dir = std::path::Path::new(&model_path)
            .parent()
            .unwrap_or(std::path::Path::new(&self.base_path));

        // Check if model is cached
        let cache_key = model_dir.to_string_lossy().to_string();
        if !self.candle_whisper_cache.contains_key(&cache_key) {
            // Load model
            let device = select_device(DeviceSelection::Auto)
                .map_err(|e| AdapterError::RuntimeError(format!("Device selection failed: {}", e)))?;

            let config = WhisperConfig {
                model_size: WhisperSize::Tiny,
                language: Some("en".to_string()),
                ..Default::default()
            };

            let model = WhisperModel::load_with_config(model_dir, &device, config)
                .map_err(|e| AdapterError::RuntimeError(format!("Failed to load Candle model: {}", e)))?;

            self.candle_whisper_cache.insert(cache_key.clone(), model);
        }

        // Run transcription
        let model = self.candle_whisper_cache.get_mut(&cache_key).unwrap();
        let text = model.transcribe_pcm(&pcm_data)
            .map_err(|e| AdapterError::InferenceFailed(format!("Whisper transcription failed: {}", e)))?;

        // Return as text output
        Ok(RawOutputs::Text(text))
    }

    /// Execute Pipeline: multi-stage execution with control flow
    fn execute_pipeline(
        &mut self,
        stages: &[PipelineStage],
        config: &HashMap<String, serde_json::Value>,
        initial_input: PreprocessedData,
        metadata: &ModelMetadata,
    ) -> ExecutorResult<RawOutputs> {
        let mut stage_outputs: HashMap<String, HashMap<String, ArrayD<f32>>> = HashMap::new();
        let mut current_data = initial_input;

        for stage in stages {
            match &stage.execution_mode {
                ExecutionMode::SingleShot => {
                    let outputs = self.execute_stage_single_shot(stage, &current_data, &stage_outputs)?;
                    stage_outputs.insert(stage.name.clone(), outputs.clone());

                    // For single-shot, pass outputs to next stage
                    // Take the first output as the current data (simplified)
                    if let Some(first_output) = outputs.values().next() {
                        current_data = PreprocessedData::Tensor(first_output.clone());
                    }
                }

                ExecutionMode::Autoregressive {
                    max_tokens,
                    start_token_id,
                    end_token_id,
                    repetition_penalty,
                } => {
                    let token_ids = self.execute_stage_autoregressive(
                        stage,
                        &stage_outputs,
                        config,
                        *max_tokens,
                        *start_token_id,
                        *end_token_id,
                        *repetition_penalty,
                    )?;

                    // Return token IDs for postprocessing
                    return Ok(RawOutputs::TokenIds(token_ids));
                }

                ExecutionMode::IterativeRefinement {
                    num_steps,
                    schedule: _,
                } => {
                    // TODO: Implement iterative refinement for diffusion models
                    return Err(AdapterError::InvalidInput(
                        format!("IterativeRefinement not yet implemented (needs {} steps)", num_steps),
                    ));
                }

                ExecutionMode::WhisperDecoder {
                    max_tokens,
                    start_token_id,
                    end_token_id,
                    language_token_id,
                    task_token_id,
                    no_timestamps_token_id,
                    suppress_tokens,
                    repetition_penalty,
                } => {
                    let token_ids = self.execute_stage_whisper_decoder(
                        stage,
                        &stage_outputs,
                        config,
                        *max_tokens,
                        *start_token_id,
                        *end_token_id,
                        *language_token_id,
                        *task_token_id,
                        *no_timestamps_token_id,
                        suppress_tokens,
                        *repetition_penalty,
                    )?;

                    // Return token IDs for postprocessing
                    return Ok(RawOutputs::TokenIds(token_ids));
                }
            }
        }

        // Return the last stage's outputs
        if let Some((_, outputs)) = stage_outputs.iter().last() {
            Ok(RawOutputs::TensorMap(outputs.clone()))
        } else {
            Err(AdapterError::InvalidInput(
                "Pipeline produced no outputs".to_string(),
            ))
        }
    }

    /// Execute a single-shot stage (run once)
    fn execute_stage_single_shot(
        &mut self,
        stage: &PipelineStage,
        current_data: &PreprocessedData,
        _stage_outputs: &HashMap<String, HashMap<String, ArrayD<f32>>>,
    ) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
        let session = self.get_or_load_session(&stage.model_file)?;

        // Convert input to tensor
        let input_tensor = current_data.to_tensor()?;

        // Query ONNX session for actual input names (robust to metadata mismatches)
        let actual_input_names = session.input_names();
        if actual_input_names.is_empty() {
            return Err(AdapterError::InvalidInput(
                "Model has no inputs".to_string(),
            ));
        }

        // Build input map using actual ONNX input name (first input)
        // This makes the system robust to metadata naming mismatches
        let mut inputs = HashMap::new();
        inputs.insert(actual_input_names[0].clone(), input_tensor);

        // Run inference
        let outputs = session.run(inputs)?;

        Ok(outputs)
    }

    /// Execute an autoregressive stage (token generation loop)
    fn execute_stage_autoregressive(
        &mut self,
        stage: &PipelineStage,
        stage_outputs: &HashMap<String, HashMap<String, ArrayD<f32>>>,
        config: &HashMap<String, serde_json::Value>,
        max_tokens: usize,
        start_token_id: i64,
        end_token_id: i64,
        repetition_penalty: f32,
    ) -> ExecutorResult<Vec<usize>> {
        // Clone the model file path to avoid borrow issues
        let model_file = stage.model_file.clone();

        // Extract KV cache shape from config
        let kv_cache_shape = if let Some(shape_value) = config.get("kv_cache_shape") {
            shape_value
                .as_array()
                .ok_or_else(|| {
                    AdapterError::InvalidInput("kv_cache_shape must be an array".to_string())
                })?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| {
                            AdapterError::InvalidInput("kv_cache_shape values must be numbers".to_string())
                        })
                        .map(|n| n as usize)
                })
                .collect::<Result<Vec<usize>, _>>()?
        } else {
            return Err(AdapterError::InvalidInput(
                "Autoregressive stage requires kv_cache_shape in config".to_string(),
            ));
        };

        // Initialize KV caches with zeros
        let kv_cache_size: usize = kv_cache_shape.iter().product();
        let kv_cache_data = vec![0.0f32; kv_cache_size];

        let mut kv_cache_k = ArrayD::<f32>::from_shape_vec(IxDyn(&kv_cache_shape), kv_cache_data.clone())
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to create KV cache K: {:?}", e)))?;
        let mut kv_cache_v = ArrayD::<f32>::from_shape_vec(IxDyn(&kv_cache_shape), kv_cache_data)
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to create KV cache V: {:?}", e)))?;

        // Get encoder outputs (cross-attention keys/values)
        let encoder_outputs = stage_outputs
            .values()
            .next()
            .ok_or_else(|| AdapterError::InvalidInput("No encoder outputs found".to_string()))?;

        // Extract cross-attention keys/values using helper (handles name variations)
        // Clone them so we don't hold a borrow of encoder_outputs
        let (cross_k, cross_v) = self.extract_encoder_cross_attention(encoder_outputs)?;
        let cross_k = cross_k.clone();
        let cross_v = cross_v.clone();

        // Get the session after extracting encoder outputs
        let session = self.get_or_load_session(&model_file)?;

        // Autoregressive loop
        let mut token_ids = vec![start_token_id as usize];
        let mut offset = 0i64;

        for _ in 0..max_tokens {
            // Create tokens tensor [batch=1, seq_len=1]
            let current_token_id = *token_ids.last().unwrap() as i64;
            let tokens_shape = vec![1, 1];
            let tokens_data = vec![current_token_id];
            let tokens_i64 = ArrayD::<i64>::from_shape_vec(IxDyn(&tokens_shape), tokens_data)
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to create tokens tensor: {:?}", e)))?;
            let tokens_value: Value = Value::from_array(tokens_i64)
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert tokens: {:?}", e)))?
                .into();

            // Convert caches to Values
            let kv_cache_k_value: Value = Value::from_array(kv_cache_k.clone())
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert KV cache K: {:?}", e)))?
                .into();
            let kv_cache_v_value: Value = Value::from_array(kv_cache_v.clone())
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert KV cache V: {:?}", e)))?
                .into();

            let cross_k_value: Value = Value::from_array(cross_k.clone())
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert cross_k: {:?}", e)))?
                .into();
            let cross_v_value: Value = Value::from_array(cross_v.clone())
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert cross_v: {:?}", e)))?
                .into();

            // Offset tensor
            let offset_shape = vec![1];
            let offset_data = vec![offset];
            let offset_i64 = ArrayD::<i64>::from_shape_vec(IxDyn(&offset_shape), offset_data)
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to create offset tensor: {:?}", e)))?;
            let offset_value: Value = Value::from_array(offset_i64)
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert offset: {:?}", e)))?
                .into();

            // Query ONNX session for actual input/output names (robust to metadata mismatches)
            let actual_input_names = session.input_names();
            let actual_output_names = session.output_names();

            if actual_input_names.len() < 6 {
                return Err(AdapterError::InvalidInput(format!(
                    "Decoder model expected 6 inputs, found {}",
                    actual_input_names.len()
                )));
            }

            if actual_output_names.len() < 3 {
                return Err(AdapterError::InvalidInput(format!(
                    "Decoder model expected 3 outputs, found {}",
                    actual_output_names.len()
                )));
            }

            // Build decoder inputs using actual ONNX input names (in order)
            // Order: [tokens, in_kv_k, in_kv_v, cross_k, cross_v, offset]
            let mut decoder_inputs = HashMap::new();
            decoder_inputs.insert(actual_input_names[0].clone(), tokens_value);
            decoder_inputs.insert(actual_input_names[1].clone(), kv_cache_k_value);
            decoder_inputs.insert(actual_input_names[2].clone(), kv_cache_v_value);
            decoder_inputs.insert(actual_input_names[3].clone(), cross_k_value);
            decoder_inputs.insert(actual_input_names[4].clone(), cross_v_value);
            decoder_inputs.insert(actual_input_names[5].clone(), offset_value);

            // Run decoder
            let decoder_outputs = session.run_with_values(decoder_inputs)
                .map_err(|e| AdapterError::InvalidInput(format!("Decoder inference failed: {:?}", e)))?;

            // Extract logits and updated KV caches using actual output names
            // Order: [logits, out_kv_k, out_kv_v]
            let logits = decoder_outputs
                .get(&actual_output_names[0])
                .ok_or_else(|| AdapterError::InvalidInput("Missing logits output".to_string()))?
                .clone();

            if let Some(updated_k) = decoder_outputs.get(&actual_output_names[1]) {
                kv_cache_k = updated_k.clone();
            }
            if let Some(updated_v) = decoder_outputs.get(&actual_output_names[2]) {
                kv_cache_v = updated_v.clone();
            }

            // Apply repetition penalty if enabled
            let mut logits = logits;
            if repetition_penalty > 0.0 && token_ids.len() > 1 {
                let recent_tokens: std::collections::HashSet<usize> =
                    token_ids.iter().rev().take(10).copied().collect();

                if let Some(logits_slice) = logits.as_slice_mut() {
                    for token_id in &recent_tokens {
                        if *token_id < logits_slice.len() {
                            logits_slice[*token_id] *= repetition_penalty;
                        }
                    }
                }
            }

            // Get next token (argmax)
            let next_token_id = Self::argmax_token(&logits)?;

            // Check for end token
            if next_token_id == end_token_id as usize {
                break;
            }

            // Check for repetition (stop if same token appears 5+ times in a row)
            if token_ids.len() >= 5 {
                let last_five: Vec<usize> = token_ids.iter().rev().take(5).copied().collect();
                if last_five.iter().all(|&id| id == next_token_id) {
                    break;
                }
            }

            token_ids.push(next_token_id);
            offset += 1;
        }

        Ok(token_ids)
    }

    /// Execute Whisper decoder stage (HuggingFace ONNX format)
    ///
    /// This handles the onnx-community/whisper-* format where:
    /// - Encoder outputs `last_hidden_state` [batch, 1500, hidden_size]
    /// - Decoder takes `input_ids` + 16 past_key_values tensors
    /// - Decoder outputs `logits` + 8 present tensors (decoder KV only)
    #[allow(clippy::too_many_arguments)]
    fn execute_stage_whisper_decoder(
        &mut self,
        stage: &PipelineStage,
        stage_outputs: &HashMap<String, HashMap<String, ArrayD<f32>>>,
        config: &HashMap<String, serde_json::Value>,
        max_tokens: usize,
        start_token_id: i64,
        end_token_id: i64,
        language_token_id: i64,
        task_token_id: i64,
        no_timestamps_token_id: i64,
        suppress_tokens: &[i64],
        repetition_penalty: f32,
    ) -> ExecutorResult<Vec<usize>> {
        use ndarray::Array2;

        // Get config values
        let num_layers = config.get("num_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        let num_heads = config.get("num_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(6) as usize;
        let head_dim = config.get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let encoder_seq_len = config.get("encoder_seq_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(1500) as usize;

        // Get encoder hidden states from previous stage
        let encoder_outputs = stage_outputs
            .get("encoder")
            .ok_or_else(|| AdapterError::InvalidInput("No encoder outputs found".to_string()))?;

        let encoder_hidden_states = encoder_outputs
            .get("last_hidden_state")
            .or_else(|| encoder_outputs.values().next())
            .ok_or_else(|| AdapterError::InvalidInput("No encoder hidden states".to_string()))?;

        // Clone encoder hidden states shape for computing encoder KV cache
        let enc_shape = encoder_hidden_states.shape();
        let batch_size = enc_shape[0];
        let _hidden_size = enc_shape[2]; // Used for validation

        // Load decoder model
        let model_file = stage.model_file.clone();
        let session = self.get_or_load_session(&model_file)?;

        // Get input/output names
        let input_names = session.input_names();
        let _output_names = session.output_names(); // Available for debugging

        // Initialize token sequence with forced decoder IDs
        // Whisper expects: <|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]
        let forced_tokens: Vec<i64> = vec![
            start_token_id,           // <|startoftranscript|> = 50258
            language_token_id,        // <|en|> = 50259
            task_token_id,            // <|transcribe|> = 50359
            no_timestamps_token_id,   // <|notimestamps|> = 50363
        ];
        let num_forced = forced_tokens.len();

        // Initialize decoder KV cache (starts empty, grows each step)
        // Shape: [batch, num_heads, 0, head_dim] → grows to [batch, num_heads, seq_len, head_dim]
        let mut decoder_kv_cache: Vec<ArrayD<f32>> = Vec::new();
        for _ in 0..(num_layers * 2) { // key + value for each layer
            let kv = ArrayD::<f32>::zeros(IxDyn(&[batch_size, num_heads, 0, head_dim]));
            decoder_kv_cache.push(kv);
        }

        // Encoder KV cache is computed once and reused
        // For HF Whisper ONNX, the encoder KV is derived from encoder_hidden_states
        // through the decoder's cross-attention layers on the first forward pass
        // We initialize with zeros and let the model populate it
        let mut encoder_kv_cache: Vec<ArrayD<f32>> = (0..(num_layers * 2))
            .map(|_| ArrayD::<f32>::zeros(IxDyn(&[batch_size, num_heads, encoder_seq_len, head_dim])))
            .collect();

        // Track generated tokens (starts with forced tokens)
        let mut generated_tokens: Vec<usize> = forced_tokens.iter().map(|&t| t as usize).collect();

        // Convert suppress_tokens to a HashSet for fast lookup
        let suppress_set: std::collections::HashSet<i64> = suppress_tokens.iter().copied().collect();


        // Autoregressive loop
        for step in 0..max_tokens {
            // Get the token to process
            // For forced tokens, use them in sequence; otherwise use the last generated token
            let current_token = if step < num_forced {
                forced_tokens[step]
            } else {
                *generated_tokens.last().unwrap() as i64
            };

            // Create input_ids tensor [batch, 1]
            let input_ids = Array2::<i64>::from_shape_vec(
                (batch_size, 1),
                vec![current_token; batch_size],
            ).map_err(|e| AdapterError::InvalidInput(format!("Failed to create input_ids: {}", e)))?;

            let input_ids_value: Value = Value::from_array(input_ids)
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert input_ids: {}", e)))?
                .into();

            // Build inputs map
            let mut inputs: HashMap<String, Value> = HashMap::new();

            // Add input_ids
            if let Some(name) = input_names.iter().find(|n| n.contains("input_ids")) {
                inputs.insert(name.clone(), input_ids_value);
            } else if !input_names.is_empty() {
                inputs.insert(input_names[0].clone(), input_ids_value);
            }

            // Add encoder hidden states (required for cross-attention)
            // The HF ONNX model needs encoder_hidden_states passed through
            let enc_hidden_value: Value = Value::from_array(encoder_hidden_states.clone())
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert encoder hidden states: {}", e)))?
                .into();

            // Find and set encoder KV cache inputs
            for name in input_names.iter() {
                if name.contains("past_key_values") {
                    // Parse layer index and type from name
                    // Format: past_key_values.{layer}.{decoder|encoder}.{key|value}
                    if let Some(captures) = parse_kv_cache_name(name) {
                        let (layer, is_encoder, is_key) = captures;

                        if is_encoder {
                            // Encoder KV cache (fixed size, computed from cross-attention)
                            let kv_idx = layer * 2 + if is_key { 0 } else { 1 };
                            if kv_idx < encoder_kv_cache.len() {
                                let kv_value: Value = Value::from_array(encoder_kv_cache[kv_idx].clone())
                                    .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert encoder KV: {}", e)))?
                                    .into();
                                inputs.insert(name.clone(), kv_value);
                            }
                        } else {
                            // Decoder KV cache (grows with each step)
                            let kv_idx = layer * 2 + if is_key { 0 } else { 1 };
                            if kv_idx < decoder_kv_cache.len() {
                                let kv_value: Value = Value::from_array(decoder_kv_cache[kv_idx].clone())
                                    .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert decoder KV: {}", e)))?
                                    .into();
                                inputs.insert(name.clone(), kv_value);
                            }
                        }
                    }
                }
            }

            // Check if we have encoder_hidden_states input
            if let Some(name) = input_names.iter().find(|n| n.contains("encoder_hidden_states")) {
                inputs.insert(name.clone(), enc_hidden_value);
            }

            // Run decoder
            let outputs = session.run_with_values(inputs)
                .map_err(|e| AdapterError::InferenceFailed(format!("Whisper decoder failed: {}", e)))?;

            // Get logits from outputs
            let logits = outputs.get("logits")
                .or_else(|| outputs.values().next())
                .ok_or_else(|| AdapterError::InvalidInput("No logits output".to_string()))?;

            // Update KV cache from present outputs
            for (name, tensor) in &outputs {
                if name.starts_with("present.") {
                    // Parse present.{layer}.{decoder|encoder}.{key|value}
                    // e.g., "present.0.decoder.key", "present.0.encoder.key"
                    if let Some((layer, is_encoder, is_key)) = parse_present_name_full(name) {
                        let kv_idx = layer * 2 + if is_key { 0 } else { 1 };
                        if is_encoder {
                            // Encoder KV cache - update only on first step
                            if step == 0 && kv_idx < encoder_kv_cache.len() {
                                encoder_kv_cache[kv_idx] = tensor.clone();
                            }
                        } else {
                            // Decoder KV cache - grows each step
                            if kv_idx < decoder_kv_cache.len() {
                                decoder_kv_cache[kv_idx] = tensor.clone();
                            }
                        }
                    }
                }
            }

            // Skip forced tokens during generation (don't process logits yet)
            if step < num_forced - 1 {
                // Still processing forced tokens
                continue;
            }

            // Apply token suppression and repetition penalty
            let mut logits_vec = logits.as_slice()
                .ok_or_else(|| AdapterError::InvalidInput("Logits not contiguous".to_string()))?
                .to_vec();

            // Suppress tokens
            for &token in &suppress_set {
                if (token as usize) < logits_vec.len() {
                    logits_vec[token as usize] = f32::NEG_INFINITY;
                }
            }

            // Apply repetition penalty
            if repetition_penalty != 1.0 && generated_tokens.len() > 4 {
                let recent: std::collections::HashSet<usize> = generated_tokens
                    .iter()
                    .rev()
                    .take(10)
                    .copied()
                    .collect();

                for token in &recent {
                    if *token < logits_vec.len() {
                        let score = logits_vec[*token];
                        logits_vec[*token] = if score > 0.0 {
                            score / repetition_penalty
                        } else {
                            score * repetition_penalty
                        };
                    }
                }
            }

            // Get next token (argmax)
            let next_token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(end_token_id as usize);

            // Check for end token
            if next_token == end_token_id as usize {
                break;
            }

            // Check for repetition loop (same token 5+ times)
            if generated_tokens.len() >= 5 {
                let last_five: Vec<usize> = generated_tokens.iter().rev().take(5).copied().collect();
                if last_five.iter().all(|&id| id == next_token) {
                    break;
                }
            }

            generated_tokens.push(next_token);
        }

        Ok(generated_tokens)
    }

    /// Apply argmax to logits to get token ID
    fn argmax_token(logits: &ArrayD<f32>) -> ExecutorResult<usize> {
        let shape = logits.shape();
        let data = logits
            .as_slice()
            .ok_or_else(|| AdapterError::InvalidInput("Logits tensor is not contiguous".to_string()))?;

        // Handle 3D logits [batch, seq_len, vocab_size]
        if shape.len() == 3 {
            let vocab_size = shape[2];
            let start_idx = 0; // First batch, first position
            let end_idx = start_idx + vocab_size;

            let slice = &data[start_idx..end_idx];
            let max_idx = slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            Ok(max_idx)
        } else {
            Err(AdapterError::InvalidInput(format!(
                "Unexpected logits shape: {:?}",
                shape
            )))
        }
    }

    /// Apply softmax to a tensor along a dimension
    fn apply_softmax(tensor: &mut ArrayD<f32>, dim: Option<usize>) -> ExecutorResult<()> {
        let shape = tensor.shape().to_vec(); // Clone shape to avoid borrow conflicts

        // Default to last dimension if not specified
        let dim = dim.unwrap_or(shape.len() - 1);

        if dim >= shape.len() {
            return Err(AdapterError::InvalidInput(format!(
                "Softmax dimension {} out of bounds for tensor with {} dimensions",
                dim, shape.len()
            )));
        }

        // For simplicity, only handle the common case of 2D tensors (batch, classes)
        // or 1D tensors (classes)
        if let Some(slice) = tensor.as_slice_mut() {
            if shape.len() == 1 {
                // 1D tensor: apply softmax directly
                Self::softmax_1d(slice);
            } else if shape.len() == 2 && dim == 1 {
                // 2D tensor: apply softmax along last dimension
                let batch_size = shape[0];
                let class_size = shape[1];

                for batch in 0..batch_size {
                    let start = batch * class_size;
                    let end = start + class_size;
                    Self::softmax_1d(&mut slice[start..end]);
                }
            } else {
                return Err(AdapterError::InvalidInput(format!(
                    "Softmax only supports 1D or 2D tensors, got shape {:?}",
                    shape
                )));
            }
        } else {
            return Err(AdapterError::InvalidInput(
                "Tensor is not contiguous, cannot apply softmax".to_string()
            ));
        }

        Ok(())
    }

    /// Apply softmax to a 1D slice
    fn softmax_1d(slice: &mut [f32]) {
        // Find max for numerical stability
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for val in slice.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }

        // Normalize
        for val in slice.iter_mut() {
            *val /= sum;
        }
    }

    /// Get top-K predictions from a tensor
    /// Returns Vec of (class_index, score) tuples
    fn top_k_predictions(
        tensor: &ArrayD<f32>,
        k: usize,
        dim: Option<usize>,
    ) -> ExecutorResult<Vec<(usize, f32)>> {
        let shape = tensor.shape();

        // Default to last dimension
        let _dim = dim.unwrap_or(shape.len() - 1);

        // Get values as slice
        let values = tensor.as_slice().ok_or_else(|| {
            AdapterError::InvalidInput("Tensor is not contiguous for TopK".to_string())
        })?;

        // For simplicity, handle the common case: 1D (classes) or 2D (batch=1, classes)
        let class_scores: &[f32] = if shape.len() == 1 {
            values
        } else if shape.len() == 2 && shape[0] == 1 {
            // Batch size 1, get the first batch
            &values[0..shape[1]]
        } else {
            return Err(AdapterError::InvalidInput(format!(
                "TopK only supports 1D or 2D (batch=1) tensors, got shape {:?}",
                shape
            )));
        };

        // Create (index, score) pairs and sort by score descending
        let mut indexed_scores: Vec<(usize, f32)> = class_scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K
        let top_k: Vec<(usize, f32)> = indexed_scores.into_iter().take(k).collect();

        Ok(top_k)
    }

    /// Run postprocessing steps from metadata
    fn run_postprocessing(
        &mut self,
        metadata: &ModelMetadata,
        outputs: RawOutputs,
    ) -> ExecutorResult<Envelope> {
        if metadata.postprocessing.is_empty() {
            return outputs.to_envelope();
        }

        let _postprocess_span = trace::SpanGuard::new("postprocessing");
        trace::add_metadata("steps", &metadata.postprocessing.len().to_string());

        let mut data = outputs;

        for step in &metadata.postprocessing {
            // Create span for each postprocessing step
            let step_name = format!("postprocessing:{}", step.step_name());
            let _step_span = trace::SpanGuard::new(&step_name);

            data = self.apply_postprocessing_step(step, data, metadata)?;
        }

        // Convert final data to Envelope
        data.to_envelope()
    }

    /// Apply a single postprocessing step
    fn apply_postprocessing_step(
        &mut self,
        step: &PostprocessingStep,
        data: RawOutputs,
        metadata: &ModelMetadata,
    ) -> ExecutorResult<RawOutputs> {
        match step {
            PostprocessingStep::BPEDecode { vocab_file } => {
                let token_ids = match data {
                    RawOutputs::TokenIds(ids) => ids,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "BPEDecode requires token IDs".to_string(),
                        ))
                    }
                };

                // Load vocab file and decode tokens
                let vocab_path = self.resolve_file_path(vocab_file);
                let text = self.decode_bpe_tokens(&token_ids, &vocab_path)?;

                Ok(RawOutputs::Text(text))
            }

            PostprocessingStep::Argmax { dim: _ } => {
                // Apply argmax to tensor outputs
                let tensor_map = match data {
                    RawOutputs::TensorMap(map) => map,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "Argmax requires tensor map".to_string(),
                        ))
                    }
                };

                // Get the first output tensor
                let tensor = tensor_map
                    .values()
                    .next()
                    .ok_or_else(|| AdapterError::InvalidInput("No outputs to apply argmax".to_string()))?;

                let class_id = Self::argmax_token(tensor)?;

                Ok(RawOutputs::ClassId(class_id))
            }

            PostprocessingStep::Softmax { dim } => {
                // Apply softmax to tensor outputs
                let mut tensor_map = match data {
                    RawOutputs::TensorMap(map) => map,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "Softmax requires tensor map".to_string(),
                        ))
                    }
                };

                // Apply softmax to each tensor in the map
                for (_name, tensor) in tensor_map.iter_mut() {
                    Self::apply_softmax(tensor, *dim)?;
                }

                Ok(RawOutputs::TensorMap(tensor_map))
            }

            PostprocessingStep::TopK { k, dim } => {
                // Get top-K predictions with scores
                let tensor_map = match data {
                    RawOutputs::TensorMap(map) => map,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "TopK requires tensor map".to_string(),
                        ))
                    }
                };

                // Get the first output tensor
                let tensor = tensor_map
                    .values()
                    .next()
                    .ok_or_else(|| AdapterError::InvalidInput("No outputs for TopK".to_string()))?;

                // Apply top-k
                let top_k_results = Self::top_k_predictions(tensor, *k, *dim)?;

                // Return as tensor map with flattened [index1, score1, index2, score2, ...]
                let mut flattened = Vec::with_capacity(k * 2);
                for (idx, score) in top_k_results {
                    flattened.push(idx as f32);
                    flattened.push(score);
                }

                // Create a 1D tensor from the flattened results
                let topk_tensor = ArrayD::from_shape_vec(
                    IxDyn(&[k * 2]),
                    flattened
                ).map_err(|e| AdapterError::InvalidInput(format!("Failed to create TopK tensor: {:?}", e)))?;

                let mut result_map = HashMap::new();
                result_map.insert("topk".to_string(), topk_tensor);

                Ok(RawOutputs::TensorMap(result_map))
            }

            PostprocessingStep::Threshold {
                threshold,
                return_indices,
            } => {
                // Apply threshold to convert probabilities to binary predictions
                let tensor_map = match data {
                    RawOutputs::TensorMap(map) => map,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "Threshold requires tensor map".to_string(),
                        ))
                    }
                };

                // Get the first output tensor
                let tensor = tensor_map
                    .values()
                    .next()
                    .ok_or_else(|| {
                        AdapterError::InvalidInput("No outputs for Threshold".to_string())
                    })?;

                let values = tensor.as_slice().ok_or_else(|| {
                    AdapterError::InvalidInput("Tensor is not contiguous for Threshold".to_string())
                })?;

                if *return_indices {
                    // Return indices where value > threshold
                    let indices: Vec<f32> = values
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &val)| {
                            if val > *threshold {
                                Some(idx as f32)
                            } else {
                                None
                            }
                        })
                        .collect();

                    let result_tensor = ArrayD::from_shape_vec(IxDyn(&[indices.len()]), indices)
                        .map_err(|e| {
                            AdapterError::InvalidInput(format!(
                                "Failed to create threshold tensor: {:?}",
                                e
                            ))
                        })?;

                    let mut result_map = HashMap::new();
                    result_map.insert("threshold_indices".to_string(), result_tensor);
                    Ok(RawOutputs::TensorMap(result_map))
                } else {
                    // Return binary mask (0 or 1)
                    let binary: Vec<f32> = values
                        .iter()
                        .map(|&val| if val > *threshold { 1.0 } else { 0.0 })
                        .collect();

                    let result_tensor = ArrayD::from_shape_vec(
                        IxDyn(tensor.shape()),
                        binary
                    ).map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create threshold mask: {:?}",
                            e
                        ))
                    })?;

                    let mut result_map = HashMap::new();
                    result_map.insert("threshold_mask".to_string(), result_tensor);
                    Ok(RawOutputs::TensorMap(result_map))
                }
            }

            PostprocessingStep::MeanPool { dim } => {
                // Apply mean pooling over token embeddings
                let tensor_map = match data {
                    RawOutputs::TensorMap(map) => map,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "MeanPool requires tensor map".to_string(),
                        ))
                    }
                };

                // Get the first output tensor (usually "last_hidden_state" or similar)
                let tensor = tensor_map
                    .values()
                    .next()
                    .ok_or_else(|| {
                        AdapterError::InvalidInput("No outputs for MeanPool".to_string())
                    })?;

                let shape = tensor.shape();

                // Expected shape: [batch, sequence_length, hidden_size]
                if shape.len() != 3 {
                    return Err(AdapterError::InvalidInput(format!(
                        "MeanPool expects 3D tensor [batch, seq_len, hidden_size], got {:?}",
                        shape
                    )));
                }

                let batch_size = shape[0];
                let seq_len = shape[1];
                let hidden_size = shape[2];

                // Pool over the sequence dimension (dim=1 by default)
                if *dim != 1 {
                    return Err(AdapterError::InvalidInput(format!(
                        "MeanPool only supports pooling over dim=1 (sequence), got dim={}",
                        dim
                    )));
                }

                // Create output tensor [batch, hidden_size]
                let mut pooled = ArrayD::<f32>::zeros(IxDyn(&[batch_size, hidden_size]));

                // Compute mean over sequence length for each batch and hidden dimension
                for b in 0..batch_size {
                    for h in 0..hidden_size {
                        let mut sum = 0.0;
                        for s in 0..seq_len {
                            sum += tensor[IxDyn(&[b, s, h])];
                        }
                        pooled[IxDyn(&[b, h])] = sum / (seq_len as f32);
                    }
                }

                // Return pooled embedding
                let mut result_map = HashMap::new();
                result_map.insert("sentence_embedding".to_string(), pooled);

                Ok(RawOutputs::TensorMap(result_map))
            }

            PostprocessingStep::TemperatureSample {
                temperature: _,
                top_k: _,
                top_p: _,
            } => {
                // TODO: Implement temperature sampling
                Ok(data)
            }

            PostprocessingStep::CTCDecode {
                vocab_file,
                blank_index,
            } => {
                // CTC decoding for Wav2Vec2-style models
                let tensor_map = match data {
                    RawOutputs::TensorMap(map) => map,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "CTCDecode requires tensor map".to_string(),
                        ))
                    }
                };

                // Get logits tensor (usually "logits" output)
                let logits = tensor_map
                    .values()
                    .next()
                    .ok_or_else(|| {
                        AdapterError::InvalidInput("No outputs for CTCDecode".to_string())
                    })?;

                let shape = logits.shape();
                // Expected shape: [batch, time_steps, vocab_size]
                if shape.len() != 3 {
                    return Err(AdapterError::InvalidInput(format!(
                        "CTCDecode expects 3D tensor [batch, time, vocab], got {:?}",
                        shape
                    )));
                }

                let _batch_size = shape[0];
                let time_steps = shape[1];
                let _vocab_size = shape[2];

                // Simple greedy CTC decoding:
                // 1. Take argmax over vocab dimension for each timestep
                // 2. Remove consecutive duplicates
                // 3. Remove blank tokens

                let mut token_ids = Vec::new();
                let mut prev_id: Option<usize> = None;

                for t in 0..time_steps {
                    // Get argmax over vocab dimension
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;

                    for v in 0.._vocab_size {
                        let val = logits[IxDyn(&[0, t, v])]; // batch=0 for simplicity
                        if val > max_val {
                            max_val = val;
                            max_idx = v;
                        }
                    }

                    // Skip blank tokens and consecutive duplicates
                    if max_idx != *blank_index && Some(max_idx) != prev_id {
                        token_ids.push(max_idx);
                    }

                    prev_id = Some(max_idx);
                }

                // Load vocabulary and decode
                let vocab_path = self.resolve_file_path(vocab_file);
                let text = self.decode_ctc_tokens(&token_ids, &vocab_path)?;

                Ok(RawOutputs::Text(text))
            }

            PostprocessingStep::Denormalize { mean: _, std: _ } => {
                // TODO: Implement denormalization
                Ok(data)
            }

            PostprocessingStep::TTSAudioEncode {
                sample_rate,
                apply_postprocessing,
            } => {
                // Convert TTS waveform tensor to audio bytes
                let tensor_map = match data {
                    RawOutputs::TensorMap(map) => map,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "TTSAudioEncode requires tensor map".to_string(),
                        ))
                    }
                };

                // Get the waveform tensor (usually named "waveform" or first output)
                let waveform = tensor_map
                    .get("waveform")
                    .or_else(|| tensor_map.values().next())
                    .ok_or_else(|| {
                        AdapterError::InvalidInput("No waveform output for TTS".to_string())
                    })?;

                // Convert tensor to f32 samples
                let samples: Vec<f32> = waveform
                    .as_slice()
                    .ok_or_else(|| {
                        AdapterError::InvalidInput("Waveform tensor not contiguous".to_string())
                    })?
                    .to_vec();

                // Apply postprocessing if enabled
                let processed_samples = if *apply_postprocessing {
                    use crate::phonemizer::postprocess_tts_audio;
                    postprocess_tts_audio(&samples, *sample_rate)
                } else {
                    samples
                };

                // Convert f32 samples to 16-bit PCM bytes
                let audio_bytes = self.samples_to_pcm16(&processed_samples);

                Ok(RawOutputs::AudioBytes(audio_bytes))
            }

            PostprocessingStep::WhisperDecode { tokenizer_file } => {
                // Decode Whisper token IDs using HuggingFace tokenizer
                let token_ids = match data {
                    RawOutputs::TokenIds(ids) => ids,
                    _ => {
                        return Err(AdapterError::InvalidInput(
                            "WhisperDecode requires token IDs".to_string(),
                        ))
                    }
                };

                let tokenizer_path = self.resolve_file_path(tokenizer_file);
                let text = self.decode_whisper_tokens(&token_ids, &tokenizer_path)?;

                Ok(RawOutputs::Text(text))
            }
        }
    }

    /// Convert f32 audio samples to 16-bit PCM bytes
    fn samples_to_pcm16(&self, samples: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);

        for &sample in samples {
            // Clamp to [-1.0, 1.0] and convert to i16
            let clamped = sample.clamp(-1.0, 1.0);
            let pcm16 = (clamped * 32767.0) as i16;
            bytes.extend_from_slice(&pcm16.to_le_bytes());
        }

        bytes
    }

    /// Decode BPE tokens to text
    fn decode_bpe_tokens(&self, token_ids: &[usize], vocab_path: &str) -> ExecutorResult<String> {
        use base64::{engine::general_purpose, Engine as _};

        // Load vocabulary
        let content = std::fs::read_to_string(vocab_path)
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to read vocab file: {}", e)))?;

        let tokens: Vec<String> = content.lines().map(|line| line.trim().to_string()).collect();

        // Decode tokens
        let mut decoded_bytes = Vec::new();

        for &id in token_ids {
            if id < tokens.len() {
                let token_line = &tokens[id];

                // Skip special tokens
                if token_line.starts_with("<|") && token_line.ends_with("|>") {
                    continue;
                }

                // Token format: "BASE64 ID"
                let base64_part = if let Some(space_idx) = token_line.find(' ') {
                    &token_line[..space_idx]
                } else {
                    token_line
                };

                // Decode base64 to bytes
                if let Ok(bytes) = general_purpose::STANDARD.decode(base64_part) {
                    decoded_bytes.extend_from_slice(&bytes);
                }
            }
        }

        Ok(String::from_utf8_lossy(&decoded_bytes).to_string())
    }

    /// Decode CTC tokens to text (for Wav2Vec2-style models)
    fn decode_ctc_tokens(&self, token_ids: &[usize], vocab_path: &str) -> ExecutorResult<String> {
        // Load vocabulary
        let content = std::fs::read_to_string(vocab_path)
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to read vocab file: {}", e)))?;

        // Try to parse as JSON first (Wav2Vec2 format: {"char": id, ...})
        let vocab: Vec<String> = if content.trim().starts_with('{') {
            // Parse JSON vocab: {"'": 27, "A": 7, "B": 24, ...}
            let json_vocab = serde_json::from_str::<std::collections::HashMap<String, usize>>(&content)
                .map_err(|e| AdapterError::InvalidInput(format!("Failed to parse vocab JSON: {}", e)))?;

            // Create reverse mapping: id -> char
            let max_id = json_vocab.values().max().copied().unwrap_or(0);
            let mut id_to_char = vec![String::new(); max_id + 1];

            for (char_str, id) in json_vocab {
                if id < id_to_char.len() {
                    id_to_char[id] = char_str;
                }
            }

            id_to_char
        } else {
            // Plain text format: one token per line
            content.lines().map(|line| line.trim().to_string()).collect()
        };

        // Build text from token IDs
        let mut text = String::new();
        for &id in token_ids {
            if id < vocab.len() {
                let token = &vocab[id];
                // Handle special Wav2Vec2 tokens
                if token == "|" {
                    text.push(' ');  // Word boundary
                } else if !token.starts_with('<') && !token.ends_with('>') {
                    // Regular character token
                    text.push_str(token);
                }
            }
        }

        // Clean up extra spaces
        Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    /// Decode Whisper tokens using HuggingFace tokenizer.json
    fn decode_whisper_tokens(&self, token_ids: &[usize], tokenizer_path: &str) -> ExecutorResult<String> {
        use tokenizers::Tokenizer;

        // Load the HuggingFace tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to load tokenizer: {}", e)))?;

        // Convert token IDs to u32 (tokenizers crate uses u32)
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

        // Filter out special tokens:
        // - Whisper special tokens are in range 50257-50364
        // - Keep only normal text tokens
        let filtered_ids: Vec<u32> = ids
            .into_iter()
            .filter(|&id| id < 50257) // Filter out special tokens
            .collect();

        // Decode using the tokenizer
        let text = tokenizer
            .decode(&filtered_ids, true) // skip_special_tokens=true
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to decode tokens: {}", e)))?;

        // Clean up whitespace
        Ok(text.trim().to_string())
    }

    /// Get or load an ONNX session
    fn get_or_load_session(&mut self, model_file: &str) -> ExecutorResult<&ONNXSession> {
        if !self.session_cache.contains_key(model_file) {
            let model_path = self.resolve_file_path(model_file);

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let use_metal = true;
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            let use_metal = false;

            let session = ONNXSession::new(&model_path, false, use_metal)?;
            self.session_cache.insert(model_file.to_string(), session);
        }

        Ok(self.session_cache.get(model_file).unwrap())
    }

    /// Resolve a file path relative to base_path
    fn resolve_file_path(&self, file: &str) -> String {
        if self.base_path.is_empty() {
            file.to_string()
        } else {
            Path::new(&self.base_path)
                .join(file)
                .to_string_lossy()
                .to_string()
        }
    }

    /// Helper: Get debug information about a session's inputs/outputs
    ///
    /// This is useful for understanding model I/O signatures when metadata doesn't match
    #[allow(dead_code)]
    fn get_session_io_info(&self, session: &ONNXSession) -> String {
        let input_names = session.input_names();
        let output_names = session.output_names();

        format!(
            "Inputs: {:?}, Outputs: {:?}",
            input_names,
            output_names
        )
    }

    /// Helper: Validate that encoder outputs contain cross-attention keys/values
    ///
    /// Returns the two outputs in the correct order (k, v)
    fn extract_encoder_cross_attention<'a>(
        &self,
        encoder_outputs: &'a HashMap<String, ArrayD<f32>>,
    ) -> ExecutorResult<(&'a ArrayD<f32>, &'a ArrayD<f32>)> {
        // Try named lookup first (preferred if metadata is accurate)
        if let (Some(k), Some(v)) = (
            encoder_outputs.get("n_layer_cross_k"),
            encoder_outputs.get("n_layer_cross_v"),
        ) {
            return Ok((k, v));
        }

        // Fallback: use first two outputs by position
        if encoder_outputs.len() < 2 {
            return Err(AdapterError::InvalidInput(format!(
                "Encoder must produce at least 2 outputs (cross_k, cross_v), found {}",
                encoder_outputs.len()
            )));
        }

        let mut values = encoder_outputs.values();
        let k = values.next().unwrap();
        let v = values.next().unwrap();

        Ok((k, v))
    }
}

impl Default for TemplateExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse KV cache input name from HuggingFace format
/// Format: past_key_values.{layer}.{decoder|encoder}.{key|value}
/// Returns: (layer_index, is_encoder, is_key)
fn parse_kv_cache_name(name: &str) -> Option<(usize, bool, bool)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 4 || parts[0] != "past_key_values" {
        return None;
    }

    let layer = parts[1].parse::<usize>().ok()?;
    let is_encoder = parts[2] == "encoder";
    let is_key = parts[3] == "key";

    Some((layer, is_encoder, is_key))
}

/// Parse present output name from HuggingFace format
/// Format: present.{layer}.decoder.{key|value}
/// Returns: (layer_index, is_key)
#[allow(dead_code)]
fn parse_present_name(name: &str) -> Option<(usize, bool)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 4 || parts[0] != "present" {
        return None;
    }

    let layer = parts[1].parse::<usize>().ok()?;
    // present outputs are always decoder (encoder KV is static)
    let is_key = parts[3] == "key";

    Some((layer, is_key))
}

/// Parse present output name from HuggingFace format (full version)
/// Format: present.{layer}.{decoder|encoder}.{key|value}
/// Returns: (layer_index, is_encoder, is_key)
fn parse_present_name_full(name: &str) -> Option<(usize, bool, bool)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 4 || parts[0] != "present" {
        return None;
    }

    let layer = parts[1].parse::<usize>().ok()?;
    let is_encoder = parts[2] == "encoder";
    let is_key = parts[3] == "key";

    Some((layer, is_encoder, is_key))
}

/// Preprocessed data intermediate representation
#[derive(Debug, Clone)]
enum PreprocessedData {
    AudioBytes(Vec<u8>),
    AudioSamples(Vec<f32>),  // Decoded PCM audio samples
    Text(String),
    Tensor(ArrayD<f32>),
    TokenIds {
        ids: Vec<usize>,
        attention_mask: Vec<usize>,
        token_type_ids: Vec<usize>,
        vocab_file: String,
        original_text: String,
    },
    PhonemeIds {
        ids: Vec<i64>,
        phonemes: String,
        original_text: String,
    },
}

impl PreprocessedData {
    /// Check if this is phoneme data (for TTS)
    fn is_phoneme_ids(&self) -> bool {
        matches!(self, PreprocessedData::PhonemeIds { .. })
    }

    /// Get phoneme IDs if this is phoneme data
    fn as_phoneme_ids(&self) -> Option<&Vec<i64>> {
        match self {
            PreprocessedData::PhonemeIds { ids, .. } => Some(ids),
            _ => None,
        }
    }
}

impl PreprocessedData {
    fn from_envelope(envelope: &Envelope) -> ExecutorResult<Self> {
        match &envelope.kind {
            EnvelopeKind::Audio(bytes) => Ok(PreprocessedData::AudioBytes(bytes.clone())),
            EnvelopeKind::Text(text) => Ok(PreprocessedData::Text(text.clone())),
            EnvelopeKind::Embedding(floats) => {
                // Convert embedding Vec<f32> to tensor
                let tensor = ArrayD::from_shape_vec(IxDyn(&[floats.len()]), floats.clone())
                    .map_err(|e| AdapterError::InvalidInput(format!("Failed to create tensor: {:?}", e)))?;
                Ok(PreprocessedData::Tensor(tensor))
            }
        }
    }

    fn to_tensor(&self) -> ExecutorResult<ArrayD<f32>> {
        match self {
            PreprocessedData::Tensor(t) => Ok(t.clone()),
            PreprocessedData::AudioSamples(samples) => {
                // Convert audio samples to tensor [batch, samples]
                let batch_size = 1;
                let num_samples = samples.len();
                let tensor = ArrayD::from_shape_vec(
                    IxDyn(&[batch_size, num_samples]),
                    samples.clone()
                ).map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create audio tensor: {:?}", e))
                })?;
                Ok(tensor)
            }
            _ => Err(AdapterError::InvalidInput(
                "Cannot convert to tensor".to_string(),
            )),
        }
    }
}

/// Raw outputs from model execution
#[derive(Debug, Clone)]
enum RawOutputs {
    TensorMap(HashMap<String, ArrayD<f32>>),
    TokenIds(Vec<usize>),
    Text(String),
    ClassId(usize),
    AudioBytes(Vec<u8>),
}

impl RawOutputs {
    fn to_envelope(&self) -> ExecutorResult<Envelope> {
        match self {
            RawOutputs::Text(text) => Ok(Envelope::new(EnvelopeKind::Text(text.clone()))),
            RawOutputs::ClassId(id) => Ok(Envelope::new(EnvelopeKind::Text(format!("Class: {}", id)))),
            RawOutputs::TensorMap(map) => {
                // Convert first tensor to embedding Vec<f32>
                let tensor = map
                    .values()
                    .next()
                    .ok_or_else(|| AdapterError::InvalidInput("No outputs".to_string()))?;

                let data = tensor
                    .as_slice()
                    .ok_or_else(|| AdapterError::InvalidInput("Tensor not contiguous".to_string()))?;

                Ok(Envelope::new(EnvelopeKind::Embedding(data.to_vec())))
            }
            RawOutputs::TokenIds(ids) => {
                // Convert token IDs to text representation
                Ok(Envelope::new(EnvelopeKind::Text(format!("{:?}", ids))))
            }
            RawOutputs::AudioBytes(bytes) => {
                // Return audio bytes directly
                Ok(Envelope::new(EnvelopeKind::Audio(bytes.clone())))
            }
        }
    }
}

// ============================================================================
// Preprocessing Step Functions
// ============================================================================

/// Convert audio samples to mel spectrogram.
///
/// # Arguments
/// - `data`: Input data (AudioSamples or AudioBytes)
/// - `preset`: Optional preset name (e.g., "whisper", "whisper-large")
/// - `n_mels`: Number of mel frequency bins
/// - `sample_rate`: Target sample rate in Hz
/// - `fft_size`: FFT window size
/// - `hop_length`: Hop length between frames
/// - `mel_scale`: Mel frequency scale (Slaney or HTK)
/// - `max_frames`: Maximum number of output frames
///
/// # Presets
/// If a preset is specified, it overrides the individual parameters.
/// Available presets: "whisper", "whisper-large"
fn mel_spectrogram_step(
    data: PreprocessedData,
    preset: Option<&str>,
    n_mels: usize,
    sample_rate: u32,
    fft_size: usize,
    hop_length: usize,
    mel_scale: MelScaleType,
    max_frames: Option<usize>,
) -> ExecutorResult<PreprocessedData> {
    // Build MelConfig from preset or individual parameters
    let config = if let Some(preset_name) = preset {
        MelConfig::from_preset(preset_name).unwrap_or_else(|| {
            eprintln!("[WARN] Unknown mel spectrogram preset '{}', using parameters", preset_name);
            build_mel_config(n_mels, sample_rate, fft_size, hop_length, mel_scale, max_frames)
        })
    } else {
        build_mel_config(n_mels, sample_rate, fft_size, hop_length, mel_scale, max_frames)
    };

    match data {
        PreprocessedData::AudioSamples(samples) => {
            let mel = compute_mel_spectrogram(&samples, &config)
                .map_err(|e| AdapterError::InvalidInput(e))?;
            Ok(PreprocessedData::Tensor(mel))
        }
        PreprocessedData::AudioBytes(bytes) => {
            // For audio bytes, we need to parse WAV first
            // Use the preprocessing module's audio_bytes_to_whisper_mel for now
            // TODO: Add audio bytes parsing to the unified API
            let mel = audio_bytes_to_whisper_mel(&bytes)?;
            Ok(PreprocessedData::Tensor(mel))
        }
        _ => {
            Err(AdapterError::InvalidInput(
                "MelSpectrogram requires audio samples or bytes input".to_string(),
            ))
        }
    }
}

/// Build MelConfig from individual parameters.
fn build_mel_config(
    n_mels: usize,
    sample_rate: u32,
    fft_size: usize,
    hop_length: usize,
    mel_scale: MelScaleType,
    max_frames: Option<usize>,
) -> MelConfig {
    MelConfig {
        n_mels,
        n_fft: fft_size,
        hop_length,
        sample_rate,
        mel_scale: match mel_scale {
            MelScaleType::Slaney => MelScale::Slaney,
            MelScaleType::Htk => MelScale::Htk,
        },
        f_min: 0.0,
        f_max: (sample_rate / 2) as f64, // Nyquist
        padding: PaddingMode::Reflect,
        max_frames,
        normalize: true,
    }
}

/// Tokenize text input for NLP models.
///
/// # Arguments
/// - `data`: Input data (Text)
/// - `tokenizer_path`: Path to tokenizer.json file
/// - `tokenizer_type`: Type of tokenizer (WordPiece, BPE, SentencePiece)
/// - `max_length`: Optional maximum sequence length
fn tokenize_step(
    data: PreprocessedData,
    tokenizer_path: &str,
    tokenizer_type: &crate::execution_template::TokenizerType,
    max_length: Option<usize>,
) -> ExecutorResult<PreprocessedData> {
    use crate::execution_template::TokenizerType;
    use tokenizers::Tokenizer;

    let text = match data {
        PreprocessedData::Text(text) => text,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Tokenize requires text input".to_string(),
            ))
        }
    };

    let tokenizer = match tokenizer_type {
        TokenizerType::WordPiece | TokenizerType::BPE => {
            Tokenizer::from_file(tokenizer_path).map_err(|e| {
                AdapterError::InvalidInput(format!(
                    "Failed to load tokenizer from {}: {}",
                    tokenizer_path, e
                ))
            })?
        }
        TokenizerType::SentencePiece => {
            return Err(AdapterError::InvalidInput(
                "SentencePiece tokenizer not yet implemented".to_string(),
            ));
        }
    };

    let encoding = tokenizer.encode(text.clone(), false).map_err(|e| {
        AdapterError::InvalidInput(format!("Tokenization failed: {}", e))
    })?;

    let mut ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
    let mut attention_mask: Vec<usize> = encoding
        .get_attention_mask()
        .iter()
        .map(|&mask| mask as usize)
        .collect();
    let mut token_type_ids: Vec<usize> = encoding
        .get_type_ids()
        .iter()
        .map(|&type_id| type_id as usize)
        .collect();

    if let Some(max_len) = max_length {
        if ids.len() > max_len {
            ids.truncate(max_len);
            attention_mask.truncate(max_len);
            token_type_ids.truncate(max_len);
        }
    }

    Ok(PreprocessedData::TokenIds {
        ids,
        attention_mask,
        token_type_ids,
        vocab_file: tokenizer_path.to_string(),
        original_text: text,
    })
}

/// Normalize tensor values using mean and standard deviation.
///
/// # Arguments
/// - `data`: Input data (Tensor)
/// - `mean`: Per-channel mean values
/// - `std`: Per-channel standard deviation values
fn normalize_step(
    data: PreprocessedData,
    mean: &[f32],
    std: &[f32],
) -> ExecutorResult<PreprocessedData> {
    let mut tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Normalize requires tensor input".to_string(),
            ))
        }
    };

    if let Some(tensor_slice) = tensor.as_slice_mut() {
        for (i, val) in tensor_slice.iter_mut().enumerate() {
            let channel = i % mean.len();
            *val = (*val - mean[channel]) / std[channel];
        }
    }

    Ok(PreprocessedData::Tensor(tensor))
}

/// Reshape tensor to target dimensions.
///
/// # Arguments
/// - `data`: Input data (Tensor)
/// - `shape`: Target shape dimensions
fn reshape_step(data: PreprocessedData, shape: &[usize]) -> ExecutorResult<PreprocessedData> {
    let tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Reshape requires tensor input".to_string(),
            ))
        }
    };

    let total_elements: usize = shape.iter().product();
    let tensor_elements = tensor.len();

    if total_elements != tensor_elements {
        return Err(AdapterError::InvalidInput(format!(
            "Cannot reshape tensor: shape {:?} requires {} elements, but tensor has {}",
            shape, total_elements, tensor_elements
        )));
    }

    #[allow(deprecated)]
    let reshaped = tensor
        .into_shape(IxDyn(shape))
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to reshape tensor: {:?}", e)))?;

    Ok(PreprocessedData::Tensor(reshaped))
}

/// Center crop image tensor to target dimensions.
///
/// # Arguments
/// - `data`: Input data (Tensor with shape [batch, channels, h, w] or [channels, h, w])
/// - `width`: Target crop width
/// - `height`: Target crop height
fn center_crop_step(
    data: PreprocessedData,
    width: usize,
    height: usize,
) -> ExecutorResult<PreprocessedData> {
    let tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "CenterCrop requires tensor input".to_string(),
            ))
        }
    };

    let shape = tensor.shape();
    if shape.len() < 3 {
        return Err(AdapterError::InvalidInput(format!(
            "CenterCrop requires at least 3D tensor (got {:?})",
            shape
        )));
    }

    let (batch_size, channels, src_h, src_w) = if shape.len() == 4 {
        (shape[0], shape[1], shape[2], shape[3])
    } else {
        (1, shape[0], shape[1], shape[2])
    };

    if height > src_h || width > src_w {
        return Err(AdapterError::InvalidInput(format!(
            "Cannot crop {}x{} from {}x{} image",
            width, height, src_w, src_h
        )));
    }

    let offset_h = (src_h - height) / 2;
    let offset_w = (src_w - width) / 2;

    let out_shape = if shape.len() == 4 {
        vec![batch_size, channels, height, width]
    } else {
        vec![channels, height, width]
    };

    let mut cropped = ArrayD::<f32>::zeros(IxDyn(&out_shape));

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_coords = if shape.len() == 4 {
                        vec![b, c, offset_h + h, offset_w + w]
                    } else {
                        vec![c, offset_h + h, offset_w + w]
                    };
                    let dst_coords = if shape.len() == 4 {
                        vec![b, c, h, w]
                    } else {
                        vec![c, h, w]
                    };

                    cropped[IxDyn(&dst_coords)] = tensor[IxDyn(&src_coords)];
                }
            }
        }
    }

    Ok(PreprocessedData::Tensor(cropped))
}

/// Resize image tensor using interpolation.
///
/// # Arguments
/// - `data`: Input data (Tensor with shape [batch, channels, h, w] or [channels, h, w])
/// - `width`: Target width
/// - `height`: Target height
/// - `interpolation`: Interpolation method (Nearest, Bilinear, Bicubic)
fn resize_step(
    data: PreprocessedData,
    width: usize,
    height: usize,
    interpolation: &crate::execution_template::InterpolationMethod,
) -> ExecutorResult<PreprocessedData> {
    use crate::execution_template::InterpolationMethod;

    let tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Resize requires tensor input".to_string(),
            ))
        }
    };

    let shape = tensor.shape();
    if shape.len() < 3 {
        return Err(AdapterError::InvalidInput(format!(
            "Resize requires at least 3D tensor (got {:?})",
            shape
        )));
    }

    let (batch_size, channels, src_h, src_w) = if shape.len() == 4 {
        (shape[0], shape[1], shape[2], shape[3])
    } else {
        (1, shape[0], shape[1], shape[2])
    };

    if channels != 3 && channels != 1 {
        return Err(AdapterError::InvalidInput(format!(
            "Resize only supports 1 or 3 channels (got {})",
            channels
        )));
    }

    let filter_type = match interpolation {
        InterpolationMethod::Nearest => image::imageops::FilterType::Nearest,
        InterpolationMethod::Bilinear => image::imageops::FilterType::Triangle,
        InterpolationMethod::Bicubic => image::imageops::FilterType::CatmullRom,
    };

    let out_shape = if shape.len() == 4 {
        vec![batch_size, channels, height as usize, width as usize]
    } else {
        vec![channels, height as usize, width as usize]
    };

    let mut resized_tensor = ArrayD::<f32>::zeros(IxDyn(&out_shape));

    for b in 0..batch_size {
        if channels == 3 {
            resized_tensor = resize_rgb_image(
                &tensor,
                resized_tensor,
                shape,
                b,
                src_h,
                src_w,
                width,
                height,
                filter_type,
            )?;
        } else {
            resized_tensor = resize_grayscale_image(
                &tensor,
                resized_tensor,
                shape,
                b,
                src_h,
                src_w,
                width,
                height,
                filter_type,
            )?;
        }
    }

    Ok(PreprocessedData::Tensor(resized_tensor))
}

/// Helper: Resize an RGB image within a tensor batch.
fn resize_rgb_image(
    tensor: &ArrayD<f32>,
    mut resized_tensor: ArrayD<f32>,
    shape: &[usize],
    b: usize,
    src_h: usize,
    src_w: usize,
    width: usize,
    height: usize,
    filter_type: image::imageops::FilterType,
) -> ExecutorResult<ArrayD<f32>> {
    use image::{ImageBuffer, Rgb, RgbImage};

    let mut img: RgbImage = ImageBuffer::new(src_w as u32, src_h as u32);
    for h in 0..src_h {
        for w in 0..src_w {
            let (r_idx, g_idx, b_idx) = if shape.len() == 4 {
                (vec![b, 0, h, w], vec![b, 1, h, w], vec![b, 2, h, w])
            } else {
                (vec![0, h, w], vec![1, h, w], vec![2, h, w])
            };

            let r = (tensor[IxDyn(&r_idx)] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[IxDyn(&g_idx)] * 255.0).clamp(0.0, 255.0) as u8;
            let b_val = (tensor[IxDyn(&b_idx)] * 255.0).clamp(0.0, 255.0) as u8;

            img.put_pixel(w as u32, h as u32, Rgb([r, g, b_val]));
        }
    }

    let resized = image::imageops::resize(&img, width as u32, height as u32, filter_type);

    for h in 0..height {
        for w in 0..width {
            let pixel = resized.get_pixel(w as u32, h as u32);
            let (r_idx, g_idx, b_idx) = if shape.len() == 4 {
                (vec![b, 0, h, w], vec![b, 1, h, w], vec![b, 2, h, w])
            } else {
                (vec![0, h, w], vec![1, h, w], vec![2, h, w])
            };

            resized_tensor[IxDyn(&r_idx)] = pixel[0] as f32 / 255.0;
            resized_tensor[IxDyn(&g_idx)] = pixel[1] as f32 / 255.0;
            resized_tensor[IxDyn(&b_idx)] = pixel[2] as f32 / 255.0;
        }
    }

    Ok(resized_tensor)
}

/// Helper: Resize a grayscale image within a tensor batch.
fn resize_grayscale_image(
    tensor: &ArrayD<f32>,
    mut resized_tensor: ArrayD<f32>,
    shape: &[usize],
    b: usize,
    src_h: usize,
    src_w: usize,
    width: usize,
    height: usize,
    filter_type: image::imageops::FilterType,
) -> ExecutorResult<ArrayD<f32>> {
    use image::{GrayImage, ImageBuffer, Luma};

    let mut img: GrayImage = ImageBuffer::new(src_w as u32, src_h as u32);
    for h in 0..src_h {
        for w in 0..src_w {
            let idx = if shape.len() == 4 {
                vec![b, 0, h, w]
            } else {
                vec![0, h, w]
            };
            let val = (tensor[IxDyn(&idx)] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(w as u32, h as u32, Luma([val]));
        }
    }

    let resized = image::imageops::resize(&img, width as u32, height as u32, filter_type);

    for h in 0..height {
        for w in 0..width {
            let pixel = resized.get_pixel(w as u32, h as u32);
            let idx = if shape.len() == 4 {
                vec![b, 0, h, w]
            } else {
                vec![0, h, w]
            };
            resized_tensor[IxDyn(&idx)] = pixel[0] as f32 / 255.0;
        }
    }

    Ok(resized_tensor)
}

/// Phonemize text input for TTS models.
///
/// Converts English text to IPA phonemes using either CMU Dictionary or espeak-ng,
/// then maps phonemes to token IDs using the provided tokens file.
///
/// # Arguments
/// - `data`: Input data (Text)
/// - `tokens_path`: Path to tokens.txt file (maps IPA symbols to token IDs)
/// - `backend`: Which phonemization backend to use
/// - `dict_path`: Optional path to CMU dictionary file (CMU backend only)
/// - `language`: Language code for espeak-ng (e.g., "en-us", "en-gb")
/// - `add_padding`: Whether to add padding tokens (0) at start and end
/// - `normalize_text`: Whether to normalize text before phonemization
fn phonemize_step(
    data: PreprocessedData,
    tokens_path: &str,
    backend: &crate::execution_template::PhonemizerBackend,
    dict_path: Option<&str>,
    language: Option<&str>,
    add_padding: bool,
    normalize_text: bool,
) -> ExecutorResult<PreprocessedData> {
    use crate::execution_template::PhonemizerBackend;
    use crate::phonemizer::load_tokens_map;

    let text = match data {
        PreprocessedData::Text(text) => text,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Phonemize requires text input".to_string(),
            ))
        }
    };

    // Load tokens mapping
    let tokens_content = std::fs::read_to_string(tokens_path).map_err(|e| {
        AdapterError::InvalidInput(format!("Failed to read tokens file {}: {}", tokens_path, e))
    })?;
    let tokens_map = load_tokens_map(&tokens_content);

    // Optionally normalize text
    let processed_text = if normalize_text {
        normalize_text_for_tts(&text)
    } else {
        text.clone()
    };

    // Convert text to IPA phonemes based on backend
    let phonemes = match backend {
        PhonemizerBackend::CmuDictionary => {
            use crate::phonemizer::Phonemizer;

            let phonemizer = if let Some(path) = dict_path {
                Phonemizer::new(path).map_err(|e| {
                    AdapterError::InvalidInput(format!(
                        "Failed to load CMU dictionary from {}: {}",
                        path, e
                    ))
                })?
            } else {
                Phonemizer::from_default_location().map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to initialize phonemizer: {}", e))
                })?
            };

            phonemizer.phonemize(&processed_text)
        }
        PhonemizerBackend::EspeakNG => {
            phonemize_with_espeak(&processed_text, language.unwrap_or("en-us"), &tokens_map)?
        }
        PhonemizerBackend::MisakiDictionary => {
            // Derive base_path from tokens_path (go up one directory from tokens.txt)
            let tokens_dir = std::path::Path::new(tokens_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            phonemize_with_misaki(&processed_text, tokens_dir.to_str().unwrap_or("."), &tokens_map)?
        }
    };

    // Convert phonemes to token IDs
    let mut ids: Vec<i64> = Vec::new();

    if add_padding {
        ids.push(0); // Start padding token
    }

    for c in phonemes.chars() {
        if let Some(&id) = tokens_map.get(&c) {
            ids.push(id);
        } else if c == ' ' {
            // Space character - check if it has a mapping
            if let Some(&id) = tokens_map.get(&' ') {
                ids.push(id);
            }
        }
        // Skip unknown characters silently
    }

    if add_padding {
        ids.push(0); // End padding token
    }

    // Return as PhonemeIds for use by TTS models
    Ok(PreprocessedData::PhonemeIds {
        ids,
        phonemes,
        original_text: text,
    })
}

/// Phonemize text using espeak-ng backend.
///
/// This function calls espeak-ng as an external command to convert text to IPA phonemes,
/// then filters the output to only include characters in the vocabulary.
///
/// Requires espeak-ng to be installed on the system:
/// - macOS: `brew install espeak-ng`
/// - Linux: `apt-get install espeak-ng`
fn phonemize_with_espeak(
    text: &str,
    language: &str,
    vocab: &std::collections::HashMap<char, i64>,
) -> ExecutorResult<String> {
    use std::process::Command;

    // Call espeak-ng with IPA output
    // --ipa outputs IPA phonemes, -q is quiet (no audio), -v sets voice/language
    let output = Command::new("espeak-ng")
        .args(["--ipa", "-q", "-v", language])
        .arg(text)
        .output()
        .map_err(|e| {
            AdapterError::InvalidInput(format!(
                "Failed to run espeak-ng. Is it installed? Error: {}. \
                Install with: brew install espeak-ng (macOS) or apt-get install espeak-ng (Linux)",
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AdapterError::InvalidInput(format!(
            "espeak-ng failed: {}",
            stderr
        )));
    }

    let phonemes = String::from_utf8_lossy(&output.stdout);

    // Filter to only characters in vocabulary
    let filtered: String = phonemes.chars().filter(|c| vocab.contains_key(c)).collect();

    Ok(filtered.trim().to_string())
}

/// Phonemize text using Misaki dictionary-based backend.
///
/// This function uses bundled JSON dictionaries (us_gold.json, us_silver.json) to convert
/// text to IPA phonemes without any system dependencies.
///
/// For out-of-vocabulary words, uses a simple letter-to-phoneme fallback.
fn phonemize_with_misaki(
    text: &str,
    base_path: &str,
    vocab: &std::collections::HashMap<char, i64>,
) -> ExecutorResult<String> {
    use std::collections::HashMap;

    // Load dictionaries (cached via lazy_static in production)
    let misaki_dir = std::path::Path::new(base_path).join("misaki");
    let gold_path = misaki_dir.join("us_gold.json");
    let silver_path = misaki_dir.join("us_silver.json");

    // Parse dictionaries
    let gold_dict: HashMap<String, serde_json::Value> = if gold_path.exists() {
        let content = std::fs::read_to_string(&gold_path).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to read misaki gold dictionary: {}", e))
        })?;
        serde_json::from_str(&content).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to parse misaki gold dictionary: {}", e))
        })?
    } else {
        HashMap::new()
    };

    let silver_dict: HashMap<String, serde_json::Value> = if silver_path.exists() {
        let content = std::fs::read_to_string(&silver_path).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to read misaki silver dictionary: {}", e))
        })?;
        serde_json::from_str(&content).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to parse misaki silver dictionary: {}", e))
        })?
    } else {
        HashMap::new()
    };

    if gold_dict.is_empty() && silver_dict.is_empty() {
        return Err(AdapterError::InvalidInput(
            "No misaki dictionaries found. Expected us_gold.json and us_silver.json in misaki/ directory".to_string()
        ));
    }

    // Simple tokenization: split by whitespace and punctuation
    let mut result = String::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        // Clean punctuation from word edges
        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let lower_word = clean_word.to_lowercase();

        // Try gold dict first, then silver
        let phonemes = lookup_word_phonemes(&lower_word, &gold_dict)
            .or_else(|| lookup_word_phonemes(&lower_word, &silver_dict))
            .or_else(|| lookup_word_phonemes(clean_word, &gold_dict))
            .or_else(|| lookup_word_phonemes(clean_word, &silver_dict))
            .unwrap_or_else(|| fallback_phonemize(&lower_word));

        result.push_str(&phonemes);

        // Add space between words (will be filtered if not in vocab)
        if i < words.len() - 1 {
            result.push(' ');
        }
    }

    // Filter to only characters in vocabulary
    let filtered: String = result.chars().filter(|c| vocab.contains_key(c)).collect();

    Ok(filtered.trim().to_string())
}

/// Look up a word's phonemes in a misaki dictionary.
fn lookup_word_phonemes(
    word: &str,
    dict: &std::collections::HashMap<String, serde_json::Value>,
) -> Option<String> {
    dict.get(word).and_then(|v| {
        match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(obj) => {
                // Has POS-specific pronunciations, use DEFAULT
                obj.get("DEFAULT")
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string())
            }
            _ => None,
        }
    })
}

/// Fallback phonemization for out-of-vocabulary words.
/// Uses simple letter-to-phoneme mapping for English.
fn fallback_phonemize(word: &str) -> String {
    // Very basic fallback: skip unknown words or use simple mapping
    // This is intentionally conservative - returns empty for unknown words
    // rather than producing bad phonemes that might sound worse
    let mut result = String::new();
    for c in word.chars() {
        match c.to_ascii_lowercase() {
            'a' => result.push_str("æ"),
            'e' => result.push_str("ɛ"),
            'i' => result.push_str("ɪ"),
            'o' => result.push_str("ɑ"),
            'u' => result.push_str("ʌ"),
            'b' => result.push_str("b"),
            'c' => result.push_str("k"),
            'd' => result.push_str("d"),
            'f' => result.push_str("f"),
            'g' => result.push_str("ɡ"),
            'h' => result.push_str("h"),
            'j' => result.push_str("ʤ"),
            'k' => result.push_str("k"),
            'l' => result.push_str("l"),
            'm' => result.push_str("m"),
            'n' => result.push_str("n"),
            'p' => result.push_str("p"),
            'q' => result.push_str("k"),
            'r' => result.push_str("ɹ"),
            's' => result.push_str("s"),
            't' => result.push_str("t"),
            'v' => result.push_str("v"),
            'w' => result.push_str("w"),
            'x' => result.push_str("ks"),
            'y' => result.push_str("j"),
            'z' => result.push_str("z"),
            _ => {} // Skip non-alphabetic characters
        }
    }
    result
}

/// Normalize text for TTS processing.
///
/// Applies common text transformations:
/// - Normalize quotes and special characters
/// - Expand common abbreviations (Dr., Mr., etc.)
/// - Clean up whitespace
fn normalize_text_for_tts(text: &str) -> String {
    let mut result = text.to_string();

    // Normalize quotes
    result = result.replace('\u{2018}', "'").replace('\u{2019}', "'");
    result = result.replace('\u{201C}', "\"").replace('\u{201D}', "\"");

    // Expand common abbreviations
    result = result.replace("Dr.", "Doctor");
    result = result.replace("Mr.", "Mister");
    result = result.replace("Mrs.", "Missus");
    result = result.replace("Ms.", "Miss");
    result = result.replace("etc.", "etcetera");

    // Normalize whitespace
    let mut prev_space = false;
    result = result
        .chars()
        .filter_map(|c| {
            if c.is_whitespace() {
                if prev_space {
                    None
                } else {
                    prev_space = true;
                    Some(' ')
                }
            } else {
                prev_space = false;
                Some(c)
            }
        })
        .collect();

    result.trim().to_string()
}

// ============================================================================
// Audio Decoding Module
// ============================================================================

/// Decode audio from various formats to float32 samples.
///
/// Supports:
/// - Pre-decoded float32 samples (from AudioEnvelope, format="float32")
/// - Raw PCM16 bytes (format="pcm16")
/// - WAV files (format="wav" or unspecified)
///
/// After decoding, converts to target sample rate and channel count.
fn decode_audio_step(
    data: PreprocessedData,
    input_envelope: &Envelope,
    target_sample_rate: u32,
    target_channels: usize,
) -> ExecutorResult<PreprocessedData> {
    // First, try to use Envelope's format-aware extraction
    if let Ok(Some(audio_samples)) = input_envelope.to_audio_samples() {
        // Audio was already decoded (float32 or pcm16 format)
        let prepared = prepare_audio_samples(
            audio_samples.samples,
            audio_samples.sample_rate,
            audio_samples.channels as usize,
            target_sample_rate,
            target_channels,
        );
        return Ok(PreprocessedData::AudioSamples(prepared));
    }

    // Fallback: decode from raw bytes (WAV format)
    let audio_bytes = match data {
        PreprocessedData::AudioBytes(bytes) => bytes,
        _ => {
            return Err(AdapterError::InvalidInput(
                "AudioDecode requires audio bytes input".to_string(),
            ))
        }
    };

    decode_wav_audio(&audio_bytes, target_sample_rate, target_channels)
}

/// Decode WAV audio bytes to float32 samples.
fn decode_wav_audio(
    audio_bytes: &[u8],
    target_sample_rate: u32,
    target_channels: usize,
) -> ExecutorResult<PreprocessedData> {
    use std::io::Cursor;
    let cursor = Cursor::new(audio_bytes);

    match hound::WavReader::new(cursor) {
        Ok(mut reader) => {
            let spec = reader.spec();
            let source_sample_rate = spec.sample_rate;
            let source_channels = spec.channels as usize;

            // Read samples as f32
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => reader
                    .samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect(),
                hound::SampleFormat::Int => {
                    let bits = spec.bits_per_sample;
                    let max_value = match (1i32).checked_shl((bits - 1) as u32) {
                        Some(val) => val as f32,
                        None => {
                            return Err(AdapterError::InvalidInput(format!(
                                "Unsupported bits_per_sample: {} (must be < 32)",
                                bits
                            )));
                        }
                    };
                    reader
                        .samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / max_value)
                        .collect()
                }
            };

            let prepared = prepare_audio_samples(
                samples,
                source_sample_rate,
                source_channels,
                target_sample_rate,
                target_channels,
            );

            Ok(PreprocessedData::AudioSamples(prepared))
        }
        Err(e) => Err(AdapterError::InvalidInput(format!(
            "Failed to decode WAV audio: {}. Only WAV format is currently supported.",
            e
        ))),
    }
}

/// Prepare audio samples by converting to target sample rate and channels.
fn prepare_audio_samples(
    samples: Vec<f32>,
    source_sample_rate: u32,
    source_channels: usize,
    target_sample_rate: u32,
    target_channels: usize,
) -> Vec<f32> {
    // Convert to mono if needed
    let mono_samples = if source_channels > 1 && target_channels == 1 {
        samples
            .chunks(source_channels)
            .map(|chunk| chunk.iter().sum::<f32>() / source_channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample if needed
    if source_sample_rate != target_sample_rate {
        let ratio = target_sample_rate as f32 / source_sample_rate as f32;
        let target_len = (mono_samples.len() as f32 * ratio) as usize;

        (0..target_len)
            .map(|i| {
                let source_idx = (i as f32 / ratio) as usize;
                mono_samples.get(source_idx).copied().unwrap_or(0.0)
            })
            .collect()
    } else {
        mono_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_executor_creation() {
        let executor = TemplateExecutor::new();
        assert_eq!(executor.base_path, "");
    }

    #[test]
    fn test_executor_with_base_path() {
        let executor = TemplateExecutor::with_base_path("/path/to/models");
        assert_eq!(executor.base_path, "/path/to/models");
    }

    #[test]
    fn test_resolve_file_path() {
        let executor = TemplateExecutor::with_base_path("/models");
        let resolved = executor.resolve_file_path("encoder.onnx");
        assert!(resolved.contains("encoder.onnx"));
    }

    // ========================================================================
    // Preprocessing Step Unit Tests
    // ========================================================================

    mod preprocessing_steps {
        use super::*;

        // --------------------------------------------------------------------
        // mel_spectrogram_step tests
        // --------------------------------------------------------------------

        #[test]
        fn test_mel_spectrogram_step_basic() {
            // Create a simple sine wave as audio input
            let sample_rate = 16000;
            let duration_secs = 0.5;
            let num_samples = (sample_rate as f32 * duration_secs) as usize;
            let frequency = 440.0; // A4 note

            let samples: Vec<f32> = (0..num_samples)
                .map(|i| {
                    let t = i as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * frequency * t).sin()
                })
                .collect();

            let data = PreprocessedData::AudioSamples(samples);
            // Use preset "whisper" for standard Whisper configuration
            let result = mel_spectrogram_step(
                data,
                Some("whisper"),
                80,
                16000,
                400,
                160,
                MelScaleType::Slaney,
                Some(3000),
            );

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    let shape = tensor.shape();
                    // Should be [1, n_mels, time_frames]
                    assert_eq!(shape.len(), 3);
                    assert_eq!(shape[0], 1); // batch
                    assert_eq!(shape[1], 80); // n_mels
                    assert!(shape[2] > 0); // time frames
                }
                _ => panic!("Expected Tensor output from mel_spectrogram_step"),
            }
        }

        #[test]
        fn test_mel_spectrogram_step_invalid_input() {
            // Text input should fail
            let data = PreprocessedData::Text("invalid".to_string());
            let result = mel_spectrogram_step(
                data,
                None,
                80,
                16000,
                400,
                160,
                MelScaleType::Slaney,
                Some(3000),
            );

            assert!(result.is_err());
        }

        #[test]
        fn test_mel_spectrogram_step_empty_audio() {
            let data = PreprocessedData::AudioSamples(vec![]);
            let result = mel_spectrogram_step(
                data,
                None,
                80,
                16000,
                400,
                160,
                MelScaleType::Slaney,
                Some(3000),
            );

            // Empty audio produces an error (cannot compute spectrogram without samples)
            assert!(result.is_err());
        }

        // --------------------------------------------------------------------
        // normalize_step tests
        // --------------------------------------------------------------------

        #[test]
        fn test_normalize_step_basic() {
            // Create a tensor with known values
            let data = Array1::from_vec(vec![0.0, 10.0, 20.0, 30.0])
                .into_dyn();
            let input = PreprocessedData::Tensor(data);

            let mean = vec![10.0];
            let std = vec![10.0];

            let result = normalize_step(input, &mean, &std);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    let values: Vec<f32> = tensor.iter().cloned().collect();
                    // (0-10)/10 = -1, (10-10)/10 = 0, (20-10)/10 = 1, (30-10)/10 = 2
                    assert!((values[0] - (-1.0)).abs() < 0.001);
                    assert!((values[1] - 0.0).abs() < 0.001);
                    assert!((values[2] - 1.0).abs() < 0.001);
                    assert!((values[3] - 2.0).abs() < 0.001);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_normalize_step_multichannel() {
            // Create a 2D tensor [batch, channels] = [2, 3]
            let data = ndarray::Array2::from_shape_vec(
                (2, 3),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap()
            .into_dyn();
            let input = PreprocessedData::Tensor(data);

            let mean = vec![1.0, 2.0, 3.0];
            let std = vec![1.0, 1.0, 1.0];

            let result = normalize_step(input, &mean, &std);

            assert!(result.is_ok());
        }

        #[test]
        fn test_normalize_step_invalid_input() {
            let data = PreprocessedData::Text("text".to_string());
            let result = normalize_step(data, &[0.0], &[1.0]);

            assert!(result.is_err());
        }

        // --------------------------------------------------------------------
        // reshape_step tests
        // --------------------------------------------------------------------

        #[test]
        fn test_reshape_step_basic() {
            // Create a 1D tensor with 12 elements
            let data = Array1::from_vec((0..12).map(|i| i as f32).collect())
                .into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = reshape_step(input, &[3, 4]);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[3, 4]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_reshape_step_multidimensional() {
            // Create a tensor with 24 elements
            let data = Array1::from_vec((0..24).map(|i| i as f32).collect())
                .into_dyn();
            let input = PreprocessedData::Tensor(data);

            // Reshape to 3D: [2, 4, 3]
            let result = reshape_step(input, &[2, 4, 3]);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[2, 4, 3]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_reshape_step_invalid_shape() {
            // Create a tensor with 10 elements
            let data = Array1::from_vec((0..10).map(|i| i as f32).collect())
                .into_dyn();
            let input = PreprocessedData::Tensor(data);

            // 10 cannot be reshaped to [3, 4] (12 elements)
            let result = reshape_step(input, &[3, 4]);

            assert!(result.is_err());
        }

        #[test]
        fn test_reshape_step_invalid_input() {
            let data = PreprocessedData::Text("text".to_string());
            let result = reshape_step(data, &[2, 2]);

            assert!(result.is_err());
        }

        // --------------------------------------------------------------------
        // center_crop_step tests
        // --------------------------------------------------------------------

        #[test]
        fn test_center_crop_step_basic() {
            // Create a 4D tensor [batch, channels, height, width] = [1, 3, 100, 100]
            let data = ndarray::Array4::<f32>::zeros((1, 3, 100, 100)).into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = center_crop_step(input, 50, 50);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[1, 3, 50, 50]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_center_crop_step_3d_tensor() {
            // Create a 3D tensor [channels, height, width] = [3, 64, 64]
            let data = ndarray::Array3::<f32>::zeros((3, 64, 64)).into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = center_crop_step(input, 32, 32);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[3, 32, 32]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_center_crop_step_larger_than_input() {
            // Crop size larger than input should fail
            let data = ndarray::Array4::<f32>::zeros((1, 3, 50, 50)).into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = center_crop_step(input, 100, 100);

            assert!(result.is_err());
        }

        #[test]
        fn test_center_crop_step_invalid_input() {
            let data = PreprocessedData::AudioSamples(vec![1.0, 2.0, 3.0]);
            let result = center_crop_step(data, 10, 10);

            assert!(result.is_err());
        }

        // --------------------------------------------------------------------
        // resize_step tests
        // --------------------------------------------------------------------

        #[test]
        fn test_resize_step_rgb_upscale() {
            use crate::execution_template::InterpolationMethod;

            // Create a small 4D RGB tensor [1, 3, 4, 4]
            let mut data = ndarray::Array4::<f32>::zeros((1, 3, 4, 4));
            // Set some pixel values
            data[[0, 0, 0, 0]] = 1.0; // Red channel
            data[[0, 1, 0, 0]] = 0.5; // Green channel
            data[[0, 2, 0, 0]] = 0.0; // Blue channel

            let input = PreprocessedData::Tensor(data.into_dyn());

            let result = resize_step(input, 8, 8, &InterpolationMethod::Bilinear);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[1, 3, 8, 8]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_resize_step_rgb_downscale() {
            use crate::execution_template::InterpolationMethod;

            let data = ndarray::Array4::<f32>::zeros((1, 3, 100, 100)).into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = resize_step(input, 50, 50, &InterpolationMethod::Nearest);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[1, 3, 50, 50]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_resize_step_grayscale() {
            use crate::execution_template::InterpolationMethod;

            let data = ndarray::Array4::<f32>::zeros((1, 1, 32, 32)).into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = resize_step(input, 64, 64, &InterpolationMethod::Bicubic);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[1, 1, 64, 64]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_resize_step_3d_tensor() {
            use crate::execution_template::InterpolationMethod;

            // 3D tensor [channels, height, width]
            let data = ndarray::Array3::<f32>::zeros((3, 20, 20)).into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = resize_step(input, 10, 10, &InterpolationMethod::Bilinear);

            assert!(result.is_ok());
            match result.unwrap() {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[3, 10, 10]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }

        #[test]
        fn test_resize_step_invalid_channels() {
            use crate::execution_template::InterpolationMethod;

            // 5 channels is not supported
            let data = ndarray::Array4::<f32>::zeros((1, 5, 10, 10)).into_dyn();
            let input = PreprocessedData::Tensor(data);

            let result = resize_step(input, 20, 20, &InterpolationMethod::Nearest);

            assert!(result.is_err());
        }

        #[test]
        fn test_resize_step_invalid_input() {
            use crate::execution_template::InterpolationMethod;

            let data = PreprocessedData::Text("text".to_string());
            let result = resize_step(data, 10, 10, &InterpolationMethod::Nearest);

            assert!(result.is_err());
        }

        // --------------------------------------------------------------------
        // Audio decoding helper tests
        // --------------------------------------------------------------------

        #[test]
        fn test_prepare_audio_samples_mono_no_resample() {
            let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
            let result = prepare_audio_samples(samples, 16000, 1, 16000, 1);

            assert_eq!(result.len(), 5);
            assert!((result[2] - 1.0).abs() < 0.001);
        }

        #[test]
        fn test_prepare_audio_samples_stereo_to_mono() {
            // Stereo samples: L, R, L, R, L, R
            let samples = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
            let result = prepare_audio_samples(samples, 16000, 2, 16000, 1);

            // Should average L+R: (1+0)/2=0.5, (0.5+0.5)/2=0.5, (0+1)/2=0.5
            assert_eq!(result.len(), 3);
            assert!((result[0] - 0.5).abs() < 0.001);
            assert!((result[1] - 0.5).abs() < 0.001);
            assert!((result[2] - 0.5).abs() < 0.001);
        }

        #[test]
        fn test_prepare_audio_samples_resample() {
            // 8000 Hz samples
            let samples: Vec<f32> = (0..80).map(|i| (i as f32 / 80.0)).collect();
            let result = prepare_audio_samples(samples, 8000, 1, 16000, 1);

            // Upsampling 2x should roughly double the length
            assert!(result.len() > 80);
        }
    }
}

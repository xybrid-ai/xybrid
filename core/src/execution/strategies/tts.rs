//! TTS execution strategy.
//!
//! Handles text-to-speech models with:
//! - Automatic text chunking for long inputs
//! - Voice embedding loading
//! - Audio concatenation

use log::{debug, info};
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

use super::{ExecutionContext, ExecutionStrategy};
use crate::execution::modes::execute_tts_inference;
use crate::execution::template::{ExecutionTemplate, ModelMetadata, PreprocessingStep};
use crate::execution::types::{ExecutorResult, PreprocessedData, RawOutputs};
use crate::execution::voice_loader::TtsVoiceLoader;
use crate::execution::{postprocessing, preprocessing};
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use crate::tracing as xybrid_trace;

/// Maximum characters per TTS chunk (Kokoro's BERT encoder has ~512 token limit).
const MAX_TTS_CHARS: usize = 350;

/// TTS execution strategy.
///
/// Handles TTS models that use Phonemize preprocessing. Supports:
/// - Automatic chunking for long text inputs
/// - Voice embedding loading via TtsVoiceLoader
/// - Audio concatenation for chunked execution
pub struct TtsStrategy {
    max_chars: usize,
}

impl TtsStrategy {
    /// Create a new TTS strategy with default settings.
    pub fn new() -> Self {
        Self {
            max_chars: MAX_TTS_CHARS,
        }
    }

    /// Create with custom max characters per chunk.
    pub fn with_max_chars(max_chars: usize) -> Self {
        Self { max_chars }
    }

    /// Check if metadata indicates a TTS model.
    fn is_tts_model(metadata: &ModelMetadata) -> bool {
        metadata
            .preprocessing
            .iter()
            .any(|step| matches!(step, PreprocessingStep::Phonemize { .. }))
    }

    /// Get the model file path from metadata.
    fn get_model_file(metadata: &ModelMetadata) -> ExecutorResult<&str> {
        match &metadata.execution_template {
            ExecutionTemplate::Onnx { model_file } => Ok(model_file),
            ExecutionTemplate::SafeTensors { model_file, .. } => Ok(model_file),
            _ => Err(AdapterError::InvalidInput(
                "TTS strategy requires ONNX or SafeTensors model".to_string(),
            )),
        }
    }

    /// Run preprocessing steps from metadata.
    fn run_preprocessing(
        &self,
        ctx: &ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<PreprocessedData> {
        if metadata.preprocessing.is_empty() {
            debug!(target: "xybrid_core", "No preprocessing steps configured");
            return PreprocessedData::from_envelope(input);
        }

        let _preprocess_span = xybrid_trace::SpanGuard::new("preprocessing");

        let mut data = PreprocessedData::from_envelope(input)?;
        for step in &metadata.preprocessing {
            data = preprocessing::apply_preprocessing_step(step, data, input, ctx.base_path)?;
        }
        Ok(data)
    }

    /// Run postprocessing steps from metadata.
    fn run_postprocessing(
        &self,
        ctx: &ExecutionContext<'_>,
        metadata: &ModelMetadata,
        outputs: RawOutputs,
    ) -> ExecutorResult<Envelope> {
        if metadata.postprocessing.is_empty() {
            debug!(target: "xybrid_core", "No postprocessing steps configured");
            return outputs.to_envelope();
        }

        let _postprocess_span = xybrid_trace::SpanGuard::new("postprocessing");

        let mut data = outputs;
        for step in &metadata.postprocessing {
            data = postprocessing::apply_postprocessing_step(step, data, ctx.base_path)?;
        }
        data.to_envelope()
    }

    /// Execute TTS for a single (short) text input.
    fn execute_single(
        &self,
        ctx: &ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
        model_path: &Path,
    ) -> ExecutorResult<Envelope> {
        // Run preprocessing
        let preprocessed = self.run_preprocessing(ctx, metadata, input)?;

        let phoneme_ids = preprocessed
            .as_phoneme_ids()
            .ok_or_else(|| AdapterError::InvalidInput("Expected phoneme IDs".to_string()))?;

        eprintln!(
            "[DEBUG TTS Single] Phoneme IDs count: {}, first 20: {:?}",
            phoneme_ids.len(),
            &phoneme_ids[..phoneme_ids.len().min(20)]
        );

        // Load voice embedding
        let voice_loader = TtsVoiceLoader::new(ctx.base_path);
        let voice_embedding = voice_loader.load(metadata, input)?;

        // Create and run TTS session
        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)?;
        let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding)?;

        // Run postprocessing
        self.run_postprocessing(ctx, metadata, RawOutputs::TensorMap(raw_outputs))
    }

    /// Execute TTS with automatic chunking for long text.
    fn execute_chunked(
        &self,
        ctx: &ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
        model_path: &Path,
    ) -> ExecutorResult<Envelope> {
        let text = match &input.kind {
            EnvelopeKind::Text(t) => t.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "TTS requires text input".to_string(),
                ))
            }
        };

        eprintln!(
            "[DEBUG TTS Chunked] Input text length: {} chars (max={})",
            text.len(),
            self.max_chars
        );

        // If text is short enough, use single execution
        if text.len() <= self.max_chars {
            return self.execute_single(ctx, metadata, input, model_path);
        }

        info!(
            target: "xybrid_core",
            "Text too long ({} chars), splitting into chunks",
            text.len()
        );

        // Split text into chunks
        let chunks = Self::chunk_text(&text, self.max_chars);
        eprintln!("[DEBUG TTS] Split into {} chunks", chunks.len());

        // Process each chunk and collect audio
        let mut all_audio: Vec<f32> = Vec::new();
        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)?;

        for (i, chunk) in chunks.iter().enumerate() {
            eprintln!(
                "[DEBUG TTS] Processing chunk {}/{}: {} chars",
                i + 1,
                chunks.len(),
                chunk.len()
            );

            // Create envelope for this chunk
            let chunk_input = Envelope {
                kind: EnvelopeKind::Text(chunk.clone()),
                metadata: input.metadata.clone(),
            };

            // Run preprocessing on chunk
            let preprocessed = self.run_preprocessing(ctx, metadata, &chunk_input)?;

            // Get phoneme IDs
            let phoneme_ids = preprocessed
                .as_phoneme_ids()
                .ok_or_else(|| AdapterError::InvalidInput("Expected phoneme IDs".to_string()))?;

            eprintln!(
                "[DEBUG TTS] Chunk {} has {} phoneme IDs",
                i + 1,
                phoneme_ids.len()
            );

            // Load voice embedding
            let voice_loader = TtsVoiceLoader::new(ctx.base_path);
            let voice_embedding = voice_loader.load(metadata, input)?;

            // Run TTS inference
            let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding)?;

            // Extract audio from outputs
            if let Some(audio_tensor) = raw_outputs.values().next() {
                let chunk_audio: Vec<f32> = audio_tensor.iter().cloned().collect();
                all_audio.extend(chunk_audio);
            }
        }

        eprintln!("[DEBUG TTS] Total audio samples: {}", all_audio.len());

        // Convert concatenated audio to envelope
        let output_names = session.output_names();
        let output_name = output_names.first().map(|s| s.as_str()).unwrap_or("audio");

        let mut combined_outputs: HashMap<String, ArrayD<f32>> = HashMap::new();
        let audio_array = ndarray::Array1::from_vec(all_audio).into_dyn();
        combined_outputs.insert(output_name.to_string(), audio_array);

        // Run postprocessing on combined audio
        self.run_postprocessing(ctx, metadata, RawOutputs::TensorMap(combined_outputs))
    }

    /// Split text into chunks at sentence boundaries.
    fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
        if text.len() <= max_chars {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        // Split into sentences (keep delimiter)
        let sentences: Vec<&str> = text
            .split_inclusive(|c| c == '.' || c == '!' || c == '?')
            .collect();

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            // If single sentence is too long, split at commas or spaces
            if sentence.len() > max_chars {
                // Flush current chunk first
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                }

                // Split long sentence at commas or spaces
                let mut remaining = sentence;
                while remaining.len() > max_chars {
                    let split_at = remaining[..max_chars]
                        .rfind(|c: char| c == ',' || c.is_whitespace())
                        .unwrap_or(max_chars);
                    chunks.push(remaining[..split_at].trim().to_string());
                    remaining = remaining[split_at..].trim_start_matches(',').trim();
                }
                if !remaining.is_empty() {
                    current_chunk = remaining.to_string();
                }
            } else if current_chunk.len() + sentence.len() + 1 > max_chars {
                // Current chunk would exceed limit, start new chunk
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }
                current_chunk = sentence.to_string();
            } else {
                // Add to current chunk
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(sentence);
            }
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }
}

impl Default for TtsStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionStrategy for TtsStrategy {
    fn can_handle(&self, metadata: &ModelMetadata) -> bool {
        Self::is_tts_model(metadata)
    }

    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        let _span = xybrid_trace::SpanGuard::new("tts_execution");

        let model_file = Self::get_model_file(metadata)?;
        let model_path = ctx.resolve_path(model_file);

        self.execute_chunked(ctx, metadata, input, &model_path)
    }

    fn name(&self) -> &'static str {
        "tts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_tts_model_true() {
        let metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx")
            .with_preprocessing(PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            });

        assert!(TtsStrategy::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_false() {
        let metadata = ModelMetadata::onnx("test-asr", "1.0", "model.onnx");
        assert!(!TtsStrategy::is_tts_model(&metadata));
    }

    #[test]
    fn test_chunk_text_short() {
        let chunks = TtsStrategy::chunk_text("Hello world.", 350);
        assert_eq!(chunks, vec!["Hello world."]);
    }

    #[test]
    fn test_chunk_text_sentences() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = TtsStrategy::chunk_text(text, 20);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_can_handle() {
        let strategy = TtsStrategy::new();

        let tts_metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx")
            .with_preprocessing(PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            });

        let other_metadata = ModelMetadata::onnx("test-other", "1.0", "model.onnx");

        assert!(strategy.can_handle(&tts_metadata));
        assert!(!strategy.can_handle(&other_metadata));
    }
}

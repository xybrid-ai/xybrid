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

        debug!(
            target: "xybrid_core",
            "TTS Single: Phoneme IDs count: {}, first 20: {:?}",
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

        debug!(
            target: "xybrid_core",
            "TTS Chunked: Input text length: {} chars (max={})",
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
        debug!(target: "xybrid_core", "TTS: Split into {} chunks", chunks.len());

        // Process each chunk and collect audio
        let mut all_audio: Vec<f32> = Vec::new();
        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)?;

        for (i, chunk) in chunks.iter().enumerate() {
            debug!(
                target: "xybrid_core",
                "TTS: Processing chunk {}/{}: {} chars",
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

            debug!(
                target: "xybrid_core",
                "TTS: Chunk {} has {} phoneme IDs",
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

        debug!(target: "xybrid_core", "TTS: Total audio samples: {}", all_audio.len());

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

    // ============================================================================
    // TTS Model Detection Tests
    // ============================================================================

    #[test]
    fn test_is_tts_model_with_phonemize() {
        let metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx").with_preprocessing(
            PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            },
        );

        assert!(TtsStrategy::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_without_phonemize() {
        let metadata = ModelMetadata::onnx("test-asr", "1.0", "model.onnx");
        assert!(!TtsStrategy::is_tts_model(&metadata));
    }

    #[test]
    fn test_can_handle_tts() {
        let strategy = TtsStrategy::new();

        let tts_metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx").with_preprocessing(
            PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            },
        );

        assert!(strategy.can_handle(&tts_metadata));
    }

    #[test]
    fn test_cannot_handle_non_tts() {
        let strategy = TtsStrategy::new();
        let other_metadata = ModelMetadata::onnx("test-other", "1.0", "model.onnx");
        assert!(!strategy.can_handle(&other_metadata));
    }

    // ============================================================================
    // Text Chunking Tests - Core Logic (These test what we implemented!)
    // ============================================================================

    #[test]
    fn test_chunk_text_under_limit_returns_single() {
        let chunks = TtsStrategy::chunk_text("Hello world.", 350);
        assert_eq!(chunks, vec!["Hello world."]);
    }

    #[test]
    fn test_chunk_text_empty_string() {
        let chunks = TtsStrategy::chunk_text("", 350);
        // Empty input should return either empty vec or vec with empty string
        assert!(chunks.is_empty() || (chunks.len() == 1 && chunks[0].is_empty()));
    }

    #[test]
    fn test_chunk_text_splits_at_sentence_boundaries() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = TtsStrategy::chunk_text(text, 20);

        // Each sentence is ~16 chars, so should split into 3 chunks
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First sentence.");
        assert_eq!(chunks[1], "Second sentence.");
        assert_eq!(chunks[2], "Third sentence.");
    }

    #[test]
    fn test_chunk_text_combines_short_sentences() {
        let text = "Hi. Hello. Hey there.";
        // Each sentence is short, they should combine until max_chars
        let chunks = TtsStrategy::chunk_text(text, 50);
        // All fit in one chunk
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_text_respects_max_chars() {
        let text = "This is a test sentence. Here is another one. And a third.";
        let max_chars = 30;
        let chunks = TtsStrategy::chunk_text(text, max_chars);

        // Most chunks should be under max_chars (some tolerance for not breaking mid-sentence)
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() <= max_chars + 15, // Allow tolerance for sentence boundaries
                "Chunk {} too long: {} chars (max {}): '{}'",
                i,
                chunk.len(),
                max_chars,
                chunk
            );
        }
    }

    #[test]
    fn test_chunk_text_long_sentence_splits_at_comma() {
        let text = "This is a very long sentence with many words, and it has a comma here, which should be a split point.";
        let chunks = TtsStrategy::chunk_text(text, 50);

        // Should split at commas
        assert!(chunks.len() >= 2, "Long sentence should be split");
    }

    #[test]
    fn test_chunk_text_long_sentence_splits_at_space() {
        // Sentence without commas
        let text = "This is a sentence without any commas that should still be split somewhere at a word boundary.";
        let chunks = TtsStrategy::chunk_text(text, 40);

        // Should split at word boundaries
        assert!(chunks.len() >= 2, "Should split at spaces");
        for chunk in &chunks {
            // No partial words - check chunk starts with valid char
            if let Some(c) = chunk.chars().next() {
                assert!(
                    c.is_alphabetic() || c == '"',
                    "Chunk starts unexpectedly: '{}'",
                    chunk
                );
            }
        }
    }

    #[test]
    fn test_chunk_text_preserves_content() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = TtsStrategy::chunk_text(text, 20);

        // All words should be in the output
        let rejoined = chunks.join(" ");
        assert!(rejoined.contains("First"));
        assert!(rejoined.contains("Second"));
        assert!(rejoined.contains("Third"));
    }

    #[test]
    fn test_chunk_text_handles_question_marks() {
        let text = "Is this a question? Yes it is! And here is a statement.";
        let chunks = TtsStrategy::chunk_text(text, 25);

        assert!(chunks.len() >= 2, "Should split at ? and !");
    }

    #[test]
    fn test_chunk_text_handles_exclamation_marks() {
        let text = "Wow! Amazing! Incredible!";
        let chunks = TtsStrategy::chunk_text(text, 10);

        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_text_real_llm_output() {
        // This is the actual use case - LLM generating long responses
        let text = "Paris is the capital of France. France is a country in Western Europe. \
            It is known for its art, culture, and cuisine. The Eiffel Tower is a famous landmark. \
            Paris has a population of over 2 million people.";

        let chunks = TtsStrategy::chunk_text(text, 100);

        assert!(
            chunks.len() >= 2,
            "LLM output of {} chars should split with max=100",
            text.len()
        );

        // Verify no content is lost
        let total_chars: usize = chunks.iter().map(|c| c.len()).sum();
        assert!(
            total_chars >= text.len() - 30, // Allow for whitespace trimming
            "Content lost: {} vs {}",
            total_chars,
            text.len()
        );
    }

    #[test]
    fn test_chunk_text_with_llm_special_tokens() {
        // LLM outputs often contain special tokens that shouldn't break chunking
        let text = "Paris.<|im_end|><|im_start|>user\nWhat else?<|im_end|><|im_start|>assistant\n\
            France has many cities.";

        let chunks = TtsStrategy::chunk_text(text, 80);

        // Should not panic on special characters
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_very_long_input() {
        // Test with ~1000 chars (like the original failing case)
        let text = "Paris is the capital. ".repeat(50); // ~1100 chars
        let chunks = TtsStrategy::chunk_text(&text, 350);

        assert!(chunks.len() >= 3, "Should split into multiple chunks");

        // All chunks should be under limit
        for chunk in &chunks {
            assert!(
                chunk.len() <= 400, // Some tolerance
                "Chunk too long: {} chars",
                chunk.len()
            );
        }
    }

    // ============================================================================
    // Strategy Configuration Tests
    // ============================================================================

    #[test]
    fn test_custom_max_chars() {
        let strategy = TtsStrategy::with_max_chars(100);
        assert_eq!(strategy.max_chars, 100);
    }

    #[test]
    fn test_default_max_chars_is_350() {
        let strategy = TtsStrategy::new();
        assert_eq!(strategy.max_chars, MAX_TTS_CHARS);
        assert_eq!(strategy.max_chars, 350);
    }

    #[test]
    fn test_strategy_name() {
        let strategy = TtsStrategy::new();
        assert_eq!(strategy.name(), "tts");
    }

    // ============================================================================
    // Model File Extraction Tests
    // ============================================================================

    #[test]
    fn test_get_model_file_onnx() {
        let metadata = ModelMetadata::onnx("test", "1.0", "custom_model.onnx");
        let result = TtsStrategy::get_model_file(&metadata);
        assert_eq!(result.unwrap(), "custom_model.onnx");
    }

    #[test]
    fn test_get_model_file_safetensors() {
        let metadata = ModelMetadata::safetensors("test", "1.0", "model.safetensors", "whisper");
        let result = TtsStrategy::get_model_file(&metadata);
        assert_eq!(result.unwrap(), "model.safetensors");
    }

    #[test]
    fn test_get_model_file_model_graph_unsupported() {
        // ModelGraph templates should not be handled by TTS strategy
        let metadata = ModelMetadata::model_graph("test", "1.0", vec![], vec![]);
        let result = TtsStrategy::get_model_file(&metadata);
        assert!(
            result.is_err(),
            "ModelGraph should not be supported for TTS"
        );
    }
}

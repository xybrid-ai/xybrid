//! Codec TTS execution strategy for NeuTTS-style models.
//!
//! Orchestrates a GGUF backbone (llama.cpp) + ONNX codec decoder in a single
//! execution: PhonemeRaw → voice codes → prompt → LLM generate → CodecDecode.

use log::{debug, info};
use std::path::Path;

use super::{ExecutionContext, ExecutionStrategy};
use crate::execution::postprocessing::codec::{
    codec_decode_step, create_codec_session, decode_tokens_to_samples, extract_speech_tokens,
};
use crate::execution::strategies::llm::{LlmGenerationParams, LlmInference, LlmModelConfig};
use crate::execution::template::{
    ExecutionTemplate, GenerationParams, ModelMetadata, PostprocessingStep,
};
use crate::execution::types::ExecutorResult;
use crate::execution::voice_loader::TtsVoiceLoader;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::AdapterError;
use crate::tracing as xybrid_trace;

/// Maximum characters per chunk for codec TTS (LLM context budget minus prompt overhead).
const CODEC_TTS_MAX_CHARS: usize = 1500;

/// Silence duration between audio chunks in milliseconds.
const INTER_CHUNK_SILENCE_MS: u32 = 200;

/// Env var: set to a directory path to dump every pipeline intermediate to disk.
/// Writes: input_phonemes.txt, ref_phonemes.txt, ref_codes.txt, prompt.txt,
/// llm_output.txt, tokens.txt, waveform.f32 (raw f32 LE samples).
const DEBUG_DUMP_ENV: &str = "XYBRID_CODEC_TTS_DUMP";

/// Write `content` to `<XYBRID_CODEC_TTS_DUMP>/<filename>` if the env var is set.
/// Silently no-ops when the env var is absent or the write fails.
fn dump(filename: &str, content: &[u8]) {
    if let Ok(dir) = std::env::var(DEBUG_DUMP_ENV) {
        let path = Path::new(&dir).join(filename);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(&path, content) {
            log::warn!(target: "xybrid_core", "dump({}) failed: {}", path.display(), e);
        } else {
            debug!(target: "xybrid_core", "dumped intermediate: {}", path.display());
        }
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::execution::strategies::llm::DefaultLlmInference;

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
type DefaultInference = DefaultLlmInference;

#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
use crate::execution::strategies::llm::NoOpLlmInference;

#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
type DefaultInference = NoOpLlmInference;

/// Codec TTS strategy for NeuTTS-style models.
///
/// Handles models with `ExecutionTemplate::Gguf` + `PostprocessingStep::CodecDecode`.
/// Pipeline: phonemize input → load voice codes → build prompt → LLM generate → codec decode.
pub struct CodecTtsStrategy<I: LlmInference = DefaultInference> {
    inference: std::sync::Mutex<I>,
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
impl CodecTtsStrategy<DefaultLlmInference> {
    pub fn new() -> Self {
        Self {
            inference: std::sync::Mutex::new(DefaultLlmInference::new()),
        }
    }
}

#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
impl CodecTtsStrategy<NoOpLlmInference> {
    pub fn new() -> Self {
        Self {
            inference: std::sync::Mutex::new(NoOpLlmInference),
        }
    }
}

impl<I: LlmInference> CodecTtsStrategy<I> {
    /// Create with a custom inference backend (for testing).
    pub fn with_inference(inference: I) -> Self {
        Self {
            inference: std::sync::Mutex::new(inference),
        }
    }

    /// Check if metadata describes a codec TTS model (GGUF + CodecDecode).
    fn is_codec_tts(metadata: &ModelMetadata) -> bool {
        matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. })
            && metadata
                .postprocessing
                .iter()
                .any(|s| matches!(s, PostprocessingStep::CodecDecode { .. }))
    }

    /// Extract GGUF config from metadata.
    fn extract_gguf_config(
        metadata: &ModelMetadata,
        base_path: &str,
    ) -> ExecutorResult<LlmModelConfig> {
        match &metadata.execution_template {
            ExecutionTemplate::Gguf {
                model_file,
                chat_template,
                context_length,
                ..
            } => {
                let model_path = Path::new(base_path).join(model_file);
                let mut config =
                    LlmModelConfig::new(model_path.to_string_lossy().to_string(), *context_length);
                if let Some(template) = chat_template {
                    let template_path = Path::new(base_path).join(template);
                    config = config.with_chat_template(template_path.to_string_lossy().to_string());
                }
                if let Some(hint) = metadata.metadata.get("backend").and_then(|v| v.as_str()) {
                    config = config.with_backend_hint(hint);
                }
                Ok(config)
            }
            _ => Err(AdapterError::InvalidInput(
                "Expected GGUF execution template".to_string(),
            )),
        }
    }

    /// Build sampling params for LLM generation, starting from NeuTTS-friendly
    /// defaults (temperature=1.0, top_k=50, no top_p or repetition filtering,
    /// stops on `<|SPEECH_GENERATION_END|>`) and overriding any field that the
    /// model's `execution_template.generation_params` declares.
    fn build_generation_params(metadata: &ModelMetadata) -> LlmGenerationParams {
        let mut params = LlmGenerationParams {
            max_tokens: 2048,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
            system_prompt: None,
            stop_sequences: vec!["<|SPEECH_GENERATION_END|>".to_string()],
        };

        if let ExecutionTemplate::Gguf {
            generation_params: Some(overrides),
            ..
        } = &metadata.execution_template
        {
            let GenerationParams {
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                stop_sequences,
            } = overrides;
            if let Some(v) = max_tokens {
                params.max_tokens = *v;
            }
            if let Some(v) = temperature {
                params.temperature = *v;
            }
            if let Some(v) = top_p {
                params.top_p = *v;
            }
            if let Some(v) = top_k {
                params.top_k = *v;
            }
            if let Some(v) = repetition_penalty {
                params.repetition_penalty = *v;
            }
            if !stop_sequences.is_empty() {
                params.stop_sequences = stop_sequences.clone();
            }
        }

        params
    }

    /// Extract CodecDecode config from postprocessing steps.
    fn extract_codec_config(metadata: &ModelMetadata) -> ExecutorResult<(&str, u32, &str, bool)> {
        metadata
            .postprocessing
            .iter()
            .find_map(|s| match s {
                PostprocessingStep::CodecDecode {
                    decoder_model,
                    sample_rate,
                    token_pattern,
                    apply_postprocessing,
                } => Some((
                    decoder_model.as_str(),
                    *sample_rate,
                    token_pattern.as_str(),
                    *apply_postprocessing,
                )),
                _ => None,
            })
            .ok_or_else(|| {
                AdapterError::InvalidInput("No CodecDecode postprocessing step found".to_string())
            })
    }

    /// Phonemize text using the backend specified in metadata.
    fn phonemize_raw(
        metadata: &ModelMetadata,
        base_path: &str,
        text: &str,
    ) -> ExecutorResult<String> {
        use crate::execution::template::PreprocessingStep;

        let (backend_config, language) = metadata
            .preprocessing
            .iter()
            .find_map(|s| match s {
                PreprocessingStep::PhonemeRaw { backend, language } => {
                    Some((backend.clone(), language.clone()))
                }
                _ => None,
            })
            .ok_or_else(|| {
                AdapterError::InvalidInput(
                    "No PhonemeRaw preprocessing step found in metadata".to_string(),
                )
            })?;

        let backend_impl = backend_config.create(base_path, None, language.as_deref());
        backend_impl.phonemize_raw(text)
    }

    /// Build the NeuTTS prompt from phonemes and reference codes.
    fn build_prompt(ref_phones: &str, input_phones: &str, ref_codes: &[i32]) -> String {
        let ref_tokens: String = ref_codes
            .iter()
            .map(|c| format!("<|speech_{}|>", c))
            .collect::<Vec<_>>()
            .join("");

        format!(
            "user: Convert the text to speech:<|TEXT_PROMPT_START|>{} {}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{}",
            ref_phones, input_phones, ref_tokens
        )
    }

    /// Split text into chunks at sentence boundaries.
    fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
        if text.len() <= max_chars {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        let sentences: Vec<&str> = text.split_inclusive(['.', '!', '?']).collect();

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            if sentence.len() > max_chars {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                }
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
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }
                current_chunk = sentence.to_string();
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(sentence);
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }
}

impl<I: LlmInference + 'static> ExecutionStrategy for CodecTtsStrategy<I> {
    fn can_handle(&self, metadata: &ModelMetadata) -> bool {
        Self::is_codec_tts(metadata)
    }

    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        let _span = xybrid_trace::SpanGuard::new("codec_tts_execution");

        let input_text = match &input.kind {
            EnvelopeKind::Text(text) => text.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "Codec TTS requires text input".to_string(),
                ))
            }
        };

        let max_chars = metadata.max_chunk_chars.unwrap_or(CODEC_TTS_MAX_CHARS);

        info!(
            target: "xybrid_core",
            "Executing codec TTS: model={}, text_len={}, max_chars={}",
            metadata.model_id, input_text.len(), max_chars
        );

        // Load shared resources once: voice codes, reference transcript, phonemes
        let voice_id = input
            .metadata
            .get("voice_id")
            .map(|s| s.as_str())
            .or_else(|| metadata.voices.as_ref().map(|v| v.default.as_str()))
            .ok_or_else(|| {
                AdapterError::InvalidInput(
                    "No voice_id specified and no default voice configured".to_string(),
                )
            })?;

        let voice_loader = TtsVoiceLoader::new(ctx.base_path);
        let (ref_codes, ref_transcript) = voice_loader.load_reference_codes(metadata, voice_id)?;

        debug!(
            target: "xybrid_core",
            "Loaded voice '{}': {} ref codes, transcript='{}'",
            voice_id, ref_codes.len(),
            &ref_transcript[..ref_transcript.len().min(50)]
        );
        dump("ref_transcript.txt", ref_transcript.as_bytes());
        dump(
            "ref_codes.txt",
            ref_codes
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ")
                .as_bytes(),
        );

        let ref_phones = Self::phonemize_raw(metadata, ctx.base_path, &ref_transcript)?;
        dump("ref_phonemes.txt", ref_phones.as_bytes());

        xybrid_trace::add_metadata("model", &metadata.model_id);
        xybrid_trace::add_metadata("voice", voice_id);

        // Load LLM once
        let config = Self::extract_gguf_config(metadata, ctx.base_path)?;
        let mut inference = self
            .inference
            .lock()
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to acquire lock: {}", e)))?;

        if !inference.is_loaded() {
            debug!(target: "xybrid_core", "Loading LLM model: {}", config.model_path);
            inference.load_model(&config)?;
        }

        let (decoder_model, sample_rate, token_pattern, apply_pp) =
            Self::extract_codec_config(metadata)?;
        let decoder_path = Path::new(ctx.base_path).join(decoder_model);

        let params = Self::build_generation_params(metadata);

        let chunks = Self::chunk_text(&input_text, max_chars);

        if chunks.len() == 1 {
            // Single chunk — same as before
            let input_phones = Self::phonemize_raw(metadata, ctx.base_path, &input_text)?;
            dump("input_phonemes.txt", input_phones.as_bytes());

            let prompt = Self::build_prompt(&ref_phones, &input_phones, &ref_codes);
            dump("prompt.txt", prompt.as_bytes());

            // Raw generation — the prompt already has NeuTTS's user:/assistant: turns
            // and control tokens. Applying another chat template on top corrupts it.
            let llm_output = inference.generate_raw(&prompt, &params)?;
            dump("llm_output.txt", llm_output.as_bytes());

            let tail_start = llm_output.len().saturating_sub(200);
            debug!(
                target: "xybrid_core",
                "LLM produced {} chars. Tail: {:?}",
                llm_output.len(),
                &llm_output[tail_start..]
            );

            // Extract tokens first so we can see how many the regex found.
            let tokens = extract_speech_tokens(&llm_output, token_pattern)?;
            dump(
                "tokens.txt",
                tokens
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
                    .as_bytes(),
            );
            info!(
                target: "xybrid_core",
                "Extracted {} speech tokens from {} chars of LLM output",
                tokens.len(), llm_output.len()
            );

            let mut decoder_session = create_codec_session(&decoder_path)?;
            let samples =
                decode_tokens_to_samples(&mut decoder_session, &tokens, sample_rate, apply_pp)?;

            if std::env::var(DEBUG_DUMP_ENV).is_ok() {
                let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                dump("waveform.f32", &bytes);
            }

            let wav_bytes = crate::audio::samples_to_wav(&samples, sample_rate);
            // Keep codec_decode_step callable for other paths, but we've already
            // done the work inline so we can observe intermediates.
            let _ = codec_decode_step;

            info!(target: "xybrid_core", "Codec TTS complete: {} bytes WAV, {} samples", wav_bytes.len(), samples.len());
            return Ok(Envelope::new(EnvelopeKind::Audio(wav_bytes)));
        }

        // Multi-chunk path
        info!(target: "xybrid_core", "Splitting into {} chunks", chunks.len());

        let mut decoder_session = create_codec_session(&decoder_path)?;
        let silence_samples = (sample_rate as usize * INTER_CHUNK_SILENCE_MS as usize) / 1000;
        let mut all_samples: Vec<f32> = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            debug!(
                target: "xybrid_core",
                "Chunk {}/{}: {} chars", i + 1, chunks.len(), chunk.len()
            );

            if i > 0 {
                all_samples.extend(std::iter::repeat_n(0.0f32, silence_samples));
            }

            let chunk_phones = Self::phonemize_raw(metadata, ctx.base_path, chunk)?;
            let prompt = Self::build_prompt(&ref_phones, &chunk_phones, &ref_codes);
            let llm_output = inference.generate_raw(&prompt, &params)?;
            let tokens = extract_speech_tokens(&llm_output, token_pattern)?;
            let chunk_samples =
                decode_tokens_to_samples(&mut decoder_session, &tokens, sample_rate, apply_pp)?;

            all_samples.extend(chunk_samples);
        }

        let wav_bytes = crate::audio::samples_to_wav(&all_samples, sample_rate);

        info!(
            target: "xybrid_core",
            "Codec TTS chunked complete: {} chunks, {} bytes WAV",
            chunks.len(), wav_bytes.len()
        );

        Ok(Envelope::new(EnvelopeKind::Audio(wav_bytes)))
    }

    fn name(&self) -> &'static str {
        "codec_tts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::strategies::llm::LlmModelConfig;
    use crate::execution::template::{PhonemizerBackend, PreprocessingStep};
    use std::collections::HashMap;

    /// Mock LLM inference for testing.
    struct MockInference {
        loaded: std::sync::atomic::AtomicBool,
        response: String,
    }

    impl MockInference {
        fn new(response: &str) -> Self {
            Self {
                loaded: std::sync::atomic::AtomicBool::new(false),
                response: response.to_string(),
            }
        }
    }

    impl LlmInference for MockInference {
        fn load_model(&mut self, _config: &LlmModelConfig) -> ExecutorResult<()> {
            self.loaded.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        fn generate(&self, _prompt: &str, _params: &LlmGenerationParams) -> ExecutorResult<String> {
            Ok(self.response.clone())
        }
        fn is_loaded(&self) -> bool {
            self.loaded.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn backend_name(&self) -> &str {
            "mock"
        }
    }

    fn create_codec_tts_metadata() -> ModelMetadata {
        ModelMetadata {
            model_id: "neutts-nano-q4".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::Gguf {
                model_file: "model.gguf".to_string(),
                chat_template: None,
                context_length: 2048,
                generation_params: None,
            },
            preprocessing: vec![PreprocessingStep::PhonemeRaw {
                backend: PhonemizerBackend::MisakiDictionary,
                language: Some("en-us".to_string()),
            }],
            postprocessing: vec![PostprocessingStep::CodecDecode {
                decoder_model: "neucodec-decoder-int8.onnx".to_string(),
                sample_rate: 24000,
                token_pattern: r"<\|speech_(\d+)\|>".to_string(),
                apply_postprocessing: true,
            }],
            files: vec!["model.gguf".to_string()],
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        }
    }

    fn create_plain_gguf_metadata() -> ModelMetadata {
        ModelMetadata {
            model_id: "test-llm".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::Gguf {
                model_file: "model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["model.gguf".to_string()],
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        }
    }

    #[test]
    fn test_can_handle_codec_tts_metadata() {
        let strategy = CodecTtsStrategy::with_inference(MockInference::new(""));
        let metadata = create_codec_tts_metadata();
        assert!(strategy.can_handle(&metadata));
    }

    #[test]
    fn test_cannot_handle_plain_gguf_metadata() {
        let strategy = CodecTtsStrategy::with_inference(MockInference::new(""));
        let metadata = create_plain_gguf_metadata();
        assert!(!strategy.can_handle(&metadata));
    }

    #[test]
    fn test_cannot_handle_onnx_metadata() {
        let strategy = CodecTtsStrategy::with_inference(MockInference::new(""));
        let metadata = ModelMetadata::onnx("test", "1.0", "model.onnx");
        assert!(!strategy.can_handle(&metadata));
    }

    #[test]
    fn test_strategy_name() {
        let strategy = CodecTtsStrategy::with_inference(MockInference::new(""));
        assert_eq!(strategy.name(), "codec_tts");
    }

    #[test]
    fn test_build_prompt() {
        let prompt =
            CodecTtsStrategy::<MockInference>::build_prompt("hɛˈloʊ", "wɝld", &[10, 20, 30]);

        assert!(prompt.contains("<|TEXT_PROMPT_START|>"));
        assert!(prompt.contains("hɛˈloʊ wɝld"));
        assert!(prompt.contains("<|TEXT_PROMPT_END|>"));
        assert!(prompt.contains("<|SPEECH_GENERATION_START|>"));
        assert!(prompt.contains("<|speech_10|>"));
        assert!(prompt.contains("<|speech_20|>"));
        assert!(prompt.contains("<|speech_30|>"));
        assert!(prompt.starts_with("user: Convert the text to speech:"));
    }

    #[test]
    fn test_extract_codec_config() {
        let metadata = create_codec_tts_metadata();
        let (decoder, sr, pattern, pp) =
            CodecTtsStrategy::<MockInference>::extract_codec_config(&metadata).unwrap();
        assert_eq!(decoder, "neucodec-decoder-int8.onnx");
        assert_eq!(sr, 24000);
        assert_eq!(pattern, r"<\|speech_(\d+)\|>");
        assert!(pp);
    }

    #[test]
    fn test_extract_codec_config_missing() {
        let metadata = create_plain_gguf_metadata();
        let result = CodecTtsStrategy::<MockInference>::extract_codec_config(&metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_codec_tts_true() {
        let metadata = create_codec_tts_metadata();
        assert!(CodecTtsStrategy::<MockInference>::is_codec_tts(&metadata));
    }

    #[test]
    fn test_is_codec_tts_false_no_codec_decode() {
        let metadata = create_plain_gguf_metadata();
        assert!(!CodecTtsStrategy::<MockInference>::is_codec_tts(&metadata));
    }

    #[test]
    fn test_is_codec_tts_false_onnx() {
        let metadata = ModelMetadata::onnx("test", "1.0", "model.onnx");
        assert!(!CodecTtsStrategy::<MockInference>::is_codec_tts(&metadata));
    }

    #[test]
    fn test_chunk_text_three_sentences() {
        let text = "First sentence. Second sentence. Third sentence.";
        // Use a small max so each sentence is its own chunk
        let chunks = CodecTtsStrategy::<MockInference>::chunk_text(text, 20);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First sentence.");
        assert_eq!(chunks[1], "Second sentence.");
        assert_eq!(chunks[2], "Third sentence.");
    }

    #[test]
    fn test_chunk_text_single_short_sentence() {
        let text = "Hello world.";
        let chunks = CodecTtsStrategy::<MockInference>::chunk_text(text, 1500);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world.");
    }

    #[test]
    fn test_chunk_text_combines_short_sentences() {
        let text = "Hi. There. Friend.";
        let chunks = CodecTtsStrategy::<MockInference>::chunk_text(text, 1500);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hi. There. Friend.");
    }

    #[test]
    fn test_chunk_text_splits_at_sentence_boundary() {
        let text = "First sentence here. Second sentence here.";
        let chunks = CodecTtsStrategy::<MockInference>::chunk_text(text, 25);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "First sentence here.");
        assert_eq!(chunks[1], "Second sentence here.");
    }

    #[test]
    fn test_chunk_text_long_sentence_splits_at_comma_or_space() {
        let text = "This is a very long sentence without any period that exceeds the limit";
        let chunks = CodecTtsStrategy::<MockInference>::chunk_text(text, 30);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_build_generation_params_defaults_to_neutts_values() {
        let metadata = create_codec_tts_metadata();
        let params = CodecTtsStrategy::<MockInference>::build_generation_params(&metadata);
        assert_eq!(params.max_tokens, 2048);
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, 50);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(
            params.stop_sequences,
            vec!["<|SPEECH_GENERATION_END|>".to_string()]
        );
    }

    #[test]
    fn test_build_generation_params_applies_metadata_overrides() {
        use crate::execution::template::GenerationParams;
        let mut metadata = create_codec_tts_metadata();
        metadata.execution_template = ExecutionTemplate::Gguf {
            model_file: "model.gguf".to_string(),
            chat_template: None,
            context_length: 2048,
            generation_params: Some(GenerationParams {
                temperature: Some(0.7),
                top_p: Some(0.85),
                stop_sequences: vec!["<|custom_stop|>".to_string()],
                ..Default::default()
            }),
        };
        let params = CodecTtsStrategy::<MockInference>::build_generation_params(&metadata);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.85);
        assert_eq!(params.top_k, 50, "unspecified fields keep defaults");
        assert_eq!(params.max_tokens, 2048, "unspecified fields keep defaults");
        assert_eq!(params.stop_sequences, vec!["<|custom_stop|>".to_string()]);
    }
}

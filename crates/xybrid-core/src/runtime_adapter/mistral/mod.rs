//! MistralBackend - LLM inference using mistral.rs
//!
//! This module provides the mistral.rs implementation of the LlmBackend trait.
//! It is feature-gated behind `llm-mistral`.
//!
//! # Note
//!
//! mistral.rs uses candle + gemm which requires `+fp16` on ARM64.
//! This causes SIGILL on Android devices without ARMv8.2-A FP16 extension.
//! For Android, use `llama_cpp` backend instead.

use crate::ir::MessageRole;
use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, LlmBackend, LlmConfig, LlmResult,
};
#[cfg(feature = "llm-mistral")]
use crate::runtime_adapter::llm_telemetry::compute_streaming_fields;
use crate::runtime_adapter::AdapterError;

#[cfg(feature = "llm-mistral")]
use mistralrs::{
    GgufModelBuilder, Model, PagedAttentionMetaBuilder, RequestBuilder, Response, TextMessageRole,
};

/// 0.0 is how mistralrs reports "below timing resolution" (e.g. very short
/// prompts on fast hardware). Treat it as `None` so dashboards don't show a
/// misleading 0 tok/s.
#[cfg(feature = "llm-mistral")]
fn nonzero(v: f32) -> Option<f32> {
    if v > 0.0 {
        Some(v)
    } else {
        None
    }
}

/// Accumulated state for the streaming response consumer. Kept pure-data so
/// [`handle_response`] can be tested sequentially without an async runtime
/// or the mistralrs `Stream<'_>` wrapper.
#[cfg(feature = "llm-mistral")]
struct StreamState {
    text: String,
    finish_reason: String,
    tokens_reported: Option<usize>,
    chunk_ts: Vec<std::time::Instant>,
    decode_tps_reported: Option<f32>,
    prefill_tps_reported: Option<f32>,
    saw_terminal: bool,
}

#[cfg(feature = "llm-mistral")]
impl StreamState {
    fn new() -> Self {
        Self {
            text: String::new(),
            finish_reason: String::from("unknown"),
            tokens_reported: None,
            chunk_ts: Vec::new(),
            decode_tps_reported: None,
            prefill_tps_reported: None,
            saw_terminal: false,
        }
    }
}

/// Pure state-machine step over a single mistralrs [`Response`]. Returns
/// `Ok(true)` when the caller should stop reading (terminal `Response::Done`
/// only — terminal `Response::Chunk` sets `saw_terminal` but doesn't stop
/// the stream, since the producer closes the channel naturally after).
///
/// Invariants documented at call site:
/// - `ts` is the moment the response was observed (not the moment the
///   function runs), so tests can inject synthetic timestamps.
/// - On `is_streaming=true`, mistralrs signals completion on the **last**
///   `Response::Chunk` by populating `usage` and `finish_reason`
///   (mistralrs-core sequence.rs:1519-1527 + sampling.rs:154,188-200).
///   The `Response::Done` arm is kept defensively for non-streaming paths
///   and any future mistralrs version that emits it.
#[cfg(feature = "llm-mistral")]
fn handle_response(
    response: Response,
    state: &mut StreamState,
    ts: std::time::Instant,
) -> Result<bool, AdapterError> {
    match response {
        Response::Chunk(chunk) => {
            state.chunk_ts.push(ts);
            if let Some(delta) = chunk
                .choices
                .first()
                .and_then(|c| c.delta.content.as_deref())
            {
                state.text.push_str(delta);
            }
            // Terminal chunk carries usage + finish_reason.
            let chunk_fr = chunk.choices.first().and_then(|c| c.finish_reason.as_ref());
            if chunk.usage.is_some() || chunk_fr.is_some() {
                state.saw_terminal = true;
                if let Some(u) = chunk.usage.as_ref() {
                    state.tokens_reported = Some(u.completion_tokens);
                    state.decode_tps_reported = nonzero(u.avg_compl_tok_per_sec);
                    state.prefill_tps_reported = nonzero(u.avg_prompt_tok_per_sec);
                }
                if let Some(fr) = chunk_fr {
                    state.finish_reason = fr.clone();
                }
            }
            Ok(false)
        }
        Response::Done(final_resp) => {
            state.saw_terminal = true;
            state.tokens_reported = Some(final_resp.usage.completion_tokens);
            state.decode_tps_reported = nonzero(final_resp.usage.avg_compl_tok_per_sec);
            state.prefill_tps_reported = nonzero(final_resp.usage.avg_prompt_tok_per_sec);
            if let Some(choice) = final_resp.choices.first() {
                state.finish_reason = choice.finish_reason.clone();
            }
            Ok(true)
        }
        Response::ModelError(msg, _partial) => {
            Err(AdapterError::InferenceFailed(format!("model: {}", msg)))
        }
        Response::InternalError(e) => {
            Err(AdapterError::InferenceFailed(format!("internal: {}", e)))
        }
        Response::ValidationError(e) => {
            Err(AdapterError::InvalidInput(format!("validation: {}", e)))
        }
        // Streaming chat path should never produce these (completion /
        // image / speech / raw / embeddings).
        _ => Err(AdapterError::InferenceFailed(
            "unexpected stream response variant for chat".to_string(),
        )),
    }
}

/// MistralBackend - LLM inference using mistral.rs.
///
/// This backend uses the mistral.rs library for pure-Rust LLM inference.
/// It supports GGUF models and provides efficient inference with features like:
/// - Paged attention for memory efficiency
/// - Metal/CUDA acceleration (via feature flags)
/// - Streaming generation (future)
///
/// # Platform Support
///
/// - **macOS/iOS**: Works with Metal acceleration
/// - **Linux/Windows**: Works with CUDA or CPU
/// - **Android**: NOT SUPPORTED - use `LlamaCppBackend` instead
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::runtime_adapter::mistral::MistralBackend;
/// use xybrid_core::runtime_adapter::llm::{LlmBackend, LlmConfig};
///
/// let mut backend = MistralBackend::new()?;
/// backend.load(&LlmConfig::new("model.gguf"))?;
/// ```
#[cfg(feature = "llm-mistral")]
pub struct MistralBackend {
    /// Loaded model (None if not loaded)
    model: Option<Model>,
    /// Current configuration
    config: Option<LlmConfig>,
    /// Context length actually in effect on the backend.
    ///
    /// The `GgufModelBuilder` in mistralrs reads context length from the
    /// GGUF file and does not expose a runtime override. Callers that set
    /// `LlmConfig::context_length` to a non-default value see that value
    /// NOT reach the backend. We report `None` in that case so callers
    /// and telemetry read an honest "unknown" rather than a value the
    /// runtime does not actually honor.
    effective_context_length: Option<usize>,
}

#[cfg(feature = "llm-mistral")]
impl MistralBackend {
    /// Create a new MistralBackend.
    pub fn new() -> LlmResult<Self> {
        Ok(Self {
            model: None,
            config: None,
            effective_context_length: None,
        })
    }

    /// Convert our MessageRole to mistral.rs TextMessageRole.
    fn convert_role(role: &MessageRole) -> TextMessageRole {
        match role {
            MessageRole::System => TextMessageRole::System,
            MessageRole::Assistant => TextMessageRole::Assistant,
            MessageRole::User => TextMessageRole::User,
        }
    }
}

#[cfg(feature = "llm-mistral")]
impl Default for MistralBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create MistralBackend")
    }
}

#[cfg(feature = "llm-mistral")]
impl LlmBackend for MistralBackend {
    fn name(&self) -> &str {
        "mistral"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["gguf"]
    }

    fn load(&mut self, config: &LlmConfig) -> LlmResult<()> {
        use std::path::Path;

        let model_path = Path::new(&config.model_path);
        if !model_path.exists() {
            return Err(AdapterError::ModelNotFound(config.model_path.clone()));
        }

        // Honest-config contract: warn on load-time fields the backend does
        // not actually wire into mistralrs, and report them as None from the
        // trait accessors rather than echoing the requested value.
        // 4096 matches `LlmConfig::default_context_length()` (types.rs).
        const DEFAULT_CONTEXT_LENGTH: usize = 4096;
        if config.context_length != DEFAULT_CONTEXT_LENGTH {
            log::warn!(
                "LlmConfig.context_length={} is not wired to the mistralrs GGUF backend; \
                 requested value ignored. Context length is derived from the GGUF file.",
                config.context_length
            );
        }
        if config.gpu_layers != 0 {
            log::warn!(
                "LlmConfig.gpu_layers={} is not wired to mistralrs; build with the \
                 appropriate feature flag (llm-mistral-metal/llm-mistral-cuda) instead.",
                config.gpu_layers
            );
        }
        self.effective_context_length = None;

        // Determine directory and filename
        let (model_dir, model_file) = if model_path.is_file() {
            let dir = model_path
                .parent()
                .ok_or_else(|| AdapterError::InvalidInput("Invalid model path".to_string()))?;
            let file = model_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| AdapterError::InvalidInput("Invalid model filename".to_string()))?;
            (dir.to_string_lossy().to_string(), file.to_string())
        } else {
            // Directory provided - look for .gguf files
            let gguf_files: Vec<_> = std::fs::read_dir(model_path)
                .map_err(AdapterError::IOError)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "gguf")
                        .unwrap_or(false)
                })
                .collect();

            if gguf_files.is_empty() {
                return Err(AdapterError::ModelNotFound(format!(
                    "No .gguf files found in {}",
                    config.model_path
                )));
            }

            let file = gguf_files[0].file_name().to_string_lossy().to_string();
            (config.model_path.clone(), file)
        };

        // Build the model using tokio runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

        let model = rt.block_on(async {
            let mut builder = GgufModelBuilder::new(&model_dir, vec![model_file]);

            // Apply chat template if provided
            if let Some(ref template) = config.chat_template {
                builder = builder.with_chat_template(template);
            }

            // Enable logging if requested
            if config.logging {
                builder = builder.with_logging();
            }

            // Enable paged attention if requested. mistralrs 0.8 takes the
            // config by value (the closure form was 0.7). `build()` still
            // returns Result, so keep the error mapping on that call.
            if config.paged_attention {
                let paged_cfg = PagedAttentionMetaBuilder::default().build().map_err(|e| {
                    AdapterError::RuntimeError(format!("Paged attention setup failed: {}", e))
                })?;
                builder = builder.with_paged_attn(paged_cfg);
            }

            builder
                .build()
                .await
                .map_err(|e| AdapterError::RuntimeError(format!("Model loading failed: {}", e)))
        })?;

        self.model = Some(model);
        self.config = Some(config.clone());

        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn unload(&mut self) -> LlmResult<()> {
        self.model = None;
        self.config = None;
        Ok(())
    }

    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        let model = self.model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load() first.".to_string())
        })?;

        // Build the request with messages
        let mut request = RequestBuilder::new();

        for msg in messages {
            request = request.add_message(Self::convert_role(&msg.role), &msg.content);
        }

        // Deterministic-by-default: when the caller has not asked for
        // sampling (`temperature <= 0.0`), use mistralrs's
        // `set_deterministic_sampler()` so local inference is reproducible.
        // `max_tokens` still applies in the deterministic path; sampling
        // knobs are only set when requested.
        request = if config.temperature <= 0.0 {
            request
                .set_deterministic_sampler()
                .set_sampler_max_len(config.max_tokens)
        } else {
            request
                .set_sampler_temperature(config.temperature as f64)
                .set_sampler_topp(config.top_p as f64)
                .set_sampler_topk(config.top_k)
                .set_sampler_max_len(config.max_tokens)
        };

        let start = std::time::Instant::now();

        // Run streaming inference. We measure per-chunk timings so we can
        // report TTFT (time to first emitted chunk) and inter-chunk latency
        // summaries.
        //
        // On `is_streaming=true`, mistralrs signals completion via the
        // **last `Response::Chunk`** (not `Response::Done`):
        //   - `ChatCompletionChunkResponse.usage: Option<Usage>` is `Some(..)`
        //     only on the terminal chunk (sequence.rs:1519-1527).
        //   - `ChunkChoice.finish_reason: Option<String>` is set only on the
        //     terminal chunk (sampling.rs:154,188-200).
        // We accept either signal (`Done` or a terminal chunk) via
        // `saw_terminal`; a stream that closes without either is an error.
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

        let state = rt.block_on(async move {
            let mut stream = model
                .stream_chat_request(request)
                .await
                .map_err(|e| AdapterError::InferenceFailed(format!("stream init: {}", e)))?;

            let mut state = StreamState::new();
            while let Some(response) = stream.next().await {
                let done = handle_response(response, &mut state, std::time::Instant::now())?;
                if done {
                    break;
                }
            }
            if !state.saw_terminal {
                return Err(AdapterError::InferenceFailed(
                    "stream closed before terminal chunk (no usage or finish_reason)".to_string(),
                ));
            }
            Ok::<_, AdapterError>(state)
        })?;

        let StreamState {
            text,
            finish_reason,
            tokens_reported,
            chunk_ts,
            decode_tps_reported,
            prefill_tps_reported,
            ..
        } = state;

        // `saw_terminal` in the async block above guarantees we had a
        // usage-bearing terminal chunk; if it was absent despite terminal
        // signals, fall back to chunk count so downstream metrics don't NaN.
        let tokens_generated = tokens_reported.unwrap_or(chunk_ts.len());

        // Shared telemetry derivation (TTFT, mean/p95 ITL, tokens_per_second).
        // `prompt_token_count = 0` because mistralrs tokenizes internally
        // and reports prefill_tps directly via `Usage.avg_prompt_tok_per_sec`
        // — we override the (always-None) derived value with the engine
        // number below via `.or()`.
        let fields = compute_streaming_fields(start, &chunk_ts, 0, tokens_generated);

        Ok(GenerationOutput {
            text,
            tokens_generated,
            generation_time_ms: fields.generation_time_ms,
            tokens_per_second: fields.tokens_per_second,
            finish_reason,
            ttft_ms: fields.ttft_ms,
            mean_itl_ms: fields.mean_itl_ms,
            p95_itl_ms: fields.p95_itl_ms,
            emitted_chunks: fields.emitted_chunks,
            inter_chunk_ms: fields.inter_chunk_ms,
            // Engine-reported values win when present; derived values
            // (decode_tps from ITL, prefill_tps = None since we passed 0
            // prompt tokens) are the fallback for any future code path.
            decode_tps: decode_tps_reported.or(fields.decode_tps),
            prefill_tps: prefill_tps_reported.or(fields.prefill_tps),
        })
    }

    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
        let messages = vec![ChatMessage::user(prompt)];
        self.generate(&messages, config)
    }

    // TODO: Implement true streaming for mistral.rs once we verify the streaming API
    // For now, uses the default implementation which falls back to non-streaming.
    // The mistral.rs streaming API (stream_chat_request) has a different response
    // structure that needs investigation.

    fn supports_streaming(&self) -> bool {
        // Return false until true streaming is implemented
        false
    }

    fn memory_usage(&self) -> Option<u64> {
        None
    }

    fn context_length(&self) -> Option<usize> {
        // Honest: we only report a context length when the backend actually
        // applies it. Today mistralrs derives context from the GGUF, which
        // we don't read, so this stays `None`. See `load()` for the contract.
        self.effective_context_length
    }
}

// =============================================================================
// Stub implementation when llm-mistral feature is not enabled
// =============================================================================

#[cfg(not(feature = "llm-mistral"))]
pub struct MistralBackend;

#[cfg(not(feature = "llm-mistral"))]
impl MistralBackend {
    pub fn new() -> LlmResult<Self> {
        Err(AdapterError::RuntimeError(
            "llm-mistral feature not enabled. Build with --features llm-mistral".to_string(),
        ))
    }
}

#[cfg(not(feature = "llm-mistral"))]
impl LlmBackend for MistralBackend {
    fn name(&self) -> &str {
        "mistral"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["gguf"]
    }

    fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> {
        Err(AdapterError::RuntimeError(
            "llm-mistral feature not enabled".to_string(),
        ))
    }

    fn is_loaded(&self) -> bool {
        false
    }

    fn unload(&mut self) -> LlmResult<()> {
        Ok(())
    }

    fn generate(
        &self,
        _messages: &[ChatMessage],
        _config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        Err(AdapterError::RuntimeError(
            "llm-mistral feature not enabled".to_string(),
        ))
    }

    fn generate_raw(
        &self,
        _prompt: &str,
        _config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        Err(AdapterError::RuntimeError(
            "llm-mistral feature not enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    // Mock-stream tests for `handle_response` / `StreamState`.
    //
    // These exercise the pure state machine that `generate()` drives
    // with real mistralrs responses. They reproduce the Codex P1
    // regression from the prior review: the terminal `Response::Chunk`
    // must populate `finish_reason` and `usage`, since mistralrs under
    // `is_streaming=true` does not emit `Response::Done`.
    //
    // Telemetry math (itl_stats, streaming field derivation) is tested
    // in `runtime_adapter::llm_telemetry` — no point repeating it here.

    #[cfg(feature = "llm-mistral")]
    mod mock_stream {
        use super::super::{handle_response, nonzero, StreamState};
        use crate::runtime_adapter::AdapterError;
        use mistralrs::{ChatCompletionChunkResponse, ChunkChoice, Delta, Response, Usage};
        use std::time::{Duration, Instant};

        fn usage(completion_tokens: usize, avg_prompt: f32, avg_compl: f32) -> Usage {
            Usage {
                completion_tokens,
                prompt_tokens: 16,
                total_tokens: 16 + completion_tokens,
                avg_tok_per_sec: (avg_prompt + avg_compl) / 2.0,
                avg_prompt_tok_per_sec: avg_prompt,
                avg_compl_tok_per_sec: avg_compl,
                total_time_sec: 1.0,
                total_prompt_time_sec: 0.1,
                total_completion_time_sec: 0.9,
            }
        }

        fn delta_chunk(content: &str) -> Response {
            Response::Chunk(ChatCompletionChunkResponse {
                id: "test".into(),
                choices: vec![ChunkChoice {
                    finish_reason: None,
                    index: 0,
                    delta: Delta {
                        content: Some(content.into()),
                        role: "assistant".into(),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                }],
                created: 0,
                model: "test-model".into(),
                system_fingerprint: String::new(),
                object: "chat.completion.chunk".into(),
                usage: None,
            })
        }

        fn terminal_chunk(
            completion_tokens: usize,
            avg_prompt: f32,
            avg_compl: f32,
            finish_reason: &str,
        ) -> Response {
            Response::Chunk(ChatCompletionChunkResponse {
                id: "test".into(),
                choices: vec![ChunkChoice {
                    finish_reason: Some(finish_reason.into()),
                    index: 0,
                    delta: Delta {
                        content: None,
                        role: "assistant".into(),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                }],
                created: 0,
                model: "test-model".into(),
                system_fingerprint: String::new(),
                object: "chat.completion.chunk".into(),
                usage: Some(usage(completion_tokens, avg_prompt, avg_compl)),
            })
        }

        /// Happy path: prefix chunks accumulate text, terminal chunk
        /// populates TTFT-comparable state, usage + finish_reason.
        #[test]
        fn happy_path_populates_all_fields() {
            let start = Instant::now();
            let mut state = StreamState::new();

            assert!(!handle_response(
                delta_chunk("Hel"),
                &mut state,
                start + Duration::from_millis(40)
            )
            .unwrap());
            assert!(!handle_response(
                delta_chunk("lo"),
                &mut state,
                start + Duration::from_millis(55)
            )
            .unwrap());
            assert!(!handle_response(
                delta_chunk(" world"),
                &mut state,
                start + Duration::from_millis(70)
            )
            .unwrap());
            // Terminal chunk (no body content, finish_reason + usage).
            assert!(!handle_response(
                terminal_chunk(3, 250.0, 120.0, "stop"),
                &mut state,
                start + Duration::from_millis(85),
            )
            .unwrap());

            assert_eq!(state.text, "Hello world");
            assert_eq!(state.chunk_ts.len(), 4);
            assert!(state.saw_terminal);
            assert_eq!(state.tokens_reported, Some(3));
            assert_eq!(state.decode_tps_reported, Some(120.0));
            assert_eq!(state.prefill_tps_reported, Some(250.0));
            assert_eq!(state.finish_reason, "stop");

            // TTFT math — first timestamp relative to start.
            let ttft = state.chunk_ts[0].duration_since(start).as_millis() as u64;
            assert_eq!(ttft, 40);
        }

        /// `avg_*_tok_per_sec == 0.0` reads as "below timing resolution"
        /// and must collapse to None at the field boundary.
        #[test]
        fn zero_tps_values_collapse_to_none() {
            let mut state = StreamState::new();
            handle_response(
                terminal_chunk(1, 0.0, 0.0, "stop"),
                &mut state,
                Instant::now(),
            )
            .unwrap();

            assert!(state.saw_terminal);
            assert_eq!(state.decode_tps_reported, None);
            assert_eq!(state.prefill_tps_reported, None);
        }

        /// If no terminal chunk is ever seen, the `saw_terminal` guard in
        /// `generate()` converts that into `AdapterError::InferenceFailed`.
        /// The `handle_response` layer's job is to leave `saw_terminal`
        /// false; the error is raised by the caller.
        #[test]
        fn stream_without_terminal_leaves_saw_terminal_false() {
            let mut state = StreamState::new();
            handle_response(delta_chunk("Partial"), &mut state, Instant::now()).unwrap();
            handle_response(delta_chunk(" response"), &mut state, Instant::now()).unwrap();

            assert!(!state.saw_terminal);
            assert_eq!(state.text, "Partial response");
            assert_eq!(state.tokens_reported, None);
            assert_eq!(state.finish_reason, "unknown");
        }

        /// `Response::Done` (not normally emitted on streaming path, kept
        /// for robustness) also populates state and returns `true` to stop.
        #[test]
        fn done_response_populates_and_stops() {
            use mistralrs::{ChatCompletionResponse, Choice, ResponseMessage};
            let mut state = StreamState::new();
            let final_resp = ChatCompletionResponse {
                id: "test".into(),
                choices: vec![Choice {
                    finish_reason: "length".into(),
                    index: 0,
                    message: ResponseMessage {
                        content: Some("done body".into()),
                        role: "assistant".into(),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                }],
                created: 0,
                model: "test-model".into(),
                system_fingerprint: String::new(),
                object: "chat.completion".into(),
                usage: usage(42, 180.0, 95.0),
            };
            let stop =
                handle_response(Response::Done(final_resp), &mut state, Instant::now()).unwrap();
            assert!(
                stop,
                "Response::Done must return true so the outer loop breaks"
            );
            assert!(state.saw_terminal);
            assert_eq!(state.tokens_reported, Some(42));
            assert_eq!(state.decode_tps_reported, Some(95.0));
            assert_eq!(state.prefill_tps_reported, Some(180.0));
            assert_eq!(state.finish_reason, "length");
        }

        /// Error variants propagate as `AdapterError`.
        #[test]
        fn error_variants_produce_adapter_error() {
            let mut state = StreamState::new();
            let err = handle_response(
                Response::ValidationError("bad request".to_string().into()),
                &mut state,
                Instant::now(),
            )
            .unwrap_err();
            assert!(matches!(err, AdapterError::InvalidInput(_)));
        }

        /// `nonzero` filter behaves as the rest of the pipeline expects.
        #[test]
        fn nonzero_filter() {
            assert_eq!(nonzero(0.0), None);
            assert_eq!(nonzero(-1.0), None);
            assert_eq!(nonzero(42.5), Some(42.5));
        }
    }
}

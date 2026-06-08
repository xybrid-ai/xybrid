//! LlamaCppBackend - LLM inference using llama.cpp
//!
//! This module provides llama.cpp bindings for LLM inference.
//! The whole module is feature-gated behind `llm-llamacpp`, which links
//! the llama.cpp runtime via `llama-cpp-sys` + `xybrid-llama`.
//!
//! # Why llama.cpp?
//!
//! llama.cpp has proper Android ARM64 support with runtime SIMD detection,
//! unlike mistral.rs/candle which require compile-time `+fp16` flags that
//! cause SIGILL on devices without ARMv8.2-A FP16 extension.
//!
//! # Architecture
//!
//! ```text
//! LlamaCppBackend (Rust)
//!     │
//!     └── llama_cpp_sys (FFI bindings)
//!             │
//!             └── llama.cpp (C/C++ library)
//!                     │
//!                     └── ggml (tensor library with runtime SIMD detection)
//! ```

// Re-export log control functions for external use. Phase 2 of the
// `llamacpp-crate-split` epic relocated them to `xybrid_llama`; we
// alias back to the historical `llama_log_*` names here so
// `crate::telemetry::events` and the `runtime_adapter::mod` re-export
// keep their existing identifiers.
mod chat;
pub use xybrid_llama::{
    get_verbosity as llama_log_get_verbosity, set_verbosity as llama_log_set_verbosity,
};

use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, LlmBackend, LlmConfig, LlmResult, PartialToken,
};
use crate::runtime_adapter::llm_telemetry::{StreamingTelemetry, StreamingTelemetryFields};
use crate::runtime_adapter::streaming_postprocess::{
    merge_stop_patterns, strip_thinking_tags, trim_partial_stop_suffix, truncate_at_first_stop,
    StreamingTextFilter, CHAT_STOP_PATTERNS, CHAT_STOP_PATTERNS_BROKEN,
};
use crate::runtime_adapter::AdapterError;
#[cfg(feature = "llm-llamacpp-vision")]
use crate::runtime_adapter::{MultimodalChatMessage, MultimodalMessagePart};
use crate::tracing as xybrid_trace;
use std::sync::Mutex;

// Backend init is idempotent through `xybrid_llama::backend_init`; the OS
// reclaims llama.cpp resources at process exit.

#[cfg(feature = "llm-llamacpp-vision")]
const MTMD_MEDIA_MARKER: &str = "<__media__>";

/// LlamaCppBackend - LLM inference using llama.cpp.
///
/// This backend uses llama.cpp for GGUF model inference with proper
/// Android ARM64 support via runtime SIMD detection.
///
/// # Platform Support
///
/// - **Android**: Full support with runtime NEON/FP16 detection
/// - **iOS**: Supported with Metal acceleration
/// - **macOS**: Supported with Metal acceleration
/// - **Linux/Windows**: Supported with CPU/CUDA
///
/// # Example
///
/// ```no_run
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
/// use xybrid_core::runtime_adapter::llm::{LlmBackend, LlmConfig};
///
/// let mut backend = LlamaCppBackend::new()?;
/// backend.load(&LlmConfig::new("model.gguf"))?;
/// # Ok(())
/// # }
/// ```
pub struct LlamaCppBackend {
    /// Pointer to loaded model (llama_model*)
    model: Option<xybrid_llama::LlamaModel>,
    /// Pointer to context (llama_context*).
    ///
    /// Wrapped in Mutex because llama_decode() mutates internal state and is
    /// not thread-safe. The LlmBackend trait requires Send + Sync, and
    /// generate() takes &self, so we need a Mutex to serialize context access.
    context: Mutex<Option<xybrid_llama::LlamaContext>>,
    /// Current configuration
    config: Option<LlmConfig>,
    /// Multi-turn KV cache reuse state. Records the tokenized prompt of
    /// the last `generate*` call so the next call can find the longest
    /// common prefix and skip re-prefilling that portion. Wrapped in
    /// `Mutex` because `generate*` are `&self` and we need cross-call
    /// mutation; lock order is always `context → kv_state` to match the
    /// natural critical section (we already hold the context lock when
    /// we mutate the cache via seq_rm).
    kv_state: Mutex<KvCacheState>,
    /// Backend-owned mtmd projector context for vision-language models.
    ///
    /// The mtmd context references the loaded llama text model, so `Drop`,
    /// `unload`, and model replacement always take this before dropping the
    /// llama context/model pair.
    #[cfg(feature = "llm-llamacpp-vision")]
    mtmd_context: Mutex<Option<MtmdContextState>>,
}

/// Cross-call state for the multi-turn KV cache reuse path. `Default::default()`
/// gives an empty cache (`n_past = 0`, `cached_tokens.is_empty()`), which
/// matches the post-load state of a fresh context.
///
/// `last_prefix_hit` records the prefix length the most recent call was able
/// to reuse — the source of truth for the future `prompt_cached_tokens`
/// telemetry field. Read it post-`generate*` to learn how many tokens were
/// served from cache.
#[derive(Default)]
struct KvCacheState {
    /// The exact token sequence currently sitting in the KV cache. Empty
    /// when the cache is unpopulated (post-load, or post full-clear).
    cached_tokens: Vec<i32>,
    /// Prefix length the most recent prepare call was able to keep from
    /// the prior cached_tokens. `None` when no `generate*` has run since
    /// load; `Some(0)` when the most recent call had no shared prefix
    /// (fully re-prefilled).
    last_prefix_hit: Option<usize>,
}

#[cfg(feature = "llm-llamacpp-vision")]
struct MtmdContextState {
    mmproj_path: String,
    _context: xybrid_llama::MtmdContext,
}

/// Bitmap payload extracted for one image part in a multimodal prompt.
///
/// `Encoded` carries container bytes that mtmd decodes itself (the original,
/// unchanged single-image path). `RawRgb` carries tightly-packed RGB pixels
/// (`width * height * 3` bytes) converted from an `ImageSource::Raw` camera
/// frame, so the raw-frame path skips a per-frame JPEG round-trip.
#[cfg(feature = "llm-llamacpp-vision")]
#[derive(Debug)]
enum MtmdPromptPayload {
    Encoded {
        bytes: Vec<u8>,
    },
    RawRgb {
        rgb: Vec<u8>,
        width: u32,
        height: u32,
    },
}

#[cfg(feature = "llm-llamacpp-vision")]
struct MtmdPromptImage {
    payload: MtmdPromptPayload,
    local_id: String,
}

#[cfg(feature = "llm-llamacpp-vision")]
impl MtmdPromptImage {
    /// Build the mtmd bitmap for this image, selecting the encoded or raw
    /// constructor based on how the source arrived.
    fn build_bitmap(
        &self,
        mtmd_context: &xybrid_llama::MtmdContext,
    ) -> LlmResult<xybrid_llama::MtmdBitmap> {
        let mut bitmap = match &self.payload {
            MtmdPromptPayload::Encoded { bytes } => {
                xybrid_llama::MtmdBitmap::from_encoded_bytes(mtmd_context, bytes)?
            }
            MtmdPromptPayload::RawRgb { rgb, width, height } => {
                xybrid_llama::MtmdBitmap::from_raw_rgb(mtmd_context, *width, *height, rgb)?
            }
        };
        bitmap.set_id(&self.local_id)?;
        Ok(bitmap)
    }
}

#[cfg(feature = "llm-llamacpp-vision")]
struct MtmdPromptInputs {
    chat_messages: Vec<ChatMessage>,
    images: Vec<MtmdPromptImage>,
}

#[cfg(feature = "llm-llamacpp-vision")]
fn mtmd_prompt_inputs_from_messages(
    messages: &[MultimodalChatMessage],
) -> LlmResult<MtmdPromptInputs> {
    let mut chat_messages = Vec::with_capacity(messages.len());
    let mut images = Vec::new();

    for message in messages {
        let content = message.marker_prompt(MTMD_MEDIA_MARKER)?;
        chat_messages.push(ChatMessage {
            role: message.role,
            content,
        });

        for part in &message.parts {
            let MultimodalMessagePart::Image(image) = part else {
                continue;
            };
            let payload = if let Some((bytes, _format)) = image.source.as_encoded() {
                // Encoded path (unchanged): mtmd decodes the container bytes.
                MtmdPromptPayload::Encoded {
                    bytes: bytes.to_vec(),
                }
            } else if let Some(raw) = image.source.as_raw() {
                // Raw path: strip stride/alpha (and convert YUV→RGB) into the
                // tightly-packed RGB layout mtmd_bitmap_init expects, skipping
                // the per-frame JPEG round-trip.
                let rgb = crate::execution::preprocessing::image::raw_image_to_packed_rgb(raw)?;
                MtmdPromptPayload::RawRgb {
                    rgb,
                    width: raw.dimensions.width,
                    height: raw.dimensions.height,
                }
            } else {
                return Err(AdapterError::InvalidInput(
                    "llama.cpp mtmd requires encoded or raw image bytes".to_string(),
                ));
            };
            images.push(MtmdPromptImage {
                payload,
                local_id: image.local_id.clone(),
            });
        }
    }

    Ok(MtmdPromptInputs {
        chat_messages,
        images,
    })
}

fn emit_filtered_partial_token(
    filter: &mut StreamingTextFilter,
    token_id: i32,
    token_text: &str,
    token_index: &mut usize,
    on_token: &mut crate::runtime_adapter::llm::StreamingCallback<'_>,
) -> Result<(), crate::runtime_adapter::llm::StreamingError> {
    if let Some(safe_text) = filter.push(token_text) {
        let partial = PartialToken::new(
            safe_text,
            *token_index,
            filter.cumulative_emitted().to_string(),
        )
        .with_token_id(token_id as i64);
        *token_index += 1;
        on_token(partial)?;
    }

    Ok(())
}

impl LlamaCppBackend {
    /// Create a new LlamaCppBackend.
    pub fn new() -> LlmResult<Self> {
        // Initialize llama.cpp backend exactly once through the safe wrapper.
        // The wrapper keeps Xybrid's log-policy hook paired with native
        // backend init while leaving the `-sys` crate policy-free.
        xybrid_llama::backend_init();

        Ok(Self {
            model: None,
            context: Mutex::new(None),
            config: None,
            kv_state: Mutex::new(KvCacheState::default()),
            #[cfg(feature = "llm-llamacpp-vision")]
            mtmd_context: Mutex::new(None),
        })
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        // Drop mtmd first, then context, then model (order matters: both mtmd
        // and llama context reference the model).
        // LlamaContext and LlamaModel implement Drop, so take() + drop handles cleanup.
        // get_mut() doesn't lock — safe because Drop has &mut self.
        #[cfg(feature = "llm-llamacpp-vision")]
        let _ = self.mtmd_context.get_mut().unwrap().take(); // drops MtmdContext
        let _ = self.context.get_mut().unwrap().take(); // drops LlamaContext
        let _ = self.model.take(); // drops LlamaModel

        // Note: We intentionally do NOT call llama_backend_free() here.
        // See BACKEND_INIT comment for rationale.
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create LlamaCppBackend")
    }
}

impl LlamaCppBackend {
    /// Acquire the model + context under the context mutex and hand both
    /// to `f`. Replaces three copies of the same five-line dance across
    /// `generate`, `generate_raw`, and `generate_streaming`. The guard
    /// is held for the duration of `f` — `LlamaContext` is non-`Sync`
    /// and `llama_decode` mutates internal state, so serialization
    /// across threads is required.
    fn with_model_and_context<R, F>(&self, f: F) -> LlmResult<R>
    where
        F: FnOnce(&xybrid_llama::LlamaModel, &xybrid_llama::LlamaContext) -> LlmResult<R>,
    {
        let model = self.model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load() first.".to_string())
        })?;
        let ctx_guard = self
            .context
            .lock()
            .map_err(|_| AdapterError::RuntimeError("Context mutex poisoned".to_string()))?;
        let context = ctx_guard.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No context. Call load() first.".to_string())
        })?;
        f(model, context)
    }

    /// Multi-turn KV cache reuse: compute the longest common prefix between
    /// the new prompt's tokens and what's already in the cache from the
    /// previous call, truncate the cache to that prefix, and return the
    /// diverged tail along with the prefix length the caller should pass
    /// as `n_past_in` to [`xybrid_llama::generate_streaming`].
    ///
    /// On a fresh context (no prior call) or when the prompts share no
    /// prefix, returns `(full_tokens, 0)` and full-clears the cache so the
    /// generate call below starts from scratch — same behaviour as before
    /// this helper existed.
    ///
    /// Safety net: when the prefilled prefix plus new tail plus
    /// `max_tokens` would overflow the context window, fall back to a
    /// full clear so the generation call doesn't trip the
    /// `n_past + n_input >= n_ctx` bail-out in the C wrapper. The simple
    /// LCP path doesn't implement context-window eviction yet — that's
    /// deliberately out of scope, the fallback keeps correctness.
    ///
    /// Updates `kv_state.cached_tokens` to reflect the post-call cache
    /// (which will be `[prefix; new_tail; generated_tokens]` after the
    /// generation runs). Records `last_prefix_hit` for the telemetry
    /// accessor [`Self::last_cached_prefix_len`].
    fn prepare_kv_cache_and_get_tail(
        &self,
        model: &xybrid_llama::LlamaModel,
        context: &xybrid_llama::LlamaContext,
        new_tokens: &[i32],
        max_new_tokens: usize,
    ) -> LlmResult<(Vec<i32>, usize)> {
        let mut state = self
            .kv_state
            .lock()
            .map_err(|_| AdapterError::RuntimeError("KV state mutex poisoned".to_string()))?;

        // Models with recurrent state — fully recurrent (Mamba, RWKV)
        // OR hybrid (LFM2 / LFM2MOE, Qwen35 / Qwen35MOE, Granite-hybrid,
        // …) — can't safely have their cache truncated by position.
        // Their recurrent layers accumulate state across positions, so
        // `llama_kv_cache_seq_rm` leaves the residual state inconsistent
        // with the new prefix length and `llama_decode` fails on the
        // diverging tail (wrapper error code -3; surfaced on LFM 2.5
        // second-turn chat).
        //
        // Gating on `has_recurrent_state` (which combines the upstream
        // `is_recurrent` and `is_hybrid` predicates) instead of
        // `is_recurrent` alone is important: LFM2 is classified hybrid,
        // not fully recurrent, so a narrower check missed it the first
        // time round. Skip prefix-reuse entirely on these models and
        // fall back to the pre-INF-99 full-clear path. The cost is
        // per-turn re-prefill of the full conversation — correct, just
        // not the prefix-reuse optimisation.
        if model.has_recurrent_state() {
            context.kv_cache_clear();
            state.cached_tokens = new_tokens.to_vec();
            state.last_prefix_hit = Some(0);
            return Ok((new_tokens.to_vec(), 0));
        }

        let n_ctx = context.n_ctx();
        let prefix_len = compute_reusable_prefix_len(&state.cached_tokens, new_tokens);

        // Safety net: if the prefilled prefix + new tail + max_new_tokens
        // would push us past the context window, the simple LCP path
        // can't help — drop the cache and let the standard "input too
        // long" check in the C wrapper produce a clear error. This also
        // covers the eviction case the issue explicitly punts on.
        let would_overflow = prefix_len
            .saturating_add(new_tokens.len() - prefix_len)
            .saturating_add(max_new_tokens)
            >= n_ctx;
        if prefix_len == 0 || would_overflow {
            context.kv_cache_clear();
            state.cached_tokens = new_tokens.to_vec();
            state.last_prefix_hit = Some(0);
            return Ok((new_tokens.to_vec(), 0));
        }

        // Truncate the cache to the prefix. seq_id = 0 because the
        // wrapper's batch.seq_id[..][0] = 0 path uses a single sequence;
        // when we add multi-sequence support the seq_id needs to flow
        // through prepare too.
        context.kv_cache_seq_rm(0, prefix_len);
        let tail = new_tokens[prefix_len..].to_vec();
        state.cached_tokens = new_tokens.to_vec();
        state.last_prefix_hit = Some(prefix_len);
        Ok((tail, prefix_len))
    }

    fn reset_kv_cache_after_failed_stream(&self, context: &xybrid_llama::LlamaContext) {
        context.kv_cache_clear();
        self.clear_cached_prefix_state();
    }

    fn clear_cached_prefix_state(&self) {
        if let Ok(mut state) = self.kv_state.lock() {
            *state = KvCacheState::default();
        }
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    fn ensure_mtmd_context_loaded_with<F>(&self, mmproj_path: &str, loader: F) -> LlmResult<bool>
    where
        F: FnOnce(&str) -> LlmResult<xybrid_llama::MtmdContext>,
    {
        let (loaded, ()) = self.with_mtmd_context_loaded_with(mmproj_path, loader, |_| Ok(()))?;
        Ok(loaded)
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    fn with_mtmd_context_loaded_with<F, G, R>(
        &self,
        mmproj_path: &str,
        loader: F,
        f: G,
    ) -> LlmResult<(bool, R)>
    where
        F: FnOnce(&str) -> LlmResult<xybrid_llama::MtmdContext>,
        G: FnOnce(&xybrid_llama::MtmdContext) -> LlmResult<R>,
    {
        let mut guard = self
            .mtmd_context
            .lock()
            .map_err(|_| AdapterError::RuntimeError("mtmd context mutex poisoned".to_string()))?;

        let mut loaded = false;
        if let Some(state) = guard.as_ref() {
            if state.mmproj_path == mmproj_path {
                let result = f(&state._context)?;
                return Ok((loaded, result));
            }
        }

        let context = loader(mmproj_path)?;
        *guard = Some(MtmdContextState {
            mmproj_path: mmproj_path.to_string(),
            _context: context,
        });
        loaded = true;

        let state = guard.as_ref().ok_or_else(|| {
            AdapterError::RuntimeError("mtmd context missing after load".to_string())
        })?;
        let result = f(&state._context)?;
        Ok((loaded, result))
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    fn with_mtmd_context_loaded<G, R>(
        &self,
        model: &xybrid_llama::LlamaModel,
        mmproj_path: &str,
        f: G,
    ) -> LlmResult<(bool, R)>
    where
        G: FnOnce(&xybrid_llama::MtmdContext) -> LlmResult<R>,
    {
        let config = self.config.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No config. Call load() first.".to_string())
        })?;
        let use_gpu = config.gpu_layers != 0;
        let warmup = false;
        let n_threads = config.n_threads;
        let flash_attn = config.flash_attn;

        self.with_mtmd_context_loaded_with(
            mmproj_path,
            |path| {
                xybrid_llama::MtmdContext::from_file(
                    path, model, use_gpu, warmup, n_threads, flash_attn,
                )
                .map_err(AdapterError::from)
            },
            f,
        )
    }

    fn tokenize_chat_prompt(
        model: &xybrid_llama::LlamaModel,
        messages: &[ChatMessage],
    ) -> LlmResult<Vec<i32>> {
        let prompt = chat::format_chat_prompt(model, messages)?;
        Ok(model.tokenize_special(&prompt, true)?)
    }

    fn tokenize_raw_prompt(model: &xybrid_llama::LlamaModel, prompt: &str) -> LlmResult<Vec<i32>> {
        Ok(model.tokenize_special(prompt, true)?)
    }

    fn prepare_generation(
        &self,
        model: &xybrid_llama::LlamaModel,
        context: &xybrid_llama::LlamaContext,
        tokens: Vec<i32>,
        config: &GenerationConfig,
        prompt_kind: PromptKind,
    ) -> LlmResult<PreparedGeneration> {
        let n_ctx = context.n_ctx();
        if tokens.len() >= n_ctx {
            return Err(AdapterError::InvalidInput(
                prompt_kind.input_too_long_message(tokens.len(), n_ctx),
            ));
        }

        let prompt_token_count = tokens.len();
        let (tail, n_past) =
            self.prepare_kv_cache_and_get_tail(model, context, &tokens, config.max_tokens)?;

        Ok(PreparedGeneration {
            prompt_token_count,
            tail,
            n_past,
        })
    }

    fn run_streaming_generation<F>(
        context: &xybrid_llama::LlamaContext,
        model: &xybrid_llama::LlamaModel,
        prepared: &PreparedGeneration,
        config: &GenerationConfig,
        stop_sequences: &[String],
        mut on_chunk: F,
    ) -> LlmResult<(Vec<i32>, bool, StreamingTelemetryFields)>
    where
        F: FnMut(
            i32,
            &str,
            &mut StreamingTelemetry,
        ) -> Result<(), crate::runtime_adapter::llm::StreamingError>,
    {
        xybrid_trace::add_metadata("tokens_in", prepared.prompt_token_count.to_string());
        let mut tel = StreamingTelemetry::new(prepared.prompt_token_count);
        let (output_tokens, stopped_by_callback) = xybrid_llama::generate_streaming(
            context,
            model,
            &prepared.tail,
            config.max_tokens,
            config.temperature,
            config.top_p,
            config.min_p,
            config.top_k,
            config.repetition_penalty,
            stop_sequences,
            |token_id, token_text| on_chunk(token_id, token_text, &mut tel),
            prepared.n_past,
        )?;
        let fields = tel.finalize(output_tokens.len());
        Ok((output_tokens, stopped_by_callback, fields))
    }
}

struct PreparedGeneration {
    prompt_token_count: usize,
    tail: Vec<i32>,
    n_past: usize,
}

enum PromptKind {
    Chat,
    Raw,
}

impl PromptKind {
    fn input_too_long_message(&self, tokens_len: usize, n_ctx: usize) -> String {
        match self {
            Self::Chat => format!(
                "Input too long: {} tokens exceeds context window of {} tokens. \
                     Reduce the prompt size or conversation history.",
                tokens_len, n_ctx
            ),
            Self::Raw => format!(
                "Input too long: {} tokens exceeds context window of {} tokens.",
                tokens_len, n_ctx
            ),
        }
    }
}

fn output_from_fields(
    text: String,
    tokens_generated: usize,
    finish_reason: String,
    fields: StreamingTelemetryFields,
) -> GenerationOutput {
    GenerationOutput {
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
        decode_tps: fields.decode_tps,
        prefill_tps: fields.prefill_tps,
        image_preprocess_ms: None,
    }
}

/// Assemble the final-cleanup stop patterns for a chat turn: the caller's
/// configured stops plus the built-in [`CHAT_STOP_PATTERNS`] and their
/// `_BROKEN` variants. Shared by `generate` and `generate_streaming` so the
/// pattern set cannot drift between the streaming and non-streaming paths.
fn chat_stop_patterns(config: &GenerationConfig) -> Vec<String> {
    let mut extras: Vec<&str> = CHAT_STOP_PATTERNS.to_vec();
    extras.extend_from_slice(CHAT_STOP_PATTERNS_BROKEN);
    merge_stop_patterns(&config.stop_sequences, &extras)
}

/// Observation-only streaming chunk handler for the non-streaming
/// `generate` / `generate_raw` paths: records per-chunk telemetry and emits
/// nothing to the caller. Shared so the two call sites can't drift.
fn record_only(
    _token_id: i32,
    _token_text: &str,
    tel: &mut StreamingTelemetry,
) -> Result<(), crate::runtime_adapter::llm::StreamingError> {
    tel.record_chunk();
    Ok(())
}

/// Longest-common-prefix length between the cached tokens and the new
/// prompt tokens, capped at `new_tokens.len() - 1` so the post-prefill
/// batch always has at least one token to feed the C decoder.
///
/// Pulled out so the LCP arithmetic is unit-testable without needing a
/// real `LlamaContext`. Behaviour:
/// - empty `new_tokens` ⇒ 0 (caller should bail before reaching the helper)
/// - empty `cached` ⇒ 0 (first call)
/// - identical sequences ⇒ `new_tokens.len() - 1` (keep last token for the C decoder)
/// - common prefix shorter than either ⇒ that prefix length
fn compute_reusable_prefix_len(cached: &[i32], new_tokens: &[i32]) -> usize {
    let max_reuse = new_tokens.len().saturating_sub(1);
    cached
        .iter()
        .zip(new_tokens.iter())
        .take(max_reuse)
        .take_while(|(a, b)| a == b)
        .count()
}

impl LlmBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama-cpp"
    }

    fn wire_label(&self) -> Option<&'static str> {
        Some("llamacpp")
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
        if let Some(vision_encoder_path) = &config.vision_encoder_path {
            let path = Path::new(vision_encoder_path);
            if !path.exists() {
                return Err(AdapterError::MissingArtifact {
                    artifact: "vision_encoder".to_string(),
                    path: vision_encoder_path.clone(),
                });
            }
        }

        // Find the GGUF file
        let gguf_path = if model_path.is_file() {
            config.model_path.clone()
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

            gguf_files[0].path().to_string_lossy().to_string()
        };

        // Load model
        let model = xybrid_llama::LlamaModel::load(&gguf_path, config.gpu_layers).map_err(|e| {
            AdapterError::RuntimeError(format!(
                "Failed to load model from {}: {}. \
                 This may indicate an unsupported GGUF architecture — \
                 check that the vendored llama.cpp version supports this model's architecture. \
                 Enable verbose logging with XYBRID_LLAMACPP_VERBOSITY=4 for C++ error details.",
                gguf_path, e
            ))
        })?;

        // Create context with thread and batch configuration
        // n_threads=0 means auto-detect in the C++ layer
        // n_batch=0 means use default (512)
        let context = xybrid_llama::LlamaContext::new(
            &model,
            config.context_length,
            config.n_threads,
            config.n_batch,
            config.flash_attn,
        )
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to create context: {}", e)))?;

        #[cfg(feature = "llm-llamacpp-vision")]
        let _ = self.mtmd_context.get_mut().unwrap().take();
        let _ = self.context.get_mut().unwrap().take();
        let _ = self.model.take();

        self.model = Some(model);
        *self.context.get_mut().unwrap() = Some(context);
        self.config = Some(config.clone());
        *self.kv_state.get_mut().unwrap() = KvCacheState::default();

        Ok(())
    }

    fn is_loaded(&self) -> bool {
        // A poisoned context lock means an interrupted `llama_decode` may have
        // left the `LlamaContext` inconsistent, so treat it as "not safely
        // loaded" rather than panicking on `unwrap`. Matches the graceful
        // `.ok()` degradation used elsewhere in this backend (e.g.
        // `last_cached_prefix_len`) instead of `into_inner()` recovery, because
        // the guarded context must not be reused once poisoned.
        self.model.is_some() && self.context.lock().map(|c| c.is_some()).unwrap_or(false)
    }

    fn unload(&mut self) -> LlmResult<()> {
        // Drop mtmd first, then context, then model (order matters).
        // LlamaContext and LlamaModel implement Drop, so take() handles cleanup.
        #[cfg(feature = "llm-llamacpp-vision")]
        let _ = self.mtmd_context.get_mut().unwrap().take();
        let _ = self.context.get_mut().unwrap().take();
        let _ = self.model.take();
        self.config = None;
        // Reset KV cache reuse state — the next load gets a fresh cache,
        // so any cached prefix from a prior model would be a use-after-free
        // waiting to happen.
        *self.kv_state.get_mut().unwrap() = KvCacheState::default();
        Ok(())
    }

    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        self.with_model_and_context(|model, context| {
            // Tokenize with special token parsing enabled — the chat template contains
            // special tokens like <|im_start|>, <end_of_turn>, etc. that must be
            // recognized as their special token IDs, not as individual characters.
            let tokens = Self::tokenize_chat_prompt(model, messages)?;
            let prepared =
                self.prepare_generation(model, context, tokens, config, PromptKind::Chat)?;

            // Per-chunk timestamps capture the streaming cadence for TTFT +
            // inter-token-latency telemetry. The closure is observation-only
            // (no external emission) — generation still returns the full
            // token vector like `llama_generate_with_stops` did. Keeps the
            // non-streaming contract of this function intact.
            // Surface prompt size on the active span BEFORE the streaming
            // loop, so cloud-fallback aborts (which short-circuit before
            // tel.finalize runs) still attach tokens_in to LocalAborted.
            // Without this the dashboard's TOKENS column shows `—` for the
            // local leg of every aborted run. Successful runs harmlessly
            // overwrite this with the same value after finalize.
            let stream_result = Self::run_streaming_generation(
                context,
                model,
                &prepared,
                config,
                &config.stop_sequences,
                record_only,
            );
            let (output_tokens, stopped_by_callback, fields) = match stream_result {
                Ok(result) => result,
                Err(err) => {
                    self.reset_kv_cache_after_failed_stream(context);
                    return Err(err);
                }
            };

            // Log generated token count and last few tokens for debugging
            log::debug!(
                target: "xybrid_core",
                "Generated {} tokens. Last 10: {:?}",
                output_tokens.len(),
                output_tokens.iter().rev().take(10).collect::<Vec<_>>()
            );

            // Decode tokens to text
            let mut text = model.detokenize(&output_tokens)?;

            // Debug: log the raw text and its bytes to understand encoding
            log::debug!(target: "xybrid_core", "LLM raw output ({} chars): {:?}", text.len(), &text[..text.len().min(200)]);
            log::debug!(target: "xybrid_core", "First 100 bytes: {:?}", text.as_bytes().iter().take(100).collect::<Vec<_>>());

            // Stop-pattern truncation + think-tag stripping live in
            // `streaming_postprocess`. The `*_BROKEN` patterns cover
            // tokenizers that split the leading `<` off a chat-template
            // marker — safe only for final-text cleanup, not streaming.
            let final_stop_patterns = chat_stop_patterns(config);
            log::debug!(target: "xybrid_core", "Searching for stop patterns: {:?}", final_stop_patterns);
            let stopped_in_text = truncate_at_first_stop(&mut text, &final_stop_patterns);
            let trimmed_partial = trim_partial_stop_suffix(&mut text, &final_stop_patterns);
            let text = strip_thinking_tags(&text).trim().to_string();
            // `stopped_by_callback` catches the C layer detecting a stop
            // before the Rust post-scan would — e.g. the user-supplied
            // stop sequences that the C layer sees first. Prior code
            // silently dropped this signal and sometimes reported
            // `length` for a clean stop.
            let finish_reason = if stopped_in_text || trimmed_partial || stopped_by_callback {
                "stop"
            } else {
                "length"
            }
            .to_string();

            // Telemetry derivation (TTFT, mean/p95 ITL, decode_tps, prefill_tps)
            // lives in `llm_telemetry::StreamingTelemetry` and is shared with
            // the mistral backend — llama.cpp's sys bindings don't expose
            // `llama_perf_context`'s `t_p_eval_ms` / `t_eval_ms`, so the
            // numbers are derived from per-chunk timestamps. See
            // `compute_streaming_fields` for formula semantics.
            Ok(output_from_fields(
                text,
                output_tokens.len(),
                finish_reason,
                fields,
            ))
        })
    }

    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
        self.with_model_and_context(|model, context| {
            // Tokenize with parse_special=true so boundary tokens like
            // <|SPEECH_GENERATION_START|>, <|TEXT_PROMPT_START|>, <|im_start|>, etc.
            // collapse to single vocab IDs instead of 8-10 subword pieces each.
            // Matches llama-cpp-python's Llama.__call__ default (special=True),
            // which is required for NeuTTS-style codec TTS models.
            let tokens = Self::tokenize_raw_prompt(model, prompt)?;
            let prepared =
                self.prepare_generation(model, context, tokens, config, PromptKind::Raw)?;

            // Use the streaming-capable API with an observation-only
            // callback so raw generation gets the same TTFT / ITL /
            // decode-tps telemetry as `generate()`. Stop handling stays
            // raw — only user-supplied sequences, no chat markers.
            // Surface prompt size on the active span BEFORE the streaming
            // loop, so cloud-fallback aborts (which short-circuit before
            // tel.finalize runs) still attach tokens_in to LocalAborted.
            // Without this the dashboard's TOKENS column shows `—` for the
            // local leg of every aborted run. Successful runs harmlessly
            // overwrite this with the same value after finalize.
            let (output_tokens, stopped_by_callback, fields) = Self::run_streaming_generation(
                context,
                model,
                &prepared,
                config,
                &config.stop_sequences,
                record_only,
            )?;

            let text = model.detokenize(&output_tokens)?;
            let text = text.trim().to_string();
            let finish_reason = if stopped_by_callback {
                "stop"
            } else {
                "length"
            }
            .to_string();

            Ok(output_from_fields(
                text,
                output_tokens.len(),
                finish_reason,
                fields,
            ))
        })
    }

    fn generate_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        on_token: crate::runtime_adapter::llm::StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        let mut on_token = on_token;

        self.with_model_and_context(|model, context| {
            // Tokenize with special token parsing — chat template contains special tokens
            let tokens = Self::tokenize_chat_prompt(model, messages)?;
            let prepared =
                self.prepare_generation(model, context, tokens, config, PromptKind::Chat)?;

            // Shared streaming state: telemetry recorder + text filter.
            // The filter owns cumulative text, think-block state, stop-pattern
            // detection, and safe-prefix buffering — this backend just feeds
            // raw chunks in and emits whatever comes out. See
            // `streaming_postprocess` for the contract.
            //
            // Stop patterns are cloned once so the filter can own them while
            // the C layer and final-text cleanup keep a reference. The
            // `_BROKEN` variants are intentionally excluded from streaming
            // (they false-positive on legitimate text) — they only run in
            // the final cleanup pass below.
            // Surface prompt size on the active span BEFORE the streaming
            // loop, so cloud-fallback aborts (which short-circuit before
            // tel.finalize runs) still attach tokens_in to LocalAborted.
            // Without this the dashboard's TOKENS column shows `—` for the
            // local leg of every aborted run. Successful runs harmlessly
            // overwrite this with the same value after finalize.
            let stop_patterns = merge_stop_patterns(&config.stop_sequences, CHAT_STOP_PATTERNS);
            let mut filter = StreamingTextFilter::new(stop_patterns.clone());
            let mut token_index = 0usize;

            let (output_tokens, stopped_by_callback, fields) = Self::run_streaming_generation(
                context,
                model,
                &prepared,
                config,
                &stop_patterns, // C layer uses these for early stop / llama_vocab_is_eog
                |token_id, token_text, tel| {
                    // Timestamp every C-layer callback, before any filtering —
                    // the stream itself is what's being measured, not the
                    // user-visible emission.
                    tel.record_chunk();

                    if let Some(safe_text) = filter.push(token_text) {
                        let partial = PartialToken::new(
                            safe_text,
                            token_index,
                            filter.cumulative_emitted().to_string(),
                        )
                        .with_token_id(token_id as i64);
                        token_index += 1;
                        on_token(partial)?;
                    }

                    Ok(())
                },
            )?;

            // Final-output cleanup: detokenize the full token vector (rather
            // than using the filter's cumulative text) as a belt-and-braces
            // guard against chunk-boundary UTF-8 edge cases, then run the
            // same truncate / trim-partial / strip-think passes used by the
            // non-streaming path. The `_BROKEN` fallback patterns are
            // included here because this is final-text only — no streaming
            // false-positive risk.
            let final_patterns = chat_stop_patterns(config);
            let mut text = model.detokenize(&output_tokens)?;
            let stopped_full = truncate_at_first_stop(&mut text, &final_patterns);
            let trimmed_partial = trim_partial_stop_suffix(&mut text, &final_patterns);
            let text = strip_thinking_tags(&text).trim().to_string();
            // `stopped_by_callback` is an independent signal from the C
            // layer that a stop sequence was hit — previously dropped.
            let finish_reason =
                if filter.is_stopped() || stopped_full || trimmed_partial || stopped_by_callback {
                    "stop".to_string()
                } else {
                    "length".to_string()
                };

            // Send final empty token with finish_reason — matches the
            // pre-refactor contract so downstream consumers see a
            // terminal signal. Guarded on `token_index > 0` to avoid
            // emitting a stray terminal chunk when nothing was ever
            // emitted (e.g. immediate stop).
            if token_index > 0 {
                let final_partial = PartialToken::new(String::new(), token_index, text.clone())
                    .with_finish_reason(&finish_reason);
                let _ = on_token(final_partial);
            }

            Ok(output_from_fields(
                text,
                output_tokens.len(),
                finish_reason,
                fields,
            ))
        })
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn memory_usage(&self) -> Option<u64> {
        // TODO: Implement via llama_get_state_size or similar
        None
    }

    fn context_length(&self) -> Option<usize> {
        self.config.as_ref().map(|c| c.context_length)
    }

    /// Surface the prefix length the most recent `generate*` reused from
    /// the persisted KV cache. `Some(0)` on a first turn or a totally
    /// divergent prompt, `None` only before any `generate*` has run.
    /// Telemetry hoists this as `prompt_cached_tokens` on inference
    /// events when the value is positive.
    fn last_cached_prefix_len(&self) -> Option<usize> {
        self.kv_state.lock().ok().and_then(|s| s.last_prefix_hit)
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    fn supports_vision(&self) -> bool {
        true
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    fn generate_multimodal(
        &self,
        messages: &[MultimodalChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        let inputs = mtmd_prompt_inputs_from_messages(messages)?;
        let mmproj_path = self
            .config
            .as_ref()
            .and_then(|config| config.vision_encoder_path.as_deref())
            .ok_or_else(|| {
                AdapterError::InvalidInput(
                    "llama.cpp vision generation requires a vision encoder artifact".to_string(),
                )
            })?
            .to_string();

        self.with_model_and_context(|model, context| {
            let prompt = chat::format_chat_prompt(model, &inputs.chat_messages)?;
            let generation_stop_patterns =
                merge_stop_patterns(&config.stop_sequences, CHAT_STOP_PATTERNS);
            let (_loaded, (output_tokens, stopped_by_callback, fields, image_preprocess_ms)) = self
                .with_mtmd_context_loaded(model, &mmproj_path, |mtmd_context| {
                    let image_preprocess_started = std::time::Instant::now();
                    let bitmaps = inputs
                        .images
                        .iter()
                        .map(|image| image.build_bitmap(mtmd_context))
                        .collect::<Result<Vec<_>, _>>()?;

                    let chunks = xybrid_llama::MtmdInputChunks::tokenize(
                        mtmd_context,
                        &prompt,
                        true,
                        true,
                        &bitmaps,
                    )?;
                    let summary = chunks.summary()?;
                    let image_preprocess_ms = if !inputs.images.is_empty() {
                        let elapsed_ms = image_preprocess_started.elapsed().as_millis().max(1);
                        let image_preprocess_ms = elapsed_ms.min(u32::MAX as u128) as u32;
                        xybrid_trace::add_metadata(
                            "image_preprocess_ms",
                            image_preprocess_ms.to_string(),
                        );
                        Some(image_preprocess_ms)
                    } else {
                        None
                    };
                    context.kv_cache_clear();
                    self.clear_cached_prefix_state();
                    xybrid_trace::add_metadata(
                        "tokens_in",
                        summary.helper_total_tokens.to_string(),
                    );
                    let n_batch = self.config.as_ref().map_or(512, |config| {
                        if config.n_batch == 0 {
                            512
                        } else {
                            config.n_batch
                        }
                    });
                    let new_n_past = xybrid_llama::mtmd_helper_eval_chunks(
                        mtmd_context,
                        context,
                        &chunks,
                        0,
                        0,
                        n_batch,
                        true,
                    )?;
                    if new_n_past < 0 {
                        return Err(AdapterError::RuntimeError(format!(
                            "mtmd helper eval returned negative n_past: {}",
                            new_n_past
                        )));
                    }

                    let mut tel = StreamingTelemetry::new(summary.helper_total_tokens);
                    let (output_tokens, stopped_by_callback) =
                        xybrid_llama::generate_from_current_logits_streaming(
                            context,
                            model,
                            config.max_tokens,
                            config.temperature,
                            config.top_p,
                            config.min_p,
                            config.top_k,
                            config.repetition_penalty,
                            &generation_stop_patterns,
                            new_n_past as usize,
                            |_token_id, _token_text| {
                                tel.record_chunk();
                                Ok(())
                            },
                        )?;
                    let fields = tel.finalize(output_tokens.len());
                    Ok((
                        output_tokens,
                        stopped_by_callback,
                        fields,
                        image_preprocess_ms,
                    ))
                })?;

            let final_stop_patterns =
                merge_stop_patterns(&generation_stop_patterns, CHAT_STOP_PATTERNS_BROKEN);
            let mut text = model.detokenize(&output_tokens)?;
            let stopped_in_text = truncate_at_first_stop(&mut text, &final_stop_patterns);
            let trimmed_partial = trim_partial_stop_suffix(&mut text, &final_stop_patterns);
            let text = strip_thinking_tags(&text).trim().to_string();
            let finish_reason = if stopped_in_text || trimmed_partial || stopped_by_callback {
                "stop"
            } else {
                "length"
            }
            .to_string();

            Ok(GenerationOutput {
                text,
                tokens_generated: output_tokens.len(),
                generation_time_ms: fields.generation_time_ms,
                tokens_per_second: fields.tokens_per_second,
                finish_reason,
                ttft_ms: fields.ttft_ms,
                mean_itl_ms: fields.mean_itl_ms,
                p95_itl_ms: fields.p95_itl_ms,
                emitted_chunks: fields.emitted_chunks,
                inter_chunk_ms: fields.inter_chunk_ms,
                decode_tps: fields.decode_tps,
                prefill_tps: fields.prefill_tps,
                image_preprocess_ms,
            })
        })
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    fn generate_multimodal_streaming(
        &self,
        messages: &[MultimodalChatMessage],
        config: &GenerationConfig,
        on_token: crate::runtime_adapter::llm::StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        let mut on_token = on_token;

        let inputs = mtmd_prompt_inputs_from_messages(messages)?;
        let mmproj_path = self
            .config
            .as_ref()
            .and_then(|config| config.vision_encoder_path.as_deref())
            .ok_or_else(|| {
                AdapterError::InvalidInput(
                    "llama.cpp vision generation requires a vision encoder artifact".to_string(),
                )
            })?
            .to_string();

        self.with_model_and_context(|model, context| {
            let prompt = chat::format_chat_prompt(model, &inputs.chat_messages)?;
            let generation_stop_patterns =
                merge_stop_patterns(&config.stop_sequences, CHAT_STOP_PATTERNS);
            let (
                _loaded,
                (
                    output_tokens,
                    stopped_by_callback,
                    fields,
                    image_preprocess_ms,
                    filter_stopped,
                    token_index,
                ),
            ) = self.with_mtmd_context_loaded(model, &mmproj_path, |mtmd_context| {
                let image_preprocess_started = std::time::Instant::now();
                let bitmaps = inputs
                    .images
                    .iter()
                    .map(|image| image.build_bitmap(mtmd_context))
                    .collect::<Result<Vec<_>, _>>()?;

                let chunks = xybrid_llama::MtmdInputChunks::tokenize(
                    mtmd_context,
                    &prompt,
                    true,
                    true,
                    &bitmaps,
                )?;
                let summary = chunks.summary()?;
                let image_preprocess_ms = if !inputs.images.is_empty() {
                    let elapsed_ms = image_preprocess_started.elapsed().as_millis().max(1);
                    let image_preprocess_ms = elapsed_ms.min(u32::MAX as u128) as u32;
                    xybrid_trace::add_metadata(
                        "image_preprocess_ms",
                        image_preprocess_ms.to_string(),
                    );
                    Some(image_preprocess_ms)
                } else {
                    None
                };

                context.kv_cache_clear();
                self.clear_cached_prefix_state();
                xybrid_trace::add_metadata("tokens_in", summary.helper_total_tokens.to_string());
                let n_batch = self.config.as_ref().map_or(512, |config| {
                    if config.n_batch == 0 {
                        512
                    } else {
                        config.n_batch
                    }
                });
                let new_n_past = xybrid_llama::mtmd_helper_eval_chunks(
                    mtmd_context,
                    context,
                    &chunks,
                    0,
                    0,
                    n_batch,
                    true,
                )?;
                if new_n_past < 0 {
                    return Err(AdapterError::RuntimeError(format!(
                        "mtmd helper eval returned negative n_past: {}",
                        new_n_past
                    )));
                }

                let mut tel = StreamingTelemetry::new(summary.helper_total_tokens);
                let mut filter = StreamingTextFilter::new(generation_stop_patterns.clone());
                let mut token_index = 0usize;
                let stream_result = xybrid_llama::generate_from_current_logits_streaming(
                    context,
                    model,
                    config.max_tokens,
                    config.temperature,
                    config.top_p,
                    config.min_p,
                    config.top_k,
                    config.repetition_penalty,
                    &generation_stop_patterns,
                    new_n_past as usize,
                    |token_id, token_text| {
                        tel.record_chunk();
                        emit_filtered_partial_token(
                            &mut filter,
                            token_id,
                            token_text,
                            &mut token_index,
                            &mut on_token,
                        )
                    },
                );
                let (output_tokens, stopped_by_callback) = match stream_result {
                    Ok(result) => result,
                    Err(err) => {
                        self.reset_kv_cache_after_failed_stream(context);
                        return Err(err.into());
                    }
                };
                let fields = tel.finalize(output_tokens.len());
                Ok((
                    output_tokens,
                    stopped_by_callback,
                    fields,
                    image_preprocess_ms,
                    filter.is_stopped(),
                    token_index,
                ))
            })?;

            let final_stop_patterns =
                merge_stop_patterns(&generation_stop_patterns, CHAT_STOP_PATTERNS_BROKEN);
            let mut text = model.detokenize(&output_tokens)?;
            let stopped_in_text = truncate_at_first_stop(&mut text, &final_stop_patterns);
            let trimmed_partial = trim_partial_stop_suffix(&mut text, &final_stop_patterns);
            let text = strip_thinking_tags(&text).trim().to_string();
            let finish_reason =
                if filter_stopped || stopped_in_text || trimmed_partial || stopped_by_callback {
                    "stop"
                } else {
                    "length"
                }
                .to_string();

            if token_index > 0 {
                let final_partial = PartialToken::new(String::new(), token_index, text.clone())
                    .with_finish_reason(&finish_reason);
                let _ = on_token(final_partial);
            }

            Ok(GenerationOutput {
                text,
                tokens_generated: output_tokens.len(),
                generation_time_ms: fields.generation_time_ms,
                tokens_per_second: fields.tokens_per_second,
                finish_reason,
                ttft_ms: fields.ttft_ms,
                mean_itl_ms: fields.mean_itl_ms,
                p95_itl_ms: fields.p95_itl_ms,
                emitted_chunks: fields.emitted_chunks,
                inter_chunk_ms: fields.inter_chunk_ms,
                decode_tps: fields.decode_tps,
                prefill_tps: fields.prefill_tps,
                image_preprocess_ms,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_stop_patterns_includes_user_stops_and_broken_variants() {
        let config = GenerationConfig {
            stop_sequences: vec!["<<END>>".to_string()],
            ..Default::default()
        };
        let patterns = chat_stop_patterns(&config);
        assert!(
            patterns.iter().any(|p| p == "<<END>>"),
            "user-supplied stop must survive the merge"
        );
        // The `_BROKEN` variants are the drift-prone set the review flagged:
        // `generate` and `generate_streaming` must both include them for
        // final-text cleanup. Guards against the two paths diverging.
        for broken in CHAT_STOP_PATTERNS_BROKEN {
            assert!(
                patterns.iter().any(|p| p == broken),
                "broken chat-marker variant {broken:?} must be present"
            );
        }
    }

    #[test]
    fn backend_reports_true_streaming_for_sdk_cancellation_gate() {
        let backend = LlamaCppBackend::new().unwrap();

        assert!(
            backend.supports_streaming(),
            "llama.cpp must stay on the true streaming path so SDK abort checks can interrupt generation"
        );
    }

    #[test]
    fn filtered_stream_callback_errors_propagate_to_native_stream() {
        let mut filter = StreamingTextFilter::new(vec![]);
        let mut token_index = 0usize;
        let mut callback: crate::runtime_adapter::llm::StreamingCallback<'_> =
            Box::new(|_| Err("user cancelled".into()));

        let err =
            emit_filtered_partial_token(&mut filter, 42, "hello", &mut token_index, &mut callback)
                .unwrap_err();

        assert_eq!(token_index, 1);
        assert!(err.to_string().contains("user cancelled"));
    }

    #[test]
    fn load_rejects_missing_vision_encoder_before_parsing_model() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"not a real gguf").unwrap();
        let missing_mmproj = dir.path().join("missing-mmproj.gguf");

        let mut backend = LlamaCppBackend::new().unwrap();
        let err = backend
            .load(
                &LlmConfig::new(model_path.to_string_lossy().to_string())
                    .with_vision_encoder(missing_mmproj.to_string_lossy().to_string()),
            )
            .unwrap_err();

        match err {
            AdapterError::MissingArtifact { artifact, path } => {
                assert_eq!(artifact, "vision_encoder");
                assert!(path.contains("missing-mmproj.gguf"));
            }
            other => panic!("expected MissingArtifact for missing mmproj, got {other:?}"),
        }
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    #[test]
    fn mtmd_context_load_is_lazy_and_reused_for_same_mmproj() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let backend = LlamaCppBackend::new().unwrap();
        let load_count = Arc::new(AtomicUsize::new(0));
        let count_for_first_load = load_count.clone();

        let first_loaded = backend
            .ensure_mtmd_context_loaded_with("/models/mmproj.gguf", move |_| {
                count_for_first_load.fetch_add(1, Ordering::SeqCst);
                Ok(xybrid_llama::MtmdContext::test_stub())
            })
            .unwrap();
        let second_loaded = backend
            .ensure_mtmd_context_loaded_with("/models/mmproj.gguf", |_| {
                panic!("same mmproj path should reuse existing mtmd context")
            })
            .unwrap();

        assert!(first_loaded);
        assert!(!second_loaded);
        assert_eq!(load_count.load(Ordering::SeqCst), 1);
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    #[test]
    fn mtmd_context_loader_exposes_reused_context_to_callers() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let backend = LlamaCppBackend::new().unwrap();
        let load_count = Arc::new(AtomicUsize::new(0));
        let closure_count = Arc::new(AtomicUsize::new(0));
        let count_for_first_load = load_count.clone();
        let count_for_first_closure = closure_count.clone();

        let (first_loaded, first_value) = backend
            .with_mtmd_context_loaded_with(
                "/models/mmproj.gguf",
                move |_| {
                    count_for_first_load.fetch_add(1, Ordering::SeqCst);
                    Ok(xybrid_llama::MtmdContext::test_stub())
                },
                move |_| {
                    count_for_first_closure.fetch_add(1, Ordering::SeqCst);
                    Ok(41)
                },
            )
            .unwrap();
        let count_for_second_closure = closure_count.clone();
        let (second_loaded, second_value) = backend
            .with_mtmd_context_loaded_with(
                "/models/mmproj.gguf",
                |_| panic!("same mmproj path should reuse existing mtmd context"),
                move |_| {
                    count_for_second_closure.fetch_add(1, Ordering::SeqCst);
                    Ok(42)
                },
            )
            .unwrap();

        assert!(first_loaded);
        assert_eq!(first_value, 41);
        assert!(!second_loaded);
        assert_eq!(second_value, 42);
        assert_eq!(load_count.load(Ordering::SeqCst), 1);
        assert_eq!(closure_count.load(Ordering::SeqCst), 2);
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    #[test]
    fn mtmd_prompt_inputs_preserve_image_order_and_marker_parity() {
        use crate::ir::{Envelope, EnvelopeKind, MessageRole};
        use crate::runtime_adapter::MultimodalChatMessage;

        fn png_image(red: u8) -> Vec<u8> {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                1,
                1,
                image::Rgb([red, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        }

        let first = Envelope::image(png_image(17), "png")
            .unwrap()
            .with_local_id("first-image");
        let second = Envelope::image(png_image(99), "png")
            .unwrap()
            .with_local_id("second-image");
        let message = Envelope::new(EnvelopeKind::MultiPart(vec![
            Envelope::new(EnvelopeKind::Text("look ".to_string())),
            first,
            Envelope::new(EnvelopeKind::Text(" compare ".to_string())),
            second,
        ]))
        .with_role(MessageRole::User);
        let messages = vec![MultimodalChatMessage::from_envelope(&message).unwrap()];

        let inputs = mtmd_prompt_inputs_from_messages(&messages).unwrap();

        assert_eq!(inputs.chat_messages.len(), 1);
        assert_eq!(inputs.chat_messages[0].role, MessageRole::User);
        assert_eq!(
            inputs.chat_messages[0].content,
            "look <__media__> compare <__media__>"
        );
        assert_eq!(inputs.images.len(), 2);
        assert_eq!(inputs.images[0].local_id, "first-image");
        assert_eq!(inputs.images[1].local_id, "second-image");
        // Encoded sources stay on the encoded payload path (byte-for-byte
        // unchanged); each carries non-empty container bytes.
        assert!(inputs.images.iter().all(|image| matches!(
            &image.payload,
            MtmdPromptPayload::Encoded { bytes } if !bytes.is_empty()
        )));
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    #[test]
    fn mtmd_prompt_inputs_route_raw_frames_to_packed_rgb() {
        use crate::ir::{Envelope, EnvelopeKind, MessageRole, PixelFormat};
        use crate::runtime_adapter::MultimodalChatMessage;

        // 2x1 raw RGBA frame -> tightly-packed RGB (alpha + nothing else stripped).
        let raw = Envelope::image_raw(
            vec![255, 0, 0, 200, 0, 128, 255, 210],
            PixelFormat::Rgba8,
            2,
            1,
            vec![crate::ir::ImagePlane {
                offset: 0,
                row_stride: 8,
                pixel_stride: 4,
                width: 2,
                height: 1,
            }],
            None,
        )
        .unwrap()
        .with_local_id("raw-frame");
        let message = Envelope::new(EnvelopeKind::MultiPart(vec![
            Envelope::new(EnvelopeKind::Text("describe ".to_string())),
            raw,
        ]))
        .with_role(MessageRole::User);
        let messages = vec![MultimodalChatMessage::from_envelope(&message).unwrap()];

        let inputs = mtmd_prompt_inputs_from_messages(&messages).unwrap();

        assert_eq!(inputs.images.len(), 1);
        assert_eq!(inputs.images[0].local_id, "raw-frame");
        match &inputs.images[0].payload {
            MtmdPromptPayload::RawRgb { rgb, width, height } => {
                assert_eq!(*width, 2);
                assert_eq!(*height, 1);
                assert_eq!(rgb.as_slice(), &[255, 0, 0, 0, 128, 255]);
            }
            other => panic!("expected packed RGB payload, got {other:?}"),
        }
    }

    #[test]
    fn failed_stream_resets_rust_kv_cache_state() {
        let backend = LlamaCppBackend::new().unwrap();
        {
            let mut state = backend.kv_state.lock().unwrap();
            state.cached_tokens = vec![1, 2, 3];
            state.last_prefix_hit = Some(2);
        }

        backend.clear_cached_prefix_state();

        let state = backend.kv_state.lock().unwrap();
        assert!(
            state.cached_tokens.is_empty(),
            "failed streaming runs must not leave reusable prompt tokens behind"
        );
        assert_eq!(
            state.last_prefix_hit, None,
            "failed streaming runs must clear prefix-hit metadata"
        );
    }

    #[test]
    fn lcp_empty_inputs_return_zero() {
        // First-call shape: nothing cached yet. The KV cache is empty so
        // there's nothing to reuse; caller falls back to a full prefill.
        assert_eq!(compute_reusable_prefix_len(&[], &[1, 2, 3]), 0);
        // Defensive: an empty new prompt would be rejected by the caller
        // before reaching here, but the helper must not panic on it.
        assert_eq!(compute_reusable_prefix_len(&[1, 2, 3], &[]), 0);
        assert_eq!(compute_reusable_prefix_len(&[], &[]), 0);
    }

    #[test]
    fn lcp_identical_prompts_keep_one_token_for_decoder() {
        // Multi-turn replay of the exact same prompt. We could in
        // principle reuse the entire cache, but the C wrapper rejects
        // an empty input slice, so the helper caps reuse at len-1 to
        // guarantee one fresh token feeds the decode loop.
        let same = vec![10, 20, 30, 40];
        assert_eq!(compute_reusable_prefix_len(&same, &same), 3);
    }

    #[test]
    fn lcp_partial_match_returns_shared_length() {
        // Typical multi-turn shape: shared system prompt + earlier turns,
        // diverging at the new user turn. The shared prefix is the part
        // worth keeping in the cache.
        let cached = vec![1, 2, 3, 4, 99, 99];
        let new = vec![1, 2, 3, 4, 50, 60, 70];
        assert_eq!(compute_reusable_prefix_len(&cached, &new), 4);
    }

    #[test]
    fn lcp_no_overlap_returns_zero() {
        // Totally divergent prompts (e.g. user starts a new conversation
        // without clearing context). The caller will full-clear the cache
        // when this returns 0.
        assert_eq!(compute_reusable_prefix_len(&[1, 2, 3], &[9, 8, 7]), 0);
    }

    #[test]
    fn lcp_caps_at_new_tokens_minus_one() {
        // Cached is longer than the new prompt and shares it entirely.
        // We still cap at len-1 so the decoder gets a fresh token to
        // process — otherwise the C wrapper bails on empty input.
        let cached = vec![1, 2, 3, 4, 5, 6];
        let new = vec![1, 2, 3];
        assert_eq!(compute_reusable_prefix_len(&cached, &new), 2);
    }

    #[test]
    fn lcp_single_token_new_prompt_returns_zero() {
        // Edge case: new prompt is one token long (rare but possible —
        // think test fixtures). max_reuse = 0 ⇒ we always re-prefill.
        assert_eq!(compute_reusable_prefix_len(&[1, 2, 3], &[1]), 0);
    }
}

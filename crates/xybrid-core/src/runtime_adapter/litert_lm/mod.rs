//! LiteRT-LM backend: `.litertlm` models through Google's LiteRT-LM runtime.
//!
//! # What it does
//!
//! Text chat, image input and tool calling, streamed or not, for models in
//! the `.litertlm` format (qualified with Gemma 4 E2B). It implements
//! [`LlmBackend`], so the executor, the SDK and every binding reach it through
//! the calls they already make; a model's format picks the runtime.
//!
//! # How it maps onto the runtime
//!
//! Every chat turn is one LiteRT-LM *conversation*: the earlier messages are
//! its starting history, the newest message is sent, and the conversation is
//! deleted when the turn ends. No state is carried from one call to the next.
//! The conversation applies the model's own prompt template, preprocesses
//! images and renders tool traffic, so this backend holds no model-specific
//! prompt code.
//!
//! What comes back is put together as the text the model emitted (the
//! `chunks` submodule), which is what the rest of xybrid works on: the shared
//! streaming filter hides tool-call blocks and applies stop sequences, and the
//! executor parses tool calls from the text exactly as it does for llama.cpp.
//! A tool call enters that text only once the runtime has parsed it, written
//! from the parsed form, so the calls a caller is given are the runtime's.
//!
//! Tool results continue through [`LlmBackend::generate_tool_continuation`]:
//! the prior calls and their results are replayed as structured messages, so
//! the model's template writes them in its own protocol.
//!
//! # Differences from the llama.cpp backend
//!
//! - **Sampling.** Greedy, or top-k then top-p at a temperature. `min_p` is
//!   not applied: the runtime has no such filter.
//! - **Raw prompts** ([`LlmBackend::generate_raw`]) are refused; the runtime's
//!   chat interface takes messages.
//! - **Grammar-constrained output** is refused.
//! - **A tool call the runtime cannot parse fails the turn.** llama.cpp
//!   output is parsed by xybrid, which skips a malformed call and keeps the
//!   text around it. Here the runtime parses, and it reports a malformed
//!   call as an error.
//! - **Accelerator.** CPU unless `XYBRID_LITERT_LM_ACCELERATOR=gpu` is set.
//!   The GPU path is not qualified: upstream has open reports of wrong output
//!   past about 2,048 tokens at its default precision, so when GPU is
//!   requested this backend asks for full-precision activations.
//! - **Context.** A `.litertlm` file has a fixed maximum context (4,096 tokens
//!   for the general Gemma 4 E2B file); a larger request is capped to it.
//! - **Accounting.** Token counts and speeds are the runtime's own. It
//!   reports no finish reason, so `length` is inferred from the counts, and
//!   no token ids. After a turn that was stopped early it has no counts, and
//!   the number of chunks stands in for the number of tokens.
//!
//! # Cancellation
//!
//! A turn stops when the token callback returns an error, as on llama.cpp:
//! generation is cancelled and the callback's error is the turn's error. The
//! callback runs for visible answer text only. While the runtime reads the
//! prompt, thinks, or writes a tool call, nothing is delivered, so a caller
//! that cancels through the callback waits for the next visible token, or for
//! the stream timeout ([`STREAM_TIMEOUT_ENV`]) if none comes.
//!
//! # The runtime library
//!
//! Opened at run time, never linked (see `xybrid-litert-lm`). Where it is
//! absent, [`LlmBackend::load`] fails with
//! [`AdapterError::UnsupportedBackendCapability`] and nothing else in the
//! build is affected. The executor loads a backend on a model's first run, so
//! through the SDK that is when the refusal arrives, not when the model's
//! directory is loaded; [`runtime_available`] asks beforehand.
//!
//! # Testing without the runtime
//!
//! With the `test-util` feature, `testing::FakeRuntime::at` puts a scripted
//! runtime behind a model path: a model loaded from that path runs on the
//! fake, and every other path still opens the real library.

mod chunks;
mod messages;

#[cfg(feature = "test-util")]
#[doc(inline)]
pub use xybrid_litert_lm::testing;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use log::{debug, warn};
use xybrid_litert_lm::{
    ActivationPrecision, Backend, ConversationConfig, Engine, EngineSettings, Library, LogLevel,
    ModelInfo, RepetitionPenalty, Sampling, TurnOptions, Usage,
};

use self::chunks::{ChunkDecoder, Piece, TOOL_CALL_CHANNEL};
use self::messages::Turn;
use crate::runtime_adapter::llm::{GenerationOutput, LlmBackend, LlmResult, ToolContinuationTurn};
use crate::runtime_adapter::llm_telemetry::compute_streaming_fields;
use crate::runtime_adapter::streaming_postprocess::{
    merge_stop_patterns, strip_and_capture_thinking_tags, trim_partial_stop_suffix,
    truncate_at_first_stop, StreamingTextFilter, CHAT_STOP_PATTERNS, CHAT_STOP_PATTERNS_BROKEN,
};
use crate::runtime_adapter::tool_call::{parse_tool_calls_with_tools, ToolCallProtocol};
use crate::runtime_adapter::types::{
    ChatMessage, GenerationConfig, LlmConfig, MultimodalChatMessage, PartialToken,
    StreamingCallback,
};
use crate::runtime_adapter::AdapterError;
use crate::tracing as xybrid_trace;

/// The backend's name, as [`LlmBackend::name`] reports it.
pub const BACKEND_NAME: &str = "litert-lm";

/// The backend hint that selects this backend
/// ([`LlmRuntimeAdapter::with_backend_hint`]). The executor passes it for
/// every `.litertlm` model, whatever the model's metadata says.
///
/// [`LlmRuntimeAdapter::with_backend_hint`]: crate::runtime_adapter::llm::LlmRuntimeAdapter::with_backend_hint
pub const BACKEND_HINT: &str = "litert-lm";

/// Names the runtime library file to open, instead of the platform's default
/// name. For desktop development and tests; a packaged app ships the library
/// where the platform loader finds it.
pub const LIBRARY_ENV: &str = xybrid_litert_lm::LIBRARY_ENV;

/// Set to `gpu` to run on the runtime's GPU backend. Not qualified.
pub const ACCELERATOR_ENV: &str = "XYBRID_LITERT_LM_ACCELERATOR";

/// Directory for the runtime's compiled-model caches. By default the runtime
/// writes them beside the model file; for Gemma 4 E2B they total about 1 GB.
pub const CACHE_DIR_ENV: &str = "XYBRID_LITERT_LM_CACHE_DIR";

/// Seconds without a chunk before a stream is treated as stalled.
pub const STREAM_TIMEOUT_ENV: &str = "XYBRID_LITERT_LM_STREAM_TIMEOUT_SECS";

/// Long enough for a full context to be read on a slow CPU before the first
/// token: a stalled stream costs the engine, so this errs on the long side.
const DEFAULT_STREAM_TIMEOUT: Duration = Duration::from_secs(600);

/// How long the runtime gets to end a stream after a cancel.
const CANCEL_GRACE: Duration = Duration::from_secs(60);

/// llama.cpp applies its repetition penalty over the last 64 tokens
/// (`llama-cpp-sys/wrapper.cpp`); the same window keeps the two backends'
/// `repetition_penalty` comparable.
const REPETITION_PENALTY_WINDOW: u32 = 64;

/// The accelerator requested through [`ACCELERATOR_ENV`].
pub(crate) fn requested_accelerator() -> Backend {
    match std::env::var(ACCELERATOR_ENV) {
        Ok(value) if value.eq_ignore_ascii_case("gpu") => Backend::Gpu,
        _ => Backend::Cpu,
    }
}

/// Check that the LiteRT-LM runtime library can be opened on this host.
///
/// Opens the library, which then stays loaded. Meant for a caller that wants
/// to know before it offers a `.litertlm` model; loading a model performs the
/// same check and reports the same reason.
///
/// # Errors
///
/// The loader's reason, as text: the library is missing, lacks a function
/// this build calls, or reports an incompatible C API version.
pub fn runtime_available() -> Result<(), String> {
    Library::open_default()
        .map(|_| ())
        .map_err(|err| err.to_string())
}

/// The runtime library a model at `model_path` runs on.
fn open_library(model_path: &Path) -> Result<Arc<Library>, xybrid_litert_lm::Error> {
    #[cfg(feature = "test-util")]
    if let Some(fake) = testing::library_for_model(model_path) {
        return Ok(fake);
    }
    #[cfg(not(feature = "test-util"))]
    let _ = model_path;
    Library::open_default()
}

fn stream_timeout() -> Duration {
    std::env::var(STREAM_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map_or(DEFAULT_STREAM_TIMEOUT, Duration::from_secs)
}

/// A loaded model.
struct Loaded {
    engine: Engine,
    info: ModelInfo,
    model_id: String,
    context_length: usize,
    reasoning: bool,
    max_image_tokens: Option<usize>,
}

/// [`LlmBackend`] over the LiteRT-LM runtime. See the [module docs](self).
pub struct LiteRtLmBackend {
    loaded: Option<Loaded>,
    /// One generation at a time per model, as the llama.cpp backend does.
    generation: Mutex<()>,
    /// How long a turn may go without a chunk ([`STREAM_TIMEOUT_ENV`]).
    stream_timeout: Duration,
    /// How long the runtime gets to end a stream after a cancel.
    cancel_grace: Duration,
}

impl std::fmt::Debug for LiteRtLmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiteRtLmBackend")
            .field("loaded", &self.loaded.as_ref().map(|l| &l.model_id))
            .finish()
    }
}

impl LiteRtLmBackend {
    /// Create an unloaded backend. The runtime library is not opened until
    /// [`LlmBackend::load`].
    ///
    /// # Errors
    ///
    /// None today; the signature matches the other backends' constructors.
    pub fn new() -> LlmResult<Self> {
        Ok(Self {
            loaded: None,
            generation: Mutex::new(()),
            stream_timeout: stream_timeout(),
            cancel_grace: CANCEL_GRACE,
        })
    }

    /// Shorter waits, for tests of a runtime that stops answering.
    #[cfg(test)]
    fn with_timeouts(mut self, stream_timeout: Duration, cancel_grace: Duration) -> Self {
        self.stream_timeout = stream_timeout;
        self.cancel_grace = cancel_grace;
        self
    }

    fn loaded(&self) -> LlmResult<&Loaded> {
        self.loaded.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load() first.".to_string())
        })
    }

    /// Refuse what the runtime cannot honour, before anything is generated.
    fn check_request(&self, loaded: &Loaded, config: &GenerationConfig) -> LlmResult<()> {
        if config.grammar.is_some() {
            return Err(unsupported(
                loaded,
                "grammar-constrained output",
                "use a GGUF build of the model for grammar or JSON-schema constraints",
            ));
        }
        if config.temperature > 0.0 && config.min_p > 0.0 {
            // Every default request carries min_p, so refusing would refuse
            // the model's own defaults. The difference is documented instead.
            debug!(
                target: "xybrid_core",
                "LiteRT-LM has no min-p filter; min_p={} is not applied",
                config.min_p
            );
        }
        Ok(())
    }

    /// Run one turn and shape its output the way the llama.cpp backend does.
    fn run_turn(
        &self,
        turn: &Turn,
        config: &GenerationConfig,
        mut on_token: Option<StreamingCallback<'_>>,
    ) -> LlmResult<GenerationOutput> {
        let loaded = self.loaded()?;
        self.check_request(loaded, config)?;
        if turn.image_count > 0 && !loaded.info.supports_vision {
            return Err(AdapterError::UnsupportedModelCapability {
                model_id: loaded.model_id.clone(),
                capability: "image input".to_string(),
                hint: "this .litertlm file has no vision encoder".to_string(),
            });
        }
        let _generation = self
            .generation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let offers_tools = !config.tools.is_empty();
        let mut conversation_config =
            ConversationConfig::new(sampling(config)).with_thinking(loaded.reasoning);
        if let Some(history) = &turn.history_json {
            conversation_config = conversation_config.with_messages_json(history.as_str());
        }
        if offers_tools {
            conversation_config = conversation_config
                .with_tools_json(messages::tools_json(&config.tools)?)
                .with_tool_call_channel(TOOL_CALL_CHANNEL);
        }
        let mut conversation = loaded
            .engine
            .create_conversation(&conversation_config)
            .map_err(|err| runtime_error(loaded, err))?;

        // Tool calls come back as text between the markers of the model's
        // protocol, which is read off the prompt its template renders.
        let protocol = if offers_tools {
            let rendered = conversation
                .render_message(&turn.message_json)
                .map_err(|err| runtime_error(loaded, err))?;
            Some(
                ToolCallProtocol::detect_from_prompt(&rendered).ok_or_else(|| {
                    AdapterError::UnsupportedModelCapability {
                        model_id: loaded.model_id.clone(),
                        capability: "tool calling".to_string(),
                        hint: "its prompt template uses a tool protocol xybrid does not parse"
                            .to_string(),
                    }
                })?,
            )
        } else {
            None
        };

        let options = TurnOptions {
            max_output_tokens: Some(u32::try_from(config.max_tokens).unwrap_or(u32::MAX)),
            repetition_penalty: (config.repetition_penalty != 1.0).then_some(RepetitionPenalty {
                penalty: config.repetition_penalty,
                window: REPETITION_PENALTY_WINDOW,
            }),
            visual_token_budget: (turn.image_count > 0)
                .then_some(loaded.max_image_tokens)
                .flatten()
                .map(|tokens| u32::try_from(tokens).unwrap_or(u32::MAX)),
        };

        let stop_patterns = merge_stop_patterns(&config.stop_sequences, CHAT_STOP_PATTERNS);
        let mut filter = StreamingTextFilter::new(stop_patterns.clone());
        if offers_tools {
            filter = filter.with_tool_call_suppression();
        }
        let decoder = ChunkDecoder::new(protocol);
        let mut raw_text = String::new();
        let mut reasoning = String::new();
        let mut chunk_times: Vec<Instant> = Vec::new();
        let mut token_index = 0usize;
        let mut failure: Option<AdapterError> = None;
        let mut stopped_on_sequence = false;
        let timeout = self.stream_timeout;

        let start = Instant::now();
        let mut stream = conversation
            .send_stream(&turn.message_json, &options)
            .map_err(|err| runtime_error(loaded, err))?;

        'stream: loop {
            let Some(chunk) = stream.next_chunk(timeout) else {
                if !stream.is_finished() {
                    failure = Some(AdapterError::InferenceFailed(format!(
                        "LiteRT-LM produced no output for {:?}",
                        timeout
                    )));
                }
                break;
            };
            if let Some(error) = chunk.error {
                failure = Some(stream_error(&error, loaded.context_length));
                break;
            }
            if let Some(json) = chunk.text {
                chunk_times.push(chunk.received_at);
                let pieces = match decoder.decode(&json) {
                    Ok(pieces) => pieces,
                    Err(err) => {
                        failure = Some(err);
                        break;
                    }
                };
                for piece in pieces {
                    match piece {
                        Piece::Reasoning(text) => {
                            // A stop sequence ends the turn wherever the
                            // model writes it, as on llama.cpp, where
                            // reasoning is part of the text the stops are
                            // matched against.
                            if push_until_stop(&mut reasoning, &text, &stop_patterns) {
                                stopped_on_sequence = true;
                                break 'stream;
                            }
                        }
                        Piece::Raw(text) => {
                            raw_text.push_str(&text);
                            let Some(safe) = filter.push(&text) else {
                                continue;
                            };
                            if let Some(on_token) = on_token.as_mut() {
                                let partial = PartialToken::new(
                                    safe,
                                    token_index,
                                    filter.cumulative_emitted().to_string(),
                                );
                                if let Err(err) = on_token(partial) {
                                    // Kept as the caller's own error so a
                                    // cloud-fallback abort stays typed.
                                    failure =
                                        Some(AdapterError::from_streaming_callback_error(err));
                                    break 'stream;
                                }
                            }
                            token_index += 1;
                        }
                    }
                }
                if filter.is_stopped() {
                    stopped_on_sequence = true;
                    break;
                }
            }
            if chunk.is_final {
                break;
            }
        }

        // Ends the stream: cancels if the model is still generating, then
        // waits until the runtime is done with it.
        let end = stream.finish(self.cancel_grace);
        if let Some(failure) = failure {
            // What stopped the turn is the turn's error, whatever teardown
            // adds: a caller's abort (a cloud fallback among them) must come
            // back as itself. A teardown that failed has already made the
            // engine refuse further work; it is logged, not returned.
            if let Err(err) = end {
                warn!(
                    target: "xybrid_core",
                    "{}: LiteRT-LM teardown failed after the turn had already failed: {err}",
                    loaded.model_id
                );
            }
            return Err(failure);
        }
        let end = end.map_err(|err| runtime_error(loaded, err))?;
        // A cancelled conversation may only be deleted, so it has no counts.
        let usage: Option<Usage> = if end.cancelled {
            None
        } else {
            conversation.usage().ok()
        };
        drop(conversation);

        let mut final_patterns: Vec<&str> = CHAT_STOP_PATTERNS.to_vec();
        final_patterns.extend_from_slice(CHAT_STOP_PATTERNS_BROKEN);
        let final_patterns = merge_stop_patterns(&config.stop_sequences, &final_patterns);
        let mut text = raw_text;
        let stopped_in_text = truncate_at_first_stop(&mut text, &final_patterns);
        trim_partial_stop_suffix(&mut text, &final_patterns);
        let (clean, tagged_reasoning) = strip_and_capture_thinking_tags(&text);
        let text = clean.trim().to_string();
        let reasoning_content = if reasoning.trim().is_empty() {
            tagged_reasoning
        } else {
            Some(reasoning.trim().to_string())
        };

        // After a cancel the chunk count stands in for the token count; the
        // runtime sends about one chunk per token.
        let tokens_generated = usage.map_or(chunk_times.len(), |usage| usage.output_tokens);
        let prompt_tokens = usage.map_or(0, |usage| usage.prompt_tokens);
        // The runtime does not say why it stopped; it ends the stream the
        // same way at the end of a turn, at the output budget and when the
        // context window is full. A turn that hit no stop sequence and used
        // its whole budget, or filled the window, was cut off. (One that
        // ended naturally on exactly its last token is misread as cut off.)
        let truncated_by_budget = !stopped_on_sequence
            && !stopped_in_text
            && usage.is_some_and(|usage| {
                usage.output_tokens >= config.max_tokens
                    || usage.prompt_tokens + usage.output_tokens >= loaded.context_length
            });

        let parsed_tool_calls = if offers_tools {
            parse_tool_calls_with_tools(&text, &config.tools)
        } else {
            Vec::new()
        };
        let finish_reason = if !parsed_tool_calls.is_empty() {
            "tool_calls"
        } else if truncated_by_budget {
            "length"
        } else {
            "stop"
        }
        .to_string();

        if let Some(on_token) = on_token.as_mut() {
            // The terminal token: how the turn ended, and its parsed calls.
            // A run that produced nothing sends none.
            if !chunk_times.is_empty() {
                let tool_block_suppressed = offers_tools && filter.saw_tool_call_block();
                let final_cumulative = if tool_block_suppressed {
                    filter.cumulative_emitted().to_string()
                } else {
                    text.clone()
                };
                let terminal = PartialToken::new(String::new(), token_index, final_cumulative)
                    .with_finish_reason(&finish_reason)
                    .with_tool_calls(parsed_tool_calls, &text);
                on_token(terminal).map_err(AdapterError::from_streaming_callback_error)?;
            }
        }

        if prompt_tokens > 0 {
            xybrid_trace::add_metadata("tokens_in", prompt_tokens.to_string());
        }
        let fields = compute_streaming_fields(start, &chunk_times, prompt_tokens, tokens_generated);
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
            // The runtime's own measurements beat estimates from chunk gaps.
            decode_tps: usage
                .and_then(|usage| usage.output_tokens_per_second)
                .map(|speed| speed as f32)
                .or(fields.decode_tps),
            prefill_tps: usage
                .and_then(|usage| usage.prompt_tokens_per_second)
                .map(|speed| speed as f32)
                .or(fields.prefill_tps),
            image_preprocess_ms: None,
            reasoning_content,
        })
    }
}

/// Append `piece` to `text`, cutting at the first stop sequence the piece
/// completes. Returns whether one was found.
fn push_until_stop(text: &mut String, piece: &str, stops: &[String]) -> bool {
    let longest = stops.iter().map(String::len).max().unwrap_or(0);
    // A stop this piece completes starts at most `longest - 1` bytes back.
    let mut from = text.len().saturating_sub(longest.saturating_sub(1));
    while !text.is_char_boundary(from) {
        from -= 1;
    }
    text.push_str(piece);
    let mut tail = text[from..].to_string();
    if truncate_at_first_stop(&mut tail, stops) {
        text.truncate(from + tail.len());
        true
    } else {
        false
    }
}

/// xybrid's sampling settings as the runtime's.
fn sampling(config: &GenerationConfig) -> Sampling {
    if config.temperature <= 0.0 {
        return Sampling::greedy();
    }
    // `top_k == 0` means "no top-k filter" in xybrid. The runtime's sampler
    // always filters, so the widest filter stands in for none.
    let top_k = if config.top_k == 0 {
        u32::MAX
    } else {
        u32::try_from(config.top_k).unwrap_or(u32::MAX)
    };
    Sampling::sampled(config.temperature, top_k, config.top_p)
}

fn model_id_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn unsupported(loaded: &Loaded, capability: &str, hint: &str) -> AdapterError {
    AdapterError::UnsupportedBackendCapability {
        model_id: loaded.model_id.clone(),
        backend: BACKEND_NAME.to_string(),
        capability: capability.to_string(),
        hint: hint.to_string(),
    }
}

fn runtime_error(loaded: &Loaded, err: xybrid_litert_lm::Error) -> AdapterError {
    map_runtime_error(&loaded.model_id, err)
}

/// An error the runtime reported through a stream chunk.
///
/// The runtime refuses a prompt that does not fit its context window with
/// `INVALID_ARGUMENT`; that is the request's fault and is reported the way
/// the llama.cpp backend reports it. It uses the same code for a tool call
/// the model wrote and it could not parse, which is the model's doing, not
/// the request's. Anything else is a failed inference.
fn stream_error(error: &str, context_length: usize) -> AdapterError {
    if error.contains("Failed to parse tool calls") {
        // The runtime's message quotes the model's whole reply; it goes to
        // the debug log, not into an error that travels.
        debug!(target: "xybrid_core", "LiteRT-LM tool-call parse failure: {error}");
        return AdapterError::InferenceFailed(
            "LiteRT-LM could not parse a tool call the model wrote".to_string(),
        );
    }
    if error.contains("Input token ids are too long") {
        return AdapterError::InvalidInput(format!(
            "Input too long: the prompt exceeds the context window of {context_length} tokens. \
             Reduce the prompt size or conversation history. (LiteRT-LM: {error})"
        ));
    }
    if error.starts_with("INVALID_ARGUMENT") {
        return AdapterError::InvalidInput(format!("LiteRT-LM: {error}"));
    }
    AdapterError::InferenceFailed(format!("LiteRT-LM: {error}"))
}

/// A runtime error as an adapter error. A missing or unusable runtime
/// library is a capability the build lacks, not a failed inference.
fn map_runtime_error(model_id: &str, err: xybrid_litert_lm::Error) -> AdapterError {
    if err.is_runtime_unavailable() {
        return AdapterError::UnsupportedBackendCapability {
            model_id: model_id.to_string(),
            backend: BACKEND_NAME.to_string(),
            capability: "the LiteRT-LM runtime library".to_string(),
            hint: format!(
                "{err}. Ship the LiteRT-LM runtime with the app, or point {LIBRARY_ENV} at it"
            ),
        };
    }
    AdapterError::InferenceFailed(format!("LiteRT-LM: {err}"))
}

impl LlmBackend for LiteRtLmBackend {
    fn name(&self) -> &str {
        BACKEND_NAME
    }

    fn wire_label(&self) -> Option<&'static str> {
        Some("litertlm")
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["litertlm"]
    }

    fn load(&mut self, config: &LlmConfig) -> LlmResult<()> {
        let model_id = model_id_from_path(&config.model_path);
        let model_path = PathBuf::from(&config.model_path);

        let library = open_library(&model_path).map_err(|err| map_runtime_error(&model_id, err))?;
        // The runtime logs every skipped model section as a warning.
        let level = if config.logging {
            LogLevel::Info
        } else {
            LogLevel::Error
        };
        if let Err(err) = library.set_min_log_level(level) {
            debug!(target: "xybrid_core", "LiteRT-LM log level not set: {err}");
        }

        let info = ModelInfo::read(&library, &model_path)
            .map_err(|err| map_runtime_error(&model_id, err))?;

        // The file's maximum is a hard limit: the runtime would silently use
        // it anyway, and reporting the larger number would let a caller plan
        // for context that is not there.
        let context_length = match info.max_context_tokens {
            Some(max) if (max as usize) < config.context_length => {
                warn!(
                    target: "xybrid_core",
                    "{model_id}: context {} exceeds the file's maximum of {max}; using {max}",
                    config.context_length
                );
                max as usize
            }
            _ => config.context_length,
        };

        let accelerator = requested_accelerator();
        let mut settings = EngineSettings::new(&model_path)
            .with_backend(accelerator)
            .with_max_num_tokens(u32::try_from(context_length).unwrap_or(u32::MAX));
        if info.supports_vision {
            settings = settings.with_vision_backend(accelerator);
        }
        if config.n_threads > 0 {
            settings = settings.with_num_threads(u32::try_from(config.n_threads).unwrap_or(1));
        }
        if accelerator == Backend::Gpu {
            // Upstream's GPU default is half precision, which is reported to
            // corrupt output past about 2,048 tokens of context.
            settings = settings.with_activation_precision(ActivationPrecision::Float32);
        }
        if let Some(dir) = std::env::var_os(CACHE_DIR_ENV).filter(|dir| !dir.is_empty()) {
            settings = settings.with_cache_dir(PathBuf::from(dir));
        }

        let engine =
            Engine::load(library, &settings).map_err(|err| map_runtime_error(&model_id, err))?;
        debug!(
            target: "xybrid_core",
            "LiteRT-LM loaded {model_id}: accelerator={}, context={context_length}, vision={}",
            accelerator.as_str(),
            info.supports_vision
        );
        self.loaded = Some(Loaded {
            engine,
            info,
            model_id,
            context_length,
            reasoning: config.reasoning,
            max_image_tokens: config.max_image_tokens,
        });
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn unload(&mut self) -> LlmResult<()> {
        self.loaded = None;
        Ok(())
    }

    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        self.run_turn(&messages::chat_turn(messages)?, config, None)
    }

    fn generate_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        on_token: StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        self.run_turn(&messages::chat_turn(messages)?, config, Some(on_token))
    }

    fn generate_raw(
        &self,
        _prompt: &str,
        _config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        Err(unsupported(
            self.loaded()?,
            "raw prompts",
            "send chat messages; the LiteRT-LM chat interface applies the model's own template",
        ))
    }

    fn render_chat_prompt(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<String> {
        let loaded = self.loaded()?;
        let turn = messages::chat_turn(messages)?;
        // Not under the generation lock: rendering runs nothing on the
        // model, and it uses a conversation of its own, which the runtime
        // allows beside one that is generating. Taking the lock would block
        // a token callback that renders a prompt for good, since the turn
        // holds it while the callback runs.
        let mut conversation_config =
            ConversationConfig::new(Sampling::greedy()).with_thinking(loaded.reasoning);
        if let Some(history) = &turn.history_json {
            conversation_config = conversation_config.with_messages_json(history.as_str());
        }
        if !config.tools.is_empty() {
            conversation_config =
                conversation_config.with_tools_json(messages::tools_json(&config.tools)?);
        }
        let mut conversation = loaded
            .engine
            .create_conversation(&conversation_config)
            .map_err(|err| runtime_error(loaded, err))?;
        conversation
            .render_message(&turn.message_json)
            .map_err(|err| runtime_error(loaded, err))
    }

    fn has_tool_continuation(&self) -> bool {
        true
    }

    fn generate_tool_continuation(
        &self,
        turn: &ToolContinuationTurn<'_>,
        config: &GenerationConfig,
        on_token: Option<StreamingCallback<'_>>,
    ) -> LlmResult<GenerationOutput> {
        let turn = messages::continuation_turn(
            turn.messages,
            turn.prior_assistant_text,
            turn.tool_responses_json,
            &config.tools,
        )?;
        self.run_turn(&turn, config, on_token)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    /// Before a model is loaded the answer is "this backend can": the
    /// executor asks before it loads. Afterwards it is whether the loaded
    /// file has a vision encoder.
    fn supports_vision(&self) -> bool {
        self.loaded
            .as_ref()
            .is_none_or(|loaded| loaded.info.supports_vision)
    }

    fn generate_multimodal(
        &self,
        messages: &[MultimodalChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        self.run_turn(&messages::multimodal_turn(messages)?, config, None)
    }

    fn generate_multimodal_streaming(
        &self,
        messages: &[MultimodalChatMessage],
        config: &GenerationConfig,
        on_token: StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        self.run_turn(
            &messages::multimodal_turn(messages)?,
            config,
            Some(on_token),
        )
    }

    fn context_length(&self) -> Option<usize> {
        self.loaded.as_ref().map(|loaded| loaded.context_length)
    }
}

#[cfg(all(test, feature = "test-util"))]
mod fake_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_is_greedy() {
        let config = GenerationConfig::greedy();
        assert_eq!(sampling(&config), Sampling::greedy());
    }

    #[test]
    fn sampled_requests_carry_temperature_top_k_and_top_p() {
        let config = GenerationConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            ..GenerationConfig::default()
        };
        assert_eq!(sampling(&config), Sampling::sampled(0.7, 50, 0.95));
    }

    #[test]
    fn a_disabled_top_k_becomes_the_widest_filter() {
        let config = GenerationConfig {
            temperature: 0.7,
            top_k: 0,
            ..GenerationConfig::default()
        };
        assert_eq!(
            sampling(&config),
            Sampling::sampled(0.7, u32::MAX, config.top_p)
        );
    }

    #[test]
    fn an_unloaded_backend_refuses_to_generate() {
        let backend = LiteRtLmBackend::new().expect("a backend");
        assert!(!backend.is_loaded());
        let err = backend
            .generate(&[ChatMessage::user("hi")], &GenerationConfig::default())
            .expect_err("nothing is loaded");
        assert!(matches!(err, AdapterError::ModelNotLoaded(_)));
    }

    #[test]
    fn a_missing_runtime_library_is_a_missing_capability() {
        let err = map_runtime_error(
            "gemma-4-E2B-it",
            xybrid_litert_lm::Library::open("/nonexistent/liblitert-lm.so")
                .expect_err("no such library"),
        );
        match err {
            AdapterError::UnsupportedBackendCapability {
                model_id,
                backend,
                hint,
                ..
            } => {
                assert_eq!(model_id, "gemma-4-E2B-it");
                assert_eq!(backend, BACKEND_NAME);
                assert!(hint.contains(xybrid_litert_lm::LIBRARY_ENV));
            }
            other => panic!("expected a missing capability, got {other:?}"),
        }
    }

    #[test]
    fn an_oversize_prompt_is_the_callers_error() {
        let err = stream_error(
            "INVALID_ARGUMENT: Input token ids are too long. Exceeding the maximum number of \
             tokens allowed: 13904 >= 4096",
            4096,
        );
        match err {
            AdapterError::InvalidInput(message) => {
                assert!(message.starts_with("Input too long"), "{message}");
                assert!(message.contains("4096"), "{message}");
                assert!(
                    message.contains("13904"),
                    "the runtime's text is kept: {message}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// The runtime reports a tool call it cannot parse with the code it uses
    /// for a bad request. The model wrote it; the caller did nothing wrong.
    #[test]
    fn a_tool_call_the_runtime_cannot_parse_is_not_the_callers_error() {
        let err = stream_error(
            "INVALID_ARGUMENT: Failed to parse tool calls from code block: call:f{\n\
             full response: the user's private text\nerror: mismatched input",
            4096,
        );
        match err {
            AdapterError::InferenceFailed(message) => {
                assert!(
                    !message.contains("private"),
                    "the model's reply stays out: {message}"
                );
            }
            other => panic!("expected InferenceFailed, got {other:?}"),
        }
    }

    #[test]
    fn a_stop_sequence_is_found_across_pieces_and_cut() {
        let stops = vec!["STOP".to_string(), "<turn|>".to_string()];
        let mut text = String::new();
        assert!(!push_until_stop(&mut text, "thinking S", &stops));
        assert!(!push_until_stop(&mut text, "TO", &stops));
        assert!(push_until_stop(&mut text, "P and more", &stops));
        assert_eq!(text, "thinking ");

        // Multi-byte text before the window's start is not split.
        let mut text = String::from("héé");
        assert!(!push_until_stop(&mut text, "é", &stops));
        assert_eq!(text, "hééé");
        // No stops, nothing to find.
        assert!(!push_until_stop(&mut text, "STOP", &[]));
    }

    #[test]
    fn other_stream_errors_keep_their_kind() {
        assert!(matches!(
            stream_error("INVALID_ARGUMENT: bad message", 4096),
            AdapterError::InvalidInput(_)
        ));
        assert!(matches!(
            stream_error("INTERNAL: something broke", 4096),
            AdapterError::InferenceFailed(_)
        ));
    }

    #[test]
    fn the_model_id_is_the_file_stem() {
        assert_eq!(
            model_id_from_path("/models/gemma-4-E2B-it.litertlm"),
            "gemma-4-E2B-it"
        );
    }
}

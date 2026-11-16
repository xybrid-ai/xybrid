//! Conversations: the runtime's chat interface.
//!
//! A conversation applies the model's own prompt template, declares tools,
//! preprocesses images and parses tool calls. Messages cross the C API as
//! JSON; this crate passes those strings through and leaves their shape to
//! the caller.

use std::ffi::{c_char, c_int, CStr};
use std::fmt;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::time::Duration;

use crate::engine::{Engine, EngineInner};
use crate::error::{Error, Result};
use crate::library::{c_string, Library};
use crate::stream::{trampoline, EndState, Shared, Stream, StreamEnd, StreamId, DEFAULT_GRACE};
use crate::sys;

/// How the next token is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: Option<i32>,
}

impl Sampling {
    /// Always take the most likely token.
    ///
    /// Expressed as top-p sampling over a single candidate, because the
    /// runtime's CPU path implements no other sampler type (see
    /// [`sys::SAMPLER_TYPE_TOP_P`]).
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: None,
        }
    }

    /// Sample at `temperature` from the `top_k` most likely tokens, cut to
    /// the smallest set whose probability reaches `top_p`.
    pub fn sampled(temperature: f32, top_k: u32, top_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            seed: None,
        }
    }

    /// Seed the random generator, making a sampled run repeatable.
    #[must_use]
    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// What a [`Conversation`] is created with.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationConfig {
    messages_json: Option<String>,
    tools_json: Option<String>,
    sampling: Sampling,
    thinking: bool,
    tool_call_channel: Option<String>,
}

impl ConversationConfig {
    /// A conversation with no history and no tools, sampled as `sampling`
    /// says, with thinking off.
    pub fn new(sampling: Sampling) -> Self {
        Self {
            messages_json: None,
            tools_json: None,
            sampling,
            thinking: false,
            tool_call_channel: None,
        }
    }

    /// Start from earlier messages: a JSON array, system message first if
    /// there is one.
    #[must_use]
    pub fn with_messages_json(mut self, messages_json: impl Into<String>) -> Self {
        self.messages_json = Some(messages_json.into());
        self
    }

    /// Declare the tools the model may call: a JSON array.
    #[must_use]
    pub fn with_tools_json(mut self, tools_json: impl Into<String>) -> Self {
        self.tools_json = Some(tools_json.into());
        self
    }

    /// Let the model think before it answers, for models that can.
    #[must_use]
    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    /// Also stream the raw text of each tool call, under the named channel
    /// of the chunk's `channels` object, before the parsed call arrives.
    #[must_use]
    pub fn with_tool_call_channel(mut self, name: impl Into<String>) -> Self {
        self.tool_call_channel = Some(name.into());
        self
    }
}

/// A penalty on tokens the model produced recently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepetitionPenalty {
    /// Values above 1 make repeats less likely; 1 changes nothing.
    pub penalty: f32,
    /// How many of the latest output tokens are considered.
    pub window: u32,
}

/// Per-turn options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TurnOptions {
    /// Stop after this many output tokens.
    pub max_output_tokens: Option<u32>,
    /// Penalize recent tokens.
    pub repetition_penalty: Option<RepetitionPenalty>,
    /// Tokens an image may take, for models with variable image resolution.
    pub visual_token_budget: Option<u32>,
}

/// Token counts and speeds of a conversation, as the runtime measured them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Usage {
    /// Tokens read as input since the conversation was created.
    pub prompt_tokens: usize,
    /// Tokens generated since the conversation was created.
    pub output_tokens: usize,
    /// Input speed of the latest turn, tokens per second.
    pub prompt_tokens_per_second: Option<f64>,
    /// Output speed of the latest turn, tokens per second.
    pub output_tokens_per_second: Option<f64>,
}

/// A C object owned for the duration of a call sequence.
struct Owned<T> {
    raw: *mut T,
    delete: unsafe extern "C" fn(*mut T),
}

impl<T> Owned<T> {
    fn create(
        library: &Library,
        call: &'static str,
        create: unsafe extern "C" fn(*mut *mut T) -> sys::Status,
        delete: unsafe extern "C" fn(*mut T),
    ) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: a valid out-pointer, as every `*_create` function requires.
        let status = unsafe { create(&mut raw) };
        library.check(call, status)?;
        Ok(Self { raw, delete })
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        // SAFETY: created by the matching `*_create`, deleted exactly once.
        // The destructors are documented as NULL-safe.
        unsafe { (self.delete)(self.raw) };
    }
}

fn clamp_c_int(value: u32) -> c_int {
    c_int::try_from(value).unwrap_or(c_int::MAX)
}

impl Engine {
    /// Create a conversation on this engine.
    ///
    /// # Errors
    ///
    /// [`Error::EngineUnusable`] after a stalled stream, and [`Error::Call`]
    /// when the runtime rejects the configuration.
    pub fn create_conversation(&self, config: &ConversationConfig) -> Result<Conversation> {
        let inner = self.inner();
        let library = &inner.library;
        let api = library.api();

        let sampler = SamplerParams::create(library, &config.sampling)?;
        let session_config = Owned::create(
            library,
            "litert_lm_session_config_create",
            api.session_config_create,
            api.session_config_delete,
        )?;
        // SAFETY: both handles are live; the callee copies the parameters.
        let status =
            unsafe { (api.session_config_set_sampler_params)(session_config.raw, sampler.raw) };
        library.check("litert_lm_session_config_set_sampler_params", status)?;

        let conversation_config = Owned::create(
            library,
            "litert_lm_conversation_config_create",
            api.conversation_config_create,
            api.conversation_config_delete,
        )?;
        // SAFETY: both handles are live; the callee copies the session config.
        let status = unsafe {
            (api.conversation_config_set_session_config)(
                conversation_config.raw,
                session_config.raw,
            )
        };
        library.check("litert_lm_conversation_config_set_session_config", status)?;

        if let Some(messages) = &config.messages_json {
            let messages = c_string("messages", messages)?;
            // SAFETY: live handle; NUL-terminated string, copied by the callee.
            let status = unsafe {
                (api.conversation_config_set_messages)(conversation_config.raw, messages.as_ptr())
            };
            library.check("litert_lm_conversation_config_set_messages", status)?;
        }
        if let Some(tools) = &config.tools_json {
            let tools = c_string("tools", tools)?;
            // SAFETY: as above.
            let status = unsafe {
                (api.conversation_config_set_tools)(conversation_config.raw, tools.as_ptr())
            };
            library.check("litert_lm_conversation_config_set_tools", status)?;
        }

        // Always set explicitly: upstream's default is thinking on, and
        // thinking spends the output budget.
        let thinking = Owned::create(
            library,
            "litert_lm_thinking_config_create",
            api.thinking_config_create,
            api.thinking_config_delete,
        )?;
        // SAFETY: live handle; plain value.
        let status =
            unsafe { (api.thinking_config_set_enable_thinking)(thinking.raw, config.thinking) };
        library.check("litert_lm_thinking_config_set_enable_thinking", status)?;
        // SAFETY: both handles are live; the callee copies the config.
        let status = unsafe {
            (api.conversation_config_set_thinking_config)(conversation_config.raw, thinking.raw)
        };
        library.check("litert_lm_conversation_config_set_thinking_config", status)?;

        if let Some(channel) = &config.tool_call_channel {
            let channel = c_string("tool-call channel", channel)?;
            // SAFETY: live handle; NUL-terminated string, copied by the callee.
            let status = unsafe {
                (api.conversation_config_set_stream_tool_calls)(
                    conversation_config.raw,
                    true,
                    channel.as_ptr(),
                )
            };
            library.check(
                "litert_lm_conversation_config_set_stream_tool_calls",
                status,
            )?;
        }

        let mut raw = ptr::null_mut();
        {
            let engine = inner.lock_handle()?;
            // SAFETY: the engine handle is held exclusively; the config is
            // live; valid out-pointer.
            let status =
                unsafe { (api.conversation_create)(engine.raw, conversation_config.raw, &mut raw) };
            library.check("litert_lm_conversation_create", status)?;
        }
        let raw = NonNull::new(raw).ok_or(Error::Protocol(
            "conversation_create succeeded with a null conversation",
        ))?;

        Ok(Conversation {
            raw,
            engine: Arc::clone(inner),
            spent: false,
            abandoned: false,
            active: None,
        })
    }
}

/// RAII handle for `LiteRtLmSamplerParams`.
struct SamplerParams<'l> {
    library: &'l Library,
    raw: *mut sys::SamplerParams,
}

impl<'l> SamplerParams<'l> {
    fn create(library: &'l Library, sampling: &Sampling) -> Result<Self> {
        let api = library.api();
        let mut raw = ptr::null_mut();
        // SAFETY: declared enum value and a valid out-pointer.
        let status = unsafe { (api.sampler_params_create)(sys::SAMPLER_TYPE_TOP_P, &mut raw) };
        library.check("litert_lm_sampler_params_create", status)?;
        let params = Self { library, raw };

        let top_k = i32::try_from(sampling.top_k).unwrap_or(i32::MAX);
        // SAFETY: live handle; plain values. The same holds for the calls
        // below.
        let status = unsafe { (api.sampler_params_set_top_k)(params.raw, top_k) };
        library.check("litert_lm_sampler_params_set_top_k", status)?;
        let status = unsafe { (api.sampler_params_set_top_p)(params.raw, sampling.top_p) };
        library.check("litert_lm_sampler_params_set_top_p", status)?;
        let status =
            unsafe { (api.sampler_params_set_temperature)(params.raw, sampling.temperature) };
        library.check("litert_lm_sampler_params_set_temperature", status)?;
        if let Some(seed) = sampling.seed {
            let status = unsafe { (api.sampler_params_set_seed)(params.raw, seed) };
            library.check("litert_lm_sampler_params_set_seed", status)?;
        }
        Ok(params)
    }
}

impl Drop for SamplerParams<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `sampler_params_create`, deleted exactly once.
        unsafe { (self.library.api().sampler_params_delete)(self.raw) };
    }
}

/// The stream a conversation is running, from the start call until the
/// runtime is known to be done with it.
struct ActiveStream {
    id: StreamId,
    shared: Arc<Shared>,
    cancel_issued: bool,
}

/// One chat with a loaded model.
///
/// Holds its engine alive. Dropping a conversation tears down a stream that
/// is still running on it, then deletes it.
pub struct Conversation {
    raw: NonNull<sys::Conversation>,
    engine: Arc<EngineInner>,
    /// A cancel was requested, or a stream ended with an error. Upstream:
    /// the conversation "MUST NOT be reused afterwards; only delete it".
    spent: bool,
    /// The runtime could not be shown to be done with it. It must never be
    /// touched or freed again.
    abandoned: bool,
    /// The running stream. Owned here, not by the [`Stream`] handle, so that
    /// teardown does not depend on that handle being dropped.
    active: Option<ActiveStream>,
}

// SAFETY: a conversation handle may be used from any thread as long as calls
// on it do not overlap (upstream's threading model). Every method that uses
// the handle takes `&mut self`, so safe code cannot overlap them.
unsafe impl Send for Conversation {}

impl fmt::Debug for Conversation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Conversation")
            .field("spent", &self.spent)
            .field("abandoned", &self.abandoned)
            .field("streaming", &self.active.is_some())
            .finish()
    }
}

impl Conversation {
    fn library(&self) -> &Library {
        &self.engine.library
    }

    /// Whether a new call may enter the runtime through this conversation.
    fn ensure_reusable(&self) -> Result<()> {
        if self.abandoned || self.engine.is_unusable() {
            // Also when another conversation made the engine unusable: from
            // then on nothing enters the runtime through any of them.
            Err(Error::EngineUnusable)
        } else if self.active.is_some() {
            // Its handle was lost without being finished or dropped.
            Err(Error::Protocol(
                "a stream is still running on this conversation",
            ))
        } else if self.spent {
            Err(Error::Protocol(
                "a cancelled or failed conversation cannot be reused",
            ))
        } else {
            Ok(())
        }
    }

    /// The prompt the runtime would build for `message_json`: everything the
    /// conversation holds so far, the message, and the opening of the
    /// model's reply. Nothing is generated.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] when the message is not valid JSON or the template
    /// rejects it.
    pub fn render_message(&mut self, message_json: &str) -> Result<String> {
        self.ensure_reusable()?;
        let message = c_string("message", message_json)?;
        let mut text: *const c_char = ptr::null();
        // SAFETY: live handle, used exclusively; NUL-terminated string; valid
        // out-pointer.
        let status = unsafe {
            (self.library().api().conversation_render_message_to_string)(
                self.raw.as_ptr(),
                message.as_ptr(),
                &mut text,
            )
        };
        self.library()
            .check("litert_lm_conversation_render_message_to_string", status)?;
        if text.is_null() {
            return Err(Error::Protocol("render succeeded with a null string"));
        }
        // SAFETY: non-null and NUL-terminated; owned by the conversation and
        // valid until its next render, which cannot happen before this copy.
        Ok(unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned())
    }

    /// Send `message_json` and stream the reply.
    ///
    /// Each chunk's text is one JSON message carrying the new part of the
    /// reply. The stream must be finished ([`Stream::finish`]) or dropped
    /// before the conversation is used again; until then the conversation
    /// refuses other calls.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] when the stream cannot start. Failures after the
    /// start arrive as a final chunk with an error.
    pub fn send_stream(&mut self, message_json: &str, options: &TurnOptions) -> Result<Stream<'_>> {
        self.ensure_reusable()?;
        let message = c_string("message", message_json)?;
        let args = self.optional_args(options)?;

        // Chunks are at most one per output token, plus the tool-call and
        // final chunks. Anything far beyond that is a runtime fault.
        let capacity = options
            .max_output_tokens
            .map_or(1 << 16, |tokens| (tokens as usize).saturating_mul(4) + 1024);
        let shared = Shared::new(self.library().api(), capacity);
        let stream_id = StreamId::register(&shared);

        // SAFETY: live handle, used exclusively; NUL-terminated message; a
        // live or null options handle; `trampoline` matches the callback
        // type and is given the callback data it expects.
        let status = unsafe {
            (self.library().api().conversation_send_message_stream)(
                self.raw.as_ptr(),
                message.as_ptr(),
                ptr::null(),
                args.as_ref()
                    .map_or(ptr::null(), |args| args.raw.cast_const()),
                trampoline,
                stream_id.as_callback_data(),
            )
        };
        if let Err(err) = self
            .library()
            .check("litert_lm_conversation_send_message_stream", status)
        {
            // Nothing will read this stream; a callback the runtime makes
            // anyway finds it gone.
            stream_id.retire();
            return Err(err);
        }
        self.active = Some(ActiveStream {
            id: stream_id,
            shared: Arc::clone(&shared),
            cancel_issued: false,
        });
        Ok(Stream::new(self, shared))
    }

    fn optional_args(&self, options: &TurnOptions) -> Result<Option<Owned<sys::OptionalArgs>>> {
        if *options == TurnOptions::default() {
            return Ok(None);
        }
        let library = self.library();
        let api = library.api();
        let args = Owned::create(
            library,
            "litert_lm_conversation_optional_args_create",
            api.conversation_optional_args_create,
            api.conversation_optional_args_delete,
        )?;
        if let Some(tokens) = options.max_output_tokens {
            // SAFETY: live handle; plain value.
            let status = unsafe {
                (api.conversation_optional_args_set_max_output_tokens)(
                    args.raw,
                    clamp_c_int(tokens),
                )
            };
            library.check(
                "litert_lm_conversation_optional_args_set_max_output_tokens",
                status,
            )?;
        }
        if let Some(budget) = options.visual_token_budget {
            // SAFETY: live handle; plain value.
            let status = unsafe {
                (api.conversation_optional_args_set_visual_token_budget)(
                    args.raw,
                    clamp_c_int(budget),
                )
            };
            library.check(
                "litert_lm_conversation_optional_args_set_visual_token_budget",
                status,
            )?;
        }
        if let Some(penalty) = options.repetition_penalty {
            let config = Owned::create(
                library,
                "litert_lm_repetition_penalty_config_create",
                api.repetition_penalty_config_create,
                api.repetition_penalty_config_delete,
            )?;
            // SAFETY: live handle; plain values. The same holds below.
            let status = unsafe {
                (api.repetition_penalty_config_set_repetition_penalty)(config.raw, penalty.penalty)
            };
            library.check(
                "litert_lm_repetition_penalty_config_set_repetition_penalty",
                status,
            )?;
            let status = unsafe {
                (api.repetition_penalty_config_set_window_size)(
                    config.raw,
                    clamp_c_int(penalty.window),
                )
            };
            library.check(
                "litert_lm_repetition_penalty_config_set_window_size",
                status,
            )?;
            // SAFETY: both handles are live; the callee deep-copies the
            // config.
            let status = unsafe {
                (api.conversation_optional_args_set_repetition_penalty_config)(args.raw, config.raw)
            };
            library.check(
                "litert_lm_conversation_optional_args_set_repetition_penalty_config",
                status,
            )?;
        }
        Ok(Some(args))
    }

    /// Token counts and speeds the runtime recorded for this conversation.
    ///
    /// # Errors
    ///
    /// [`Error::Protocol`] after a cancel or a failed stream (such a
    /// conversation may only be deleted) and while a stream is running, and
    /// [`Error::Call`] when the runtime has no record.
    pub fn usage(&mut self) -> Result<Usage> {
        self.ensure_reusable()?;
        let library = self.library();
        let api = library.api();
        let mut raw = ptr::null_mut();
        // SAFETY: live handle, used exclusively; valid out-pointer.
        let status = unsafe { (api.conversation_get_benchmark_info)(self.raw.as_ptr(), &mut raw) };
        library.check("litert_lm_conversation_get_benchmark_info", status)?;
        let info = Owned {
            raw,
            delete: api.benchmark_info_delete,
        };

        let mut usage = Usage::default();
        let mut turns = 0usize;
        // SAFETY: `info.raw` is live until `info` drops; valid out-pointers.
        // The same holds for every accessor below.
        let status = unsafe { (api.benchmark_info_get_num_prefill_turns)(info.raw, &mut turns) };
        library.check("litert_lm_benchmark_info_get_num_prefill_turns", status)?;
        for index in 0..turns {
            let mut count = 0usize;
            let status = unsafe {
                (api.benchmark_info_get_prefill_token_count_at)(info.raw, index, &mut count)
            };
            library.check(
                "litert_lm_benchmark_info_get_prefill_token_count_at",
                status,
            )?;
            usage.prompt_tokens += count;
        }
        if let Some(last) = turns.checked_sub(1) {
            let mut speed = 0f64;
            let status = unsafe {
                (api.benchmark_info_get_prefill_tokens_per_sec_at)(info.raw, last, &mut speed)
            };
            if status == sys::STATUS_OK && speed.is_finite() && speed > 0.0 {
                usage.prompt_tokens_per_second = Some(speed);
            }
        }

        let mut turns = 0usize;
        let status = unsafe { (api.benchmark_info_get_num_decode_turns)(info.raw, &mut turns) };
        library.check("litert_lm_benchmark_info_get_num_decode_turns", status)?;
        for index in 0..turns {
            let mut count = 0usize;
            let status = unsafe {
                (api.benchmark_info_get_decode_token_count_at)(info.raw, index, &mut count)
            };
            library.check("litert_lm_benchmark_info_get_decode_token_count_at", status)?;
            usage.output_tokens += count;
        }
        if let Some(last) = turns.checked_sub(1) {
            let mut speed = 0f64;
            let status = unsafe {
                (api.benchmark_info_get_decode_tokens_per_sec_at)(info.raw, last, &mut speed)
            };
            if status == sys::STATUS_OK && speed.is_finite() && speed > 0.0 {
                usage.output_tokens_per_second = Some(speed);
            }
        }
        Ok(usage)
    }

    /// Ask the runtime to stop the running stream. Only the first request
    /// per stream reaches it.
    pub(crate) fn cancel_stream(&mut self) -> Result<()> {
        if self.engine.is_unusable() {
            return Err(Error::EngineUnusable);
        }
        let Some(active) = self.active.as_mut() else {
            return Ok(());
        };
        if active.cancel_issued {
            return Ok(());
        }
        active.cancel_issued = true;
        self.spent = true;
        // SAFETY: live handle. Cancelling while the stream's callbacks run on
        // the runtime's threads is what this function is for.
        let status =
            unsafe { (self.engine.library.api().conversation_cancel_process)(self.raw.as_ptr()) };
        self.engine
            .library
            .check("litert_lm_conversation_cancel_process", status)
    }

    /// End the running stream and bring the runtime to rest. See
    /// [`Stream::finish`], which this implements. Does nothing when no
    /// stream is running.
    pub(crate) fn finish_stream(&mut self, grace: Duration) -> Result<StreamEnd> {
        let Some(active) = self.active.as_ref() else {
            return Ok(StreamEnd { cancelled: false });
        };
        if self.engine.is_unusable() {
            // Another conversation could not be brought to rest. Nothing more
            // enters the runtime through this engine, this teardown included.
            self.abandon();
            return Err(Error::EngineUnusable);
        }
        let shared = Arc::clone(&active.shared);
        let state = shared.end_state();
        if state == EndState::Clean && !active.cancel_issued {
            // The runtime ended the turn itself. Wait for it to settle, so
            // that its counters can be read and the next turn starts from a
            // conversation at rest.
            // SAFETY: live handle, used exclusively, not cancelled.
            let status = unsafe {
                (self.engine.library.api().conversation_wait_until_done)(self.raw.as_ptr())
            };
            if let Err(err) = self
                .engine
                .library
                .check("litert_lm_conversation_wait_until_done", status)
            {
                // The runtime may still be working on this conversation.
                self.abandon();
                return Err(err);
            }
        } else {
            // Stopped early, or ended with an error, which does not prove the
            // runtime is done. Either way the conversation is spent: ask the
            // runtime to stop whatever still runs. A request that fails
            // changes nothing below.
            let _ = self.cancel_stream();
            if state == EndState::Running && !shared.wait_end(grace) {
                self.abandon();
                return Err(Error::Stalled);
            }
        }

        let Some(active) = self.active.take() else {
            return Ok(StreamEnd { cancelled: false });
        };
        // Whatever the runtime still sends for this stream is dropped on
        // arrival, and the queue is freed.
        active.id.retire();
        if let Some(fault) = shared.protocol_fault() {
            return Err(Error::Protocol(fault));
        }
        Ok(StreamEnd {
            cancelled: active.cancel_issued,
        })
    }

    /// Give up on a conversation the runtime could not be shown to be done
    /// with: never call into it again, never free it, and stop the engine and
    /// its other conversations from being used or freed under it. What the
    /// runtime still sends for its stream is dropped on arrival.
    fn abandon(&mut self) {
        self.abandoned = true;
        self.engine.mark_unusable();
        if let Some(active) = self.active.take() {
            active.id.retire();
        }
    }
}

impl Drop for Conversation {
    fn drop(&mut self) {
        if self.active.is_some() {
            // The stream's handle was lost (`mem::forget`) before it was
            // finished or dropped: tear the stream down here, by the same
            // rules. Errors cannot be reported from a destructor.
            let _ = self.finish_stream(DEFAULT_GRACE);
        }
        if self.abandoned || self.engine.is_unusable() {
            // The runtime may still be using this handle, or the engine it
            // belongs to: it is leaked, and so is the engine (see
            // `EngineInner::drop`).
            return;
        }
        // SAFETY: created by `conversation_create`, deleted exactly once. No
        // stream is running: either none was, or the runtime ended it, or it
        // was cancelled, which is the sequence upstream documents before a
        // delete.
        unsafe { (self.engine.library.api().conversation_delete)(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_one_candidate() {
        let greedy = Sampling::greedy();
        assert_eq!(greedy.top_k, 1);
        assert_eq!(greedy.seed, None);
    }

    #[test]
    fn default_turn_options_set_nothing() {
        assert_eq!(
            TurnOptions::default(),
            TurnOptions {
                max_output_tokens: None,
                repetition_penalty: None,
                visual_token_budget: None,
            }
        );
    }
}

/// What a conversation keeps for its stream, against the scripted runtime.
#[cfg(all(test, feature = "test-util"))]
mod stream_bookkeeping {
    use super::*;
    use crate::testing::{FakeRuntime, Script, Step};

    const WAIT: Duration = Duration::from_secs(10);

    fn fake() -> FakeRuntime {
        let fake = FakeRuntime::new();
        fake.script(
            "normal",
            Script::new([Step::text("a"), Step::text("b"), Step::End]),
        );
        fake
    }

    /// Nothing of a finished stream stays behind: not its registration with
    /// the callback, and not its queue with the chunks nobody read.
    #[test]
    fn a_finished_stream_is_retired_with_its_unread_chunks() {
        let fake = fake();
        let engine = fake.engine();
        let mut conversation = engine
            .create_conversation(&ConversationConfig::new(Sampling::greedy()))
            .expect("a conversation");

        for _ in 0..3 {
            let stream = conversation
                .send_stream("script:normal", &TurnOptions::default())
                .expect("the stream starts");
            // Losing the handle releases the borrow; the conversation still
            // owns the stream. The lost handle keeps one reference to the
            // queue for good, which is the leak `mem::forget` asks for.
            std::mem::forget(stream);
            let active = conversation.active.as_ref().expect("a running stream");
            let (id, shared) = (active.id, Arc::clone(&active.shared));
            assert!(id.is_registered());
            // Let the reply end by itself, so the conversation stays usable.
            assert!(shared.wait_end(WAIT));

            conversation.finish_stream(WAIT).expect("teardown succeeds");

            assert!(conversation.active.is_none());
            assert!(!id.is_registered(), "the callback can no longer reach it");
            assert_eq!(
                Arc::strong_count(&shared),
                2,
                "only the lost handle and this test still hold the queue"
            );
        }
    }

    #[test]
    fn an_abandoned_stream_is_retired_too() {
        let fake = fake();
        fake.script(
            "stall",
            Script::new([Step::text("a"), Step::WaitForRelease]),
        );
        let engine = fake.engine();
        let mut conversation = engine
            .create_conversation(&ConversationConfig::new(Sampling::greedy()))
            .expect("a conversation");
        let stream = conversation
            .send_stream("script:stall", &TurnOptions::default())
            .expect("the stream starts");
        std::mem::forget(stream);
        let id = conversation.active.as_ref().expect("a running stream").id;

        let err = conversation
            .finish_stream(Duration::from_millis(50))
            .expect_err("the stream never ends");

        assert!(matches!(err, Error::Stalled));
        assert!(conversation.abandoned);
        assert!(!id.is_registered());
        fake.release();
    }
}

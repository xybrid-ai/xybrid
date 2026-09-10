//! Tool-result continuation support for the executor's LLM paths.
//!
//! One `run` is always one model turn. When an input envelope carries tool
//! results (see [`Envelope::tool_results`]), the executor replays the prior
//! assistant turn plus the results as a raw, protocol-faithful continuation
//! prompt through the backend's `generate_raw` / `generate_raw_streaming`
//! path. The helpers here are pure orchestration glue shared by every LLM
//! path that can compose one — plain text, conversation context, streaming,
//! streaming-with-context, and the text-only vision-language variants; the
//! protocol text itself (composition, parsing, turn markers) is owned by
//! `runtime_adapter::tool_call`.
//!
//! The one shape that stays closed is a conversation whose replayed turns
//! contain images: `generate_raw` is a text-only surface, so those image
//! embeddings cannot be re-evaluated from the composed prompt. That is a
//! property of the replay mechanism, not an unfinished path — see
//! [`text_messages_from_multimodal`].

use crate::ir::Envelope;
use crate::runtime_adapter::llm::GenerationOutput;
// llamacpp-only on purpose: `streaming_postprocess` is not compiled for
// mistral-only builds, and the mistral backend rejects tool-bearing
// requests outright, so a continuation can never execute there.
#[cfg(feature = "llm-llamacpp")]
use crate::runtime_adapter::streaming_postprocess::strip_and_capture_thinking_tags;
use crate::runtime_adapter::tool_call::{
    compose_tool_continuation, truncate_at_turn_marker, TURN_MARKERS,
};
use crate::runtime_adapter::{
    AdapterError, ChatMessage, GenerationConfig, LlmBackend, MultimodalChatMessage,
    StreamingCallback,
};

use super::types::ExecutorResult;

/// Run one tool-result continuation turn through the backend's raw path.
///
/// The chat prefix is re-rendered byte-identically (maximizing KV-prefix
/// reuse), then the prior assistant turn and the tool responses are appended
/// in the model's own protocol, detected from the rendered base by
/// `compose_tool_continuation`, and generation continues through `generate_raw`.
///
/// Reasoning models: the raw path itself never splits `<think>` blocks
/// (its other callers feed non-chat prompts), so this function does it —
/// a continuation is a chat turn, and reasoning models (e.g.
/// LFM2.5-1.2B-Thinking) deliberate before the continuation answer just
/// like any other turn. Known v1 limitation: the rendered base may end
/// with a primed `<think>` channel that belongs to a fresh assistant turn
/// rather than the replayed prior turn; the models tested self-open the
/// tag anyway, and the dangling-close handling in
/// `strip_and_capture_thinking_tags` covers the primed shape.
pub(crate) fn run_tool_continuation(
    backend: &dyn LlmBackend,
    messages: &[ChatMessage],
    gen_config: &GenerationConfig,
    input: &Envelope,
    responses_json: &str,
) -> ExecutorResult<GenerationOutput> {
    let (prompt, raw_config) =
        compose_continuation_prompt(backend, messages, gen_config, input, responses_json)?;
    let out = backend.generate_raw(&prompt, &raw_config)?;
    Ok(finish_continuation_output(out))
}

/// Streaming sibling of [`run_tool_continuation`].
///
/// Composes the identical prompt and runs it through the backend's
/// `generate_raw_streaming`, so a continuation turn emits tokens as it
/// generates instead of arriving as one block. The emitted stream is already
/// filtered (reasoning and tool-protocol blocks suppressed) by the backend's
/// streaming filter; the returned output gets the same final cleanup as the
/// batch path so envelope and stream agree.
pub(crate) fn run_tool_continuation_streaming(
    backend: &dyn LlmBackend,
    messages: &[ChatMessage],
    gen_config: &GenerationConfig,
    input: &Envelope,
    responses_json: &str,
    on_token: StreamingCallback<'_>,
) -> ExecutorResult<GenerationOutput> {
    let (prompt, raw_config) =
        compose_continuation_prompt(backend, messages, gen_config, input, responses_json)?;
    let out = backend.generate_raw_streaming(&prompt, &raw_config, on_token)?;
    Ok(finish_continuation_output(out))
}

/// Render the chat prefix, append the replayed assistant turn plus the tool
/// responses in the model's own protocol, and return the composed raw prompt
/// alongside the generation config to run it with.
fn compose_continuation_prompt(
    backend: &dyn LlmBackend,
    messages: &[ChatMessage],
    gen_config: &GenerationConfig,
    input: &Envelope,
    responses_json: &str,
) -> ExecutorResult<(String, GenerationConfig)> {
    let prior_text = input
        .metadata
        .get(Envelope::TOOL_PRIOR_TEXT_METADATA_KEY)
        .ok_or_else(|| {
            AdapterError::InvalidInput(
                "tool_responses metadata requires tool_prior_text; build continuation \
                 envelopes with Envelope::tool_results"
                    .to_string(),
            )
        })?;
    let base = backend.render_chat_prompt(messages, gen_config)?;
    let prompt = compose_tool_continuation(&base, prior_text, responses_json)?;
    // generate_raw applies only caller-supplied stop sequences (no
    // chat-marker merging like the chat path). The composed continuation
    // ends at the next turn marker of whichever protocol the template
    // speaks; supported families do not legitimately emit one another's
    // markers, so the shared set is safe. Belt-and-braces alongside each
    // model's own end-of-turn token.
    let mut raw_config = gen_config.clone();
    for stop in TURN_MARKERS {
        if !raw_config.stop_sequences.iter().any(|s| s == stop) {
            raw_config.stop_sequences.push((*stop).to_string());
        }
    }
    Ok((prompt, raw_config))
}

/// Final cleanup shared by the batch and streaming continuation paths.
fn finish_continuation_output(mut out: GenerationOutput) -> GenerationOutput {
    // The raw path keeps stop-marker text in the output (the chat path
    // truncates it). Cut it so the answer is clean and a further
    // continuation doesn't double the turn marker.
    truncate_at_turn_marker(&mut out.text);
    // Split the reasoning channel the way the chat paths do — a no-op for
    // non-reasoning models (no tags → text unchanged, reasoning `None`).
    #[cfg(feature = "llm-llamacpp")]
    {
        let (clean, reasoning) = strip_and_capture_thinking_tags(&out.text);
        out.text = clean;
        out.reasoning_content = reasoning.or(out.reasoning_content);
    }
    out
}

/// Flatten backend-neutral multimodal messages to plain text chat messages
/// for the raw tool-continuation path. Image (or other non-text) parts are
/// an invalid-input error: `generate_raw` is a text-only surface, so a
/// conversation whose replayed turns contain images cannot continue through
/// it — the image embeddings cannot be re-evaluated from text.
pub(crate) fn text_messages_from_multimodal(
    messages: &[MultimodalChatMessage],
) -> ExecutorResult<Vec<ChatMessage>> {
    messages
        .iter()
        .map(|message| {
            let mut content = String::new();
            for part in &message.parts {
                match part.as_text() {
                    Some(text) => content.push_str(text),
                    None => {
                        return Err(AdapterError::InvalidInput(
                            "tool-result continuation requires a text-only conversation. \
                             A continuation replays the prior turns as a composed text \
                             prompt, and image embeddings cannot be re-evaluated from \
                             text — so an image-bearing conversation cannot continue on \
                             any path, streaming or not. Run the tool loop on a text-only \
                             conversation, or describe the image in text first."
                                .to_string(),
                        ))
                    }
                }
            }
            Ok(ChatMessage {
                role: message.role,
                content,
            })
        })
        .collect()
}

/// The tool-result payload carried by a continuation envelope, resolved once
/// at the executor's entry point and threaded down to whichever
/// message-based execution function ends up running the turn.
///
/// Those functions never see the envelope itself, so without this the
/// continuation metadata would be silently dropped and the turn would run as
/// a fresh question — a plausible but ungrounded answer.
#[derive(Clone, Copy)]
pub(crate) struct ToolContinuation<'a> {
    /// The continuation envelope, for `tool_prior_text` and turn replay.
    pub input: &'a Envelope,
    /// Serialized tool responses (`Envelope::TOOL_RESPONSES_METADATA_KEY`).
    pub responses_json: &'a str,
}

impl<'a> ToolContinuation<'a> {
    /// Resolve the continuation payload from `input`, or `None` when the
    /// envelope is an ordinary first-turn request.
    ///
    /// Only the envelope's own metadata is inspected. Continuation metadata
    /// buried inside a `MultiPart` part is a different shape and is refused
    /// by [`reject_nested_tool_continuation_parts`], because the multimodal
    /// conversion discards part metadata.
    pub fn from_input(input: &'a Envelope) -> Option<Self> {
        input
            .metadata
            .get(Envelope::TOOL_RESPONSES_METADATA_KEY)
            .map(|responses_json| Self {
                input,
                responses_json,
            })
    }
}

/// Refuse continuation metadata buried inside `MultiPart` parts.
///
/// Outer continuations are composed by every LLM path now; a *nested* one is
/// not a supported envelope shape, and the multimodal conversion discards
/// part metadata, so a nested continuation would silently lose its tool
/// results. Fail loudly instead.
pub(crate) fn reject_nested_tool_continuation_parts(
    input: &Envelope,
    path: &str,
) -> Result<(), AdapterError> {
    if let crate::ir::EnvelopeKind::MultiPart(parts) = &input.kind {
        if parts.iter().any(carries_tool_continuation) {
            return Err(AdapterError::InvalidInput(format!(
                "tool-result continuation metadata inside a MultiPart part is not a supported \
                 envelope shape on the {path} path; build the turn with Envelope::tool_results \
                 so the continuation rides the envelope itself"
            )));
        }
    }
    Ok(())
}

/// Whether `input` (or any nested `MultiPart` part) carries continuation
/// metadata.
fn carries_tool_continuation(input: &Envelope) -> bool {
    if input
        .metadata
        .contains_key(Envelope::TOOL_RESPONSES_METADATA_KEY)
    {
        return true;
    }
    if let crate::ir::EnvelopeKind::MultiPart(parts) = &input.kind {
        return parts.iter().any(carries_tool_continuation);
    }
    false
}

//! Tool-result continuation support for the executor's LLM paths.
//!
//! One `run` is always one model turn. When an input envelope carries tool
//! results (see [`Envelope::tool_results`]), the executor replays the prior
//! assistant turn plus the results as a raw, protocol-faithful continuation
//! prompt through the backend's `generate_raw` path. The helpers here are
//! pure orchestration glue shared by the text path (`execute_llm`) and the
//! text-only vision-language path; the protocol text itself (composition,
//! parsing, turn markers) is owned by `runtime_adapter::tool_call`.

use crate::ir::Envelope;
use crate::runtime_adapter::llm::GenerationOutput;
use crate::runtime_adapter::tool_call::{
    compose_tool_continuation, truncate_at_turn_marker, TURN_MARKERS,
};
use crate::runtime_adapter::{
    AdapterError, ChatMessage, GenerationConfig, LlmBackend, MultimodalChatMessage,
};

use super::types::ExecutorResult;

/// Run one tool-result continuation turn through the backend's raw path.
///
/// The chat prefix is re-rendered byte-identically (maximizing KV-prefix
/// reuse), then the prior assistant turn and the tool responses are appended
/// in the model's own protocol — ChatML/LFM2 or gemma-4, detected from the
/// rendered base by `compose_tool_continuation` — and generation continues
/// through `generate_raw`.
///
/// Known v1 limitation: for a reasoning model the rendered base ends with
/// the primed `<think>` channel, which belongs to a FRESH assistant turn —
/// not to the replayed prior turn appended next. No LFM2- or gemma-4-family
/// tool model is a reasoning model today; revisit priming placement when a
/// reasoning+tools model lands.
pub(crate) fn run_tool_continuation(
    backend: &dyn LlmBackend,
    messages: &[ChatMessage],
    gen_config: &GenerationConfig,
    input: &Envelope,
    responses_json: &str,
) -> ExecutorResult<GenerationOutput> {
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
    // speaks; neither family legitimately emits the other's markers, so the
    // shared marker set is safe. Belt-and-braces alongside the model's own
    // EOS (`<|im_end|>` for LFM2-family, `<turn|>` for gemma-4-family).
    let mut raw_config = gen_config.clone();
    for stop in TURN_MARKERS {
        if !raw_config.stop_sequences.iter().any(|s| s == stop) {
            raw_config.stop_sequences.push((*stop).to_string());
        }
    }
    let mut out = backend.generate_raw(&prompt, &raw_config)?;
    // The raw path keeps stop-marker text in the output (the chat path
    // truncates it). Cut it so the answer is clean and a further
    // continuation doesn't double the turn marker.
    truncate_at_turn_marker(&mut out.text);
    Ok(out)
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
                            "tool-result continuation requires a text-only conversation; \
                             image inputs cannot replay through the raw continuation path"
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

/// Reject tool-result continuation envelopes on paths that do not compose
/// the raw continuation prompt. Without this guard those paths would treat
/// the envelope as a fresh first turn — silently dropping the prior
/// assistant turn and the tool results — and produce a plausible but wrong
/// answer. v1 supports continuation on the non-streaming paths only (text,
/// and text-only vision-language turns).
pub(crate) fn reject_tool_continuation_input(
    input: &Envelope,
    path: &str,
) -> Result<(), AdapterError> {
    if input
        .metadata
        .contains_key(Envelope::TOOL_RESPONSES_METADATA_KEY)
    {
        return Err(AdapterError::InvalidInput(format!(
            "tool-result continuation envelopes are not supported on the {path} path; \
             run the continuation turn through the non-streaming text path"
        )));
    }
    Ok(())
}

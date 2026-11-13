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
use crate::runtime_adapter::streaming_postprocess::{
    is_reasoning_primed, strip_and_capture_thinking_tags_primed,
};
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
/// like any other turn. LFM3 with thinking on reopens `<think>` on the
/// continuation's fresh assistant turn, so its reasoning arrives without an
/// opener and the split is primed: a thought the budget cut short is
/// reasoning, not the answer. Known v1 limitation (ChatML): the rendered
/// base may end with a primed `<think>` channel that belongs to a fresh
/// assistant turn rather than the replayed prior turn, and the continuation
/// opens its own turn unprimed; the models tested self-open the tag anyway,
/// and the dangling-close handling in `strip_and_capture_thinking_tags`
/// covers the primed shape.
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
    Ok(finish_continuation_output(out, Some(&prompt)))
}

/// Streaming sibling of [`run_tool_continuation`].
///
/// Composes the identical prompt and runs it through the backend's
/// `generate_raw_streaming`, so a continuation turn emits tokens as it
/// generates instead of arriving as one block. The emitted stream is already
/// filtered (reasoning and tool-protocol blocks suppressed) by the backend's
/// streaming filter, primed when the prompt ends inside `<think>`; the
/// returned output gets the same final cleanup as the batch path so envelope
/// and stream agree.
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
    // The backend split the reasoning off its text with the same priming.
    Ok(finish_continuation_output(out, None))
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
///
/// `raw_prompt` is the prompt `out.text` was generated from while that text
/// is still the model's output as generated (`generate_raw`), and `None` once
/// the backend has split the reasoning off (`generate_raw_streaming` does).
fn finish_continuation_output(
    mut out: GenerationOutput,
    raw_prompt: Option<&str>,
) -> GenerationOutput {
    // The raw path keeps stop-marker text in the output (the chat path
    // truncates it). Cut it so the answer is clean and a further
    // continuation doesn't double the turn marker.
    truncate_at_turn_marker(&mut out.text);
    // Split the reasoning channel the way the chat paths do — a no-op for
    // non-reasoning models (no tags → text unchanged, reasoning `None`).
    // After a prompt that ends inside `<think>`, a thought the budget cut
    // short has no marker at all. Priming text that is already split would
    // instead turn a truncated answer into reasoning.
    #[cfg(feature = "llm-llamacpp")]
    {
        let primed = raw_prompt.is_some_and(is_reasoning_primed);
        let truncated_by_budget = out.finish_reason == "length";
        let (clean, reasoning) =
            strip_and_capture_thinking_tags_primed(&out.text, primed, truncated_by_budget);
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

// The reasoning split under test is llama.cpp-only (see the import above).
#[cfg(all(test, feature = "llm-llamacpp"))]
mod tests {
    use super::*;
    use crate::ir::ToolCallResult;
    use crate::runtime_adapter::{LlmConfig, LlmResult};
    use std::sync::Mutex;

    /// LFM3's template render of one weather question with a `get_weather`
    /// tool, up to the thinking state.
    const LFM3_PROMPT: &str = "<|startoftext|><|im_start|>system\n<tools>\n{\"name\": \"get_weather\", \"description\": \"Get the current weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}\n</tools><|im_end|>\n<|im_start|>user\nWhat's the weather in Paris?<|im_end|>\n<|im_start|>assistant\n";
    /// LFM3-1.7B's call for that question.
    const LFM3_CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    /// A ChatML render primed for a reasoning model, as LFM2.5-Thinking's is.
    const CHATML_PRIMED_PROMPT: &str =
        "<|im_start|>user\nWhat's the weather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n";
    const LFM2_CALL: &str = "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>";

    /// Backend with a fixed chat render and a fixed raw output, which records
    /// the raw prompts it is handed.
    struct RawOutputBackend {
        rendered: String,
        output: GenerationOutput,
        raw_prompts: Mutex<Vec<String>>,
    }

    impl RawOutputBackend {
        fn new(rendered: &str, text: &str, finish_reason: &str) -> Self {
            Self::with_output(rendered, output(text, finish_reason, None))
        }

        fn with_output(rendered: &str, output: GenerationOutput) -> Self {
            Self {
                rendered: rendered.to_string(),
                output,
                raw_prompts: Mutex::new(Vec::new()),
            }
        }

        fn raw_prompt(&self) -> String {
            self.raw_prompts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .first()
                .cloned()
                .expect("the raw path ran")
        }
    }

    impl LlmBackend for RawOutputBackend {
        fn name(&self) -> &str {
            "raw-output"
        }

        fn supported_formats(&self) -> Vec<&'static str> {
            vec!["gguf"]
        }

        fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> {
            Ok(())
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn unload(&mut self) -> LlmResult<()> {
            Ok(())
        }

        fn generate(
            &self,
            _messages: &[ChatMessage],
            _config: &GenerationConfig,
        ) -> LlmResult<GenerationOutput> {
            panic!("a continuation must not run through the chat path")
        }

        fn render_chat_prompt(
            &self,
            _messages: &[ChatMessage],
            _config: &GenerationConfig,
        ) -> LlmResult<String> {
            Ok(self.rendered.clone())
        }

        fn generate_raw(
            &self,
            prompt: &str,
            _config: &GenerationConfig,
        ) -> LlmResult<GenerationOutput> {
            self.raw_prompts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(prompt.to_string());
            Ok(self.output.clone())
        }
    }

    fn output(text: &str, finish_reason: &str, reasoning: Option<&str>) -> GenerationOutput {
        GenerationOutput {
            text: text.to_string(),
            tokens_generated: 16,
            generation_time_ms: 100,
            tokens_per_second: 160.0,
            finish_reason: finish_reason.to_string(),
            ttft_ms: None,
            mean_itl_ms: None,
            p95_itl_ms: None,
            emitted_chunks: None,
            inter_chunk_ms: Vec::new(),
            decode_tps: None,
            prefill_tps: None,
            image_preprocess_ms: None,
            reasoning_content: reasoning.map(str::to_string),
        }
    }

    fn weather_results(prior_text: &str) -> Envelope {
        Envelope::tool_results(
            "What's the weather in Paris?",
            prior_text,
            &[ToolCallResult {
                call_id: "call_0".to_string(),
                name: "get_weather".to_string(),
                content: serde_json::json!({"temperature_c": 21, "condition": "sunny"}),
            }],
        )
    }

    fn run_batch(backend: &RawOutputBackend, prior_text: &str) -> GenerationOutput {
        let input = weather_results(prior_text);
        let continuation = ToolContinuation::from_input(&input).expect("a continuation");
        run_tool_continuation(
            backend,
            &[ChatMessage::user("What's the weather in Paris?")],
            &GenerationConfig::default(),
            continuation.input,
            continuation.responses_json,
        )
        .expect("the continuation runs")
    }

    #[test]
    fn lfm3_continuation_out_of_budget_mid_thought_has_no_answer() {
        let backend = RawOutputBackend::new(
            &format!("{LFM3_PROMPT}<think>"),
            "The tool says it is sunny and",
            "length",
        );

        let out = run_batch(&backend, LFM3_CALL);

        assert!(
            backend
                .raw_prompt()
                .ends_with("<|im_start|>assistant\n<think>"),
            "the continuation reopens the thought: {}",
            backend.raw_prompt()
        );
        assert_eq!(out.text, "");
        assert_eq!(
            out.reasoning_content.as_deref(),
            Some("The tool says it is sunny and")
        );
    }

    #[test]
    fn lfm3_continuation_splits_a_closed_thought() {
        let backend = RawOutputBackend::new(
            &format!("{LFM3_PROMPT}<think>"),
            "The tool says 21.</think>It is sunny and 21°C in Paris.<|im_end|>",
            "stop",
        );

        let out = run_batch(&backend, LFM3_CALL);

        assert_eq!(out.text, "It is sunny and 21°C in Paris.");
        assert_eq!(out.reasoning_content.as_deref(), Some("The tool says 21."));
    }

    /// The chat paths' rule: a rare thought that ends without `</think>`
    /// reads as the answer rather than vanishing.
    #[test]
    fn lfm3_continuation_ending_without_closing_its_thought_keeps_the_text() {
        let backend = RawOutputBackend::new(
            &format!("{LFM3_PROMPT}<think>"),
            "It is sunny and 21°C in Paris.<|im_end|>",
            "stop",
        );

        let out = run_batch(&backend, LFM3_CALL);

        assert_eq!(out.text, "It is sunny and 21°C in Paris.");
        assert_eq!(out.reasoning_content, None);
    }

    #[test]
    fn lfm3_continuation_with_thinking_off_is_not_primed() {
        let backend = RawOutputBackend::new(
            &format!("{LFM3_PROMPT}<think></think>"),
            "It is sunny and",
            "length",
        );

        let out = run_batch(&backend, LFM3_CALL);

        assert_eq!(out.text, "It is sunny and");
        assert_eq!(out.reasoning_content, None);
    }

    /// ChatML continuations open a fresh, unprimed turn, even after a primed
    /// base: unchanged.
    #[test]
    fn chatml_continuation_is_not_primed() {
        let backend = RawOutputBackend::new(CHATML_PRIMED_PROMPT, "It is sunny and", "length");

        let out = run_batch(&backend, LFM2_CALL);

        assert!(backend.raw_prompt().ends_with("<|im_start|>assistant\n"));
        assert_eq!(out.text, "It is sunny and");
        assert_eq!(out.reasoning_content, None);
    }

    /// The streaming backend splits the reasoning itself; priming its split
    /// text again would turn a truncated answer into reasoning.
    #[test]
    fn streamed_continuation_keeps_the_backend_split() {
        struct SplitStreamBackend(RawOutputBackend);

        impl LlmBackend for SplitStreamBackend {
            fn name(&self) -> &str {
                self.0.name()
            }

            fn supported_formats(&self) -> Vec<&'static str> {
                self.0.supported_formats()
            }

            fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> {
                Ok(())
            }

            fn is_loaded(&self) -> bool {
                true
            }

            fn unload(&mut self) -> LlmResult<()> {
                Ok(())
            }

            fn generate(
                &self,
                messages: &[ChatMessage],
                config: &GenerationConfig,
            ) -> LlmResult<GenerationOutput> {
                self.0.generate(messages, config)
            }

            fn render_chat_prompt(
                &self,
                messages: &[ChatMessage],
                config: &GenerationConfig,
            ) -> LlmResult<String> {
                self.0.render_chat_prompt(messages, config)
            }

            fn generate_raw(
                &self,
                prompt: &str,
                config: &GenerationConfig,
            ) -> LlmResult<GenerationOutput> {
                self.0.generate_raw(prompt, config)
            }

            fn generate_raw_streaming(
                &self,
                prompt: &str,
                config: &GenerationConfig,
                _on_token: StreamingCallback<'_>,
            ) -> LlmResult<GenerationOutput> {
                self.0.generate_raw(prompt, config)
            }
        }

        // What the llama.cpp backend returns for `The tool says 21.</think>It
        // is sunny and` cut off by the budget.
        let backend = SplitStreamBackend(RawOutputBackend::with_output(
            &format!("{LFM3_PROMPT}<think>"),
            output("It is sunny and", "length", Some("The tool says 21.")),
        ));
        let input = weather_results(LFM3_CALL);
        let continuation = ToolContinuation::from_input(&input).expect("a continuation");

        let out = run_tool_continuation_streaming(
            &backend,
            &[ChatMessage::user("What's the weather in Paris?")],
            &GenerationConfig::default(),
            continuation.input,
            continuation.responses_json,
            Box::new(|_token| Ok(())),
        )
        .expect("the continuation runs");

        assert_eq!(out.text, "It is sunny and");
        assert_eq!(out.reasoning_content.as_deref(), Some("The tool says 21."));
    }
}

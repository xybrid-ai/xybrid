//! The REPL's tool-calling turn engine.
//!
//! One `run` is one model turn; this loop is deliberately plain app code
//! (xybrid ships no loop API — see AGENTS.md). Per query it drives:
//! tools-offering turn → execute returned calls → feed results back via
//! `Envelope::tool_results` → repeat, within a hard turn budget, then a
//! forced tools-off synthesis if the model never settles.
//!
//! Path rules the loop encodes (v1 core constraints):
//! - The tools-offering first turn may use the conversation-context paths
//!   (tool-call parsing works there) and may stream (tool-call blocks are
//!   suppressed from the token stream).
//! - Continuation turns run through the **non-context, non-streaming**
//!   `run` — the executor composes the prior assistant turn itself and
//!   rejects continuations everywhere else.
//! - Continuations replay only the immediately-prior turn, so a running
//!   findings log is folded into the (replayed) user text each turn.
//!
//! The context/non-context mix is deliberate: the first turn doubles as the
//! plain-chat path, and it is what lets a follow-up like "what year was
//! that novel published?" resolve "that novel" from history — running the
//! whole loop context-free would strip history from every query. The cost
//! is an asymmetry: continuation turns (tool-bearing queries only) do not
//! see conversation history until core grows context-aware continuation
//! composition (v2). Misrouting a continuation through a context path is a
//! hard error, not a silent degradation.

use std::collections::HashSet;
use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use xybrid_core::conversation::ConversationContext;
use xybrid_core::gateway::Tool;
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole, ToolCallResult};
use xybrid_core::runtime_adapter::tool_call::strip_tool_calls;
use xybrid_core::runtime_adapter::types::GenerationConfig;
use xybrid_sdk::model::XybridModel;

use super::tools::{self, ToolBox};
use crate::ui;

/// Tool-turn budget per query. After this many tool-calling turns the loop
/// stops offering tools and forces a final answer from the notes — a small
/// model can loop indefinitely without settling on its own.
const MAX_TOOL_TURNS: usize = 4;

/// Layer session options over the model's resolved defaults. An explicit
/// config replaces model defaults wholesale at the executor, so it must start
/// from them or a one-field override (or tools alone) would silently reset
/// template sampling params and the reasoning-budget floor to generic defaults.
fn session_generation_config(
    mut base: GenerationConfig,
    tools: Option<Vec<Tool>>,
    max_tokens: Option<usize>,
) -> GenerationConfig {
    if let Some(tools) = tools {
        base = base.with_tools(tools);
    }
    if let Some(max_tokens) = max_tokens {
        base = base.with_max_tokens(max_tokens);
    }
    base
}

/// Default system prompt when tools are active and the user supplied none.
/// Capabilities-first, tools-second: a small model told it "has tools"
/// narrows its self-concept to tool dispatch and starts refusing ordinary
/// requests (writing, brainstorming), so normal abilities are licensed
/// explicitly before tools are mentioned. Override with `--system`.
pub(crate) const TOOL_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. You write, brainstorm, explain, and chat \
     directly — always do these yourself, never refuse them. You also have \
     tools for looking things up: for questions about facts, people, or \
     events, call web_search once with a short, focused query, then answer \
     from the results. When information may be newer than your knowledge or \
     you find you do not know, call web_search instead of saying the \
     information is unavailable. Never repeat a call you already made. For \
     everything else, answer without tools.";

pub(crate) struct QueryOutcome {
    pub answer: String,
    pub finish_reason: Option<String>,
    pub reasoning_present: bool,
    /// True when the answer's tokens were already printed live by the
    /// streaming callback — the caller must not print it again.
    pub already_printed: bool,
    /// Tool calls actually executed (repeats and nudges included).
    pub tool_calls_run: usize,
    /// Token throughput of the answer turn, when it streamed.
    pub stream_stats: Option<StreamStats>,
}

/// Streaming throughput of an answer turn, for the caller's stats line.
pub(crate) struct StreamStats {
    pub tokens: usize,
    /// Time to first token, measured from the turn's start.
    pub ttft: Option<std::time::Duration>,
}

struct FirstTurn {
    text: String,
    calls: Vec<xybrid_core::gateway::ToolCall>,
    already_printed: bool,
    stream_stats: Option<StreamStats>,
    reasoning_content: Option<String>,
    finish_reason: Option<String>,
}

/// Run one REPL query end-to-end: the tools-offering turn, any tool
/// continuation turns, and the forced synthesis fallback.
///
/// `context` is read-only — the caller pushes the user/assistant turns into
/// its `ConversationContext` after this returns (pushing before would double
/// the user turn in the prompt).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_query(
    model: &XybridModel,
    context: Option<&ConversationContext>,
    question: &str,
    toolbox: Option<&ToolBox>,
    system_prompt: Option<&str>,
    stream: bool,
    show_reasoning: bool,
    max_tokens: Option<usize>,
    verbose: u8,
) -> Result<QueryOutcome> {
    // Keeping `None` when no session option is set preserves the executor's
    // no-config path; otherwise layering starts from the same resolved defaults.
    let config = (toolbox.is_some() || max_tokens.is_some()).then(|| {
        session_generation_config(
            model.default_generation_config(),
            toolbox.map(|toolbox| toolbox.definitions()),
            max_tokens,
        )
    });

    // ── Tools-offering turn (context-aware; may stream) ─────────────────────
    let mut input = Envelope::new(EnvelopeKind::Text(question.to_string()));
    if context.is_some() {
        input = input.with_role(MessageRole::User);
    } else if let Some(system) = system_prompt {
        input
            .metadata
            .insert("system_prompt".to_string(), system.to_string());
    }

    let first_turn = run_first_turn(model, context, &input, config.as_ref(), stream)?;

    if first_turn.calls.is_empty() {
        let answer = strip_tool_calls(&first_turn.text);
        // The rendered tool declarations alone make some small models
        // refuse ordinary requests they handle fine without tools ("I
        // can't write a poem"). A zero-tool-call capability refusal gets
        // one retry with the declarations withheld — same system prompt
        // and context, since only the declarations cause the narrowing.
        // Streamed turns already printed their text, so they keep it.
        if !stream && toolbox.is_some() && looks_like_capability_refusal(&answer) {
            if let Some(outcome) =
                retry_without_tools(model, context, &input, show_reasoning, max_tokens)
            {
                return Ok(outcome);
            }
        }
        print_turn_reasoning(show_reasoning, first_turn.reasoning_content.as_deref());
        return Ok(QueryOutcome {
            answer,
            finish_reason: first_turn.finish_reason,
            reasoning_present: first_turn
                .reasoning_content
                .as_deref()
                .is_some_and(|r| !r.trim().is_empty()),
            already_printed: first_turn.already_printed,
            tool_calls_run: 0,
            stream_stats: first_turn.stream_stats,
        });
    }

    if stream {
        // Cue the stream→batch transition. Continuations can stream, but the
        // REPL runs them batched on purpose: the agent loop interleaves tool
        // output, duplicate-call detection, and a forced synthesis turn, and
        // partial text between those would read as interleaved noise. A
        // well-behaved model also calls its tool almost immediately, so often
        // nothing streamed at all — without a cue the drop to silence reads
        // as a hang.
        if first_turn.already_printed {
            println!();
        }
        ui::hint("tool turns print after generation");
    }

    // Tool calls can only come back when tools were offered; treat the
    // impossible case as a plain answer rather than panicking.
    let Some(toolbox) = toolbox else {
        print_turn_reasoning(show_reasoning, first_turn.reasoning_content.as_deref());
        return Ok(QueryOutcome {
            answer: strip_tool_calls(&first_turn.text),
            finish_reason: first_turn.finish_reason,
            reasoning_present: first_turn
                .reasoning_content
                .as_deref()
                .is_some_and(|r| !r.trim().is_empty()),
            already_printed: first_turn.already_printed,
            tool_calls_run: 0,
            stream_stats: None,
        });
    };

    // The reasoning behind the tool decision is the interesting part of a
    // thinking model's tool turn — show it next to the calls it produced.
    print_turn_reasoning(show_reasoning, first_turn.reasoning_content.as_deref());

    // ── Tool continuation turns (batched by choice — see the cue above) ────
    let mut findings: Vec<String> = Vec::new();
    let mut seen_calls: HashSet<String> = HashSet::new();
    let mut prior_text = first_turn.text;
    let mut prior_calls = first_turn.calls;
    let mut tool_calls_run = 0usize;
    let mut turn = 1usize;

    loop {
        let mut results = Vec::with_capacity(prior_calls.len());
        let mut saw_new_result = false;
        for call in &prior_calls {
            tool_calls_run += 1;
            let (result, is_new) =
                execute_and_display(call, toolbox, &mut seen_calls, &mut findings);
            saw_new_result |= is_new;
            results.push(result);
        }

        // A turn made of nothing but repeated calls produced no new
        // information — replaying the duplicate-call notes gets them echoed
        // back as the "answer" on small models. Jump straight to the forced
        // synthesis from the real notes instead.
        if !saw_new_result {
            if verbose > 0 {
                ui::hint("no new tool results — forcing a final answer from the notes");
            }
            break;
        }

        if turn >= MAX_TOOL_TURNS {
            break;
        }
        turn += 1;

        let mut envelope = Envelope::tool_results(fold(question, &findings), &prior_text, &results);
        if let Some(system) = system_prompt {
            envelope
                .metadata
                .insert("system_prompt".to_string(), system.to_string());
        }

        let response = model
            .run(&envelope, config.as_ref())
            .context("tool continuation turn failed")?;
        let text = response
            .text()
            .context("expected text output from LLM turn")?
            .to_string();
        let calls = response.tool_calls();
        if verbose > 0 {
            ui::hint(&format!("turn {turn}: {} tool call(s)", calls.len()));
        }

        if calls.is_empty() {
            print_turn_reasoning(show_reasoning, response.reasoning_content());
            return Ok(QueryOutcome {
                answer: strip_tool_calls(&text),
                finish_reason: response.metadata("finish_reason").cloned(),
                reasoning_present: response
                    .reasoning_content()
                    .is_some_and(|r| !r.trim().is_empty()),
                already_printed: false,
                tool_calls_run,
                stream_stats: None,
            });
        }

        // Another tool turn — surface its deliberation next to its calls.
        print_turn_reasoning(show_reasoning, response.reasoning_content());

        prior_text = text;
        prior_calls = calls;
    }

    // ── Forced synthesis: budget spent without the model settling ───────────
    // Ask as a FRESH turn — no tools, no continuation framing — so the model
    // treats the notes as plain context to summarize rather than a
    // conversation to continue with more tool calls.
    if verbose > 0 {
        ui::hint("tool budget spent — forcing a final answer from the notes");
    }
    let notes = if findings.is_empty() {
        "(no results were found)".to_string()
    } else {
        findings.join("\n\n")
    };
    let prompt = format!(
        "Question: {question}\n\nSearch notes:\n{notes}\n\nUsing only the notes \
         above, answer the question in one or two sentences."
    );
    let mut envelope = Envelope::new(EnvelopeKind::Text(prompt));
    envelope.metadata.insert(
        "system_prompt".to_string(),
        "You are a concise assistant. Answer the user's question using only the \
         provided search notes."
            .to_string(),
    );
    // The prompt and EOS enforce answer brevity; the generation budget must
    // still leave room for a thinking model's reasoning channel.
    let synthesis_config =
        session_generation_config(model.default_generation_config(), None, max_tokens);
    let response = model
        .run(&envelope, Some(&synthesis_config))
        .context("forced synthesis turn failed")?;
    let answer = strip_tool_calls(
        response
            .text()
            .context("expected text output from synthesis turn")?,
    );
    print_turn_reasoning(show_reasoning, response.reasoning_content());

    Ok(QueryOutcome {
        answer,
        finish_reason: response.metadata("finish_reason").cloned(),
        reasoning_present: response
            .reasoning_content()
            .is_some_and(|r| !r.trim().is_empty()),
        already_printed: false,
        tool_calls_run,
        stream_stats: None,
    })
}

/// Run the tools-offering turn, streaming tokens live when asked to.
/// Returns the raw assistant text (tool-call block included — continuations
/// need it verbatim), the parsed calls, whether tokens were printed, and
/// stream throughput when the turn was the final answer.
fn run_first_turn(
    model: &XybridModel,
    context: Option<&ConversationContext>,
    input: &Envelope,
    config: Option<&GenerationConfig>,
    stream: bool,
) -> Result<FirstTurn> {
    if stream {
        let started = Instant::now();
        let mut tokens = 0usize;
        let mut first_token_at: Option<Instant> = None;
        let result = match context {
            Some(ctx) => model.run_streaming_with_context(input, ctx, config, |token| {
                if tokens == 0 {
                    first_token_at = Some(Instant::now());
                }
                tokens += 1;
                print!("{}", token.token);
                std::io::stdout().flush()?;
                Ok(())
            }),
            None => model.run_streaming(input, config, |token| {
                if tokens == 0 {
                    first_token_at = Some(Instant::now());
                }
                tokens += 1;
                print!("{}", token.token);
                std::io::stdout().flush()?;
                Ok(())
            }),
        }
        .context("streaming turn failed")?;
        let text = result
            .text()
            .context("expected text output from LLM turn")?
            .to_string();
        let calls = result.tool_calls();
        let reasoning_content = result.reasoning_content().map(str::to_string);
        let finish_reason = result.metadata("finish_reason").cloned();
        let (printed_any, stats) = if calls.is_empty() {
            // The whole answer streamed.
            let stats = StreamStats {
                tokens,
                ttft: first_token_at.map(|at| at.duration_since(started)),
            };
            (true, Some(stats))
        } else {
            // Only the (usually empty) preamble streamed; the tool-call
            // block itself was suppressed.
            (!strip_tool_calls(&text).trim().is_empty(), None)
        };
        Ok(FirstTurn {
            text,
            calls,
            already_printed: printed_any,
            stream_stats: stats,
            reasoning_content,
            finish_reason,
        })
    } else {
        let result = match context {
            Some(ctx) => model.run_with_context(input, ctx, config),
            None => model.run(input, config),
        }
        .context("inference failed")?;
        let text = result
            .text()
            .context("expected text output from LLM turn")?
            .to_string();
        let calls = result.tool_calls();
        let reasoning_content = result.reasoning_content().map(str::to_string);
        let finish_reason = result.metadata("finish_reason").cloned();
        Ok(FirstTurn {
            text,
            calls,
            already_printed: false,
            stream_stats: None,
            reasoning_content,
            finish_reason,
        })
    }
}

/// Execute one tool call with the REPL's display treatment: a spinner while
/// it runs, then a dimmed `⚙ name("arg") · 640ms` line and a one-line result
/// preview. Repeated calls short-circuit to a nudge without networking.
fn execute_and_display(
    call: &xybrid_core::gateway::ToolCall,
    toolbox: &ToolBox,
    seen_calls: &mut HashSet<String>,
    findings: &mut Vec<String>,
) -> (ToolCallResult, bool) {
    let label = tools::display_call(call);

    // Small models re-issue the same call instead of answering. Return a
    // machine-shaped duplicate marker instead of hitting the network again
    // — deliberately not phrased as an instruction, because small models
    // echo instruction-shaped tool results back as their answer.
    if !seen_calls.insert(tools::dedup_key(call)) {
        println!(
            "  {} {}",
            ui::secondary(&format!("⚙ {label}")),
            ui::dim("· repeated — skipped")
        );
        let result = ToolCallResult {
            call_id: call.id.clone(),
            name: call.function.name.clone(),
            content: serde_json::json!({
                "status": "duplicate_call",
                "detail": "this exact call already ran; its results are in the notes",
            }),
        };
        return (result, false);
    }

    let spinner = ui::spinner(&format!("⚙ {label}"));
    let started = Instant::now();
    let result = toolbox.execute(call);
    spinner.finish_and_clear();
    let elapsed_ms = started.elapsed().as_millis();

    println!(
        "  {} {}",
        ui::secondary(&format!("⚙ {label}")),
        ui::dim(&format!("· {elapsed_ms}ms"))
    );
    let summary = tools::summarize(&result);
    ui::sub(&preview_line(&summary, 120));
    findings.push(summary);

    (result, true)
}

/// Conservative, head-anchored detector for "I can't do that" answers a
/// small model gives to ordinary requests when tool declarations are in
/// its prompt. Used only to trigger a tools-off retry, so a false positive
/// costs one extra (usually better) turn.
fn looks_like_capability_refusal(answer: &str) -> bool {
    let head: String = answer.chars().take(120).collect::<String>().to_lowercase();
    head.contains("i'm sorry, but i can")
        || head.contains("i am sorry, but i can")
        || head.contains("i don't have the capability")
        || head.contains("i do not have the capability")
        || head.contains("i don't have access to")
        || head.contains("i do not have access to")
        || head.contains("i can't generate")
        || head.contains("i cannot generate")
}

/// Re-run the turn without tool declarations; returns only a real answer.
fn retry_without_tools(
    model: &XybridModel,
    context: Option<&ConversationContext>,
    input: &Envelope,
    show_reasoning: bool,
    max_tokens: Option<usize>,
) -> Option<QueryOutcome> {
    let config = max_tokens.map(|max_tokens| {
        session_generation_config(model.default_generation_config(), None, Some(max_tokens))
    });
    let result = match context {
        Some(ctx) => model.run_with_context(input, ctx, config.as_ref()),
        None => model.run(input, config.as_ref()),
    }
    .ok()?;
    let answer = strip_tool_calls(result.text()?);
    if answer.trim().is_empty() || looks_like_capability_refusal(&answer) {
        return None;
    }
    ui::hint("answered without tools");
    print_turn_reasoning(show_reasoning, result.reasoning_content());
    Some(QueryOutcome {
        answer,
        finish_reason: result.metadata("finish_reason").cloned(),
        reasoning_present: result
            .reasoning_content()
            .is_some_and(|r| !r.trim().is_empty()),
        already_printed: false,
        tool_calls_run: 0,
        stream_stats: None,
    })
}

/// Show one turn's chain-of-thought as a dimmed block directly above
/// whatever the turn produced — its `⚙` tool calls or its answer. One
/// uniform presentation for every turn keeps `--show-reasoning` output
/// chronological (think → act → think → answer) instead of repeating a
/// separate reasoning section per query.
fn print_turn_reasoning(show_reasoning: bool, reasoning: Option<&str>) {
    if !show_reasoning {
        return;
    }
    let Some(reasoning) = reasoning.map(str::trim).filter(|r| !r.is_empty()) else {
        return;
    };
    for line in reasoning.lines() {
        println!("    {}", ui::dim(line));
    }
}

/// Fold the running findings into the user turn so the model retains every
/// prior result despite the single-turn continuation replay.
fn fold(question: &str, findings: &[String]) -> String {
    if findings.is_empty() {
        question.to_string()
    } else {
        format!(
            "{question}\n\nNotes from earlier searches:\n{}",
            findings.join("\n\n")
        )
    }
}

/// First `max` characters of `text` on a single line.
fn preview_line(text: &str, max: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut line: String = collapsed.chars().take(max).collect();
    if line.chars().count() < collapsed.chars().count() {
        line.push('…');
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tool() -> Tool {
        Tool::function(
            "lookup",
            "Look up a value",
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }

    #[test]
    fn session_config_preserves_base_fields_when_overriding_max_tokens() {
        let base = GenerationConfig {
            temperature: 0.7,
            top_k: 12,
            stop_sequences: vec!["<end>".to_string()],
            ..Default::default()
        };

        let config = session_generation_config(base, None, Some(64));

        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 12);
        assert_eq!(config.stop_sequences, vec!["<end>".to_string()]);
    }

    #[test]
    fn session_config_attaches_tools_without_clearing_base_fields() {
        let base = GenerationConfig {
            temperature: 0.7,
            top_p: 0.8,
            repetition_penalty: 1.25,
            ..Default::default()
        };

        let config = session_generation_config(base, Some(vec![test_tool()]), None);

        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.8);
        assert_eq!(config.repetition_penalty, 1.25);
        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].function.name, "lookup");
    }

    #[test]
    fn session_config_without_overrides_returns_base_unchanged() {
        let base = GenerationConfig {
            max_tokens: 3584,
            temperature: 0.7,
            top_p: 0.8,
            min_p: 0.02,
            top_k: 12,
            repetition_penalty: 1.25,
            stop_sequences: vec!["<end>".to_string()],
            grammar: Some("root ::= \"ok\"".to_string()),
            tools: vec![test_tool()],
        };

        let config = session_generation_config(base, None, None);

        assert_eq!(config.max_tokens, 3584);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.8);
        assert_eq!(config.min_p, 0.02);
        assert_eq!(config.top_k, 12);
        assert_eq!(config.repetition_penalty, 1.25);
        assert_eq!(config.stop_sequences, vec!["<end>".to_string()]);
        assert_eq!(config.grammar.as_deref(), Some("root ::= \"ok\""));
        assert_eq!(config.tools.len(), 1);
    }

    #[test]
    fn fold_appends_findings_to_the_question() {
        assert_eq!(fold("q?", &[]), "q?");
        let folded = fold("q?", &["Search \"a\":\n- t: s".to_string()]);
        assert!(folded.starts_with("q?\n\nNotes from earlier searches:"));
        assert!(folded.contains("- t: s"));
    }

    #[test]
    fn capability_refusal_detector_is_head_anchored_and_conservative() {
        assert!(looks_like_capability_refusal(
            "I'm sorry, but I can't write a poem. However, I can help you brainstorm!"
        ));
        assert!(looks_like_capability_refusal(
            "I don't have the capability to generate poems directly."
        ));
        assert!(looks_like_capability_refusal(
            "I don't have access to real-time sports data."
        ));

        // Real answers pass through untouched.
        assert!(!looks_like_capability_refusal(
            "The 2022 FIFA World Cup final was won by Argentina."
        ));
        assert!(!looks_like_capability_refusal(
            "Here's a poem about the sea:\nWaves that wander, wide and free…"
        ));
        // Refusal-ish phrasing deep in an otherwise-real answer is ignored.
        let long_tail = format!(
            "{} I'm sorry, but I can't say more.",
            "A fine answer sentence. ".repeat(10)
        );
        assert!(!looks_like_capability_refusal(&long_tail));
    }

    #[test]
    fn preview_line_collapses_and_caps() {
        assert_eq!(preview_line("a  b\nc", 120), "a b c");
        let long = "word ".repeat(50);
        let preview = preview_line(&long, 20);
        assert!(preview.chars().count() <= 21); // cap + ellipsis
        assert!(preview.ends_with('…'));
    }
}

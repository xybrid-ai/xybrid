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
/// Token cap for the forced synthesis turn.
const SYNTHESIS_MAX_TOKENS: usize = 256;

/// Default system prompt when tools are active and the user supplied none.
/// Deliberation-first: the model should decide whether a tool is needed at
/// all before reaching for one, with a clear stop condition once results
/// arrive. Override with `--system`.
pub(crate) const TOOL_SYSTEM_PROMPT: &str =
    "You are a helpful assistant with tools. Before answering, decide \
     whether a tool is needed. Greetings, chit-chat, and simple math: answer \
     directly, no tools. Questions about facts, people, or events: call \
     web_search once with a short, focused query, then answer from the \
     results. Never repeat a call you already made, and never claim you \
     cannot look something up — web_search is available. Once the results \
     answer the question, stop calling tools and answer concisely.";

pub(crate) struct QueryOutcome {
    pub answer: String,
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
    verbose: u8,
) -> Result<QueryOutcome> {
    // Tools ride the config; `Default` matches what the executor uses when
    // no config is passed, so plain-chat behavior is unchanged.
    let config =
        toolbox.map(|toolbox| GenerationConfig::default().with_tools(toolbox.definitions()));

    // ── Tools-offering turn (context-aware; may stream) ─────────────────────
    let mut input = Envelope::new(EnvelopeKind::Text(question.to_string()));
    if context.is_some() {
        input = input.with_role(MessageRole::User);
    } else if let Some(system) = system_prompt {
        input
            .metadata
            .insert("system_prompt".to_string(), system.to_string());
    }

    let (first_text, first_calls, first_streamed, stream_stats) =
        run_first_turn(model, context, &input, config.as_ref(), stream)?;

    if first_calls.is_empty() {
        return Ok(QueryOutcome {
            answer: strip_tool_calls(&first_text),
            already_printed: first_streamed,
            tool_calls_run: 0,
            stream_stats,
        });
    }

    if stream {
        // Cue the stream→batch transition: continuation answers arrive
        // whole (streaming rejects them), and a well-behaved model calls
        // its tool almost immediately, so often nothing streamed at all —
        // without a cue the drop to silence reads as a hang.
        if first_streamed {
            println!();
        }
        ui::hint("tool turns print after generation");
    }

    // Tool calls can only come back when tools were offered; treat the
    // impossible case as a plain answer rather than panicking.
    let Some(toolbox) = toolbox else {
        return Ok(QueryOutcome {
            answer: strip_tool_calls(&first_text),
            already_printed: first_streamed,
            tool_calls_run: 0,
            stream_stats: None,
        });
    };

    // ── Tool continuation turns (non-context, non-streaming) ────────────────
    let mut findings: Vec<String> = Vec::new();
    let mut seen_calls: HashSet<String> = HashSet::new();
    let mut prior_text = first_text;
    let mut prior_calls = first_calls;
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
            return Ok(QueryOutcome {
                answer: strip_tool_calls(&text),
                already_printed: false,
                tool_calls_run,
                stream_stats: None,
            });
        }

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
    let synthesis_config = GenerationConfig::default().with_max_tokens(SYNTHESIS_MAX_TOKENS);
    let response = model
        .run(&envelope, Some(&synthesis_config))
        .context("forced synthesis turn failed")?;
    let answer = strip_tool_calls(
        response
            .text()
            .context("expected text output from synthesis turn")?,
    );

    Ok(QueryOutcome {
        answer,
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
) -> Result<(
    String,
    Vec<xybrid_core::gateway::ToolCall>,
    bool,
    Option<StreamStats>,
)> {
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
        Ok((text, calls, printed_any, stats))
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
        Ok((text, calls, false, None))
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

    #[test]
    fn fold_appends_findings_to_the_question() {
        assert_eq!(fold("q?", &[]), "q?");
        let folded = fold("q?", &["Search \"a\":\n- t: s".to_string()]);
        assert!(folded.starts_with("q?\n\nNotes from earlier searches:"));
        assert!(folded.contains("- t: s"));
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

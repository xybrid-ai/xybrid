//! Tool calling in a streaming chat: tokens, history, and a continuation.
//!
//! This is the shape a chat UI actually needs, and the one that used to fail
//! closed. It runs the whole loop through `execute_streaming_with_context`:
//!
//! 1. Turn 1 streams. Tool-call blocks never reach the token feed — they are
//!    protocol traffic, not answer text — and the terminal `PartialToken`
//!    carries `finish_reason: "tool_calls"`, the parsed `tool_calls`, and
//!    `raw_text` (the turn's output *with* the block, which is what the
//!    continuation replays).
//! 2. The tool runs in ordinary application code.
//! 3. Turn 2 feeds an `Envelope::tool_results` envelope back through the same
//!    streaming-with-context call. It streams too, and it still sees history.
//!
//! Fetch the fixture first:
//!   ./integration-tests/download.sh functiongemma-270m-it
//!
//! Run with:
//!   cargo run --example functiongemma_tools_streaming_context \
//!       -p xybrid-core --features llm-llamacpp

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use xybrid_core::conversation::ConversationContext;
use xybrid_core::execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor};
use xybrid_core::gateway::Tool;
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole, ToolCallResult};
use xybrid_core::runtime_adapter::types::{GenerationConfig, PartialToken};
use xybrid_core::testing::model_fixtures;

const MODEL_ID: &str = "functiongemma-270m-it";
const SYSTEM_PROMPT: &str = "Use the available function and report its result clearly.";
const USER_PROMPT: &str = "What's the current temperature in London?";

fn weather_tool() -> Tool {
    Tool::function(
        "get_current_temperature",
        "Gets the current temperature for a given location.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco"
                }
            },
            "required": ["location"]
        }),
    )
}

fn model_dir() -> PathBuf {
    model_fixtures::model_path(MODEL_ID)
        .unwrap_or_else(|| PathBuf::from("integration-tests/fixtures/models").join(MODEL_ID))
}

/// What a streaming turn hands back: the text the UI painted, plus the typed
/// handoff on the terminal token.
#[derive(Default)]
struct StreamedTurn {
    emitted: String,
    terminal: Option<PartialToken>,
}

/// Run one turn through the streaming-with-context path, collecting the token
/// feed and the terminal token. The `Arc<Mutex<..>>` is only because the
/// callback is `Send`; a real UI would push straight into its own state.
fn stream_turn(
    executor: &mut TemplateExecutor,
    metadata: &ModelMetadata,
    input: &Envelope,
    context: &ConversationContext,
    config: &GenerationConfig,
) -> Result<(StreamedTurn, Envelope), Box<dyn std::error::Error>> {
    let turn = Arc::new(Mutex::new(StreamedTurn::default()));
    let sink = turn.clone();

    let response = executor.execute_streaming_with_context(
        metadata,
        input,
        context,
        Box::new(move |token: PartialToken| {
            let mut turn = sink.lock().unwrap_or_else(|e| e.into_inner());
            print!("{}", token.token);
            use std::io::Write;
            std::io::stdout().flush().ok();
            turn.emitted.push_str(&token.token);
            if token.finish_reason.is_some() {
                turn.terminal = Some(token);
            }
            Ok(())
        }),
        Some(config),
    )?;
    println!();

    let turn = Arc::try_unwrap(turn)
        .map_err(|_| "streaming callback outlived the run")?
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    Ok((turn, response))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  FunctionGemma — streaming tool calling, with context");
    println!("═══════════════════════════════════════════════════════\n");

    let dir = model_dir();
    let metadata_path = dir.join("model_metadata.json");
    if !metadata_path.exists() {
        return Err(format!(
            "metadata not found at {}; run ./integration-tests/download.sh {MODEL_ID}",
            metadata_path.display()
        )
        .into());
    }
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let model_file = match &metadata.execution_template {
        ExecutionTemplate::Gguf { model_file, .. } => model_file,
        other => return Err(format!("expected Gguf metadata, got {other:?}").into()),
    };
    if !dir.join(model_file).exists() {
        return Err(format!(
            "model not found at {}; run ./integration-tests/download.sh {MODEL_ID}",
            dir.join(model_file).display()
        )
        .into());
    }

    let mut executor =
        TemplateExecutor::with_base_path(dir.to_str().ok_or("model dir is not valid UTF-8")?);
    let config = GenerationConfig::greedy()
        .with_max_tokens(128)
        .with_tools(vec![weather_tool()]);

    // A chat screen's state: system turn plus whatever has been said so far.
    let mut context = ConversationContext::new().with_system(
        Envelope::new(EnvelopeKind::Text(SYSTEM_PROMPT.to_string())).with_role(MessageRole::System),
    );

    // ── Turn 1: stream, and halt at the tool-call boundary ─────────────────
    println!("── turn 1 (streaming) ──");
    let request = Envelope::new(EnvelopeKind::Text(USER_PROMPT.to_string()));
    let (turn, _response) = stream_turn(&mut executor, &metadata, &request, &context, &config)?;

    let terminal = turn
        .terminal
        .ok_or("streaming turn emitted no terminal token")?;
    if terminal.tool_calls.is_empty() {
        return Err("model emitted no parseable tool call".into());
    }
    if turn.emitted.contains("<start_function_call>") {
        return Err("tool-call protocol text leaked into the token feed".into());
    }
    let raw_text = terminal
        .raw_text
        .as_deref()
        .ok_or("terminal token carried calls but no raw text to replay")?;

    println!("finish_reason: {:?}", terminal.finish_reason);
    println!("parsed tool calls: {:#?}", terminal.tool_calls);
    println!("raw turn text (replayed next): {raw_text}");

    // ── Execute the tools: ordinary application code ───────────────────────
    let results: Vec<ToolCallResult> = terminal
        .tool_calls
        .iter()
        .map(|call| ToolCallResult {
            call_id: call.id.clone(),
            name: call.function.name.clone(),
            content: serde_json::json!({"location": "London", "temperature_c": 21}),
        })
        .collect();

    // NOTE: the user turn is deliberately NOT pushed to history yet. The
    // continuation envelope replays `USER_PROMPT` itself, so pushing it here
    // would put the question in the prompt twice. History gets the completed
    // exchange once the loop settles, below.

    // ── Turn 2: the continuation, streamed, still context-aware ────────────
    println!("\n── turn 2 (continuation, streaming, same context) ──");
    let continuation = Envelope::tool_results(USER_PROMPT, raw_text, &results);
    let (final_turn, final_response) =
        stream_turn(&mut executor, &metadata, &continuation, &context, &config)?;

    let final_text = match &final_response.kind {
        EnvelopeKind::Text(text) => text.trim().to_string(),
        other => return Err(format!("expected Text output, got {other:?}").into()),
    };
    if final_text.is_empty() {
        return Err("continuation produced no final answer".into());
    }
    if final_turn.emitted.trim().is_empty() {
        return Err("continuation returned text but streamed nothing".into());
    }

    // The exchange is settled: commit it to history as one user turn plus one
    // assistant turn. The next question then sees this turn as context.
    context.push(
        Envelope::new(EnvelopeKind::Text(USER_PROMPT.to_string())).with_role(MessageRole::User),
    );
    context.push(
        Envelope::new(EnvelopeKind::Text(final_text.clone())).with_role(MessageRole::Assistant),
    );

    println!("\nfinal answer: {final_text}");
    println!(
        "\nStatus: PASS - {} call(s) streamed and answered without leaving the streaming path",
        results.len()
    );
    Ok(())
}

//! Real constrained generation required before preparing a llama.cpp update release.
//!
//! Locally this skips without the FunctionGemma fixture. CI downloads it and
//! sets `XYBRID_REQUIRE_MODELS`, so missing weights fail instead of skipping.
//!
//! Greedy decoding under a grammar makes each output below deterministic, so
//! these tests also pin how a generation reports its end. llama.cpp's loop
//! stops on an end-of-generation token, on a stop sequence matched against the
//! last generated tokens, or when the budget runs out, and `finish_reason` is
//! `length` only in the last case.

#![cfg(feature = "llm-llamacpp")]

use std::path::{Path, PathBuf};

use integration_tests::fixtures;
use xybrid_core::{
    execution::{ModelMetadata, TemplateExecutor},
    ir::{Envelope, EnvelopeKind},
    runtime_adapter::{
        llama_cpp::LlamaCppBackend,
        llm::{GenerationOutput, LlmBackend},
        ChatMessage, GenerationConfig, LlmConfig, PartialToken, StreamingCallback,
    },
};

const MODEL_ID: &str = "functiongemma-270m-it";
const MODEL_FILE: &str = "functiongemma-270m-it-q8_0.gguf";

/// Under this grammar, greedy decoding of either prompt below writes exactly
/// two tokens, "The" then " rest". (Without the priming, FunctionGemma spells
/// the answer with other tokens, such as "T" "he" " rest".)
const THE_REST: &str = r#"root ::= "The rest""#;
const RAW_PROMPT: &str = "The rest\nThe rest\nThe rest\n";
const CHAT_PROMPT: &str = "Repeat after me: The rest";

/// The FunctionGemma fixture directory, or `None` to skip. Its metadata is
/// committed, so the weights are checked as well.
fn functiongemma_dir() -> Option<PathBuf> {
    let model_dir = fixtures::model_for_test(MODEL_ID)?;
    let model_file = model_dir.join(MODEL_FILE);
    if !model_file.is_file() {
        assert!(
            std::env::var_os(fixtures::REQUIRE_MODELS_ENV).is_none(),
            "required GGUF is missing: {}",
            model_file.display()
        );
        eprintln!("Skipping: FunctionGemma weights are not downloaded");
        return None;
    }
    Some(model_dir)
}

/// A freshly loaded backend, so no KV-cache prefix carries over from another
/// case. `reasoning` primes the `<think>` channel, as a thinking model's
/// metadata does.
fn functiongemma(model_dir: &Path, reasoning: bool) -> LlamaCppBackend {
    let model_file = model_dir.join(MODEL_FILE);
    let mut backend = LlamaCppBackend::new().expect("llama.cpp backend initializes");
    backend
        .load(
            &LlmConfig::new(model_file.to_str().expect("fixture path is UTF-8"))
                .with_context_length(512)
                .with_reasoning(reasoning),
        )
        .expect("FunctionGemma loads");
    backend
}

/// A streaming callback that records every chunk it receives.
fn collect_into(chunks: &mut Vec<PartialToken>) -> StreamingCallback<'_> {
    Box::new(move |chunk| {
        chunks.push(chunk);
        Ok(())
    })
}

/// Greedy decoding under `grammar`, within `max_tokens`, with `stops`.
fn constrained(grammar: &str, max_tokens: usize, stops: &[&str]) -> GenerationConfig {
    let mut config = GenerationConfig::greedy()
        .with_max_tokens(max_tokens)
        .with_grammar(grammar);
    config.stop_sequences = stops.iter().map(|stop| stop.to_string()).collect();
    config
}

#[test]
fn text_generation_obeys_grammar() {
    let Some(model_dir) = functiongemma_dir() else {
        return;
    };
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("model_metadata.json"))
            .expect("fixture metadata is readable"),
    )
    .expect("fixture metadata is valid");
    let mut executor =
        TemplateExecutor::with_base_path(model_dir.to_str().expect("fixture path is UTF-8"));
    let config = GenerationConfig::greedy()
        .with_max_tokens(16)
        .with_grammar("root ::= \"ready\"");
    let output = executor
        .execute(
            &metadata,
            &Envelope::new(EnvelopeKind::Text("Say ready.".to_string())),
            Some(&config),
        )
        .expect("constrained inference succeeds");
    match output.kind {
        EnvelopeKind::Text(text) => assert_eq!(text.trim(), "ready"),
        other => panic!("expected constrained text, got {other:?}"),
    }
}

/// `generate_raw` keeps its text as generated, so its finish reason has to be
/// the native loop's own. It used to report `length` for every case; matching
/// stops against the trimmed text instead got the empty, substring and
/// whitespace cases wrong.
#[test]
fn raw_generation_reports_how_the_native_loop_ended() {
    let Some(model_dir) = functiongemma_dir() else {
        return;
    };
    // (grammar, budget, stops, finish_reason, tokens generated)
    let cases: &[(&str, usize, &[&str], &str, usize)] = &[
        (THE_REST, 2, &[], "length", 2),
        // "" tokenizes to nothing, and the loop ignores such a stop.
        (THE_REST, 2, &[""], "length", 2),
        // "he" is text inside the "The" token, never a token suffix.
        (THE_REST, 2, &["he"], "length", 2),
        (THE_REST, 2, &["The"], "stop", 1),
        // The loop checks for a stop before the budget.
        (THE_REST, 2, &[" rest"], "stop", 2),
        // Trimming the text erases this stop; the token suffix keeps it.
        (r#"root ::= "\n""#, 1, &["\n"], "stop", 1),
        // The grammar is complete after "The", so the second token is an EOG.
        (r#"root ::= "The""#, 2, &[], "stop", 2),
    ];
    for &(grammar, max_tokens, stops, finish_reason, tokens) in cases {
        let output = functiongemma(&model_dir, false)
            .generate_raw(RAW_PROMPT, &constrained(grammar, max_tokens, stops))
            .expect("raw generation succeeds");
        let case = format!(
            "{grammar}, budget {max_tokens}, stops {stops:?}: {:?} in {} tokens",
            output.text, output.tokens_generated
        );
        assert_eq!(output.tokens_generated, tokens, "{case}");
        assert_eq!(output.finish_reason, finish_reason, "{case}");
    }
}

/// The chat and raw-streaming paths cut their text at a stop pattern, but an
/// empty stop is not one: it used to match at offset 0, erase the answer and
/// report `stop`, and a stream emitted nothing at all, not even its terminal
/// token. A partial stop trimmed off the end is not one either, because the
/// budget ran out before the output completed it.
#[test]
fn text_paths_ignore_empty_and_partial_stops() {
    let Some(model_dir) = functiongemma_dir() else {
        return;
    };
    let messages = [ChatMessage::user(CHAT_PROMPT)];
    for (stops, text) in [(&[""][..], "The rest"), (&[" rest."][..], "The")] {
        let config = constrained(THE_REST, 2, stops);
        let ended = |path: &str, output: &GenerationOutput| {
            assert_eq!(
                (output.text.as_str(), output.finish_reason.as_str()),
                (text, "length"),
                "{path}, stops {stops:?}"
            );
        };

        let output = functiongemma(&model_dir, false)
            .generate(&messages, &config)
            .expect("chat generation succeeds");
        ended("generate", &output);

        let mut chunks: Vec<PartialToken> = Vec::new();
        let output = functiongemma(&model_dir, false)
            .generate_streaming(&messages, &config, collect_into(&mut chunks))
            .expect("streaming chat generation succeeds");
        ended("generate_streaming", &output);
        let terminal = chunks
            .last()
            .expect("the stream ends with a terminal token");
        assert_eq!(terminal.finish_reason.as_deref(), Some("length"));
        assert_eq!(terminal.cumulative_text, text);

        let mut chunks: Vec<PartialToken> = Vec::new();
        let output = functiongemma(&model_dir, false)
            .generate_raw_streaming(RAW_PROMPT, &config, collect_into(&mut chunks))
            .expect("streaming raw generation succeeds");
        ended("generate_raw_streaming", &output);
        let terminal = chunks
            .last()
            .expect("the stream ends with a terminal token");
        assert_eq!(terminal.finish_reason.as_deref(), Some("length"));
        assert_eq!(terminal.cumulative_text, text);
    }
}

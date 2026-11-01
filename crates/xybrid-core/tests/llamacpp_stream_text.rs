//! What the llama.cpp backend streams, against the text it returns, for
//! output that token boundaries split.
//!
//! A byte-level vocabulary writes some characters as several tokens, none of
//! them UTF-8 on its own: qwen3 splits "🙂" and the CJK of "日本語", and
//! FunctionGemma falls back to bytes for rarer characters such as "🦩" and
//! "ꙮ". Those characters used to vanish from the stream and from the final
//! text. Each test skips without its fixture, and fails instead when
//! `XYBRID_REQUIRE_MODELS` is set.
//!
//! Run with:
//!
//! ```sh
//! ./integration-tests/download.sh functiongemma-270m-it
//! ./integration-tests/download.sh qwen3-0.6b
//! XYBRID_REQUIRE_MODELS=1 cargo test -p xybrid-core --features llm-llamacpp \
//!   --test llamacpp_stream_text
//! ```

#![cfg(feature = "llm-llamacpp")]

use std::path::{Path, PathBuf};

use xybrid_core::runtime_adapter::{
    llama_cpp::LlamaCppBackend, llm::LlmBackend, ChatMessage, GenerationConfig, LlmConfig,
    PartialToken, StreamingCallback,
};
use xybrid_core::testing::model_fixtures;

/// The fixture's GGUF, or `None` to skip.
fn gguf(model_id: &str, file: &str) -> Option<PathBuf> {
    match model_fixtures::model_or_skip(model_id) {
        Some(dir) => Some(dir.join(file)),
        None => {
            assert!(
                !model_fixtures::models_required(),
                "{} is set but model '{model_id}' is not available",
                model_fixtures::ENV_REQUIRE_MODELS
            );
            None
        }
    }
}

/// A loaded backend. Two threads apiece keep the tests in this file from
/// fighting over a CI runner's cores when they run in parallel.
fn backend(gguf: &Path) -> LlamaCppBackend {
    let mut backend = LlamaCppBackend::new().expect("llama.cpp backend initializes");
    backend
        .load(
            &LlmConfig::new(gguf.to_str().expect("fixture path is UTF-8"))
                .with_context_length(512)
                .with_threads(2),
        )
        .expect("the fixture loads");
    backend
}

/// A streaming callback that records every chunk it receives.
fn collect_into(chunks: &mut Vec<PartialToken>) -> StreamingCallback<'_> {
    Box::new(move |chunk| {
        chunks.push(chunk);
        Ok(())
    })
}

/// The deltas a stream emitted, joined.
fn streamed(chunks: &[PartialToken]) -> String {
    chunks.iter().map(|chunk| chunk.token.as_str()).collect()
}

/// Greedy decoding constrained to exactly `text`, with room for all of its
/// tokens and the end-of-generation token after them.
fn exactly(text: &str) -> GenerationConfig {
    GenerationConfig::greedy()
        .with_max_tokens(32)
        .with_grammar(format!("root ::= \"{text}\""))
}

/// Every text path returns `texts` whole, and the stream's deltas add up to
/// them. The grammar fixes the text, so the KV-cache prefix one call leaves
/// for the next cannot change it.
fn assert_split_characters_survive(backend: &LlamaCppBackend, texts: &[&str]) {
    let messages = [ChatMessage::user("Say it.")];
    for &text in texts {
        let config = exactly(text);

        let output = backend
            .generate(&messages, &config)
            .expect("chat generation succeeds");
        assert_eq!(output.text, text, "generate");

        let mut chunks = Vec::new();
        let output = backend
            .generate_streaming(&messages, &config, collect_into(&mut chunks))
            .expect("streaming chat generation succeeds");
        assert_eq!(output.text, text, "generate_streaming");
        assert_eq!(streamed(&chunks), text, "generate_streaming deltas");

        let mut chunks = Vec::new();
        let output = backend
            .generate_raw_streaming("Reply:", &config, collect_into(&mut chunks))
            .expect("streaming raw generation succeeds");
        assert_eq!(output.text, text, "generate_raw_streaming");
        assert_eq!(streamed(&chunks), text, "generate_raw_streaming deltas");

        // The raw path keeps the end-of-generation token's text.
        let output = backend
            .generate_raw("Reply:", &config)
            .expect("raw generation succeeds");
        assert!(
            output.text.starts_with(text),
            "generate_raw returned {:?}",
            output.text
        );
    }
}

/// FunctionGemma is the fixture the llama.cpp CI job downloads.
#[test]
fn characters_split_into_byte_tokens_survive_on_functiongemma() {
    let Some(gguf) = gguf("functiongemma-270m-it", "functiongemma-270m-it-q8_0.gguf") else {
        return;
    };
    assert_split_characters_survive(&backend(&gguf), &["🦩 ok", "ꙮ ok"]);
}

/// The reported case: "🙂 ok" came back as "ok", and "日本語" as "本".
#[test]
fn characters_split_across_tokens_survive_on_qwen3() {
    let Some(gguf) = gguf("qwen3-0.6b", "Qwen3-0.6B-Q8_0.gguf") else {
        return;
    };
    assert_split_characters_survive(&backend(&gguf), &["🙂 ok", "日本語"]);
}

/// The text a token carries ahead of a stop it completes is streamed.
/// Primed this way, FunctionGemma writes "The rest" as "The" then " rest",
/// so "The" completes the stop "he": the stream used to emit nothing while
/// the final text was "T".
#[test]
fn text_ahead_of_a_stop_inside_a_token_is_streamed() {
    let Some(gguf) = gguf("functiongemma-270m-it", "functiongemma-270m-it-q8_0.gguf") else {
        return;
    };
    let backend = backend(&gguf);
    let mut config = GenerationConfig::greedy()
        .with_max_tokens(2)
        .with_grammar(r#"root ::= "The rest""#);
    config.stop_sequences = vec!["he".to_string()];

    let mut chunks = Vec::new();
    let output = backend
        .generate_streaming(
            &[ChatMessage::user("Repeat after me: The rest")],
            &config,
            collect_into(&mut chunks),
        )
        .expect("streaming chat generation succeeds");
    assert_eq!(output.text, "T", "generate_streaming");
    assert_eq!(streamed(&chunks), "T", "generate_streaming deltas");

    let mut chunks = Vec::new();
    let output = backend
        .generate_raw_streaming(
            "The rest\nThe rest\nThe rest\n",
            &config,
            collect_into(&mut chunks),
        )
        .expect("streaming raw generation succeeds");
    assert_eq!(output.text, "T", "generate_raw_streaming");
    assert_eq!(streamed(&chunks), "T", "generate_raw_streaming deltas");
}

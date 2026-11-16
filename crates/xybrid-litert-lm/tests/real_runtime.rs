//! Tests against the real LiteRT-LM runtime and a real model.
//!
//! They need two things this repository does not contain, so they skip
//! (with a message, and passing) unless both are provided:
//!
//! - `XYBRID_LITERT_LM_LIBRARY`: the runtime library file for this host.
//! - `XYBRID_LITERT_LM_TEST_MODEL`: a `.litertlm` model file.
//!
//! `XYBRID_LITERT_LM_CACHE_DIR` optionally redirects the runtime's caches,
//! which it otherwise writes beside the model.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use xybrid_litert_lm::{
    Chunk, Conversation, ConversationConfig, Engine, EngineSettings, Error, Library, LogLevel,
    ModelInfo, Sampling, TurnOptions,
};

const CHUNK_TIMEOUT: Duration = Duration::from_secs(120);
const GRACE: Duration = Duration::from_secs(60);

struct Fixture {
    library: Arc<Library>,
    model: PathBuf,
}

fn fixture() -> Option<Fixture> {
    let model = std::env::var_os("XYBRID_LITERT_LM_TEST_MODEL").map(PathBuf::from);
    let library_set = std::env::var_os(xybrid_litert_lm::LIBRARY_ENV).is_some();
    let (Some(model), true) = (model, library_set) else {
        eprintln!(
            "SKIP: set {} and XYBRID_LITERT_LM_TEST_MODEL to run the real-runtime tests",
            xybrid_litert_lm::LIBRARY_ENV
        );
        return None;
    };
    let library = Library::open_default().expect("the configured runtime library opens");
    library
        .set_min_log_level(LogLevel::Error)
        .expect("log level is accepted");
    Some(Fixture { library, model })
}

fn settings(fixture: &Fixture) -> EngineSettings {
    let mut settings = EngineSettings::new(&fixture.model).with_max_num_tokens(1024);
    if let Some(dir) = std::env::var_os("XYBRID_LITERT_LM_CACHE_DIR") {
        settings = settings.with_cache_dir(PathBuf::from(dir));
    }
    settings
}

fn drain(conversation: &mut Conversation, message: &str, max_tokens: u32) -> Vec<Chunk> {
    let options = TurnOptions {
        max_output_tokens: Some(max_tokens),
        ..TurnOptions::default()
    };
    let mut stream = conversation
        .send_stream(message, &options)
        .expect("the stream starts");
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next_chunk(CHUNK_TIMEOUT) {
        chunks.push(chunk);
    }
    assert!(stream.is_finished(), "the stream ended with a final chunk");
    let end = stream.finish(GRACE).expect("teardown succeeds");
    assert!(!end.cancelled);
    chunks
}

#[test]
fn reports_the_c_api_version_and_reads_model_info() {
    let Some(fixture) = fixture() else { return };
    let (major, _, _) = fixture.library.c_api_version();
    assert_eq!(major, xybrid_litert_lm::C_API_MAJOR);

    let info = ModelInfo::read(&fixture.library, &fixture.model).expect("model info is readable");
    eprintln!("model info: {info:?}");
}

#[test]
fn a_missing_model_fails_with_a_runtime_error_not_a_crash() {
    let Some(fixture) = fixture() else { return };
    let err = ModelInfo::read(
        &fixture.library,
        &PathBuf::from("/nonexistent/model.litertlm"),
    )
    .expect_err("a missing model is an error");
    assert!(matches!(err, Error::Call { .. }), "got {err:?}");
    assert!(!err.is_runtime_unavailable());
}

#[test]
fn streams_a_reply_and_reports_usage() {
    let Some(fixture) = fixture() else { return };
    let engine =
        Engine::load(Arc::clone(&fixture.library), &settings(&fixture)).expect("the model loads");
    let mut conversation = engine
        .create_conversation(&ConversationConfig::new(Sampling::greedy()))
        .expect("a conversation is created");

    let message = r#"{"role":"user","content":"Say hello."}"#;
    let rendered = conversation
        .render_message(message)
        .expect("the message renders");
    assert!(rendered.contains("Say hello."), "rendered: {rendered:?}");

    let chunks = drain(&mut conversation, message, 16);
    let last = chunks.last().expect("at least the final chunk arrives");
    assert!(last.is_final);
    assert_eq!(last.error, None, "the stream ended without an error");
    let payloads: Vec<&str> = chunks.iter().filter_map(|c| c.text.as_deref()).collect();
    assert!(!payloads.is_empty(), "the model produced output");
    assert!(
        payloads.iter().all(|json| json.starts_with('{')),
        "every payload is a JSON message: {payloads:?}"
    );

    let usage = conversation.usage().expect("usage is recorded");
    assert!(usage.prompt_tokens > 0, "{usage:?}");
    assert!(
        usage.output_tokens > 0 && usage.output_tokens <= 16,
        "{usage:?}"
    );
}

#[test]
fn greedy_decoding_is_repeatable_across_conversations() {
    let Some(fixture) = fixture() else { return };
    let engine =
        Engine::load(Arc::clone(&fixture.library), &settings(&fixture)).expect("the model loads");
    let message = r#"{"role":"user","content":"Name one colour."}"#;
    let mut replies = Vec::new();
    for _ in 0..2 {
        let mut conversation = engine
            .create_conversation(&ConversationConfig::new(Sampling::greedy()))
            .expect("a conversation is created");
        let chunks = drain(&mut conversation, message, 12);
        let reply: String = chunks.into_iter().filter_map(|c| c.text).collect();
        replies.push(reply);
    }
    assert_eq!(replies[0], replies[1]);
}

#[test]
fn a_cancelled_stream_tears_down_and_the_engine_stays_usable() {
    let Some(fixture) = fixture() else { return };
    let engine =
        Engine::load(Arc::clone(&fixture.library), &settings(&fixture)).expect("the model loads");

    {
        let mut conversation = engine
            .create_conversation(&ConversationConfig::new(Sampling::greedy()))
            .expect("a conversation is created");
        let options = TurnOptions {
            max_output_tokens: Some(256),
            ..TurnOptions::default()
        };
        let mut stream = conversation
            .send_stream(
                r#"{"role":"user","content":"Count from 1 to 200, one number per line."}"#,
                &options,
            )
            .expect("the stream starts");
        let first = stream
            .next_chunk(CHUNK_TIMEOUT)
            .expect("a first chunk arrives");
        assert!(!first.is_final, "the reply is longer than one chunk");
        stream.cancel().expect("cancel is accepted");
        stream.cancel().expect("a second cancel is a no-op");
        let end = stream
            .finish(GRACE)
            .expect("teardown succeeds after cancel");
        assert!(end.cancelled);

        // A cancelled conversation may only be deleted.
        assert!(matches!(conversation.usage(), Err(Error::Protocol(_))));
        assert!(matches!(
            conversation.send_stream("{}", &TurnOptions::default()),
            Err(Error::Protocol(_))
        ));
    }

    assert!(engine.is_usable());
    let mut conversation = engine
        .create_conversation(&ConversationConfig::new(Sampling::greedy()))
        .expect("the engine still creates conversations");
    let chunks = drain(
        &mut conversation,
        r#"{"role":"user","content":"Say hello."}"#,
        8,
    );
    assert!(chunks.iter().any(|c| c.text.is_some()));
}

#[test]
fn dropping_a_stream_mid_reply_does_not_hang_or_crash() {
    let Some(fixture) = fixture() else { return };
    let engine =
        Engine::load(Arc::clone(&fixture.library), &settings(&fixture)).expect("the model loads");
    let mut conversation = engine
        .create_conversation(&ConversationConfig::new(Sampling::greedy()))
        .expect("a conversation is created");
    let options = TurnOptions {
        max_output_tokens: Some(256),
        ..TurnOptions::default()
    };
    let mut stream = conversation
        .send_stream(
            r#"{"role":"user","content":"Count from 1 to 200, one number per line."}"#,
            &options,
        )
        .expect("the stream starts");
    let _ = stream.next_chunk(CHUNK_TIMEOUT);
    drop(stream);
    drop(conversation);
    assert!(engine.is_usable());
}

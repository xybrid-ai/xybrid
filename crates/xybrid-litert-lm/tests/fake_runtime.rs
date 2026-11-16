//! The stream bridge against a scripted runtime that lives in this process
//! (`xybrid_litert_lm::testing`).
//!
//! These tests prove the Rust side: teardown order, at most one cancel, what
//! happens when the runtime cannot be shown to be at rest, and that a
//! callback arriving after everything was torn down is harmless. How the
//! real runtime behaves is `real_runtime.rs`'s subject.
//!
//! Every test has its own [`FakeRuntime`], so they can run in parallel.

use std::time::{Duration, Instant};

use xybrid_litert_lm::testing::{FakeRuntime, Script, Step};
use xybrid_litert_lm::{
    Chunk, Conversation, ConversationConfig, Engine, EngineSettings, Error, Sampling, StatusCode,
    Stream, TurnOptions,
};

/// How long a test waits for something the fake is about to do.
const WAIT: Duration = Duration::from_secs(10);

/// Chunks the "flood" script sends: more than the queue holds for a turn
/// with an output budget of one token (1,028).
const FLOOD_CHUNKS: usize = 1_200;

const DEADLINE_EXCEEDED: i32 = 4;
const INTERNAL: i32 = 13;

/// A fake with the scripts these tests share.
fn fake() -> FakeRuntime {
    let fake = FakeRuntime::new();
    fake.script(
        "normal",
        Script::new([Step::text("a"), Step::text("b"), Step::text("c"), Step::End]),
    )
    // Keeps talking until it is cancelled, then ends the way the real
    // runtime ends a cancelled turn.
    .script(
        "until-cancel",
        Script::new([Step::UntilCancel("x".into()), Step::cancelled()]),
    )
    // Says one thing and never ends, until the test releases it.
    .script(
        "stall",
        Script::new([
            Step::text("a"),
            Step::WaitForRelease,
            Step::text("after the stall"),
            Step::MarkLate,
        ]),
    );
    fake
}

fn conversation(engine: &Engine) -> Conversation {
    engine
        .create_conversation(&ConversationConfig::new(Sampling::greedy()))
        .expect("the fake creates a conversation")
}

fn start<'c>(conversation: &'c mut Conversation, script: &str) -> Stream<'c> {
    conversation
        .send_stream(&format!("script:{script}"), &TurnOptions::default())
        .expect("the stream starts")
}

fn drain(stream: &mut Stream<'_>) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next_chunk(WAIT) {
        chunks.push(chunk);
    }
    chunks
}

fn texts(chunks: &[Chunk]) -> Vec<&str> {
    chunks.iter().filter_map(|c| c.text.as_deref()).collect()
}

fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting until {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

// ---------------------------------------------------------------------------
// A reply that ends by itself
// ---------------------------------------------------------------------------

#[test]
fn a_reply_arrives_in_order_and_ends_with_the_final_chunk() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "normal");

    let chunks = drain(&mut stream);

    assert_eq!(texts(&chunks), ["a", "b", "c"]);
    let last = chunks.last().expect("a final chunk");
    assert!(last.is_final && last.text.is_none() && last.error.is_none());
    assert!(chunks[..chunks.len() - 1].iter().all(|c| !c.is_final));
    assert!(stream.is_finished());
    assert!(stream.next_chunk(Duration::from_millis(10)).is_none());
    let end = stream.finish(WAIT).expect("teardown succeeds");
    assert!(!end.cancelled);

    drop(conversation);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["wait", "conversation_delete", "engine_delete"],
        "no cancel; the conversation goes before its engine"
    );
}

#[test]
fn a_final_chunk_may_carry_text() {
    let fake = fake();
    fake.script(
        "final-with-text",
        Script::new([
            Step::text("a"),
            Step::Chunk {
                text: Some("b".into()),
                error: None,
                is_final: true,
            },
        ]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "final-with-text");

    let chunks = drain(&mut stream);

    assert_eq!(texts(&chunks), ["a", "b"]);
    assert!(chunks[1].is_final);
    stream.finish(WAIT).expect("teardown succeeds");
}

#[test]
fn callbacks_made_before_the_start_call_returns_are_not_lost() {
    let fake = fake();
    fake.script(
        "before-return",
        Script::new([Step::text("a"), Step::text("b"), Step::End]).inline(),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "before-return");

    let chunks = drain(&mut stream);

    assert_eq!(texts(&chunks), ["a", "b"]);
    assert!(chunks.last().expect("a final chunk").is_final);
    stream.finish(WAIT).expect("teardown succeeds");
}

#[test]
fn several_turns_on_one_conversation_each_get_their_own_stream() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);

    for _ in 0..3 {
        let mut stream = start(&mut conversation, "normal");
        assert_eq!(texts(&drain(&mut stream)), ["a", "b", "c"]);
        stream.finish(WAIT).expect("teardown succeeds");
    }

    drop(conversation);
    assert_eq!(
        fake.events(),
        ["wait", "wait", "wait", "conversation_delete"]
    );
}

#[test]
fn finishing_without_reading_the_reply_still_ends_it() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    // The reply may be over or not when `finish` runs; both are a clean end
    // for the caller, and the conversation is deleted either way.
    let stream = start(&mut conversation, "normal");
    stream.finish(WAIT).expect("teardown succeeds");
    drop(conversation);
    assert_eq!(fake.count("conversation_delete"), 1);
    assert!(engine.is_usable());
}

// ---------------------------------------------------------------------------
// A reply that fails
// ---------------------------------------------------------------------------

#[test]
fn a_failure_after_the_start_arrives_as_a_final_chunk_with_an_error() {
    let fake = fake();
    fake.script(
        "error",
        Script::new([Step::text("a"), Step::error("INTERNAL: boom")]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "error");

    let chunks = drain(&mut stream);

    assert_eq!(texts(&chunks), ["a"]);
    let last = chunks.last().expect("a final chunk");
    assert!(last.is_final);
    assert_eq!(last.error.as_deref(), Some("INTERNAL: boom"));
    // An error does not prove the runtime stopped: it is told to, and the
    // conversation is spent.
    let end = stream.finish(WAIT).expect("teardown succeeds");
    assert!(end.cancelled);
    assert!(matches!(conversation.usage(), Err(Error::Protocol(_))));
    drop(conversation);
    assert_eq!(fake.events(), ["cancel", "conversation_delete"]);
    assert!(engine.is_usable());
}

/// Upstream reports a tool call it cannot parse as an error and keeps
/// decoding. The final chunk is therefore not the end of the native work.
#[test]
fn an_error_reported_while_the_runtime_is_still_decoding_is_cancelled_before_the_delete() {
    let fake = fake();
    fake.script(
        "error-then-more",
        Script::new([
            Step::text("a"),
            Step::error("INVALID_ARGUMENT: Failed to parse tool calls"),
            Step::UntilCancel("still decoding".into()),
            Step::cancelled(),
        ]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "error-then-more");

    let chunks = drain(&mut stream);
    assert_eq!(
        texts(&chunks),
        ["a"],
        "nothing after the error is delivered"
    );
    let end = stream.finish(WAIT).expect("teardown succeeds");

    assert!(end.cancelled);
    // The fake's delete joins its worker, which only ends once cancelled.
    drop(conversation);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["cancel", "conversation_delete", "engine_delete"]
    );
}

#[test]
fn text_and_error_in_one_chunk_are_both_kept() {
    let fake = fake();
    fake.script(
        "text-and-error",
        Script::new([Step::Chunk {
            text: Some("partial".into()),
            error: Some("INTERNAL: boom".into()),
            is_final: true,
        }]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "text-and-error");

    let chunks = drain(&mut stream);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text.as_deref(), Some("partial"));
    assert_eq!(chunks[0].error.as_deref(), Some("INTERNAL: boom"));
    assert!(chunks[0].is_final);
    stream.finish(WAIT).expect("teardown succeeds");
}

#[test]
fn a_null_chunk_ends_the_stream_with_an_error() {
    let fake = fake();
    fake.script("null-chunk", Script::new([Step::NullChunk]));
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "null-chunk");

    let chunks = drain(&mut stream);

    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].is_final);
    assert!(chunks[0].error.is_some());
    stream.finish(WAIT).expect("teardown succeeds");
}

#[test]
fn chunks_after_the_final_chunk_are_ignored() {
    let fake = fake();
    fake.script(
        "extra-after-final",
        Script::new([Step::text("a"), Step::End, Step::text("ghost"), Step::End]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "extra-after-final");

    let chunks = drain(&mut stream);
    // `finish` waits for the script, ghost chunks included.
    stream.finish(WAIT).expect("teardown succeeds");

    assert_eq!(texts(&chunks), ["a"]);
    assert_eq!(chunks.len(), 2);
}

#[test]
fn more_chunks_than_the_output_budget_allows_is_a_protocol_error() {
    let fake = fake();
    fake.script(
        "flood",
        Script::new([Step::Repeat("x".into(), FLOOD_CHUNKS), Step::End]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let options = TurnOptions {
        max_output_tokens: Some(1),
        ..TurnOptions::default()
    };
    let stream = conversation
        .send_stream("script:flood", &options)
        .expect("the stream starts");
    // Nothing reads the queue, so the fake's chunks overrun it. The grace is
    // long because an interpreter (Miri) plays the flood slowly; a native
    // run returns as soon as the final chunk arrives.
    let err = stream
        .finish(Duration::from_secs(300))
        .expect_err("the queue overflowed");

    assert!(matches!(err, Error::Protocol(_)), "got {err:?}");
    drop(conversation);
    assert!(engine.is_usable(), "an overrun is an error, not a stall");
}

#[test]
fn a_stream_that_cannot_start_reports_the_runtimes_message() {
    let fake = fake();
    fake.script("start-fails", Script::failing_start(INTERNAL));
    let engine = fake.engine();
    let mut conversation = conversation(&engine);

    let err = conversation
        .send_stream("script:start-fails", &TurnOptions::default())
        .map(|_| ())
        .expect_err("the fake refuses to start");

    match err {
        Error::Call {
            call,
            code,
            message,
        } => {
            assert_eq!(call, "litert_lm_conversation_send_message_stream");
            assert_eq!(code, StatusCode::Internal);
            assert_eq!(message, "INTERNAL: scripted failure");
        }
        other => panic!("expected a failed call, got {other:?}"),
    }

    // The conversation was not cancelled and still works.
    let mut stream = start(&mut conversation, "normal");
    assert_eq!(texts(&drain(&mut stream)), ["a", "b", "c"]);
    stream.finish(WAIT).expect("teardown succeeds");
    assert_eq!(fake.count("cancel"), 0);
}

// ---------------------------------------------------------------------------
// Stopping early
// ---------------------------------------------------------------------------

#[test]
fn cancel_is_requested_once_and_the_conversation_is_only_deleted_afterwards() {
    let fake = fake();
    fake.script(
        "cancel-then-more",
        Script::new([
            Step::UntilCancel("x".into()),
            Step::text("after cancel"),
            Step::cancelled(),
        ]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "cancel-then-more");
    assert!(stream.next_chunk(WAIT).is_some());

    stream.cancel().expect("cancel is accepted");
    stream.cancel().expect("a second cancel is a no-op");

    // What the runtime produced before it noticed still arrives, then the
    // final chunk, which reports the cancellation.
    let rest = drain(&mut stream);
    assert_eq!(
        texts(&rest).last().copied(),
        Some("after cancel"),
        "chunks sent after the cancel are delivered"
    );
    let last = rest.last().expect("a final chunk");
    assert!(last.is_final);
    assert_eq!(last.error.as_deref(), Some("CANCELLED: Task cancelled"));
    let end = stream.finish(WAIT).expect("teardown succeeds");
    assert!(end.cancelled);

    // A cancelled conversation may only be deleted.
    assert!(matches!(conversation.usage(), Err(Error::Protocol(_))));
    assert!(matches!(
        conversation.render_message("{}"),
        Err(Error::Protocol(_))
    ));
    assert!(matches!(
        conversation.send_stream("script:normal", &TurnOptions::default()),
        Err(Error::Protocol(_))
    ));
    drop(conversation);
    assert_eq!(
        fake.events(),
        ["cancel", "conversation_delete"],
        "one cancel, no wait, then the delete"
    );
    assert!(engine.is_usable());
}

#[test]
fn finishing_early_cancels_and_waits_for_the_final_chunk() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let stream = start(&mut conversation, "until-cancel");

    let end = stream.finish(WAIT).expect("teardown succeeds");

    assert!(end.cancelled);
    assert_eq!(fake.count("cancel"), 1);
}

/// A turn that is silent when the caller stops it: nothing has arrived, and
/// nothing will until the runtime is told to stop.
#[test]
fn a_silent_turn_can_be_stopped() {
    let fake = fake();
    fake.script(
        "silent",
        Script::new([Step::WaitForCancel, Step::cancelled()]),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "silent");
    assert!(stream.next_chunk(Duration::from_millis(30)).is_none());
    assert!(!stream.is_finished());

    let end = stream.finish(WAIT).expect("teardown succeeds");

    assert!(end.cancelled);
    drop(conversation);
    assert_eq!(fake.events(), ["cancel", "conversation_delete"]);
}

#[test]
fn dropping_a_stream_mid_reply_cancels_it() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "until-cancel");
    assert!(stream.next_chunk(WAIT).is_some());

    drop(stream);
    drop(conversation);
    drop(engine);

    assert_eq!(
        fake.events(),
        ["cancel", "conversation_delete", "engine_delete"]
    );
}

#[test]
fn a_panicking_consumer_still_tears_the_stream_down() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut stream = start(&mut conversation, "until-cancel");
        let _ = stream.next_chunk(WAIT);
        panic!("the consumer failed");
    }));

    assert!(result.is_err());
    assert_eq!(fake.count("cancel"), 1);
    drop(conversation);
    assert!(engine.is_usable());
    drop(engine);
    assert_eq!(
        fake.events(),
        ["cancel", "conversation_delete", "engine_delete"]
    );
}

// ---------------------------------------------------------------------------
// A stream handle that is never dropped
// ---------------------------------------------------------------------------

/// `mem::forget` is safe code. It must not leave a conversation that can be
/// used, or freed, while the runtime is still generating on it.
#[test]
fn a_forgotten_stream_leaves_a_conversation_that_refuses_work_until_it_is_dropped() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "until-cancel");
    assert!(stream.next_chunk(WAIT).is_some());

    std::mem::forget(stream);

    // The borrow is gone; the generation is not.
    assert!(matches!(conversation.usage(), Err(Error::Protocol(_))));
    assert!(matches!(
        conversation.render_message("{}"),
        Err(Error::Protocol(_))
    ));
    assert!(matches!(
        conversation.send_stream("script:normal", &TurnOptions::default()),
        Err(Error::Protocol(_))
    ));
    assert_eq!(
        fake.events(),
        [] as [&str; 0],
        "nothing entered the runtime"
    );

    // Dropping the conversation stops the stream by the usual rules.
    drop(conversation);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["cancel", "conversation_delete", "engine_delete"]
    );
}

#[test]
fn a_forgotten_stream_that_had_ended_is_settled_when_its_conversation_is_dropped() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "normal");
    assert_eq!(texts(&drain(&mut stream)), ["a", "b", "c"]);

    std::mem::forget(stream);

    assert!(matches!(conversation.usage(), Err(Error::Protocol(_))));
    drop(conversation);
    assert_eq!(fake.events(), ["wait", "conversation_delete"]);
}

/// Once the engine is unusable nothing more may enter the runtime, a
/// forgotten stream's teardown included.
#[test]
fn a_forgotten_stream_on_an_unusable_engine_is_abandoned_without_another_call() {
    let fake = fake();
    let engine = fake.engine();
    let mut forgotten = conversation(&engine);
    let mut stream = start(&mut forgotten, "until-cancel");
    assert!(stream.next_chunk(WAIT).is_some());
    std::mem::forget(stream);

    let mut stalled = conversation(&engine);
    let err = start(&mut stalled, "stall")
        .finish(Duration::from_millis(100))
        .expect_err("the stream never ends");
    assert!(matches!(err, Error::Stalled), "got {err:?}");

    drop(forgotten);
    drop(stalled);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["cancel"],
        "the stalled stream's cancel, and no call or delete after it"
    );
    fake.release();
}

// ---------------------------------------------------------------------------
// A runtime that cannot be shown to be at rest
// ---------------------------------------------------------------------------

#[test]
fn a_stream_that_never_ends_leaks_instead_of_freeing() {
    let fake = fake();
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "stall");
    assert_eq!(
        stream.next_chunk(WAIT).and_then(|c| c.text).as_deref(),
        Some("a")
    );
    assert!(
        stream.next_chunk(Duration::from_millis(50)).is_none(),
        "nothing more arrives"
    );
    assert!(!stream.is_finished());

    let err = stream
        .finish(Duration::from_millis(200))
        .expect_err("the stream never ends");

    assert!(matches!(err, Error::Stalled), "got {err:?}");
    assert!(!engine.is_usable());
    assert!(matches!(conversation.usage(), Err(Error::EngineUnusable)));
    assert!(matches!(
        engine.create_conversation(&ConversationConfig::new(Sampling::greedy())),
        Err(Error::EngineUnusable)
    ));
    drop(conversation);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["cancel"],
        "nothing was waited on, and nothing the runtime still uses was freed"
    );

    // The runtime wakes up later and calls back into a stream that is gone.
    fake.release();
    wait_until("the late callback has run", || fake.late_callbacks() == 1);
}

/// The runtime's wait has a deadline of its own. When it fails, the turn's
/// final chunk has arrived but the native work is not known to be over.
#[test]
fn a_wait_that_fails_leaves_nothing_reusable_and_nothing_freed() {
    let fake = fake();
    fake.script(
        "wait-fails",
        Script::new([
            Step::text("a"),
            Step::End,
            Step::WaitForRelease,
            Step::text("still working"),
            Step::MarkLate,
        ])
        .failing_wait(DEADLINE_EXCEEDED),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "wait-fails");
    assert_eq!(texts(&drain(&mut stream)), ["a"]);

    let err = stream.finish(WAIT).expect_err("the wait fails");

    assert_eq!(err.status(), Some(StatusCode::DeadlineExceeded));
    assert!(!engine.is_usable());
    assert!(matches!(conversation.usage(), Err(Error::EngineUnusable)));
    assert!(matches!(
        conversation.send_stream("script:normal", &TurnOptions::default()),
        Err(Error::EngineUnusable)
    ));
    drop(conversation);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["wait"],
        "neither the conversation nor its engine was deleted"
    );

    // The native work the wait gave up on carries on, harmlessly.
    fake.release();
    wait_until("the late callback has run", || fake.late_callbacks() == 1);
}

/// One stalled conversation retires the engine for every conversation on it.
#[test]
fn a_stall_stops_the_other_conversations_of_the_engine() {
    let fake = fake();
    let engine = fake.engine();
    let mut stalled = conversation(&engine);
    let mut idle = conversation(&engine);
    let mut busy = conversation(&engine);
    let mut busy_stream = start(&mut busy, "until-cancel");
    assert!(busy_stream.next_chunk(WAIT).is_some());

    let err = start(&mut stalled, "stall")
        .finish(Duration::from_millis(100))
        .expect_err("the stream never ends");
    assert!(matches!(err, Error::Stalled), "got {err:?}");

    // A conversation that was doing nothing refuses everything.
    assert!(matches!(idle.usage(), Err(Error::EngineUnusable)));
    assert!(matches!(
        idle.render_message("{}"),
        Err(Error::EngineUnusable)
    ));
    assert!(matches!(
        idle.send_stream("script:normal", &TurnOptions::default()),
        Err(Error::EngineUnusable)
    ));
    // One that was streaming cannot cancel or settle through the runtime.
    assert!(matches!(busy_stream.cancel(), Err(Error::EngineUnusable)));
    let err = busy_stream.finish(WAIT).expect_err("the engine is gone");
    assert!(matches!(err, Error::EngineUnusable), "got {err:?}");

    drop(stalled);
    drop(idle);
    drop(busy);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["cancel"],
        "the stalled stream's cancel, and no call or delete after it"
    );
    fake.release();
}

// ---------------------------------------------------------------------------
// Late callbacks and ownership
// ---------------------------------------------------------------------------

#[test]
fn a_callback_after_everything_was_torn_down_is_dropped() {
    let fake = fake();
    fake.script(
        "detached",
        Script::new([
            Step::text("a"),
            Step::End,
            Step::WaitForRelease,
            Step::text("after delete"),
            Step::End,
            Step::MarkLate,
        ])
        .detached(),
    );
    let engine = fake.engine();
    let mut conversation = conversation(&engine);
    let mut stream = start(&mut conversation, "detached");
    let chunks = drain(&mut stream);
    assert_eq!(texts(&chunks), ["a"]);
    stream.finish(WAIT).expect("teardown succeeds");
    drop(conversation);
    drop(engine);
    assert_eq!(
        fake.events(),
        ["wait", "conversation_delete", "engine_delete"]
    );

    // A runtime that does not join its callbacks calls back now, with the
    // callback data of a stream, a conversation and an engine that are gone.
    fake.release();
    wait_until("the late callback has run", || fake.late_callbacks() == 1);
}

#[test]
fn an_engine_outlives_its_handle_while_a_conversation_needs_it() {
    let fake = fake();
    let engine = fake.engine();
    let conversation = conversation(&engine);

    drop(engine);
    assert!(
        fake.events().is_empty(),
        "the engine is kept while a conversation exists"
    );

    drop(conversation);
    assert_eq!(fake.events(), ["conversation_delete", "engine_delete"]);
}

#[test]
fn an_unknown_model_is_a_failed_call_with_the_runtimes_message() {
    let fake = fake();
    let err = Engine::load(fake.library(), &EngineSettings::new("fake://nobody"))
        .expect_err("no fake is registered under that path");
    assert_eq!(err.status(), Some(StatusCode::NotFound));
    assert!(!err.is_runtime_unavailable());
    assert!(err.to_string().contains("no fake is registered"), "{err}");
}

// ---------------------------------------------------------------------------
// What the fake records, which tests in other crates rely on
// ---------------------------------------------------------------------------

#[test]
fn the_fake_records_what_a_turn_was_given() {
    use xybrid_litert_lm::RepetitionPenalty;

    let fake = fake();
    fake.set_usage(12, 3).set_render_prefix("<|turn>user\n");
    let engine = fake.engine();
    let config = ConversationConfig::new(Sampling::sampled(0.7, 40, 0.9).with_seed(7))
        .with_messages_json(r#"[{"role":"system","content":"Be brief."}]"#)
        .with_tools_json("[]")
        .with_thinking(true)
        .with_tool_call_channel("tool_call");
    let mut conversation = engine
        .create_conversation(&config)
        .expect("the fake creates a conversation");
    assert_eq!(
        conversation
            .render_message("script:normal")
            .expect("a prompt"),
        "<|turn>user\nscript:normal"
    );
    let options = TurnOptions {
        max_output_tokens: Some(64),
        repetition_penalty: Some(RepetitionPenalty {
            penalty: 1.1,
            window: 64,
        }),
        visual_token_budget: Some(280),
    };
    let mut stream = conversation
        .send_stream("script:normal", &options)
        .expect("the stream starts");
    drain(&mut stream);
    stream.finish(WAIT).expect("teardown succeeds");

    let usage = conversation.usage().expect("the fake has usage figures");
    assert_eq!((usage.prompt_tokens, usage.output_tokens), (12, 3));

    let turns = fake.turns();
    assert_eq!(turns.len(), 1);
    let turn = &turns[0];
    assert_eq!(turn.message_json, "script:normal");
    assert_eq!(
        turn.history_json.as_deref(),
        Some(r#"[{"role":"system","content":"Be brief."}]"#)
    );
    assert_eq!(turn.tools_json.as_deref(), Some("[]"));
    assert_eq!(turn.thinking, Some(true));
    assert_eq!(turn.tool_call_channel.as_deref(), Some("tool_call"));
    assert_eq!(turn.max_output_tokens, Some(64));
    assert_eq!(turn.repetition_penalty, Some((1.1, 64)));
    assert_eq!(turn.visual_token_budget, Some(280));
    let sampler = turn.sampler.expect("a sampler");
    assert_eq!(sampler.top_k, Some(40));
    assert_eq!(sampler.top_p, Some(0.9));
    assert_eq!(sampler.temperature, Some(0.7));
    assert_eq!(sampler.seed, Some(7));

    let engines = fake.engines();
    assert_eq!(engines.len(), 1);
    assert_eq!(engines[0].backend, "cpu");
    assert!(engines[0].benchmark, "token counts need benchmarking on");
}

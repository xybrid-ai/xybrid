//! End-to-end cancellation coverage at the Bolt foreign-binding boundary.
//!
//! Run with:
//!   cargo test -p integration-tests --features llm-llamacpp \
//!     --test native_cancellation_integration -- --nocapture
//!
//! Download the model first:
//!   ./integration-tests/download.sh functiongemma-270m-it

#![cfg(feature = "llm-llamacpp")]

use std::time::{Duration, Instant};

use integration_tests::fixtures;
use xybrid_bolt::{
    XybridCancellationToken, XybridConversationContext, XybridEnvelope, XybridEnvelopeKind,
    XybridModel, XybridStreamEventKind,
};

const MODEL: &str = "functiongemma-270m-it";

#[test]
fn bolt_cancellation_reaches_real_llama_batch_context_and_stream_paths() {
    let Some(model_dir) = fixtures::model_for_test(MODEL) else {
        eprintln!("Skipping: {MODEL} not downloaded. Run: ./integration-tests/download.sh {MODEL}");
        return;
    };

    let model = XybridModel::from_directory(model_dir.to_string_lossy().into_owned())
        .expect("the FunctionGemma fixture should load through Bolt");

    let pre_cancelled = XybridCancellationToken::new();
    pre_cancelled.cancel();
    let started = Instant::now();
    let batch_error = match model.run(prompt(), None, &pre_cancelled) {
        Ok(_) => panic!("a pre-cancelled batch run must not enter inference"),
        Err(error) => error,
    };
    assert_cancelled(batch_error);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "pre-cancellation should short-circuit before model execution"
    );

    let context = XybridConversationContext::new();
    let context_error = match model.run_with_context(prompt(), &context, None, &pre_cancelled) {
        Ok(_) => panic!("context runs must receive the same cancellation token"),
        Err(error) => error,
    };
    assert_cancelled(context_error);

    let in_flight = XybridCancellationToken::new();
    let cancellation_latency = std::thread::scope(|scope| {
        let worker = scope.spawn(|| model.run(prompt(), None, &in_flight));

        // Give prompt evaluation and generation time to enter the native
        // backend. This specifically distinguishes in-flight cancellation
        // from the pre-run checks above.
        std::thread::sleep(Duration::from_millis(250));
        let cancelled_at = Instant::now();
        in_flight.cancel();

        let error = match worker.join().expect("the batch worker must not panic") {
            Ok(_) => panic!("an in-flight batch run ignored cancellation"),
            Err(error) => error,
        };
        assert_cancelled(error);
        cancelled_at.elapsed()
    });
    assert!(
        cancellation_latency < Duration::from_secs(2),
        "batch cancellation should stop at a token boundary, not max_tokens"
    );

    let context_in_flight = XybridCancellationToken::new();
    let context_cancellation_latency = std::thread::scope(|scope| {
        let worker =
            scope.spawn(|| model.run_with_context(prompt(), &context, None, &context_in_flight));

        std::thread::sleep(Duration::from_millis(250));
        let cancelled_at = Instant::now();
        context_in_flight.cancel();

        let error = match worker
            .join()
            .expect("the context batch worker must not panic")
        {
            Ok(_) => panic!("an in-flight context run ignored cancellation"),
            Err(error) => error,
        };
        assert_cancelled(error);
        cancelled_at.elapsed()
    });
    assert!(
        context_cancellation_latency < Duration::from_secs(2),
        "context cancellation should stop at a token boundary, not max_tokens"
    );

    let streaming = XybridCancellationToken::new();
    let stream_id = model
        .run_stream(prompt(), None, &streaming)
        .expect("the real llama stream should start");
    let first = model
        .stream_next(stream_id)
        .expect("the stream should produce at least one token before cancellation");
    assert_eq!(first.kind, XybridStreamEventKind::Token);

    streaming.cancel();
    let stream_error = loop {
        match model.stream_next(stream_id) {
            Ok(event) if event.kind == XybridStreamEventKind::Token => continue,
            Ok(_) => panic!("a cancelled stream completed instead of reporting cancellation"),
            Err(error) => break error,
        }
    };
    assert_cancelled(stream_error);

    let close_token = XybridCancellationToken::new();
    let close_stream = model
        .run_stream(prompt(), None, &close_token)
        .expect("a second stream should start on the same loaded model");
    model.stream_close(close_stream);
    assert!(
        close_token.is_cancelled(),
        "closing a raw pull stream must cancel its native worker"
    );
}

fn prompt() -> XybridEnvelope {
    XybridEnvelope {
        kind: XybridEnvelopeKind::Text {
            text: "Count from one to one hundred, spelling out every number.".into(),
        },
        metadata: Vec::new(),
    }
}

fn assert_cancelled(error: xybrid_bolt::XybridError) {
    let message = format!("{error:?}").to_ascii_lowercase();
    assert!(
        message.contains("cancel"),
        "expected a cancellation error, got {error:?}"
    );
}

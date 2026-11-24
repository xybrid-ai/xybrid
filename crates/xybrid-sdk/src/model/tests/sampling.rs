//! Runs whose spans are not exported still count in metrics (reliability C1,
//! C3, C4).
//!
//! - A suppressed run (`TraceMode::Suppress`) exports no span, not even a
//!   stage, and still records one duration point with its outcome and the
//!   usage its runtime reported from inside a core stage.
//! - A run under an incoming unsampled parent (`traceparent` flags `00`)
//!   exports no span either, keeps a reference in the parent's trace, counts
//!   exactly, and its reference takes evaluations and feedback.
//! - Live frames: the first frame of each `live_trace_interval` of a frame
//!   session is traced, every frame is counted. The session's live clock is
//!   driven by the test, so which frame opens an interval never depends on
//!   how fast the test runs.
//!
//! Each test whose runs export no span ends with a traced anchor run on the
//! same session, which must export: what is missing was dropped, not lost.
//!
//! Every run is pinned to its session. Only the unsampled-parent test
//! installs a process default, under the default-session lock, to record
//! evaluations through the public free functions. It installs a second
//! session of its own, not the run's: a run pinned to no session in a test
//! beside it records into that default too (the test makes one), so it reads
//! back only the records linked to its run's span there.

use super::*;
use crate::observability::semconv::{self, attr, metric};
use crate::observability::test_support::{
    default_session_lock, memory_session_with_signals, memory_session_with_signals_configured,
    parent_id, span_id, trace_id, DefaultSession, Signals,
};
use crate::observability::{
    Evaluation, Feedback, ObservationContext, ObservationOptions, ObservationRef, TraceMode,
};
use std::collections::{BTreeMap, VecDeque};
use xybrid_core::runtime_adapter::AdapterError;

/// How a run ends.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Outcome {
    /// Succeeds after its runtime reported usage from a core stage.
    Done,
    /// Fails with an inference error.
    Fail,
    /// Ends cancelled.
    Cancel,
}

impl Outcome {
    /// The `error.type` the run is counted under.
    fn error_type(self) -> Option<&'static str> {
        match self {
            Self::Done => None,
            Self::Fail => Some("inference_error"),
            Self::Cancel => Some("cancelled"),
        }
    }
}

/// Reports usage the way a local runtime does: metadata on a core stage
/// running on the run's thread.
fn report_usage(tokens_in: Option<u64>, tokens_out: Option<u64>) {
    let _stage = xybrid_core::tracing::SpanGuard::new("fake_decode");
    if let Some(n) = tokens_in {
        xybrid_core::tracing::add_metadata("tokens_in", n.to_string());
    }
    if let Some(n) = tokens_out {
        xybrid_core::tracing::add_metadata("tokens_out", n.to_string());
    }
}

/// Fake `"onnx"` runtime: each `execute()` reports 3 input and 5 output
/// tokens from a core stage and succeeds, or fails when scripted to.
struct UsageRuntime {
    /// Whether each next `execute()` fails; once empty, every call succeeds.
    failures: VecDeque<bool>,
}

impl xybrid_core::runtime_adapter::ModelRuntime for UsageRuntime {
    fn name(&self) -> &str {
        "onnx"
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["onnx"]
    }

    fn load(
        &mut self,
        _model_path: &std::path::Path,
    ) -> xybrid_core::runtime_adapter::AdapterResult<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(
        &mut self,
        _input: &xybrid_core::ir::Envelope,
    ) -> xybrid_core::runtime_adapter::AdapterResult<xybrid_core::ir::Envelope> {
        if self.failures.pop_front().unwrap_or(false) {
            return Err(AdapterError::RuntimeError("decoder failed".into()));
        }
        report_usage(Some(3), Some(5));
        Ok(text_envelope("done"))
    }
}

/// A chat model named `id` over [`UsageRuntime`] scripted with `failures`.
fn usage_model(id: &str, failures: &[bool]) -> XybridModel {
    let mut model = as_chat(test_model_with_runtime(Box::new(UsageRuntime {
        failures: failures.iter().copied().collect(),
    })));
    model.model_id = id.to_owned();
    model
}

/// Run options pinned to the test's session, with `observation`.
fn pinned(signals: &Signals, observation: ObservationOptions) -> RunOptions {
    RunOptions::new().with_observation(observation.with_session(&signals.session))
}

fn suppressed() -> ObservationOptions {
    ObservationOptions {
        trace: TraceMode::Suppress,
        ..ObservationOptions::default()
    }
}

/// Count (histograms) or value (counters) of the metric `name`, summed per
/// `error.type`.
fn by_error_type(signals: &Signals, name: &str) -> BTreeMap<Option<String>, u64> {
    let mut out = BTreeMap::new();
    for point in signals.points(name) {
        *out.entry(point.attr(attr::ERROR_TYPE).map(str::to_owned))
            .or_default() += point.count;
    }
    out
}

/// The expected [`by_error_type`] for `runs` counted once each.
fn counted(runs: &[Outcome]) -> BTreeMap<Option<String>, u64> {
    let mut out = BTreeMap::new();
    for run in runs {
        *out.entry(run.error_type().map(str::to_owned)).or_default() += 1;
    }
    out
}

/// The sum of every point of the counter `name`, asserting each is for text.
fn text_tokens(signals: &Signals, name: &str) -> u64 {
    signals
        .points(name)
        .iter()
        .map(|p| {
            assert_eq!(p.attr(attr::TOKEN_MODALITY), Some("text"), "{p:?}");
            p.count
        })
        .sum()
}

/// Trace id, span id and whether the sampled flag is set, of a log record
/// linked to a span.
type Linked = (String, String, bool);

/// Event names and trace context of every emitted log record.
fn log_records(signals: &Signals) -> Vec<(String, Option<Linked>)> {
    signals
        .logs
        .get_emitted_logs()
        .expect("logs")
        .iter()
        .map(|l| {
            (
                l.record.event_name().unwrap_or_default().to_owned(),
                l.record.trace_context().map(|t| {
                    (
                        t.trace_id.to_string(),
                        t.span_id.to_string(),
                        t.trace_flags.is_some_and(|f| f.is_sampled()),
                    )
                }),
            )
        })
        .collect()
}

/// Asserts the only spans exported so far are those of `anchor`, a traced
/// root run that reported usage: its run span and its `fake_decode` stage.
fn assert_only_anchor_exported(signals: &Signals, anchor: &ObservationRef) {
    let spans = signals.spans.spans();
    let exported: Vec<_> = spans
        .iter()
        .map(|s| (s.name.to_string(), trace_id(s)))
        .collect();
    assert!(
        spans.iter().all(|s| trace_id(s) == anchor.trace_id),
        "only the anchor's trace is exported, have {exported:?} for {anchor:?}"
    );
    signals.spans.by_ref(anchor);
    signals.spans.named("fake_decode");
}

#[test]
fn suppressed_runs_export_no_span_but_count_outcomes_and_usage() {
    // C1 through RunObservation: the stage recorder harvests usage while the
    // run and its stages are suppressed.
    let signals = memory_session_with_signals();
    let model = as_chat(test_loaded_model(true));
    let options = pinned(&signals, suppressed());
    let observe = || model.observe(&text_envelope("hi"), &[], Some(&options), None, false, None);

    let run = observe();
    assert!(run.context().is_none(), "a suppressed run has no context");
    report_usage(Some(3), Some(5));
    assert_eq!(run.succeed(&text_envelope("done")), None);

    let run = observe();
    assert_eq!(run.succeed_without_output(), None);

    let run = observe();
    report_usage(Some(7), None);
    run.fail(&SdkError::inference("runtime failed"));

    let run = observe();
    assert_eq!(run.cancel(), None);

    assert_eq!(
        by_error_type(&signals, metric::INFERENCE_DURATION),
        counted(&[Outcome::Done, Outcome::Done, Outcome::Fail, Outcome::Cancel]),
        "one duration point per run, by outcome"
    );
    assert_eq!(text_tokens(&signals, metric::USAGE_INPUT_TOKENS), 3 + 7);
    assert_eq!(text_tokens(&signals, metric::USAGE_OUTPUT_TOKENS), 5);
    assert_eq!(
        by_error_type(&signals, metric::OPERATION_INPUT_TOKENS),
        counted(&[Outcome::Done, Outcome::Fail]),
        "input usage of the successful and the failed run"
    );
    assert_eq!(
        log_records(&signals),
        [(semconv::event::OPERATION_EXCEPTION.to_owned(), None)],
        "the failure's exception is the only record, attached to no span"
    );

    // Anchor, after the metrics were read: the same path, traced, exports
    // its run and its stage, and nothing else was exported.
    let anchor = model.observe(
        &text_envelope("hi"),
        &[],
        Some(&pinned(&signals, ObservationOptions::default())),
        None,
        false,
        None,
    );
    report_usage(Some(3), Some(5));
    let anchor = anchor
        .succeed(&text_envelope("done"))
        .expect("a traced run has a reference");
    assert_only_anchor_exported(&signals, &anchor);
}

#[test]
fn suppressed_model_runs_export_no_span_but_count_outcomes_and_usage() {
    // C1 through the automatic run path: `run_with_options` over a runtime
    // that reports usage from a core stage, then fails once.
    let signals = memory_session_with_signals();
    let model = usage_model("suppressed-runs", &[false, true, false]);
    let options = pinned(&signals, suppressed());

    let first = model
        .run_with_options(&text_envelope("hi"), &options)
        .expect("ok");
    let failed = model.run_with_options(&text_envelope("hi"), &options);
    let last = model
        .run_with_options(&text_envelope("hi"), &options)
        .expect("ok");
    assert!(failed.is_err(), "the runtime fails the second run");
    assert_eq!(
        first.observation_ref(),
        None,
        "a suppressed run has no reference"
    );
    assert_eq!(last.observation_ref(), None);

    assert_eq!(
        by_error_type(&signals, metric::INFERENCE_DURATION),
        counted(&[Outcome::Done, Outcome::Fail, Outcome::Done]),
    );
    assert_eq!(text_tokens(&signals, metric::USAGE_INPUT_TOKENS), 2 * 3);
    assert_eq!(text_tokens(&signals, metric::USAGE_OUTPUT_TOKENS), 2 * 5);
    assert_eq!(
        log_records(&signals),
        [(semconv::event::OPERATION_EXCEPTION.to_owned(), None)],
    );

    // Anchor: the same model, traced, exports its run and its runtime stage;
    // no run and no stage of the suppressed runs was exported.
    let anchor = model
        .run_with_options(
            &text_envelope("hi"),
            &pinned(&signals, ObservationOptions::default()),
        )
        .expect("ok");
    assert_only_anchor_exported(
        &signals,
        anchor
            .observation_ref()
            .expect("a traced run has a reference"),
    );
}

#[test]
fn an_unsampled_parent_run_keeps_its_reference_counts_and_evaluations() {
    // C3 through the automatic run path: under a `-00` parent the run and
    // its runtime stage are dropped, its reference is in the parent's trace,
    // its metrics are exact, and evaluations and feedback recorded through
    // the public `observability::record_*` functions link to it.
    const TRACE: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
    const PARENT_SPAN: &str = "00f067aa0ba902b7";
    let signals = memory_session_with_signals();
    let model = usage_model("unsampled-parent", &[]);
    let parent =
        ObservationContext::from_traceparent(&format!("00-{TRACE}-{PARENT_SPAN}-00"), None)
            .unwrap();
    let options = pinned(
        &signals,
        ObservationOptions::default().with_parent(parent.clone()),
    );

    let result = model
        .run_with_options(&text_envelope("hi"), &options)
        .expect("ok");
    let reference: ObservationRef = result
        .observation_ref()
        .cloned()
        .expect("an unsampled reference");
    assert_eq!(reference.trace_id, TRACE, "in the parent's trace");
    assert_ne!(reference.span_id, PARENT_SPAN, "a span of its own");
    let linked = Some((TRACE.to_owned(), reference.span_id.clone(), false));
    {
        // The public free functions record on the process default, which
        // also takes the records of any run pinned to no session meanwhile.
        let evaluations = memory_session_with_signals();
        let _default = default_session_lock();
        let _installed = DefaultSession::install(&evaluations.session);
        run_an_unpinned_neighbour_that_fails();
        assert!(crate::observability::record_evaluation(
            &reference,
            &Evaluation {
                name: "relevance".into(),
                score_value: Some(0.5),
                ..Evaluation::default()
            }
        ));
        assert!(crate::observability::record_feedback(
            &reference,
            &Feedback {
                score: Some(1.0),
                ..Feedback::default()
            }
        ));
        let of_the_run: Vec<_> = log_records(&evaluations)
            .into_iter()
            .filter(|(_, link)| {
                link.as_ref()
                    .is_some_and(|(_, span, _)| *span == reference.span_id)
            })
            .collect();
        assert_eq!(
            of_the_run,
            [
                (semconv::event::EVALUATION_RESULT.to_owned(), linked.clone()),
                (semconv::event::EVALUATION_RESULT.to_owned(), linked),
            ],
            "evaluation and feedback on the run's reference, trace flags 0"
        );
    }

    assert_eq!(
        by_error_type(&signals, metric::INFERENCE_DURATION),
        counted(&[Outcome::Done])
    );
    assert_eq!(text_tokens(&signals, metric::USAGE_INPUT_TOKENS), 3);
    assert_eq!(text_tokens(&signals, metric::USAGE_OUTPUT_TOKENS), 5);
    let own = log_records(&signals);
    assert!(own.is_empty(), "the run itself emits no record: {own:?}");

    // Anchor: the same model as a traced root run exports its run and its
    // runtime stage; nothing of the unsampled run was exported.
    let anchor = model
        .run_with_options(
            &text_envelope("hi"),
            &pinned(&signals, ObservationOptions::default()),
        )
        .expect("ok");
    let anchor = anchor
        .observation_ref()
        .expect("a traced run has a reference");
    assert_ne!(anchor.trace_id, TRACE, "the anchor is a root of its own");
    assert_only_anchor_exported(&signals, anchor);
}

const CAM_1: &str = "cam-1";
const CAM_2: &str = "cam-2";

/// Runs one live frame of `session` at the live clock's current time and
/// ends it with `outcome`; returns whether the frame was traced.
fn live_frame(model: &XybridModel, signals: &Signals, session: &str, outcome: Outcome) -> bool {
    let options = pinned(signals, ObservationOptions::default()).with_frame_session(session);
    let run = model.observe(
        &text_envelope("frame"),
        &[],
        Some(&options),
        None,
        true,
        None,
    );
    let traced = run.context().is_some();
    let reference = match outcome {
        Outcome::Done => {
            report_usage(Some(4), Some(2));
            run.succeed(&text_envelope("seen"))
        }
        Outcome::Fail => {
            run.fail(&SdkError::inference("frame failed"));
            return traced;
        }
        Outcome::Cancel => run.cancel(),
    };
    assert_eq!(reference.is_some(), traced, "{session} {outcome:?}");
    traced
}

#[test]
fn live_frames_trace_the_first_frame_per_interval_and_count_every_frame() {
    // C4: two frame sessions over three intervals, through RunObservation.
    use Outcome::{Cancel, Done, Fail};
    const INTERVAL: Duration = Duration::from_secs(10);
    let mut signals =
        memory_session_with_signals_configured(|config| config.live_trace_interval(INTERVAL));
    let start = Instant::now();
    let clock = Arc::new(Mutex::new(start));
    signals.session.drive_live_clock(clock.clone());
    let model = as_chat(test_loaded_model(true));
    let run_span = run_span_name(&model);

    let s = Duration::from_secs;
    // Inside the interval a session's last traced frame opened.
    let just_before = |k: u32| INTERVAL * k - Duration::from_nanos(1);
    // (arrival, frame session, outcome, traced)
    let frames = [
        (s(0), CAM_1, Done, true),
        (s(1), CAM_1, Fail, false),
        (s(2), CAM_2, Done, true),
        (s(3), CAM_2, Cancel, false),
        (s(5), CAM_1, Done, false),
        (just_before(1), CAM_1, Cancel, false),
        (INTERVAL, CAM_1, Fail, true),
        (INTERVAL + s(1), CAM_1, Done, false),
        (INTERVAL + s(2), CAM_2, Cancel, true),
        (INTERVAL + s(3), CAM_2, Done, false),
        (INTERVAL + s(5), CAM_1, Done, false),
        (just_before(2), CAM_1, Fail, false),
        (INTERVAL * 2, CAM_1, Done, true),
        (INTERVAL * 2 + s(1), CAM_1, Cancel, false),
        (INTERVAL * 2 + s(2), CAM_2, Done, true),
        (INTERVAL * 2 + s(3), CAM_2, Fail, false),
        (INTERVAL * 2 + s(5), CAM_1, Done, false),
        (just_before(3), CAM_1, Done, false),
    ];
    for (i, (at, session, outcome, traced)) in frames.iter().enumerate() {
        *clock.lock().unwrap() = start + *at;
        assert_eq!(
            live_frame(&model, &signals, session, *outcome),
            *traced,
            "frame {i}: {session} at {at:?}"
        );
    }

    // One run span per interval per session: each opening frame's, in order.
    let spans = signals.spans.spans();
    let runs: Vec<_> = spans.iter().filter(|s| s.name == run_span).collect();
    let traced: Vec<_> = runs
        .iter()
        .map(|s| {
            (
                crate::observability::test_support::attr(s, "xybrid.live.session_id"),
                crate::observability::test_support::attr(s, attr::ERROR_TYPE),
            )
        })
        .collect();
    let expected: Vec<_> = frames
        .iter()
        .filter(|(_, _, _, traced)| *traced)
        .map(|(_, session, outcome, _)| {
            (
                Some((*session).to_owned()),
                outcome.error_type().map(str::to_owned),
            )
        })
        .collect();
    assert_eq!(traced, expected);
    // Only traced frames export their runtime stage.
    let run_ids: Vec<_> = runs.iter().map(|s| span_id(s)).collect();
    let stages: Vec<_> = spans.iter().filter(|s| s.name != run_span).collect();
    for stage in &stages {
        assert!(
            run_ids.contains(&parent_id(stage)),
            "{} is not a stage of a traced frame",
            stage.name
        );
    }
    let traced_done = frames
        .iter()
        .filter(|(_, _, outcome, traced)| *traced && *outcome == Done)
        .count();
    assert_eq!(stages.len(), traced_done, "one stage per traced success");

    // Every frame is counted, by outcome, with its usage.
    let outcomes: Vec<_> = frames.iter().map(|(_, _, outcome, _)| *outcome).collect();
    let done = outcomes.iter().filter(|o| **o == Done).count() as u64;
    assert_eq!(
        by_error_type(&signals, metric::INFERENCE_DURATION),
        counted(&outcomes)
    );
    assert_eq!(text_tokens(&signals, metric::USAGE_INPUT_TOKENS), 4 * done);
    assert_eq!(text_tokens(&signals, metric::USAGE_OUTPUT_TOKENS), 2 * done);
    assert_eq!(
        by_error_type(&signals, metric::OPERATION_OUTPUT_TOKENS),
        BTreeMap::from([(None, done)])
    );
}

/// Runs four frames of one frame session at a single instant of a frozen
/// live clock under `interval`; returns whether each was traced, and how
/// many run spans were exported.
fn frames_at_one_instant(interval: Duration) -> (Vec<bool>, usize) {
    let mut signals =
        memory_session_with_signals_configured(|config| config.live_trace_interval(interval));
    signals
        .session
        .drive_live_clock(Arc::new(Mutex::new(Instant::now())));
    let model = as_chat(test_loaded_model(true));
    let traced = [Outcome::Done, Outcome::Fail, Outcome::Cancel, Outcome::Done]
        .into_iter()
        .map(|outcome| live_frame(&model, &signals, CAM_1, outcome))
        .collect();
    let run_span = run_span_name(&model);
    let runs = signals
        .spans
        .spans()
        .iter()
        .filter(|s| s.name == run_span)
        .count();
    (traced, runs)
}

#[test]
fn a_zero_live_interval_traces_every_frame() {
    // `now - last < 0` never holds: every frame opens an interval, even with
    // the clock standing still.
    assert_eq!(frames_at_one_instant(Duration::ZERO), (vec![true; 4], 4));
    // Control: the same frames go through the live sampler. One nanosecond
    // later than zero, only the first frame opens an interval.
    assert_eq!(
        frames_at_one_instant(Duration::from_nanos(1)),
        (vec![true, false, false, false], 1)
    );
}

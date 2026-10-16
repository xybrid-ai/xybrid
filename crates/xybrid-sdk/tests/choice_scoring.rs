//! Choice requests through the SDK's public surface.
//!
//! A choice scorer accepts one input, a `ChoiceRequest`, through one kind of
//! entry point, a batch run, where it scores the candidates. Everything else
//! is refused before any cloud leg or backend call: a choice request sent to
//! another model, other input sent to a scorer, every streaming and
//! conversation entry point, and a choice kind buried in a multi-part message
//! or in conversation history.
//!
//! The scorers run the fake ONNX scorers of
//! `integration-tests/fixtures/choice/fake/`, copied under the FORMS spec's
//! file name. Tests that run one skip without ONNX Runtime, unless
//! `XYBRID_REQUIRE_MODELS` is set, where they fail. The chat model is a
//! directory holding only `model_metadata.json`, so a path that tried to open
//! weights would fail with a load error instead of the refusal each test
//! expects.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use tokio_stream::StreamExt;
use xybrid_core::execution::choice::onnx::OnnxChoiceScorer;
use xybrid_core::execution::template::ChoiceScorerSpec;
use xybrid_core::runtime_adapter::onnx::{ExecutionProviderKind, ONNXSession, SessionOptions};
use xybrid_core::runtime_adapter::{AdapterResult, CloudStreaming};
use xybrid_core::testing::model_fixtures::{models_required, ENV_REQUIRE_MODELS};
use xybrid_sdk::ir::{Envelope, EnvelopeKind, MessageRole};
use xybrid_sdk::{
    CancellationToken, Choice, ChoiceScore, ChoiceScores, ConversationContext, InferenceResult,
    ModelLoader, OutputType, PartialToken, RunOptions, SdkError, StreamConfig, StreamEvent,
    StreamingCallback, XybridModel,
};

const CANARY: &str = "canary-9c41d7";

type TokenResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn choice_fixtures() -> std::path::PathBuf {
    // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs in
    // a sandbox where the compile-time path points nowhere.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
    std::path::Path::new(&manifest_dir).join("../../integration-tests/fixtures/choice")
}

fn forms_spec() -> serde_json::Value {
    let path = choice_fixtures().join("specs/cua-s1-forms.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the FORMS spec is JSON")
}

/// Whether ONNX Runtime can run here: it is linked in on Apple platforms and
/// loaded from `ORT_DYLIB_PATH` elsewhere. Without it a test that runs a
/// scorer skips, unless models are required, where it fails instead.
fn onnx_runtime() -> bool {
    let available = cfg!(any(target_os = "macos", target_os = "ios"))
        || std::env::var_os("ORT_DYLIB_PATH")
            .is_some_and(|path| std::path::Path::new(&path).is_file());
    if !available {
        assert!(
            !models_required(),
            "{ENV_REQUIRE_MODELS} is set but ONNX Runtime is not loadable; set ORT_DYLIB_PATH"
        );
        eprintln!("Skipping: ONNX Runtime is not loadable (set ORT_DYLIB_PATH)");
    }
    available
}

fn model_dir(id: &str, template: serde_json::Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let metadata = serde_json::json!({
        "model_id": id,
        "version": "1.0",
        "execution_template": template,
        "files": [],
    });
    std::fs::write(dir.path().join("model_metadata.json"), metadata.to_string()).unwrap();
    dir
}

/// A metadata-only model; the directory must outlive it.
fn load(id: &str, template: serde_json::Value) -> (tempfile::TempDir, XybridModel) {
    let dir = model_dir(id, template);
    let model = ModelLoader::from_directory(dir.path())
        .unwrap()
        .load()
        .unwrap();
    (dir, model)
}

/// A scorer with the FORMS spec whose model file is a copy of the fake scorer
/// `fake`; the directory must outlive it.
fn scorer_with(fake: &str) -> (tempfile::TempDir, XybridModel) {
    scorer_with_spec(fake, forms_spec())
}

/// A scorer with `spec` whose model file is a copy of the fake scorer `fake`;
/// the directory must outlive it.
fn scorer_with_spec(fake: &str, spec: serde_json::Value) -> (tempfile::TempDir, XybridModel) {
    let dir = model_dir(
        "forms-scorer",
        serde_json::json!({ "type": "ChoiceScorer", "scorer": spec }),
    );
    let model_file = spec["model_file"]
        .as_str()
        .expect("the spec names its model");
    std::fs::copy(
        choice_fixtures().join("fake").join(fake),
        dir.path().join(model_file),
    )
    .unwrap();
    let model = ModelLoader::from_directory(dir.path())
        .unwrap()
        .load()
        .unwrap();
    (dir, model)
}

fn scorer() -> (tempfile::TempDir, XybridModel) {
    scorer_with("dynamic.onnx")
}

/// What the fake scorer `fake` returns for `input` when run straight through
/// xybrid-core, which the SDK must reproduce.
fn scored_directly(fake: &str, input: &Envelope) -> ChoiceScores {
    let Ok(ChoiceScorerSpec::OnnxByteOptionScorer(spec)) = serde_json::from_value(forms_spec())
    else {
        panic!("the FORMS spec is an ONNX byte scorer");
    };
    let EnvelopeKind::ChoiceRequest(request) = &input.kind else {
        panic!("a choice request");
    };
    let EnvelopeKind::Text(context) = &request.context().kind else {
        panic!("a text context");
    };
    let path = choice_fixtures().join("fake").join(fake);
    let session = ONNXSession::build(
        path.to_str().unwrap(),
        ExecutionProviderKind::Cpu,
        SessionOptions::default(),
    )
    .unwrap();
    OnnxChoiceScorer::bind(&session, &spec)
        .unwrap()
        .score(&session, context, request.choices())
        .unwrap()
}

fn chat_model() -> (tempfile::TempDir, XybridModel) {
    load(
        "chat",
        serde_json::json!({ "type": "Gguf", "model_file": "model.gguf" }),
    )
}

fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().unwrap())
}

fn text(content: &str) -> Envelope {
    Envelope::new(EnvelopeKind::Text(content.to_string()))
}

fn request() -> Envelope {
    Envelope::choice_request(
        text("FORM Intake\nELEMENT Edit \"Phone\" value=\"\""),
        vec![
            Choice::new("tel", "fill Tel: (503) 555-0142"),
            Choice::new("name", "fill Name: Jane Doe"),
        ],
    )
}

fn scores() -> Envelope {
    Envelope::new(EnvelopeKind::ChoiceScores(ChoiceScores {
        entries: vec![ChoiceScore {
            id: "tel".to_string(),
            logit: 1.0,
            score: 1.0,
            fixed: false,
        }],
        label_mass: None,
    }))
}

/// Every choice kind a model that is not a scorer must refuse.
fn choice_kinds() -> [Envelope; 3] {
    [
        request(),
        Envelope::new(EnvelopeKind::MultiPart(vec![text("look"), request()])),
        scores(),
    ]
}

/// The capability an `UnsupportedModelCapability` refusal names.
#[track_caller]
fn capability<T: std::fmt::Debug>(result: Result<T, SdkError>) -> String {
    match result {
        Err(SdkError::UnsupportedModelCapability { capability, .. }) => capability,
        other => panic!("expected UnsupportedModelCapability, got {other:?}"),
    }
}

fn no_token(_token: PartialToken) -> TokenResult {
    panic!("a refused request must not stream a token")
}

/// Cloud leg that counts calls; a refused request must never reach it.
#[derive(Default)]
struct CountingCloud {
    calls: AtomicUsize,
}

impl CloudStreaming for CountingCloud {
    fn execute_streaming(
        &self,
        input: &Envelope,
        _on_token: StreamingCallback<'_>,
    ) -> AdapterResult<Envelope> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(input.clone())
    }
}

/// Runs `input` through the batch entry points (`run`, `run_with_options`,
/// `run_async`) and returns the capability each refused it with.
fn batch_refusals(model: &XybridModel, input: &Envelope) -> Vec<String> {
    vec![
        capability(model.run(input, None)),
        capability(model.run_with_options(input, &RunOptions::new())),
        capability(runtime().block_on(model.run_async(input, None))),
    ]
}

/// Runs `input` through every public run entry point but the batch ones and
/// returns the capability each refused it with, in a fixed order.
fn refusals(model: &XybridModel, input: &Envelope, context: &ConversationContext) -> Vec<String> {
    let options = RunOptions::new();
    let preempt = RunOptions::new().with_cancellation_token(CancellationToken::new());
    let cloud = CountingCloud::default();

    let streamed: Vec<StreamEvent> = runtime().block_on(async {
        model
            .run_stream(input.clone(), None)
            .collect::<Vec<_>>()
            .await
    });
    let run_stream = match streamed.as_slice() {
        [StreamEvent::Error(message)] => message.clone(),
        other => panic!("run_stream must yield one error event, got {other:?}"),
    };

    let refused = vec![
        capability(model.run_with_context(input, context, None)),
        capability(model.run_with_context_options(input, context, &options)),
        capability(model.run_streaming(input, None, no_token)),
        capability(model.run_streaming_with_options(input, &options, no_token)),
        capability(model.run_streaming_with_options_preempt(input, &preempt, true, no_token)),
        capability(model.run_streaming_with_context(input, context, None, no_token)),
        capability(model.run_streaming_with_context_options(input, context, &options, no_token)),
        capability(
            model.run_streaming_with_context_options_preempt(
                input, context, &preempt, true, no_token,
            ),
        ),
        capability(model.run_tts_streaming(input, &options, |_, _| {
            panic!("a refused request must not synthesize")
        })),
        capability(model.run_streaming_with_fallback(
            input,
            &options,
            &cloud,
            &mut no_token,
            &mut |_| panic!("a refused request has no cloud seam"),
        )),
    ];
    assert_eq!(cloud.calls.load(Ordering::SeqCst), 0, "the cloud leg ran");
    assert!(
        run_stream.contains("Unsupported model capability"),
        "{run_stream}"
    );
    refused
}

// ---------------------------------------------------------------------------
// Loading a scorer
// ---------------------------------------------------------------------------

#[test]
fn a_scorer_loads_and_describes_itself() {
    let (_dir, model) = scorer();
    assert!(model.is_choice_scorer());
    let fixed: Vec<String> = model.fixed_choices().into_iter().map(|c| c.id).collect();
    assert_eq!(fixed, ["check", "click", "skip"]);
    assert_eq!(model.output_type(), OutputType::ChoiceScores);
    assert!(!model.supports_streaming());
    assert!(!model.supports_token_streaming());
    assert!(!model.is_llm());

    let (_dir, chat) = chat_model();
    assert!(!chat.is_choice_scorer());
    assert!(chat.fixed_choices().is_empty());
}

#[test]
fn an_invalid_scorer_spec_fails_at_load() {
    let mut spec = forms_spec();
    spec["max_choices"] = 1.into();
    let dir = model_dir(
        "bad-scorer",
        serde_json::json!({ "type": "ChoiceScorer", "scorer": spec }),
    );
    let error = ModelLoader::from_directory(dir.path())
        .unwrap()
        .load()
        .err()
        .expect("an invalid spec must not load");
    match error {
        SdkError::MetadataInvalid(message) => {
            assert!(message.contains("max_choices"), "{message}")
        }
        other => panic!("expected MetadataInvalid, got {other:?}"),
    }
}

#[test]
fn choice_scores_results_are_typed() {
    let result = InferenceResult::new(scores(), "forms-scorer", 3);
    assert_eq!(result.output_type(), OutputType::ChoiceScores);
    assert_eq!(result.output_type().to_string(), "choice_scores");
    let scores = result.choice_scores().expect("scores");
    assert_eq!(scores.best(), Some(0));
    assert!(result.text().is_none());

    let text_result = InferenceResult::new(text("hello"), "chat", 3);
    assert!(text_result.choice_scores().is_none());
}

// ---------------------------------------------------------------------------
// The scorer's own entry point
// ---------------------------------------------------------------------------

#[test]
fn valid_requests_are_scored_through_every_batch_entry() {
    if !onnx_runtime() {
        return;
    }
    let (_dir, model) = scorer();
    let expected = scored_directly("dynamic.onnx", &request());
    let ids: Vec<&str> = expected.entries.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, ["tel", "name", "check", "click", "skip"]);

    let results = [
        model.run(&request(), None).unwrap(),
        model
            .run_with_options(&request(), &RunOptions::new())
            .unwrap(),
        runtime()
            .block_on(model.run_async(&request(), None))
            .unwrap(),
    ];
    for result in results {
        assert_eq!(result.output_type(), OutputType::ChoiceScores);
        assert_eq!(result.choice_scores(), Some(&expected));
        assert!(result.text().is_none());
    }

    // FORMS has three fixed choices, so a request may offer none.
    let fixed_only = Envelope::choice_request(text("FORM Intake"), Vec::new());
    assert_eq!(
        model.run(&fixed_only, None).unwrap().choice_scores(),
        Some(&scored_directly("dynamic.onnx", &fixed_only))
    );
}

#[test]
fn scorers_warm_up_on_a_request_they_accept() {
    if !onnx_runtime() {
        return;
    }
    let (_dir, model) = scorer();
    model.warmup().unwrap();
    runtime().block_on(model.warmup_async()).unwrap();
    // Warming up changes no later score.
    assert_eq!(
        model.run(&request(), None).unwrap().choice_scores(),
        Some(&scored_directly("dynamic.onnx", &request()))
    );
}

#[test]
fn scorers_keep_their_own_models() {
    if !onnx_runtime() {
        return;
    }
    // Two scorers on one spec, in one process: the byte-table model fails on
    // letters, the dynamic one does not, whichever ran first.
    let (_dynamic_dir, dynamic) = scorer();
    let (_table_dir, table) = scorer_with("byte_table.onnx");
    for _ in 0..2 {
        assert!(dynamic.run(&request(), None).is_ok());
        let error = table
            .run(&request(), None)
            .expect_err("the byte table refuses letters");
        assert!(
            matches!(&error, SdkError::InferenceError { .. })
                && format!("{error:?}").contains("withheld"),
            "{error:?}"
        );
    }
}

#[test]
fn invalid_requests_are_inference_errors_without_caller_content() {
    let (_dir, model) = scorer();
    let invalid = [
        Envelope::choice_request(text(""), vec![Choice::new(CANARY, CANARY)]),
        Envelope::choice_request(
            text(CANARY),
            vec![Choice::new(CANARY, CANARY), Choice::new(CANARY, CANARY)],
        ),
        Envelope::choice_request(text(CANARY), vec![Choice::new("skip", CANARY)]),
        Envelope::choice_request(request(), vec![Choice::new(CANARY, CANARY)]),
    ];
    for input in invalid {
        let error = model
            .run(&input, None)
            .expect_err("an invalid request must fail");
        assert!(
            matches!(error, SdkError::InferenceError { .. }),
            "{error:?}"
        );
        let rendered = format!("{error} {error:?}");
        assert!(!rendered.contains(CANARY), "{rendered}");
    }
}

#[test]
fn scorers_refuse_every_other_entry_point() {
    let (_dir, model) = scorer();
    let refused = refusals(&model, &request(), &ConversationContext::new());
    let expected = [
        "conversation context",
        "conversation context",
        "streaming",
        "streaming",
        "streaming",
        "streaming",
        "streaming",
        "streaming",
        "streaming",
        "streaming",
    ];
    assert_eq!(refused, expected);

    assert!(matches!(
        model.stream(StreamConfig::default()),
        Err(SdkError::StreamingNotSupported)
    ));
}

#[test]
fn scorers_refuse_every_input_but_a_request() {
    let (_dir, model) = scorer();
    for (input, kind) in [
        (text("hello"), "Text input"),
        (
            Envelope::new(EnvelopeKind::Audio(vec![0; 16])),
            "Audio input",
        ),
        (scores(), "ChoiceScores input"),
    ] {
        assert!(
            batch_refusals(&model, &input).iter().all(|c| c == kind),
            "{kind}"
        );
    }
}

// ---------------------------------------------------------------------------
// Every other model
// ---------------------------------------------------------------------------

#[test]
fn other_models_refuse_choice_kinds_on_every_entry_point() {
    let (_dir, model) = chat_model();
    for input in choice_kinds() {
        let mut refused = batch_refusals(&model, &input);
        refused.extend(refusals(&model, &input, &ConversationContext::new()));
        assert_eq!(refused.len(), 13);
        assert!(
            refused.iter().all(|c| c == "choice requests"),
            "{refused:?}"
        );
    }
}

#[test]
fn choice_kinds_in_history_are_refused() {
    let (_dir, model) = chat_model();
    let user = text("hello").with_role(MessageRole::User);

    let mut in_history = ConversationContext::new();
    in_history.push(text("earlier").with_role(MessageRole::User));
    in_history.push(Envelope::new(EnvelopeKind::MultiPart(vec![request()])));
    let as_system = ConversationContext::new().with_system(scores());

    let options = RunOptions::new();
    for context in [&in_history, &as_system] {
        let refused = [
            capability(model.run_with_context(&user, context, None)),
            capability(model.run_with_context_options(&user, context, &options)),
            capability(model.run_streaming_with_context(&user, context, None, no_token)),
            capability(
                model.run_streaming_with_context_options(&user, context, &options, no_token),
            ),
        ];
        assert!(
            refused.iter().all(|c| c == "choice requests"),
            "{refused:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Privacy: only the result carries the caller's content
// ---------------------------------------------------------------------------

/// Records every log line, so a test can check what was logged.
struct RecordingLogger;

static LOG_LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

impl log::Log for RecordingLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        LOG_LINES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(format!("{} {}", record.target(), record.args()));
    }

    fn flush(&self) {}
}

/// What ONNX Runtime logs about the fake scorers' runs, when it logs them:
/// the byte table's failure quotes the offending byte id, and
/// `shape_warning.onnx`'s warning quotes a length taken from the first context
/// byte.
const RUNTIME_MESSAGES: [&str; 4] = [
    "out of data bounds",
    "idx=",
    "VerifyOutputSizes",
    "actual shape",
];

/// The shape of `shape_warning.onnx`'s extra output for the canary request,
/// whose context starts with 'c' (byte id 100). Read as the logits, it fails
/// the run, and nothing may quote it.
const CANARY_SHAPE: &str = "[100]";

/// Set in the child process that `caller_content_reaches_only_the_result`
/// starts.
const PRIVACY_CHILD: &str = "XYBRID_TEST_CHOICE_PRIVACY_CHILD";

/// No surface but the result carries the caller's content: not errors,
/// `Debug`, telemetry and its spans, execution events, logs, the trace or
/// stderr.
///
/// The logger, the span collector, the telemetry senders and the execution
/// listener are process-wide, and another test's telemetry resets the span
/// collector, which could drop this test's spans before they are checked. So
/// the checks run alone, in a child copy of this test binary; this test reads
/// the child's result and stderr. The child starts ONNX Runtime with its
/// environment logging warnings (`ORT_LOG=warning`), so a scorer that let
/// ONNX Runtime log would show on that stderr.
#[test]
fn caller_content_reaches_only_the_result() {
    if std::env::var_os(PRIVACY_CHILD).is_some() {
        check_every_surface_for_caller_content();
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "caller_content_reaches_only_the_result",
            "--exact",
            "--nocapture",
        ])
        .env(PRIVACY_CHILD, "1")
        .env("ORT_LOG", "warning")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "the child run failed or matched no test:\n{stdout}\n{stderr}"
    );
    for leak in [CANARY, CANARY_SHAPE].into_iter().chain(RUNTIME_MESSAGES) {
        assert!(
            !stderr.contains(leak),
            "the child's stderr carries {leak:?}:\n{stderr}"
        );
    }
}

/// The child's side of `caller_content_reaches_only_the_result`, alone in its
/// process.
fn check_every_surface_for_caller_content() {
    static LOGGER: RecordingLogger = RecordingLogger;
    log::set_logger(&LOGGER).expect("the child process installs the only logger");
    log::set_max_level(log::LevelFilter::Trace);
    xybrid_core::tracing::init_tracing(true);

    let (telemetry_tx, telemetry_rx) = std::sync::mpsc::channel();
    xybrid_sdk::register_telemetry_sender(telemetry_tx);
    let listened = std::sync::Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = listened.clone();
    xybrid_sdk::set_execution_listener(move |event| {
        sink.lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(format!("{event:?}"));
    });

    let mut canary_context = text(&format!("context {CANARY}"));
    canary_context.set_metadata("note".to_string(), CANARY.to_string());
    let canary_choices = vec![
        Choice::new(CANARY, format!("text {CANARY}")),
        Choice::new(format!("{CANARY}-2"), CANARY),
    ];
    let canary_request = Envelope::choice_request(canary_context.clone(), canary_choices.clone());
    let nested = Envelope::new(EnvelopeKind::MultiPart(vec![canary_request.clone()]));
    let mut history = ConversationContext::new();
    history.push(canary_request.clone());

    // Refusals, before any model runs.
    let (scorer_dir, scorer) = scorer();
    let (_chat_dir, chat) = chat_model();
    let mut errors = Vec::new();
    for (model, inputs) in [
        (&scorer, vec![nested.clone()]),
        (&chat, vec![canary_request.clone(), nested.clone()]),
    ] {
        for input in inputs {
            errors.extend(
                [
                    model.run(&input, None).err(),
                    model.run_streaming(&input, None, no_token).err(),
                    model.run_with_context(&input, &history, None).err(),
                ]
                .into_iter()
                .flatten()
                .map(|error| format!("{error} {error:?}")),
            );
        }
        errors.extend(
            [
                model.run_streaming(&canary_request, None, no_token).err(),
                model.run_with_context(&text("hello"), &history, None).err(),
            ]
            .into_iter()
            .flatten()
            .map(|error| format!("{error} {error:?}")),
        );
        errors.push(format!("{canary_request:?} {history:?}"));
    }
    // The executor also refuses on its own, as the CLI calls it directly.
    let mut executor = xybrid_core::execution::TemplateExecutor::with_base_path(
        scorer_dir.path().to_str().unwrap(),
    );
    let metadata: xybrid_core::execution::ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(scorer_dir.path().join("model_metadata.json")).unwrap(),
    )
    .unwrap();
    let mut repeated = canary_choices.clone();
    repeated.push(canary_choices[0].clone());
    let duplicate = Envelope::choice_request(canary_context.clone(), repeated);
    errors.extend(
        [&nested, &duplicate]
            .into_iter()
            .filter_map(|input| executor.execute(&metadata, input, None).err())
            .map(|error| format!("{error} {error:?}")),
    );
    assert!(errors.len() >= 17, "{errors:#?}");

    // A request scored through the SDK and the executor, and a run ONNX
    // Runtime fails: the byte table refuses letters with a message quoting
    // the byte. The SDK also scores it with `shape_warning.onnx`, whose extra
    // output is as long as the first context byte id (100, for 'c'), which
    // ONNX Runtime warns about at the child's log level.
    let ran = onnx_runtime();
    if ran {
        let (_warning_dir, warning) = scorer_with("shape_warning.onnx");
        for model in [&scorer, &warning] {
            let scored = model
                .run(&canary_request, None)
                .expect("a valid request is scored");
            let first = &scored.choice_scores().expect("scores").entries[0];
            assert_eq!(first.id, CANARY, "the result names the candidates");
        }
        assert!(executor.execute(&metadata, &canary_request, None).is_ok());

        let (_table_dir, table) = scorer_with("byte_table.onnx");
        let error = table
            .run(&canary_request, None)
            .expect_err("the byte table refuses letters");
        assert!(format!("{error:?}").contains("withheld"), "{error:?}");
        errors.push(format!("{error} {error:?}"));

        // Read as the logits, `shape_warning.onnx`'s extra output has a shape
        // taken from the context, which fails the run.
        let mut spec = forms_spec();
        spec["logits_output"] = serde_json::json!("auxiliary");
        let (_shape_dir, shape) = scorer_with_spec("shape_warning.onnx", spec);
        let error = shape
            .run(&canary_request, None)
            .expect_err("a wrong-shaped output fails the run");
        errors.push(format!("{error} {error:?}"));
    }

    xybrid_sdk::clear_execution_listener();
    let telemetry: Vec<_> = telemetry_rx.try_iter().collect();
    let trace = xybrid_core::tracing::get_stages_json();
    let logs = LOG_LINES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let listened = listened.lock().unwrap_or_else(|e| e.into_inner()).clone();

    // Each request the SDK scored has its own completion, which names the
    // decision, counts its five candidates and carries its spans; the
    // executor's run and the failed one leave theirs in the trace. Both are
    // then checked below, so this is the evidence that those checks saw the
    // scorers' traces.
    let decisions: Vec<serde_json::Value> = telemetry
        .iter()
        .filter(|event| event.event_type == "ModelComplete")
        .filter_map(|event| serde_json::from_str(event.data.as_deref()?).ok())
        .filter(|data: &serde_json::Value| data["output_type"] == "ChoiceScores")
        .collect();
    let scoring_span = |span: &serde_json::Value| span["name"] == "choice_scoring";
    if ran && !xybrid_sdk::is_telemetry_opted_out() {
        assert_eq!(decisions.len(), 2, "two scored requests: {decisions:?}");
        for data in &decisions {
            assert_eq!(data["task"], "decision", "{data}");
            assert_eq!(data["backend"], "ort", "{data}");
            assert_eq!(data["choice_count"], 5, "{data}");
            let spans = data["spans"]
                .as_array()
                .expect("the completion carries its spans");
            assert!(
                spans
                    .iter()
                    .any(|span| scoring_span(span) && span["metadata"]["choice_count"] == "5"),
                "{data}"
            );
        }
    } else {
        assert!(decisions.is_empty(), "{decisions:?}");
    }
    if ran {
        // The wrong-shaped run's failure, which names its output, reached the
        // execution listener, which hands telemetry the same message, so the
        // checks below saw it.
        assert!(
            listened
                .iter()
                .any(|event| event.contains("output 'auxiliary'")),
            "{listened:?}"
        );
        let spans = trace["spans"]
            .as_array()
            .expect("the trace lists its spans");
        assert!(spans.iter().any(scoring_span), "{trace}");
    }

    let telemetry: Vec<String> = telemetry.iter().map(|event| format!("{event:?}")).collect();
    for (surface, lines) in [
        ("errors and Debug", errors),
        ("telemetry", telemetry),
        ("execution events", listened),
        ("logs", logs),
        ("trace", vec![trace.to_string()]),
    ] {
        for line in lines {
            assert!(
                !line.contains(CANARY),
                "{surface} leaked caller content: {line}"
            );
            assert!(
                !line.contains(CANARY_SHAPE),
                "{surface} quotes a shape taken from the request: {line}"
            );
            for message in RUNTIME_MESSAGES {
                assert!(
                    !line.contains(message),
                    "{surface} carries ONNX Runtime's message: {line}"
                );
            }
        }
    }
}

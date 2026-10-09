//! Choice requests through the SDK's public surface.
//!
//! A choice scorer accepts one input, a `ChoiceRequest`, through one kind of
//! entry point, a batch run. This build validates such requests but does not
//! run scorers yet, so even a valid request is refused before any model is
//! opened. Everything else is refused before any cloud leg or backend call:
//! a choice request sent to another model, other input sent to a scorer, every
//! streaming and conversation entry point, and a choice kind buried in a
//! multi-part message or in conversation history.
//!
//! The models here are directories holding only `model_metadata.json`, so a
//! path that tried to open weights would fail with a load error instead of the
//! refusal each test expects.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use tokio_stream::StreamExt;
use xybrid_core::runtime_adapter::{AdapterResult, CloudStreaming};
use xybrid_sdk::ir::{Envelope, EnvelopeKind, MessageRole};
use xybrid_sdk::{
    CancellationToken, Choice, ChoiceScore, ChoiceScores, ConversationContext, InferenceResult,
    ModelLoader, OutputType, PartialToken, RunOptions, SdkError, StreamConfig, StreamEvent,
    StreamingCallback, XybridModel,
};

const CANARY: &str = "canary-9c41d7";

type TokenResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn forms_spec() -> serde_json::Value {
    // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs in
    // a sandbox where the compile-time path points nowhere.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
    let path = std::path::Path::new(&manifest_dir)
        .join("../../integration-tests/fixtures/choice/specs/cua-s1-forms.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the FORMS spec is JSON")
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

fn scorer() -> (tempfile::TempDir, XybridModel) {
    load(
        "forms-scorer",
        serde_json::json!({ "type": "ChoiceScorer", "scorer": forms_spec() }),
    )
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

/// Runs `input` through every public run entry point and returns the
/// capability each refused it with, in a fixed order.
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
        capability(model.run(input, None)),
        capability(model.run_with_options(input, &options)),
        capability(runtime().block_on(model.run_async(input, None))),
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
fn a_valid_request_is_refused_until_scorers_run() {
    let (_dir, model) = scorer();
    assert_eq!(capability(model.run(&request(), None)), "choice scoring");
    assert_eq!(
        capability(runtime().block_on(model.run_async(&request(), None))),
        "choice scoring"
    );
    // FORMS has three fixed choices, so a request may offer none.
    let fixed_only = Envelope::choice_request(text("FORM Intake"), Vec::new());
    assert_eq!(capability(model.run(&fixed_only, None)), "choice scoring");

    assert_eq!(capability(model.warmup()), "choice scoring");
    assert_eq!(
        capability(runtime().block_on(model.warmup_async())),
        "choice scoring"
    );
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
    // run, run_with_options and run_async are the batch entries: there the
    // valid request reaches the not-yet-enabled scorer.
    let expected = [
        "choice scoring",
        "choice scoring",
        "choice scoring",
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
        assert_eq!(capability(model.run(&input, None)), kind);
        assert_eq!(
            capability(runtime().block_on(model.run_async(&input, None))),
            kind
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
        let refused = refusals(&model, &input, &ConversationContext::new());
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
// Privacy: refusals never carry the caller's content
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

#[test]
fn refusals_leak_no_caller_content() {
    static LOGGER: RecordingLogger = RecordingLogger;
    // Only this test installs a logger; `set_logger` fails if one exists.
    let _ = log::set_logger(&LOGGER);
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
    let canary_request = Envelope::choice_request(
        canary_context,
        vec![
            Choice::new(CANARY, format!("text {CANARY}")),
            Choice::new(format!("{CANARY}-2"), CANARY),
        ],
    );
    let mut history = ConversationContext::new();
    history.push(canary_request.clone());

    let (_scorer_dir, scorer) = scorer();
    let (_chat_dir, chat) = chat_model();
    let mut errors = Vec::new();
    for model in [&scorer, &chat] {
        for input in [
            canary_request.clone(),
            Envelope::new(EnvelopeKind::MultiPart(vec![canary_request.clone()])),
        ] {
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
            model
                .run_with_context(&text("hello"), &history, None)
                .err()
                .map(|error| format!("{error} {error:?}")),
        );
        errors.push(format!("{canary_request:?} {history:?}"));
    }
    // The executor also refuses on its own, as the CLI calls it directly.
    let mut executor = xybrid_core::execution::TemplateExecutor::with_base_path(
        _scorer_dir.path().to_str().unwrap(),
    );
    let metadata: xybrid_core::execution::ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(_scorer_dir.path().join("model_metadata.json")).unwrap(),
    )
    .unwrap();
    errors.extend(
        executor
            .execute(&metadata, &canary_request, None)
            .err()
            .map(|error| format!("{error} {error:?}")),
    );
    assert!(errors.len() >= 13, "{errors:#?}");

    xybrid_sdk::clear_execution_listener();
    let telemetry: Vec<String> = telemetry_rx
        .try_iter()
        .map(|event| format!("{event:?}"))
        .collect();
    let trace = xybrid_core::tracing::get_stages_json().to_string();
    let logs = LOG_LINES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let listened = listened.lock().unwrap_or_else(|e| e.into_inner()).clone();

    for (surface, lines) in [
        ("errors and Debug", errors),
        ("telemetry", telemetry),
        ("execution events", listened),
        ("logs", logs),
        ("trace", vec![trace]),
    ] {
        for line in lines {
            assert!(
                !line.contains(CANARY),
                "{surface} leaked caller content: {line}"
            );
        }
    }
}

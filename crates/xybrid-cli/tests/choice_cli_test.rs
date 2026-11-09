//! `xybrid run --choice` through the built CLI: a choice scorer scores the
//! candidates, and an invalid request or a model that cannot score them is
//! refused before anything runs.
//!
//! The scorers are the fake ONNX scorers of
//! `integration-tests/fixtures/choice/fake/`, copied under the FORMS spec's
//! file name, and CUA-S1-FORMS itself once its export is staged
//! (`./integration-tests/download.sh cua-s1-forms`). Every run gets its own
//! `HOME` and a registry URL on a closed loopback port, so a cached model
//! runs offline and a registry call fails at once.
//!
//! The refusals need no ONNX Runtime: their scorers have no model file, so a
//! request that got past the checks would fail to load instead. The scored
//! runs compare the CLI's `--output` JSON with the same request run through
//! the Rust SDK in this process; they skip without ONNX Runtime unless
//! `XYBRID_REQUIRE_MODELS` is set, where they fail instead.

use std::ffi::OsStr;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use xybrid_core::testing::model_fixtures::{models_required, staged_artifact, ENV_REQUIRE_MODELS};
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::{Choice, ChoiceScores, ModelLoader};

const CANARY: &str = "canary-3d8e51";

/// The context of the plan's pinned FORMS command.
const FORMS_CONTEXT: &str = "TASK fill the form from the document, then submit\n\
                             FORM Intake\n\
                             ELEMENT Edit \"Phone\" value=\"\"";

/// The message every refusal of a FORMS-shaped scorer opens with.
const NUMBERING: &str = "choice request refused (counting from 0, candidates are the --choice \
                         flags in order and then the model's fixed choices check, click, skip; \
                         caller choices are the --choice flags; fixed choices are check, \
                         click, skip)";

fn xybrid_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_xybrid"))
}

fn choice_fixtures() -> PathBuf {
    // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs in
    // a sandbox where the compile-time path points nowhere.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
    Path::new(&manifest_dir).join("../../integration-tests/fixtures/choice")
}

fn forms_spec() -> serde_json::Value {
    let path = choice_fixtures().join("specs/cua-s1-forms.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the FORMS spec is JSON")
}

/// Whether ONNX Runtime can run here: it is linked in on Apple platforms and
/// loaded from `ORT_DYLIB_PATH` elsewhere. The CLI child inherits the same
/// environment. Without it a scored test skips, unless models are required,
/// where it fails instead.
fn onnx_runtime() -> bool {
    let available = cfg!(any(target_os = "macos", target_os = "ios"))
        || std::env::var_os("ORT_DYLIB_PATH").is_some_and(|path| Path::new(&path).is_file());
    if !available {
        assert!(
            !models_required(),
            "{ENV_REQUIRE_MODELS} is set but ONNX Runtime is not loadable; set ORT_DYLIB_PATH"
        );
        eprintln!("Skipping: ONNX Runtime is not loadable (set ORT_DYLIB_PATH)");
    }
    available
}

/// Writes a scorer with `spec` into `dir`: its metadata and, when `fake`
/// names one, a copy of that fake ONNX scorer as its model file. Without
/// `fake` the model file is absent.
fn write_scorer(dir: &Path, model_id: &str, spec: &serde_json::Value, fake: Option<&str>) {
    std::fs::create_dir_all(dir).unwrap();
    let metadata = serde_json::json!({
        "model_id": model_id,
        "version": "1.0",
        "execution_template": { "type": "ChoiceScorer", "scorer": spec },
        "files": [],
    });
    std::fs::write(dir.join("model_metadata.json"), metadata.to_string()).unwrap();
    if let Some(fake) = fake {
        let model_file = spec["model_file"]
            .as_str()
            .expect("the spec names its model");
        std::fs::copy(
            choice_fixtures().join("fake").join(fake),
            dir.join(model_file),
        )
        .unwrap();
    }
}

/// A FORMS-shaped scorer in a fresh directory; see [`write_scorer`].
fn scorer(fake: Option<&str>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_scorer(dir.path(), "forms-scorer", &forms_spec(), fake);
    dir
}

/// One isolated environment for `xybrid run`: its own `HOME` (and so its own
/// model cache), no API key, telemetry off, plain output, and the registry on
/// a loopback port nothing listens on.
struct Cli {
    home: tempfile::TempDir,
    registry_url: String,
}

impl Cli {
    fn new() -> Self {
        // Bound, then released: nothing listens there afterwards.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        Self {
            home: tempfile::tempdir().unwrap(),
            registry_url: format!("http://127.0.0.1:{port}"),
        }
    }

    /// Runs `xybrid run <args>`. The child keeps this process's working
    /// directory: under Bazel the binary's path is relative to it.
    fn run<I, S>(&self, args: I) -> Output
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        Command::new(xybrid_bin())
            .arg("run")
            .args(args)
            .env("HOME", self.home.path())
            .env("XYBRID_REGISTRY_URL", &self.registry_url)
            .env("NO_COLOR", "1")
            // Would override NO_COLOR and color the table the tests parse.
            .env_remove("CLICOLOR_FORCE")
            .env("XYBRID_TELEMETRY_OPTOUT", "1")
            .env_remove("XYBRID_API_KEY")
            .env_remove("XYBRID_GATEWAY_URL")
            .env_remove("XYBRID_PLATFORM_URL")
            .env_remove("RUST_BACKTRACE")
            .env_remove("RUST_LIB_BACKTRACE")
            .env_remove("HTTP_PROXY")
            .env_remove("HTTPS_PROXY")
            .env_remove("http_proxy")
            .env_remove("https_proxy")
            .env_remove("ALL_PROXY")
            .output()
            .expect("spawn the xybrid CLI")
    }

    /// Seeds the model cache with a scorer, as `RegistryClient::resolve_offline`
    /// expects it: `<HOME>/.xybrid/cache/extracted/<id>/`. Unix only: on
    /// Windows the cache follows the user profile, not `HOME`.
    #[cfg(unix)]
    fn cache_scorer(&self, model_id: &str, fake: Option<&str>) {
        let cache = self.home.path().join(".xybrid").join("cache");
        std::fs::create_dir_all(cache.join("models")).unwrap();
        write_scorer(
            &cache.join("extracted").join(model_id),
            model_id,
            &forms_spec(),
            fake,
        );
    }
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[track_caller]
fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "xybrid run failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout(output),
        stderr(output)
    );
}

/// Asserts a refusal: exit code `code`, `expected` in stderr, and no inference.
#[track_caller]
fn assert_refused(output: &Output, code: i32, expected: &str) {
    let (out, err) = (stdout(output), stderr(output));
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout:\n{out}\nstderr:\n{err}"
    );
    assert!(err.contains(expected), "expected {expected:?} in:\n{err}");
    assert!(!out.contains("Results"), "nothing may run:\n{out}");
}

fn choice_args(choices: &[(&str, &str)]) -> Vec<String> {
    choices
        .iter()
        .flat_map(|(id, text)| ["--choice".to_string(), format!("{id}={text}")])
        .collect()
}

/// One row of the printed score table.
#[derive(Debug, PartialEq)]
struct Row {
    id: String,
    score: String,
    logit: String,
    note: String,
}

/// The rows of the score table in `stdout`, in printed order.
fn table_rows(stdout: &str) -> Vec<Row> {
    let mut lines = stdout.lines();
    lines
        .by_ref()
        .find(|line| line.split_whitespace().eq(["ID", "Score", "Logit"]))
        .unwrap_or_else(|| panic!("no score table in:\n{stdout}"));
    lines.next().expect("the table's separator line");
    lines
        .take_while(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            let mut field = || fields.next().expect("a full row").to_string();
            let (id, score, logit) = (field(), field(), field());
            Row {
                id,
                score,
                logit,
                note: fields.collect::<Vec<_>>().join(" "),
            }
        })
        .collect()
}

/// The rows the table should print for `scores`.
fn expected_rows(scores: &ChoiceScores) -> Vec<Row> {
    let best = scores.best().expect("scores have entries");
    scores
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| Row {
            id: entry.id.clone(),
            score: format!("{:.4}", entry.score),
            logit: format!("{:.4}", entry.logit),
            note: match (index == best, entry.fixed) {
                (true, true) => "best, fixed",
                (true, false) => "best",
                (false, true) => "fixed",
                (false, false) => "",
            }
            .to_string(),
        })
        .collect()
}

fn read_scores(path: &Path) -> ChoiceScores {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}\n{text}", path.display()))
}

/// The same request through the Rust SDK, in this process.
fn scored_by_the_sdk(model_dir: &Path, context: &str, choices: &[(&str, &str)]) -> ChoiceScores {
    let model = ModelLoader::from_directory(model_dir)
        .expect("the scorer's metadata parses")
        .load()
        .expect("the scorer loads");
    let request = Envelope::choice_request(
        Envelope::new(EnvelopeKind::Text(context.to_string())),
        choices
            .iter()
            .map(|(id, text)| Choice::new(*id, *text))
            .collect(),
    );
    model
        .run(&request, None)
        .expect("the SDK scores the request")
        .choice_scores()
        .expect("a scorer returns choice scores")
        .clone()
}

/// Scores `choices` against `context` with the CLI on the scorer in
/// `model_dir`, checks the printed table against the saved JSON, and returns
/// the saved scores.
fn score_with_the_cli(
    cli: &Cli,
    model_dir: &Path,
    context: &str,
    choices: &[(&str, &str)],
) -> ChoiceScores {
    let saved = cli.home.path().join("scores.json");
    let mut args = vec![
        "--directory".into(),
        model_dir.display().to_string(),
        "--input-text".into(),
        context.to_string(),
        "--output".into(),
        saved.display().to_string(),
    ];
    args.extend(choice_args(choices));
    let output = cli.run(&args);
    assert_success(&output);

    let scores = read_scores(&saved);
    assert_eq!(table_rows(&stdout(&output)), expected_rows(&scores));
    scores
}

/// Asserts the CLI saved exactly the scores the SDK returns. Compared as
/// JSON, because the types' `Debug` hides ids and would hide the difference.
#[track_caller]
fn assert_same_scores(cli: &ChoiceScores, sdk: &ChoiceScores) {
    assert_eq!(
        serde_json::to_value(cli).unwrap(),
        serde_json::to_value(sdk).unwrap(),
        "the CLI and the SDK disagree"
    );
}

fn ids(scores: &ChoiceScores) -> Vec<&str> {
    scores
        .entries
        .iter()
        .map(|entry| entry.id.as_str())
        .collect()
}

// ---------------------------------------------------------------------------
// Refusals (no ONNX Runtime needed)
// ---------------------------------------------------------------------------

#[test]
fn choice_flags_fail_before_anything_loads() {
    let cli = Cli::new();
    // A directory that does not exist: parsing fails before any load.
    let missing = cli.home.path().join("missing");
    let missing = missing.to_str().unwrap();

    let no_context = cli.run(["--directory", missing, "--choice", "a=b"]);
    assert_refused(&no_context, 2, "--input-text");

    let malformed = cli.run([
        "--directory",
        missing,
        "--input-text",
        "ctx",
        "--choice",
        "ab",
    ]);
    assert_refused(&malformed, 2, "expected ID=TEXT");

    let conflict = cli.run([
        "--directory",
        missing,
        "--input-text",
        "ctx",
        "--choice",
        "a=b",
        "--voice",
        "af_bella",
    ]);
    assert_refused(&conflict, 2, "cannot be used with");
}

#[test]
fn other_models_refuse_choices_before_opening_their_files() {
    let cli = Cli::new();
    // A chat model with metadata only: opening its weights would fail with a
    // load error instead of the refusal.
    let chat = tempfile::tempdir().unwrap();
    let metadata = serde_json::json!({
        "model_id": "chat",
        "version": "1.0",
        "execution_template": { "type": "Gguf", "model_file": "model.gguf" },
        "files": [],
    });
    std::fs::write(
        chat.path().join("model_metadata.json"),
        metadata.to_string(),
    )
    .unwrap();

    let output = cli.run(
        [
            "--directory",
            chat.path().to_str().unwrap(),
            "--input-text",
            CANARY,
        ]
        .into_iter()
        .map(String::from)
        .chain(choice_args(&[(CANARY, CANARY)])),
    );
    assert_refused(
        &output,
        1,
        "'chat' is not a choice scorer, so it cannot score --choice candidates",
    );
    let err = stderr(&output);
    assert!(err.contains("does not support choice requests"), "{err}");
    assert!(!err.contains(CANARY), "{err}");
}

#[test]
fn scorers_refuse_invalid_requests_before_running() {
    let cli = Cli::new();
    // No model file: a request that got past the checks would fail to load.
    let forms = scorer(None);
    let mut unfixed_spec = forms_spec();
    unfixed_spec["fixed_choices"] = serde_json::json!([]);
    let unfixed = tempfile::tempdir().unwrap();
    write_scorer(unfixed.path(), "unfixed-scorer", &unfixed_spec, None);

    let too_many: Vec<(String, String)> = (0..62)
        .map(|index| (format!("c{index}"), CANARY.to_string()))
        .collect();
    let too_many: Vec<(&str, &str)> = too_many
        .iter()
        .map(|(id, text)| (id.as_str(), text.as_str()))
        .collect();

    /// One invalid request and the refusal it must draw.
    struct Case<'a> {
        scorer: &'a Path,
        context: &'a str,
        choices: &'a [(&'a str, &'a str)],
        refusal: &'a str,
    }
    let cases = [
        Case {
            scorer: forms.path(),
            context: "",
            choices: &[("tel", CANARY)],
            refusal: "the context is empty; a byte choice scorer needs at least one byte",
        },
        Case {
            scorer: forms.path(),
            context: CANARY,
            choices: &[("skip", CANARY)],
            refusal: "caller choice 0 reuses the id of fixed choice 2",
        },
        Case {
            scorer: forms.path(),
            context: CANARY,
            choices: &[(CANARY, "a"), (CANARY, "b")],
            refusal: "candidate 1 repeats the id of candidate 0",
        },
        Case {
            scorer: forms.path(),
            context: CANARY,
            choices: &too_many,
            refusal:
                "65 candidates offered including fixed choices; this scorer accepts at most 64",
        },
        Case {
            scorer: unfixed.path(),
            context: CANARY,
            choices: &[("a", CANARY)],
            refusal: "1 candidate(s) offered; a choice scorer needs at least 2",
        },
    ];
    for case in cases {
        let mut args = vec![
            "--directory".to_string(),
            case.scorer.display().to_string(),
            "--input-text".to_string(),
            case.context.to_string(),
        ];
        args.extend(choice_args(case.choices));
        let output = cli.run(&args);
        assert_refused(&output, 1, case.refusal);

        let err = stderr(&output);
        let numbering = if case.scorer == unfixed.path() {
            "choice request refused (counting from 0, candidates are the --choice flags in order)"
        } else {
            NUMBERING
        };
        assert!(err.contains(numbering), "{err}");
        assert!(!err.contains(CANARY), "caller text in stderr:\n{err}");
    }
}

#[test]
fn scorers_refuse_flags_they_cannot_use() {
    let cli = Cli::new();
    let forms = scorer(None);
    let forms = forms.path().to_str().unwrap();
    // Without --choice, clap's conflicts do not apply; the scorer refuses the
    // flags itself, before reading the audio file (which does not exist).
    let output = cli.run([
        "--directory",
        forms,
        "--input-text",
        CANARY,
        "--input-audio",
        "missing.wav",
        "--voice",
        "af_bella",
        "--max-tokens",
        "8",
    ]);
    assert_refused(
        &output,
        1,
        "'forms-scorer' is a choice scorer, which reads only --input-text and --choice; it \
         cannot use --input-audio, --voice, --max-tokens",
    );
    assert!(!stderr(&output).contains(CANARY));
}

#[test]
fn dry_runs_check_choice_requests_like_runs() {
    let cli = Cli::new();
    let forms = scorer(None);
    let forms = forms.path().to_str().unwrap();
    let chat = tempfile::tempdir().unwrap();
    let metadata = serde_json::json!({
        "model_id": "chat",
        "version": "1.0",
        "execution_template": { "type": "Gguf", "model_file": "model.gguf" },
        "files": [],
    });
    std::fs::write(
        chat.path().join("model_metadata.json"),
        metadata.to_string(),
    )
    .unwrap();

    let valid = cli.run([
        "--directory",
        forms,
        "--input-text",
        CANARY,
        "--choice",
        "tel=fill Tel",
        "--dry-run",
    ]);
    assert_success(&valid);
    assert!(
        stdout(&valid).contains("Model is valid and ready for execution"),
        "{}",
        stdout(&valid)
    );

    let reserved = cli.run([
        "--directory",
        forms,
        "--input-text",
        CANARY,
        "--choice",
        "skip=skip it",
        "--dry-run",
    ]);
    assert_refused(
        &reserved,
        1,
        "caller choice 0 reuses the id of fixed choice 2",
    );
    assert!(!stdout(&reserved).contains("ready for execution"));

    let other = cli.run([
        "--directory",
        chat.path().to_str().unwrap(),
        "--input-text",
        CANARY,
        "--choice",
        "a=b",
        "--dry-run",
    ]);
    assert_refused(&other, 1, "'chat' is not a choice scorer");
    assert!(!stdout(&other).contains("ready for execution"));
}

#[cfg(unix)]
#[test]
fn speculative_cloud_refuses_a_cached_scorer() {
    let cli = Cli::new();
    cli.cache_scorer("cached-scorer", None);

    // Clap keeps --choice away from --speculative-cloud; a scorer run without
    // it would score the fixed choices alone, so the CLI refuses the scorer.
    // Its dry run plans on another path and must refuse it too.
    for dry_run in [false, true] {
        let mut args = vec!["--model", "cached-scorer", "--input-text", CANARY];
        args.push("--speculative-cloud");
        if dry_run {
            args.push("--dry-run");
        }
        let output = cli.run(&args);
        assert_refused(
            &output,
            1,
            "'cached-scorer' is a choice scorer; --speculative-cloud streams a text answer",
        );
        assert!(!stderr(&output).contains(CANARY));
        assert!(
            !stdout(&output).contains("ready for execution"),
            "{}",
            stdout(&output)
        );
    }
}

// ---------------------------------------------------------------------------
// Scored runs (ONNX Runtime)
// ---------------------------------------------------------------------------

#[test]
fn scorers_print_and_save_scores_in_offered_order() {
    if !onnx_runtime() {
        return;
    }
    let cli = Cli::new();
    let model = scorer(Some("dynamic.onnx"));
    let choices = [
        ("tel", "fill Tel: (503) 555-0142"),
        ("name", "fill Name: Jane Doe"),
    ];

    let scores = score_with_the_cli(&cli, model.path(), FORMS_CONTEXT, &choices);
    assert_eq!(ids(&scores), ["tel", "name", "check", "click", "skip"]);
    let fixed: Vec<bool> = scores.entries.iter().map(|entry| entry.fixed).collect();
    assert_eq!(fixed, [false, false, true, true, true]);
    assert_eq!(scores.label_mass, None);
    assert_same_scores(
        &scores,
        &scored_by_the_sdk(model.path(), FORMS_CONTEXT, &choices),
    );
}

#[test]
fn identical_choices_put_best_on_the_first() {
    if !onnx_runtime() {
        return;
    }
    let cli = Cli::new();
    let model = scorer(Some("dynamic.onnx"));
    // The fake scores a candidate by its bytes, so equal texts tie exactly,
    // and these outscore the short fixed choices.
    let choices = [
        ("first", "fill the phone field"),
        ("second", "fill the phone field"),
    ];

    let scores = score_with_the_cli(&cli, model.path(), "FORM Intake", &choices);
    assert_eq!(scores.entries[0].logit, scores.entries[1].logit);
    assert_eq!(scores.best(), Some(0));
    assert_eq!(scores.margin(), Some(0.0));
}

#[test]
fn a_scorer_without_choices_scores_its_fixed_choices() {
    if !onnx_runtime() {
        return;
    }
    let cli = Cli::new();
    let model = scorer(Some("dynamic.onnx"));

    let scores = score_with_the_cli(&cli, model.path(), "FORM Intake", &[]);
    assert_eq!(ids(&scores), ["check", "click", "skip"]);
    assert!(scores.entries.iter().all(|entry| entry.fixed));
    assert_same_scores(
        &scores,
        &scored_by_the_sdk(model.path(), "FORM Intake", &[]),
    );
}

#[cfg(unix)]
#[test]
fn a_cached_scorer_runs_by_model_id() {
    if !onnx_runtime() {
        return;
    }
    let cli = Cli::new();
    cli.cache_scorer("cached-scorer", Some("dynamic.onnx"));
    let saved = cli.home.path().join("by-id.json");

    // The registry is unreachable: the model must come from the cache.
    let output = cli.run([
        "--model",
        "cached-scorer",
        "--input-text",
        "FORM Intake",
        "--choice",
        "tel=fill Tel: (503) 555-0142",
        "--output",
        saved.to_str().unwrap(),
    ]);
    assert_success(&output);
    assert!(
        stdout(&output).contains("Using locally cached model"),
        "{}",
        stdout(&output)
    );

    let scores = read_scores(&saved);
    let cached = cli
        .home
        .path()
        .join(".xybrid/cache/extracted/cached-scorer");
    assert_same_scores(
        &scores,
        &scored_by_the_sdk(
            &cached,
            "FORM Intake",
            &[("tel", "fill Tel: (503) 555-0142")],
        ),
    );
}

#[test]
fn forms_scores_the_pinned_command_like_the_sdk() {
    let Some(onnx) = staged_artifact("cua-s1-forms", "cua-s1-forms.onnx") else {
        return;
    };
    if !onnx_runtime() {
        return;
    }
    let forms = onnx
        .parent()
        .expect("the export sits in its model directory");
    let choices = [
        ("tel", "fill Tel: (503) 555-0142"),
        ("name", "fill Name: Jane Doe"),
    ];

    let scores = score_with_the_cli(&Cli::new(), forms, FORMS_CONTEXT, &choices);
    assert_eq!(ids(&scores), ["tel", "name", "check", "click", "skip"]);
    assert_same_scores(&scores, &scored_by_the_sdk(forms, FORMS_CONTEXT, &choices));
}

//! Label scoring through [`LlamaCppBackend::score_labels`], checked against
//! the conformance fixtures in `integration-tests/fixtures/choice`.
//!
//! The model-free tests pin the scoring render. The rest run on the
//! Qwen3.5-0.8B conformance GGUFs and skip when a model is not staged, unless
//! `XYBRID_REQUIRE_MODELS` is set:
//!
//! ```sh
//! tools/conformance/prepare.sh qwen3.5-0.8b-f16-cf qwen3.5-0.8b-q4km-cf
//! export XYBRID_LLAMA_HARNESS="$(tools/conformance/llama_harness/build.sh bindings | tail -n 1)"
//! cargo test -p xybrid-core --features llm-llamacpp --lib -- \
//!     runtime_adapter::llama_cpp::choice_scoring_tests --test-threads=1
//! ```
//!
//! - **G0:** the GGUF embeds the template the goldens were rendered from, and
//!   the scoring path renders, tokenizes and resolves labels exactly as
//!   Hugging Face did.
//! - **G1:** the F16 GGUF, with the backend's default settings, against the
//!   Hugging Face float32 reference, within `gates.labels.g1`.
//! - **G2:** the same GGUF through the C++ harness, with the backend's context
//!   and batch sizes on the CPU, within `gates.labels.g2`. It needs
//!   `XYBRID_LLAMA_HARNESS`, which is required whenever models are.
//! - **G3:** the Q4_K_M GGUF against the reference, written to
//!   `target/choice-conformance/g3.json`; a report that never fails on its
//!   numbers.
//! - **Spread:** the F16 GGUF at 1, 4 and 8 threads, on the CPU and with the
//!   default GPU offload, five repeats each, written to
//!   `target/choice-conformance/label-spread.json`: the measurements
//!   `gates.labels` is frozen from.
//! - **State:** scores and generations match a fresh backend's whatever ran
//!   before them, refusals included.

use super::chat::render_scoring_template;
use super::jinja_template::{JinjaChatTemplate, RenderOptions};
use super::{chat, label_token_ids, LlamaCppBackend};
use crate::execution::choice::math::label_scores;
use crate::ir::ChoiceScores;
use crate::runtime_adapter::llm::{
    local_execution_provider, ChatMessage, GenerationConfig, LabelLogits, LlmBackend, LlmConfig,
    ScoringRequest,
};
use crate::runtime_adapter::AdapterError;
use crate::testing::model_fixtures::{models_required, staged_artifact, ENV_REQUIRE_MODELS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc};
use std::time::Duration;

const F16_ID: &str = "qwen3.5-0.8b-f16-cf";
const F16_FILE: &str = "Qwen3.5-0.8B-F16.gguf";
const Q4_ID: &str = "qwen3.5-0.8b-q4km-cf";
const Q4_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";
const CANARY: &str = "canary-7f3a9c";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Goldens {
    provenance: GoldenProvenance,
    cases: Vec<Golden>,
}

#[derive(Deserialize)]
struct GoldenProvenance {
    template_sha256: String,
}

#[derive(Deserialize)]
struct Golden {
    id: String,
    messages: Vec<ChatMessage>,
    rendered: String,
    token_ids: Vec<i32>,
    labels: Vec<String>,
    label_ids: Vec<i32>,
    label_logits: Vec<f64>,
    logsumexp: f64,
    scores: Vec<f64>,
    label_mass: f64,
    best: usize,
    margin: f64,
}

#[derive(Deserialize)]
struct Provenance {
    gates: Gates,
}

#[derive(Deserialize)]
struct Gates {
    labels: LabelGates,
}

#[derive(Deserialize)]
struct LabelGates {
    status: String,
    g1: Bounds,
    g2: Bounds,
}

#[derive(Deserialize, Clone, Copy, Debug)]
struct Bounds {
    max_abs_logit: f64,
    max_abs_score: f64,
    max_abs_label_mass: f64,
    near_tie_margin: f64,
}

fn workspace_root() -> PathBuf {
    // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs
    // in a sandbox where the compile-time path points nowhere.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
    PathBuf::from(manifest_dir).join("../..")
}

fn read_choice_fixture(relative: &str) -> String {
    let path = workspace_root()
        .join("integration-tests/fixtures/choice")
        .join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn goldens() -> Goldens {
    serde_json::from_str(&read_choice_fixture("goldens/labels.json"))
        .expect("goldens/labels.json parses")
}

fn label_gates() -> LabelGates {
    let provenance: Provenance = serde_json::from_str(&read_choice_fixture("provenance.json"))
        .expect("provenance.json parses");
    let gates = provenance.gates.labels;
    assert!(
        gates.status == "provisional" || gates.status == "frozen",
        "unknown gates.labels.status {:?}",
        gates.status
    );
    gates
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn file_sha256(path: &Path) -> String {
    let mut file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// One single-token group per label, as every Qwen case uses.
fn singleton_groups(labels: &[String]) -> Vec<Vec<String>> {
    labels.iter().map(|label| vec![label.clone()]).collect()
}

fn request(case: &Golden) -> ScoringRequest {
    ScoringRequest::new(case.messages.clone(), singleton_groups(&case.labels))
}

fn ids(case: &Golden) -> Vec<(String, bool)> {
    case.labels
        .iter()
        .map(|label| (label.clone(), false))
        .collect()
}

fn scores(case: &Golden, logits: &LabelLogits) -> ChoiceScores {
    label_scores(ids(case), &logits.label_logits, logits.logsumexp, None)
        .unwrap_or_else(|e| panic!("{}: {e}", case.id))
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// A backend loaded from `path` with `configure` applied to the default
/// settings.
fn load(path: &Path, configure: impl FnOnce(&mut LlmConfig)) -> LlamaCppBackend {
    let mut config = LlmConfig::new(path.to_str().expect("a UTF-8 model path"));
    configure(&mut config);
    let mut backend = LlamaCppBackend::new().expect("create the backend");
    backend.load(&config).expect("load the GGUF");
    backend
}

/// The CPU with flash attention off, as the C++ harness runs, and four
/// threads, as it is given.
fn cpu(config: &mut LlmConfig) {
    config.gpu_layers = 0;
    config.flash_attn = false;
    config.n_threads = 4;
}

fn greedy(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        min_p: 0.0,
        top_k: 0,
        repetition_penalty: 1.0,
        ..GenerationConfig::default()
    }
}

fn score_all(backend: &LlamaCppBackend, goldens: &Goldens) -> Vec<LabelLogits> {
    goldens
        .cases
        .iter()
        .map(|case| {
            backend
                .score_labels(&request(case))
                .unwrap_or_else(|e| panic!("{}: {e}", case.id))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Drift against the reference
// ---------------------------------------------------------------------------

/// How far one configuration's scores are from the reference goldens.
#[derive(Debug, Default, Serialize)]
struct Drift {
    max_abs_logit: f64,
    max_abs_logsumexp: f64,
    max_abs_score: f64,
    max_abs_label_mass: f64,
    /// Cases whose best label differs from the reference's.
    argmax_differs: Vec<String>,
    /// Cases closer than the near-tie margin, whose best label is not
    /// compared.
    near_ties: Vec<String>,
}

impl Drift {
    fn of(logits: &[LabelLogits], goldens: &Goldens, near_tie_margin: f64) -> Self {
        let mut drift = Self::default();
        for (case, logits) in goldens.cases.iter().zip(logits) {
            let scores = scores(case, logits);
            for (index, entry) in scores.entries.iter().enumerate() {
                drift.max_abs_logit = drift
                    .max_abs_logit
                    .max((f64::from(entry.logit) - case.label_logits[index]).abs());
                drift.max_abs_score = drift
                    .max_abs_score
                    .max((f64::from(entry.score) - case.scores[index]).abs());
            }
            drift.max_abs_logsumexp = drift
                .max_abs_logsumexp
                .max((f64::from(logits.logsumexp) - case.logsumexp).abs());
            let label_mass = f64::from(scores.label_mass.expect("label scores carry a mass"));
            drift.max_abs_label_mass = drift
                .max_abs_label_mass
                .max((label_mass - case.label_mass).abs());
            if case.margin < near_tie_margin {
                drift.near_ties.push(case.id.clone());
            } else if scores.best() != Some(case.best) {
                drift.argmax_differs.push(case.id.clone());
            }
        }
        drift
    }

    /// Widens `self` to cover `other`: the larger of each maximum, and every
    /// case either lists.
    fn include(&mut self, other: &Self) {
        self.max_abs_logit = self.max_abs_logit.max(other.max_abs_logit);
        self.max_abs_logsumexp = self.max_abs_logsumexp.max(other.max_abs_logsumexp);
        self.max_abs_score = self.max_abs_score.max(other.max_abs_score);
        self.max_abs_label_mass = self.max_abs_label_mass.max(other.max_abs_label_mass);
        for (mine, theirs) in [
            (&mut self.argmax_differs, &other.argmax_differs),
            (&mut self.near_ties, &other.near_ties),
        ] {
            for id in theirs {
                if !mine.contains(id) {
                    mine.push(id.clone());
                }
            }
        }
    }

    fn assert_within(&self, bounds: Bounds, what: &str) {
        let checks = [
            ("label logit", self.max_abs_logit, bounds.max_abs_logit),
            ("score", self.max_abs_score, bounds.max_abs_score),
            (
                "label mass",
                self.max_abs_label_mass,
                bounds.max_abs_label_mass,
            ),
        ];
        for (name, measured, bound) in checks {
            assert!(
                measured <= bound,
                "{what}: max |Δ{name}| {measured:.3e} > {bound:.1e}"
            );
        }
        assert!(
            self.argmax_differs.is_empty(),
            "{what}: the best label differs from the reference in {:?}",
            self.argmax_differs
        );
    }
}

/// Writes a report under the workspace's `target/choice-conformance/`.
fn write_report(name: &str, report: &serde_json::Value) {
    let dir = workspace_root().join("target/choice-conformance");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let path = dir.join(name);
    let text = serde_json::to_string_pretty(report).expect("the report serializes") + "\n";
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

fn platform() -> String {
    format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
}

// ---------------------------------------------------------------------------
// The scoring render (no model)
// ---------------------------------------------------------------------------

/// The scoring prompt is the template's output and nothing else: the
/// committed Qwen3.5 template renders every case as Hugging Face did with the
/// thinking flag omitted, the flag reaches a template as given, `bos_token`
/// is empty, `eos_token` is passed through, and no `<think>` prime follows.
#[test]
fn the_scoring_render_is_the_template_alone() {
    let template = read_choice_fixture("templates/qwen3.5.jinja");
    let goldens = goldens();
    assert_eq!(
        sha256_hex(template.as_bytes()),
        goldens.provenance.template_sha256,
        "templates/qwen3.5.jinja is not the template the goldens were rendered from"
    );
    for case in &goldens.cases {
        let rendered = render_scoring_template(Some(&template), None, &case.messages, None)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(rendered, case.rendered, "{}: render differs", case.id);
    }

    let probe = "{{ bos_token }}|{{ eos_token }}|{% if enable_thinking is defined %}\
                 {{ enable_thinking }}{% else %}undefined{% endif %}";
    let render = |enable_thinking| {
        render_scoring_template(
            Some(probe),
            Some("</s>".to_string()),
            &[ChatMessage::user("x")],
            enable_thinking,
        )
        .expect("the probe renders")
    };
    assert_eq!(render(None), "|</s>|undefined");
    assert_eq!(render(Some(false)), "|</s>|False");
    assert_eq!(render(Some(true)), "|</s>|True");
}

/// Without an embedded template there is no scoring prompt, and a template
/// that fails to render has the engine's message withheld.
#[test]
fn the_scoring_render_refuses_without_quoting_the_request() {
    let messages = [ChatMessage::user(CANARY)];
    match render_scoring_template(None, None, &messages, None) {
        Err(AdapterError::InvalidInput(message)) => {
            assert!(message.contains("no embedded chat template"), "{message}");
            assert!(!message.contains(CANARY), "{message}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // minijinja quotes names from the template text, such as an unknown
    // function's; none of the engine's message reaches the scoring error.
    let marker = "probe_7f3a9c";
    let failing = format!("{{{{ {marker}(messages[0].content) }}}}");
    let raw = JinjaChatTemplate::new(failing.clone(), None, None)
        .render(&messages, &RenderOptions::default())
        .expect_err("the probe fails to render")
        .to_string();
    assert!(raw.contains(marker), "the probe's error names it: {raw}");
    match render_scoring_template(Some(&failing), None, &messages, None) {
        Err(AdapterError::RuntimeError(message)) => {
            assert!(message.contains("failed to render"), "{message}");
            assert!(
                !message.contains(marker) && !message.contains(CANARY),
                "{message}"
            );
        }
        other => panic!("expected RuntimeError, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// G0–G3 on the conformance GGUFs
// ---------------------------------------------------------------------------

/// G0: the GGUF embeds the template the goldens were rendered from, and the
/// scoring path renders, tokenizes and resolves the labels of every case
/// exactly as Hugging Face did.
#[test]
fn g0_prompts_tokens_and_labels_match_the_goldens() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let backend = load(&path, cpu);
    let model = backend.loaded_model().expect("a loaded model");
    let goldens = goldens();
    let template = model
        .chat_template()
        .expect("the GGUF embeds a chat template");
    assert_eq!(
        sha256_hex(template.as_bytes()),
        goldens.provenance.template_sha256,
        "the GGUF's embedded template is not the one the goldens were rendered from"
    );

    for case in &goldens.cases {
        let rendered = chat::render_scoring_prompt(model, &case.messages, None)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(rendered, case.rendered, "{}: rendered prompt", case.id);
        let tokens = model
            .tokenize_special(&rendered, false)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(tokens, case.token_ids, "{}: prompt tokens", case.id);
        let labels = label_token_ids(model, &singleton_groups(&case.labels))
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(labels, case.label_ids, "{}: label tokens", case.id);
    }
}

/// G1: the F16 GGUF, with the backend's default settings, against the
/// Hugging Face float32 reference, within `gates.labels.g1`. Near ties are
/// exempt from the argmax check only; the numeric bounds apply to every case.
#[test]
fn g1_f16_scores_match_the_reference() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let gates = label_gates();
    let goldens = goldens();
    let backend = load(&path, |_| {});
    let drift = Drift::of(
        &score_all(&backend, &goldens),
        &goldens,
        gates.g1.near_tie_margin,
    );
    eprintln!(
        "labels G1 (F16, default settings, {}, gates {}): {drift:?}",
        local_execution_provider("llama-cpp"),
        gates.status
    );
    drift.assert_within(gates.g1, "G1");
    assert_eq!(drift.near_ties, ["system-prompt", "near-tie-season"]);
}

/// G2: the backend and the C++ harness, given the same GGUF, prompt tokens,
/// context and batch sizes on the CPU, read the same final row: label logits
/// and scores within `gates.labels.g2`.
#[test]
fn g2_scores_match_the_harness_on_the_same_gguf() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let Some(harness) = harness() else {
        return;
    };
    let gates = label_gates();
    let goldens = goldens();
    let backend = load(&path, cpu);
    let n_ctx = backend.config.as_ref().expect("a config").context_length;
    let n_batch = {
        let context = backend.context.lock().expect("the context lock");
        context.as_ref().expect("a context").n_batch()
    };
    let model = backend.loaded_model().expect("a loaded model");

    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    for case in &goldens.cases {
        let groups = singleton_groups(&case.labels);
        let prompt = chat::render_scoring_prompt(model, &case.messages, None)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let tokens = model
            .tokenize_special(&prompt, false)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let labels = label_token_ids(model, &groups).unwrap_or_else(|e| panic!("{}: {e}", case.id));

        ours.push(
            backend
                .score_labels(&request(case))
                .unwrap_or_else(|e| panic!("{}: {e}", case.id)),
        );
        let (logsumexp, label_logits) =
            harness_row(&harness, &path, n_ctx, n_batch, &tokens, &labels);
        theirs.push(LabelLogits::new(
            label_logits.into_iter().map(|logit| vec![logit]).collect(),
            logsumexp,
        ));
    }

    let mut drift = Drift::default();
    for ((case, ours), theirs) in goldens.cases.iter().zip(&ours).zip(&theirs) {
        let (our_scores, their_scores) = (scores(case, ours), scores(case, theirs));
        for (our, their) in our_scores.entries.iter().zip(&their_scores.entries) {
            drift.max_abs_logit = drift
                .max_abs_logit
                .max((f64::from(our.logit) - f64::from(their.logit)).abs());
            drift.max_abs_score = drift
                .max_abs_score
                .max((f64::from(our.score) - f64::from(their.score)).abs());
        }
        drift.max_abs_logsumexp = drift
            .max_abs_logsumexp
            .max((f64::from(ours.logsumexp) - f64::from(theirs.logsumexp)).abs());
        drift.max_abs_label_mass = drift.max_abs_label_mass.max(
            (f64::from(our_scores.label_mass.expect("a mass"))
                - f64::from(their_scores.label_mass.expect("a mass")))
            .abs(),
        );
        let margin = their_scores.margin().map_or(f64::INFINITY, f64::from);
        if margin < gates.g2.near_tie_margin {
            drift.near_ties.push(case.id.clone());
        } else if our_scores.best() != their_scores.best() {
            drift.argmax_differs.push(case.id.clone());
        }
    }
    eprintln!(
        "labels G2 (F16, CPU, n_ctx {n_ctx}, n_batch {n_batch}, bit for bit: {}): {drift:?}",
        ours == theirs
    );
    drift.assert_within(gates.g2, "G2");
}

/// G3: the Q4_K_M GGUF, with the backend's default settings, against the
/// reference. A drift report written to `target/choice-conformance/g3.json`,
/// not a gate: it never fails on the numbers.
#[test]
fn g3_reports_the_q4_k_m_drift() {
    let Some(path) = staged_artifact(Q4_ID, Q4_FILE) else {
        return;
    };
    let gates = label_gates();
    let goldens = goldens();
    let backend = load(&path, |_| {});
    let drift = Drift::of(
        &score_all(&backend, &goldens),
        &goldens,
        gates.g1.near_tie_margin,
    );
    eprintln!("labels G3 (Q4_K_M, default settings): {drift:?}");
    write_report(
        "g3.json",
        &serde_json::json!({
            "gate": "G3: Q4_K_M against the Hugging Face float32 reference (report only)",
            "model": format!("{Q4_ID}/{Q4_FILE}"),
            "sha256": file_sha256(&path),
            "platform": platform(),
            "execution_provider": local_execution_provider("llama-cpp"),
            "settings": "LlmConfig defaults",
            "near_tie_margin": gates.g1.near_tie_margin,
            "drift": drift,
        }),
    );
}

/// The spread `gates.labels` is frozen from: the F16 GGUF against the
/// reference at 1, 4 and 8 threads, on the CPU and with the default GPU
/// offload, five repeats each. Written to
/// `target/choice-conformance/label-spread.json`; never fails on the numbers.
#[test]
fn label_spread_report() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let gates = label_gates();
    let goldens = goldens();
    let default_gpu_layers = LlmConfig::default().gpu_layers;

    let mut runs = Vec::new();
    let mut worst = Drift::default();
    let mut first_cpu_run: Option<Vec<LabelLogits>> = None;
    for gpu_layers in [0, default_gpu_layers] {
        for threads in [1, 4, 8] {
            let backend = load(&path, |config| {
                config.gpu_layers = gpu_layers;
                config.n_threads = threads;
            });
            let first = score_all(&backend, &goldens);
            let repeats_equal = (1..5).all(|_| score_all(&backend, &goldens) == first);
            let drift = Drift::of(&first, &goldens, gates.g1.near_tie_margin);
            worst.include(&drift);
            let equals_first_cpu_run = match &first_cpu_run {
                Some(cpu_run) => *cpu_run == first,
                None => {
                    first_cpu_run = Some(first.clone());
                    true
                }
            };
            eprintln!(
                "labels spread: gpu_layers {gpu_layers}, {threads} threads: repeats equal \
                 {repeats_equal}, equal to the first CPU run {equals_first_cpu_run}, {drift:?}"
            );
            runs.push(serde_json::json!({
                "gpu_layers": gpu_layers,
                "execution_provider": if gpu_layers == 0 {
                    "cpu"
                } else {
                    local_execution_provider("llama-cpp")
                },
                "threads": threads,
                "repeats": 5,
                "repeats_bit_for_bit_equal": repeats_equal,
                "equal_to_the_first_cpu_run": equals_first_cpu_run,
                "drift": drift,
            }));
        }
    }
    eprintln!("labels spread, worst over every run: {worst:?}");
    write_report(
        "label-spread.json",
        &serde_json::json!({
            "measures": "F16 against the Hugging Face float32 reference, per configuration",
            "model": format!("{F16_ID}/{F16_FILE}"),
            "sha256": file_sha256(&path),
            "platform": platform(),
            "settings": "LlmConfig defaults except gpu_layers and n_threads",
            "gates_status": gates.status,
            "worst": worst,
            "runs": runs,
        }),
    );
}

/// The C++ harness binary, or `None` (with a note) when it is not built.
///
/// # Panics
///
/// When models are required: G2 is the same-artifact gate.
fn harness() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XYBRID_LLAMA_HARNESS") {
        return Some(PathBuf::from(path));
    }
    let reason = "XYBRID_LLAMA_HARNESS is not set; build the harness with \
                  tools/conformance/llama_harness/build.sh and set it to the path that prints";
    if models_required() {
        panic!("{ENV_REQUIRE_MODELS} is set but {reason}");
    }
    eprintln!("Skipping test: {reason}");
    None
}

/// Runs the harness on one request: the row's log-sum-exp and each label's
/// logit.
fn harness_row(
    harness: &Path,
    model: &Path,
    n_ctx: usize,
    n_batch: usize,
    prompt: &[i32],
    labels: &[i32],
) -> (f32, Vec<f32>) {
    let join = |ids: &[i32]| ids.iter().map(i32::to_string).collect::<Vec<_>>().join(" ");
    let mut child = Command::new(harness)
        .arg(model)
        .args([n_ctx.to_string(), n_batch.to_string(), "4".to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the harness");
    child
        .stdin
        .take()
        .expect("the harness's stdin")
        .write_all(format!("{}\n{}\n", join(prompt), join(labels)).as_bytes())
        .expect("send the request");
    let output = child.wait_with_output().expect("run the harness");
    assert!(
        output.status.success(),
        "the harness failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8(output.stdout).expect("UTF-8 output");
    let values = |line: Option<&str>, name: &str| -> Vec<f32> {
        let line = line.unwrap_or_else(|| panic!("no {name} line"));
        let rest = line
            .strip_prefix(name)
            .unwrap_or_else(|| panic!("expected {name}, got {line:.40}"));
        rest.split_whitespace()
            .map(|bits| f32::from_bits(bits.parse().expect("an f32 bit pattern")))
            .collect()
    };
    let mut lines = text.lines();
    let logsumexp = values(lines.next(), "logsumexp");
    let labels = values(lines.next(), "labels");
    assert_eq!(logsumexp.len(), 1, "one log-sum-exp");
    (logsumexp[0], labels)
}

// ---------------------------------------------------------------------------
// State and request rules on the F16 GGUF
// ---------------------------------------------------------------------------

/// Scores do not depend on what the backend ran before: scoring A, then B,
/// then generating, then A again gives what a fresh backend gives each time,
/// bit for bit.
#[test]
fn scores_do_not_depend_on_what_ran_before() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let goldens = goldens();
    let (a, b) = (request(&goldens.cases[0]), request(&goldens.cases[1]));
    let fresh_a = load(&path, cpu).score_labels(&a).expect("score A");
    let fresh_b = load(&path, cpu).score_labels(&b).expect("score B");

    let backend = load(&path, cpu);
    assert_eq!(
        backend.score_labels(&a).expect("score A"),
        fresh_a,
        "A first"
    );
    assert_eq!(
        backend.score_labels(&b).expect("score B"),
        fresh_b,
        "B after A"
    );
    backend
        .generate(&goldens.cases[1].messages, &greedy(8))
        .expect("generate");
    assert_eq!(
        backend.score_labels(&a).expect("score A"),
        fresh_a,
        "A after a generation"
    );
}

/// Scoring forgets the prompt prefix the last generation cached, since it
/// clears the native memory that held it, and a generation after scoring is
/// a fresh backend's, token for token.
#[test]
fn generation_after_scoring_matches_a_fresh_generation() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let goldens = goldens();
    let case = &goldens.cases[1];
    let config = greedy(16);
    let fresh = load(&path, cpu)
        .generate(&case.messages, &config)
        .expect("generate");

    let backend = load(&path, cpu);
    backend.generate(&case.messages, &config).expect("generate");
    assert!(
        !backend.kv_state.lock().unwrap().cached_tokens.is_empty(),
        "a generation records the prompt it cached"
    );
    backend.score_labels(&request(case)).expect("score");
    {
        let state = backend.kv_state.lock().unwrap();
        assert!(state.cached_tokens.is_empty(), "scoring forgets the prefix");
        assert_eq!(state.last_prefix_hit, None);
    }
    let after = backend.generate(&case.messages, &config).expect("generate");
    assert_eq!(after.text, fresh.text);
    assert_eq!(after.tokens_generated, fresh.tokens_generated);
    assert_eq!(after.finish_reason, fresh.finish_reason);
}

/// Every refusal names positions and counts, never the request's text, and
/// leaves nothing a later request can see: afterwards a valid request scores
/// as on a fresh backend.
#[test]
fn refused_requests_leave_the_backend_as_it_was() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let goldens = goldens();
    let case = &goldens.cases[0];
    let fresh = load(&path, cpu)
        .score_labels(&request(case))
        .expect("score");

    let backend = load(&path, cpu);
    let messages = vec![ChatMessage::user(format!("{CANARY} which one?"))];
    let label = |text: &str| vec![text.to_string()];
    let refusals = [
        ("no groups", messages.clone(), vec![]),
        (
            "an empty group",
            messages.clone(),
            vec![label("A"), Vec::new()],
        ),
        (
            "a multi-token label",
            messages.clone(),
            vec![label("A"), label(CANARY)],
        ),
        (
            "a repeated token",
            messages.clone(),
            vec![label("A"), label("A")],
        ),
        (
            "a prompt longer than the context",
            vec![ChatMessage::user(format!("{CANARY} ").repeat(2048))],
            vec![label("A"), label("B")],
        ),
    ];
    for (what, messages, groups) in refusals {
        match backend.score_labels(&ScoringRequest::new(messages, groups)) {
            Err(AdapterError::InvalidInput(message)) => {
                assert!(!message.contains(CANARY), "{what}: {message}");
                eprintln!("{what}: {message}");
            }
            other => panic!("{what}: expected InvalidInput, got {other:?}"),
        }
    }
    assert_eq!(
        backend.score_labels(&request(case)).expect("score"),
        fresh,
        "a valid request after the refusals"
    );
}

/// A group's logits are the per-token logits of the same row, and the group
/// scores its largest: `"A"` with `" A"` against each read on its own.
#[test]
fn groups_read_each_label_from_the_same_row() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let goldens = goldens();
    let case = &goldens.cases[1];
    let backend = load(&path, cpu);
    let labels = |groups: &[&[&str]]| -> Vec<Vec<String>> {
        groups
            .iter()
            .map(|group| group.iter().map(|label| (*label).to_string()).collect())
            .collect()
    };
    let grouped = backend
        .score_labels(&ScoringRequest::new(
            case.messages.clone(),
            labels(&[&["A", " A"], &["B", " B"]]),
        ))
        .expect("score groups");
    let single = backend
        .score_labels(&ScoringRequest::new(
            case.messages.clone(),
            labels(&[&["A"], &[" A"], &["B"], &[" B"]]),
        ))
        .expect("score each label");

    assert_eq!(grouped.logsumexp, single.logsumexp);
    let each: Vec<f32> = single.label_logits.iter().map(|group| group[0]).collect();
    assert_eq!(
        grouped.label_logits,
        [vec![each[0], each[1]], vec![each[2], each[3]]]
    );
    let scored = label_scores(
        vec![("a".into(), false), ("b".into(), false)],
        &grouped.label_logits,
        grouped.logsumexp,
        None,
    )
    .expect("scores");
    assert_eq!(scored.entries[0].logit, each[0].max(each[1]));
    assert_eq!(scored.entries[1].logit, each[2].max(each[3]));
}

/// Reordering the label groups permutes the logits and changes nothing else.
#[test]
fn reordered_groups_permute_the_logits() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let case = goldens()
        .cases
        .into_iter()
        .max_by_key(|case| case.labels.len())
        .expect("a case");
    let backend = load(&path, cpu);
    let groups = singleton_groups(&case.labels);
    let reversed: Vec<Vec<String>> = groups.iter().rev().cloned().collect();
    let forward = backend
        .score_labels(&ScoringRequest::new(case.messages.clone(), groups))
        .expect("score");
    let backward = backend
        .score_labels(&ScoringRequest::new(case.messages.clone(), reversed))
        .expect("score reversed");

    assert_eq!(forward.logsumexp, backward.logsumexp);
    let mut unreversed = backward.label_logits;
    unreversed.reverse();
    assert_eq!(forward.label_logits, unreversed);
}

/// Scoring takes the context lock, then the prefix-cache lock, as generation
/// does; holding the second while asking for it again would hang instead of
/// failing, so a worker thread must return within a deadline.
#[test]
fn scoring_returns_within_a_deadline() {
    let Some(path) = staged_artifact(F16_ID, F16_FILE) else {
        return;
    };
    let backend = Arc::new(load(&path, cpu));
    let scoring = request(&goldens().cases[0]);
    let (sender, receiver) = mpsc::channel();
    let worker = Arc::clone(&backend);
    std::thread::spawn(move || {
        let result = worker.score_labels(&scoring).map_err(|e| e.to_string());
        let _ = sender.send(result);
    });
    match receiver.recv_timeout(Duration::from_secs(120)) {
        Ok(result) => {
            result.expect("scoring succeeds");
        }
        Err(_) => panic!("score_labels did not return within 120 s; is a lock taken twice?"),
    }
}

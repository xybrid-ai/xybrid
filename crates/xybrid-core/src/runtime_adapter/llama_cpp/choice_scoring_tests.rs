//! Label scoring through [`LlamaCppBackend`] on the Qwen3.5-0.8B
//! conformance GGUFs, against `integration-tests/fixtures/choice` (pinned by
//! its `provenance.json`):
//!
//! - state: scoring and generation interleave as they would on fresh
//!   backends, a refused request included;
//! - label rules: groups resolve to distinct single tokens, take the largest
//!   of their tokens' logits, and follow the request's order;
//! - G0: the GGUF's embedded template, the rendered prompt bytes, the prompt's
//!   token ids and the label ids equal `goldens/labels.json`;
//! - G1: the F16 GGUF against the float32 reference, within
//!   `gates.labels.g1`, on the CPU and with the backend's default offload;
//! - G2: the same GGUF and settings through the C++ harness in
//!   `tools/conformance/llama_harness`, within `gates.labels.g2`;
//! - G3: the Q4_K_M GGUF's drift from the reference, written to
//!   `target/choice-conformance/reports/label-token-g3.json` and never failing.
//!
//! Stage the models with `tools/conformance/prepare.sh qwen3.5-0.8b-f16-cf
//! qwen3.5-0.8b-q4km-cf`. G2 needs the harness: set `XYBRID_LLAMA_HARNESS` to
//! the path `tools/conformance/llama_harness/build.sh` prints. A test skips
//! without what it needs, unless `XYBRID_REQUIRE_MODELS` is set, where it
//! fails instead.
//!
//! On a build with Metal, llama.cpp runs large matrix products on the GPU
//! even with no GPU layers (its default operation offload). Set
//! `GGML_OP_OFFLOAD_MIN_BATCH=1000000` so the CPU settings measure the CPU,
//! as CPU-only builds run; G1 requires it there when models are required.

use super::scoring::scoring_tokens;
use super::{chat, LlamaCppBackend};
use crate::execution::choice::math::{choice_scores, label_mass};
use crate::ir::ChoiceScores;
use crate::runtime_adapter::llm::{
    local_execution_provider, ChatMessage, GenerationConfig, LabelGroup, LabelLogits, LlmBackend,
    LlmConfig, ScoringRequest,
};
use crate::runtime_adapter::AdapterError;
use crate::testing::model_fixtures::{models_required, staged_artifact};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc};
use std::time::Duration;

const CANARY: &str = "canary-3f9a1c";

/// The context and batch every test loads with: the batch covers each
/// golden prompt in one decode, so every setting cuts the prompts alike.
const N_CTX: usize = 2048;
const N_BATCH: usize = 512;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A staged conformance GGUF.
struct Artifact {
    id: &'static str,
    file: &'static str,
}

const F16: Artifact = Artifact {
    id: "qwen3.5-0.8b-f16-cf",
    file: "Qwen3.5-0.8B-F16.gguf",
};

const Q4_K_M: Artifact = Artifact {
    id: "qwen3.5-0.8b-q4km-cf",
    file: "Qwen3.5-0.8B-Q4_K_M.gguf",
};

impl Artifact {
    /// Its path, or `None` (skipped) when it is not staged.
    fn staged(&self) -> Option<PathBuf> {
        staged_artifact(self.id, self.file)
    }
}

fn workspace_root() -> PathBuf {
    // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs in
    // a sandbox where the compile-time path points nowhere.
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

fn read_json<T: for<'de> Deserialize<'de>>(relative: &str) -> T {
    serde_json::from_str(&read_choice_fixture(relative))
        .unwrap_or_else(|e| panic!("parse {relative}: {e}"))
}

#[derive(Deserialize)]
struct LabelGoldens {
    provenance: GoldenProvenance,
    cases: Vec<LabelCase>,
}

#[derive(Deserialize)]
struct GoldenProvenance {
    template_sha256: String,
}

#[derive(Deserialize)]
struct LabelCase {
    id: String,
    messages: Vec<ChatMessage>,
    rendered: String,
    token_ids: Vec<i32>,
    labels: Vec<String>,
    label_ids: Vec<i32>,
    label_logits: Vec<f64>,
    scores: Vec<f64>,
    label_mass: f64,
    best: usize,
    margin: f64,
}

impl LabelCase {
    /// The case as the goldens scored it: one single-label group per letter,
    /// the template's thinking flag omitted.
    fn request(&self) -> ScoringRequest {
        ScoringRequest::new(
            self.messages.clone(),
            self.labels.iter().map(LabelGroup::new).collect(),
        )
    }
}

fn goldens() -> LabelGoldens {
    let goldens: LabelGoldens = read_json("goldens/labels.json");
    assert!(
        !goldens.cases.is_empty(),
        "goldens/labels.json has no cases"
    );
    goldens
}

fn golden(id: &str) -> LabelCase {
    goldens()
        .cases
        .into_iter()
        .find(|case| case.id == id)
        .unwrap_or_else(|| panic!("goldens/labels.json has no case {id}"))
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
    g1: G1Bounds,
    g2: Bounds,
}

/// Largest allowed absolute differences; 0 means bit for bit.
#[derive(Deserialize, Clone, Copy)]
struct Bounds {
    max_abs_logit: f64,
    max_abs_score: f64,
    max_abs_label_mass: f64,
}

#[derive(Deserialize, Clone, Copy)]
struct G1Bounds {
    #[serde(flatten)]
    within: Bounds,
    /// Below this reference margin the best label may differ.
    near_tie_margin: f64,
}

/// The label track's bounds, which must have been measured and frozen.
fn label_gates() -> LabelGates {
    let gates = read_json::<Provenance>("provenance.json").gates.labels;
    assert_eq!(
        gates.status, "frozen",
        "the label bounds were measured and frozen; loosening them needs a justification, \
         not a status change"
    );
    gates
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// The load settings the scores depend on.
#[derive(Clone, Copy, Debug)]
struct Settings {
    gpu_layers: i32,
    flash_attn: bool,
    threads: usize,
}

/// No GPU layers, flash attention off, as the harness runs by default.
/// Repeated runs are equal bit for bit. On a Metal build this is the CPU
/// only when operation offload is off (see the module docs).
const CPU: Settings = Settings {
    gpu_layers: 0,
    flash_attn: false,
    threads: 4,
};

/// No GPU layers with flash attention on, the backend's default: what a
/// build without a GPU backend runs.
const CPU_FLASH: Settings = Settings {
    flash_attn: true,
    ..CPU
};

/// The backend's own defaults: every layer offloaded where a GPU backend is
/// built in (Metal on Apple), flash attention on.
fn defaults() -> Settings {
    let config = LlmConfig::default();
    Settings {
        gpu_layers: config.gpu_layers,
        flash_attn: config.flash_attn,
        threads: 4,
    }
}

/// Every setting the numerical gates cover.
fn gated_settings() -> [Settings; 3] {
    [CPU, CPU_FLASH, defaults()]
}

/// Whether the settings without GPU layers really run on the CPU. On a Metal
/// build, llama.cpp's operation offload sends matrix products with a batch
/// of `GGML_OP_OFFLOAD_MIN_BATCH` (32 by default) or more to the GPU even
/// then; a threshold above the context size keeps every one on the CPU.
fn cpu_settings_run_on_the_cpu() -> bool {
    local_execution_provider("llama-cpp") != "metal"
        || std::env::var("GGML_OP_OFFLOAD_MIN_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|min_batch| min_batch > N_CTX)
}

fn load(model: &Path, settings: Settings) -> LlamaCppBackend {
    let mut config = LlmConfig::new(model.to_str().expect("a UTF-8 model path"))
        .with_context_length(N_CTX)
        .with_gpu_layers(settings.gpu_layers);
    config.n_threads = settings.threads;
    config.n_batch = N_BATCH;
    config.flash_attn = settings.flash_attn;
    let mut backend = LlamaCppBackend::new().expect("create the backend");
    backend.load(&config).expect("load the GGUF");
    backend
}

fn score(backend: &dyn LlmBackend, request: &ScoringRequest) -> LabelLogits {
    backend
        .score_labels(request)
        .unwrap_or_else(|error| panic!("score: {error}"))
}

/// The scores a label scorer reports from `read`: the softmax of the group
/// logits at temperature 1, with the label mass.
fn choice_scores_of(read: &LabelLogits) -> ChoiceScores {
    let ids = (0..read.logits.len())
        .map(|index| (index.to_string(), false))
        .collect();
    let mut scores = choice_scores(ids, &read.logits, None).expect("finite label logits");
    scores.label_mass = Some(read.label_mass);
    scores
}

fn single_labels(labels: &[&str]) -> Vec<LabelGroup> {
    labels.iter().copied().map(LabelGroup::new).collect()
}

fn greedy(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        ..GenerationConfig::greedy()
    }
}

// ---------------------------------------------------------------------------
// State: every call runs as on a fresh backend
// ---------------------------------------------------------------------------

/// Scoring another prompt in between changes nothing, and a used backend
/// scores like a fresh one.
#[test]
fn scoring_repeats_bit_for_bit_after_other_scoring() {
    let Some(model) = F16.staged() else { return };
    let (first, second) = (golden("forms-phone"), golden("multilingual"));
    let backend = load(&model, CPU);

    let a = score(&backend, &first.request());
    let b = score(&backend, &second.request());
    let a_again = score(&backend, &first.request());
    let fresh = score(&load(&model, CPU), &first.request());

    assert_ne!(a, b, "different prompts score differently");
    assert_eq!(
        a, a_again,
        "scoring another prompt in between changes nothing"
    );
    assert_eq!(a, fresh, "a used backend scores like a fresh one");
}

/// Generation then scoring, and scoring then generation, give what each
/// gives on a fresh backend. Runs on another thread through the trait object,
/// the route the executor takes, with a deadline: a lock-order deadlock fails
/// here instead of hanging the suite.
#[test]
fn scoring_and_generation_interleave_as_on_fresh_backends() {
    let Some(model) = F16.staged() else { return };
    let request = golden("binary").request();
    let chat = vec![ChatMessage::user("Name the colour of a clear sky.")];
    let config = greedy(8);
    let backend: Arc<dyn LlmBackend> = Arc::new(load(&model, CPU));

    let (sender, receiver) = mpsc::channel();
    let worker = Arc::clone(&backend);
    let (worker_request, worker_chat, worker_config) =
        (request.clone(), chat.clone(), config.clone());
    std::thread::spawn(move || {
        let generated = worker
            .generate(&worker_chat, &worker_config)
            .map(|out| out.text);
        let scored = worker.score_labels(&worker_request);
        let generated_again = worker
            .generate(&worker_chat, &worker_config)
            .map(|out| out.text);
        let scored_again = worker.score_labels(&worker_request);
        let _ = sender.send((generated, scored, generated_again, scored_again));
    });
    let (generated, scored, generated_again, scored_again) = receiver
        .recv_timeout(Duration::from_secs(300))
        .expect("scoring and generation finish within the deadline");
    let generated = generated.expect("generate on a fresh backend");
    let scored = scored.expect("score after generating");

    let fresh = load(&model, CPU);
    assert_eq!(
        scored,
        score(&fresh, &request),
        "scoring after generation matches a fresh backend"
    );
    assert_eq!(
        generated_again.expect("generate after scoring"),
        generated,
        "generation after scoring matches the first, fresh generation"
    );
    assert_eq!(
        scored_again.expect("score after generating again"),
        scored,
        "scoring repeats after another generation"
    );
}

/// Generation records the prompt it left in the cache; scoring clears the
/// cache, so it must forget that record too, or the next generation would
/// reuse a prefix that is no longer there.
#[test]
fn scoring_forgets_the_cached_prefix() {
    let Some(model) = F16.staged() else { return };
    let backend = load(&model, CPU);
    backend
        .generate(&[ChatMessage::user("Say hi.")], &greedy(2))
        .expect("generate");
    assert!(
        !backend.kv_state.lock().unwrap().cached_tokens.is_empty(),
        "generation records the prompt it cached"
    );

    score(&backend, &golden("binary").request());

    let state = backend.kv_state.lock().unwrap();
    assert!(
        state.cached_tokens.is_empty(),
        "no cached prefix is recorded"
    );
    assert_eq!(state.last_prefix_hit, None);
}

/// A request refused while the readout checks it, here a prompt that fills
/// the context window, finds the memory cleared, so it must also forget the
/// prefix generation recorded; the backend then scores and generates as a
/// fresh one does. A request refused earlier, before the readout starts,
/// keeps that record: the memory is untouched.
#[test]
fn a_refused_request_leaves_nothing_behind() {
    let Some(model) = F16.staged() else { return };
    let request = golden("near-tie-season").request();
    let chat = [ChatMessage::user("Name a fruit that is yellow.")];
    let fresh = load(&model, CPU);
    let expected_scores = score(&fresh, &request);
    let expected_text = fresh.generate(&chat, &greedy(8)).expect("generate").text;
    let too_long = ScoringRequest::new(
        vec![ChatMessage::user(format!("{CANARY} ").repeat(N_CTX))],
        single_labels(&["A", "B"]),
    );
    let bad_labels = ScoringRequest::new(chat.to_vec(), single_labels(&["A", "A"]));
    let cached = |backend: &LlamaCppBackend| backend.kv_state.lock().unwrap().cached_tokens.clone();

    let backend = load(&model, CPU);
    for _ in 0..2 {
        backend.generate(&chat, &greedy(8)).expect("generate");
        let recorded = cached(&backend);
        assert!(
            !recorded.is_empty(),
            "generation records the prompt it cached"
        );

        backend
            .score_labels(&bad_labels)
            .expect_err("a token two groups share is refused");
        assert_eq!(
            cached(&backend),
            recorded,
            "a refusal before the readout keeps the record"
        );

        let refused = backend
            .score_labels(&too_long)
            .expect_err("a prompt that fills the context is refused");
        match &refused {
            AdapterError::InvalidInput(message) => {
                assert!(message.contains("no free position"), "{message}");
                assert!(!message.contains(CANARY), "{message}");
            }
            other => panic!("expected invalid input, got {other:?}"),
        }
        assert!(
            cached(&backend).is_empty(),
            "the readout cleared the memory, so the record is gone too"
        );
    }
    assert_eq!(score(&backend, &request), expected_scores);
    backend.score_labels(&too_long).expect_err("refused again");
    assert_eq!(
        backend.generate(&chat, &greedy(8)).expect("generate").text,
        expected_text
    );
}

// ---------------------------------------------------------------------------
// Label rules
// ---------------------------------------------------------------------------

/// Every group needs a single-token label, and no two groups may share a
/// token. Refusals name group indexes only.
#[test]
fn label_groups_must_be_distinct_single_tokens() {
    let Some(model) = F16.staged() else { return };
    let backend = load(&model, CPU);
    let messages = vec![ChatMessage::user(format!("{CANARY}: pick one."))];
    let refusals: [(Vec<LabelGroup>, &str); 5] = [
        (Vec::new(), "no label groups"),
        (
            single_labels(&["A", "canary-3f9a1c is several tokens"]),
            "label group 1 is not a single token",
        ),
        (
            single_labels(&["", "B"]),
            "label group 0 is not a single token",
        ),
        (
            single_labels(&["A", "B", "A"]),
            "label groups 0 and 2 share a token",
        ),
        (
            vec![
                LabelGroup::new("A"),
                LabelGroup::with_alternatives("B", ["A"]),
            ],
            "label groups 0 and 1 share a token",
        ),
    ];

    for (groups, expected) in refusals {
        let refused = backend
            .score_labels(&ScoringRequest::new(messages.clone(), groups))
            .expect_err(expected);
        match &refused {
            AdapterError::InvalidInput(message) => {
                assert!(message.contains(expected), "{message}");
                assert!(!message.contains(CANARY), "{message}");
            }
            other => panic!("expected invalid input, got {other:?}"),
        }
    }
    score(
        &backend,
        &ScoringRequest::new(messages, single_labels(&["A", "B"])),
    );
}

/// A group scores the largest of its tokens' logits, as reading each token
/// on its own shows, and the label mass counts each token once. Spellings
/// that are not one token, or repeat one, change nothing.
#[test]
fn a_group_scores_the_largest_of_its_tokens() {
    let Some(model) = F16.staged() else { return };
    let messages = golden("binary").messages;
    let backend = load(&model, CPU);
    let request = |groups| ScoringRequest::new(messages.clone(), groups);
    let grouped = request(vec![
        LabelGroup::with_alternatives("A", [" A"]),
        LabelGroup::with_alternatives("B", [" B"]),
    ]);
    let model_ref = backend.model.as_ref().expect("loaded");
    assert_eq!(
        scoring_tokens(model_ref, &grouped)
            .expect("resolve the groups")
            .group_sizes,
        [2, 2],
        "each spaced letter is a token of its own"
    );

    let each = score(&backend, &request(single_labels(&["A", " A", "B", " B"])));
    let pooled = score(&backend, &grouped);
    // The same tokens with the spaced form as the label: in one of the two
    // orders the larger logit is not the group's first token.
    let spaced_first = score(
        &backend,
        &request(vec![
            LabelGroup::with_alternatives(" A", ["A"]),
            LabelGroup::with_alternatives(" B", ["B"]),
        ]),
    );
    let padded = score(
        &backend,
        &request(vec![
            LabelGroup::with_alternatives("A", [" A", "A", "Answer A"]),
            LabelGroup::with_alternatives("B", ["B b", " B", " B"]),
        ]),
    );
    let maxima = [
        each.logits[0].max(each.logits[1]),
        each.logits[2].max(each.logits[3]),
    ];

    assert!(
        each.logits[0] != each.logits[1] && each.logits[2] != each.logits[3],
        "each group's two tokens score differently"
    );
    assert_eq!(pooled.logits, maxima);
    assert_eq!(spaced_first.logits, maxima);
    assert_eq!(pooled.label_mass, each.label_mass);
    assert!((spaced_first.label_mass - each.label_mass).abs() <= 1e-6);
    assert_eq!(padded, pooled);
}

/// The logits follow the groups' order: reversing the groups over the same
/// prompt reverses them exactly.
#[test]
fn reordering_the_groups_reorders_the_logits() {
    let Some(model) = F16.staged() else { return };
    let case = golden("forms-phone");
    let backend = load(&model, CPU);
    let mut reversed_labels = case.labels.clone();
    reversed_labels.reverse();

    let forward = score(&backend, &case.request());
    let mut reversed = score(
        &backend,
        &ScoringRequest::new(
            case.messages.clone(),
            reversed_labels.iter().map(LabelGroup::new).collect(),
        ),
    );
    reversed.logits.reverse();

    assert_eq!(reversed.logits, forward.logits);
    assert!((reversed.label_mass - forward.label_mass).abs() <= 1e-6);
}

/// On the CPU the thread count moves nothing: one, four and eight threads
/// give the same bits on every golden case.
#[test]
fn cpu_scores_do_not_move_with_the_thread_count() {
    let Some(model) = F16.staged() else { return };
    let cases = goldens().cases;
    let reference: Vec<LabelLogits> = {
        let backend = load(&model, CPU);
        cases
            .iter()
            .map(|case| score(&backend, &case.request()))
            .collect()
    };

    for threads in [1, 8] {
        let backend = load(&model, Settings { threads, ..CPU });
        for (case, expected) in cases.iter().zip(&reference) {
            assert_eq!(
                &score(&backend, &case.request()),
                expected,
                "{}: {threads} threads",
                case.id
            );
        }
    }
}

// ---------------------------------------------------------------------------
// G0: template, prompt bytes, token and label ids
// ---------------------------------------------------------------------------

fn sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// The GGUF embeds the pinned template, and the scoring path renders and
/// tokenizes every golden case exactly as the reference did, labels included.
#[test]
fn g0_template_prompt_and_label_ids_match_the_goldens() {
    let Some(model) = F16.staged() else { return };
    let goldens = goldens();
    let backend = load(&model, CPU);
    let llama = backend.model.as_ref().expect("loaded");

    let embedded = llama
        .chat_template()
        .expect("the GGUF embeds a chat template");
    assert_eq!(
        sha256_hex(&embedded),
        goldens.provenance.template_sha256,
        "the GGUF does not embed the template the goldens were rendered from"
    );
    assert_eq!(
        embedded,
        read_choice_fixture("templates/qwen3.5.jinja"),
        "the GGUF's template differs from the pinned templates/qwen3.5.jinja"
    );

    for case in &goldens.cases {
        let rendered = chat::render_scoring_prompt(llama, &case.messages, None)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(rendered, case.rendered, "{}: rendered prompt", case.id);
        let tokens =
            scoring_tokens(llama, &case.request()).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(
            tokens.prompt, case.token_ids,
            "{}: prompt token ids",
            case.id
        );
        assert_eq!(tokens.labels, case.label_ids, "{}: label ids", case.id);
        assert!(
            tokens.group_sizes.iter().all(|&size| size == 1),
            "{}: one token per letter",
            case.id
        );
    }
}

// ---------------------------------------------------------------------------
// G1: F16 against the float32 reference
// ---------------------------------------------------------------------------

/// The largest absolute differences seen.
#[derive(Default, Clone, Copy, Debug)]
struct Drift {
    logit: f64,
    score: f64,
    label_mass: f64,
}

impl Drift {
    fn of(case: &LabelCase, read: &LabelLogits, scores: &ChoiceScores) -> Self {
        let largest = |pairs: Vec<(f64, f64)>| {
            pairs
                .into_iter()
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max)
        };
        assert_eq!(read.logits.len(), case.label_logits.len(), "{}", case.id);
        Self {
            logit: largest(
                read.logits
                    .iter()
                    .map(|&logit| f64::from(logit))
                    .zip(case.label_logits.iter().copied())
                    .collect(),
            ),
            score: largest(
                scores
                    .entries
                    .iter()
                    .map(|entry| f64::from(entry.score))
                    .zip(case.scores.iter().copied())
                    .collect(),
            ),
            label_mass: (f64::from(read.label_mass) - case.label_mass).abs(),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            logit: self.logit.max(other.logit),
            score: self.score.max(other.score),
            label_mass: self.label_mass.max(other.label_mass),
        }
    }

    fn assert_within(&self, bounds: &Bounds, what: &str) {
        assert!(
            self.logit <= bounds.max_abs_logit,
            "{what}: |Δlogit| {:.3e} > {:.1e}",
            self.logit,
            bounds.max_abs_logit
        );
        assert!(
            self.score <= bounds.max_abs_score,
            "{what}: |Δscore| {:.3e} > {:.1e}",
            self.score,
            bounds.max_abs_score
        );
        assert!(
            self.label_mass <= bounds.max_abs_label_mass,
            "{what}: |Δlabel_mass| {:.3e} > {:.1e}",
            self.label_mass,
            bounds.max_abs_label_mass
        );
    }
}

/// The F16 GGUF scores every golden case within the frozen G1 bounds, on the
/// CPU with flash attention off and on and with the backend's defaults, and
/// picks the reference's best label wherever the reference's margin is not a
/// near tie.
#[test]
fn g1_f16_scores_match_the_reference() {
    let Some(model) = F16.staged() else { return };
    let bounds = label_gates().g1;
    let cases = goldens().cases;
    let cpu_only = cpu_settings_run_on_the_cpu();
    if models_required() {
        assert!(
            cpu_only,
            "set GGML_OP_OFFLOAD_MIN_BATCH above {N_CTX}: on a Metal build llama.cpp otherwise \
             runs large matrix products on the GPU with no GPU layers, so the CPU settings \
             would not measure the CPU"
        );
    }

    for settings in gated_settings() {
        let backend = load(&model, settings);
        let mut worst = Drift::default();
        let mut exempt = Vec::new();
        for case in &cases {
            let read = score(&backend, &case.request());
            let scores = choice_scores_of(&read);
            let drift = Drift::of(case, &read, &scores);
            drift.assert_within(&bounds.within, &format!("{} ({settings:?})", case.id));
            worst = worst.max(drift);
            if case.margin < bounds.near_tie_margin {
                exempt.push(case.id.clone());
            } else {
                assert_eq!(
                    scores.best(),
                    Some(case.best),
                    "{} ({settings:?}): argmax",
                    case.id
                );
            }
        }
        eprintln!(
            "label G1 {settings:?} (CPU settings on the CPU only: {cpu_only}): max |Δlogit| \
             {:.3e}, |Δscore| {:.3e}, |Δlabel_mass| {:.3e}; argmax exempt (margin < {}): \
             {exempt:?}",
            worst.logit, worst.score, worst.label_mass, bounds.near_tie_margin
        );
        assert_eq!(exempt, ["system-prompt", "near-tie-season"]);
    }
}

// ---------------------------------------------------------------------------
// G2: the same artifact through the C++ harness
// ---------------------------------------------------------------------------

/// The harness binary, or `None` (skipped) when it is not built. Required
/// whenever models are.
fn harness() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XYBRID_LLAMA_HARNESS") {
        return Some(PathBuf::from(path));
    }
    let reason = "XYBRID_LLAMA_HARNESS is not set to the path \
                  tools/conformance/llama_harness/build.sh prints";
    if models_required() {
        panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
    }
    eprintln!("Skipping test: {reason}");
    None
}

/// Runs the harness on one prompt with `settings`: the log-sum-exp and each
/// label's logit.
fn harness_row(
    harness: &Path,
    model: &Path,
    settings: Settings,
    prompt: &[i32],
    labels: &[i32],
) -> (f32, Vec<f32>) {
    let join = |ids: &[i32]| ids.iter().map(i32::to_string).collect::<Vec<_>>().join(" ");
    let mut child = Command::new(harness)
        .arg(model)
        .args([
            N_CTX.to_string(),
            N_BATCH.to_string(),
            settings.threads.to_string(),
            u8::from(settings.flash_attn).to_string(),
            settings.gpu_layers.to_string(),
        ])
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
        line.strip_prefix(name)
            .unwrap_or_else(|| panic!("expected {name}, got {line:.40}"))
            .split_whitespace()
            .map(|bits| f32::from_bits(bits.parse().expect("an f32 bit pattern")))
            .collect()
    };
    let mut lines = text.lines();
    let logsumexp = values(lines.next(), "logsumexp");
    let labels = values(lines.next(), "labels");
    assert_eq!(logsumexp.len(), 1, "one log-sum-exp");
    (logsumexp[0], labels)
}

/// A G2 request: what the backend scores, and the ids the harness reads.
struct HarnessCase {
    id: String,
    request: ScoringRequest,
    prompt: Vec<i32>,
    labels: Vec<i32>,
    group_sizes: Vec<usize>,
}

/// The largest of each consecutive run of `sizes` values.
fn group_maxima(values: &[f32], sizes: &[usize]) -> Vec<f32> {
    let mut start = 0;
    sizes
        .iter()
        .map(|&size| {
            let group = &values[start..start + size];
            start += size;
            group.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

/// The backend reads what the harness reads on the same GGUF, settings and
/// tokens, bit for bit under the frozen bounds: every golden case (with the
/// goldens' own token and label ids, independent of the backend's
/// tokenization) and a grouped request, on the CPU with flash attention off
/// and on, and with the backend's defaults.
#[test]
fn g2_scores_equal_the_harness_on_the_same_artifact() {
    let Some(model) = F16.staged() else { return };
    let Some(harness) = harness() else { return };
    let bounds = label_gates().g2;
    let cases = goldens().cases;
    let grouped = ScoringRequest::new(
        golden("max-26").messages,
        ('A'..='Z')
            .map(|letter| LabelGroup::with_alternatives(letter, [format!(" {letter}")]))
            .collect(),
    );

    for settings in gated_settings() {
        let backend = load(&model, settings);
        let llama = backend.model.as_ref().expect("loaded");
        let mut worst = Drift::default();
        let mut requests: Vec<HarnessCase> = cases
            .iter()
            .map(|case| HarnessCase {
                id: case.id.clone(),
                request: case.request(),
                prompt: case.token_ids.clone(),
                labels: case.label_ids.clone(),
                group_sizes: vec![1; case.label_ids.len()],
            })
            .collect();
        let tokens = scoring_tokens(llama, &grouped).expect("resolve the grouped request");
        assert!(
            tokens.group_sizes.iter().all(|&size| size == 2),
            "every spaced letter is a token of its own"
        );
        requests.push(HarnessCase {
            id: "max-26 grouped".to_string(),
            request: grouped.clone(),
            prompt: tokens.prompt,
            labels: tokens.labels,
            group_sizes: tokens.group_sizes,
        });

        for case in &requests {
            let read = score(&backend, &case.request);
            let (logsumexp, label_logits) =
                harness_row(&harness, &model, settings, &case.prompt, &case.labels);
            let expected_logits = group_maxima(&label_logits, &case.group_sizes);
            let expected_mass = label_mass(&label_logits, logsumexp).expect("finite harness row");
            let expected_scores =
                choice_scores_of(&LabelLogits::new(expected_logits.clone(), expected_mass));
            let scores = choice_scores_of(&read);

            let largest = |pairs: Vec<(f32, f32)>| {
                pairs
                    .into_iter()
                    .map(|(a, b)| (f64::from(a) - f64::from(b)).abs())
                    .fold(0.0, f64::max)
            };
            let drift = Drift {
                logit: largest(read.logits.iter().copied().zip(expected_logits).collect()),
                score: largest(
                    scores
                        .entries
                        .iter()
                        .zip(&expected_scores.entries)
                        .map(|(ours, theirs)| (ours.score, theirs.score))
                        .collect(),
                ),
                label_mass: (f64::from(read.label_mass) - f64::from(expected_mass)).abs(),
            };
            drift.assert_within(&bounds, &format!("{} ({settings:?})", case.id));
            worst = worst.max(drift);
        }
        eprintln!(
            "label G2 {settings:?}: max |Δlogit| {:.3e}, |Δscore| {:.3e}, |Δlabel_mass| {:.3e} \
             over {} requests",
            worst.logit,
            worst.score,
            worst.label_mass,
            requests.len()
        );
    }
}

// ---------------------------------------------------------------------------
// G3: Q4_K_M drift, reported only
// ---------------------------------------------------------------------------

fn g3_report_path() -> PathBuf {
    std::env::var_os("XYBRID_CONFORMANCE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("target/choice-conformance"))
        .join("reports/label-token-g3.json")
}

/// Writes how far the Q4_K_M GGUF scores from the float32 reference, per case
/// and setting, to `reports/label-token-g3.json`. Quantization drift is
/// expected: nothing here fails on the numbers.
#[test]
fn g3_q4_k_m_drift_is_reported() {
    let Some(model) = Q4_K_M.staged() else { return };
    let cases = goldens().cases;
    let mut runs = Vec::new();

    for settings in gated_settings() {
        let backend = load(&model, settings);
        let mut worst = Drift::default();
        let mut agree = 0;
        let mut rows = Vec::new();
        for case in &cases {
            let read = score(&backend, &case.request());
            let scores = choice_scores_of(&read);
            let drift = Drift::of(case, &read, &scores);
            worst = worst.max(drift);
            let best = scores.best();
            agree += usize::from(best == Some(case.best));
            rows.push(serde_json::json!({
                "id": case.id,
                "max_abs_logit": drift.logit,
                "max_abs_score": drift.score,
                "abs_label_mass": drift.label_mass,
                "best": best,
                "reference_best": case.best,
                "reference_margin": case.margin,
            }));
        }
        eprintln!(
            "label G3 Q4_K_M {settings:?}: max |Δlogit| {:.3e}, |Δscore| {:.3e}, \
             |Δlabel_mass| {:.3e}; argmax agrees {agree}/{}",
            worst.logit,
            worst.score,
            worst.label_mass,
            cases.len()
        );
        runs.push(serde_json::json!({
            "settings": {
                "gpu_layers": settings.gpu_layers,
                "flash_attn": settings.flash_attn,
                "threads": settings.threads,
                "n_ctx": N_CTX,
                "n_batch": N_BATCH,
            },
            "max_abs_logit": worst.logit,
            "max_abs_score": worst.score,
            "max_abs_label_mass": worst.label_mass,
            "argmax_agreement": agree,
            "cases": rows,
        }));
    }

    let report = serde_json::json!({
        "artifact": Q4_K_M.id,
        "reference": "integration-tests/fixtures/choice/goldens/labels.json",
        "note": "Q4_K_M drift from the float32 reference; reported, never a gate.",
        "runs": runs,
    });
    let path = g3_report_path();
    std::fs::create_dir_all(path.parent().expect("a reports directory"))
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&report).expect("serialize the report") + "\n",
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("label G3 report: {}", path.display());
}

//! Golden-token parity oracle (Phase 0.3 of the MLX perf plan).
//!
//! Guards decode numerics during performance iteration. Two tiers:
//!
//! - **Tier A (`self-greedy-*.json`, gating):** goldens captured from *this*
//!   implementation at a known-good commit, on this machine. Same stack,
//!   same kernels, same machine → greedy decode is deterministic, so exact
//!   token-id equality is required. Any flip means the change altered decode
//!   numerics and must be explained or reverted.
//! - **Tier B (`upstream-greedy-*.json`, informational):** captured from
//!   pinned Python `mlx-lm` on the same bundle. Cross-stack comparison is
//!   margin-aware (near-tie argmax flips across MLX versions / prefill
//!   topologies are expected); run explicitly via `--ignored`.
//!
//! Golden files live at `<bundle>/.golden/` next to the (gitignored) model
//! fixtures; they are local artifacts, never committed. Capture them with
//! `tools/scripts/capture-golden-tokens.sh`. All tests skip cleanly when
//! bundles or goldens are absent, so fixture-less CI is unaffected.

#![cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use serde::{Deserialize, Serialize};
use xybrid_core::runtime_adapter::llm::GenerationConfig;
use xybrid_core::runtime_adapter::mlx::{
    generate::{self, GenerateParams},
    MlxLlmAdapter, MlxLlmConfig,
};

// =============================================================================
// Oracle prompts (canonical — captured goldens record them verbatim)
// =============================================================================

/// Long realistic prompt (~250 tokens) matching the perf-bench protocol.
const BENCH_PROMPT: &str = "Imagine the compact benchmarking harness we are running right now. \
    Write a long, continuous, well-formed paragraph about caches, warm-ups, variance, and why \
    you should always discard the first run of a benchmark. Discuss how thermal conditions, \
    background processes, and memory pressure conspire to make single measurements unreliable, \
    and explain the discipline of taking medians across repeated rounds. Then describe how a \
    careful engineer reports results: with the hardware, the software versions, the exact \
    command line, and the variance band, so that a colleague can reproduce the numbers months \
    later on the same machine. Keep going until you have produced roughly one hundred and \
    twenty-eight output tokens of flowing prose.";

/// Short continuation-style prompt that pushes multi-byte / byte-fallback
/// output tails. Continuation form (not an instruction) because the oracle
/// bypasses chat templates: instruct models answer a bare instruction with
/// an immediate EOS, which would capture a worthless zero-length golden.
const BYTE_FALLBACK_PROMPT: &str = "A multilingual greeting card collection. Card one: \
    Hello! \u{1F60A} Card two: \u{4F60}\u{597D}\u{FF01}\u{1F389} Card three: Bonjour! \u{2728} Card four:";

const ORACLE_MAX_TOKENS: usize = 128;

fn oracle_config() -> GenerationConfig {
    GenerationConfig {
        max_tokens: ORACLE_MAX_TOKENS,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        seed: None,
        stop_sequences: Vec::new(),
    }
}

// =============================================================================
// Golden file model
// =============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct GoldenFile {
    tier: String,
    prompt: String,
    prompt_ids: Vec<i64>,
    generated_ids: Vec<i64>,
    eos_ids: Vec<i64>,
    terminated_by: String,
    params: GoldenParams,
    /// Tier B only: upstream top-2 logprob margin at every decode step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    margins_top2: Option<Vec<f64>>,
    provenance: Provenance,
}

#[derive(Debug, Serialize, Deserialize)]
struct GoldenParams {
    max_tokens: usize,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Provenance {
    commit: String,
    /// Version string of the MLX library linked into the capturing binary
    /// (Tier A) — verification hard-fails when the running build differs.
    #[serde(default)]
    mlx_runtime: Option<String>,
    /// Tier B: pip package versions used for the upstream capture.
    #[serde(default)]
    mlx_lm: Option<String>,
    #[serde(default)]
    mlx_pip: Option<String>,
    machine: String,
    macos: String,
    date: String,
    prefill_mode: String,
}

/// FNV-1a 64-bit, hex-encoded; first 8 chars tag the golden file name.
/// Mirrored by the Python upstream capture in
/// `tools/scripts/capture-golden-tokens.sh` — keep in sync.
fn prompt_tag(prompt: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in prompt.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")[..8].to_string()
}

fn golden_dir(bundle: &Path) -> PathBuf {
    bundle.join(".golden")
}

fn load_goldens(bundle: &Path, prefix: &str) -> Vec<(PathBuf, GoldenFile)> {
    let dir = golden_dir(bundle);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(prefix) || !name.ends_with(".json") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read golden {}: {e}", path.display()));
        let golden: GoldenFile = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("parse golden {}: {e}", path.display()));
        out.push((path, golden));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

// =============================================================================
// Bundle discovery (mirrors mlx_llm_chat.rs)
// =============================================================================

fn staged_bundles() -> Vec<(String, PathBuf)> {
    const BUNDLE_ENV_VARS: &[&str] = &[
        "XYBRID_MLX_LFM25_DIR",
        "XYBRID_MLX_QWEN_4B_DIR",
        "XYBRID_MLX_QWEN_DIR",
        "XYBRID_MLX_GEMMA_DIR",
    ];
    BUNDLE_ENV_VARS
        .iter()
        .filter_map(|var| std::env::var_os(var).map(|dir| ((*var).to_string(), PathBuf::from(dir))))
        .filter(|(_, dir)| dir.join("config.json").is_file())
        .collect()
}

fn mlx_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("MLX golden test lock poisoned")
}

// =============================================================================
// Generation helper: greedy decode collecting token ids
// =============================================================================

struct GreedyRun {
    /// Generated ids with a terminal EOS (if any) stripped — golden form.
    generated_ids: Vec<i64>,
    terminated_by: &'static str,
}

fn run_greedy_from_ids(adapter: &MlxLlmAdapter, prompt_ids: &[i64]) -> GreedyRun {
    let collected: Mutex<Vec<(i64, Option<String>)>> = Mutex::new(Vec::new());
    let config = oracle_config();
    let params = GenerateParams::new(&config);
    let out = generate::generate_tokens_from_ids(
        adapter,
        prompt_ids,
        params,
        Some(Box::new(|t| {
            collected.lock().expect("collector lock").push((
                t.token_id.expect("MLX path always sets token_id"),
                t.finish_reason,
            ));
            Ok(())
        })),
    )
    .expect("oracle greedy generation");

    let mut ids: Vec<(i64, Option<String>)> = collected.into_inner().expect("collector lock");
    assert_eq!(
        ids.len(),
        out.tokens_generated,
        "callback must see every generated token"
    );
    let terminated_by = if out.finish_reason == "stop" {
        let (_, last_finish) = ids.last().expect("stop implies at least one token");
        assert_eq!(
            last_finish.as_deref(),
            Some("stop"),
            "terminal token must carry the stop finish_reason"
        );
        ids.pop();
        "eos"
    } else {
        "length"
    };
    GreedyRun {
        generated_ids: ids.into_iter().map(|(id, _)| id).collect(),
        terminated_by,
    }
}

// =============================================================================
// Stream comparison (unit-testable; the canary below exercises it)
// =============================================================================

#[derive(Debug, PartialEq, Eq)]
enum StreamDiff {
    Equal,
    DivergesAt {
        index: usize,
        golden: i64,
        ours: i64,
    },
    LengthMismatch {
        golden_len: usize,
        ours_len: usize,
    },
}

fn compare_streams(golden: &[i64], ours: &[i64]) -> StreamDiff {
    for (index, (g, o)) in golden.iter().zip(ours.iter()).enumerate() {
        if g != o {
            return StreamDiff::DivergesAt {
                index,
                golden: *g,
                ours: *o,
            };
        }
    }
    if golden.len() != ours.len() {
        return StreamDiff::LengthMismatch {
            golden_len: golden.len(),
            ours_len: ours.len(),
        };
    }
    StreamDiff::Equal
}

fn decode_window(tokenizer: &tokenizers::Tokenizer, ids: &[i64], center: usize) -> String {
    let start = center.saturating_sub(20);
    let end = (center + 20).min(ids.len());
    let window: Vec<u32> = ids[start..end]
        .iter()
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();
    tokenizer
        .decode(&window, true)
        .unwrap_or_else(|e| format!("<decode failed: {e}>"))
}

fn sorted(ids: &[i64]) -> Vec<i64> {
    let mut v = ids.to_vec();
    v.sort_unstable();
    v
}

// =============================================================================
// Tests
// =============================================================================

/// Tier A preflight + exact parity: the gating oracle.
#[test]
fn golden_self_parity() {
    let _guard = mlx_test_lock();
    let bundles = staged_bundles();
    if bundles.is_empty() {
        eprintln!("skip: no MLX bundles staged");
        return;
    }
    let mut compared = 0usize;
    for (var, bundle) in &bundles {
        let goldens = load_goldens(bundle, "self-greedy-");
        if goldens.is_empty() {
            eprintln!("skip: {var} has no self goldens (run capture-golden-tokens.sh --self)");
            continue;
        }
        // The capture writes one golden per canonical prompt; a partial set
        // means coverage silently shrank (deleted file, aborted capture).
        assert!(
            goldens.len() >= 2,
            "{var}: found {} self golden(s), expected one per canonical prompt (2) — \
             re-run capture-golden-tokens.sh --self",
            goldens.len()
        );
        let adapter = MlxLlmAdapter::load(bundle, &MlxLlmConfig::default()).expect("load bundle");
        let tokenizer = adapter.tokenizer().expect("loaded tokenizer");
        let runtime_version = xybrid_mlx::version().expect("mlx runtime version");

        for (path, golden) in goldens {
            assert_eq!(golden.tier, "self", "{}: wrong tier", path.display());
            // Preflight 1: MLX runtime identity. Tier A is same-stack by
            // definition — a version mismatch means the golden predates an
            // MLX pin bump and must be re-captured, not compared.
            let captured = golden.provenance.mlx_runtime.as_deref().unwrap_or("");
            assert_eq!(
                captured,
                runtime_version,
                "{}: golden captured on MLX {captured} but this build links MLX \
                 {runtime_version} — re-capture goldens for the new pin",
                path.display()
            );
            // Preflight 2: EOS-set identity (config-driven resolution).
            let adapter_eos = adapter.eos_token_ids().expect("loaded eos set");
            assert_eq!(
                sorted(&golden.eos_ids),
                sorted(adapter_eos),
                "{}: EOS set changed since capture (golden {:?} vs adapter {:?})",
                path.display(),
                golden.eos_ids,
                adapter_eos
            );

            let run = run_greedy_from_ids(&adapter, &golden.prompt_ids);
            assert_eq!(
                run.terminated_by,
                golden.terminated_by,
                "{}: termination mode changed",
                path.display()
            );
            match compare_streams(&golden.generated_ids, &run.generated_ids) {
                StreamDiff::Equal => {
                    compared += 1;
                    eprintln!(
                        "golden ok: {} ({} ids, terminated_by={})",
                        path.display(),
                        run.generated_ids.len(),
                        run.terminated_by
                    );
                }
                StreamDiff::DivergesAt {
                    index,
                    golden: g,
                    ours: o,
                } => {
                    let hint = if index == 0 {
                        "index 0 → prefill numerics changed"
                    } else {
                        "mid-stream → decode-step numerics or accumulated drift"
                    };
                    panic!(
                        "{}: decode numerics changed — first divergence at index {index} \
                         (golden id {g}, ours {o}; {hint}).\n\
                         golden context: ...{}...\n\
                         ours   context: ...{}...\n\
                         golden provenance: {:?}\n\
                         If this change is an intentional numerics change, re-capture \
                         goldens in the same change and note it in the experiments CSV.",
                        path.display(),
                        decode_window(tokenizer, &golden.generated_ids, index),
                        decode_window(tokenizer, &run.generated_ids, index),
                        golden.provenance
                    );
                }
                StreamDiff::LengthMismatch {
                    golden_len,
                    ours_len,
                } => panic!(
                    "{}: stream length changed (golden {golden_len}, ours {ours_len}) — \
                     one side terminated early; check EOS behavior",
                    path.display()
                ),
            }
        }
    }
    eprintln!("golden_self_parity: {compared} golden(s) compared");
}

/// Tokenizer parity: our Rust-side encode of the golden's prompt string must
/// reproduce the prompt ids recorded at capture time. Decoupled from the
/// numeric oracle (which feeds ids directly).
#[test]
fn golden_tokenizer_parity() {
    let bundles = staged_bundles();
    if bundles.is_empty() {
        eprintln!("skip: no MLX bundles staged");
        return;
    }
    for (var, bundle) in &bundles {
        let goldens = load_goldens(bundle, "self-greedy-");
        if goldens.is_empty() {
            eprintln!("skip: {var} has no self goldens");
            continue;
        }
        let tokenizer = tokenizers::Tokenizer::from_file(bundle.join("tokenizer.json"))
            .expect("load bundle tokenizer");
        for (path, golden) in goldens {
            let encoding = tokenizer
                .encode(golden.prompt.as_str(), false)
                .expect("encode golden prompt");
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&u| i64::from(u)).collect();
            assert_eq!(
                ids,
                golden.prompt_ids,
                "{}: tokenizer encode diverged from capture-time prompt ids",
                path.display()
            );
        }
    }
}

/// Tier B: margin-aware audit of self goldens against pinned upstream
/// mlx-lm captures. Informational — run explicitly:
/// `cargo test --test mlx_golden_parity -- --ignored golden_upstream_audit`
#[test]
#[ignore = "informational cross-stack audit; requires upstream captures"]
fn golden_upstream_audit() {
    let bundles = staged_bundles();
    let margin_threshold: f64 = std::env::var("XYBRID_GOLDEN_MARGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05);
    let mut audited = 0usize;
    for (var, bundle) in &bundles {
        let upstream = load_goldens(bundle, "upstream-greedy-");
        if upstream.is_empty() {
            eprintln!("skip: {var} has no upstream goldens (run capture --upstream)");
            continue;
        }
        for (up_path, up) in upstream {
            let tag = prompt_tag(&up.prompt);
            let self_path = golden_dir(bundle).join(format!("self-greedy-{tag}.json"));
            let Ok(text) = std::fs::read_to_string(&self_path) else {
                eprintln!("skip: no self golden pairing {}", up_path.display());
                continue;
            };
            let own: GoldenFile = serde_json::from_str(&text).expect("parse self golden");
            assert_eq!(
                own.prompt_ids,
                up.prompt_ids,
                "{}: prompt ids differ between stacks — tokenizer divergence, \
                 not a numerics result",
                up_path.display()
            );
            // EOS-set parity between our config-driven resolution and
            // upstream's tokenizer-config-driven set (plan acceptance #5).
            assert_eq!(
                sorted(&own.eos_ids),
                sorted(&up.eos_ids),
                "{}: EOS sets differ between stacks (ours {:?}, upstream {:?})",
                up_path.display(),
                own.eos_ids,
                up.eos_ids
            );
            match compare_streams(&up.generated_ids, &own.generated_ids) {
                StreamDiff::Equal => {
                    audited += 1;
                    eprintln!("upstream parity exact: {}", up_path.display());
                }
                StreamDiff::DivergesAt {
                    index,
                    golden,
                    ours,
                } => {
                    let margin = up.margins_top2.as_ref().and_then(|m| m.get(index)).copied();
                    // Index-0 divergence means prefill numerics differ — the
                    // design hard-fails this regardless of margin (a near-tie
                    // at the first token still leaves every later position
                    // unvalidated, so the audit would certify nothing).
                    assert_ne!(
                        index,
                        0,
                        "{}: divergence at index 0 (prefill numerics; upstream id {golden}, \
                         ours {ours}, margin {margin:?}) — investigate before trusting Tier A",
                        up_path.display()
                    );
                    match margin {
                        Some(m) if m.abs() < margin_threshold => {
                            audited += 1;
                            eprintln!(
                                "upstream parity NEAR-TIE divergence at {index} \
                                 (margin {m:.4} < {margin_threshold}): {} — acceptable \
                                 cross-stack noise (upstream {golden}, ours {ours}); \
                                 streams legitimately differ after this point",
                                up_path.display()
                            );
                        }
                        Some(m) => panic!(
                            "{}: LARGE-MARGIN divergence from upstream at index {index} \
                             (margin {m:.4} ≥ {margin_threshold}, upstream id {golden}, \
                             ours {ours}) — real numerics gap, not a near-tie",
                            up_path.display()
                        ),
                        None => panic!(
                            "{}: divergence at index {index} but no margin recorded — \
                             re-capture upstream goldens with full margins",
                            up_path.display()
                        ),
                    }
                }
                StreamDiff::LengthMismatch {
                    golden_len,
                    ours_len,
                } => panic!(
                    "{}: length mismatch (upstream {golden_len}, ours {ours_len}) — \
                     EOS behavior differs between stacks",
                    up_path.display()
                ),
            }
        }
    }
    eprintln!("golden_upstream_audit: {audited} pair(s) audited");
}

/// Tier A capture helper, driven by `capture-golden-tokens.sh --self`.
/// Ignored by default so plain test runs never write goldens.
#[test]
#[ignore = "capture helper; run via tools/scripts/capture-golden-tokens.sh"]
fn capture_self_goldens() {
    let _guard = mlx_test_lock();
    let bundles = staged_bundles();
    assert!(!bundles.is_empty(), "capture requires staged bundles");
    let commit = std::env::var("XYBRID_GOLDEN_COMMIT")
        .expect("capture must be driven by the script (XYBRID_GOLDEN_COMMIT unset)");
    let provenance_base = Provenance {
        commit,
        mlx_runtime: Some(xybrid_mlx::version().expect("mlx runtime version")),
        mlx_lm: None,
        mlx_pip: None,
        machine: std::env::var("XYBRID_GOLDEN_MACHINE").unwrap_or_else(|_| "unknown".into()),
        macos: std::env::var("XYBRID_GOLDEN_MACOS").unwrap_or_else(|_| "unknown".into()),
        date: std::env::var("XYBRID_GOLDEN_DATE").unwrap_or_else(|_| "unknown".into()),
        prefill_mode: "full-prompt".into(),
    };

    for (var, bundle) in &bundles {
        let adapter = MlxLlmAdapter::load(bundle, &MlxLlmConfig::default()).expect("load bundle");
        let tokenizer = adapter.tokenizer().expect("loaded tokenizer");
        let eos_ids = adapter.eos_token_ids().expect("loaded eos set").to_vec();
        std::fs::create_dir_all(golden_dir(bundle)).expect("create .golden dir");

        for prompt in [BENCH_PROMPT, BYTE_FALLBACK_PROMPT] {
            let encoding = tokenizer.encode(prompt, false).expect("encode prompt");
            let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&u| i64::from(u)).collect();

            // Determinism check at capture time: two runs must agree before
            // a golden is worth writing.
            let first = run_greedy_from_ids(&adapter, &prompt_ids);
            let second = run_greedy_from_ids(&adapter, &prompt_ids);
            assert_eq!(
                first.generated_ids, second.generated_ids,
                "{var}: greedy decode not deterministic across back-to-back runs; \
                 refusing to capture a golden"
            );
            assert!(
                first.generated_ids.len() >= 16,
                "{var}: prompt {:?}... produced only {} token(s) — a near-empty \
                 golden guards nothing; choose a prompt the model continues",
                &prompt[..prompt.len().min(40)],
                first.generated_ids.len()
            );

            let golden = GoldenFile {
                tier: "self".into(),
                prompt: prompt.to_string(),
                prompt_ids,
                generated_ids: first.generated_ids,
                eos_ids: eos_ids.clone(),
                terminated_by: first.terminated_by.to_string(),
                params: GoldenParams {
                    max_tokens: ORACLE_MAX_TOKENS,
                    temperature: 0.0,
                },
                margins_top2: None,
                provenance: Provenance {
                    commit: provenance_base.commit.clone(),
                    mlx_runtime: provenance_base.mlx_runtime.clone(),
                    mlx_lm: None,
                    mlx_pip: None,
                    machine: provenance_base.machine.clone(),
                    macos: provenance_base.macos.clone(),
                    date: provenance_base.date.clone(),
                    prefill_mode: provenance_base.prefill_mode.clone(),
                },
            };
            let path = golden_dir(bundle).join(format!("self-greedy-{}.json", prompt_tag(prompt)));
            std::fs::write(&path, serde_json::to_string_pretty(&golden).unwrap())
                .expect("write golden");
            eprintln!(
                "captured {} ({} ids, terminated_by={})",
                path.display(),
                golden.generated_ids.len(),
                golden.terminated_by
            );
        }
    }
}

// =============================================================================
// Harness canary: the comparator itself must report divergence precisely.
// (Numerics *sensitivity* is not claimable by any cheap canary — this
// validates the comparison machinery deterministically.)
// =============================================================================

#[test]
fn comparator_canary_reports_exact_divergence_index() {
    let golden: Vec<i64> = (0..128).collect();
    let mut mutated = golden.clone();
    mutated[57] = 9999;
    assert_eq!(
        compare_streams(&golden, &mutated),
        StreamDiff::DivergesAt {
            index: 57,
            golden: 57,
            ours: 9999
        }
    );
    assert_eq!(compare_streams(&golden, &golden), StreamDiff::Equal);
    assert_eq!(
        compare_streams(&golden, &golden[..100]),
        StreamDiff::LengthMismatch {
            golden_len: 128,
            ours_len: 100
        }
    );
    // Prefix equality with different length must report the length, not a
    // bogus divergence index.
    let mut longer = golden.clone();
    longer.push(1);
    assert_eq!(
        compare_streams(&golden, &longer),
        StreamDiff::LengthMismatch {
            golden_len: 128,
            ours_len: 129
        }
    );
}

/// Empty prompt-id slices must be rejected as a validation error before
/// reaching MLX prefill (a `[1, 0]` tensor has undefined behavior there).
/// Fixture-gated: needs a loadable bundle to get past the NotLoaded checks.
#[test]
fn empty_prompt_ids_rejected_before_prefill() {
    let _guard = mlx_test_lock();
    let bundles = staged_bundles();
    let Some((_, bundle)) = bundles.first() else {
        eprintln!("skip: no MLX bundles staged");
        return;
    };
    let adapter = MlxLlmAdapter::load(bundle, &MlxLlmConfig::default()).expect("load bundle");
    let config = oracle_config();
    let err = generate::generate_tokens_from_ids(&adapter, &[], GenerateParams::new(&config), None)
        .expect_err("empty prompt ids must be rejected");
    assert!(
        err.to_string().contains("zero tokens"),
        "expected validation error, got: {err}"
    );
}

#[test]
fn prompt_tag_is_stable_fnv1a() {
    // Locked value: the Python capture in capture-golden-tokens.sh
    // implements the same FNV-1a 64 and must produce identical tags.
    assert_eq!(prompt_tag("hello"), "a430d846");
    assert_ne!(prompt_tag(BENCH_PROMPT), prompt_tag(BYTE_FALLBACK_PROMPT));
}

//! Run & score records + the on-disk run store.
//!
//! A run is `evalset × candidate → verdicts`: per-case output/score/latency plus
//! an aggregate [`Scores`] block that carries quality, the gate verdict, the
//! confidence interval, and the full **on-device SLO** field set. Those SLO
//! fields and the judge-identity fields are **frozen here even though their
//! capture is deferred** — they are part of the gate-contract surface and
//! expensive to retrofit.
//!
//! OTel-GenAI naming: the on-disk record keeps these field names (which already
//! track common GenAI/observability conventions — latency, token counts,
//! evaluation scores). The dotted `gen_ai.*` attribute mapping lives at the
//! telemetry/exporter boundary, not in this nested JSON record.
//!
//! Run storage is **dependency-injected** ([`EvalRunStore::with_dir`]) so tests
//! never touch the real `$HOME`; production resolves
//! `~/.xybrid/eval-runs/<run_id>/`.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::eval::format::EvalError;
use crate::eval::grader::{CaseGrade, Verdict};
use crate::eval::stats::{percentile, ConfidenceInterval, GatePolicy, GateVerdict, LatencyStats};

/// Scorer schema version recorded on every run (bumps invalidate comparisons).
pub const SCORER_VERSION: &str = "eval-scorer-v0";

/// Run-record filename inside a run directory.
pub const RUN_FILE: &str = "run.json";

/// Maximum `run.json` size accepted by [`EvalRunStore::load`] (DoS guard): a
/// FIFO/device file reports `len()==0` and would hang an unbounded read, and a
/// pathologically large file would exhaust memory — both are rejected up front.
const MAX_RUN_FILE_BYTES: u64 = 16 * 1024 * 1024;

fn default_true() -> bool {
    true
}

// ============================================================================
// Candidate & environment
// ============================================================================

fn default_temperature() -> f64 {
    0.0
}
fn default_candidate_seed() -> u64 {
    42
}

/// Generation config pinned for a run (determinism by default: greedy, seed 42).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateConfig {
    /// Sampling temperature (0.0 = greedy).
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Decode seed.
    #[serde(default = "default_candidate_seed")]
    pub seed: u64,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            seed: 42,
        }
    }
}

/// A thing being evaluated — a draft deployment to a named target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRef {
    /// Model id (optionally with variant/quant in the id).
    pub model_id: String,
    /// Resolved model content hash, for exact attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    /// Prompt version (server-side prompt library; absent in the local store).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_id: Option<String>,
    /// Pinned generation config.
    #[serde(default)]
    pub config: CandidateConfig,
}

impl CandidateRef {
    /// A candidate for `model_id` with default (deterministic) config.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            model_sha256: None,
            prompt_id: None,
            config: CandidateConfig::default(),
        }
    }
}

/// The environment a run executed in — pins reproducibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    /// Host triple-ish (e.g. `macos-arm64`).
    pub host: String,
    /// Inference backend (`llamacpp`, `ort`, `candle`, `cloud`, …).
    pub backend: String,
    /// Execution provider (`metal`, `coreml`, `cpu`, …).
    pub execution_provider: String,
    /// SDK version that produced the run.
    pub sdk_version: String,
}

// ============================================================================
// Judge identity (governance — frozen schema, capture deferred)
// ============================================================================

/// Judge identity recorded on a judge-backed run (trust layer → judge trust). A
/// judge-model/rubric change invalidates exactly the history it should.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeIdentity {
    /// Stable grader id.
    pub grader_id: String,
    /// Rubric version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric_version: Option<String>,
    /// Judge model id.
    pub judge_model: String,
    /// Hash of the exact judge prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_prompt_hash: Option<String>,
    /// Judge seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Judge temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

// ============================================================================
// Scores (aggregate)
// ============================================================================

/// Aggregate quality + on-device SLOs for a run. Every SLO field is `Option`
/// (populated-or-null) — the schema is frozen now; capture lands incrementally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    /// Normalized aggregate quality (0..1).
    pub quality: f64,
    /// Cases that passed.
    pub pass: usize,
    /// Cases that failed.
    pub fail: usize,
    /// Golden cases with no blessed reference (excluded from `quality`).
    #[serde(default)]
    pub unblessed: usize,
    /// The gate verdict for the run.
    pub verdict: GateVerdict,
    /// Confidence interval on `quality`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci: Option<ConfidenceInterval>,
    /// Whether the candidate was flagged flaky across repeats.
    #[serde(default)]
    pub flaky: bool,
    /// Mean quality for each repeat, when `gate.repeats` was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_qualities: Option<Vec<f64>>,

    // ---- on-device SLOs (Option = not captured) ----
    /// p50 wall-clock latency, ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_p50_ms: Option<f64>,
    /// p95 wall-clock latency, ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_p95_ms: Option<f64>,
    /// p95 time-to-first-token, ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_p95_ms: Option<f64>,
    /// p95 inter-token latency, ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_p95_ms: Option<f64>,
    /// Cold-start time, ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cold_start_ms: Option<f64>,
    /// Model load time, ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_load_ms: Option<f64>,
    /// Peak memory, MB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_memory_mb: Option<f64>,
    /// Bundle size, MB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_mb: Option<f64>,
    /// Energy delta, mWh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_delta_mwh: Option<f64>,
    /// Thermal state label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_state: Option<String>,
    /// Crash / timeout count.
    #[serde(default)]
    pub crash_or_timeout: u32,
    /// Whether the candidate ran fully offline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offline_ok: Option<bool>,
    /// Estimated cloud cost (when target = cloud).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cloud_cost: Option<f64>,

    // ---- governance ----
    /// Judge identity, when a judge backed this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeIdentity>,
    /// Scorer schema version.
    pub scorer_version: String,
}

impl Default for Scores {
    fn default() -> Self {
        Self {
            quality: 0.0,
            pass: 0,
            fail: 0,
            unblessed: 0,
            verdict: GateVerdict::Inconclusive,
            ci: None,
            flaky: false,
            repeat_qualities: None,
            latency_p50_ms: None,
            latency_p95_ms: None,
            ttft_p95_ms: None,
            itl_p95_ms: None,
            cold_start_ms: None,
            model_load_ms: None,
            peak_memory_mb: None,
            bundle_mb: None,
            energy_delta_mwh: None,
            thermal_state: None,
            crash_or_timeout: 0,
            offline_ok: None,
            estimated_cloud_cost: None,
            judge: None,
            scorer_version: SCORER_VERSION.to_string(),
        }
    }
}

// ============================================================================
// Per-case record + run
// ============================================================================

/// One per-case result in a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunCase {
    /// Case id.
    pub id: String,
    /// Captured output (privacy-gated in production; may be absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    /// Pass / fail / unblessed.
    pub verdict: Verdict,
    /// Normalized score (0..1).
    pub score: f64,
    /// Per-case latency, ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u32>,
    /// Grader detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Whether this case was eligible for gate scoring when the run was made.
    #[serde(default = "default_true")]
    pub counts_for_gate: bool,
}

/// A complete run record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// Unique run id.
    pub run_id: String,
    /// Evalset name.
    pub evalset: String,
    /// Evalset version the run scored against.
    pub evalset_version: u32,
    /// The evaluated candidate.
    pub candidate: CandidateRef,
    /// Execution environment.
    pub environment: Environment,
    /// Aggregate scores.
    pub scores: Scores,
    /// Per-case results.
    pub cases: Vec<RunCase>,
    /// ISO-8601 creation timestamp (injected — never stamped implicitly).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
}

// ============================================================================
// Aggregation
// ============================================================================

/// Build the aggregate [`Scores`] from per-case grades + latencies under a gate
/// policy.
///
/// `quality`, the CI, and the verdict are computed on the **unweighted** scorable
/// sample (Unblessed cases excluded) so the three numbers stay mutually
/// consistent. Per-case `weight` is honored by the separate
/// [`weighted_quality`] helper for Tier-3 weighted scoring.
pub fn aggregate_scores(
    grades: &[CaseGrade],
    latencies_ms: &[f64],
    policy: &GatePolicy,
    baseline_quality: Option<f64>,
) -> Scores {
    aggregate_scores_with_repeats(grades, latencies_ms, policy, baseline_quality, None)
}

pub(crate) fn aggregate_scores_with_repeats(
    grades: &[CaseGrade],
    latencies_ms: &[f64],
    policy: &GatePolicy,
    baseline_quality: Option<f64>,
    repeat_qualities: Option<Vec<f64>>,
) -> Scores {
    let mut pass = 0;
    let mut fail = 0;
    let mut unblessed = 0;
    let mut scorable = Vec::new();
    for g in grades {
        // Sanitize: a Tier-3 grader could emit a non-finite score; let it through
        // and `mean` yields NaN, serde writes `null`, and reload fails. Treat any
        // non-finite score as 0.0 so the run record stays finite and round-trips.
        let score = if g.score.is_finite() { g.score } else { 0.0 };
        match g.verdict {
            Verdict::Pass => {
                pass += 1;
                scorable.push(score);
            }
            Verdict::Fail => {
                fail += 1;
                scorable.push(score);
            }
            Verdict::Unblessed => unblessed += 1,
        }
    }

    let p95 = percentile(latencies_ms, 95.0);
    let latency = LatencyStats {
        p95_ms: p95,
        measured_cases: latencies_ms.len(),
        scorable_cases: scorable.len(),
    };
    let decision = policy.evaluate_with_latency_and_repeats(
        &scorable,
        latency,
        baseline_quality,
        repeat_qualities.as_deref(),
    );

    Scores {
        quality: decision.quality,
        pass,
        fail,
        unblessed,
        verdict: decision.verdict,
        ci: decision.ci,
        flaky: decision.flaky,
        repeat_qualities,
        latency_p50_ms: percentile(latencies_ms, 50.0),
        latency_p95_ms: p95,
        ..Scores::default()
    }
}

/// Weighted aggregate quality over `(score, weight)` pairs. Pairs with
/// non-positive total weight fall back to the unweighted mean. Caller excludes
/// Unblessed cases.
pub fn weighted_quality(scored: &[(f64, f64)]) -> f64 {
    let total_w: f64 = scored.iter().map(|(_, w)| w.max(0.0)).sum();
    if total_w <= 0.0 {
        if scored.is_empty() {
            return 0.0;
        }
        return scored.iter().map(|(s, _)| s).sum::<f64>() / scored.len() as f64;
    }
    scored.iter().map(|(s, w)| s * w.max(0.0)).sum::<f64>() / total_w
}

// ============================================================================
// Run store (injected base dir)
// ============================================================================

/// Validate that `run_id` is a single, safe path component (no separators, no
/// `..`, not absolute), so a user-supplied id (`eval show/diff/gate --run`)
/// can never traverse out of the store's base directory.
fn validate_run_id(run_id: &str) -> Result<&str, EvalError> {
    let mut comps = Path::new(run_id).components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(c)), None) if c == std::ffi::OsStr::new(run_id) => Ok(run_id),
        _ => Err(EvalError::Invalid(format!(
            "invalid run id {run_id:?}: must be a single path component"
        ))),
    }
}

/// On-disk store for run records. Construct with [`EvalRunStore::with_dir`] in
/// tests; [`EvalRunStore::default_location`] resolves `~/.xybrid/eval-runs`.
#[derive(Debug, Clone)]
pub struct EvalRunStore {
    base: PathBuf,
}

impl EvalRunStore {
    /// A store rooted at an explicit base directory.
    pub fn with_dir(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// The default store at `~/.xybrid/eval-runs`.
    pub fn default_location() -> Result<Self, EvalError> {
        let home = dirs::home_dir()
            .ok_or_else(|| EvalError::Io("could not resolve home directory".into()))?;
        Ok(Self::with_dir(home.join(".xybrid").join("eval-runs")))
    }

    /// The base directory.
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Persist a run as `<base>/<run_id>/run.json`; returns the run directory.
    pub fn save(&self, run: &Run) -> Result<PathBuf, EvalError> {
        let run_id = validate_run_id(&run.run_id)?;
        let dir = self.base.join(run_id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| EvalError::Io(format!("{}: {e}", dir.display())))?;
        let path = dir.join(RUN_FILE);
        let json = serde_json::to_string_pretty(run)
            .map_err(|e| EvalError::Io(format!("serialize run: {e}")))?;
        let tmp = dir.join(format!(".{RUN_FILE}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, json).map_err(|e| EvalError::Io(format!("{}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            EvalError::Io(format!("{}: {e}", path.display()))
        })?;
        Ok(dir)
    }

    /// Load a previously-saved run by id.
    pub fn load(&self, run_id: &str) -> Result<Run, EvalError> {
        let run_id = validate_run_id(run_id)?;
        let path = self.base.join(run_id).join(RUN_FILE);
        // DoS guard: require a regular file within the size cap before reading,
        // so a FIFO `run.json` can't hang `gate`/`show`/`diff` and an oversized
        // one can't exhaust memory.
        let meta = std::fs::metadata(&path)
            .map_err(|e| EvalError::Io(format!("{}: {e}", path.display())))?;
        if !meta.is_file() {
            return Err(EvalError::Invalid(format!(
                "run {run_id}: {} is not a regular file",
                path.display()
            )));
        }
        if meta.len() > MAX_RUN_FILE_BYTES {
            return Err(EvalError::Invalid(format!(
                "run {run_id}: run.json too large ({} bytes)",
                meta.len()
            )));
        }
        let src = std::fs::read_to_string(&path)
            .map_err(|e| EvalError::Io(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&src).map_err(|e| EvalError::Invalid(format!("run {run_id}: {e}")))
    }

    /// List saved run ids (directories containing a `run.json`).
    pub fn list(&self) -> Result<Vec<String>, EvalError> {
        if !self.base.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(&self.base)
            .map_err(|e| EvalError::Io(format!("{}: {e}", self.base.display())))?
        {
            let entry = entry.map_err(|e| EvalError::Io(e.to_string()))?;
            if entry.path().join(RUN_FILE).exists() {
                if let Some(name) = entry.file_name().to_str() {
                    ids.push(name.to_string());
                }
            }
        }
        ids.sort();
        Ok(ids)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::grader::CaseGrade;
    use tempfile::TempDir;

    fn env() -> Environment {
        Environment {
            host: "macos-arm64".into(),
            backend: "llamacpp".into(),
            execution_provider: "metal".into(),
            sdk_version: "0.1.2".into(),
        }
    }

    #[test]
    fn aggregate_counts_and_excludes_unblessed() {
        let grades = vec![
            CaseGrade::pass(),
            CaseGrade::pass(),
            CaseGrade::fail("nope"),
            CaseGrade::unblessed("golden"),
        ];
        let policy = GatePolicy {
            min_cases: 1,
            bootstrap_iterations: 200,
            ..GatePolicy::default()
        };
        let scores = aggregate_scores(&grades, &[], &policy, None);
        assert_eq!(scores.pass, 2);
        assert_eq!(scores.fail, 1);
        assert_eq!(scores.unblessed, 1);
        // quality over the 3 scorable cases = 2/3.
        assert!((scores.quality - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(scores.scorer_version, SCORER_VERSION);
    }

    #[test]
    fn aggregate_populates_latency_percentiles() {
        let grades = vec![CaseGrade::pass(); 20];
        let latencies: Vec<f64> = (1..=20).map(|x| x as f64 * 10.0).collect();
        let policy = GatePolicy {
            min_cases: 1,
            bootstrap_iterations: 100,
            ..GatePolicy::default()
        };
        let scores = aggregate_scores(&grades, &latencies, &policy, None);
        assert_eq!(scores.latency_p50_ms, Some(100.0)); // 10th value
        assert_eq!(scores.latency_p95_ms, Some(190.0)); // 19th value
    }

    #[test]
    fn aggregate_marks_latency_slo_partial_coverage_inconclusive() {
        let grades = vec![CaseGrade::pass(); 3];
        let policy = GatePolicy {
            min_cases: 1,
            min_quality: Some(0.5),
            max_p95_latency_ms: Some(800.0),
            bootstrap_iterations: 100,
            ..GatePolicy::default()
        };
        let scores = aggregate_scores(&grades, &[100.0], &policy, None);
        assert_eq!(scores.verdict, GateVerdict::Inconclusive);
        assert_eq!(scores.latency_p95_ms, Some(100.0));
    }

    #[test]
    fn aggregate_sanitizes_non_finite_score() {
        // C6: a grader emits NaN/Inf on otherwise-passing cases. The aggregate
        // must stay finite (NaN→0.0) and round-trip through serde.
        let mut nan_grade = CaseGrade::pass();
        nan_grade.score = f64::NAN;
        let mut inf_grade = CaseGrade::pass();
        inf_grade.score = f64::INFINITY;
        let grades = vec![CaseGrade::pass(), nan_grade, inf_grade];
        let policy = GatePolicy {
            min_cases: 1,
            bootstrap_iterations: 100,
            ..GatePolicy::default()
        };
        let scores = aggregate_scores(&grades, &[], &policy, None);
        assert!(scores.quality.is_finite(), "quality must stay finite");
        // two sanitized-to-0.0 + one real 1.0 over 3 scorable = 1/3.
        assert!((scores.quality - 1.0 / 3.0).abs() < 1e-9);
        // serde must round-trip (a NaN quality would serialize as null).
        let json = serde_json::to_string(&scores).unwrap();
        let back: Scores = serde_json::from_str(&json).unwrap();
        assert_eq!(scores, back);
    }

    #[test]
    fn aggregate_wires_gate_verdict() {
        let grades = vec![CaseGrade::pass(); 50];
        let policy = GatePolicy {
            min_cases: 10,
            min_quality: Some(0.9),
            bootstrap_iterations: 200,
            ..GatePolicy::default()
        };
        let scores = aggregate_scores(&grades, &[], &policy, None);
        assert_eq!(scores.verdict, GateVerdict::Pass);
        assert!(scores.ci.is_some());
    }

    #[test]
    fn weighted_quality_honors_weight() {
        // a perfect-but-light case and a failing-but-heavy case.
        let scored = vec![(1.0, 1.0), (0.0, 3.0)];
        // weighted = (1*1 + 0*3) / 4 = 0.25
        assert!((weighted_quality(&scored) - 0.25).abs() < 1e-9);
        // zero total weight → unweighted mean.
        assert!((weighted_quality(&[(1.0, 0.0), (0.0, 0.0)]) - 0.5).abs() < 1e-9);
        assert_eq!(weighted_quality(&[]), 0.0);
    }

    #[test]
    fn scores_round_trip_with_all_slo_fields() {
        let mut scores = Scores {
            quality: 0.84,
            pass: 42,
            fail: 8,
            verdict: GateVerdict::Fail,
            latency_p50_ms: Some(210.0),
            latency_p95_ms: Some(640.0),
            ttft_p95_ms: Some(95.0),
            itl_p95_ms: Some(28.0),
            cold_start_ms: Some(1840.0),
            model_load_ms: Some(1120.0),
            peak_memory_mb: Some(890.0),
            bundle_mb: Some(612.0),
            energy_delta_mwh: Some(41.2),
            thermal_state: Some("nominal".into()),
            crash_or_timeout: 0,
            offline_ok: Some(true),
            ..Scores::default()
        };
        scores.judge = Some(JudgeIdentity {
            grader_id: "g1".into(),
            rubric_version: Some("v3".into()),
            judge_model: "overlap-judge-v0".into(),
            judge_prompt_hash: Some("abc123".into()),
            seed: Some(42),
            temperature: Some(0.0),
        });
        let json = serde_json::to_string(&scores).unwrap();
        let back: Scores = serde_json::from_str(&json).unwrap();
        assert_eq!(scores, back);
        assert_eq!(back.thermal_state.as_deref(), Some("nominal"));
        assert_eq!(back.judge.unwrap().judge_model, "overlap-judge-v0");
    }

    #[test]
    fn run_round_trips() {
        let run = Run {
            run_id: "run_test_1".into(),
            evalset: "intent-classifier".into(),
            evalset_version: 3,
            candidate: CandidateRef::new("qwen3.5-0.8b"),
            environment: env(),
            scores: Scores::default(),
            cases: vec![RunCase {
                id: "c1".into(),
                output: Some(serde_json::json!({"label": "refund"})),
                verdict: Verdict::Pass,
                score: 1.0,
                latency_ms: Some(188),
                detail: None,
                counts_for_gate: true,
            }],
            created: Some("2026-06-14".into()),
        };
        let json = serde_json::to_string(&run).unwrap();
        let back: Run = serde_json::from_str(&json).unwrap();
        assert_eq!(run, back);
    }

    #[test]
    fn run_case_counts_for_gate_defaults_true_for_old_records() {
        let json = r#"{"id":"c1","verdict":"pass","score":1.0}"#;
        let case: RunCase = serde_json::from_str(json).unwrap();
        assert!(case.counts_for_gate);
    }

    // ---- store isolation ----

    #[test]
    fn store_round_trips_in_temp_dir() {
        let dir = TempDir::new().unwrap();
        let store = EvalRunStore::with_dir(dir.path());
        assert!(store.list().unwrap().is_empty());

        let run = Run {
            run_id: "run_abc".into(),
            evalset: "s".into(),
            evalset_version: 1,
            candidate: CandidateRef::new("m"),
            environment: env(),
            scores: Scores::default(),
            cases: vec![],
            created: None,
        };
        let run_dir = store.save(&run).unwrap();
        assert!(run_dir.starts_with(dir.path())); // never escapes the injected base
        assert_eq!(store.list().unwrap(), vec!["run_abc".to_string()]);

        let loaded = store.load("run_abc").unwrap();
        assert_eq!(loaded, run);
        assert!(std::fs::read_dir(&run_dir).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")));
    }

    #[test]
    fn store_default_ctor_never_touches_home_in_test() {
        // The store under test writes only under the injected base — proving a
        // run save can't leak into the real `$HOME/.xybrid`.
        let dir = TempDir::new().unwrap();
        let store = EvalRunStore::with_dir(dir.path());
        let run = Run {
            run_id: "x".into(),
            evalset: "s".into(),
            evalset_version: 1,
            candidate: CandidateRef::new("m"),
            environment: env(),
            scores: Scores::default(),
            cases: vec![],
            created: None,
        };
        store.save(&run).unwrap();
        assert!(dir.path().join("x").join(RUN_FILE).exists());
    }

    #[test]
    fn list_empty_when_base_absent() {
        let store = EvalRunStore::with_dir("/nonexistent/xybrid/eval/base");
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn store_rejects_traversal_run_ids() {
        let dir = TempDir::new().unwrap();
        let store = EvalRunStore::with_dir(dir.path());
        // load with a malicious id must error, never read outside the base.
        for bad in [
            "../escape",
            "../../etc/passwd",
            "/etc/passwd",
            "a/b",
            "..",
            ".",
            "",
        ] {
            assert!(store.load(bad).is_err(), "load should reject {bad:?}");
        }
        // save with a traversal id must error and write nothing outside the base.
        let mut run = Run {
            run_id: "../evil".into(),
            evalset: "s".into(),
            evalset_version: 1,
            candidate: CandidateRef::new("m"),
            environment: env(),
            scores: Scores::default(),
            cases: vec![],
            created: None,
        };
        assert!(store.save(&run).is_err());
        assert!(!dir.path().parent().unwrap().join("evil").exists());
        // a normal id still works.
        run.run_id = "run_ok".into();
        assert!(store.save(&run).is_ok());
        assert_eq!(store.load("run_ok").unwrap().run_id, "run_ok");
    }

    #[test]
    fn load_rejects_non_file_run_json() {
        // S2: a `run.json` that is a directory (stand-in for a FIFO/device file —
        // both fail the is_file check) must be rejected, never read.
        let dir = TempDir::new().unwrap();
        let store = EvalRunStore::with_dir(dir.path());
        // Create `<base>/weird/run.json` AS A DIRECTORY.
        std::fs::create_dir_all(dir.path().join("weird").join(RUN_FILE)).unwrap();
        let err = store.load("weird").unwrap_err();
        assert!(
            matches!(err, EvalError::Invalid(_)),
            "expected Invalid, got {err:?}"
        );
    }

    #[test]
    fn load_rejects_oversized_run_json() {
        // S2: a `run.json` over the size cap is rejected before reading.
        let dir = TempDir::new().unwrap();
        let store = EvalRunStore::with_dir(dir.path());
        let run_dir = dir.path().join("big");
        std::fs::create_dir_all(&run_dir).unwrap();
        let path = run_dir.join(RUN_FILE);
        // Allocate a sparse file just past the 16 MiB cap without writing 16 MiB.
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(MAX_RUN_FILE_BYTES + 1).unwrap();
        let err = store.load("big").unwrap_err();
        assert!(
            matches!(err, EvalError::Invalid(_)),
            "expected Invalid, got {err:?}"
        );
    }
}

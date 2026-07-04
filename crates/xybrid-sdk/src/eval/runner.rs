//! Evalset runner — orchestrates `evalset × candidate → Run`.
//!
//! The model-execution step is an **injected closure** (`infer`), so the
//! orchestration — iterate cases, grade, count crashes, aggregate, attach judge
//! identity — is fully unit-testable without loading a model. The CLI supplies a
//! real `infer` that runs the candidate through the SDK; tests supply a
//! deterministic mock.

use crate::eval::format::{Case, Evalset, Gate, LoadedEvalset};
use crate::eval::grader::{grade_case, CaseGrade, GradeOutput, Judge, Verdict};
use crate::eval::run::{
    aggregate_scores_with_repeats, CandidateRef, Environment, JudgeIdentity, Run, RunCase, Scores,
};
use crate::eval::stats::{
    GatePolicy, DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_CONFIDENCE, DEFAULT_SEED,
};

const MAX_CASE_DETAIL_CHARS: usize = 200;
const TRUNCATED_SUFFIX: &str = "…(truncated)";

/// What running a candidate on one case produced.
#[derive(Debug, Clone)]
pub struct CaseOutcome {
    /// The model output to grade.
    pub output: GradeOutput,
    /// Per-case latency, ms.
    pub latency_ms: Option<u32>,
}

impl CaseOutcome {
    /// Convenience: a text outcome with a latency.
    pub fn text(s: impl Into<String>, latency_ms: u32) -> Self {
        Self {
            output: GradeOutput::Text(s.into()),
            latency_ms: Some(latency_ms),
        }
    }
}

/// Options controlling a run.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Whether to capture per-case outputs into the run record. Off ⇒ the run
    /// stores verdicts + scores only (privacy-conservative default).
    pub capture_outputs: bool,
    /// Baseline quality for non-inferiority (compare mode); `None` for absolute.
    pub baseline_quality: Option<f64>,
    /// Today's date (`YYYY-MM-DD`) for expiry exclusion. `None` disables expiry
    /// filtering (quarantined cases are always excluded regardless).
    pub today: Option<String>,
}

/// Today's date as `YYYY-MM-DD` (UTC) — for the CLI to pass into [`RunOptions`]
/// so expired cases are excluded from a run.
pub fn today_utc() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

/// Translate the manifest's optional [`Gate`] into a [`GatePolicy`], filling
/// engine defaults for anything the manifest leaves unset.
pub fn gate_policy(manifest: &Evalset) -> GatePolicy {
    let gate: Option<&Gate> = manifest.gate.as_ref();
    let non_inferiority_margin = gate.and_then(|g| g.non_inferiority_margin).unwrap_or(0.0);
    if non_inferiority_margin < 0.0 {
        eprintln!(
            "warning: clamping negative non-inferiority margin {non_inferiority_margin} to 0.0"
        );
    }
    GatePolicy {
        min_cases: gate.and_then(|g| g.min_cases).unwrap_or(1),
        min_quality: gate.and_then(|g| g.min_quality),
        max_p95_latency_ms: gate.and_then(|g| g.max_p95_latency_ms),
        non_inferiority_margin: non_inferiority_margin.max(0.0),
        flaky_std_threshold: 0.1,
        seed: DEFAULT_SEED,
        bootstrap_iterations: DEFAULT_BOOTSTRAP_ITERATIONS,
        confidence: DEFAULT_CONFIDENCE,
    }
}

/// Run an evalset against a candidate, scoring each case via `infer`.
///
/// Inference failures are counted as `crash_or_timeout` and fail their case
/// (never abort the whole run). The judge identity is recorded when a judge is
/// supplied. `run_id`/`created` are caller-supplied (never stamped implicitly,
/// for determinism).
// CLIPPY-ALLOW: run_evalset is the public orchestration seam for injected inference.
#[allow(clippy::too_many_arguments)]
pub fn run_evalset<F>(
    set: &LoadedEvalset,
    candidate: CandidateRef,
    environment: Environment,
    policy: &GatePolicy,
    judge: Option<&dyn Judge>,
    options: &RunOptions,
    run_id: impl Into<String>,
    mut infer: F,
) -> Run
where
    F: FnMut(&Case) -> Result<CaseOutcome, String>,
{
    let repeats = set
        .manifest
        .gate
        .as_ref()
        .and_then(|gate| gate.repeats)
        .unwrap_or(1)
        .max(1);
    let mut grades: Vec<CaseGrade> = Vec::with_capacity(set.cases.len() * repeats as usize);
    let mut latencies: Vec<f64> = Vec::new();
    let mut run_cases: Vec<RunCase> = Vec::with_capacity(set.cases.len());
    let mut crashes: u32 = 0;
    let mut repeat_qualities = Vec::with_capacity(repeats as usize);

    for repeat_idx in 0..repeats {
        let mut repeat_scores = Vec::new();
        for case in &set.cases {
            let counts_for_gate = counts_for_gate(case, options.today.as_deref());
            match infer(case) {
                Ok(outcome) => {
                    let mut grade = grade_case(&set.manifest, case, &outcome.output, judge);
                    grade.score = finite_score(grade.score);
                    grade.detail = grade.detail.map(cap_case_detail);
                    if counts_for_gate {
                        if let Some(ms) = outcome.latency_ms {
                            latencies.push(ms as f64);
                        }
                        if grade.verdict != Verdict::Unblessed {
                            repeat_scores.push(grade.score);
                        }
                        grades.push(grade.clone());
                    }
                    if repeat_idx == 0 {
                        run_cases.push(RunCase {
                            id: case.id.clone(),
                            output: if options.capture_outputs {
                                Some(output_to_json(&outcome.output))
                            } else {
                                None
                            },
                            verdict: grade.verdict,
                            score: grade.score,
                            latency_ms: outcome.latency_ms,
                            detail: grade.detail.clone(),
                            counts_for_gate,
                        });
                    }
                }
                Err(err) => {
                    let grade = CaseGrade::fail(format!("inference error: {err}"));
                    let detail = grade.detail.clone().map(cap_case_detail);
                    if counts_for_gate {
                        crashes += 1;
                        repeat_scores.push(0.0);
                        grades.push(CaseGrade {
                            detail: detail.clone(),
                            ..grade
                        });
                    }
                    if repeat_idx == 0 {
                        run_cases.push(RunCase {
                            id: case.id.clone(),
                            output: None,
                            verdict: Verdict::Fail,
                            score: 0.0,
                            latency_ms: None,
                            detail,
                            counts_for_gate,
                        });
                    }
                }
            }
        }
        repeat_qualities.push(mean_or_zero(&repeat_scores));
    }

    let repeat_qualities = (repeats > 1).then_some(repeat_qualities);
    let mut scores: Scores = aggregate_scores_with_repeats(
        &grades,
        &latencies,
        policy,
        options.baseline_quality,
        repeat_qualities,
    );
    scores.crash_or_timeout = crashes;
    if let Some(j) = judge {
        scores.judge = Some(JudgeIdentity {
            grader_id: "task-default".to_string(),
            rubric_version: None,
            judge_model: j.judge_model().to_string(),
            judge_prompt_hash: None,
            seed: None,
            temperature: None,
        });
    }

    Run {
        run_id: run_id.into(),
        evalset: set.manifest.name.clone(),
        evalset_version: set.manifest.version,
        candidate,
        environment,
        scores,
        cases: run_cases,
        created: None,
    }
}

fn counts_for_gate(case: &Case, today: Option<&str>) -> bool {
    !case.is_quarantined()
        && case.split == crate::eval::format::Split::Regression
        && today.is_none_or(|today| !case.is_expired_on(today))
}

fn mean_or_zero(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// Serialize a grade output for storage in a run record.
fn output_to_json(output: &GradeOutput) -> serde_json::Value {
    match output {
        GradeOutput::Text(t) => serde_json::Value::String(t.clone()),
        GradeOutput::Json(v) => v.clone(),
        GradeOutput::Embedding(v) => serde_json::json!(v),
    }
}

fn finite_score(score: f64) -> f64 {
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

fn cap_case_detail(detail: String) -> String {
    if detail.chars().count() <= MAX_CASE_DETAIL_CHARS {
        return detail;
    }
    let head_len = MAX_CASE_DETAIL_CHARS.saturating_sub(TRUNCATED_SUFFIX.chars().count());
    let head: String = detail.chars().take(head_len).collect();
    format!("{head}{TRUNCATED_SUFFIX}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::format::{Case, CaseInput, Evalset, Expected, LoadedEvalset, Split, TaskType};
    use crate::eval::grader::OverlapJudge;
    use crate::eval::stats::GateVerdict;

    fn env() -> Environment {
        Environment {
            host: "macos-arm64".into(),
            backend: "mock".into(),
            execution_provider: "cpu".into(),
            sdk_version: "0.1.2".into(),
        }
    }

    fn classify_set() -> LoadedEvalset {
        let mut manifest = Evalset::new("intent", TaskType::Classify);
        manifest.labels = vec!["refund".into(), "cancel".into()];
        let cases = vec![
            regression_case("c1", "refund please", "refund"),
            regression_case("c2", "cancel my order", "cancel"),
        ];
        LoadedEvalset {
            manifest,
            cases,
            root: std::path::PathBuf::from("."),
        }
    }

    fn regression_case(id: &str, input: &str, label: &str) -> Case {
        let mut case = Case::new(id, CaseInput::Text(input.into()))
            .with_expected(Expected::Label(label.into()));
        case.split = Split::Regression;
        case
    }

    #[test]
    fn perfect_oracle_scores_all_pass() {
        let set = classify_set();
        let policy = gate_policy(&set.manifest);
        // Oracle: echo the expected label per input keyword.
        let run = run_evalset(
            &set,
            CandidateRef::new("oracle"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_1",
            |case| match &case.input {
                CaseInput::Text(t) if t.contains("refund") => Ok(CaseOutcome::text("refund", 10)),
                CaseInput::Text(_) => Ok(CaseOutcome::text("cancel", 12)),
                _ => Err("unsupported".into()),
            },
        );
        assert_eq!(run.scores.pass, 2);
        assert_eq!(run.scores.fail, 0);
        assert_eq!(run.scores.quality, 1.0);
        assert_eq!(run.cases.len(), 2);
        assert_eq!(run.scores.crash_or_timeout, 0);
        assert!(run.cases[0].output.is_none());
    }

    #[test]
    fn wrong_oracle_fails_cases() {
        let set = classify_set();
        let policy = gate_policy(&set.manifest);
        let run = run_evalset(
            &set,
            CandidateRef::new("bad"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_2",
            |_| Ok(CaseOutcome::text("refund", 10)), // always refund → c2 fails
        );
        assert_eq!(run.scores.pass, 1);
        assert_eq!(run.scores.fail, 1);
        assert!((run.scores.quality - 0.5).abs() < 1e-9);
    }

    #[test]
    fn inference_error_counts_as_crash_and_fails_case() {
        let set = classify_set();
        let policy = gate_policy(&set.manifest);
        let run = run_evalset(
            &set,
            CandidateRef::new("flaky"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_3",
            |case| match &case.input {
                CaseInput::Text(t) if t.contains("refund") => Ok(CaseOutcome::text("refund", 10)),
                _ => Err("backend timeout".into()),
            },
        );
        assert_eq!(run.scores.pass, 1);
        assert_eq!(run.scores.fail, 1);
        assert_eq!(run.scores.crash_or_timeout, 1);
        // The errored case has no output and a detail explaining the failure.
        let errored = run.cases.iter().find(|c| c.id == "c2").unwrap();
        assert!(errored.output.is_none());
        assert!(errored.detail.as_ref().unwrap().contains("inference error"));
    }

    #[test]
    fn capture_outputs_true_records_payloads() {
        let set = classify_set();
        let policy = gate_policy(&set.manifest);
        let opts = RunOptions {
            capture_outputs: true,
            ..RunOptions::default()
        };
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            None,
            &opts,
            "run_4",
            |_| Ok(CaseOutcome::text("refund", 10)),
        );
        assert!(run.cases.iter().all(|c| c.output.is_some()));
    }

    #[test]
    fn judge_identity_recorded_when_judge_present() {
        let mut manifest = Evalset::new("chat", TaskType::Chat);
        manifest.gate = None;
        let cases = vec![Case::new("c1", CaseInput::Text("hi".into()))
            .with_expected(Expected::Text("hello there".into()))];
        let set = LoadedEvalset {
            manifest,
            cases,
            root: ".".into(),
        };
        let judge = OverlapJudge::default();
        let policy = gate_policy(&set.manifest);
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            Some(&judge),
            &RunOptions::default(),
            "run_5",
            |_| Ok(CaseOutcome::text("hello there", 5)),
        );
        let judge_id = run.scores.judge.expect("judge identity recorded");
        assert_eq!(judge_id.judge_model, "overlap-judge-v0");
    }

    struct NanJudge;

    impl Judge for NanJudge {
        fn grade(
            &self,
            _input: &CaseInput,
            _reference: Option<&Expected>,
            _output: &GradeOutput,
        ) -> CaseGrade {
            CaseGrade {
                score: f64::NAN,
                verdict: Verdict::Pass,
                detail: None,
            }
        }

        fn judge_model(&self) -> &str {
            "nan-judge"
        }
    }

    #[test]
    fn run_case_sanitizes_non_finite_grade_score() {
        let mut manifest = Evalset::new("chat", TaskType::Chat);
        manifest.gate = Some(Gate {
            min_quality: Some(0.5),
            min_cases: Some(1),
            ..Gate::default()
        });
        let cases = vec![Case::new("c1", CaseInput::Text("hi".into()))
            .with_expected(Expected::Text("hello".into()))];
        let set = LoadedEvalset {
            manifest,
            cases,
            root: ".".into(),
        };
        let policy = gate_policy(&set.manifest);
        let judge = NanJudge;
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            Some(&judge),
            &RunOptions::default(),
            "run_nan",
            |_| Ok(CaseOutcome::text("hello", 5)),
        );
        assert_eq!(run.cases[0].score, 0.0);
        assert!(run.scores.quality.is_finite());
    }

    #[test]
    fn inference_error_detail_is_capped() {
        let set = classify_set();
        let policy = gate_policy(&set.manifest);
        let run = run_evalset(
            &set,
            CandidateRef::new("flaky"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_detail",
            |_| Err("é".repeat(400)),
        );
        let detail = run.cases[0].detail.as_ref().unwrap();
        assert_eq!(detail.chars().count(), MAX_CASE_DETAIL_CHARS);
        assert!(detail.ends_with(TRUNCATED_SUFFIX));
    }

    #[test]
    fn gate_policy_maps_manifest_thresholds() {
        let mut manifest = Evalset::new("s", TaskType::Classify);
        manifest.gate = Some(Gate {
            min_quality: Some(0.92),
            max_p95_latency_ms: Some(800.0),
            min_cases: Some(30),
            non_inferiority_margin: Some(0.02),
            repeats: Some(3),
        });
        let policy = gate_policy(&manifest);
        assert_eq!(policy.min_quality, Some(0.92));
        assert_eq!(policy.max_p95_latency_ms, Some(800.0));
        assert_eq!(policy.min_cases, 30);
        assert!((policy.non_inferiority_margin - 0.02).abs() < 1e-9);
    }

    #[test]
    fn gate_policy_clamps_negative_non_inferiority_margin() {
        // C10: a negative margin from the manifest would invert tie logic; it
        // must be clamped to 0.0.
        let mut manifest = Evalset::new("s", TaskType::Classify);
        manifest.gate = Some(Gate {
            min_quality: None,
            max_p95_latency_ms: None,
            min_cases: None,
            non_inferiority_margin: Some(-0.05),
            repeats: None,
        });
        let policy = gate_policy(&manifest);
        assert_eq!(policy.non_inferiority_margin, 0.0);
    }

    #[test]
    fn run_verdict_reflects_gate() {
        let mut set = classify_set();
        set.manifest.gate = Some(Gate {
            min_quality: Some(0.9),
            max_p95_latency_ms: None,
            min_cases: Some(1),
            non_inferiority_margin: None,
            repeats: None,
        });
        let policy = gate_policy(&set.manifest);
        // Both pass → quality 1.0 ≥ 0.9 → Pass.
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_6",
            |case| match &case.input {
                CaseInput::Text(t) if t.contains("refund") => Ok(CaseOutcome::text("refund", 10)),
                _ => Ok(CaseOutcome::text("cancel", 10)),
            },
        );
        assert_eq!(run.scores.verdict, GateVerdict::Pass);
    }

    #[test]
    fn dev_quarantined_and_expired_cases_run_but_do_not_feed_gate() {
        let mut manifest = Evalset::new("s", TaskType::Classify);
        manifest.labels = vec!["a".into()];
        let mut good = Case::new("good", CaseInput::Text("x".into()))
            .with_expected(Expected::Label("a".into()));
        good.split = Split::Regression;
        let dev = Case::new("dev", CaseInput::Text("x".into()))
            .with_expected(Expected::Label("a".into()));
        let mut quarantined = Case::new("bad", CaseInput::Text("x".into()))
            .with_expected(Expected::Label("a".into()));
        quarantined.split = Split::Regression;
        quarantined.quarantine_reason = Some("mislabeled".into());
        let mut expired = Case::new("old", CaseInput::Text("x".into()))
            .with_expected(Expected::Label("a".into()));
        expired.split = Split::Regression;
        expired.expires_at = Some("2020-01-01".into());
        let set = LoadedEvalset {
            manifest,
            cases: vec![good, dev, quarantined, expired],
            root: ".".into(),
        };
        let policy = gate_policy(&set.manifest);
        let opts = RunOptions {
            today: Some("2026-06-14".into()),
            ..RunOptions::default()
        };
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            None,
            &opts,
            "run_gov",
            |_| Ok(CaseOutcome::text("a", 5)),
        );
        assert_eq!(run.cases.len(), 4);
        assert_eq!(
            run.cases
                .iter()
                .filter(|case| case.counts_for_gate)
                .map(|case| case.id.as_str())
                .collect::<Vec<_>>(),
            vec!["good"]
        );
        assert_eq!(run.scores.pass, 1);
    }

    #[test]
    fn repeated_gate_marks_flaky_candidate_inconclusive() {
        let mut set = classify_set();
        set.manifest.gate = Some(Gate {
            min_quality: Some(0.0),
            min_cases: Some(1),
            repeats: Some(2),
            ..Gate::default()
        });
        let policy = gate_policy(&set.manifest);
        let mut calls = 0;
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_repeats",
            |case| {
                calls += 1;
                let first_repeat = calls <= set.cases.len();
                match (&case.input, first_repeat) {
                    (CaseInput::Text(t), true) if t.contains("refund") => {
                        Ok(CaseOutcome::text("refund", 10))
                    }
                    (CaseInput::Text(_), true) => Ok(CaseOutcome::text("cancel", 20)),
                    _ => Ok(CaseOutcome::text("wrong", 30)),
                }
            },
        );
        assert_eq!(calls, set.cases.len() * 2);
        assert_eq!(run.cases.len(), 2);
        assert_eq!(run.scores.repeat_qualities, Some(vec![1.0, 0.0]));
        assert!(run.scores.flaky);
        assert_eq!(run.scores.verdict, GateVerdict::Inconclusive);
        assert_eq!(run.scores.ci.as_ref().unwrap().repeats, 2);
        assert_eq!(run.scores.latency_p95_ms, Some(30.0));
    }

    #[test]
    fn latency_gate_with_all_crashes_is_inconclusive_not_pass() {
        // End-to-end coverage for the C5 fix: a manifest with a latency SLO where
        // every case crashes (no latencies recorded) must resolve to
        // Inconclusive — an unmeasured hard SLO can never silently pass.
        let mut manifest = Evalset::new("s", TaskType::Classify);
        manifest.labels = vec!["a".into()];
        manifest.gate = Some(Gate {
            max_p95_latency_ms: Some(800.0),
            min_cases: Some(1),
            ..Gate::default()
        });
        let cases = vec![
            regression_case("c1", "x", "a"),
            regression_case("c2", "y", "a"),
        ];
        let set = LoadedEvalset {
            manifest,
            cases,
            root: ".".into(),
        };
        let policy = gate_policy(&set.manifest);
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_crash",
            |_| Err("backend down".to_string()),
        );
        assert_eq!(run.scores.crash_or_timeout, 2);
        assert_eq!(run.scores.latency_p95_ms, None);
        assert_eq!(run.scores.verdict, GateVerdict::Inconclusive);
    }

    #[test]
    fn latency_gate_with_partial_latency_coverage_is_inconclusive() {
        let mut set = classify_set();
        set.manifest.gate = Some(Gate {
            max_p95_latency_ms: Some(800.0),
            min_quality: Some(0.5),
            min_cases: Some(1),
            ..Gate::default()
        });
        let policy = gate_policy(&set.manifest);
        let run = run_evalset(
            &set,
            CandidateRef::new("m"),
            env(),
            &policy,
            None,
            &RunOptions::default(),
            "run_partial_latency",
            |case| match &case.input {
                CaseInput::Text(t) if t.contains("refund") => Ok(CaseOutcome::text("refund", 10)),
                _ => Ok(CaseOutcome {
                    output: GradeOutput::Text("cancel".into()),
                    latency_ms: None,
                }),
            },
        );
        assert_eq!(run.scores.pass, 2);
        assert_eq!(run.scores.latency_p95_ms, Some(10.0));
        assert_eq!(run.scores.verdict, GateVerdict::Inconclusive);
    }
}

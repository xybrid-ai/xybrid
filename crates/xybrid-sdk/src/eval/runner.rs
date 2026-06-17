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
    aggregate_scores, CandidateRef, Environment, JudgeIdentity, Run, RunCase, Scores,
};
use crate::eval::stats::{
    GatePolicy, DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_CONFIDENCE, DEFAULT_SEED,
};

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
#[derive(Debug, Clone)]
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

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            capture_outputs: true,
            baseline_quality: None,
            today: None,
        }
    }
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
    GatePolicy {
        min_cases: gate.and_then(|g| g.min_cases).unwrap_or(1),
        min_quality: gate.and_then(|g| g.min_quality),
        max_p95_latency_ms: gate.and_then(|g| g.max_p95_latency_ms),
        // Clamp to >= 0: a negative margin would invert the tie logic (a clear
        // loss read as a pass). A margin is a non-negative slack band.
        non_inferiority_margin: gate
            .and_then(|g| g.non_inferiority_margin)
            .unwrap_or(0.0)
            .max(0.0),
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
    let mut grades: Vec<CaseGrade> = Vec::with_capacity(set.cases.len());
    let mut latencies: Vec<f64> = Vec::new();
    let mut run_cases: Vec<RunCase> = Vec::with_capacity(set.cases.len());
    let mut crashes: u32 = 0;

    for case in &set.cases {
        // Governance: never score a quarantined (known-bad) case, and skip
        // expired (stale) cases when a date is supplied. Both are retained on
        // disk for audit but excluded from scoring/gates.
        if case.is_quarantined() {
            continue;
        }
        if options
            .today
            .as_deref()
            .is_some_and(|today| case.is_expired_on(today))
        {
            continue;
        }
        match infer(case) {
            Ok(outcome) => {
                let grade = grade_case(&set.manifest, case, &outcome.output, judge);
                if let Some(ms) = outcome.latency_ms {
                    latencies.push(ms as f64);
                }
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
                });
                grades.push(grade);
            }
            Err(err) => {
                crashes += 1;
                let grade = CaseGrade::fail(format!("inference error: {err}"));
                run_cases.push(RunCase {
                    id: case.id.clone(),
                    output: None,
                    verdict: Verdict::Fail,
                    score: 0.0,
                    latency_ms: None,
                    detail: grade.detail.clone(),
                });
                grades.push(grade);
            }
        }
    }

    let mut scores: Scores =
        aggregate_scores(&grades, &latencies, policy, options.baseline_quality);
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

/// Serialize a grade output for storage in a run record.
fn output_to_json(output: &GradeOutput) -> serde_json::Value {
    match output {
        GradeOutput::Text(t) => serde_json::Value::String(t.clone()),
        GradeOutput::Json(v) => v.clone(),
        GradeOutput::Embedding(v) => serde_json::json!(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::format::{Case, CaseInput, Evalset, Expected, LoadedEvalset, TaskType};
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
            Case::new("c1", CaseInput::Text("refund please".into()))
                .with_expected(Expected::Label("refund".into())),
            Case::new("c2", CaseInput::Text("cancel my order".into()))
                .with_expected(Expected::Label("cancel".into())),
        ];
        LoadedEvalset {
            manifest,
            cases,
            root: std::path::PathBuf::from("."),
        }
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
        // outputs captured by default
        assert!(run.cases[0].output.is_some());
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
    fn capture_outputs_false_omits_payloads() {
        let set = classify_set();
        let policy = gate_policy(&set.manifest);
        let opts = RunOptions {
            capture_outputs: false,
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
        assert!(run.cases.iter().all(|c| c.output.is_none()));
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
    fn quarantined_and_expired_cases_are_excluded_from_a_run() {
        let mut manifest = Evalset::new("s", TaskType::Classify);
        manifest.labels = vec!["a".into()];
        let good = Case::new("good", CaseInput::Text("x".into()))
            .with_expected(Expected::Label("a".into()));
        let mut quarantined = Case::new("bad", CaseInput::Text("x".into()))
            .with_expected(Expected::Label("a".into()));
        quarantined.quarantine_reason = Some("mislabeled".into());
        let mut expired = Case::new("old", CaseInput::Text("x".into()))
            .with_expected(Expected::Label("a".into()));
        expired.expires_at = Some("2020-01-01".into());
        let set = LoadedEvalset {
            manifest,
            cases: vec![good, quarantined, expired],
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
        // Only the one healthy case is scored; the quarantined + expired cases
        // are excluded from the run entirely (retained on disk for audit).
        assert_eq!(run.cases.len(), 1);
        assert_eq!(run.cases[0].id, "good");
        assert_eq!(run.scores.pass, 1);
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
            Case::new("c1", CaseInput::Text("x".into())).with_expected(Expected::Label("a".into())),
            Case::new("c2", CaseInput::Text("y".into())).with_expected(Expected::Label("a".into())),
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
}

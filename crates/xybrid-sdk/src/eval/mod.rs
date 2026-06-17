//! Eval-driven development for Xybrid (the eval harness loop).
//!
//! This module is the local-first engine behind the five-verb loop
//! `flag → collect → compare → gate → ship`. It owns the evalset file format,
//! the task-implied graders, the run/score records, and the statistical gate
//! policy — everything needed to score a candidate against an evalset offline,
//! with no platform account.
//!
//! Layers:
//! - [`format`] — on-disk schema (`Case`, `Evalset`) + loader with path-traversal
//!   containment.
//! - [`grader`] — task-implied graders (label match, WER, golden mode, …) that
//!   normalize every result to a `0..=1` quality score + per-case verdict.
//! - [`stats`] — deterministic bootstrap confidence intervals and the
//!   pass/fail/inconclusive gate policy (min-N, non-inferiority, flaky guard).
//! - [`run`] — run/score records and the on-disk run store.
//!
//! Design stance: the task type implies the grader, quality is always a
//! normalized `0..=1` score, determinism is pinned by seed, and the trust-layer
//! schemas (case governance, judge identity, statistical gate) are frozen here.

pub mod deploy;
pub mod format;
pub mod grader;
pub mod monitor;
pub mod prompt;
pub mod run;
pub mod runner;
pub mod stats;

pub use deploy::{
    now_rfc3339, rollback_decision, DeploymentStatus, DeploymentStore, PromotionRecord,
    RollbackTrigger,
};
pub use format::{
    Case, CaseInput, CaseSource, EvalError, Evalset, EvalsetKind, Expected, Gate, GraderConfig,
    LoadedEvalset, PrivacyClass, ReviewStatus, Severity, Split, TaskType,
};
pub use grader::{
    bless, grade_case, wer, CaseGrade, GradeOutput, Grader, Judge, OverlapJudge, Verdict,
};
pub use monitor::{structural_signals, BehavioralSignal, StructuralSignals};
pub use prompt::{
    estimate_tokens, lint_prompt, model_profile, ModelProfile, PromptLibrary, PromptSuggestion,
    PromptVersion, SuggestionLevel,
};
pub use run::{
    aggregate_scores, weighted_quality, CandidateConfig, CandidateRef, Environment, EvalRunStore,
    JudgeIdentity, Run, RunCase, Scores,
};
pub use runner::{gate_policy, run_evalset, today_utc, CaseOutcome, RunOptions};
pub use stats::{
    bootstrap_ci, is_flaky, ConfidenceInterval, GateDecision, GatePolicy, GateVerdict, Rng,
};

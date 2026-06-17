//! Task-implied graders.
//!
//! The task type implies the grader — Tier 1 never chooses one.
//! Every grader normalizes its result to a **finite `0..=1` quality score** plus
//! a per-case [`Verdict`], so the CLI table, the console, and the gate contract
//! never depend on which grader produced the number.
//!
//! Implemented here (deterministic, no model needed to *grade*):
//! - `classify` — normalized label match with an alias map from `evalset.labels`.
//! - `asr` — word-level Word Error Rate, `quality = clamp(1 - WER, 0, 1)`.
//! - `extract` — per-field match over a reference JSON object.
//! - golden mode — a case with no reference is **`Unblessed`** (cannot be
//!   scored); blessing writes the output into `expected` (see [`bless`]).
//! - `chat` / `summarize` / `vlm` — routed to a [`Judge`] when one is supplied;
//!   the real LLM judge is deferred, so an offline deterministic [`OverlapJudge`]
//!   stands in. With a reference but no judge, deterministic exact-diff golden.
//!
//! Deferred (need the runner / a real judge / labeled pairs): `tts` perceptual,
//! `embedding` recall@k, and the calibrated LLM judge.

use serde::{Deserialize, Serialize};

use crate::eval::format::{Case, CaseInput, Evalset, Expected, ReviewStatus, TaskType};

// ============================================================================
// Verdict & grade
// ============================================================================

/// Per-case verdict. `Unblessed` is a first-class state for golden-mode cases
/// with no blessed reference yet — they cannot be passed or failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Met the bar.
    Pass,
    /// Missed the bar.
    Fail,
    /// Golden case with no blessed reference — not scorable.
    Unblessed,
}

/// The result of grading one case: a normalized `0..=1` score + a verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseGrade {
    /// Quality in `0..=1` (always finite).
    pub score: f64,
    /// Pass / fail / unblessed.
    pub verdict: Verdict,
    /// Human-readable reason (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl CaseGrade {
    /// A passing grade (score 1.0).
    pub fn pass() -> Self {
        Self {
            score: 1.0,
            verdict: Verdict::Pass,
            detail: None,
        }
    }

    /// A failing grade (score 0.0) with a reason.
    pub fn fail(detail: impl Into<String>) -> Self {
        Self {
            score: 0.0,
            verdict: Verdict::Fail,
            detail: Some(detail.into()),
        }
    }

    /// An unblessed (unscorable) golden case.
    pub fn unblessed(detail: impl Into<String>) -> Self {
        Self {
            score: 0.0,
            verdict: Verdict::Unblessed,
            detail: Some(detail.into()),
        }
    }

    /// A graded result: clamp `score` to a finite `0..=1` and derive the verdict
    /// from `threshold` (default pass bar).
    pub fn scored(score: f64, threshold: f64) -> Self {
        let score = clamp01(score);
        let verdict = if score >= threshold {
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        Self {
            score,
            verdict,
            detail: None,
        }
    }

    /// Whether this grade counts toward an aggregate (Unblessed does not).
    pub fn is_scorable(&self) -> bool {
        self.verdict != Verdict::Unblessed
    }
}

/// Clamp to a finite `0..=1`; map NaN to 0.0 (never propagate NaN into a gate).
pub fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// Default pass threshold for a scored grade when the manifest sets none.
const DEFAULT_PASS_THRESHOLD: f64 = 0.5;

// ============================================================================
// Candidate output
// ============================================================================

/// The output produced by running a candidate on a case — the thing graders
/// score. Mirrors the relevant `Envelope` kinds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GradeOutput {
    /// Text output (classify label, ASR transcript, chat answer).
    Text(String),
    /// Structured output (extraction).
    Json(serde_json::Value),
    /// Embedding vector.
    Embedding(Vec<f32>),
}

impl GradeOutput {
    /// View the output as text, if it is text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            GradeOutput::Text(t) => Some(t),
            _ => None,
        }
    }
}

// ============================================================================
// Grader trait + judge seam
// ============================================================================

/// A grader scores one case's output. The dispatcher [`grade_case`] selects the
/// right one from the task type; Tier 3 can supply a custom implementation.
pub trait Grader {
    /// Score `output` for `case` under `manifest`.
    fn grade(&self, manifest: &Evalset, case: &Case, output: &GradeOutput) -> CaseGrade;
}

/// An LLM-as-judge. The calibrated implementation is deferred; this trait is the
/// seam the runner calls. [`OverlapJudge`] is a deterministic offline stand-in.
pub trait Judge {
    /// Score `output` against the `input` and optional `reference`.
    fn grade(
        &self,
        input: &CaseInput,
        reference: Option<&Expected>,
        output: &GradeOutput,
    ) -> CaseGrade;

    /// The judge model identifier (recorded on the run for governance).
    fn judge_model(&self) -> &str;
}

// ============================================================================
// Dispatcher
// ============================================================================

/// Grade a single case by its evalset's task type.
///
/// Golden mode (no `expected`): reference-free judge for chat/summarize/vlm when
/// a `judge` is supplied, otherwise `Unblessed`. With a reference, the
/// task-implied deterministic grader runs (judge for chat/summarize/vlm).
pub fn grade_case(
    manifest: &Evalset,
    case: &Case,
    output: &GradeOutput,
    judge: Option<&dyn Judge>,
) -> CaseGrade {
    // Per-case pass bar is a fixed default — it is NOT the manifest's aggregate
    // `gate.min_quality` (that applies to the run mean in the gate policy).
    // Conflating them would silently turn a 0.92 aggregate gate into a 0.92
    // per-case bar for ASR/judge tasks.
    let threshold = DEFAULT_PASS_THRESHOLD;

    match &case.expected {
        // ---- Golden mode: no reference yet ----
        None => match manifest.task {
            TaskType::Chat | TaskType::Summarize | TaskType::Vlm => match judge {
                Some(j) => j.grade(&case.input, None, output),
                None => CaseGrade::unblessed("no reference and no judge configured"),
            },
            _ => CaseGrade::unblessed("golden case awaiting a blessed reference"),
        },

        // ---- Reference present ----
        Some(expected) => match manifest.task {
            TaskType::Classify => classify_grade(&manifest.labels, expected, output),
            TaskType::Asr => asr_grade(expected, output, threshold),
            TaskType::Extract => extract_grade(expected, output),
            TaskType::Chat | TaskType::Summarize | TaskType::Vlm => match judge {
                Some(j) => j.grade(&case.input, Some(expected), output),
                // Deterministic golden fallback: exact normalized diff.
                None => golden_text_grade(expected, output),
            },
            TaskType::Tts => CaseGrade::unblessed("tts grading deferred (needs audio runner)"),
            TaskType::Embedding => {
                CaseGrade::unblessed("embedding recall@k deferred (needs ranked output)")
            }
        },
    }
}

// ============================================================================
// Blessing (golden-mode pin) — a mutation, separate from scoring
// ============================================================================

/// Pin a candidate's `output` as the golden reference on `case`: write it into
/// `expected` and mark the case `review_status: golden`. This is the "approve a
/// good output" curation step, deliberately separate from grading.
pub fn bless(case: &mut Case, output: &GradeOutput) {
    case.expected = Some(match output {
        GradeOutput::Text(t) => Expected::Text(t.clone()),
        GradeOutput::Json(v) => Expected::Json(v.clone()),
        GradeOutput::Embedding(v) => Expected::Json(serde_json::json!(v)),
    });
    case.review_status = ReviewStatus::Golden;
}

// ============================================================================
// Deterministic graders
// ============================================================================

/// Classify: normalize the output, resolve it against the declared labels (the
/// alias map), and compare to the expected label. An output that resolves to no
/// declared label fails (not an error).
fn classify_grade(labels: &[String], expected: &Expected, output: &GradeOutput) -> CaseGrade {
    let expected_label = match expected {
        Expected::Label(l) | Expected::Text(l) => l.clone(),
        Expected::Json(v) => v.as_str().map(str::to_string).unwrap_or_default(),
    };
    let Some(out) = output.as_text() else {
        return CaseGrade::fail("classify output is not text");
    };
    let norm_out = normalize(out);
    let norm_expected = normalize(&expected_label);

    // With declared labels, the output must resolve to one of them.
    if !labels.is_empty() {
        let resolved = labels.iter().find(|l| normalize(l) == norm_out);
        return match resolved {
            Some(l) if normalize(l) == norm_expected => CaseGrade::pass(),
            Some(l) => CaseGrade::fail(format!("predicted '{l}', expected '{expected_label}'")),
            None => CaseGrade::fail(format!("'{out}' matched no declared label")),
        };
    }
    // No declared labels: direct normalized comparison.
    if norm_out == norm_expected {
        CaseGrade::pass()
    } else {
        CaseGrade::fail(format!("predicted '{out}', expected '{expected_label}'"))
    }
}

/// ASR: word-level WER, `quality = clamp(1 - WER, 0, 1)`.
fn asr_grade(expected: &Expected, output: &GradeOutput, threshold: f64) -> CaseGrade {
    let reference = match expected {
        Expected::Text(t) | Expected::Label(t) => t.clone(),
        Expected::Json(v) => v.as_str().map(str::to_string).unwrap_or_default(),
    };
    let Some(out) = output.as_text() else {
        return CaseGrade::fail("asr output is not text");
    };
    let rate = wer(&reference, out);
    let mut grade = CaseGrade::scored(1.0 - rate, threshold);
    grade.detail = Some(format!("WER {:.3}", rate));
    grade
}

/// Extract: fraction of reference object fields that match the output object.
/// All fields matching ⇒ pass.
fn extract_grade(expected: &Expected, output: &GradeOutput) -> CaseGrade {
    let expected_json = match expected {
        Expected::Json(v) => v.clone(),
        Expected::Text(t) | Expected::Label(t) => serde_json::Value::String(t.clone()),
    };
    let out_json = match output {
        GradeOutput::Json(v) => v.clone(),
        GradeOutput::Text(t) => match serde_json::from_str::<serde_json::Value>(t) {
            Ok(v) => v,
            Err(_) => return CaseGrade::fail("extract output is not valid JSON"),
        },
        GradeOutput::Embedding(_) => return CaseGrade::fail("extract output is an embedding"),
    };

    match expected_json.as_object() {
        Some(fields) if !fields.is_empty() => {
            let total = fields.len();
            let matched = fields
                .iter()
                .filter(|(k, v)| out_json.get(*k) == Some(*v))
                .count();
            let score = matched as f64 / total as f64;
            CaseGrade {
                score: clamp01(score),
                verdict: if matched == total {
                    Verdict::Pass
                } else {
                    Verdict::Fail
                },
                detail: Some(format!("{matched}/{total} fields matched")),
            }
        }
        // Non-object reference: exact equality.
        _ => {
            if out_json == expected_json {
                CaseGrade::pass()
            } else {
                CaseGrade::fail("extract output did not equal reference")
            }
        }
    }
}

/// Deterministic golden diff for text tasks: exact match after normalization.
fn golden_text_grade(expected: &Expected, output: &GradeOutput) -> CaseGrade {
    let reference = match expected {
        Expected::Text(t) | Expected::Label(t) => t.clone(),
        Expected::Json(v) => v.to_string(),
    };
    let out = match output {
        GradeOutput::Text(t) => t.clone(),
        GradeOutput::Json(v) => v.to_string(),
        GradeOutput::Embedding(_) => return CaseGrade::fail("golden output is an embedding"),
    };
    if normalize(&reference) == normalize(&out) {
        CaseGrade::pass()
    } else {
        CaseGrade::fail("output differs from blessed golden reference")
    }
}

// ============================================================================
// Text normalization + WER (no `regex` dependency — hand-rolled)
// ============================================================================

/// Normalize text for comparison: lowercase, replace any non-alphanumeric /
/// non-whitespace char with a space (punctuation strip), and collapse runs of
/// (Unicode) whitespace to single spaces.
fn normalize(s: &str) -> String {
    let lowered = s.to_lowercase();
    let spaced: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `c` belongs to a script that doesn't delimit words with spaces, so
/// WER must fall back to character-level tokens (otherwise a whole sentence is
/// one token and any single-char error scores WER 1.0).
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | // CJK ideographs + Ext A
        0x3040..=0x309F | 0x30A0..=0x30FF | // Hiragana, Katakana
        0xAC00..=0xD7AF |                   // Hangul syllables
        0x0E00..=0x0E7F) // Thai
}

/// Tokenize for WER: normalize then split on whitespace. A whitespace-delimited
/// word that contains any no-space-script (CJK/JP/KO/Thai) char is exploded into
/// per-character tokens; Latin/whitespace text tokenizes word-by-word as before.
fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in normalize(s).split_whitespace() {
        if word.chars().any(is_cjk) {
            tokens.extend(word.chars().map(|c| c.to_string()));
        } else {
            tokens.push(word.to_string());
        }
    }
    tokens
}

/// Word Error Rate = word edit distance / reference length. Unbounded above (an
/// insertion-heavy hypothesis can exceed 1.0). Empty-reference rule: an empty
/// hypothesis is perfect (0.0), any non-empty hypothesis is fully wrong (1.0).
pub fn wer(reference: &str, hypothesis: &str) -> f64 {
    let r = tokenize(reference);
    let h = tokenize(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    word_edit_distance(&r, &h) as f64 / r.len() as f64
}

/// Levenshtein edit distance over word tokens (two-row DP, O(n) memory).
fn word_edit_distance(a: &[String], b: &[String]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ai) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let cost = if ai == bj { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost) // substitute / match
                .min(prev[j + 1] + 1) // delete from a
                .min(curr[j] + 1); // insert into a
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

// ============================================================================
// Deterministic offline judge (placeholder for the calibrated LLM judge)
// ============================================================================

/// A deterministic, model-free judge used offline and in tests. It scores the
/// token-overlap (Jaccard) between the output and the reference (or, reference-
/// free, the input text). **Not a calibrated judge** — a stand-in so the runner
/// seam and determinism are exercised before the real judge lands.
#[derive(Debug, Clone)]
pub struct OverlapJudge {
    /// Pass threshold on the overlap score.
    pub threshold: f64,
}

impl Default for OverlapJudge {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl Judge for OverlapJudge {
    fn grade(
        &self,
        input: &CaseInput,
        reference: Option<&Expected>,
        output: &GradeOutput,
    ) -> CaseGrade {
        let out = output.as_text().unwrap_or_default();
        let basis = match reference {
            Some(Expected::Text(t)) | Some(Expected::Label(t)) => t.clone(),
            Some(Expected::Json(v)) => v.to_string(),
            None => match input {
                CaseInput::Text(t) => t.clone(),
                _ => String::new(),
            },
        };
        let score = jaccard(&tokenize(&basis), &tokenize(out));
        let mut grade = CaseGrade::scored(score, self.threshold);
        grade.detail = Some(format!("overlap {:.3}", score));
        grade
    }

    fn judge_model(&self) -> &str {
        "overlap-judge-v0"
    }
}

/// Jaccard similarity over two token bags (by unique tokens). Empty/empty ⇒ 1.0.
fn jaccard(a: &[String], b: &[String]) -> f64 {
    use std::collections::BTreeSet;
    let sa: BTreeSet<&String> = a.iter().collect();
    let sb: BTreeSet<&String> = b.iter().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::format::{Case, CaseInput, EvalsetKind};
    use serde_json::json;

    fn classify_set(labels: &[&str]) -> Evalset {
        let mut s = Evalset::new("s", TaskType::Classify);
        s.labels = labels.iter().map(|l| l.to_string()).collect();
        s
    }

    fn case_with(expected: Option<Expected>) -> Case {
        let mut c = Case::new("c", CaseInput::Text("input".into()));
        c.expected = expected;
        c
    }

    // ---- classify ----

    #[test]
    fn classify_exact_label_passes() {
        let set = classify_set(&["refund", "cancel", "other"]);
        let case = case_with(Some(Expected::Label("refund".into())));
        let g = grade_case(&set, &case, &GradeOutput::Text("refund".into()), None);
        assert_eq!(g.verdict, Verdict::Pass);
        assert_eq!(g.score, 1.0);
    }

    #[test]
    fn classify_is_case_and_whitespace_insensitive() {
        let set = classify_set(&["refund", "cancel"]);
        let case = case_with(Some(Expected::Label("refund".into())));
        let g = grade_case(&set, &case, &GradeOutput::Text("  Refund ".into()), None);
        assert_eq!(g.verdict, Verdict::Pass);
    }

    #[test]
    fn classify_wrong_label_fails() {
        let set = classify_set(&["refund", "cancel"]);
        let case = case_with(Some(Expected::Label("refund".into())));
        let g = grade_case(&set, &case, &GradeOutput::Text("cancel".into()), None);
        assert_eq!(g.verdict, Verdict::Fail);
        assert_eq!(g.score, 0.0);
    }

    #[test]
    fn classify_undeclared_label_fails_not_errors() {
        let set = classify_set(&["refund", "cancel"]);
        let case = case_with(Some(Expected::Label("refund".into())));
        let g = grade_case(
            &set,
            &case,
            &GradeOutput::Text("refund_request".into()),
            None,
        );
        assert_eq!(g.verdict, Verdict::Fail);
        assert!(g.detail.unwrap().contains("matched no declared label"));
    }

    #[test]
    fn classify_without_declared_labels_compares_directly() {
        let set = Evalset::new("s", TaskType::Classify); // no labels
        let case = case_with(Some(Expected::Label("spam".into())));
        assert_eq!(
            grade_case(&set, &case, &GradeOutput::Text("SPAM".into()), None).verdict,
            Verdict::Pass
        );
        assert_eq!(
            grade_case(&set, &case, &GradeOutput::Text("ham".into()), None).verdict,
            Verdict::Fail
        );
    }

    // ---- WER / asr ----

    #[test]
    fn wer_identical_is_zero() {
        assert_eq!(wer("the cat sat", "the cat sat"), 0.0);
    }

    #[test]
    fn wer_normalizes_case_and_punctuation() {
        assert_eq!(wer("Hello, world!", "hello world"), 0.0);
    }

    #[test]
    fn wer_collapses_unicode_whitespace() {
        assert_eq!(wer("a  b\tc", "a b c"), 0.0);
    }

    #[test]
    fn wer_single_edits() {
        // one substitution out of 3 ref words
        assert!((wer("the cat sat", "the dog sat") - 1.0 / 3.0).abs() < 1e-9);
        // one deletion
        assert!((wer("the cat sat", "the sat") - 1.0 / 3.0).abs() < 1e-9);
        // one insertion
        assert!((wer("the cat sat", "the big cat sat") - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn wer_empty_reference_rules() {
        assert_eq!(wer("", ""), 0.0);
        assert_eq!(wer("", "word"), 1.0);
        assert_eq!(wer("a b", ""), 1.0);
    }

    #[test]
    fn wer_can_exceed_one_on_insertions_but_quality_clamps() {
        let rate = wer("hi", "hi a b c d e"); // 5 insertions / 1 ref word = 5.0
        assert!(rate > 1.0);
        let set = Evalset::new("s", TaskType::Asr);
        let case = case_with(Some(Expected::Text("hi".into())));
        let g = grade_case(&set, &case, &GradeOutput::Text("hi a b c d e".into()), None);
        assert_eq!(g.score, 0.0); // clamped, never negative
        assert_eq!(g.verdict, Verdict::Fail);
    }

    #[test]
    fn wer_cjk_falls_back_to_character_tokens() {
        // C4: `今天天气很好` vs `今天天气很差` — 1 of 6 chars wrong. Word-level
        // tokenization would make each side a single token (WER 1.0); char-level
        // fallback gives WER 1/6 → quality ≈ 0.833 (a Pass), not 0.0.
        let rate = wer("今天天气很好", "今天天气很差");
        assert!((rate - 1.0 / 6.0).abs() < 1e-9, "WER was {rate}");
        let set = Evalset::new("s", TaskType::Asr);
        let case = case_with(Some(Expected::Text("今天天气很好".into())));
        let g = grade_case(&set, &case, &GradeOutput::Text("今天天气很差".into()), None);
        assert!(
            (g.score - (1.0 - 1.0 / 6.0)).abs() < 1e-9,
            "score {}",
            g.score
        );
        assert_eq!(g.verdict, Verdict::Pass);
    }

    #[test]
    fn wer_latin_unaffected_by_cjk_fallback() {
        // Latin text must still tokenize word-by-word (regression guard for C4).
        assert_eq!(wer("the cat sat", "the cat sat"), 0.0);
        assert!((wer("the cat sat", "the dog sat") - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn asr_perfect_transcript_passes() {
        let set = Evalset::new("s", TaskType::Asr);
        let case = case_with(Some(Expected::Text("the quick brown fox".into())));
        let g = grade_case(
            &set,
            &case,
            &GradeOutput::Text("The quick brown fox.".into()),
            None,
        );
        assert_eq!(g.verdict, Verdict::Pass);
        assert_eq!(g.score, 1.0);
    }

    // ---- extract ----

    #[test]
    fn extract_all_fields_match_passes() {
        let set = Evalset::new("s", TaskType::Extract);
        let case = case_with(Some(Expected::Json(json!({"name":"Ada","year":1815}))));
        let g = grade_case(
            &set,
            &case,
            &GradeOutput::Json(json!({"name":"Ada","year":1815,"extra":true})),
            None,
        );
        assert_eq!(g.verdict, Verdict::Pass);
        assert_eq!(g.score, 1.0);
    }

    #[test]
    fn extract_partial_match_scores_fraction() {
        let set = Evalset::new("s", TaskType::Extract);
        let case = case_with(Some(Expected::Json(json!({"name":"Ada","year":1815}))));
        let g = grade_case(
            &set,
            &case,
            &GradeOutput::Json(json!({"name":"Ada","year":1900})),
            None,
        );
        assert_eq!(g.verdict, Verdict::Fail);
        assert!((g.score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn extract_from_text_output_parses_json() {
        let set = Evalset::new("s", TaskType::Extract);
        let case = case_with(Some(Expected::Json(json!({"k":"v"}))));
        let g = grade_case(&set, &case, &GradeOutput::Text(r#"{"k":"v"}"#.into()), None);
        assert_eq!(g.verdict, Verdict::Pass);
    }

    #[test]
    fn extract_non_json_text_fails() {
        let set = Evalset::new("s", TaskType::Extract);
        let case = case_with(Some(Expected::Json(json!({"k":"v"}))));
        let g = grade_case(&set, &case, &GradeOutput::Text("not json".into()), None);
        assert_eq!(g.verdict, Verdict::Fail);
    }

    // ---- golden mode ----

    #[test]
    fn golden_case_without_reference_is_unblessed() {
        let set = Evalset::new("s", TaskType::Summarize);
        let case = case_with(None);
        let g = grade_case(&set, &case, &GradeOutput::Text("anything".into()), None);
        assert_eq!(g.verdict, Verdict::Unblessed);
        assert!(!g.is_scorable());
    }

    #[test]
    fn classify_without_reference_is_unblessed() {
        let set = classify_set(&["a", "b"]);
        let case = case_with(None);
        let g = grade_case(&set, &case, &GradeOutput::Text("a".into()), None);
        assert_eq!(g.verdict, Verdict::Unblessed);
    }

    #[test]
    fn bless_then_grade_passes_on_same_output() {
        let set = Evalset::new("s", TaskType::Summarize);
        let mut case = case_with(None);
        let output = GradeOutput::Text("a crisp summary".into());
        // Unblessed before blessing.
        assert_eq!(
            grade_case(&set, &case, &output, None).verdict,
            Verdict::Unblessed
        );
        bless(&mut case, &output);
        assert_eq!(case.review_status, ReviewStatus::Golden);
        // Deterministic golden diff (no judge) now passes on the same output.
        assert_eq!(
            grade_case(&set, &case, &output, None).verdict,
            Verdict::Pass
        );
        // A different output fails.
        assert_eq!(
            grade_case(
                &set,
                &case,
                &GradeOutput::Text("totally different".into()),
                None
            )
            .verdict,
            Verdict::Fail
        );
    }

    // ---- judge seam ----

    #[test]
    fn overlap_judge_is_deterministic() {
        let judge = OverlapJudge::default();
        let set = Evalset::new("s", TaskType::Chat);
        let case = case_with(Some(Expected::Text(
            "the capital of france is paris".into(),
        )));
        let out = GradeOutput::Text("paris is the capital of france".into());
        let a = grade_case(&set, &case, &out, Some(&judge));
        let b = grade_case(&set, &case, &out, Some(&judge));
        assert_eq!(a, b);
        assert_eq!(a.verdict, Verdict::Pass);
        assert_eq!(judge.judge_model(), "overlap-judge-v0");
    }

    #[test]
    fn chat_reference_free_uses_judge_against_input() {
        let judge = OverlapJudge { threshold: 0.1 };
        let set = Evalset::new("s", TaskType::Chat);
        let mut case = Case::new("c", CaseInput::Text("tell me about rust".into()));
        case.expected = None; // reference-free
        let g = grade_case(
            &set,
            &case,
            &GradeOutput::Text("rust is a language".into()),
            Some(&judge),
        );
        assert_ne!(g.verdict, Verdict::Unblessed); // judge scored it
    }

    #[test]
    fn clamp01_handles_nan_and_range() {
        assert_eq!(clamp01(f64::NAN), 0.0);
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
        assert_eq!(clamp01(0.42), 0.42);
    }

    #[test]
    fn kind_field_is_unused_by_grader_but_present() {
        // Sanity: grading does not depend on evalset kind.
        let mut set = Evalset::new("s", TaskType::Classify);
        set.kind = EvalsetKind::Safety;
        set.labels = vec!["safe".into(), "unsafe".into()];
        let case = case_with(Some(Expected::Label("safe".into())));
        assert_eq!(
            grade_case(&set, &case, &GradeOutput::Text("safe".into()), None).verdict,
            Verdict::Pass
        );
    }
}

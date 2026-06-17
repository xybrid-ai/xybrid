//! Continuous quality monitoring — the always-on implicit signals that
//! complement explicit `result.report()` (continuous quality monitoring).
//!
//! Two cheap, on-device tiers live here:
//! - **Structural guards** ([`structural_signals`]): truncation, empty,
//!   repetition loops, refusal patterns, format validity — derived from the
//!   output text + finish reason on every call, no user interaction.
//! - **Behavioral signals** ([`BehavioralSignal`]): implicit feedback
//!   the app wires (`markRegenerated` ≈ a soft 👎, `markEdited`, `markDismissed`,
//!   `markUsed`, `markCopied`).
//!
//! Both feed the **same** failure inbox as the explicit loop, as lightweight,
//! `trace_id`-joinable, **metadata-only** telemetry signals. The server-side
//! tuning these feed (baselines, sampling, drift, alerts) is out of scope here
//! (tracked separately); this module is purely the device-side capture.

use serde::{Deserialize, Serialize};

/// Repetition score above which an output is considered a degenerate loop.
pub const REPETITION_THRESHOLD: f64 = 0.5;

/// Structural quality guards computed from an output. All are cheap and
/// reference-free — they read only the output text and the finish reason.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StructuralSignals {
    /// Output is empty / whitespace-only.
    pub empty: bool,
    /// Generation hit the token budget (finish reason indicates length).
    pub truncated: bool,
    /// Fraction of the output consumed by the longest consecutive n-gram repeat
    /// (`0.0`–`1.0`); high values indicate a degenerate loop.
    pub repetition_score: f64,
    /// The output looks like a refusal ("I can't…", "As an AI…").
    pub refusal_suspected: bool,
    /// When a structured format was expected, whether the output parsed as it
    /// (`None` when no format was expected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format_valid: Option<bool>,
}

impl StructuralSignals {
    /// Whether any guard tripped — i.e. this output is a monitoring candidate
    /// failure worth auto-flagging.
    pub fn has_issue(&self) -> bool {
        self.empty
            || self.truncated
            || self.refusal_suspected
            || self.repetition_score > REPETITION_THRESHOLD
            || self.format_valid == Some(false)
    }
}

/// Compute the structural guards for an output.
///
/// `finish_reason` is the backend's stop reason (e.g. `"length"`, `"stop"`);
/// `expect_json` enables the format-validity check.
pub fn structural_signals(
    text: &str,
    finish_reason: Option<&str>,
    expect_json: bool,
) -> StructuralSignals {
    let empty = text.trim().is_empty();
    let truncated = finish_reason
        .map(|r| {
            let r = r.to_ascii_lowercase();
            r.contains("length") || r.contains("max_tokens") || r == "truncated"
        })
        .unwrap_or(false);
    let format_valid = if expect_json {
        Some(serde_json::from_str::<serde_json::Value>(text.trim()).is_ok())
    } else {
        None
    };
    StructuralSignals {
        empty,
        truncated,
        repetition_score: repetition_score(text),
        refusal_suspected: refusal_suspected(text),
        format_valid,
    }
}

/// Fraction of the output consumed by the longest run of a consecutively-repeated
/// word window (size 1–4). `"go go go go"` → ~1.0; normal prose → near 0.
fn repetition_score(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();
    if n < 4 {
        return 0.0;
    }
    let mut best = 0usize;
    for w in 1..=4usize {
        let mut i = 0;
        while i + 2 * w <= n {
            let mut reps = 1usize;
            while i + (reps + 1) * w <= n
                && words[i..i + w] == words[i + reps * w..i + (reps + 1) * w]
            {
                reps += 1;
            }
            if reps >= 2 {
                best = best.max(reps * w);
                i += reps * w;
            } else {
                i += 1;
            }
        }
    }
    best as f64 / n as f64
}

/// A small set of refusal markers (case-insensitive substring match). Cheap and
/// intentionally conservative — it flags *suspected* refusals for review, not a
/// verdict.
const REFUSAL_MARKERS: &[&str] = &[
    "i can't",
    "i cannot",
    "i can not",
    "i'm sorry, but",
    "i am sorry, but",
    "i'm unable",
    "i am unable",
    "i won't be able",
    "as an ai",
    "i'm not able to",
    "i am not able to",
];

fn refusal_suspected(text: &str) -> bool {
    let lower = text.to_lowercase();
    REFUSAL_MARKERS.iter().any(|m| lower.contains(m))
}

/// Implicit behavioral feedback on a prior result. The app wires these
/// like it wired the thumbs-down; a `Regenerated` is treated as a soft negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BehavioralSignal {
    /// The user accepted/consumed the result (a soft positive).
    Used,
    /// The user asked to regenerate (a soft negative).
    Regenerated,
    /// The user edited the result before using it.
    Edited,
    /// The user copied the result.
    Copied,
    /// The user dismissed the result.
    Dismissed,
}

impl BehavioralSignal {
    /// Wire label.
    pub fn as_str(self) -> &'static str {
        match self {
            BehavioralSignal::Used => "used",
            BehavioralSignal::Regenerated => "regenerated",
            BehavioralSignal::Edited => "edited",
            BehavioralSignal::Copied => "copied",
            BehavioralSignal::Dismissed => "dismissed",
        }
    }

    /// Whether this signal is a soft-negative quality signal (a candidate
    /// failure for the inbox).
    pub fn is_soft_negative(self) -> bool {
        matches!(
            self,
            BehavioralSignal::Regenerated | BehavioralSignal::Dismissed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_output_flagged() {
        let s = structural_signals("   \n  ", Some("stop"), false);
        assert!(s.empty);
        assert!(s.has_issue());
    }

    #[test]
    fn truncated_detected_from_finish_reason() {
        assert!(structural_signals("a long answer", Some("length"), false).truncated);
        assert!(structural_signals("x", Some("max_tokens"), false).truncated);
        assert!(!structural_signals("done.", Some("stop"), false).truncated);
        assert!(!structural_signals("done.", None, false).truncated);
    }

    #[test]
    fn repetition_loop_detected() {
        let s = structural_signals("go go go go go go", Some("stop"), false);
        assert!(s.repetition_score > REPETITION_THRESHOLD, "{s:?}");
        assert!(s.has_issue());
        // window-2 loop
        let s2 = structural_signals("the cat the cat the cat the cat", Some("stop"), false);
        assert!(s2.repetition_score > REPETITION_THRESHOLD, "{s2:?}");
    }

    #[test]
    fn normal_prose_has_low_repetition() {
        let s = structural_signals(
            "The quick brown fox jumps over the lazy dog near the river.",
            Some("stop"),
            false,
        );
        assert!(s.repetition_score <= REPETITION_THRESHOLD, "{s:?}");
        assert!(!s.has_issue());
    }

    #[test]
    fn short_outputs_not_flagged_as_repetition() {
        assert_eq!(
            structural_signals("hi there", Some("stop"), false).repetition_score,
            0.0
        );
    }

    #[test]
    fn refusal_detected() {
        assert!(
            structural_signals("I can't help with that.", Some("stop"), false).refusal_suspected
        );
        assert!(
            structural_signals("As an AI language model, I cannot…", Some("stop"), false)
                .refusal_suspected
        );
        assert!(
            !structural_signals("Sure, here's the answer.", Some("stop"), false).refusal_suspected
        );
    }

    #[test]
    fn format_validity_only_when_expected() {
        assert_eq!(
            structural_signals("not json", Some("stop"), false).format_valid,
            None
        );
        assert_eq!(
            structural_signals("not json", Some("stop"), true).format_valid,
            Some(false)
        );
        assert_eq!(
            structural_signals(r#"{"k":"v"}"#, Some("stop"), true).format_valid,
            Some(true)
        );
    }

    #[test]
    fn clean_output_has_no_issue() {
        let s = structural_signals("The capital of France is Paris.", Some("stop"), false);
        assert!(!s.has_issue());
    }

    #[test]
    fn behavioral_signal_labels_and_polarity() {
        assert_eq!(BehavioralSignal::Regenerated.as_str(), "regenerated");
        assert!(BehavioralSignal::Regenerated.is_soft_negative());
        assert!(BehavioralSignal::Dismissed.is_soft_negative());
        assert!(!BehavioralSignal::Used.is_soft_negative());
        assert!(!BehavioralSignal::Copied.is_soft_negative());
    }
}

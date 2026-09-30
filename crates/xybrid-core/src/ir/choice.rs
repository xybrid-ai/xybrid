//! Choice-scoring data: the candidates a caller offers and the scores a
//! decision model returns for them.
//!
//! A choice scorer takes one context plus a closed list of candidate actions
//! and returns one score per candidate from a single forward pass, without
//! generating text. These types carry that request and result. They are plain
//! data: validation lives in [`crate::execution::choice`], and the runtimes
//! that produce [`ChoiceScores`] live next to their backends.
//!
//! Choice ids and texts are caller content. The [`Debug`] impls here print
//! lengths, counts and numbers only, so logging a request or result never
//! leaks what the caller offered.
//!
//! # Example
//!
//! ```
//! use xybrid_core::ir::{Choice, ChoiceScore, ChoiceScores};
//!
//! let offered = [Choice::new("tel", "fill Tel: (503) 555-0142"), Choice::new("skip", "skip")];
//! let scores = ChoiceScores {
//!     entries: vec![
//!         ChoiceScore { id: offered[0].id.clone(), logit: 3.0, score: 0.95, fixed: false },
//!         ChoiceScore { id: offered[1].id.clone(), logit: 0.0, score: 0.05, fixed: true },
//!     ],
//!     label_mass: None,
//! };
//! assert_eq!(scores.best(), Some(0));
//! assert!((scores.margin().unwrap() - 0.9).abs() < 1e-6);
//! assert!(!format!("{scores:?}").contains("tel"));
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(feature = "schema")]
use schemars::JsonSchema;

/// One candidate offered to a choice scorer.
///
/// `id` names the candidate in the result; `text` is what the model reads.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
pub struct Choice {
    /// Caller-chosen identifier, unique within one request.
    pub id: String,
    /// The candidate as the model sees it.
    pub text: String,
}

impl Choice {
    /// Creates a choice from its id and text.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

impl fmt::Debug for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Choice")
            .field("id_bytes", &self.id.len())
            .field("text_bytes", &self.text.len())
            .finish()
    }
}

/// The score of one candidate.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceScore {
    /// The candidate's id, as offered.
    pub id: String,
    /// The model's raw logit for the candidate.
    pub logit: f32,
    /// Softmax of the logits over the offered candidates.
    ///
    /// A relative preference among the candidates, not a calibrated
    /// probability that the candidate is correct.
    pub score: f32,
    /// `true` when the candidate came from the model's fixed choices rather
    /// than from the caller.
    pub fixed: bool,
}

impl fmt::Debug for ChoiceScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChoiceScore")
            .field("logit", &self.logit)
            .field("score", &self.score)
            .field("fixed", &self.fixed)
            .finish()
    }
}

/// Scores for every offered candidate, in the order they were offered
/// (caller choices first, then the model's fixed choices).
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceScores {
    /// One entry per candidate.
    pub entries: Vec<ChoiceScore>,
    /// Diagnostic for label-token scorers: the probability mass the full
    /// vocabulary puts on the candidate labels before normalization. `None`
    /// for scorers without a vocabulary.
    pub label_mass: Option<f32>,
}

impl ChoiceScores {
    /// Index of the best-scoring entry.
    ///
    /// Scores are rounded to `f32`, so two nearly equal candidates can share
    /// one; the higher logit breaks that tie, and the first entry wins an exact
    /// one. `None` when there are no entries.
    pub fn best(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let better = best.is_none_or(|b| {
                let current = &self.entries[b];
                entry.score > current.score
                    || (entry.score == current.score && entry.logit > current.logit)
            });
            if better {
                best = Some(index);
            }
        }
        best
    }

    /// The best score minus the second-best score.
    ///
    /// `0.0` for an exact tie; `None` with fewer than two entries.
    pub fn margin(&self) -> Option<f32> {
        let best = self.best()?;
        let runner_up = self
            .entries
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != best)
            .map(|(_, entry)| entry.score)
            .reduce(f32::max)?;
        Some(self.entries[best].score - runner_up)
    }
}

impl fmt::Debug for ChoiceScores {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChoiceScores")
            .field("count", &self.entries.len())
            .field("best", &self.best())
            .field("entries", &self.entries)
            .field("label_mass", &self.label_mass)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANARY: &str = "canary-7f3a9c";

    fn scores(values: &[f32]) -> ChoiceScores {
        ChoiceScores {
            entries: values
                .iter()
                .enumerate()
                .map(|(index, &score)| ChoiceScore {
                    id: format!("{CANARY}-{index}"),
                    logit: score,
                    score,
                    fixed: false,
                })
                .collect(),
            label_mass: None,
        }
    }

    #[test]
    fn best_prefers_the_first_of_equal_scores() {
        assert_eq!(scores(&[0.2, 0.4, 0.4]).best(), Some(1));
        assert_eq!(scores(&[0.5, 0.5]).best(), Some(0));
        assert_eq!(scores(&[]).best(), None);
    }

    #[test]
    fn best_breaks_rounded_score_ties_by_logit() {
        // Logits 0 and 1e-8 both round to a score of 0.5.
        let mut rounded = scores(&[0.5, 0.5]);
        rounded.entries[0].logit = 0.0;
        rounded.entries[1].logit = 1e-8;
        assert_eq!(rounded.best(), Some(1));
        assert_eq!(rounded.margin(), Some(0.0));
    }

    #[test]
    fn margin_is_top_minus_runner_up() {
        let margin = scores(&[0.1, 0.6, 0.3]).margin().unwrap();
        assert!((margin - 0.3).abs() < 1e-6);
        assert_eq!(scores(&[0.5, 0.5]).margin(), Some(0.0));
        assert_eq!(scores(&[1.0]).margin(), None);
    }

    #[test]
    fn debug_never_prints_ids_or_text() {
        let choice = Choice::new(CANARY, format!("text {CANARY}"));
        let rendered = format!("{choice:?} {:?}", scores(&[0.7, 0.3]));
        assert!(!rendered.contains(CANARY), "{rendered}");
        assert!(rendered.contains("count: 2"), "{rendered}");
    }

    #[test]
    fn serde_round_trips() {
        let original = scores(&[0.25, 0.75]);
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(
            serde_json::from_str::<ChoiceScores>(&json).unwrap(),
            original
        );
        let choice: Choice = serde_json::from_str(r#"{"id":"a","text":"b"}"#).unwrap();
        assert_eq!(choice, Choice::new("a", "b"));
    }
}

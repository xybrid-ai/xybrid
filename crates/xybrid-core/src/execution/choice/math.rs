//! Score math shared by every choice scorer.
//!
//! Scores are the softmax of the candidates' logits, divided by the scoring
//! temperature first when one is set. The arithmetic runs in `f64` with the
//! maximum shifted out, so large logits cannot overflow; the results are
//! stored as `f32`. Ties go to the higher logit, then the first candidate
//! (see [`ChoiceScores::best`]). A NaN or infinite logit on an offered candidate is
//! an error, never a silently skewed distribution. A static model's padding
//! rows must be dropped **before** calling in here: they may be NaN.
//!
//! A label-token scorer reads a group of label tokens per candidate:
//! [`label_scores`] takes the largest logit in each group as the candidate's
//! logit and reports how much of the whole next-token distribution the labels
//! hold.
//!
//! # Example
//!
//! ```
//! use xybrid_core::execution::choice::math::softmax_scores;
//!
//! let scores = softmax_scores(&[1.0, 1.0, 0.0], None)?;
//! assert_eq!(scores[0], scores[1]);
//! assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-6);
//! assert!(softmax_scores(&[1.0, f32::NAN], None).is_err());
//! # Ok::<(), xybrid_core::execution::choice::ChoiceError>(())
//! ```

use super::{ChoiceError, ChoiceResult};
use crate::ir::{ChoiceScore, ChoiceScores};

/// Softmax of `logits / temperature` (`temperature` defaults to 1).
///
/// # Errors
///
/// [`ChoiceError::NoCandidates`] for an empty slice,
/// [`ChoiceError::NonFiniteLogit`] with the first offending index, and
/// [`ChoiceError::InvalidSpec`] for a temperature that is not finite and
/// positive.
pub fn softmax_scores(logits: &[f32], temperature: Option<f32>) -> ChoiceResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(ChoiceError::NoCandidates);
    }
    if let Some(index) = logits.iter().position(|logit| !logit.is_finite()) {
        return Err(ChoiceError::NonFiniteLogit { index });
    }
    let temperature = f64::from(temperature.unwrap_or(1.0));
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ChoiceError::InvalidSpec(
            "scoring_temperature must be finite and greater than 0".into(),
        ));
    }

    let scaled: Vec<f64> = logits
        .iter()
        .map(|&logit| f64::from(logit) / temperature)
        .collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|x| (x - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    Ok(exps.iter().map(|e| (e / total) as f32).collect())
}

/// Pairs E, as `(id, fixed)` in E order, with its logits and scores.
///
/// `logits` holds exactly one value per candidate, in E order — a static
/// model's padding rows already removed.
///
/// # Errors
///
/// As [`softmax_scores`].
///
/// # Panics
///
/// When `ids` and `logits` differ in length: the caller must have checked the
/// model's output shape first.
pub fn choice_scores(
    ids: Vec<(String, bool)>,
    logits: &[f32],
    temperature: Option<f32>,
) -> ChoiceResult<ChoiceScores> {
    assert_eq!(
        ids.len(),
        logits.len(),
        "choice_scores needs one logit per candidate; check the output shape first"
    );
    let scores = softmax_scores(logits, temperature)?;
    let entries = ids
        .into_iter()
        .zip(logits.iter().zip(scores))
        .map(|((id, fixed), (&logit, score))| ChoiceScore {
            id,
            logit,
            score,
            fixed,
        })
        .collect();
    Ok(ChoiceScores {
        entries,
        label_mass: None,
    })
}

/// Scores a label-token readout: each candidate's logit is the largest logit
/// in its label group, scored as in [`choice_scores`]. `label_mass` is the
/// share of the whole next-token distribution the labels hold,
/// `Σ exp(logit − logsumexp)` over every label token, summed in `f64`.
///
/// `ids` and `groups` are both in E order, one group per candidate;
/// `logsumexp` is that of the full vocabulary row the logits were read from.
///
/// # Errors
///
/// [`ChoiceError::NonFiniteLogit`] with the candidate's index when any of its
/// label logits is NaN or infinite, [`ChoiceError::NonFiniteLogSumExp`] when
/// `logsumexp` is, and otherwise as [`softmax_scores`].
///
/// # Panics
///
/// When `ids` and `groups` differ in length or a group is empty: the caller
/// builds one non-empty group per candidate.
pub fn label_scores(
    ids: Vec<(String, bool)>,
    groups: &[Vec<f32>],
    logsumexp: f32,
    temperature: Option<f32>,
) -> ChoiceResult<ChoiceScores> {
    assert_eq!(
        ids.len(),
        groups.len(),
        "label_scores needs one label group per candidate"
    );
    let mut logits = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        assert!(!group.is_empty(), "label group {index} is empty");
        if group.iter().any(|logit| !logit.is_finite()) {
            return Err(ChoiceError::NonFiniteLogit { index });
        }
        logits.push(group.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    }
    if !logsumexp.is_finite() {
        return Err(ChoiceError::NonFiniteLogSumExp);
    }

    let mut scores = choice_scores(ids, &logits, temperature)?;
    let logsumexp = f64::from(logsumexp);
    let mass: f64 = groups
        .iter()
        .flatten()
        .map(|&logit| (f64::from(logit) - logsumexp).exp())
        .sum();
    scores.label_mass = Some(mass as f32);
    Ok(scores)
}

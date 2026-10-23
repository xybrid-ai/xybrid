//! Score math shared by every choice scorer.
//!
//! Scores are the softmax of the candidates' logits, divided by the scoring
//! temperature first when one is set. The arithmetic runs in `f64` with the
//! maximum shifted out, so large logits cannot overflow; the results are
//! stored as `f32`. Ties go to the higher logit, then the first candidate
//! (see [`ChoiceScores::best`]). A NaN or infinite logit on an offered candidate is
//! an error, never a silently skewed distribution. A static model's padding
//! rows must be dropped **before** calling in here: they may be NaN.
//! [`label_mass`] is the label scorers' diagnostic: the share of the full
//! next-token distribution that went to any label.
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

/// The probability a next-token distribution puts on a set of label
/// tokens: `Σ exp(logit − logsumexp)`, summed in `f64` and stored as `f32`.
///
/// `token_logits` holds each label token's logit once (a repeated token
/// would count twice), at temperature 1, and `logsumexp` is the log-sum-exp
/// of the whole row they were read from. This is the
/// [`ChoiceScores::label_mass`] diagnostic: how much of the model's
/// next-token probability went to any label at all, before the candidates
/// are normalized against each other. `None` for an empty slice or any
/// non-finite input.
///
/// # Example
///
/// ```
/// use xybrid_core::execution::choice::math::label_mass;
///
/// // Two labels holding e^0 and e^1 of a row whose total is e^0 + e^1 + e^2.
/// let lse = (1.0_f64 + 1.0_f64.exp() + 2.0_f64.exp()).ln() as f32;
/// let mass = label_mass(&[0.0, 1.0], lse).unwrap();
/// assert!((mass - 0.33).abs() < 0.01);
/// assert_eq!(label_mass(&[], lse), None);
/// ```
pub fn label_mass(token_logits: &[f32], logsumexp: f32) -> Option<f32> {
    if token_logits.is_empty()
        || !logsumexp.is_finite()
        || token_logits.iter().any(|logit| !logit.is_finite())
    {
        return None;
    }
    let logsumexp = f64::from(logsumexp);
    let mass: f64 = token_logits
        .iter()
        .map(|&logit| (f64::from(logit) - logsumexp).exp())
        .sum();
    Some(mass as f32)
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

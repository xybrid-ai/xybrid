//! Independent selected-logit validation and Liquid answer finalization.

use super::liquid::DecisionProtocolError as Error;
use super::prepare::{AnswerKind, PreparedQuestion};
use crate::ir::decision::{
    DecisionAnswer, DecisionOrigin, DecisionScore, DecisionValue, ScoreStatistic,
};

/// Reduces aliases by maximum and constructs a validated answer in caller order.
///
/// Every selected score is checked before alias reduction, including losing
/// aliases. Computation follows the reference's ordered f64 softmax and median.
pub fn finalize(
    prepared: &PreparedQuestion,
    selected_logits: &[f32],
) -> Result<DecisionAnswer, Error> {
    let plan = &prepared.plan;
    if selected_logits.len() != prepared.select.len() || plan.groups.is_empty() {
        return Err(Error::InvalidOutput);
    }
    let (valid_count, max_aliases) = match plan.kind {
        AnswerKind::Choice => ((1..=255).contains(&plan.groups.len()), 2),
        AnswerKind::Noul => (plan.groups.len() == 2, 3),
        AnswerKind::Score => ((2..=10).contains(&plan.groups.len()), 1),
    };
    if !valid_count {
        return Err(Error::InvalidOutput);
    }
    if selected_logits
        .iter()
        .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err(Error::NonFiniteScores);
    }
    if !plan.temperature.is_finite() || plan.temperature <= 0.0 {
        return Err(Error::InvalidMetadata);
    }
    let mut scores = Vec::with_capacity(plan.groups.len());
    let mut next = 0;
    for group in &plan.groups {
        if group.range.start != next || group.range.is_empty() || group.range.len() > max_aliases {
            return Err(Error::InvalidOutput);
        }
        let aliases = selected_logits
            .get(group.range.clone())
            .ok_or(Error::InvalidOutput)?;
        scores.push(aliases.iter().copied().fold(f32::NEG_INFINITY, first_max));
        next = group.range.end;
    }
    if next != selected_logits.len() {
        return Err(Error::InvalidOutput);
    }
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, first_max);
    if !maximum.is_finite() {
        return Err(Error::NonFiniteScores);
    }
    let temperature = f64::from(plan.temperature);
    let mut probabilities = Vec::with_capacity(scores.len());
    let mut total = 0.0_f64;
    for score in &scores {
        let probability = ((f64::from(*score) - f64::from(maximum)) / temperature).exp();
        probabilities.push(probability);
        total += probability;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::NonFiniteScores);
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    // Strictly greater preserves the first option at an exact tie.
    let mut best = 0;
    for index in 1..probabilities.len() {
        if probabilities[index] > probabilities[best] {
            best = index;
        }
    }
    let (value, confidence) = match plan.kind {
        AnswerKind::Choice => (
            DecisionValue::choice(&plan.groups[best].key),
            Some(probabilities[best]),
        ),
        AnswerKind::Noul => {
            if plan.groups.len() != 2
                || plan.groups[0].key != "true"
                || plan.groups[1].key != "false"
            {
                return Err(Error::InvalidOutput);
            }
            (DecisionValue::yes_no(probabilities[0]), None)
        }
        AnswerKind::Score => {
            let mut cumulative = 0.0;
            let mut median = probabilities.len() - 1;
            for (index, probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if cumulative >= 0.5 {
                    median = index;
                    break;
                }
            }
            (
                DecisionValue::score(median as f64, ScoreStatistic::Median),
                Some(probabilities[best]),
            )
        }
    };
    let distribution = plan
        .groups
        .iter()
        .zip(scores)
        .zip(probabilities)
        .map(|((group, raw), probability)| {
            DecisionScore::new(&group.key, raw, probability, DecisionOrigin::Caller)
        })
        .collect();
    DecisionAnswer::new(&plan.question, distribution, value, confidence).map_err(Into::into)
}

fn first_max(first: f32, second: f32) -> f32 {
    if second > first {
        second
    } else {
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::decision::prepare::{AnswerPlan, LabelGroup};

    fn prepared(kind: AnswerKind, lengths: &[usize]) -> PreparedQuestion {
        let mut cursor = 0;
        let groups = lengths
            .iter()
            .enumerate()
            .map(|(i, len)| {
                let key = if kind == AnswerKind::Noul {
                    ["true", "false"][i].to_owned()
                } else {
                    i.to_string()
                };
                let range = cursor..cursor + len;
                cursor += len;
                LabelGroup {
                    code: key.clone(),
                    key,
                    range,
                }
            })
            .collect();
        PreparedQuestion {
            tokens: vec![1],
            select: vec![0; cursor],
            plan: AnswerPlan {
                question: "q".into(),
                kind,
                groups,
                temperature: 1.0,
                temperature_text: None,
            },
        }
    }

    #[test]
    fn rejects_non_finite_losing_aliases_before_max() {
        let pass = prepared(AnswerKind::Choice, &[2, 1]);
        for bad in [f32::NAN, f32::INFINITY] {
            assert_eq!(
                finalize(&pass, &[1.0, bad, 0.0]).unwrap_err(),
                Error::NonFiniteScores
            );
        }
    }

    #[test]
    fn negative_infinity_is_allowed_unless_all_options_are_impossible() {
        let pass = prepared(AnswerKind::Choice, &[2, 1]);
        let answer = finalize(&pass, &[f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0]).unwrap();
        assert_eq!(answer.distribution()[0].raw_score(), f32::NEG_INFINITY);
        assert_eq!(answer.distribution()[0].probability(), 0.0);
        assert_eq!(answer.value().as_choice(), Some("1"));
        assert_eq!(
            finalize(&pass, &[f32::NEG_INFINITY; 3]).unwrap_err(),
            Error::NonFiniteScores
        );
        assert!(serde_json::to_string(&answer)
            .unwrap()
            .contains("-Infinity"));
    }

    #[test]
    fn alias_max_does_not_reward_more_aliases() {
        let pass = prepared(AnswerKind::Choice, &[2, 1]);
        let answer = finalize(&pass, &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(answer.distribution()[0].probability(), 0.5);
        assert_eq!(answer.value().as_choice(), Some("0"));
        assert_eq!(answer.confidence(), Some(0.5));
    }

    #[test]
    fn score_exact_half_uses_first_crossing_and_survives_serialization() {
        let answer = finalize(&prepared(AnswerKind::Score, &[1, 1]), &[0.0, 0.0]).unwrap();
        assert_eq!(
            answer.value().as_score(),
            Some((0.0, ScoreStatistic::Median))
        );
        let encoded = serde_json::to_string(&answer).unwrap();
        let decoded: DecisionAnswer = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, answer);
    }

    #[test]
    fn noul_has_no_confidence_and_caller_origins() {
        let answer = finalize(&prepared(AnswerKind::Noul, &[1, 1]), &[0.0, 0.0]).unwrap();
        assert_eq!(answer.value().as_yes_no(), Some(0.5));
        assert_eq!(answer.confidence(), None);
        assert!(answer
            .distribution()
            .iter()
            .all(|s| s.origin() == DecisionOrigin::Caller));
    }

    #[test]
    fn debug_never_exposes_content_or_selected_logits() {
        let mut pass = prepared(AnswerKind::Choice, &[1]);
        pass.plan.question = "PRIVATE-STATE".into();
        pass.plan.groups[0].key = "PRIVATE-OPTION".into();
        assert!(!format!("{pass:?}").contains("PRIVATE"));
    }

    #[test]
    fn equal_aliases_retain_first_signed_zero_like_upstream_max() {
        let pass = prepared(AnswerKind::Choice, &[2]);
        let answer = finalize(&pass, &[-0.0, 0.0]).unwrap();
        assert_eq!(
            answer.distribution()[0].raw_score().to_bits(),
            (-0.0_f32).to_bits()
        );
    }
}

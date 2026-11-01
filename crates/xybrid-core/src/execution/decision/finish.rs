//! Answers from readouts: LFM3 d1's arithmetic, in llama.cpp's operation
//! order (`format_answer`, `tools/server/server-decision.cpp:1651-1747`).

use super::plan::{AnswerPlan, FinalizeRule, RequestDigest};
use super::protocol::option_count;
use super::DecisionPlan;
use crate::ir::decision::{
    DecisionAnswer, DecisionError, DecisionRequest, DecisionScore, DecisionValue, Decisions,
    ScoreStatistic,
};

pub(super) fn finish(
    request: &DecisionRequest,
    plan: &DecisionPlan,
    readouts: Vec<Vec<f32>>,
) -> Result<Decisions, DecisionError> {
    assert!(
        plan.request == RequestDigest::of(request),
        "finish was given a plan prepared for another request"
    );
    if readouts.len() != plan.passes.len() {
        return Err(DecisionError::ReadoutCount {
            expected: plan.passes.len(),
            found: readouts.len(),
        });
    }
    for (pass, (entry, readout)) in plan.passes.iter().zip(&readouts).enumerate() {
        if readout.len() != entry.select.len() {
            return Err(DecisionError::ReadoutLength {
                pass,
                expected: entry.select.len(),
                found: readout.len(),
            });
        }
    }
    assert_eq!(
        plan.answers.len(),
        request.questions().len(),
        "finish was given a plan prepared for another request"
    );

    let mut answers = Vec::with_capacity(plan.answers.len());
    for answer in &plan.answers {
        let question = &request.questions()[answer.question];
        assert_eq!(
            answer.groups.len(),
            option_count(question.kind()),
            "finish was given a plan prepared for another request"
        );
        let readout = &readouts[answer.pass];
        answers.push(read_answer(question.id(), answer, readout)?);
    }
    Ok(Decisions::new(answers, plan.usage()))
}

/// The answer of one question from its pass's readout.
fn read_answer(
    question_id: &str,
    answer: &AnswerPlan,
    readout: &[f32],
) -> Result<DecisionAnswer, DecisionError> {
    let refuse = || DecisionError::NonFiniteScores {
        question: answer.question,
    };
    // Stricter than upstream, which only checks the group maxima and so
    // answers by label order when an alias is NaN: any NaN refuses. A +inf
    // is its group's maximum, which upstream refuses too
    // (`server-decision.cpp:1671-1673`).
    if readout
        .iter()
        .any(|&logit| logit.is_nan() || logit == f32::INFINITY)
    {
        return Err(refuse());
    }
    // Each option scores the highest logit of its group, the first of equal
    // ones (`std::max_element`, `server-context.cpp:2537-2545`).
    let scores: Vec<f32> = answer
        .groups
        .iter()
        .map(|group| {
            readout[group.select.clone()]
                .iter()
                .copied()
                .reduce(|best, logit| if best < logit { logit } else { best })
                .expect("prepare gives every group at least one token")
        })
        .collect();
    // All -inf leaves no distribution (upstream would divide 0 by 0).
    if scores.iter().all(|&score| score == f32::NEG_INFINITY) {
        return Err(refuse());
    }

    let probabilities = softmax(&scores, answer.temperature.value);
    let distribution = answer
        .groups
        .iter()
        .zip(&scores)
        .zip(&probabilities)
        .map(|((group, &score), &probability)| {
            DecisionScore::new(group.key.clone(), score, probability, group.origin)
        })
        .collect();
    // The first maximum, as `std::max_element` finds it.
    let (best, &max_probability) = probabilities
        .iter()
        .enumerate()
        .reduce(|best, entry| if best.1 < entry.1 { entry } else { best })
        .expect("a question has at least one option");
    let (value, confidence) = match answer.rule {
        FinalizeRule::Choice => (
            DecisionValue::choice(answer.groups[best].key.clone()),
            Some(max_probability),
        ),
        // Protocol order puts "true" first (`server-decision.cpp:1704-1709`).
        FinalizeRule::YesNo => (DecisionValue::yes_no(probabilities[0]), None),
        FinalizeRule::Score => (
            DecisionValue::score(median(&probabilities) as f64, ScoreStatistic::Median),
            Some(max_probability),
        ),
    };
    DecisionAnswer::new(question_id, distribution, value, confidence)
}

/// The distribution over `scores` at `temperature`, as upstream computes it
/// (`server-decision.cpp:1674-1691`): each score widened to `f64`, the
/// maximum subtracted *before* dividing by the temperature (an `f32`,
/// widened), `exp` of each in option order, then each divided by their sum,
/// added in option order.
///
/// Upstream also divides by the number of variants and adds into a zeroed
/// accumulator; with LFM3 d1's single variant that is `/ 1.0` and `0.0 +`,
/// which change no bit.
fn softmax(scores: &[f32], temperature: f32) -> Vec<f64> {
    let temperature = f64::from(temperature);
    let logits: Vec<f64> = scores.iter().copied().map(f64::from).collect();
    let max = logits
        .iter()
        .copied()
        .reduce(|best, logit| if best < logit { logit } else { best })
        .expect("a question has at least one option");
    let weights: Vec<f64> = logits
        .iter()
        .map(|&logit| ((logit - max) / temperature).exp())
        .collect();
    let sum = weights.iter().fold(0.0, |sum, weight| sum + weight);
    weights.iter().map(|weight| weight / sum).collect()
}

/// The first level whose cumulative probability, summed in level order,
/// reaches one half; the last level if none does (LFM3 d1's median,
/// `server-decision.cpp:1732-1740`).
fn median(probabilities: &[f64]) -> usize {
    let mut mass = 0.0;
    let mut level = probabilities.len() - 1;
    for (index, &probability) in probabilities.iter().enumerate() {
        if mass >= 0.5 {
            break;
        }
        mass += probability;
        level = index;
    }
    level
}

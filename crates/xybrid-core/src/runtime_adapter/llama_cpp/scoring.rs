//! Label scoring for the llama.cpp backend: a [`ScoringRequest`] turned into
//! prompt and label token ids, read through
//! [`xybrid_llama::LlamaContext::score_final_position`], and reduced to one
//! logit per label group.
//!
//! Errors name group indexes and counts only: labels and messages never
//! reach a message.

use super::chat;
use crate::execution::choice::math;
use crate::runtime_adapter::llm::{LabelGroup, LabelLogits, LlmResult, ScoringRequest};
use crate::runtime_adapter::AdapterError;
use std::collections::HashMap;

/// A [`ScoringRequest`] in token ids.
pub(super) struct ScoringTokens {
    /// The rendered prompt, tokenized as generation tokenizes a chat prompt.
    pub(super) prompt: Vec<i32>,
    /// Every label token, group after group, each group's label first.
    pub(super) labels: Vec<i32>,
    /// How many of `labels` belong to each group, in group order.
    pub(super) group_sizes: Vec<usize>,
}

/// Resolves the request's label groups, then renders and tokenizes its
/// prompt. Touches neither the context nor its memory.
pub(super) fn scoring_tokens(
    model: &xybrid_llama::LlamaModel,
    request: &ScoringRequest,
) -> LlmResult<ScoringTokens> {
    let (labels, group_sizes) = label_tokens(model, &request.label_groups)?;
    let prompt = chat::render_scoring_prompt(model, &request.messages, request.enable_thinking)?;
    // Special-token parsing on, so the template's markers become their ids,
    // and nothing added: the template already holds every special token, as
    // Hugging Face's `apply_chat_template` tokenizes.
    let prompt = model.tokenize_special(&prompt, false)?;
    Ok(ScoringTokens {
        prompt,
        labels,
        group_sizes,
    })
}

/// Each group's token ids, its label first and then each alternative that
/// is one token and new to the group, as llama.cpp's decision server builds
/// LiquidAI d1's groups (`d1_labels` in `server-decision.cpp`). A token
/// that two groups share would give both options the same evidence, so it
/// is refused.
fn label_tokens(
    model: &xybrid_llama::LlamaModel,
    groups: &[LabelGroup],
) -> LlmResult<(Vec<i32>, Vec<usize>)> {
    if groups.is_empty() {
        return Err(AdapterError::InvalidInput(
            "no label groups to score".to_string(),
        ));
    }
    let mut labels = Vec::new();
    let mut group_sizes = Vec::with_capacity(groups.len());
    let mut owners: HashMap<i32, usize> = HashMap::new();
    for (index, group) in groups.iter().enumerate() {
        let label = single_token(model, &group.label)?.ok_or_else(|| {
            AdapterError::InvalidInput(format!(
                "the label of label group {index} is not a single token"
            ))
        })?;
        let mut members = vec![label];
        for alternative in &group.alternatives {
            if let Some(token) = single_token(model, alternative)? {
                if !members.contains(&token) {
                    members.push(token);
                }
            }
        }
        for token in &members {
            if let Some(first) = owners.insert(*token, index) {
                return Err(AdapterError::InvalidInput(format!(
                    "label groups {first} and {index} share a token"
                )));
            }
        }
        group_sizes.push(members.len());
        labels.extend(members);
    }
    Ok((labels, group_sizes))
}

/// The token `text` is, when it is exactly one. Plain text: a special-token
/// string is not read as that special token.
fn single_token(model: &xybrid_llama::LlamaModel, text: &str) -> LlmResult<Option<i32>> {
    Ok(match model.tokenize(text, false)?.as_slice() {
        [token] => Some(*token),
        _ => None,
    })
}

/// One logit per group, the largest among its tokens', and the mass the
/// row puts on every label token. `label_logits` and `logsumexp` are what
/// [`xybrid_llama::FinalLogits`] read for the labels of [`ScoringTokens`].
pub(super) fn reduce(
    label_logits: &[f32],
    logsumexp: f32,
    group_sizes: &[usize],
) -> LlmResult<LabelLogits> {
    let logits = group_maxima(label_logits, group_sizes);
    // Groups share no token, so each label token is counted once.
    let label_mass = math::label_mass(label_logits, logsumexp).ok_or_else(|| {
        AdapterError::InferenceFailed("the label logits have no finite label mass".to_string())
    })?;
    Ok(LabelLogits::new(logits, label_mass))
}

/// The largest of each consecutive run of `group_sizes` logits.
///
/// # Panics
///
/// When the sizes do not add up to `logits.len()` or a size is zero: the
/// sizes come from [`label_tokens`], which never produces either.
fn group_maxima(logits: &[f32], group_sizes: &[usize]) -> Vec<f32> {
    assert_eq!(
        group_sizes.iter().sum::<usize>(),
        logits.len(),
        "label group sizes cover every label logit"
    );
    let mut rest = logits;
    group_sizes
        .iter()
        .map(|&size| {
            assert!(size > 0, "a label group has at least its label");
            let (group, tail) = rest.split_at(size);
            rest = tail;
            group.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_maxima_take_the_largest_of_each_run() {
        let logits = [1.0, 4.0, -2.0, 3.0, 3.5, 0.0];

        assert_eq!(group_maxima(&logits, &[2, 1, 3]), [4.0, -2.0, 3.5]);
        assert_eq!(group_maxima(&logits, &[1; 6]), logits);
    }

    #[test]
    #[should_panic(expected = "cover every label logit")]
    fn group_maxima_need_sizes_covering_every_logit() {
        group_maxima(&[1.0, 2.0], &[1]);
    }

    #[test]
    fn reduce_counts_every_label_token_in_the_mass() {
        let reduced = reduce(&[0.0, 1.0, 2.0], 3.0, &[2, 1]).expect("finite logits reduce");

        assert_eq!(reduced.logits, [1.0, 2.0]);
        let expected = [0.0_f64, 1.0, 2.0]
            .iter()
            .map(|logit| (logit - 3.0).exp())
            .sum::<f64>();
        assert!((f64::from(reduced.label_mass) - expected).abs() < 1e-6);
    }
}

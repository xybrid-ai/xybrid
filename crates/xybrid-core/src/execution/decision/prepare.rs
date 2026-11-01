//! `prepare`: a whole request, checked, rendered, tokenized and labelled,
//! before anything is evaluated (the work `fill_task` does per question,
//! `tools/server/server-decision.cpp:1190-1267`, for every question first).

use std::sync::Arc;

use super::escape::{Escaped, Spellings};
use super::labels::{self, Label, LabelTokens};
use super::plan::{AnswerPlan, FinalizeRule, LabelGroup, RequestDigest};
use super::render::{RenderError, Renderer};
use super::{protocol, DecisionModelFacts, DecisionPass, DecisionPlan, DecisionVocab, Token};
use crate::ir::decision::{
    ContentPart, DecisionContent, DecisionError, DecisionInstructions, DecisionOrigin,
    DecisionQuestion, DecisionQuestionKind, DecisionRequest,
};

/// [`super::prepare`]; with `prompts`, each rendered prompt is also kept
/// there, in pass order (for the conformance tests).
pub(super) fn prepare(
    request: &DecisionRequest,
    facts: &DecisionModelFacts,
    vocab: &dyn DecisionVocab,
    mut prompts: Option<&mut Vec<String>>,
) -> Result<DecisionPlan, DecisionError> {
    let limits = facts.limits();
    protocol::check(request, limits)?;
    let spellings = Spellings::new(vocab.control_spellings())?;
    let state = Arc::new(Escaped::content(
        &spellings,
        request.state().body(),
        ContentPart::State,
    )?);
    // No prompt that fits the budget is longer than this: every token covers
    // at most `max_token_bytes` bytes of it (`DecisionVocab::max_token_bytes`).
    let max_bytes = (limits.token_budget as usize).saturating_mul(vocab.max_token_bytes().max(1));
    let renderer = Renderer::new(max_bytes).map_err(|_| DecisionError::RenderFailed)?;
    let mut label_tokens = LabelTokens::new(vocab);

    let count = request.questions().len();
    let mut passes = Vec::with_capacity(count);
    let mut answers = Vec::with_capacity(count);
    let mut total: u64 = 0;
    for (index, question) in request.questions().iter().enumerate() {
        let QuestionLabels {
            rule,
            type_name,
            keys,
            labels,
        } = labels_of(index, question, &mut label_tokens)?;
        let input = template_input(index, question, type_name, &labels, &state, &spellings)?;
        let prompt = renderer
            .render(Arc::new(input))
            .map_err(|error| match error {
                RenderError::TooLong => DecisionError::RenderedTooLong {
                    question: index,
                    max_bytes,
                },
                RenderError::Failed => DecisionError::RenderFailed,
            })?;

        // The rendered text spells its special tokens; the caller's text no
        // longer spells CONTROL ones (`server-decision.cpp:1250`).
        let tokens = vocab.tokenize(&prompt, false, true)?;
        if tokens.is_empty() {
            return Err(DecisionError::TokenizationFailed);
        }
        if tokens.len() > limits.token_budget as usize {
            return Err(DecisionError::BudgetExceeded {
                question: index,
                tokens: tokens.len(),
                budget: limits.token_budget,
            });
        }
        total += tokens.len() as u64;
        if total > limits.max_total_tokens {
            return Err(DecisionError::TotalBudgetExceeded {
                tokens: total,
                max: limits.max_total_tokens,
            });
        }

        let mut select: Vec<Token> = Vec::new();
        let groups = keys
            .into_iter()
            .zip(labels)
            .map(|(key, label)| {
                let start = select.len();
                select.extend_from_slice(&label.group);
                LabelGroup {
                    key,
                    code: label.code,
                    select: start..select.len(),
                    origin: DecisionOrigin::Caller,
                }
            })
            .collect::<Vec<_>>();
        let temperature = facts.temperatures.lookup(type_name, groups.len());
        answers.push(AnswerPlan {
            question: index,
            pass: passes.len(),
            groups,
            rule,
            temperature,
        });
        passes.push(DecisionPass { tokens, select });
        if let Some(prompts) = prompts.as_deref_mut() {
            prompts.push(prompt);
        }
    }
    Ok(DecisionPlan {
        passes,
        answers,
        request: RequestDigest::of(request),
    })
}

/// What a question's kind decides: its rule, its type as the template names
/// it, its option keys in protocol order, and their labels.
struct QuestionLabels {
    rule: FinalizeRule,
    type_name: &'static str,
    keys: Vec<String>,
    labels: Vec<Label>,
}

fn labels_of(
    index: usize,
    question: &DecisionQuestion,
    tokens: &mut LabelTokens<'_>,
) -> Result<QuestionLabels, DecisionError> {
    Ok(match question.kind() {
        DecisionQuestionKind::Choice { options } => {
            let keys: Vec<&str> = options.iter().map(|option| option.key()).collect();
            QuestionLabels {
                rule: FinalizeRule::Choice,
                type_name: "choice",
                labels: labels::choice(index, &keys, tokens)?,
                keys: keys.into_iter().map(str::to_string).collect(),
            }
        }
        DecisionQuestionKind::YesNo { .. } => QuestionLabels {
            rule: FinalizeRule::YesNo,
            type_name: "noul",
            labels: labels::yes_no(index, tokens)?.to_vec(),
            keys: vec!["true".to_string(), "false".to_string()],
        },
        DecisionQuestionKind::Score { levels } => QuestionLabels {
            rule: FinalizeRule::Score,
            type_name: "score",
            labels: labels::score(index, levels.len(), tokens)?,
            keys: (0..levels.len()).map(|level| level.to_string()).collect(),
        },
    })
}

/// The template input of one question, escaped as upstream escapes it
/// (`render`, `server-decision.cpp:1101-1188`): `id`, `type`,
/// `instructions`, `state` and `options` (each `{key, description,
/// label}`), every string and key through `d1_break`; then `images`, empty,
/// added after the escaping.
fn template_input(
    index: usize,
    question: &DecisionQuestion,
    type_name: &str,
    labels: &[Label],
    state: &Arc<Escaped>,
    spellings: &Spellings,
) -> Result<Escaped, DecisionError> {
    let instructions = match question.instructions() {
        Some(DecisionInstructions::Content(content)) => content,
        // `protocol::check` refused it.
        None => return Err(DecisionError::MissingInstructions { question: index }),
    };
    let instructions = Escaped::content(
        spellings,
        Some(instructions),
        ContentPart::Instructions { question: index },
    )?;

    // Each option: its key as the caller wrote it, its description (null
    // for none) and its label text, in protocol order.
    let described: Vec<(&str, Option<&DecisionContent>, ContentPart)> = match question.kind() {
        DecisionQuestionKind::Choice { options } => options
            .iter()
            .enumerate()
            .map(|(option, entry)| {
                (
                    entry.key(),
                    entry.description(),
                    ContentPart::OptionDescription {
                        question: index,
                        option,
                    },
                )
            })
            .collect(),
        // True first (`noul_true_first`, `server-decision.cpp:167`, `:349-351`).
        DecisionQuestionKind::YesNo { yes, no } => vec![
            (
                "true",
                yes.as_ref(),
                ContentPart::YesNoDescription {
                    question: index,
                    yes: true,
                },
            ),
            (
                "false",
                no.as_ref(),
                ContentPart::YesNoDescription {
                    question: index,
                    yes: false,
                },
            ),
        ],
        DecisionQuestionKind::Score { levels } => levels
            .iter()
            .enumerate()
            .map(|(level, entry)| {
                (
                    // `labels::score` codes each level by its number.
                    labels[level].code.as_str(),
                    entry.description(),
                    ContentPart::LevelDescription {
                        question: index,
                        level,
                    },
                )
            })
            .collect(),
    };
    let options = described
        .into_iter()
        .zip(labels)
        .map(|((key, description, part), label)| {
            Ok(Arc::new(Escaped::structure(
                spellings,
                vec![
                    ("key", Arc::new(Escaped::text(spellings, key)?)),
                    (
                        "description",
                        Arc::new(Escaped::content(spellings, description, part)?),
                    ),
                    ("label", Arc::new(Escaped::text(spellings, &label.code)?)),
                ],
            )?))
        })
        .collect::<Result<Vec<_>, DecisionError>>()?;

    let mut input = Escaped::structure(
        spellings,
        vec![
            ("id", Arc::new(Escaped::text(spellings, question.id())?)),
            ("type", Arc::new(Escaped::text(spellings, type_name)?)),
            ("instructions", Arc::new(instructions)),
            ("state", Arc::clone(state)),
            ("options", Arc::new(Escaped::Array(options))),
        ],
    )?;
    if let Escaped::Object(members) = &mut input {
        members.push(("images".to_string(), Arc::new(Escaped::Array(Vec::new()))));
    }
    Ok(input)
}

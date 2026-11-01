//! What LFM3 d1 reads: the checks `prepare` runs on a request before it
//! renders anything.

use super::DecisionLimits;
use crate::ir::decision::{
    validate_structure, ContentPart, DecisionContent, DecisionError, DecisionInstructions,
    DecisionQuestionKind, DecisionRequest, JsonNumber, JsonView, OrderedJson, StructuralLimits,
};

/// The most options of a choice question LFM3 d1 reads
/// (`n_options_max`, `tools/server/server-decision.cpp:166`).
pub(super) const MAX_OPTIONS: usize = 255;

/// The levels of a score question LFM3 d1 reads
/// (`server-decision.cpp:317-319`).
pub(super) const MIN_LEVELS: usize = 2;
pub(super) const MAX_LEVELS: usize = 10;

/// The deepest JSON LFM3 d1 reads in the state and in instructions, and in
/// option, yes/no and level descriptions, counting a value itself as
/// [`OrderedJson::depth`] does.
///
/// llama.cpp escapes the whole template input with a recursion limit of
/// 128 (`RECURSIVE_LIMIT`, `server-decision.cpp:27`, checked in
/// `decision_escape`, `:606-628`) starting from the input object: the state
/// and the instructions sit one level below it, descriptions three (input,
/// `options`, option).
pub(super) const MAX_STATE_DEPTH: usize = 127;
pub(super) const MAX_INSTRUCTIONS_DEPTH: usize = 127;
pub(super) const MAX_DESCRIPTION_DEPTH: usize = 125;

/// Checks `request` against what LFM3 d1 reads under `limits`, in order:
/// structure ([`validate_structure`], with the limits' question and option
/// ceilings), media, then per question instructions, option and level
/// counts, and the depth and integers of every piece of JSON content.
pub(super) fn check(
    request: &DecisionRequest,
    limits: &DecisionLimits,
) -> Result<(), DecisionError> {
    let structural = StructuralLimits::default()
        .with_max_questions(limits.max_questions)
        .with_max_options(limits.max_options);
    validate_structure(request, &structural)?;
    if u32::try_from(request.questions().len()).is_err() {
        return Err(DecisionError::TooManyQuestions {
            count: request.questions().len(),
            max: u32::MAX as usize,
        });
    }
    if !request.state().media().is_empty() {
        return Err(DecisionError::MediaUnsupported);
    }
    check_content(request.state().body(), ContentPart::State, MAX_STATE_DEPTH)?;

    for (question, entry) in request.questions().iter().enumerate() {
        let instructions = match entry.instructions() {
            Some(DecisionInstructions::Content(content)) => content,
            None => return Err(DecisionError::MissingInstructions { question }),
        };
        check_content(
            Some(instructions),
            ContentPart::Instructions { question },
            MAX_INSTRUCTIONS_DEPTH,
        )?;
        match entry.kind() {
            DecisionQuestionKind::Choice { options } => {
                if options.len() > MAX_OPTIONS {
                    return Err(DecisionError::TooManyOptions {
                        question,
                        count: options.len(),
                        max: MAX_OPTIONS,
                    });
                }
                for (option, entry) in options.iter().enumerate() {
                    check_content(
                        entry.description(),
                        ContentPart::OptionDescription { question, option },
                        MAX_DESCRIPTION_DEPTH,
                    )?;
                }
            }
            DecisionQuestionKind::YesNo { yes, no } => {
                for (description, yes) in [(yes, true), (no, false)] {
                    check_content(
                        description.as_ref(),
                        ContentPart::YesNoDescription { question, yes },
                        MAX_DESCRIPTION_DEPTH,
                    )?;
                }
            }
            DecisionQuestionKind::Score { levels } => {
                if !(MIN_LEVELS..=MAX_LEVELS).contains(&levels.len()) {
                    return Err(DecisionError::LevelCount {
                        question,
                        count: levels.len(),
                        min: MIN_LEVELS,
                        max: MAX_LEVELS,
                    });
                }
                for (level, entry) in levels.iter().enumerate() {
                    check_content(
                        entry.description(),
                        ContentPart::LevelDescription { question, level },
                        MAX_DESCRIPTION_DEPTH,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// JSON content no deeper than `max_depth`, holding no integer above
/// `i64::MAX` (upstream's engine reads every integer as an `int64_t`, so it
/// would print `18446744073709551615` as `-1`). Text has neither problem.
fn check_content(
    content: Option<&DecisionContent>,
    part: ContentPart,
    max_depth: usize,
) -> Result<(), DecisionError> {
    let Some(value) = content.and_then(DecisionContent::as_json) else {
        return Ok(());
    };
    let depth = value.depth();
    if depth > max_depth {
        return Err(DecisionError::ContentTooDeep {
            part,
            depth,
            max: max_depth,
        });
    }
    if holds_u64(value) {
        return Err(DecisionError::NumberOutOfRange { part });
    }
    Ok(())
}

/// Whether `value` holds a [`JsonNumber::U64`]: an integer in
/// (`i64::MAX`, `u64::MAX`]. Larger integer spellings are already floats.
fn holds_u64(value: &OrderedJson) -> bool {
    match value.view() {
        JsonView::Number(JsonNumber::U64(_)) => true,
        JsonView::Array(items) => items.iter().any(holds_u64),
        JsonView::Object(members) => members.iter().any(|(_, item)| holds_u64(item)),
        _ => false,
    }
}

/// The options a question's answer has: its options, 2 for yes/no, its
/// levels.
pub(super) fn option_count(kind: &DecisionQuestionKind) -> usize {
    match kind {
        DecisionQuestionKind::Choice { options } => options.len(),
        DecisionQuestionKind::YesNo { .. } => 2,
        DecisionQuestionKind::Score { levels } => levels.len(),
    }
}

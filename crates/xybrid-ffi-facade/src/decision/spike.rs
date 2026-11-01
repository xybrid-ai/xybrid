//! Round-trip hooks shared by the Bolt and Flutter spike exports (P4 only).
//!
//! A foreign test sends a request, then compares what it decoded against
//! what Rust reports here, so a codec bug that cancels out in an echo still
//! shows. These return caller content on purpose (to the caller that sent
//! it); nothing here is a log or an error message.

use super::stand_in::{self as ir, DecisionContentView};
use super::{first_bit_difference, DecisionValue, Decisions};
use crate::Result;

fn content(content: Option<&ir::DecisionContent>) -> String {
    match content.map(ir::DecisionContent::view) {
        None => "none".into(),
        Some(DecisionContentView::Text(text)) => format!("text:{text}"),
        Some(DecisionContentView::Json(json)) => format!("json:{}", json.to_json_text()),
    }
}

/// What the IR holds, one line per item in request order.
pub fn outline(request: &ir::DecisionRequest) -> Vec<String> {
    let mut lines = vec![format!("state={}", content(request.state().body()))];
    for (q, question) in request.questions().iter().enumerate() {
        let instructions = match question.instructions() {
            None => "none".to_string(),
            Some(ir::DecisionInstructions::Content(c)) => content(Some(c)),
        };
        let id = question.id();
        match question.kind() {
            ir::DecisionQuestionKind::Choice { options } => {
                lines.push(format!("q{q}:id={id}:choice:instructions={instructions}"));
                for (o, option) in options.iter().enumerate() {
                    lines.push(format!(
                        "q{q}.o{o}:key={}:{}",
                        option.key(),
                        content(option.description())
                    ));
                }
            }
            ir::DecisionQuestionKind::YesNo { yes, no } => {
                lines.push(format!("q{q}:id={id}:yes_no:instructions={instructions}"));
                lines.push(format!("q{q}.yes:{}", content(yes.as_ref())));
                lines.push(format!("q{q}.no:{}", content(no.as_ref())));
            }
            ir::DecisionQuestionKind::Score { levels } => {
                lines.push(format!("q{q}:id={id}:score:instructions={instructions}"));
                for (l, level) in levels.iter().enumerate() {
                    lines.push(format!("q{q}.l{l}:{}", content(level.as_ref())));
                }
            }
        }
    }
    lines
}

/// The stub engine's result for `request`.
pub fn decide(request: &ir::DecisionRequest) -> Decisions {
    Decisions::from_ir(&ir::stub_decide(request))
}

/// Every float of a result as hex bit patterns: `a<i>.s<j>.raw=<hex8>`,
/// `a<i>.s<j>.p=<hex16>`, then the value and the confidence per answer.
pub fn bits(decisions: &Decisions) -> Vec<String> {
    let mut lines = Vec::new();
    for (a, answer) in decisions.answers.iter().enumerate() {
        for (s, score) in answer.distribution.iter().enumerate() {
            lines.push(format!("a{a}.s{s}.raw={:08x}", score.raw_score.to_bits()));
            lines.push(format!("a{a}.s{s}.p={:016x}", score.probability.to_bits()));
        }
        lines.push(match &answer.value {
            DecisionValue::Choice { key } => format!("a{a}.value=choice:{key}"),
            DecisionValue::YesNo { p_yes } => format!("a{a}.value=yes_no:{:016x}", p_yes.to_bits()),
            DecisionValue::Score { value, statistic } => {
                format!("a{a}.value=score:{:016x}:{statistic:?}", value.to_bits())
            }
        });
        lines.push(match answer.confidence {
            Some(c) => format!("a{a}.confidence={:016x}", c.to_bits()),
            None => format!("a{a}.confidence=none"),
        });
    }
    lines
}

/// Compare a result a foreign side re-encoded with the stub engine's own,
/// bit for bit, then rebuild it through the invariant-checking constructor.
/// `None` means identical; otherwise the path of the first difference.
///
/// # Errors
/// The received result breaks an answer invariant.
pub fn verify(request: &ir::DecisionRequest, received: Decisions) -> Result<Option<String>> {
    if let Some(path) = first_bit_difference(&decide(request), &received) {
        return Ok(Some(path));
    }
    received.into_ir()?;
    Ok(None)
}

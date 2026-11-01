//! How a question's answer is read from its pass, and which request a plan
//! was prepared for.

use std::fmt;
use std::ops::Range;

use sha2::{Digest, Sha256};

use super::facts::Temperature;
use crate::ir::decision::{DecisionOrigin, DecisionRequest};

/// Which request a plan was prepared for: the SHA-256 of the request's
/// bincode serialization, which covers every question id, option key and
/// piece of content. Equal requests (a clone, a decoded copy) share one.
///
/// No `Debug`: a digest of the caller's content is not content-free.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct RequestDigest([u8; 32]);

impl RequestDigest {
    pub(super) fn of(request: &DecisionRequest) -> Self {
        let mut hasher = Sha256::new();
        bincode::serialize_into(&mut hasher, request)
            .expect("bincode serializes every decision request, and a hasher takes any bytes");
        Self(hasher.finalize().into())
    }
}

/// How the answer to one question is read.
#[derive(Clone)]
pub(super) struct AnswerPlan {
    /// Position of the question in the request.
    pub(super) question: usize,
    /// Position of the pass it reads.
    pub(super) pass: usize,
    /// One group per option, in protocol order: option order for a choice,
    /// level order for a score, true then false for yes/no.
    pub(super) groups: Vec<LabelGroup>,
    /// What the distribution becomes.
    pub(super) rule: FinalizeRule,
    /// The temperature of the distribution.
    pub(super) temperature: Temperature,
}

/// The label tokens of one option, as a range of its pass's `select`.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct LabelGroup {
    /// The option's key as requested (not escaped): an option key, a level
    /// (`"0"`, `"1"`, …), or `"true"` / `"false"`.
    pub(super) key: String,
    /// The label text: the code shown in the prompt, or the key for yes/no
    /// and score.
    pub(super) code: String,
    /// Where its tokens are in the pass's `select`.
    pub(super) select: Range<usize>,
    /// Where the option came from.
    pub(super) origin: DecisionOrigin,
}

/// What a question's distribution becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FinalizeRule {
    /// The first most probable option; confidence is its probability.
    Choice,
    /// The probability of `"true"`; no confidence.
    YesNo,
    /// The median level; confidence is the highest probability.
    Score,
}

/// Counts only: keys and codes are the caller's.
impl fmt::Debug for AnswerPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnswerPlan")
            .field("question", &self.question)
            .field("pass", &self.pass)
            .field("groups", &self.groups.len())
            .field("rule", &self.rule)
            .field("temperature", &self.temperature.value)
            .finish()
    }
}

/// Lengths only: keys and codes are the caller's.
impl fmt::Debug for LabelGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LabelGroup")
            .field("key_bytes", &self.key.len())
            .field("code_bytes", &self.code.len())
            .field("select", &self.select)
            .field("origin", &self.origin)
            .finish()
    }
}

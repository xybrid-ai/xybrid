//! Choice-scoring primitives: candidate rules, input encoding, score math and
//! the ONNX byte scorer.
//!
//! A choice scorer reads one context and a closed list of candidates and
//! returns one score per candidate. This module holds the pieces every scorer
//! shares, so each runtime only adds its readout:
//!
//! | Module | Contents |
//! |--------|----------|
//! | (this) | [`ChoiceError`], [`effective_candidates`], [`validate_scorer_spec`] |
//! | [`encode`] | UTF-8 byte-id rows and masks for byte scorers |
//! | [`math`] | Softmax with temperature, tie and non-finite rules |
//! | [`onnx`] | Binding a byte-scorer spec to an ONNX session and running it |
//!
//! Nothing here is reachable through [`crate::ir::EnvelopeKind`] or
//! [`crate::execution::ExecutionTemplate`] yet: these are the primitives the
//! executor integration builds on.
//!
//! # Candidates
//!
//! The candidates E are the caller's choices followed by the scorer's fixed
//! choices. Ids must be non-empty and unique across E, so a caller may not
//! reuse a fixed id, and `2 <= |E| <= max_choices` must hold after the fixed
//! choices are appended. Scores come back in E order.
//!
//! # Errors carry no content
//!
//! Choice ids, choice texts and the context are caller content. Every
//! [`ChoiceError`] message names positions, counts and spec fields only.
//!
//! # Example
//!
//! ```
//! use xybrid_core::execution::choice::{effective_candidates, ChoiceError};
//! use xybrid_core::ir::Choice;
//!
//! let fixed = [Choice::new("skip", "skip")];
//! let caller = [Choice::new("tel", "fill Tel: (503) 555-0142")];
//! let e = effective_candidates(&caller, &fixed, 64)?;
//! assert_eq!(e.len(), 2);
//! assert!(e[1].fixed);
//!
//! let clash = [Choice::new("skip", "skip it")];
//! assert!(matches!(
//!     effective_candidates(&clash, &fixed, 64),
//!     Err(ChoiceError::ReservedChoiceId { index: 0, fixed_index: 0 })
//! ));
//! # Ok::<(), ChoiceError>(())
//! ```

pub mod encode;
pub mod math;
pub mod onnx;

#[cfg(test)]
mod tests;

use crate::execution::template::{
    ByteFieldSpec, ByteOverflow, ChoiceScorerSpec, OnnxByteOptionScorerSpec,
};
use crate::ir::Choice;
use crate::runtime_adapter::AdapterError;
use std::collections::HashMap;
use std::fmt;

/// Most ids one request may encode across the context row and every choice
/// row: 1 Mi ids, i.e. 8 MiB of int64 plus 1 MiB of masks. It bounds the
/// tensors a spec (or a static model axis) can make a request allocate;
/// CUA-S1-FORMS needs 224 + 64 × 96 = 6,368.
pub const MAX_ENCODED_IDS: usize = 1 << 20;

/// Result alias for choice-scoring operations.
pub type ChoiceResult<T> = Result<T, ChoiceError>;

/// A text field of a choice request, named by position only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceField {
    /// The context.
    Context,
    /// The text of candidate `index` in E order.
    Choice(usize),
}

impl fmt::Display for ChoiceField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context => f.write_str("the context"),
            Self::Choice(index) => write!(f, "candidate {index}"),
        }
    }
}

/// Why a choice request, spec or model could not be scored.
///
/// Messages hold positions, counts and spec field names, never choice ids,
/// choice texts or the context.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ChoiceError {
    /// A candidate has an empty id.
    #[error("candidate {index} has an empty id")]
    EmptyChoiceId {
        /// Position in E.
        index: usize,
    },
    /// Two candidates share an id.
    #[error("candidate {index} repeats the id of candidate {first}")]
    DuplicateChoiceId {
        /// Position in E of the repeat.
        index: usize,
        /// Position in E of the first use.
        first: usize,
    },
    /// A caller choice reuses the id of one of the scorer's fixed choices.
    #[error("caller choice {index} reuses the id of fixed choice {fixed_index}")]
    ReservedChoiceId {
        /// Position among the caller's choices.
        index: usize,
        /// Position among the fixed choices.
        fixed_index: usize,
    },
    /// Fewer than two candidates after the fixed choices were appended.
    #[error("{count} candidate(s) offered; a choice scorer needs at least 2")]
    TooFewCandidates {
        /// |E|.
        count: usize,
    },
    /// More candidates than the scorer accepts.
    #[error(
        "{count} candidates offered including fixed choices; this scorer accepts at most {max}"
    )]
    TooManyCandidates {
        /// |E|.
        count: usize,
        /// The limit.
        max: usize,
    },
    /// The context is empty. A byte scorer then masks every context position,
    /// which the exported models do not reproduce faithfully.
    #[error("the context is empty; a byte choice scorer needs at least one byte")]
    EmptyContext,
    /// A field is longer than its spec allows and the spec says `Reject`.
    #[error("{field} is {len} bytes; this scorer rejects more than {max_len}")]
    InputTooLong {
        /// Which field.
        field: ChoiceField,
        /// Its length in UTF-8 bytes.
        len: usize,
        /// The spec's `max_len`.
        max_len: usize,
    },
    /// The scorer spec itself is invalid.
    #[error("invalid choice scorer spec: {0}")]
    InvalidSpec(String),
    /// The model does not have the inputs or output the spec names.
    #[error("model does not match its choice scorer spec: {0}")]
    ModelContract(String),
    /// The model returned logits of an unexpected shape.
    #[error("output '{output}' has shape {actual:?}, expected {expected:?}")]
    OutputShape {
        /// Output name from the spec.
        output: String,
        /// The shape the request implies.
        expected: Vec<usize>,
        /// The shape the model returned.
        actual: Vec<usize>,
    },
    /// An offered candidate got a NaN or infinite logit.
    #[error("candidate {index} has a non-finite logit")]
    NonFiniteLogit {
        /// Position in E.
        index: usize,
    },
    /// No logits to normalise.
    #[error("no candidate logits to score")]
    NoCandidates,
    /// ONNX Runtime failed to run the model. Its message is withheld because
    /// it can quote input values, which are bytes of the request.
    #[error("the choice scorer failed to run (runtime message withheld)")]
    InferenceFailed,
    /// The runtime failed to run the model.
    #[error(transparent)]
    Runtime(#[from] AdapterError),
}

impl From<ChoiceError> for AdapterError {
    /// Request, spec and model-contract problems become
    /// [`AdapterError::InvalidInput`], a bad model output
    /// [`AdapterError::InferenceFailed`]; runtime errors pass through.
    /// Entry-point refusals (`UnsupportedModelCapability`) are raised by the
    /// callers that know the model id, not here.
    fn from(error: ChoiceError) -> Self {
        match error {
            ChoiceError::Runtime(inner) => inner,
            ChoiceError::OutputShape { .. }
            | ChoiceError::NonFiniteLogit { .. }
            | ChoiceError::NoCandidates
            | ChoiceError::InferenceFailed => Self::InferenceFailed(error.to_string()),
            _ => Self::InvalidInput(error.to_string()),
        }
    }
}

const CHOICE_ROUTE_HINT: &str =
    "send a ChoiceRequest only to a ChoiceScorer model, through a batch run";

/// What a runtime returns for a choice request or choice scores, so no
/// runtime turns one into text, tensors or a prompt.
pub(crate) fn choice_kind_not_runtime_input() -> AdapterError {
    AdapterError::InvalidInput(format!(
        "choice requests and choice scores are not runtime input; {CHOICE_ROUTE_HINT}"
    ))
}

/// One member of E: a candidate and whether it is a fixed choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate<'a> {
    /// The candidate.
    pub choice: &'a Choice,
    /// `true` for one of the scorer's fixed choices.
    pub fixed: bool,
}

/// Builds E, the caller's choices followed by the fixed choices, and checks the
/// candidate rules (see the [module docs](self)).
///
/// # Errors
///
/// [`ChoiceError::TooFewCandidates`] / [`ChoiceError::TooManyCandidates`] when
/// `|E|` is outside `2..=max_choices`; [`ChoiceError::EmptyChoiceId`],
/// [`ChoiceError::ReservedChoiceId`] or [`ChoiceError::DuplicateChoiceId`] for
/// the first offending id.
pub fn effective_candidates<'a>(
    caller: &'a [Choice],
    fixed: &'a [Choice],
    max_choices: usize,
) -> ChoiceResult<Vec<Candidate<'a>>> {
    let count = caller.len() + fixed.len();
    if count < 2 {
        return Err(ChoiceError::TooFewCandidates { count });
    }
    if count > max_choices {
        return Err(ChoiceError::TooManyCandidates {
            count,
            max: max_choices,
        });
    }

    let candidates: Vec<Candidate<'a>> = caller
        .iter()
        .map(|choice| Candidate {
            choice,
            fixed: false,
        })
        .chain(fixed.iter().map(|choice| Candidate {
            choice,
            fixed: true,
        }))
        .collect();

    let mut first_use: HashMap<&str, usize> = HashMap::with_capacity(count);
    for (index, candidate) in candidates.iter().enumerate() {
        let id = candidate.choice.id.as_str();
        if id.is_empty() {
            return Err(ChoiceError::EmptyChoiceId { index });
        }
        if let Some(&first) = first_use.get(id) {
            // Caller choices come first, so a caller id that a fixed choice
            // repeats was taken by the caller.
            return Err(if first < caller.len() && index >= caller.len() {
                ChoiceError::ReservedChoiceId {
                    index: first,
                    fixed_index: index - caller.len(),
                }
            } else {
                ChoiceError::DuplicateChoiceId { index, first }
            });
        }
        first_use.insert(id, index);
    }
    Ok(candidates)
}

/// Checks a scorer spec before any model is opened.
///
/// For [`ChoiceScorerSpec::OnnxByteOptionScorer`]: a non-empty model file;
/// five non-empty, distinct input names and a non-empty output name; byte
/// fields with `offset >= 1` and `max_len >= 1`; `max_choices >= 2`; at most
/// [`MAX_ENCODED_IDS`] ids per request; fixed choices with non-empty, unique
/// ids, no more of them than `max_choices` and none too long for a `Reject`
/// field; a finite, positive `scoring_temperature` when set.
///
/// # Errors
///
/// [`ChoiceError::InvalidSpec`] naming the first failing field.
pub fn validate_scorer_spec(spec: &ChoiceScorerSpec) -> ChoiceResult<()> {
    match spec {
        ChoiceScorerSpec::OnnxByteOptionScorer(onnx) => validate_onnx_spec(onnx),
    }
}

pub(crate) fn validate_onnx_spec(spec: &OnnxByteOptionScorerSpec) -> ChoiceResult<()> {
    let invalid = |reason: String| Err(ChoiceError::InvalidSpec(reason));

    if spec.model_file.is_empty() {
        return invalid("model_file is empty".into());
    }
    let names = [
        ("context.ids_input", &spec.context.ids_input),
        ("context.mask_input", &spec.context.mask_input),
        ("choice.ids_input", &spec.choice.ids_input),
        ("choice.mask_input", &spec.choice.mask_input),
        ("choice_mask_input", &spec.choice_mask_input),
    ];
    for (index, (field, name)) in names.iter().enumerate() {
        if name.is_empty() {
            return invalid(format!("{field} is empty"));
        }
        if let Some((other, _)) = names[..index].iter().find(|(_, earlier)| earlier == name) {
            return invalid(format!("{field} names the same input as {other}"));
        }
    }
    if spec.logits_output.is_empty() {
        return invalid("logits_output is empty".into());
    }

    validate_byte_field("context", &spec.context)?;
    validate_byte_field("choice", &spec.choice)?;

    if spec.max_choices < 2 {
        return invalid(format!("max_choices is {}; at least 2", spec.max_choices));
    }
    encoded_ids(spec, spec.max_choices).ok_or_else(|| {
        ChoiceError::InvalidSpec(format!(
            "a request could encode more than {MAX_ENCODED_IDS} ids"
        ))
    })?;

    if spec.fixed_choices.len() > spec.max_choices {
        return invalid(format!(
            "{} fixed choices exceed max_choices {}",
            spec.fixed_choices.len(),
            spec.max_choices
        ));
    }
    for (index, choice) in spec.fixed_choices.iter().enumerate() {
        if choice.id.is_empty() {
            return invalid(format!("fixed choice {index} has an empty id"));
        }
        if let Some(first) = spec.fixed_choices[..index]
            .iter()
            .position(|earlier| earlier.id == choice.id)
        {
            return invalid(format!("fixed choice {index} repeats the id of {first}"));
        }
        if spec.choice.overflow == ByteOverflow::Reject && choice.text.len() > spec.choice.max_len {
            return invalid(format!(
                "fixed choice {index} is {} bytes; choice.max_len is {} with Reject",
                choice.text.len(),
                spec.choice.max_len
            ));
        }
    }

    if let Some(temperature) = spec.scoring_temperature {
        if !temperature.is_finite() || temperature <= 0.0 {
            return invalid("scoring_temperature must be finite and greater than 0".into());
        }
    }
    Ok(())
}

fn validate_byte_field(name: &str, field: &ByteFieldSpec) -> ChoiceResult<()> {
    if field.offset == 0 {
        return Err(ChoiceError::InvalidSpec(format!(
            "{name}.offset is 0; byte ids must stay clear of the padding id 0"
        )));
    }
    if field.max_len == 0 {
        return Err(ChoiceError::InvalidSpec(format!("{name}.max_len is 0")));
    }
    Ok(())
}

/// Ids one request encodes with `rows` choice rows, or `None` above
/// [`MAX_ENCODED_IDS`] (or on overflow).
pub(crate) fn encoded_ids(spec: &OnnxByteOptionScorerSpec, rows: usize) -> Option<usize> {
    rows.checked_mul(spec.choice.max_len)?
        .checked_add(spec.context.max_len)
        .filter(|&ids| ids <= MAX_ENCODED_IDS)
}

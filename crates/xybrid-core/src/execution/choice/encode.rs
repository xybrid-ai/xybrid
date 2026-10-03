//! UTF-8 byte-id encoding for byte choice scorers.
//!
//! Each field becomes a fixed-length row: the text's raw UTF-8 bytes, cut at
//! the field's `max_len` (a multi-byte character may be split), each shifted
//! by the field's `offset`, then zero padded. The mask is `true` exactly where
//! a byte was written. Because a validated offset is at least 1, that equals
//! the reference collator's `ids != 0` mask.
//!
//! Work is bounded by the spec, not the input: an overlong text is sliced, never
//! copied whole, and a `Reject` field refuses it before anything is allocated.
//!
//! # Example
//!
//! ```
//! use xybrid_core::execution::choice::{effective_candidates, encode::encode_request};
//! use xybrid_core::execution::template::ChoiceScorerSpec;
//! use xybrid_core::ir::Choice;
//!
//! # let json = r#"{"type":"OnnxByteOptionScorer","model_file":"m.onnx",
//! #   "context":{"offset":1,"max_len":4,"overflow":"Truncate","ids_input":"c","mask_input":"cm"},
//! #   "choice":{"offset":1,"max_len":3,"overflow":"Truncate","ids_input":"o","mask_input":"om"},
//! #   "choice_mask_input":"m","logits_output":"logits","max_choices":8}"#;
//! # let ChoiceScorerSpec::OnnxByteOptionScorer(spec) = serde_json::from_str(json)? else { unreachable!() };
//! let choices = [Choice::new("a", "é"), Choice::new("b", "ok")];
//! let candidates = effective_candidates(&choices, &[], spec.max_choices)?;
//! let encoded = encode_request(&spec, "hello", &candidates, None)?;
//! assert_eq!(encoded.context_ids.as_slice().unwrap(), &[105, 102, 109, 109]); // "hell" + 1
//! assert!(encoded.context_truncated);
//! assert_eq!(encoded.choice_ids.as_slice().unwrap(), &[196, 170, 0, 112, 108, 0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::{encoded_ids, Candidate, ChoiceError, ChoiceField, ChoiceResult, MAX_ENCODED_IDS};
use crate::execution::template::{ByteFieldSpec, ByteOverflow, OnnxByteOptionScorerSpec};
use ndarray::{Array2, Array3, ArrayViewMut1, Axis};
use std::fmt;

/// The five tensors of one byte-scorer request, plus what truncation dropped.
///
/// The ids spell out the request's text, so [`Debug`] prints shapes and
/// truncation flags only.
#[derive(Clone, PartialEq)]
pub struct EncodedRequest {
    /// `[1, context.max_len]` byte ids.
    pub context_ids: Array2<i64>,
    /// `[1, context.max_len]`, `true` where `context_ids` holds a byte.
    pub context_mask: Array2<bool>,
    /// `[1, rows, choice.max_len]` byte ids, one row per candidate, then
    /// all-zero padding rows.
    pub choice_ids: Array3<i64>,
    /// `[1, rows, choice.max_len]`, `true` where `choice_ids` holds a byte.
    pub choice_token_mask: Array3<bool>,
    /// `[1, rows]`, `true` for the rows holding a candidate.
    pub choice_mask: Array2<bool>,
    /// Whether the context was cut at `context.max_len`.
    pub context_truncated: bool,
    /// Per candidate in E order, whether its text was cut at `choice.max_len`.
    pub choice_truncated: Vec<bool>,
}

impl fmt::Debug for EncodedRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodedRequest")
            .field("context_ids", &self.context_ids.shape())
            .field("choice_ids", &self.choice_ids.shape())
            .field("context_truncated", &self.context_truncated)
            .field("choice_truncated", &self.choice_truncated)
            .finish()
    }
}

/// Encodes a context and its candidates into the byte scorer's tensors.
///
/// `rows` is the model's fixed choice axis, if it has one; the rows after the
/// candidates are zero with a `false` mask. `None` gives one row per candidate.
///
/// Expects a spec that passed
/// [`validate_scorer_spec`](super::validate_scorer_spec) and candidates from
/// [`effective_candidates`](super::effective_candidates): with an offset of 0
/// a NUL byte would encode as the padding id, and the candidate rules are not
/// re-checked here. [`OnnxChoiceScorer`](super::onnx::OnnxChoiceScorer)
/// guarantees both.
///
/// # Errors
///
/// [`ChoiceError::EmptyContext`] for an empty context;
/// [`ChoiceError::InputTooLong`] when a `Reject` field is over its `max_len`;
/// [`ChoiceError::TooManyCandidates`] when there are more candidates than
/// `rows`; [`ChoiceError::InvalidSpec`] when the request would encode more
/// than [`MAX_ENCODED_IDS`] ids.
pub fn encode_request(
    spec: &OnnxByteOptionScorerSpec,
    context: &str,
    candidates: &[Candidate<'_>],
    rows: Option<usize>,
) -> ChoiceResult<EncodedRequest> {
    if context.is_empty() {
        return Err(ChoiceError::EmptyContext);
    }
    check_length(context, &spec.context, ChoiceField::Context)?;
    for (index, candidate) in candidates.iter().enumerate() {
        check_length(
            &candidate.choice.text,
            &spec.choice,
            ChoiceField::Choice(index),
        )?;
    }

    let rows = rows.unwrap_or(candidates.len());
    if candidates.len() > rows {
        return Err(ChoiceError::TooManyCandidates {
            count: candidates.len(),
            max: rows,
        });
    }
    if encoded_ids(spec, rows).is_none() {
        return Err(ChoiceError::InvalidSpec(format!(
            "{rows} choice rows would encode more than {MAX_ENCODED_IDS} ids"
        )));
    }

    let mut context_ids = Array2::<i64>::zeros((1, spec.context.max_len));
    let mut context_mask = Array2::from_elem((1, spec.context.max_len), false);
    let context_truncated = write_row(
        context,
        &spec.context,
        context_ids.row_mut(0),
        context_mask.row_mut(0),
    );

    let mut choice_ids = Array3::<i64>::zeros((1, rows, spec.choice.max_len));
    let mut choice_token_mask = Array3::from_elem((1, rows, spec.choice.max_len), false);
    let mut choice_mask = Array2::from_elem((1, rows), false);
    let mut choice_truncated = Vec::with_capacity(candidates.len());
    let mut ids_rows = choice_ids.index_axis_mut(Axis(0), 0);
    let mut mask_rows = choice_token_mask.index_axis_mut(Axis(0), 0);
    for (((candidate, ids), mask), offered) in candidates
        .iter()
        .zip(ids_rows.outer_iter_mut())
        .zip(mask_rows.outer_iter_mut())
        .zip(choice_mask.row_mut(0).iter_mut())
    {
        choice_truncated.push(write_row(&candidate.choice.text, &spec.choice, ids, mask));
        *offered = true;
    }

    Ok(EncodedRequest {
        context_ids,
        context_mask,
        choice_ids,
        choice_token_mask,
        choice_mask,
        context_truncated,
        choice_truncated,
    })
}

fn check_length(text: &str, field: &ByteFieldSpec, which: ChoiceField) -> ChoiceResult<()> {
    if field.overflow == ByteOverflow::Reject && text.len() > field.max_len {
        return Err(ChoiceError::InputTooLong {
            field: which,
            len: text.len(),
            max_len: field.max_len,
        });
    }
    Ok(())
}

/// Writes `text` into a zeroed row; returns whether it was truncated.
fn write_row(
    text: &str,
    field: &ByteFieldSpec,
    mut ids: ArrayViewMut1<'_, i64>,
    mut mask: ArrayViewMut1<'_, bool>,
) -> bool {
    let bytes = text.as_bytes();
    let kept = &bytes[..bytes.len().min(field.max_len)];
    let offset = i64::from(field.offset);
    for ((id, set), &byte) in ids.iter_mut().zip(mask.iter_mut()).zip(kept) {
        *id = i64::from(byte) + offset;
        *set = true;
    }
    kept.len() < bytes.len()
}

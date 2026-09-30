//! Running an [`OnnxByteOptionScorerSpec`] against an [`ONNXSession`].
//!
//! Binding checks the model against the spec once: every named input exists
//! with the right element type and rank, fixed dimensions match the spec, the
//! three per-choice inputs agree on their choice axis, and the logits output
//! exists and is float32. ONNX output shapes are not declared reliably, so the
//! logits shape is checked again on every call.
//!
//! The choice axis is either dynamic (one row per candidate) or static: then
//! every request fills `N` rows, masks the unused tail, and drops the tail's
//! logits before they are checked or normalised, since a model may return
//! anything there (NaN included).
//!
//! Scoring runs in two steps so a caller can prepare a request, which needs
//! only the bound scorer, before borrowing the session:
//! [`OnnxChoiceScorer::prepare`] validates and encodes, then
//! [`PreparedChoices::run`] runs the model. [`OnnxChoiceScorer::score`] does
//! both.
//!
//! # Example
//!
//! ```no_run
//! use xybrid_core::execution::choice::onnx::OnnxChoiceScorer;
//! use xybrid_core::execution::template::ChoiceScorerSpec;
//! use xybrid_core::ir::Choice;
//! use xybrid_core::runtime_adapter::onnx::{ExecutionProviderKind, ONNXSession, SessionOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let spec: ChoiceScorerSpec = serde_json::from_str(&std::fs::read_to_string("spec.json")?)?;
//! let ChoiceScorerSpec::OnnxByteOptionScorer(spec) = spec else { unreachable!() };
//! let session = ONNXSession::build("cua-s1-forms.onnx", ExecutionProviderKind::Cpu, SessionOptions::default())?;
//! let scorer = OnnxChoiceScorer::bind(&session, &spec)?;
//! let scores = scorer.score(
//!     &session,
//!     "FORM Intake\nELEMENT Edit \"Phone\" value=\"\"",
//!     &[Choice::new("tel", "fill Tel: (503) 555-0142"), Choice::new("name", "fill Name: Jane Doe")],
//! )?;
//! let best = &scores.entries[scores.best().unwrap()];
//! println!("{} {:.3}", best.id, best.score);
//! # Ok(())
//! # }
//! ```

use super::encode::{encode_request, EncodedRequest};
use super::math::choice_scores;
use super::{effective_candidates, encoded_ids, validate_onnx_spec, ChoiceError, ChoiceResult};
use crate::execution::template::OnnxByteOptionScorerSpec;
use crate::ir::{Choice, ChoiceScores};
use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{Array, Dimension};
use ort::tensor::{PrimitiveTensorElementType, TensorElementType};
use ort::value::{Tensor, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// The size of a model's choice axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceAxis {
    /// One row per candidate.
    Dynamic,
    /// Always this many rows; the rows after the candidates are masked.
    Static(usize),
}

/// A byte-scorer spec checked against one ONNX model.
///
/// Bind once per model file and reuse it for every request to that model.
/// Cloning is cheap (the spec is shared).
#[derive(Debug, Clone)]
pub struct OnnxChoiceScorer {
    spec: Arc<OnnxByteOptionScorerSpec>,
    axis: ChoiceAxis,
}

impl OnnxChoiceScorer {
    /// Validates `spec` and checks that `session`'s model matches it.
    ///
    /// # Errors
    ///
    /// [`ChoiceError::InvalidSpec`] for a spec that fails
    /// [`validate_scorer_spec`](super::validate_scorer_spec);
    /// [`ChoiceError::ModelContract`] when the model's inputs or output differ
    /// from the spec.
    pub fn bind(session: &ONNXSession, spec: &OnnxByteOptionScorerSpec) -> ChoiceResult<Self> {
        validate_onnx_spec(spec)?;
        let contract = |reason: String| ChoiceError::ModelContract(reason);

        let names = session.input_names();
        if names.len() != 5 {
            return Err(contract(format!(
                "the model has {} inputs; the spec names 5",
                names.len()
            )));
        }

        let int64 = TensorElementType::Int64;
        let bool_ = TensorElementType::Bool;
        let context_len = spec.context.max_len;
        let choice_len = spec.choice.max_len;
        let context_ids = declared_input(session, &spec.context.ids_input, int64, 2)?;
        let context_mask = declared_input(session, &spec.context.mask_input, bool_, 2)?;
        let choice_ids = declared_input(session, &spec.choice.ids_input, int64, 3)?;
        let choice_token_mask = declared_input(session, &spec.choice.mask_input, bool_, 3)?;
        let choice_mask = declared_input(session, &spec.choice_mask_input, bool_, 2)?;

        for input in [
            &context_ids,
            &context_mask,
            &choice_ids,
            &choice_token_mask,
            &choice_mask,
        ] {
            input.expect_dim(0, 1)?;
        }
        for input in [&context_ids, &context_mask] {
            input.expect_dim(1, context_len)?;
        }
        for input in [&choice_ids, &choice_token_mask] {
            input.expect_dim(2, choice_len)?;
        }

        let axis = match (
            choice_ids.dim(1),
            choice_token_mask.dim(1),
            choice_mask.dim(1),
        ) {
            (None, None, None) => ChoiceAxis::Dynamic,
            (Some(a), Some(b), Some(c)) if a == b && b == c => ChoiceAxis::Static(a),
            _ => {
                return Err(contract(format!(
                    "inputs '{}', '{}' and '{}' disagree on the choice axis",
                    spec.choice.ids_input, spec.choice.mask_input, spec.choice_mask_input
                )))
            }
        };
        if let ChoiceAxis::Static(rows) = axis {
            if rows < spec.max_choices {
                return Err(contract(format!(
                    "the choice axis holds {rows} rows; the spec allows {} choices",
                    spec.max_choices
                )));
            }
            if encoded_ids(spec, rows).is_none() {
                return Err(contract(format!(
                    "a choice axis of {rows} rows would encode more than {} ids",
                    super::MAX_ENCODED_IDS
                )));
            }
        }

        let output = session
            .output_names()
            .iter()
            .position(|name| *name == spec.logits_output)
            .ok_or_else(|| contract(format!("the model has no output '{}'", spec.logits_output)))?;
        match session.output_dtypes().get(output).copied().flatten() {
            Some(TensorElementType::Float32) => {}
            other => {
                return Err(contract(format!(
                    "output '{}' is {}; the spec needs float32",
                    spec.logits_output,
                    describe(other)
                )))
            }
        }

        Ok(Self {
            spec: Arc::new(spec.clone()),
            axis,
        })
    }

    /// The spec this scorer was bound with.
    pub fn spec(&self) -> &OnnxByteOptionScorerSpec {
        &self.spec
    }

    /// The model's choice axis.
    pub fn axis(&self) -> ChoiceAxis {
        self.axis
    }

    /// Checks the candidates and encodes the request, without the session.
    ///
    /// # Errors
    ///
    /// The candidate errors of
    /// [`effective_candidates`] and the encoding
    /// errors of [`encode_request`].
    pub fn prepare(&self, context: &str, choices: &[Choice]) -> ChoiceResult<PreparedChoices> {
        let candidates =
            effective_candidates(choices, &self.spec.fixed_choices, self.spec.max_choices)?;
        let rows = match self.axis {
            ChoiceAxis::Dynamic => None,
            ChoiceAxis::Static(rows) => Some(rows),
        };
        let encoded = encode_request(&self.spec, context, &candidates, rows)?;
        Ok(PreparedChoices {
            spec: Arc::clone(&self.spec),
            ids: candidates
                .iter()
                .map(|candidate| (candidate.choice.id.clone(), candidate.fixed))
                .collect(),
            encoded,
        })
    }

    /// Scores `choices` (then the fixed choices) against `context`.
    ///
    /// `session` must hold the model this scorer was bound to: only the output
    /// shape is re-checked per call.
    ///
    /// # Errors
    ///
    /// As [`OnnxChoiceScorer::prepare`] and [`PreparedChoices::run`].
    pub fn score(
        &self,
        session: &ONNXSession,
        context: &str,
        choices: &[Choice],
    ) -> ChoiceResult<ChoiceScores> {
        self.prepare(context, choices)?.run(session)
    }
}

/// A validated, encoded choice request, ready to run.
///
/// Its [`Debug`] output shows counts and shapes, never candidate ids or text.
#[derive(Clone)]
pub struct PreparedChoices {
    spec: Arc<OnnxByteOptionScorerSpec>,
    /// E as `(id, fixed)`.
    ids: Vec<(String, bool)>,
    encoded: EncodedRequest,
}

impl std::fmt::Debug for PreparedChoices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedChoices")
            .field("candidates", &self.ids.len())
            .field("encoded", &self.encoded)
            .finish()
    }
}

impl PreparedChoices {
    /// The encoded tensors.
    pub fn encoded(&self) -> &EncodedRequest {
        &self.encoded
    }

    /// Runs the model and turns its logits into scores in E order.
    ///
    /// `session` must hold the model the scorer was bound to: only the output
    /// shape is re-checked here.
    ///
    /// # Errors
    ///
    /// [`ChoiceError::InferenceFailed`] when ONNX Runtime fails (its message is
    /// withheld: it can quote input values, which are bytes of the request);
    /// [`ChoiceError::Runtime`] when the logits output is missing or not
    /// float32; [`ChoiceError::OutputShape`] unless the logits are `[1, rows]`
    /// for the encoded rows; [`ChoiceError::NonFiniteLogit`] for a NaN or
    /// infinite logit on an offered candidate.
    pub fn run(self, session: &ONNXSession) -> ChoiceResult<ChoiceScores> {
        let spec = &self.spec;
        let rows = self.encoded.choice_mask.len();
        let EncodedRequest {
            context_ids,
            context_mask,
            choice_ids,
            choice_token_mask,
            choice_mask,
            ..
        } = self.encoded;
        let inputs = HashMap::from([
            (spec.context.ids_input.clone(), tensor(context_ids)?),
            (spec.context.mask_input.clone(), tensor(context_mask)?),
            (spec.choice.ids_input.clone(), tensor(choice_ids)?),
            (spec.choice.mask_input.clone(), tensor(choice_token_mask)?),
            (spec.choice_mask_input.clone(), tensor(choice_mask)?),
        ]);

        let logits = session
            .run_single_output_f32(inputs, &spec.logits_output)
            .map_err(|error| match error {
                AdapterError::InferenceFailed(_) => ChoiceError::InferenceFailed,
                other => ChoiceError::Runtime(other),
            })?;
        if logits.shape() != [1, rows] {
            return Err(ChoiceError::OutputShape {
                output: spec.logits_output.clone(),
                expected: vec![1, rows],
                actual: logits.shape().to_vec(),
            });
        }
        // Only the candidates' rows count; a static axis's tail is padding.
        let active: Vec<f32> = logits.iter().take(self.ids.len()).copied().collect();
        choice_scores(self.ids, &active, spec.scoring_temperature)
    }
}

fn tensor<T, D>(array: Array<T, D>) -> ChoiceResult<Value>
where
    T: PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static,
    D: Dimension + 'static,
{
    Ok(Tensor::from_array(array)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to build an input tensor: {e}")))?
        .into_dyn())
}

fn describe(dtype: Option<TensorElementType>) -> String {
    dtype.map_or_else(|| "not a tensor".to_string(), |dtype| format!("{dtype:?}"))
}

/// One input the spec names, as the model declares it.
struct DeclaredInput<'s> {
    name: &'s str,
    shape: &'s [i64],
}

/// Finds input `name` and checks its element type and rank.
fn declared_input<'s>(
    session: &'s ONNXSession,
    name: &'s str,
    dtype: TensorElementType,
    rank: usize,
) -> ChoiceResult<DeclaredInput<'s>> {
    let contract = |reason: String| ChoiceError::ModelContract(reason);
    let index = session
        .input_names()
        .iter()
        .position(|declared| declared == name)
        .ok_or_else(|| contract(format!("the model has no input '{name}'")))?;
    let declared = session.input_dtypes().get(index).copied().flatten();
    if declared != Some(dtype) {
        return Err(contract(format!(
            "input '{name}' is {}; the spec needs {dtype:?}",
            describe(declared)
        )));
    }
    let shape = session.input_shapes()[index].as_slice();
    if shape.len() != rank {
        return Err(contract(format!(
            "input '{name}' has rank {}; the spec needs {rank}",
            shape.len()
        )));
    }
    Ok(DeclaredInput { name, shape })
}

impl DeclaredInput<'_> {
    /// The fixed size of `axis`, or `None` when it is dynamic.
    fn dim(&self, axis: usize) -> Option<usize> {
        usize::try_from(self.shape[axis]).ok()
    }

    /// Accepts a dynamic `axis` or one fixed at `size`.
    fn expect_dim(&self, axis: usize, size: usize) -> ChoiceResult<()> {
        match self.dim(axis) {
            Some(fixed) if fixed != size => Err(ChoiceError::ModelContract(format!(
                "input '{}' fixes axis {axis} at {fixed}; the spec needs {size}",
                self.name
            ))),
            _ => Ok(()),
        }
    }
}

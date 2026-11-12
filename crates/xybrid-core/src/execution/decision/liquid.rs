//! Liquid text/v1 admission and content-free protocol refusals.

use crate::error::XybridError;
use crate::ir::decision::{
    validate_structure, DecisionContent, DecisionError, DecisionQuestionKind, DecisionRequest,
    JsonErrorKind, JsonNumber, JsonView, OrderedJson, StructuralLimits,
};
use crate::runtime_adapter::AdapterError;

/// Hash of the audited named `systemone` template.
pub const TEMPLATE_SHA256: &str =
    "ee5fc62461a9bbbcb8389c57f4cd2cd694ccb0dec19521576bd1419049490d8b";
/// Maximum rendered bytes per independently evaluated question.
pub const MAX_RENDERED_BYTES: usize = 8 << 20;
/// Inclusive whole-prompt token ceiling.
pub const MAX_INPUT_TOKENS: usize = 4096;

/// A stable refusal category, containing no caller or model content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DecisionProtocolError {
    #[error("decision invalid_json")]
    InvalidJson,
    #[error("decision invalid_request")]
    InvalidRequest,
    #[error("decision duplicate_key")]
    DuplicateKey,
    #[error("decision unknown_field")]
    UnknownField,
    #[error("decision missing_state")]
    MissingState,
    #[error("decision invalid_questions")]
    InvalidQuestions,
    #[error("decision invalid_question")]
    InvalidQuestion,
    #[error("decision missing_instructions")]
    MissingInstructions,
    #[error("decision invalid_question_type")]
    InvalidQuestionType,
    #[error("decision invalid_criteria")]
    InvalidCriteria,
    #[error("decision level_count")]
    LevelCount,
    #[error("decision too_many_options")]
    TooManyOptions,
    #[error("decision empty_question_id")]
    EmptyQuestionId,
    #[error("decision empty_option_key")]
    EmptyOptionKey,
    #[error("decision media_unsupported")]
    MediaUnsupported,
    #[error("decision spoken_instructions")]
    SpokenInstructions,
    #[error("decision json_too_deep")]
    JsonTooDeep,
    #[error("decision number_out_of_range")]
    NumberOutOfRange,
    #[error("decision escaped_key_collision")]
    EscapedKeyCollision,
    #[error("decision no_single_token_label")]
    NoSingleTokenLabel,
    #[error("decision budget_exceeded")]
    BudgetExceeded,
    #[error("decision non_finite_scores")]
    NonFiniteScores,
    #[error("decision invalid_metadata")]
    InvalidMetadata,
    #[error("decision unsupported_metadata")]
    UnsupportedMetadata,
    #[error("decision invalid_output")]
    InvalidOutput,
    #[error("decision native_failure")]
    NativeFailure,
    #[error("decision cancelled")]
    Cancelled,
}

impl DecisionProtocolError {
    /// Stable, content-free fixture and telemetry refusal category.
    pub fn category(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidRequest => "invalid_request",
            Self::DuplicateKey => "duplicate_key",
            Self::UnknownField => "unknown_field",
            Self::MissingState => "missing_state",
            Self::InvalidQuestions => "invalid_questions",
            Self::InvalidQuestion => "invalid_question",
            Self::MissingInstructions => "missing_instructions",
            Self::InvalidQuestionType => "invalid_question_type",
            Self::InvalidCriteria => "invalid_criteria",
            Self::LevelCount => "level_count",
            Self::TooManyOptions => "too_many_options",
            Self::EmptyQuestionId => "empty_question_id",
            Self::EmptyOptionKey => "empty_option_key",
            Self::MediaUnsupported => "media_unsupported",
            Self::SpokenInstructions => "spoken_instructions",
            Self::JsonTooDeep => "json_too_deep",
            Self::NumberOutOfRange => "number_out_of_range",
            Self::EscapedKeyCollision => "escaped_key_collision",
            Self::NoSingleTokenLabel => "no_single_token_label",
            Self::BudgetExceeded => "budget_exceeded",
            Self::NonFiniteScores => "non_finite_scores",
            Self::InvalidMetadata => "invalid_metadata",
            Self::UnsupportedMetadata => "unsupported_metadata",
            Self::InvalidOutput => "invalid_output",
            Self::NativeFailure => "native_failure",
            Self::Cancelled => "cancelled",
        }
    }
}

impl From<DecisionError> for DecisionProtocolError {
    fn from(error: DecisionError) -> Self {
        match error {
            DecisionError::InvalidJson {
                kind: JsonErrorKind::DuplicateKey,
                ..
            } => Self::DuplicateKey,
            DecisionError::InvalidJson {
                kind: JsonErrorKind::NumberOutOfRange,
                ..
            } => Self::NumberOutOfRange,
            DecisionError::InvalidJson { .. } => Self::InvalidJson,
            DecisionError::JsonTooDeep { .. } => Self::JsonTooDeep,
            DecisionError::EmptyQuestionId { .. } => Self::EmptyQuestionId,
            DecisionError::EmptyOptionKey { .. } => Self::EmptyOptionKey,
            DecisionError::DuplicateQuestionId { .. }
            | DecisionError::DuplicateOptionKey { .. } => Self::DuplicateKey,
            DecisionError::NoQuestions => Self::InvalidQuestions,
            DecisionError::NoOptions { .. } => Self::InvalidCriteria,
            DecisionError::TooManyOptions { .. } => Self::TooManyOptions,
            DecisionError::NoLevels { .. } | DecisionError::TooManyLevels { .. } => {
                Self::LevelCount
            }
            DecisionError::InvalidAnswer(_) => Self::InvalidOutput,
            _ => Self::BudgetExceeded,
        }
    }
}

impl From<DecisionProtocolError> for AdapterError {
    fn from(error: DecisionProtocolError) -> Self {
        match error {
            DecisionProtocolError::Cancelled => Self::Cancelled,
            DecisionProtocolError::NonFiniteScores
            | DecisionProtocolError::InvalidOutput
            | DecisionProtocolError::NativeFailure => Self::InferenceFailed(error.to_string()),
            _ => Self::InvalidInput(error.to_string()),
        }
    }
}

impl From<DecisionProtocolError> for XybridError {
    fn from(error: DecisionProtocolError) -> Self {
        AdapterError::from(error).into()
    }
}

/// Validates typed requests against the text/v1 schema before preparation.
pub fn validate_request(request: &DecisionRequest) -> Result<(), DecisionProtocolError> {
    validate(request, true)
}

pub(super) fn validate_wire_request(
    request: &DecisionRequest,
) -> Result<(), DecisionProtocolError> {
    validate(request, false)
}

fn validate(request: &DecisionRequest, preparation: bool) -> Result<(), DecisionProtocolError> {
    validate_structure(request, &StructuralLimits::default())?;
    let outer_depth = if preparation { 127 } else { 128 };
    let description_depth = if preparation { 125 } else { 128 };
    validate_content(request.state().body(), outer_depth)?;
    if !request.state().media().is_empty() {
        return Err(DecisionProtocolError::MediaUnsupported);
    }
    if let Some(state) = request.state().body().and_then(DecisionContent::as_json) {
        reject_media(state)?;
    }
    for question in request.questions() {
        let instructions = question
            .instructions()
            .and_then(|i| i.as_content())
            .ok_or(DecisionProtocolError::MissingInstructions)?;
        validate_content(Some(instructions), outer_depth)?;
        if instructions
            .as_json()
            .and_then(|v| v.get("type"))
            .and_then(OrderedJson::as_str)
            == Some("input_audio")
        {
            return Err(DecisionProtocolError::SpokenInstructions);
        }
        match question.kind() {
            DecisionQuestionKind::Choice { options } => {
                for option in options {
                    validate_content(option.description(), description_depth)?;
                }
            }
            DecisionQuestionKind::YesNo { yes, no } => {
                validate_content(yes.as_ref(), description_depth)?;
                validate_content(no.as_ref(), description_depth)?;
            }
            DecisionQuestionKind::Score { levels } => {
                if !(2..=10).contains(&levels.len()) {
                    return Err(DecisionProtocolError::LevelCount);
                }
                for level in levels {
                    validate_content(level.description(), description_depth)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_content(
    content: Option<&DecisionContent>,
    depth: usize,
) -> Result<(), DecisionProtocolError> {
    if let Some(json) = content.and_then(DecisionContent::as_json) {
        if json.depth() > depth {
            return Err(DecisionProtocolError::JsonTooDeep);
        }
        validate_numbers(json)?;
    }
    Ok(())
}

fn validate_numbers(value: &OrderedJson) -> Result<(), DecisionProtocolError> {
    match value.view() {
        // The pinned Jinja value representation has signed integers only. The
        // neutral IR retains u64; this profile must refuse rather than wrap it.
        JsonView::Number(JsonNumber::U64(_)) => {
            return Err(DecisionProtocolError::NumberOutOfRange)
        }
        JsonView::Array(items) => {
            for item in items {
                validate_numbers(item)?;
            }
        }
        JsonView::Object(items) => {
            for (_, item) in items {
                validate_numbers(item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn reject_media(state: &OrderedJson) -> Result<(), DecisionProtocolError> {
    let messages = state.get("messages").unwrap_or(state);
    if let Some(messages) = messages.as_array() {
        for message in messages {
            if let Some(parts) = message.get("content").and_then(OrderedJson::as_array) {
                for part in parts {
                    if matches!(
                        part.get("type").and_then(OrderedJson::as_str),
                        Some("image_url" | "video_url" | "input_audio")
                    ) {
                        return Err(DecisionProtocolError::MediaUnsupported);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "llm-llamacpp")]
pub use native::{DecisionVocabulary, LiquidProtocol};

#[cfg(feature = "llm-llamacpp")]
mod native {
    use super::DecisionProtocolError as Error;
    use crate::execution::decision::prepare::{
        self, PreparationBackend, PreparedQuestion, Profile,
    };
    use crate::ir::decision::{DecisionQuestion, DecisionRequest, DecisionState};
    use std::sync::atomic::{AtomicBool, Ordering};
    use xybrid_llama::{DecisionRenderer, LlamaModel, VocabularyModel};

    /// Read-only native vocabulary access, also implemented by tensor-free models.
    pub trait DecisionVocabulary {
        fn decision_metadata(&self) -> Result<Vec<(String, String)>, Error>;
        fn decision_template(&self) -> Result<String, Error>;
        fn decision_controls(&self) -> Result<Vec<String>, Error>;
        fn decision_tokenize(&self, text: &str, parse_special: bool) -> Result<Vec<i32>, Error>;
    }

    macro_rules! vocabulary {
        ($model:ty) => {
            impl DecisionVocabulary for $model {
                fn decision_metadata(&self) -> Result<Vec<(String, String)>, Error> {
                    self.metadata_entries().map_err(|_| Error::InvalidMetadata)
                }
                fn decision_template(&self) -> Result<String, Error> {
                    self.named_chat_template("systemone")
                        .map_err(|_| Error::InvalidMetadata)?
                        .ok_or(Error::InvalidMetadata)
                }
                fn decision_controls(&self) -> Result<Vec<String>, Error> {
                    let count =
                        i32::try_from(self.n_vocab()).map_err(|_| Error::InvalidMetadata)?;
                    if count > 1 << 20 {
                        return Err(Error::UnsupportedMetadata);
                    }
                    let mut controls = Vec::new();
                    let mut bytes = 0usize;
                    for token in (0..count).filter(|token| self.token_is_control(*token)) {
                        if controls.len() == prepare::MAX_CONTROL_SPELLINGS {
                            return Err(Error::UnsupportedMetadata);
                        }
                        let text = self.token_text(token).map_err(|_| Error::InvalidMetadata)?;
                        bytes = bytes
                            .checked_add(text.len())
                            .ok_or(Error::UnsupportedMetadata)?;
                        if bytes > prepare::MAX_CONTROL_BYTES {
                            return Err(Error::UnsupportedMetadata);
                        }
                        controls.push(text);
                    }
                    Ok(controls)
                }
                fn decision_tokenize(
                    &self,
                    text: &str,
                    parse_special: bool,
                ) -> Result<Vec<i32>, Error> {
                    if parse_special {
                        self.tokenize_special(text, false)
                    } else {
                        self.tokenize(text, false)
                    }
                    .map_err(|_| Error::NativeFailure)
                }
            }
        };
    }
    vocabulary!(LlamaModel);
    vocabulary!(VocabularyModel);

    /// One compiled renderer and immutable admitted profile per loaded model.
    #[derive(Debug)]
    pub struct LiquidProtocol {
        profile: Profile,
        renderer: DecisionRenderer,
    }

    impl LiquidProtocol {
        /// Validates native metadata and compiles the audited template once.
        pub fn load(model: &impl DecisionVocabulary) -> Result<Self, Error> {
            let template = model.decision_template()?;
            let profile = Profile::from_metadata(
                &template,
                &model.decision_metadata()?,
                model.decision_controls()?,
            )?;
            let renderer = DecisionRenderer::new(&template).map_err(|_| Error::InvalidMetadata)?;
            Ok(Self { profile, renderer })
        }

        /// Preflights a complete request before model evaluation.
        pub fn prepare(
            &mut self,
            model: &impl DecisionVocabulary,
            request: &DecisionRequest,
        ) -> Result<Vec<PreparedQuestion>, Error> {
            self.prepare_controlled(model, request, None)
        }

        /// Preflights all passes, checking cancellation between questions.
        pub fn prepare_controlled(
            &mut self,
            model: &impl DecisionVocabulary,
            request: &DecisionRequest,
            abort: Option<&AtomicBool>,
        ) -> Result<Vec<PreparedQuestion>, Error> {
            let check = || {
                if abort.is_some_and(|a| a.load(Ordering::Acquire)) {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            };
            check()?;
            super::validate_request(request)?;
            let mut backend = NativePreparation {
                model,
                renderer: &mut self.renderer,
            };
            let mut prepared = Vec::with_capacity(request.questions().len());
            for question in request.questions() {
                check()?;
                let (_, pass) =
                    prepare::prepare_one(request.state(), question, &self.profile, &mut backend)?;
                prepared.push(pass);
            }
            check()?;
            Ok(prepared)
        }

        /// Returns prompt bytes for a single independently checked G0 pass.
        pub fn prepare_one(
            &mut self,
            model: &impl DecisionVocabulary,
            state: &DecisionState,
            question: &DecisionQuestion,
        ) -> Result<(String, PreparedQuestion), Error> {
            let mut backend = NativePreparation {
                model,
                renderer: &mut self.renderer,
            };
            prepare::prepare_one(state, question, &self.profile, &mut backend)
        }
    }

    struct NativePreparation<'a, V> {
        model: &'a V,
        renderer: &'a mut DecisionRenderer,
    }

    impl<V: DecisionVocabulary> PreparationBackend for NativePreparation<'_, V> {
        fn tokenize(&self, text: &str, parse_special: bool) -> Result<Vec<i32>, Error> {
            self.model.decision_tokenize(text, parse_special)
        }
        fn render(&mut self, ordered_json: &str, max_output_bytes: usize) -> Result<String, Error> {
            self.renderer
                .render(ordered_json, max_output_bytes)
                .map_err(|_| Error::NativeFailure)
        }
    }
}

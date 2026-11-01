//! [`DecisionError`]: every refusal of the decision IR, with messages that
//! carry no caller content.

use super::ScoreStatistic;
use crate::runtime_adapter::AdapterError;
use std::fmt;

/// Why a JSON text could not be parsed, named without quoting the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JsonErrorKind {
    /// The text ended inside a value.
    UnexpectedEnd,
    /// No value starts here.
    ExpectedValue,
    /// An object member does not start with a string key.
    ExpectedKey,
    /// An object key is not followed by `:`.
    ExpectedColon,
    /// An array item or object member is not followed by `,` or the closing
    /// bracket.
    ExpectedCommaOrEnd,
    /// A word other than `null`, `true` or `false`.
    InvalidLiteral,
    /// A number outside the JSON grammar (a leading zero, `+`, a bare `.` or
    /// `e`).
    InvalidNumber,
    /// A number too large for a finite `f64`. nlohmann::json refuses these
    /// too ("number overflow").
    NumberOutOfRange,
    /// A backslash escape outside the JSON grammar.
    InvalidEscape,
    /// A `\u` escape of a UTF-16 surrogate that is not half of a pair.
    UnpairedSurrogate,
    /// A control character (U+0000 to U+001F) written raw inside a string.
    ControlCharacter,
    /// Something other than whitespace after the value.
    TrailingCharacters,
    /// An object repeats a key (after unescaping).
    DuplicateKey,
}

impl JsonErrorKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::UnexpectedEnd => "unexpected end of input",
            Self::ExpectedValue => "expected a value",
            Self::ExpectedKey => "expected a string key",
            Self::ExpectedColon => "expected ':' after an object key",
            Self::ExpectedCommaOrEnd => "expected ',' or a closing bracket",
            Self::InvalidLiteral => "invalid literal",
            Self::InvalidNumber => "invalid number",
            Self::NumberOutOfRange => "number out of range",
            Self::InvalidEscape => "invalid escape sequence",
            Self::UnpairedSurrogate => "unpaired UTF-16 surrogate escape",
            Self::ControlCharacter => "unescaped control character in a string",
            Self::TrailingCharacters => "characters after the value",
            Self::DuplicateKey => "an object repeats a key",
        }
    }
}

impl fmt::Display for JsonErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An invariant a [`DecisionAnswer`](super::DecisionAnswer) breaks.
///
/// Entries are named by their position in the distribution; keys and
/// question ids are never quoted.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AnswerInvariant {
    /// The question id is empty.
    EmptyQuestionId,
    /// The distribution has no entries.
    EmptyDistribution,
    /// An entry has an empty key.
    EmptyKey {
        /// Position in the distribution.
        index: usize,
    },
    /// Two entries share a key.
    DuplicateKey {
        /// Position of the repeat.
        index: usize,
        /// Position of the first use.
        first: usize,
    },
    /// An entry's raw score is NaN or `+inf`. `-inf` is allowed.
    InvalidRawScore {
        /// Position in the distribution.
        index: usize,
    },
    /// Every raw score is `-inf`: the model ruled every entry out, so no
    /// distribution follows from them.
    NoFiniteRawScore,
    /// An entry's probability is not a finite number in `[0, 1]`.
    InvalidProbability {
        /// Position in the distribution.
        index: usize,
    },
    /// The probabilities do not sum to 1 within
    /// [`PROBABILITY_TOLERANCE`](super::PROBABILITY_TOLERANCE).
    ProbabilitySum {
        /// Their sum, added in distribution order.
        sum: f64,
    },
    /// The chosen key is not in the distribution.
    ChoiceNotInDistribution,
    /// The chosen key does not have the highest probability.
    ChoiceNotHighest,
    /// A yes/no distribution does not have exactly the keys `"true"` and
    /// `"false"`.
    YesNoKeys,
    /// `p_yes` is not the probability of `"true"`.
    PYesMismatch,
    /// Entry `index` of a score distribution is not keyed `"index"`.
    ScoreKey {
        /// Position in the distribution.
        index: usize,
    },
    /// The score value is not the statistic it names.
    ScoreMismatch {
        /// The statistic the value claims to be.
        statistic: ScoreStatistic,
    },
    /// The confidence is not a finite number in `[0, 1]`.
    InvalidConfidence,
}

impl fmt::Display for AnswerInvariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuestionId => f.write_str("the question id is empty"),
            Self::EmptyDistribution => f.write_str("the distribution is empty"),
            Self::EmptyKey { index } => write!(f, "distribution entry {index} has an empty key"),
            Self::DuplicateKey { index, first } => {
                write!(
                    f,
                    "distribution entry {index} repeats the key of entry {first}"
                )
            }
            Self::InvalidRawScore { index } => {
                write!(f, "distribution entry {index} has a NaN or +inf raw score")
            }
            Self::NoFiniteRawScore => f.write_str("no raw score is finite"),
            Self::InvalidProbability { index } => {
                write!(
                    f,
                    "distribution entry {index} has a probability outside [0, 1]"
                )
            }
            Self::ProbabilitySum { sum } => write!(
                f,
                "the probabilities sum to {sum}, not 1 within {}",
                super::PROBABILITY_TOLERANCE
            ),
            Self::ChoiceNotInDistribution => {
                f.write_str("the chosen key is not in the distribution")
            }
            Self::ChoiceNotHighest => {
                f.write_str("the chosen key does not have the highest probability")
            }
            Self::YesNoKeys => {
                f.write_str("a yes/no distribution needs exactly the keys \"true\" and \"false\"")
            }
            Self::PYesMismatch => f.write_str("p_yes is not the probability of \"true\""),
            Self::ScoreKey { index } => {
                write!(
                    f,
                    "score distribution entry {index} is not keyed \"{index}\""
                )
            }
            Self::ScoreMismatch { statistic } => {
                write!(
                    f,
                    "the score value is not the {statistic} of the distribution"
                )
            }
            Self::InvalidConfidence => f.write_str("the confidence is outside [0, 1]"),
        }
    }
}

/// Where in a request a piece of content sits, named by position only.
///
/// A protocol refusal points at the content it refuses with this, never by
/// quoting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ContentPart {
    /// The state's body.
    State,
    /// The instructions of a question.
    Instructions {
        /// Position of the question.
        question: usize,
    },
    /// The description of an option of a choice question.
    OptionDescription {
        /// Position of the question.
        question: usize,
        /// Position of the option in its question.
        option: usize,
    },
    /// The description of yes (`true`) or no (`false`) of a yes/no question.
    YesNoDescription {
        /// Position of the question.
        question: usize,
        /// `true` for the description of yes.
        yes: bool,
    },
    /// The description of a level of a score question.
    LevelDescription {
        /// Position of the question.
        question: usize,
        /// The level.
        level: usize,
    },
}

impl fmt::Display for ContentPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::State => f.write_str("the state"),
            Self::Instructions { question } => write!(f, "the instructions of question {question}"),
            Self::OptionDescription { question, option } => {
                write!(
                    f,
                    "the description of option {option} of question {question}"
                )
            }
            Self::YesNoDescription { question, yes } => write!(
                f,
                "the description of {} of question {question}",
                if yes { "yes" } else { "no" }
            ),
            Self::LevelDescription { question, level } => {
                write!(f, "the description of level {level} of question {question}")
            }
        }
    }
}

/// The category of a refusal, as the decision conformance cases name it.
///
/// Callers that react to a refusal (the SDK's error mapping, a transport
/// answering a request) match on [`DecisionError::kind`], never on a
/// message. [`as_str`](Self::as_str) is the spelling of the cases' refusal
/// kinds (`integration-tests/fixtures/decision/README.md`, "Refusal kinds"),
/// plus [`RequestTooLarge`](Self::RequestTooLarge), which no case uses yet.
///
/// Some kinds belong to a transport (`unknown_field`, `missing_state`, …):
/// no [`DecisionError`] has them, because the request types cannot hold the
/// requests they refuse, but a transport reports them with this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecisionRefusalKind {
    /// The body is not one JSON value.
    InvalidJson,
    /// The body is JSON but not an object.
    InvalidRequest,
    /// An object repeats a key.
    DuplicateKey,
    /// A field the transport does not define.
    UnknownField,
    /// No state.
    MissingState,
    /// The questions are missing, not an object, or empty.
    InvalidQuestions,
    /// A question that is not an object.
    InvalidQuestion,
    /// Instructions missing, or null.
    MissingInstructions,
    /// A question type missing, not a string, or unknown.
    InvalidQuestionType,
    /// Criteria of the wrong shape for the question's type.
    InvalidCriteria,
    /// A score with a number of levels the model does not read.
    LevelCount,
    /// More options than the model supports.
    TooManyOptions,
    /// An empty question id.
    EmptyQuestionId,
    /// An empty option key.
    EmptyOptionKey,
    /// Media, which the model does not read.
    MediaUnsupported,
    /// Spoken instructions, which the model does not read.
    SpokenInstructions,
    /// JSON nested deeper than the model reads.
    JsonTooDeep,
    /// A number the model cannot read faithfully.
    NumberOutOfRange,
    /// Two keys of one object that escaping makes equal.
    EscapedKeyCollision,
    /// No single-token label left for an option.
    NoSingleTokenLabel,
    /// A pass longer than the token budget, or passes longer than the
    /// request's token total.
    BudgetExceeded,
    /// NaN or `+inf` among a question's scores, or no finite score.
    NonFiniteScores,
    /// Decision metadata, a template or a vocabulary this build does not
    /// implement.
    UnsupportedMetadata,
    /// Decision metadata no model may carry.
    InvalidMetadata,
    /// A request over a size ceiling: bytes, JSON values or questions. Not
    /// a kind of the conformance cases yet.
    RequestTooLarge,
}

impl DecisionRefusalKind {
    /// Every kind, in declaration order.
    pub const ALL: &'static [Self] = &[
        Self::InvalidJson,
        Self::InvalidRequest,
        Self::DuplicateKey,
        Self::UnknownField,
        Self::MissingState,
        Self::InvalidQuestions,
        Self::InvalidQuestion,
        Self::MissingInstructions,
        Self::InvalidQuestionType,
        Self::InvalidCriteria,
        Self::LevelCount,
        Self::TooManyOptions,
        Self::EmptyQuestionId,
        Self::EmptyOptionKey,
        Self::MediaUnsupported,
        Self::SpokenInstructions,
        Self::JsonTooDeep,
        Self::NumberOutOfRange,
        Self::EscapedKeyCollision,
        Self::NoSingleTokenLabel,
        Self::BudgetExceeded,
        Self::NonFiniteScores,
        Self::UnsupportedMetadata,
        Self::InvalidMetadata,
        Self::RequestTooLarge,
    ];

    /// The kind's name, as the conformance cases spell it:
    /// `"invalid_json"`, `"budget_exceeded"`, ….
    pub fn as_str(self) -> &'static str {
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
            Self::UnsupportedMetadata => "unsupported_metadata",
            Self::InvalidMetadata => "invalid_metadata",
            Self::RequestTooLarge => "request_too_large",
        }
    }
}

impl fmt::Display for DecisionRefusalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why JSON text, a decision request, a decision model or a decision answer
/// was refused, or why running a decision failed.
///
/// Every variant carries byte offsets, positions, counts, limits and kinds
/// only, so its message, its `Debug` and the [`AdapterError`] it converts
/// into never hold a question id, an option key, text, any part of a JSON
/// value, or text from a model file, a template engine or a native library.
/// A variant added later keeps to that rule; there is no passthrough for
/// another error's message.
///
/// [`DecisionError::kind`] names the category of a refusal, and
/// [`DecisionError::into_adapter_error`] maps each kind to the runtime's
/// error categories (none of which is retryable).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DecisionError {
    /// The JSON text breaks the JSON grammar.
    #[error("invalid JSON at byte {offset}: {kind}")]
    InvalidJson {
        /// What is wrong.
        kind: JsonErrorKind,
        /// Byte offset of the problem in the text.
        offset: usize,
    },
    /// The JSON text is longer than its [`JsonLimits`](super::JsonLimits)
    /// allow.
    #[error("JSON text is {len} bytes; at most {max} are accepted")]
    JsonTooLong {
        /// Length of the text in bytes.
        len: usize,
        /// [`JsonLimits::max_bytes`](super::JsonLimits::max_bytes).
        max: usize,
    },
    /// The JSON text holds more values than its
    /// [`JsonLimits`](super::JsonLimits) allow.
    #[error("JSON text has more than {max} values")]
    JsonTooManyNodes {
        /// [`JsonLimits::max_nodes`](super::JsonLimits::max_nodes).
        max: usize,
    },
    /// The JSON text nests deeper than
    /// [`MAX_JSON_DEPTH`](super::MAX_JSON_DEPTH).
    #[error("JSON is nested more than {max} levels deep")]
    JsonTooDeep {
        /// [`MAX_JSON_DEPTH`](super::MAX_JSON_DEPTH).
        max: usize,
    },
    /// The request asks no question.
    #[error("a decision request needs at least one question")]
    NoQuestions,
    /// The request asks more questions than the limits allow.
    #[error("{count} questions; at most {max} are accepted")]
    TooManyQuestions {
        /// Number of questions.
        count: usize,
        /// [`StructuralLimits::max_questions`](super::StructuralLimits::max_questions).
        max: usize,
    },
    /// A question has an empty id.
    #[error("question {question} has an empty id")]
    EmptyQuestionId {
        /// Position of the question.
        question: usize,
    },
    /// Two questions share an id.
    #[error("question {question} repeats the id of question {first}")]
    DuplicateQuestionId {
        /// Position of the repeat.
        question: usize,
        /// Position of the first use.
        first: usize,
    },
    /// A choice question has no options.
    #[error("choice question {question} has no options")]
    NoOptions {
        /// Position of the question.
        question: usize,
    },
    /// A choice question has more options than the limits, or the model's
    /// protocol, allow.
    #[error("choice question {question} has {count} options; at most {max} are accepted")]
    TooManyOptions {
        /// Position of the question.
        question: usize,
        /// Number of options.
        count: usize,
        /// [`StructuralLimits::max_options`](super::StructuralLimits::max_options),
        /// or the protocol's own ceiling when it is lower.
        max: usize,
    },
    /// An option has an empty key.
    #[error("option {option} of question {question} has an empty key")]
    EmptyOptionKey {
        /// Position of the question.
        question: usize,
        /// Position of the option in its question.
        option: usize,
    },
    /// Two options of one question share a key.
    #[error("option {option} of question {question} repeats the key of option {first}")]
    DuplicateOptionKey {
        /// Position of the question.
        question: usize,
        /// Position of the repeat.
        option: usize,
        /// Position of the first use.
        first: usize,
    },
    /// A score question has no levels.
    #[error("score question {question} has no levels")]
    NoLevels {
        /// Position of the question.
        question: usize,
    },
    /// A score question has more levels than the limits allow.
    #[error("score question {question} has {count} levels; at most {max} are accepted")]
    TooManyLevels {
        /// Position of the question.
        question: usize,
        /// Number of levels.
        count: usize,
        /// [`StructuralLimits::max_options`](super::StructuralLimits::max_options).
        max: usize,
    },
    /// The request holds more content than the limits allow.
    #[error("the request holds {bytes} bytes of content; at most {max} are accepted")]
    RequestTooLarge {
        /// Content bytes, counted as
        /// [`StructuralLimits::max_bytes`](super::StructuralLimits::max_bytes)
        /// describes.
        bytes: usize,
        /// [`StructuralLimits::max_bytes`](super::StructuralLimits::max_bytes).
        max: usize,
    },
    /// The request's JSON content holds more values than the limits allow.
    #[error("the request holds {count} JSON values; at most {max} are accepted")]
    TooManyJsonNodes {
        /// JSON values across the request.
        count: usize,
        /// [`StructuralLimits::max_json_nodes`](super::StructuralLimits::max_json_nodes).
        max: usize,
    },
    /// An answer breaks one of its invariants.
    #[error("invalid decision answer: {0}")]
    InvalidAnswer(AnswerInvariant),
    // Protocol refusals: what a model's protocol does not read, found
    // before anything runs.
    /// A question has no instructions, and the model's protocol needs them.
    #[error("question {question} has no instructions, which this model needs")]
    MissingInstructions {
        /// Position of the question.
        question: usize,
    },
    /// A score question has a number of levels the model's protocol does
    /// not read.
    #[error("score question {question} has {count} levels; this model reads {min} to {max}")]
    LevelCount {
        /// Position of the question.
        question: usize,
        /// Number of levels.
        count: usize,
        /// Fewest levels the protocol reads.
        min: usize,
        /// Most levels the protocol reads.
        max: usize,
    },
    /// The state has media, and the model's protocol reads none.
    #[error("the state has media, which this model does not read")]
    MediaUnsupported,
    /// JSON content nests deeper than the model's protocol reads it. The
    /// depth counts the value itself, as
    /// [`OrderedJson::depth`](super::OrderedJson::depth) does.
    #[error("JSON in {part} nests {depth} levels deep; this model reads at most {max}")]
    ContentTooDeep {
        /// Where the content is.
        part: ContentPart,
        /// Its depth.
        depth: usize,
        /// The deepest the protocol reads there.
        max: usize,
    },
    /// JSON content holds an integer above `i64::MAX`, which the model's
    /// template engine cannot read faithfully.
    #[error("an integer in {part} is above 9223372036854775807, which this model cannot read")]
    NumberOutOfRange {
        /// Where the content is.
        part: ContentPart,
    },
    /// Two keys of one JSON object become equal once the model's
    /// special-token spellings are escaped, so one would replace the other.
    #[error(
        "two keys of one object in {part} become equal once special-token spellings are escaped"
    )]
    EscapedKeyCollision {
        /// Where the object is.
        part: ContentPart,
    },
    /// No single-token label is left for an option (or no single token
    /// spells a yes/no or score label).
    #[error("option {option} of question {question} has no single-token label left")]
    NoSingleTokenLabel {
        /// Position of the question.
        question: usize,
        /// Position of the option (yes is 0 and no 1; a level is its
        /// number).
        option: usize,
    },
    /// A question's pass is longer than the model's token budget.
    #[error("question {question} renders to {tokens} tokens; the token budget is {budget}")]
    BudgetExceeded {
        /// Position of the question.
        question: usize,
        /// Tokens in its pass.
        tokens: usize,
        /// The budget, inclusive.
        budget: u32,
    },
    /// The passes together are longer than the request's token total.
    #[error("the request renders to at least {tokens} tokens; at most {max} are accepted")]
    TotalBudgetExceeded {
        /// Tokens counted when the total was exceeded.
        tokens: u64,
        /// The total, inclusive.
        max: u64,
    },

    // Model refusals: what a model carries that the protocol does not
    // implement.
    /// The model carries decision metadata the profile does not implement
    /// (a score bias, say).
    #[error("the model carries decision metadata this build does not implement")]
    UnsupportedMetadata,
    /// The model's decision metadata is invalid: no decision model may carry
    /// it (a temperature that reads as zero or less, say).
    #[error("the model's decision metadata is invalid")]
    InvalidMetadata,
    /// The model's decision template is not one the profile implements.
    #[error("the model's decision template is not one this build implements")]
    UnsupportedTemplate,
    /// The model's metadata does not classify as the profile it was given
    /// as.
    #[error("the model's metadata does not classify as the decision profile it was loaded as")]
    ProfileMismatch,
    /// The model's vocabulary has special-token spellings the protocol
    /// cannot escape (not UTF-8, or holding the escape character itself).
    #[error("the model's vocabulary has special-token spellings this protocol cannot escape")]
    UnsupportedVocabulary,

    // Runtime failures, named without the runtime's own text.
    /// Rendering a prompt failed.
    #[error("rendering a decision prompt failed")]
    RenderFailed,
    /// Tokenizing a prompt or a label failed.
    #[error("tokenizing a decision prompt failed")]
    TokenizationFailed,
    /// The model returned a different number of readouts than the plan has
    /// passes.
    #[error("{found} readouts for {expected} passes")]
    ReadoutCount {
        /// Passes in the plan.
        expected: usize,
        /// Readouts received.
        found: usize,
    },
    /// A readout has a different number of values than its pass selects.
    #[error("readout {pass} has {found} values; its pass selects {expected}")]
    ReadoutLength {
        /// Position of the pass.
        pass: usize,
        /// Tokens the pass selects.
        expected: usize,
        /// Values received.
        found: usize,
    },
    /// A question's scores hold NaN or `+inf`, or no finite score: the model
    /// could not evaluate it.
    #[error("the model's scores for question {question} are not finite")]
    NonFiniteScores {
        /// Position of the question.
        question: usize,
    },
    /// Reading the model's vocabulary (a token's attributes or text) failed.
    #[error("reading the model's vocabulary failed")]
    VocabularyFailed,
}

/// The model id an [`AdapterError::UnsupportedModelCapability`] names when
/// the conversion does not know it ([`From<DecisionError>`]); callers that
/// know it use [`DecisionError::into_adapter_error`].
const UNKNOWN_MODEL_ID: &str = "unknown";

/// The capability a refused decision model lacks, for
/// [`AdapterError::UnsupportedModelCapability`].
const DECISION_PROFILE_CAPABILITY: &str = "the lfm3-d1/text/v1 decision profile";

impl DecisionError {
    /// The category of the refusal, or `None` for what is not one: an
    /// invalid answer, a runtime failure, a readout of the wrong shape.
    ///
    /// Size ceilings (bytes, JSON values, questions) are
    /// [`DecisionRefusalKind::RequestTooLarge`]; the structural refusals map
    /// to the kind a transport would report for the same request (a
    /// repeated question id is a repeated key, no options is invalid
    /// criteria, too few or too many levels a level count).
    pub fn kind(&self) -> Option<DecisionRefusalKind> {
        use DecisionRefusalKind as Kind;
        Some(match self {
            Self::InvalidJson {
                kind: JsonErrorKind::DuplicateKey,
                ..
            } => Kind::DuplicateKey,
            Self::InvalidJson {
                kind: JsonErrorKind::NumberOutOfRange,
                ..
            } => Kind::NumberOutOfRange,
            Self::InvalidJson { .. } => Kind::InvalidJson,
            Self::JsonTooLong { .. }
            | Self::JsonTooManyNodes { .. }
            | Self::TooManyQuestions { .. }
            | Self::RequestTooLarge { .. }
            | Self::TooManyJsonNodes { .. } => Kind::RequestTooLarge,
            Self::JsonTooDeep { .. } | Self::ContentTooDeep { .. } => Kind::JsonTooDeep,
            Self::NoQuestions => Kind::InvalidQuestions,
            Self::EmptyQuestionId { .. } => Kind::EmptyQuestionId,
            Self::DuplicateQuestionId { .. } | Self::DuplicateOptionKey { .. } => {
                Kind::DuplicateKey
            }
            Self::NoOptions { .. } => Kind::InvalidCriteria,
            Self::TooManyOptions { .. } => Kind::TooManyOptions,
            Self::EmptyOptionKey { .. } => Kind::EmptyOptionKey,
            Self::NoLevels { .. } | Self::TooManyLevels { .. } | Self::LevelCount { .. } => {
                Kind::LevelCount
            }
            Self::MissingInstructions { .. } => Kind::MissingInstructions,
            Self::MediaUnsupported => Kind::MediaUnsupported,
            Self::NumberOutOfRange { .. } => Kind::NumberOutOfRange,
            Self::EscapedKeyCollision { .. } => Kind::EscapedKeyCollision,
            Self::NoSingleTokenLabel { .. } => Kind::NoSingleTokenLabel,
            Self::BudgetExceeded { .. } | Self::TotalBudgetExceeded { .. } => Kind::BudgetExceeded,
            Self::NonFiniteScores { .. } => Kind::NonFiniteScores,
            Self::UnsupportedMetadata
            | Self::UnsupportedTemplate
            | Self::ProfileMismatch
            | Self::UnsupportedVocabulary => Kind::UnsupportedMetadata,
            Self::InvalidMetadata => Kind::InvalidMetadata,
            Self::InvalidAnswer(_)
            | Self::RenderFailed
            | Self::TokenizationFailed
            | Self::VocabularyFailed
            | Self::ReadoutCount { .. }
            | Self::ReadoutLength { .. } => return None,
        })
    }

    /// The runtime error for this refusal, naming `model_id` where the
    /// category names a model.
    ///
    /// - A model the protocol does not implement (unsupported metadata,
    ///   template, profile or vocabulary) is
    ///   [`AdapterError::UnsupportedModelCapability`].
    /// - Invalid decision metadata is [`AdapterError::RuntimeError`], the
    ///   category of a model that fails to load.
    /// - An answer the model's output cannot make, non-finite scores, a
    ///   readout of the wrong shape, and a failed render, tokenization or
    ///   vocabulary read are [`AdapterError::InferenceFailed`].
    /// - Every other refusal is about the request, budgets included:
    ///   [`AdapterError::InvalidInput`].
    ///
    /// None of these is retryable: the same request on the same model is
    /// refused again. Messages are this error's own, content-free.
    pub fn into_adapter_error(self, model_id: &str) -> AdapterError {
        let unsupported = |hint: &str| AdapterError::UnsupportedModelCapability {
            model_id: model_id.to_string(),
            capability: DECISION_PROFILE_CAPABILITY.to_string(),
            hint: hint.to_string(),
        };
        match self {
            Self::UnsupportedMetadata => {
                unsupported("it carries decision metadata this build does not implement")
            }
            Self::UnsupportedTemplate => {
                unsupported("its decision template is not one this build implements")
            }
            Self::ProfileMismatch => {
                unsupported("its metadata does not classify as the profile it was loaded as")
            }
            Self::UnsupportedVocabulary => {
                unsupported("its special-token spellings cannot be escaped")
            }
            Self::InvalidMetadata => AdapterError::RuntimeError(self.to_string()),
            Self::InvalidAnswer(_)
            | Self::NonFiniteScores { .. }
            | Self::ReadoutCount { .. }
            | Self::ReadoutLength { .. }
            | Self::RenderFailed
            | Self::TokenizationFailed
            | Self::VocabularyFailed => AdapterError::InferenceFailed(self.to_string()),
            _ => AdapterError::InvalidInput(self.to_string()),
        }
    }
}

impl From<DecisionError> for AdapterError {
    /// [`DecisionError::into_adapter_error`], with an unknown model id.
    fn from(error: DecisionError) -> Self {
        error.into_adapter_error(UNKNOWN_MODEL_ID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_and_request_refusals_become_invalid_input() {
        for error in [
            DecisionError::InvalidJson {
                kind: JsonErrorKind::DuplicateKey,
                offset: 7,
            },
            DecisionError::JsonTooDeep { max: 128 },
            DecisionError::NoQuestions,
            DecisionError::DuplicateOptionKey {
                question: 0,
                option: 2,
                first: 1,
            },
            DecisionError::RequestTooLarge { bytes: 10, max: 9 },
        ] {
            let message = error.to_string();
            match AdapterError::from(error) {
                AdapterError::InvalidInput(text) => assert_eq!(text, message),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn invalid_answers_become_inference_failures() {
        let error = DecisionError::InvalidAnswer(AnswerInvariant::ChoiceNotHighest);
        assert!(matches!(
            AdapterError::from(error),
            AdapterError::InferenceFailed(text)
                if text == "invalid decision answer: the chosen key does not have the highest probability"
        ));
    }

    const CANARY: &str = "PRIVATE-REQUEST-CANARY";
    const NUMBER_CANARY: &str = "918273645";

    /// Every variant and kind, refusing input full of caller content.
    fn every_refusal() -> Vec<DecisionError> {
        use crate::ir::decision::{
            validate_structure, DecisionAnswer, DecisionContent, DecisionLevel, DecisionOption,
            DecisionOrigin, DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionScore,
            DecisionState, DecisionValue, JsonLimits, OrderedJson, StructuralLimits,
        };
        let c = CANARY;
        let n = NUMBER_CANARY;
        let mut refusals = Vec::new();

        // JSON text, one refusal per syntax kind, then each ceiling.
        for text in [
            format!(r#""{c}"#),
            format!(r#"["{c}", ]"#),
            format!(r#"{{"{c}": {n},}}"#),
            format!(r#"{{"{c}"}}"#),
            format!(r#"["{c}" {n}]"#),
            format!(r#"["{c}", nul]"#),
            format!(r#"["{c}", 0{n}]"#),
            format!(r#"["{c}", {n}e999]"#),
            format!(r#""{c}\q""#),
            format!(r#""{c}\uD800""#),
            format!("\"{c}\t\""),
            format!(r#""{c}" {n}"#),
            format!(r#"{{"{c}": 1, "{c}": {n}}}"#),
        ] {
            refusals.push(OrderedJson::parse(&text).unwrap_err());
        }
        let text = format!(r#"["{c}", {n}]"#);
        let tight = JsonLimits::default();
        refusals.push(OrderedJson::parse_with_limits(&text, &tight.with_max_bytes(4)).unwrap_err());
        refusals.push(OrderedJson::parse_with_limits(&text, &tight.with_max_nodes(2)).unwrap_err());
        let deep = format!("{}\"{c}\"{}", "[".repeat(200), "]".repeat(200));
        refusals.push(OrderedJson::parse(&deep).unwrap_err());

        // Request structure.
        let text = |text: String| Some(DecisionContent::text(text));
        let yes_no = |id: String| {
            DecisionQuestion::new(id, None, DecisionQuestionKind::yes_no(text(c.into()), None))
        };
        let choice = |keys: Vec<String>| {
            let options = keys
                .into_iter()
                .map(|key| DecisionOption::new(key, text(c.into())))
                .collect();
            DecisionQuestion::new(c, None, DecisionQuestionKind::choice(options))
        };
        let score = |levels: usize| {
            DecisionQuestion::new(
                c,
                None,
                DecisionQuestionKind::score(vec![DecisionLevel::new(text(c.into())); levels]),
            )
        };
        let state = || {
            DecisionState::from_json(
                OrderedJson::parse(&format!(r#"{{"{c}": [{n}, "{c}"]}}"#)).unwrap(),
            )
        };
        let check = |questions: Vec<DecisionQuestion>, limits: StructuralLimits| {
            validate_structure(&DecisionRequest::new(state(), questions), &limits).unwrap_err()
        };
        let limits = StructuralLimits::default;
        refusals.push(check(vec![], limits()));
        refusals.push(check(
            (0..17).map(|i| yes_no(format!("{c}-{i}"))).collect(),
            limits(),
        ));
        refusals.push(check(
            vec![yes_no(c.into()), yes_no(String::new())],
            limits(),
        ));
        refusals.push(check(vec![yes_no(c.into()), yes_no(c.into())], limits()));
        refusals.push(check(vec![choice(vec![])], limits()));
        refusals.push(check(
            vec![choice((0..256).map(|i| format!("{c}-{i}")).collect())],
            limits(),
        ));
        refusals.push(check(vec![choice(vec![c.into(), String::new()])], limits()));
        refusals.push(check(vec![choice(vec![c.into(), c.into()])], limits()));
        refusals.push(check(vec![score(0)], limits()));
        refusals.push(check(vec![score(256)], limits()));
        refusals.push(check(vec![yes_no(c.into())], limits().with_max_bytes(8)));
        refusals.push(check(
            vec![yes_no(c.into())],
            limits().with_max_json_nodes(1),
        ));

        // Answers, one refusal per invariant.
        let entry =
            |key: &str, raw: f32, p: f64| DecisionScore::new(key, raw, p, DecisionOrigin::Caller);
        let answer =
            |id: &str, distribution: Vec<DecisionScore>, value: DecisionValue, confidence| {
                DecisionAnswer::new(id, distribution, value, confidence).unwrap_err()
            };
        let pick = || DecisionValue::choice(c);
        let one = || vec![entry(c, 0.0, 1.0)];
        refusals.push(answer("", one(), pick(), None));
        refusals.push(answer(c, vec![], pick(), None));
        refusals.push(answer(c, vec![entry("", 0.0, 1.0)], pick(), None));
        refusals.push(answer(
            c,
            vec![entry(c, 0.0, 0.5), entry(c, 0.0, 0.5)],
            pick(),
            None,
        ));
        refusals.push(answer(c, vec![entry(c, f32::NAN, 1.0)], pick(), None));
        refusals.push(answer(
            c,
            vec![entry(c, f32::NEG_INFINITY, 1.0)],
            pick(),
            None,
        ));
        refusals.push(answer(c, vec![entry(c, 0.0, 2.0)], pick(), None));
        refusals.push(answer(c, vec![entry(c, 0.0, 0.5)], pick(), None));
        refusals.push(answer(
            c,
            one(),
            DecisionValue::choice(format!("{c}-other")),
            None,
        ));
        refusals.push(answer(
            c,
            vec![entry(c, 0.0, 0.25), entry("b", 0.0, 0.75)],
            pick(),
            None,
        ));
        refusals.push(answer(c, one(), DecisionValue::yes_no(1.0), None));
        let yes_no_entries = vec![entry("true", 0.0, 0.75), entry("false", 0.0, 0.25)];
        refusals.push(answer(c, yes_no_entries, DecisionValue::yes_no(0.5), None));
        refusals.push(answer(
            c,
            one(),
            DecisionValue::score(0.0, ScoreStatistic::Median),
            None,
        ));
        let half = vec![entry("0", 0.0, 0.5), entry("1", 0.0, 0.5)];
        refusals.push(answer(
            c,
            half,
            DecisionValue::score(1.0, ScoreStatistic::Median),
            None,
        ));
        refusals.push(answer(c, one(), pick(), Some(2.0)));

        // A decision protocol's refusals and failures. They hold positions,
        // counts and limits only, so they are built directly here; the LFM3
        // d1 protocol's tests (`execution/decision/tests`) derive each from
        // requests and models full of content.
        let part = ContentPart::YesNoDescription {
            question: 1,
            yes: false,
        };
        refusals.extend([
            DecisionError::MissingInstructions { question: 2 },
            DecisionError::LevelCount {
                question: 0,
                count: 11,
                min: 2,
                max: 10,
            },
            DecisionError::MediaUnsupported,
            DecisionError::ContentTooDeep {
                part,
                depth: 126,
                max: 125,
            },
            DecisionError::NumberOutOfRange { part },
            DecisionError::EscapedKeyCollision { part },
            DecisionError::NoSingleTokenLabel {
                question: 0,
                option: 52,
            },
            DecisionError::BudgetExceeded {
                question: 0,
                tokens: 4097,
                budget: 4096,
            },
            DecisionError::TotalBudgetExceeded {
                tokens: 16385,
                max: 16384,
            },
            DecisionError::UnsupportedMetadata,
            DecisionError::InvalidMetadata,
            DecisionError::UnsupportedTemplate,
            DecisionError::ProfileMismatch,
            DecisionError::UnsupportedVocabulary,
            DecisionError::RenderFailed,
            DecisionError::TokenizationFailed,
            DecisionError::ReadoutCount {
                expected: 2,
                found: 1,
            },
            DecisionError::ReadoutLength {
                pass: 0,
                expected: 5,
                found: 4,
            },
            DecisionError::NonFiniteScores { question: 3 },
            DecisionError::VocabularyFailed,
        ]);
        refusals
    }

    /// One name per variant and kind. The matches are exhaustive, so a new
    /// variant stops this from compiling until it is named here and
    /// `every_refusal` produces it.
    fn variant(error: &DecisionError) -> String {
        let name = match error {
            DecisionError::InvalidJson { kind, .. } => {
                return format!(
                    "InvalidJson::{}",
                    match kind {
                        JsonErrorKind::UnexpectedEnd => "UnexpectedEnd",
                        JsonErrorKind::ExpectedValue => "ExpectedValue",
                        JsonErrorKind::ExpectedKey => "ExpectedKey",
                        JsonErrorKind::ExpectedColon => "ExpectedColon",
                        JsonErrorKind::ExpectedCommaOrEnd => "ExpectedCommaOrEnd",
                        JsonErrorKind::InvalidLiteral => "InvalidLiteral",
                        JsonErrorKind::InvalidNumber => "InvalidNumber",
                        JsonErrorKind::NumberOutOfRange => "NumberOutOfRange",
                        JsonErrorKind::InvalidEscape => "InvalidEscape",
                        JsonErrorKind::UnpairedSurrogate => "UnpairedSurrogate",
                        JsonErrorKind::ControlCharacter => "ControlCharacter",
                        JsonErrorKind::TrailingCharacters => "TrailingCharacters",
                        JsonErrorKind::DuplicateKey => "DuplicateKey",
                    }
                )
            }
            DecisionError::JsonTooLong { .. } => "JsonTooLong",
            DecisionError::JsonTooManyNodes { .. } => "JsonTooManyNodes",
            DecisionError::JsonTooDeep { .. } => "JsonTooDeep",
            DecisionError::NoQuestions => "NoQuestions",
            DecisionError::TooManyQuestions { .. } => "TooManyQuestions",
            DecisionError::EmptyQuestionId { .. } => "EmptyQuestionId",
            DecisionError::DuplicateQuestionId { .. } => "DuplicateQuestionId",
            DecisionError::NoOptions { .. } => "NoOptions",
            DecisionError::TooManyOptions { .. } => "TooManyOptions",
            DecisionError::EmptyOptionKey { .. } => "EmptyOptionKey",
            DecisionError::DuplicateOptionKey { .. } => "DuplicateOptionKey",
            DecisionError::NoLevels { .. } => "NoLevels",
            DecisionError::TooManyLevels { .. } => "TooManyLevels",
            DecisionError::RequestTooLarge { .. } => "RequestTooLarge",
            DecisionError::TooManyJsonNodes { .. } => "TooManyJsonNodes",
            DecisionError::InvalidAnswer(invariant) => {
                return format!(
                    "InvalidAnswer::{}",
                    match invariant {
                        AnswerInvariant::EmptyQuestionId => "EmptyQuestionId",
                        AnswerInvariant::EmptyDistribution => "EmptyDistribution",
                        AnswerInvariant::EmptyKey { .. } => "EmptyKey",
                        AnswerInvariant::DuplicateKey { .. } => "DuplicateKey",
                        AnswerInvariant::InvalidRawScore { .. } => "InvalidRawScore",
                        AnswerInvariant::NoFiniteRawScore => "NoFiniteRawScore",
                        AnswerInvariant::InvalidProbability { .. } => "InvalidProbability",
                        AnswerInvariant::ProbabilitySum { .. } => "ProbabilitySum",
                        AnswerInvariant::ChoiceNotInDistribution => "ChoiceNotInDistribution",
                        AnswerInvariant::ChoiceNotHighest => "ChoiceNotHighest",
                        AnswerInvariant::YesNoKeys => "YesNoKeys",
                        AnswerInvariant::PYesMismatch => "PYesMismatch",
                        AnswerInvariant::ScoreKey { .. } => "ScoreKey",
                        AnswerInvariant::ScoreMismatch { .. } => "ScoreMismatch",
                        AnswerInvariant::InvalidConfidence => "InvalidConfidence",
                    }
                )
            }
            DecisionError::MissingInstructions { .. } => "MissingInstructions",
            DecisionError::LevelCount { .. } => "LevelCount",
            DecisionError::MediaUnsupported => "MediaUnsupported",
            DecisionError::ContentTooDeep { .. } => "ContentTooDeep",
            DecisionError::NumberOutOfRange { .. } => "NumberOutOfRange",
            DecisionError::EscapedKeyCollision { .. } => "EscapedKeyCollision",
            DecisionError::NoSingleTokenLabel { .. } => "NoSingleTokenLabel",
            DecisionError::BudgetExceeded { .. } => "BudgetExceeded",
            DecisionError::TotalBudgetExceeded { .. } => "TotalBudgetExceeded",
            DecisionError::UnsupportedMetadata => "UnsupportedMetadata",
            DecisionError::InvalidMetadata => "InvalidMetadata",
            DecisionError::UnsupportedTemplate => "UnsupportedTemplate",
            DecisionError::ProfileMismatch => "ProfileMismatch",
            DecisionError::UnsupportedVocabulary => "UnsupportedVocabulary",
            DecisionError::RenderFailed => "RenderFailed",
            DecisionError::TokenizationFailed => "TokenizationFailed",
            DecisionError::ReadoutCount { .. } => "ReadoutCount",
            DecisionError::ReadoutLength { .. } => "ReadoutLength",
            DecisionError::NonFiniteScores { .. } => "NonFiniteScores",
            DecisionError::VocabularyFailed => "VocabularyFailed",
        };
        name.to_string()
    }

    /// The text of `error` and of everything its `source()` chain reaches.
    fn texts(error: &dyn std::error::Error) -> Vec<String> {
        let mut texts = vec![
            error.to_string(),
            format!("{error:?}"),
            format!("{error:#?}"),
        ];
        let mut source = error.source();
        while let Some(inner) = source {
            texts.extend([inner.to_string(), format!("{inner:?}")]);
            source = inner.source();
        }
        texts
    }

    #[test]
    fn no_refusal_carries_caller_content() {
        let refusals = every_refusal();
        let names: std::collections::BTreeSet<String> = refusals.iter().map(variant).collect();
        // 13 JSON syntax kinds, 3 JSON ceilings, 12 request refusals, 15
        // answer invariants and 20 protocol refusals and failures: every one,
        // each once.
        assert_eq!(names.len(), 63, "{names:#?}");
        assert_eq!(refusals.len(), names.len());

        for error in refusals {
            let name = variant(&error);
            let mut seen = texts(&error);
            let adapter = AdapterError::from(error);
            assert!(
                matches!(
                    adapter,
                    AdapterError::InvalidInput(_)
                        | AdapterError::InferenceFailed(_)
                        | AdapterError::UnsupportedModelCapability { .. }
                        | AdapterError::RuntimeError(_)
                ),
                "{name}"
            );
            seen.extend(texts(&adapter));
            for text in seen {
                assert!(!text.contains(CANARY), "{name}: {text}");
                assert!(!text.contains(NUMBER_CANARY), "{name}: {text}");
            }
        }
    }

    #[test]
    fn messages_name_offsets_positions_and_counts() {
        let cases: [(DecisionError, &str); 8] = [
            (
                DecisionError::InvalidJson {
                    kind: JsonErrorKind::UnpairedSurrogate,
                    offset: 12,
                },
                "invalid JSON at byte 12: unpaired UTF-16 surrogate escape",
            ),
            (
                DecisionError::JsonTooLong { len: 9, max: 8 },
                "JSON text is 9 bytes; at most 8 are accepted",
            ),
            (
                DecisionError::JsonTooManyNodes { max: 4 },
                "JSON text has more than 4 values",
            ),
            (
                DecisionError::TooManyQuestions { count: 17, max: 16 },
                "17 questions; at most 16 are accepted",
            ),
            (
                DecisionError::DuplicateQuestionId {
                    question: 3,
                    first: 1,
                },
                "question 3 repeats the id of question 1",
            ),
            (
                DecisionError::TooManyLevels {
                    question: 0,
                    count: 300,
                    max: 255,
                },
                "score question 0 has 300 levels; at most 255 are accepted",
            ),
            (
                DecisionError::InvalidAnswer(AnswerInvariant::ScoreMismatch {
                    statistic: ScoreStatistic::Median,
                }),
                "invalid decision answer: the score value is not the median of the distribution",
            ),
            (
                DecisionError::InvalidAnswer(AnswerInvariant::ProbabilitySum { sum: 0.5 }),
                "invalid decision answer: the probabilities sum to 0.5, not 1 within 0.000001",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }

    /// The fixture README's refusal kinds ("Refusal kinds" in
    /// `integration-tests/fixtures/decision/README.md`), in its order. The
    /// decision conformance tests also read the table itself.
    const README_KINDS: [&str; 24] = [
        "invalid_json",
        "invalid_request",
        "duplicate_key",
        "unknown_field",
        "missing_state",
        "invalid_questions",
        "invalid_question",
        "missing_instructions",
        "invalid_question_type",
        "invalid_criteria",
        "level_count",
        "too_many_options",
        "empty_question_id",
        "empty_option_key",
        "media_unsupported",
        "spoken_instructions",
        "json_too_deep",
        "number_out_of_range",
        "escaped_key_collision",
        "no_single_token_label",
        "budget_exceeded",
        "non_finite_scores",
        "unsupported_metadata",
        "invalid_metadata",
    ];

    #[test]
    fn refusal_kinds_are_spelled_as_the_cases_spell_them() {
        let names: Vec<&str> = DecisionRefusalKind::ALL
            .iter()
            .map(|kind| kind.as_str())
            .collect();
        assert_eq!(names[..README_KINDS.len()], README_KINDS);
        assert_eq!(names[README_KINDS.len()..], ["request_too_large"]);
        for kind in DecisionRefusalKind::ALL {
            assert_eq!(kind.to_string(), kind.as_str());
        }
    }

    /// One error of every variant this module defines, with its kind and
    /// the runtime category it maps to.
    fn every_variant() -> Vec<(DecisionError, Option<&'static str>, &'static str)> {
        let part = ContentPart::OptionDescription {
            question: 1,
            option: 2,
        };
        vec![
            (
                DecisionError::InvalidJson {
                    kind: JsonErrorKind::ExpectedValue,
                    offset: 0,
                },
                Some("invalid_json"),
                "InvalidInput",
            ),
            (
                DecisionError::InvalidJson {
                    kind: JsonErrorKind::DuplicateKey,
                    offset: 3,
                },
                Some("duplicate_key"),
                "InvalidInput",
            ),
            (
                DecisionError::InvalidJson {
                    kind: JsonErrorKind::NumberOutOfRange,
                    offset: 3,
                },
                Some("number_out_of_range"),
                "InvalidInput",
            ),
            (
                DecisionError::JsonTooLong { len: 9, max: 8 },
                Some("request_too_large"),
                "InvalidInput",
            ),
            (
                DecisionError::JsonTooManyNodes { max: 4 },
                Some("request_too_large"),
                "InvalidInput",
            ),
            (
                DecisionError::JsonTooDeep { max: 128 },
                Some("json_too_deep"),
                "InvalidInput",
            ),
            (
                DecisionError::NoQuestions,
                Some("invalid_questions"),
                "InvalidInput",
            ),
            (
                DecisionError::TooManyQuestions { count: 17, max: 16 },
                Some("request_too_large"),
                "InvalidInput",
            ),
            (
                DecisionError::EmptyQuestionId { question: 0 },
                Some("empty_question_id"),
                "InvalidInput",
            ),
            (
                DecisionError::DuplicateQuestionId {
                    question: 1,
                    first: 0,
                },
                Some("duplicate_key"),
                "InvalidInput",
            ),
            (
                DecisionError::NoOptions { question: 0 },
                Some("invalid_criteria"),
                "InvalidInput",
            ),
            (
                DecisionError::TooManyOptions {
                    question: 0,
                    count: 256,
                    max: 255,
                },
                Some("too_many_options"),
                "InvalidInput",
            ),
            (
                DecisionError::EmptyOptionKey {
                    question: 0,
                    option: 1,
                },
                Some("empty_option_key"),
                "InvalidInput",
            ),
            (
                DecisionError::DuplicateOptionKey {
                    question: 0,
                    option: 2,
                    first: 1,
                },
                Some("duplicate_key"),
                "InvalidInput",
            ),
            (
                DecisionError::NoLevels { question: 0 },
                Some("level_count"),
                "InvalidInput",
            ),
            (
                DecisionError::TooManyLevels {
                    question: 0,
                    count: 300,
                    max: 255,
                },
                Some("level_count"),
                "InvalidInput",
            ),
            (
                DecisionError::RequestTooLarge { bytes: 10, max: 9 },
                Some("request_too_large"),
                "InvalidInput",
            ),
            (
                DecisionError::TooManyJsonNodes { count: 10, max: 9 },
                Some("request_too_large"),
                "InvalidInput",
            ),
            (
                DecisionError::InvalidAnswer(AnswerInvariant::ChoiceNotHighest),
                None,
                "InferenceFailed",
            ),
            (
                DecisionError::MissingInstructions { question: 2 },
                Some("missing_instructions"),
                "InvalidInput",
            ),
            (
                DecisionError::LevelCount {
                    question: 0,
                    count: 11,
                    min: 2,
                    max: 10,
                },
                Some("level_count"),
                "InvalidInput",
            ),
            (
                DecisionError::MediaUnsupported,
                Some("media_unsupported"),
                "InvalidInput",
            ),
            (
                DecisionError::ContentTooDeep {
                    part,
                    depth: 126,
                    max: 125,
                },
                Some("json_too_deep"),
                "InvalidInput",
            ),
            (
                DecisionError::NumberOutOfRange { part },
                Some("number_out_of_range"),
                "InvalidInput",
            ),
            (
                DecisionError::EscapedKeyCollision { part },
                Some("escaped_key_collision"),
                "InvalidInput",
            ),
            (
                DecisionError::NoSingleTokenLabel {
                    question: 0,
                    option: 254,
                },
                Some("no_single_token_label"),
                "InvalidInput",
            ),
            (
                DecisionError::BudgetExceeded {
                    question: 1,
                    tokens: 4097,
                    budget: 4096,
                },
                Some("budget_exceeded"),
                "InvalidInput",
            ),
            (
                DecisionError::TotalBudgetExceeded {
                    tokens: 16400,
                    max: 16384,
                },
                Some("budget_exceeded"),
                "InvalidInput",
            ),
            (
                DecisionError::UnsupportedMetadata,
                Some("unsupported_metadata"),
                "UnsupportedModelCapability",
            ),
            (
                DecisionError::InvalidMetadata,
                Some("invalid_metadata"),
                "RuntimeError",
            ),
            (
                DecisionError::UnsupportedTemplate,
                Some("unsupported_metadata"),
                "UnsupportedModelCapability",
            ),
            (
                DecisionError::ProfileMismatch,
                Some("unsupported_metadata"),
                "UnsupportedModelCapability",
            ),
            (
                DecisionError::UnsupportedVocabulary,
                Some("unsupported_metadata"),
                "UnsupportedModelCapability",
            ),
            (DecisionError::RenderFailed, None, "InferenceFailed"),
            (DecisionError::TokenizationFailed, None, "InferenceFailed"),
            (
                DecisionError::ReadoutCount {
                    expected: 2,
                    found: 1,
                },
                None,
                "InferenceFailed",
            ),
            (
                DecisionError::ReadoutLength {
                    pass: 0,
                    expected: 5,
                    found: 4,
                },
                None,
                "InferenceFailed",
            ),
            (
                DecisionError::NonFiniteScores { question: 3 },
                Some("non_finite_scores"),
                "InferenceFailed",
            ),
            (DecisionError::VocabularyFailed, None, "InferenceFailed"),
        ]
    }

    fn category(error: &AdapterError) -> &'static str {
        match error {
            AdapterError::InvalidInput(_) => "InvalidInput",
            AdapterError::InferenceFailed(_) => "InferenceFailed",
            AdapterError::RuntimeError(_) => "RuntimeError",
            AdapterError::UnsupportedModelCapability { .. } => "UnsupportedModelCapability",
            other => panic!("unexpected category {other:?}"),
        }
    }

    #[test]
    fn every_refusal_has_its_kind_and_category() {
        for (error, kind, expected) in every_variant() {
            let message = error.to_string();
            assert_eq!(
                error.kind().map(DecisionRefusalKind::as_str),
                kind,
                "{message}"
            );
            let adapted = AdapterError::from(error);
            assert_eq!(category(&adapted), expected, "{message}");
            // The runtime error says what this error says.
            assert!(
                adapted.to_string().contains(&message) || expected == "UnsupportedModelCapability",
                "{adapted}"
            );
        }
    }

    #[test]
    fn unsupported_models_name_the_model_when_it_is_known() {
        let error = DecisionError::UnsupportedTemplate.into_adapter_error("d1-2b");
        let AdapterError::UnsupportedModelCapability {
            model_id,
            capability,
            hint,
        } = error
        else {
            panic!("{error:?}");
        };
        assert_eq!(model_id, "d1-2b");
        assert_eq!(capability, "the lfm3-d1/text/v1 decision profile");
        assert_eq!(
            hint,
            "its decision template is not one this build implements"
        );
        let unknown = AdapterError::from(DecisionError::UnsupportedMetadata);
        assert!(
            matches!(&unknown, AdapterError::UnsupportedModelCapability { model_id, .. } if model_id == "unknown"),
            "{unknown:?}"
        );
    }

    #[test]
    fn protocol_messages_name_parts_positions_and_counts() {
        let cases: [(DecisionError, &str); 9] = [
            (
                DecisionError::MissingInstructions { question: 2 },
                "question 2 has no instructions, which this model needs",
            ),
            (
                DecisionError::LevelCount {
                    question: 0,
                    count: 1,
                    min: 2,
                    max: 10,
                },
                "score question 0 has 1 levels; this model reads 2 to 10",
            ),
            (
                DecisionError::ContentTooDeep {
                    part: ContentPart::State,
                    depth: 128,
                    max: 127,
                },
                "JSON in the state nests 128 levels deep; this model reads at most 127",
            ),
            (
                DecisionError::NumberOutOfRange {
                    part: ContentPart::Instructions { question: 1 },
                },
                "an integer in the instructions of question 1 is above 9223372036854775807, which this model cannot read",
            ),
            (
                DecisionError::EscapedKeyCollision {
                    part: ContentPart::YesNoDescription {
                        question: 0,
                        yes: false,
                    },
                },
                "two keys of one object in the description of no of question 0 become equal once special-token spellings are escaped",
            ),
            (
                DecisionError::EscapedKeyCollision {
                    part: ContentPart::LevelDescription {
                        question: 4,
                        level: 3,
                    },
                },
                "two keys of one object in the description of level 3 of question 4 become equal once special-token spellings are escaped",
            ),
            (
                DecisionError::BudgetExceeded {
                    question: 1,
                    tokens: 4097,
                    budget: 4096,
                },
                "question 1 renders to 4097 tokens; the token budget is 4096",
            ),
            (
                DecisionError::ReadoutLength {
                    pass: 0,
                    expected: 5,
                    found: 4,
                },
                "readout 0 has 4 values; its pass selects 5",
            ),
            (
                DecisionError::NonFiniteScores { question: 3 },
                "the model's scores for question 3 are not finite",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }
}

//! A SystemOne request body to a [`DecisionRequest`]: the SystemOne JSON
//! subset v1.
//!
//! See the [module docs](super) for the subset and the order of the checks.

use super::body::{self, BodyError, Value};
use std::fmt;
use xybrid_core::ir::decision::{JsonLimits, DEFAULT_MAX_OPTIONS, DEFAULT_MAX_REQUEST_JSON_NODES};
use xybrid_core::ir::{
    DecisionContent, DecisionError, DecisionInstructions, DecisionLevel, DecisionOption,
    DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionState, JsonErrorKind,
    JsonNumber, JsonView, OrderedJson,
};

/// Default [`RequestLimits::max_body_bytes`]: 8 MiB.
///
/// A request's content may hold 1 MiB under
/// [`StructuralLimits`](xybrid_core::ir::decision::StructuralLimits)'
/// defaults; the body adds the request's own objects, indentation and
/// escapes (`\u0001` is six bytes for one), and 8 MiB leaves room for them.
/// A token budget refuses far smaller requests anyway.
pub const DEFAULT_MAX_BODY_BYTES: usize = 8 << 20;

/// Default [`RequestLimits::max_body_values`]: 131,072, twice the JSON values
/// [`StructuralLimits`](xybrid_core::ir::decision::StructuralLimits)
/// accepts in a request's content, which leaves room for the request's own
/// objects, keys and strings.
pub const DEFAULT_MAX_BODY_VALUES: usize = 2 * DEFAULT_MAX_REQUEST_JSON_NODES;

/// The highest JSON depth content may have here, counting the value itself
/// (a scalar is 1). Content is an [`OrderedJson`], which holds no more; the
/// protocol's own limits are lower and are checked when it prepares the
/// request.
const MAX_CONTENT_DEPTH: usize = xybrid_core::ir::decision::MAX_JSON_DEPTH;

/// Fewest and most levels of a score question (llama.cpp's
/// `parse_questions`, every decision type but h2o-lightning).
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;

/// Ceilings on reading one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestLimits {
    /// Most bytes in a request body.
    pub max_body_bytes: usize,
    /// Most JSON values in a request body, counted as
    /// [`JsonLimits::max_nodes`] counts them.
    pub max_body_values: usize,
    /// Most options of a choice question: LFM3 d1's ceiling in llama.cpp
    /// (`n_options_max`), 255.
    pub max_options: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_body_values: DEFAULT_MAX_BODY_VALUES,
            max_options: DEFAULT_MAX_OPTIONS,
        }
    }
}

/// A media field of the request, or a media part of the state's messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaAt {
    /// The request's `videos` field.
    Videos,
    /// The request's `files` field.
    Files,
    /// The request's `images` field.
    Images,
    /// An `image_url`, `input_audio` or `video_url` part of a chat message
    /// in the state.
    StatePart {
        /// Position of the message.
        message: usize,
        /// Position of the part in the message's content.
        part: usize,
    },
}

/// A member the subset does not define, by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldAt {
    /// A member of the request.
    Request {
        /// Position of the member.
        field: usize,
    },
    /// A member of a question.
    Question {
        /// Position of the question.
        question: usize,
        /// Position of the member in the question.
        field: usize,
    },
    /// A member of a noul question's criteria.
    NoulCriteria {
        /// Position of the question.
        question: usize,
        /// Position of the member in the criteria.
        field: usize,
    },
}

/// Where a piece of content sits in the request. Positions count from 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentAt {
    /// The state.
    State,
    /// The instructions of a question.
    Instructions {
        /// Position of the question.
        question: usize,
    },
    /// The description of an option of a choice question.
    Option {
        /// Position of the question.
        question: usize,
        /// Position of the option.
        option: usize,
    },
    /// The description of `true` of a noul question.
    Yes {
        /// Position of the question.
        question: usize,
    },
    /// The description of `false` of a noul question.
    No {
        /// Position of the question.
        question: usize,
    },
    /// The description of a level of a score question.
    Level {
        /// Position of the question.
        question: usize,
        /// The level.
        level: usize,
    },
}

/// What a question's `criteria` must be, for its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CriteriaShape {
    /// A choice: a non-empty object.
    Choice,
    /// A score: an array.
    Score,
    /// A noul: null, absent, or an object.
    Noul,
}

/// The kind of a parse-stage refusal, named as the decision fixtures name
/// it (`integration-tests/fixtures/decision/README.md`).
///
/// Code matches kinds, never messages. These are the kinds a request can
/// be refused with at parse; the fixtures' later kinds (`budget_exceeded`,
/// `non_finite_scores`, ...) belong to the runtime. One kind is the
/// transport's own and in no fixture: [`RefusalKind::RequestTooLarge`], for
/// its ceilings on what one request may make it read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefusalKind {
    /// `invalid_json`: the body is not one JSON value.
    InvalidJson,
    /// `invalid_request`: the body is JSON but not an object.
    InvalidRequest,
    /// `duplicate_key`: an object repeats a key.
    DuplicateKey,
    /// `unknown_field`: a field the subset does not define.
    UnknownField,
    /// `missing_state`: no `state`.
    MissingState,
    /// `invalid_questions`: `questions` missing, not an object, or empty.
    InvalidQuestions,
    /// `invalid_question`: a question that is not an object.
    InvalidQuestion,
    /// `missing_instructions`: `instructions` missing or null.
    MissingInstructions,
    /// `invalid_question_type`: `type` missing, not a string, or unknown.
    InvalidQuestionType,
    /// `invalid_criteria`: criteria of the wrong JSON type for the question.
    InvalidCriteria,
    /// `level_count`: a score with fewer than 2 or more than 10 levels.
    LevelCount,
    /// `too_many_options`: more options than the model takes.
    TooManyOptions,
    /// `empty_question_id`: a question id that is the empty string.
    EmptyQuestionId,
    /// `empty_option_key`: an option key that is the empty string.
    EmptyOptionKey,
    /// `media_unsupported`: media fields or media parts.
    MediaUnsupported,
    /// `spoken_instructions`: instructions that are an audio part.
    SpokenInstructions,
    /// `json_too_deep`: content nested beyond the limit.
    JsonTooDeep,
    /// `number_out_of_range`: a number the model cannot read faithfully.
    NumberOutOfRange,
    /// `request_too_large`: more bytes or JSON values than the transport
    /// reads ([`RequestLimits`], and the value ceiling of one piece of
    /// content). The transport's own kind: no fixture case reaches it.
    RequestTooLarge,
}

impl RefusalKind {
    /// The kind's name in the fixtures, such as `"unknown_field"`.
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
            Self::RequestTooLarge => "request_too_large",
        }
    }
}

impl fmt::Display for RefusalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a SystemOne request was refused before anything ran.
///
/// Messages name byte offsets, positions (counting from 0), counts and the
/// subset's own field names only: never a question id, an option key, a
/// field the caller made up, or any part of the content.
/// [`RequestError::kind`] names every refusal's [`RefusalKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestError {
    /// The body is longer than the transport reads.
    TooLarge {
        /// Bytes in the body.
        len: usize,
        /// [`RequestLimits::max_body_bytes`].
        max: usize,
    },
    /// The body holds more JSON values than the transport reads.
    TooManyValues {
        /// [`RequestLimits::max_body_values`].
        max: usize,
    },
    /// The body is not UTF-8.
    NotUtf8 {
        /// Where the first byte that is not UTF-8 starts.
        offset: usize,
    },
    /// The body is not one JSON value.
    InvalidJson {
        /// What is wrong.
        kind: JsonErrorKind,
        /// Where, in bytes.
        offset: usize,
    },
    /// An object repeats a key.
    DuplicateKey {
        /// Where the first repeat starts, in bytes.
        offset: usize,
    },
    /// A number is beyond the range of `f64`.
    NumberOverflow {
        /// Where the number starts, in bytes.
        offset: usize,
    },
    /// The body is JSON, but not an object.
    NotAnObject,
    /// The request has no `state` (a null state is one).
    MissingState,
    /// `questions` is missing, not an object, or empty.
    InvalidQuestions,
    /// A question is not an object.
    InvalidQuestion {
        /// Position of the question.
        question: usize,
    },
    /// A question's `instructions` are missing or null.
    MissingInstructions {
        /// Position of the question.
        question: usize,
    },
    /// A question's instructions are an `input_audio` part.
    SpokenInstructions {
        /// Position of the question.
        question: usize,
    },
    /// A question's `type` is missing, not a string, or not choice, noul or
    /// score.
    InvalidQuestionType {
        /// Position of the question.
        question: usize,
    },
    /// A question's `criteria` have the wrong JSON type for its type.
    InvalidCriteria {
        /// Position of the question.
        question: usize,
        /// What its type needs.
        expected: CriteriaShape,
    },
    /// A score question has fewer than 2 or more than 10 levels.
    LevelCount {
        /// Position of the question.
        question: usize,
        /// Its number of levels.
        count: usize,
    },
    /// A choice question has more options than the model takes.
    TooManyOptions {
        /// Position of the question.
        question: usize,
        /// Its number of options.
        count: usize,
        /// [`RequestLimits::max_options`].
        max: usize,
    },
    /// The request carries media, which the text profile does not take.
    MediaUnsupported(MediaAt),
    /// A member the subset does not define.
    UnknownField(FieldAt),
    /// A question id is the empty string.
    EmptyQuestionId {
        /// Position of the question.
        question: usize,
    },
    /// An option key is the empty string.
    EmptyOptionKey {
        /// Position of the question.
        question: usize,
        /// Position of the option.
        option: usize,
    },
    /// Content holds an integer above `i64::MAX`, which llama.cpp's
    /// template engine reads wrapped to a negative number.
    IntegerOutOfRange(ContentAt),
    /// Content nests deeper than an [`OrderedJson`] can.
    TooDeep(ContentAt),
    /// One piece of content holds more JSON values than an [`OrderedJson`]
    /// read here may.
    ContentTooManyValues {
        /// Which content.
        at: ContentAt,
        /// The ceiling.
        max: usize,
    },
    /// Content that [`OrderedJson::parse`] refuses although the body was
    /// read as JSON: the two grammars disagree (they are tested to agree).
    InvalidContent(ContentAt),
}

impl RequestError {
    /// The refusal's kind: as the decision fixtures name it, or
    /// [`RefusalKind::RequestTooLarge`] for the transport's own ceilings
    /// ([`RequestError::TooLarge`], [`RequestError::TooManyValues`],
    /// [`RequestError::ContentTooManyValues`]).
    pub fn kind(&self) -> RefusalKind {
        match self {
            Self::TooLarge { .. }
            | Self::TooManyValues { .. }
            | Self::ContentTooManyValues { .. } => RefusalKind::RequestTooLarge,
            Self::NotUtf8 { .. } | Self::InvalidJson { .. } | Self::InvalidContent(_) => {
                RefusalKind::InvalidJson
            }
            Self::DuplicateKey { .. } => RefusalKind::DuplicateKey,
            Self::NumberOverflow { .. } | Self::IntegerOutOfRange(_) => {
                RefusalKind::NumberOutOfRange
            }
            Self::NotAnObject => RefusalKind::InvalidRequest,
            Self::MissingState => RefusalKind::MissingState,
            Self::InvalidQuestions => RefusalKind::InvalidQuestions,
            Self::InvalidQuestion { .. } => RefusalKind::InvalidQuestion,
            Self::MissingInstructions { .. } => RefusalKind::MissingInstructions,
            Self::SpokenInstructions { .. } => RefusalKind::SpokenInstructions,
            Self::InvalidQuestionType { .. } => RefusalKind::InvalidQuestionType,
            Self::InvalidCriteria { .. } => RefusalKind::InvalidCriteria,
            Self::LevelCount { .. } => RefusalKind::LevelCount,
            Self::TooManyOptions { .. } => RefusalKind::TooManyOptions,
            Self::MediaUnsupported(_) => RefusalKind::MediaUnsupported,
            Self::UnknownField(_) => RefusalKind::UnknownField,
            Self::EmptyQuestionId { .. } => RefusalKind::EmptyQuestionId,
            Self::EmptyOptionKey { .. } => RefusalKind::EmptyOptionKey,
            Self::TooDeep(_) => RefusalKind::JsonTooDeep,
        }
    }

    /// The refusal for a body that is not JSON the transport reads.
    fn from_body(error: BodyError) -> Self {
        match error {
            BodyError::Json { kind, offset } => Self::from_json(kind, offset),
            BodyError::TooManyValues { max } => Self::TooManyValues { max },
        }
    }

    fn from_json(kind: JsonErrorKind, offset: usize) -> Self {
        match kind {
            JsonErrorKind::DuplicateKey => Self::DuplicateKey { offset },
            JsonErrorKind::NumberOutOfRange => Self::NumberOverflow { offset },
            kind => Self::InvalidJson { kind, offset },
        }
    }
}

impl fmt::Display for MediaAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Videos => f.write_str("the request has a \"videos\" field"),
            Self::Files => f.write_str("the request has a \"files\" field"),
            Self::Images => f.write_str("the request has an \"images\" field"),
            Self::StatePart { message, part } => write!(
                f,
                "part {part} of message {message} of the state is an image, audio or video part"
            ),
        }
    }
}

impl fmt::Display for FieldAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request { field } => write!(
                f,
                "member {field} of the request is not a field of the subset (state, questions, \
                 model)"
            ),
            Self::Question { question, field } => write!(
                f,
                "member {field} of question {question} is not a question field (type, \
                 instructions, criteria)"
            ),
            Self::NoulCriteria { question, field } => write!(
                f,
                "member {field} of the criteria of question {question} is not \"true\" or \
                 \"false\", the only answers a noul question describes"
            ),
        }
    }
}

impl fmt::Display for ContentAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State => f.write_str("the state"),
            Self::Instructions { question } => {
                write!(f, "the instructions of question {question}")
            }
            Self::Option { question, option } => {
                write!(
                    f,
                    "the description of option {option} of question {question}"
                )
            }
            Self::Yes { question } => {
                write!(f, "the description of \"true\" of question {question}")
            }
            Self::No { question } => {
                write!(f, "the description of \"false\" of question {question}")
            }
            Self::Level { question, level } => {
                write!(f, "the description of level {level} of question {question}")
            }
        }
    }
}

impl fmt::Display for CriteriaShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Choice => "a choice question's \"criteria\" must be a non-empty object",
            Self::Score => "a score question's \"criteria\" must be an array of 2 to 10 levels",
            Self::Noul => "a noul question's \"criteria\" must be null or an object",
        })
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { len, max } => {
                write!(f, "the request is {len} bytes; at most {max} are read")
            }
            Self::TooManyValues { max } => {
                write!(f, "the request holds more than {max} JSON values")
            }
            Self::NotUtf8 { offset } => write!(f, "the request is not UTF-8 (byte {offset})"),
            Self::InvalidJson { kind, offset } => {
                write!(f, "invalid JSON at byte {offset}: {kind}")
            }
            Self::DuplicateKey { offset } => {
                write!(f, "an object repeats a key (byte {offset})")
            }
            Self::NumberOverflow { offset } => {
                write!(
                    f,
                    "a number is beyond the range of a 64-bit float (byte {offset})"
                )
            }
            Self::NotAnObject => f.write_str("the request is not a JSON object"),
            Self::MissingState => {
                f.write_str("the request has no \"state\" (use null for no state)")
            }
            Self::InvalidQuestions => f.write_str("\"questions\" must be a non-empty object"),
            Self::InvalidQuestion { question } => {
                write!(f, "question {question} is not an object")
            }
            Self::MissingInstructions { question } => {
                write!(f, "question {question} has no \"instructions\"")
            }
            Self::SpokenInstructions { question } => write!(
                f,
                "the instructions of question {question} are spoken (an input_audio part), \
                 which needs a model with audio input"
            ),
            Self::InvalidQuestionType { question } => write!(
                f,
                "question {question}: \"type\" must be one of choice, score, noul"
            ),
            Self::InvalidCriteria { question, expected } => {
                write!(f, "question {question}: {expected}")
            }
            Self::LevelCount { question, count } => write!(
                f,
                "question {question}: a score question takes 2 to 10 levels, not {count}"
            ),
            Self::TooManyOptions {
                question,
                count,
                max,
            } => write!(
                f,
                "question {question} has {count} options; the model takes at most {max}"
            ),
            Self::MediaUnsupported(at) => {
                write!(f, "{at}; decision requests take no media in this version")
            }
            Self::UnknownField(at) => at.fmt(f),
            Self::EmptyQuestionId { question } => {
                write!(f, "question {question} has an empty id")
            }
            Self::EmptyOptionKey { question, option } => {
                write!(f, "option {option} of question {question} has an empty key")
            }
            Self::IntegerOutOfRange(at) => write!(
                f,
                "{at} holds an integer above {}, which the model cannot read faithfully",
                i64::MAX
            ),
            Self::TooDeep(at) => {
                write!(
                    f,
                    "{at} is nested more than {MAX_CONTENT_DEPTH} levels deep"
                )
            }
            Self::ContentTooManyValues { at, max } => {
                write!(f, "{at} holds more than {max} JSON values")
            }
            Self::InvalidContent(at) => write!(f, "{at} could not be read as JSON"),
        }
    }
}

impl std::error::Error for RequestError {}

/// Reads a SystemOne request (subset v1) into a [`DecisionRequest`], with
/// questions in request order.
///
/// See the [module docs](super) for the subset and the order of the checks.
/// The result still needs
/// [`validate_structure`](xybrid_core::ir::decision::validate_structure),
/// and the model's protocol checks it again before anything runs.
///
/// # Errors
///
/// The first [`RequestError`] in the documented order.
pub fn parse_request(body: &[u8], limits: &RequestLimits) -> Result<DecisionRequest, RequestError> {
    if body.len() > limits.max_body_bytes {
        return Err(RequestError::TooLarge {
            len: body.len(),
            max: limits.max_body_bytes,
        });
    }
    let text = std::str::from_utf8(body).map_err(|error| RequestError::NotUtf8 {
        offset: error.valid_up_to(),
    })?;
    let root = body::read(text, limits.max_body_values).map_err(RequestError::from_body)?;
    Reader { text, limits }.request(&root)
}

/// The members a question may have.
const QUESTION_FIELDS: [&str; 3] = ["type", "instructions", "criteria"];

/// A question's alternatives, as its type reads them from its `criteria`.
enum Alternatives<'v> {
    /// A choice's options: the criteria's members, in order.
    Choice(&'v [(String, Value)]),
    /// A noul's descriptions: the criteria's members, if it has criteria.
    Noul(&'v [(String, Value)]),
    /// A score's levels: the criteria's items, in order.
    Score(&'v [Value]),
}

/// A question as llama.cpp's checks read it, before its content is parsed.
struct RawQuestion<'v> {
    id: &'v str,
    question: &'v Value,
    instructions: &'v Value,
    alternatives: Alternatives<'v>,
}

struct Reader<'a> {
    text: &'a str,
    limits: &'a RequestLimits,
}

impl Reader<'_> {
    fn request(&self, root: &Value) -> Result<DecisionRequest, RequestError> {
        let Some(fields) = root.as_object() else {
            return Err(RequestError::NotAnObject);
        };

        // The steps of the module docs; step 1, the body as JSON, is done.
        // 2. llama.cpp's own checks, in its order (`parse_questions`, then
        //    `parse_state` and the media check), so a request it refuses is
        //    refused here for the same reason.
        let state = root.get("state").ok_or(RequestError::MissingState)?;
        let questions = match root.get("questions").and_then(Value::as_object) {
            Some(questions) if !questions.is_empty() => questions,
            _ => return Err(RequestError::InvalidQuestions),
        };
        let questions = questions
            .iter()
            .enumerate()
            .map(|(index, (id, question))| self.read_question(index, id, question))
            .collect::<Result<Vec<_>, _>>()?;
        check_media(root, state)?;

        // 3. What llama.cpp accepts and the subset does not: fields it
        //    ignores, empty ids and keys.
        if let Some(field) = fields
            .iter()
            .position(|(name, _)| !matches!(name.as_str(), "state" | "questions" | "model"))
        {
            return Err(RequestError::UnknownField(FieldAt::Request { field }));
        }
        for (index, read) in questions.iter().enumerate() {
            check_question_shape(index, read)?;
        }

        // 4. The content, in request order.
        let state = DecisionState::from_json(self.content(state, ContentAt::State)?);
        let questions = questions
            .iter()
            .enumerate()
            .map(|(index, read)| self.question(index, read))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DecisionRequest::new(state, questions))
    }

    /// llama.cpp's checks of one question, in its order: an object, with
    /// instructions that are not spoken, a type, criteria of the type's
    /// shape, and no more options than the model takes.
    fn read_question<'v>(
        &self,
        index: usize,
        id: &'v str,
        question: &'v Value,
    ) -> Result<RawQuestion<'v>, RequestError> {
        if question.as_object().is_none() {
            return Err(RequestError::InvalidQuestion { question: index });
        }
        let instructions = question
            .get("instructions")
            .filter(|value| !value.is_null())
            .ok_or(RequestError::MissingInstructions { question: index })?;
        if is_spoken(instructions) {
            return Err(RequestError::SpokenInstructions { question: index });
        }
        // `json_value(q, "type", "")`: a type that is not a string reads as
        // no type. Null criteria read as absent ones.
        let criteria = question.get("criteria").filter(|value| !value.is_null());
        let invalid = |expected| RequestError::InvalidCriteria {
            question: index,
            expected,
        };
        let alternatives = match question.get("type").and_then(Value::as_str) {
            Some("choice") => match criteria.and_then(Value::as_object) {
                Some(options) if !options.is_empty() => Alternatives::Choice(options),
                _ => return Err(invalid(CriteriaShape::Choice)),
            },
            Some("score") => match criteria.map(|criteria| criteria.as_array()) {
                Some(Some(levels)) if SCORE_LEVELS.contains(&levels.len()) => {
                    Alternatives::Score(levels)
                }
                Some(Some(levels)) => {
                    return Err(RequestError::LevelCount {
                        question: index,
                        count: levels.len(),
                    })
                }
                _ => return Err(invalid(CriteriaShape::Score)),
            },
            Some("noul") => match criteria.map(|criteria| criteria.as_object()) {
                None => Alternatives::Noul(&[]),
                Some(Some(members)) => Alternatives::Noul(members),
                Some(None) => return Err(invalid(CriteriaShape::Noul)),
            },
            _ => return Err(RequestError::InvalidQuestionType { question: index }),
        };
        if let Alternatives::Choice(options) = alternatives {
            if options.len() > self.limits.max_options {
                return Err(RequestError::TooManyOptions {
                    question: index,
                    count: options.len(),
                    max: self.limits.max_options,
                });
            }
        }
        Ok(RawQuestion {
            id,
            question,
            instructions,
            alternatives,
        })
    }

    /// The IR of a question that passed every check of its shape.
    fn question(
        &self,
        index: usize,
        read: &RawQuestion<'_>,
    ) -> Result<DecisionQuestion, RequestError> {
        let instructions = self.content(
            read.instructions,
            ContentAt::Instructions { question: index },
        )?;
        let kind = match read.alternatives {
            Alternatives::Choice(options) => DecisionQuestionKind::choice(
                options
                    .iter()
                    .enumerate()
                    .map(|(option, (key, description))| {
                        let at = ContentAt::Option {
                            question: index,
                            option,
                        };
                        Ok(DecisionOption::new(
                            key.as_str(),
                            self.description(description, at)?,
                        ))
                    })
                    .collect::<Result<_, RequestError>>()?,
            ),
            Alternatives::Noul(members) => {
                let description = |name: &str, at| {
                    members
                        .iter()
                        .find(|(key, _)| key == name)
                        .map_or(Ok(None), |(_, value)| self.description(value, at))
                };
                DecisionQuestionKind::yes_no(
                    description("true", ContentAt::Yes { question: index })?,
                    description("false", ContentAt::No { question: index })?,
                )
            }
            Alternatives::Score(levels) => DecisionQuestionKind::score(
                levels
                    .iter()
                    .enumerate()
                    .map(|(level, description)| {
                        let at = ContentAt::Level {
                            question: index,
                            level,
                        };
                        Ok(DecisionLevel::new(self.description(description, at)?))
                    })
                    .collect::<Result<_, RequestError>>()?,
            ),
        };
        Ok(DecisionQuestion::new(
            read.id,
            DecisionContent::from_json(instructions).map(DecisionInstructions::from),
            kind,
        ))
    }

    /// A description: absent for JSON null.
    fn description(
        &self,
        value: &Value,
        at: ContentAt,
    ) -> Result<Option<DecisionContent>, RequestError> {
        Ok(DecisionContent::from_json(self.content(value, at)?))
    }

    /// A piece of content, read from its text by [`OrderedJson::parse`].
    fn content(&self, value: &Value, at: ContentAt) -> Result<OrderedJson, RequestError> {
        // No byte ceiling beyond the body's own; the default value ceiling,
        // so the content decodes again from its serde form.
        let limits = JsonLimits::default().with_max_bytes(usize::MAX);
        let json = OrderedJson::parse_with_limits(&self.text[value.span.clone()], &limits)
            .map_err(|error| match error {
                DecisionError::JsonTooDeep { .. } => RequestError::TooDeep(at),
                DecisionError::JsonTooManyNodes { max } => {
                    RequestError::ContentTooManyValues { at, max }
                }
                _ => RequestError::InvalidContent(at),
            })?;
        if holds_u64(&json) {
            return Err(RequestError::IntegerOutOfRange(at));
        }
        Ok(json)
    }
}

/// The checks of the subset on one question's shape, in reading order:
/// a non-empty id, only the question fields, and non-empty option keys or
/// noul criteria that describe `true` and `false` only.
fn check_question_shape(index: usize, read: &RawQuestion<'_>) -> Result<(), RequestError> {
    if read.id.is_empty() {
        return Err(RequestError::EmptyQuestionId { question: index });
    }
    if let Some(field) = read.question.as_object().and_then(|fields| {
        fields
            .iter()
            .position(|(name, _)| !QUESTION_FIELDS.contains(&name.as_str()))
    }) {
        return Err(RequestError::UnknownField(FieldAt::Question {
            question: index,
            field,
        }));
    }
    match read.alternatives {
        Alternatives::Choice(options) => {
            if let Some(option) = options.iter().position(|(key, _)| key.is_empty()) {
                return Err(RequestError::EmptyOptionKey {
                    question: index,
                    option,
                });
            }
        }
        Alternatives::Noul(members) => {
            if let Some(field) = members
                .iter()
                .position(|(name, _)| name != "true" && name != "false")
            {
                return Err(RequestError::UnknownField(FieldAt::NoulCriteria {
                    question: index,
                    field,
                }));
            }
        }
        Alternatives::Score(_) => {}
    }
    Ok(())
}

/// llama.cpp's `decision_is_spoken`: instructions that are an object whose
/// `type` is the string `"input_audio"`.
fn is_spoken(instructions: &Value) -> bool {
    instructions.get("type").and_then(Value::as_str) == Some("input_audio")
}

/// Media in the request, as llama.cpp's `parse_state` finds it: a `videos`,
/// `files` or `images` field (any value: the subset takes none, not even an
/// empty list), or an image, audio or video part in the content of a chat
/// message of the state. Messages are the state itself when it is an array,
/// or its `messages` member when it is an object that has one.
fn check_media(root: &Value, state: &Value) -> Result<(), RequestError> {
    for (field, at) in [
        ("videos", MediaAt::Videos),
        ("files", MediaAt::Files),
        ("images", MediaAt::Images),
    ] {
        if root.get(field).is_some() {
            return Err(RequestError::MediaUnsupported(at));
        }
    }
    let Some(messages) = state.get("messages").unwrap_or(state).as_array() else {
        return Ok(());
    };
    for (message, entry) in messages.iter().enumerate() {
        let Some(parts) = entry.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (part, entry) in parts.iter().enumerate() {
            let media = match entry.get("type").and_then(Value::as_str) {
                Some(kind @ ("image_url" | "input_audio" | "video_url")) => {
                    entry.get(kind).is_some()
                }
                _ => false,
            };
            if media {
                return Err(RequestError::MediaUnsupported(MediaAt::StatePart {
                    message,
                    part,
                }));
            }
        }
    }
    Ok(())
}

/// Whether `value` holds an integer above `i64::MAX` anywhere.
fn holds_u64(value: &OrderedJson) -> bool {
    match value.view() {
        JsonView::Number(JsonNumber::U64(_)) => true,
        JsonView::Array(items) => items.iter().any(holds_u64),
        JsonView::Object(members) => members.iter().any(|(_, value)| holds_u64(value)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_core::ir::DecisionContentView;

    const CANARY: &str = "canary-5c1e07";

    fn parse(text: &str) -> Result<DecisionRequest, RequestError> {
        parse_request(text.as_bytes(), &RequestLimits::default())
    }

    fn refused(text: &str) -> RequestError {
        parse(text).expect_err(text)
    }

    /// The name of `error`'s kind.
    fn kind(error: RequestError) -> &'static str {
        error.kind().as_str()
    }

    fn json(text: &str) -> OrderedJson {
        OrderedJson::parse(text).unwrap()
    }

    /// A request with `state` and one question whose members are `question`.
    fn one(state: &str, question: &str) -> String {
        format!(r#"{{"state": {state}, "questions": {{"q": {question}}}}}"#)
    }

    const NOUL: &str = r#"{"type": "noul", "instructions": "Refund?"}"#;

    #[test]
    fn a_full_request_keeps_order_and_content() {
        let request = parse(
            r#"{"model": "ignored", "state": {"b": 1, "a": [1.0, -0, null]},
                "questions": {
                  "zeta": {"type": "choice", "instructions": "Pick", "criteria": {"y": "Yes", "b_c": null, "x": {"k": 0.5}}},
                  "alpha": {"instructions": {"task": "t"}, "type": "noul", "criteria": {"false": "No", "true": null}},
                  "mid": {"criteria": [null, "Low", 3], "type": "score", "instructions": 7}
                }}"#,
        )
        .unwrap();

        assert_eq!(
            request.state().body().and_then(DecisionContent::as_json),
            Some(&json(r#"{"b": 1, "a": [1.0, 0, null]}"#))
        );
        let ids: Vec<&str> = request
            .questions()
            .iter()
            .map(DecisionQuestion::id)
            .collect();
        assert_eq!(ids, ["zeta", "alpha", "mid"]);

        let [zeta, alpha, mid] = request.questions() else {
            unreachable!()
        };
        let options = zeta.kind().as_choice().unwrap();
        let keys: Vec<&str> = options.iter().map(DecisionOption::key).collect();
        assert_eq!(keys, ["y", "b_c", "x"]);
        assert_eq!(
            options[0].description(),
            Some(&DecisionContent::text("Yes"))
        );
        assert_eq!(options[1].description(), None);
        assert_eq!(
            options[2].description().and_then(DecisionContent::as_json),
            Some(&json(r#"{"k": 0.5}"#))
        );
        let instructions = |question: &DecisionQuestion| {
            question
                .instructions()
                .and_then(DecisionInstructions::as_content)
                .cloned()
        };
        assert_eq!(instructions(zeta), Some(DecisionContent::text("Pick")));

        assert_eq!(
            alpha.kind().as_yes_no(),
            Some((None, Some(&DecisionContent::text("No"))))
        );
        assert_eq!(
            instructions(alpha)
                .as_ref()
                .and_then(DecisionContent::as_json),
            Some(&json(r#"{"task": "t"}"#))
        );

        let levels: Vec<Option<&DecisionContent>> = mid
            .kind()
            .as_score()
            .unwrap()
            .iter()
            .map(DecisionLevel::description)
            .collect();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], None);
        assert_eq!(levels[1], Some(&DecisionContent::text("Low")));
        assert_eq!(
            levels[2].and_then(DecisionContent::as_json),
            Some(&json("3"))
        );
        assert_eq!(
            instructions(mid)
                .as_ref()
                .and_then(DecisionContent::as_json),
            Some(&json("7"))
        );
    }

    #[test]
    fn the_state_may_be_null_empty_text_or_any_json() {
        let state = |text: &str| parse(&one(text, NOUL)).unwrap().into_parts().0;
        assert_eq!(state("null"), DecisionState::default());
        assert_eq!(state(r#""""#), DecisionState::text(""));
        assert_eq!(state(r#"" \n ""#), DecisionState::text(" \n "));
        for value in [
            "0",
            "false",
            "[]",
            "{}",
            r#"[{"role": "user", "content": "hi"}]"#,
        ] {
            assert_eq!(
                state(value),
                DecisionState::from_json(json(value)),
                "{value}"
            );
        }
    }

    #[test]
    fn noul_criteria_may_be_absent_null_empty_or_partial() {
        let yes_no = |criteria: &str| {
            let question = format!(r#"{{"type": "noul", "instructions": "Refund?"{criteria}}}"#);
            let request = parse(&one("null", &question)).unwrap();
            let (yes, no) = request.questions()[0].kind().as_yes_no().unwrap();
            (yes.cloned(), no.cloned())
        };
        assert_eq!(yes_no(""), (None, None));
        assert_eq!(yes_no(r#", "criteria": null"#), (None, None));
        assert_eq!(yes_no(r#", "criteria": {}"#), (None, None));
        assert_eq!(
            yes_no(r#", "criteria": {"true": ""}"#),
            (Some(DecisionContent::text("")), None)
        );
        assert_eq!(
            yes_no(r#", "criteria": {"false": [0]}"#),
            (None, DecisionContent::from_json(json("[0]")))
        );
    }

    #[test]
    fn falsy_choice_descriptions_stay_content_and_null_is_absent() {
        let request = parse(&one(
            "null",
            r#"{"type": "choice", "instructions": "x", "criteria": {"a": "", "b": false, "c": 0, "d": [], "e": {}, "f": null}}"#,
        ))
        .unwrap();
        let options = request.questions()[0].kind().as_choice().unwrap();
        let descriptions: Vec<Option<DecisionContentView<'_>>> = options
            .iter()
            .map(|option| option.description().map(DecisionContent::view))
            .collect();
        assert!(matches!(
            descriptions[0],
            Some(DecisionContentView::Text(""))
        ));
        for (index, text) in [(1, "false"), (2, "0"), (3, "[]"), (4, "{}")] {
            assert!(
                matches!(descriptions[index], Some(DecisionContentView::Json(value)) if *value == json(text)),
                "{text}"
            );
        }
        assert!(descriptions[5].is_none());
    }

    #[test]
    fn the_body_must_be_one_json_object() {
        for (text, expected) in [
            ("", "invalid_json"),
            ("state: refund", "invalid_json"),
            ("{} {}", "invalid_json"),
            ("{\"state\": 01}", "invalid_json"),
            ("[{\"state\": \"x\"}]", "invalid_request"),
            ("\"a request\"", "invalid_request"),
            ("null", "invalid_request"),
            ("{\"state\": 1e400}", "number_out_of_range"),
            ("{\"state\": {\"a\": 1, \"a\": 2}}", "duplicate_key"),
        ] {
            assert_eq!(kind(refused(text)), expected, "{text}");
        }
        assert_eq!(
            parse_request(b"{\"state\": \"\xc3\"}", &RequestLimits::default()),
            Err(RequestError::NotUtf8 { offset: 11 })
        );
    }

    #[test]
    fn upstream_refusals_keep_upstream_order() {
        let cases: &[(&str, RequestError)] = &[
            (r#"{"questions": {"q": {}}}"#, RequestError::MissingState),
            (r#"{"state": null}"#, RequestError::InvalidQuestions),
            (
                r#"{"state": null, "questions": []}"#,
                RequestError::InvalidQuestions,
            ),
            (
                r#"{"state": null, "questions": {}}"#,
                RequestError::InvalidQuestions,
            ),
            (
                r#"{"state": null, "questions": {"q": "text"}}"#,
                RequestError::InvalidQuestion { question: 0 },
            ),
            // Instructions are read before the type.
            (
                r#"{"state": null, "questions": {"q": {"type": "bogus"}}}"#,
                RequestError::MissingInstructions { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": null}}}"#,
                RequestError::MissingInstructions { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": 1, "instructions": {"type": "input_audio"}}}}"#,
                RequestError::SpokenInstructions { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": null, "instructions": "x"}}}"#,
                RequestError::InvalidQuestionType { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "Choice", "instructions": "x"}}}"#,
                RequestError::InvalidQuestionType { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": null}}}"#,
                RequestError::InvalidCriteria {
                    question: 0,
                    expected: CriteriaShape::Choice,
                },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "score", "instructions": "x"}}}"#,
                RequestError::InvalidCriteria {
                    question: 0,
                    expected: CriteriaShape::Score,
                },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "score", "instructions": "x", "criteria": []}}}"#,
                RequestError::LevelCount {
                    question: 0,
                    count: 0,
                },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": "yes"}}}"#,
                RequestError::InvalidCriteria {
                    question: 0,
                    expected: CriteriaShape::Noul,
                },
            ),
            // Question 1's refusal comes before media and before question
            // 0's unknown field, which only the subset refuses.
            (
                r#"{"state": null, "files": [], "questions": {"a": {"type": "noul", "instructions": "x", "extra": 1}, "b": {"type": "noul"}}}"#,
                RequestError::MissingInstructions { question: 1 },
            ),
            (
                r#"{"state": null, "extra": 1, "images": null, "questions": {"q": {"type": "noul", "instructions": "x"}}}"#,
                RequestError::MediaUnsupported(MediaAt::Images),
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(&refused(text), expected, "{text}");
        }
    }

    #[test]
    fn spoken_instructions_follow_upstream() {
        // Only an object whose type is the string "input_audio".
        for instructions in [
            r#"{"type": "input_audio"}"#,
            r#"{"type": "input_audio", "input_audio": {"data": "UklGRg=="}}"#,
        ] {
            let text = one(
                "null",
                &format!(r#"{{"type": "noul", "instructions": {instructions}}}"#),
            );
            assert_eq!(
                kind(refused(&text)),
                "spoken_instructions",
                "{instructions}"
            );
        }
        for instructions in [
            r#"{"type": ["input_audio"]}"#,
            r#"{"kind": "input_audio"}"#,
            r#"["input_audio"]"#,
            r#""input_audio""#,
        ] {
            let text = one(
                "null",
                &format!(r#"{{"type": "noul", "instructions": {instructions}}}"#),
            );
            assert!(parse(&text).is_ok(), "{instructions}");
        }
    }

    #[test]
    fn choice_option_counts_follow_the_model() {
        let choice = |count: usize| {
            let options: Vec<String> = (0..count).map(|i| format!(r#""o{i}": null"#)).collect();
            one(
                "null",
                &format!(
                    r#"{{"type": "choice", "instructions": "x", "criteria": {{{}}}}}"#,
                    options.join(", ")
                ),
            )
        };
        assert_eq!(
            parse(&choice(1)).unwrap().questions()[0]
                .kind()
                .as_choice()
                .map(<[_]>::len),
            Some(1)
        );
        assert!(parse(&choice(255)).is_ok());
        assert_eq!(
            refused(&choice(256)),
            RequestError::TooManyOptions {
                question: 0,
                count: 256,
                max: 255
            }
        );
        let limits = RequestLimits {
            max_options: 3,
            ..RequestLimits::default()
        };
        assert!(parse_request(choice(3).as_bytes(), &limits).is_ok());
        assert_eq!(
            kind(parse_request(choice(4).as_bytes(), &limits).unwrap_err()),
            "too_many_options"
        );
    }

    #[test]
    fn score_levels_run_from_two_to_ten() {
        let score = |count: usize| {
            let levels = vec!["null"; count].join(", ");
            one(
                "null",
                &format!(r#"{{"type": "score", "instructions": "x", "criteria": [{levels}]}}"#),
            )
        };
        for count in [2, 10] {
            assert_eq!(
                parse(&score(count)).unwrap().questions()[0]
                    .kind()
                    .as_score()
                    .map(<[_]>::len),
                Some(count)
            );
        }
        for count in [0, 1, 11] {
            assert_eq!(
                refused(&score(count)),
                RequestError::LevelCount { question: 0, count }
            );
        }
        let object = one(
            "null",
            r#"{"type": "score", "instructions": "x", "criteria": {"0": "a", "1": "b"}}"#,
        );
        assert_eq!(kind(refused(&object)), "invalid_criteria");
    }

    #[test]
    fn media_is_refused_wherever_llama_cpp_finds_it() {
        let with_state = |state: &str| refused(&one(state, NOUL));
        for field in ["files", "images", "videos"] {
            for value in ["[]", "null", r#"["data:image/png;base64,AAAA"]"#, "{}"] {
                let text = format!(
                    r#"{{"state": null, "{field}": {value}, "questions": {{"q": {NOUL}}}}}"#
                );
                assert_eq!(
                    kind(refused(&text)),
                    "media_unsupported",
                    "{field}: {value}"
                );
            }
        }
        // A part of a message's content, unwrapped or under "messages".
        assert_eq!(
            with_state(
                r#"[{"role": "user", "content": "x"}, {"content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": null}]}]"#
            ),
            RequestError::MediaUnsupported(MediaAt::StatePart {
                message: 1,
                part: 1
            })
        );
        assert_eq!(
            with_state(
                r#"{"channel": "chat", "messages": [{"content": [{"type": "input_audio", "input_audio": {"data": "AAAA"}}]}]}"#
            ),
            RequestError::MediaUnsupported(MediaAt::StatePart {
                message: 0,
                part: 0
            })
        );
        assert_eq!(
            with_state(r#"[{"content": [{"type": "video_url", "video_url": "x"}]}]"#),
            RequestError::MediaUnsupported(MediaAt::StatePart {
                message: 0,
                part: 0
            })
        );
        // What llama.cpp keeps as JSON stays content.
        for state in [
            r#"[{"content": [{"type": "image_url"}]}]"#,
            r#"[{"content": [{"type": "image", "image_url": "x"}]}]"#,
            r#"[{"content": {"type": "image_url", "image_url": "x"}}]"#,
            r#"[[{"type": "image_url", "image_url": "x"}]]"#,
            r#"{"history": [{"content": [{"type": "image_url", "image_url": "x"}]}]}"#,
            r#"{"messages": {"content": [{"type": "image_url", "image_url": "x"}]}}"#,
            r#"[{"content": [{"type": {"t": "image_url"}, "image_url": "x"}]}]"#,
        ] {
            assert!(parse(&one(state, NOUL)).is_ok(), "{state}");
        }
    }

    #[test]
    fn the_subset_refuses_what_llama_cpp_ignores() {
        let cases: &[(&str, RequestError)] = &[
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}}, "temperature": 0.1}"#,
                RequestError::UnknownField(FieldAt::Request { field: 2 }),
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "choice", "instructions": "x", "multiple": true, "criteria": {"a": null}}}}"#,
                RequestError::UnknownField(FieldAt::Question {
                    question: 0,
                    field: 2,
                }),
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "a", "yes": "b"}}}}"#,
                RequestError::UnknownField(FieldAt::NoulCriteria {
                    question: 0,
                    field: 1,
                }),
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}, "": {"type": "noul", "instructions": "x"}}}"#,
                RequestError::EmptyQuestionId { question: 1 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": null, "": "b"}}}}"#,
                RequestError::EmptyOptionKey {
                    question: 0,
                    option: 1,
                },
            ),
            // Fields first, then each question in order, then the content.
            (
                r#"{"state": {"n": 18446744073709551615}, "x": 1, "questions": {"": {"type": "noul", "instructions": "x"}}}"#,
                RequestError::UnknownField(FieldAt::Request { field: 1 }),
            ),
            (
                r#"{"state": {"n": 18446744073709551615}, "questions": {"q": {"type": "noul", "instructions": "x"}, "": {"type": "noul", "instructions": "x"}}}"#,
                RequestError::EmptyQuestionId { question: 1 },
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(&refused(text), expected, "{text}");
        }
        // The model field is ignored, whatever it holds.
        for model in [
            r#""d1-2B""#,
            "null",
            "18446744073709551615",
            r#"{"a": [1]}"#,
        ] {
            let text =
                format!(r#"{{"model": {model}, "state": null, "questions": {{"q": {NOUL}}}}}"#);
            assert!(parse(&text).is_ok(), "{model}");
        }
    }

    #[test]
    fn integers_above_i64_are_refused_wherever_content_holds_them() {
        let cases: &[(&str, ContentAt)] = &[
            (
                &one(r#"{"a": [18446744073709551615]}"#, NOUL),
                ContentAt::State,
            ),
            (
                &one(
                    "null",
                    r#"{"type": "noul", "instructions": 9223372036854775808}"#,
                ),
                ContentAt::Instructions { question: 0 },
            ),
            (
                &one(
                    "null",
                    r#"{"type": "choice", "instructions": "x", "criteria": {"a": null, "b": {"n": 9223372036854775808}}}"#,
                ),
                ContentAt::Option {
                    question: 0,
                    option: 1,
                },
            ),
            (
                &one(
                    "null",
                    r#"{"type": "noul", "instructions": "x", "criteria": {"false": 1, "true": [9223372036854775808]}}"#,
                ),
                ContentAt::Yes { question: 0 },
            ),
            (
                &one(
                    "null",
                    r#"{"type": "noul", "instructions": "x", "criteria": {"false": 9223372036854775808}}"#,
                ),
                ContentAt::No { question: 0 },
            ),
            (
                &one(
                    "null",
                    r#"{"type": "score", "instructions": "x", "criteria": [0, 9223372036854775808]}"#,
                ),
                ContentAt::Level {
                    question: 0,
                    level: 1,
                },
            ),
        ];
        for (text, at) in cases {
            assert_eq!(
                refused(text),
                RequestError::IntegerOutOfRange(*at),
                "{text}"
            );
        }
        // i64's own range, integers past u64 (floats to llama.cpp) and
        // floats are content.
        let state = r#"{"max": 9223372036854775807, "min": -9223372036854775808, "past_u64": 18446744073709551616, "below": -9223372036854775809, "big": 1e300}"#;
        assert!(parse(&one(state, NOUL)).is_ok());
    }

    #[test]
    fn content_depth_counts_from_the_content() {
        // Content may nest 128 levels counting from itself, wherever it sits
        // (the protocol, not the transport, caps it lower), so each of these
        // bodies is deeper than an `OrderedJson` holds, and the content
        // comes through exactly. The score level opens right after the
        // criteria's own bracket.
        let nested = |depth: usize| format!("{}1{}", "[".repeat(depth - 1), "]".repeat(depth - 1));
        let places: [(&str, ContentAt); 6] = [
            (
                r#"{"state": #, "questions": {"q": {"type": "noul", "instructions": "x"}}}"#,
                ContentAt::State,
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": #}}}"#,
                ContentAt::Instructions { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": null, "b": #}}}}"#,
                ContentAt::Option {
                    question: 0,
                    option: 1,
                },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": #}}}}"#,
                ContentAt::Yes { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": null, "false": #}}}}"#,
                ContentAt::No { question: 0 },
            ),
            (
                r#"{"state": null, "questions": {"q": {"type": "score", "instructions": "x", "criteria": [#, null]}}}"#,
                ContentAt::Level {
                    question: 0,
                    level: 0,
                },
            ),
        ];
        let content_at = |request: &DecisionRequest, at: ContentAt| -> Option<DecisionContent> {
            let kind = || request.questions()[0].kind();
            match at {
                ContentAt::State => request.state().body().cloned(),
                ContentAt::Instructions { .. } => request.questions()[0]
                    .instructions()
                    .and_then(DecisionInstructions::as_content)
                    .cloned(),
                ContentAt::Option { option, .. } => {
                    kind().as_choice()?[option].description().cloned()
                }
                ContentAt::Yes { .. } => kind().as_yes_no()?.0.cloned(),
                ContentAt::No { .. } => kind().as_yes_no()?.1.cloned(),
                ContentAt::Level { level, .. } => kind().as_score()?[level].description().cloned(),
            }
        };
        for (template, at) in places {
            let deepest = nested(128);
            let request = parse(&template.replace('#', &deepest))
                .unwrap_or_else(|error| panic!("{at:?}: {error}"));
            let content = content_at(&request, at).unwrap_or_else(|| panic!("{at:?}"));
            let json = content.as_json().unwrap_or_else(|| panic!("{at:?}"));
            assert_eq!(json.depth(), 128, "{at:?}");
            assert_eq!(json.to_json_text(), deepest, "{at:?}");
            assert_eq!(
                refused(&template.replace('#', &nested(129))),
                RequestError::TooDeep(at)
            );
        }
    }

    #[test]
    fn the_transport_ceilings_hold() {
        let text = one("null", NOUL);
        let limits = RequestLimits {
            max_body_bytes: text.len() - 1,
            ..RequestLimits::default()
        };
        assert_eq!(
            parse_request(text.as_bytes(), &limits),
            Err(RequestError::TooLarge {
                len: text.len(),
                max: text.len() - 1
            })
        );
        let limits = RequestLimits {
            max_body_values: 4,
            ..RequestLimits::default()
        };
        assert_eq!(
            parse_request(text.as_bytes(), &limits),
            Err(RequestError::TooManyValues { max: 4 })
        );
        // One piece of content may hold the default value ceiling, no more.
        let state = format!("[{}]", vec!["0"; 65_536].join(","));
        assert_eq!(
            refused(&one(&state, NOUL)),
            RequestError::ContentTooManyValues {
                at: ContentAt::State,
                max: 65_536
            }
        );
        assert_eq!(kind(refused(&one(&state, NOUL))), "request_too_large");
    }

    #[test]
    fn refusals_never_quote_the_request() {
        let texts = [
            format!(r#"{{"{CANARY}": 1, "state": null, "questions": {{"q": {NOUL}}}}}"#),
            format!(r#"{{"state": "{CANARY}", "questions": {{"{CANARY}": "x"}}}}"#),
            format!(
                r#"{{"state": null, "questions": {{"{CANARY}": {{"type": "{CANARY}", "instructions": "{CANARY}"}}}}}}"#
            ),
            format!(
                r#"{{"state": null, "questions": {{"{CANARY}": {{"type": "noul", "instructions": "x", "{CANARY}": 1}}}}}}"#
            ),
            format!(
                r#"{{"state": null, "questions": {{"q": {{"type": "noul", "instructions": "x", "criteria": {{"{CANARY}": 1}}}}}}}}"#
            ),
            format!(
                r#"{{"state": {{"{CANARY}": 18446744073709551615}}, "questions": {{"q": {NOUL}}}}}"#
            ),
            format!(r#"{{"state": "{CANARY}", "{CANARY}": [1, 2,]}}"#),
            format!(r#"{{"state": {{"{CANARY}": 1, "{CANARY}": 2}}}}"#),
            format!(
                r#"{{"state": [{{"content": [{{"type": "image_url", "image_url": "{CANARY}"}}]}}], "questions": {{"q": {NOUL}}}}}"#
            ),
        ];
        for text in &texts {
            let error = refused(text);
            for rendered in [error.to_string(), format!("{error:?}")] {
                assert!(!rendered.contains(CANARY), "{rendered}");
            }
        }
    }

    #[test]
    fn every_refusal_has_a_kind_and_only_the_ceilings_have_the_transports() {
        // The parse-stage kinds of the fixtures' README.
        let names = [
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
        ];
        let at = ContentAt::State;
        for error in [
            RequestError::TooLarge { len: 2, max: 1 },
            RequestError::TooManyValues { max: 1 },
            RequestError::NotUtf8 { offset: 0 },
            RequestError::InvalidJson {
                kind: JsonErrorKind::UnexpectedEnd,
                offset: 0,
            },
            RequestError::DuplicateKey { offset: 0 },
            RequestError::NumberOverflow { offset: 0 },
            RequestError::NotAnObject,
            RequestError::MissingState,
            RequestError::InvalidQuestions,
            RequestError::InvalidQuestion { question: 0 },
            RequestError::MissingInstructions { question: 0 },
            RequestError::SpokenInstructions { question: 0 },
            RequestError::InvalidQuestionType { question: 0 },
            RequestError::InvalidCriteria {
                question: 0,
                expected: CriteriaShape::Choice,
            },
            RequestError::LevelCount {
                question: 0,
                count: 1,
            },
            RequestError::TooManyOptions {
                question: 0,
                count: 2,
                max: 1,
            },
            RequestError::MediaUnsupported(MediaAt::Files),
            RequestError::UnknownField(FieldAt::Request { field: 0 }),
            RequestError::EmptyQuestionId { question: 0 },
            RequestError::EmptyOptionKey {
                question: 0,
                option: 0,
            },
            RequestError::IntegerOutOfRange(at),
            RequestError::TooDeep(at),
            RequestError::ContentTooManyValues { at, max: 1 },
            RequestError::InvalidContent(at),
        ] {
            let kind = error.kind();
            let ceiling = matches!(
                error,
                RequestError::TooLarge { .. }
                    | RequestError::TooManyValues { .. }
                    | RequestError::ContentTooManyValues { .. }
            );
            if ceiling {
                assert_eq!(kind, RefusalKind::RequestTooLarge, "{error:?}");
                assert_eq!(kind.as_str(), "request_too_large");
            } else {
                assert!(names.contains(&kind.as_str()), "{error:?}");
            }
            assert_eq!(kind.to_string(), kind.as_str());
            assert!(!error.to_string().is_empty());
        }
    }
}

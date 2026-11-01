//! Decision requests: one state and the questions asked about it.

use super::OrderedJson;
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;

#[cfg(feature = "schema")]
use schemars::JsonSchema;

/// Text, or a JSON value that is not a string.
///
/// A field that holds content is an `Option<DecisionContent>`, and JSON
/// null is never content: [`DecisionContent::from_json`] turns `null` into
/// `None` and a JSON string into text, so each value has one form. `Some`
/// of an empty text is still content and stays distinct from `None`; nulls
/// nested inside a JSON value are kept.
///
/// `Debug` prints the kind and a length, never the content.
#[derive(Clone, PartialEq, Eq)]
pub struct DecisionContent(Content);

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename = "DecisionContent")]
enum Content {
    Text(String),
    Json(OrderedJson),
}

/// A borrowed view of [`DecisionContent`].
///
/// `Debug` prints the kind and a length, never the content.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DecisionContentView<'a> {
    /// Text.
    Text(&'a str),
    /// A JSON value that is neither null nor a string.
    Json(&'a OrderedJson),
}

impl DecisionContent {
    /// Text content.
    pub fn text(text: impl Into<String>) -> Self {
        Self(Content::Text(text.into()))
    }

    /// Content from a JSON value: a string becomes text, `null` becomes
    /// `None` (absent content), and anything else stays JSON.
    pub fn from_json(value: OrderedJson) -> Option<Self> {
        if value.is_null() {
            return None;
        }
        Some(Self(match value.into_string() {
            Ok(text) => Content::Text(text),
            Err(value) => Content::Json(value),
        }))
    }

    /// A borrowed view, for matching on text versus JSON.
    pub fn view(&self) -> DecisionContentView<'_> {
        match &self.0 {
            Content::Text(text) => DecisionContentView::Text(text),
            Content::Json(value) => DecisionContentView::Json(value),
        }
    }

    /// The text, when the content is text.
    pub fn as_text(&self) -> Option<&str> {
        match &self.0 {
            Content::Text(text) => Some(text),
            Content::Json(_) => None,
        }
    }

    /// The JSON value, when the content is JSON. Never null or a string.
    pub fn as_json(&self) -> Option<&OrderedJson> {
        match &self.0 {
            Content::Text(_) => None,
            Content::Json(value) => Some(value),
        }
    }
}

impl From<String> for DecisionContent {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for DecisionContent {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl DecisionContentView<'_> {
    /// Kind and length only: content is the caller's.
    fn redacted(&self, name: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = f.debug_struct(name);
        match self {
            Self::Text(text) => out.field("kind", &"text").field("bytes", &text.len()),
            Self::Json(value) => out.field("kind", &"json").field("json", value),
        };
        out.finish()
    }
}

impl fmt::Debug for DecisionContentView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.redacted("DecisionContentView", f)
    }
}

impl fmt::Debug for DecisionContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.view().redacted("DecisionContent", f)
    }
}

/// `{"Text": "..."}` or `{"Json": "<compact JSON text>"}`.
impl Serialize for DecisionContent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

/// Reads what [`Serialize`] writes. A `Json` payload that is a string
/// decodes as text; one that is `null` is refused, because absent content is
/// the `None` of the field that holds it.
impl<'de> Deserialize<'de> for DecisionContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Content::deserialize(deserializer)? {
            Content::Text(text) => Ok(Self(Content::Text(text))),
            Content::Json(value) => Self::from_json(value).ok_or_else(|| {
                de::Error::custom("JSON content is null; absent content is the field's null")
            }),
        }
    }
}

#[cfg(feature = "schema")]
impl JsonSchema for DecisionContent {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "DecisionContent".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Content::json_schema(generator)
    }
}

/// Media in a decision state. Reserved: there is no variant yet, so no
/// value exists, and every protocol refuses media until one is added.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[non_exhaustive]
pub enum DecisionMedia {}

impl fmt::Debug for DecisionMedia {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {}
    }
}

/// What the questions are asked about: a body of content, plus media
/// (reserved).
///
/// A `None` body is a null state, which some protocols accept (LFM3 d1
/// then leaves out the part of the prompt that introduces the state); an
/// empty text is a state too, and differs from a null one.
///
/// `Debug` prints the kind and size of the body and the number of media.
#[derive(Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DecisionState {
    body: Option<DecisionContent>,
    media: Vec<DecisionMedia>,
}

impl DecisionState {
    /// A state with `body` and no media. `None` is the null state, as is
    /// [`DecisionState::default`].
    pub fn new(body: Option<DecisionContent>) -> Self {
        Self {
            body,
            media: Vec::new(),
        }
    }

    /// A text state.
    pub fn text(text: impl Into<String>) -> Self {
        Self::new(Some(DecisionContent::text(text)))
    }

    /// A state from a JSON value, normalized as
    /// [`DecisionContent::from_json`] does: `null` is the null state and a
    /// string a text state.
    pub fn from_json(value: OrderedJson) -> Self {
        Self::new(DecisionContent::from_json(value))
    }

    /// The body; `None` for a null state.
    pub fn body(&self) -> Option<&DecisionContent> {
        self.body.as_ref()
    }

    /// The media, in order. Always empty: no media kind exists yet.
    pub fn media(&self) -> &[DecisionMedia] {
        &self.media
    }

    /// Takes the body out.
    pub fn into_body(self) -> Option<DecisionContent> {
        self.body
    }
}

impl fmt::Debug for DecisionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionState")
            .field("body", &self.body)
            .field("media", &self.media.len())
            .finish()
    }
}

/// The instructions of a question: what it asks.
///
/// Only written content exists today; spoken instructions are planned and
/// will be a new variant, which is why matches need a wildcard arm.
///
/// `Debug` prints the kind and a length, never the content.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[non_exhaustive]
pub enum DecisionInstructions {
    /// Text or JSON.
    Content(DecisionContent),
}

impl DecisionInstructions {
    /// The written content; `None` for instructions that are not written
    /// (none exist yet).
    pub fn as_content(&self) -> Option<&DecisionContent> {
        match self {
            Self::Content(content) => Some(content),
        }
    }
}

impl From<DecisionContent> for DecisionInstructions {
    fn from(content: DecisionContent) -> Self {
        Self::Content(content)
    }
}

impl fmt::Debug for DecisionInstructions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Content(content) => f.debug_tuple("Content").field(content).finish(),
        }
    }
}

/// One option of a choice question.
///
/// Unlike a legacy [`Choice`](crate::ir::Choice) id, the key may affect
/// inference: a protocol can show it to the model (LFM3 d1 uses
/// single-letter keys as the labels it reads, and prints the key, with `_`
/// as a space, in place of a missing or empty description). Renaming a key
/// can therefore change the answer.
///
/// `Debug` prints the key's length and the description's kind and length.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DecisionOption {
    key: String,
    description: Option<DecisionContent>,
}

impl DecisionOption {
    /// An option with its key and optional description.
    pub fn new(key: impl Into<String>, description: Option<DecisionContent>) -> Self {
        Self {
            key: key.into(),
            description,
        }
    }

    /// The key, which names the option in the answer.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The description, if any.
    pub fn description(&self) -> Option<&DecisionContent> {
        self.description.as_ref()
    }
}

impl fmt::Debug for DecisionOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionOption")
            .field("key_bytes", &self.key.len())
            .field("description", &self.description)
            .finish()
    }
}

/// What a question asks for, with its alternatives.
///
/// Which kinds, how many alternatives and which descriptions a model
/// accepts is up to its protocol; [`validate_structure`](super::validate_structure)
/// only checks what every protocol needs.
///
/// `Debug` prints counts, never content.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub enum DecisionQuestionKind {
    /// Pick one option. The answer's distribution has one entry per option,
    /// keyed by its key, in option order; a model with fixed choices of its
    /// own appends them after, as [`DecisionOrigin::Fixed`](super::DecisionOrigin::Fixed).
    Choice {
        /// The options, in the order the caller offers them.
        options: Vec<DecisionOption>,
    },
    /// Yes or no, each optionally described. The answer's distribution is
    /// keyed `"true"` and `"false"`, in the protocol's order.
    YesNo {
        /// What "yes" means.
        yes: Option<DecisionContent>,
        /// What "no" means.
        no: Option<DecisionContent>,
    },
    /// A level on a scale. Level `i` is keyed `"i"` (decimal, from `"0"`)
    /// in the answer's distribution; a level without a description is
    /// still a level.
    Score {
        /// The level descriptions, lowest first.
        levels: Vec<Option<DecisionContent>>,
    },
}

impl DecisionQuestionKind {
    /// The kind's name: `"choice"`, `"yes/no"` or `"score"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Choice { .. } => "choice",
            Self::YesNo { .. } => "yes/no",
            Self::Score { .. } => "score",
        }
    }
}

impl fmt::Debug for DecisionQuestionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice { options } => f
                .debug_struct("Choice")
                .field("options", &options.len())
                .finish(),
            Self::YesNo { yes, no } => f
                .debug_struct("YesNo")
                .field("yes", yes)
                .field("no", no)
                .finish(),
            Self::Score { levels } => f
                .debug_struct("Score")
                .field("levels", &levels.len())
                .finish(),
        }
    }
}

/// One question: an id, optional instructions, and its kind.
///
/// The id names the question's answer. A protocol may also show it to the
/// model (LFM3 d1's template receives it).
///
/// `Debug` prints the id's length, the instructions' kind and length, and
/// the kind's counts.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DecisionQuestion {
    id: String,
    instructions: Option<DecisionInstructions>,
    kind: DecisionQuestionKind,
}

impl DecisionQuestion {
    /// A question with its id, optional instructions and kind.
    pub fn new(
        id: impl Into<String>,
        instructions: Option<DecisionInstructions>,
        kind: DecisionQuestionKind,
    ) -> Self {
        Self {
            id: id.into(),
            instructions,
            kind,
        }
    }

    /// The id, which names the answer.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The instructions, if any.
    pub fn instructions(&self) -> Option<&DecisionInstructions> {
        self.instructions.as_ref()
    }

    /// What the question asks for.
    pub fn kind(&self) -> &DecisionQuestionKind {
        &self.kind
    }
}

impl fmt::Debug for DecisionQuestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionQuestion")
            .field("id_bytes", &self.id.len())
            .field("instructions", &self.instructions)
            .field("kind", &self.kind)
            .finish()
    }
}

/// A decision request: one state and the questions about it, in order.
///
/// Each question is answered on its own, and the answers come back in
/// question order ([`Decisions`](super::Decisions)). The request is plain
/// data; [`validate_structure`](super::validate_structure) checks what every
/// protocol needs, and the model's protocol checks the rest before anything
/// runs.
///
/// `Debug` prints the state's kind and size and the number of questions.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    state: DecisionState,
    questions: Vec<DecisionQuestion>,
}

impl DecisionRequest {
    /// A request asking `questions` about `state`.
    pub fn new(state: DecisionState, questions: Vec<DecisionQuestion>) -> Self {
        Self { state, questions }
    }

    /// The state.
    pub fn state(&self) -> &DecisionState {
        &self.state
    }

    /// The questions, in order.
    pub fn questions(&self) -> &[DecisionQuestion] {
        &self.questions
    }

    /// Splits the request into its state and questions.
    pub fn into_parts(self) -> (DecisionState, Vec<DecisionQuestion>) {
        (self.state, self.questions)
    }
}

impl fmt::Debug for DecisionRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionRequest")
            .field("state", &self.state)
            .field("questions", &self.questions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANARY: &str = "canary-7f3a9c";

    fn json(text: &str) -> OrderedJson {
        OrderedJson::parse(text).unwrap()
    }

    #[test]
    fn json_content_turns_strings_into_text_and_null_into_none() {
        assert_eq!(DecisionContent::from_json(json("null")), None);
        assert_eq!(
            DecisionContent::from_json(json(r#""a \"text\"""#)),
            Some(DecisionContent::text("a \"text\""))
        );
        assert_eq!(
            DecisionContent::from_json(json(r#""""#)),
            Some(DecisionContent::text(""))
        );
        for text in ["0", "-0.0", "false", "[]", "{}", "[null]", r#"{"a":null}"#] {
            let content = DecisionContent::from_json(json(text)).unwrap();
            assert_eq!(content.as_json(), Some(&json(text)), "{text}");
            assert!(content.as_text().is_none(), "{text}");
        }
        // Nulls inside a JSON value are content, and are kept.
        let nested = DecisionContent::from_json(json(r#"[null,{"k":null}]"#)).unwrap();
        assert_eq!(
            nested.as_json().map(OrderedJson::to_json_text).as_deref(),
            Some(r#"[null,{"k":null}]"#)
        );
    }

    #[test]
    fn content_views_follow_the_kind() {
        let text = DecisionContent::from("words");
        assert!(matches!(text.view(), DecisionContentView::Text("words")));
        assert_eq!(text, DecisionContent::from(String::from("words")));
        assert_eq!(text.as_text(), Some("words"));
        assert!(text.as_json().is_none());

        let value = DecisionContent::from_json(json(r#"{"k":[1]}"#)).unwrap();
        assert!(
            matches!(value.view(), DecisionContentView::Json(inner) if inner.get("k").is_some())
        );
        assert_ne!(
            DecisionContent::text("1"),
            DecisionContent::from_json(json("1")).unwrap()
        );
    }

    #[test]
    fn a_null_state_differs_from_an_empty_one() {
        let null = DecisionState::default();
        assert_eq!(null, DecisionState::new(None));
        assert_eq!(null, DecisionState::from_json(json("null")));
        assert!(null.body().is_none());

        let empty = DecisionState::text("");
        assert_ne!(empty, null);
        assert_eq!(empty.body().and_then(DecisionContent::as_text), Some(""));
        assert_eq!(DecisionState::from_json(json(r#""""#)), empty);
        assert!(empty.media().is_empty());

        let state = DecisionState::from_json(json(r#"{"b":[1,null],"a":"x"}"#));
        let body = state.clone().into_body().unwrap();
        assert_eq!(state.body(), Some(&body));
        assert_eq!(
            body.as_json().map(OrderedJson::to_json_text).as_deref(),
            Some(r#"{"b":[1,null],"a":"x"}"#)
        );
    }

    #[test]
    fn accessors_return_what_was_set() {
        let instructions = DecisionInstructions::from(DecisionContent::text("Which one?"));
        let options = vec![
            DecisionOption::new("a", Some(DecisionContent::text("first"))),
            DecisionOption::new("b_c", None),
        ];
        let question = DecisionQuestion::new(
            "q1",
            Some(instructions.clone()),
            DecisionQuestionKind::Choice {
                options: options.clone(),
            },
        );
        assert_eq!(question.id(), "q1");
        assert_eq!(question.instructions(), Some(&instructions));
        assert_eq!(
            instructions.as_content().and_then(DecisionContent::as_text),
            Some("Which one?")
        );
        assert!(
            matches!(question.kind(), DecisionQuestionKind::Choice { options: o } if *o == options)
        );
        assert_eq!(options[0].key(), "a");
        assert_eq!(
            options[0].description().and_then(DecisionContent::as_text),
            Some("first")
        );
        assert!(options[1].description().is_none());

        let state = DecisionState::text("context");
        let request = DecisionRequest::new(state.clone(), vec![question.clone()]);
        assert_eq!(request.state(), &state);
        assert_eq!(request.questions(), std::slice::from_ref(&question));
        assert_eq!(request.into_parts(), (state, vec![question]));
    }

    #[test]
    fn kind_names() {
        let names = [
            DecisionQuestionKind::Choice { options: vec![] },
            DecisionQuestionKind::YesNo {
                yes: None,
                no: None,
            },
            DecisionQuestionKind::Score { levels: vec![] },
        ]
        .map(|kind| kind.name());
        assert_eq!(names, ["choice", "yes/no", "score"]);
    }

    #[test]
    fn content_serde_form_is_tagged_text_or_json_text() {
        let text = DecisionContent::text("hi");
        assert_eq!(serde_json::to_string(&text).unwrap(), r#"{"Text":"hi"}"#);
        let value = DecisionContent::from_json(json(r#"{"b":1,"a":1.0}"#)).unwrap();
        let encoded = serde_json::to_string(&value).unwrap();
        assert_eq!(encoded, r#"{"Json":"{\"b\":1,\"a\":1.0}"}"#);
        assert_eq!(
            serde_json::from_str::<DecisionContent>(&encoded).unwrap(),
            value
        );
        for content in [text, value] {
            let bytes = bincode::serialize(&content).unwrap();
            assert_eq!(
                bincode::deserialize::<DecisionContent>(&bytes).unwrap(),
                content
            );
        }

        // A JSON string under the Json tag decodes as text; null is refused.
        let decoded: DecisionContent = serde_json::from_str(r#"{"Json":"\"abc\""}"#).unwrap();
        assert_eq!(decoded, DecisionContent::text("abc"));
        let error = serde_json::from_str::<DecisionContent>(r#"{"Json":"null"}"#).unwrap_err();
        assert!(
            error.to_string().contains("JSON content is null"),
            "{error}"
        );
        let error =
            serde_json::from_str::<DecisionContent>(&format!(r#"{{"Json":"[\"{CANARY}\","}}"#))
                .unwrap_err();
        assert!(!error.to_string().contains(CANARY), "{error}");
        assert!(serde_json::from_str::<DecisionContent>(r#"{"Html":"x"}"#).is_err());
    }

    #[test]
    fn every_struct_and_struct_variant_refuses_unknown_fields() {
        let valid = [
            (r#"{"body":null,"media":[]}"#, "state"),
            (r#"{"key":"a","description":null}"#, "option"),
            (
                r#"{"id":"q","instructions":null,"kind":{"YesNo":{"yes":null,"no":null}}}"#,
                "question",
            ),
            (
                r#"{"state":{"body":{"Text":"s"},"media":[]},"questions":[]}"#,
                "request",
            ),
        ];
        for (text, what) in valid {
            let extra = format!("{},\"extra\":1}}", &text[..text.len() - 1]);
            let (ok, refused) = match what {
                "state" => (
                    serde_json::from_str::<DecisionState>(text).is_ok(),
                    serde_json::from_str::<DecisionState>(&extra).is_err(),
                ),
                "option" => (
                    serde_json::from_str::<DecisionOption>(text).is_ok(),
                    serde_json::from_str::<DecisionOption>(&extra).is_err(),
                ),
                "question" => (
                    serde_json::from_str::<DecisionQuestion>(text).is_ok(),
                    serde_json::from_str::<DecisionQuestion>(&extra).is_err(),
                ),
                _ => (
                    serde_json::from_str::<DecisionRequest>(text).is_ok(),
                    serde_json::from_str::<DecisionRequest>(&extra).is_err(),
                ),
            };
            assert!(ok && refused, "{what}");
        }
        for kind in [
            r#"{"Choice":{"options":[],"extra":1}}"#,
            r#"{"YesNo":{"yes":null,"no":null,"extra":1}}"#,
            r#"{"Score":{"levels":[],"extra":1}}"#,
            r#"{"Ranking":{"options":[]}}"#,
        ] {
            assert!(
                serde_json::from_str::<DecisionQuestionKind>(kind).is_err(),
                "{kind}"
            );
        }
        assert!(serde_json::from_str::<DecisionInstructions>(r#"{"Spoken":"x"}"#).is_err());
    }

    #[test]
    fn media_cannot_be_decoded_yet() {
        assert!(
            serde_json::from_str::<DecisionState>(r#"{"body":null,"media":[{"Image":"x"}]}"#)
                .is_err()
        );
        // bincode: a media list of length 1 has nothing to decode into.
        let mut bytes = bincode::serialize(&DecisionState::default()).unwrap();
        let last = bytes.len() - 8;
        bytes[last] = 1;
        assert!(bincode::deserialize::<DecisionState>(&bytes).is_err());
    }

    #[test]
    fn debug_never_prints_content() {
        let content = DecisionContent::from_json(json(&format!(
            r#"{{"{CANARY}": ["{CANARY}", 918273645]}}"#
        )))
        .unwrap();
        let question = DecisionQuestion::new(
            CANARY,
            Some(DecisionInstructions::Content(DecisionContent::text(CANARY))),
            DecisionQuestionKind::Choice {
                options: vec![
                    DecisionOption::new(CANARY, Some(content.clone())),
                    DecisionOption::new(format!("{CANARY}-2"), Some(DecisionContent::text(CANARY))),
                ],
            },
        );
        let yes_no = DecisionQuestionKind::YesNo {
            yes: Some(DecisionContent::text(CANARY)),
            no: Some(content.clone()),
        };
        let score = DecisionQuestionKind::Score {
            levels: vec![
                None,
                Some(DecisionContent::text(CANARY)),
                Some(content.clone()),
            ],
        };
        let request = DecisionRequest::new(
            DecisionState::new(Some(content.clone())),
            vec![question.clone()],
        );
        let rendered = [
            format!("{content:?}"),
            format!("{:?}", content.view()),
            format!("{question:?}"),
            format!("{yes_no:?}"),
            format!("{score:?}"),
            format!("{request:?}"),
            format!("{:?}", request.state()),
            format!("{:?}", question.instructions()),
        ];
        for text in &rendered {
            assert!(!text.contains(CANARY), "{text}");
            assert!(!text.contains("918273645"), "{text}");
        }
        assert_eq!(
            rendered[0],
            r#"DecisionContent { kind: "json", json: OrderedJson { kind: "object", members: 1 } }"#
        );
        assert!(rendered[2].contains("id_bytes: 13"), "{}", rendered[2]);
        assert!(
            rendered[2].contains("Choice { options: 2 }"),
            "{}",
            rendered[2]
        );
        assert!(rendered[4].contains("levels: 3"), "{}", rendered[4]);
        assert!(rendered[5].contains("questions: 1"), "{}", rendered[5]);
        assert!(rendered[6].contains("media: 0"), "{}", rendered[6]);
        assert_eq!(
            format!("{:?}", DecisionContent::text(CANARY)),
            r#"DecisionContent { kind: "text", bytes: 13 }"#
        );
    }
}

//! Decision results: one answer per question, each with the distribution it
//! was read from.

use super::{AnswerInvariant, DecisionError};
use serde::de::{self, Deserializer, Visitor};
use serde::ser::{self, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "schema")]
use schemars::JsonSchema;

/// How far from 1 a distribution's probabilities may sum, and (scaled by
/// the number of levels) how far an [`ScoreStatistic::Expected`] value may
/// sit from the expectation it reports.
///
/// Probabilities normalized in `f64` sum to 1 within about `1e-13`;
/// rounding each to `f32` first moves the sum by at most about `6e-8`
/// (each rounding error is relative to its own probability). `1e-6` accepts
/// both and refuses a distribution that was never normalized. It bounds
/// validity, not precision: protocols that promise exact answers
/// (LFM3 d1) are held to them by their conformance tests.
pub const PROBABILITY_TOLERANCE: f64 = 1e-6;

/// Where a distribution entry came from.
///
/// `#[non_exhaustive]`: match it with a wildcard arm for origins added later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[non_exhaustive]
pub enum DecisionOrigin {
    /// From the request: an option, a level, or yes/no.
    Caller,
    /// From the model's own fixed choices, appended after the caller's (a
    /// legacy byte scorer's fixed choices, for example).
    Fixed,
}

/// Which statistic of a score distribution a score value is.
///
/// `#[non_exhaustive]`: match it with a wildcard arm for statistics added
/// later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[non_exhaustive]
pub enum ScoreStatistic {
    /// The first level whose cumulative probability reaches one half
    /// (LFM3 d1).
    Median,
    /// The expected level, the sum of `i * p_i`.
    Expected,
}

impl fmt::Display for ScoreStatistic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Median => "median",
            Self::Expected => "expected value",
        })
    }
}

/// One entry of an answer's distribution.
///
/// `raw_score` is what the protocol read from the model for this entry: for
/// a vocabulary readout such as LFM3 d1's, the highest logit among the
/// entry's label tokens; for another protocol (an embedding head, say) its
/// own score, not necessarily a logit. `-inf` is valid (the model ruled the
/// entry out); NaN and `+inf` never appear in a valid answer.
///
/// `probability` is the entry's share of the distribution, in `f64`, at the
/// temperature the model's publisher set. It is the protocol's canonical
/// distribution, not a calibrated probability that the entry is correct.
///
/// In human-readable serde formats `raw_score` is a number, or the string
/// `"-Infinity"` for `-inf` (never `null`), and `probability` is a decimal
/// string read back exactly (see [`DecisionAnswer`]); binary formats keep
/// the `f32` and the `f64`.
///
/// `Debug` prints the key's length and the numbers, never the key.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DecisionScore {
    key: String,
    #[serde(with = "raw_score")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "raw_score::schema"))]
    raw_score: f32,
    #[serde(with = "exact_f64")]
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    probability: f64,
    origin: DecisionOrigin,
}

impl DecisionScore {
    /// An entry. It is checked when it joins an answer
    /// ([`DecisionAnswer::new`]), not here.
    pub fn new(
        key: impl Into<String>,
        raw_score: f32,
        probability: f64,
        origin: DecisionOrigin,
    ) -> Self {
        Self {
            key: key.into(),
            raw_score,
            probability,
            origin,
        }
    }

    /// The entry's key: an option key, a level (`"0"`, `"1"`, ...), or
    /// `"true"` / `"false"`.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// What the protocol read from the model for the entry.
    pub fn raw_score(&self) -> f32 {
        self.raw_score
    }

    /// The entry's probability in the distribution.
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Where the entry came from.
    pub fn origin(&self) -> DecisionOrigin {
        self.origin
    }
}

impl fmt::Debug for DecisionScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionScore")
            .field("key_bytes", &self.key.len())
            .field("raw_score", &self.raw_score)
            .field("probability", &self.probability)
            .field("origin", &self.origin)
            .finish()
    }
}

/// The answer to one question, derived from its distribution.
///
/// The enum and each of its variants are `#[non_exhaustive]`. Outside this
/// crate, build a value with [`choice`](Self::choice), [`yes_no`](Self::yes_no)
/// or [`score`](Self::score), and read it with the `as_*` accessors or with
/// `{ .. }` patterns and a wildcard arm, so a value appended later (see the
/// module's compatibility rules) breaks no caller:
///
/// ```
/// use xybrid_core::ir::{DecisionValue, ScoreStatistic};
///
/// let values = [
///     DecisionValue::choice("refund"),
///     DecisionValue::yes_no(0.75),
///     DecisionValue::score(2.0, ScoreStatistic::Median),
/// ];
/// assert_eq!(values[0].as_choice(), Some("refund"));
/// assert_eq!(values[1].as_yes_no(), Some(0.75));
/// assert_eq!(values[2].as_score(), Some((2.0, ScoreStatistic::Median)));
/// let kinds: Vec<&str> = values
///     .iter()
///     .map(|value| match value {
///         DecisionValue::Choice { .. } => "choice",
///         DecisionValue::YesNo { p_yes, .. } if *p_yes > 0.5 => "yes",
///         DecisionValue::YesNo { .. } => "no",
///         DecisionValue::Score { statistic: ScoreStatistic::Median, .. } => "median",
///         _ => "other", // a value or statistic added later
///     })
///     .collect();
/// assert_eq!(kinds, ["choice", "yes", "median"]);
/// ```
///
/// Struct expressions do not compile outside this crate:
///
/// ```compile_fail,E0639
/// use xybrid_core::ir::DecisionValue;
///
/// let value = DecisionValue::YesNo { p_yes: 0.5 };
/// ```
///
/// `Debug` prints the chosen key's length and the numbers, never the key.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub enum DecisionValue {
    /// The chosen option: the key with the highest probability.
    #[non_exhaustive]
    Choice {
        /// Its key.
        key: String,
    },
    /// The probability of yes: the distribution's `"true"` entry.
    #[non_exhaustive]
    YesNo {
        /// That probability.
        #[serde(with = "exact_f64")]
        #[cfg_attr(feature = "schema", schemars(with = "String"))]
        p_yes: f64,
    },
    /// A level, as the named statistic of the distribution.
    #[non_exhaustive]
    Score {
        /// The level (whole for [`ScoreStatistic::Median`]).
        #[serde(with = "exact_f64")]
        #[cfg_attr(feature = "schema", schemars(with = "String"))]
        value: f64,
        /// Which statistic it is.
        statistic: ScoreStatistic,
    },
}

impl DecisionValue {
    /// The chosen option, by key.
    pub fn choice(key: impl Into<String>) -> Self {
        Self::Choice { key: key.into() }
    }

    /// The probability of yes.
    pub fn yes_no(p_yes: f64) -> Self {
        Self::YesNo { p_yes }
    }

    /// A level, as `statistic` of the distribution.
    pub fn score(value: f64, statistic: ScoreStatistic) -> Self {
        Self::Score { value, statistic }
    }

    /// The chosen key of a choice.
    pub fn as_choice(&self) -> Option<&str> {
        match self {
            Self::Choice { key } => Some(key),
            _ => None,
        }
    }

    /// The probability of yes of a yes/no answer.
    pub fn as_yes_no(&self) -> Option<f64> {
        match self {
            Self::YesNo { p_yes } => Some(*p_yes),
            _ => None,
        }
    }

    /// The level and statistic of a score.
    pub fn as_score(&self) -> Option<(f64, ScoreStatistic)> {
        match self {
            Self::Score { value, statistic } => Some((*value, *statistic)),
            _ => None,
        }
    }
}

impl fmt::Debug for DecisionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice { key } => f
                .debug_struct("Choice")
                .field("key_bytes", &key.len())
                .finish(),
            Self::YesNo { p_yes } => f.debug_struct("YesNo").field("p_yes", p_yes).finish(),
            Self::Score { value, statistic } => f
                .debug_struct("Score")
                .field("value", value)
                .field("statistic", statistic)
                .finish(),
        }
    }
}

/// The answer to one question: its distribution, the value derived from
/// it, and a confidence.
///
/// [`DecisionAnswer::new`] is the only way to build one, and deserializing
/// goes through it in every format, so every answer is consistent with its
/// distribution, exactly.
///
/// That holds across a text round trip because human-readable formats never
/// carry an `f64` here as a number: a probability, `p_yes`, a score value
/// and a confidence are decimal strings with the shortest digits that read
/// back as the same `f64`, read back with a correctly rounded parser.
/// serde_json's own float parsing is best-effort unless its
/// `float_roundtrip` feature is on, and can move a probability by an ulp,
/// enough to move the level where the mass reaches one half. Binary formats
/// carry the `f64` itself.
///
/// `Debug` prints lengths and numbers, never the question id or keys.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(try_from = "AnswerFields")]
pub struct DecisionAnswer {
    question_id: String,
    distribution: Vec<DecisionScore>,
    value: DecisionValue,
    #[serde(with = "exact_f64::option")]
    #[cfg_attr(feature = "schema", schemars(with = "Option<String>"))]
    confidence: Option<f64>,
}

/// The serde form of [`DecisionAnswer`], checked on the way in.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields, rename = "DecisionAnswer")]
struct AnswerFields {
    question_id: String,
    distribution: Vec<DecisionScore>,
    value: DecisionValue,
    #[serde(with = "exact_f64::option")]
    #[cfg_attr(feature = "schema", schemars(with = "Option<String>"))]
    confidence: Option<f64>,
}

impl TryFrom<AnswerFields> for DecisionAnswer {
    type Error = DecisionError;

    fn try_from(fields: AnswerFields) -> Result<Self, Self::Error> {
        Self::new(
            fields.question_id,
            fields.distribution,
            fields.value,
            fields.confidence,
        )
    }
}

impl DecisionAnswer {
    /// An answer to question `question_id`, checked against its
    /// distribution.
    ///
    /// The distribution lists entries in the protocol's order: option order
    /// for a choice, level order for a score, the protocol's order of
    /// `"true"` and `"false"` for yes/no (LFM3 d1: true first). `confidence`
    /// is protocol-defined (LFM3 d1: the highest probability for a choice or
    /// a score, none for yes/no) and is not comparable across protocols.
    ///
    /// # Errors
    ///
    /// [`DecisionError::InvalidAnswer`] naming the first broken invariant:
    ///
    /// - the question id and every key are non-empty, and keys are unique;
    /// - the distribution has at least one entry; raw scores are finite or
    ///   `-inf`, and at least one is finite; probabilities are finite, in
    ///   `[0, 1]`, and sum to 1 within [`PROBABILITY_TOLERANCE`];
    /// - [`DecisionValue::Choice`]: the key is in the distribution with the
    ///   highest probability (on a tie the protocol picks among the tied
    ///   keys; LFM3 d1 takes the first);
    /// - [`DecisionValue::YesNo`]: the keys are exactly `"true"` and
    ///   `"false"`, and `p_yes` equals the probability of `"true"`;
    /// - [`DecisionValue::Score`]: entry `i` is keyed `"i"`; a
    ///   [`ScoreStatistic::Median`] value equals the first level whose
    ///   cumulative probability (summed in level order) reaches 0.5, or the
    ///   last level if none does; an [`ScoreStatistic::Expected`] value is
    ///   within `PROBABILITY_TOLERANCE * max(1, levels - 1)` of the sum of
    ///   `i * p_i`;
    /// - the confidence, if any, is finite and in `[0, 1]`.
    ///
    /// # Example
    ///
    /// ```
    /// use xybrid_core::ir::{DecisionAnswer, DecisionOrigin, DecisionScore, DecisionValue, ScoreStatistic};
    ///
    /// // Float32 logits [0, 1e-8] at the published temperature 1.3373981, in f64:
    /// let probabilities = [0.4999999981306987, 0.5000000018693014];
    /// let levels: Vec<DecisionScore> = probabilities.iter().enumerate()
    ///     .map(|(i, &p)| DecisionScore::new(i.to_string(), 0.0, p, DecisionOrigin::Caller))
    ///     .collect();
    ///
    /// // The first level reaching half the mass is level 1, not level 0.
    /// let median = |value| DecisionValue::score(value, ScoreStatistic::Median);
    /// let answer = DecisionAnswer::new("severity", levels.clone(), median(1.0), Some(0.5000000018693014))?;
    /// assert_eq!(answer.value(), &median(1.0));
    /// assert!(DecisionAnswer::new("severity", levels, median(0.0), None).is_err());
    /// # Ok::<(), xybrid_core::ir::DecisionError>(())
    /// ```
    pub fn new(
        question_id: impl Into<String>,
        distribution: Vec<DecisionScore>,
        value: DecisionValue,
        confidence: Option<f64>,
    ) -> Result<Self, DecisionError> {
        let answer = Self {
            question_id: question_id.into(),
            distribution,
            value,
            confidence,
        };
        answer.check().map_err(DecisionError::InvalidAnswer)?;
        Ok(answer)
    }

    fn check(&self) -> Result<(), AnswerInvariant> {
        if self.question_id.is_empty() {
            return Err(AnswerInvariant::EmptyQuestionId);
        }
        if self.distribution.is_empty() {
            return Err(AnswerInvariant::EmptyDistribution);
        }
        let mut first_use = HashMap::with_capacity(self.distribution.len());
        for (index, entry) in self.distribution.iter().enumerate() {
            if entry.key.is_empty() {
                return Err(AnswerInvariant::EmptyKey { index });
            }
            if let Some(&first) = first_use.get(entry.key.as_str()) {
                return Err(AnswerInvariant::DuplicateKey { index, first });
            }
            first_use.insert(entry.key.as_str(), index);
            if entry.raw_score.is_nan() || entry.raw_score == f32::INFINITY {
                return Err(AnswerInvariant::InvalidRawScore { index });
            }
            if !(entry.probability.is_finite() && (0.0..=1.0).contains(&entry.probability)) {
                return Err(AnswerInvariant::InvalidProbability { index });
            }
        }
        if self
            .distribution
            .iter()
            .all(|entry| entry.raw_score == f32::NEG_INFINITY)
        {
            return Err(AnswerInvariant::NoFiniteRawScore);
        }
        let sum = self.probabilities().fold(0.0, |sum, p| sum + p);
        if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(AnswerInvariant::ProbabilitySum { sum });
        }

        match &self.value {
            DecisionValue::Choice { key } => {
                let chosen = first_use
                    .get(key.as_str())
                    .map(|&index| self.distribution[index].probability)
                    .ok_or(AnswerInvariant::ChoiceNotInDistribution)?;
                if self.probabilities().any(|p| p > chosen) {
                    return Err(AnswerInvariant::ChoiceNotHighest);
                }
            }
            DecisionValue::YesNo { p_yes } => {
                let (Some(&yes), Some(_), 2) = (
                    first_use.get("true"),
                    first_use.get("false"),
                    self.distribution.len(),
                ) else {
                    return Err(AnswerInvariant::YesNoKeys);
                };
                if *p_yes != self.distribution[yes].probability {
                    return Err(AnswerInvariant::PYesMismatch);
                }
            }
            DecisionValue::Score { value, statistic } => {
                for (index, entry) in self.distribution.iter().enumerate() {
                    if entry.key != index.to_string() {
                        return Err(AnswerInvariant::ScoreKey { index });
                    }
                }
                let matches = match statistic {
                    ScoreStatistic::Median => is_median(*value, self.probabilities()),
                    ScoreStatistic::Expected => {
                        let levels = self.distribution.len();
                        let tolerance = PROBABILITY_TOLERANCE * (levels.max(2) - 1) as f64;
                        (value - expected_level(self.probabilities())).abs() <= tolerance
                    }
                };
                if !matches {
                    return Err(AnswerInvariant::ScoreMismatch {
                        statistic: *statistic,
                    });
                }
            }
        }

        if let Some(confidence) = self.confidence {
            if !(confidence.is_finite() && (0.0..=1.0).contains(&confidence)) {
                return Err(AnswerInvariant::InvalidConfidence);
            }
        }
        Ok(())
    }

    fn probabilities(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.distribution.iter().map(|entry| entry.probability)
    }

    /// The id of the question this answers.
    pub fn question_id(&self) -> &str {
        &self.question_id
    }

    /// The distribution, in the protocol's order.
    pub fn distribution(&self) -> &[DecisionScore] {
        &self.distribution
    }

    /// The value derived from the distribution.
    pub fn value(&self) -> &DecisionValue {
        &self.value
    }

    /// The protocol's confidence, if it defines one for this kind.
    pub fn confidence(&self) -> Option<f64> {
        self.confidence
    }
}

/// Whether `value` is the median level: the first level whose cumulative
/// probability, summed in level order in `f64` from 0, reaches 0.5, or the
/// last level if none does. That is llama.cpp's loop for LFM3 d1
/// (`format_answer` in `tools/server/server-decision.cpp`), checked against
/// a transcription of it in the tests.
fn is_median(value: f64, probabilities: impl ExactSizeIterator<Item = f64>) -> bool {
    let last = probabilities.len().saturating_sub(1);
    let mut below = 0.0; // the cumulative mass before `level`
    for (level, probability) in probabilities.enumerate() {
        let through = below + probability;
        if value == level as f64 {
            // The mass before the level has not reached one half, and the
            // mass through it has (or no level does and this is the last).
            return (level == 0 || below < 0.5) && (through >= 0.5 || level == last);
        }
        below = through;
    }
    false
}

/// The sum of `i * p_i`, in level order.
fn expected_level(probabilities: impl Iterator<Item = f64>) -> f64 {
    probabilities
        .enumerate()
        .fold(0.0, |sum, (index, probability)| {
            sum + index as f64 * probability
        })
}

impl fmt::Debug for DecisionAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionAnswer")
            .field("question_id_bytes", &self.question_id.len())
            .field("distribution", &self.distribution)
            .field("value", &self.value)
            .field("confidence", &self.confidence)
            .finish()
    }
}

/// Content-free counts of the work a request took.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DecisionUsage {
    passes: u32,
    evaluated_tokens: u64,
}

impl DecisionUsage {
    /// Usage of `passes` model evaluations over `evaluated_tokens` tokens in
    /// total.
    pub fn new(passes: u32, evaluated_tokens: u64) -> Self {
        Self {
            passes,
            evaluated_tokens,
        }
    }

    /// Model evaluations run (LFM3 d1: one per question).
    pub fn passes(&self) -> u32 {
        self.passes
    }

    /// Tokens evaluated across all passes.
    pub fn evaluated_tokens(&self) -> u64 {
        self.evaluated_tokens
    }
}

/// The answers to a [`DecisionRequest`](super::DecisionRequest), one per
/// question in question order, and what they took.
///
/// `Debug` prints the answers' lengths and numbers, never ids or keys.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Decisions {
    answers: Vec<DecisionAnswer>,
    usage: DecisionUsage,
}

impl Decisions {
    /// The answers, in question order, and the usage.
    pub fn new(answers: Vec<DecisionAnswer>, usage: DecisionUsage) -> Self {
        Self { answers, usage }
    }

    /// The answers, in question order.
    pub fn answers(&self) -> &[DecisionAnswer] {
        &self.answers
    }

    /// The answer to question `question_id`: the first with that id (a
    /// protocol answers each question of a request once).
    pub fn answer(&self, question_id: &str) -> Option<&DecisionAnswer> {
        self.answers
            .iter()
            .find(|answer| answer.question_id == question_id)
    }

    /// What the request took.
    pub fn usage(&self) -> DecisionUsage {
        self.usage
    }

    /// Takes the answers out.
    pub fn into_answers(self) -> Vec<DecisionAnswer> {
        self.answers
    }
}

impl fmt::Debug for Decisions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decisions")
            .field("answers", &self.answers)
            .field("usage", &self.usage)
            .finish()
    }
}

/// The serde form of the `f64` fields (a probability, `p_yes`, a score
/// value, a confidence): in human-readable formats a decimal string with
/// the shortest digits that read back as the same `f64`, read back with a
/// correctly rounded parser of JSON's number grammar; in binary formats the
/// `f64` itself. A text round trip is therefore exact in every format,
/// serde_json included, and decoding needs no tolerance. NaN and infinities
/// never reach a human-readable wire.
mod exact_f64 {
    use super::*;
    use crate::ir::decision::json::{float_text, parse_float};

    pub(super) fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        if !serializer.is_human_readable() {
            serializer.serialize_f64(*value)
        } else if value.is_finite() {
            serializer.serialize_str(&float_text(*value))
        } else {
            Err(ser::Error::custom(
                "a probability, level or confidence is NaN or infinite, which a valid answer never holds",
            ))
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        if deserializer.is_human_readable() {
            deserializer.deserialize_str(DecimalVisitor)
        } else {
            f64::deserialize(deserializer)
        }
    }

    struct DecimalVisitor;

    impl Visitor<'_> for DecimalVisitor {
        type Value = f64;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a finite decimal number written as a string")
        }

        fn visit_str<E: de::Error>(self, text: &str) -> Result<f64, E> {
            parse_float(text).ok_or_else(|| {
                E::custom(
                    "a probability, level or confidence string is not a finite decimal number",
                )
            })
        }
    }

    /// `Option<f64>` in the same form, `None` as the format's null.
    pub(super) mod option {
        use super::*;

        pub(in super::super) fn serialize<S: Serializer>(
            value: &Option<f64>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            match value {
                None => serializer.serialize_none(),
                Some(value) => serializer.serialize_some(&Exact(*value)),
            }
        }

        pub(in super::super) fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Option<f64>, D::Error> {
            Ok(Option::<Exact>::deserialize(deserializer)?.map(|Exact(value)| value))
        }

        struct Exact(f64);

        impl Serialize for Exact {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                super::serialize(&self.0, serializer)
            }
        }

        impl<'de> Deserialize<'de> for Exact {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                super::deserialize(deserializer).map(Exact)
            }
        }
    }
}

/// `raw_score`'s serde form: in human-readable formats a number, or
/// `"-Infinity"` for `-inf`; in binary formats the `f32` itself.
mod raw_score {
    use super::*;

    const NEG_INFINITY: &str = "-Infinity";

    pub(super) fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
        if !serializer.is_human_readable() || value.is_finite() {
            serializer.serialize_f32(*value)
        } else if *value == f32::NEG_INFINITY {
            serializer.serialize_str(NEG_INFINITY)
        } else {
            Err(ser::Error::custom(
                "a raw score is NaN or +inf, which a valid answer never holds",
            ))
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
        if deserializer.is_human_readable() {
            deserializer.deserialize_any(RawScoreVisitor)
        } else {
            f32::deserialize(deserializer)
        }
    }

    struct RawScoreVisitor;

    impl Visitor<'_> for RawScoreVisitor {
        type Value = f32;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "a finite number or \"{NEG_INFINITY}\"")
        }

        fn visit_f64<E: de::Error>(self, value: f64) -> Result<f32, E> {
            let score = value as f32;
            if score.is_finite() {
                Ok(score)
            } else {
                Err(E::custom("a raw score is out of the f32 range"))
            }
        }

        fn visit_i64<E: de::Error>(self, value: i64) -> Result<f32, E> {
            self.visit_f64(value as f64)
        }

        fn visit_u64<E: de::Error>(self, value: u64) -> Result<f32, E> {
            self.visit_f64(value as f64)
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<f32, E> {
            if value == NEG_INFINITY {
                Ok(f32::NEG_INFINITY)
            } else {
                Err(E::custom(
                    "the only string a raw score may be is \"-Infinity\"",
                ))
            }
        }
    }

    #[cfg(feature = "schema")]
    pub(super) fn schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "oneOf": [{ "type": "number" }, { "const": NEG_INFINITY }]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANARY: &str = "canary-7f3a9c";

    fn entry(key: &str, raw_score: f32, probability: f64) -> DecisionScore {
        DecisionScore::new(key, raw_score, probability, DecisionOrigin::Caller)
    }

    fn levels(probabilities: &[f64]) -> Vec<DecisionScore> {
        probabilities
            .iter()
            .enumerate()
            .map(|(level, &p)| entry(&level.to_string(), 0.0, p))
            .collect()
    }

    fn choice(key: &str) -> DecisionValue {
        DecisionValue::Choice { key: key.into() }
    }

    fn median(value: f64) -> DecisionValue {
        DecisionValue::Score {
            value,
            statistic: ScoreStatistic::Median,
        }
    }

    fn expected(value: f64) -> DecisionValue {
        DecisionValue::Score {
            value,
            statistic: ScoreStatistic::Expected,
        }
    }

    fn yes_no(p_yes: f64) -> DecisionValue {
        DecisionValue::YesNo { p_yes }
    }

    fn answer(
        distribution: Vec<DecisionScore>,
        value: DecisionValue,
    ) -> Result<DecisionAnswer, DecisionError> {
        DecisionAnswer::new("q", distribution, value, None)
    }

    fn answer_of(
        probabilities: &[f64],
        value: DecisionValue,
    ) -> Result<DecisionAnswer, DecisionError> {
        answer(levels(probabilities), value)
    }

    fn broken(result: Result<DecisionAnswer, DecisionError>) -> AnswerInvariant {
        match result {
            Err(DecisionError::InvalidAnswer(invariant)) => invariant,
            other => panic!("expected an invalid answer, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_choice_answer_keeps_what_it_was_given() {
        let distribution = vec![
            entry("a", 2.0, 0.7),
            entry("b", -1.0, 0.2),
            DecisionScore::new("skip", f32::NEG_INFINITY, 0.1, DecisionOrigin::Fixed),
        ];
        let answer =
            DecisionAnswer::new("pick", distribution.clone(), choice("a"), Some(0.7)).unwrap();
        assert_eq!(answer.question_id(), "pick");
        assert_eq!(answer.distribution(), distribution.as_slice());
        assert_eq!(answer.value(), &choice("a"));
        assert_eq!(answer.confidence(), Some(0.7));
        let skip = &answer.distribution()[2];
        assert_eq!(
            (
                skip.key(),
                skip.raw_score(),
                skip.probability(),
                skip.origin()
            ),
            ("skip", f32::NEG_INFINITY, 0.1, DecisionOrigin::Fixed)
        );
    }

    #[test]
    fn the_choice_must_be_offered_and_most_probable() {
        let distribution = || vec![entry("a", 1.0, 0.6), entry("b", 0.0, 0.4)];
        assert_eq!(
            broken(answer(distribution(), choice("c"))),
            AnswerInvariant::ChoiceNotInDistribution
        );
        assert_eq!(
            broken(answer(distribution(), choice("b"))),
            AnswerInvariant::ChoiceNotHighest
        );
        // On a tie, any of the tied keys is a valid choice.
        let tied = || vec![entry("a", 0.0, 0.5), entry("b", 0.0, 0.5)];
        assert!(answer(tied(), choice("a")).is_ok());
        assert!(answer(tied(), choice("b")).is_ok());
    }

    #[test]
    fn yes_no_needs_true_and_false_and_the_probability_of_true() {
        let d1_order = || vec![entry("true", 1.0, 0.75), entry("false", 0.0, 0.25)];
        assert!(answer(d1_order(), yes_no(0.75)).is_ok());
        let other_order = vec![entry("false", 0.0, 0.25), entry("true", 1.0, 0.75)];
        assert!(answer(other_order, yes_no(0.75)).is_ok());

        assert_eq!(
            broken(answer(d1_order(), yes_no(0.25))),
            AnswerInvariant::PYesMismatch
        );
        let one_ulp_off = f64::from_bits(0.75f64.to_bits() + 1);
        assert_eq!(
            broken(answer(d1_order(), yes_no(one_ulp_off))),
            AnswerInvariant::PYesMismatch
        );
        assert_eq!(
            broken(answer(d1_order(), yes_no(f64::NAN))),
            AnswerInvariant::PYesMismatch
        );

        for keys in [
            vec![entry("yes", 1.0, 0.75), entry("no", 0.0, 0.25)],
            vec![entry("true", 1.0, 1.0)],
            vec![
                entry("true", 1.0, 0.5),
                entry("false", 0.0, 0.25),
                entry("maybe", 0.0, 0.25),
            ],
        ] {
            assert_eq!(
                broken(answer(keys, yes_no(0.75))),
                AnswerInvariant::YesNoKeys
            );
        }
    }

    #[test]
    fn score_entries_are_keyed_by_level() {
        let swapped = vec![entry("1", 0.0, 0.5), entry("0", 0.0, 0.5)];
        assert_eq!(
            broken(answer(swapped, median(0.0))),
            AnswerInvariant::ScoreKey { index: 0 }
        );
        let padded = vec![entry("0", 0.0, 0.5), entry("01", 0.0, 0.5)];
        assert_eq!(
            broken(answer(padded, median(0.0))),
            AnswerInvariant::ScoreKey { index: 1 }
        );
    }

    #[test]
    fn the_median_is_the_first_level_reaching_half_the_mass() {
        let cases: [(&[f64], f64); 6] = [
            (&[1.0], 0.0),
            (&[0.5, 0.5], 0.0),
            (&[0.2, 0.3, 0.5], 1.0),
            (&[0.1, 0.2, 0.7], 2.0),
            (&[0.0, 0.0, 0.0, 1.0], 3.0),
            (&[0.25, 0.2, 0.05, 0.5], 2.0),
        ];
        for (probabilities, level) in cases {
            assert!(
                answer(levels(probabilities), median(level)).is_ok(),
                "{probabilities:?}"
            );
            assert_eq!(
                broken(answer(levels(probabilities), median(level + 1.0))),
                AnswerInvariant::ScoreMismatch {
                    statistic: ScoreStatistic::Median
                }
            );
        }
        // A median is a whole level, compared exactly.
        assert!(answer(levels(&[0.2, 0.3, 0.5]), median(1.0 + 1e-12)).is_err());
    }

    #[test]
    fn the_median_follows_the_f64_distribution_not_its_f32_rounding() {
        // Review B3: float32 logits [0, 1e-8] at the effective temperature
        // 1.3373980522155762, in llama.cpp's double arithmetic.
        let temperature = f64::from(1.337_398_f32);
        let logits = [0.0f64, f64::from(1e-8f32)];
        let max = logits[1];
        let weights = logits.map(|logit| ((logit - max) / temperature).exp());
        let sum: f64 = weights.iter().sum();
        let probabilities = weights.map(|weight| weight / sum);
        assert!(probabilities[0] < 0.5 && probabilities[1] > 0.5);
        assert!((probabilities[0] - 0.4999999981306987).abs() < 1e-15);

        assert!(answer(levels(&probabilities), median(1.0)).is_ok());
        assert!(answer(levels(&probabilities), median(0.0)).is_err());
        // Rounded to f32 first, the same logits would give level 0.
        let rounded = probabilities.map(|p| f64::from(p as f32));
        assert_eq!(rounded, [0.5, 0.5]);
        assert!(answer(levels(&rounded), median(0.0)).is_ok());
    }

    /// llama.cpp's median loop for LFM3 d1 (`format_answer`), verbatim.
    fn median_level(probabilities: &[f64]) -> usize {
        let n = probabilities.len();
        let mut mass = 0.0;
        let mut level = n - 1;
        let mut i = 0;
        while i < n && mass < 0.5 {
            mass += probabilities[i];
            level = i;
            i += 1;
        }
        level
    }

    #[test]
    fn the_median_rule_is_upstreams_loop() {
        // Deterministic sample, unnormalized too, so the fallback to the
        // last level (no level reaching one half) is exercised.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20_000 {
            let levels = 1 + (next() * 12.0) as usize;
            let scale = if next() < 0.2 { 0.5 * next() } else { 1.0 };
            let mut probabilities: Vec<f64> = (0..levels)
                .map(|_| if next() < 0.2 { 0.0 } else { next() })
                .collect();
            let total: f64 = probabilities.iter().sum();
            if total > 0.0 {
                probabilities
                    .iter_mut()
                    .for_each(|p| *p = *p / total * scale);
            }
            let median = median_level(&probabilities);
            for level in 0..levels {
                assert_eq!(
                    is_median(level as f64, probabilities.iter().copied()),
                    level == median,
                    "{probabilities:?} level {level}"
                );
            }
            for odd in [-1.0, 0.5, levels as f64, f64::NAN] {
                assert!(!is_median(odd, probabilities.iter().copied()));
            }
        }
        assert_eq!(median_level(&[0.1, 0.1, 0.1]), 2);
        assert!(is_median(2.0, [0.1, 0.1, 0.1].into_iter()));
    }

    #[test]
    fn the_expected_value_is_the_mean_level_within_tolerance() {
        let probabilities = [0.25, 0.25, 0.5];
        assert!(answer(levels(&probabilities), expected(1.25)).is_ok());
        let tolerance = PROBABILITY_TOLERANCE * 2.0;
        assert!(answer(levels(&probabilities), expected(1.25 + tolerance * 0.9)).is_ok());
        assert_eq!(
            broken(answer(
                levels(&probabilities),
                expected(1.25 + tolerance * 1.1)
            )),
            AnswerInvariant::ScoreMismatch {
                statistic: ScoreStatistic::Expected
            }
        );
        assert!(answer(levels(&[1.0]), expected(0.0)).is_ok());
        assert!(answer(levels(&[0.5, 0.5]), expected(f64::NAN)).is_err());
    }

    #[test]
    fn probabilities_must_form_a_distribution() {
        let cases: [(Vec<DecisionScore>, AnswerInvariant); 7] = [
            (vec![], AnswerInvariant::EmptyDistribution),
            (
                vec![entry("", 0.0, 1.0)],
                AnswerInvariant::EmptyKey { index: 0 },
            ),
            (
                vec![
                    entry("a", 0.0, 0.5),
                    entry("b", 0.0, 0.25),
                    entry("a", 0.0, 0.25),
                ],
                AnswerInvariant::DuplicateKey { index: 2, first: 0 },
            ),
            (
                vec![entry("a", 0.0, 1.5), entry("b", 0.0, -0.5)],
                AnswerInvariant::InvalidProbability { index: 0 },
            ),
            (
                vec![
                    entry("a", 0.0, 1.0),
                    entry("b", 0.0, -0.0),
                    entry("c", 0.0, -1e-300),
                ],
                AnswerInvariant::InvalidProbability { index: 2 },
            ),
            (
                vec![entry("a", 0.0, f64::NAN), entry("b", 0.0, 1.0)],
                AnswerInvariant::InvalidProbability { index: 0 },
            ),
            (
                vec![entry("a", 0.0, 0.5), entry("b", 0.0, 0.5 - 2e-6)],
                AnswerInvariant::ProbabilitySum {
                    sum: 0.5 + (0.5 - 2e-6),
                },
            ),
        ];
        for (distribution, invariant) in cases {
            assert_eq!(broken(answer(distribution, choice("a"))), invariant);
        }
        assert!(answer(
            vec![entry("a", 0.0, 0.5), entry("b", 0.0, 0.5 - 5e-7)],
            choice("a")
        )
        .is_ok());
        assert!(answer(vec![entry("a", 0.0, f64::INFINITY)], choice("a")).is_err());
    }

    #[test]
    fn raw_scores_may_be_minus_infinity_but_not_nan_or_plus_infinity() {
        for raw in [f32::NEG_INFINITY, f32::MIN, f32::MAX, 0.0, -0.0] {
            let distribution = vec![entry("a", 0.5, 1.0), entry("b", raw, 0.0)];
            assert!(answer(distribution, choice("a")).is_ok(), "{raw}");
        }
        for raw in [f32::NAN, f32::INFINITY] {
            assert_eq!(
                broken(answer(vec![entry("a", raw, 1.0)], choice("a"))),
                AnswerInvariant::InvalidRawScore { index: 0 }
            );
        }
        // Contract: -inf is allowed only beside at least one finite score.
        let ruled_out = vec![
            entry("a", f32::NEG_INFINITY, 0.5),
            entry("b", f32::NEG_INFINITY, 0.5),
        ];
        assert_eq!(
            broken(answer(ruled_out, choice("a"))),
            AnswerInvariant::NoFiniteRawScore
        );
    }

    #[test]
    fn confidence_and_question_id_are_checked() {
        let one = || vec![entry("a", 0.0, 1.0)];
        for confidence in [None, Some(0.0), Some(1.0), Some(0.3)] {
            assert!(DecisionAnswer::new("q", one(), choice("a"), confidence).is_ok());
        }
        for confidence in [-0.1, 1.000001, f64::NAN, f64::INFINITY] {
            assert_eq!(
                broken(DecisionAnswer::new(
                    "q",
                    one(),
                    choice("a"),
                    Some(confidence)
                )),
                AnswerInvariant::InvalidConfidence
            );
        }
        assert_eq!(
            broken(DecisionAnswer::new("", one(), choice("a"), None)),
            AnswerInvariant::EmptyQuestionId
        );
    }

    #[test]
    fn raw_scores_are_numbers_or_minus_infinity_in_json_and_f32_in_bincode() {
        let finite = entry("a", 1.5, 0.25);
        assert_eq!(
            serde_json::to_string(&finite).unwrap(),
            r#"{"key":"a","raw_score":1.5,"probability":"0.25","origin":"Caller"}"#
        );
        let ruled_out = entry("a", f32::NEG_INFINITY, 0.0);
        let text = serde_json::to_string(&ruled_out).unwrap();
        assert_eq!(
            text,
            r#"{"key":"a","raw_score":"-Infinity","probability":"0.0","origin":"Caller"}"#
        );
        assert_eq!(
            serde_json::from_str::<DecisionScore>(&text).unwrap(),
            ruled_out
        );
        for score in [
            finite,
            ruled_out,
            entry("a", 3.4028235e38, 1.0),
            entry("a", 1e-45, 1.0),
        ] {
            let text = serde_json::to_string(&score).unwrap();
            assert_eq!(
                serde_json::from_str::<DecisionScore>(&text).unwrap(),
                score,
                "{text}"
            );
            let bytes = bincode::serialize(&score).unwrap();
            assert_eq!(
                bincode::deserialize::<DecisionScore>(&bytes).unwrap(),
                score
            );
        }

        let integer: DecisionScore = serde_json::from_str(
            r#"{"key":"a","raw_score":3,"probability":"1.0","origin":"Fixed"}"#,
        )
        .unwrap();
        assert_eq!(
            (integer.raw_score(), integer.origin()),
            (3.0, DecisionOrigin::Fixed)
        );
        for raw in [
            r#""Infinity""#,
            r#""NaN""#,
            r#""-inf""#,
            "null",
            "1e39",
            "[]",
        ] {
            let text =
                format!(r#"{{"key":"a","raw_score":{raw},"probability":"1.0","origin":"Caller"}}"#);
            assert!(
                serde_json::from_str::<DecisionScore>(&text).is_err(),
                "{raw}"
            );
        }

        // NaN and +inf never reach the JSON wire; bincode carries any f32.
        for raw in [f32::NAN, f32::INFINITY] {
            assert!(serde_json::to_string(&entry("a", raw, 1.0)).is_err());
            let bytes = bincode::serialize(&entry("a", raw, 1.0)).unwrap();
            let back: DecisionScore = bincode::deserialize(&bytes).unwrap();
            assert_eq!(back.raw_score().to_bits(), raw.to_bits());
        }
    }

    #[test]
    fn f64_fields_are_decimal_strings_in_human_readable_formats() {
        let accepted: [(&str, f64); 8] = [
            ("0.25", 0.25),
            ("1.0", 1.0),
            ("1", 1.0),
            ("0", 0.0),
            ("-0.0", -0.0),
            ("1E-3", 0.001),
            ("2.5e-1", 0.25),
            ("0.18475726705416062", 0.18475726705416062),
        ];
        for (text, value) in accepted {
            let json =
                format!(r#"{{"key":"a","raw_score":0,"probability":"{text}","origin":"Caller"}}"#);
            let score: DecisionScore = serde_json::from_str(&json).unwrap();
            assert_eq!(score.probability().to_bits(), value.to_bits(), "{text}");
        }
        // A JSON number is refused (serde_json would have rounded it already),
        // and so is any string outside JSON's number grammar.
        for probability in [
            "0.25",
            r#""NaN""#,
            r#""inf""#,
            r#""+1""#,
            r#"".5""#,
            r#""0x1""#,
            r#""""#,
            r#""1e400""#,
            r#""0.5 0.5""#,
            "null",
        ] {
            let json = format!(
                r#"{{"key":"a","raw_score":0,"probability":{probability},"origin":"Caller"}}"#
            );
            assert!(
                serde_json::from_str::<DecisionScore>(&json).is_err(),
                "{probability}"
            );
        }
        let value = r#"{"Score":{"value":2.0,"statistic":"Median"}}"#;
        assert!(serde_json::from_str::<DecisionValue>(value).is_err());
        let value = r#"{"Score":{"value":"2.0","statistic":"Median"}}"#;
        assert_eq!(
            serde_json::from_str::<DecisionValue>(value).unwrap(),
            median(2.0)
        );
        assert_eq!(
            serde_json::to_string(&yes_no(0.125)).unwrap(),
            r#"{"YesNo":{"p_yes":"0.125"}}"#
        );
        // NaN and infinities never reach the JSON wire; bincode carries them
        // and the answer check refuses them.
        for probability in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(serde_json::to_string(&entry("a", 0.0, probability)).is_err());
            assert!(serde_json::to_string(&yes_no(probability)).is_err());
        }
        let wire = Wire::new(vec![entry("a", 0.0, f64::NAN)], choice("a"));
        let bytes = bincode::serialize(&wire).unwrap();
        assert!(bincode::deserialize::<DecisionAnswer>(&bytes).is_err());
    }

    #[test]
    fn deserializing_an_answer_checks_it() {
        let valid = r#"{"question_id":"q","distribution":[{"key":"true","raw_score":2.0,"probability":"0.75","origin":"Caller"},{"key":"false","raw_score":"-Infinity","probability":"0.25","origin":"Caller"}],"value":{"YesNo":{"p_yes":"0.75"}},"confidence":null}"#;
        let answer: DecisionAnswer = serde_json::from_str(valid).unwrap();
        assert_eq!(serde_json::to_string(&answer).unwrap(), valid);
        let bytes = bincode::serialize(&answer).unwrap();
        assert_eq!(
            bincode::deserialize::<DecisionAnswer>(&bytes).unwrap(),
            answer
        );

        let mismatch = valid.replace(r#""p_yes":"0.75""#, r#""p_yes":"0.5""#);
        let error = serde_json::from_str::<DecisionAnswer>(&mismatch).unwrap_err();
        assert!(
            error.to_string().contains("p_yes is not the probability"),
            "{error}"
        );
        let extra = valid.replace(r#""confidence":null"#, r#""confidence":null,"extra":1"#);
        assert!(serde_json::from_str::<DecisionAnswer>(&extra).is_err());
        let unknown_value = valid.replace(
            r#"{"YesNo":{"p_yes":"0.75"}}"#,
            r#"{"YesNo":{"p_yes":"0.75","p_no":"0.25"}}"#,
        );
        assert!(serde_json::from_str::<DecisionAnswer>(&unknown_value).is_err());
        let confident = valid.replace(r#""confidence":null"#, r#""confidence":"0.75""#);
        let decoded: DecisionAnswer = serde_json::from_str(&confident).unwrap();
        assert_eq!(decoded.confidence(), Some(0.75));
        let number = valid.replace(r#""confidence":null"#, r#""confidence":0.75"#);
        assert!(serde_json::from_str::<DecisionAnswer>(&number).is_err());

        // bincode goes through the same check.
        let mut invalid = answer.clone();
        invalid.confidence = Some(2.0);
        let bytes = bincode::serialize(&invalid).unwrap();
        assert!(bincode::deserialize::<DecisionAnswer>(&bytes).is_err());
    }

    /// An answer's serde form, written without the checks, to put invalid
    /// answers on the wire.
    #[derive(Serialize)]
    struct Wire {
        question_id: String,
        distribution: Vec<DecisionScore>,
        value: DecisionValue,
        #[serde(with = "super::exact_f64::option")]
        confidence: Option<f64>,
    }

    impl Wire {
        fn new(distribution: Vec<DecisionScore>, value: DecisionValue) -> Self {
            Self {
                question_id: "q".into(),
                distribution,
                value,
                confidence: None,
            }
        }

        /// The answer decoded from JSON and from bincode.
        fn decode(&self) -> [Result<DecisionAnswer, String>; 2] {
            let text = serde_json::to_string(self).unwrap();
            let bytes = bincode::serialize(self).unwrap();
            [
                serde_json::from_str::<DecisionAnswer>(&text).map_err(|error| error.to_string()),
                bincode::deserialize::<DecisionAnswer>(&bytes).map_err(|error| error.to_string()),
            ]
        }
    }

    /// `results` are all refusals naming `reason`.
    fn refused(results: [Result<DecisionAnswer, String>; 2], reason: &str) {
        for (format, result) in ["json", "bincode"].into_iter().zip(results) {
            let error = result.expect_err(format);
            assert!(error.contains(reason), "{format}: {error}");
        }
    }

    fn accepted(results: [Result<DecisionAnswer, String>; 2]) {
        for (format, result) in ["json", "bincode"].into_iter().zip(results) {
            assert!(result.is_ok(), "{format}: {result:?}");
        }
    }

    #[test]
    fn decoding_is_exactly_as_strict_as_new_in_every_format() {
        // Exact half: [0.5, 0.5] reaches one half at level 0, so the median
        // is level 0 and only level 0.
        let half = || levels(&[0.5, 0.5]);
        accepted(Wire::new(half(), median(0.0)).decode());
        refused(Wire::new(half(), median(1.0)).decode(), "not the median");

        // Near the boundary: the mass through level 0 falls 1e-13 short of
        // one half, so the median is level 1.
        let short = || levels(&[0.5 - 1e-13, 0.5 + 1e-13]);
        refused(Wire::new(short(), median(0.0)).decode(), "not the median");
        accepted(Wire::new(short(), median(1.0)).decode());

        // A choice losing by 1e-13, or by one ulp, is not the choice.
        let close = || vec![entry("a", 0.0, 0.5 + 1e-13), entry("b", 0.0, 0.5 - 1e-13)];
        refused(
            Wire::new(close(), choice("b")).decode(),
            "highest probability",
        );
        accepted(Wire::new(close(), choice("a")).decode());
        let ulp = || vec![entry("a", 0.0, 0.5), entry("b", 0.0, 0.5f64.next_up())];
        refused(
            Wire::new(ulp(), choice("a")).decode(),
            "highest probability",
        );
        accepted(Wire::new(ulp(), choice("b")).decode());

        // Review B3: the median read from the float32-rounded distribution.
        let b3 = || levels(&[0.4999999981306987, 0.5000000018693014]);
        refused(Wire::new(b3(), median(0.0)).decode(), "not the median");
        accepted(Wire::new(b3(), median(1.0)).decode());
    }

    #[test]
    fn text_round_trips_reproduce_every_bit() {
        // Score logits [0, 0.5343, 0.5343, 0] at T = 1, in llama.cpp's
        // arithmetic: the mass through level 1 falls one ulp short of one
        // half, so the median is level 2. serde_json, reading the
        // probabilities as numbers, would read level 0's one ulp high and
        // move the median; as decimal strings they come back exactly.
        let logits = [0.0f32, 0.5343, 0.5343, 0.0].map(f64::from);
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = logits.map(|logit| ((logit - max) / 1.0).exp());
        let sum = weights.iter().fold(0.0, |sum, weight| sum + weight);
        let probabilities = weights.map(|weight| weight / sum);
        assert!(probabilities[0] + probabilities[1] < 0.5);
        let scored = DecisionAnswer::new(
            "q",
            levels(&probabilities),
            median(2.0),
            Some(probabilities[1]),
        )
        .unwrap();

        let text = serde_json::to_string(&scored).unwrap();
        assert!(
            text.contains(r#""probability":"0.18475726705416062""#),
            "{text}"
        );
        let bytes = bincode::serialize(&scored).unwrap();
        for back in [
            serde_json::from_str::<DecisionAnswer>(&text).unwrap(),
            bincode::deserialize::<DecisionAnswer>(&bytes).unwrap(),
        ] {
            assert_eq!(back.value(), &median(2.0));
            assert_eq!(
                back.confidence().map(f64::to_bits),
                Some(probabilities[1].to_bits())
            );
            let bits = |answer: &DecisionAnswer| {
                answer
                    .distribution()
                    .iter()
                    .map(|entry| entry.probability().to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&back), bits(&scored));
        }

        // One ulp more on level 0 makes level 1 the median, and only level 1.
        let mut nudged = probabilities;
        nudged[0] = nudged[0].next_up();
        assert!(nudged[0] + nudged[1] >= 0.5);
        assert!(answer_of(&nudged, median(2.0)).is_err());
        refused(
            Wire::new(levels(&nudged), median(2.0)).decode(),
            "not the median",
        );
        accepted(Wire::new(levels(&nudged), median(1.0)).decode());
    }

    #[test]
    fn decisions_keep_answers_in_order() {
        let first = answer(vec![entry("a", 0.0, 1.0)], choice("a")).unwrap();
        let second =
            DecisionAnswer::new("level", levels(&[0.25, 0.75]), median(1.0), Some(0.75)).unwrap();
        let usage = DecisionUsage::new(2, 731);
        assert_eq!((usage.passes(), usage.evaluated_tokens()), (2, 731));
        let decisions = Decisions::new(vec![first.clone(), second.clone()], usage);
        assert_eq!(
            decisions.answers(),
            [first.clone(), second.clone()].as_slice()
        );
        assert_eq!(decisions.answer("level"), Some(&second));
        assert!(decisions.answer("missing").is_none());
        assert_eq!(decisions.usage(), usage);
        assert_eq!(decisions.into_answers(), vec![first, second]);
        assert_eq!(DecisionUsage::default(), DecisionUsage::new(0, 0));
    }

    #[test]
    fn debug_never_prints_ids_or_keys() {
        let answer = DecisionAnswer::new(
            CANARY,
            vec![
                entry(CANARY, 1.25, 0.75),
                entry(&format!("{CANARY}-b"), f32::NEG_INFINITY, 0.25),
            ],
            choice(CANARY),
            Some(0.75),
        )
        .unwrap();
        let decisions = Decisions::new(vec![answer.clone()], DecisionUsage::new(1, 9));
        let rendered = format!("{answer:?} {decisions:?} {:?}", answer.value());
        assert!(!rendered.contains(CANARY), "{rendered}");
        assert!(rendered.contains("question_id_bytes: 13"), "{rendered}");
        assert!(rendered.contains("key_bytes: 15"), "{rendered}");
        assert!(rendered.contains("raw_score: -inf"), "{rendered}");
        assert!(rendered.contains("evaluated_tokens: 9"), "{rendered}");
        let score = DecisionValue::Score {
            value: 2.0,
            statistic: ScoreStatistic::Median,
        };
        assert_eq!(
            format!("{score:?}"),
            "Score { value: 2.0, statistic: Median }"
        );
        assert_eq!(format!("{:?}", yes_no(0.5)), "YesNo { p_yes: 0.5 }");
    }

    #[test]
    fn invariant_messages_never_quote_keys() {
        let cases = [
            answer(vec![entry(CANARY, 0.0, 1.0)], choice("missing")),
            answer(
                vec![entry(CANARY, 0.0, 0.5), entry(CANARY, 0.0, 0.5)],
                choice(CANARY),
            ),
            DecisionAnswer::new(
                CANARY,
                vec![entry(CANARY, f32::NAN, 1.0)],
                choice(CANARY),
                None,
            ),
            answer(
                vec![entry(CANARY, 0.0, 0.25), entry("x", 0.0, 0.75)],
                choice(CANARY),
            ),
            answer(vec![entry(CANARY, 0.0, 1.0)], median(0.0)),
        ];
        for result in cases {
            let error = result.unwrap_err();
            for text in [error.to_string(), format!("{error:?}")] {
                assert!(!text.contains(CANARY), "{text}");
            }
        }
    }
}

//! Exact Liquid label construction and bounded, ordered prompt preparation.

use super::liquid::{
    validate_request, DecisionProtocolError as Error, MAX_INPUT_TOKENS, MAX_RENDERED_BYTES,
    TEMPLATE_SHA256,
};
use crate::ir::decision::{
    DecisionContent, DecisionContentView, DecisionQuestion, DecisionQuestionKind, DecisionRequest,
    DecisionState, JsonNumber, JsonView, OrderedJson,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::ops::Range;

/// Admission ceilings for vocabulary CONTROL spellings (the artifact has 124).
pub const MAX_CONTROL_SPELLINGS: usize = 256;
/// Aggregate UTF-8 bytes across admitted CONTROL spellings.
pub const MAX_CONTROL_BYTES: usize = 64 << 10;

/// The answer operation of this profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerKind {
    Choice,
    Noul,
    Score,
}

impl AnswerKind {
    /// The Liquid wire name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Noul => "noul",
            Self::Score => "score",
        }
    }
}

/// One output's aliases within the selected-logit array.
#[derive(Clone)]
pub struct LabelGroup {
    pub key: String,
    pub code: String,
    pub range: Range<usize>,
}

impl fmt::Debug for LabelGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LabelGroup")
            .field("aliases", &self.range.len())
            .finish_non_exhaustive()
    }
}

/// The immutable finalization plan for one question.
#[derive(Clone)]
pub struct AnswerPlan {
    pub question: String,
    pub kind: AnswerKind,
    pub groups: Vec<LabelGroup>,
    pub temperature: f32,
    /// The original native metadata string, before the upstream f32 parse.
    pub temperature_text: Option<String>,
}

impl fmt::Debug for AnswerPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnswerPlan")
            .field("kind", &self.kind)
            .field("groups", &self.groups.len())
            .finish_non_exhaustive()
    }
}

/// A complete pass, retaining no rendered prompt and exposing no native pointer.
#[derive(Clone)]
pub struct PreparedQuestion {
    pub tokens: Vec<i32>,
    pub select: Vec<i32>,
    pub plan: AnswerPlan,
}

impl fmt::Debug for PreparedQuestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedQuestion")
            .field("input_tokens", &self.tokens.len())
            .field("selected_logits", &self.select.len())
            .field("plan", &self.plan)
            .finish()
    }
}

/// Validated metadata and control spellings shared by every request.
#[derive(Clone)]
pub struct Profile {
    controls: Vec<String>,
    temperatures: BTreeMap<String, (String, f32)>,
}

impl fmt::Debug for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Profile")
            .field("control_spellings", &self.controls.len())
            .field("temperature_entries", &self.temperatures.len())
            .finish()
    }
}

impl Profile {
    /// Admits the pinned architecture, decision type, template and calibration.
    pub fn from_metadata(
        template: &str,
        metadata: &[(String, String)],
        mut control_spellings: Vec<String>,
    ) -> Result<Self, Error> {
        let get = |key: &str| {
            metadata
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        if get("general.architecture") != Some("lfm3")
            || get("lfm3.decision.type") != Some("lfm3-d1")
        {
            return Err(Error::UnsupportedMetadata);
        }
        if format!("{:x}", Sha256::digest(template.as_bytes())) != TEMPLATE_SHA256 {
            return Err(Error::UnsupportedMetadata);
        }
        let mut temperatures = BTreeMap::new();
        let mut seen = HashSet::new();
        for (key, value) in metadata {
            if !seen.insert(key) {
                return Err(Error::InvalidMetadata);
            }
            if key.starts_with("lfm3.decision.score_bias.") {
                return Err(Error::UnsupportedMetadata);
            }
            if let Some(name) = key.strip_prefix("lfm3.decision.temperature.") {
                // Image calibration is irrelevant to this text-only profile.
                if name == "image" {
                    continue;
                }
                let (kind, bucket) = name
                    .split_once('.')
                    .map_or((name, None), |(k, b)| (k, Some(b)));
                if !matches!(kind, "choice" | "noul" | "score")
                    || bucket.is_some_and(|b| !matches!(b, "2" | "3_5" | "6_10" | "11"))
                {
                    return Err(Error::UnsupportedMetadata);
                }
                let temperature: f32 = value.parse().map_err(|_| Error::InvalidMetadata)?;
                if !temperature.is_finite() || temperature <= 0.0 {
                    return Err(Error::InvalidMetadata);
                }
                temperatures.insert(name.to_owned(), (value.clone(), temperature));
            } else if key.starts_with("lfm3.decision.") && key != "lfm3.decision.type" {
                return Err(Error::UnsupportedMetadata);
            }
        }
        control_spellings.retain(|s| s.len() >= 2);
        if control_spellings.len() > MAX_CONTROL_SPELLINGS
            || control_spellings
                .iter()
                .try_fold(0usize, |total, spelling| total.checked_add(spelling.len()))
                .is_none_or(|total| total > MAX_CONTROL_BYTES)
        {
            return Err(Error::UnsupportedMetadata);
        }
        // A one-character spelling cannot be broken after its first character.
        // A spelling containing the inserted separator could make escaping cycle.
        if control_spellings
            .iter()
            .any(|s| s.chars().count() < 2 || s.contains('\u{200b}'))
        {
            return Err(Error::UnsupportedMetadata);
        }
        control_spellings.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Ok(Self {
            controls: control_spellings,
            temperatures,
        })
    }

    fn temperature(&self, kind: AnswerKind, count: usize) -> (Option<String>, f32) {
        let bucket = match count {
            0..=2 => "2",
            3..=5 => "3_5",
            6..=10 => "6_10",
            _ => "11",
        };
        for name in [
            format!("{}.{}", kind.name(), bucket),
            kind.name().to_owned(),
        ] {
            if let Some((text, value)) = self.temperatures.get(&name) {
                return (Some(text.clone()), *value);
            }
        }
        (None, 1.0)
    }
}

/// Narrow preparation boundary, allowing vocabulary-only conformance checks.
pub trait PreparationBackend {
    /// Tokenizes with `add_special=false` and explicit special-token parsing.
    fn tokenize(&self, text: &str, parse_special: bool) -> Result<Vec<i32>, Error>;
    /// Renders ordered JSON through the pinned native Jinja implementation.
    fn render(&mut self, ordered_json: &str, max_output_bytes: usize) -> Result<String, Error>;
}

/// Preflights all questions before returning any executable passes.
pub fn prepare(
    request: &DecisionRequest,
    profile: &Profile,
    backend: &mut impl PreparationBackend,
) -> Result<Vec<PreparedQuestion>, Error> {
    validate_request(request)?;
    request
        .questions()
        .iter()
        .map(|question| {
            let (_, pass) = prepare_one(request.state(), question, profile, backend)?;
            Ok(pass)
        })
        .collect()
}

/// Prepares a question and returns prompt bytes for isolated conformance tests.
///
/// Runtime callers use [`prepare`], which releases each prompt immediately.
pub fn prepare_one(
    state: &DecisionState,
    question: &DecisionQuestion,
    profile: &Profile,
    backend: &mut impl PreparationBackend,
) -> Result<(String, PreparedQuestion), Error> {
    // This API is also independently callable by the conformance runner.
    validate_request(&DecisionRequest::new(state.clone(), vec![question.clone()]))?;
    let (kind, options) = question_options(question);
    let (codes, aliases) = labels(kind, &options, backend)?;
    let mut select = Vec::new();
    let mut groups = Vec::with_capacity(options.len());
    for ((key, _), (code, aliases)) in options.iter().zip(codes.into_iter().zip(aliases)) {
        let start = select.len();
        select.extend(aliases);
        groups.push(LabelGroup {
            key: key.clone(),
            code,
            range: start..select.len(),
        });
    }
    let mut unique = HashSet::new();
    if select.iter().any(|token| !unique.insert(*token)) {
        return Err(Error::NoSingleTokenLabel);
    }
    let instructions = question
        .instructions()
        .and_then(|i| i.as_content())
        .ok_or(Error::MissingInstructions)?;
    let mut input = String::from("{");
    write_field("id", &profile.controls, &mut input)?;
    write_text(question.id(), &profile.controls, &mut input)?;
    input.push(',');
    write_field("type", &profile.controls, &mut input)?;
    write_text(kind.name(), &profile.controls, &mut input)?;
    input.push(',');
    write_field("instructions", &profile.controls, &mut input)?;
    write_content(Some(instructions), &profile.controls, &mut input)?;
    input.push(',');
    write_field("state", &profile.controls, &mut input)?;
    write_content(state.body(), &profile.controls, &mut input)?;
    input.push(',');
    write_field("options", &profile.controls, &mut input)?;
    input.push('[');
    for (index, ((key, description), group)) in options.iter().zip(&groups).enumerate() {
        if index > 0 {
            input.push(',');
        }
        input.push('{');
        write_field("key", &profile.controls, &mut input)?;
        write_text(key, &profile.controls, &mut input)?;
        input.push(',');
        write_field("description", &profile.controls, &mut input)?;
        write_content(*description, &profile.controls, &mut input)?;
        input.push(',');
        write_field("label", &profile.controls, &mut input)?;
        write_text(&group.code, &profile.controls, &mut input)?;
        input.push('}');
    }
    input.push_str("],\"images\":[]}");
    if input.len() > MAX_RENDERED_BYTES {
        return Err(Error::BudgetExceeded);
    }
    // The template is admitted by hash. Each source value appears at most once;
    // this upper bound includes JSON escapes, numbers, separators and the only
    // pretty-printed field (state), before entering the native allocator.
    let value = OrderedJson::parse_with_limits(
        &input,
        &crate::ir::decision::JsonLimits::default()
            .with_max_bytes(MAX_RENDERED_BYTES)
            .with_max_nodes(65536 + 4096),
    )?;
    let bound = render_bound(&value, 0).saturating_add(4096);
    if bound > MAX_RENDERED_BYTES {
        return Err(Error::BudgetExceeded);
    }
    let prompt = backend.render(&input, MAX_RENDERED_BYTES)?;
    if prompt.len() > MAX_RENDERED_BYTES {
        return Err(Error::BudgetExceeded);
    }
    let tokens = backend.tokenize(&prompt, true)?;
    if tokens.is_empty() || tokens.len() > MAX_INPUT_TOKENS {
        return Err(Error::BudgetExceeded);
    }
    let (temperature_text, temperature) = profile.temperature(kind, options.len());
    Ok((
        prompt,
        PreparedQuestion {
            tokens,
            select,
            plan: AnswerPlan {
                question: question.id().to_owned(),
                kind,
                groups,
                temperature,
                temperature_text,
            },
        },
    ))
}

type OptionView<'a> = (String, Option<&'a DecisionContent>);

fn question_options(question: &DecisionQuestion) -> (AnswerKind, Vec<OptionView<'_>>) {
    match question.kind() {
        DecisionQuestionKind::Choice { options } => (
            AnswerKind::Choice,
            options
                .iter()
                .map(|o| (o.key().to_owned(), o.description()))
                .collect(),
        ),
        DecisionQuestionKind::YesNo { yes, no } => (
            AnswerKind::Noul,
            vec![
                ("true".to_owned(), yes.as_ref()),
                ("false".to_owned(), no.as_ref()),
            ],
        ),
        DecisionQuestionKind::Score { levels } => (
            AnswerKind::Score,
            levels
                .iter()
                .enumerate()
                .map(|(i, l)| (i.to_string(), l.description()))
                .collect(),
        ),
    }
}

fn labels(
    kind: AnswerKind,
    options: &[OptionView<'_>],
    backend: &impl PreparationBackend,
) -> Result<(Vec<String>, Vec<Vec<i32>>), Error> {
    let mut cache = HashMap::<String, Vec<i32>>::new();
    let mut tokenize = |text: &str| -> Result<Vec<i32>, Error> {
        if let Some(tokens) = cache.get(text) {
            return Ok(tokens.clone());
        }
        let tokens = backend.tokenize(text, false)?;
        cache.insert(text.to_owned(), tokens.clone());
        Ok(tokens)
    };
    let mut texts = Vec::with_capacity(options.len());
    let mut groups = Vec::with_capacity(options.len());
    if kind != AnswerKind::Choice {
        for (key, _) in options {
            let forms: Vec<&str> = match kind {
                AnswerKind::Score => vec![key],
                _ if key == "true" => vec!["yes", "Yes", "YES"],
                _ => vec!["no", "No", "NO"],
            };
            let mut group = Vec::new();
            for form in forms {
                let tokens = tokenize(form)?;
                if tokens.len() == 1 && !group.contains(&tokens[0]) {
                    group.push(tokens[0]);
                }
            }
            if group.is_empty() {
                return Err(Error::NoSingleTokenLabel);
            }
            texts.push(key.clone());
            groups.push(group);
        }
        return Ok((texts, groups));
    }
    let is_letters = options
        .iter()
        .all(|(key, _)| key.len() == 1 && key.as_bytes()[0].is_ascii_alphabetic());
    let codes: Vec<String> = options
        .iter()
        .enumerate()
        .map(|(i, (key, _))| {
            if is_letters {
                key.clone()
            } else if i < 26 {
                char::from(b'A' + i as u8).to_string()
            } else {
                format!(
                    "{}{}",
                    char::from(b'A' + ((i - 26) / 26) as u8),
                    char::from(b'A' + ((i - 26) % 26) as u8)
                )
            }
        })
        .collect();
    let mut pool: Vec<String> = (b'A'..=b'Z').map(|c| char::from(c).to_string()).collect();
    pool.extend((0..100).map(|i| format!("{i:02}")));
    pool.extend((b'a'..=b'z').map(|c| char::from(c).to_string()));
    pool.extend((0..200).map(|i| format!("#{i}")));
    pool.extend(
        (b'A'..=b'Z').flat_map(|a| {
            (b'A'..=b'Z').map(move |b| format!("{}{}", char::from(a), char::from(b)))
        }),
    );
    let mut used = HashSet::new();
    for preferred in codes {
        let mut taken = false;
        for code in std::iter::once(&preferred).chain(pool.iter()) {
            let primary = tokenize(code)?;
            if primary.len() != 1 || used.contains(&primary[0]) {
                continue;
            }
            used.insert(primary[0]);
            let mut group = primary;
            let alias = tokenize(&format!(" {code}"))?;
            if alias.len() == 1 && alias[0] != group[0] {
                group.push(alias[0]);
            }
            texts.push(code.clone());
            groups.push(group);
            taken = true;
            break;
        }
        if !taken {
            return Err(Error::NoSingleTokenLabel);
        }
    }
    Ok((texts, groups))
}

/// Breaks only CONTROL spellings, longest first, including overlapping matches.
pub fn escape_controls(text: &str, spellings: &[String]) -> Result<String, Error> {
    if text.len() > MAX_RENDERED_BYTES {
        return Err(Error::BudgetExceeded);
    }
    if spellings.len() > MAX_CONTROL_SPELLINGS
        || spellings
            .iter()
            .try_fold(0usize, |total, spelling| total.checked_add(spelling.len()))
            .is_none_or(|total| total > MAX_CONTROL_BYTES)
    {
        return Err(Error::UnsupportedMetadata);
    }
    if spellings
        .iter()
        .any(|s| s.chars().count() < 2 || s.contains('\u{200b}'))
    {
        return Err(Error::UnsupportedMetadata);
    }
    let mut spellings: Vec<&str> = spellings.iter().map(String::as_str).collect();
    spellings.sort_by_key(|spelling| std::cmp::Reverse(spelling.len()));
    let mut first_bytes = [false; 256];
    for spelling in &spellings {
        first_bytes[usize::from(spelling.as_bytes()[0])] = true;
    }
    let mut current = text.to_owned();
    loop {
        let mut output = String::with_capacity(current.len());
        let mut position = 0;
        let mut changed = false;
        while position < current.len() {
            let remaining = &current[position..];
            let hit = first_bytes[usize::from(remaining.as_bytes()[0])]
                .then(|| spellings.iter().find(|s| remaining.starts_with(**s)))
                .flatten();
            if let Some(hit) = hit {
                let first = hit
                    .chars()
                    .next()
                    .ok_or(Error::UnsupportedMetadata)?
                    .len_utf8();
                output.push_str(&hit[..first]);
                output.push('\u{200b}');
                output.push_str(&hit[first..]);
                position += hit.len();
                changed = true;
            } else {
                let width = remaining
                    .chars()
                    .next()
                    .ok_or(Error::InvalidRequest)?
                    .len_utf8();
                output.push_str(&remaining[..width]);
                position += width;
            }
            if output.len() > MAX_RENDERED_BYTES {
                return Err(Error::BudgetExceeded);
            }
        }
        if !changed {
            return Ok(output);
        }
        current = output;
    }
}

fn write_text(text: &str, controls: &[String], out: &mut String) -> Result<(), Error> {
    let escaped = escape_controls(text, controls)?;
    out.push_str(&serde_json::to_string(&escaped).map_err(|_| Error::InvalidRequest)?);
    if out.len() > MAX_RENDERED_BYTES {
        return Err(Error::BudgetExceeded);
    }
    Ok(())
}

fn write_field(name: &str, controls: &[String], out: &mut String) -> Result<(), Error> {
    // Upstream escapes the entire template input, including its field names.
    write_text(name, controls, out)?;
    out.push(':');
    Ok(())
}

fn write_content(
    content: Option<&DecisionContent>,
    controls: &[String],
    out: &mut String,
) -> Result<(), Error> {
    match content.map(DecisionContent::view) {
        None => out.push_str("null"),
        Some(DecisionContentView::Text(text)) => write_text(text, controls, out)?,
        Some(DecisionContentView::Json(json)) => write_json(json, controls, out)?,
    }
    Ok(())
}

fn write_json(value: &OrderedJson, controls: &[String], out: &mut String) -> Result<(), Error> {
    match value.view() {
        JsonView::String(text) => write_text(text, controls, out)?,
        JsonView::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, controls, out)?;
            }
            out.push(']');
        }
        JsonView::Object(items) => {
            out.push('{');
            let mut keys = HashSet::new();
            for (i, (key, item)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let key = escape_controls(key, controls)?;
                if !keys.insert(key.clone()) {
                    return Err(Error::EscapedKeyCollision);
                }
                out.push_str(&serde_json::to_string(&key).map_err(|_| Error::InvalidRequest)?);
                out.push(':');
                write_json(item, controls, out)?;
            }
            out.push('}');
        }
        _ => out.push_str(&value.to_json_text()),
    }
    if out.len() > MAX_RENDERED_BYTES {
        return Err(Error::BudgetExceeded);
    }
    Ok(())
}

fn render_bound(value: &OrderedJson, depth: usize) -> usize {
    let indent = depth.saturating_mul(4).saturating_add(8);
    match value.view() {
        JsonView::Null | JsonView::Bool(_) => 5,
        JsonView::Number(JsonNumber::F64(_)) => 400,
        JsonView::Number(_) => 21,
        JsonView::String(text) => text.len().saturating_mul(6).saturating_add(2),
        JsonView::Array(items) => items.iter().fold(2, |total: usize, item| {
            total
                .saturating_add(indent)
                .saturating_add(render_bound(item, depth + 1))
        }),
        JsonView::Object(items) => items.iter().fold(2, |total: usize, (key, item)| {
            total
                .saturating_add(indent)
                .saturating_add(key.len().saturating_mul(6))
                .saturating_add(4)
                .saturating_add(render_bound(item, depth + 1))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaping_repeats_for_overlaps_and_preserves_nul() {
        let controls = vec!["<|start|>".to_owned(), "|>".to_owned()];
        assert_eq!(
            escape_controls("\0<|start|>\u{200b}", &controls).unwrap(),
            "\0<\u{200b}|start|\u{200b}>\u{200b}"
        );
        assert_eq!(
            escape_controls("<user-defined>", &controls).unwrap(),
            "<user-defined>"
        );
    }

    #[test]
    fn escaped_object_collisions_are_refused() {
        let value = OrderedJson::parse("{\"<x>\":1,\"<\u{200b}x>\":2}").unwrap();
        assert_eq!(
            write_json(&value, &["<x>".to_owned()], &mut String::new()).unwrap_err(),
            Error::EscapedKeyCollision
        );
    }

    #[test]
    fn bound_counts_indentation_and_float_expansion() {
        let value = OrderedJson::parse("[[[[1e200]]]]").unwrap();
        assert!(render_bound(&value, 0) > 400);
    }

    struct FakeBackend {
        prompt_tokens: usize,
        alias_collision: bool,
        render_calls: usize,
    }

    impl PreparationBackend for FakeBackend {
        fn tokenize(&self, text: &str, parse_special: bool) -> Result<Vec<i32>, Error> {
            if parse_special {
                return Ok(vec![1; self.prompt_tokens]);
            }
            let value = match text {
                "A" => 1,
                "B" => 2,
                " A" if self.alias_collision => 2,
                " A" => 3,
                " B" => 4,
                _ => return Ok(vec![1, 2]),
            };
            Ok(vec![value])
        }
        fn render(&mut self, _: &str, _: usize) -> Result<String, Error> {
            self.render_calls += 1;
            Ok("prompt".into())
        }
    }

    fn fake_profile() -> Profile {
        Profile {
            controls: vec![],
            temperatures: BTreeMap::new(),
        }
    }

    fn request_with_state(state: &str) -> DecisionRequest {
        super::super::wire::decode(&format!(r#"{{"state":{state},"questions":{{"q":{{"type":"choice","instructions":"q","criteria":{{"first":null,"second":null}}}}}}}}"#)).unwrap()
    }

    #[test]
    fn token_limit_is_inclusive_and_no_next_token_slot_is_reserved() {
        for count in [4096, 4097] {
            let mut backend = FakeBackend {
                prompt_tokens: count,
                alias_collision: false,
                render_calls: 0,
            };
            let result = prepare(&request_with_state("null"), &fake_profile(), &mut backend);
            assert_eq!(result.is_ok(), count == 4096);
            assert_eq!(backend.render_calls, 1);
        }
    }

    #[test]
    fn colliding_aliases_fail_before_rendering() {
        let mut backend = FakeBackend {
            prompt_tokens: 1,
            alias_collision: true,
            render_calls: 0,
        };
        assert_eq!(
            prepare(&request_with_state("null"), &fake_profile(), &mut backend).unwrap_err(),
            Error::NoSingleTokenLabel
        );
        assert_eq!(backend.render_calls, 0);
    }

    #[test]
    fn indentation_budget_is_checked_before_native_rendering() {
        // The content is below both neutral aggregate ceilings, but deep pretty
        // printing would allocate far beyond the rendered-prompt ceiling.
        let state = format!(
            "{}{}{}",
            "[".repeat(120),
            vec!["0"; 50000].join(","),
            "]".repeat(120)
        );
        let request = request_with_state(&state);
        let mut backend = FakeBackend {
            prompt_tokens: 1,
            alias_collision: false,
            render_calls: 0,
        };
        assert_eq!(
            prepare(&request, &fake_profile(), &mut backend).unwrap_err(),
            Error::BudgetExceeded
        );
        assert_eq!(backend.render_calls, 0);
    }

    #[test]
    fn unbreakable_special_spellings_are_refused_without_looping() {
        assert_eq!(
            escape_controls("xxx", &["x".into()]).unwrap_err(),
            Error::UnsupportedMetadata
        );
        assert_eq!(
            escape_controls("x\u{200b}y", &["x\u{200b}y".into()]).unwrap_err(),
            Error::UnsupportedMetadata
        );
        assert_eq!(
            escape_controls("x", &vec!["<x>".into(); MAX_CONTROL_SPELLINGS + 1]).unwrap_err(),
            Error::UnsupportedMetadata
        );
        assert_eq!(
            escape_controls("x", &["x".repeat(MAX_CONTROL_BYTES + 1)]).unwrap_err(),
            Error::UnsupportedMetadata
        );
    }
}

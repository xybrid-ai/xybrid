//! Decision-model classification of a GGUF header. The rule is documented
//! in the [module documentation](super#the-classification-rule); the tests
//! at the end of this file are its test vectors.

use std::fmt;

use super::error::GgufError;
use super::header::{GgufHeader, GgufValue};
use super::native::effective_f32;

/// A decision protocol profile this build implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecisionProfileId {
    /// `lfm3-d1/text/v1`: LiquidAI's LFM3 d1 (`lfm3.decision.type =
    /// "lfm3-d1"`) with a text, JSON or absent state, and no media.
    Lfm3D1Text,
}

impl DecisionProfileId {
    /// The profile's versioned name, as provenance records pin it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lfm3D1Text => "lfm3-d1/text/v1",
        }
    }
}

impl fmt::Display for DecisionProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a GGUF is, for the purpose of loading it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecisionClassification {
    /// Not a decision model: the complete header has no
    /// `<arch>.decision.type` key and no other decision key.
    NotDecision,
    /// A decision model of a profile this build implements.
    Supported(DecisionProfileId),
    /// A decision model this build does not run. Loading it as anything
    /// else (a chat model, say) would be wrong, so it is refused.
    Unsupported {
        /// The file's `<arch>.decision.type`.
        decision_type: String,
        /// Why no implemented profile admits it.
        reason: UnsupportedReason,
    },
}

/// Why a decision model is [`DecisionClassification::Unsupported`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedReason {
    /// The decision type names a protocol no profile here implements.
    Protocol,
    /// The decision type names a protocol a later profile will admit, but
    /// no profile in this build does yet: `lfm3-d1-omni` (LiquidAI's
    /// d1-omni), which returns with its own pins alongside media support.
    ProtocolNotYetAdmitted,
    /// The decision type is `lfm3-d1`, but the architecture is not `lfm3`.
    Architecture,
    /// Attention is causal: `lfm3.attention.causal` is `true`, or absent,
    /// which llama.cpp reads as causal.
    CausalAttention,
    /// The `tokenizer.chat_template.systemone` template is missing.
    MissingTemplate,
    /// A decision key the profile does not implement, such as a
    /// `score_bias`, which llama.cpp would apply.
    DecisionKey {
        /// The first such key, in file order.
        key: String,
    },
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol => f.write_str("no supported profile implements this decision type"),
            Self::ProtocolNotYetAdmitted => {
                f.write_str("this decision type is not admitted by any profile in this build yet")
            }
            Self::Architecture => f.write_str("the architecture is not lfm3"),
            Self::CausalAttention => f.write_str("the model's attention is causal"),
            Self::MissingTemplate => f.write_str("the systemone template is missing"),
            Self::DecisionKey { key } => write!(f, "decision key {key:?} is not supported"),
        }
    }
}

const ARCHITECTURE_KEY: &str = "general.architecture";
const LFM3_D1: &str = "lfm3-d1";
/// Decision types a later profile admits: [`UnsupportedReason::ProtocolNotYetAdmitted`].
const NOT_YET_ADMITTED: &[&str] = &["lfm3-d1-omni"];
const LFM3: &str = "lfm3";
const SYSTEMONE_TEMPLATE_KEY: &str = "tokenizer.chat_template.systemone";

/// Classify a GGUF from its complete header.
///
/// See the [module documentation](super#the-classification-rule) for the
/// rule.
///
/// # Errors
///
/// [`GgufError::MissingKey`] or [`GgufError::WrongType`] when
/// `general.architecture` is not a string, [`GgufError::WrongType`] when a
/// decision or profile key has the wrong type (an array-valued
/// `<arch>.decision.type` included: it is malformed, not absent),
/// [`GgufError::Malformed`] for a string that is not UTF-8, and
/// [`GgufError::InvalidDecision`] for a decision type under another
/// architecture's prefix, decision keys without a decision type, or a
/// decision temperature llama.cpp would refuse.
pub fn classify(header: &GgufHeader) -> Result<DecisionClassification, GgufError> {
    let arch = utf8_string(header, ARCHITECTURE_KEY)?.ok_or_else(|| GgufError::MissingKey {
        key: ARCHITECTURE_KEY.to_string(),
    })?;

    for (key, _) in header.metadata() {
        if let Some(prefix) = key.strip_suffix(".decision.type") {
            if prefix != arch {
                return Err(GgufError::InvalidDecision(format!(
                    "{key:?} does not belong to the {arch:?} architecture"
                )));
            }
        }
    }

    let decision_prefix = format!("{arch}.decision.");
    let type_key = format!("{decision_prefix}type");
    let Some(decision_type) = utf8_string(header, &type_key)? else {
        if let Some((key, _)) = header
            .metadata()
            .find(|(key, _)| key.starts_with(&decision_prefix))
        {
            return Err(GgufError::InvalidDecision(format!(
                "{key:?} is present but {type_key:?} is not"
            )));
        }
        return Ok(DecisionClassification::NotDecision);
    };

    let unsupported = |reason| {
        Ok(DecisionClassification::Unsupported {
            decision_type: decision_type.to_string(),
            reason,
        })
    };
    if NOT_YET_ADMITTED.contains(&decision_type) {
        return unsupported(UnsupportedReason::ProtocolNotYetAdmitted);
    }
    if decision_type != LFM3_D1 {
        return unsupported(UnsupportedReason::Protocol);
    }
    if arch != LFM3 {
        return unsupported(UnsupportedReason::Architecture);
    }

    // Types first, so a malformed file is an error even where it is also
    // unsupported.
    let causal_key = format!("{arch}.attention.causal");
    let causal = match header.get(&causal_key) {
        None => None,
        Some(GgufValue::Bool(causal)) => Some(*causal),
        Some(other) => return Err(wrong_type(&causal_key, "bool", other)),
    };
    let has_template = match header.get(SYSTEMONE_TEMPLATE_KEY) {
        None => false,
        Some(GgufValue::String(_)) => true,
        Some(other) => return Err(wrong_type(SYSTEMONE_TEMPLATE_KEY, "str", other)),
    };
    let temperature_prefix = format!("{decision_prefix}temperature.");
    let mut unsupported_key = None;
    for (key, value) in header.metadata() {
        if !key.starts_with(&decision_prefix) || key == type_key {
            continue;
        }
        let Some(name) = key.strip_prefix(&temperature_prefix) else {
            unsupported_key.get_or_insert(key);
            continue;
        };
        let GgufValue::F32(stored) = *value else {
            return Err(wrong_type(key, "f32", value));
        };
        if name.is_empty() {
            return Err(GgufError::InvalidDecision(format!(
                "{key:?} names no temperature"
            )));
        }
        // llama.cpp refuses a temperature that reads as zero or less; one
        // that is not finite would poison every probability.
        let effective = effective_f32(stored);
        if !(effective.is_finite() && effective > 0.0) {
            return Err(GgufError::InvalidDecision(format!(
                "{key:?} reads as {effective}, not a positive temperature"
            )));
        }
    }

    if causal != Some(false) {
        return unsupported(UnsupportedReason::CausalAttention);
    }
    if !has_template {
        return unsupported(UnsupportedReason::MissingTemplate);
    }
    if let Some(key) = unsupported_key {
        return unsupported(UnsupportedReason::DecisionKey {
            key: key.to_string(),
        });
    }
    Ok(DecisionClassification::Supported(
        DecisionProfileId::Lfm3D1Text,
    ))
}

/// A string-typed key as UTF-8 text, `None` when absent.
fn utf8_string<'h>(header: &'h GgufHeader, key: &str) -> Result<Option<&'h str>, GgufError> {
    match header.get(key) {
        None => Ok(None),
        Some(GgufValue::String(bytes)) => std::str::from_utf8(bytes)
            .map(Some)
            .map_err(|_| GgufError::Malformed(format!("{key:?} is not UTF-8"))),
        Some(other) => Err(wrong_type(key, "str", other)),
    }
}

fn wrong_type(key: &str, expected: &'static str, found: &GgufValue) -> GgufError {
    GgufError::WrongType {
        key: key.to_string(),
        expected,
        found: found.type_name(),
    }
}

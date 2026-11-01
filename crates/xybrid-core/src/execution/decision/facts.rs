//! The facts LFM3 d1 reads from a GGUF header: the classification, the
//! pinned template, and the temperature table.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::SYSTEMONE_TEMPLATE_SHA256;
use crate::gguf::{
    classify, effective_f32, native_f32, DecisionClassification, DecisionProfileId, GgufHeader,
    GgufValue, UnsupportedReason,
};
use crate::ir::decision::DecisionError;

/// The key of the decision template (`llama_model_chat_template(model,
/// "systemone")`).
const SYSTEMONE_TEMPLATE_KEY: &str = "tokenizer.chat_template.systemone";

/// The prefix of the temperature keys: `<arch>.decision.temperature.` with
/// the profile's architecture (`server-decision.cpp:58`).
const LFM3_TEMPERATURE_PREFIX: &str = "lfm3.decision.temperature.";

/// The pinned `systemone` template, byte for byte ([`SYSTEMONE_TEMPLATE_SHA256`];
/// LiquidAI's d1-2B-GGUF @ `e968f657`, written by `conversion/lfm3.py:153-157`
/// from the LFM2 d1 prompt plus `<think></think>`). A header passes
/// [`read`] only if its template is these bytes, so the renderer renders
/// this constant.
pub(super) const SYSTEMONE_TEMPLATE: &str = concat!(
    "<|startoftext|><|im_start|>user\n",
    r"{% for image in images %}{{ image }}{% endfor %}{% if state is not none %}",
    r"{% if state is string %}{{ state }}{% else %}{{ state | tojson(indent=2) }}{% endif %}",
    r"{{ '\n\n\nQUESTION:\n' }}{% endif %}",
    r"{{ instructions if instructions is string else instructions | tojson }}",
    r"{% if type == 'choice' %}{{ '\n\nOptions:\n' }}{% for o in options %}{{ o.label }} ",
    r"{% if o.description %}",
    r"{{ o.description if o.description is string else o.description | tojson }}{% else %}",
    r"{{ o.key | replace('_', ' ') }}{% endif %}{% if not loop.last %}{{ '\n' }}{% endif %}",
    r"{% endfor %}{{ '\n\nReply with the option code only.' }}{% elif type == 'noul' %}",
    r"{% set ns = namespace(criteria=false) %}{% for o in options %}",
    r"{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}{% endfor %}",
    r"{% if ns.criteria %}{% for o in options %}",
    r"{{ '\nYes: ' if o.key == 'true' else '\nNo: ' }}{% if o.description is none %}None",
    r"{% else %}{{ o.description if o.description is string else o.description | tojson }}",
    r"{% endif %}{% endfor %}{% endif %}{{ '\n\nReply with yes or no only.' }}{% else %}",
    r"{{ '\n\n' }}{% for o in options %}{{ o.key }} ",
    r"{{ o.description if o.description is string else o.description | tojson }}{{ '\n' }}",
    r"{% endfor %}{{ '\nReply with a single digit 0-' }}{{ options | length - 1 }}",
    r"{{ ' only.' }}{% endif %}{{ '<|im_end|>\n<|im_start|>assistant\n' }}<think></think>",
);

/// One temperature, as llama.cpp's decision server holds it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Temperature {
    /// The metadata string it was read from, as
    /// `llama_model_meta_val_str` returns it (`"1.337398"`); `None` when no
    /// key applies and the temperature is 1.
    pub(super) source: Option<String>,
    /// The value: `strtof` of the string.
    pub(super) value: f32,
}

impl Temperature {
    /// llama.cpp's default when no key applies (`server-decision.cpp:1590`).
    fn default_one() -> Self {
        Self {
            source: None,
            value: 1.0,
        }
    }
}

/// Every `lfm3.decision.temperature.<name>` of a model, by name.
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct Temperatures(BTreeMap<String, Temperature>);

impl Temperatures {
    /// The temperature of a question of `type_name` (`"choice"`, `"noul"` or
    /// `"score"`) with `options` options, levels or answers: the key
    /// `<type>.<bucket>`, else `<type>`, else 1 (`get_temperature`,
    /// `server-decision.cpp:1572-1591`). The `image` key is for requests
    /// with media, which the text profile has none of.
    pub(super) fn lookup(&self, type_name: &str, options: usize) -> Temperature {
        let bucket = match options {
            0..=2 => "2",
            3..=5 => "3_5",
            6..=10 => "6_10",
            _ => "11",
        };
        [format!("{type_name}.{bucket}"), type_name.to_string()]
            .iter()
            .find_map(|name| self.0.get(name).cloned())
            .unwrap_or_else(Temperature::default_one)
    }

    #[cfg(test)]
    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

/// Checks that `header` is a model of `profile` this build renders, and
/// reads its temperatures.
pub(super) fn read(
    profile: DecisionProfileId,
    header: &GgufHeader,
) -> Result<Temperatures, DecisionError> {
    match classify(header) {
        Ok(DecisionClassification::Supported(found)) if found == profile => {}
        Ok(DecisionClassification::Unsupported {
            reason: UnsupportedReason::DecisionKey { .. },
            ..
        }) => return Err(DecisionError::UnsupportedMetadata),
        Ok(_) => return Err(DecisionError::ProfileMismatch),
        Err(_) => return Err(DecisionError::InvalidMetadata),
    }
    match profile {
        DecisionProfileId::Lfm3D1Text => {}
    }

    // Classification proved the template is a string.
    let template = header
        .get(SYSTEMONE_TEMPLATE_KEY)
        .and_then(GgufValue::as_bytes)
        .ok_or(DecisionError::ProfileMismatch)?;
    if sha256_hex(template) != SYSTEMONE_TEMPLATE_SHA256 {
        return Err(DecisionError::UnsupportedTemplate);
    }

    let mut temperatures = BTreeMap::new();
    for (key, value) in header.metadata() {
        let Some(name) = key.strip_prefix(LFM3_TEMPERATURE_PREFIX) else {
            continue;
        };
        // Classification proved every temperature an f32 that reads back
        // finite and positive.
        let GgufValue::F32(stored) = *value else {
            return Err(DecisionError::InvalidMetadata);
        };
        temperatures.insert(
            name.to_string(),
            Temperature {
                source: Some(native_f32(stored)),
                value: effective_f32(stored),
            },
        );
    }
    Ok(Temperatures(temperatures))
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

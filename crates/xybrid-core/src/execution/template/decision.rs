//! Versioned, bounded metadata for local decision inference.

use serde::{Deserialize, Serialize};

/// The audited Liquid d1-2B text/JSON protocol.
pub const LIQUID_TEXT_V1: &str = "lfm3-d1/text/v1";

/// A decision artifact and its whole-prompt CPU execution configuration.
///
/// The first profile requires these exact settings. New model families and
/// accelerated configurations require separate qualification, not larger defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DecisionTemplate {
    /// Versioned protocol, currently `lfm3-d1/text/v1`.
    pub protocol: String,
    /// GGUF file relative to the model directory.
    pub model_file: String,
    /// Maximum tokens in a whole prompt, inclusive.
    #[serde(default = "tokens")]
    pub context_length: u32,
    /// Logical batch capacity. Must equal the context capacity.
    #[serde(default = "tokens")]
    pub batch_size: u32,
    /// Physical batch capacity. The prompt must never be chunked.
    #[serde(default = "tokens")]
    pub ubatch_size: u32,
    /// CPU threads for both inference thread settings.
    #[serde(default = "threads")]
    pub n_threads: i32,
}

const fn tokens() -> u32 {
    4096
}
const fn threads() -> i32 {
    4
}

impl DecisionTemplate {
    /// Selects the audited text profile and its reference CPU configuration.
    pub fn new(model_file: impl Into<String>) -> Self {
        Self {
            protocol: LIQUID_TEXT_V1.into(),
            model_file: model_file.into(),
            context_length: tokens(),
            batch_size: tokens(),
            ubatch_size: tokens(),
            n_threads: threads(),
        }
    }

    /// Refuses unknown profiles and configurations before opening an artifact.
    pub fn validate(&self) -> Result<(), crate::runtime_adapter::AdapterError> {
        if self.protocol != LIQUID_TEXT_V1
            || self.model_file.is_empty()
            || self.context_length != tokens()
            || self.batch_size != tokens()
            || self.ubatch_size != tokens()
            || self.n_threads != threads()
        {
            return Err(crate::runtime_adapter::AdapterError::InvalidInput(
                "unsupported decision profile or execution configuration".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profile_defaults_are_bounded_and_fail_closed() {
        let spec: DecisionTemplate =
            serde_json::from_str(r#"{"protocol":"lfm3-d1/text/v1","model_file":"model.gguf"}"#)
                .unwrap();
        assert_eq!(spec, DecisionTemplate::new("model.gguf"));
        spec.validate().unwrap();
        let mut bad = spec.clone();
        bad.ubatch_size = 512;
        assert!(bad.validate().is_err());
        bad = spec;
        bad.protocol = "lfm3-d1/text/v2".into();
        assert!(bad.validate().is_err());
        assert!(serde_json::from_str::<DecisionTemplate>(
            r#"{"protocol":"lfm3-d1/text/v1","model_file":"model.gguf","gpu":true}"#
        )
        .is_err());
    }
}

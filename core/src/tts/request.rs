//! TTS request types.

use serde::{Deserialize, Serialize};

/// Predefined voice options for TTS synthesis.
///
/// KittenTTS supports 8 voices (4 male, 4 female). 
/// We'll have to adpat this as we integrate more models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// Female voice 1 (default)
    Female1,
    /// Female voice 2
    Female2,
    /// Female voice 3
    Female3,
    /// Female voice 4
    Female4,
    /// Male voice 1
    Male1,
    /// Male voice 2
    Male2,
    /// Male voice 3
    Male3,
    /// Male voice 4
    Male4,
}

impl Voice {
    /// Get the voice index (0-7) for loading from voices.bin
    pub fn index(&self) -> usize {
        match self {
            Voice::Female1 => 0,
            Voice::Female2 => 1,
            Voice::Female3 => 2,
            Voice::Female4 => 3,
            Voice::Male1 => 4,
            Voice::Male2 => 5,
            Voice::Male3 => 6,
            Voice::Male4 => 7,
        }
    }

    /// Create voice from index (0-7)
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Voice::Female1),
            1 => Some(Voice::Female2),
            2 => Some(Voice::Female3),
            3 => Some(Voice::Female4),
            4 => Some(Voice::Male1),
            5 => Some(Voice::Male2),
            6 => Some(Voice::Male3),
            7 => Some(Voice::Male4),
            _ => None,
        }
    }

    /// Get all available voices
    pub fn all() -> &'static [Voice] {
        &[
            Voice::Female1,
            Voice::Female2,
            Voice::Female3,
            Voice::Female4,
            Voice::Male1,
            Voice::Male2,
            Voice::Male3,
            Voice::Male4,
        ]
    }
}

impl Default for Voice {
    fn default() -> Self {
        Voice::Female1
    }
}

impl std::fmt::Display for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Voice::Female1 => write!(f, "female1"),
            Voice::Female2 => write!(f, "female2"),
            Voice::Female3 => write!(f, "female3"),
            Voice::Female4 => write!(f, "female4"),
            Voice::Male1 => write!(f, "male1"),
            Voice::Male2 => write!(f, "male2"),
            Voice::Male3 => write!(f, "male3"),
            Voice::Male4 => write!(f, "male4"),
        }
    }
}

impl std::str::FromStr for Voice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "female1" | "f1" | "0" => Ok(Voice::Female1),
            "female2" | "f2" | "1" => Ok(Voice::Female2),
            "female3" | "f3" | "2" => Ok(Voice::Female3),
            "female4" | "f4" | "3" => Ok(Voice::Female4),
            "male1" | "m1" | "4" => Ok(Voice::Male1),
            "male2" | "m2" | "5" => Ok(Voice::Male2),
            "male3" | "m3" | "6" => Ok(Voice::Male3),
            "male4" | "m4" | "7" => Ok(Voice::Male4),
            _ => Err(format!("Unknown voice: {}", s)),
        }
    }
}

/// Request for TTS synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisRequest {
    /// Text to synthesize
    pub text: String,

    /// Voice to use (default: Female1)
    pub voice: Voice,

    /// Speech speed multiplier (0.5 - 2.0, default: 1.0)
    pub speed: f32,

    /// Whether to apply audio postprocessing (default: true)
    /// Includes: high-pass filter, silence trim, loudness normalization
    pub postprocess: bool,
}

impl SynthesisRequest {
    /// Create a new synthesis request with default options.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: Voice::default(),
            speed: 1.0,
            postprocess: true,
        }
    }

    /// Set the voice.
    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = voice;
        self
    }

    /// Set the speech speed (0.5 - 2.0).
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.clamp(0.5, 2.0);
        self
    }

    /// Enable or disable postprocessing.
    pub fn with_postprocess(mut self, postprocess: bool) -> Self {
        self.postprocess = postprocess;
        self
    }

    /// Validate the request.
    pub fn validate(&self) -> Result<(), super::TtsError> {
        if self.text.is_empty() {
            return Err(super::TtsError::EmptyText);
        }

        // KittenTTS has a practical limit around 500 characters
        const MAX_TEXT_LENGTH: usize = 500;
        if self.text.len() > MAX_TEXT_LENGTH {
            return Err(super::TtsError::TextTooLong {
                length: self.text.len(),
                max: MAX_TEXT_LENGTH,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_index() {
        assert_eq!(Voice::Female1.index(), 0);
        assert_eq!(Voice::Male4.index(), 7);
    }

    #[test]
    fn test_voice_from_index() {
        assert_eq!(Voice::from_index(0), Some(Voice::Female1));
        assert_eq!(Voice::from_index(7), Some(Voice::Male4));
        assert_eq!(Voice::from_index(8), None);
    }

    #[test]
    fn test_voice_parse() {
        assert_eq!("female1".parse::<Voice>().unwrap(), Voice::Female1);
        assert_eq!("m2".parse::<Voice>().unwrap(), Voice::Male2);
        assert_eq!("5".parse::<Voice>().unwrap(), Voice::Male2);
    }

    #[test]
    fn test_request_builder() {
        let request = SynthesisRequest::new("Hello")
            .with_voice(Voice::Male1)
            .with_speed(1.2);

        assert_eq!(request.text, "Hello");
        assert_eq!(request.voice, Voice::Male1);
        assert_eq!(request.speed, 1.2);
    }

    #[test]
    fn test_request_validation() {
        // Empty text should fail
        let request = SynthesisRequest::new("");
        assert!(request.validate().is_err());

        // Normal text should pass
        let request = SynthesisRequest::new("Hello world");
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_speed_clamping() {
        let request = SynthesisRequest::new("Test").with_speed(3.0);
        assert_eq!(request.speed, 2.0);

        let request = SynthesisRequest::new("Test").with_speed(0.1);
        assert_eq!(request.speed, 0.5);
    }
}

//! Stage resolver module - Parses stage names to extract model ID and version.
//!
//! This module provides utilities for parsing stage names like "whisper-tiny@1.2"
//! into model ID and version components for registry lookup.

/// Parses a stage name to extract model ID and optional version.
///
/// # Examples
///
/// ```
/// use xybrid_core::stage_resolver::parse_stage_name;
///
/// let (id, version) = parse_stage_name("whisper-tiny@1.2");
/// assert_eq!(id, "whisper-tiny");
/// assert_eq!(version, Some("1.2".to_string()));
///
/// let (id, version) = parse_stage_name("whisper-tiny");
/// assert_eq!(id, "whisper-tiny");
/// assert_eq!(version, None);
/// ```
pub fn parse_stage_name(stage_name: &str) -> (String, Option<String>) {
    if let Some(at_pos) = stage_name.rfind('@') {
        // Split at the last '@' to handle cases like "model@1.2.3"
        let (id, version) = stage_name.split_at(at_pos);
        let version_str = &version[1..]; // Skip the '@'

        // Validate that version is not empty
        if !version_str.is_empty() {
            return (id.to_string(), Some(version_str.to_string()));
        }
    }

    // No version found or empty version, return entire string as ID
    (stage_name.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stage_name_with_version() {
        let (id, version) = parse_stage_name("whisper-tiny@1.2");
        assert_eq!(id, "whisper-tiny");
        assert_eq!(version, Some("1.2".to_string()));
    }

    #[test]
    fn test_parse_stage_name_without_version() {
        let (id, version) = parse_stage_name("whisper-tiny");
        assert_eq!(id, "whisper-tiny");
        assert_eq!(version, None);
    }

    #[test]
    fn test_parse_stage_name_with_semver() {
        let (id, version) = parse_stage_name("model@1.2.3");
        assert_eq!(id, "model");
        assert_eq!(version, Some("1.2.3".to_string()));
    }

    #[test]
    fn test_parse_stage_name_with_multiple_at_signs() {
        // Should split at the last '@'
        let (id, version) = parse_stage_name("model@1.2@3.4");
        assert_eq!(id, "model@1.2");
        assert_eq!(version, Some("3.4".to_string()));
    }

    #[test]
    fn test_parse_stage_name_empty_version() {
        let (id, version) = parse_stage_name("model@");
        assert_eq!(id, "model@");
        assert_eq!(version, None);
    }

    #[test]
    fn test_parse_stage_name_empty_string() {
        let (id, version) = parse_stage_name("");
        assert_eq!(id, "");
        assert_eq!(version, None);
    }
}

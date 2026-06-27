//! Resolves a model file path relative to a base path.
//!
//! Shared by the executor and the pre/post-processing pipelines so the
//! empty-base-path fallback behaves identically across all three call sites.

use std::path::Path;

/// Resolves `file` against `base_path`.
///
/// Returns `file` unchanged when `base_path` is empty; otherwise joins them
/// and returns the lossy UTF-8 form of the resulting path.
pub(crate) fn resolve_file_path(base_path: &str, file: &str) -> String {
    if base_path.is_empty() {
        file.to_string()
    } else {
        Path::new(base_path)
            .join(file)
            .to_string_lossy()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_base_returns_file_unchanged() {
        assert_eq!(resolve_file_path("", "encoder.onnx"), "encoder.onnx");
    }

    #[test]
    fn non_empty_base_is_joined() {
        let resolved = resolve_file_path("/models", "encoder.onnx");
        assert!(resolved.contains("encoder.onnx"));
        assert!(resolved.contains("models"));
    }
}

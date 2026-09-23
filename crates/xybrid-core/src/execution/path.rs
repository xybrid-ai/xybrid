//! Resolves a model file path relative to a base path, and fingerprints files
//! for the executor's load caches.
//!
//! Shared by the executor and the pre/post-processing pipelines so the
//! empty-base-path fallback behaves identically across all three call sites.

use std::path::Path;
use std::time::SystemTime;

/// `(len, modified)` of a file, used to spot a file replaced in place.
pub(crate) type FileIdentity = (u64, SystemTime);

/// Reads a file's [`FileIdentity`].
///
/// `None` when the metadata cannot be read; caches treat that as
/// "unverifiable" and rebuild rather than trust a possibly-stale entry.
pub(crate) fn file_identity(path: &Path) -> Option<FileIdentity> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.len(), meta.modified().ok()?))
}

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

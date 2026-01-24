//! Model fixtures for integration testing.
//!
//! Provides utilities for locating test models in a consistent way across
//! different environments (local development, CI, standalone repo builds).
//!
//! ## Resolution Order
//!
//! Model paths are resolved in the following order:
//! 1. `$XYBRID_TEST_MODELS/<model>` - Environment variable (highest priority)
//! 2. `$CARGO_MANIFEST_DIR/../integration-tests/fixtures/models/<model>` - Relative to crate
//! 3. `$CARGO_MANIFEST_DIR/../../integration-tests/fixtures/models/<model>` - Workspace layout
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::testing::model_fixtures;
//!
//! // Get model path (panics if not found)
//! let model_dir = model_fixtures::require_model("kokoro-82m");
//!
//! // Get model path (returns Option)
//! if let Some(model_dir) = model_fixtures::model_path("kokoro-82m") {
//!     // Use model
//! }
//!
//! // Check if model is available
//! if model_fixtures::model_available("kokoro-82m") {
//!     // Run model-dependent test
//! }
//! ```
//!
//! ## Environment Variables
//!
//! - `XYBRID_TEST_MODELS`: Override the default model search path

use std::path::PathBuf;
use std::sync::OnceLock;

/// Environment variable name for custom test models path.
pub const ENV_TEST_MODELS: &str = "XYBRID_TEST_MODELS";

/// Cached models directory path.
static MODELS_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Find the integration-tests fixtures/models directory.
///
/// Searches in order:
/// 1. `$XYBRID_TEST_MODELS` environment variable
/// 2. Relative to CARGO_MANIFEST_DIR (for core crate)
/// 3. Common workspace layouts
fn find_models_dir() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(env_path) = std::env::var(ENV_TEST_MODELS) {
        let path = PathBuf::from(env_path);
        if path.exists() {
            return Some(path);
        }
    }

    // Try relative paths from CARGO_MANIFEST_DIR
    // This works when running from xybrid-core crate
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let manifest_path = PathBuf::from(&manifest_dir);

    // Path is always relative to CARGO_MANIFEST_DIR (core/)
    // core/ -> ../integration-tests/fixtures/models
    let fixtures_path = manifest_path.join("../integration-tests/fixtures/models");

    if fixtures_path.exists() && fixtures_path.is_dir() {
        // Canonicalize to get absolute path
        if let Ok(canonical) = fixtures_path.canonicalize() {
            return Some(canonical);
        }
        return Some(fixtures_path);
    }

    None
}

/// Get the models directory path.
///
/// Returns the cached models directory, finding it on first call.
pub fn models_dir() -> Option<&'static PathBuf> {
    MODELS_DIR.get_or_init(find_models_dir).as_ref()
}

/// Get the path to a specific model directory.
///
/// Returns `None` if the models directory is not found or the model doesn't exist.
///
/// # Example
///
/// ```rust,ignore
/// if let Some(path) = model_fixtures::model_path("kokoro-82m") {
///     let metadata = path.join("model_metadata.json");
/// }
/// ```
pub fn model_path(model_name: &str) -> Option<PathBuf> {
    let models = models_dir()?;
    let path = models.join(model_name);
    if path.exists() && path.is_dir() {
        Some(path)
    } else {
        None
    }
}

/// Check if a model is available (directory exists with model_metadata.json).
///
/// # Example
///
/// ```rust,ignore
/// if model_fixtures::model_available("kokoro-82m") {
///     // Run integration test
/// } else {
///     eprintln!("Skipping: kokoro-82m not downloaded");
/// }
/// ```
pub fn model_available(model_name: &str) -> bool {
    model_path(model_name)
        .map(|p| p.join("model_metadata.json").exists() || p.join("model.onnx").exists())
        .unwrap_or(false)
}

/// Get the path to a model, panicking with a helpful message if not found.
///
/// Use this in examples and tests where the model is required.
///
/// # Panics
///
/// Panics if the model directory doesn't exist or models directory is not found.
///
/// # Example
///
/// ```rust,ignore
/// let model_dir = model_fixtures::require_model("kokoro-82m");
/// let metadata_path = model_dir.join("model_metadata.json");
/// ```
pub fn require_model(model_name: &str) -> PathBuf {
    if let Some(path) = model_path(model_name) {
        if path.join("model_metadata.json").exists() || path.join("model.onnx").exists() {
            return path;
        }
    }

    let models_dir_info = models_dir()
        .map(|p| format!("Models directory: {}", p.display()))
        .unwrap_or_else(|| "Models directory: NOT FOUND".to_string());

    panic!(
        r#"
Model '{}' not found!

{}

To download test models, run:
  ./integration-tests/download.sh {}

Or download all models:
  ./integration-tests/download.sh --all

You can also set XYBRID_TEST_MODELS environment variable to a custom path.
"#,
        model_name, models_dir_info, model_name
    );
}

/// Get model path or skip the test with a message.
///
/// Returns `None` and prints a skip message if the model is not available.
/// Useful for tests that should be skipped rather than fail when models are missing.
///
/// # Example
///
/// ```rust,ignore
/// #[test]
/// fn test_tts_inference() {
///     let Some(model_dir) = model_fixtures::model_or_skip("kokoro-82m") else {
///         return; // Test skipped
///     };
///     // ... run test with model_dir
/// }
/// ```
pub fn model_or_skip(model_name: &str) -> Option<PathBuf> {
    if let Some(path) = model_path(model_name) {
        if path.join("model_metadata.json").exists() || path.join("model.onnx").exists() {
            return Some(path);
        }
    }

    eprintln!(
        "Skipping test: model '{}' not downloaded. Run: ./integration-tests/download.sh {}",
        model_name, model_name
    );
    None
}

/// Get the integration-tests fixtures directory (parent of models directory).
///
/// Returns the fixtures directory which contains `models/` and `input/` subdirectories.
///
/// # Example
///
/// ```rust,ignore
/// if let Some(fixtures) = model_fixtures::fixtures_dir() {
///     let test_audio = fixtures.join("input/test_audio.wav");
/// }
/// ```
pub fn fixtures_dir() -> Option<PathBuf> {
    models_dir().and_then(|m| m.parent().map(|p| p.to_path_buf()))
}

/// List all available models in the models directory.
pub fn list_available_models() -> Vec<String> {
    let Some(models) = models_dir() else {
        return vec![];
    };

    let Ok(entries) = std::fs::read_dir(models) else {
        return vec![];
    };

    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter(|e| {
            let path = e.path();
            path.join("model_metadata.json").exists() || path.join("model.onnx").exists()
        })
        .filter_map(|e| e.file_name().into_string().ok())
        .collect()
}

// ============================================================================
// Input Fixtures (test audio, text files)
// ============================================================================

/// Get the input fixtures directory.
///
/// Returns the directory containing test input files (audio, text).
///
/// # Example
///
/// ```rust,ignore
/// if let Some(input_dir) = model_fixtures::input_dir() {
///     let test_audio = input_dir.join("test_audio.wav");
/// }
/// ```
pub fn input_dir() -> Option<PathBuf> {
    fixtures_dir().map(|f| f.join("input"))
}

/// Get path to a test audio file.
///
/// # Example
///
/// ```rust,ignore
/// if let Some(audio_path) = model_fixtures::test_audio("test_audio.wav") {
///     let audio_bytes = std::fs::read(&audio_path)?;
/// }
/// ```
pub fn test_audio(filename: &str) -> Option<PathBuf> {
    input_dir().map(|d| d.join(filename)).filter(|p| p.exists())
}

/// Get path to the default test audio file (test_audio.wav).
///
/// # Example
///
/// ```rust,ignore
/// let audio_path = model_fixtures::default_test_audio()
///     .expect("test_audio.wav should exist");
/// ```
pub fn default_test_audio() -> Option<PathBuf> {
    test_audio("test_audio.wav")
}

/// Get path to a test text file.
///
/// # Example
///
/// ```rust,ignore
/// if let Some(text_path) = model_fixtures::test_text("sample.txt") {
///     let text = std::fs::read_to_string(&text_path)?;
/// }
/// ```
pub fn test_text(filename: &str) -> Option<PathBuf> {
    input_dir().map(|d| d.join(filename)).filter(|p| p.exists())
}

/// Get the pipeline configurations directory.
///
/// Returns the directory containing pipeline YAML files.
pub fn pipelines_dir() -> Option<PathBuf> {
    fixtures_dir().map(|f| f.join("pipelines"))
}

/// Get path to a pipeline configuration file.
///
/// # Example
///
/// ```rust,ignore
/// if let Some(pipeline_path) = model_fixtures::pipeline("tts_pipeline.yaml") {
///     let yaml = std::fs::read_to_string(&pipeline_path)?;
/// }
/// ```
pub fn pipeline(filename: &str) -> Option<PathBuf> {
    pipelines_dir().map(|d| d.join(filename)).filter(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_dir_found() {
        // This test may fail if run outside the xybrid workspace
        // That's expected - it validates the resolution logic works in-workspace
        if let Some(dir) = models_dir() {
            assert!(dir.exists(), "Models directory should exist: {:?}", dir);
            assert!(dir.is_dir(), "Models path should be a directory");
        }
    }

    #[test]
    fn test_model_available_returns_false_for_nonexistent() {
        assert!(!model_available("nonexistent-model-xyz"));
    }

    #[test]
    fn test_model_path_returns_none_for_nonexistent() {
        assert!(model_path("nonexistent-model-xyz").is_none());
    }

    #[test]
    fn test_list_available_models() {
        let models = list_available_models();
        // Just verify it doesn't panic and returns a vec
        // The actual count depends on which models are downloaded
        drop(models);
    }

    #[test]
    fn test_model_or_skip_returns_none_for_missing() {
        let result = model_or_skip("definitely-not-a-real-model");
        assert!(result.is_none());
    }
}

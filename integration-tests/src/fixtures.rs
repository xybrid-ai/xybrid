use std::path::PathBuf;

/// Root fixtures directory
///
/// Resolves `CARGO_MANIFEST_DIR` at runtime rather than with `env!`: the
/// compile-time value is baked into the binary and does not point anywhere
/// useful when a test runs in a sandbox. Both cargo and Bazel set the variable
/// for test execution.
pub fn fixtures_dir() -> PathBuf {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
    PathBuf::from(manifest_dir).join("fixtures")
}

/// Directory containing test input files (audio, text samples)
pub fn input_dir() -> PathBuf {
    fixtures_dir().join("input")
}

/// Directory containing downloaded models
pub fn models_dir() -> PathBuf {
    fixtures_dir().join("models")
}

/// Directory containing pipeline YAML configurations
pub fn pipelines_dir() -> PathBuf {
    fixtures_dir().join("pipelines")
}

/// Get path to a specific model directory
pub fn model_path(model_name: &str) -> PathBuf {
    models_dir().join(model_name)
}

/// Check if a model is available (downloaded)
pub fn model_available(model_name: &str) -> bool {
    let model_dir = model_path(model_name);
    model_dir.exists() && model_dir.join("model_metadata.json").exists()
}

/// Get model path or panic with helpful message
///
/// Use this in tests that require models:
/// ```rust,ignore
/// let model_dir = require_model("kitten-tts-nano-0.2");
/// ```
pub fn require_model(model_name: &str) -> PathBuf {
    let path = model_path(model_name);
    if !model_available(model_name) {
        panic!(
            "Model '{}' not found. Run: ./integration-tests/download.sh {}",
            model_name, model_name
        );
    }
    path
}

/// Skip test if model is not available (returns None)
///
/// Use with early return in tests:
/// ```rust,ignore
/// let Some(model_dir) = model_if_available("kitten-tts-nano-0.2") else {
///     eprintln!("Skipping: kitten-tts-nano-0.2 not downloaded");
///     return;
/// };
/// ```
pub fn model_if_available(model_name: &str) -> Option<PathBuf> {
    if model_available(model_name) {
        Some(model_path(model_name))
    } else {
        None
    }
}

/// Name of the environment variable that turns a missing model into a failure.
pub const REQUIRE_MODELS_ENV: &str = "XYBRID_REQUIRE_MODELS";

/// Get a model path, skipping locally but failing where models are mandatory.
///
/// [`model_if_available`] alone makes every model-gated test a no-op on a
/// machine without the weights — convenient locally, useless on CI, where a
/// download step that quietly failed leaves the whole suite green while
/// testing nothing. Setting `XYBRID_REQUIRE_MODELS` (to any value) flips the
/// missing-model case from "return `None` and skip" to [`require_model`]'s
/// panic, so CI has to actually have the model.
///
/// Use with early return in tests:
/// ```rust,ignore
/// let Some(model_dir) = model_for_test("whisper-tiny") else {
///     eprintln!("Skipping: whisper-tiny not downloaded");
///     return;
/// };
/// ```
pub fn model_for_test(model_name: &str) -> Option<PathBuf> {
    if std::env::var_os(REQUIRE_MODELS_ENV).is_some() {
        return Some(require_model(model_name));
    }
    model_if_available(model_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixtures_dir_exists() {
        assert!(fixtures_dir().exists());
    }

    #[test]
    fn test_models_json_exists() {
        assert!(models_dir().join("models.json").exists());
    }

    #[test]
    fn test_input_dir_exists() {
        assert!(input_dir().exists());
    }

    #[test]
    fn test_pipelines_dir_exists() {
        assert!(pipelines_dir().exists());
    }
}

use std::path::PathBuf;

/// Root fixtures directory
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
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
/// let model_dir = require_model("kitten-tts");
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
/// let Some(model_dir) = model_if_available("kitten-tts") else {
///     eprintln!("Skipping: kitten-tts not downloaded");
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

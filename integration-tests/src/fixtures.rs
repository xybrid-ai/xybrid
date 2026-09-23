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
    fn test_mlx_manifest_entries_are_staged_until_real_fixtures_ship() {
        let manifest_path = models_dir().join("models.json");
        let raw = std::fs::read_to_string(&manifest_path).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let models = manifest
            .get("models")
            .and_then(|m| m.as_object())
            .expect("models.json must contain a models object");

        for id in [
            "qwen3-4b-mlx",
            "gemma4-2b",
            "lfm2-350m-bf16",
            "lfm2.5-1.2b-instruct-mlx",
            "nomic-embed-text-v1.5",
        ] {
            let entry = models.get(id).unwrap_or_else(|| panic!("{id} missing"));
            assert_eq!(
                entry.get("source").and_then(|v| v.as_str()),
                Some("staged"),
                "{id} must not be listed as a downloadable registry fixture before real MLX fixture staging ships"
            );
            assert_eq!(
                entry.get("feature_gate").and_then(|v| v.as_str()),
                Some("llm-mlx-runtime"),
                "{id} requires the runtime feature, not the skeleton-only llm-mlx gate"
            );
            let expected_status = match id {
                "qwen3-4b-mlx" => "runtime",
                "gemma4-2b"
                | "lfm2-350m-bf16"
                | "lfm2.5-1.2b-instruct-mlx"
                | "nomic-embed-text-v1.5" => "partial-runtime",
                _ => unreachable!("unexpected MLX fixture id"),
            };
            let expected_env_var = match id {
                "qwen3-4b-mlx" => "XYBRID_MLX_QWEN_4B_DIR",
                "gemma4-2b" => "XYBRID_MLX_GEMMA_DIR",
                "lfm2-350m-bf16" => "XYBRID_MLX_LFM_DIR",
                "lfm2.5-1.2b-instruct-mlx" => "XYBRID_MLX_LFM25_DIR",
                "nomic-embed-text-v1.5" => "XYBRID_MLX_NOMIC_DIR",
                _ => unreachable!("unexpected MLX fixture id"),
            };
            assert_eq!(
                entry.get("status").and_then(|v| v.as_str()),
                Some(expected_status),
                "{id} status must reflect the current MLX runtime coverage"
            );
            assert_eq!(
                entry.get("test_env_var").and_then(|v| v.as_str()),
                Some(expected_env_var),
                "{id} must declare the env var used by the real MLX runtime tests"
            );
            let required_files = entry
                .get("required_files")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("{id} must list required staged fixture files"));
            for file in ["config.json", "tokenizer.json"] {
                assert!(
                    required_files
                        .iter()
                        .any(|required| required.as_str() == Some(file)),
                    "{id} must require {file}"
                );
            }
            let has_single_weights = required_files
                .iter()
                .any(|required| required.as_str() == Some("model.safetensors"));
            let has_indexed_weights = required_files
                .iter()
                .any(|required| required.as_str() == Some("model.safetensors.index.json"));
            assert!(
                has_single_weights || has_indexed_weights,
                "{id} must require either model.safetensors or model.safetensors.index.json"
            );
            let platforms = entry
                .get("platforms")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("{id} must list supported runtime platforms"));
            assert!(
                platforms
                    .iter()
                    .any(|platform| platform.as_str() == Some("macos")),
                "{id} must remain listed for the macOS MLX runtime"
            );
            assert!(
                !platforms
                    .iter()
                    .any(|platform| platform.as_str() == Some("ios")),
                "{id} must not advertise iOS MLX runtime readiness while iOS slices are skeleton-only"
            );
            assert!(
                entry
                    .get("notes")
                    .and_then(|v| v.as_str())
                    .is_some_and(|notes| notes.contains("Not downloaded by --all")
                        && notes.contains("iOS remains skeleton-only")),
                "{id} notes must make staged download and iOS skeleton-only behavior explicit"
            );
        }
    }

    #[test]
    fn test_mlx_staged_fixture_env_vars_are_unambiguous() {
        let manifest_path = models_dir().join("models.json");
        let raw = std::fs::read_to_string(&manifest_path).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let models = manifest
            .get("models")
            .and_then(|m| m.as_object())
            .expect("models.json must contain a models object");

        let mut seen = std::collections::HashMap::new();
        for id in [
            "qwen3-4b-mlx",
            "gemma4-2b",
            "lfm2-350m-bf16",
            "lfm2.5-1.2b-instruct-mlx",
            "nomic-embed-text-v1.5",
        ] {
            let env_var = models
                .get(id)
                .and_then(|entry| entry.get("test_env_var"))
                .and_then(|value| value.as_str())
                .unwrap_or_else(|| panic!("{id} must declare test_env_var"));
            if let Some(previous_id) = seen.insert(env_var, id) {
                panic!(
                    "{id} and {previous_id} share {env_var}; staged MLX benchmark fixtures must not resolve through ambiguous env vars"
                );
            }
        }
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

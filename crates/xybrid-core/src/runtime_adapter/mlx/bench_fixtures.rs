//! Fixture resolution shared by the MLX benchmark harness and tests.
//!
//! Staged MLX fixtures are often too large to keep under
//! `integration-tests/fixtures/models/<id>` during local development. The
//! manifest already declares the env var used by runtime smoke tests, so the
//! benchmark resolver accepts that same path when present.

use std::path::{Component, Path, PathBuf};

pub fn fixtures_root() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("XYBRID_BENCH_FIXTURES_ROOT") {
        let p = PathBuf::from(root);
        return p.exists().then_some(p);
    }

    let candidates = [
        PathBuf::from("repos/xybrid/integration-tests/fixtures/models"),
        PathBuf::from("integration-tests/fixtures/models"),
        PathBuf::from("../integration-tests/fixtures/models"),
        PathBuf::from("../../integration-tests/fixtures/models"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

pub fn resolve_mlx_dir(fixture_id: &str) -> Option<PathBuf> {
    let root = fixtures_root()?;
    let entry = manifest_entry(&root, fixture_id);
    let dir = root.join(fixture_id);
    if mlx_dir_is_ready(&dir, entry.as_ref()) {
        Some(dir)
    } else {
        resolve_mlx_dir_from_manifest_env(&root, fixture_id, entry.as_ref())
    }
}

fn manifest_entry(root: &Path, fixture_id: &str) -> Option<serde_json::Value> {
    let raw = std::fs::read_to_string(root.join("models.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&raw).ok()?;
    manifest.get("models")?.get(fixture_id).cloned()
}

fn resolve_mlx_dir_from_manifest_env(
    root: &Path,
    fixture_id: &str,
    entry: Option<&serde_json::Value>,
) -> Option<PathBuf> {
    let owned_entry;
    let entry = match entry {
        Some(entry) => entry,
        None => {
            owned_entry = manifest_entry(root, fixture_id)?;
            &owned_entry
        }
    };
    let env_var = entry.get("test_env_var")?.as_str()?;
    let dir = PathBuf::from(std::env::var_os(env_var)?);
    mlx_dir_is_ready(&dir, Some(entry)).then_some(dir)
}

fn mlx_dir_is_ready(dir: &Path, entry: Option<&serde_json::Value>) -> bool {
    if !dir.is_dir() {
        return false;
    }

    if let Some(entry) = entry {
        let required_files = entry
            .get("required_files")
            .and_then(|files| files.as_array())
            .into_iter()
            .flatten()
            .filter_map(|file| file.as_str());

        let mut saw_required_file = false;
        for file in required_files {
            saw_required_file = true;
            if !dir.join(file).is_file() {
                return false;
            }
        }

        if saw_required_file {
            return indexed_shards_are_present(dir);
        }
    }

    dir.join("config.json").is_file()
        && dir.join("tokenizer.json").is_file()
        && (dir.join("model.safetensors").is_file()
            || (dir.join("model.safetensors.index.json").is_file()
                && indexed_shards_are_present(dir)))
}

fn indexed_shards_are_present(dir: &Path) -> bool {
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return true;
    }

    let raw = match std::fs::read_to_string(&index_path) {
        Ok(raw) => raw,
        Err(_) => return false,
    };
    let index: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(index) => index,
        Err(_) => return false,
    };
    let Some(weight_map) = index.get("weight_map").and_then(|map| map.as_object()) else {
        return false;
    };
    if weight_map.is_empty() {
        return false;
    }

    weight_map.values().all(|file| {
        let Some(file) = file.as_str() else {
            return false;
        };
        let path = Path::new(file);
        if path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return false;
        }
        dir.join(path).is_file()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn with_env_var<R>(key: &str, value: Option<&std::path::Path>, f: impl FnOnce() -> R) -> R {
        let old = std::env::var_os(key);
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        let out = f();
        match old {
            Some(old) => std::env::set_var(key, old),
            None => std::env::remove_var(key),
        }
        out
    }

    fn write_manifest(root: &std::path::Path) {
        std::fs::write(
            root.join("models.json"),
            r#"{
              "models": {
                "gemma4-2b": {
                  "test_env_var": "XYBRID_MLX_GEMMA_DIR",
                  "required_files": [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "model.safetensors.index.json"
                  ]
                }
              }
            }"#,
        )
        .unwrap();
    }

    fn write_index(dir: &std::path::Path, shard: &str) {
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": shard
                }
            })
            .to_string(),
        )
        .unwrap();
    }

    #[test]
    fn resolve_mlx_dir_uses_manifest_test_env_var() {
        let _guard = env_lock();
        let root = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir().unwrap();
        write_manifest(root.path());
        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ] {
            std::fs::write(staged.path().join(file), "{}").unwrap();
        }
        write_index(staged.path(), "model-00001-of-00001.safetensors");
        std::fs::write(
            staged.path().join("model-00001-of-00001.safetensors"),
            b"placeholder",
        )
        .unwrap();

        with_env_var("XYBRID_BENCH_FIXTURES_ROOT", Some(root.path()), || {
            with_env_var("XYBRID_MLX_GEMMA_DIR", Some(staged.path()), || {
                assert_eq!(
                    resolve_mlx_dir("gemma4-2b"),
                    Some(staged.path().to_path_buf())
                );
            })
        });
    }

    #[test]
    fn resolve_mlx_dir_rejects_indexed_bundle_with_missing_shard() {
        let _guard = env_lock();
        let root = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir().unwrap();
        write_manifest(root.path());
        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ] {
            std::fs::write(staged.path().join(file), "{}").unwrap();
        }
        write_index(staged.path(), "model-00001-of-00001.safetensors");

        with_env_var("XYBRID_BENCH_FIXTURES_ROOT", Some(root.path()), || {
            with_env_var("XYBRID_MLX_GEMMA_DIR", Some(staged.path()), || {
                assert_eq!(resolve_mlx_dir("gemma4-2b"), None);
            })
        });
    }

    #[test]
    fn resolve_mlx_dir_rejects_env_bundle_missing_required_chat_file() {
        let _guard = env_lock();
        let root = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir().unwrap();
        write_manifest(root.path());
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            std::fs::write(staged.path().join(file), "{}").unwrap();
        }
        write_index(staged.path(), "model-00001-of-00001.safetensors");
        std::fs::write(
            staged.path().join("model-00001-of-00001.safetensors"),
            b"placeholder",
        )
        .unwrap();

        with_env_var("XYBRID_BENCH_FIXTURES_ROOT", Some(root.path()), || {
            with_env_var("XYBRID_MLX_GEMMA_DIR", Some(staged.path()), || {
                assert_eq!(resolve_mlx_dir("gemma4-2b"), None);
            })
        });
    }
}

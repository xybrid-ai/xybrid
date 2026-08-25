//! Reuse of the standard shared Hugging Face hub cache.
//!
//! The Hugging Face ecosystem (`huggingface-cli download`, the `transformers`
//! and `diffusers` libraries, ...) keeps downloaded models in a shared hub
//! cache at `~/.cache/huggingface/hub` (overridable via `HF_HOME` or
//! `HUGGINGFACE_HUB_CACHE`). When xybrid is asked to load a model the user
//! already downloaded with those tools, re-downloading wastes bandwidth.
//!
//! This module probes that shared cache read-only — zero network I/O — before
//! the network download path in
//! [`crate::model::ModelLoader::load_from_huggingface`] runs. On a hit, the
//! files xybrid needs are materialized (symlinked on unix, copied elsewhere)
//! into xybrid's own materialized layout and the model loads from there.
//!
//! The probe is desktop-only: on iOS/Android [`find_shared_snapshot`] always
//! returns `None`, keeping mobile behavior unchanged.

use std::path::{Path, PathBuf};

/// A snapshot of a model repo found in the shared Hugging Face hub cache.
#[derive(Debug, Clone)]
pub(crate) struct SharedSnapshot {
    /// Resolved commit hash the snapshot corresponds to.
    pub(crate) commit: String,
    /// Snapshot directory: `<hub root>/<repo dir>/snapshots/<commit>`.
    pub(crate) dir: PathBuf,
}

/// Resolve the shared Hugging Face hub cache root, honoring
/// `HUGGINGFACE_HUB_CACHE`, then `HF_HOME`, then `~/.cache/huggingface/hub`.
///
/// Returns `None` when no home directory can be determined; callers then
/// silently skip the probe. Unlike `hf_hub::Cache::from_env()` /
/// `hf_hub::Cache::default()` this never panics on a missing home directory.
#[cfg(not(any(target_os = "ios", target_os = "android")))]
pub(crate) fn shared_cache_root() -> Option<PathBuf> {
    resolve_shared_cache_root(
        std::env::var("HUGGINGFACE_HUB_CACHE").ok(),
        std::env::var("HF_HOME").ok(),
        dirs::home_dir(),
    )
}

/// Pure resolution of the shared cache root from explicit inputs (testable
/// without touching process env vars).
///
/// Order: non-empty `HUGGINGFACE_HUB_CACHE` (the hub root itself), non-empty
/// `HF_HOME` (hub root is `<HF_HOME>/hub`), then `<home>/.cache/huggingface/hub`.
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn resolve_shared_cache_root(
    hub_cache: Option<String>,
    hf_home: Option<String>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(value) = hub_cache {
        if !value.is_empty() {
            return Some(PathBuf::from(value));
        }
    }
    if let Some(value) = hf_home {
        if !value.is_empty() {
            return Some(PathBuf::from(value).join("hub"));
        }
    }
    home.map(|home| home.join(".cache").join("huggingface").join("hub"))
}

/// Find a usable snapshot of `repo` in the shared Hugging Face hub cache
/// without any network I/O.
///
/// Resolution order: (a) the requested revision, read from
/// `<repo dir>/refs/<revision>` — or, when the revision is a full commit SHA,
/// probed directly as `snapshots/<revision>` (tools write refs under branch
/// names, so a SHA may have no ref entry); (b) without a requested revision,
/// the `main` ref, then the `master` ref. The resolved commit's snapshot
/// directory must exist and contain at least one model payload file
/// (`.gguf`/`.onnx`/`.safetensors` or `model_metadata.json`).
///
/// Returns `None` (with a debug log) when the repo is absent from the shared
/// cache, the revision cannot be resolved, or the snapshot holds no model
/// payload — the caller then falls through to the normal download path
/// unchanged.
#[cfg(not(any(target_os = "ios", target_os = "android")))]
pub(crate) fn find_shared_snapshot(repo: &str, revision: Option<&str>) -> Option<SharedSnapshot> {
    let Some(root) = shared_cache_root() else {
        log::debug!(
            target: "xybrid_sdk",
            "No shared Hugging Face cache root; skipping shared-cache probe for '{}'",
            repo
        );
        return None;
    };
    find_shared_snapshot_with_root(&root, repo, revision)
}

/// Mobile: the shared-cache probe is a desktop feature, so never hit it.
#[cfg(any(target_os = "ios", target_os = "android"))]
pub(crate) fn find_shared_snapshot(_repo: &str, _revision: Option<&str>) -> Option<SharedSnapshot> {
    None
}

/// Probe a specific hub root directly (used by tests to avoid process env).
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn find_shared_snapshot_with_root(
    root: &Path,
    repo: &str,
    revision: Option<&str>,
) -> Option<SharedSnapshot> {
    use hf_hub::{Cache, Repo, RepoType};

    let cache = Cache::new(root.to_path_buf());
    let repo_cache = cache.repo(Repo::new(repo.to_string(), RepoType::Model));
    let folder_name = Repo::new(repo.to_string(), RepoType::Model).folder_name();

    let commit = resolve_commit(root, &folder_name, repo, revision)?;
    let snapshot_dir = repo_cache.pointer_path(&commit);
    if !snapshot_dir.is_dir() {
        log::debug!(
            target: "xybrid_sdk",
            "Shared Hugging Face cache ref for '{}' resolves to commit {} but no snapshot exists at {}",
            repo, commit, snapshot_dir.display()
        );
        return None;
    }
    if !snapshot_has_model_payload(&snapshot_dir) {
        log::debug!(
            target: "xybrid_sdk",
            "Shared Hugging Face snapshot for '{}' at {} contains no model files",
            repo, snapshot_dir.display()
        );
        return None;
    }
    Some(SharedSnapshot {
        commit,
        dir: snapshot_dir,
    })
}

/// Resolve a commit hash from the shared cache's refs without network I/O.
///
/// With a requested revision: read `refs/<revision>`; if that ref is absent
/// and the revision is a full 40-hex commit SHA, use it directly (a SHA may
/// not have a ref entry of its own). Without a requested revision: try
/// `main`, then `master`.
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn resolve_commit(
    root: &Path,
    folder_name: &str,
    repo: &str,
    revision: Option<&str>,
) -> Option<String> {
    if let Some(requested) = revision {
        let resolved = read_ref(root, folder_name, requested);
        if resolved.is_none() && is_full_commit_sha(requested) {
            log::debug!(
                target: "xybrid_sdk",
                "No shared Hugging Face ref for '{}'@{}; probing commit SHA directly",
                repo, requested
            );
            return Some(requested.to_string());
        }
        if resolved.is_none() {
            log::debug!(
                target: "xybrid_sdk",
                "No shared Hugging Face ref '{}' for '{}'",
                requested, repo
            );
        }
        return resolved;
    }
    read_ref(root, folder_name, "main").or_else(|| read_ref(root, folder_name, "master"))
}

/// Read and trim `refs/<revision>` inside the repo folder.
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn read_ref(root: &Path, folder_name: &str, revision: &str) -> Option<String> {
    let value = std::fs::read_to_string(root.join(folder_name).join("refs").join(revision)).ok()?;
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn is_full_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether the snapshot contains at least one model payload file
/// (`.gguf`/`.onnx`/`.safetensors` or `model_metadata.json`).
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn snapshot_has_model_payload(snapshot_dir: &Path) -> bool {
    list_snapshot_files(snapshot_dir)
        .iter()
        .any(|filename| is_model_payload_file(filename))
}

fn is_model_payload_file(filename: &str) -> bool {
    let basename = filename.rsplit('/').next().unwrap_or(filename);
    basename == "model_metadata.json"
        || filename.ends_with(".gguf")
        || filename.ends_with(".onnx")
        || filename.ends_with(".safetensors")
}

/// Recursively list the files under `dir` as repo-relative paths (POSIX
/// separators), sorted for determinism.
///
/// Symlink entries are listed even when their target is missing, so callers
/// can distinguish "snapshot has this file" (entry present) from "the file's
/// payload is actually readable" (see the completeness check in
/// [`select_snapshot_files`]).
pub(crate) fn list_snapshot_files(dir: &Path) -> Vec<String> {
    fn walk(dir: &Path, prefix: &str, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            let relative = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            if file_type.is_dir() {
                walk(&entry.path(), &relative, out);
            } else if file_type.is_file() || file_type.is_symlink() {
                out.push(relative);
            }
        }
    }

    let mut files = Vec::new();
    walk(dir, "", &mut files);
    files
}

/// Run the same file-selection pipeline the network download path uses
/// against a shared-cache snapshot, and return the selected files only when
/// every one of them is a readable file inside the snapshot.
///
/// Returns `None` (with a debug log) when the selection fails or any selected
/// file is missing from the snapshot — the caller then falls through to the
/// existing download path unchanged. The selection helpers live in
/// `crate::model` and are reused as-is; nothing is duplicated here.
pub(crate) fn select_snapshot_files(
    snapshot: &SharedSnapshot,
    repo: &str,
    variant: Option<&str>,
) -> Option<Vec<String>> {
    let all_filenames = list_snapshot_files(&snapshot.dir);
    if all_filenames.is_empty() {
        log::debug!(
            target: "xybrid_sdk",
            "Shared Hugging Face snapshot for '{}' is empty",
            repo
        );
        return None;
    }
    let all_filenames: Vec<&str> = all_filenames.iter().map(String::as_str).collect();
    let all_gguf_files: Vec<&str> = all_filenames
        .iter()
        .filter(|filename| filename.ends_with(".gguf"))
        .copied()
        .collect();
    let gguf_files: Vec<&str> = all_gguf_files
        .iter()
        .copied()
        .filter(|filename| !crate::model::is_gguf_companion(filename))
        .collect();

    // Mirrors the download path's guard: companion-only snapshots (e.g. an
    // mmproj without a language model) are not usable.
    if !all_gguf_files.is_empty() && gguf_files.is_empty() {
        log::debug!(
            target: "xybrid_sdk",
            "Shared Hugging Face snapshot for '{}' has GGUF companions but no language model",
            repo
        );
        return None;
    }

    let selected_gguf = if gguf_files.is_empty() {
        None
    } else {
        Some(crate::model::select_gguf_variant(&gguf_files, variant).ok()?)
    };
    let selected_projector = crate::model::select_vision_projector(&all_gguf_files);

    let selected_files = crate::model::select_huggingface_files_to_download(
        repo,
        &all_filenames,
        selected_gguf.as_deref(),
        selected_projector,
    )
    .ok()?;
    if selected_files
        .iter()
        .any(|filename| !snapshot.dir.join(filename).is_file())
    {
        log::debug!(
            target: "xybrid_sdk",
            "Shared Hugging Face snapshot for '{}' is missing a selected file; falling back to download",
            repo
        );
        return None;
    }
    Some(selected_files.into_iter().map(str::to_string).collect())
}

/// Materialize `filenames` from `snapshot` into `target_dir`, mirroring the
/// download path's layout exactly: existing targets are left untouched,
/// missing files are symlinked on unix and copied elsewhere.
pub(crate) fn materialize_from_shared(
    snapshot: &SharedSnapshot,
    filenames: &[&str],
    target_dir: &Path,
) -> std::io::Result<()> {
    for filename in filenames {
        let target_path = target_dir.join(filename);
        if target_path.exists() {
            continue;
        }
        // A dangling symlink from an earlier materialization (its source cache
        // was pruned) reports `!exists()` but still occupies the directory
        // entry, so linking over it would fail; remove it first.
        if target_path.symlink_metadata().is_ok() {
            std::fs::remove_file(&target_path)?;
        }
        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let source_path = snapshot.dir.join(filename);
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&source_path, &target_path)?;
        }
        #[cfg(not(unix))]
        {
            std::fs::copy(&source_path, &target_path)?;
        }
    }
    Ok(())
}

/// Detect (and best-effort remove) dangling symlinks under a materialized
/// cache directory, returning how many were found.
///
/// On unix both materialization paths symlink instead of copying — the
/// download path into xybrid's own hub cache, the shared-cache path into the
/// user's Hugging Face hub cache. Pruning either source cache (e.g.
/// `huggingface-cli delete-cache`) leaves those links dangling while the
/// materialized directory still looks complete to the marker/metadata checks,
/// so a load from it would fail hard. Callers run this before trusting a
/// materialized directory: a non-zero return means the payload is gone and
/// they must fall through to re-materialization instead of loading.
///
/// Removal is best-effort (failures are logged and still counted) so callers
/// always learn the directory is unusable even when cleanup itself fails.
/// Regular files and symlinks that still resolve are never touched.
pub(crate) fn remove_dangling_files(dir: &Path) -> usize {
    fn walk(dir: &Path, dangling: &mut usize) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() {
                walk(&path, dangling);
            } else if file_type.is_symlink() && std::fs::metadata(&path).is_err() {
                *dangling += 1;
                if let Err(e) = std::fs::remove_file(&path) {
                    log::debug!(
                        target: "xybrid_sdk",
                        "Failed to remove dangling symlink {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }
    }

    let mut dangling = 0;
    walk(dir, &mut dangling);
    dangling
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const REPO: &str = "org/model";
    const COMMIT_MAIN: &str = "1111111111111111111111111111111111111111";
    const COMMIT_V1: &str = "2222222222222222222222222222222222222222";

    fn repo_folder(repo: &str) -> String {
        hf_hub::Repo::new(repo.to_string(), hf_hub::RepoType::Model).folder_name()
    }

    fn write_file(root: &Path, rel: &str, contents: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn find_with_root(root: &Path, revision: Option<&str>) -> Option<SharedSnapshot> {
        find_shared_snapshot_with_root(root, REPO, revision)
    }

    fn build_snapshot(root: &Path, commit: &str) {
        let folder = repo_folder(REPO);
        write_file(
            root,
            &format!("{folder}/snapshots/{commit}/model-Q4_K_M.gguf"),
            "payload",
        );
        write_file(
            root,
            &format!("{folder}/snapshots/{commit}/config.json"),
            "{}",
        );
    }

    #[test]
    fn finds_snapshot_via_requested_revision_ref() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(
            tmp.path(),
            &format!("{folder}/refs/v1.0"),
            &format!("{COMMIT_V1}\n"),
        );
        build_snapshot(tmp.path(), COMMIT_V1);

        let snapshot = find_with_root(tmp.path(), Some("v1.0")).unwrap();
        assert_eq!(snapshot.commit, COMMIT_V1);
        assert_eq!(
            snapshot.dir,
            tmp.path().join(&folder).join("snapshots").join(COMMIT_V1)
        );
    }

    #[test]
    fn falls_back_to_main_ref_without_requested_revision() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(tmp.path(), &format!("{folder}/refs/main"), COMMIT_MAIN);
        write_file(tmp.path(), &format!("{folder}/refs/master"), COMMIT_V1);
        build_snapshot(tmp.path(), COMMIT_MAIN);
        build_snapshot(tmp.path(), COMMIT_V1);

        let snapshot = find_with_root(tmp.path(), None).unwrap();
        assert_eq!(snapshot.commit, COMMIT_MAIN, "main must win over master");
    }

    #[test]
    fn falls_back_to_master_ref_when_main_is_absent() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(tmp.path(), &format!("{folder}/refs/master"), COMMIT_V1);
        build_snapshot(tmp.path(), COMMIT_V1);

        let snapshot = find_with_root(tmp.path(), None).unwrap();
        assert_eq!(snapshot.commit, COMMIT_V1);
    }

    #[test]
    fn returns_none_when_repo_absent_from_shared_cache() {
        let tmp = TempDir::new().unwrap();
        assert!(find_with_root(tmp.path(), None).is_none());
        assert!(find_with_root(tmp.path(), Some("main")).is_none());
    }

    #[test]
    fn returns_none_when_ref_is_present_but_snapshot_missing() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        // Ref exists, but no snapshots/<commit> directory was ever created.
        write_file(tmp.path(), &format!("{folder}/refs/main"), COMMIT_MAIN);

        assert!(find_with_root(tmp.path(), None).is_none());
    }

    #[test]
    fn returns_none_when_snapshot_has_no_model_payload() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(tmp.path(), &format!("{folder}/refs/main"), COMMIT_MAIN);
        // Only a README — no model file, no model_metadata.json.
        write_file(
            tmp.path(),
            &format!("{folder}/snapshots/{COMMIT_MAIN}/README.md"),
            "# model\n",
        );

        assert!(find_with_root(tmp.path(), None).is_none());
    }

    #[test]
    fn probes_commit_sha_directly_without_a_ref_entry() {
        let tmp = TempDir::new().unwrap();
        // No refs at all — but the snapshot directory for the full SHA exists
        // (e.g. populated by a tool that writes snapshots without refs).
        build_snapshot(tmp.path(), COMMIT_V1);

        let snapshot = find_with_root(tmp.path(), Some(COMMIT_V1)).unwrap();
        assert_eq!(snapshot.commit, COMMIT_V1);
    }

    #[test]
    fn select_snapshot_files_returns_none_when_selected_file_missing() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(tmp.path(), &format!("{folder}/refs/main"), COMMIT_MAIN);
        // The snapshot holds a different quantization than the one requested:
        // the file the user selected is missing, so the shared path must
        // abandon rather than materialize a partial snapshot.
        write_file(
            tmp.path(),
            &format!("{folder}/snapshots/{COMMIT_MAIN}/model-Q8_0.gguf"),
            "payload",
        );
        write_file(
            tmp.path(),
            &format!("{folder}/snapshots/{COMMIT_MAIN}/config.json"),
            "{}",
        );

        let snapshot = find_with_root(tmp.path(), None).unwrap();
        assert!(
            select_snapshot_files(&snapshot, REPO, Some("Q4_K_M")).is_none(),
            "a snapshot without the requested variant must not be materialized"
        );
    }

    #[cfg(unix)]
    #[test]
    fn select_snapshot_files_treats_dangling_symlink_as_missing() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(tmp.path(), &format!("{folder}/refs/main"), COMMIT_MAIN);
        let snapshot_dir = tmp.path().join(&folder).join("snapshots").join(COMMIT_MAIN);
        // The model file is a dangling symlink (blob evicted): it exists as a
        // snapshot entry but its payload is unreadable.
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::os::unix::fs::symlink(
            tmp.path().join("gone-blob"),
            snapshot_dir.join("model-Q4_K_M.gguf"),
        )
        .unwrap();
        write_file(&snapshot_dir, "config.json", "{}");

        let snapshot = find_with_root(tmp.path(), None).unwrap();
        assert!(select_snapshot_files(&snapshot, REPO, Some("Q4_K_M")).is_none());
    }

    #[test]
    fn select_snapshot_files_returns_complete_selection() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(tmp.path(), &format!("{folder}/refs/main"), COMMIT_MAIN);
        write_file(
            tmp.path(),
            &format!("{folder}/snapshots/{COMMIT_MAIN}/model-Q8_0.gguf"),
            "payload",
        );
        write_file(
            tmp.path(),
            &format!("{folder}/snapshots/{COMMIT_MAIN}/model-Q4_K_M.gguf"),
            "payload",
        );
        write_file(
            tmp.path(),
            &format!("{folder}/snapshots/{COMMIT_MAIN}/config.json"),
            "{}",
        );

        let snapshot = find_with_root(tmp.path(), None).unwrap();
        let files = select_snapshot_files(&snapshot, REPO, Some("Q8_0")).unwrap();
        assert!(files.contains(&"model-Q8_0.gguf".to_string()));
        assert!(!files.contains(&"model-Q4_K_M.gguf".to_string()));
        assert!(files.contains(&"config.json".to_string()));
    }

    #[test]
    fn colliding_repo_ids_resolve_snapshots_without_cross_reads() {
        let tmp = TempDir::new().unwrap();
        let first_repo = "a/b--c";
        let second_repo = "a--b/c";
        // The collision premise: both ids sanitize to the same hub folder.
        assert_eq!(repo_folder(first_repo), repo_folder(second_repo));

        // Two separate hub roots: neither repo may read the other's snapshot.
        let root_a = tmp.path().join("hub-a");
        let root_b = tmp.path().join("hub-b");
        let folder = repo_folder(first_repo);
        write_file(&root_a, &format!("{folder}/refs/main"), COMMIT_MAIN);
        write_file(
            &root_a,
            &format!("{folder}/snapshots/{COMMIT_MAIN}/model.gguf"),
            "payload",
        );
        write_file(&root_b, &format!("{folder}/refs/main"), COMMIT_V1);
        write_file(
            &root_b,
            &format!("{folder}/snapshots/{COMMIT_V1}/model.gguf"),
            "payload",
        );

        let first = find_shared_snapshot_with_root(&root_a, first_repo, Some("main")).unwrap();
        assert_eq!(first.commit, COMMIT_MAIN);
        let second = find_shared_snapshot_with_root(&root_b, second_repo, Some("main")).unwrap();
        assert_eq!(second.commit, COMMIT_V1);

        // Within a single shared root both ids resolve the same hub folder —
        // that is the shared cache's own layout, not ours to fix; the folder
        // name must not be mangled per-repo.
        let first_in_b = find_shared_snapshot_with_root(&root_b, first_repo, Some("main")).unwrap();
        assert_eq!(first_in_b.commit, COMMIT_V1);
    }

    #[test]
    fn shared_cache_root_resolution_order() {
        assert_eq!(
            resolve_shared_cache_root(
                Some("/hub/custom".into()),
                Some("/hf/home".into()),
                Some("/home".into())
            ),
            Some(PathBuf::from("/hub/custom"))
        );
        // An empty HUGGINGFACE_HUB_CACHE falls through to HF_HOME.
        assert_eq!(
            resolve_shared_cache_root(
                Some(String::new()),
                Some("/hf/home".into()),
                Some("/home".into())
            ),
            Some(PathBuf::from("/hf/home").join("hub"))
        );
        // HF_HOME beats the home-directory default.
        assert_eq!(
            resolve_shared_cache_root(None, Some("/hf/home".into()), Some("/home".into())),
            Some(PathBuf::from("/hf/home").join("hub"))
        );
        // No env at all: ~/.cache/huggingface/hub.
        assert_eq!(
            resolve_shared_cache_root(None, None, Some("/home".into())),
            Some(PathBuf::from("/home/.cache/huggingface/hub"))
        );
        // No home directory: None — the probe is skipped, never a panic.
        assert_eq!(resolve_shared_cache_root(None, None, None), None);
    }

    #[cfg(unix)]
    #[test]
    fn materialize_from_shared_symlinks_files_into_target() {
        let tmp = TempDir::new().unwrap();
        let snapshot = SharedSnapshot {
            commit: COMMIT_MAIN.to_string(),
            dir: tmp.path().join("snapshots").join(COMMIT_MAIN),
        };
        write_file(&snapshot.dir, "model.gguf", "payload");
        write_file(&snapshot.dir, "sub/config.json", "{}");
        let target = tmp.path().join("target");

        materialize_from_shared(&snapshot, &["model.gguf", "sub/config.json"], &target).unwrap();

        let model_link = target.join("model.gguf");
        assert!(model_link
            .symlink_metadata()
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read(&model_link).unwrap(), b"payload");
        assert_eq!(
            std::fs::read(target.join("sub/config.json")).unwrap(),
            b"{}"
        );

        // Existing targets are left untouched, not replaced.
        std::fs::write(target.join("sub/config.json"), "custom").unwrap();
        materialize_from_shared(&snapshot, &["sub/config.json"], &target).unwrap();
        assert_eq!(
            std::fs::read(target.join("sub/config.json")).unwrap(),
            b"custom"
        );
    }

    #[cfg(unix)]
    #[test]
    fn materialize_from_shared_replaces_dangling_target() {
        let tmp = TempDir::new().unwrap();
        let snapshot = SharedSnapshot {
            commit: COMMIT_MAIN.to_string(),
            dir: tmp.path().join("snapshots").join(COMMIT_MAIN),
        };
        write_file(&snapshot.dir, "model.gguf", "payload");
        let target = tmp.path().join("target");
        // The target already holds a dangling symlink for the file (its
        // previous source was pruned): materialization must replace it
        // instead of failing with EEXIST.
        std::fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(tmp.path().join("gone-blob"), target.join("model.gguf"))
            .unwrap();

        materialize_from_shared(&snapshot, &["model.gguf"], &target).unwrap();
        assert_eq!(
            std::fs::read(target.join("model.gguf")).unwrap(),
            b"payload"
        );
    }

    #[cfg(unix)]
    #[test]
    fn remove_dangling_files_removes_only_dangling_entries() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("materialized");
        write_file(&dir, "model_metadata.json", "{}");
        write_file(tmp.path(), "blob", "payload");
        std::os::unix::fs::symlink(tmp.path().join("blob"), dir.join("model.gguf")).unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::os::unix::fs::symlink(tmp.path().join("gone"), dir.join("sub/tokenizer.json"))
            .unwrap();

        assert_eq!(remove_dangling_files(&dir), 1);
        // The dangling entry is gone; the real file and the live symlink stay.
        assert!(dir.join("sub/tokenizer.json").symlink_metadata().is_err());
        assert_eq!(std::fs::read(dir.join("model.gguf")).unwrap(), b"payload");
        assert_eq!(
            std::fs::read(dir.join("model_metadata.json")).unwrap(),
            b"{}"
        );
        // A second pass finds nothing left to heal.
        assert_eq!(remove_dangling_files(&dir), 0);
    }

    #[test]
    fn remove_dangling_files_is_zero_for_regular_files_and_missing_dirs() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("materialized");
        write_file(&dir, "model.gguf", "payload");

        assert_eq!(remove_dangling_files(&dir), 0);
        assert_eq!(remove_dangling_files(&tmp.path().join("absent")), 0);
    }
}

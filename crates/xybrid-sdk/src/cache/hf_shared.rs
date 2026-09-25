//! Reuse of the standard shared Hugging Face hub cache.
//!
//! The Hugging Face ecosystem (`huggingface-cli download`, the `transformers`
//! and `diffusers` libraries, ...) keeps downloaded models in a shared hub
//! cache at `~/.cache/huggingface/hub` by default, overridable through the
//! same env vars those tools honor (see [`shared_cache_root`]). When xybrid
//! is asked to load a model the user already downloaded with those tools,
//! re-downloading wastes bandwidth.
//!
//! This module probes that shared cache read-only and
//! [`crate::model::ModelLoader::load_from_huggingface`] uses it three ways:
//!
//! 1. **Per-file reuse (the normal online path).** The requested revision is
//!    first resolved to its current commit SHA through the Hub API — so a
//!    stale local `main` ref can never win over the remote — and file
//!    selection runs against the authoritative repo manifest. Each selected
//!    file already present in the shared snapshot at that exact commit is
//!    then materialized from it (symlinked on unix, copied elsewhere). A
//!    file with no snapshot at that commit can still be reused
//!    content-addressed from the repo's `blobs/` store when its hash matches
//!    the Hub's manifest (e.g. the commit moved but the file didn't); only
//!    files the shared cache truly lacks are downloaded.
//! 2. **Explicit offline mode.** With `HF_HUB_OFFLINE` set, the locally
//!    cached refs resolve the revision and the snapshot loads with zero
//!    network I/O — with a warning, since neither freshness nor completeness
//!    against the remote repo can be verified offline.
//! 3. **Connectivity fallback.** When the Hub API fails with a transport
//!    error or a server-side 5xx, the same local-refs path is tried; client
//!    errors (bad repo, bad revision, bad auth) propagate instead of being
//!    masked by a stale local copy.
//!
//! The probe is desktop-only (macOS/Linux/Windows allowlist): on every other
//! target — iOS, Android, and anything else non-desktop —
//! [`find_shared_snapshot`] is a stub returning `None`, keeping their
//! behavior unchanged.

use std::path::{Path, PathBuf};

/// A snapshot of a model repo found in the shared Hugging Face hub cache.
#[derive(Debug, Clone)]
pub(crate) struct SharedSnapshot {
    /// Resolved commit hash the snapshot corresponds to.
    pub(crate) commit: String,
    /// Snapshot directory: `<hub root>/<repo dir>/snapshots/<commit>`.
    pub(crate) dir: PathBuf,
}

/// Resolve the shared Hugging Face hub cache root the way current
/// `huggingface_hub` does: `HF_HUB_CACHE`, then the legacy
/// `HUGGINGFACE_HUB_CACHE`, then `HF_HOME/hub`, then
/// `XDG_CACHE_HOME/huggingface/hub`, then `~/.cache/huggingface/hub`.
///
/// Returns `None` when no home directory can be determined; callers then
/// silently skip the probe. Unlike `hf_hub::Cache::from_env()` /
/// `hf_hub::Cache::default()` this never panics on a missing home directory.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) fn shared_cache_root() -> Option<PathBuf> {
    resolve_shared_cache_root(
        std::env::var("HF_HUB_CACHE").ok(),
        std::env::var("HUGGINGFACE_HUB_CACHE").ok(),
        std::env::var("HF_HOME").ok(),
        std::env::var("XDG_CACHE_HOME").ok(),
        dirs::home_dir(),
    )
    .and_then(absolutize)
}

/// Anchor a relative cache root (e.g. `HF_HUB_CACHE=./hf-cache`) to the
/// working directory. Left relative, it would later become a relative
/// symlink *target*, which the OS resolves against the link's own directory
/// instead of the working directory — a silently dangling link.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn absolutize(path: PathBuf) -> Option<PathBuf> {
    if path.is_absolute() {
        Some(path)
    } else {
        std::env::current_dir().ok().map(|cwd| cwd.join(path))
    }
}

/// Pure resolution of the shared cache root from explicit inputs (testable
/// without touching process env vars). Empty values fall through to the next
/// candidate; a leading `~`/`~/` is expanded against the home directory (a
/// tilde value with no known home also falls through).
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn resolve_shared_cache_root(
    hub_cache: Option<String>,
    legacy_hub_cache: Option<String>,
    hf_home: Option<String>,
    xdg_cache_home: Option<String>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    let home_dir = home.as_deref();
    for value in [hub_cache, legacy_hub_cache] {
        if let Some(path) = env_path(value, home_dir) {
            return Some(path);
        }
    }
    if let Some(path) = env_path(hf_home, home_dir) {
        return Some(path.join("hub"));
    }
    if let Some(path) = env_path(xdg_cache_home, home_dir) {
        return Some(path.join("huggingface").join("hub"));
    }
    home.map(|home| home.join(".cache").join("huggingface").join("hub"))
}

/// Turn an env-var value into a usable path: `None` for unset/empty (or a
/// tilde value when no home directory is known), tilde-expanded otherwise.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn env_path(value: Option<String>, home: Option<&Path>) -> Option<PathBuf> {
    let value = value?;
    if value.is_empty() {
        return None;
    }
    if value == "~" {
        return home.map(Path::to_path_buf);
    }
    if let Some(rest) = value.strip_prefix("~/") {
        return home.map(|home| home.join(rest));
    }
    Some(PathBuf::from(value))
}

/// Find a usable snapshot of `repo` in the shared Hugging Face hub cache
/// without any network I/O.
///
/// Resolution order: (a) the requested revision, read from
/// `<repo dir>/refs/<revision>` — or, when the revision is a full commit SHA,
/// probed directly as `snapshots/<revision>` (tools write refs under branch
/// names, so a SHA may have no ref entry); (b) without a requested revision,
/// the `main` ref, then the `master` ref. The resolved commit's snapshot
/// directory must exist and contain at least one model weight file
/// (`.gguf`/`.onnx`/`.safetensors`).
///
/// Returns `None` (with a debug log) when the repo is absent from the shared
/// cache, the revision cannot be resolved, or the snapshot holds no model
/// payload — the caller then falls through to the normal download path
/// unchanged.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
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

/// Non-desktop targets (iOS, Android, and anything else outside the
/// macOS/Linux/Windows allowlist): the shared-cache probe is a desktop
/// feature, so never hit it.
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) fn find_shared_snapshot(_repo: &str, _revision: Option<&str>) -> Option<SharedSnapshot> {
    None
}

/// Whether the shared cache holds a `blobs/` store for `repo` at all — a
/// cheap local pre-check before the caller pays a network request for the
/// repo's file hashes.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) fn shared_repo_has_blobs(repo: &str) -> bool {
    let Some(root) = shared_cache_root() else {
        return false;
    };
    if !is_safe_repo_id(repo) {
        return false;
    }
    root.join(repo_folder_name(repo)).join("blobs").is_dir()
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) fn shared_repo_has_blobs(_repo: &str) -> bool {
    false
}

/// Locate one content-addressed blob in the shared cache's per-repo
/// `blobs/` store.
///
/// The hub cache names blobs by content hash — the LFS sha256 for large
/// files, the git blob sha1 for small ones — so a hit means the shared cache
/// already holds these exact bytes even when no snapshot of the wanted
/// commit exists (e.g. the commit moved but this file didn't). Both inputs
/// are validated before being joined into a path.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) fn shared_repo_blob_path(repo: &str, oid: &str) -> Option<PathBuf> {
    let root = shared_cache_root()?;
    shared_repo_blob_path_with_root(&root, repo, oid)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) fn shared_repo_blob_path(_repo: &str, _oid: &str) -> Option<PathBuf> {
    None
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn shared_repo_blob_path_with_root(root: &Path, repo: &str, oid: &str) -> Option<PathBuf> {
    if !is_safe_repo_id(repo) || !is_hex_oid(oid) {
        return None;
    }
    let path = root.join(repo_folder_name(repo)).join("blobs").join(oid);
    path.is_file().then_some(path)
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn repo_folder_name(repo: &str) -> String {
    hf_hub::Repo::new(repo.to_string(), hf_hub::RepoType::Model).folder_name()
}

/// A git blob sha1 (40 hex) or an LFS sha256 (64 hex) — the two shapes the
/// hub cache uses for blob filenames.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn is_hex_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64) && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Probe a specific hub root directly (used by tests to avoid process env).
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn find_shared_snapshot_with_root(
    root: &Path,
    repo: &str,
    revision: Option<&str>,
) -> Option<SharedSnapshot> {
    use hf_hub::{Cache, Repo, RepoType};

    // Hugging Face repo ids are `owner/name` over [A-Za-z0-9._-]; anything
    // else (backslashes, dot-dot segments, ...) must not be turned into a
    // filesystem probe path.
    if !is_safe_repo_id(repo) {
        log::debug!(
            target: "xybrid_sdk",
            "Skipping shared Hugging Face cache probe for unsafe repo id '{}'",
            repo
        );
        return None;
    }

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
/// A revision that already is a full 40-hex commit SHA is returned as-is and
/// never looked up through `refs/` — a planted `refs/<sha>` file must not be
/// able to redirect an immutable request to a different commit. Other
/// revisions read `refs/<revision>`; without a requested revision, `main`
/// is tried, then `master`.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn resolve_commit(
    root: &Path,
    folder_name: &str,
    repo: &str,
    revision: Option<&str>,
) -> Option<String> {
    if let Some(requested) = revision {
        if is_full_commit_sha(requested) {
            return Some(requested.to_string());
        }
        let resolved = read_ref(root, folder_name, requested);
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

/// Read `refs/<revision>` inside the repo folder and return the commit SHA it
/// names.
///
/// The revision is validated before it is joined into a path (relative,
/// `Normal` components only — nested refs like `pr/123` are fine, traversal
/// and absolute paths are not), the read is capped, and the contents must be
/// exactly a 40-hex commit SHA: the result is later joined under
/// `snapshots/`, so a ref file holding an arbitrary path must never redirect
/// the snapshot lookup outside the cache.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn read_ref(root: &Path, folder_name: &str, revision: &str) -> Option<String> {
    use std::io::Read;

    if !is_safe_ref_name(revision) {
        log::debug!(
            target: "xybrid_sdk",
            "Ignoring unsafe Hugging Face revision '{}'",
            revision
        );
        return None;
    }
    let path = root.join(folder_name).join("refs").join(revision);
    let mut value = String::new();
    std::fs::File::open(path)
        .ok()?
        .take(256)
        .read_to_string(&mut value)
        .ok()?;
    let value = value.trim();
    if is_full_commit_sha(value) {
        Some(value.to_string())
    } else {
        log::debug!(
            target: "xybrid_sdk",
            "Shared Hugging Face ref '{}' does not name a commit SHA; ignoring",
            revision
        );
        None
    }
}

/// A revision usable as a relative `refs/` path: no traversal, no absolute
/// paths, no backslashes. Nested names (`pr/123`) are allowed.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn is_safe_ref_name(revision: &str) -> bool {
    use std::path::Component;
    !revision.is_empty()
        && !revision.contains('\\')
        && Path::new(revision)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Mirror Hugging Face's canonical repo-id validation: at most one `/`, at
/// most 96 chars, segments over [A-Za-z0-9._-] that don't start or end with
/// `.`/`-`, and no `--`, `..`, or `.git` suffix.
///
/// The `--` rule matters beyond hygiene: the hub cache encodes `/` as `--`
/// in folder names, so an id containing `--` (e.g. `org--model`) would alias
/// another repo's cache folder (`org/model`) and let a lookup under one name
/// silently return the other repo's snapshot.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn is_safe_repo_id(repo: &str) -> bool {
    if repo.is_empty() || repo.len() > 96 {
        return false;
    }
    if repo.contains("--") || repo.contains("..") || repo.ends_with(".git") {
        return false;
    }
    repo.split('/').count() <= 2
        && repo.split('/').all(|segment| {
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                && !segment.starts_with(['.', '-'])
                && !segment.ends_with(['.', '-'])
        })
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn is_full_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether the snapshot contains at least one model weight file
/// (`.gguf`/`.onnx`/`.safetensors`). A snapshot holding only metadata or
/// config files has nothing worth reusing.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn snapshot_has_model_payload(snapshot_dir: &Path) -> bool {
    list_snapshot_files(snapshot_dir)
        .iter()
        .any(|filename| is_model_weight_file(filename))
}

fn is_model_weight_file(filename: &str) -> bool {
    filename.ends_with(".gguf") || filename.ends_with(".onnx") || filename.ends_with(".safetensors")
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
        link_or_copy(&snapshot.dir.join(filename), &target_dir.join(filename))?;
    }
    Ok(())
}

/// Materialize one file: symlink on unix, copy elsewhere.
///
/// Idempotent and race-tolerant: an existing target is left untouched, a
/// dangling symlink occupying the entry is replaced, and losing an
/// `AlreadyExists` race against a concurrent materialization of the same
/// target counts as success.
pub(crate) fn link_or_copy(source_path: &Path, target_path: &Path) -> std::io::Result<()> {
    if target_path.exists() {
        return Ok(());
    }
    // A dangling symlink from an earlier materialization (its source cache
    // was pruned) reports `!exists()` but still occupies the directory
    // entry, so linking over it would fail; remove it first.
    if target_path.symlink_metadata().is_ok() {
        std::fs::remove_file(target_path)?;
    }
    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(source_path, target_path);
    #[cfg(not(unix))]
    let result = std::fs::copy(source_path, target_path).map(|_| ());
    match result {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && target_path.exists() => Ok(()),
        other => other,
    }
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
/// Removal only happens when the target is definitively gone
/// (`ErrorKind::NotFound`); a target that errors for another reason
/// (permissions, symlink loop, unreachable mount) is counted — so the caller
/// still refuses to load from the directory — but not deleted, since the
/// error may be transient. Removal itself is best-effort (failures are
/// logged and still counted). Regular files and symlinks that resolve are
/// never touched.
pub(crate) fn remove_dangling_files(dir: &Path) -> usize {
    fn walk(dir: &Path, unusable: &mut usize) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() {
                walk(&path, unusable);
            } else if file_type.is_symlink() {
                match std::fs::metadata(&path) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        *unusable += 1;
                        if let Err(e) = std::fs::remove_file(&path) {
                            log::debug!(
                                target: "xybrid_sdk",
                                "Failed to remove dangling symlink {}: {}",
                                path.display(),
                                e
                            );
                        }
                    }
                    Err(e) => {
                        *unusable += 1;
                        log::debug!(
                            target: "xybrid_sdk",
                            "Symlink {} is unreadable ({}); leaving it in place",
                            path.display(),
                            e
                        );
                    }
                }
            }
        }
    }

    let mut unusable = 0;
    walk(dir, &mut unusable);
    unusable
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
    fn aliasing_repo_ids_cannot_read_another_repos_snapshot() {
        let tmp = TempDir::new().unwrap();
        // The hub cache encodes '/' as "--" in folder names, so the invalid
        // id "org--model" would resolve to the SAME folder as "org/model".
        assert_eq!(repo_folder("org--model"), repo_folder(REPO));
        write_file(
            tmp.path(),
            &format!("{}/refs/main", repo_folder(REPO)),
            COMMIT_MAIN,
        );
        build_snapshot(tmp.path(), COMMIT_MAIN);

        // The valid id reads its own snapshot; the aliasing id is rejected
        // by validation and must never reach that folder.
        assert!(find_shared_snapshot_with_root(tmp.path(), REPO, Some("main")).is_some());
        assert!(find_shared_snapshot_with_root(tmp.path(), "org--model", Some("main")).is_none());
    }

    #[test]
    fn shared_cache_root_resolution_order() {
        let home = || Some(PathBuf::from("/home"));
        // HF_HUB_CACHE wins over everything.
        assert_eq!(
            resolve_shared_cache_root(
                Some("/hub/new".into()),
                Some("/hub/legacy".into()),
                Some("/hf/home".into()),
                Some("/xdg".into()),
                home()
            ),
            Some(PathBuf::from("/hub/new"))
        );
        // Legacy HUGGINGFACE_HUB_CACHE is next.
        assert_eq!(
            resolve_shared_cache_root(
                None,
                Some("/hub/legacy".into()),
                Some("/hf/home".into()),
                Some("/xdg".into()),
                home()
            ),
            Some(PathBuf::from("/hub/legacy"))
        );
        // Empty values fall through instead of resolving to "".
        assert_eq!(
            resolve_shared_cache_root(
                Some(String::new()),
                Some(String::new()),
                Some("/hf/home".into()),
                None,
                home()
            ),
            Some(PathBuf::from("/hf/home").join("hub"))
        );
        // HF_HOME beats XDG_CACHE_HOME and the home default.
        assert_eq!(
            resolve_shared_cache_root(
                None,
                None,
                Some("/hf/home".into()),
                Some("/xdg".into()),
                home()
            ),
            Some(PathBuf::from("/hf/home").join("hub"))
        );
        // XDG_CACHE_HOME beats the home default.
        assert_eq!(
            resolve_shared_cache_root(None, None, None, Some("/xdg".into()), home()),
            Some(PathBuf::from("/xdg/huggingface/hub"))
        );
        // No env at all: ~/.cache/huggingface/hub.
        assert_eq!(
            resolve_shared_cache_root(None, None, None, None, home()),
            Some(PathBuf::from("/home/.cache/huggingface/hub"))
        );
        // Tilde values expand against the home directory.
        assert_eq!(
            resolve_shared_cache_root(Some("~/hf-hub".into()), None, None, None, home()),
            Some(PathBuf::from("/home/hf-hub"))
        );
        assert_eq!(
            resolve_shared_cache_root(None, None, Some("~".into()), None, home()),
            Some(PathBuf::from("/home").join("hub"))
        );
        // A tilde value with no known home falls through to the next source.
        assert_eq!(
            resolve_shared_cache_root(
                Some("~/hf-hub".into()),
                Some("/hub/legacy".into()),
                None,
                None,
                None
            ),
            Some(PathBuf::from("/hub/legacy"))
        );
        // No home directory: None — the probe is skipped, never a panic.
        assert_eq!(
            resolve_shared_cache_root(None, None, None, None, None),
            None
        );
    }

    #[test]
    fn unsafe_revisions_and_ref_contents_are_rejected() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        build_snapshot(tmp.path(), COMMIT_MAIN);

        // A traversal or absolute revision must never be joined into a path.
        // Plant a well-formed "ref file" outside refs/ to prove it is not
        // read: the lookup must fail on the name alone.
        write_file(tmp.path(), &format!("{folder}/outside"), COMMIT_MAIN);
        assert!(find_with_root(tmp.path(), Some("../outside")).is_none());
        assert!(find_with_root(tmp.path(), Some("refs/../../outside")).is_none());
        let absolute = tmp.path().join(&folder).join("outside");
        assert!(find_with_root(tmp.path(), Some(absolute.to_str().unwrap())).is_none());

        // Ref contents that are not a 40-hex commit SHA are ignored: an
        // absolute path, a traversal, or junk must never become a snapshot
        // lookup key.
        for (name, contents) in [
            ("abs", "/etc/passwd"),
            ("traversal", "../../../outside"),
            ("junk", "not-a-commit"),
            ("empty", ""),
            ("oversized", &"a".repeat(4096) as &str),
        ] {
            write_file(tmp.path(), &format!("{folder}/refs/{name}"), contents);
            assert!(
                find_with_root(tmp.path(), Some(name)).is_none(),
                "ref '{name}' with contents {contents:?} must be rejected"
            );
        }

        // Nested ref names (Hugging Face PR refs) still work.
        write_file(tmp.path(), &format!("{folder}/refs/pr/123"), COMMIT_MAIN);
        let snapshot = find_with_root(tmp.path(), Some("pr/123")).unwrap();
        assert_eq!(snapshot.commit, COMMIT_MAIN);
    }

    #[test]
    fn unsafe_repo_ids_are_not_probed() {
        let tmp = TempDir::new().unwrap();
        for repo in [
            "../escape/x",
            "a/../b",
            "org\\model",
            "",
            "/abs/x",
            "a//b",
            "a/b/c",
            "org--model",
            "org/model.git",
            "-org/model",
            "org/model-",
            ".org/model",
            "org/model.",
        ] {
            assert!(
                find_shared_snapshot_with_root(tmp.path(), repo, None).is_none(),
                "repo id {repo:?} must not be probed"
            );
        }
        let oversized = format!("org/{}", "a".repeat(96));
        assert!(find_shared_snapshot_with_root(tmp.path(), &oversized, None).is_none());
    }

    #[test]
    fn full_commit_sha_ignores_planted_ref_files() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        // A planted ref file named after commit A points at commit B.
        write_file(
            tmp.path(),
            &format!("{folder}/refs/{COMMIT_MAIN}"),
            COMMIT_V1,
        );
        build_snapshot(tmp.path(), COMMIT_MAIN);
        build_snapshot(tmp.path(), COMMIT_V1);

        // Requesting immutable commit A must resolve to A — never through
        // refs, which could redirect it.
        let snapshot = find_with_root(tmp.path(), Some(COMMIT_MAIN)).unwrap();
        assert_eq!(snapshot.commit, COMMIT_MAIN);
    }

    #[test]
    fn absolutize_anchors_relative_roots() {
        assert_eq!(
            absolutize(PathBuf::from("/abs/root")),
            Some(PathBuf::from("/abs/root"))
        );
        let anchored = absolutize(PathBuf::from("rel/root")).unwrap();
        assert!(anchored.is_absolute());
        assert!(anchored.ends_with("rel/root"));
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

    #[cfg(unix)]
    #[test]
    fn remove_dangling_files_counts_but_keeps_unreadable_symlinks() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("materialized");
        std::fs::create_dir_all(&dir).unwrap();
        // A symlink loop errors with something other than NotFound: the dir
        // must be reported unusable, but the entry must not be deleted.
        std::os::unix::fs::symlink(dir.join("loop"), dir.join("loop")).unwrap();

        assert_eq!(remove_dangling_files(&dir), 1);
        assert!(dir.join("loop").symlink_metadata().is_ok(), "not deleted");
    }

    #[test]
    fn metadata_only_snapshot_is_not_reused() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        write_file(tmp.path(), &format!("{folder}/refs/main"), COMMIT_MAIN);
        // model_metadata.json without any weight file is not a usable model.
        write_file(
            tmp.path(),
            &format!("{folder}/snapshots/{COMMIT_MAIN}/model_metadata.json"),
            "{}",
        );

        assert!(find_with_root(tmp.path(), None).is_none());
    }

    #[test]
    fn shared_repo_blob_path_validates_inputs_and_finds_blobs() {
        let tmp = TempDir::new().unwrap();
        let folder = repo_folder(REPO);
        let sha256 = "7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4";
        let sha1 = "0123456789abcdef0123456789abcdef01234567";
        write_file(tmp.path(), &format!("{folder}/blobs/{sha256}"), "payload");
        write_file(tmp.path(), &format!("{folder}/blobs/{sha1}"), "small");

        // Both hash shapes resolve when the blob exists.
        assert!(shared_repo_blob_path_with_root(tmp.path(), REPO, sha256).is_some());
        assert!(shared_repo_blob_path_with_root(tmp.path(), REPO, sha1).is_some());
        // A missing blob is a miss, not an error.
        assert!(shared_repo_blob_path_with_root(tmp.path(), REPO, &"a".repeat(64)).is_none());
        // Invalid hashes and repo ids must never become probe paths.
        for oid in ["", "xyz", "..", "../../etc/passwd", &"a".repeat(63)] {
            assert!(
                shared_repo_blob_path_with_root(tmp.path(), REPO, oid).is_none(),
                "oid {oid:?} must be rejected"
            );
        }
        assert!(shared_repo_blob_path_with_root(tmp.path(), "a/../b", sha256).is_none());
    }
}

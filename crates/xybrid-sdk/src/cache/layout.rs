use crate::model::SdkError;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(super) const HF_REVISIONS_DIR: &str = ".revisions";
const HF_REFS_DIR: &str = ".refs";
const HF_REPO_ID_FILE: &str = ".repo-id";
const HF_RESOLVED_REVISION_FILE: &str = ".resolved-revision";
const HF_VARIANT_FILE: &str = ".variant";
const HF_REPO_DIR_PREFIX: &str = "repo--";

/// Logical cache location for a model entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheEntryLocation {
    /// Registry bundle cache under `models/`.
    Registry,
    /// Runtime-ready extraction cache under `extracted/`.
    Extracted,
    /// Direct Hugging Face file cache under `hf/`.
    HuggingFace,
    /// Hugging Face hub blob cache under `hf-hub/`.
    HuggingFaceHub,
}

impl CacheEntryLocation {
    /// Return the stable CLI/API label for this cache location.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Registry => "models",
            Self::Extracted => "extracted",
            Self::HuggingFace => "hf",
            Self::HuggingFaceHub => "hf-hub",
        }
    }
}

/// Summary of one cached model entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntryInfo {
    /// Model or repository identifier inferred from the cache directory name.
    pub model_id: String,
    /// Cache location where this entry was found.
    pub location: CacheEntryLocation,
    /// Path to the cached model entry.
    pub path: PathBuf,
    /// Recursive size of this cache entry in bytes.
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct CacheLayout {
    registry_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheEntryRoot {
    pub(crate) location: CacheEntryLocation,
    pub(crate) path: PathBuf,
}

impl CacheLayout {
    pub(crate) fn from_registry_root(registry_root: PathBuf) -> Self {
        Self { registry_root }
    }

    pub(crate) fn registry_root(&self) -> &Path {
        &self.registry_root
    }

    pub(crate) fn cache_root(&self) -> &Path {
        if is_models_dir(&self.registry_root) {
            // `Path::parent` returns `Some("")` (an empty path), not `None`,
            // for a bare single-component relative root like "models". Guard
            // against that empty parent too — otherwise the sibling
            // `extracted/`, `hf/`, and `hf-hub/` roots would resolve
            // CWD-relative and split away from the registry bundles.
            self.registry_root
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or(&self.registry_root)
        } else {
            &self.registry_root
        }
    }

    pub(crate) fn extracted_root(&self) -> PathBuf {
        self.cache_root().join("extracted")
    }

    pub(crate) fn extracted_roots(&self) -> Vec<PathBuf> {
        let mut roots = vec![self.extracted_root()];
        if let Some(root) = self.legacy_parent_extracted_root() {
            roots.push(root);
        }
        dedup_paths(roots)
    }

    pub(crate) fn extraction_dir(&self, model_id: &str) -> PathBuf {
        self.extracted_root().join(model_id)
    }

    pub(crate) fn extraction_dirs(&self, model_id: &str) -> Vec<PathBuf> {
        self.extracted_roots()
            .into_iter()
            .map(|root| root.join(model_id))
            .collect()
    }

    pub(crate) fn registry_bundle_path(&self, hf_repo: &str, file: &str) -> PathBuf {
        self.registry_root.join(repo_leaf(hf_repo)).join(file)
    }

    pub(crate) fn huggingface_root(&self) -> PathBuf {
        self.cache_root().join("hf")
    }

    pub(crate) fn huggingface_repo_dir(&self, repo: &str) -> PathBuf {
        self.huggingface_root()
            .join(huggingface_repo_cache_dir_name(repo))
    }

    pub(crate) fn huggingface_repo_revision_dir(
        &self,
        repo: &str,
        resolved_revision: &str,
        variant: Option<&str>,
    ) -> PathBuf {
        let revision_hash = hash_huggingface_revision(resolved_revision, variant);
        self.huggingface_repo_dir(repo)
            .join(HF_REVISIONS_DIR)
            .join(revision_hash)
    }

    pub(crate) fn materialized_huggingface_revision_dir(
        &self,
        repo: &str,
        resolved_revision: &str,
        variant: Option<&str>,
    ) -> Option<PathBuf> {
        let revision_hash = hash_huggingface_revision(resolved_revision, variant);
        self.huggingface_repo_dirs(repo)
            .into_iter()
            .filter(|repo_dir| huggingface_repo_marker_matches(repo_dir, repo))
            .map(|repo_dir| repo_dir.join(HF_REVISIONS_DIR).join(&revision_hash))
            .find(|revision_dir| {
                if !revision_dir.join("model_metadata.json").is_file() {
                    return false;
                }
                let revision_matches =
                    std::fs::read_to_string(revision_dir.join(HF_RESOLVED_REVISION_FILE))
                        .is_ok_and(|value| value.trim() == resolved_revision);
                let variant_matches = match variant {
                    Some(expected) => std::fs::read_to_string(revision_dir.join(HF_VARIANT_FILE))
                        .is_ok_and(|value| value == expected),
                    None => !revision_dir.join(HF_VARIANT_FILE).exists(),
                };
                revision_matches && variant_matches
            })
    }

    pub(crate) fn record_huggingface_revision(
        &self,
        repo: &str,
        requested_revision: &str,
        resolved_revision: &str,
        variant: Option<&str>,
    ) -> Result<(), SdkError> {
        self.record_huggingface_repo(repo)?;
        let revision_dir = self.huggingface_repo_revision_dir(repo, resolved_revision, variant);
        std::fs::create_dir_all(&revision_dir)?;
        std::fs::write(
            revision_dir.join(HF_RESOLVED_REVISION_FILE),
            resolved_revision,
        )?;
        if let Some(variant) = variant {
            std::fs::write(revision_dir.join(HF_VARIANT_FILE), variant)?;
        }

        let refs_dir = self.huggingface_repo_dir(repo).join(HF_REFS_DIR);
        std::fs::create_dir_all(&refs_dir)?;
        std::fs::write(
            refs_dir.join(hash_huggingface_revision(requested_revision, variant)),
            resolved_revision,
        )?;
        Ok(())
    }

    pub(crate) fn cached_huggingface_revision(
        &self,
        repo: &str,
        requested_revision: &str,
        variant: Option<&str>,
    ) -> Result<Option<String>, SdkError> {
        for repo_dir in self.huggingface_repo_dirs(repo) {
            if !huggingface_repo_marker_matches(&repo_dir, repo) {
                continue;
            }
            let path = repo_dir
                .join(HF_REFS_DIR)
                .join(hash_huggingface_revision(requested_revision, variant));
            match std::fs::read_to_string(path) {
                Ok(value) => {
                    let value = value.trim();
                    if !value.is_empty() {
                        return Ok(Some(value.to_string()));
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(None)
    }

    pub(crate) fn is_huggingface_revision_materialized(
        &self,
        repo: &str,
        resolved_revision: &str,
        variant: Option<&str>,
    ) -> bool {
        self.materialized_huggingface_revision_dir(repo, resolved_revision, variant)
            .is_some()
    }

    pub(crate) fn huggingface_repo_dirs(&self, repo: &str) -> Vec<PathBuf> {
        let sanitized_repo = sanitize_repo_id(repo);
        let current = self.huggingface_repo_dir(repo);
        let mut dirs = vec![current];
        dirs.extend(
            [
                self.huggingface_root().join(&sanitized_repo),
                self.registry_root.join("hf").join(&sanitized_repo),
            ]
            .into_iter()
            .filter(|path| huggingface_repo_marker_matches(path, repo)),
        );
        dedup_paths(dirs)
    }

    pub(crate) fn record_huggingface_repo(&self, repo: &str) -> Result<(), SdkError> {
        let repo_dir = self.huggingface_repo_dir(repo);
        std::fs::create_dir_all(&repo_dir)?;
        std::fs::write(repo_dir.join(HF_REPO_ID_FILE), repo)?;
        Ok(())
    }

    pub(crate) fn is_huggingface_repo_materialized(&self, repo: &str, path: &Path) -> bool {
        huggingface_repo_marker_matches(path, repo) && path.join("model_metadata.json").is_file()
    }

    pub(crate) fn huggingface_hub_root(&self) -> PathBuf {
        self.cache_root().join("hf-hub")
    }

    pub(crate) fn huggingface_hub_repo_root(&self, repo: &str) -> PathBuf {
        self.huggingface_hub_root()
            .join(huggingface_repo_cache_dir_name(repo))
    }

    pub(crate) fn prepare_huggingface_hub_repo_root(
        &self,
        repo: &str,
    ) -> Result<PathBuf, SdkError> {
        let repo_root = self.huggingface_hub_repo_root(repo);
        std::fs::create_dir_all(&repo_root)?;
        std::fs::write(repo_root.join(HF_REPO_ID_FILE), repo)?;
        Ok(repo_root)
    }

    pub(crate) fn entry_roots(&self) -> Vec<CacheEntryRoot> {
        let extracted_roots = self
            .extracted_roots()
            .into_iter()
            .map(|path| CacheEntryRoot {
                location: CacheEntryLocation::Extracted,
                path,
            });
        self.dedup_roots(
            std::iter::once(CacheEntryRoot {
                location: CacheEntryLocation::Registry,
                path: self.registry_root.clone(),
            })
            .chain(extracted_roots)
            .chain([
                CacheEntryRoot {
                    location: CacheEntryLocation::HuggingFace,
                    path: self.huggingface_root(),
                },
                CacheEntryRoot {
                    location: CacheEntryLocation::HuggingFaceHub,
                    path: self.huggingface_hub_root(),
                },
                CacheEntryRoot {
                    location: CacheEntryLocation::HuggingFace,
                    path: self.registry_root.join("hf"),
                },
                CacheEntryRoot {
                    location: CacheEntryLocation::HuggingFaceHub,
                    path: self.registry_root.join("hf-hub"),
                },
            ])
            .collect(),
        )
    }

    pub(crate) fn model_roots(&self, model_id: &str) -> Vec<PathBuf> {
        let sanitized_repo = sanitize_repo_id(model_id);
        let mut roots = vec![
            self.registry_root.join(model_id),
            self.huggingface_root().join(model_id),
            self.huggingface_repo_dir(model_id),
            self.huggingface_hub_repo_root(model_id),
            self.registry_root.join("hf").join(model_id),
        ];
        roots.extend(
            [
                self.huggingface_root().join(&sanitized_repo),
                self.registry_root.join("hf").join(&sanitized_repo),
            ]
            .into_iter()
            .filter(|path| huggingface_repo_marker_matches(path, model_id)),
        );
        roots.extend(self.extraction_dirs(model_id));
        dedup_paths(roots)
    }

    pub(crate) fn cache_entries(&self) -> Result<Vec<CacheEntryInfo>, SdkError> {
        let mut entries = Vec::new();

        for root in self.entry_roots() {
            entries.extend(cache_entries_for_root(root.location, &root.path)?);
        }

        entries.sort_by(|left, right| {
            left.model_id
                .cmp(&right.model_id)
                .then_with(|| left.location.as_str().cmp(right.location.as_str()))
                .then_with(|| left.path.cmp(&right.path))
        });

        Ok(entries)
    }

    fn dedup_roots(&self, roots: Vec<CacheEntryRoot>) -> Vec<CacheEntryRoot> {
        let mut seen = HashSet::new();
        roots
            .into_iter()
            .filter(|root| seen.insert(root.path.clone()))
            .collect()
    }

    fn legacy_parent_extracted_root(&self) -> Option<PathBuf> {
        if is_models_dir(&self.registry_root) {
            return None;
        }

        let root = self.registry_root.parent()?.join("extracted");
        (root != self.extracted_root()).then_some(root)
    }

    fn huggingface_hub_roots(&self) -> Vec<PathBuf> {
        dedup_paths(vec![
            self.huggingface_hub_root(),
            self.registry_root.join("hf-hub"),
        ])
    }
}

fn cache_entries_for_root(
    location: CacheEntryLocation,
    root: &Path,
) -> Result<Vec<CacheEntryInfo>, SdkError> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let Some(model_id) = entry.file_name().into_string().ok() else {
            continue;
        };

        if model_id.starts_with('.') || is_reserved_registry_dir(location, &model_id) {
            continue;
        }

        match location {
            CacheEntryLocation::HuggingFace => {
                entries.extend(huggingface_cache_entries(&model_id, &entry.path())?);
            }
            CacheEntryLocation::HuggingFaceHub => {
                entries.extend(huggingface_hub_cache_entries(&model_id, &entry.path())?);
            }
            CacheEntryLocation::Registry | CacheEntryLocation::Extracted => {
                entries.push(CacheEntryInfo {
                    model_id,
                    location,
                    path: entry.path(),
                    size_bytes: dir_size(&entry.path())?,
                });
            }
        }
    }

    Ok(entries)
}

fn huggingface_hub_cache_entries(
    model_id: &str,
    repo_root: &Path,
) -> Result<Vec<CacheEntryInfo>, SdkError> {
    let recorded_repo = match std::fs::read_to_string(repo_root.join(HF_REPO_ID_FILE)) {
        Ok(value) if !value.is_empty() => Some(value),
        Ok(_) => None,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    // A hash-named directory without its identity marker is incomplete and
    // cannot be attributed safely. Pre-hash hf-hub directories use the
    // `models--owner--repo` form instead; keep those visible under that legacy
    // label so users can list and explicitly evict them after upgrading.
    if model_id.starts_with(HF_REPO_DIR_PREFIX) && recorded_repo.is_none() {
        return Ok(Vec::new());
    }
    Ok(vec![CacheEntryInfo {
        model_id: recorded_repo.as_deref().unwrap_or(model_id).to_string(),
        location: CacheEntryLocation::HuggingFaceHub,
        path: repo_root.to_path_buf(),
        size_bytes: dir_size_excluding(repo_root, &[HF_REPO_ID_FILE])?,
    }])
}

fn huggingface_cache_entries(
    model_id: &str,
    repo_dir: &Path,
) -> Result<Vec<CacheEntryInfo>, SdkError> {
    let recorded_repo = match std::fs::read_to_string(repo_dir.join(HF_REPO_ID_FILE)) {
        Ok(value) if !value.is_empty() => Some(value),
        Ok(_) => None,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    // Do not infer an original `org/repo` from an unmarked legacy label: the
    // old slash-to-`--` encoding was not injective. Listing the on-disk label
    // still makes that cache discoverable and explicitly evictable without
    // trusting it as model input.
    if model_id.starts_with(HF_REPO_DIR_PREFIX) && recorded_repo.is_none() {
        return Ok(Vec::new());
    }
    let model_id = recorded_repo.as_deref().unwrap_or(model_id);
    let mut entries = Vec::new();
    if has_direct_materialization(repo_dir)? {
        entries.push(CacheEntryInfo {
            model_id: model_id.to_string(),
            location: CacheEntryLocation::HuggingFace,
            path: repo_dir.to_path_buf(),
            size_bytes: dir_size_excluding(
                repo_dir,
                &[HF_REVISIONS_DIR, HF_REFS_DIR, HF_REPO_ID_FILE],
            )?,
        });
    }

    let revisions_dir = repo_dir.join(HF_REVISIONS_DIR);
    if !revisions_dir.is_dir() {
        return Ok(entries);
    }

    for entry in std::fs::read_dir(revisions_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() || !entry.path().join("model_metadata.json").is_file() {
            continue;
        }
        let Ok(resolved_revision) =
            std::fs::read_to_string(entry.path().join(HF_RESOLVED_REVISION_FILE))
        else {
            continue;
        };
        let resolved_revision = resolved_revision.trim();
        if resolved_revision.is_empty() {
            continue;
        }
        let revision_label = encode_cache_label_component(resolved_revision);
        let variant = match std::fs::read_to_string(entry.path().join(HF_VARIANT_FILE)) {
            Ok(value) => Some(value),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let variant_label = variant
            .as_deref()
            .map(|value| format!("#{}", encode_cache_label_component(value)))
            .unwrap_or_default();
        entries.push(CacheEntryInfo {
            model_id: format!("{model_id}@{revision_label}{variant_label}"),
            location: CacheEntryLocation::HuggingFace,
            path: entry.path(),
            size_bytes: dir_size(&entry.path())?,
        });
    }
    Ok(entries)
}

fn has_direct_materialization(repo_dir: &Path) -> Result<bool, SdkError> {
    for entry in std::fs::read_dir(repo_dir)? {
        let name = entry?.file_name();
        if name != HF_REVISIONS_DIR && name != HF_REFS_DIR && name != HF_REPO_ID_FILE {
            return Ok(true);
        }
    }
    Ok(false)
}

fn dir_size(path: &Path) -> Result<u64, SdkError> {
    let mut total: u64 = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += dir_size(&entry.path())?;
        }
    }
    Ok(total)
}

fn dir_size_excluding(path: &Path, excluded: &[&str]) -> Result<u64, SdkError> {
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if excluded.iter().any(|name| entry.file_name() == *name) {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += dir_size(&entry.path())?;
        }
    }
    Ok(total)
}

fn hash_huggingface_revision(revision: &str, variant: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    hasher.update((revision.len() as u64).to_le_bytes());
    hasher.update(revision.as_bytes());
    match variant {
        Some(variant) => {
            hasher.update([1]);
            hasher.update(variant.as_bytes());
        }
        None => hasher.update([0]),
    }
    format!("{:x}", hasher.finalize())
}

fn huggingface_repo_cache_dir_name(repo: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(repo.as_bytes());
    format!("{HF_REPO_DIR_PREFIX}{:x}", hasher.finalize())
}

fn huggingface_repo_marker_matches(path: &Path, repo: &str) -> bool {
    std::fs::read_to_string(path.join(HF_REPO_ID_FILE)).is_ok_and(|value| value == repo)
}

fn encode_cache_label_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write;
            write!(&mut encoded, "%{byte:02X}").expect("writing to a String cannot fail");
        }
    }
    encoded
}

fn repo_leaf(repo: &str) -> &str {
    repo.split('/').next_back().unwrap_or(repo)
}

fn sanitize_repo_id(repo: &str) -> String {
    repo.replace('/', "--")
}

fn is_models_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("models"))
}

fn is_reserved_registry_dir(location: CacheEntryLocation, model_id: &str) -> bool {
    location == CacheEntryLocation::Registry && matches!(model_id, "extracted" | "hf" | "hf-hub")
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

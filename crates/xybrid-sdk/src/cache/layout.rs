use crate::model::SdkError;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

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
        self.huggingface_root().join(sanitize_repo_id(repo))
    }

    pub(crate) fn huggingface_repo_dirs(&self, repo: &str) -> Vec<PathBuf> {
        let sanitized_repo = sanitize_repo_id(repo);
        dedup_paths(vec![
            self.huggingface_root().join(&sanitized_repo),
            self.registry_root.join("hf").join(&sanitized_repo),
        ])
    }

    pub(crate) fn huggingface_hub_root(&self) -> PathBuf {
        self.cache_root().join("hf-hub")
    }

    pub(crate) fn preferred_huggingface_hub_root(&self, repo: &str) -> PathBuf {
        let repo_dir = huggingface_hub_repo_dir_name(repo);
        self.huggingface_hub_roots()
            .into_iter()
            .find(|root| root.join(&repo_dir).exists())
            .unwrap_or_else(|| self.huggingface_hub_root())
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
        let hf_hub_repo = huggingface_hub_repo_dir_name(model_id);
        let mut roots = vec![
            self.registry_root.join(model_id),
            self.huggingface_root().join(model_id),
            self.huggingface_root().join(&sanitized_repo),
            self.huggingface_hub_root().join(&hf_hub_repo),
            self.registry_root.join("hf").join(model_id),
            self.registry_root.join("hf").join(&sanitized_repo),
            self.registry_root.join("hf-hub").join(&hf_hub_repo),
        ];
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

        entries.push(CacheEntryInfo {
            model_id,
            location,
            path: entry.path(),
            size_bytes: dir_size(&entry.path())?,
        });
    }

    Ok(entries)
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

fn repo_leaf(repo: &str) -> &str {
    repo.split('/').next_back().unwrap_or(repo)
}

fn sanitize_repo_id(repo: &str) -> String {
    repo.replace('/', "--")
}

fn huggingface_hub_repo_dir_name(repo: &str) -> String {
    format!("models--{}", sanitize_repo_id(repo))
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

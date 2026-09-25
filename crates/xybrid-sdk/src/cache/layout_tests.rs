use super::layout::{CacheEntryLocation, CacheLayout};
#[cfg(feature = "huggingface")]
use hf_hub::{Cache, Repo, RepoType};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn layout_uses_model_cache_parent_as_cache_root() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("xybrid");
    let models_dir = cache_root.join("models");
    let layout = CacheLayout::from_registry_root(models_dir.clone());

    assert_eq!(layout.cache_root(), cache_root.as_path());
    assert_eq!(layout.registry_root(), models_dir.as_path());
    assert_eq!(layout.extracted_root(), cache_root.join("extracted"));
    assert_eq!(
        layout.extraction_dir("lfm2.5-350m"),
        cache_root.join("extracted").join("lfm2.5-350m")
    );
    let repo_dir = layout.huggingface_repo_dir("owner/repo");
    assert_eq!(repo_dir.parent(), Some(cache_root.join("hf").as_path()));
    let repo_dir_name = repo_dir.file_name().unwrap().to_str().unwrap();
    assert!(repo_dir_name.starts_with("repo--"));
    assert_eq!(repo_dir_name.len(), "repo--".len() + 64);
    assert_eq!(layout.huggingface_hub_root(), cache_root.join("hf-hub"));
    let hub_repo_root = layout.huggingface_hub_repo_root("owner/repo");
    assert_eq!(
        hub_repo_root.parent(),
        Some(cache_root.join("hf-hub").as_path())
    );
    let hub_repo_name = hub_repo_root.file_name().unwrap().to_str().unwrap();
    assert!(hub_repo_name.starts_with("repo--"));
    assert_eq!(hub_repo_name.len(), "repo--".len() + 64);
}

#[test]
fn layout_keeps_custom_non_models_root_self_contained() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("custom-cache");
    let layout = CacheLayout::from_registry_root(cache_root.clone());

    assert_eq!(layout.cache_root(), cache_root.as_path());
    assert_eq!(layout.registry_root(), cache_root.as_path());
    assert_eq!(layout.extracted_root(), cache_root.join("extracted"));
    let repo_dir = layout.huggingface_repo_dir("owner/repo");
    assert_eq!(repo_dir.parent(), Some(cache_root.join("hf").as_path()));

    let roots: Vec<_> = layout
        .entry_roots()
        .into_iter()
        .map(|root| (root.location, root.path))
        .collect();

    assert!(roots.contains(&(CacheEntryLocation::HuggingFace, cache_root.join("hf"))));
    assert!(!roots.contains(&(CacheEntryLocation::HuggingFace, temp_dir.path().join("hf"))));
}

#[test]
fn custom_non_models_roots_include_legacy_parent_extracted_cache() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("custom-cache");
    let layout = CacheLayout::from_registry_root(cache_root.clone());

    let roots: Vec<_> = layout
        .entry_roots()
        .into_iter()
        .map(|root| (root.location, root.path))
        .collect();

    assert!(roots.contains(&(CacheEntryLocation::Extracted, cache_root.join("extracted"))));
    assert!(roots.contains(&(
        CacheEntryLocation::Extracted,
        temp_dir.path().join("extracted")
    )));

    let model_roots = layout.model_roots("legacy-model");
    assert!(model_roots.contains(&cache_root.join("extracted").join("legacy-model")));
    assert!(model_roots.contains(&temp_dir.path().join("extracted").join("legacy-model")));
}

#[test]
fn registry_bundle_path_uses_repo_leaf_under_models() {
    let temp_dir = TempDir::new().unwrap();
    let models_dir = temp_dir.path().join("cache").join("models");
    let layout = CacheLayout::from_registry_root(models_dir.clone());

    assert_eq!(
        layout.registry_bundle_path("xybrid-ai/kokoro-82m", "universal.xyb"),
        models_dir.join("kokoro-82m").join("universal.xyb")
    );
}

#[test]
fn direct_huggingface_roots_include_legacy_nested_locations() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("cache");
    let models_dir = cache_root.join("models");
    let layout = CacheLayout::from_registry_root(models_dir.clone());

    let current_repo = layout.huggingface_repo_dir("owner/repo");
    assert_eq!(
        layout.huggingface_repo_dirs("owner/repo"),
        vec![current_repo.clone()]
    );

    let canonical_legacy_repo = cache_root.join("hf").join("owner--repo");
    let nested_legacy_repo = models_dir.join("hf").join("owner--repo");
    for legacy_repo in [&canonical_legacy_repo, &nested_legacy_repo] {
        fs::create_dir_all(legacy_repo).unwrap();
        fs::write(legacy_repo.join(".repo-id"), "owner/repo").unwrap();
    }
    assert_eq!(
        layout.huggingface_repo_dirs("owner/repo"),
        vec![current_repo, canonical_legacy_repo, nested_legacy_repo]
    );

    let hub_repo_root = layout
        .prepare_huggingface_hub_repo_root("owner/repo")
        .unwrap();
    assert_eq!(
        hub_repo_root,
        layout.huggingface_hub_repo_root("owner/repo")
    );
    assert_eq!(
        fs::read_to_string(hub_repo_root.join(".repo-id")).unwrap(),
        "owner/repo"
    );
}

#[test]
fn marked_legacy_revision_cache_remains_readable() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("cache");
    let layout = CacheLayout::from_registry_root(cache_root.join("models"));
    let repo = "owner/repo";
    let requested_revision = "main";
    let resolved_revision = "commit-a";

    let legacy_repo = cache_root.join("hf").join("owner--repo");
    let revision_dir = legacy_repo.join(".revisions").join(
        layout
            .huggingface_repo_revision_dir(repo, resolved_revision, None)
            .file_name()
            .unwrap(),
    );
    let refs_dir = legacy_repo.join(".refs");
    fs::create_dir_all(&revision_dir).unwrap();
    fs::create_dir_all(&refs_dir).unwrap();
    fs::write(legacy_repo.join(".repo-id"), repo).unwrap();
    fs::write(revision_dir.join(".resolved-revision"), resolved_revision).unwrap();
    fs::write(revision_dir.join("model_metadata.json"), b"{}").unwrap();

    layout
        .record_huggingface_revision(repo, requested_revision, resolved_revision, None)
        .unwrap();
    let current_ref = layout
        .huggingface_repo_dir(repo)
        .join(".refs")
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .file_name();
    fs::write(refs_dir.join(current_ref), resolved_revision).unwrap();
    fs::remove_dir_all(layout.huggingface_repo_dir(repo)).unwrap();

    assert_eq!(
        layout
            .cached_huggingface_revision(repo, requested_revision, None)
            .unwrap(),
        Some(resolved_revision.to_string())
    );
    assert_eq!(
        layout.materialized_huggingface_revision_dir(repo, resolved_revision, None),
        Some(revision_dir)
    );
}

#[test]
fn direct_huggingface_revisions_have_isolated_safe_directories() {
    const SHA256_HEX_LENGTH: usize = 64;

    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("cache");
    let layout = CacheLayout::from_registry_root(cache_root.join("models"));

    let main = layout.huggingface_repo_revision_dir("owner/repo", "main", None);
    let release = layout.huggingface_repo_revision_dir("owner/repo", "release/v1", None);
    let traversal = layout.huggingface_repo_revision_dir("owner/repo", "../../outside", None);

    assert_ne!(main, release);
    assert_ne!(main, traversal);
    for path in [&main, &release, &traversal] {
        assert_eq!(path.parent().unwrap().file_name().unwrap(), ".revisions");
        assert_eq!(
            path.parent().unwrap().parent().unwrap(),
            layout.huggingface_repo_dir("owner/repo")
        );
        assert_eq!(path.file_name().unwrap().len(), SHA256_HEX_LENGTH);
    }
}

#[test]
fn resolved_huggingface_revision_is_shared_listed_and_sized_once() {
    let temp_dir = TempDir::new().unwrap();
    let layout = CacheLayout::from_registry_root(temp_dir.path().join("cache/models"));
    let commit = "0123456789abcdef";

    layout
        .record_huggingface_revision("owner/repo", "main", commit, None)
        .unwrap();
    layout
        .record_huggingface_revision("owner/repo", "release", commit, None)
        .unwrap();
    let revision_dir = layout.huggingface_repo_revision_dir("owner/repo", commit, None);
    fs::write(revision_dir.join("model_metadata.json"), b"{}").unwrap();
    fs::write(revision_dir.join("model.gguf"), b"weights").unwrap();

    assert_eq!(
        layout
            .cached_huggingface_revision("owner/repo", "main", None)
            .unwrap(),
        Some(commit.to_string())
    );
    assert_eq!(
        layout
            .cached_huggingface_revision("owner/repo", "release", None)
            .unwrap(),
        Some(commit.to_string())
    );

    let entries = layout.cache_entries().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].model_id, format!("owner/repo@{commit}"));
    assert_eq!(entries[0].path, revision_dir);
    assert_eq!(entries[0].size_bytes, (2 + 7 + commit.len()) as u64);
}

#[test]
fn incomplete_huggingface_revision_is_not_listed_as_cached() {
    let temp_dir = TempDir::new().unwrap();
    let layout = CacheLayout::from_registry_root(temp_dir.path().join("cache/models"));
    let revision_dir = layout.huggingface_repo_revision_dir("owner/repo", "commit-a", None);
    fs::create_dir_all(&revision_dir).unwrap();
    fs::write(revision_dir.join("model_metadata.json"), b"{}").unwrap();

    assert!(!layout.is_huggingface_revision_materialized("owner/repo", "commit-a", None));
    assert!(layout.cache_entries().unwrap().is_empty());
}

#[test]
fn resolved_huggingface_variants_have_distinct_cache_entries() {
    let temp_dir = TempDir::new().unwrap();
    let layout = CacheLayout::from_registry_root(temp_dir.path().join("cache/models"));
    let commit = "0123456789abcdef";

    for variant in ["Q4_K_M", "Q8_0"] {
        layout
            .record_huggingface_revision("owner/repo", "main", commit, Some(variant))
            .unwrap();
        let revision_dir =
            layout.huggingface_repo_revision_dir("owner/repo", commit, Some(variant));
        fs::write(revision_dir.join("model_metadata.json"), b"{}").unwrap();
    }

    let q4 = layout.huggingface_repo_revision_dir("owner/repo", commit, Some("Q4_K_M"));
    let q8 = layout.huggingface_repo_revision_dir("owner/repo", commit, Some("Q8_0"));
    assert_ne!(q4, q8);
    assert_eq!(
        layout
            .cached_huggingface_revision("owner/repo", "main", Some("Q4_K_M"))
            .unwrap(),
        Some(commit.to_string())
    );
    assert_eq!(
        layout
            .cached_huggingface_revision("owner/repo", "main", Some("Q8_0"))
            .unwrap(),
        Some(commit.to_string())
    );

    let mut ids: Vec<_> = layout
        .cache_entries()
        .unwrap()
        .into_iter()
        .map(|entry| entry.model_id)
        .collect();
    ids.sort();
    assert_eq!(
        ids,
        [
            format!("owner/repo@{commit}#Q4_K_M"),
            format!("owner/repo@{commit}#Q8_0"),
        ]
    );
}

#[test]
fn colliding_legacy_repo_labels_have_distinct_cache_namespaces() {
    let temp_dir = TempDir::new().unwrap();
    let layout = CacheLayout::from_registry_root(temp_dir.path().join("cache/models"));
    let first_repo = "a/b--c";
    let second_repo = "a--b/c";
    let requested_revision = "main";
    let first_commit = "commit-a";
    let second_commit = "commit-b";

    assert_eq!(
        first_repo.replace('/', "--"),
        second_repo.replace('/', "--"),
        "the previous cache layout aliased these repository IDs"
    );
    assert_ne!(
        layout.huggingface_repo_dir(first_repo),
        layout.huggingface_repo_dir(second_repo)
    );
    assert_ne!(
        layout.huggingface_repo_revision_dir(first_repo, first_commit, None),
        layout.huggingface_repo_revision_dir(second_repo, first_commit, None)
    );
    let first_hub_root = layout
        .prepare_huggingface_hub_repo_root(first_repo)
        .unwrap();
    let second_hub_root = layout
        .prepare_huggingface_hub_repo_root(second_repo)
        .unwrap();
    assert_ne!(first_hub_root, second_hub_root);
    assert_eq!(
        fs::read_to_string(first_hub_root.join(".repo-id")).unwrap(),
        first_repo
    );
    assert_eq!(
        fs::read_to_string(second_hub_root.join(".repo-id")).unwrap(),
        second_repo
    );
    let hub_entries = layout.cache_entries().unwrap();
    assert!(hub_entries.iter().any(|entry| {
        entry.location == CacheEntryLocation::HuggingFaceHub
            && entry.model_id == first_repo
            && entry.path == first_hub_root
    }));
    assert!(hub_entries.iter().any(|entry| {
        entry.location == CacheEntryLocation::HuggingFaceHub
            && entry.model_id == second_repo
            && entry.path == second_hub_root
    }));

    layout
        .record_huggingface_revision(first_repo, requested_revision, first_commit, None)
        .unwrap();
    assert_eq!(
        layout
            .cached_huggingface_revision(second_repo, requested_revision, None)
            .unwrap(),
        None,
        "one repository must never consume another repository's cached ref"
    );
    layout
        .record_huggingface_revision(second_repo, requested_revision, second_commit, None)
        .unwrap();
    assert_eq!(
        layout
            .cached_huggingface_revision(first_repo, requested_revision, None)
            .unwrap(),
        Some(first_commit.to_string())
    );
    assert_eq!(
        layout
            .cached_huggingface_revision(second_repo, requested_revision, None)
            .unwrap(),
        Some(second_commit.to_string())
    );
}

#[cfg(feature = "huggingface")]
#[test]
fn colliding_repo_ids_cannot_read_each_others_hub_snapshots() {
    let temp_dir = TempDir::new().unwrap();
    let layout = CacheLayout::from_registry_root(temp_dir.path().join("cache/models"));
    let first_repo = "a/b--c";
    let second_repo = "a--b/c";
    let commit = "commit-a";
    let filename = "model.gguf";

    let first_root = layout
        .prepare_huggingface_hub_repo_root(first_repo)
        .unwrap();
    let second_root = layout
        .prepare_huggingface_hub_repo_root(second_repo)
        .unwrap();
    let first_cache = Cache::new(first_root);
    let second_cache = Cache::new(second_root);
    let first = first_cache.repo(Repo::new(first_repo.to_string(), RepoType::Model));
    let second = second_cache.repo(Repo::new(second_repo.to_string(), RepoType::Model));

    first.create_ref(commit).unwrap();
    let first_snapshot = first.pointer_path(commit);
    fs::create_dir_all(&first_snapshot).unwrap();
    fs::write(first_snapshot.join(filename), b"attacker-controlled").unwrap();

    assert_eq!(
        fs::read(first.get(filename).expect("primed repository should hit")).unwrap(),
        b"attacker-controlled"
    );
    assert_eq!(
        second.get(filename),
        None,
        "a colliding repository ID must not consume another repo's hub snapshot"
    );
}

#[test]
fn entry_roots_include_canonical_and_legacy_direct_hf_locations() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("cache");
    let models_dir = cache_root.join("models");
    let layout = CacheLayout::from_registry_root(models_dir.clone());

    let roots: Vec<_> = layout
        .entry_roots()
        .into_iter()
        .map(|root| (root.location, root.path))
        .collect();

    assert!(roots.contains(&(CacheEntryLocation::Registry, models_dir.clone())));
    assert!(roots.contains(&(CacheEntryLocation::Extracted, cache_root.join("extracted"))));
    assert!(roots.contains(&(CacheEntryLocation::HuggingFace, cache_root.join("hf"))));
    assert!(roots.contains(&(
        CacheEntryLocation::HuggingFaceHub,
        cache_root.join("hf-hub")
    )));
    assert!(roots.contains(&(CacheEntryLocation::HuggingFace, models_dir.join("hf"))));
    assert!(roots.contains(&(
        CacheEntryLocation::HuggingFaceHub,
        models_dir.join("hf-hub")
    )));
}

#[test]
fn model_clear_roots_cover_canonical_and_legacy_repo_locations() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("cache");
    let models_dir = cache_root.join("models");
    let layout = CacheLayout::from_registry_root(models_dir.clone());

    let roots = layout.model_roots("owner/repo");

    assert!(roots.contains(&models_dir.join("owner/repo")));
    assert!(roots.contains(&cache_root.join("extracted").join("owner/repo")));
    assert!(roots.contains(&cache_root.join("hf").join("owner/repo")));
    assert!(roots.contains(&layout.huggingface_repo_dir("owner/repo")));
    assert!(!roots.contains(&cache_root.join("hf").join("owner--repo")));
    assert!(roots.contains(&layout.huggingface_hub_repo_root("owner/repo")));
    assert!(roots.contains(&models_dir.join("hf").join("owner/repo")));
    assert!(!roots.contains(&models_dir.join("hf").join("owner--repo")));
    assert!(!roots.contains(&models_dir.join("hf-hub").join("models--owner--repo")));
}

#[test]
fn layout_keeps_siblings_co_located_for_bare_relative_models_root() {
    // Regression guard for the `cache_root()` empty-parent bug: `Path::parent`
    // of a bare single-component "models" is `Some("")`, not `None`. Without
    // the empty-parent guard, the sibling extracted/hf/hf-hub roots would
    // resolve CWD-relative and split away from the registry bundles.
    let layout = CacheLayout::from_registry_root(PathBuf::from("models"));

    // cache_root() collapses to the registry root itself, never an empty path.
    assert_eq!(layout.cache_root(), Path::new("models"));
    assert_eq!(
        layout.extracted_root(),
        Path::new("models").join("extracted")
    );
    assert_eq!(layout.huggingface_root(), Path::new("models").join("hf"));
    assert_eq!(
        layout.huggingface_hub_root(),
        Path::new("models").join("hf-hub")
    );

    // Every sibling root stays nested under the registry root — none escapes to
    // a bare CWD-relative directory.
    for root in [
        layout.extracted_root(),
        layout.huggingface_root(),
        layout.huggingface_hub_root(),
    ] {
        assert!(
            root.starts_with("models"),
            "sibling root {:?} escaped the registry root",
            root
        );
    }
}

#[test]
fn cache_root_never_splits_siblings_for_non_models_roots() {
    // Any registry root whose leaf is not "models" is treated as the cache root
    // itself, so the sibling areas always nest under it — never an empty or
    // CWD-relative path. Covers the edge inputs around the bare-"models" guard:
    // a current-dir ".", an absolute non-models dir, and a trailing-slash
    // relative dir.
    for raw in [".", "/var/cache/xybrid", "relative/custom/"] {
        let root = PathBuf::from(raw);
        let layout = CacheLayout::from_registry_root(root.clone());

        assert_eq!(
            layout.cache_root(),
            root.as_path(),
            "cache_root should equal the registry root for {raw:?}"
        );
        assert_eq!(layout.extracted_root(), root.join("extracted"));
        assert_eq!(layout.huggingface_root(), root.join("hf"));
        assert_eq!(layout.huggingface_hub_root(), root.join("hf-hub"));
        assert!(
            layout.extracted_root().starts_with(&root),
            "extracted root escaped {raw:?}"
        );
    }
}

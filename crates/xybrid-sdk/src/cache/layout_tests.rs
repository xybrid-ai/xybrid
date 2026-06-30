use super::layout::{CacheEntryLocation, CacheLayout};
use std::fs;
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
    assert_eq!(
        layout.huggingface_repo_dir("owner/repo"),
        cache_root.join("hf").join("owner--repo")
    );
    assert_eq!(layout.huggingface_hub_root(), cache_root.join("hf-hub"));
}

#[test]
fn layout_keeps_custom_non_models_root_self_contained() {
    let temp_dir = TempDir::new().unwrap();
    let cache_root = temp_dir.path().join("custom-cache");
    let layout = CacheLayout::from_registry_root(cache_root.clone());

    assert_eq!(layout.cache_root(), cache_root.as_path());
    assert_eq!(layout.registry_root(), cache_root.as_path());
    assert_eq!(layout.extracted_root(), cache_root.join("extracted"));
    assert_eq!(
        layout.huggingface_repo_dir("owner/repo"),
        cache_root.join("hf").join("owner--repo")
    );

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

    let repo_dirs = layout.huggingface_repo_dirs("owner/repo");
    assert_eq!(
        repo_dirs,
        vec![
            cache_root.join("hf").join("owner--repo"),
            models_dir.join("hf").join("owner--repo")
        ]
    );

    let legacy_hub_repo = models_dir.join("hf-hub").join("models--owner--repo");
    fs::create_dir_all(&legacy_hub_repo).unwrap();

    assert_eq!(
        layout.preferred_huggingface_hub_root("owner/repo"),
        models_dir.join("hf-hub")
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
    assert!(roots.contains(&cache_root.join("hf").join("owner--repo")));
    assert!(roots.contains(&cache_root.join("hf-hub").join("models--owner--repo")));
    assert!(roots.contains(&models_dir.join("hf").join("owner/repo")));
    assert!(roots.contains(&models_dir.join("hf").join("owner--repo")));
    assert!(roots.contains(&models_dir.join("hf-hub").join("models--owner--repo")));
}

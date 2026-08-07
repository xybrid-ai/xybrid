//! Integration tests for ModelLoader::from_directory()

use std::path::PathBuf;
use xybrid_sdk::model::{ModelLoader, SdkError};

/// Helper to get the path to fixture models from the workspace root.
fn fixtures_dir() -> PathBuf {
    // Tests run from the crate directory; fixtures are at the workspace level.
    // Resolve at RUNTIME rather than with `env!`: the compile-time value is baked
    // into the binary and does not point anywhere useful when the test runs in a
    // sandbox. Both cargo and Bazel set the variable for test execution.
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests"),
    );
    manifest_dir
        .parent() // crates/
        .unwrap()
        .parent() // repos/xybrid/
        .unwrap()
        .join("integration-tests/fixtures/models")
}

#[test]
fn from_directory_loads_valid_fixture() {
    let model_dir = fixtures_dir().join("mnist");
    let loader = ModelLoader::from_directory(&model_dir).expect("should load mnist fixture");
    assert_eq!(loader.source_type(), "directory");
    // load() would require ORT runtime, so just verify from_directory succeeds
}

#[test]
fn from_directory_returns_directory_not_found() {
    let bad_path = PathBuf::from("/tmp/xybrid-test-nonexistent-dir-12345");
    let err = ModelLoader::from_directory(&bad_path).unwrap_err();
    assert!(
        matches!(err, SdkError::DirectoryNotFound(_)),
        "expected DirectoryNotFound, got: {:?}",
        err
    );
}

#[test]
fn from_directory_returns_metadata_not_found() {
    // Use a real directory that has no model_metadata.json
    let dir = std::env::temp_dir().join("xybrid-test-no-metadata");
    std::fs::create_dir_all(&dir).unwrap();
    // Ensure no model_metadata.json exists
    let _ = std::fs::remove_file(dir.join("model_metadata.json"));

    let err = ModelLoader::from_directory(&dir).unwrap_err();
    assert!(
        matches!(err, SdkError::MetadataNotFound(_)),
        "expected MetadataNotFound, got: {:?}",
        err
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn from_directory_returns_metadata_invalid() {
    // Create a directory with an invalid model_metadata.json
    let dir = std::env::temp_dir().join("xybrid-test-bad-metadata");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model_metadata.json"), "{ not valid json }").unwrap();

    let err = ModelLoader::from_directory(&dir).unwrap_err();
    assert!(
        matches!(err, SdkError::MetadataInvalid(_)),
        "expected MetadataInvalid, got: {:?}",
        err
    );

    let _ = std::fs::remove_dir_all(&dir);
}

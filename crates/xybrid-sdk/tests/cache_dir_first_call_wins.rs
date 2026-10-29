//! `init_sdk_cache_dir` keeps the first folder, environment included, so an
//! SDK's default (Unity sets one on Android) never overrides an app's earlier
//! choice. A test binary of its own, because the folder is process-wide.

#[test]
fn a_later_call_changes_neither_the_folder_nor_the_environment() {
    let first = tempfile::tempdir().expect("temp dir");
    let second = tempfile::tempdir().expect("temp dir");

    xybrid_sdk::init_sdk_cache_dir(first.path().join("models"));
    let hf_home = std::env::var("HF_HOME").expect("the first call sets HF_HOME");

    xybrid_sdk::init_sdk_cache_dir(second.path().join("models"));

    assert_eq!(
        xybrid_sdk::get_sdk_cache_dir(),
        Some(first.path().join("models"))
    );
    assert_eq!(std::env::var("HF_HOME").ok(), Some(hf_home));
}

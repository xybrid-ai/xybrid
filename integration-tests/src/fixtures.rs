use std::path::PathBuf;

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

pub fn input_dir() -> PathBuf {
    fixtures_dir().join("input")
}

pub fn models_dir() -> PathBuf {
    fixtures_dir().join("models")
}

pub fn pipelines_dir() -> PathBuf {
    fixtures_dir().join("pipelines")
}

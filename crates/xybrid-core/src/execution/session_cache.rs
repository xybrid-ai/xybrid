//! Single-slot cache for the ONNX sessions the executor builds itself.
//!
//! The TTS path and the token-ID (BERT-style) path of the `TemplateExecutor`
//! build their own CPU session instead of going through the ONNX runtime.
//! Building one loads the graph and runs Level-3 optimization, which dominates
//! a short call: 44 of ~52 ms for a MiniLM embedding on an M4 Max, against
//! 2 ms of inference. [`OnnxSessionCache`] keeps the session for the next call.
//!
//! It holds one session, not a map: the SDK gives every loaded model its own
//! executor, and a map would pin each model's session when core callers run
//! several models through one executor. Like the tokenizer cache, the session
//! is reused only while the files it was built from keep a readable, unchanged
//! `(len, modified)`, so a model replaced in place is rebuilt rather than
//! served stale. Those files are the graph and the external weights named
//! after it (`model.onnx_data`, `model.onnx.data`), where large exports keep
//! their weights; weights under another name are not checked. The session
//! lives as long as the executor and drops with it on unload.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::path::{file_identity, FileIdentity};
use super::session_factory::OnnxSessionFactory;
use super::types::ExecutorResult;
use crate::runtime_adapter::onnx::{ExecutionProviderKind, ONNXSession, SessionOptions};

/// The last session built, keyed by model path and the identity of its files.
pub(crate) struct OnnxSessionCache<S = ONNXSession> {
    slot: Option<CachedSession<S>>,
}

struct CachedSession<S> {
    model_path: PathBuf,
    /// External weight files next to the graph when it was built.
    weight_files: Vec<PathBuf>,
    /// Identity of the graph, then of each weight file.
    identity: Vec<FileIdentity>,
    session: Arc<S>,
}

impl<S> Default for OnnxSessionCache<S> {
    fn default() -> Self {
        Self { slot: None }
    }
}

impl OnnxSessionCache {
    /// Returns the CPU session for `model_path`, building it on a miss or when
    /// one of its files changed.
    ///
    /// Both callers run on CPU with default options, so the path and the files'
    /// identity are a sufficient key.
    ///
    /// # Errors
    ///
    /// Returns the build error. A failed build is not cached, so the next call
    /// retries.
    pub(crate) fn get_or_build(&mut self, model_path: &Path) -> ExecutorResult<Arc<ONNXSession>> {
        self.get_or_build_with(model_path, |path| {
            OnnxSessionFactory::create_session(
                path,
                ExecutionProviderKind::Cpu,
                SessionOptions::default(),
            )
        })
    }
}

impl<S> OnnxSessionCache<S> {
    /// [`OnnxSessionCache::get_or_build`] with the session built by `build`.
    fn get_or_build_with(
        &mut self,
        model_path: &Path,
        build: impl FnOnce(&Path) -> ExecutorResult<S>,
    ) -> ExecutorResult<Arc<S>> {
        if let Some(cached) = &self.slot {
            if cached.model_path == model_path
                && fingerprint(model_path, &cached.weight_files).as_ref() == Some(&cached.identity)
            {
                return Ok(Arc::clone(&cached.session));
            }
        }

        let weight_files = external_weight_files(model_path);
        let identity = fingerprint(model_path, &weight_files);
        let session = Arc::new(build(model_path)?);
        // An unreadable identity serves this build but is never trusted later.
        self.slot = identity.map(|identity| CachedSession {
            model_path: model_path.to_path_buf(),
            weight_files,
            identity,
            session: Arc::clone(&session),
        });
        Ok(session)
    }
}

/// Identity of the graph followed by each weight file, or `None` if any of
/// them cannot be read.
fn fingerprint(model_path: &Path, weight_files: &[PathBuf]) -> Option<Vec<FileIdentity>> {
    std::iter::once(model_path)
        .chain(weight_files.iter().map(PathBuf::as_path))
        .map(file_identity)
        .collect()
}

/// Files next to the graph whose name extends the graph's (`model.onnx_data`,
/// `model.onnx.data`, `model.onnx_data_1`), the names exporters give the
/// external data that holds a large model's weights.
fn external_weight_files(model_path: &Path) -> Vec<PathBuf> {
    let Some(name) = model_path.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    let dir = match model_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|other| other.len() > name.len() && other.starts_with(name))
        })
        .map(|entry| entry.path())
        .collect();
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_adapter::AdapterError;

    /// Builds sessions numbered in build order, so a test can tell a rebuild
    /// from a reuse.
    #[derive(Default)]
    struct Builds(u32);

    impl Builds {
        fn build(&mut self, _path: &Path) -> ExecutorResult<u32> {
            self.0 += 1;
            Ok(self.0)
        }
    }

    fn write(path: &Path, contents: &str) {
        std::fs::write(path, contents).expect("model file is written");
    }

    #[test]
    fn unchanged_file_reuses_the_session() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.onnx");
        write(&path, "graph");
        let mut cache = OnnxSessionCache::default();
        let mut builds = Builds::default();

        let first = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();
        let second = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(builds.0, 1);
    }

    #[test]
    fn file_replaced_in_place_is_rebuilt() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.onnx");
        write(&path, "graph");
        let mut cache = OnnxSessionCache::default();
        let mut builds = Builds::default();
        let before = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        // A longer file changes the identity even on filesystems with coarse
        // modification times.
        write(&path, "a larger graph");
        let after = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        assert_eq!((*before, *after), (1, 2));
    }

    #[test]
    fn replaced_external_weights_are_rebuilt() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.onnx");
        let weights = dir.path().join("model.onnx_data");
        write(&path, "graph");
        write(&weights, "weights");
        let mut cache = OnnxSessionCache::default();
        let mut builds = Builds::default();
        let before = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        // The graph is untouched; only the weights it points to change.
        write(&weights, "other weights");
        let after = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        assert_eq!((*before, *after), (1, 2));
    }

    #[test]
    fn removed_external_weights_are_rebuilt() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.onnx");
        let weights = dir.path().join("model.onnx.data");
        write(&path, "graph");
        write(&weights, "weights");
        let mut cache = OnnxSessionCache::default();
        let mut builds = Builds::default();
        cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        std::fs::remove_file(&weights).expect("weights are removed");
        cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        assert_eq!(builds.0, 2);
    }

    #[test]
    fn other_files_in_the_folder_do_not_rebuild() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.onnx");
        let tokenizer = dir.path().join("tokenizer.json");
        write(&path, "graph");
        write(&tokenizer, "{}");
        let mut cache = OnnxSessionCache::default();
        let mut builds = Builds::default();
        let before = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        write(&tokenizer, "{\"changed\": true}");
        let after = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        assert!(Arc::ptr_eq(&before, &after));
        assert_eq!(builds.0, 1);
    }

    #[test]
    fn another_path_replaces_the_cached_session() {
        let dir = tempfile::tempdir().expect("temp dir");
        let first_path = dir.path().join("first.onnx");
        let second_path = dir.path().join("second.onnx");
        write(&first_path, "graph");
        write(&second_path, "graph");
        let mut cache = OnnxSessionCache::default();
        let mut builds = Builds::default();

        let first = cache
            .get_or_build_with(&first_path, |p| builds.build(p))
            .unwrap();
        let second = cache
            .get_or_build_with(&second_path, |p| builds.build(p))
            .unwrap();
        // One slot: going back to the first model builds it again.
        let first_again = cache
            .get_or_build_with(&first_path, |p| builds.build(p))
            .unwrap();

        assert_eq!((*first, *second, *first_again), (1, 2, 3));
    }

    #[test]
    fn failed_build_is_not_cached() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.onnx");
        write(&path, "graph");
        let mut cache = OnnxSessionCache::default();
        let mut builds = Builds::default();

        let failed = cache.get_or_build_with(&path, |_| {
            Err::<u32, _>(AdapterError::RuntimeError("build failed".to_string()))
        });
        let retried = cache.get_or_build_with(&path, |p| builds.build(p)).unwrap();

        assert!(failed.is_err());
        assert_eq!((*retried, builds.0), (1, 1));
    }
}

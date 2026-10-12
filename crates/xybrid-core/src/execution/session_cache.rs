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
//! is reused only while the model file's `(len, modified)` is readable and
//! unchanged, so a model replaced in place is rebuilt rather than served stale.
//! The session lives as long as the executor and drops with it on unload.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::path::{file_identity, FileIdentity};
use super::session_factory::OnnxSessionFactory;
use super::types::ExecutorResult;
use crate::runtime_adapter::onnx::{ExecutionProviderKind, ONNXSession, SessionOptions};

/// The last session built, keyed by model path and file identity.
pub(crate) struct OnnxSessionCache<S = ONNXSession> {
    slot: Option<CachedSession<S>>,
}

struct CachedSession<S> {
    model_path: PathBuf,
    identity: FileIdentity,
    session: Arc<S>,
}

impl<S> Default for OnnxSessionCache<S> {
    fn default() -> Self {
        Self { slot: None }
    }
}

impl OnnxSessionCache {
    /// Returns the CPU session for `model_path`, building it on a miss or when
    /// the file changed.
    ///
    /// Both callers run on CPU with default options, so the path and the file
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
        let identity = file_identity(model_path);
        if let (Some(identity), Some(cached)) = (identity, &self.slot) {
            if cached.model_path == model_path && cached.identity == identity {
                return Ok(Arc::clone(&cached.session));
            }
        }

        let session = Arc::new(build(model_path)?);
        // An unreadable identity serves this build but is never trusted later.
        self.slot = identity.map(|identity| CachedSession {
            model_path: model_path.to_path_buf(),
            identity,
            session: Arc::clone(&session),
        });
        Ok(session)
    }
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

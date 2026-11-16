//! Model files, engine settings and the engine.

use std::ffi::{c_char, c_int, CStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::{Error, Result};
use crate::library::{c_path, c_string, Library};
use crate::sys;

/// What a `.litertlm` file declares about itself, read without loading the
/// model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// The file contains a vision encoder, so image input is possible.
    pub supports_vision: bool,
    /// The file contains an audio encoder.
    pub supports_audio: bool,
    /// The model's metadata declares tool (function) calling.
    pub supports_function_calling: bool,
    /// The largest context the file supports, when it declares one.
    pub max_context_tokens: Option<u32>,
    /// The oldest runtime release the file says it needs, when it says so.
    pub min_runtime_version: Option<String>,
}

impl ModelInfo {
    /// Read the capabilities of the model file at `model_path`.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] when the file cannot be opened or is not a LiteRT-LM
    /// model.
    pub fn read(library: &Library, model_path: &Path) -> Result<Self> {
        let api = library.api();
        let path = c_path("model path", model_path)?;
        let mut raw = ptr::null_mut();
        // SAFETY: a NUL-terminated path and a valid out-pointer.
        let status = unsafe { (api.loaded_file_create)(path.as_ptr(), &mut raw) };
        library.check("litert_lm_loaded_file_create", status)?;
        let file = LoadedFile { library, raw };

        let supports_vision = file.supports_modality(sys::MODALITY_VISION)?;
        let supports_audio = file.supports_modality(sys::MODALITY_AUDIO)?;

        let mut supports_function_calling = false;
        // SAFETY: `file.raw` is live until `file` drops; valid out-pointer.
        let status = unsafe {
            (api.loaded_file_supports_function_calling)(file.raw, &mut supports_function_calling)
        };
        library.check("litert_lm_loaded_file_supports_function_calling", status)?;

        let mut max_context = 0u32;
        // SAFETY: as above.
        let status = unsafe { (api.loaded_file_max_context_tokens)(file.raw, &mut max_context) };
        library.check("litert_lm_loaded_file_max_context_tokens", status)?;

        let mut min_version: *const c_char = ptr::null();
        // SAFETY: as above. The string is owned by the loaded file, which is
        // still alive when it is copied below.
        let status = unsafe { (api.loaded_file_min_runtime_version)(file.raw, &mut min_version) };
        library.check("litert_lm_loaded_file_min_runtime_version", status)?;
        let min_runtime_version = if min_version.is_null() {
            None
        } else {
            // SAFETY: non-null and NUL-terminated per the C API contract.
            let text = unsafe { CStr::from_ptr(min_version) }.to_string_lossy();
            (!text.is_empty()).then(|| text.into_owned())
        };

        Ok(Self {
            supports_vision,
            supports_audio,
            supports_function_calling,
            max_context_tokens: (max_context > 0).then_some(max_context),
            min_runtime_version,
        })
    }
}

/// RAII handle for `LiteRtLmLoadedFile`.
struct LoadedFile<'l> {
    library: &'l Library,
    raw: *mut sys::LoadedFile,
}

impl LoadedFile<'_> {
    fn supports_modality(&self, modality: c_int) -> Result<bool> {
        let mut supported = false;
        // SAFETY: `self.raw` is live; valid out-pointer.
        let status = unsafe {
            (self.library.api().loaded_file_supports_input_modality)(
                self.raw,
                modality,
                &mut supported,
            )
        };
        self.library
            .check("litert_lm_loaded_file_supports_input_modality", status)?;
        Ok(supported)
    }
}

impl Drop for LoadedFile<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `loaded_file_create`, deleted exactly once.
        unsafe { (self.library.api().loaded_file_delete)(self.raw) };
    }
}

/// Where a model, or its vision encoder, runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// CPU (XNNPACK).
    Cpu,
    /// The runtime's GPU backend (OpenCL, Metal or WebGPU, as the platform
    /// provides).
    Gpu,
}

impl Backend {
    /// The backend string the C API expects.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// Numeric precision of the activations the runtime computes with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationPrecision {
    /// 32-bit floats.
    Float32,
    /// 16-bit floats. The GPU default upstream; reported to corrupt long
    /// contexts on several GPUs with Gemma 4.
    Float16,
}

/// Settings an [`Engine`] is created from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineSettings {
    model_path: PathBuf,
    backend: Backend,
    vision_backend: Option<Backend>,
    max_num_tokens: Option<u32>,
    num_threads: Option<u32>,
    cache_dir: Option<PathBuf>,
    activation: Option<ActivationPrecision>,
}

impl EngineSettings {
    /// Settings for the model at `model_path`, on CPU, text only.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            backend: Backend::Cpu,
            vision_backend: None,
            max_num_tokens: None,
            num_threads: None,
            cache_dir: None,
            activation: None,
        }
    }

    /// Run the language model on `backend`.
    #[must_use]
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Load the vision encoder on `backend`, enabling image input. Without
    /// this the engine is text only, whatever the file contains.
    #[must_use]
    pub fn with_vision_backend(mut self, backend: Backend) -> Self {
        self.vision_backend = Some(backend);
        self
    }

    /// Cap the context (prompt plus output) at `tokens`. The runtime reserves
    /// memory for the whole of it, so pass what will be used, not what the
    /// model could support.
    #[must_use]
    pub fn with_max_num_tokens(mut self, tokens: u32) -> Self {
        self.max_num_tokens = Some(tokens);
        self
    }

    /// Use `threads` CPU threads.
    #[must_use]
    pub fn with_num_threads(mut self, threads: u32) -> Self {
        self.num_threads = Some(threads);
        self
    }

    /// Write the runtime's compiled-model caches under `dir`. By default the
    /// runtime writes them beside the model file.
    #[must_use]
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Compute activations at `precision` instead of the backend's default.
    #[must_use]
    pub fn with_activation_precision(mut self, precision: ActivationPrecision) -> Self {
        self.activation = Some(precision);
        self
    }

    /// The model file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// The language model's backend.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The vision encoder's backend, when image input is enabled.
    pub fn vision_backend(&self) -> Option<Backend> {
        self.vision_backend
    }
}

/// A loaded model. Heavyweight: it holds the weights.
///
/// Cloning is cheap and shares the same loaded model. The engine is deleted
/// when the last clone and the last [`crate::Conversation`] created from it
/// are gone, which is the order the C API requires.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

impl fmt::Debug for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("library", &self.inner.library)
            .field("usable", &self.is_usable())
            .finish()
    }
}

impl Engine {
    /// Load the model described by `settings`.
    ///
    /// Benchmark bookkeeping is always enabled: it is how token counts are
    /// read back after a turn.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] when the settings are rejected or the model cannot be
    /// loaded.
    pub fn load(library: Arc<Library>, settings: &EngineSettings) -> Result<Self> {
        let api = library.api();
        let model_path = c_path("model path", &settings.model_path)?;
        let backend = c_string("backend", settings.backend.as_str())?;
        let vision_backend = settings
            .vision_backend
            .map(|backend| c_string("vision backend", backend.as_str()))
            .transpose()?;

        let mut raw_settings = ptr::null_mut();
        // SAFETY: NUL-terminated strings (or null for an absent backend) and
        // a valid out-pointer.
        let status = unsafe {
            (api.engine_settings_create)(
                model_path.as_ptr(),
                backend.as_ptr(),
                vision_backend.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                ptr::null(),
                &mut raw_settings,
            )
        };
        library.check("litert_lm_engine_settings_create", status)?;
        let raw_settings = SettingsGuard {
            library: &library,
            raw: raw_settings,
        };

        if let Some(tokens) = settings.max_num_tokens {
            // SAFETY: live settings handle; plain value.
            let status = unsafe {
                (api.engine_settings_set_max_num_tokens)(raw_settings.raw, clamp_c_int(tokens))
            };
            library.check("litert_lm_engine_settings_set_max_num_tokens", status)?;
        }
        if let Some(threads) = settings.num_threads {
            // SAFETY: as above.
            let status = unsafe {
                (api.engine_settings_set_num_threads)(raw_settings.raw, clamp_c_int(threads))
            };
            library.check("litert_lm_engine_settings_set_num_threads", status)?;
        }
        if let Some(dir) = &settings.cache_dir {
            let dir = c_path("cache directory", dir)?;
            // SAFETY: live settings handle; NUL-terminated string, copied by
            // the callee.
            let status =
                unsafe { (api.engine_settings_set_cache_dir)(raw_settings.raw, dir.as_ptr()) };
            library.check("litert_lm_engine_settings_set_cache_dir", status)?;
        }
        if let Some(precision) = settings.activation {
            let raw_precision = match precision {
                ActivationPrecision::Float32 => sys::ACTIVATION_FLOAT32,
                ActivationPrecision::Float16 => sys::ACTIVATION_FLOAT16,
            };
            // SAFETY: live settings handle; declared enum value.
            let status = unsafe {
                (api.engine_settings_set_activation_data_type)(raw_settings.raw, raw_precision)
            };
            library.check("litert_lm_engine_settings_set_activation_data_type", status)?;
        }
        // SAFETY: live settings handle.
        let status = unsafe { (api.engine_settings_enable_benchmark)(raw_settings.raw) };
        library.check("litert_lm_engine_settings_enable_benchmark", status)?;

        let mut raw_engine = ptr::null_mut();
        // SAFETY: live settings handle and a valid out-pointer.
        let status = unsafe { (api.engine_create)(raw_settings.raw, &mut raw_engine) };
        library.check("litert_lm_engine_create", status)?;
        let raw = NonNull::new(raw_engine).ok_or(Error::Protocol(
            "engine_create succeeded with a null engine",
        ))?;
        drop(raw_settings);

        Ok(Self {
            inner: Arc::new(EngineInner {
                library,
                raw,
                calls: Mutex::new(()),
                unusable: AtomicBool::new(false),
            }),
        })
    }

    /// The library this engine was created from.
    pub fn library(&self) -> &Arc<Library> {
        &self.inner.library
    }

    /// `false` once the runtime could not be brought to rest on this engine
    /// (a stream stalled, or a wait failed). An unusable engine refuses new
    /// conversations, its existing conversations refuse further work, and
    /// all of them are leaked instead of deleted.
    pub fn is_usable(&self) -> bool {
        !self.inner.is_unusable()
    }

    pub(crate) fn inner(&self) -> &Arc<EngineInner> {
        &self.inner
    }
}

fn clamp_c_int(value: u32) -> c_int {
    c_int::try_from(value).unwrap_or(c_int::MAX)
}

struct SettingsGuard<'l> {
    library: &'l Library,
    raw: *mut sys::EngineSettings,
}

impl Drop for SettingsGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `engine_settings_create`, deleted exactly once.
        unsafe { (self.library.api().engine_settings_delete)(self.raw) };
    }
}

/// The engine handle and what guards it.
pub(crate) struct EngineInner {
    pub(crate) library: Arc<Library>,
    raw: NonNull<sys::Engine>,
    /// Serializes calls that take the engine handle. Upstream: "calls that
    /// use the same handle MUST NOT overlap".
    calls: Mutex<()>,
    /// Set when a stream stalled. See [`Engine::is_usable`].
    unusable: AtomicBool,
}

// SAFETY: the engine handle may be used from any thread as long as calls on
// it do not overlap (upstream's threading model). Every use of `raw` goes
// through `lock_handle`, which holds `calls`, or happens in `drop`, which has
// exclusive access.
unsafe impl Send for EngineInner {}
// SAFETY: as above; shared references only reach `raw` through the mutex.
unsafe impl Sync for EngineInner {}

impl EngineInner {
    /// Exclusive use of the engine handle for one C call.
    pub(crate) fn lock_handle(&self) -> Result<EngineHandle<'_>> {
        if self.is_unusable() {
            return Err(Error::EngineUnusable);
        }
        let guard = self
            .calls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(EngineHandle {
            raw: self.raw.as_ptr(),
            _guard: guard,
        })
    }

    /// Record that the runtime could not be brought to rest on this engine:
    /// a stream stalled, or a wait failed. From then on nothing enters the
    /// runtime through this engine or its conversations, and none of them is
    /// freed.
    pub(crate) fn mark_unusable(&self) {
        self.unusable.store(true, Ordering::Release);
    }

    /// Whether [`Self::mark_unusable`] was called.
    pub(crate) fn is_unusable(&self) -> bool {
        self.unusable.load(Ordering::Acquire)
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        if self.is_unusable() {
            // A conversation the runtime may still be working on was leaked.
            // Deleting the engine under it would be a use-after-free inside
            // the runtime, so the engine is leaked too.
            return;
        }
        // SAFETY: every conversation holds an `Arc<EngineInner>`, so none is
        // alive here; the handle is deleted exactly once.
        unsafe { (self.library.api().engine_delete)(self.raw.as_ptr()) };
    }
}

/// The engine handle, held exclusively.
pub(crate) struct EngineHandle<'e> {
    pub(crate) raw: *mut sys::Engine,
    _guard: MutexGuard<'e, ()>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_default_to_cpu_text_only() {
        let settings = EngineSettings::new("/models/m.litertlm");
        assert_eq!(settings.backend(), Backend::Cpu);
        assert_eq!(settings.vision_backend(), None);
        assert_eq!(settings.model_path(), Path::new("/models/m.litertlm"));
    }

    #[test]
    fn backend_strings_match_the_c_api() {
        assert_eq!(Backend::Cpu.as_str(), "cpu");
        assert_eq!(Backend::Gpu.as_str(), "gpu");
    }

    #[test]
    fn oversized_counts_are_clamped_not_wrapped() {
        assert_eq!(clamp_c_int(4096), 4096);
        assert_eq!(clamp_c_int(u32::MAX), c_int::MAX);
    }
}

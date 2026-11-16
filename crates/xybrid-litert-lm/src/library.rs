//! Opening the runtime library and checking that it can be called.

use std::ffi::{c_int, CStr, CString, OsString};
use std::fmt;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result, StatusCode};
use crate::sys;

/// Environment variable naming the runtime library file to open.
///
/// Takes precedence over the platform's default name. Meant for desktop
/// development and tests; a packaged app ships the library where the
/// platform loader finds it by name.
pub const LIBRARY_ENV: &str = "XYBRID_LITERT_LM_LIBRARY";

/// The C API major version these bindings were written against.
pub const C_API_MAJOR: i32 = 1;

/// The file name the platform loader is asked for when no path is given.
///
/// On Android the name resolves inside the app's own native library
/// directory, so the library must be packaged with the app.
pub fn default_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "litert-lm.dll"
    } else if cfg!(any(target_os = "macos", target_os = "ios")) {
        "liblitert-lm.dylib"
    } else {
        "liblitert-lm.so"
    }
}

/// How much the runtime writes to its own log (standard error, or logcat on
/// Android). The setting is global to the loaded library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Everything.
    Verbose,
    /// Informational messages and above.
    Info,
    /// Warnings and errors.
    Warning,
    /// Errors only.
    Error,
    /// Nothing.
    Silent,
}

impl LogLevel {
    fn raw(self) -> c_int {
        match self {
            Self::Verbose => sys::LOG_VERBOSE,
            Self::Info => sys::LOG_INFO,
            Self::Warning => sys::LOG_WARNING,
            Self::Error => sys::LOG_ERROR,
            Self::Silent => sys::LOG_SILENT,
        }
    }
}

/// An opened LiteRT-LM runtime library whose C API version was accepted.
///
/// Shared through an [`Arc`]: every [`crate::Engine`] holds one, so the
/// function table outlives everything created from it. The library is never
/// unloaded, because the runtime owns threads that may still be running when
/// the last handle goes away.
pub struct Library {
    api: sys::Api,
    origin: String,
    version: (i32, i32, i32),
    /// Keeps the mapping alive. Deliberately never dropped.
    _handle: Option<ManuallyDrop<libloading::Library>>,
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library")
            .field("origin", &self.origin)
            .field("c_api_version", &self.version)
            .finish()
    }
}

impl Library {
    /// Open the runtime library at `path`, or by bare name through the
    /// platform loader's search rules.
    ///
    /// # Errors
    ///
    /// [`Error::LibraryUnavailable`] when the file cannot be opened,
    /// [`Error::MissingSymbol`] when it lacks a function these bindings
    /// call, and [`Error::IncompatibleVersion`] when it reports a C API major
    /// version other than [`C_API_MAJOR`].
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let path = path.as_ref();
        let origin = path.display().to_string();
        // SAFETY: loading a shared library runs its initializers. The caller
        // names the file, and on a packaged app it is one the app itself
        // ships. Nothing is called until the version check below passes.
        let handle =
            unsafe { libloading::Library::new(path) }.map_err(|err| Error::LibraryUnavailable {
                tried: origin.clone(),
                reason: err.to_string(),
            })?;
        // SAFETY: the signatures are those of C API 1.x, which the version
        // check in `from_parts` confirms before any other function is used.
        // `litert_lm_get_api_version` itself is guaranteed stable across
        // every version by upstream's header.
        let api =
            unsafe { sys::Api::load(&handle) }.map_err(|(symbol, err)| Error::MissingSymbol {
                symbol: symbol.to_string(),
                reason: err.to_string(),
            })?;
        Self::from_parts(api, origin, Some(ManuallyDrop::new(handle))).map(Arc::new)
    }

    /// Open the library named by [`LIBRARY_ENV`], or else the platform's
    /// default name ([`default_library_name`]).
    ///
    /// # Errors
    ///
    /// As [`Library::open`].
    pub fn open_default() -> Result<Arc<Self>> {
        match std::env::var_os(LIBRARY_ENV).filter(|value| !value.is_empty()) {
            Some(path) => Self::open(path),
            None => Self::open(OsString::from(default_library_name())),
        }
    }

    /// Wrap an in-process function table, for tests that replace the runtime
    /// with a fake. The version check still applies.
    ///
    /// # Safety
    ///
    /// Every function in `api` is called as the LiteRT-LM C API function of
    /// the same name, so each must keep that function's contract as the
    /// upstream headers state it: write its out-parameters as documented,
    /// return strings that stay valid for as long as the header says, create
    /// and destroy objects consistently with the other functions of the
    /// table, tolerate calls from any thread under the header's threading
    /// model, and invoke stream callbacks only with the callback data it was
    /// given. The functions must stay callable for the rest of the process.
    ///
    /// # Errors
    ///
    /// [`Error::IncompatibleVersion`] when the table reports another major
    /// version.
    pub unsafe fn from_api(api: sys::Api) -> Result<Arc<Self>> {
        Self::from_parts(api, "in-process".to_string(), None).map(Arc::new)
    }

    fn from_parts(
        api: sys::Api,
        origin: String,
        handle: Option<ManuallyDrop<libloading::Library>>,
    ) -> Result<Self> {
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        // SAFETY: three valid out-pointers, as the function requires.
        let status = unsafe { (api.get_api_version)(&mut major, &mut minor, &mut patch) };
        let library = Self {
            api,
            origin,
            version: (major, minor, patch),
            _handle: handle,
        };
        library.check("litert_lm_get_api_version", status)?;
        if major != C_API_MAJOR {
            return Err(Error::IncompatibleVersion {
                major,
                minor,
                patch,
                needed_major: C_API_MAJOR,
            });
        }
        Ok(library)
    }

    /// The C API version the library reports, as `(major, minor, patch)`.
    ///
    /// This is the version of the C interface, not of the LiteRT-LM release:
    /// the C API has no call that reports the release.
    pub fn c_api_version(&self) -> (i32, i32, i32) {
        self.version
    }

    /// Where the library came from: the path or name it was opened with.
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// Set how much the runtime logs. Global to the loaded library.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the runtime rejects the level.
    pub fn set_min_log_level(&self, level: LogLevel) -> Result<()> {
        // SAFETY: plain value argument.
        let status = unsafe { (self.api.set_min_log_level)(level.raw()) };
        self.check("litert_lm_set_min_log_level", status)
    }

    /// The raw function table, for C API calls this crate does not wrap.
    ///
    /// Calling through it is `unsafe` and outside the handle discipline the
    /// safe types keep; see upstream's headers for each function's contract.
    pub fn api(&self) -> &sys::Api {
        &self.api
    }

    /// Turn a status into a `Result`, reading the runtime's last error
    /// message. Must run on the thread that made the failing call, right
    /// after it: the message is thread-local.
    pub(crate) fn check(&self, call: &'static str, status: sys::Status) -> Result<()> {
        if status == sys::STATUS_OK {
            return Ok(());
        }
        // SAFETY: no arguments; the returned pointer is owned by the library
        // and valid until the next failing call on this thread.
        let message = unsafe { (self.api.get_last_error_message)() };
        let message = if message.is_null() {
            String::new()
        } else {
            // SAFETY: non-null, NUL-terminated per the C API contract.
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::Call {
            call,
            code: StatusCode::from_raw(status),
            message,
        })
    }
}

/// Convert a Rust string for a C call, naming the argument on failure.
pub(crate) fn c_string(what: &'static str, value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| Error::InteriorNul { what })
}

/// Convert a path for a C call.
pub(crate) fn c_path(what: &'static str, path: &Path) -> Result<CString> {
    c_string(what, &path.to_string_lossy())
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn version_1_0_0(
        major: *mut c_int,
        minor: *mut c_int,
        patch: *mut c_int,
    ) -> sys::Status {
        *major = 1;
        *minor = 0;
        *patch = 0;
        sys::STATUS_OK
    }

    unsafe extern "C" fn version_2_0_0(
        major: *mut c_int,
        minor: *mut c_int,
        patch: *mut c_int,
    ) -> sys::Status {
        *major = 2;
        *minor = 0;
        *patch = 0;
        sys::STATUS_OK
    }

    #[test]
    fn accepts_the_major_version_it_was_written_against() {
        // SAFETY: the stub's functions take nothing they dereference and
        // return failure, null or nothing; the version function writes its
        // three out-parameters. The same holds for the tables below.
        let library = unsafe {
            Library::from_api(sys::Api {
                get_api_version: version_1_0_0,
                ..sys::Api::stub()
            })
        }
        .expect("C API 1.0.0 is accepted");
        assert_eq!(library.c_api_version(), (1, 0, 0));
        assert_eq!(library.origin(), "in-process");
    }

    #[test]
    fn refuses_another_major_version() {
        // SAFETY: as in the test above.
        let err = unsafe {
            Library::from_api(sys::Api {
                get_api_version: version_2_0_0,
                ..sys::Api::stub()
            })
        }
        .expect_err("C API 2.x is refused");
        assert!(matches!(
            err,
            Error::IncompatibleVersion {
                major: 2,
                needed_major: 1,
                ..
            }
        ));
        assert!(err.is_runtime_unavailable());
    }

    #[test]
    fn a_library_without_a_version_call_is_refused() {
        // The stub's version function fails, as a pre-1.0 library's absence
        // of the symbol would.
        // SAFETY: as in the tests above.
        let err =
            unsafe { Library::from_api(sys::Api::stub()) }.expect_err("no version, no library");
        assert_eq!(err.status(), Some(StatusCode::Unimplemented));
    }

    #[test]
    fn a_missing_file_is_reported_as_unavailable() {
        let err = Library::open("/nonexistent/liblitert-lm-test.so").expect_err("no such file");
        assert!(err.is_runtime_unavailable());
        assert!(err
            .to_string()
            .contains("/nonexistent/liblitert-lm-test.so"));
    }

    #[test]
    fn interior_nul_is_rejected_by_name() {
        let err = c_string("model path", "a\0b").expect_err("NUL is rejected");
        assert_eq!(err.to_string(), "model path contains a NUL byte");
    }
}

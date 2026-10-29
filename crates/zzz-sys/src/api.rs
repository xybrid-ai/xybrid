//! One owned dispatch table for the statically linked engine; tests substitute a mock table.

use std::ffi::{c_char, c_void};
use std::sync::Arc;

use crate::{AudioCallback, EmbedError, EmbedOptions, EmbedResult, KittenOptions};

pub(crate) type AbiVersion = unsafe extern "C" fn() -> u32;
pub(crate) type SupportsModel = unsafe extern "C" fn(*const c_char) -> i32;
pub(crate) type DefaultOptions = unsafe extern "C" fn(*mut EmbedOptions);
pub(crate) type Open = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *const EmbedOptions,
    *mut *mut c_void,
    *mut EmbedError,
) -> i32;
pub(crate) type OpenKitten =
    unsafe extern "C" fn(*const KittenOptions, *mut *mut c_void, *mut EmbedError) -> i32;
pub(crate) type Synthesize = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const c_char,
    Option<AudioCallback>,
    *mut c_void,
    *mut EmbedResult,
    *mut EmbedError,
) -> i32;
pub(crate) type SessionCall = unsafe extern "C" fn(*mut c_void);

/// Why the linked engine cannot run in this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum UnavailableReason {
    /// The static Linux archive needs a v3 CPU and OS-enabled AVX state.
    #[error("Linux x86_64 requires an x86-64-v3 CPU and OS-enabled AVX/XMM/YMM state")]
    UnsupportedCpu,
}

#[derive(Clone, Copy)]
pub(crate) struct Functions {
    pub(crate) abi_version: AbiVersion,
    pub(crate) supports_model: SupportsModel,
    pub(crate) default_options: DefaultOptions,
    pub(crate) open: Open,
    pub(crate) open_kitten: OpenKitten,
    pub(crate) synthesize: Synthesize,
    pub(crate) cancel: SessionCall,
    pub(crate) close: SessionCall,
}

pub(crate) struct EngineApi {
    pub(crate) functions: Functions,
}

impl EngineApi {
    pub(crate) fn require_cpu(&self) -> Result<(), UnavailableReason> {
        require_cpu()
    }
}

fn require_cpu() -> Result<(), UnavailableReason> {
    if crate::cpu::available() {
        Ok(())
    } else {
        Err(UnavailableReason::UnsupportedCpu)
    }
}

pub(crate) fn load() -> Result<Arc<EngineApi>, UnavailableReason> {
    require_cpu()?;
    #[cfg(test)]
    if let Some(result) = TEST_API.with(|slot| slot.borrow().clone()) {
        return result;
    }
    #[cfg(feature = "bindings")]
    {
        use std::sync::OnceLock;
        static API: OnceLock<Arc<EngineApi>> = OnceLock::new();
        Ok(Arc::clone(API.get_or_init(|| {
            Arc::new(EngineApi {
                functions: static_functions(),
            })
        })))
    }
    // Only unit tests compile this module without `bindings`. Nothing is linked
    // then, so reaching here is a test bug: it must install a mock table.
    #[cfg(not(feature = "bindings"))]
    {
        panic!("no zzz engine is linked without `bindings`; wrap the test in `with_test_api`")
    }
}

/// Return runtime availability without opening model files.
///
/// # Errors
/// Returns [`UnavailableReason::UnsupportedCpu`] when this host cannot run the
/// linked archive.
pub fn availability() -> Result<(), UnavailableReason> {
    load().map(|_| ())
}

#[cfg(feature = "bindings")]
fn static_functions() -> Functions {
    extern "C" {
        fn zzz_embed_abi_version() -> u32;
        fn zzz_embed_supports_model(model: *const c_char) -> i32;
        fn zzz_embed_default_options(options: *mut EmbedOptions);
        fn zzz_embed_open(
            model: *const c_char,
            path: *const c_char,
            voice: *const c_char,
            options: *const EmbedOptions,
            session: *mut *mut c_void,
            error: *mut EmbedError,
        ) -> i32;
        fn zzz_embed_open_kitten(
            options: *const KittenOptions,
            session: *mut *mut c_void,
            error: *mut EmbedError,
        ) -> i32;
        fn zzz_embed_synthesize(
            session: *mut c_void,
            text: *const c_char,
            language: *const c_char,
            callback: Option<AudioCallback>,
            user: *mut c_void,
            result: *mut EmbedResult,
            error: *mut EmbedError,
        ) -> i32;
        fn zzz_embed_cancel(session: *mut c_void);
        fn zzz_embed_close(session: *mut c_void);
    }
    Functions {
        abi_version: zzz_embed_abi_version,
        supports_model: zzz_embed_supports_model,
        default_options: zzz_embed_default_options,
        open: zzz_embed_open,
        open_kitten: zzz_embed_open_kitten,
        synthesize: zzz_embed_synthesize,
        cancel: zzz_embed_cancel,
        close: zzz_embed_close,
    }
}

#[cfg(test)]
thread_local! {
    static TEST_API: std::cell::RefCell<Option<Result<Arc<EngineApi>, UnavailableReason>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_test_api<T>(
    value: Result<Arc<EngineApi>, UnavailableReason>,
    run: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<Result<Arc<EngineApi>, UnavailableReason>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_API.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(TEST_API.with(|slot| slot.replace(Some(value))));
    run()
}

#[cfg(test)]
pub(crate) mod tests;

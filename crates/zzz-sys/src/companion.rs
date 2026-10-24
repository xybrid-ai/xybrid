//! SDK-owned Android companion handshake; production adoption remains disabled.
//!
//! ABI 1 contains only C-layout data and nullable function pointers. No caller
//! may use a slot until the entire fixed contract and runtime probes pass.

use std::sync::Arc;

use crate::api::{self, EngineApi, Functions, UnavailableReason};

const COMPANION_ABI: u32 = 1;
const ANDROID_API: u32 = 29;
const SELECTED_MODELS: u32 = 3; // bit 0: Kitten TTS 2, bit 1: public Phonon 31M

// Intentionally UNCONFIGURED. A future reviewed adoption must pin an actual
// notice-bearing selected companion contract. ABI 1 or rc.8 alone is not it.
const EXPECTED_CONTRACT: Option<[u8; 32]> = None;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CompanionApiV1 {
    struct_size: u32,
    companion_abi: u32,
    engine_abi: u32,
    android_api: u32,
    models: u32,
    reserved: u32,
    contract_id: [u8; 32],
    abi_version: Option<api::AbiVersion>,
    supports_model: Option<api::SupportsModel>,
    default_options: Option<api::DefaultOptions>,
    open: Option<api::Open>,
    open_kitten: Option<api::OpenKitten>,
    synthesize: Option<api::Synthesize>,
    cancel: Option<api::SessionCall>,
    close: Option<api::SessionCall>,
}

// Getter copies into caller-owned storage, writes no more than out_size, and
// returns zero only for this exact ABI/size. A failure leaves no usable table.
type Getter = unsafe extern "C" fn(u32, u32, *mut CompanionApiV1) -> i32;

const _: () = {
    assert!(std::mem::offset_of!(CompanionApiV1, contract_id) == 24);
    assert!(std::mem::offset_of!(CompanionApiV1, abi_version) == 56);
    assert!(std::mem::size_of::<CompanionApiV1>() == 56 + 8 * std::mem::size_of::<usize>());
    assert!(std::mem::size_of::<Option<Getter>>() == std::mem::size_of::<usize>());
};

impl CompanionApiV1 {
    fn functions(self, expected: [u8; 32]) -> Result<Functions, UnavailableReason> {
        if self.struct_size as usize != std::mem::size_of::<Self>()
            || self.companion_abi != COMPANION_ABI
            || self.engine_abi != crate::ZZZ_EMBED_ABI_VERSION
            || self.android_api != ANDROID_API
            || self.models != SELECTED_MODELS
            || self.reserved != 0
            || self.contract_id != expected
        {
            return Err(UnavailableReason::InvalidTable);
        }
        let missing = UnavailableReason::InvalidTable;
        Ok(Functions {
            abi_version: self.abi_version.ok_or(missing)?,
            supports_model: self.supports_model.ok_or(missing)?,
            default_options: self.default_options.ok_or(missing)?,
            open: self.open.ok_or(missing)?,
            open_kitten: self.open_kitten.ok_or(missing)?,
            synthesize: self.synthesize.ok_or(missing)?,
            cancel: self.cancel.ok_or(missing)?,
            close: self.close.ok_or(missing)?,
        })
    }
}

#[derive(Clone, Copy)]
enum Platform {
    AndroidArm64,
    OtherAndroidAbi,
    OtherOs,
}

fn platform() -> Platform {
    if !cfg!(target_os = "android") {
        Platform::OtherOs
    } else if cfg!(all(target_arch = "aarch64", target_pointer_width = "64")) {
        Platform::AndroidArm64
    } else {
        Platform::OtherAndroidAbi
    }
}

/// An open handle anchors every copied function pointer. The successful
/// production instance is retained by api::load's process-lifetime OnceLock.
pub(crate) struct Library {
    #[cfg(all(
        target_os = "android",
        target_arch = "aarch64",
        target_pointer_width = "64"
    ))]
    handle: Option<std::ptr::NonNull<std::ffi::c_void>>,
    #[cfg(test)]
    drop_count: Arc<std::sync::atomic::AtomicUsize>,
}

// SAFETY: the handle is immutable after dlopen; Android permits dlsym/dlclose
// across threads. Engine session serialization lives in the safe session type.
unsafe impl Send for Library {}
// SAFETY: sharing keeps the loader reference alive; no mutable handle escapes.
unsafe impl Sync for Library {}

impl Drop for Library {
    fn drop(&mut self) {
        #[cfg(all(
            target_os = "android",
            target_arch = "aarch64",
            target_pointer_width = "64"
        ))]
        // SAFETY: Library owns one successful dlopen reference and this is its
        // final owner. No accepted pointer can outlive its EngineApi owner.
        if let Some(handle) = self.handle {
            unsafe {
                native::dlclose(handle.as_ptr());
            }
        }
        #[cfg(test)]
        self.drop_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

// This small boundary is injectable only inside the crate's unit tests. It
// never accepts a caller-selected filename, API getter or expected identity.
trait Loader {
    fn device_api(&self) -> Option<i32>;
    fn open(&self) -> Result<Library, UnavailableReason>;
    fn getter(&self, library: &Library) -> Result<Getter, UnavailableReason>;
}

fn load_with(
    platform: Platform,
    expected: Option<[u8; 32]>,
    loader: &impl Loader,
) -> Result<Arc<EngineApi>, UnavailableReason> {
    match platform {
        Platform::OtherOs => return Err(UnavailableReason::UnsupportedPlatform),
        Platform::OtherAndroidAbi => return Err(UnavailableReason::UnsupportedAbi),
        Platform::AndroidArm64 => {}
    }
    let expected = expected.ok_or(UnavailableReason::Unconfigured)?;
    match loader.device_api() {
        Some(api) if api >= ANDROID_API as i32 => {}
        Some(api) if api >= 0 => return Err(UnavailableReason::UnsupportedAndroidApi),
        _ => return Err(UnavailableReason::UnknownAndroidApi),
    }
    let library = loader.open()?;
    let getter = loader.getter(&library)?;
    let mut table = CompanionApiV1::default();
    // SAFETY: getter comes from this open SDK library handle. It accepts the
    // stated size and writes only C-layout table storage alive for the call.
    let status = unsafe {
        getter(
            COMPANION_ABI,
            std::mem::size_of_val(&table) as u32,
            &mut table,
        )
    };
    if status != 0 {
        return Err(UnavailableReason::GetterRejected);
    }
    let functions = table.functions(expected)?;
    // SAFETY: all slots are present and match the fixed ABI/identity. Library
    // remains owned locally until transferred to the accepted EngineApi.
    if unsafe { (functions.abi_version)() } != crate::ZZZ_EMBED_ABI_VERSION {
        return Err(UnavailableReason::EngineAbiMismatch);
    }
    // SAFETY: both model names are static NUL-terminated strings, and the
    // immutable table's library is still open for both synchronous probes.
    if unsafe { (functions.supports_model)(c"phonon".as_ptr()) } == 0
        || unsafe { (functions.supports_model)(c"kitten-tts2".as_ptr()) } == 0
    {
        return Err(UnavailableReason::MissingModels);
    }
    Ok(Arc::new(EngineApi {
        functions,
        _library: Some(library),
    }))
}

pub(crate) fn load() -> Result<Arc<EngineApi>, UnavailableReason> {
    load_with(platform(), EXPECTED_CONTRACT, &NativeLoader)
}

struct NativeLoader;

#[cfg(not(all(
    target_os = "android",
    target_arch = "aarch64",
    target_pointer_width = "64"
)))]
impl Loader for NativeLoader {
    fn device_api(&self) -> Option<i32> {
        None
    }
    fn open(&self) -> Result<Library, UnavailableReason> {
        Err(UnavailableReason::UnsupportedAbi)
    }
    fn getter(&self, _: &Library) -> Result<Getter, UnavailableReason> {
        Err(UnavailableReason::UnsupportedAbi)
    }
}

#[cfg(all(
    target_os = "android",
    target_arch = "aarch64",
    target_pointer_width = "64"
))]
mod native {
    use super::*;
    use std::ffi::{c_char, c_int, c_void};
    use std::ptr;

    const RTLD_NOW: c_int = 2;
    const RTLD_LOCAL: c_int = 0;

    #[link(name = "dl")]
    extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub(super) fn dlclose(handle: *mut c_void) -> c_int;
    }

    impl Loader for NativeLoader {
        fn device_api(&self) -> Option<i32> {
            // SAFETY: Android's RTLD_DEFAULT is null; this lookup touches only
            // the already loaded system namespace. The API-29 symbol is never
            // a required relocation (older OSes have only a header inline).
            let symbol =
                unsafe { dlsym(ptr::null_mut(), c"android_get_device_api_level".as_ptr()) };
            if symbol.is_null() {
                return None;
            }
            // SAFETY: this fixed bionic symbol has exactly int(void) signature.
            let read_api: unsafe extern "C" fn() -> c_int = unsafe { std::mem::transmute(symbol) };
            // SAFETY: libc remains loaded for the entire process.
            Some(unsafe { read_api() })
        }

        fn open(&self) -> Result<Library, UnavailableReason> {
            // SAFETY: fixed SDK SONAME, no caller-controlled path or environment
            // override. load_with has already checked actual ARM64/API-29 OS.
            let handle = unsafe { dlopen(c"libxybrid_zzz_tts.so".as_ptr(), RTLD_NOW | RTLD_LOCAL) };
            let handle =
                std::ptr::NonNull::new(handle).ok_or(UnavailableReason::LibraryNotFound)?;
            Ok(Library {
                handle: Some(handle),
                #[cfg(test)]
                drop_count: Arc::default(),
            })
        }

        fn getter(&self, library: &Library) -> Result<Getter, UnavailableReason> {
            // SAFETY: resolve only from this owned library's handle, never the
            // global namespace. The exact getter ABI is fixed above.
            let symbol = unsafe {
                dlsym(
                    library
                        .handle
                        .ok_or(UnavailableReason::GetterNotFound)?
                        .as_ptr(),
                    c"xybrid_tts_get_api_v1".as_ptr(),
                )
            };
            if symbol.is_null() {
                return Err(UnavailableReason::GetterNotFound);
            }
            // SAFETY: an accepted SDK companion exports this fixed C signature.
            Ok(unsafe { std::mem::transmute::<*mut c_void, Getter>(symbol) })
        }
    }
}

#[cfg(test)]
mod tests;

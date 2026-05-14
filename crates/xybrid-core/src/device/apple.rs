//! Apple platform detection (macOS/iOS).
//!
//! This module handles Metal and CoreML Neural Engine detection
//! for Apple platforms.

use super::types::DetectionConfidence;

// Pull the CoreGraphics framework link into xybrid-core. `objc2-metal`
// documents `MTLCreateSystemDefaultDevice` as requiring CoreGraphics at
// link time, and `objc2-core-graphics` declares `#[link(name =
// "CoreGraphics", kind = "framework")]` for that purpose. Without
// referencing the crate from code, the linker can GC the directive
// because nothing in xybrid-core uses any symbol from
// objc2-core-graphics — which would leave the binary missing the
// framework on builds that don't transitively pull it in (release,
// stripped, LTO). The unused-import suppression is intentional and
// load-bearing.
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc2_core_graphics as _;

/// iOS/macOS device family detection result.
#[derive(Debug, Clone)]
pub struct AppleDeviceInfo {
    /// Device identifier (e.g., "iPhone12,1", "MacBookPro18,1")
    pub device_model: Option<String>,
    /// Whether Neural Engine is likely available
    pub has_neural_engine: bool,
    /// Detection confidence
    pub confidence: DetectionConfidence,
}

/// Real Metal probe via MTLCreateSystemDefaultDevice.
///
/// Returns `(present, confidence)`. On Apple platforms this is a real
/// runtime device probe — confidence is High. Off Apple, returns
/// `(false, Unknown)`.
pub fn detect_metal_with_confidence() -> (bool, DetectionConfidence) {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // MTLCreateSystemDefaultDevice in objc2-metal 0.3 is:
        //   pub extern "C-unwind" fn MTLCreateSystemDefaultDevice()
        //       -> Option<Retained<ProtocolObject<dyn MTLDevice>>>
        // It is NOT `unsafe fn` — no unsafe block needed. The
        // objc2-core-graphics sibling dep is in tree to satisfy the
        // CoreGraphics framework link this call requires.
        let device = objc2_metal::MTLCreateSystemDefaultDevice();
        (device.is_some(), DetectionConfidence::High)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        (false, DetectionConfidence::Unknown)
    }
}

/// Compatibility wrapper. Existing callers consuming a bare bool stay working;
/// new code should prefer `detect_metal_with_confidence`.
pub fn detect_metal_availability() -> bool {
    detect_metal_with_confidence().0
}

/// Detect Apple device model from environment variables.
///
/// On iOS, the device model can be obtained from various sources:
/// 1. `DEVICE_MODEL` environment variable (may be set by Flutter/runtime)
/// 2. `SIMULATOR_MODEL_IDENTIFIER` for iOS Simulator
///
/// Returns device info with Neural Engine availability inference.
pub fn detect_apple_device() -> AppleDeviceInfo {
    // 1) Env-var hint from the host runtime (Flutter sets DEVICE_MODEL,
    //    Xcode sets SIMULATOR_MODEL_IDENTIFIER). Medium confidence — we
    //    can't verify the variable wasn't lying.
    let env_model = std::env::var("DEVICE_MODEL")
        .or_else(|_| std::env::var("SIMULATOR_MODEL_IDENTIFIER"))
        .or_else(|_| std::env::var("APPLE_DEVICE_MODEL"))
        .ok();

    if let Some(ref model_str) = env_model {
        let has_ne = has_neural_engine_by_model(model_str);
        return AppleDeviceInfo {
            device_model: Some(model_str.clone()),
            has_neural_engine: has_ne,
            confidence: DetectionConfidence::Medium,
        };
    }

    // 2) sysctl `hw.machine` — reads the hardware identifier straight
    //    from the kernel (e.g. "iPhone10,3", "iPad8,1", "Mac15,3").
    //    Pure Rust, no JNI / no host bridge, allowed by Apple's App
    //    Store review guidelines. This is the authoritative path.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if let Some(model) = read_hw_machine() {
        let has_ne = has_neural_engine_by_model(&model);
        return AppleDeviceInfo {
            device_model: Some(model),
            has_neural_engine: has_ne,
            confidence: DetectionConfidence::High,
        };
    }

    // 3) Final fallback. Reached only if sysctl is unavailable, which
    //    is an exotic environment. Branch carefully — aarch64 alone
    //    is NOT a guarantee of Neural Engine on iOS (A9..A10X lack
    //    ANE; only A11+ have it).
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        // Every aarch64 Mac is Apple Silicon → ANE present.
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: true,
            confidence: DetectionConfidence::High,
        }
    }
    #[cfg(all(target_arch = "aarch64", target_os = "ios"))]
    {
        // aarch64 iOS spans A9 through A17; ANE only on A11+. Without
        // a model identifier we cannot tell — signal Unknown honestly
        // rather than fabricating a value the routing engine could act on.
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: false,
            confidence: DetectionConfidence::Unknown,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Intel Mac — no Apple device has ANE on x86_64.
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: false,
            confidence: DetectionConfidence::High,
        }
    }
    // Non-Apple aarch64 hosts (Linux aarch64, Android aarch64). The
    // function isn't *expected* to run here — these hosts can't actually
    // host an Apple device — but the file is compiled for every target
    // in the workspace, so this branch is needed to keep the function
    // a `-> AppleDeviceInfo` rather than `-> ()`. Env-var hints above
    // are the only signal that could have produced a real value; if we
    // reached this point, we had none.
    #[cfg(all(
        target_arch = "aarch64",
        not(any(target_os = "macos", target_os = "ios"))
    ))]
    {
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: false,
            confidence: DetectionConfidence::Unknown,
        }
    }
}

/// Read the hardware-model identifier via `sysctlbyname`. Returns the
/// kernel's identifier — `"iPhone10,3"` on iOS, `"Mac15,3"` on macOS.
///
/// Apple's sysctl keys are platform-specific:
/// - **iOS / iPadOS / tvOS / watchOS**: `hw.machine` returns the
///   device identifier (`"iPhone15,2"`).
/// - **macOS**: `hw.machine` returns the architecture (`"arm64"`),
///   not the model. The model lives under `hw.model` (`"Mac15,3"`).
///
/// In the iOS Simulator the kernel returns the host Mac's architecture
/// rather than the simulated device's identifier;
/// `has_neural_engine_by_model("arm64")` falls through to the
/// conservative "unknown device" default. The env-var path
/// (`SIMULATOR_MODEL_IDENTIFIER`) catches the simulated model first
/// when Xcode sets it.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn read_hw_machine() -> Option<String> {
    use std::ffi::CStr;

    #[cfg(target_os = "macos")]
    let key: &CStr = c"hw.model";
    #[cfg(target_os = "ios")]
    let key: &CStr = c"hw.machine";

    // First call: query the required buffer size.
    let mut size: libc::size_t = 0;
    // SAFETY: passing a null oldp with a valid oldlenp asks sysctl to
    // write the required size into oldlenp without writing data. Name
    // is a NUL-terminated string literal.
    let ret = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 || size == 0 {
        return None;
    }

    let mut buf = vec![0u8; size];
    // SAFETY: buf is sized exactly to `size` from the previous query.
    let ret = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            buf.as_mut_ptr() as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return None;
    }

    CStr::from_bytes_until_nul(&buf)
        .ok()?
        .to_str()
        .ok()
        .map(String::from)
}

/// Real Neural Engine probe via Core ML's MLAllComputeDevices.
///
/// Returns `(present, confidence)`. On macOS 14+ / iOS 17+ this calls
/// `MLAllComputeDevices` and looks for an `MLNeuralEngineComputeDevice`
/// in the result — confidence is High. On older OS where the class
/// doesn't exist in the Objective-C runtime, falls through to the
/// device-model/sysctl path and preserves that fallback's confidence.
/// Off Apple, returns `(false, Unknown)`.
pub fn detect_neural_engine_with_confidence() -> (bool, DetectionConfidence) {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use objc2::rc::Retained;
        use objc2::runtime::{AnyClass, NSObjectProtocol, ProtocolObject};
        use objc2_core_ml::MLComputeDeviceProtocol;
        use objc2_foundation::NSArray;

        // First gate (cheap): MLNeuralEngineComputeDevice was added in
        // macOS 14 / iOS 17. AnyClass::get returns None if the class
        // isn't registered with the Objective-C runtime.
        // c-string literals (`c"..."`) require Rust 1.77+. The
        // workspace has no pinned MSRV (xybrid-core/Cargo.toml) and
        // CI uses `dtolnay/rust-toolchain@stable`, so this is fine;
        // clippy's `manual_c_str_literals` lint also prefers this form
        // over the `CStr::from_bytes_with_nul` alternative.
        let Some(ne_class) = AnyClass::get(c"MLNeuralEngineComputeDevice") else {
            let info = detect_apple_device();
            return (info.has_neural_engine, info.confidence);
        };

        // Second gate: resolve MLAllComputeDevices at runtime via dlsym
        // rather than through the `objc2-core-ml` generated wrapper. The
        // wrapper emits a strong extern reference to the C symbol, which
        // can cause dyld to refuse loading the binary on macOS 13 / iOS
        // 16 where the symbol doesn't exist. We deliberately do NOT
        // enable the `MLAllComputeDevices` feature in Cargo.toml; this
        // dlsym path is the only entry point.
        //
        // Type the function pointer with a RAW *mut NSArray return,
        // matching the underlying C ABI. Wrapping the return in
        // `Retained<...>` directly would skip the autorelease retain
        // handoff that the generated wrapper performs, causing
        // ownership corruption on the returned object.
        type AllComputeDevicesFn = unsafe extern "C-unwind" fn() -> *mut NSArray<
            ProtocolObject<dyn MLComputeDeviceProtocol>,
        >;

        // SAFETY: dlsym with RTLD_DEFAULT searches the process's loaded
        // libraries for the named symbol. The call has no caller-side
        // preconditions; the return is either a valid function pointer
        // or null. We only invoke the pointer after a null check.
        let raw = unsafe {
            libc::dlsym(
                libc::RTLD_DEFAULT,
                c"MLAllComputeDevices".as_ptr() as *const _,
            )
        };
        if raw.is_null() {
            let info = detect_apple_device();
            return (info.has_neural_engine, info.confidence);
        }

        // SAFETY: `raw` is non-null (checked above). The transmute
        // matches the C symbol's documented ABI: no arguments, returns
        // an autoreleased `*mut NSArray<id<MLComputeDeviceProtocol>>`.
        let all_compute_devices: AllComputeDevicesFn = unsafe { std::mem::transmute(raw) };
        // SAFETY: the function has no preconditions per Apple's docs.
        // The returned pointer is autoreleased (+0 ownership).
        let raw_array = unsafe { all_compute_devices() };

        // SAFETY: `retain_autoreleased` balances the autorelease and
        // gives us a `Retained<...>` we own. It returns `None` if the
        // pointer is null, which we treat as "Apple gave us nothing"
        // and fall back to the device-model/sysctl path.
        let Some(devices) = (unsafe { Retained::retain_autoreleased(raw_array) }) else {
            let info = detect_apple_device();
            return (info.has_neural_engine, info.confidence);
        };

        // `isKindOfClass:` is exposed as safe in objc2-foundation 0.3.
        let has_ne = devices.iter().any(|device| device.isKindOfClass(ne_class));
        (has_ne, DetectionConfidence::High)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        (false, DetectionConfidence::Unknown)
    }
}

/// Check if a device model has Neural Engine based on identifier.
///
/// Neural Engine was introduced with:
/// - iPhone: A11 Bionic (iPhone 8/X, 2017) - "iPhone10,x"
/// - iPad: A12X Bionic (iPad Pro 2018) - "iPad8,x"
/// - Mac: M1 (2020) - "MacBookPro17,x", "MacBookAir10,x", etc.
pub fn has_neural_engine_by_model(model: &str) -> bool {
    let model_lower = model.to_lowercase();

    // iPhone detection
    // iPhone10,x = iPhone 8/X (A11) - first with Neural Engine
    // All iPhones after iPhone 10,x have Neural Engine
    if model_lower.starts_with("iphone") {
        if let Some(version_str) = model
            .strip_prefix("iPhone")
            .or_else(|| model.strip_prefix("iphone"))
        {
            if let Some((major_str, _)) = version_str.split_once(',') {
                if let Ok(major) = major_str.parse::<u32>() {
                    // iPhone10+ has Neural Engine (iPhone 8/X and later)
                    return major >= 10;
                }
            }
        }
        // Unknown iPhone format - conservative: assume newer iPhone
        return true;
    }

    // iPad detection
    // iPad8,x = iPad Pro 2018 (A12X) - first iPad with Neural Engine
    // iPad mini 5 (iPad11,1) also has NE
    if model_lower.starts_with("ipad") {
        if let Some(version_str) = model
            .strip_prefix("iPad")
            .or_else(|| model.strip_prefix("ipad"))
        {
            if let Some((major_str, _)) = version_str.split_once(',') {
                if let Ok(major) = major_str.parse::<u32>() {
                    // iPad8+ has Neural Engine
                    return major >= 8;
                }
            }
        }
        // Unknown iPad format - conservative: assume no NE (safer)
        return false;
    }

    // Mac detection - All Apple Silicon Macs have Neural Engine
    // M1 Macs: MacBookPro17, MacBookAir10, Macmini9, iMac21, Mac13
    // M2+ Macs: MacBookPro18+, MacBookAir11+, etc.
    if model_lower.contains("mac") {
        // Check for known Apple Silicon identifiers
        // These are ARM64 Macs with Neural Engine
        let apple_silicon_patterns = [
            "macbookpro17",
            "macbookpro18",
            "macbookpro19",
            "macbookpro20",
            "macbookair10",
            "macbookair11",
            "macbookair12",
            "macmini9",
            "macmini10",
            "imac21",
            "imac22",
            "imac23",
            "imac24",
            "mac13",
            "mac14",
            "mac15", // Mac Studio, Mac Pro
        ];

        for pattern in &apple_silicon_patterns {
            if model_lower.contains(pattern) {
                return true;
            }
        }

        // Unknown Mac - check architecture at compile time
        #[cfg(target_arch = "aarch64")]
        return true; // ARM64 Mac = Apple Silicon
        #[cfg(not(target_arch = "aarch64"))]
        return false; // Intel Mac
    }

    // Apple TV - A10X Fusion (2017) and later have some ANE capability
    // But it's limited, so conservative: false
    if model_lower.starts_with("appletv") {
        return false;
    }

    // Apple Watch - S4 (2018) and later have NE, but we likely won't run on Watch
    if model_lower.starts_with("watch") {
        return false;
    }

    // Unknown device - conservative default
    false
}

/// Detects CoreML Neural Engine availability (macOS/iOS only).
///
/// Compatibility wrapper. New code should prefer
/// `detect_neural_engine_with_confidence`.
pub fn detect_coreml_availability() -> bool {
    detect_neural_engine_with_confidence().0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::types::DetectionConfidence;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn test_metal_probe_returns_high_on_apple() {
        let (_present, confidence) = detect_metal_with_confidence();
        assert_eq!(confidence, DetectionConfidence::High);
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_metal_probe_returns_unknown_off_apple() {
        let (present, confidence) = detect_metal_with_confidence();
        assert!(!present);
        assert_eq!(confidence, DetectionConfidence::Unknown);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn test_neural_engine_probe_uses_runtime_class_lookup() {
        use objc2::runtime::AnyClass;
        let class_present = AnyClass::get(c"MLNeuralEngineComputeDevice").is_some();
        let (_has_ne, confidence) = detect_neural_engine_with_confidence();
        if class_present {
            assert_eq!(
                confidence,
                DetectionConfidence::High,
                "Class present → real Core ML probe ran → High confidence",
            );
        } else {
            let fallback = detect_apple_device();
            assert_eq!(
                confidence, fallback.confidence,
                "Class absent → fallback confidence should be preserved",
            );
        }
    }
}

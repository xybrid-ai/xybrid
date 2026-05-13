//! Windows platform detection.
//!
//! DXGI adapter enumeration via the `windows` crate. Software adapters
//! (WARP) are filtered out — the routing engine wants a real hardware
//! accelerator before preferring local execution (see
//! pipeline/resolver.rs:336 comment).

use super::types::DetectionConfidence;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, DXGI_ADAPTER_FLAG, DXGI_ADAPTER_FLAG_SOFTWARE,
};

/// Returns `(hardware_gpu_present, confidence)`.
///
/// Confidence is `High` whenever `CreateDXGIFactory1` succeeds —
/// including a result where the only adapter is WARP, which we
/// filter to `(false, High)`. Confidence is `Unknown` only when
/// DXGI itself cannot be loaded.
pub fn detect_gpu_with_confidence() -> (bool, DetectionConfidence) {
    // SAFETY: CreateDXGIFactory1 is a COM constructor with no
    // preconditions; the `windows` crate's Drop releases the COM ref.
    let factory_result: windows::core::Result<IDXGIFactory1> = unsafe { CreateDXGIFactory1() };
    let Ok(factory) = factory_result else {
        return (false, DetectionConfidence::Unknown);
    };

    let mut index: u32 = 0;
    loop {
        // SAFETY: EnumAdapters1 on a valid factory is safe to call with
        // any u32 index; out-of-range returns DXGI_ERROR_NOT_FOUND
        // which becomes Err in the windows crate's binding.
        let adapter_result: windows::core::Result<IDXGIAdapter1> =
            unsafe { factory.EnumAdapters1(index) };
        let Ok(adapter) = adapter_result else {
            // Walked all adapters; none were hardware.
            return (false, DetectionConfidence::High);
        };

        // SAFETY: GetDesc1 on a valid adapter has no preconditions. In
        // `windows` 0.61 it returns `Result<DXGI_ADAPTER_DESC1>` (the
        // struct is the Ok value; there is no out-pointer parameter).
        if let Ok(desc) = unsafe { adapter.GetDesc1() } {
            let is_software =
                (DXGI_ADAPTER_FLAG(desc.Flags as i32).0 & DXGI_ADAPTER_FLAG_SOFTWARE.0) != 0;
            if !is_software {
                return (true, DetectionConfidence::High);
            }
        }
        index += 1;
        // Sanity bound — DXGI on any real machine returns < 16 adapters.
        if index > 16 {
            return (false, DetectionConfidence::High);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_returns_one_of_two_documented_states() {
        let (_present, confidence) = detect_gpu_with_confidence();
        assert!(matches!(
            confidence,
            DetectionConfidence::High | DetectionConfidence::Unknown,
        ));
    }
}

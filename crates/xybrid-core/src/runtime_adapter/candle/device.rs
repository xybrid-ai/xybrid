//! Device selection for Candle inference.
//!
//! This module handles device selection (CPU, Metal, CUDA) based on
//! available hardware and feature flags.

use candle_core::Device;
use log::{debug, info, warn};

/// Device selection preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSelection {
    /// Automatically select best available device
    #[default]
    Auto,
    /// Force CPU execution
    Cpu,
    /// Force Metal (macOS/iOS) - requires candle-metal feature
    Metal,
    /// Force CUDA (NVIDIA) - requires candle-cuda feature
    Cuda(usize),
}

/// Check if running on iOS Simulator.
///
/// Metal is not properly supported on iOS Simulator and will crash when
/// trying to allocate buffers. We detect this at runtime to fall back to CPU.
fn is_ios_simulator() -> bool {
    // iOS Simulator runs on x86_64 or arm64 macOS, but reports as "ios" target
    // The definitive check is the TARGET_OS_SIMULATOR environment variable
    // which is set by Xcode when building for simulator
    #[cfg(target_os = "ios")]
    {
        // At runtime, we can check if we're on a simulator by looking at
        // environment or using the SIMULATOR_DEVICE_NAME env var
        if std::env::var("SIMULATOR_DEVICE_NAME").is_ok() {
            return true;
        }

        // Alternative: check if the device model is "simulator"
        // This uses sysctl on Darwin systems
        #[cfg(target_vendor = "apple")]
        {
            use std::ffi::CStr;
            use std::os::raw::c_char;

            extern "C" {
                fn sysctlbyname(
                    name: *const c_char,
                    oldp: *mut u8,
                    oldlenp: *mut usize,
                    newp: *const u8,
                    newlen: usize,
                ) -> i32;
            }

            let mut buffer = [0u8; 256];
            let mut size = buffer.len();
            let name = b"hw.model\0";

            unsafe {
                if sysctlbyname(
                    name.as_ptr() as *const c_char,
                    buffer.as_mut_ptr(),
                    &mut size,
                    std::ptr::null(),
                    0,
                ) == 0
                {
                    if let Ok(model) = CStr::from_ptr(buffer.as_ptr() as *const c_char).to_str() {
                        // Simulator models typically contain "Mac" or specific patterns
                        // Real iOS devices have models like "iPhone14,2"
                        if model.contains("Mac") || model.contains("x86") {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    #[cfg(not(target_os = "ios"))]
    {
        false
    }
}

/// Select the best available device based on preferences and features.
///
/// # Arguments
///
/// * `preference` - Device selection preference
///
/// # Returns
///
/// The selected `candle_core::Device`
///
/// # Device Selection Order (Auto mode)
///
/// 1. Metal (if available and candle-metal feature enabled)
/// 2. CUDA (if available and candle-cuda feature enabled)
/// 3. CPU (always available)
pub fn select_device(preference: DeviceSelection) -> candle_core::Result<Device> {
    match preference {
        DeviceSelection::Cpu => Ok(Device::Cpu),

        DeviceSelection::Metal => {
            #[cfg(feature = "candle-metal")]
            {
                // Check if running on iOS Simulator - Metal doesn't work there
                if is_ios_simulator() {
                    warn!("Metal not supported on iOS Simulator, falling back to CPU");
                    return Ok(Device::Cpu);
                }
                Device::new_metal(0)
            }
            #[cfg(not(feature = "candle-metal"))]
            {
                warn!("Metal requested but candle-metal feature not enabled, falling back to CPU");
                Ok(Device::Cpu)
            }
        }

        DeviceSelection::Cuda(ordinal) => {
            #[cfg(feature = "candle-cuda")]
            {
                Device::new_cuda(ordinal)
            }
            #[cfg(not(feature = "candle-cuda"))]
            {
                let _ = ordinal;
                warn!("CUDA requested but candle-cuda feature not enabled, falling back to CPU");
                Ok(Device::Cpu)
            }
        }

        DeviceSelection::Auto => {
            // Check if running on iOS Simulator first - skip Metal entirely
            if is_ios_simulator() {
                info!("Running on iOS Simulator, using CPU device");
                return Ok(Device::Cpu);
            }

            // Try Metal first (macOS/iOS real device)
            #[cfg(feature = "candle-metal")]
            {
                match Device::new_metal(0) {
                    Ok(device) => {
                        info!("Auto-selected Metal device");
                        return Ok(device);
                    }
                    Err(e) => {
                        debug!("Metal device not available: {}, trying alternatives", e);
                    }
                }
            }

            // Try CUDA next (NVIDIA GPUs)
            #[cfg(feature = "candle-cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(device) => {
                        info!("Auto-selected CUDA device 0");
                        return Ok(device);
                    }
                    Err(e) => {
                        debug!("CUDA device not available: {}, falling back to CPU", e);
                    }
                }
            }

            // Fall back to CPU
            info!("Using CPU device");
            Ok(Device::Cpu)
        }
    }
}

/// Get device name for display
pub fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_selection() {
        let device = select_device(DeviceSelection::Cpu).unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_auto_selection() {
        // Auto should always succeed (at minimum returns CPU)
        let device = select_device(DeviceSelection::Auto).unwrap();
        // Just verify we got a valid device
        let _ = device_name(&device);
    }

    #[test]
    fn test_device_name() {
        assert_eq!(device_name(&Device::Cpu), "CPU");
    }
}

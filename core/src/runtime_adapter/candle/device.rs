//! Device selection for Candle inference.
//!
//! This module handles device selection (CPU, Metal, CUDA) based on
//! available hardware and feature flags.

use candle_core::Device;

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
                Device::new_metal(0)
            }
            #[cfg(not(feature = "candle-metal"))]
            {
                tracing_or_log("Metal requested but candle-metal feature not enabled, falling back to CPU");
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
                tracing_or_log("CUDA requested but candle-cuda feature not enabled, falling back to CPU");
                Ok(Device::Cpu)
            }
        }

        DeviceSelection::Auto => {
            // Try Metal first (macOS/iOS)
            #[cfg(feature = "candle-metal")]
            {
                match Device::new_metal(0) {
                    Ok(device) => {
                        tracing_or_log("Auto-selected Metal device");
                        return Ok(device);
                    }
                    Err(_) => {
                        tracing_or_log("Metal device not available, trying alternatives");
                    }
                }
            }

            // Try CUDA next (NVIDIA GPUs)
            #[cfg(feature = "candle-cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(device) => {
                        tracing_or_log("Auto-selected CUDA device 0");
                        return Ok(device);
                    }
                    Err(_) => {
                        tracing_or_log("CUDA device not available, falling back to CPU");
                    }
                }
            }

            // Fall back to CPU
            tracing_or_log("Using CPU device");
            Ok(Device::Cpu)
        }
    }
}

/// Log message (stub - could integrate with tracing crate)
fn tracing_or_log(msg: &str) {
    // For now, just use eprintln in debug builds
    #[cfg(debug_assertions)]
    eprintln!("[candle] {}", msg);
    #[cfg(not(debug_assertions))]
    let _ = msg;
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

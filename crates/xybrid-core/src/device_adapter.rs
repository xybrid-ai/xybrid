//! Device Adapter module - Collects live device metrics for routing decisions.
//!
//! The DeviceAdapter provides a platform-agnostic interface for collecting
//! device metrics such as network latency, battery level, and temperature.
//!
//! For MVP, this module provides a LocalDeviceAdapter implementation
//! for Linux and macOS that attempts to read real system metrics,
//! with sensible fallbacks when metrics cannot be determined.

use crate::context::DeviceMetrics;
use std::process::Command;
use std::time::Instant;

/// Trait for device metric collection adapters.
///
/// Implementations can collect metrics from various sources:
/// - Local device sensors (battery, temperature)
/// - Network measurements (RTT)
/// - System APIs (macOS, Linux, Windows)
pub trait DeviceAdapter {
    /// Collect current device metrics.
    ///
    /// Returns a `DeviceMetrics` struct containing:
    /// - Network round-trip time (RTT) in milliseconds
    /// - Battery level (0-100)
    /// - Device temperature in Celsius
    fn collect_metrics(&self) -> DeviceMetrics;
}

/// Default device adapter for local Linux/macOS systems.
///
/// This adapter attempts to collect real system metrics but falls back
/// to reasonable defaults when metrics cannot be determined.
///
/// # Metrics Collection
///
/// - **Network RTT**: Measures ping to localhost or default gateway
/// - **Battery**: Reads from system power management (macOS: `pmset`, Linux: `/sys/class/power_supply`)
/// - **Temperature**: Reads from system thermal sensors (macOS: `powermetrics`, Linux: `/sys/class/thermal`)
pub struct LocalDeviceAdapter;

impl LocalDeviceAdapter {
    /// Create a new LocalDeviceAdapter instance.
    pub fn new() -> Self {
        Self
    }

    /// Measure network round-trip time by pinging localhost.
    ///
    /// Falls back to a default value if ping is not available or fails.
    fn measure_network_rtt(&self) -> u32 {
        // Try to ping localhost to measure network latency
        // On Unix systems (Linux/macOS), we can use `ping` command
        #[cfg(unix)]
        {
            let start = Instant::now();
            let output = Command::new("ping")
                .arg("-c")
                .arg("1")
                .arg("-W")
                .arg("1000") // 1 second timeout
                .arg("127.0.0.1") // localhost
                .output();

            match output {
                Ok(output) => {
                    if output.status.success() {
                        // Extract RTT from ping output if available
                        // For now, use measured ping time as approximation
                        let elapsed_ms = start.elapsed().as_millis() as u32;
                        // Ping adds some overhead, so use a reasonable default
                        // In a real implementation, we'd parse the ping output
                        elapsed_ms.saturating_add(5) // Add small overhead estimate
                    } else {
                        // Ping failed, use default
                        self.default_network_rtt()
                    }
                }
                Err(_) => {
                    // ping command not available, use default
                    self.default_network_rtt()
                }
            }
        }

        #[cfg(not(unix))]
        {
            // Non-Unix platform, use default
            self.default_network_rtt()
        }
    }

    /// Default network RTT value when measurement fails.
    fn default_network_rtt(&self) -> u32 {
        // Reasonable default: 100ms for good network conditions
        100
    }

    /// Read battery level from system.
    ///
    /// Attempts to read from system power management APIs.
    /// Returns 100 (100%) if battery information is not available
    /// (e.g., desktop systems or device is plugged in).
    fn read_battery_level(&self) -> u8 {
        #[cfg(target_os = "macos")]
        {
            // On macOS, use pmset to get battery level
            let output = Command::new("pmset").arg("-g").arg("batt").output();

            if let Ok(output) = output {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    // Parse output like: " -InternalBattery-0 (id=12345) 75%; charging;"
                    if let Some(percent_start) = stdout.find('%') {
                        let line = &stdout[..percent_start];
                        if let Some(space_pos) = line.rfind(' ') {
                            let percent_str = &line[space_pos + 1..];
                            if let Ok(percent) = percent_str.parse::<u8>() {
                                return percent.min(100);
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            // On Linux, try to read from /sys/class/power_supply
            // Common paths: /sys/class/power_supply/BAT0/capacity
            let paths = vec![
                "/sys/class/power_supply/BAT0/capacity",
                "/sys/class/power_supply/BAT1/capacity",
            ];

            for path in paths {
                if let Ok(contents) = std::fs::read_to_string(path) {
                    if let Ok(percent) = contents.trim().parse::<u8>() {
                        return percent.min(100);
                    }
                }
            }
        }

        // Fallback: assume 100% (plugged in or desktop system)
        100
    }

    /// Read device temperature from system thermal sensors.
    ///
    /// Attempts to read from system thermal APIs.
    /// Returns a default temperature if sensors are not available.
    fn read_temperature(&self) -> f32 {
        #[cfg(target_os = "macos")]
        {
            // On macOS, try to use powermetrics or ioreg
            // For MVP, we'll use a default since powermetrics requires root
            // In a production implementation, we'd use proper thermal APIs
            self.default_temperature()
        }

        #[cfg(target_os = "linux")]
        {
            // On Linux, try to read from thermal sensors
            // Common paths: /sys/class/thermal/thermal_zone0/temp
            let paths = vec![
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp",
                "/sys/class/hwmon/hwmon0/temp1_input",
            ];

            for path in paths {
                if let Ok(contents) = std::fs::read_to_string(path) {
                    // Temperature is typically in millidegrees Celsius
                    if let Ok(temp_milli) = contents.trim().parse::<i32>() {
                        return (temp_milli as f32) / 1000.0;
                    }
                }
            }

            self.default_temperature()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            self.default_temperature()
        }
    }

    /// Default temperature value when measurement fails.
    fn default_temperature(&self) -> f32 {
        // Reasonable default: 25.0°C (room temperature)
        25.0
    }
}

impl Default for LocalDeviceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceAdapter for LocalDeviceAdapter {
    fn collect_metrics(&self) -> DeviceMetrics {
        DeviceMetrics {
            network_rtt: self.measure_network_rtt(),
            battery: self.read_battery_level(),
            temperature: self.read_temperature(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_device_adapter_creation() {
        let adapter = LocalDeviceAdapter::new();
        let _metrics = adapter.collect_metrics();
        // Just verify it doesn't panic and returns valid metrics
    }

    #[test]
    fn test_collect_metrics_returns_valid_values() {
        let adapter = LocalDeviceAdapter::new();
        let metrics = adapter.collect_metrics();

        // Verify metrics are within reasonable bounds
        assert!(metrics.battery <= 100, "Battery should be <= 100%");
        assert!(metrics.network_rtt > 0, "Network RTT should be > 0");
        assert!(
            metrics.temperature >= -40.0 && metrics.temperature <= 100.0,
            "Temperature should be reasonable (between -40°C and 100°C)"
        );
    }

    #[test]
    fn test_default_network_rtt() {
        let adapter = LocalDeviceAdapter::new();
        let rtt = adapter.default_network_rtt();
        assert_eq!(rtt, 100);
    }

    #[test]
    fn test_default_temperature() {
        let adapter = LocalDeviceAdapter::new();
        let temp = adapter.default_temperature();
        assert_eq!(temp, 25.0);
    }
}

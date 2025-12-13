//! Example demonstrating device metric collection using DeviceAdapter.
//!
//! This example shows how to use the LocalDeviceAdapter to collect
//! real-time device metrics (network RTT, battery, temperature).
//!
//! Run with: `cargo run --example device_metrics`

use xybrid_core::device_adapter::{DeviceAdapter, LocalDeviceAdapter};

fn main() {
    println!("📊 Device Metrics Collection Example");
    println!("{}", "=".repeat(60));
    println!();

    let adapter = LocalDeviceAdapter::new();

    println!("Collecting device metrics...");
    println!();

    let metrics = adapter.collect_metrics();

    println!("📊 Device Metrics:");
    println!("   Network RTT: {}ms", metrics.network_rtt);
    println!("   Battery: {}%", metrics.battery);
    println!("   Temperature: {:.1}°C", metrics.temperature);
    println!();

    println!("💡 Note: These are real metrics from your system.");
    println!("   If some metrics cannot be determined, defaults are used:");
    println!("   - Network RTT: 100ms (if ping unavailable)");
    println!("   - Battery: 100% (if not available, e.g., desktop)");
    println!("   - Temperature: 25.0°C (if sensors unavailable)");
}

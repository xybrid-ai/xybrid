//! Common detection functionality (cross-platform).
//!
//! This module provides memory and CPU detection using the sysinfo crate,
//! which works across all supported platforms.

use super::types::DetectionConfidence;
use sysinfo::System;

/// Memory detection result with confidence.
pub struct MemoryInfo {
    pub available_mb: u64,
    pub total_mb: u64,
    pub confidence: DetectionConfidence,
}

/// Detect available and total memory in MB using sysinfo.
///
/// Uses the cross-platform `sysinfo` crate for accurate memory detection.
pub fn detect_memory() -> MemoryInfo {
    let mut sys = System::new();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();

    if total_bytes > 0 {
        MemoryInfo {
            available_mb: available_bytes / (1024 * 1024),
            total_mb: total_bytes / (1024 * 1024),
            confidence: DetectionConfidence::High, // sysinfo provides accurate values
        }
    } else {
        // Fallback to defaults
        MemoryInfo {
            available_mb: 4096,
            total_mb: 8192,
            confidence: DetectionConfidence::Low,
        }
    }
}

/// Legacy memory detection for backwards compatibility.
/// Prefer `detect_memory()` for new code.
#[allow(dead_code)]
pub fn detect_memory_legacy() -> (u64, u64) {
    let info = detect_memory();
    (info.available_mb, info.total_mb)
}

/// CPU detection result.
pub struct CpuInfo {
    pub usage_percent: f32,
    pub cores: u32,
    pub confidence: DetectionConfidence,
}

/// Detect CPU usage and core count using sysinfo.
pub fn detect_cpu() -> CpuInfo {
    let mut sys = System::new();
    sys.refresh_cpu_all();

    // Need a brief delay to get accurate CPU usage
    // For now, we'll get core count immediately (accurate)
    // CPU usage will be approximate on first call
    let cores = sys.cpus().len() as u32;
    let usage = sys.global_cpu_usage();

    if cores > 0 {
        CpuInfo {
            usage_percent: usage,
            cores,
            confidence: if usage > 0.0 {
                DetectionConfidence::High
            } else {
                DetectionConfidence::Medium // First call may not have accurate usage
            },
        }
    } else {
        CpuInfo {
            usage_percent: 0.0,
            cores: 1,
            confidence: DetectionConfidence::Low,
        }
    }
}

/// Detects GPU/Vulkan acceleration availability.
///
/// This is a stub implementation that returns true.
/// Real implementation would check Vulkan device availability.
#[allow(dead_code)]
pub fn detect_gpu_availability() -> bool {
    // Stub: Always return true for now
    // TODO: Real implementation would check:
    // - Vulkan device availability
    // - GPU compute shader support
    true
}

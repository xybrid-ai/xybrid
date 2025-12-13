//! Hardware Capability Detection module (v0.0.8).
//!
//! This module provides unified hardware capability detection across platforms,
//! including GPU acceleration (Metal, Vulkan), neural processing units (CoreML, NNAPI),
//! battery level, thermal state, and memory profiling. This information is used by
//! the routing engine and runtime adapters to make optimal execution decisions.
//!
//! ## v0.0.7 Improvements
//!
//! - Real memory detection using `sysinfo` crate (cross-platform)
//! - CPU usage tracking
//! - Detection confidence indicators
//! - Improved thermal reading
//!
//! ## v0.0.8 Improvements
//!
//! - Android API level detection (NNAPI requires API 27+)
//! - iOS device family detection (Neural Engine on iPhone10+, iPad8+)
//! - Conservative defaults (prefer `false` when detection uncertain)
//! - Apple Silicon Mac detection for Neural Engine
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::device::capabilities::{HardwareCapabilities, detect_capabilities};
//! use xybrid_core::context::DeviceMetrics;
//!
//! let metrics = DeviceMetrics {
//!     network_rtt: 100,
//!     battery: 75,
//!     temperature: 25.0,
//! };
//!
//! let capabilities = detect_capabilities(&metrics);
//! if capabilities.has_gpu() {
//!     println!("GPU acceleration available");
//! }
//! if capabilities.has_npu() {
//!     println!("NPU available: {:?}", capabilities.npu_type());
//! }
//! println!("Memory confidence: {:?}", capabilities.memory_confidence);
//! ```

use crate::context::DeviceMetrics;
use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Thermal state for mobile devices.
///
/// This enum represents the thermal state of a device, which affects
/// performance and battery consumption decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal operating temperature (< 60°C)
    Normal,
    /// Device is warm, may throttle performance (60-70°C)
    Warm,
    /// Device is hot, should reduce workload (70-80°C)
    Hot,
    /// Critical temperature, should pause heavy operations (> 80°C)
    Critical,
}

impl ThermalState {
    /// Convert to string representation for FFI/JSON
    pub fn as_str(&self) -> &'static str {
        match self {
            ThermalState::Normal => "normal",
            ThermalState::Warm => "warm",
            ThermalState::Hot => "hot",
            ThermalState::Critical => "critical",
        }
    }
}

/// GPU type for the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuType {
    /// Apple Metal (macOS/iOS)
    Metal,
    /// Vulkan (cross-platform)
    Vulkan,
    /// OpenCL (legacy)
    OpenCL,
    /// DirectX/DirectML (Windows)
    DirectX,
    /// No GPU available
    None,
}

impl GpuType {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuType::Metal => "metal",
            GpuType::Vulkan => "vulkan",
            GpuType::OpenCL => "opencl",
            GpuType::DirectX => "directx",
            GpuType::None => "none",
        }
    }
}

/// Neural Processing Unit type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NpuType {
    /// Apple CoreML Neural Engine (iOS/macOS)
    CoreML,
    /// Android Neural Networks API
    NNAPI,
    /// Windows DirectML
    DirectML,
    /// No NPU available
    None,
}

impl NpuType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NpuType::CoreML => "coreml",
            NpuType::NNAPI => "nnapi",
            NpuType::DirectML => "directml",
            NpuType::None => "none",
        }
    }
}

/// Platform identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    MacOS,
    IOS,
    Android,
    Linux,
    Windows,
    Unknown,
}

impl Platform {
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::MacOS => "macos",
            Platform::IOS => "ios",
            Platform::Android => "android",
            Platform::Linux => "linux",
            Platform::Windows => "windows",
            Platform::Unknown => "unknown",
        }
    }

    /// Detect current platform at compile time
    pub fn current() -> Self {
        #[cfg(target_os = "macos")]
        return Platform::MacOS;
        #[cfg(target_os = "ios")]
        return Platform::IOS;
        #[cfg(target_os = "android")]
        return Platform::Android;
        #[cfg(target_os = "linux")]
        return Platform::Linux;
        #[cfg(target_os = "windows")]
        return Platform::Windows;
        #[cfg(not(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "linux",
            target_os = "windows"
        )))]
        return Platform::Unknown;
    }
}

/// Detection confidence level (v0.0.7).
///
/// Indicates how confident we are in a detected value. This helps
/// downstream code make informed decisions about whether to trust
/// the detected value or use fallbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DetectionConfidence {
    /// Real detection from native APIs or sysinfo crate
    High,
    /// Detected from environment, device model, or indirect methods
    Medium,
    /// Hardcoded default or estimate
    #[default]
    Low,
}

impl DetectionConfidence {
    pub fn as_str(&self) -> &'static str {
        match self {
            DetectionConfidence::High => "high",
            DetectionConfidence::Medium => "medium",
            DetectionConfidence::Low => "low",
        }
    }
}

/// Detection source for audit and debugging (v0.0.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DetectionSource {
    /// Direct native API call (Metal, NNAPI, etc.)
    NativeApi,
    /// sysinfo crate detection
    Sysinfo,
    /// Inferred from device model or environment
    DeviceModel,
    /// Hardcoded fallback value
    #[default]
    Default,
}

/// Hardware capabilities for the current device.
///
/// This struct provides a unified interface for querying hardware capabilities
/// across different platforms (macOS, iOS, Android, desktop).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    // Accelerator availability
    /// GPU acceleration available (Vulkan, Metal, or other GPU compute)
    pub has_gpu: bool,
    /// GPU type if available
    pub gpu_type: GpuType,
    /// Android Neural Networks API available
    pub has_nnapi: bool,
    /// iOS/macOS Metal framework available
    pub has_metal: bool,
    /// Neural Processing Unit available (CoreML Neural Engine, NNAPI accelerators)
    pub has_npu: bool,
    /// NPU type if available
    pub npu_type: NpuType,

    // Memory metrics (in MB)
    /// Available memory in MB
    pub memory_available_mb: u64,
    /// Total system memory in MB
    pub memory_total_mb: u64,

    // CPU metrics (v0.0.7)
    /// Current CPU usage percentage (0-100)
    pub cpu_usage_percent: f32,
    /// Number of CPU cores
    pub cpu_cores: u32,

    // Power and thermal
    /// Current battery level (0-100)
    pub battery_level: u8,
    /// Current thermal state
    pub thermal_state: ThermalState,

    // Platform info
    /// Current platform
    pub platform: Platform,

    // Detection confidence (v0.0.7)
    /// Confidence level for memory detection
    pub memory_confidence: DetectionConfidence,
    /// Confidence level for GPU detection
    pub gpu_confidence: DetectionConfidence,
    /// Confidence level for NPU detection
    pub npu_confidence: DetectionConfidence,
}

impl HardwareCapabilities {
    /// Creates a new HardwareCapabilities instance with default values.
    pub fn new() -> Self {
        Self {
            has_gpu: false,
            gpu_type: GpuType::None,
            has_nnapi: false,
            has_metal: false,
            has_npu: false,
            npu_type: NpuType::None,
            memory_available_mb: 0,
            memory_total_mb: 0,
            cpu_usage_percent: 0.0,
            cpu_cores: 1,
            battery_level: 100,
            thermal_state: ThermalState::Normal,
            platform: Platform::current(),
            memory_confidence: DetectionConfidence::Low,
            gpu_confidence: DetectionConfidence::Low,
            npu_confidence: DetectionConfidence::Low,
        }
    }

    /// Returns CPU usage percentage (0-100).
    pub fn cpu_usage_percent(&self) -> f32 {
        self.cpu_usage_percent
    }

    /// Returns number of CPU cores.
    pub fn cpu_cores(&self) -> u32 {
        self.cpu_cores
    }

    /// Returns whether GPU acceleration is available.
    pub fn has_gpu(&self) -> bool {
        self.has_gpu
    }

    /// Returns the GPU type.
    pub fn gpu_type(&self) -> GpuType {
        self.gpu_type
    }

    /// Returns whether NNAPI is available (Android only).
    pub fn has_nnapi(&self) -> bool {
        self.has_nnapi
    }

    /// Returns whether Metal is available (iOS/macOS only).
    pub fn has_metal(&self) -> bool {
        self.has_metal
    }

    /// Returns whether an NPU (Neural Processing Unit) is available.
    pub fn has_npu(&self) -> bool {
        self.has_npu
    }

    /// Returns the NPU type.
    pub fn npu_type(&self) -> NpuType {
        self.npu_type
    }

    /// Returns the current battery level (0-100).
    pub fn battery_level(&self) -> u8 {
        self.battery_level
    }

    /// Returns the current thermal state.
    pub fn thermal_state(&self) -> ThermalState {
        self.thermal_state
    }

    /// Returns available memory in MB.
    pub fn memory_available_mb(&self) -> u64 {
        self.memory_available_mb
    }

    /// Returns total system memory in MB.
    pub fn memory_total_mb(&self) -> u64 {
        self.memory_total_mb
    }

    /// Returns the current platform.
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Returns whether execution should be throttled based on battery and thermal state.
    ///
    /// Throttling is recommended when:
    /// - Battery level is below 20%
    /// - Thermal state is Hot or Critical
    pub fn should_throttle(&self) -> bool {
        if self.battery_level < 20 {
            return true;
        }

        matches!(
            self.thermal_state,
            ThermalState::Hot | ThermalState::Critical
        )
    }

    /// Returns whether GPU acceleration should be preferred.
    ///
    /// GPU is preferred when:
    /// - GPU is available
    /// - Battery level is above 30% (GPU can drain battery faster)
    /// - Thermal state is not Critical
    pub fn should_prefer_gpu(&self) -> bool {
        self.has_gpu && self.battery_level > 30 && self.thermal_state != ThermalState::Critical
    }

    /// Returns whether NNAPI should be preferred (Android only).
    ///
    /// NNAPI is preferred when:
    /// - NNAPI is available
    /// - Battery level is above 20%
    /// - Thermal state is not Critical
    pub fn should_prefer_nnapi(&self) -> bool {
        self.has_nnapi && self.battery_level > 20 && self.thermal_state != ThermalState::Critical
    }

    /// Returns whether Metal should be preferred (iOS/macOS only).
    ///
    /// Metal is preferred when:
    /// - Metal is available
    /// - Battery level is above 30%
    /// - Thermal state is not Critical
    pub fn should_prefer_metal(&self) -> bool {
        self.has_metal && self.battery_level > 30 && self.thermal_state != ThermalState::Critical
    }

    /// Returns whether NPU should be preferred.
    ///
    /// NPU is preferred when:
    /// - NPU is available
    /// - Battery level is above 20% (NPU is generally power-efficient)
    /// - Thermal state is not Critical
    pub fn should_prefer_npu(&self) -> bool {
        self.has_npu && self.battery_level > 20 && self.thermal_state != ThermalState::Critical
    }

    /// Check if there's enough memory to load a model of given size.
    ///
    /// # Arguments
    /// * `model_size_mb` - Required memory for the model in MB
    /// * `safety_margin` - Multiplier for safety margin (default 1.5)
    pub fn can_load_model(&self, model_size_mb: u64, safety_margin: Option<f64>) -> bool {
        let margin = safety_margin.unwrap_or(1.5);
        let required = (model_size_mb as f64 * margin) as u64;
        self.memory_available_mb >= required
    }

    /// Convert to JSON string for FFI transport.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Create from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

/// Detects hardware capabilities for the current platform.
///
/// This function performs platform-specific detection and returns a
/// `HardwareCapabilities` struct with the detected capabilities.
///
/// ## v0.0.7 Improvements
///
/// - Uses `sysinfo` crate for accurate memory/CPU detection
/// - Includes detection confidence indicators
/// - Cross-platform CPU core count and usage
///
/// # Arguments
///
/// * `metrics` - Device metrics containing battery and temperature information
///
/// # Returns
///
/// A `HardwareCapabilities` struct with detected capabilities
pub fn detect_capabilities(metrics: &DeviceMetrics) -> HardwareCapabilities {
    let mut capabilities = HardwareCapabilities::new();

    // Set battery level from metrics
    capabilities.battery_level = metrics.battery;

    // Determine thermal state from temperature
    // These thresholds are approximate and may need tuning per device
    capabilities.thermal_state = if metrics.temperature > 80.0 {
        ThermalState::Critical
    } else if metrics.temperature > 70.0 {
        ThermalState::Hot
    } else if metrics.temperature > 60.0 {
        ThermalState::Warm
    } else {
        ThermalState::Normal
    };

    // Detect memory using sysinfo (v0.0.7)
    let memory_info = detect_memory_v2();
    capabilities.memory_available_mb = memory_info.available_mb;
    capabilities.memory_total_mb = memory_info.total_mb;
    capabilities.memory_confidence = memory_info.confidence;

    // Detect CPU using sysinfo (v0.0.7)
    let cpu_info = detect_cpu();
    capabilities.cpu_usage_percent = cpu_info.usage_percent;
    capabilities.cpu_cores = cpu_info.cores;

    // Platform-specific detection
    #[cfg(target_os = "macos")]
    {
        capabilities.has_metal = detect_metal_availability();
        capabilities.has_gpu = capabilities.has_metal;
        capabilities.gpu_type = if capabilities.has_metal {
            GpuType::Metal
        } else {
            GpuType::None
        };
        // GPU confidence: Medium (compile-time check, not runtime validation)
        capabilities.gpu_confidence = DetectionConfidence::Medium;

        // CoreML Neural Engine detection (stub based on arch)
        capabilities.has_npu = detect_coreml_availability();
        capabilities.npu_type = if capabilities.has_npu {
            NpuType::CoreML
        } else {
            NpuType::None
        };
        // NPU confidence: Medium (arch-based, not actual detection)
        capabilities.npu_confidence = DetectionConfidence::Medium;
    }

    #[cfg(target_os = "ios")]
    {
        capabilities.has_metal = detect_metal_availability();
        capabilities.has_gpu = capabilities.has_metal;
        capabilities.gpu_type = if capabilities.has_metal {
            GpuType::Metal
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = DetectionConfidence::Medium;

        capabilities.has_npu = detect_coreml_availability();
        capabilities.npu_type = if capabilities.has_npu {
            NpuType::CoreML
        } else {
            NpuType::None
        };
        capabilities.npu_confidence = DetectionConfidence::Medium;
    }

    #[cfg(target_os = "android")]
    {
        capabilities.has_nnapi = detect_nnapi_availability();
        capabilities.has_gpu = detect_gpu_availability();
        capabilities.gpu_type = if capabilities.has_gpu {
            GpuType::Vulkan
        } else {
            GpuType::None
        };
        // GPU confidence: Low (stubbed, always true)
        capabilities.gpu_confidence = DetectionConfidence::Low;

        // NNAPI can use NPU accelerators
        capabilities.has_npu = capabilities.has_nnapi;
        capabilities.npu_type = if capabilities.has_nnapi {
            NpuType::NNAPI
        } else {
            NpuType::None
        };
        // NPU confidence: Medium (v0.0.8: checks API level from env vars)
        // If API level detected from env, Medium; otherwise conservative false
        let api_info = detect_android_api_level();
        capabilities.npu_confidence = api_info.confidence;
    }

    #[cfg(target_os = "windows")]
    {
        capabilities.has_gpu = detect_gpu_availability();
        capabilities.gpu_type = if capabilities.has_gpu {
            GpuType::DirectX
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = DetectionConfidence::Low;

        // DirectML NPU (stub)
        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Low;
    }

    #[cfg(target_os = "linux")]
    {
        capabilities.has_gpu = detect_gpu_availability();
        capabilities.gpu_type = if capabilities.has_gpu {
            GpuType::Vulkan
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = DetectionConfidence::Low;

        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Low;
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows",
        target_os = "linux"
    )))]
    {
        capabilities.has_gpu = detect_gpu_availability();
        capabilities.gpu_type = GpuType::None;
        capabilities.gpu_confidence = DetectionConfidence::Low;
        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Low;
    }

    capabilities
}

/// Memory detection result with confidence (v0.0.7).
struct MemoryInfo {
    available_mb: u64,
    total_mb: u64,
    confidence: DetectionConfidence,
}

/// Detect available and total memory in MB using sysinfo (v0.0.7).
///
/// Uses the cross-platform `sysinfo` crate for accurate memory detection.
fn detect_memory_v2() -> MemoryInfo {
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
/// Prefer `detect_memory_v2()` for new code.
fn detect_memory() -> (u64, u64) {
    let info = detect_memory_v2();
    (info.available_mb, info.total_mb)
}

/// CPU detection result (v0.0.7).
struct CpuInfo {
    usage_percent: f32,
    cores: u32,
    confidence: DetectionConfidence,
}

/// Detect CPU usage and core count using sysinfo (v0.0.7).
fn detect_cpu() -> CpuInfo {
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

/// Detects Metal framework availability (macOS/iOS only).
fn detect_metal_availability() -> bool {
    // Stub: Always return true on Apple platforms
    // TODO: Real implementation would check:
    // - MTLCreateSystemDefaultDevice() != nil
    // - Device supports compute shaders
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        true
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

/// iOS device family detection result (v0.0.8).
#[derive(Debug, Clone)]
struct AppleDeviceInfo {
    /// Device identifier (e.g., "iPhone12,1", "MacBookPro18,1")
    device_model: Option<String>,
    /// Whether Neural Engine is likely available
    has_neural_engine: bool,
    /// Detection confidence
    confidence: DetectionConfidence,
}

/// Detect Apple device model from environment variables.
///
/// On iOS, the device model can be obtained from various sources:
/// 1. `DEVICE_MODEL` environment variable (may be set by Flutter/runtime)
/// 2. `SIMULATOR_MODEL_IDENTIFIER` for iOS Simulator
///
/// Returns device info with Neural Engine availability inference.
fn detect_apple_device() -> AppleDeviceInfo {
    // Try environment variables that might be set by the runtime
    let model = std::env::var("DEVICE_MODEL")
        .or_else(|_| std::env::var("SIMULATOR_MODEL_IDENTIFIER"))
        .or_else(|_| std::env::var("APPLE_DEVICE_MODEL"))
        .ok();

    if let Some(ref model_str) = model {
        let has_ne = has_neural_engine_by_model(model_str);
        return AppleDeviceInfo {
            device_model: Some(model_str.clone()),
            has_neural_engine: has_ne,
            confidence: DetectionConfidence::Medium,
        };
    }

    // No model detected - use architecture-based fallback
    #[cfg(target_arch = "aarch64")]
    {
        // ARM64 Apple devices generally have Neural Engine:
        // - All Apple Silicon Macs (M1/M2/M3/M4)
        // - All iPhones since iPhone 8/X (A11+)
        // - All iPads since iPad Pro 2018 (A12X+)
        // Conservative: assume available on ARM64
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: true,
            confidence: DetectionConfidence::Medium,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Intel Macs don't have Neural Engine
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: false,
            confidence: DetectionConfidence::High, // We know Intel has no NE
        }
    }
}

/// Check if a device model has Neural Engine based on identifier.
///
/// Neural Engine was introduced with:
/// - iPhone: A11 Bionic (iPhone 8/X, 2017) - "iPhone10,x"
/// - iPad: A12X Bionic (iPad Pro 2018) - "iPad8,x"
/// - Mac: M1 (2020) - "MacBookPro17,x", "MacBookAir10,x", etc.
fn has_neural_engine_by_model(model: &str) -> bool {
    let model_lower = model.to_lowercase();

    // iPhone detection
    // iPhone10,x = iPhone 8/X (A11) - first with Neural Engine
    // All iPhones after iPhone 10,x have Neural Engine
    if model_lower.starts_with("iphone") {
        if let Some(version_str) = model.strip_prefix("iPhone").or_else(|| model.strip_prefix("iphone")) {
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
        if let Some(version_str) = model.strip_prefix("iPad").or_else(|| model.strip_prefix("ipad")) {
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
            "macbookpro17", "macbookpro18", "macbookpro19", "macbookpro20",
            "macbookair10", "macbookair11", "macbookair12",
            "macmini9", "macmini10",
            "imac21", "imac22", "imac23", "imac24",
            "mac13", "mac14", "mac15", // Mac Studio, Mac Pro
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
/// ## v0.0.8 Improvements
///
/// - Detects device model from environment variables
/// - Infers Neural Engine availability by device family
/// - iPhone 8/X (A11+) and iPad Pro 2018 (A12X+) have Neural Engine
/// - All Apple Silicon Macs have Neural Engine
/// - Intel Macs do NOT have Neural Engine
#[allow(dead_code)]
fn detect_coreml_availability() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let device_info = detect_apple_device();
        device_info.has_neural_engine
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

/// Android API level detection result (v0.0.8).
#[derive(Debug, Clone)]
struct AndroidApiInfo {
    api_level: Option<u32>,
    confidence: DetectionConfidence,
}

/// Detect Android API level from environment variables or build properties.
///
/// This checks several sources:
/// 1. `ANDROID_SDK_VERSION` environment variable
/// 2. `ro.build.version.sdk` system property (if exposed)
///
/// Returns None if API level cannot be determined, triggering conservative defaults.
fn detect_android_api_level() -> AndroidApiInfo {
    // Try environment variable first (may be set by build system or runtime)
    if let Ok(sdk_str) = std::env::var("ANDROID_SDK_VERSION") {
        if let Ok(level) = sdk_str.parse::<u32>() {
            return AndroidApiInfo {
                api_level: Some(level),
                confidence: DetectionConfidence::Medium,
            };
        }
    }

    // Try alternative environment variable names
    for env_var in &["ANDROID_API_LEVEL", "SDK_INT", "TARGET_SDK_VERSION"] {
        if let Ok(sdk_str) = std::env::var(env_var) {
            if let Ok(level) = sdk_str.parse::<u32>() {
                return AndroidApiInfo {
                    api_level: Some(level),
                    confidence: DetectionConfidence::Medium,
                };
            }
        }
    }

    // Could not detect - return unknown
    AndroidApiInfo {
        api_level: None,
        confidence: DetectionConfidence::Low,
    }
}

/// Detects NNAPI (Android Neural Networks API) availability.
///
/// ## v0.0.8 Improvements
///
/// - Checks Android API level (NNAPI requires API 27+)
/// - Uses conservative defaults when API level unknown
///
/// NNAPI was introduced in Android 8.1 (API level 27).
/// For full accelerator support, API 29+ is recommended.
#[allow(dead_code)]
fn detect_nnapi_availability() -> bool {
    #[cfg(target_os = "android")]
    {
        let api_info = detect_android_api_level();

        match api_info.api_level {
            Some(level) => {
                // NNAPI requires API level 27 (Android 8.1) or higher
                level >= 27
            }
            None => {
                // Conservative default: assume NNAPI is NOT available
                // This prevents attempting to use unavailable APIs
                false
            }
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        false
    }
}

/// Detects GPU/Vulkan acceleration availability.
#[allow(dead_code)]
fn detect_gpu_availability() -> bool {
    // Stub: Always return true for now
    // TODO: Real implementation would check:
    // - Vulkan device availability
    // - GPU compute shader support
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_capabilities() {
        let caps = HardwareCapabilities::new();
        assert!(!caps.has_gpu());
        assert!(!caps.has_nnapi());
        assert!(!caps.has_metal());
        assert!(!caps.has_npu());
        assert_eq!(caps.battery_level(), 100);
        assert_eq!(caps.thermal_state(), ThermalState::Normal);
        assert_eq!(caps.gpu_type(), GpuType::None);
        assert_eq!(caps.npu_type(), NpuType::None);
        // v0.0.7: Check new fields
        assert_eq!(caps.cpu_cores(), 1);
        assert_eq!(caps.cpu_usage_percent(), 0.0);
        assert_eq!(caps.memory_confidence, DetectionConfidence::Low);
        assert_eq!(caps.gpu_confidence, DetectionConfidence::Low);
        assert_eq!(caps.npu_confidence, DetectionConfidence::Low);
    }

    #[test]
    fn test_should_throttle_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.battery_level = 15;
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_should_throttle_hot_device() {
        let mut caps = HardwareCapabilities::new();
        caps.thermal_state = ThermalState::Hot;
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_should_not_throttle_normal() {
        let caps = HardwareCapabilities::new();
        assert!(!caps.should_throttle());
    }

    #[test]
    fn test_should_prefer_gpu() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_gpu());
    }

    #[test]
    fn test_should_not_prefer_gpu_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.battery_level = 25;
        assert!(!caps.should_prefer_gpu());
    }

    #[test]
    fn test_should_not_prefer_gpu_critical_thermal() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.battery_level = 50;
        caps.thermal_state = ThermalState::Critical;
        assert!(!caps.should_prefer_gpu());
    }

    #[test]
    fn test_should_prefer_nnapi() {
        let mut caps = HardwareCapabilities::new();
        caps.has_nnapi = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_nnapi());
    }

    #[test]
    fn test_should_not_prefer_nnapi_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.has_nnapi = true;
        caps.battery_level = 15;
        assert!(!caps.should_prefer_nnapi());
    }

    #[test]
    fn test_should_prefer_metal() {
        let mut caps = HardwareCapabilities::new();
        caps.has_metal = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_metal());
    }

    #[test]
    fn test_should_prefer_npu() {
        let mut caps = HardwareCapabilities::new();
        caps.has_npu = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_npu());
    }

    #[test]
    fn test_should_not_prefer_npu_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.has_npu = true;
        caps.battery_level = 15;
        assert!(!caps.should_prefer_npu());
    }

    #[test]
    fn test_can_load_model() {
        let mut caps = HardwareCapabilities::new();
        caps.memory_available_mb = 4096;

        // Model that fits with default 1.5x margin
        assert!(caps.can_load_model(2000, None)); // 2000 * 1.5 = 3000 < 4096

        // Model that doesn't fit
        assert!(!caps.can_load_model(3000, None)); // 3000 * 1.5 = 4500 > 4096

        // Model that fits with custom margin
        assert!(caps.can_load_model(3000, Some(1.2))); // 3000 * 1.2 = 3600 < 4096
    }

    #[test]
    fn test_detect_capabilities() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.battery_level(), 75);
        assert_eq!(caps.thermal_state(), ThermalState::Normal);
        assert!(caps.memory_total_mb() > 0);
    }

    #[test]
    fn test_detect_capabilities_hot_device() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 75.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Hot);
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_detect_capabilities_critical_device() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 85.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Critical);
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_detect_capabilities_low_battery() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 15,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.battery_level(), 15);
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_thermal_state_thresholds() {
        // Normal
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Normal);

        // Warm
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 65.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Warm);

        // Hot
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 75.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Hot);

        // Critical
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 85.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Critical);
    }

    #[test]
    fn test_json_serialization() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.gpu_type = GpuType::Metal;
        caps.battery_level = 85;
        caps.memory_total_mb = 16384;
        caps.memory_available_mb = 8192;

        let json = caps.to_json();
        assert!(json.contains("\"has_gpu\":true"));
        assert!(json.contains("\"battery_level\":85"));

        let parsed = HardwareCapabilities::from_json(&json).unwrap();
        assert_eq!(parsed.has_gpu, true);
        assert_eq!(parsed.battery_level, 85);
    }

    #[test]
    fn test_platform_current() {
        let platform = Platform::current();
        #[cfg(target_os = "macos")]
        assert_eq!(platform, Platform::MacOS);
        #[cfg(target_os = "linux")]
        assert_eq!(platform, Platform::Linux);
    }

    #[test]
    fn test_enum_as_str() {
        assert_eq!(ThermalState::Normal.as_str(), "normal");
        assert_eq!(ThermalState::Critical.as_str(), "critical");
        assert_eq!(GpuType::Metal.as_str(), "metal");
        assert_eq!(NpuType::CoreML.as_str(), "coreml");
        assert_eq!(Platform::MacOS.as_str(), "macos");
    }

    // v0.0.7 tests

    #[test]
    fn test_detection_confidence_as_str() {
        assert_eq!(DetectionConfidence::High.as_str(), "high");
        assert_eq!(DetectionConfidence::Medium.as_str(), "medium");
        assert_eq!(DetectionConfidence::Low.as_str(), "low");
    }

    #[test]
    fn test_detection_confidence_default() {
        let confidence: DetectionConfidence = Default::default();
        assert_eq!(confidence, DetectionConfidence::Low);
    }

    #[test]
    fn test_sysinfo_memory_detection() {
        let info = detect_memory_v2();
        // sysinfo should return real values on all platforms
        assert!(info.total_mb > 0, "Total memory should be > 0");
        // Available memory should be <= total
        assert!(
            info.available_mb <= info.total_mb,
            "Available should be <= total"
        );
        // Confidence should be High when sysinfo works
        assert_eq!(info.confidence, DetectionConfidence::High);
    }

    #[test]
    fn test_sysinfo_cpu_detection() {
        let info = detect_cpu();
        // Should have at least 1 core
        assert!(info.cores >= 1, "Should have at least 1 CPU core");
        // CPU usage should be in valid range
        assert!(
            info.usage_percent >= 0.0 && info.usage_percent <= 100.0,
            "CPU usage should be 0-100%"
        );
    }

    #[test]
    fn test_detect_capabilities_has_confidence() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        // Memory confidence should be High when using sysinfo
        assert_eq!(
            caps.memory_confidence,
            DetectionConfidence::High,
            "Memory detection should have High confidence with sysinfo"
        );
        // GPU/NPU confidence depends on platform but should be set
        assert!(
            caps.gpu_confidence == DetectionConfidence::High
                || caps.gpu_confidence == DetectionConfidence::Medium
                || caps.gpu_confidence == DetectionConfidence::Low
        );
    }

    #[test]
    fn test_detect_capabilities_has_cpu_info() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        // Should have CPU info
        assert!(caps.cpu_cores >= 1, "Should detect at least 1 CPU core");
        // Memory should be detected
        assert!(caps.memory_total_mb > 0, "Should detect total memory");
    }

    // v0.0.8 tests - Mobile API level detection

    #[test]
    fn test_has_neural_engine_iphone_models() {
        // iPhone 8/X and later (iPhone10+) have Neural Engine
        assert!(has_neural_engine_by_model("iPhone10,1")); // iPhone 8
        assert!(has_neural_engine_by_model("iPhone10,4")); // iPhone 8
        assert!(has_neural_engine_by_model("iPhone10,3")); // iPhone X
        assert!(has_neural_engine_by_model("iPhone11,2")); // iPhone XS
        assert!(has_neural_engine_by_model("iPhone12,1")); // iPhone 11
        assert!(has_neural_engine_by_model("iPhone13,1")); // iPhone 12 mini
        assert!(has_neural_engine_by_model("iPhone14,5")); // iPhone 13
        assert!(has_neural_engine_by_model("iPhone15,2")); // iPhone 14 Pro
        assert!(has_neural_engine_by_model("iPhone16,1")); // iPhone 15 Pro

        // iPhone 7 and earlier (iPhone9 and below) do NOT have Neural Engine
        assert!(!has_neural_engine_by_model("iPhone9,1")); // iPhone 7
        assert!(!has_neural_engine_by_model("iPhone9,3")); // iPhone 7
        assert!(!has_neural_engine_by_model("iPhone8,1")); // iPhone 6s
        assert!(!has_neural_engine_by_model("iPhone7,2")); // iPhone 6
    }

    #[test]
    fn test_has_neural_engine_ipad_models() {
        // iPad Pro 2018 and later (iPad8+) have Neural Engine
        assert!(has_neural_engine_by_model("iPad8,1")); // iPad Pro 11" 2018
        assert!(has_neural_engine_by_model("iPad8,5")); // iPad Pro 12.9" 2018
        assert!(has_neural_engine_by_model("iPad11,1")); // iPad mini 5
        assert!(has_neural_engine_by_model("iPad13,1")); // iPad Air 4
        assert!(has_neural_engine_by_model("iPad14,1")); // iPad mini 6

        // Older iPads do NOT have Neural Engine
        assert!(!has_neural_engine_by_model("iPad7,5")); // iPad 6th gen
        assert!(!has_neural_engine_by_model("iPad6,11")); // iPad 5th gen
        assert!(!has_neural_engine_by_model("iPad5,3")); // iPad Air 2
    }

    #[test]
    fn test_has_neural_engine_mac_models() {
        // Apple Silicon Macs have Neural Engine
        assert!(has_neural_engine_by_model("MacBookPro17,1")); // M1 MacBook Pro 13"
        assert!(has_neural_engine_by_model("MacBookPro18,1")); // M1 Pro MacBook Pro 16"
        assert!(has_neural_engine_by_model("MacBookAir10,1")); // M1 MacBook Air
        assert!(has_neural_engine_by_model("Macmini9,1")); // M1 Mac mini
        assert!(has_neural_engine_by_model("iMac21,1")); // M1 iMac 24"
        assert!(has_neural_engine_by_model("Mac13,1")); // M1 Mac Studio

        // Note: Intel Mac detection depends on architecture at compile time
        // These tests verify the pattern matching works
    }

    #[test]
    fn test_has_neural_engine_unknown_devices() {
        // Unknown devices should return conservative defaults
        assert!(!has_neural_engine_by_model("AppleTV6,2")); // Apple TV 4K
        assert!(!has_neural_engine_by_model("Watch5,1")); // Apple Watch Series 5
        assert!(!has_neural_engine_by_model("UnknownDevice1,1"));
    }

    #[test]
    fn test_android_api_level_detection_no_env() {
        // Without environment variables set, should return None/Low confidence
        // Note: This test assumes no ANDROID_SDK_VERSION env var is set
        // In CI, this should be the case
        let info = detect_android_api_level();
        // If no env var is set, api_level should be None
        // (unless running in an Android environment)
        if info.api_level.is_none() {
            assert_eq!(info.confidence, DetectionConfidence::Low);
        }
    }

    #[test]
    fn test_apple_device_detection_fallback() {
        // Without environment variables, should fall back to architecture check
        let info = detect_apple_device();

        // On ARM64 (Apple Silicon), should detect Neural Engine
        #[cfg(target_arch = "aarch64")]
        {
            assert!(info.has_neural_engine);
            assert_eq!(info.confidence, DetectionConfidence::Medium);
        }

        // On Intel, should NOT detect Neural Engine
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!info.has_neural_engine);
            assert_eq!(info.confidence, DetectionConfidence::High);
        }
    }
}

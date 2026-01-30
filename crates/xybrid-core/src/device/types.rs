//! Device capability types and enums.
//!
//! This module contains all the data types for hardware capability detection,
//! including thermal state, GPU types, NPU types, and the main HardwareCapabilities struct.

use serde::{Deserialize, Serialize};

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

/// Detection confidence level.
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

/// Detection source for audit and debugging.
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

    // CPU metrics
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

    // Detection confidence
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

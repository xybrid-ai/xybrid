//! Execution Provider configuration for ONNX Runtime.
//!
//! This module defines execution provider types and configurations for hardware
//! acceleration in ONNX Runtime inference. Execution providers allow models to
//! run on different hardware backends (CPU, GPU, Neural Engine, etc.).
//!
//! # Supported Providers
//!
//! | Provider | Platform | Hardware | Feature Flag |
//! |----------|----------|----------|--------------|
//! | CPU | All | CPU | (default) |
//! | CoreML | macOS/iOS | Neural Engine, GPU, CPU | `coreml-ep` |
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::runtime_adapter::onnx::{ExecutionProviderKind, CoreMLConfig, CoreMLComputeUnits};
//!
//! // Use CPU (default)
//! let cpu_provider = ExecutionProviderKind::Cpu;
//!
//! // Use CoreML with Neural Engine
//! let coreml_provider = ExecutionProviderKind::CoreML(CoreMLConfig {
//!     compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
//!     ..Default::default()
//! });
//! ```

use std::fmt;

/// Execution provider selection for ONNX Runtime.
///
/// Determines which hardware backend to use for model inference.
/// If the selected provider is unavailable, ONNX Runtime automatically
/// falls back to CPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProviderKind {
    /// CPU execution (default, always available)
    Cpu,

    /// CoreML execution provider (macOS/iOS only)
    ///
    /// Enables hardware acceleration via Apple's CoreML framework,
    /// which can utilize the Neural Engine, GPU, or CPU depending
    /// on the configured compute units.
    ///
    /// Requires the `coreml-ep` feature flag.
    #[cfg(feature = "coreml-ep")]
    CoreML(CoreMLConfig),
}

impl Default for ExecutionProviderKind {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for ExecutionProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            #[cfg(feature = "coreml-ep")]
            Self::CoreML(config) => write!(f, "coreml-{}", config.compute_units),
        }
    }
}

impl ExecutionProviderKind {
    /// Returns the name of this execution provider.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "coreml-ep")]
            Self::CoreML(_) => "coreml",
        }
    }

    /// Returns whether this provider requires specific hardware.
    pub fn requires_hardware(&self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(feature = "coreml-ep")]
            Self::CoreML(_) => true,
        }
    }
}

/// CoreML execution provider configuration.
///
/// Controls how CoreML executes model inference, including which
/// compute units (CPU, GPU, Neural Engine) to use.
#[cfg(feature = "coreml-ep")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreMLConfig {
    /// Which compute units to use for inference.
    pub compute_units: CoreMLComputeUnits,

    /// Whether to enable CoreML on subgraphs (control flow operators).
    ///
    /// When enabled, CoreML can accelerate parts of models that contain
    /// control flow operations like If, Loop, etc.
    pub use_subgraphs: bool,

    /// Whether to require static input shapes.
    ///
    /// When enabled, models must have fixed input dimensions.
    /// This can improve performance but reduces flexibility.
    pub require_static_shapes: bool,
}

#[cfg(feature = "coreml-ep")]
impl Default for CoreMLConfig {
    fn default() -> Self {
        Self {
            compute_units: CoreMLComputeUnits::default(),
            use_subgraphs: true,
            require_static_shapes: false,
        }
    }
}

#[cfg(feature = "coreml-ep")]
impl CoreMLConfig {
    /// Creates a new CoreML configuration with Neural Engine acceleration.
    pub fn with_neural_engine() -> Self {
        Self {
            compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
            ..Default::default()
        }
    }

    /// Creates a new CoreML configuration with GPU acceleration.
    pub fn with_gpu() -> Self {
        Self {
            compute_units: CoreMLComputeUnits::CpuAndGpu,
            ..Default::default()
        }
    }

    /// Creates a new CoreML configuration using CPU only (for testing/comparison).
    pub fn cpu_only() -> Self {
        Self {
            compute_units: CoreMLComputeUnits::CpuOnly,
            ..Default::default()
        }
    }
}

/// CoreML compute unit selection.
///
/// Determines which hardware units CoreML should use for inference.
/// The Neural Engine provides the best performance for ML workloads
/// on supported devices (A12+ chips, M1+ Macs).
#[cfg(feature = "coreml-ep")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLComputeUnits {
    /// CPU only - no hardware acceleration.
    ///
    /// Useful for testing or when other compute units are unavailable.
    CpuOnly,

    /// CPU and GPU acceleration.
    ///
    /// Uses Metal for GPU compute. Good for older devices without
    /// Neural Engine support.
    CpuAndGpu,

    /// CPU and Neural Engine acceleration (recommended).
    ///
    /// Uses Apple's Neural Engine for optimal ML inference performance.
    /// Available on A12+ (iPhone XS and later) and M1+ Macs.
    CpuAndNeuralEngine,

    /// All available compute units.
    ///
    /// Lets CoreML decide the optimal execution strategy based on
    /// the model and available hardware.
    All,
}

#[cfg(feature = "coreml-ep")]
impl Default for CoreMLComputeUnits {
    fn default() -> Self {
        // Default to Neural Engine for best performance on Apple Silicon
        Self::CpuAndNeuralEngine
    }
}

#[cfg(feature = "coreml-ep")]
impl fmt::Display for CoreMLComputeUnits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuOnly => write!(f, "cpu"),
            Self::CpuAndGpu => write!(f, "gpu"),
            Self::CpuAndNeuralEngine => write!(f, "ane"),
            Self::All => write!(f, "all"),
        }
    }
}

#[cfg(feature = "coreml-ep")]
impl CoreMLComputeUnits {
    /// Parses a compute unit from a string.
    ///
    /// Accepts: "cpu", "gpu", "ane", "neural-engine", "all"
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" | "cpu-only" => Some(Self::CpuOnly),
            "gpu" | "cpu-gpu" => Some(Self::CpuAndGpu),
            "ane" | "neural-engine" | "cpu-ane" => Some(Self::CpuAndNeuralEngine),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_provider_is_cpu() {
        let provider = ExecutionProviderKind::default();
        assert_eq!(provider, ExecutionProviderKind::Cpu);
        assert_eq!(provider.name(), "cpu");
        assert!(!provider.requires_hardware());
    }

    #[test]
    fn test_cpu_provider_display() {
        let provider = ExecutionProviderKind::Cpu;
        assert_eq!(format!("{}", provider), "cpu");
    }

    #[cfg(feature = "coreml-ep")]
    #[test]
    fn test_coreml_config_default() {
        let config = CoreMLConfig::default();
        assert_eq!(config.compute_units, CoreMLComputeUnits::CpuAndNeuralEngine);
        assert!(config.use_subgraphs);
        assert!(!config.require_static_shapes);
    }

    #[cfg(feature = "coreml-ep")]
    #[test]
    fn test_coreml_compute_units_from_str() {
        assert_eq!(
            CoreMLComputeUnits::from_str("ane"),
            Some(CoreMLComputeUnits::CpuAndNeuralEngine)
        );
        assert_eq!(
            CoreMLComputeUnits::from_str("gpu"),
            Some(CoreMLComputeUnits::CpuAndGpu)
        );
        assert_eq!(
            CoreMLComputeUnits::from_str("cpu"),
            Some(CoreMLComputeUnits::CpuOnly)
        );
        assert_eq!(
            CoreMLComputeUnits::from_str("all"),
            Some(CoreMLComputeUnits::All)
        );
        assert_eq!(CoreMLComputeUnits::from_str("invalid"), None);
    }

    #[cfg(feature = "coreml-ep")]
    #[test]
    fn test_coreml_provider_display() {
        let provider = ExecutionProviderKind::CoreML(CoreMLConfig::with_neural_engine());
        assert_eq!(format!("{}", provider), "coreml-ane");
        assert_eq!(provider.name(), "coreml");
        assert!(provider.requires_hardware());
    }
}

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

// ============================================================================
// Auto-Selection Heuristics
// ============================================================================

/// Model characteristics used for execution provider selection.
///
/// These hints help the auto-selection algorithm choose the optimal
/// execution provider based on model properties.
#[derive(Debug, Clone, Default)]
pub struct ModelHints {
    /// Model task type (e.g., "image_classification", "text-to-speech", "asr")
    pub task: Option<String>,

    /// Input tensor shapes (if known). Static shapes benefit from ANE.
    pub input_shapes: Option<Vec<Vec<i64>>>,

    /// Whether all input shapes are static (no dynamic dimensions).
    /// Models with static shapes perform better on Neural Engine.
    pub static_shapes: Option<bool>,

    /// Model file size in MB (approximate).
    /// Very small models (<1MB) often run faster on CPU due to dispatch overhead.
    pub model_size_mb: Option<f32>,

    /// Whether the model uses autoregressive decoding (e.g., TTS, LLM).
    /// Autoregressive models typically don't benefit from ANE.
    pub autoregressive: Option<bool>,

    /// Explicit execution provider preference from pipeline YAML.
    /// Overrides auto-selection when specified.
    pub explicit_provider: Option<String>,
}

impl ModelHints {
    /// Creates hints from model metadata JSON.
    pub fn from_metadata(metadata: &serde_json::Value) -> Self {
        let mut hints = Self::default();

        // Extract task
        if let Some(task) = metadata.get("task").and_then(|v| v.as_str()) {
            hints.task = Some(task.to_string());
        }

        // Extract input_shape and determine if static
        if let Some(input_shape) = metadata.get("input_shape").and_then(|v| v.as_array()) {
            let shape: Vec<i64> = input_shape
                .iter()
                .filter_map(|v| v.as_i64())
                .collect();
            if !shape.is_empty() {
                // Check if all dimensions are positive (static)
                let is_static = shape.iter().all(|&d| d > 0);
                hints.static_shapes = Some(is_static);
                hints.input_shapes = Some(vec![shape]);
            }
        }

        // Extract model_size_mb
        if let Some(size) = metadata.get("model_size_mb").and_then(|v| v.as_f64()) {
            hints.model_size_mb = Some(size as f32);
        }

        hints
    }

    /// Checks if this model is likely a vision/CNN model.
    pub fn is_vision_model(&self) -> bool {
        if let Some(ref task) = self.task {
            let task_lower = task.to_lowercase();
            return task_lower.contains("image")
                || task_lower.contains("vision")
                || task_lower.contains("classification")
                || task_lower.contains("detection")
                || task_lower.contains("segmentation");
        }

        // Heuristic: 4D input [batch, channels, height, width] suggests vision
        if let Some(ref shapes) = self.input_shapes {
            if let Some(shape) = shapes.first() {
                if shape.len() == 4 && shape[1] <= 4 {
                    // Likely [B, C, H, W] where C <= 4 (RGB/RGBA)
                    return true;
                }
            }
        }

        false
    }

    /// Checks if this model is likely a TTS model.
    pub fn is_tts_model(&self) -> bool {
        if let Some(ref task) = self.task {
            let task_lower = task.to_lowercase();
            return task_lower.contains("tts")
                || task_lower.contains("text-to-speech")
                || task_lower.contains("speech-synthesis");
        }
        false
    }

    /// Checks if this model is likely an embedding model.
    pub fn is_embedding_model(&self) -> bool {
        if let Some(ref task) = self.task {
            let task_lower = task.to_lowercase();
            return task_lower.contains("embedding")
                || task_lower.contains("encoder")
                || task_lower.contains("sentence");
        }
        false
    }

    /// Checks if this model is too small to benefit from hardware acceleration.
    pub fn is_tiny_model(&self) -> bool {
        if let Some(size) = self.model_size_mb {
            return size < 1.0; // Less than 1MB
        }
        false
    }
}

/// Selects the optimal execution provider based on model hints.
///
/// This implements the auto-selection heuristics:
/// - Static shapes + vision/CNN → CoreML ANE
/// - Embedding models with fixed sequence → CoreML ANE
/// - TTS/autoregressive models → CPU
/// - Tiny models (<1MB) → CPU
/// - Dynamic shapes → CPU
///
/// # Arguments
///
/// * `hints` - Model characteristics for decision making
///
/// # Returns
///
/// The recommended execution provider kind
pub fn select_optimal_provider(hints: &ModelHints) -> ExecutionProviderKind {
    // Check for explicit override first
    if let Some(ref explicit) = hints.explicit_provider {
        return parse_provider_string(explicit);
    }

    // Only consider CoreML on Apple platforms with the feature enabled
    #[cfg(all(feature = "coreml-ep", any(target_os = "macos", target_os = "ios")))]
    {
        // Rule 1: Tiny models → CPU (dispatch overhead)
        if hints.is_tiny_model() {
            return ExecutionProviderKind::Cpu;
        }

        // Rule 2: TTS/autoregressive → CPU (dynamic output)
        if hints.is_tts_model() || hints.autoregressive == Some(true) {
            return ExecutionProviderKind::Cpu;
        }

        // Rule 3: Dynamic shapes → CPU
        if hints.static_shapes == Some(false) {
            return ExecutionProviderKind::Cpu;
        }

        // Rule 4: Vision models with static shapes → CoreML ANE
        if hints.is_vision_model() && hints.static_shapes != Some(false) {
            return ExecutionProviderKind::CoreML(CoreMLConfig::with_neural_engine());
        }

        // Rule 5: Embedding models → CoreML ANE (usually fixed sequence)
        if hints.is_embedding_model() {
            return ExecutionProviderKind::CoreML(CoreMLConfig::with_neural_engine());
        }

        // Rule 6: Unknown model with static shapes and reasonable size → try CoreML
        if hints.static_shapes == Some(true) {
            if let Some(size) = hints.model_size_mb {
                if size >= 1.0 && size <= 500.0 {
                    return ExecutionProviderKind::CoreML(CoreMLConfig::with_neural_engine());
                }
            }
        }
    }

    // Default to CPU for safety
    ExecutionProviderKind::Cpu
}

/// Parses an execution provider from a string (for YAML override).
///
/// Accepts:
/// - "cpu" → CPU
/// - "coreml", "coreml-ane", "ane" → CoreML with Neural Engine
/// - "coreml-gpu", "gpu" → CoreML with GPU
/// - "coreml-all" → CoreML with all compute units
/// - "auto" → triggers auto-selection (returns CPU as placeholder)
pub fn parse_provider_string(s: &str) -> ExecutionProviderKind {
    let s_lower = s.to_lowercase();

    match s_lower.as_str() {
        "cpu" => ExecutionProviderKind::Cpu,

        #[cfg(feature = "coreml-ep")]
        "coreml" | "coreml-ane" | "ane" | "neural-engine" => {
            ExecutionProviderKind::CoreML(CoreMLConfig::with_neural_engine())
        }

        #[cfg(feature = "coreml-ep")]
        "coreml-gpu" | "gpu" => ExecutionProviderKind::CoreML(CoreMLConfig::with_gpu()),

        #[cfg(feature = "coreml-ep")]
        "coreml-all" | "all" => ExecutionProviderKind::CoreML(CoreMLConfig {
            compute_units: CoreMLComputeUnits::All,
            ..Default::default()
        }),

        // Auto or unknown → CPU (auto-selection happens elsewhere)
        _ => ExecutionProviderKind::Cpu,
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

    // Auto-selection tests

    #[test]
    fn test_model_hints_from_metadata_vision() {
        let metadata = serde_json::json!({
            "task": "image_classification",
            "input_shape": [1, 3, 224, 224],
            "model_size_mb": 13.3
        });

        let hints = ModelHints::from_metadata(&metadata);
        assert_eq!(hints.task, Some("image_classification".to_string()));
        assert_eq!(hints.static_shapes, Some(true));
        assert_eq!(hints.model_size_mb, Some(13.3));
        assert!(hints.is_vision_model());
        assert!(!hints.is_tts_model());
    }

    #[test]
    fn test_model_hints_from_metadata_tts() {
        let metadata = serde_json::json!({
            "task": "text-to-speech",
            "model_size_mb": 170.0
        });

        let hints = ModelHints::from_metadata(&metadata);
        assert!(hints.is_tts_model());
        assert!(!hints.is_vision_model());
    }

    #[test]
    fn test_model_hints_vision_by_shape() {
        // No task, but 4D input shape suggests vision
        let hints = ModelHints {
            input_shapes: Some(vec![vec![1, 3, 224, 224]]),
            ..Default::default()
        };
        assert!(hints.is_vision_model());
    }

    #[test]
    fn test_tiny_model_detection() {
        let hints = ModelHints {
            model_size_mb: Some(0.5),
            ..Default::default()
        };
        assert!(hints.is_tiny_model());

        let hints_large = ModelHints {
            model_size_mb: Some(13.0),
            ..Default::default()
        };
        assert!(!hints_large.is_tiny_model());
    }

    #[test]
    fn test_parse_provider_string() {
        assert_eq!(parse_provider_string("cpu"), ExecutionProviderKind::Cpu);
        assert_eq!(parse_provider_string("CPU"), ExecutionProviderKind::Cpu);
        assert_eq!(parse_provider_string("unknown"), ExecutionProviderKind::Cpu);
    }

    #[cfg(feature = "coreml-ep")]
    #[test]
    fn test_parse_provider_string_coreml() {
        match parse_provider_string("coreml-ane") {
            ExecutionProviderKind::CoreML(config) => {
                assert_eq!(config.compute_units, CoreMLComputeUnits::CpuAndNeuralEngine);
            }
            _ => panic!("Expected CoreML provider"),
        }

        match parse_provider_string("gpu") {
            ExecutionProviderKind::CoreML(config) => {
                assert_eq!(config.compute_units, CoreMLComputeUnits::CpuAndGpu);
            }
            _ => panic!("Expected CoreML GPU provider"),
        }
    }

    #[test]
    fn test_select_optimal_provider_explicit_override() {
        let hints = ModelHints {
            explicit_provider: Some("cpu".to_string()),
            task: Some("image_classification".to_string()),
            static_shapes: Some(true),
            model_size_mb: Some(13.0),
            ..Default::default()
        };

        // Even though it's a vision model, explicit override should win
        let provider = select_optimal_provider(&hints);
        assert_eq!(provider, ExecutionProviderKind::Cpu);
    }

    #[cfg(all(feature = "coreml-ep", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_select_optimal_provider_vision() {
        let hints = ModelHints {
            task: Some("image_classification".to_string()),
            static_shapes: Some(true),
            model_size_mb: Some(13.0),
            ..Default::default()
        };

        let provider = select_optimal_provider(&hints);
        match provider {
            ExecutionProviderKind::CoreML(config) => {
                assert_eq!(config.compute_units, CoreMLComputeUnits::CpuAndNeuralEngine);
            }
            _ => panic!("Expected CoreML ANE for vision model"),
        }
    }

    #[cfg(all(feature = "coreml-ep", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_select_optimal_provider_tts_returns_cpu() {
        let hints = ModelHints {
            task: Some("text-to-speech".to_string()),
            model_size_mb: Some(170.0),
            ..Default::default()
        };

        let provider = select_optimal_provider(&hints);
        assert_eq!(provider, ExecutionProviderKind::Cpu);
    }

    #[cfg(all(feature = "coreml-ep", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_select_optimal_provider_tiny_model_returns_cpu() {
        let hints = ModelHints {
            task: Some("image_classification".to_string()),
            static_shapes: Some(true),
            model_size_mb: Some(0.02), // 20KB - too small
            ..Default::default()
        };

        let provider = select_optimal_provider(&hints);
        assert_eq!(provider, ExecutionProviderKind::Cpu);
    }
}

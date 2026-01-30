//! ONNX Mobile Runtime Adapter implementation.
//!
//! This module provides a stub implementation of RuntimeAdapter for ONNX models
//! optimized for mobile platforms (Android, iOS). It includes mobile-specific
//! features like NNAPI delegate detection, battery-aware throttling, and GPU/Vulkan support.
//!
//! For MVP, it simulates ONNX inference without requiring the actual ONNX Runtime library.
//! Future versions will integrate with ort (ONNX Runtime) or similar crates with
//! mobile-specific optimizations.
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::runtime_adapter::onnx::ONNXMobileRuntimeAdapter;
//! use xybrid_core::runtime_adapter::RuntimeAdapter;
//!
//! let mut adapter = ONNXMobileRuntimeAdapter::new();
//! adapter.load_model("/path/to/model.onnx")?;
//! ```

use super::session::ONNXSession;
use crate::device::capabilities::ThermalState;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::tensor_utils::{envelope_to_tensors, tensors_to_envelope};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use std::collections::HashMap;
use std::path::Path;

/// ONNX Mobile Runtime Adapter.
///
/// This adapter provides real ONNX model loading and inference optimized for mobile platforms.
/// It includes mobile-specific features:
/// - NNAPI delegate detection (Android Neural Networks API)
/// - Battery-aware execution throttling
/// - GPU/Vulkan acceleration detection
/// - Thermal state management
///
/// # Behavior
///
/// - `load_model()`: Loads ONNX model using ONNX Runtime and stores session
/// - `execute()`: Runs real inference using ONNX Runtime with battery/thermal awareness
/// - Supports multiple models loaded simultaneously via `RuntimeAdapterExt`
pub struct ONNXMobileRuntimeAdapter {
    /// Map of loaded models (model_id -> metadata)
    models: HashMap<String, ModelMetadata>,
    /// Map of ONNX Runtime sessions (model_id -> session)
    sessions: HashMap<String, ONNXSession>,
    /// Currently active model (for simple single-model execution)
    current_model: Option<String>,
    /// NNAPI availability (stub: always true for now)
    nnapi_available: bool,
    /// GPU/Vulkan availability (stub: always true for now)
    gpu_available: bool,
    /// Current battery level (0-100)
    battery_level: u8,
    /// Current thermal state
    thermal_state: ThermalState,
}

impl ONNXMobileRuntimeAdapter {
    /// Creates a new ONNX Mobile Runtime Adapter instance.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            sessions: HashMap::new(),
            current_model: None,
            nnapi_available: Self::detect_nnapi_availability(),
            gpu_available: Self::detect_gpu_availability(),
            battery_level: 100, // Default to full battery
            thermal_state: ThermalState::Normal,
        }
    }

    /// Creates a new adapter with specified battery level and thermal state.
    ///
    /// Useful for testing mobile-specific scenarios.
    pub fn with_conditions(battery_level: u8, thermal_state: ThermalState) -> Self {
        Self {
            models: HashMap::new(),
            sessions: HashMap::new(),
            current_model: None,
            nnapi_available: Self::detect_nnapi_availability(),
            gpu_available: Self::detect_gpu_availability(),
            battery_level,
            thermal_state,
        }
    }

    /// Detects NNAPI (Android Neural Networks API) availability.
    ///
    /// For MVP, this is a stub that returns true.
    /// Real implementation would check:
    /// - Android API level >= 27 (Android 8.1)
    /// - NNAPI runtime availability via JNI
    /// - Hardware accelerator support (DSP, NPU, GPU)
    fn detect_nnapi_availability() -> bool {
        // Stub: Always return true for now
        // TODO: Real implementation would check:
        // - Android API level
        // - NNAPI runtime via JNI calls
        // - Available accelerators
        #[cfg(target_os = "android")]
        {
            true
        }
        #[cfg(not(target_os = "android"))]
        {
            false // NNAPI is Android-specific
        }
    }

    /// Detects GPU/Vulkan acceleration availability.
    ///
    /// For MVP, this is a stub that returns true.
    /// Real implementation would check:
    /// - Vulkan API availability
    /// - GPU compute capability
    /// - Memory constraints
    fn detect_gpu_availability() -> bool {
        // Stub: Always return true for now
        // TODO: Real implementation would check:
        // - Vulkan device availability
        // - GPU compute shader support
        // - Sufficient memory available
        true
    }

    /// Returns whether NNAPI is available.
    pub fn has_nnapi(&self) -> bool {
        self.nnapi_available
    }

    /// Returns whether GPU acceleration is available.
    pub fn has_gpu(&self) -> bool {
        self.gpu_available
    }

    /// Returns the current battery level (0-100).
    pub fn battery_level(&self) -> u8 {
        self.battery_level
    }

    /// Updates the battery level.
    ///
    /// Used for testing or when battery state changes.
    pub fn set_battery_level(&mut self, level: u8) {
        self.battery_level = level.min(100);
    }

    /// Returns the current thermal state.
    pub fn thermal_state(&self) -> ThermalState {
        self.thermal_state
    }

    /// Updates the thermal state.
    ///
    /// Used for testing or when thermal state changes.
    pub fn set_thermal_state(&mut self, state: ThermalState) {
        self.thermal_state = state;
    }

    /// Validates that a model file exists and is accessible.
    fn validate_model_file(&self, model_path: &str) -> AdapterResult<()> {
        let path = Path::new(model_path);

        if !path.exists() {
            return Err(AdapterError::ModelNotFound(format!(
                "Model file not found: {}",
                model_path
            )));
        }

        if !path.is_file() {
            return Err(AdapterError::InvalidInput(format!(
                "Path is not a file: {}",
                model_path
            )));
        }

        // Check if it's an ONNX file (basic validation)
        if let Some(ext) = path.extension() {
            if ext != "onnx" && ext != "ONNX" {
                // Warn but don't fail (some models might have different extensions)
            }
        }

        Ok(())
    }

    /// Extracts model ID from file path (for internal tracking).
    fn extract_model_id(&self, path: &str) -> String {
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Determines if execution should be throttled based on battery and thermal state.
    ///
    /// Returns true if execution should be throttled (reduced performance).
    pub fn should_throttle(&self) -> bool {
        // Throttle if battery is low (< 20%)
        if self.battery_level < 20 {
            return true;
        }

        // Throttle if device is hot or critical
        matches!(
            self.thermal_state,
            ThermalState::Hot | ThermalState::Critical
        )
    }

    /// Simulates inference execution with mobile-specific optimizations.
    ///
    /// For MVP, this generates mock outputs based on input kind.
    /// Real implementation would:
    /// 1. Convert Envelope to ONNX tensor format
    /// 2. Select execution provider (NNAPI, GPU, CPU) based on availability and battery
    /// 3. Run inference via ONNX Runtime with mobile optimizations
    /// 4. Convert output tensors back to Envelope
    /// 5. Handle throttling based on battery/thermal state
    #[allow(dead_code)]
    fn simulate_inference(&self, input: &Envelope) -> Envelope {
        // Mock inference: transform input kind to output kind
        // Mobile-optimized inference with battery awareness
        let output_text = if self.should_throttle() {
            // Throttled execution: slower but more battery-efficient
            match &input.kind {
                EnvelopeKind::Audio(_) => "onnx-mobile-throttled-transcribed text".to_string(),
                EnvelopeKind::Text(text) => format!("onnx-mobile-throttled-{}-output", text),
                EnvelopeKind::Embedding(_) => "onnx-mobile-throttled-similarity result".to_string(),
            }
        } else {
            // Normal execution: full performance
            match &input.kind {
                EnvelopeKind::Audio(_) => "onnx-mobile-transcribed text".to_string(),
                EnvelopeKind::Text(text) => format!("onnx-mobile-{}-output", text),
                EnvelopeKind::Embedding(_) => "onnx-mobile-similarity result".to_string(),
            }
        };

        Envelope::new(EnvelopeKind::Text(output_text))
    }

    /// Runs real ONNX Runtime inference.
    ///
    /// # Arguments
    ///
    /// * `session` - ONNX Runtime session for the model
    /// * `input` - Input envelope
    ///
    /// # Returns
    ///
    /// Output envelope with inference results
    fn real_inference(&self, session: &ONNXSession, input: &Envelope) -> AdapterResult<Envelope> {
        // Convert Envelope to tensors
        let input_shapes: Vec<Vec<i64>> = session.input_shapes().to_vec();
        let input_names: Vec<String> = session.input_names().to_vec();

        let input_tensors = envelope_to_tensors(input, &input_shapes, &input_names)?;

        // Run inference
        let output_tensors = session.run(input_tensors).map_err(|e| {
            AdapterError::InferenceFailed(format!("ONNX Runtime inference failed: {}", e))
        })?;

        // Convert tensors back to Envelope
        let output_names: Vec<String> = session.output_names().to_vec();
        let output = tensors_to_envelope(&output_tensors, &output_names)?;

        Ok(output)
    }
}

impl Default for ONNXMobileRuntimeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for ONNXMobileRuntimeAdapter {
    fn name(&self) -> &str {
        "onnx-mobile"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["onnx", "onnx.gz", "onnx.quantized"]
    }

    fn load_model(&mut self, path: &str) -> AdapterResult<()> {
        // Validate model file exists
        self.validate_model_file(path)?;

        // Extract model ID from path
        let model_id = self.extract_model_id(path);

        // Check if model is already loaded - just log and continue
        if self.models.contains_key(&model_id) {
            log::warn!("Model '{}' is already loaded, skipping reload", model_id);
            return Ok(());
        }

        // Create ONNX Runtime session
        // Use NNAPI on Android if available and battery is sufficient
        let use_nnapi = self.nnapi_available && self.battery_level > 20;
        let use_metal = false; // Metal is for macOS/iOS, not Android

        let session = ONNXSession::new(path, use_nnapi, use_metal)?;

        // Extract real input/output shapes from session
        let input_shapes = session.input_shapes();
        let output_shapes = session.output_shapes();
        let input_names = session.input_names();
        let output_names = session.output_names();

        // Create metadata with real shapes from session
        let mut input_schema = HashMap::new();
        for (i, name) in input_names.iter().enumerate() {
            if let Some(shape) = input_shapes.get(i) {
                input_schema.insert(name.clone(), shape.iter().map(|&s| s as u64).collect());
            }
        }

        let mut output_schema = HashMap::new();
        for (i, name) in output_names.iter().enumerate() {
            if let Some(shape) = output_shapes.get(i) {
                output_schema.insert(name.clone(), shape.iter().map(|&s| s as u64).collect());
            }
        }

        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            version: "1.0.0".to_string(), // Default version
            runtime_type: "onnx-mobile".to_string(),
            model_path: path.to_string(),
            input_schema,
            output_schema,
        };

        // Store session and metadata
        self.sessions.insert(model_id.clone(), session);
        self.models.insert(model_id.clone(), metadata);
        self.current_model = Some(model_id);

        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        // Check if a model is loaded
        let model_id = self.current_model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load_model() first.".to_string())
        })?;

        // Get session for current model
        let session = self.sessions.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Session for model '{}' not found", model_id))
        })?;

        // Run real inference
        self.real_inference(session, input)
    }
}

impl RuntimeAdapterExt for ONNXMobileRuntimeAdapter {
    fn is_loaded(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    fn get_metadata(&self, model_id: &str) -> AdapterResult<&ModelMetadata> {
        self.models.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Model '{}' is not loaded", model_id))
        })
    }

    fn infer(&self, model_id: &str, input: &Envelope) -> AdapterResult<Envelope> {
        // Check if model is loaded
        if !self.is_loaded(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' is not loaded. Call load_model() first.",
                model_id
            )));
        }

        // Get session for model
        let session = self.sessions.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Session for model '{}' not found", model_id))
        })?;

        // Run real inference
        self.real_inference(session, input)
    }

    fn unload_model(&mut self, model_id: &str) -> AdapterResult<()> {
        if !self.models.contains_key(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' is not loaded",
                model_id
            )));
        }

        // Remove session (will be dropped automatically, freeing resources)
        self.sessions.remove(model_id);

        // Remove metadata
        self.models.remove(model_id);

        // Clear current model if it was the one being unloaded
        if self.current_model.as_ref() == Some(&model_id.to_string()) {
            self.current_model = None;
        }

        Ok(())
    }

    fn list_loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_adapter() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        assert!(adapter.list_loaded_models().is_empty());
    }

    #[test]
    fn test_adapter_name() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        assert_eq!(adapter.name(), "onnx-mobile");
    }

    #[test]
    fn test_supported_formats() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        let formats = adapter.supported_formats();
        assert!(formats.contains(&"onnx"));
        assert!(formats.contains(&"onnx.gz"));
        assert!(formats.contains(&"onnx.quantized"));
    }

    #[test]
    fn test_nnapi_detection() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        // NNAPI should be false on non-Android platforms
        #[cfg(not(target_os = "android"))]
        assert!(!adapter.has_nnapi());
        #[cfg(target_os = "android")]
        assert!(adapter.has_nnapi());
    }

    #[test]
    fn test_gpu_detection() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        // Stub always returns true
        assert!(adapter.has_gpu());
    }

    #[test]
    fn test_battery_level() {
        let mut adapter = ONNXMobileRuntimeAdapter::new();
        assert_eq!(adapter.battery_level(), 100);
        adapter.set_battery_level(50);
        assert_eq!(adapter.battery_level(), 50);
        adapter.set_battery_level(150); // Should cap at 100
        assert_eq!(adapter.battery_level(), 100);
    }

    #[test]
    fn test_thermal_state() {
        let mut adapter = ONNXMobileRuntimeAdapter::new();
        assert_eq!(adapter.thermal_state(), ThermalState::Normal);
        adapter.set_thermal_state(ThermalState::Hot);
        assert_eq!(adapter.thermal_state(), ThermalState::Hot);
    }

    #[test]
    fn test_should_throttle_low_battery() {
        let mut adapter = ONNXMobileRuntimeAdapter::new();
        adapter.set_battery_level(15); // Low battery
        assert!(adapter.should_throttle());
    }

    #[test]
    fn test_should_throttle_hot_device() {
        let mut adapter = ONNXMobileRuntimeAdapter::new();
        adapter.set_thermal_state(ThermalState::Hot);
        assert!(adapter.should_throttle());
    }

    #[test]
    fn test_should_throttle_critical_device() {
        let mut adapter = ONNXMobileRuntimeAdapter::new();
        adapter.set_thermal_state(ThermalState::Critical);
        assert!(adapter.should_throttle());
    }

    #[test]
    fn test_should_not_throttle_normal() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        assert!(!adapter.should_throttle());
    }

    #[test]
    fn test_load_model_not_found() {
        let mut adapter = ONNXMobileRuntimeAdapter::new();
        let result = adapter.load_model("/nonexistent/model.onnx");
        assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
    }

    #[test]
    fn test_execute_no_model_loaded() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = adapter.execute(&input);
        assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
    }

    #[test]
    fn test_infer_model_not_loaded() {
        let adapter = ONNXMobileRuntimeAdapter::new();
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = adapter.infer("nonexistent-model", &input);
        assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
    }
}

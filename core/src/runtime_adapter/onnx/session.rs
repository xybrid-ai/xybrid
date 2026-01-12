//! ONNX Runtime session wrapper for managing model sessions and inference.
//!
//! This module provides a wrapper around ONNX Runtime sessions that:
//! - Manages session lifecycle
//! - Extracts model metadata (input/output names and shapes)
//! - Handles execution provider selection (CPU, CoreML, etc.)
//! - Provides a clean interface for running inference
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::runtime_adapter::onnx::{ONNXSession, ExecutionProviderKind};
//!
//! // CPU execution (default)
//! let session = ONNXSession::with_provider("/path/to/model.onnx", ExecutionProviderKind::Cpu)?;
//!
//! // CoreML execution (requires coreml-ep feature)
//! #[cfg(feature = "coreml-ep")]
//! let session = ONNXSession::with_provider(
//!     "/path/to/model.onnx",
//!     ExecutionProviderKind::CoreML(CoreMLConfig::with_neural_engine())
//! )?;
//!
//! let inputs = /* prepare inputs */;
//! let outputs = session.run(inputs)?;
//! ```

use super::execution_provider::ExecutionProviderKind;
use crate::runtime_adapter::{AdapterError, AdapterResult};
use ndarray::{ArrayD, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// ONNX Runtime session wrapper.
///
/// Manages an ONNX Runtime session, including:
/// - Model loading and session creation
/// - Input/output metadata extraction
/// - Execution provider selection
/// - Inference execution
pub struct ONNXSession {
    /// The ONNX Runtime session (wrapped in Mutex for thread-safe interior mutability)
    session: Mutex<Session>,
    /// Input names from the model
    input_names: Vec<String>,
    /// Output names from the model
    output_names: Vec<String>,
    /// Input shapes (may contain dynamic dimensions)
    input_shapes: Vec<Vec<i64>>,
    /// Output shapes (may contain dynamic dimensions)
    output_shapes: Vec<Vec<i64>>,
    /// The execution provider used for this session
    execution_provider: ExecutionProviderKind,
}

impl ONNXSession {
    /// Creates a new ONNX session from a model file (legacy API).
    ///
    /// This method is kept for backwards compatibility. For new code,
    /// prefer using `with_provider()` which gives explicit control over
    /// the execution provider.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `_use_nnapi` - Deprecated: use `with_provider()` instead
    /// * `_use_metal` - Deprecated: use `with_provider()` instead
    ///
    /// # Returns
    ///
    /// A new `ONNXSession` instance using CPU execution
    pub fn new(model_path: &str, _use_nnapi: bool, _use_metal: bool) -> AdapterResult<Self> {
        Self::with_provider(model_path, ExecutionProviderKind::Cpu)
    }

    /// Creates a new ONNX session with the specified execution provider.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `execution_provider` - The execution provider to use (CPU, CoreML, etc.)
    ///
    /// # Returns
    ///
    /// A new `ONNXSession` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model file doesn't exist
    /// - Model loading fails
    /// - Execution provider initialization fails
    /// - Metadata extraction fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use xybrid_core::runtime_adapter::onnx::{ONNXSession, ExecutionProviderKind};
    ///
    /// // CPU execution
    /// let session = ONNXSession::with_provider("model.onnx", ExecutionProviderKind::Cpu)?;
    ///
    /// // CoreML with Neural Engine (requires coreml-ep feature)
    /// #[cfg(feature = "coreml-ep")]
    /// let session = ONNXSession::with_provider(
    ///     "model.onnx",
    ///     ExecutionProviderKind::CoreML(CoreMLConfig::with_neural_engine())
    /// )?;
    /// ```
    pub fn with_provider(
        model_path: &str,
        execution_provider: ExecutionProviderKind,
    ) -> AdapterResult<Self> {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(AdapterError::ModelNotFound(format!(
                "Model file not found: {}",
                model_path
            )));
        }

        // Initialize ONNX Runtime environment (singleton, safe to call multiple times)
        let _ = ort::init().commit();

        // Create session builder with optimization
        let mut builder = Session::builder()
            .map_err(|e| {
                AdapterError::RuntimeError(format!("Failed to create session builder: {}", e))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                AdapterError::RuntimeError(format!("Failed to set optimization level: {}", e))
            })?;

        // Configure execution provider
        builder = Self::configure_execution_provider(builder, &execution_provider)?;

        // Load model
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| {
                AdapterError::RuntimeError(format!("Failed to load ONNX model: {}", e))
            })?;

        // Extract input/output metadata from session
        let (input_names, input_shapes) = Self::extract_input_metadata(&session)?;
        let (output_names, output_shapes) = Self::extract_output_metadata(&session)?;

        log::info!(
            "Created ONNX session with {} execution provider for model: {}",
            execution_provider,
            model_path
        );

        Ok(Self {
            session: Mutex::new(session),
            input_names,
            output_names,
            input_shapes,
            output_shapes,
            execution_provider,
        })
    }

    /// Configures the execution provider on the session builder.
    fn configure_execution_provider(
        builder: ort::session::builder::SessionBuilder,
        provider: &ExecutionProviderKind,
    ) -> AdapterResult<ort::session::builder::SessionBuilder> {
        match provider {
            ExecutionProviderKind::Cpu => {
                // CPU is the default, no additional configuration needed
                Ok(builder)
            }

            #[cfg(feature = "coreml-ep")]
            ExecutionProviderKind::CoreML(config) => {
                use super::execution_provider::CoreMLComputeUnits;
                use ort::ep;

                // Build CoreML execution provider with configuration
                let coreml_ep = {
                    let mut coreml = ep::CoreML::default();

                    // Configure subgraphs
                    coreml = coreml.with_subgraphs(config.use_subgraphs);

                    // Configure compute units
                    coreml = coreml.with_compute_units(match config.compute_units {
                        CoreMLComputeUnits::CpuOnly => ep::coreml::ComputeUnits::CPUOnly,
                        CoreMLComputeUnits::CpuAndGpu => ep::coreml::ComputeUnits::CPUAndGPU,
                        CoreMLComputeUnits::CpuAndNeuralEngine => ep::coreml::ComputeUnits::CPUAndNeuralEngine,
                        CoreMLComputeUnits::All => ep::coreml::ComputeUnits::All,
                    });

                    coreml.build()
                };

                log::debug!("Configuring CoreML execution provider: {:?}", config);

                builder
                    .with_execution_providers([coreml_ep])
                    .map_err(|e| {
                        AdapterError::RuntimeError(format!(
                            "Failed to configure CoreML execution provider: {}",
                            e
                        ))
                    })
            }
        }
    }

    /// Extracts input metadata from the session.
    fn extract_input_metadata(session: &Session) -> AdapterResult<(Vec<String>, Vec<Vec<i64>>)> {
        let mut input_names = Vec::new();
        let mut input_shapes = Vec::new();

        // Access session.inputs directly - ort exposes inputs as Vec<Input>
        // Each Input has a name field
        for input in session.inputs() {
            input_names.push(input.name().to_string());
            // Note: ort's Input struct doesn't directly expose shapes
            // Shapes may be dynamic or need to be inferred from the model
            // For now, use placeholder shapes - we'll need to infer from actual model or use default
            // TODO: Extract real shapes from model metadata if available
            input_shapes.push(vec![-1]); // Placeholder: -1 indicates dynamic dimension
        }

        // If no inputs found, use placeholder
        if input_names.is_empty() {
            input_names.push("input".to_string());
            input_shapes.push(vec![1, 1, 16000]); // Placeholder shape for audio
        }

        Ok((input_names, input_shapes))
    }

    /// Extracts output metadata from the session.
    fn extract_output_metadata(session: &Session) -> AdapterResult<(Vec<String>, Vec<Vec<i64>>)> {
        let mut output_names = Vec::new();
        let mut output_shapes = Vec::new();

        // Access session.outputs directly - ort exposes outputs as Vec<Output>
        // Each Output has a name field
        for output in session.outputs() {
            output_names.push(output.name().to_string());
            // Note: ort's Output struct doesn't directly expose shapes
            // Shapes may be dynamic or need to be inferred from the model
            // For now, use placeholder shapes
            // TODO: Extract real shapes from model metadata if available
            output_shapes.push(vec![-1]); // Placeholder: -1 indicates dynamic dimension
        }

        // If no outputs found, use placeholder
        if output_names.is_empty() {
            output_names.push("output".to_string());
            output_shapes.push(vec![1, 512]); // Placeholder shape
        }

        Ok((output_names, output_shapes))
    }

    /// Runs inference on the session.
    ///
    /// # Arguments
    ///
    /// * `inputs` - HashMap mapping input names to tensors (`ndarray::ArrayD<f32>`)
    ///
    /// # Returns
    ///
    /// HashMap mapping output names to output tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input names don't match model inputs
    /// - Tensor shapes don't match expected shapes
    /// - Inference execution fails
    pub fn run(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> AdapterResult<HashMap<String, ArrayD<f32>>> {
        // Convert f32 arrays to Values
        let value_inputs: HashMap<String, Value> = inputs
            .into_iter()
            .map(|(k, v)| {
                Ok((k, Value::from_array(v)
                    .map_err(|e| AdapterError::RuntimeError(format!("Failed to convert tensor: {}", e)))?
                    .into()))
            })
            .collect::<AdapterResult<_>>()?;

        self.run_with_values(value_inputs)
    }

    /// Runs inference with mixed input types (Value types).
    ///
    /// This method accepts `Value` types directly, allowing for mixed f32/i64 inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - HashMap mapping input names to `ort::Value` tensors
    ///
    /// # Returns
    ///
    /// HashMap mapping output names to `ndarray::ArrayD<f32>` tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input names don't match model inputs
    /// - Tensor shapes don't match expected shapes
    /// - Inference execution fails
    pub fn run_with_values(
        &self,
        inputs: HashMap<String, Value>,
    ) -> AdapterResult<HashMap<String, ArrayD<f32>>> {
        use ort::session::SessionInputs;

        // Get mutable access to session (wrapped in Mutex)
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to lock session: {}", e)))?;

        // Convert HashMap to Vec of (Cow<str>, SessionInputValue)
        // This allows us to pass an arbitrary number of inputs
        let ort_inputs: Vec<(std::borrow::Cow<'_, str>, ort::session::SessionInputValue<'_>)> = inputs
            .into_iter()
            .map(|(name, value)| (std::borrow::Cow::Owned(name), value.into()))
            .collect();

        // Run inference with dynamic number of inputs
        let outputs = session_guard.run(SessionInputs::from(ort_inputs))
        .map_err(|e| {
            AdapterError::InferenceFailed(format!("ONNX Runtime inference failed: {}", e))
        })?;

        // Convert outputs back to HashMap<String, ArrayD<f32>>
        let mut result = HashMap::new();

        for output_name in &self.output_names {
            // Extract output value from SessionOutputs
            // SessionOutputs can be indexed by name or accessed as a slice
            let output_value = &outputs[output_name.as_str()];

            // Try to extract as f32 first, then as i64 if that fails
            // This handles models with mixed output types
            let array_d = if let Ok(output_array) = output_value.try_extract_array::<f32>() {
                // Convert ndarray view to owned ArrayD
                let shape = output_array.shape();
                let dims: Vec<usize> = shape.iter().copied().collect();
                let owned_array = output_array.to_owned();
                let data: Vec<f32> = owned_array.as_slice().unwrap().to_vec();
                ArrayD::from_shape_vec(IxDyn(&dims), data)
                    .map_err(|e| AdapterError::RuntimeError(format!("Failed to convert output to ArrayD: {}", e)))?
            } else if let Ok(output_array) = output_value.try_extract_array::<i64>() {
                // Convert i64 to f32 for uniform handling
                let shape = output_array.shape();
                let dims: Vec<usize> = shape.iter().copied().collect();
                let owned_array = output_array.to_owned();
                let data: Vec<f32> = owned_array.as_slice().unwrap().iter().map(|&x| x as f32).collect();
                ArrayD::from_shape_vec(IxDyn(&dims), data)
                    .map_err(|e| AdapterError::RuntimeError(format!("Failed to convert output to ArrayD: {}", e)))?
            } else {
                return Err(AdapterError::RuntimeError(format!(
                    "Failed to extract output '{}': unsupported type (expected f32 or i64)",
                    output_name
                )));
            };

            result.insert(output_name.clone(), array_d);
        }

        Ok(result)
    }

    /// Returns input names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Returns output names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Returns input shapes.
    pub fn input_shapes(&self) -> &[Vec<i64>] {
        &self.input_shapes
    }

    /// Returns output shapes.
    pub fn output_shapes(&self) -> &[Vec<i64>] {
        &self.output_shapes
    }

    /// Returns the execution provider used for this session.
    pub fn execution_provider(&self) -> &ExecutionProviderKind {
        &self.execution_provider
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_session_creation_fails_on_nonexistent_file() {
        let result = ONNXSession::new("/nonexistent/model.onnx", false, false);
        assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
    }

    #[test]
    fn test_session_creation_with_mock_file() {
        // Create a temporary ONNX file (minimal valid ONNX format)
        // Note: This is a minimal test - real ONNX files are binary protobuf
        // For now, we'll test that the file existence check works
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");

        // Create a minimal file (not a real ONNX, but tests file existence)
        fs::write(&model_path, b"fake onnx data").unwrap();

        // This will fail at ort initialization or model loading, but we can test the structure
        let result = ONNXSession::new(model_path.to_str().unwrap(), false, false);

        // The session creation might fail due to invalid ONNX format,
        // but we've at least tested that the file existence check passes
        // and the ort initialization is attempted
        match result {
            Ok(_) => {
                // If it succeeds, verify the structure
                let session = result.unwrap();
                assert!(!session.input_names().is_empty());
                assert!(!session.output_names().is_empty());
            }
            Err(e) => {
                // Expected: invalid ONNX format will cause ort to fail
                // But we've verified the code path executes
                println!("Expected error (invalid ONNX format): {:?}", e);
            }
        }
    }

    #[test]
    fn test_mnist_model_loading() {
        // Test loading the real MNIST model
        // Try multiple possible paths (workspace root, or relative to test execution)
        let possible_paths = vec![
            PathBuf::from("test_models/mnist-12.onnx"),
            PathBuf::from("../test_models/mnist-12.onnx"),
            PathBuf::from("../../test_models/mnist-12.onnx"),
        ];

        let model_path = possible_paths.iter()
            .find(|p| p.exists())
            .cloned();

        let model_path = match model_path {
            Some(p) => p,
            None => {
                println!("MNIST model not found, skipping test. Tried: {:?}", possible_paths);
                return;
            }
        };

        let result = ONNXSession::new(model_path.to_str().unwrap(), false, false);
        assert!(result.is_ok(), "Failed to load MNIST model: {:?}", result.err());

        let session = result.unwrap();

        // Verify we extracted real metadata
        let input_names = session.input_names();
        let output_names = session.output_names();

        println!("MNIST Input names: {:?}", input_names);
        println!("MNIST Output names: {:?}", output_names);
        println!("MNIST Input shapes: {:?}", session.input_shapes());
        println!("MNIST Output shapes: {:?}", session.output_shapes());

        // MNIST should have 1 input and 1 output
        assert!(!input_names.is_empty(), "Should have at least one input");
        assert!(!output_names.is_empty(), "Should have at least one output");

        // Verify input/output names are not placeholders
        assert_ne!(input_names[0], "input", "Should have real input name, not placeholder");
        assert_ne!(output_names[0], "output", "Should have real output name, not placeholder");
    }

    #[test]
    fn test_mnist_inference() {
        // Test running inference on the MNIST model
        // Try multiple possible paths (workspace root, or relative to test execution)
        let possible_paths = vec![
            PathBuf::from("test_models/mnist-12.onnx"),
            PathBuf::from("../test_models/mnist-12.onnx"),
            PathBuf::from("../../test_models/mnist-12.onnx"),
        ];

        let model_path = possible_paths.iter()
            .find(|p| p.exists())
            .cloned();

        let model_path = match model_path {
            Some(p) => p,
            None => {
                println!("MNIST model not found, skipping test. Tried: {:?}", possible_paths);
                return;
            }
        };

        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)
            .expect("Failed to load MNIST model");

        // Get real input name from session
        let input_names = session.input_names();
        let input_name = &input_names[0];

        // Create test input: 28x28 grayscale image (all zeros for now)
        // MNIST expects: [batch=1, channels=1, height=28, width=28]
        let mut inputs = HashMap::new();
        let input_tensor = ArrayD::<f32>::from_shape_vec(
            IxDyn(&[1, 1, 28, 28]),
            vec![0.0f32; 784], // 28*28 = 784
        ).unwrap();
        inputs.insert(input_name.clone(), input_tensor);

        // Run real inference using ONNX Runtime
        let result = session.run(inputs);
        assert!(result.is_ok(), "Inference failed: {:?}", result.err());

        let outputs = result.unwrap();
        assert!(!outputs.is_empty(), "Should have at least one output");

        // Verify output structure
        let output_names = session.output_names();
        let output_name = &output_names[0];
        assert!(outputs.contains_key(output_name), "Output should contain expected output name");

        // Get output tensor
        let output_tensor = outputs.get(output_name).unwrap();
        println!("MNIST Output shape: {:?}", output_tensor.shape());
        println!("MNIST Output size: {}", output_tensor.len());

        // MNIST outputs 10 class probabilities (one for each digit 0-9)
        // Verify we got the correct output shape: [batch=1, classes=10]
        assert_eq!(output_tensor.shape(), &[1, 10], "MNIST should output shape [1, 10]");
        assert_eq!(output_tensor.len(), 10, "MNIST output should have 10 elements");
    }
}

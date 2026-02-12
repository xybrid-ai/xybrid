//! Session factory for ONNX inference sessions.
//!
//! This module provides traits for creating and using inference sessions,
//! enabling:
//! - Mockable session creation for unit tests
//! - Abstraction over different session implementations
//! - Dependency injection in the executor

use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

use super::types::ExecutorResult;
use crate::runtime_adapter::AdapterError;

/// Trait for running inference on a loaded session.
///
/// This trait abstracts the inference interface, allowing for mock
/// implementations in tests without requiring actual ONNX models.
pub trait InferenceSession: Send + Sync {
    /// Run inference with the given inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input names to tensor data
    ///
    /// # Returns
    ///
    /// Map of output names to tensor data
    fn run(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> ExecutorResult<HashMap<String, ArrayD<f32>>>;

    /// Get the names of output tensors.
    fn output_names(&self) -> &[String];

    /// Get the names of input tensors.
    fn input_names(&self) -> &[String];
}

/// Trait for creating inference sessions.
///
/// This trait abstracts session creation, allowing the executor to
/// be tested without actual model files.
pub trait SessionFactory: Send + Sync {
    /// The type of session this factory creates.
    type Session: InferenceSession;

    /// Create a new session from a model file path.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file
    ///
    /// # Returns
    ///
    /// A new inference session
    fn create(&self, model_path: &Path) -> ExecutorResult<Self::Session>;
}

// ============================================================================
// ONNX Implementation
// ============================================================================

use crate::runtime_adapter::onnx::ONNXSession;

/// Wrapper to implement InferenceSession for ONNXSession.
pub struct OnnxInferenceSession {
    session: ONNXSession,
}

impl InferenceSession for OnnxInferenceSession {
    fn run(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
        self.session.run(inputs).map_err(|e| e)
    }

    fn output_names(&self) -> &[String] {
        self.session.output_names()
    }

    fn input_names(&self) -> &[String] {
        self.session.input_names()
    }
}

/// Default session factory that creates real ONNX sessions.
#[derive(Default)]
pub struct OnnxSessionFactory;

impl SessionFactory for OnnxSessionFactory {
    type Session = OnnxInferenceSession;

    fn create(&self, model_path: &Path) -> ExecutorResult<Self::Session> {
        let path_str = model_path
            .to_str()
            .ok_or_else(|| AdapterError::InvalidInput("Invalid model path encoding".to_string()))?;

        let session = ONNXSession::new(path_str, false, false)?;
        Ok(OnnxInferenceSession { session })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    // ============================================================================
    // Mock Implementations for Testing
    // ============================================================================

    /// Mock session that returns predefined outputs.
    pub struct MockSession {
        pub input_names: Vec<String>,
        pub output_names: Vec<String>,
        pub outputs: HashMap<String, ArrayD<f32>>,
    }

    impl MockSession {
        pub fn new() -> Self {
            Self {
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                outputs: HashMap::new(),
            }
        }

        pub fn with_output(mut self, name: &str, tensor: ArrayD<f32>) -> Self {
            self.outputs.insert(name.to_string(), tensor);
            self
        }

        pub fn with_input_names(mut self, names: Vec<String>) -> Self {
            self.input_names = names;
            self
        }

        pub fn with_output_names(mut self, names: Vec<String>) -> Self {
            self.output_names = names;
            self
        }
    }

    impl InferenceSession for MockSession {
        fn run(
            &self,
            _inputs: HashMap<String, ArrayD<f32>>,
        ) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
            Ok(self.outputs.clone())
        }

        fn output_names(&self) -> &[String] {
            &self.output_names
        }

        fn input_names(&self) -> &[String] {
            &self.input_names
        }
    }

    /// Mock factory that creates MockSessions with configurable outputs.
    pub struct MockSessionFactory {
        pub session_template: MockSession,
    }

    impl MockSessionFactory {
        pub fn new() -> Self {
            Self {
                session_template: MockSession::new(),
            }
        }

        pub fn with_output(mut self, name: &str, tensor: ArrayD<f32>) -> Self {
            self.session_template
                .outputs
                .insert(name.to_string(), tensor);
            self
        }
    }

    impl SessionFactory for MockSessionFactory {
        type Session = MockSession;

        fn create(&self, _model_path: &Path) -> ExecutorResult<Self::Session> {
            Ok(MockSession {
                input_names: self.session_template.input_names.clone(),
                output_names: self.session_template.output_names.clone(),
                outputs: self.session_template.outputs.clone(),
            })
        }
    }

    // ============================================================================
    // Tests
    // ============================================================================

    #[test]
    fn test_mock_session_run() {
        let output_tensor = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let session = MockSession::new().with_output("output", output_tensor.clone());

        let inputs = HashMap::new();
        let result = session.run(inputs).unwrap();

        assert_eq!(result.get("output").unwrap(), &output_tensor);
    }

    #[test]
    fn test_mock_session_names() {
        let session = MockSession::new()
            .with_input_names(vec!["tokens".to_string(), "mask".to_string()])
            .with_output_names(vec!["logits".to_string()]);

        assert_eq!(session.input_names(), &["tokens", "mask"]);
        assert_eq!(session.output_names(), &["logits"]);
    }

    #[test]
    fn test_mock_factory_creates_session() {
        let output_tensor = Array1::from_vec(vec![4.0, 5.0, 6.0]).into_dyn();
        let factory = MockSessionFactory::new().with_output("result", output_tensor.clone());

        let session = factory.create(Path::new("/fake/model.onnx")).unwrap();
        let result = session.run(HashMap::new()).unwrap();

        assert_eq!(result.get("result").unwrap(), &output_tensor);
    }
}

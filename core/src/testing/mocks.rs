//! Mock implementations for testing.
//!
//! Provides mock versions of runtime adapters and other components
//! that can be used for unit testing without real model files.

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;

/// A mock model runtime that returns configurable outputs.
///
/// This mock can be configured to:
/// - Return fixed output values
/// - Track how many times it was called
/// - Simulate errors
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::testing::mocks::MockRuntime;
///
/// let mut runtime = MockRuntime::with_embedding(vec![0.1, 0.2, 0.3]);
/// let output = runtime.execute(&input_envelope)?;
/// assert_eq!(runtime.call_count(), 1);
/// ```
pub struct MockRuntime {
    /// Output to return from execute()
    output: MockOutput,
    /// Number of times execute() was called
    call_count: usize,
    /// Whether to simulate an error
    simulate_error: Option<String>,
    /// Loaded model path (if any)
    loaded_model: Option<String>,
}

/// Types of mock outputs
pub enum MockOutput {
    /// Return an embedding vector
    Embedding(Vec<f32>),
    /// Return text
    Text(String),
    /// Return audio bytes
    Audio(Vec<u8>),
    /// Return a tensor map
    TensorMap(HashMap<String, ArrayD<f32>>),
}

impl MockRuntime {
    /// Create a new mock runtime with embedding output.
    pub fn with_embedding(values: Vec<f32>) -> Self {
        Self {
            output: MockOutput::Embedding(values),
            call_count: 0,
            simulate_error: None,
            loaded_model: None,
        }
    }

    /// Create a new mock runtime with text output.
    pub fn with_text(text: impl Into<String>) -> Self {
        Self {
            output: MockOutput::Text(text.into()),
            call_count: 0,
            simulate_error: None,
            loaded_model: None,
        }
    }

    /// Create a new mock runtime with audio output.
    pub fn with_audio(bytes: Vec<u8>) -> Self {
        Self {
            output: MockOutput::Audio(bytes),
            call_count: 0,
            simulate_error: None,
            loaded_model: None,
        }
    }

    /// Create a new mock runtime with tensor output.
    pub fn with_tensor(name: impl Into<String>, tensor: ArrayD<f32>) -> Self {
        let mut map = HashMap::new();
        map.insert(name.into(), tensor);
        Self {
            output: MockOutput::TensorMap(map),
            call_count: 0,
            simulate_error: None,
            loaded_model: None,
        }
    }

    /// Configure the mock to simulate an error.
    pub fn with_error(mut self, error_message: impl Into<String>) -> Self {
        self.simulate_error = Some(error_message.into());
        self
    }

    /// Get the number of times execute() was called.
    pub fn call_count(&self) -> usize {
        self.call_count
    }

    /// Get the loaded model path (if any).
    pub fn loaded_model(&self) -> Option<&str> {
        self.loaded_model.as_deref()
    }

    /// Reset the call count.
    pub fn reset(&mut self) {
        self.call_count = 0;
    }
}

impl ModelRuntime for MockRuntime {
    fn name(&self) -> &str {
        "MockRuntime"
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["onnx", "safetensors"]
    }

    fn load(&mut self, model_path: &Path) -> Result<(), AdapterError> {
        if let Some(ref error) = self.simulate_error {
            return Err(AdapterError::RuntimeError(error.clone()));
        }
        self.loaded_model = Some(model_path.to_string_lossy().to_string());
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(&mut self, _input: &Envelope) -> Result<Envelope, AdapterError> {
        if let Some(ref error) = self.simulate_error {
            return Err(AdapterError::RuntimeError(error.clone()));
        }

        self.call_count += 1;

        let kind = match &self.output {
            MockOutput::Embedding(v) => EnvelopeKind::Embedding(v.clone()),
            MockOutput::Text(t) => EnvelopeKind::Text(t.clone()),
            MockOutput::Audio(b) => EnvelopeKind::Audio(b.clone()),
            MockOutput::TensorMap(_) => {
                // For tensor output, we return an embedding for simplicity
                EnvelopeKind::Embedding(vec![0.0])
            }
        };

        Ok(Envelope {
            kind,
            metadata: HashMap::new(),
        })
    }
}

/// A mock ONNX session for testing preprocessing/postprocessing.
pub struct MockOnnxOutputs {
    pub outputs: HashMap<String, ArrayD<f32>>,
}

impl MockOnnxOutputs {
    /// Create mock outputs with a single logits tensor.
    pub fn with_logits(vocab_size: usize, seq_len: usize) -> Self {
        let logits = ArrayD::zeros(IxDyn(&[1, seq_len, vocab_size]));
        let mut outputs = HashMap::new();
        outputs.insert("logits".to_string(), logits);
        Self { outputs }
    }

    /// Create mock outputs with a waveform tensor.
    pub fn with_waveform(num_samples: usize) -> Self {
        let waveform = ArrayD::zeros(IxDyn(&[1, num_samples]));
        let mut outputs = HashMap::new();
        outputs.insert("waveform".to_string(), waveform);
        Self { outputs }
    }

    /// Create mock outputs with hidden states.
    pub fn with_hidden_states(batch: usize, seq_len: usize, hidden_size: usize) -> Self {
        let hidden = ArrayD::zeros(IxDyn(&[batch, seq_len, hidden_size]));
        let mut outputs = HashMap::new();
        outputs.insert("last_hidden_state".to_string(), hidden);
        Self { outputs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_runtime_with_embedding() {
        let mut runtime = MockRuntime::with_embedding(vec![1.0, 2.0, 3.0]);

        // Load should succeed
        assert!(runtime.load(Path::new("/fake/model.onnx")).is_ok());
        assert_eq!(runtime.loaded_model(), Some("/fake/model.onnx"));

        // Execute should return embedding
        let input = Envelope {
            kind: EnvelopeKind::Text("test".to_string()),
            metadata: HashMap::new(),
        };
        let output = runtime.execute(&input).unwrap();
        match output.kind {
            EnvelopeKind::Embedding(v) => assert_eq!(v, vec![1.0, 2.0, 3.0]),
            _ => panic!("Expected Embedding output"),
        }
        assert_eq!(runtime.call_count(), 1);
    }

    #[test]
    fn test_mock_runtime_with_error() {
        let mut runtime = MockRuntime::with_text("test").with_error("Simulated failure");

        // Load should fail
        let result = runtime.load(Path::new("/fake/model.onnx"));
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_onnx_outputs() {
        let outputs = MockOnnxOutputs::with_logits(1000, 50);
        assert!(outputs.outputs.contains_key("logits"));
        assert_eq!(outputs.outputs["logits"].shape(), &[1, 50, 1000]);
    }
}

use crate::ir::Envelope;
use crate::runtime_adapter::onnx::OnnxRuntimeAdapter;
use crate::runtime_adapter::{AdapterResult, ModelRuntime, RuntimeAdapter};
use std::path::Path;

/// ModelRuntime implementation for generic ONNX models.
/// Wraps the existing OnnxRuntimeAdapter.
pub struct OnnxRuntime {
    adapter: OnnxRuntimeAdapter,
}

impl OnnxRuntime {
    pub fn new() -> Self {
        Self {
            adapter: OnnxRuntimeAdapter::new(),
        }
    }

    pub fn get_session(
        &self,
        model_path: &str,
    ) -> AdapterResult<&crate::runtime_adapter::onnx::ONNXSession> {
        // We rely on adapter internals usually, but adapter.sessions is private.
        // We need to extend OnnxRuntimeAdapter to expose get_session or proxy it.
        // Assuming adapter has get_session logic or we add it.
        self.adapter.get_session(model_path)
    }
}

impl ModelRuntime for OnnxRuntime {
    fn name(&self) -> &str {
        "onnx"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supported_formats(&self) -> Vec<&str> {
        self.adapter.supported_formats()
    }

    fn load(&mut self, model_path: &Path) -> AdapterResult<()> {
        let path_str = model_path.to_string_lossy();
        self.adapter.load_model(&path_str)
    }

    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        self.adapter.execute(input)
    }
}

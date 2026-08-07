use crate::ir::Envelope;
use crate::runtime_adapter::AdapterResult;
use std::path::Path;

/// Trait for model runtime implementations used by TemplateExecutor.
///
/// This trait abstracts the specific execution logic (Candle, ONNX, TTS, etc.)
/// from the generic TemplateExecutor logic.
pub trait ModelRuntime: Send + Sync {
    /// Returns the name of the runtime (e.g., "candle-whisper", "kokoro-tts")
    fn name(&self) -> &str;

    /// Returns supported file extensions
    fn supported_formats(&self) -> Vec<&str>;

    /// Load the model from the specified path
    fn load(&mut self, model_path: &Path) -> AdapterResult<()>;

    /// Whether the model at `model_path` is already loaded in this runtime's
    /// cache.
    ///
    /// Runtimes that keep no cache report `false` (the default), which reads
    /// as "a load (and its cost) would happen". Used to decide whether a
    /// warm-up inference is worth paying.
    fn is_loaded(&self, _model_path: &Path) -> bool {
        false
    }

    /// Downcast to concrete type
    fn as_any(&self) -> &dyn std::any::Any;

    /// Execute inference on the input envelope
    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope>;
}

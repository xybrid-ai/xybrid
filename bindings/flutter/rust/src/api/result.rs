//! Inference result FFI wrappers for Flutter.
use xybrid_sdk::InferenceResult;

/// FFI wrapper for inference results.
/// Fields are public and accessible directly via FRB-generated bindings.
#[derive(Clone)]
pub struct FfiResult {
    pub success: bool,
    pub text: Option<String>,
    pub audio_bytes: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
    pub latency_ms: u32,
}

impl FfiResult {
    pub(crate) fn from_inference_result(r: &InferenceResult) -> Self {
        Self {
            success: true,
            text: r.text().map(|s| s.to_string()),
            audio_bytes: r.audio_bytes().map(|b| b.to_vec()),
            embedding: r.embedding().map(|e| e.to_vec()),
            latency_ms: r.latency_ms(),
        }
    }
}

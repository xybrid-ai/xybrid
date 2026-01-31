//! Inference result FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use xybrid_sdk::InferenceResult;

/// FFI wrapper for inference results.
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

    #[frb(sync)]
    pub fn success(&self) -> bool {
        self.success
    }

    #[frb(sync)]
    pub fn text(&self) -> Option<String> {
        self.text.clone()
    }

    #[frb(sync)]
    pub fn audio_bytes(&self) -> Option<Vec<u8>> {
        self.audio_bytes.clone()
    }

    #[frb(sync)]
    pub fn embedding(&self) -> Option<Vec<f32>> {
        self.embedding.clone()
    }

    #[frb(sync)]
    pub fn latency_ms(&self) -> u32 {
        self.latency_ms
    }
}

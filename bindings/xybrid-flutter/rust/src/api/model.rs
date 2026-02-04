//! Model loading FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::sync::Arc;
use xybrid_sdk::{ModelLoader, XybridModel};

use crate::frb_generated::StreamSink;

use super::result::FfiResult;

/// Event emitted during model loading with progress.
#[derive(Clone)]
pub enum FfiLoadEvent {
    /// Download progress update (0.0 to 1.0)
    Progress(f64),
    /// Model loaded successfully - contains the model handle ID
    Complete,
    /// An error occurred during loading
    Error(String),
}

/// Event emitted during streaming inference.
/// Follows the "everything is a stream" pattern from the SDK.
#[derive(Clone)]
pub enum FfiStreamEvent {
    /// A token was generated
    Token(FfiStreamToken),
    /// Inference completed with final result
    Complete(FfiResult),
    /// An error occurred
    Error(String),
}

/// Token received during streaming inference.
/// Mirrors the SDK's StreamToken structure for FFI.
#[derive(Clone)]
pub struct FfiStreamToken {
    /// The generated token text
    pub token: String,
    /// The token ID (if available)
    pub token_id: Option<i64>,
    /// Index of this token in the sequence
    pub index: u32,
    /// Cumulative text generated so far
    pub cumulative_text: String,
    /// Reason for stopping (if this is the final token)
    pub finish_reason: Option<String>,
}

/// FFI wrapper for ModelLoader (preparatory step before loading).
#[frb(opaque)]
pub struct FfiModelLoader(ModelLoader);

/// FFI wrapper for a loaded XybridModel ready for inference.
#[frb(opaque)]
pub struct FfiModel(Arc<XybridModel>);

impl From<xybrid_sdk::StreamEvent> for FfiStreamEvent {
    fn from(event: xybrid_sdk::StreamEvent) -> Self {
        match event {
            xybrid_sdk::StreamEvent::Token(token) => FfiStreamEvent::Token(FfiStreamToken {
                token: token.token,
                token_id: token.token_id,
                index: token.index as u32,
                cumulative_text: token.cumulative_text,
                finish_reason: token.finish_reason,
            }),
            xybrid_sdk::StreamEvent::Complete(result) => {
                FfiStreamEvent::Complete(FfiResult::from_inference_result(&result))
            }
            xybrid_sdk::StreamEvent::Error(e) => FfiStreamEvent::Error(e),
        }
    }
}

impl FfiModelLoader {
    #[frb(sync)]
    pub fn from_registry(model_id: String) -> FfiModelLoader {
        FfiModelLoader(ModelLoader::from_registry(&model_id))
    }

    #[frb(sync)]
    pub fn from_bundle(path: String) -> Result<FfiModelLoader, String> {
        ModelLoader::from_bundle(&path)
            .map(FfiModelLoader)
            .map_err(|e| e.to_string())
    }

    /// Load the model without progress updates.
    pub async fn load(&self) -> Result<FfiModel, String> {
        self.0
            .load_async()
            .await
            .map(|m| FfiModel(Arc::new(m)))
            .map_err(|e| e.to_string())
    }

    /// Load the model with download progress updates.
    ///
    /// Streams FfiLoadEvent during download:
    /// - `Progress(f64)` for download progress (0.0 to 1.0)
    /// - `Complete` when the model is ready
    /// - `Error(String)` if loading fails
    ///
    /// After receiving `Complete`, call `load()` to get the cached model instantly.
    pub fn load_with_progress(&self, sink: StreamSink<FfiLoadEvent>) {
        let loader = self.0.clone();

        // Run loading in a background thread to not block
        std::thread::spawn(move || {
            let result = loader.load_with_progress(|progress| {
                // Send progress as f64 (0.0 to 1.0)
                let _ = sink.add(FfiLoadEvent::Progress(progress as f64));
            });

            match result {
                Ok(_) => {
                    // Model is now cached, send complete event
                    let _ = sink.add(FfiLoadEvent::Complete);
                }
                Err(e) => {
                    let _ = sink.add(FfiLoadEvent::Error(e.to_string()));
                }
            }
        });
    }
}

impl FfiModel {
    /// Run batch inference (non-streaming).
    pub fn run(&self, envelope: super::envelope::FfiEnvelope) -> Result<FfiResult, String> {
        let result = self
            .0
            .run(&envelope.into_envelope())
            .map_err(|e| e.to_string())?;
        Ok(FfiResult::from_inference_result(&result))
    }

    /// Run inference with streaming output.
    ///
    /// Returns a stream of events:
    /// - `FfiStreamEvent::Token` for each generated token (LLM models)
    /// - `FfiStreamEvent::Complete` when inference finishes
    /// - `FfiStreamEvent::Error` if an error occurs
    ///
    /// For non-LLM models, a single Token event is emitted with the full result.
    pub fn run_stream(
        &self,
        envelope: super::envelope::FfiEnvelope,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        use tokio_stream::StreamExt;

        let model = self.0.clone();
        let env = envelope.into_envelope();

        // Spawn a background thread with its own Tokio runtime
        // (same pattern as load_with_progress which works)
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(format!(
                        "Failed to create runtime: {}",
                        e
                    )));
                    return;
                }
            };

            rt.block_on(async move {
                let mut stream = model.run_stream(env);

                while let Some(event) = stream.next().await {
                    let ffi_event = FfiStreamEvent::from(event);
                    // Send to Dart stream (ignore errors if sink is closed)
                    if sink.add(ffi_event).is_err() {
                        break;
                    }
                }
            });
        });
    }

    /// Check if this model supports true token-by-token streaming.
    ///
    /// Returns `true` for LLM models (GGUF), `false` for other model types.
    #[frb(sync)]
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    pub fn supports_token_streaming(&self) -> bool {
        self.0.supports_token_streaming()
    }
}

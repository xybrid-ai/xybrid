//! Model loading FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::sync::Arc;
use xybrid_sdk::{GenerationConfig, ModelLoader, XybridModel};

use crate::frb_generated::StreamSink;

use super::context::FfiConversationContext;
use super::result::FfiResult;

/// Generation parameters for LLM inference.
///
/// All fields are optional. When `None`, the model's default value is used.
/// This is non-opaque so FRB auto-generates a plain Dart data class.
pub struct FfiGenerationConfig {
    /// Maximum tokens to generate. Default: 2048
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 = deterministic, higher = more random). Default: 0.7
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold. Default: 0.9
    pub top_p: Option<f32>,
    /// Min-p sampling threshold. Default: 0.05
    pub min_p: Option<f32>,
    /// Top-k sampling (0 = disabled). Default: 40
    pub top_k: Option<u32>,
    /// Repetition penalty (1.0 = disabled). Default: 1.1
    pub repetition_penalty: Option<f32>,
    /// Stop sequences. When `None` or empty, only EOS token stops generation.
    pub stop_sequences: Option<Vec<String>>,
}

impl FfiGenerationConfig {
    /// Create a greedy decoding config (deterministic, temperature=0).
    #[frb(sync)]
    pub fn greedy() -> FfiGenerationConfig {
        FfiGenerationConfig {
            max_tokens: None,
            temperature: Some(0.0),
            top_p: Some(1.0),
            min_p: None,
            top_k: Some(0),
            repetition_penalty: None,
            stop_sequences: None,
        }
    }

    /// Create a creative generation config (higher temperature).
    #[frb(sync)]
    pub fn creative() -> FfiGenerationConfig {
        FfiGenerationConfig {
            max_tokens: None,
            temperature: Some(0.9),
            top_p: Some(0.95),
            min_p: None,
            top_k: Some(50),
            repetition_penalty: None,
            stop_sequences: None,
        }
    }

    pub(crate) fn to_sdk(&self) -> GenerationConfig {
        let mut config = GenerationConfig::default();
        if let Some(v) = self.max_tokens {
            config.max_tokens = v as usize;
        }
        if let Some(v) = self.temperature {
            config.temperature = v;
        }
        if let Some(v) = self.top_p {
            config.top_p = v;
        }
        if let Some(v) = self.min_p {
            config.min_p = v;
        }
        if let Some(v) = self.top_k {
            config.top_k = v as usize;
        }
        if let Some(v) = self.repetition_penalty {
            config.repetition_penalty = v;
        }
        if let Some(ref v) = self.stop_sequences {
            config.stop_sequences = v.clone();
        }
        config
    }
}

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

    #[frb(sync)]
    pub fn from_directory(path: String) -> Result<FfiModelLoader, String> {
        ModelLoader::from_directory(&path)
            .map(FfiModelLoader)
            .map_err(|e| e.to_string())
    }

    /// Create a loader for a model from a HuggingFace Hub repository.
    ///
    /// Downloads model files from HuggingFace and caches them locally.
    /// Requires the `huggingface` feature flag to be enabled.
    #[frb(sync)]
    pub fn from_huggingface(repo: String) -> FfiModelLoader {
        FfiModelLoader(ModelLoader::from_huggingface(&repo))
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
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run(
        &self,
        envelope: super::envelope::FfiEnvelope,
        config: Option<FfiGenerationConfig>,
    ) -> Result<FfiResult, String> {
        let sdk_config = config.as_ref().map(|c| c.to_sdk());
        let result = self
            .0
            .run(&envelope.into_envelope(), sdk_config.as_ref())
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
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run_stream(
        &self,
        envelope: super::envelope::FfiEnvelope,
        config: Option<FfiGenerationConfig>,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        use tokio_stream::StreamExt;

        let model = self.0.clone();
        let env = envelope.into_envelope();
        let sdk_config = config.map(|c| c.to_sdk());

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
                let mut stream = model.run_stream(env, sdk_config);

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

    /// Run inference with conversation context.
    ///
    /// The context provides conversation history which is formatted into
    /// the prompt using the model's chat template.
    ///
    /// Note: The context is NOT automatically updated with the result.
    /// Call `context.push_text(result.text(), FfiMessageRole::Assistant)` to add the response.
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run_with_context(
        &self,
        envelope: super::envelope::FfiEnvelope,
        context: &FfiConversationContext,
        config: Option<FfiGenerationConfig>,
    ) -> Result<FfiResult, String> {
        let sdk_config = config.as_ref().map(|c| c.to_sdk());
        let ctx_guard = context
            .0
            .read()
            .map_err(|e| format!("Failed to read context: {}", e))?;

        let result = self
            .0
            .run_with_context(&envelope.into_envelope(), &ctx_guard, sdk_config.as_ref())
            .map_err(|e| e.to_string())?;

        Ok(FfiResult::from_inference_result(&result))
    }

    /// Run inference with streaming output and conversation context.
    ///
    /// Combines streaming output with multi-turn conversation memory.
    /// The model sees the full conversation history when generating responses.
    ///
    /// Returns a stream of events:
    /// - `FfiStreamEvent::Token` for each generated token (LLM models)
    /// - `FfiStreamEvent::Complete` when inference finishes
    /// - `FfiStreamEvent::Error` if an error occurs
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run_stream_with_context(
        &self,
        envelope: super::envelope::FfiEnvelope,
        context: &FfiConversationContext,
        config: Option<FfiGenerationConfig>,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        let model = self.0.clone();
        let env = envelope.into_envelope();
        let ctx = context.0.clone();
        let sdk_config = config.map(|c| c.to_sdk());

        // Spawn a background thread
        std::thread::spawn(move || {
            // Get read lock on context
            let ctx_guard = match ctx.read() {
                Ok(guard) => guard,
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(format!(
                        "Failed to read context: {}",
                        e
                    )));
                    return;
                }
            };

            // Track token index for the stream
            let mut token_index = 0u32;

            // Use callback-based streaming
            let result =
                model.run_streaming_with_context(&env, &ctx_guard, sdk_config.as_ref(), |token| {
                    let ffi_token = FfiStreamToken {
                        token: token.token.clone(),
                        token_id: token.token_id,
                        index: token_index,
                        cumulative_text: token.cumulative_text.clone(),
                        finish_reason: token.finish_reason.clone(),
                    };
                    token_index += 1;
                    let _ = sink.add(FfiStreamEvent::Token(ffi_token));
                    Ok(())
                });

            match result {
                Ok(inference_result) => {
                    let ffi_result = FfiResult::from_inference_result(&inference_result);
                    let _ = sink.add(FfiStreamEvent::Complete(ffi_result));
                }
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(e.to_string()));
                }
            }
        });
    }
}

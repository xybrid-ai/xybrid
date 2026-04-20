//! Executor module - Executes model inference stages using runtime adapters.
//!
//! The Executor is the **mid-level** execution layer that maintains a registry of runtime
//! adapters and delegates inference execution to the appropriate adapter based on the target.
//!
//! See [`EXECUTION_LAYERS.md`](./EXECUTION_LAYERS.md) for the full architecture.
//!
//! ## Responsibility
//!
//! The executor handles:
//! - **Adapter registry**: Maintain available runtime adapters
//! - **Target selection**: Choose adapter based on execution target
//! - **Model execution**: Execute models from DIRECTORIES (pre-extracted)
//! - **LLM integration**: Handle cloud API calls (OpenAI, Anthropic)
//!
//! ## Architectural Boundary
//!
//! **IMPORTANT**: Core only accepts directories, NOT `.xyb` bundle files.
//! Bundle extraction must be done by SDK's `CacheManager.ensure_extracted()` before calling Core.
//!
//! ```text
//! SDK Layer                          Core Layer
//! ┌─────────────────────────┐        ┌─────────────────────────┐
//! │ CacheManager            │        │ Executor                │
//! │ - ensure_extracted()    │───────►│ - Only accepts dirs     │
//! │ - Returns directory     │        │ - Rejects .xyb files    │
//! └─────────────────────────┘        └─────────────────────────┘
//! ```
//!
//! ## Cross-Layer Execution
//!
//! The executor supports cross-layer pipelines where different stages run on different targets:
//! - **Device/Local**: On-device inference from extracted directories (via [`TemplateExecutor`])
//! - **Integration**: Third-party API calls (OpenAI, Anthropic, etc.) via [`CloudRuntimeAdapter`]
//! - **Cloud/Server**: Xybrid-hosted inference (future)

use crate::context::StageDescriptor;
use crate::execution::{ModelMetadata, TemplateExecutor};
use crate::ir::Envelope;
use crate::runtime_adapter::{AdapterError, CloudRuntimeAdapter, RuntimeAdapter};
use crate::tracing as trace;
use log::debug;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

use tokio::task;

/// Error type for executor operations.
#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("Adapter not found: {0}")]
    AdapterNotFound(String),
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Invalid target: {0}")]
    InvalidTarget(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Adapter error: {0}")]
    AdapterError(#[from] AdapterError),
    #[error("Integration error: {0}")]
    IntegrationError(String),
    #[error("Provider not configured: {0}")]
    ProviderNotConfigured(String),
    #[error("Bundle must be extracted first: {0}. Use SDK's CacheManager.ensure_extracted() before calling Core.")]
    BundleNotExtracted(String),
    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for executor operations.
pub type ExecutorResult<T> = Result<T, ExecutorError>;

/// Metadata about stage execution.
#[derive(Debug, Clone)]
pub struct StageMetadata {
    /// Name of the adapter used for execution
    pub adapter: String,
    /// Target where execution occurred (local, edge, cloud)
    pub target: String,
    /// Execution latency in milliseconds
    pub latency_ms: u128,
}

/// Executor for managing runtime adapters and executing inference stages.
///
/// The executor maintains a registry of runtime adapters and selects the
/// appropriate adapter based on the target. It handles model loading,
/// inference execution, and metadata collection.
///
/// **Note**: The executor works with pre-extracted model directories.
/// Bundle download and extraction is handled by SDK's `CacheManager` before invoking the executor.
pub struct Executor {
    /// Registry of runtime adapters by name
    adapters: HashMap<String, Arc<dyn RuntimeAdapter>>,
    /// Default adapter name for local execution
    default_local_adapter: Option<String>,
    /// Default adapter name for cloud execution
    default_cloud_adapter: Option<String>,
    /// Cached TemplateExecutor instances keyed by base_path.
    /// This avoids recreating executors (and reloading models) on every call.
    template_executor_cache: HashMap<String, TemplateExecutor>,
}

impl Clone for Executor {
    fn clone(&self) -> Self {
        Self {
            adapters: self.adapters.clone(),
            default_local_adapter: self.default_local_adapter.clone(),
            default_cloud_adapter: self.default_cloud_adapter.clone(),
            template_executor_cache: HashMap::new(), // Don't clone cache (stateful)
        }
    }
}

impl Executor {
    /// Creates a new Executor instance.
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            default_local_adapter: None,
            default_cloud_adapter: None,
            template_executor_cache: HashMap::new(),
        }
    }

    /// Registers a runtime adapter with the executor.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The runtime adapter to register (wrapped in Arc for shared ownership)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xybrid_core::executor::Executor;
    /// use xybrid_core::runtime_adapter::OnnxRuntimeAdapter;
    /// use std::sync::Arc;
    ///
    /// let mut executor = Executor::new();
    /// let adapter = Arc::new(OnnxRuntimeAdapter::new());
    /// executor.register_adapter(adapter);
    /// ```
    pub fn register_adapter(&mut self, adapter: Arc<dyn RuntimeAdapter>) {
        let name = adapter.name().to_string();
        if self.default_local_adapter.is_none() && name == "onnx" {
            self.default_local_adapter = Some(name.clone());
        }
        if self.default_cloud_adapter.is_none() && name == "cloud" {
            self.default_cloud_adapter = Some(name.clone());
        }
        self.adapters.insert(name, adapter);
    }

    /// Gets an adapter by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The adapter name (e.g., "onnx", "coreml", "cloud")
    ///
    /// # Returns
    ///
    /// `Some(adapter)` if found, `None` otherwise
    pub fn get_adapter(&self, name: &str) -> Option<&Arc<dyn RuntimeAdapter>> {
        self.adapters.get(name)
    }

    /// Executes a stage using the specified target.
    ///
    /// # Arguments
    ///
    /// * `stage` - Stage descriptor containing stage information
    /// * `input` - Input envelope containing the inference data
    /// * `target` - Target where execution should occur ("local", "edge", "cloud")
    ///
    /// # Returns
    ///
    /// A tuple containing the output envelope and stage metadata
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xybrid_core::executor::Executor;
    /// use xybrid_core::context::StageDescriptor;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind};
    ///
    /// let executor = Executor::new();
    /// let stage = StageDescriptor::new("asr");
    /// let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));
    ///
    /// let (output, metadata) = executor.execute_stage(&stage, &input, "local")?;
    /// ```
    pub fn execute_stage(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        target: &str,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
        let start_time = Instant::now();

        // Check if this is a cloud stage (third-party API like OpenAI/Anthropic)
        if stage.is_cloud() {
            return self.execute_cloud(stage, input, start_time);
        }

        // Select adapter based on target
        let adapter_name = self.select_adapter(target)?;

        // For cloud adapter, skip model loading (legacy path - prefer integration)
        if adapter_name == "cloud" {
            let adapter = self
                .get_adapter(&adapter_name)
                .ok_or_else(|| ExecutorError::AdapterNotFound(adapter_name.clone()))?;

            let output = adapter
                .execute(input)
                .map_err(ExecutorError::AdapterError)?;

            let latency_ms = start_time.elapsed().as_millis();
            let metadata = StageMetadata {
                adapter: adapter_name,
                target: target.to_string(),
                latency_ms,
            };
            return Ok((output, metadata));
        }

        // Try bundle_path for metadata-driven execution
        // IMPORTANT: Core only accepts directories, not .xyb files.
        // Bundle extraction must be done by SDK's CacheManager.ensure_extracted() before calling Core.
        if let Some(bundle_path_str) = &stage.bundle_path {
            let bundle_path = PathBuf::from(bundle_path_str);
            debug!(
                target: "xybrid_core",
                "Stage '{}' has bundle_path: {:?}",
                stage.name,
                bundle_path
            );

            // BOUNDARY ENFORCEMENT: Reject .xyb files - check extension BEFORE checking existence
            // This catches the error early even if the file doesn't exist yet
            let ext = bundle_path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if ext == "xyb" || ext == "bundle" {
                return Err(ExecutorError::BundleNotExtracted(format!(
                    "Received .xyb bundle path '{}'. Core only accepts extracted directories. \
                     Use SDK's CacheManager.ensure_extracted() to extract the bundle first.",
                    bundle_path.display()
                )));
            }

            if bundle_path.exists() {
                debug!(target: "xybrid_core", "Path extension: '{}'", ext);

                // bundle_path is a directory - check for model_metadata.json
                if bundle_path.is_dir() {
                    let metadata_path = bundle_path.join("model_metadata.json");
                    if metadata_path.exists() {
                        // Load metadata from directory
                        let metadata_content = fs::read_to_string(&metadata_path).map_err(|e| {
                            ExecutorError::Other(format!(
                                "Failed to read model_metadata.json: {}",
                                e
                            ))
                        })?;
                        let model_metadata: ModelMetadata = serde_json::from_str(&metadata_content)
                            .map_err(|e| {
                                ExecutorError::Other(format!(
                                    "Failed to parse model_metadata.json: {}",
                                    e
                                ))
                            })?;

                        debug!(
                            target: "xybrid_core",
                            "Found model_metadata.json in directory. Template: {:?}",
                            model_metadata.execution_template
                        );

                        // Use TemplateExecutor for metadata-driven inference
                        let base_path = bundle_path.to_str().ok_or_else(|| {
                            ExecutorError::Other("Invalid bundle dir path".to_string())
                        })?;

                        // Get or create cached TemplateExecutor for this base_path
                        let base_path_key = base_path.to_string();
                        if !self.template_executor_cache.contains_key(&base_path_key) {
                            debug!(
                                target: "xybrid_core",
                                "Creating new TemplateExecutor for base_path: {}",
                                base_path
                            );
                            self.template_executor_cache
                                .insert(base_path_key.clone(), TemplateExecutor::new(base_path));
                        } else {
                            debug!(
                                target: "xybrid_core",
                                "Reusing cached TemplateExecutor for base_path: {}",
                                base_path
                            );
                        }

                        let template_executor = self
                            .template_executor_cache
                            .get_mut(&base_path_key)
                            .expect("TemplateExecutor was just inserted");

                        let output = template_executor
                            .execute(&model_metadata, input, None)
                            .map_err(ExecutorError::AdapterError)?;

                        let latency_ms = start_time.elapsed().as_millis();
                        let metadata = StageMetadata {
                            adapter: "template-executor".to_string(),
                            target: target.to_string(),
                            latency_ms,
                        };

                        return Ok((output, metadata));
                    } else {
                        debug!(
                            target: "xybrid_core",
                            "Directory exists but NO model_metadata.json found at {:?}. Falling back to raw adapter.",
                            metadata_path
                        );
                    }
                }
            } else {
                debug!(
                    target: "xybrid_core",
                    "Bundle path does not exist: {:?}",
                    bundle_path
                );
            }
        } else {
            debug!(
                target: "xybrid_core",
                "Stage '{}' has no bundle_path set",
                stage.name
            );
        }

        // Raw adapter fallback for externally-preloaded adapters (test
        // harness or advanced embedders). Unlike the old code path, we
        // no longer auto-create a zero-byte mock `.onnx` file or swap
        // the `ModelNotLoaded` error for a `mock-output-<stage>-<input>`
        // envelope. If the adapter isn't pre-loaded the real
        // `ModelNotLoaded` error propagates — the user must call
        // `Pipeline::load_models()` first (or pre-load the adapter).
        debug!(
            target: "xybrid_core",
            "Stage '{}' has no bundle_path; falling back to raw adapter '{}' (adapter must be pre-loaded)",
            stage.name,
            adapter_name,
        );

        let adapter = self
            .get_adapter(&adapter_name)
            .ok_or_else(|| ExecutorError::AdapterNotFound(adapter_name.clone()))?;

        let output = adapter.execute(input).map_err(|e| match e {
            AdapterError::ModelNotLoaded(msg) => ExecutorError::Other(format!(
                "Stage '{}' has no bundle_path and the adapter is not loaded: {}. \
                 Call `Pipeline::load_models()` before `Pipeline::run()`, or pre-load \
                 the adapter with `adapter.load_model(path)` before driving the \
                 orchestrator directly.",
                stage.name, msg
            )),
            other => ExecutorError::AdapterError(other),
        })?;

        let latency_ms = start_time.elapsed().as_millis();
        let metadata = StageMetadata {
            adapter: adapter_name,
            target: target.to_string(),
            latency_ms,
        };

        Ok((output, metadata))
    }

    /// Executes a stage asynchronously using the specified target.
    ///
    /// This is an async wrapper around `execute_stage` that runs the sync
    /// adapter execution in a blocking thread pool to avoid blocking the async runtime.
    ///
    /// # Arguments
    ///
    /// * `stage` - Stage descriptor containing stage information
    /// * `input` - Input envelope containing the inference data
    /// * `target` - Target where execution should occur ("local", "edge", "cloud")
    ///
    /// # Returns
    ///
    /// A future that resolves to a tuple containing the output envelope and stage metadata
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xybrid_core::executor::Executor;
    /// use xybrid_core::context::StageDescriptor;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let executor = Executor::new();
    /// let stage = StageDescriptor::new("asr");
    /// let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));
    ///
    /// let (output, metadata) = executor.execute_stage_async(&stage, &input, "local").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_stage_async(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        target: &str,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
        // Clone what we need for the blocking task
        let stage = stage.clone();
        let input = input.clone();
        let target = target.to_string();

        // Clone executor for blocking task (temp dir won't be cloned, but that's ok)
        let mut executor = self.clone();

        task::spawn_blocking(move || executor.execute_stage(&stage, &input, &target))
            .await
            .map_err(|e| ExecutorError::Other(format!("Task join error: {}", e)))?
    }

    /// Executes a stage via third-party cloud API (OpenAI, Anthropic, etc.).
    ///
    /// This method handles cross-layer pipeline execution where a stage runs on
    /// a remote cloud provider rather than locally on-device.
    ///
    /// Delegates to [`CloudRuntimeAdapter`] after enriching the envelope with
    /// stage configuration metadata.
    ///
    /// # Arguments
    ///
    /// * `stage` - Stage descriptor with provider info
    /// * `input` - Input envelope (expects Text)
    /// * `start_time` - Timer for latency measurement
    ///
    /// # Returns
    ///
    /// Output envelope with cloud response and stage metadata
    fn execute_cloud(
        &self,
        stage: &StageDescriptor,
        input: &Envelope,
        start_time: Instant,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
        // Extract provider from stage descriptor
        let provider = stage.provider.ok_or_else(|| {
            ExecutorError::ProviderNotConfigured("Integration stage requires provider".to_string())
        })?;

        // Start tracing span for cloud execution
        let model_name = stage.model.clone().unwrap_or_else(|| "unknown".to_string());
        let _exec_span = trace::SpanGuard::new(format!("execute:{}", model_name));
        trace::add_metadata("provider", provider.as_str());
        trace::add_metadata("target", "cloud");
        if let Some(ref model) = stage.model {
            trace::add_metadata("model", model);
        }

        // Enrich envelope with stage configuration for CloudRuntimeAdapter
        let mut enriched_input = input.clone();
        enriched_input
            .metadata
            .insert("provider".to_string(), provider.as_str().to_string());

        if let Some(ref model) = stage.model {
            enriched_input
                .metadata
                .insert("model".to_string(), model.clone());
        }

        // Apply stage options to metadata
        if let Some(ref options) = stage.options {
            if let Some(backend) = options.get::<String>("backend") {
                enriched_input
                    .metadata
                    .insert("backend".to_string(), backend);
            }
            if let Some(gateway_url) = options.get::<String>("gateway_url") {
                enriched_input
                    .metadata
                    .insert("gateway_url".to_string(), gateway_url);
            }
            if let Some(api_key) = options.get::<String>("api_key") {
                enriched_input
                    .metadata
                    .insert("api_key".to_string(), api_key);
            }
            if let Some(timeout) = options.timeout_ms() {
                enriched_input
                    .metadata
                    .insert("timeout_ms".to_string(), timeout.to_string());
            }
            if let Some(debug) = options.get::<bool>("debug") {
                enriched_input
                    .metadata
                    .insert("debug".to_string(), debug.to_string());
            }
            if let Some(system) = options.system_prompt() {
                enriched_input
                    .metadata
                    .insert("system_prompt".to_string(), system);
            }
            if let Some(temp) = options.temperature() {
                enriched_input
                    .metadata
                    .insert("temperature".to_string(), temp.to_string());
            }
            if let Some(max) = options.max_tokens() {
                enriched_input
                    .metadata
                    .insert("max_tokens".to_string(), max.to_string());
            }
        }

        // Use registered cloud adapter or create a new one
        let output = if let Some(adapter) = self.get_adapter("cloud") {
            adapter
                .execute(&enriched_input)
                .map_err(ExecutorError::AdapterError)?
        } else {
            // Create a temporary adapter if none registered
            let adapter = CloudRuntimeAdapter::new();
            adapter
                .execute(&enriched_input)
                .map_err(ExecutorError::AdapterError)?
        };

        // Calculate latency
        let latency_ms = start_time.elapsed().as_millis();

        // Build metadata (include backend info from output)
        let backend_info = output
            .metadata
            .get("backend")
            .cloned()
            .unwrap_or_else(|| "gateway".to_string());
        let metadata = StageMetadata {
            adapter: format!("cloud:{}:{}", provider, backend_info),
            target: "cloud".to_string(),
            latency_ms,
        };

        Ok((output, metadata))
    }

    /// Selects an adapter name based on the target.
    ///
    /// # Arguments
    ///
    /// * `target` - Target string ("local", "edge", "cloud")
    ///
    /// # Returns
    ///
    /// Adapter name to use
    fn select_adapter(&self, target: &str) -> ExecutorResult<String> {
        match target {
            "local" => {
                // Prefer ONNX for local execution
                if let Some(name) = &self.default_local_adapter {
                    if self.adapters.contains_key(name) {
                        return Ok(name.clone());
                    }
                }
                // Fallback to first available adapter
                self.adapters
                    .keys()
                    .next()
                    .ok_or_else(|| {
                        ExecutorError::AdapterNotFound("No adapters registered".to_string())
                    })
                    .cloned()
            }
            "cloud" => {
                // Prefer cloud adapter if available
                if let Some(name) = &self.default_cloud_adapter {
                    if self.adapters.contains_key(name) {
                        return Ok(name.clone());
                    }
                }
                // Fallback to first available adapter
                self.adapters
                    .keys()
                    .next()
                    .ok_or_else(|| {
                        ExecutorError::AdapterNotFound("No adapters registered".to_string())
                    })
                    .cloned()
            }
            "edge" => {
                // Edge is similar to local, prefer ONNX
                if let Some(name) = &self.default_local_adapter {
                    if self.adapters.contains_key(name) {
                        return Ok(name.clone());
                    }
                }
                self.adapters
                    .keys()
                    .next()
                    .ok_or_else(|| {
                        ExecutorError::AdapterNotFound("No adapters registered".to_string())
                    })
                    .cloned()
            }
            _ => Err(ExecutorError::InvalidTarget(format!(
                "Unknown target: {}",
                target
            ))),
        }
    }

    /// Lists all registered adapter names.
    ///
    /// # Returns
    ///
    /// Vector of adapter names
    pub fn list_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::EnvelopeKind;
    use crate::runtime_adapter::OnnxRuntimeAdapter;
    use crate::testing::mocks::MockRuntimeAdapter;
    use std::sync::Arc;

    /// Create a test executor with a mock adapter (returns text output).
    fn create_test_executor() -> Executor {
        let mut executor = Executor::new();
        let mut adapter = MockRuntimeAdapter::with_text_output("mock output");
        adapter.load_model("/mock/model.onnx").unwrap();
        executor.register_adapter(Arc::new(adapter));
        executor
    }

    /// Create a test executor with a real ONNX adapter (for tests that need adapter metadata).
    fn create_onnx_executor() -> Executor {
        let mut executor = Executor::new();
        let adapter = Arc::new(OnnxRuntimeAdapter::new());
        executor.register_adapter(adapter);
        executor
    }

    #[test]
    fn test_executor_creation() {
        let executor = Executor::new();
        assert!(executor.list_adapters().is_empty());
    }

    #[test]
    fn test_register_adapter() {
        let mut executor = Executor::new();
        let adapter = Arc::new(OnnxRuntimeAdapter::new());
        executor.register_adapter(adapter);

        let adapters = executor.list_adapters();
        assert_eq!(adapters.len(), 1);
        assert!(adapters.contains(&"onnx".to_string()));
    }

    #[test]
    fn test_get_adapter() {
        let mut executor = Executor::new();
        let adapter = Arc::new(OnnxRuntimeAdapter::new());
        executor.register_adapter(adapter);

        let retrieved = executor.get_adapter("onnx");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "onnx");

        let not_found = executor.get_adapter("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_execute_stage_local() -> ExecutorResult<()> {
        let mut executor = Executor::new();

        // Create and register mock adapter that returns text (simulates ASR)
        let mut adapter = MockRuntimeAdapter::with_text_output("transcribed text");
        adapter.load_model("/mock/asr.onnx")?;
        executor.register_adapter(Arc::new(adapter));

        let stage = StageDescriptor::new("asr");
        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));

        let (output, metadata) = executor.execute_stage(&stage, &input, "local")?;

        // Verify output
        assert_eq!(output.kind_str(), "Text"); // Mock returns text (simulating ASR)
        assert_eq!(metadata.target, "local");
        assert_eq!(metadata.adapter, "mock");
        Ok(())
    }

    /// Regression test for A1 — `execute_stage` used to silently return
    /// `mock-output-<stage>-<input>` envelopes when the adapter was
    /// registered but not loaded and the stage had no bundle_path. That
    /// made `Pipeline::run()` look like it worked end-to-end even though
    /// every stage was producing fake text. The fix surfaces the real
    /// `ModelNotLoaded` error (wrapped with a message pointing at
    /// `Pipeline::load_models()`) instead.
    #[test]
    fn test_execute_stage_unloaded_adapter_errors_instead_of_mocking() {
        let mut executor = Executor::new();
        // Register adapter WITHOUT calling load_model — simulates the
        // post-A1 state where Pipeline::run drives the orchestrator
        // without first running load_models(). The mock's `execute()`
        // returns `AdapterError::ModelNotLoaded` when `is_loaded` is
        // false, which is exactly the path the old fallback masked.
        let adapter = MockRuntimeAdapter::with_text_output("(never returned)");
        executor.register_adapter(Arc::new(adapter));

        let stage = StageDescriptor::new("asr");
        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));

        let result = executor.execute_stage(&stage, &input, "local");

        let err = result.expect_err(
            "execute_stage must surface the real ModelNotLoaded error; \
             previously it returned `mock-output-asr-...` and silently succeeded",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("load_models") || msg.contains("not loaded"),
            "error should point the user at Pipeline::load_models(), got: {msg}"
        );
        // And obviously not the old mock envelope.
        assert!(
            !msg.contains("mock-output"),
            "error must not leak the old mock fallback string"
        );
    }

    #[test]
    fn test_execute_stage_cloud_target() -> ExecutorResult<()> {
        let mut executor = Executor::new();

        // Create and register mock adapter
        let mut adapter = MockRuntimeAdapter::with_text_output("cloud response");
        adapter.load_model("/mock/model.onnx")?;
        executor.register_adapter(Arc::new(adapter));

        let stage = StageDescriptor::new("motivator");
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let (_output, metadata) = executor.execute_stage(&stage, &input, "cloud")?;

        // Cloud target still routes through the adapter (no provider set = not integration)
        assert_eq!(metadata.target, "cloud");

        Ok(())
    }

    #[test]
    fn test_execute_stage_no_adapter() {
        let mut executor = Executor::new();
        let stage = StageDescriptor::new("test");
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = executor.execute_stage(&stage, &input, "local");
        assert!(matches!(result, Err(ExecutorError::AdapterNotFound(_))));
    }

    #[test]
    fn test_execute_stage_invalid_target() {
        let mut executor = create_test_executor();
        let stage = StageDescriptor::new("test");
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = executor.execute_stage(&stage, &input, "invalid_target");
        assert!(matches!(result, Err(ExecutorError::InvalidTarget(_))));
    }

    #[test]
    fn test_list_adapters() {
        let mut executor = Executor::new();
        let adapter1 = Arc::new(OnnxRuntimeAdapter::new());
        executor.register_adapter(adapter1);

        let adapters = executor.list_adapters();
        assert_eq!(adapters.len(), 1);
        assert!(adapters.contains(&"onnx".to_string()));
    }

    #[test]
    fn test_select_adapter() {
        let mut executor = Executor::new();
        let adapter = Arc::new(OnnxRuntimeAdapter::new());
        executor.register_adapter(adapter);

        // Test local target
        let adapter_name = executor.select_adapter("local").unwrap();
        assert_eq!(adapter_name, "onnx");

        // Test cloud target
        let adapter_name = executor.select_adapter("cloud").unwrap();
        assert_eq!(adapter_name, "onnx");

        // Test invalid target
        let result = executor.select_adapter("invalid");
        assert!(matches!(result, Err(ExecutorError::InvalidTarget(_))));
    }

    // ============================================================================
    // Bundle Extraction Unique Naming Tests
    // ============================================================================

    /// Test that bundles with the same filename from different directories
    /// get unique extraction paths. This prevents collision when multiple
    /// bundles are named "universal.xyb".
    #[test]
    fn test_bundle_extraction_unique_naming() {
        use std::path::Path;

        // Helper function that mirrors the naming logic in extract_bundle_with_metadata
        fn compute_unique_name(bundle_path: &Path) -> String {
            let parent_name = bundle_path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let bundle_stem = bundle_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("bundle");
            format!("{}_{}", parent_name, bundle_stem)
        }

        // Simulate two bundles with same filename but different parent directories
        let whisper_bundle = Path::new("/cache/models/whisper-tiny/universal.xyb");
        let qwen_bundle = Path::new("/cache/models/Qwen2.5-0.5B-Instruct-GGUF/universal.xyb");
        let kokoro_bundle = Path::new("/cache/models/Kokoro-82M-v1.0-ONNX/universal.xyb");

        let whisper_name = compute_unique_name(whisper_bundle);
        let qwen_name = compute_unique_name(qwen_bundle);
        let kokoro_name = compute_unique_name(kokoro_bundle);

        // All names should be different
        assert_ne!(
            whisper_name, qwen_name,
            "whisper and qwen should have different names"
        );
        assert_ne!(
            whisper_name, kokoro_name,
            "whisper and kokoro should have different names"
        );
        assert_ne!(
            qwen_name, kokoro_name,
            "qwen and kokoro should have different names"
        );

        // Names should include parent directory name
        assert!(
            whisper_name.contains("whisper"),
            "Name should contain parent dir: {}",
            whisper_name
        );
        assert!(
            qwen_name.contains("Qwen"),
            "Name should contain parent dir: {}",
            qwen_name
        );
        assert!(
            kokoro_name.contains("Kokoro"),
            "Name should contain parent dir: {}",
            kokoro_name
        );

        // Names should include bundle stem
        assert!(
            whisper_name.contains("universal"),
            "Name should contain bundle stem: {}",
            whisper_name
        );
    }

    #[test]
    fn test_bundle_extraction_handles_missing_parent() {
        use std::path::Path;

        fn compute_unique_name(bundle_path: &Path) -> String {
            let parent_name = bundle_path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let bundle_stem = bundle_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("bundle");
            format!("{}_{}", parent_name, bundle_stem)
        }

        // Edge case: bundle at root level
        let root_bundle = Path::new("universal.xyb");
        let name = compute_unique_name(root_bundle);

        // Should use "unknown" for missing parent
        assert!(
            name.contains("unknown") || name.contains("universal"),
            "Should handle missing parent gracefully: {}",
            name
        );
    }

    #[test]
    fn test_bundle_extraction_different_bundle_names() {
        use std::path::Path;

        fn compute_unique_name(bundle_path: &Path) -> String {
            let parent_name = bundle_path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let bundle_stem = bundle_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("bundle");
            format!("{}_{}", parent_name, bundle_stem)
        }

        // Different bundle names in same directory
        let bundle1 = Path::new("/models/model_a.xyb");
        let bundle2 = Path::new("/models/model_b.xyb");

        let name1 = compute_unique_name(bundle1);
        let name2 = compute_unique_name(bundle2);

        assert_ne!(
            name1, name2,
            "Different bundle names should produce different extract dirs"
        );
        assert!(
            name1.contains("model_a"),
            "Name should contain bundle stem: {}",
            name1
        );
        assert!(
            name2.contains("model_b"),
            "Name should contain bundle stem: {}",
            name2
        );
    }

    // ============================================================================
    // Boundary Enforcement Tests
    // ============================================================================
    // These tests enforce the architectural boundary:
    // - Core only accepts DIRECTORIES (extracted bundles)
    // - SDK is responsible for extracting .xyb bundles via CacheManager
    // ============================================================================

    #[test]
    fn test_boundary_rejects_xyb_bundle_file() {
        let mut executor = create_test_executor();

        // Create a stage with a .xyb bundle path (should be rejected)
        let mut stage = StageDescriptor::new("test-model");
        stage.bundle_path = Some("/path/to/model.xyb".to_string());

        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = executor.execute_stage(&stage, &input, "local");

        // Should fail with BundleNotExtracted error
        assert!(
            matches!(result, Err(ExecutorError::BundleNotExtracted(_))),
            "Expected BundleNotExtracted error for .xyb file, got: {:?}",
            result
        );

        // Error message should mention SDK's CacheManager
        if let Err(ExecutorError::BundleNotExtracted(msg)) = result {
            assert!(
                msg.contains("CacheManager"),
                "Error should mention CacheManager: {}",
                msg
            );
        }
    }

    #[test]
    fn test_boundary_rejects_bundle_extension() {
        let mut executor = create_test_executor();

        // Test with .bundle extension too
        let mut stage = StageDescriptor::new("test-model");
        stage.bundle_path = Some("/path/to/model.bundle".to_string());

        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = executor.execute_stage(&stage, &input, "local");

        assert!(
            matches!(result, Err(ExecutorError::BundleNotExtracted(_))),
            "Expected BundleNotExtracted error for .bundle file, got: {:?}",
            result
        );
    }

    #[test]
    fn test_boundary_accepts_directory_path() {
        use tempfile::TempDir;

        let mut executor = create_test_executor();

        // Create a temp directory with model_metadata.json
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        // Create a minimal model_metadata.json
        let metadata = r#"{
            "model_id": "test-model",
            "version": "1.0",
            "execution_template": { "type": "Onnx", "model_file": "model.onnx" },
            "preprocessing": [],
            "postprocessing": [],
            "files": ["model.onnx"],
            "metadata": {}
        }"#;
        std::fs::write(model_dir.join("model_metadata.json"), metadata).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake onnx").unwrap();

        // Create a stage with a directory path (should be accepted)
        let mut stage = StageDescriptor::new("test-model");
        stage.bundle_path = Some(model_dir.to_str().unwrap().to_string());

        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        // This will fail during actual execution (no real model), but it should NOT
        // fail with BundleNotExtracted - that boundary check should pass
        let result = executor.execute_stage(&stage, &input, "local");

        // Should NOT be a BundleNotExtracted error
        assert!(
            !matches!(result, Err(ExecutorError::BundleNotExtracted(_))),
            "Directory paths should be accepted, not rejected as bundles: {:?}",
            result
        );
    }

    #[test]
    fn test_boundary_error_message_is_actionable() {
        // Verify the error message tells developers exactly what to do
        let error = ExecutorError::BundleNotExtracted("/path/to/bundle.xyb".to_string());
        let msg = error.to_string();

        // Should contain actionable guidance
        assert!(msg.contains("CacheManager"), "Should mention CacheManager");
        assert!(
            msg.contains("ensure_extracted"),
            "Should mention ensure_extracted()"
        );
        assert!(msg.contains("SDK"), "Should mention SDK layer");
    }
}

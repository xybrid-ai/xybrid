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

impl ExecutorError {
    pub fn cloud_fallback_abort_reason(&self) -> Option<crate::abort::AbortReason> {
        match self {
            Self::AdapterError(error) => error.cloud_fallback_abort_reason(),
            _ => None,
        }
    }
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
    /// ```no_run
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
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// use xybrid_core::executor::Executor;
    /// use xybrid_core::context::StageDescriptor;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind};
    ///
    /// let mut executor = Executor::new();
    /// let stage = StageDescriptor::new("asr");
    /// let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));
    ///
    /// let (output, metadata) = executor.execute_stage(&stage, &input, "local")?;
    /// # let _ = (output, metadata);
    /// # Ok(())
    /// # }
    /// ```
    pub fn execute_stage(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        target: &str,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
        let prepared = prepare_stage_input(stage, input)?;
        self.execute_prepared(stage, &prepared, target)
    }

    /// Executes a stage whose input has already been through
    /// [`prepare_stage_input`].
    ///
    /// The orchestrator prepares once, evaluates policy against the prepared
    /// envelope, and then calls this so the merge is not repeated. Dispatch
    /// is driven by `target` alone — the routing decision is authoritative.
    /// A stage carrying a `provider` is *not* automatically a cloud stage:
    /// when it is routed `local` it must have a usable local bundle, and it
    /// never falls through to a different model or to cloud.
    pub fn execute_prepared(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        target: &str,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
        let start_time = Instant::now();
        match target {
            "cloud" => {
                if stage.provider.is_some() {
                    return self.execute_cloud(stage, input, start_time);
                }
                self.execute_registered_cloud_adapter(stage, input, target, start_time)
            }
            "local" | "edge" => self.execute_local(stage, input, target, start_time),
            other => Err(ExecutorError::InvalidTarget(format!(
                "Unknown target: {}",
                other
            ))),
        }
    }

    /// Provider-free stage routed to cloud: only an explicitly registered
    /// cloud adapter may serve it. This never falls back to a local adapter.
    fn execute_registered_cloud_adapter(
        &self,
        stage: &StageDescriptor,
        input: &Envelope,
        target: &str,
        start_time: Instant,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
        let adapter_name = self
            .default_cloud_adapter
            .clone()
            .filter(|name| self.adapters.contains_key(name))
            .ok_or_else(|| {
                ExecutorError::AdapterNotFound(format!(
                    "stage '{}' was routed to cloud but has no provider and no cloud adapter \
                     is registered",
                    stage.name
                ))
            })?;
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
        Ok((output, metadata))
    }

    /// Local execution: an extracted model directory drives the
    /// [`TemplateExecutor`]; provider-free stages without a bundle may fall
    /// back to a pre-loaded raw adapter.
    fn execute_local(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        target: &str,
        start_time: Instant,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
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

            // A hybrid stage (local bundle + cloud provider) must not fall
            // through to an unrelated adapter when its bundle is unusable —
            // that would silently run a different model.
            if stage.provider.is_some() {
                return Err(ExecutorError::ExecutionFailed(format!(
                    "stage '{}' was routed local but its bundle at '{}' is not an extracted \
                     model directory (missing model_metadata.json); not falling back to \
                     another model or to cloud",
                    stage.name,
                    bundle_path.display()
                )));
            }
        } else {
            debug!(
                target: "xybrid_core",
                "Stage '{}' has no bundle_path set",
                stage.name
            );
            if stage.provider.is_some() {
                return Err(ExecutorError::ExecutionFailed(format!(
                    "stage '{}' was routed local but has no local bundle (cloud leg denied by \
                     policy?)",
                    stage.name
                )));
            }
        }

        // Raw adapter fallback for externally-preloaded adapters (test
        // harness or advanced embedders). Unlike the old code path, we
        // no longer auto-create a zero-byte mock `.onnx` file or swap
        // the `ModelNotLoaded` error for a `mock-output-<stage>-<input>`
        // envelope. If the adapter isn't pre-loaded the real
        // `ModelNotLoaded` error propagates — the user must call
        // `Pipeline::load_models()` first (or pre-load the adapter).
        let adapter_name = self.select_local_adapter()?;
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
    /// ```no_run
    /// # async fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// use xybrid_core::executor::Executor;
    /// use xybrid_core::context::StageDescriptor;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind};
    ///
    /// let mut executor = Executor::new();
    /// let stage = StageDescriptor::new("asr");
    /// let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));
    ///
    /// let (output, metadata) = executor.execute_stage_async(&stage, &input, "local").await?;
    /// # let _ = (output, metadata);
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
    /// the stage's *transport* configuration. Shared generation options
    /// (`system_prompt`, `temperature`, `max_tokens`, `top_p`) were already
    /// applied by [`prepare_stage_input`] and are not overwritten here.
    ///
    /// # Arguments
    ///
    /// * `stage` - Stage descriptor with provider info
    /// * `input` - Prepared input envelope (expects Text)
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

        // The model sent to the provider. A hybrid stage names its local
        // bundle in `model` and its cloud model in the `cloud_model` option.
        let effective_model = stage
            .options
            .as_ref()
            .and_then(|options| options.get::<String>("cloud_model"))
            .or_else(|| stage.model.clone());

        // Start tracing span for cloud execution
        let model_name = effective_model
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let _exec_span = trace::SpanGuard::new(format!("execute:{}", model_name));
        trace::add_metadata("provider", provider.as_str());
        trace::add_metadata("target", "cloud");
        if let Some(ref model) = effective_model {
            trace::add_metadata("model", model);
        }

        // Enrich envelope with transport configuration for CloudRuntimeAdapter
        let mut enriched_input = input.clone();
        enriched_input
            .metadata
            .insert("provider".to_string(), provider.as_str().to_string());

        if let Some(model) = effective_model {
            enriched_input.metadata.insert("model".to_string(), model);
        }

        if let Some(ref options) = stage.options {
            for key in ["backend", "gateway_url", "api_key"] {
                if let Some(value) = options.get::<String>(key) {
                    enriched_input.metadata.insert(key.to_string(), value);
                }
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
            // Provider-specific request capability; the adapter validates
            // the value and the provider it is used with.
            match options.values.get("thinking") {
                None => {}
                Some(serde_json::Value::String(mode)) => {
                    enriched_input
                        .metadata
                        .insert("thinking".to_string(), mode.clone());
                }
                Some(other) => {
                    return Err(ExecutorError::Other(format!(
                        "stage '{}': option 'thinking' must be a string (enabled|disabled), got {}",
                        stage.name, other
                    )));
                }
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

    /// Selects the adapter for local execution when no bundle drives a
    /// [`TemplateExecutor`].
    ///
    /// Prefers the default local adapter (`onnx`); otherwise the
    /// alphabetically first registered adapter that is not the cloud adapter,
    /// so the choice is deterministic and a cloud adapter never serves a
    /// local route.
    fn select_local_adapter(&self) -> ExecutorResult<String> {
        if let Some(name) = &self.default_local_adapter {
            if self.adapters.contains_key(name) {
                return Ok(name.clone());
            }
        }
        let mut candidates: Vec<&String> = self
            .adapters
            .keys()
            .filter(|name| {
                name.as_str() != "cloud" && Some(*name) != self.default_cloud_adapter.as_ref()
            })
            .collect();
        candidates.sort();
        candidates
            .first()
            .map(|name| (*name).clone())
            .ok_or_else(|| {
                ExecutorError::AdapterNotFound("No local adapters registered".to_string())
            })
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

/// Generation options a stage may declare in pipeline YAML that apply to
/// *both* legs of a hybrid stage, and the string form the backends parse
/// from envelope metadata.
const SHARED_GENERATION_OPTIONS: [&str; 4] =
    ["system_prompt", "temperature", "max_tokens", "top_p"];

/// Apply a stage's shared generation options to the input envelope.
///
/// Precedence, highest first: metadata already on the input (caller or CLI
/// overrides), then the stage's YAML options, then the model template's own
/// defaults (left to the backend). Unrelated metadata and the payload are
/// preserved. A recognised option with an invalid value is an error, never
/// silently dropped. Both the local [`TemplateExecutor`] and the cloud
/// adapter read the resulting metadata keys, so a hybrid stage generates
/// with the same settings on either leg.
pub fn prepare_stage_input(stage: &StageDescriptor, input: &Envelope) -> ExecutorResult<Envelope> {
    let Some(options) = stage.options.as_ref() else {
        return Ok(input.clone());
    };
    let mut prepared = input.clone();
    for key in SHARED_GENERATION_OPTIONS {
        let Some(value) = options.values.get(key) else {
            continue;
        };
        let rendered = render_generation_option(&stage.name, key, value)?;
        prepared.metadata.entry(key.to_string()).or_insert(rendered);
    }
    Ok(prepared)
}

fn render_generation_option(
    stage: &str,
    key: &str,
    value: &serde_json::Value,
) -> ExecutorResult<String> {
    use serde_json::Value;
    let invalid = |expected: &str| {
        ExecutorError::Other(format!(
            "stage '{stage}': option '{key}' must be {expected}, got {value}"
        ))
    };
    match key {
        "system_prompt" => match value {
            Value::String(text) => Ok(text.clone()),
            _ => Err(invalid("a string")),
        },
        "temperature" | "top_p" => match value.as_f64() {
            Some(number) if number.is_finite() && number >= 0.0 => Ok(number.to_string()),
            _ => Err(invalid("a non-negative number")),
        },
        "max_tokens" => match value.as_u64() {
            Some(count) if count > 0 => Ok(count.to_string()),
            _ => Err(invalid("a positive integer")),
        },
        other => Err(invalid(&format!("a known option (internal: '{other}')"))),
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

        // A provider-free cloud route is served only by an adapter registered
        // under the "cloud" name.
        let mut adapter = MockRuntimeAdapter::with_text_output("cloud response").with_name("cloud");
        adapter.load_model("/mock/model.onnx")?;
        executor.register_adapter(Arc::new(adapter));

        let stage = StageDescriptor::new("motivator");
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let (_output, metadata) = executor.execute_stage(&stage, &input, "cloud")?;

        assert_eq!(metadata.target, "cloud");
        assert_eq!(metadata.adapter, "cloud");

        Ok(())
    }

    #[test]
    fn cloud_target_without_provider_requires_registered_cloud_adapter() {
        // Only a local adapter is registered: a provider-free cloud route must
        // error rather than quietly run on the local adapter.
        let mut executor = Executor::new();
        let mut local = MockRuntimeAdapter::with_text_output("local").with_name("onnx");
        local.load_model("/mock/model.onnx").unwrap();
        let local = Arc::new(local);
        executor.register_adapter(local.clone());

        let stage = StageDescriptor::new("motivator");
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let result = executor.execute_stage(&stage, &input, "cloud");

        assert!(
            matches!(result, Err(ExecutorError::AdapterNotFound(_))),
            "got {result:?}"
        );
        assert_eq!(
            local.call_count(),
            0,
            "local adapter must not serve a cloud route"
        );
    }

    #[test]
    fn local_selection_never_picks_the_cloud_adapter() {
        let mut executor = Executor::new();
        let mut cloud = MockRuntimeAdapter::with_text_output("cloud").with_name("cloud");
        cloud.load_model("/mock/cloud").unwrap();
        let cloud = Arc::new(cloud);
        executor.register_adapter(cloud.clone());

        let stage = StageDescriptor::new("asr");
        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 16]));

        let result = executor.execute_stage(&stage, &input, "local");

        assert!(
            matches!(result, Err(ExecutorError::AdapterNotFound(_))),
            "got {result:?}"
        );
        assert_eq!(cloud.call_count(), 0);
    }

    #[test]
    fn local_fallback_adapter_selection_is_deterministic() {
        // No "onnx" adapter: the alphabetically-first non-cloud adapter wins,
        // regardless of HashMap iteration order.
        let mut executor = Executor::new();
        for name in ["zeta", "alpha", "cloud", "mid"] {
            let mut adapter = MockRuntimeAdapter::with_text_output(name).with_name(name);
            adapter.load_model("/mock/model").unwrap();
            executor.register_adapter(Arc::new(adapter));
        }

        assert_eq!(executor.select_local_adapter().unwrap(), "alpha");

        let stage = StageDescriptor::new("asr");
        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 16]));
        let (output, metadata) = executor.execute_stage(&stage, &input, "local").unwrap();
        assert_eq!(metadata.adapter, "alpha");
        assert_eq!(output.as_text(), Some("alpha"));
    }

    fn hybrid_stage(bundle_path: Option<&str>) -> StageDescriptor {
        let mut options = crate::pipeline::StageOptions::new();
        options.set("cloud_model", "deepseek-flash");
        options.set("system_prompt", "Be terse.");
        options.set("temperature", 0.0);
        options.set("max_tokens", 16);
        options.set("thinking", "disabled");
        options.set("gateway_url", "http://127.0.0.1:9/v1");
        options.set("api_key", "$DEEPSEEK_API_KEY");
        let mut stage = StageDescriptor::new("llm")
            .with_model("functiongemma-270m-it")
            .with_target(crate::pipeline::ExecutionTarget::Auto)
            .with_provider(crate::pipeline::IntegrationProvider::DeepSeek)
            .with_options(options);
        stage.bundle_path = bundle_path.map(str::to_string);
        stage
    }

    #[test]
    fn cloud_target_with_provider_uses_cloud_adapter_and_cloud_model() {
        let mut executor = Executor::new();
        let mut cloud =
            MockRuntimeAdapter::with_text_output("FAKE_DEEPSEEK_REPLY").with_name("cloud");
        cloud.load_model("/mock/cloud").unwrap();
        let cloud = Arc::new(cloud);
        executor.register_adapter(cloud.clone());
        let mut local = MockRuntimeAdapter::with_text_output("local").with_name("onnx");
        local.load_model("/mock/model.onnx").unwrap();
        let local = Arc::new(local);
        executor.register_adapter(local.clone());

        let stage = hybrid_stage(None);
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let (output, metadata) = executor.execute_stage(&stage, &input, "cloud").unwrap();

        assert_eq!(output.as_text(), Some("FAKE_DEEPSEEK_REPLY"));
        assert_eq!(metadata.adapter, "cloud:deepseek:gateway");
        assert_eq!(metadata.target, "cloud");
        assert_eq!(local.call_count(), 0);

        let sent = cloud.captured_inputs();
        assert_eq!(sent.len(), 1);
        let meta = &sent[0].metadata;
        // Effective cloud model, not the local bundle id.
        assert_eq!(
            meta.get("model").map(String::as_str),
            Some("deepseek-flash")
        );
        assert_eq!(meta.get("provider").map(String::as_str), Some("deepseek"));
        assert_eq!(meta.get("thinking").map(String::as_str), Some("disabled"));
        assert_eq!(
            meta.get("gateway_url").map(String::as_str),
            Some("http://127.0.0.1:9/v1")
        );
        assert_eq!(
            meta.get("api_key").map(String::as_str),
            Some("$DEEPSEEK_API_KEY")
        );
        // Shared generation options arrive via prepare_stage_input.
        assert_eq!(
            meta.get("system_prompt").map(String::as_str),
            Some("Be terse.")
        );
        assert_eq!(meta.get("temperature").map(String::as_str), Some("0"));
        assert_eq!(meta.get("max_tokens").map(String::as_str), Some("16"));
    }

    #[test]
    fn local_target_with_provider_but_no_bundle_errors_without_calling_adapters() {
        let mut executor = Executor::new();
        let mut local = MockRuntimeAdapter::with_text_output("local").with_name("onnx");
        local.load_model("/mock/model.onnx").unwrap();
        let local = Arc::new(local);
        executor.register_adapter(local.clone());
        let mut cloud = MockRuntimeAdapter::with_text_output("cloud").with_name("cloud");
        cloud.load_model("/mock/cloud").unwrap();
        let cloud = Arc::new(cloud);
        executor.register_adapter(cloud.clone());

        let stage = hybrid_stage(None);
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let err = executor
            .execute_stage(&stage, &input, "local")
            .expect_err("hybrid stage without a bundle cannot run locally");

        assert!(
            matches!(err, ExecutorError::ExecutionFailed(_)),
            "got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("routed local but has no local bundle"),
            "{msg}"
        );
        assert!(msg.contains("llm"), "{msg}");
        assert_eq!(
            local.call_count(),
            0,
            "raw adapter must not substitute for the bundle"
        );
        assert_eq!(cloud.call_count(), 0, "cloud must not be retried");
    }

    #[test]
    fn local_target_with_provider_and_invalid_bundle_errors() {
        let mut executor = Executor::new();
        let mut local = MockRuntimeAdapter::with_text_output("local").with_name("onnx");
        local.load_model("/mock/model.onnx").unwrap();
        let local = Arc::new(local);
        executor.register_adapter(local.clone());

        // An existing directory with no model_metadata.json is not a bundle.
        let empty_dir = tempfile::tempdir().unwrap();
        let stage = hybrid_stage(Some(empty_dir.path().to_str().unwrap()));
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let err = executor
            .execute_stage(&stage, &input, "local")
            .expect_err("invalid hybrid bundle must not fall through");
        assert!(
            matches!(err, ExecutorError::ExecutionFailed(_)),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("missing model_metadata.json"),
            "{err}"
        );
        assert_eq!(local.call_count(), 0);

        // A missing directory is rejected the same way.
        let stage = hybrid_stage(Some("/definitely/not/here"));
        let err = executor
            .execute_stage(&stage, &input, "local")
            .expect_err("missing hybrid bundle must not fall through");
        assert!(
            matches!(err, ExecutorError::ExecutionFailed(_)),
            "got {err:?}"
        );
        assert_eq!(local.call_count(), 0);
    }

    #[test]
    fn prepare_stage_input_applies_stage_options_when_input_is_silent() {
        let stage = hybrid_stage(None);
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let prepared = prepare_stage_input(&stage, &input).unwrap();

        assert_eq!(prepared.as_text(), Some("Hello"));
        assert_eq!(
            prepared.metadata.get("system_prompt").map(String::as_str),
            Some("Be terse.")
        );
        assert_eq!(
            prepared.metadata.get("temperature").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            prepared.metadata.get("max_tokens").map(String::as_str),
            Some("16")
        );
        // Transport keys are NOT shared generation options.
        assert!(!prepared.metadata.contains_key("gateway_url"));
        assert!(!prepared.metadata.contains_key("cloud_model"));
        assert!(!prepared.metadata.contains_key("thinking"));
        // The prepared values are what the local LLM strategy parses.
        let params = crate::execution::strategies::LlmGenerationParams::from_envelope_metadata(
            &prepared.metadata,
        );
        assert_eq!(params.max_tokens, 16);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.system_prompt.as_deref(), Some("Be terse."));
    }

    #[test]
    fn prepare_stage_input_keeps_explicit_input_metadata() {
        let stage = hybrid_stage(None);
        let mut input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
        input
            .metadata
            .insert("max_tokens".to_string(), "512".to_string());
        input
            .metadata
            .insert("unrelated".to_string(), "kept".to_string());

        let prepared = prepare_stage_input(&stage, &input).unwrap();

        // Caller/CLI override wins over YAML.
        assert_eq!(
            prepared.metadata.get("max_tokens").map(String::as_str),
            Some("512")
        );
        // YAML still fills the keys the caller left unset.
        assert_eq!(
            prepared.metadata.get("temperature").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            prepared.metadata.get("unrelated").map(String::as_str),
            Some("kept")
        );

        // No options at all: the input is returned untouched.
        let bare = StageDescriptor::new("bare");
        let untouched = prepare_stage_input(&bare, &input).unwrap();
        assert_eq!(untouched.metadata, input.metadata);
    }

    #[test]
    fn prepare_stage_input_rejects_invalid_options() {
        let cases: [(&str, serde_json::Value, &str); 6] = [
            (
                "temperature",
                serde_json::json!("hot"),
                "non-negative number",
            ),
            (
                "temperature",
                serde_json::json!(-0.5),
                "non-negative number",
            ),
            ("top_p", serde_json::json!(true), "non-negative number"),
            ("max_tokens", serde_json::json!(0), "positive integer"),
            ("max_tokens", serde_json::json!(1.5), "positive integer"),
            ("system_prompt", serde_json::json!(42), "a string"),
        ];
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
        for (key, value, expected) in cases {
            let mut options = crate::pipeline::StageOptions::new();
            options.values.insert(key.to_string(), value.clone());
            let stage = StageDescriptor::new("llm").with_options(options);
            let err = prepare_stage_input(&stage, &input)
                .expect_err(&format!("{key}={value} must be rejected"));
            let msg = err.to_string();
            assert!(msg.contains(key) && msg.contains(expected), "{msg}");
        }
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
    fn test_select_local_adapter_prefers_onnx() {
        let mut executor = Executor::new();
        executor.register_adapter(Arc::new(OnnxRuntimeAdapter::new()));
        let mut other = MockRuntimeAdapter::with_text_output("x").with_name("aaa");
        other.load_model("/mock").unwrap();
        executor.register_adapter(Arc::new(other));

        assert_eq!(executor.select_local_adapter().unwrap(), "onnx");

        let empty = Executor::new();
        assert!(matches!(
            empty.select_local_adapter(),
            Err(ExecutorError::AdapterNotFound(_))
        ));
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

        // Executes an Onnx stage past the boundary check, which reaches real
        // ort initialization — under load-dynamic a missing libonnxruntime
        // panics inside ort, so skip on runners without the binary.
        if !crate::runtime_adapter::onnx::ort_runtime_available() {
            eprintln!("skipping: onnxruntime dylib not available in this environment");
            return;
        }

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

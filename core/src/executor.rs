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
//! - **Model loading**: Load models from .xyb bundles (already downloaded)
//! - **LLM integration**: Handle cloud API calls (OpenAI, Anthropic)
//!
//! Note: Model downloading is NOT the executor's responsibility. Models should be
//! downloaded via the SDK's `RegistryClient` before invoking the executor.
//!
//! ## Cross-Layer Execution
//!
//! The executor supports cross-layer pipelines where different stages run on different targets:
//! - **Device/Local**: On-device inference using .xyb bundles (delegates to [`TemplateExecutor`])
//! - **Integration**: Third-party API calls (OpenAI, Anthropic, etc.) via [`Llm`]
//! - **Cloud/Server**: Xybrid-hosted inference (future)

use crate::bundler::{BundleManifest, XyBundle};
use crate::context::StageDescriptor;
use crate::llm::{Llm, LlmConfig, LlmBackend, CompletionRequest};
use crate::execution_template::ModelMetadata;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, RuntimeAdapter};
use crate::template_executor::TemplateExecutor;
use crate::tracing as trace;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

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
/// **Note**: The executor works with already-downloaded models. Model downloading
/// is handled by the SDK's `RegistryClient` before invoking the executor.
pub struct Executor {
    /// Registry of runtime adapters by name
    adapters: HashMap<String, Arc<dyn RuntimeAdapter>>,
    /// Default adapter name for local execution
    default_local_adapter: Option<String>,
    /// Default adapter name for cloud execution
    default_cloud_adapter: Option<String>,
    /// Temporary directory for mock model files (demo only)
    _mock_models_dir: Option<TempDir>,
    /// Temporary directory for extracted bundles
    _extracted_bundles_dir: Option<TempDir>,
}

impl Clone for Executor {
    fn clone(&self) -> Self {
        Self {
            adapters: self.adapters.clone(),
            default_local_adapter: self.default_local_adapter.clone(),
            default_cloud_adapter: self.default_cloud_adapter.clone(),
            _mock_models_dir: None, // Don't clone temp dir
            _extracted_bundles_dir: None, // Don't clone temp dir
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
            _mock_models_dir: None,
            _extracted_bundles_dir: None,
        }
    }

    /// Resolves a stage to a model file path.
    ///
    /// If the stage has a bundle_path set, extracts and uses that bundle.
    /// Otherwise, falls back to mock model creation (for demo/testing).
    ///
    /// **Note**: Model downloading is handled by the SDK's `RegistryClient`.
    /// The executor expects models to already be downloaded.
    fn resolve_stage_to_model_path(
        &mut self,
        stage: &StageDescriptor,
        adapter_name: &str,
    ) -> ExecutorResult<PathBuf> {
        // Check if stage has a bundle_path (set by SDK after downloading)
        if let Some(bundle_path) = &stage.bundle_path {
            let path = PathBuf::from(bundle_path);
            if path.exists() {
                // Check if it's a bundle file (.xyb) or direct model file
                if path.extension().and_then(|s| s.to_str()) == Some("xyb")
                    || path.extension().and_then(|s| s.to_str()) == Some("bundle")
                {
                    // Extract bundle and get model path
                    return self.extract_bundle_and_get_model_path(&path, adapter_name);
                } else {
                    // Direct model file path (not a bundle)
                    return Ok(path);
                }
            }
        }

        // Fallback: Create mock model file (for demo/testing)
        self.resolve_stage_to_model_path_mock(&stage.name, adapter_name)
    }

    /// Creates a mock model file for demo purposes.
    fn resolve_stage_to_model_path_mock(&mut self, stage_name: &str, adapter_name: &str) -> ExecutorResult<PathBuf> {
        // Parse stage name (e.g., "whisper-tiny@1.2" -> "whisper-tiny")
        let model_name = stage_name.split('@').next().unwrap_or(stage_name);
        
        // Create temp directory for mock models if not exists
        if self._mock_models_dir.is_none() {
            self._mock_models_dir = Some(TempDir::new().map_err(|e| {
                ExecutorError::Other(format!("Failed to create temp dir: {}", e))
            })?);
        }
        
        let temp_dir = self._mock_models_dir.as_ref().unwrap();
        
        // Determine file extension based on adapter
        let extension = match adapter_name {
            "coreml" => "mlpackage",
            "onnx" | "onnx-mobile" => "onnx",
            "cloud" => "onnx", // Cloud uses ONNX format
            _ => "onnx", // Default to ONNX
        };
        
        let model_path = temp_dir.path().join(format!("{}.{}", model_name, extension));
        
        // Create mock model file if it doesn't exist
        if !model_path.exists() {
            if extension == "mlpackage" {
                // Create .mlpackage directory structure
                fs::create_dir_all(&model_path).map_err(|e| {
                    ExecutorError::Other(format!("Failed to create mlpackage dir: {}", e))
                })?;
                // Create minimal manifest.json
                let manifest_path = model_path.join("manifest.json");
                fs::write(&manifest_path, b"{}").map_err(|e| {
                    ExecutorError::Other(format!("Failed to write manifest: {}", e))
                })?;
            } else {
                // Create mock ONNX file
                fs::write(&model_path, b"mock onnx model data").map_err(|e| {
                    ExecutorError::Other(format!("Failed to write model file: {}", e))
                })?;
            }
        }
        
        Ok(model_path)
    }

    /// Extracts a bundle and returns the path to the model file.
    ///
    /// # Arguments
    ///
    /// * `bundle_path` - Path to the .xyb bundle file
    /// * `adapter_name` - Name of the adapter (to determine model file type)
    ///
    /// # Returns
    ///
    /// Path to the extracted model file
    fn extract_bundle_and_get_model_path(
        &mut self,
        bundle_path: &Path,
        adapter_name: &str,
    ) -> ExecutorResult<PathBuf> {
        // Create temp directory for extracted bundles if not exists
        if self._extracted_bundles_dir.is_none() {
            self._extracted_bundles_dir = Some(TempDir::new().map_err(|e| {
                ExecutorError::Other(format!("Failed to create extracted bundles dir: {}", e))
            })?);
        }

        let extract_base_dir = self._extracted_bundles_dir.as_ref().unwrap();

        // Create unique extraction directory for this bundle
        let bundle_stem = bundle_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("bundle");
        let extract_dir = extract_base_dir.path().join(bundle_stem);

        // Extract bundle if not already extracted
        if !extract_dir.exists() {
            fs::create_dir_all(&extract_dir).map_err(|e| {
                ExecutorError::Other(format!("Failed to create extract dir: {}", e))
            })?;

            // Load bundle
            let bundle = XyBundle::load(bundle_path).map_err(|e| {
                ExecutorError::Other(format!("Failed to load bundle: {}", e))
            })?;

            // Extract bundle contents
            bundle.extract_to(&extract_dir).map_err(|e| {
                ExecutorError::Other(format!("Failed to extract bundle: {}", e))
            })?;

            // Write manifest.json (bundler's extract_to doesn't include manifest.json in files)
            let manifest_path = extract_dir.join("manifest.json");
            let manifest_json = serde_json::to_string_pretty(bundle.manifest())
                .map_err(|e| ExecutorError::Other(format!("Failed to serialize manifest: {}", e)))?;
            fs::write(&manifest_path, manifest_json).map_err(|e| {
                ExecutorError::Other(format!("Failed to write manifest: {}", e))
            })?;
        }

        // Find model file in extracted bundle
        self.find_model_file_in_extracted_bundle(&extract_dir, adapter_name)
    }

    /// Finds the model file in an extracted bundle directory.
    ///
    /// # Arguments
    ///
    /// * `extract_dir` - Directory containing extracted bundle files
    /// * `adapter_name` - Name of the adapter (to determine model file type)
    ///
    /// # Returns
    ///
    /// Path to the model file
    fn find_model_file_in_extracted_bundle(
        &self,
        extract_dir: &Path,
        adapter_name: &str,
    ) -> ExecutorResult<PathBuf> {
        // Read manifest to get file list
        let manifest_path = extract_dir.join("manifest.json");
        let manifest_content = fs::read_to_string(&manifest_path).map_err(|e| {
            ExecutorError::Other(format!("Failed to read manifest: {}", e))
        })?;
        let manifest: BundleManifest = serde_json::from_str(&manifest_content).map_err(|e| {
            ExecutorError::Other(format!("Failed to parse manifest: {}", e))
        })?;

        // Determine expected file extensions based on adapter
        let expected_extensions: Vec<&str> = match adapter_name {
            "coreml" => vec!["mlpackage", "mlmodel"],
            "onnx" | "onnx-mobile" => vec!["onnx"],
            "cloud" => vec!["onnx"],
            _ => vec!["onnx"],
        };

        // Look for model file in manifest.files
        for file_name in &manifest.files {
            let file_path = extract_dir.join(file_name);
            if file_path.exists() {
                // Check if file extension matches
                if let Some(ext) = file_path.extension().and_then(|s| s.to_str()) {
                    if expected_extensions.contains(&ext) {
                        return Ok(file_path);
                    }
                }
            }
        }

        // If not found in manifest, try scanning directory
        let entries = fs::read_dir(extract_dir).map_err(|e| {
            ExecutorError::Other(format!("Failed to read extract dir: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                ExecutorError::Other(format!("Failed to read dir entry: {}", e))
            })?;
            let path = entry.path();
            
            // Skip manifest.json
            if path.file_name().and_then(|s| s.to_str()) == Some("manifest.json") {
                continue;
            }

            // Check if extension matches
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if expected_extensions.contains(&ext) {
                    return Ok(path);
                }
            }

            // Check if it's a directory (for .mlpackage)
            if path.is_dir() && expected_extensions.contains(&"mlpackage") {
                // Check if it looks like a .mlpackage directory
                if path.extension().and_then(|s| s.to_str()) == Some("mlpackage") {
                    return Ok(path);
                }
            }
        }

        Err(ExecutorError::Other(format!(
            "Model file not found in bundle for adapter: {}",
            adapter_name
        )))
    }

    /// Extracts a bundle and returns the directory path and optional metadata.
    ///
    /// # Arguments
    ///
    /// * `bundle_path` - Path to the .xyb bundle file
    ///
    /// # Returns
    ///
    /// Tuple of (extract_dir, Optional<ModelMetadata>)
    fn extract_bundle_with_metadata(
        &mut self,
        bundle_path: &Path,
    ) -> ExecutorResult<(PathBuf, Option<ModelMetadata>)> {
        // Create temp directory for extracted bundles if not exists
        if self._extracted_bundles_dir.is_none() {
            self._extracted_bundles_dir = Some(TempDir::new().map_err(|e| {
                ExecutorError::Other(format!("Failed to create extracted bundles dir: {}", e))
            })?);
        }

        let extract_base_dir = self._extracted_bundles_dir.as_ref().unwrap();

        // Create unique extraction directory for this bundle
        let bundle_stem = bundle_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("bundle");
        let extract_dir = extract_base_dir.path().join(bundle_stem);

        // Extract bundle if not already extracted
        if !extract_dir.exists() {
            fs::create_dir_all(&extract_dir).map_err(|e| {
                ExecutorError::Other(format!("Failed to create extract dir: {}", e))
            })?;

            // Load bundle
            let bundle = XyBundle::load(bundle_path).map_err(|e| {
                ExecutorError::Other(format!("Failed to load bundle: {}", e))
            })?;

            // Extract bundle contents
            bundle.extract_to(&extract_dir).map_err(|e| {
                ExecutorError::Other(format!("Failed to extract bundle: {}", e))
            })?;

            // Write manifest.json
            let manifest_path = extract_dir.join("manifest.json");
            let manifest_json = serde_json::to_string_pretty(bundle.manifest())
                .map_err(|e| ExecutorError::Other(format!("Failed to serialize manifest: {}", e)))?;
            fs::write(&manifest_path, manifest_json).map_err(|e| {
                ExecutorError::Other(format!("Failed to write manifest: {}", e))
            })?;
        }

        // Try to load model_metadata.json if it exists
        let metadata_path = extract_dir.join("model_metadata.json");
        let metadata = if metadata_path.exists() {
            let metadata_content = fs::read_to_string(&metadata_path).map_err(|e| {
                ExecutorError::Other(format!("Failed to read model_metadata.json: {}", e))
            })?;
            let meta: ModelMetadata = serde_json::from_str(&metadata_content).map_err(|e| {
                ExecutorError::Other(format!("Failed to parse model_metadata.json: {}", e))
            })?;
            Some(meta)
        } else {
            None
        };

        Ok((extract_dir, metadata))
    }

    /// Registers a runtime adapter with the executor.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The runtime adapter to register (wrapped in Arc for shared ownership)
    ///
    /// # Example
    ///
    /// ```rust,no_run
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
    /// * `target` - Target where execution should occur ("local", "edge", "cloud", "integration")
    ///
    /// # Returns
    ///
    /// A tuple containing the output envelope and stage metadata
    ///
    /// # Example
    ///
    /// ```rust,no_run
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

        // Check if this is an integration stage (third-party API like OpenAI/Anthropic)
        if stage.is_integration() {
            return self.execute_integration(stage, input, start_time);
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
                .map_err(|e| ExecutorError::AdapterError(e))?;

            let latency_ms = start_time.elapsed().as_millis();
            let metadata = StageMetadata {
                adapter: adapter_name,
                target: target.to_string(),
                latency_ms,
            };
            return Ok((output, metadata));
        }

        // Try bundle_path for metadata-driven execution (bundles are pre-downloaded by SDK)
        if let Some(bundle_path_str) = &stage.bundle_path {
            let bundle_path = PathBuf::from(bundle_path_str);
            if bundle_path.exists() {
                let ext = bundle_path.extension().and_then(|s| s.to_str()).unwrap_or("");
                if ext == "xyb" || ext == "bundle" {
                    // Extract bundle and check for model_metadata.json
                    if let Ok((extract_dir, Some(model_metadata))) = self.extract_bundle_with_metadata(&bundle_path) {
                        // Use TemplateExecutor for metadata-driven inference
                        let base_path = extract_dir.to_str().ok_or_else(|| {
                            ExecutorError::Other("Invalid extract dir path".to_string())
                        })?;

                        let mut template_executor = TemplateExecutor::new(base_path);

                        let output = template_executor.execute(&model_metadata, input)
                            .map_err(|e| ExecutorError::AdapterError(e))?;

                        let latency_ms = start_time.elapsed().as_millis();
                        let metadata = StageMetadata {
                            adapter: "template-executor".to_string(),
                            target: target.to_string(),
                            latency_ms,
                        };

                        return Ok((output, metadata));
                    }
                }
            }
        }

        // Fallback: Use raw adapter execution (no metadata-driven processing)
        let model_path = self.resolve_stage_to_model_path(stage, &adapter_name)?;
        let model_path_str = model_path.to_str().ok_or_else(|| {
            ExecutorError::Other("Invalid model path".to_string())
        })?;

        // Try to load model if adapter is mutable
        if let Some(adapter_arc) = self.adapters.get_mut(&adapter_name) {
            if let Some(adapter_mut) = Arc::get_mut(adapter_arc) {
                // Try to load model - ignore errors (will handle ModelNotLoaded below)
                let _ = adapter_mut.load_model(model_path_str);
            }
        }

        let adapter = self
            .get_adapter(&adapter_name)
            .ok_or_else(|| ExecutorError::AdapterNotFound(adapter_name.clone()))?;

        // Execute inference via adapter
        // Handle ModelNotLoaded error gracefully for demo
        let output = match adapter.execute(input) {
            Ok(output) => output,
            Err(AdapterError::ModelNotLoaded(_)) => {
                // Model not loaded - return mock output for demo
                // In production, models should be pre-loaded
                match &input.kind {
                    crate::ir::EnvelopeKind::Audio(_) => {
                        Envelope::new(crate::ir::EnvelopeKind::Text(format!("mock-asr-output-{}", stage.name)))
                    }
                    crate::ir::EnvelopeKind::Text(t) => {
                        Envelope::new(crate::ir::EnvelopeKind::Text(format!("mock-output-{}-{}", stage.name, t)))
                    }
                    crate::ir::EnvelopeKind::Embedding(_) => {
                        Envelope::new(crate::ir::EnvelopeKind::Text(format!("mock-embedding-output-{}", stage.name)))
                    }
                }
            }
            Err(e) => return Err(ExecutorError::AdapterError(e)),
        };

        // Calculate latency
        let latency_ms = start_time.elapsed().as_millis();

        // Create metadata
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
    /// ```rust,no_run
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

    /// Executes a stage via third-party integration (OpenAI, Anthropic, etc.).
    ///
    /// This method handles cross-layer pipeline execution where a stage runs on
    /// a remote LLM provider rather than locally on-device.
    ///
    /// # Arguments
    ///
    /// * `stage` - Stage descriptor with provider info
    /// * `input` - Input envelope (expects Text)
    /// * `start_time` - Timer for latency measurement
    ///
    /// # Returns
    ///
    /// Output envelope with LLM response and stage metadata
    fn execute_integration(
        &self,
        stage: &StageDescriptor,
        input: &Envelope,
        start_time: Instant,
    ) -> ExecutorResult<(Envelope, StageMetadata)> {
        // Extract provider from stage descriptor
        let provider = stage.provider.ok_or_else(|| {
            ExecutorError::ProviderNotConfigured(
                "Integration stage requires provider".to_string()
            )
        })?;

        // Start tracing span for integration execution
        let model_name = stage.model.clone().unwrap_or_else(|| "unknown".to_string());
        let _exec_span = trace::SpanGuard::new(format!("execute:{}", model_name));
        trace::add_metadata("provider", provider.as_str());
        trace::add_metadata("target", "integration");
        if let Some(ref model) = stage.model {
            trace::add_metadata("model", model);
        }

        // Build LLM configuration
        // Default: Gateway (recommended for production)
        // Direct: Only if explicitly requested via options.backend = "direct"
        let mut llm_config = LlmConfig::default(); // Gateway by default

        // Apply stage options if available
        if let Some(ref options) = stage.options {
            // Check for backend override (gateway is default, direct for dev/testing)
            if let Some(backend) = options.get::<String>("backend") {
                match backend.to_lowercase().as_str() {
                    "direct" => {
                        // Use direct API calls (requires provider API key)
                        llm_config.backend = LlmBackend::Direct;
                        llm_config.direct_provider = Some(provider.as_str().to_string());
                    }
                    "local" => {
                        llm_config.backend = LlmBackend::Local;
                        if let Some(model_path) = options.get::<String>("local_model_path") {
                            llm_config.local_model_path = Some(model_path);
                        }
                    }
                    "auto" => {
                        llm_config.backend = LlmBackend::Auto;
                        llm_config.direct_provider = Some(provider.as_str().to_string());
                    }
                    _ => {
                        // Default to gateway
                        llm_config.backend = LlmBackend::Gateway;
                    }
                }
            }

            // Custom gateway URL (for self-hosted gateway)
            if let Some(gateway_url) = options.get::<String>("gateway_url") {
                llm_config.gateway_url = gateway_url;
            }

            // Explicit API key (for gateway or direct)
            if let Some(api_key) = options.get::<String>("api_key") {
                llm_config.api_key = Some(api_key);
            }

            // Timeout override
            if let Some(timeout) = options.timeout_ms() {
                llm_config.timeout_ms = timeout;
            }

            // Debug mode
            if let Some(debug) = options.get::<bool>("debug") {
                llm_config.debug = debug;
            }
        }

        // For direct backend, also set provider for fallback in Auto mode
        if llm_config.backend == LlmBackend::Auto && llm_config.direct_provider.is_none() {
            llm_config.direct_provider = Some(provider.as_str().to_string());
        }

        // Capture backend string for tracing before config is consumed
        let backend_str = match llm_config.backend {
            LlmBackend::Gateway => "gateway",
            LlmBackend::Direct => "direct",
            LlmBackend::Local => "local",
            LlmBackend::Auto => "auto",
        };

        // Create LLM client (gateway-aware)
        let client = Llm::with_config(llm_config).map_err(|e| {
            ExecutorError::IntegrationError(format!("Failed to create LLM client: {}", e))
        })?;

        // Extract text input
        let input_text = match &input.kind {
            EnvelopeKind::Text(text) => text.clone(),
            other => {
                return Err(ExecutorError::IntegrationError(format!(
                    "Integration stages expect Text input, got: {:?}",
                    other
                )));
            }
        };

        // Build LLM request
        let mut request = CompletionRequest::new(&input_text);

        // Set model if specified in stage
        if let Some(ref model) = stage.model {
            request = request.with_model(model);
        }

        // Apply stage options to request
        if let Some(ref options) = stage.options {
            if let Some(system) = options.system_prompt() {
                request = request.with_system(&system);
            }
            if let Some(temp) = options.temperature() {
                request = request.with_temperature(temp);
            }
            if let Some(max) = options.max_tokens() {
                request = request.with_max_tokens(max);
            }
        }

        // Execute LLM request (via gateway by default)
        let response = {
            let _llm_span = trace::SpanGuard::new("llm_inference");
            trace::add_metadata("backend", backend_str);
            client.complete(request).map_err(|e| {
                ExecutorError::IntegrationError(format!("LLM request failed: {}", e))
            })?
        };

        // Build output envelope
        let output = Envelope::new(EnvelopeKind::Text(response.text));

        // Calculate latency
        let latency_ms = start_time.elapsed().as_millis();

        // Build metadata (include backend info)
        let backend_info = response.backend.unwrap_or_else(|| "gateway".to_string());
        let metadata = StageMetadata {
            adapter: format!("integration:{}:{}", provider, backend_info),
            target: "integration".to_string(),
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
                    .map(|k| k.clone())
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
                    .map(|k| k.clone())
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
                    .map(|k| k.clone())
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
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn create_test_executor() -> Executor {
        let mut executor = Executor::new();
        let adapter = Arc::new(OnnxRuntimeAdapter::new());
        executor.register_adapter(adapter);
        executor
    }

    fn create_mock_onnx_file() -> (TempDir, String) {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");
        fs::write(&model_path, b"fake onnx model data").unwrap();
        (temp_dir, model_path.to_string_lossy().to_string())
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
        let (_temp_dir, model_path) = create_mock_onnx_file();
        let mut executor = Executor::new();

        // Create and register adapter
        let mut adapter = OnnxRuntimeAdapter::new();
        adapter.load_model(&model_path)?;
        executor.register_adapter(Arc::new(adapter));

        let stage = StageDescriptor::new("asr");
        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));

        let (output, metadata) = executor.execute_stage(&stage, &input, "local")?;

        // Verify output
        assert_eq!(output.kind_str(), "Text"); // ASR converts audio to text
        assert_eq!(metadata.target, "local");
        assert_eq!(metadata.adapter, "onnx");
        Ok(())
    }

    #[test]
    fn test_execute_stage_cloud() -> ExecutorResult<()> {
        let (_temp_dir, model_path) = create_mock_onnx_file();
        let mut executor = Executor::new();

        let mut adapter = OnnxRuntimeAdapter::new();
        adapter.load_model(&model_path)?;
        executor.register_adapter(Arc::new(adapter));

        let stage = StageDescriptor::new("motivator");
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let (_output, metadata) = executor.execute_stage(&stage, &input, "cloud")?;

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
}

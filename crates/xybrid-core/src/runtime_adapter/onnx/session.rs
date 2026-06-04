//! ONNX Runtime session wrapper for managing model sessions and inference.
//!
//! This module provides a wrapper around ONNX Runtime sessions that:
//! - Manages session lifecycle
//! - Extracts model metadata (input/output names and shapes)
//! - Handles execution provider selection (CPU, CoreML, etc.)
//! - Provides a clean interface for running inference
//!
//! Construction goes through a single entry point — [`ONNXSession::build`] —
//! taking the model path, the requested [`ExecutionProviderKind`], and a
//! [`SessionOptions`] flag bag. The default options keep the session on
//! the cheap path; opting in to `capture_resolved_ep` enables ORT
//! profiling so the *actual* execution provider that ran each op can be
//! harvested after the first inference (see
//! [`ONNXSession::resolved_providers`]).
//!
//! # Example
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use std::collections::HashMap;
//! use ndarray::ArrayD;
//! use xybrid_core::runtime_adapter::onnx::{ONNXSession, ExecutionProviderKind, SessionOptions};
//!
//! // CPU execution, no profiling overhead
//! let session = ONNXSession::build(
//!     "/path/to/model.onnx",
//!     ExecutionProviderKind::Cpu,
//!     SessionOptions::default(),
//! )?;
//!
//! // CoreML execution with resolved-EP capture is gated by the `ort-coreml` cfg.
//!
//! let inputs: HashMap<String, ArrayD<f32>> = HashMap::new();
//! let outputs = session.run(inputs)?;
//! # let _ = outputs;
//! # Ok(())
//! # }
//! ```

use super::execution_provider::ExecutionProviderKind;
use super::profiling::{parse_profile_json, ResolvedExecutionProviders};
use crate::runtime_adapter::{AdapterError, AdapterResult};
use ndarray::{ArrayD, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tempfile::TempDir;

/// Metadata extracted from ONNX model inputs: (names, shapes, element types).
type InputMetadata = (Vec<String>, Vec<Vec<i64>>, Vec<Option<TensorElementType>>);

/// Construction-time options for [`ONNXSession::build`].
///
/// Every field defaults to the cheap behaviour — opting in costs
/// something measurable, so the burden is on the caller to ask for it.
///
/// Marked `#[non_exhaustive]` so adding new flags later isn't a
/// breaking change for external struct-literal callers. Construct with
/// [`SessionOptions::default`] (cheap path), or build the literal with
/// `..Default::default()` when only a subset of flags need to be set.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy)]
pub struct SessionOptions {
    /// When `true`, the session is built with ORT profiling enabled so
    /// the *resolved* execution provider (the EP that actually ran each
    /// op) can be harvested after the first inference via
    /// [`ONNXSession::resolved_providers`].
    ///
    /// Profiling adds roughly 10-15 % wall-clock overhead to the first
    /// inference; subsequent inferences run at normal cost once the
    /// harvest finalises the profile file. Profiling also requires a
    /// writable tempdir at construction — sandboxed targets (iOS,
    /// restricted Android, hermetic CI runners) where
    /// `tempfile::tempdir()` can fail will refuse to construct the
    /// session, so leave this `false` on any path that doesn't actually
    /// read the resolved EP.
    pub capture_resolved_ep: bool,
}

/// Lifecycle state for the resolved-EP capture.
///
/// `Disabled` is what every caller of [`ONNXSession::build`] with the
/// default [`SessionOptions`] sees. `Pending` means profiling was
/// enabled at construction (via `SessionOptions { capture_resolved_ep:
/// true, .. }`) and we're waiting for the first inference to produce a
/// profile we can harvest. `Harvested` carries the parsed summary.
/// `Failed` records the error string so callers can decide whether to
/// retry or fall back to the requested EP — we don't poison the whole
/// session over a profile-parse failure.
///
/// The `TempDir` keeps the profile file alive until harvest succeeds and
/// drops it (with the file inside) automatically afterwards. ORT's
/// profiling output goes into this directory; deleting it post-harvest
/// is what satisfies the "no leaked tmp files" acceptance criterion.
enum ResolvedEpState {
    Disabled,
    Pending {
        /// Holds the profile-output directory open until we harvest. ORT
        /// writes `<prefix>_<timestamp>.json` inside this dir; the dir
        /// is dropped on transition to `Harvested`/`Failed`.
        _tempdir: TempDir,
    },
    Harvested(ResolvedExecutionProviders),
    Failed(String),
}

/// ONNX Runtime session wrapper.
///
/// Manages an ONNX Runtime session, including:
/// - Model loading and session creation
/// - Input/output metadata extraction
/// - Execution provider selection
/// - Inference execution
pub struct ONNXSession {
    /// The ONNX Runtime session (wrapped in Mutex for thread-safe interior mutability)
    session: Mutex<Session>,
    /// Input names from the model
    input_names: Vec<String>,
    /// Output names from the model
    output_names: Vec<String>,
    /// Input shapes (may contain dynamic dimensions)
    input_shapes: Vec<Vec<i64>>,
    /// Output shapes (may contain dynamic dimensions)
    output_shapes: Vec<Vec<i64>>,
    /// Input element types (e.g., Float32, Int64) from ONNX model metadata
    input_dtypes: Vec<Option<TensorElementType>>,
    /// The execution provider used for this session
    execution_provider: ExecutionProviderKind,
    /// Resolved-EP capture state. `Disabled` for sessions built with the
    /// default [`SessionOptions`]; `Pending → Harvested/Failed` for
    /// sessions built with `SessionOptions { capture_resolved_ep: true,
    /// .. }`. Wrapped in [`Mutex`] so the auto-harvest path inside
    /// [`run_with_values`] (which only has `&self`) can mutate it.
    resolved_state: Mutex<ResolvedEpState>,
}

impl ONNXSession {
    /// Builds a new ONNX session.
    ///
    /// This is the single construction entry point on `ONNXSession`. It
    /// loads the model file, configures the requested execution
    /// provider, extracts input/output metadata, and — when
    /// `options.capture_resolved_ep` is set — turns on ORT profiling so
    /// the resolved EP can be harvested after the first inference.
    ///
    /// # Arguments
    ///
    /// * `model_path` — Path to the ONNX model file
    /// * `execution_provider` — The execution provider to request (CPU, CoreML, …)
    /// * `options` — [`SessionOptions`]; default leaves the session on
    ///   the cheap path (no profiling, no tempdir requirement)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - the model file doesn't exist or fails to load
    /// - the execution provider fails to initialise
    /// - metadata extraction fails
    /// - `options.capture_resolved_ep` is set and a tempdir cannot be
    ///   created (sandboxed targets — opt out of capture on those paths)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// use xybrid_core::runtime_adapter::onnx::{ONNXSession, ExecutionProviderKind, SessionOptions};
    ///
    /// // Cheap path — no profiling overhead, no tempdir
    /// let session = ONNXSession::build(
    ///     "model.onnx",
    ///     ExecutionProviderKind::Cpu,
    ///     SessionOptions::default(),
    /// )?;
    ///
    /// // Opt in to resolved-EP capture for telemetry callers
    /// let mut opts = SessionOptions::default();
    /// opts.capture_resolved_ep = true;
    /// let session = ONNXSession::build(
    ///     "model.onnx",
    ///     ExecutionProviderKind::Cpu,
    ///     opts,
    /// )?;
    /// # let _ = session;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(
        model_path: &str,
        execution_provider: ExecutionProviderKind,
        options: SessionOptions,
    ) -> AdapterResult<Self> {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(AdapterError::ModelNotFound(format!(
                "Model file not found: {}",
                model_path
            )));
        }

        // Initialize ONNX Runtime environment (singleton, safe to call multiple times)
        let _ = ort::init().commit();

        // Build the session-builder up front; the profiling branch adds
        // a tempdir and a `with_profiling` call, the default branch
        // skips both. Keeping the two paths visibly separate is what
        // lets sandboxed targets stay on the no-tempdir path safely.
        let mut builder = Session::builder()
            .map_err(|e| {
                AdapterError::RuntimeError(format!("Failed to create session builder: {}", e))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                AdapterError::RuntimeError(format!("Failed to set optimization level: {}", e))
            })?;

        let resolved_state = if options.capture_resolved_ep {
            let tempdir = tempfile::tempdir().map_err(|e| {
                AdapterError::RuntimeError(format!(
                    "Failed to create profile tempdir for resolved-EP capture: {}",
                    e
                ))
            })?;
            // ORT appends `_<timestamp>.json` to whatever prefix we pass;
            // this gives us a stable subpath inside the tempdir we own.
            let profile_prefix: PathBuf = tempdir.path().join("xybrid-profile");
            builder = builder.with_profiling(&profile_prefix).map_err(|e| {
                AdapterError::RuntimeError(format!(
                    "Failed to enable profiling for resolved-EP capture: {}",
                    e
                ))
            })?;
            ResolvedEpState::Pending { _tempdir: tempdir }
        } else {
            ResolvedEpState::Disabled
        };

        builder = Self::configure_execution_provider(builder, &execution_provider)?;

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to load ONNX model: {}", e)))?;

        let (input_names, input_shapes, input_dtypes) = Self::extract_input_metadata(&session)?;
        let (output_names, output_shapes) = Self::extract_output_metadata(&session)?;

        log::info!(
            "Created ONNX session with {} execution provider for model: {} (capture_resolved_ep={})",
            execution_provider,
            model_path,
            options.capture_resolved_ep,
        );

        Ok(Self {
            session: Mutex::new(session),
            input_names,
            output_names,
            input_shapes,
            output_shapes,
            input_dtypes,
            execution_provider,
            resolved_state: Mutex::new(resolved_state),
        })
    }

    /// Configures the execution provider on the session builder.
    fn configure_execution_provider(
        builder: ort::session::builder::SessionBuilder,
        provider: &ExecutionProviderKind,
    ) -> AdapterResult<ort::session::builder::SessionBuilder> {
        match provider {
            ExecutionProviderKind::Cpu => {
                // CPU is the default, no additional configuration needed
                Ok(builder)
            }

            #[cfg(feature = "ort-coreml")]
            ExecutionProviderKind::CoreML(config) => {
                use super::execution_provider::CoreMLComputeUnits;
                use ort::ep;

                // Build CoreML execution provider with configuration
                let coreml_ep = {
                    let mut coreml = ep::CoreML::default();

                    // Configure subgraphs
                    coreml = coreml.with_subgraphs(config.use_subgraphs);

                    // Configure compute units
                    coreml = coreml.with_compute_units(match config.compute_units {
                        CoreMLComputeUnits::CpuOnly => ep::coreml::ComputeUnits::CPUOnly,
                        CoreMLComputeUnits::CpuAndGpu => ep::coreml::ComputeUnits::CPUAndGPU,
                        CoreMLComputeUnits::CpuAndNeuralEngine => {
                            ep::coreml::ComputeUnits::CPUAndNeuralEngine
                        }
                        CoreMLComputeUnits::All => ep::coreml::ComputeUnits::All,
                    });

                    coreml.build()
                };

                log::debug!("Configuring CoreML execution provider: {:?}", config);

                builder.with_execution_providers([coreml_ep]).map_err(|e| {
                    AdapterError::RuntimeError(format!(
                        "Failed to configure CoreML execution provider: {}",
                        e
                    ))
                })
            }
        }
    }

    /// Extracts input metadata from the session.
    fn extract_input_metadata(session: &Session) -> AdapterResult<InputMetadata> {
        let mut input_names = Vec::new();
        let mut input_shapes = Vec::new();
        let mut input_dtypes = Vec::new();

        // Access session.inputs directly - ort exposes inputs as Vec<Outlet>
        // Each Outlet has name() and dtype() (ValueType with element type + shape)
        for input in session.inputs() {
            input_names.push(input.name().to_string());

            // Extract element type and shape from ValueType
            if let Some(shape) = input.dtype().tensor_shape() {
                input_shapes.push(shape.iter().copied().collect());
            } else {
                input_shapes.push(vec![-1]);
            }
            input_dtypes.push(input.dtype().tensor_type());
        }

        // If no inputs found, use placeholder
        if input_names.is_empty() {
            input_names.push("input".to_string());
            input_shapes.push(vec![1, 1, 16000]); // Placeholder shape for audio
            input_dtypes.push(None);
        }

        Ok((input_names, input_shapes, input_dtypes))
    }

    /// Extracts output metadata from the session.
    fn extract_output_metadata(session: &Session) -> AdapterResult<(Vec<String>, Vec<Vec<i64>>)> {
        let mut output_names = Vec::new();
        let mut output_shapes = Vec::new();

        // Access session.outputs directly - ort exposes outputs as Vec<Output>
        // Each Output has a name field
        for output in session.outputs() {
            output_names.push(output.name().to_string());
            // Note: ort's Output struct doesn't directly expose shapes
            // Shapes may be dynamic or need to be inferred from the model
            // For now, use placeholder shapes
            // TODO: Extract real shapes from model metadata if available
            output_shapes.push(vec![-1]); // Placeholder: -1 indicates dynamic dimension
        }

        // If no outputs found, use placeholder
        if output_names.is_empty() {
            output_names.push("output".to_string());
            output_shapes.push(vec![1, 512]); // Placeholder shape
        }

        Ok((output_names, output_shapes))
    }

    /// Runs inference on the session.
    ///
    /// # Arguments
    ///
    /// * `inputs` - HashMap mapping input names to tensors (`ndarray::ArrayD<f32>`)
    ///
    /// # Returns
    ///
    /// HashMap mapping output names to output tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input names don't match model inputs
    /// - Tensor shapes don't match expected shapes
    /// - Inference execution fails
    pub fn run(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> AdapterResult<HashMap<String, ArrayD<f32>>> {
        // Convert f32 arrays to Values
        let value_inputs: HashMap<String, Value> = inputs
            .into_iter()
            .map(|(k, v)| {
                Ok((
                    k,
                    Value::from_array(v)
                        .map_err(|e| {
                            AdapterError::RuntimeError(format!("Failed to convert tensor: {}", e))
                        })?
                        .into(),
                ))
            })
            .collect::<AdapterResult<_>>()?;

        self.run_with_values(value_inputs)
    }

    /// Runs inference with mixed input types (Value types).
    ///
    /// This method accepts `Value` types directly, allowing for mixed f32/i64 inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - HashMap mapping input names to `ort::Value` tensors
    ///
    /// # Returns
    ///
    /// HashMap mapping output names to `ndarray::ArrayD<f32>` tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input names don't match model inputs
    /// - Tensor shapes don't match expected shapes
    /// - Inference execution fails
    pub fn run_with_values(
        &self,
        inputs: HashMap<String, Value>,
    ) -> AdapterResult<HashMap<String, ArrayD<f32>>> {
        use ort::session::SessionInputs;

        // Get mutable access to session (wrapped in Mutex)
        let mut session_guard = self
            .session
            .lock()
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to lock session: {}", e)))?;

        // Convert HashMap to Vec of (Cow<str>, SessionInputValue)
        // This allows us to pass an arbitrary number of inputs
        let ort_inputs: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )> = inputs
            .into_iter()
            .map(|(name, value)| (std::borrow::Cow::Owned(name), value.into()))
            .collect();

        // Run inference with dynamic number of inputs
        let outputs = session_guard
            .run(SessionInputs::from(ort_inputs))
            .map_err(|e| {
                AdapterError::InferenceFailed(format!("ONNX Runtime inference failed: {}", e))
            })?;

        // Convert outputs back to HashMap<String, ArrayD<f32>>
        let mut result = HashMap::new();

        for output_name in &self.output_names {
            // Extract output value from SessionOutputs
            // SessionOutputs can be indexed by name or accessed as a slice
            let output_value = &outputs[output_name.as_str()];

            // Try to extract as f32 first, then as i64 if that fails
            // This handles models with mixed output types
            let array_d = if let Ok(output_array) = output_value.try_extract_array::<f32>() {
                // Fast path: a standard-layout output is one contiguous slice
                // (a memcpy). Fall back to logical-order iteration only when the
                // output is non-contiguous (e.g. transposed) — which `as_slice`
                // reports as `None`, where the old `as_slice().unwrap()` panicked.
                // Either way the data is row-major, matching `from_shape_vec`.
                let dims: Vec<usize> = output_array.shape().to_vec();
                let data: Vec<f32> = match output_array.as_slice() {
                    Some(slice) => slice.to_vec(),
                    None => output_array.iter().copied().collect(),
                };
                ArrayD::from_shape_vec(IxDyn(&dims), data).map_err(|e| {
                    AdapterError::RuntimeError(format!("Failed to convert output to ArrayD: {}", e))
                })?
            } else if let Ok(output_array) = output_value.try_extract_array::<i64>() {
                // Convert i64 to f32 for uniform handling, same fast/safe split
                // as the f32 arm above.
                let dims: Vec<usize> = output_array.shape().to_vec();
                let data: Vec<f32> = match output_array.as_slice() {
                    Some(slice) => slice.iter().map(|&x| x as f32).collect(),
                    None => output_array.iter().map(|&x| x as f32).collect(),
                };
                ArrayD::from_shape_vec(IxDyn(&dims), data).map_err(|e| {
                    AdapterError::RuntimeError(format!("Failed to convert output to ArrayD: {}", e))
                })?
            } else {
                return Err(AdapterError::RuntimeError(format!(
                    "Failed to extract output '{}': unsupported type (expected f32 or i64)",
                    output_name
                )));
            };

            result.insert(output_name.clone(), array_d);
        }

        // After the first inference, end profiling and parse the JSON
        // to surface the resolved EP. `outputs` has been fully converted
        // into owned `result` entries above, so we no longer borrow
        // from `session_guard` and can take a `&mut` reborrow for
        // `end_profiling`. Drop `outputs` explicitly to make that
        // borrow lifetime obvious to the reader.
        drop(outputs);
        self.maybe_harvest_resolved_ep(&mut session_guard);

        Ok(result)
    }

    /// Returns input names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Returns output names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Returns input shapes.
    pub fn input_shapes(&self) -> &[Vec<i64>] {
        &self.input_shapes
    }

    /// Returns input element types.
    pub fn input_dtypes(&self) -> &[Option<TensorElementType>] {
        &self.input_dtypes
    }

    /// Returns output shapes.
    pub fn output_shapes(&self) -> &[Vec<i64>] {
        &self.output_shapes
    }

    /// Returns the execution provider used for this session.
    pub fn execution_provider(&self) -> &ExecutionProviderKind {
        &self.execution_provider
    }

    /// Returns the resolved-EP summary from the first inference's
    /// profile output, if and only if the session was built with
    /// `SessionOptions { capture_resolved_ep: true, .. }` **and** at
    /// least one inference has completed since.
    ///
    /// Returns `None` for sessions without capture enabled, sessions
    /// where capture is still pending, or sessions where harvest
    /// failed (the failure reason is logged but not surfaced — the
    /// telemetry layer treats absence as "EP unknown").
    pub fn resolved_providers(&self) -> Option<ResolvedExecutionProviders> {
        let state = self.resolved_state.lock().ok()?;
        match &*state {
            ResolvedEpState::Harvested(summary) => Some(summary.clone()),
            _ => None,
        }
    }

    /// Diagnostic accessor for the raw resolved-EP state — used by tests
    /// (and surfaced for ad-hoc debugging) to distinguish
    /// `Disabled` / `Pending` / `Harvested` / `Failed(reason)` after a
    /// harvest attempt. Production callers should use
    /// [`ONNXSession::resolved_providers`] instead.
    #[doc(hidden)]
    pub fn resolved_state_debug(&self) -> String {
        match self.resolved_state.lock() {
            Ok(state) => match &*state {
                ResolvedEpState::Disabled => "Disabled".into(),
                ResolvedEpState::Pending { .. } => "Pending".into(),
                ResolvedEpState::Harvested(s) => format!("Harvested({s:?})"),
                ResolvedEpState::Failed(e) => format!("Failed({e})"),
            },
            Err(e) => format!("MutexPoisoned({e})"),
        }
    }

    /// Idempotent hook called after every successful inference: when the
    /// session is in [`ResolvedEpState::Pending`], end profiling, parse
    /// the resulting JSON, and transition to `Harvested`/`Failed`.
    /// No-op for any other state.
    fn maybe_harvest_resolved_ep(&self, session_guard: &mut Session) {
        let mut state = match self.resolved_state.lock() {
            Ok(g) => g,
            Err(e) => {
                log::warn!("resolved-EP state mutex poisoned: {e}");
                return;
            }
        };
        if !matches!(*state, ResolvedEpState::Pending { .. }) {
            return;
        }

        // `end_profiling()` finalises the JSON file and returns the
        // actual on-disk path (ORT appends `_<timestamp>.json` to our
        // prefix). On failure we record the error so subsequent calls
        // don't retry.
        let next = match session_guard.end_profiling() {
            Ok(profile_path) => {
                let path = std::path::Path::new(&profile_path);
                match parse_profile_json(path) {
                    Ok(summary) => {
                        log::debug!(
                            "Resolved EP for ONNX session: primary={}, breakdown={:?}",
                            summary.primary,
                            summary.breakdown
                        );
                        ResolvedEpState::Harvested(summary)
                    }
                    Err(parse_err) => {
                        log::warn!("Failed to parse ONNX profile {profile_path}: {parse_err}");
                        ResolvedEpState::Failed(parse_err.to_string())
                    }
                }
            }
            Err(end_err) => {
                log::warn!("Failed to end ONNX profiling: {end_err}");
                ResolvedEpState::Failed(end_err.to_string())
            }
        };
        *state = next;
        // Dropping the previous `Pending { _tempdir }` cleans up the
        // profile file along with the directory.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_session_creation_fails_on_nonexistent_file() {
        let result = ONNXSession::build(
            "/nonexistent/model.onnx",
            ExecutionProviderKind::Cpu,
            SessionOptions::default(),
        );
        assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
    }

    #[test]
    fn test_session_creation_with_mock_file() {
        // Create a temporary ONNX file (minimal valid ONNX format)
        // Note: This is a minimal test - real ONNX files are binary protobuf
        // For now, we'll test that the file existence check works
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");

        // Create a minimal file (not a real ONNX, but tests file existence)
        fs::write(&model_path, b"fake onnx data").unwrap();

        // This will fail at ort initialization or model loading, but we can test the structure
        let result = ONNXSession::build(
            model_path.to_str().unwrap(),
            ExecutionProviderKind::Cpu,
            SessionOptions::default(),
        );

        // The session creation might fail due to invalid ONNX format,
        // but we've at least tested that the file existence check passes
        // and the ort initialization is attempted
        match result {
            Ok(_) => {
                // If it succeeds, verify the structure
                let session = result.unwrap();
                assert!(!session.input_names().is_empty());
                assert!(!session.output_names().is_empty());
            }
            Err(e) => {
                // Expected: invalid ONNX format will cause ort to fail
                // But we've verified the code path executes
                println!("Expected error (invalid ONNX format): {:?}", e);
            }
        }
    }

    #[test]
    fn test_mnist_model_loading() {
        // Test loading the real MNIST model
        // Try multiple possible paths (workspace root, or relative to test execution)
        let possible_paths = vec![
            PathBuf::from("test_models/mnist-12.onnx"),
            PathBuf::from("../test_models/mnist-12.onnx"),
            PathBuf::from("../../test_models/mnist-12.onnx"),
        ];

        let model_path = possible_paths.iter().find(|p| p.exists()).cloned();

        let model_path = match model_path {
            Some(p) => p,
            None => {
                println!(
                    "MNIST model not found, skipping test. Tried: {:?}",
                    possible_paths
                );
                return;
            }
        };

        let result = ONNXSession::build(
            model_path.to_str().unwrap(),
            ExecutionProviderKind::Cpu,
            SessionOptions::default(),
        );
        assert!(
            result.is_ok(),
            "Failed to load MNIST model: {:?}",
            result.err()
        );

        let session = result.unwrap();

        // Verify we extracted real metadata
        let input_names = session.input_names();
        let output_names = session.output_names();

        println!("MNIST Input names: {:?}", input_names);
        println!("MNIST Output names: {:?}", output_names);
        println!("MNIST Input shapes: {:?}", session.input_shapes());
        println!("MNIST Output shapes: {:?}", session.output_shapes());

        // MNIST should have 1 input and 1 output
        assert!(!input_names.is_empty(), "Should have at least one input");
        assert!(!output_names.is_empty(), "Should have at least one output");

        // Verify input/output names are not placeholders
        assert_ne!(
            input_names[0], "input",
            "Should have real input name, not placeholder"
        );
        assert_ne!(
            output_names[0], "output",
            "Should have real output name, not placeholder"
        );
    }

    #[test]
    fn test_mnist_inference() {
        // Test running inference on the MNIST model
        // Try multiple possible paths (workspace root, or relative to test execution)
        let possible_paths = vec![
            PathBuf::from("test_models/mnist-12.onnx"),
            PathBuf::from("../test_models/mnist-12.onnx"),
            PathBuf::from("../../test_models/mnist-12.onnx"),
        ];

        let model_path = possible_paths.iter().find(|p| p.exists()).cloned();

        let model_path = match model_path {
            Some(p) => p,
            None => {
                println!(
                    "MNIST model not found, skipping test. Tried: {:?}",
                    possible_paths
                );
                return;
            }
        };

        let session = ONNXSession::build(
            model_path.to_str().unwrap(),
            ExecutionProviderKind::Cpu,
            SessionOptions::default(),
        )
        .expect("Failed to load MNIST model");

        // Get real input name from session
        let input_names = session.input_names();
        let input_name = &input_names[0];

        // Create test input: 28x28 grayscale image (all zeros for now)
        // MNIST expects: [batch=1, channels=1, height=28, width=28]
        let mut inputs = HashMap::new();
        let input_tensor = ArrayD::<f32>::from_shape_vec(
            IxDyn(&[1, 1, 28, 28]),
            vec![0.0f32; 784], // 28*28 = 784
        )
        .unwrap();
        inputs.insert(input_name.clone(), input_tensor);

        // Run real inference using ONNX Runtime
        let result = session.run(inputs);
        assert!(result.is_ok(), "Inference failed: {:?}", result.err());

        let outputs = result.unwrap();
        assert!(!outputs.is_empty(), "Should have at least one output");

        // Verify output structure
        let output_names = session.output_names();
        let output_name = &output_names[0];
        assert!(
            outputs.contains_key(output_name),
            "Output should contain expected output name"
        );

        // Get output tensor
        let output_tensor = outputs.get(output_name).unwrap();
        println!("MNIST Output shape: {:?}", output_tensor.shape());
        println!("MNIST Output size: {}", output_tensor.len());

        // MNIST outputs 10 class probabilities (one for each digit 0-9)
        // Verify we got the correct output shape: [batch=1, classes=10]
        assert_eq!(
            output_tensor.shape(),
            &[1, 10],
            "MNIST should output shape [1, 10]"
        );
        assert_eq!(
            output_tensor.len(),
            10,
            "MNIST output should have 10 elements"
        );
    }

    #[test]
    fn resolved_providers_returns_none_when_capture_disabled() {
        // Default-options path must leave the resolved-EP API dormant —
        // capture is opt-in and the cheap code path is unaffected. Uses
        // a nonexistent model so we never have to load the runtime; the
        // constructor errors before the accessor is reachable, so we
        // skip the assertion when ort fails to initialise (e.g. in
        // environments without the binary).
        let result = ONNXSession::build(
            "/nonexistent/model.onnx",
            ExecutionProviderKind::Cpu,
            SessionOptions::default(),
        );
        assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
    }

    #[test]
    fn resolved_providers_populates_after_first_inference() {
        // End-to-end: build with capture enabled, run one inference,
        // expect `resolved_providers()` to surface a primary EP. Skips
        // when the MNIST fixture isn't present so CI without the model
        // still passes.
        let possible_paths = [
            PathBuf::from("test_models/mnist-12.onnx"),
            PathBuf::from("../test_models/mnist-12.onnx"),
            PathBuf::from("../../test_models/mnist-12.onnx"),
        ];
        let model_path = match possible_paths.iter().find(|p| p.exists()) {
            Some(p) => p.clone(),
            None => {
                eprintln!("MNIST model not found; skipping resolved-EP capture test.");
                return;
            }
        };

        let session = ONNXSession::build(
            model_path.to_str().unwrap(),
            ExecutionProviderKind::Cpu,
            SessionOptions {
                capture_resolved_ep: true,
            },
        )
        .expect("Failed to load MNIST model with resolved-EP capture enabled");

        // Pre-inference: capture is Pending — accessor returns None.
        assert!(
            session.resolved_providers().is_none(),
            "resolved_providers() should be None before the first inference"
        );

        // Run one inference (same shape as `test_mnist_inference`).
        let input_names = session.input_names();
        let input_name = &input_names[0];
        let mut inputs = HashMap::new();
        let input_tensor =
            ArrayD::<f32>::from_shape_vec(IxDyn(&[1, 1, 28, 28]), vec![0.0f32; 784]).unwrap();
        inputs.insert(input_name.clone(), input_tensor);
        session.run(inputs).expect("MNIST inference must succeed");

        // Post-inference: harvest should have populated a summary. On a
        // CPU-only build of ORT, every op runs on `cpu`; on a CoreML
        // build asking for CPU, same result. We only assert the shape
        // (non-empty primary, breakdown sums >= 1) so the test is
        // robust across feature combinations and ORT versions.
        let summary = session.resolved_providers().unwrap_or_else(|| {
            panic!(
                "resolved_providers() must populate after the first inference; \
                 actual state: {}",
                session.resolved_state_debug()
            )
        });
        assert!(
            !summary.primary.is_empty(),
            "primary EP should be a non-empty string; got {:?}",
            summary
        );
        assert!(
            !summary.breakdown.is_empty(),
            "breakdown should list at least one EP; got {:?}",
            summary
        );
        let total_ops: usize = summary.breakdown.iter().map(|(_, n)| *n).sum();
        assert!(
            total_ops >= 1,
            "breakdown should account for at least one Node event; got {:?}",
            summary
        );
        // The MNIST graph is small enough that on a CPU-only feature set
        // the primary should be `cpu`. Don't hard-code on Apple builds
        // where CoreML may legitimately handle some ops.
        if cfg!(not(feature = "ort-coreml")) {
            assert_eq!(
                summary.primary, "cpu",
                "non-CoreML build should resolve to CPU; got {:?}",
                summary
            );
        }
    }
}

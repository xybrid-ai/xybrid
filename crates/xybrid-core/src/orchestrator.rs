//! Orchestrator module - Coordinates the execution of hybrid cloud-edge AI inference pipelines.
//!
//! The orchestrator is the **highest-level** execution layer that manages the lifecycle of
//! inference requests, coordinating between the policy engine, routing engine, stream manager,
//! and executor.
//!
//! See [`EXECUTION_LAYERS.md`](./EXECUTION_LAYERS.md) for the full architecture.
//!
//! ## Responsibility
//!
//! The orchestrator handles:
//! - **Policy evaluation**: Should this request be allowed?
//! - **Routing decisions**: Local vs edge vs cloud
//! - **Stream management**: Chunk buffering for real-time audio
//! - **Telemetry**: Event emission for observability
//!
//! ## Runtime Flow
//!
//! 1. Receive input envelope
//! 2. Evaluate policy
//! 3. Decide route
//! 4. Execute model (delegates to [`Executor`])
//! 5. Emit telemetry
//!
//! The orchestrator supports both batch and streaming execution modes, following the
//! architecture appendix: "Build local first, orchestrate distributed later."

// ─────────────────────────────────────────────────────────────────────────────
// Module declarations (must come first)
// ─────────────────────────────────────────────────────────────────────────────
pub mod authority;
pub mod bootstrap;
pub mod policy_engine;
pub mod routing_engine;

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for public API
// ─────────────────────────────────────────────────────────────────────────────
pub use authority::{
    AbortReason, AuthorityDecision, DecisionSource, ExecutionOutcome, LocalAuthority,
    ModelConstraints, ModelRequest, ModelSelection, ModelSource, OrchestrationAuthority,
    OutcomeCategory, PolicyOutcome, PolicyRequest, RemoteAuthority, ResolvedTarget, SignalContext,
    StageContext, TargetResolution,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal imports
// ─────────────────────────────────────────────────────────────────────────────
use crate::context::{DeviceMetrics, StageDescriptor};
use crate::control_sync::ControlSync;
use crate::device::ResourceMonitor;
use crate::event_bus::{EventBus, OrchestratorEvent};
use crate::executor::{Executor, ExecutorError};
use crate::ir::Envelope;
use crate::streaming::manager::{StreamManager, StreamManagerConfig as StreamConfig};
use crate::telemetry::Telemetry;
use crate::tracing as trace;
use policy_engine::{DefaultPolicyEngine, PolicyEngine};
use routing_engine::{
    DefaultRoutingEngine, LocalAvailability, RouteTarget, RoutingDecision, RoutingEngine,
};
use std::sync::Arc;
use thiserror::Error;
use tokio::task;

/// Error type for orchestrator operations.
#[derive(Error, Debug, Clone)]
pub enum OrchestratorError {
    #[error("Policy evaluation failed: {0}")]
    PolicyEvaluationFailed(String),
    #[error("Routing failed: {0}")]
    RoutingFailed(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Invalid stage: {0}")]
    InvalidStage(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Orchestrator error: {0}")]
    Other(String),
}

/// Result type for orchestrator operations.
pub type OrchestratorResult<T> = Result<T, OrchestratorError>;

/// Execution result from a pipeline stage.
#[derive(Debug, Clone)]
pub struct StageExecutionResult {
    pub stage: String,
    pub output: Envelope,
    pub routing_decision: RoutingDecision,
    pub latency_ms: u32,
}

/// Execution mode for the orchestrator.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionMode {
    /// Batch mode: Process complete envelopes
    Batch,
    /// Streaming mode: Process chunks with buffering
    Streaming,
}

/// Main orchestrator struct that coordinates pipeline execution.
///
/// ## Authority-Based Decisions
///
/// The Orchestrator uses an `OrchestrationAuthority` for all routing and policy decisions.
/// By default, it uses `LocalAuthority` which works completely offline with no phone-home.
///
/// For smarter decisions based on fleet-wide data, configure a `RemoteAuthority` which
/// will automatically fall back to `LocalAuthority` when the backend is unavailable.
///
/// ```rust,ignore
/// use xybrid_core::orchestrator::{Orchestrator, RemoteAuthority};
///
/// // Default: LocalAuthority (fully offline)
/// let orchestrator = Orchestrator::new();
///
/// // With RemoteAuthority (smart routing, with fallback)
/// let authority = Box::new(RemoteAuthority::new("https://api.xybrid.dev"));
/// let orchestrator = Orchestrator::with_authority(authority);
/// ```
pub struct Orchestrator {
    /// The orchestration authority for routing and policy decisions.
    /// Default: LocalAuthority (offline, no phone-home).
    authority: Box<dyn OrchestrationAuthority>,
    /// Policy engine for backward compatibility (load_policies, redact).
    policy_engine: Box<dyn PolicyEngine>,
    /// Routing engine for backward compatibility (record_feedback).
    routing_engine: Box<dyn RoutingEngine>,
    executor: Executor,
    stream_manager: StreamManager,
    event_bus: EventBus,
    telemetry: Arc<Telemetry>,
    resource_monitor: Arc<ResourceMonitor>,
    control_sync: Option<ControlSync>,
    execution_mode: ExecutionMode,
}

impl Orchestrator {
    fn effective_model_id(stage: &StageDescriptor) -> String {
        stage.model.clone().unwrap_or_else(|| stage.name.clone())
    }

    fn build_execution_outcome(
        stage: &StageDescriptor,
        resolution: &TargetResolution,
        latency_ms: u64,
        success: bool,
        error: Option<String>,
        category: Option<OutcomeCategory>,
    ) -> ExecutionOutcome {
        ExecutionOutcome {
            stage_id: stage.name.clone(),
            target: resolution.decision.result.clone(),
            latency_ms,
            success,
            error,
            category,
            model_id: Some(resolution.effective_model_id.clone()),
            signal_context: resolution.signal_context,
        }
    }

    fn outcome_category_from_executor_error(error: &ExecutorError) -> Option<OutcomeCategory> {
        error
            .cloud_fallback_abort_reason()
            .map(|reason| OutcomeCategory::AbortedForCloudFallback { reason })
    }

    /// Creates a new orchestrator with custom components.
    pub fn with_all(
        authority: Box<dyn OrchestrationAuthority>,
        policy_engine: Box<dyn PolicyEngine>,
        routing_engine: Box<dyn RoutingEngine>,
        executor: Executor,
        stream_manager: StreamManager,
        event_bus: EventBus,
        telemetry: Arc<Telemetry>,
        resource_monitor: Arc<ResourceMonitor>,
        control_sync: Option<ControlSync>,
        execution_mode: ExecutionMode,
    ) -> Self {
        Self {
            authority,
            policy_engine,
            routing_engine,
            executor,
            stream_manager,
            event_bus,
            telemetry,
            resource_monitor,
            control_sync,
            execution_mode,
        }
    }

    /// Creates a new orchestrator with default components.
    ///
    /// Uses `LocalAuthority` by default - fully offline, no phone-home.
    pub fn new() -> Self {
        Self::bootstrap(None)
            .expect("orchestrator bootstrap with default configuration should succeed")
    }

    /// Creates a new orchestrator with a custom authority.
    ///
    /// Use this to configure smart routing via `RemoteAuthority`:
    ///
    /// ```rust,ignore
    /// use xybrid_core::orchestrator::{Orchestrator, RemoteAuthority};
    ///
    /// // RemoteAuthority automatically falls back to LocalAuthority
    /// // when the backend is unavailable.
    /// let authority = Box::new(RemoteAuthority::new("https://api.xybrid.dev"));
    /// let orchestrator = Orchestrator::with_authority(authority);
    /// ```
    pub fn with_authority(authority: Box<dyn OrchestrationAuthority>) -> Self {
        let telemetry = Arc::new(Telemetry::new());
        let resource_monitor = ResourceMonitor::global();
        Self {
            authority,
            policy_engine: Box::new(DefaultPolicyEngine::with_default_policy()),
            routing_engine: Box::new(DefaultRoutingEngine::new()),
            executor: Executor::new(),
            stream_manager: StreamManager::new(),
            event_bus: EventBus::new(),
            telemetry,
            resource_monitor,
            control_sync: None,
            execution_mode: ExecutionMode::Batch,
        }
    }

    /// Creates a new orchestrator with custom policy and routing engines.
    ///
    /// Note: This uses `LocalAuthority` internally. For custom authority,
    /// use `with_authority()` instead.
    pub fn with_engines(
        policy_engine: Box<dyn PolicyEngine>,
        routing_engine: Box<dyn RoutingEngine>,
    ) -> Self {
        let telemetry = Arc::new(Telemetry::new());
        let resource_monitor = ResourceMonitor::global();
        Self {
            authority: Box::new(LocalAuthority::new()),
            policy_engine,
            routing_engine,
            executor: Executor::new(),
            stream_manager: StreamManager::new(),
            event_bus: EventBus::new(),
            telemetry,
            resource_monitor,
            control_sync: None,
            execution_mode: ExecutionMode::Batch,
        }
    }

    /// Creates a new orchestrator configured for streaming execution.
    ///
    /// Uses `LocalAuthority` by default - fully offline, no phone-home.
    pub fn with_streaming(config: StreamConfig) -> Self {
        let telemetry = Arc::new(Telemetry::new());
        let resource_monitor = ResourceMonitor::global();
        Self {
            authority: Box::new(LocalAuthority::new()),
            policy_engine: Box::new(DefaultPolicyEngine::with_default_policy()),
            routing_engine: Box::new(DefaultRoutingEngine::new()),
            executor: Executor::new(),
            stream_manager: StreamManager::with_config(config),
            event_bus: EventBus::new(),
            telemetry,
            resource_monitor,
            control_sync: None,
            execution_mode: ExecutionMode::Streaming,
        }
    }

    /// Execute a single pipeline stage.
    ///
    /// This method orchestrates the full lifecycle according to the architecture:
    /// 1. Receive input envelope
    /// 2. Evaluate policy (via OrchestrationAuthority)
    /// 3. Decide route (via OrchestrationAuthority)
    /// 4. Execute model
    /// 5. Emit telemetry
    /// 6. Record outcome for learning
    ///
    /// All routing and policy decisions go through the `OrchestrationAuthority`.
    /// By default, `LocalAuthority` is used (fully offline, no phone-home).
    pub fn execute_stage(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        metrics: &DeviceMetrics,
        _availability: &LocalAvailability,
    ) -> OrchestratorResult<StageExecutionResult> {
        let _start_time = std::time::Instant::now();

        // Step 1: Receive input envelope
        // Emit stage start event
        self.event_bus.publish(OrchestratorEvent::StageStart {
            stage_name: stage.name.clone(),
        });
        self.telemetry.log_stage_start(&stage.name);

        // Step 2: Evaluate policy via OrchestrationAuthority
        let policy_request = PolicyRequest {
            stage_id: stage.name.clone(),
            envelope: input.clone(),
            metrics: metrics.clone(),
        };
        let policy_decision = self.authority.apply_policy(&policy_request);
        let policy_allowed = policy_decision.result.is_allowed();
        let needs_transform = matches!(&policy_decision.result, PolicyOutcome::Transform { .. });

        // Emit policy evaluation event
        self.event_bus.publish(OrchestratorEvent::PolicyEvaluated {
            stage_name: stage.name.clone(),
            allowed: policy_allowed,
            reason: Some(policy_decision.reason.clone()),
        });
        self.telemetry.log_policy_evaluation(
            &stage.name,
            policy_allowed,
            Some(&policy_decision.reason),
        );

        // Apply redaction if transforms needed (use policy_engine for actual redaction)
        let mut redacted_input = input.clone();
        if needs_transform {
            self.policy_engine.redact(&mut redacted_input);
        }

        // Step 3: Resolve target via OrchestrationAuthority
        let stage_context = StageContext {
            stage_id: stage.name.clone(),
            model_id: Self::effective_model_id(stage),
            input_kind: input.kind.clone(),
            metrics: metrics.clone(),
            resource_monitor: self.resource_monitor.clone(),
            explicit_target: stage.target.clone(),
        };
        let target_resolution = self.authority.resolve_target_with_feedback(&stage_context);

        // Convert ResolvedTarget to RoutingDecision for backward compatibility
        let routing_decision =
            self.resolved_target_to_routing_decision(&stage.name, &target_resolution);

        // Emit routing decision event
        self.event_bus.publish(OrchestratorEvent::RoutingDecided {
            stage_name: stage.name.clone(),
            target: routing_decision.target.to_json_string(),
            reason: routing_decision.reason.clone(),
        });
        self.telemetry.log_routing_decision(
            &stage.name,
            &routing_decision.target.to_json_string(),
            &routing_decision.reason,
        );

        // Step 4: Execute model based on routing decision
        self.event_bus.publish(OrchestratorEvent::ExecutionStarted {
            stage_name: stage.name.clone(),
            target: routing_decision.target.to_json_string(),
        });
        self.telemetry
            .log_execution_start(&stage.name, &routing_decision.target.to_json_string());

        let target = routing_decision.target.to_json_string();
        let execution_result = self.executor.execute_stage(stage, &redacted_input, &target);

        let (output, stage_metadata, success, error_msg) = match execution_result {
            Ok((out, meta)) => (out, meta, true, None),
            Err(e) => {
                let error_msg = format!("{:?}", e);
                let category = Self::outcome_category_from_executor_error(&e);
                // Publish a structured event before recording the outcome so
                // listener-driven telemetry sees a terminal counterpart to
                // ExecutionStarted. AbortedForCloudFallback errors have their
                // listener event suppressed by TemplateExecutor (the
                // suppression is contractually paired with this richer event).
                if let Some(OutcomeCategory::AbortedForCloudFallback { reason }) = &category {
                    self.event_bus.publish(OrchestratorEvent::LocalAborted {
                        stage_name: stage.name.clone(),
                        target: routing_decision.target.to_json_string(),
                        reason: reason.as_str().to_string(),
                    });
                } else {
                    self.event_bus.publish(OrchestratorEvent::ExecutionFailed {
                        stage_name: stage.name.clone(),
                        target: routing_decision.target.to_json_string(),
                        error: error_msg.clone(),
                    });
                }
                // Record failure outcome
                let outcome = Self::build_execution_outcome(
                    stage,
                    &target_resolution,
                    0,
                    false,
                    Some(error_msg.clone()),
                    category,
                );
                self.authority.record_outcome(&outcome);
                return Err(OrchestratorError::ExecutionFailed(error_msg));
            }
        };

        let latency_ms = stage_metadata.latency_ms as u32;

        // Step 5: Emit telemetry and events
        self.event_bus
            .publish(OrchestratorEvent::ExecutionCompleted {
                stage_name: stage.name.clone(),
                target: routing_decision.target.to_json_string(),
                execution_time_ms: latency_ms,
            });
        self.telemetry.log_execution_complete(
            &stage.name,
            &routing_decision.target.to_json_string(),
            latency_ms,
        );

        // Step 6: Record outcome for learning (via OrchestrationAuthority)
        let outcome = Self::build_execution_outcome(
            stage,
            &target_resolution,
            latency_ms as u64,
            success,
            error_msg,
            None,
        );
        self.authority.record_outcome(&outcome);

        // Also record feedback for backward compatibility with routing engine
        self.routing_engine
            .record_feedback(&routing_decision, latency_ms);

        // Emit stage completion event and telemetry
        self.event_bus.publish(OrchestratorEvent::StageComplete {
            stage_name: stage.name.clone(),
            target: routing_decision.target.to_json_string(),
            latency_ms,
        });
        self.telemetry.log_stage_complete(
            &stage.name,
            &routing_decision.target.to_json_string(),
            latency_ms,
            None,
        );

        Ok(StageExecutionResult {
            stage: stage.name.clone(),
            output,
            routing_decision,
            latency_ms,
        })
    }

    /// Execute a multi-stage pipeline.
    ///
    /// Stages are executed sequentially, with each stage's output
    /// becoming the next stage's input.
    pub fn execute_pipeline(
        &mut self,
        stages: &[StageDescriptor],
        initial_input: &Envelope,
        metrics: &DeviceMetrics,
        availability_fn: &dyn Fn(&str) -> LocalAvailability,
    ) -> OrchestratorResult<Vec<StageExecutionResult>> {
        let pipeline_start = std::time::Instant::now();
        let stage_names: Vec<String> = stages.iter().map(|s| s.name.clone()).collect();

        // Start top-level pipeline span
        let pipeline_name = stage_names.join(" → ");
        let _pipeline_span = trace::SpanGuard::new(format!("pipeline:{}", pipeline_name));
        trace::add_metadata("stages", stage_names.len().to_string());

        // Emit pipeline start event
        self.event_bus.publish(OrchestratorEvent::PipelineStart {
            stages: stage_names.clone(),
        });

        let mut results = Vec::new();
        let mut current_input = initial_input.clone();

        for stage in stages {
            let availability = availability_fn(&stage.name);
            let result = self.execute_stage(stage, &current_input, metrics, &availability)?;
            current_input = result.output.clone();
            results.push(result);
        }

        let total_latency_ms = pipeline_start.elapsed().as_millis() as u32;

        // Emit pipeline complete event
        self.event_bus
            .publish(OrchestratorEvent::PipelineComplete { total_latency_ms });

        Ok(results)
    }

    /// Execute a single pipeline stage asynchronously.
    ///
    /// This is an async wrapper around `execute_stage` that runs the sync
    /// orchestrator logic in a blocking thread pool.
    ///
    /// All routing and policy decisions go through the `OrchestrationAuthority`.
    /// By default, `LocalAuthority` is used (fully offline, no phone-home).
    ///
    /// # Arguments
    ///
    /// * `stage` - Stage descriptor
    /// * `input` - Input envelope
    /// * `metrics` - Device metrics
    /// * `availability` - Local availability (deprecated, decisions now via authority)
    ///
    /// # Returns
    ///
    /// A future that resolves to the stage execution result
    pub async fn execute_stage_async(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        metrics: &DeviceMetrics,
        _availability: &LocalAvailability,
    ) -> OrchestratorResult<StageExecutionResult> {
        // Emit stage start event (consistent with sync execute_stage)
        self.event_bus.publish(OrchestratorEvent::StageStart {
            stage_name: stage.name.clone(),
        });
        self.telemetry.log_stage_start(&stage.name);

        // Step 2: Evaluate policy via OrchestrationAuthority
        let policy_request = PolicyRequest {
            stage_id: stage.name.clone(),
            envelope: input.clone(),
            metrics: metrics.clone(),
        };
        let policy_decision = self.authority.apply_policy(&policy_request);
        let policy_allowed = policy_decision.result.is_allowed();
        let needs_transform = matches!(&policy_decision.result, PolicyOutcome::Transform { .. });

        // Emit policy evaluation event
        self.event_bus.publish(OrchestratorEvent::PolicyEvaluated {
            stage_name: stage.name.clone(),
            allowed: policy_allowed,
            reason: Some(policy_decision.reason.clone()),
        });
        self.telemetry.log_policy_evaluation(
            &stage.name,
            policy_allowed,
            Some(&policy_decision.reason),
        );

        // Apply redaction if transforms needed (use policy_engine for actual redaction)
        let mut redacted_input = input.clone();
        if needs_transform {
            self.policy_engine.redact(&mut redacted_input);
        }

        // Step 3: Resolve target via OrchestrationAuthority
        let stage_context = StageContext {
            stage_id: stage.name.clone(),
            model_id: Self::effective_model_id(stage),
            input_kind: input.kind.clone(),
            metrics: metrics.clone(),
            resource_monitor: self.resource_monitor.clone(),
            explicit_target: stage.target.clone(),
        };
        let target_resolution = self.authority.resolve_target_with_feedback(&stage_context);

        // Convert ResolvedTarget to RoutingDecision for backward compatibility
        let routing_decision =
            self.resolved_target_to_routing_decision(&stage.name, &target_resolution);

        // Emit routing decision event
        self.event_bus.publish(OrchestratorEvent::RoutingDecided {
            stage_name: stage.name.clone(),
            target: routing_decision.target.to_json_string(),
            reason: routing_decision.reason.clone(),
        });
        self.telemetry.log_routing_decision(
            &stage.name,
            &routing_decision.target.to_json_string(),
            &routing_decision.reason,
        );

        // Execute model in blocking thread pool (adapter execution may be CPU-bound)
        let stage_clone = stage.clone();
        let redacted_input_clone = redacted_input.clone();
        let target = routing_decision.target.to_json_string();

        self.event_bus.publish(OrchestratorEvent::ExecutionStarted {
            stage_name: stage.name.clone(),
            target: routing_decision.target.to_json_string(),
        });
        self.telemetry
            .log_execution_start(&stage.name, &routing_decision.target.to_json_string());

        let mut executor_clone = self.executor.clone();
        let execution_result = task::spawn_blocking(move || {
            executor_clone.execute_stage(&stage_clone, &redacted_input_clone, &target)
        })
        .await
        .map_err(|e| OrchestratorError::ExecutionFailed(format!("Task join error: {}", e)))?;

        let (output, stage_metadata, success, error_msg) = match execution_result {
            Ok((out, meta)) => (out, meta, true, None),
            Err(e) => {
                let error_msg = format!("{:?}", e);
                let category = Self::outcome_category_from_executor_error(&e);
                // Mirror execute_stage: emit a structured terminal event so
                // listener-driven telemetry sees a counterpart to
                // ExecutionStarted even when TemplateExecutor suppresses its
                // own listener event for cooperative cloud-fallback aborts.
                if let Some(OutcomeCategory::AbortedForCloudFallback { reason }) = &category {
                    self.event_bus.publish(OrchestratorEvent::LocalAborted {
                        stage_name: stage.name.clone(),
                        target: routing_decision.target.to_json_string(),
                        reason: reason.as_str().to_string(),
                    });
                } else {
                    self.event_bus.publish(OrchestratorEvent::ExecutionFailed {
                        stage_name: stage.name.clone(),
                        target: routing_decision.target.to_json_string(),
                        error: error_msg.clone(),
                    });
                }
                // Record failure outcome
                let outcome = Self::build_execution_outcome(
                    stage,
                    &target_resolution,
                    0,
                    false,
                    Some(error_msg.clone()),
                    category,
                );
                self.authority.record_outcome(&outcome);
                return Err(OrchestratorError::ExecutionFailed(error_msg));
            }
        };

        let latency_ms = stage_metadata.latency_ms as u32;

        // Emit telemetry and events
        self.event_bus
            .publish(OrchestratorEvent::ExecutionCompleted {
                stage_name: stage.name.clone(),
                target: routing_decision.target.to_json_string(),
                execution_time_ms: latency_ms,
            });
        self.telemetry.log_execution_complete(
            &stage.name,
            &routing_decision.target.to_json_string(),
            latency_ms,
        );

        // Step 6: Record outcome for learning (via OrchestrationAuthority)
        let outcome = Self::build_execution_outcome(
            stage,
            &target_resolution,
            latency_ms as u64,
            success,
            error_msg,
            None,
        );
        self.authority.record_outcome(&outcome);

        // Also record feedback for backward compatibility with routing engine
        self.routing_engine
            .record_feedback(&routing_decision, latency_ms);

        // Emit stage completion event
        self.event_bus.publish(OrchestratorEvent::StageComplete {
            stage_name: stage.name.clone(),
            target: routing_decision.target.to_json_string(),
            latency_ms,
        });
        self.telemetry.log_stage_complete(
            &stage.name,
            &routing_decision.target.to_json_string(),
            latency_ms,
            None,
        );

        Ok(StageExecutionResult {
            stage: stage.name.clone(),
            output,
            routing_decision,
            latency_ms,
        })
    }

    /// Execute a multi-stage pipeline asynchronously.
    ///
    /// This is an async wrapper around `execute_pipeline` that runs stages
    /// sequentially in a blocking thread pool.
    ///
    /// # Arguments
    ///
    /// * `stages` - Stage descriptors
    /// * `initial_input` - Initial input envelope
    /// * `metrics` - Device metrics
    /// * `availability_fn` - Availability function
    ///
    /// # Returns
    ///
    /// A future that resolves to a vector of stage execution results
    pub async fn execute_pipeline_async(
        &mut self,
        stages: &[StageDescriptor],
        initial_input: &Envelope,
        metrics: &DeviceMetrics,
        availability_fn: &dyn Fn(&str) -> LocalAvailability,
    ) -> OrchestratorResult<Vec<StageExecutionResult>> {
        let pipeline_start = std::time::Instant::now();
        let stage_names: Vec<String> = stages.iter().map(|s| s.name.clone()).collect();

        // Start top-level pipeline span
        let pipeline_name = stage_names.join(" → ");
        let _pipeline_span = trace::SpanGuard::new(format!("pipeline:{}", pipeline_name));
        trace::add_metadata("stages", stage_names.len().to_string());

        // Emit pipeline start event
        self.event_bus.publish(OrchestratorEvent::PipelineStart {
            stages: stage_names.clone(),
        });

        let mut results = Vec::new();
        let mut current_input = initial_input.clone();

        // Execute stages sequentially (can be parallelized in future)
        for stage in stages {
            let availability = availability_fn(&stage.name);
            let result = self
                .execute_stage_async(stage, &current_input, metrics, &availability)
                .await?;
            current_input = result.output.clone();
            results.push(result);
        }

        let total_latency_ms = pipeline_start.elapsed().as_millis() as u32;

        // Emit pipeline complete event
        self.event_bus
            .publish(OrchestratorEvent::PipelineComplete { total_latency_ms });

        Ok(results)
    }

    /// Execute a streaming pipeline stage.
    ///
    /// Processes chunks from the stream manager, executing them through the pipeline.
    pub fn execute_streaming_stage(
        &mut self,
        stage: &StageDescriptor,
        metrics: &DeviceMetrics,
        availability: &LocalAvailability,
    ) -> OrchestratorResult<Option<StageExecutionResult>> {
        // Get the next chunk from input buffer
        let Some(input_chunk) = self.stream_manager.pop_input_chunk() else {
            return Ok(None); // No chunks available
        };

        // Process the chunk through the stage
        let result = self.execute_stage(stage, &input_chunk.data, metrics, availability)?;

        // Push output to output buffer
        self.stream_manager
            .push_output_chunk(result.output.clone(), input_chunk.is_last)
            .map_err(|e| OrchestratorError::StreamError(e.to_string()))?;

        Ok(Some(result))
    }

    /// Push a chunk into the streaming pipeline.
    ///
    /// This is used for streaming input (e.g., audio chunks from microphone).
    pub fn push_stream_chunk(
        &mut self,
        envelope: Envelope,
        is_last: bool,
    ) -> OrchestratorResult<()> {
        if self.execution_mode != ExecutionMode::Streaming {
            return Err(OrchestratorError::Other(
                "Orchestrator not in streaming mode".to_string(),
            ));
        }

        self.stream_manager
            .push_input_chunk(envelope, is_last)
            .map_err(|e| OrchestratorError::StreamError(e.to_string()))?;

        Ok(())
    }

    /// Pop a processed chunk from the streaming pipeline.
    ///
    /// This is used to retrieve processed output chunks.
    pub fn pop_stream_output(&mut self) -> Option<crate::streaming::manager::StreamChunk> {
        if self.execution_mode != ExecutionMode::Streaming {
            return None;
        }
        self.stream_manager.pop_output_chunk()
    }

    /// Load policies into the policy engine.
    pub fn load_policies(&mut self, bundle_bytes: Vec<u8>) -> OrchestratorResult<()> {
        self.policy_engine
            .load_policies(bundle_bytes)
            .map_err(OrchestratorError::PolicyEvaluationFailed)
    }

    /// Get a reference to the event bus for subscribing to events.
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    /// Get a mutable reference to the stream manager.
    pub fn stream_manager_mut(&mut self) -> &mut StreamManager {
        &mut self.stream_manager
    }

    /// Get a reference to the executor.
    pub fn executor(&self) -> &Executor {
        &self.executor
    }

    /// Get a mutable reference to the executor.
    pub fn executor_mut(&mut self) -> &mut Executor {
        &mut self.executor
    }

    /// Get the execution mode.
    pub fn execution_mode(&self) -> &ExecutionMode {
        &self.execution_mode
    }

    /// Set the execution mode.
    pub fn set_execution_mode(&mut self, mode: ExecutionMode) {
        self.execution_mode = mode;
    }

    /// Get the authority name (for debugging/logging).
    pub fn authority_name(&self) -> &str {
        self.authority.name()
    }

    /// Invalidate any cached authority decisions.
    ///
    /// Call this when conditions change significantly (e.g., network status change).
    pub fn invalidate_authority_cache(&self) {
        self.authority.invalidate_cache();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────────

    /// Convert a ResolvedTarget to a RoutingDecision for backward compatibility.
    fn resolved_target_to_routing_decision(
        &self,
        stage_name: &str,
        resolution: &TargetResolution,
    ) -> RoutingDecision {
        let decision = &resolution.decision;
        let target = match &decision.result {
            ResolvedTarget::Device => RouteTarget::Local,
            ResolvedTarget::Cloud { .. } => RouteTarget::Cloud,
            ResolvedTarget::Server { endpoint } => RouteTarget::Fallback(endpoint.clone()),
        };

        RoutingDecision {
            stage: stage_name.to_string(),
            target,
            reason: format!(
                "[{}] {} (confidence: {:.0}%)",
                decision.source,
                decision.reason,
                decision.confidence * 100.0
            ),
            timestamp_ms: decision.timestamp_ms,
            local_reliability_hint: resolution.local_reliability_hint.unwrap_or_default(),
        }
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Envelope, EnvelopeKind};
    use crate::pipeline::ExecutionTarget;
    use crate::runtime_adapter::{AdapterError, AdapterResult, RuntimeAdapter};
    use crate::testing::mocks::MockRuntimeAdapter;
    use std::sync::{Arc, Mutex};

    fn text_envelope(value: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(value.to_string()))
    }

    fn audio_envelope(bytes: &[u8]) -> Envelope {
        Envelope::new(EnvelopeKind::Audio(bytes.to_vec()))
    }

    #[derive(Clone, Copy)]
    enum FailureKind {
        CloudFallbackAbort(AbortReason),
        Runtime,
    }

    struct FailingRuntimeAdapter {
        kind: FailureKind,
        loaded: Mutex<bool>,
    }

    impl FailingRuntimeAdapter {
        fn new(kind: FailureKind) -> Self {
            Self {
                kind,
                loaded: Mutex::new(false),
            }
        }
    }

    impl RuntimeAdapter for FailingRuntimeAdapter {
        fn name(&self) -> &str {
            "failing"
        }

        fn supported_formats(&self) -> Vec<&'static str> {
            vec!["onnx"]
        }

        fn load_model(&mut self, _path: &str) -> AdapterResult<()> {
            *self.loaded.lock().unwrap() = true;
            Ok(())
        }

        fn execute(&self, _input: &Envelope) -> AdapterResult<Envelope> {
            assert!(
                *self.loaded.lock().unwrap(),
                "test adapter should be loaded before execution"
            );
            match self.kind {
                FailureKind::CloudFallbackAbort(reason) => {
                    Err(AdapterError::AbortedForCloudFallback { reason })
                }
                FailureKind::Runtime => Err(AdapterError::RuntimeError("boom".to_string())),
            }
        }
    }

    struct RecordingAuthority {
        inner: Arc<LocalAuthority>,
        outcomes: Arc<Mutex<Vec<ExecutionOutcome>>>,
    }

    impl RecordingAuthority {
        fn new(inner: Arc<LocalAuthority>, outcomes: Arc<Mutex<Vec<ExecutionOutcome>>>) -> Self {
            Self { inner, outcomes }
        }
    }

    impl OrchestrationAuthority for RecordingAuthority {
        fn apply_policy(&self, request: &PolicyRequest) -> AuthorityDecision<PolicyOutcome> {
            self.inner.apply_policy(request)
        }

        fn resolve_target(&self, context: &StageContext) -> AuthorityDecision<ResolvedTarget> {
            self.inner.resolve_target(context)
        }

        fn resolve_target_with_feedback(&self, context: &StageContext) -> TargetResolution {
            self.inner.resolve_target_with_feedback(context)
        }

        fn select_model(&self, request: &ModelRequest) -> AuthorityDecision<ModelSelection> {
            self.inner.select_model(request)
        }

        fn record_outcome(&self, outcome: &ExecutionOutcome) {
            self.outcomes.lock().unwrap().push(outcome.clone());
            self.inner.record_outcome(outcome);
        }

        fn name(&self) -> &str {
            "recording"
        }
    }

    /// Helper to create an orchestrator with a pre-loaded mock adapter registered.
    ///
    /// For `Batch` we deliberately use `with_authority(LocalAuthority::new())`
    /// instead of `Orchestrator::new()` — the latter bootstraps the real
    /// ONNX/cloud adapters and wires them as the default for `local`/`cloud`
    /// targets, so a subsequently-added mock would never be selected. The
    /// fresh-executor path leaves the mock as the only registered adapter.
    fn orchestrator_with_mock_adapter(execution_mode: ExecutionMode) -> Orchestrator {
        let mut orchestrator = match execution_mode {
            ExecutionMode::Streaming => Orchestrator::with_streaming(StreamConfig::default()),
            ExecutionMode::Batch => Orchestrator::with_authority(Box::new(LocalAuthority::new())),
        };

        // Register a mock adapter that returns text output
        let mut adapter = MockRuntimeAdapter::with_text_output("mock output");
        adapter.load_model("/mock/model.onnx").unwrap();
        orchestrator
            .executor_mut()
            .register_adapter(Arc::new(adapter));

        orchestrator
    }

    fn orchestrator_with_failing_adapter(
        kind: FailureKind,
        outcomes: Arc<Mutex<Vec<ExecutionOutcome>>>,
        inner: Arc<LocalAuthority>,
    ) -> Orchestrator {
        let authority = RecordingAuthority::new(inner, outcomes);
        let mut orchestrator = Orchestrator::with_authority(Box::new(authority));
        let mut adapter = FailingRuntimeAdapter::new(kind);
        adapter.load_model("/mock/model.onnx").unwrap();
        orchestrator
            .executor_mut()
            .register_adapter(Arc::new(adapter));
        orchestrator
    }

    fn local_routing_metrics() -> DeviceMetrics {
        DeviceMetrics::default()
    }

    fn hysteresis_probe_context(model_id: &str) -> StageContext {
        StageContext {
            stage_id: "abort_stage".to_string(),
            model_id: model_id.to_string(),
            input_kind: EnvelopeKind::Text("probe".to_string()),
            metrics: DeviceMetrics::default(),
            resource_monitor: ResourceMonitor::global(),
            explicit_target: None,
        }
    }

    fn assert_hysteresis_active(authority: &LocalAuthority, model_id: &str, reason: AbortReason) {
        let decision = authority.resolve_target(&hysteresis_probe_context(model_id));

        assert!(matches!(decision.result, ResolvedTarget::Cloud { .. }));
        assert!(decision.reason.contains("hysteresis"));
        assert!(decision.reason.contains(reason.as_str()));
    }

    fn assert_no_hysteresis(authority: &LocalAuthority, model_id: &str) {
        let decision = authority.resolve_target(&hysteresis_probe_context(model_id));

        assert!(!decision.reason.contains("hysteresis"));
    }

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = Orchestrator::new();
        assert_eq!(*orchestrator.execution_mode(), ExecutionMode::Batch);
        drop(orchestrator);
    }

    #[test]
    fn test_execute_single_stage() {
        let mut orchestrator = Orchestrator::new();
        let stage = StageDescriptor::new("test_stage");
        let input = text_envelope("Text");
        let metrics = DeviceMetrics::default();
        let availability = LocalAvailability::new(true);

        let result = orchestrator.execute_stage(&stage, &input, &metrics, &availability);

        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert_eq!(exec_result.stage, "test_stage");
        match &exec_result.output.kind {
            EnvelopeKind::Text(text) => assert!(text.contains("output")),
            other => panic!("expected text output, got {:?}", other),
        }
    }

    #[test]
    fn test_execute_pipeline() {
        // Pipeline mixes locally-available (asr/tts) and cloud-only
        // (motivator) stages; the local stages need a pre-loaded adapter
        // since the mock-output fallback is gone.
        let mut orchestrator = orchestrator_with_mock_adapter(ExecutionMode::Batch);
        let stages = vec![
            StageDescriptor::new("asr"),
            StageDescriptor::new("motivator"),
            StageDescriptor::new("tts"),
        ];
        let input = audio_envelope(&[0u8; 4]);
        let metrics = DeviceMetrics::default();

        let availability_fn = |stage: &str| -> LocalAvailability {
            // Simulate Hiiipe demo: ASR and TTS available locally, motivator only in cloud
            match stage {
                "asr" | "tts" => LocalAvailability::new(true),
                _ => LocalAvailability::new(false),
            }
        };

        let results = orchestrator.execute_pipeline(&stages, &input, &metrics, &availability_fn);

        assert!(results.is_ok());
        let pipeline_results = results.unwrap();
        assert_eq!(pipeline_results.len(), 3);
        assert_eq!(pipeline_results[0].stage, "asr");
        assert_eq!(pipeline_results[1].stage, "motivator");
        assert_eq!(pipeline_results[2].stage, "tts");
    }

    #[test]
    fn test_model_unavailable_routes_to_cloud() {
        // With the new authority-based routing, LocalAuthority checks if the model
        // actually exists locally. Since there's no actual model for "test_stage",
        // it routes to cloud for execution.
        let mut orchestrator = Orchestrator::new();
        let stage = StageDescriptor::new("test_stage");
        let input = audio_envelope(&[9, 9, 9, 9]);
        let metrics = DeviceMetrics::default();
        let availability = LocalAvailability::new(true);

        let result = orchestrator.execute_stage(&stage, &input, &metrics, &availability);

        assert!(result.is_ok());
        let exec_result = result.unwrap();
        // LocalAuthority routes to cloud when model is not found locally
        assert_eq!(exec_result.routing_decision.target.as_str(), "cloud");
        assert!(exec_result
            .routing_decision
            .reason
            .contains("model_unavailable"));
    }

    #[test]
    fn test_streaming_mode() {
        let config = StreamConfig::default();
        let orchestrator = Orchestrator::with_streaming(config);
        assert_eq!(*orchestrator.execution_mode(), ExecutionMode::Streaming);
    }

    #[test]
    fn test_push_and_execute_stream_chunk() {
        // Create orchestrator with mock adapter for streaming mode
        let mut orchestrator = orchestrator_with_mock_adapter(ExecutionMode::Streaming);
        let stage = StageDescriptor::new("asr");
        let envelope = audio_envelope(&[1, 2, 3, 4]);

        // Push chunk
        orchestrator
            .push_stream_chunk(envelope.clone(), false)
            .unwrap();

        // Execute streaming stage
        let metrics = DeviceMetrics::default();
        let availability = LocalAvailability::new(true);

        let result = orchestrator.execute_streaming_stage(&stage, &metrics, &availability);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // Pop output
        let output_chunk = orchestrator.pop_stream_output();
        assert!(output_chunk.is_some());
    }

    #[test]
    fn test_event_bus_access() {
        let orchestrator = Orchestrator::new();
        let _bus = orchestrator.event_bus();
        // Just verify we can access the event bus
    }

    #[test]
    fn test_stream_manager_access() {
        let mut orchestrator = Orchestrator::new();
        let _manager = orchestrator.stream_manager_mut();
        // Just verify we can access the stream manager
    }

    #[test]
    fn typed_cloud_fallback_abort_records_hysteresis_outcome_sync() {
        let outcomes = Arc::new(Mutex::new(Vec::new()));
        let authority = Arc::new(LocalAuthority::new());
        let mut orchestrator = orchestrator_with_failing_adapter(
            FailureKind::CloudFallbackAbort(AbortReason::StressMemory),
            outcomes.clone(),
            authority.clone(),
        );
        let stage = StageDescriptor::new("abort_stage")
            .with_model("effective-model")
            .with_target(ExecutionTarget::Device);
        let input = text_envelope("Text");
        let availability = LocalAvailability::new(true);

        let result =
            orchestrator.execute_stage(&stage, &input, &local_routing_metrics(), &availability);

        assert!(matches!(result, Err(OrchestratorError::ExecutionFailed(_))));
        let recorded = outcomes.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let outcome = &recorded[0];
        assert_eq!(outcome.effective_model_id(), "effective-model");
        assert_eq!(outcome.target, ResolvedTarget::Device);
        assert!(outcome.signal_context.is_some());
        assert_eq!(
            outcome.category,
            Some(OutcomeCategory::AbortedForCloudFallback {
                reason: AbortReason::StressMemory
            })
        );
        drop(recorded);
        assert_hysteresis_active(&authority, "effective-model", AbortReason::StressMemory);
    }

    #[test]
    fn typed_cloud_fallback_abort_records_hysteresis_outcome_async() {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let outcomes = Arc::new(Mutex::new(Vec::new()));
                let authority = Arc::new(LocalAuthority::new());
                let mut orchestrator = orchestrator_with_failing_adapter(
                    FailureKind::CloudFallbackAbort(AbortReason::StressThermal),
                    outcomes.clone(),
                    authority.clone(),
                );
                let stage = StageDescriptor::new("abort_stage")
                    .with_model("async-model")
                    .with_target(ExecutionTarget::Device);
                let input = text_envelope("Text");
                let availability = LocalAvailability::new(true);

                let result = orchestrator
                    .execute_stage_async(&stage, &input, &local_routing_metrics(), &availability)
                    .await;

                assert!(matches!(result, Err(OrchestratorError::ExecutionFailed(_))));
                let recorded = outcomes.lock().unwrap();
                assert_eq!(recorded.len(), 1);
                let outcome = &recorded[0];
                assert_eq!(outcome.effective_model_id(), "async-model");
                assert_eq!(outcome.target, ResolvedTarget::Device);
                assert!(outcome.signal_context.is_some());
                assert_eq!(
                    outcome.category,
                    Some(OutcomeCategory::AbortedForCloudFallback {
                        reason: AbortReason::StressThermal
                    })
                );
                drop(recorded);
                assert_hysteresis_active(&authority, "async-model", AbortReason::StressThermal);
            });
    }

    #[test]
    fn non_abort_failure_records_hard_fail_without_hysteresis() {
        let outcomes = Arc::new(Mutex::new(Vec::new()));
        let authority = Arc::new(LocalAuthority::new());
        let mut orchestrator = orchestrator_with_failing_adapter(
            FailureKind::Runtime,
            outcomes.clone(),
            authority.clone(),
        );
        let stage = StageDescriptor::new("abort_stage")
            .with_model("hard-fail-model")
            .with_target(ExecutionTarget::Device);
        let input = text_envelope("Text");
        let availability = LocalAvailability::new(true);

        let result =
            orchestrator.execute_stage(&stage, &input, &local_routing_metrics(), &availability);

        assert!(matches!(result, Err(OrchestratorError::ExecutionFailed(_))));
        let recorded = outcomes.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let outcome = &recorded[0];
        assert_eq!(outcome.effective_model_id(), "hard-fail-model");
        assert_eq!(outcome.target, ResolvedTarget::Device);
        assert!(matches!(
            outcome.effective_category(),
            OutcomeCategory::HardFail { .. }
        ));
        assert_eq!(outcome.category, None);
        drop(recorded);
        assert_no_hysteresis(&authority, "hard-fail-model");
    }
}

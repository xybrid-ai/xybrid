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

use crate::context::{DeviceMetrics, StageDescriptor};
use crate::control_sync::ControlSync;
use crate::event_bus::{EventBus, OrchestratorEvent};
use crate::executor::Executor;
use crate::ir::Envelope;
use crate::pipeline::ExecutionTarget;
use self::policy_engine::{DefaultPolicyEngine, PolicyEngine};
use self::routing_engine::{
    DefaultRoutingEngine, LocalAvailability, RouteTarget, RoutingDecision, RoutingEngine,
};
use crate::streaming::manager::{StreamManager, StreamManagerConfig as StreamConfig};
use crate::telemetry::Telemetry;
use crate::tracing as trace;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

/// Error type for orchestrator operations.
#[derive(Debug, Clone)]
pub enum OrchestratorError {
    PolicyEvaluationFailed(String),
    RoutingFailed(String),
    ExecutionFailed(String),
    InvalidStage(String),
    StreamError(String),
    Other(String),
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorError::PolicyEvaluationFailed(msg) => {
                write!(f, "Policy evaluation failed: {}", msg)
            }
            OrchestratorError::RoutingFailed(msg) => {
                write!(f, "Routing failed: {}", msg)
            }
            OrchestratorError::ExecutionFailed(msg) => {
                write!(f, "Execution failed: {}", msg)
            }
            OrchestratorError::InvalidStage(msg) => {
                write!(f, "Invalid stage: {}", msg)
            }
            OrchestratorError::StreamError(msg) => {
                write!(f, "Stream error: {}", msg)
            }
            OrchestratorError::Other(msg) => {
                write!(f, "Orchestrator error: {}", msg)
            }
        }
    }
}

impl std::error::Error for OrchestratorError {}

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
pub struct Orchestrator {
    policy_engine: Box<dyn PolicyEngine>,
    routing_engine: Box<dyn RoutingEngine>,
    executor: Executor,
    stream_manager: StreamManager,
    event_bus: EventBus,
    telemetry: Arc<Telemetry>,
    control_sync: Option<ControlSync>,
    execution_mode: ExecutionMode,
}

impl Orchestrator {
    /// Creates a new orchestrator with custom components.
    pub fn with_all(
        policy_engine: Box<dyn PolicyEngine>,
        routing_engine: Box<dyn RoutingEngine>,
        executor: Executor,
        stream_manager: StreamManager,
        event_bus: EventBus,
        telemetry: Arc<Telemetry>,
        control_sync: Option<ControlSync>,
        execution_mode: ExecutionMode,
    ) -> Self {
        Self {
            policy_engine,
            routing_engine,
            executor,
            stream_manager,
            event_bus,
            telemetry,
            control_sync,
            execution_mode,
        }
    }

    /// Creates a new orchestrator with default components.
    pub fn new() -> Self {
        Self::bootstrap(None)
            .expect("orchestrator bootstrap with default configuration should succeed")
    }

    /// Creates a new orchestrator with custom policy and routing engines.
    pub fn with_engines(
        policy_engine: Box<dyn PolicyEngine>,
        routing_engine: Box<dyn RoutingEngine>,
    ) -> Self {
        let telemetry = Arc::new(Telemetry::new());
        Self {
            policy_engine,
            routing_engine,
            executor: Executor::new(),
            stream_manager: StreamManager::new(),
            event_bus: EventBus::new(),
            telemetry,
            control_sync: None,
            execution_mode: ExecutionMode::Batch,
        }
    }

    /// Creates a new orchestrator configured for streaming execution.
    pub fn with_streaming(config: StreamConfig) -> Self {
        let telemetry = Arc::new(Telemetry::new());
        Self {
            policy_engine: Box::new(DefaultPolicyEngine::with_default_policy()),
            routing_engine: Box::new(DefaultRoutingEngine::new()),
            executor: Executor::new(),
            stream_manager: StreamManager::with_config(config),
            event_bus: EventBus::new(),
            telemetry,
            control_sync: None,
            execution_mode: ExecutionMode::Streaming,
        }
    }

    /// Execute a single pipeline stage.
    ///
    /// This method orchestrates the full lifecycle according to the architecture:
    /// 1. Receive input envelope
    /// 2. Evaluate policy
    /// 3. Decide route
    /// 4. Execute model
    /// 5. Emit telemetry
    pub fn execute_stage(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        metrics: &DeviceMetrics,
        availability: &LocalAvailability,
    ) -> OrchestratorResult<StageExecutionResult> {
        let _start_time = std::time::Instant::now();

        // Step 1: Receive input envelope
        // Emit stage start event
        self.event_bus.publish(OrchestratorEvent::StageStart {
            stage_name: stage.name.clone(),
        });
        self.telemetry.log_stage_start(&stage.name);

        // Step 2: Evaluate policy
        let policy_result = self.policy_engine.evaluate(&stage.name, input, metrics);

        // Emit policy evaluation event
        self.event_bus.publish(OrchestratorEvent::PolicyEvaluated {
            stage_name: stage.name.clone(),
            allowed: policy_result.allowed,
            reason: policy_result.reason.clone(),
        });
        self.telemetry.log_policy_evaluation(
            &stage.name,
            policy_result.allowed,
            policy_result.reason.as_deref(),
        );

        // Apply redaction if needed
        let mut redacted_input = input.clone();
        if !policy_result.transforms_applied.is_empty() {
            self.policy_engine.redact(&mut redacted_input);
        }

        // Step 3: Decide route
        // If stage has an explicit target (not Auto), use it directly
        // Otherwise, use the routing engine for dynamic routing
        let routing_decision = match &stage.target {
            Some(ExecutionTarget::Device) => {
                // Explicit device target - bypass routing engine
                RoutingDecision {
                    stage: stage.name.clone(),
                    target: RouteTarget::Local,
                    reason: "explicit_target: stage declares target=device".to_string(),
                    timestamp_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }
            Some(ExecutionTarget::Server) => {
                // Explicit server target - use cloud adapter
                RoutingDecision {
                    stage: stage.name.clone(),
                    target: RouteTarget::Cloud,
                    reason: "explicit_target: stage declares target=server".to_string(),
                    timestamp_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }
            Some(ExecutionTarget::Integration) => {
                // Integration target - handled specially by executor
                // Return a routing decision that indicates integration
                RoutingDecision {
                    stage: stage.name.clone(),
                    target: RouteTarget::Cloud, // Will be handled as integration by executor
                    reason: "explicit_target: stage declares target=integration".to_string(),
                    timestamp_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }
            Some(ExecutionTarget::Auto) | None => {
                // Auto or unspecified - use routing engine for dynamic decision
                self.routing_engine
                    .decide(&stage.name, metrics, &policy_result, availability)
            }
        };

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
        let (output, stage_metadata) = self
            .executor
            .execute_stage(stage, &redacted_input, &target)
            .map_err(|e| OrchestratorError::ExecutionFailed(format!("{:?}", e)))?;

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

        // Record feedback for adaptive routing
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
        trace::add_metadata("stages", &stage_names.len().to_string());

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
    /// # Arguments
    ///
    /// * `stage` - Stage descriptor
    /// * `input` - Input envelope
    /// * `metrics` - Device metrics
    /// * `availability` - Local availability
    ///
    /// # Returns
    ///
    /// A future that resolves to the stage execution result
    pub async fn execute_stage_async(
        &mut self,
        stage: &StageDescriptor,
        input: &Envelope,
        metrics: &DeviceMetrics,
        availability: &LocalAvailability,
    ) -> OrchestratorResult<StageExecutionResult> {
        // Emit stage start event (consistent with sync execute_stage)
        self.event_bus.publish(OrchestratorEvent::StageStart {
            stage_name: stage.name.clone(),
        });
        self.telemetry.log_stage_start(&stage.name);

        // Run policy and routing on the async runtime (they're fast)
        let policy_result = self.policy_engine.evaluate(&stage.name, input, metrics);

        // Emit policy evaluation event
        self.event_bus.publish(OrchestratorEvent::PolicyEvaluated {
            stage_name: stage.name.clone(),
            allowed: policy_result.allowed,
            reason: policy_result.reason.clone(),
        });
        self.telemetry.log_policy_evaluation(
            &stage.name,
            policy_result.allowed,
            policy_result.reason.as_deref(),
        );

        // Apply redaction if needed
        let mut redacted_input = input.clone();
        if !policy_result.transforms_applied.is_empty() {
            self.policy_engine.redact(&mut redacted_input);
        }

        // Decide route
        // If stage has an explicit target (not Auto), use it directly
        // Otherwise, use the routing engine for dynamic routing
        let routing_decision = match &stage.target {
            Some(ExecutionTarget::Device) => {
                // Explicit device target - bypass routing engine
                RoutingDecision {
                    stage: stage.name.clone(),
                    target: RouteTarget::Local,
                    reason: "explicit_target: stage declares target=device".to_string(),
                    timestamp_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }
            Some(ExecutionTarget::Server) => {
                // Explicit server target - use cloud adapter
                RoutingDecision {
                    stage: stage.name.clone(),
                    target: RouteTarget::Cloud,
                    reason: "explicit_target: stage declares target=server".to_string(),
                    timestamp_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }
            Some(ExecutionTarget::Integration) => {
                // Integration target - handled specially by executor
                RoutingDecision {
                    stage: stage.name.clone(),
                    target: RouteTarget::Cloud, // Will be handled as integration by executor
                    reason: "explicit_target: stage declares target=integration".to_string(),
                    timestamp_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }
            Some(ExecutionTarget::Auto) | None => {
                // Auto or unspecified - use routing engine for dynamic decision
                self.routing_engine
                    .decide(&stage.name, metrics, &policy_result, availability)
            }
        };

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
        let (output, stage_metadata) = task::spawn_blocking(move || {
            executor_clone.execute_stage(&stage_clone, &redacted_input_clone, &target)
        })
        .await
        .map_err(|e| OrchestratorError::ExecutionFailed(format!("Task join error: {}", e)))?
        .map_err(|e| OrchestratorError::ExecutionFailed(format!("{:?}", e)))?;

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

        // Record feedback for adaptive routing
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
        trace::add_metadata("stages", &stage_names.len().to_string());

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
            .map_err(|e| OrchestratorError::PolicyEvaluationFailed(e))
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
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// Bootstrap module for orchestrator initialization
pub mod bootstrap;
pub mod policy_engine;
pub mod routing_engine;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Envelope, EnvelopeKind};

    fn text_envelope(value: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(value.to_string()))
    }

    fn audio_envelope(bytes: &[u8]) -> Envelope {
        Envelope::new(EnvelopeKind::Audio(bytes.to_vec()))
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
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
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
        let mut orchestrator = Orchestrator::new();
        let stages = vec![
            StageDescriptor::new("asr"),
            StageDescriptor::new("motivator"),
            StageDescriptor::new("tts"),
        ];
        let input = audio_envelope(&[0u8; 4]);
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };

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
    fn test_policy_deny_routes_to_local() {
        let mut orchestrator = Orchestrator::new();
        let stage = StageDescriptor::new("test_stage");
        let input = audio_envelope(&[9, 9, 9, 9]); // This should be denied by default policy
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
        let availability = LocalAvailability::new(true);

        let result = orchestrator.execute_stage(&stage, &input, &metrics, &availability);

        assert!(result.is_ok());
        let exec_result = result.unwrap();
        // Policy should deny cloud execution, routing should choose local
        assert_eq!(exec_result.routing_decision.target.as_str(), "local");
    }

    #[test]
    fn test_high_rtt_routes_to_local() {
        let mut orchestrator = Orchestrator::new();
        let stage = StageDescriptor::new("test_stage");
        let input = text_envelope("Text");
        let metrics = DeviceMetrics {
            network_rtt: 300, // High RTT should trigger local routing
            battery: 50,
            temperature: 25.0,
        };
        let availability = LocalAvailability::new(true);

        let result = orchestrator.execute_stage(&stage, &input, &metrics, &availability);

        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert_eq!(exec_result.routing_decision.target.as_str(), "local");
        assert!(exec_result.routing_decision.reason.contains("high_latency"));
    }

    #[test]
    fn test_streaming_mode() {
        let config = StreamConfig::default();
        let orchestrator = Orchestrator::with_streaming(config);
        assert_eq!(*orchestrator.execution_mode(), ExecutionMode::Streaming);
    }

    #[test]
    fn test_push_and_execute_stream_chunk() {
        let mut orchestrator = Orchestrator::with_streaming(StreamConfig::default());
        let stage = StageDescriptor::new("asr");
        let envelope = audio_envelope(&[1, 2, 3, 4]);

        // Push chunk
        orchestrator
            .push_stream_chunk(envelope.clone(), false)
            .unwrap();

        // Execute streaming stage
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
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
}

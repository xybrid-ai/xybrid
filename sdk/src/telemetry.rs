//! Telemetry event bridge - Converts OrchestratorEvent to TelemetryEvent
//!
//! This module bridges events from the orchestrator's event bus to the
//! telemetry stream used by Flutter and other consumers. It also supports
//! exporting telemetry to the Xybrid Platform for analytics and monitoring.
//!
//! # Span Collection
//!
//! This module integrates with `xybrid_core::tracing` to capture execution spans.
//! When a pipeline completes, the span tree is automatically included in the
//! `PipelineComplete` telemetry event and sent to the Platform for visualization.
//!
//! # Resilience Features
//!
//! The HTTP exporter includes production-hardening features:
//! - **Circuit breaker**: Prevents hammering failing endpoints
//! - **Automatic retry**: Exponential backoff with jitter for transient failures
//! - **Failed event queue**: Retries failed events in the background

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use xybrid_core::event_bus::OrchestratorEvent;
use xybrid_core::http::{CircuitBreaker, CircuitConfig, RetryPolicy};
use xybrid_core::tracing as core_tracing;

/// Telemetry event type (simplified for FFI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Event type name
    pub event_type: String,
    /// Stage name (if applicable)
    pub stage_name: Option<String>,
    /// Target (local/cloud/fallback)
    pub target: Option<String>,
    /// Latency in milliseconds (if applicable)
    pub latency_ms: Option<u32>,
    /// Error message (if applicable)
    pub error: Option<String>,
    /// Additional event data as JSON string
    pub data: Option<String>,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
}

/// Global telemetry event channel for streaming
pub type TelemetrySender = mpsc::Sender<TelemetryEvent>;

static TELEMETRY_SENDERS: Mutex<Vec<TelemetrySender>> = Mutex::new(Vec::new());

// ============================================================================
// HTTP Platform Exporter
// ============================================================================

/// Maximum number of events to keep in the failed queue
const MAX_FAILED_QUEUE_SIZE: usize = 1000;

/// Connection timeout for telemetry requests (5 seconds)
const CONNECT_TIMEOUT_MS: u64 = 5000;

/// Request timeout for telemetry requests (10 seconds)
const REQUEST_TIMEOUT_MS: u64 = 10000;

/// Configuration for the HTTP telemetry exporter
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Platform API endpoint URL (e.g., "https://api.xybrid.dev")
    pub endpoint: String,
    /// API key for authentication
    pub api_key: String,
    /// Session ID for grouping events (generated if not provided)
    pub session_id: Uuid,
    /// Device identifier
    pub device_id: Option<String>,
    /// Platform name (e.g., "ios", "android", "macos")
    pub platform: Option<String>,
    /// App version string
    pub app_version: Option<String>,
    /// Batch size before flushing (default: 10)
    pub batch_size: usize,
    /// Flush interval in seconds (default: 5)
    pub flush_interval_secs: u64,
    /// Maximum retry attempts for failed batches (default: 3)
    pub max_retries: u32,
    /// Enable retry queue for failed events (default: true)
    pub enable_retry_queue: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            api_key: String::new(),
            session_id: Uuid::new_v4(),
            device_id: None,
            platform: None,
            app_version: None,
            batch_size: 10,
            flush_interval_secs: 5,
            max_retries: 3,
            enable_retry_queue: true,
        }
    }
}

impl TelemetryConfig {
    /// Create a new config with endpoint and API key
    pub fn new(endpoint: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Set the session ID
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = session_id;
        self
    }

    /// Set device metadata
    pub fn with_device(mut self, device_id: impl Into<String>, platform: impl Into<String>) -> Self {
        self.device_id = Some(device_id.into());
        self.platform = Some(platform.into());
        self
    }

    /// Set app version
    pub fn with_app_version(mut self, version: impl Into<String>) -> Self {
        self.app_version = Some(version.into());
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set flush interval
    pub fn with_flush_interval(mut self, secs: u64) -> Self {
        self.flush_interval_secs = secs;
        self
    }
}

/// Event payload for platform API (matches IngestTelemetryEvent)
#[derive(Debug, Clone, Serialize)]
struct PlatformEvent {
    session_id: Uuid,
    event_type: String,
    payload: serde_json::Value,
    device_id: Option<String>,
    platform: Option<String>,
    app_version: Option<String>,
    timestamp: Option<String>,
    pipeline_id: Option<Uuid>,
    trace_id: Option<Uuid>,
    stages: Option<serde_json::Value>,
}

/// Batch payload for platform API
#[derive(Debug, Serialize)]
struct PlatformEventBatch {
    events: Vec<PlatformEvent>,
}

/// HTTP telemetry exporter that sends events to the Xybrid Platform
///
/// # Resilience Features
///
/// - **Circuit breaker**: Opens after 3 consecutive failures, stays open for 30s
/// - **Automatic retry**: Up to 3 attempts with exponential backoff
/// - **Failed event queue**: Stores up to 1000 failed events for later retry
pub struct HttpTelemetryExporter {
    config: TelemetryConfig,
    buffer: Arc<Mutex<Vec<TelemetryEvent>>>,
    running: Arc<AtomicBool>,
    /// Current pipeline context for enriching events
    pipeline_id: Arc<RwLock<Option<Uuid>>>,
    trace_id: Arc<RwLock<Option<Uuid>>>,
    /// HTTP agent with timeouts configured
    agent: ureq::Agent,
    /// Circuit breaker for the telemetry endpoint
    circuit: Arc<CircuitBreaker>,
    /// Retry policy for batch submissions
    retry_policy: RetryPolicy,
    /// Queue for failed events that will be retried
    failed_queue: Arc<Mutex<VecDeque<PlatformEvent>>>,
    /// Counter for dropped events (when queue is full)
    dropped_count: Arc<AtomicU32>,
}

impl HttpTelemetryExporter {
    /// Create a new HTTP exporter with the given configuration
    pub fn new(config: TelemetryConfig) -> Self {
        // Create HTTP agent with timeouts
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .build();

        // Circuit breaker: open after 3 failures, stay open for 30s
        let circuit = Arc::new(CircuitBreaker::new(CircuitConfig::default()));

        // Retry policy with configurable max attempts
        let retry_policy = RetryPolicy {
            max_attempts: config.max_retries,
            initial_delay_ms: 500,
            max_delay_ms: 5000,
            jitter_factor: 0.3,
        };

        Self {
            config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            pipeline_id: Arc::new(RwLock::new(None)),
            trace_id: Arc::new(RwLock::new(None)),
            agent,
            circuit,
            retry_policy,
            failed_queue: Arc::new(Mutex::new(VecDeque::new())),
            dropped_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Create from environment variables
    ///
    /// Reads:
    /// - `XYBRID_API_KEY` - Required API key
    /// - `XYBRID_PLATFORM_URL` - Platform endpoint (default: https://api.xybrid.dev)
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("XYBRID_API_KEY").ok()?;
        let endpoint = std::env::var("XYBRID_PLATFORM_URL")
            .unwrap_or_else(|_| "https://api.xybrid.dev".to_string());

        let config = TelemetryConfig::new(endpoint, api_key);
        Some(Self::new(config))
    }

    /// Set the current pipeline context for event enrichment
    pub fn set_pipeline_context(&self, pipeline_id: Option<Uuid>, trace_id: Option<Uuid>) {
        if let Ok(mut pid) = self.pipeline_id.write() {
            *pid = pipeline_id;
        }
        if let Ok(mut tid) = self.trace_id.write() {
            *tid = trace_id;
        }
    }

    /// Check if the circuit breaker is open (blocking requests).
    pub fn is_circuit_open(&self) -> bool {
        self.circuit.is_open()
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset_circuit(&self) {
        self.circuit.reset();
    }

    /// Get the number of events waiting in the failed queue.
    pub fn failed_queue_size(&self) -> usize {
        self.failed_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Get the number of events that were dropped due to queue overflow.
    pub fn dropped_count(&self) -> u32 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Start the background flush thread
    pub fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return; // Already running
        }

        let buffer = Arc::clone(&self.buffer);
        let running = Arc::clone(&self.running);
        let config = self.config.clone();
        let flush_interval = Duration::from_secs(config.flush_interval_secs);
        let pipeline_id = Arc::clone(&self.pipeline_id);
        let trace_id = Arc::clone(&self.trace_id);
        let agent = self.agent.clone();
        let circuit = Arc::clone(&self.circuit);
        let retry_policy = self.retry_policy.clone();
        let failed_queue = Arc::clone(&self.failed_queue);
        let dropped_count = Arc::clone(&self.dropped_count);

        thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                thread::sleep(flush_interval);

                // First, try to send any failed events from the queue
                if config.enable_retry_queue {
                    retry_failed_events(
                        &failed_queue,
                        &config,
                        &agent,
                        &circuit,
                        &retry_policy,
                    );
                }

                // Then flush the current buffer
                flush_buffer_with_retry(
                    &buffer,
                    &config,
                    &pipeline_id,
                    &trace_id,
                    &agent,
                    &circuit,
                    &retry_policy,
                    &failed_queue,
                    &dropped_count,
                );
            }
        });
    }

    /// Stop the background flush thread
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        // Final flush with retry
        flush_buffer_with_retry(
            &self.buffer,
            &self.config,
            &self.pipeline_id,
            &self.trace_id,
            &self.agent,
            &self.circuit,
            &self.retry_policy,
            &self.failed_queue,
            &self.dropped_count,
        );
    }

    /// Add an event to the buffer
    pub fn push(&self, event: TelemetryEvent) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.push(event);

        // Flush if buffer is full
        if buffer.len() >= self.config.batch_size {
            let events: Vec<TelemetryEvent> = buffer.drain(..).collect();
            drop(buffer); // Release lock before HTTP call
            send_batch_with_retry(
                &events,
                &self.config,
                &self.pipeline_id,
                &self.trace_id,
                &self.agent,
                &self.circuit,
                &self.retry_policy,
                &self.failed_queue,
                &self.dropped_count,
            );
        }
    }

    /// Force flush all buffered events
    pub fn flush(&self) {
        flush_buffer_with_retry(
            &self.buffer,
            &self.config,
            &self.pipeline_id,
            &self.trace_id,
            &self.agent,
            &self.circuit,
            &self.retry_policy,
            &self.failed_queue,
            &self.dropped_count,
        );
    }

    /// Create a telemetry sender that feeds into this exporter
    pub fn create_sender(&self) -> TelemetrySender {
        let (tx, rx) = mpsc::channel::<TelemetryEvent>();
        let buffer = Arc::clone(&self.buffer);
        let batch_size = self.config.batch_size;
        let config = self.config.clone();
        let pipeline_id = Arc::clone(&self.pipeline_id);
        let trace_id = Arc::clone(&self.trace_id);
        let agent = self.agent.clone();
        let circuit = Arc::clone(&self.circuit);
        let retry_policy = self.retry_policy.clone();
        let failed_queue = Arc::clone(&self.failed_queue);
        let dropped_count = Arc::clone(&self.dropped_count);

        thread::spawn(move || {
            for event in rx {
                let mut buf = buffer.lock().unwrap();
                buf.push(event);

                if buf.len() >= batch_size {
                    let events: Vec<TelemetryEvent> = buf.drain(..).collect();
                    drop(buf);
                    send_batch_with_retry(
                        &events,
                        &config,
                        &pipeline_id,
                        &trace_id,
                        &agent,
                        &circuit,
                        &retry_policy,
                        &failed_queue,
                        &dropped_count,
                    );
                }
            }
        });

        tx
    }
}

impl Drop for HttpTelemetryExporter {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Flush all buffered events to the platform with retry logic.
fn flush_buffer_with_retry(
    buffer: &Arc<Mutex<Vec<TelemetryEvent>>>,
    config: &TelemetryConfig,
    pipeline_id: &Arc<RwLock<Option<Uuid>>>,
    trace_id: &Arc<RwLock<Option<Uuid>>>,
    agent: &ureq::Agent,
    circuit: &Arc<CircuitBreaker>,
    retry_policy: &RetryPolicy,
    failed_queue: &Arc<Mutex<VecDeque<PlatformEvent>>>,
    dropped_count: &Arc<AtomicU32>,
) {
    let events: Vec<TelemetryEvent> = {
        let mut buf = buffer.lock().unwrap();
        buf.drain(..).collect()
    };

    if !events.is_empty() {
        send_batch_with_retry(
            &events,
            config,
            pipeline_id,
            trace_id,
            agent,
            circuit,
            retry_policy,
            failed_queue,
            dropped_count,
        );
    }
}

/// Send a batch of events to the platform API with retry and circuit breaker.
fn send_batch_with_retry(
    events: &[TelemetryEvent],
    config: &TelemetryConfig,
    pipeline_id: &Arc<RwLock<Option<Uuid>>>,
    trace_id: &Arc<RwLock<Option<Uuid>>>,
    agent: &ureq::Agent,
    circuit: &Arc<CircuitBreaker>,
    retry_policy: &RetryPolicy,
    failed_queue: &Arc<Mutex<VecDeque<PlatformEvent>>>,
    dropped_count: &Arc<AtomicU32>,
) {
    if events.is_empty() || config.endpoint.is_empty() || config.api_key.is_empty() {
        return;
    }

    // Check circuit breaker
    if !circuit.can_execute() {
        // Circuit is open, queue events for later
        if config.enable_retry_queue {
            let pid = pipeline_id.read().ok().and_then(|g| *g);
            let tid = trace_id.read().ok().and_then(|g| *g);
            let platform_events: Vec<PlatformEvent> = events
                .iter()
                .map(|e| convert_to_platform_event(e, config, pid, tid))
                .collect();
            queue_failed_events(platform_events, failed_queue, dropped_count);
        }
        return;
    }

    let pid = pipeline_id.read().ok().and_then(|g| *g);
    let tid = trace_id.read().ok().and_then(|g| *g);

    let platform_events: Vec<PlatformEvent> = events
        .iter()
        .map(|e| convert_to_platform_event(e, config, pid, tid))
        .collect();

    // Try to send with retry
    let result = send_batch_inner(&platform_events, config, agent, circuit, retry_policy);

    if let Err(failed_events) = result {
        // Queue failed events for later retry
        if config.enable_retry_queue {
            queue_failed_events(failed_events, failed_queue, dropped_count);
        }
    }
}

/// Inner send function that returns the events on failure for queueing.
fn send_batch_inner(
    events: &[PlatformEvent],
    config: &TelemetryConfig,
    agent: &ureq::Agent,
    circuit: &Arc<CircuitBreaker>,
    retry_policy: &RetryPolicy,
) -> Result<(), Vec<PlatformEvent>> {
    let batch = PlatformEventBatch {
        events: events.to_vec(),
    };

    let url = format!("{}/v1/telemetry/batch", config.endpoint.trim_end_matches('/'));

    for attempt in 0..retry_policy.max_attempts {
        // Calculate delay for this attempt
        let delay = retry_policy.delay_for_attempt(attempt);
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }

        // Check circuit breaker again
        if !circuit.can_execute() {
            return Err(events.to_vec());
        }

        // Send HTTP request
        let result = agent
            .post(&url)
            .set("Authorization", &format!("Bearer {}", config.api_key))
            .set("Content-Type", "application/json")
            .send_json(&batch);

        match result {
            Ok(response) => {
                let status = response.status();
                if status == 200 || status == 201 {
                    circuit.record_success();
                    return Ok(());
                } else if is_retryable_status(status) {
                    circuit.record_failure();
                    // Continue to retry
                } else {
                    // Non-retryable error (4xx client errors)
                    circuit.record_success(); // Don't trip circuit for client errors
                    log::warn!(
                        target: "xybrid_telemetry",
                        "Platform returned status {}",
                        status
                    );
                    return Ok(()); // Don't retry or queue client errors
                }
            }
            Err(ureq::Error::Status(status, _)) => {
                if status == 429 {
                    circuit.record_rate_limited();
                } else if is_retryable_status(status) {
                    circuit.record_failure();
                } else {
                    // Non-retryable status
                    circuit.record_success();
                    log::warn!(
                        target: "xybrid_telemetry",
                        "Platform returned status {}",
                        status
                    );
                    return Ok(());
                }
            }
            Err(ureq::Error::Transport(_)) => {
                circuit.record_failure();
                // Continue to retry
            }
        }
    }

    // All retries exhausted
    Err(events.to_vec())
}

/// Check if an HTTP status code is retryable.
fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 502 | 503 | 504)
}

/// Queue failed events for later retry.
fn queue_failed_events(
    events: Vec<PlatformEvent>,
    failed_queue: &Arc<Mutex<VecDeque<PlatformEvent>>>,
    dropped_count: &Arc<AtomicU32>,
) {
    let mut queue = failed_queue.lock().unwrap();

    for event in events {
        if queue.len() >= MAX_FAILED_QUEUE_SIZE {
            // Queue is full, drop oldest event
            queue.pop_front();
            dropped_count.fetch_add(1, Ordering::Relaxed);
        }
        queue.push_back(event);
    }
}

/// Retry sending failed events from the queue.
fn retry_failed_events(
    failed_queue: &Arc<Mutex<VecDeque<PlatformEvent>>>,
    config: &TelemetryConfig,
    agent: &ureq::Agent,
    circuit: &Arc<CircuitBreaker>,
    retry_policy: &RetryPolicy,
) {
    // Don't retry if circuit is open
    if !circuit.can_execute() {
        return;
    }

    // Take a batch of events from the queue
    let events: Vec<PlatformEvent> = {
        let mut queue = failed_queue.lock().unwrap();
        let batch_size = config.batch_size.min(queue.len());
        queue.drain(..batch_size).collect()
    };

    if events.is_empty() {
        return;
    }

    // Try to send the batch
    if let Err(failed_events) = send_batch_inner(&events, config, agent, circuit, retry_policy) {
        // Put them back at the front of the queue
        let mut queue = failed_queue.lock().unwrap();
        for event in failed_events.into_iter().rev() {
            queue.push_front(event);
        }
    }
}

/// Convert SDK TelemetryEvent to Platform format
fn convert_to_platform_event(
    event: &TelemetryEvent,
    config: &TelemetryConfig,
    pipeline_id: Option<Uuid>,
    trace_id: Option<Uuid>,
) -> PlatformEvent {
    // Build payload from event fields
    let mut payload = serde_json::json!({});

    if let Some(stage) = &event.stage_name {
        payload["stage_name"] = serde_json::json!(stage);
    }
    if let Some(target) = &event.target {
        payload["target"] = serde_json::json!(target);
    }
    if let Some(latency) = event.latency_ms {
        payload["latency_ms"] = serde_json::json!(latency);
    }
    if let Some(error) = &event.error {
        payload["error"] = serde_json::json!(error);
        payload["status"] = serde_json::json!("error");
    } else {
        payload["status"] = serde_json::json!("success");
    }
    if let Some(data) = &event.data {
        // Try to parse as JSON, otherwise store as string
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
            payload["data"] = parsed;
        } else {
            payload["data"] = serde_json::json!(data);
        }
    }

    // Convert timestamp
    let timestamp = chrono::DateTime::from_timestamp_millis(event.timestamp_ms as i64)
        .map(|dt| dt.to_rfc3339());

    // Capture spans for PipelineComplete and ModelComplete events
    // This includes the full span tree from TemplateExecutor instrumentation
    let stages = if (event.event_type == "PipelineComplete" || event.event_type == "ModelComplete")
        && core_tracing::is_tracing_enabled()
    {
        let spans = core_tracing::get_stages_json();
        // Reset tracing for next execution
        core_tracing::reset_tracing();
        Some(spans)
    } else {
        None
    };

    PlatformEvent {
        session_id: config.session_id,
        event_type: event.event_type.clone(),
        payload,
        device_id: config.device_id.clone(),
        platform: config.platform.clone(),
        app_version: config.app_version.clone(),
        timestamp,
        pipeline_id,
        trace_id,
        stages,
    }
}

// ============================================================================
// Global Platform Exporter
// ============================================================================

static PLATFORM_EXPORTER: RwLock<Option<HttpTelemetryExporter>> = RwLock::new(None);

/// Initialize the global platform telemetry exporter
///
/// This also enables span tracing in xybrid-core for detailed execution profiling.
/// Spans are automatically captured and included in `PipelineComplete` events.
///
/// # Example
///
/// ```rust,no_run
/// use xybrid_sdk::telemetry::{init_platform_telemetry, TelemetryConfig};
///
/// let config = TelemetryConfig::new("https://api.xybrid.dev", "your-api-key")
///     .with_device("device-123", "ios")
///     .with_app_version("1.0.0");
///
/// init_platform_telemetry(config);
/// ```
pub fn init_platform_telemetry(config: TelemetryConfig) {
    // Enable span tracing in xybrid-core for execution profiling
    core_tracing::init_tracing(true);

    let exporter = HttpTelemetryExporter::new(config);
    exporter.start();

    // Register as a telemetry sender
    let sender = exporter.create_sender();
    register_telemetry_sender(sender);

    if let Ok(mut global) = PLATFORM_EXPORTER.write() {
        *global = Some(exporter);
    }
}

/// Initialize platform telemetry from environment variables
///
/// Returns `true` if initialization succeeded, `false` if XYBRID_API_KEY is not set.
/// Also enables span tracing in xybrid-core for detailed execution profiling.
pub fn init_platform_telemetry_from_env() -> bool {
    if let Some(exporter) = HttpTelemetryExporter::from_env() {
        // Enable span tracing in xybrid-core for execution profiling
        core_tracing::init_tracing(true);

        exporter.start();
        let sender = exporter.create_sender();
        register_telemetry_sender(sender);

        if let Ok(mut global) = PLATFORM_EXPORTER.write() {
            *global = Some(exporter);
        }
        true
    } else {
        false
    }
}

/// Set pipeline context for event enrichment
pub fn set_telemetry_pipeline_context(pipeline_id: Option<Uuid>, trace_id: Option<Uuid>) {
    if let Ok(exporter) = PLATFORM_EXPORTER.read() {
        if let Some(exp) = exporter.as_ref() {
            exp.set_pipeline_context(pipeline_id, trace_id);
        }
    }
}

/// Flush all pending telemetry events
pub fn flush_platform_telemetry() {
    if let Ok(exporter) = PLATFORM_EXPORTER.read() {
        if let Some(exp) = exporter.as_ref() {
            exp.flush();
        }
    }
}

/// Shutdown platform telemetry exporter
///
/// This also disables span tracing in xybrid-core.
pub fn shutdown_platform_telemetry() {
    // Disable span tracing
    core_tracing::init_tracing(false);

    if let Ok(mut exporter) = PLATFORM_EXPORTER.write() {
        if let Some(exp) = exporter.take() {
            exp.stop();
        }
    }
}

/// Register a telemetry event sender
pub fn register_telemetry_sender(sender: TelemetrySender) {
    let mut senders = TELEMETRY_SENDERS.lock().unwrap();
    senders.push(sender);
}

/// Convert OrchestratorEvent to TelemetryEvent
pub fn convert_orchestrator_event(event: &OrchestratorEvent) -> TelemetryEvent {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    match event {
        OrchestratorEvent::PipelineStart { stages } => TelemetryEvent {
            event_type: "PipelineStart".to_string(),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: Some(serde_json::json!({"stages": stages}).to_string()),
            timestamp_ms,
        },
        OrchestratorEvent::PipelineComplete { total_latency_ms } => TelemetryEvent {
            event_type: "PipelineComplete".to_string(),
            stage_name: None,
            target: None,
            latency_ms: Some(*total_latency_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageStart { stage_name } => TelemetryEvent {
            event_type: "StageStart".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageComplete {
            stage_name,
            target,
            latency_ms,
        } => TelemetryEvent {
            event_type: "StageComplete".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: Some(*latency_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageError { stage_name, error } => TelemetryEvent {
            event_type: "StageError".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: Some(error.clone()),
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::RoutingDecided {
            stage_name,
            target,
            reason,
        } => TelemetryEvent {
            event_type: "RoutingDecided".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: None,
            data: Some(serde_json::json!({"reason": reason}).to_string()),
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionStarted { stage_name, target } => TelemetryEvent {
            event_type: "ExecutionStarted".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionCompleted {
            stage_name,
            target,
            execution_time_ms,
        } => TelemetryEvent {
            event_type: "ExecutionCompleted".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: Some(*execution_time_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionFailed {
            stage_name,
            target,
            error,
        } => TelemetryEvent {
            event_type: "ExecutionFailed".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: Some(error.clone()),
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::PolicyEvaluated {
            stage_name,
            allowed,
            reason,
        } => TelemetryEvent {
            event_type: "PolicyEvaluated".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: if *allowed {
                None
            } else {
                reason.clone().or(Some("Policy violation".to_string()))
            },
            data: Some(
                serde_json::json!({
                    "allowed": allowed,
                    "reason": reason
                })
                .to_string(),
            ),
            timestamp_ms,
        },
        _ => TelemetryEvent {
            event_type: format!("{:?}", event),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: Some(format!("{:?}", event)),
            timestamp_ms,
        },
    }
}

/// Publish a telemetry event to all registered subscribers
pub fn publish_telemetry_event(event: TelemetryEvent) {
    let senders = TELEMETRY_SENDERS.lock().unwrap();
    let mut dead_senders = Vec::new();

    for (idx, sender) in senders.iter().enumerate() {
        if sender.send(event.clone()).is_err() {
            dead_senders.push(idx);
        }
    }

    // Remove dead senders
    drop(senders);
    if !dead_senders.is_empty() {
        let mut senders = TELEMETRY_SENDERS.lock().unwrap();
        for idx in dead_senders.iter().rev() {
            senders.remove(*idx);
        }
    }
}

/// Bridge orchestrator events to telemetry stream
///
/// This function subscribes to orchestrator events and converts them
/// to telemetry events, publishing them to all registered subscribers.
pub fn bridge_orchestrator_events(orchestrator: &xybrid_core::orchestrator::Orchestrator) {
    let event_bus = orchestrator.event_bus();
    let subscription = event_bus.subscribe();

    thread::spawn(move || {
        loop {
            match subscription.recv() {
                Ok(event) => {
                    let telemetry_event = convert_orchestrator_event(&event);
                    publish_telemetry_event(telemetry_event);
                }
                Err(_) => break, // Event bus closed
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_stage_start_event() {
        let event = OrchestratorEvent::StageStart {
            stage_name: "asr".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageStart");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert!(telemetry.target.is_none());
        assert!(telemetry.latency_ms.is_none());
        assert!(telemetry.error.is_none());
        assert!(telemetry.timestamp_ms > 0);
    }

    #[test]
    fn test_convert_stage_complete_event() {
        let event = OrchestratorEvent::StageComplete {
            stage_name: "tts".to_string(),
            target: "local".to_string(),
            latency_ms: 150,
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageComplete");
        assert_eq!(telemetry.stage_name, Some("tts".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
        assert_eq!(telemetry.latency_ms, Some(150));
        assert!(telemetry.error.is_none());
    }

    #[test]
    fn test_convert_stage_error_event() {
        let event = OrchestratorEvent::StageError {
            stage_name: "asr".to_string(),
            error: "Model not found".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageError");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.error, Some("Model not found".to_string()));
    }

    #[test]
    fn test_convert_pipeline_start_event() {
        let event = OrchestratorEvent::PipelineStart {
            stages: vec!["asr".to_string(), "llm".to_string(), "tts".to_string()],
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PipelineStart");
        assert!(telemetry.stage_name.is_none());
        assert!(telemetry.data.is_some());
        let data = telemetry.data.unwrap();
        assert!(data.contains("asr"));
        assert!(data.contains("llm"));
        assert!(data.contains("tts"));
    }

    #[test]
    fn test_convert_pipeline_complete_event() {
        let event = OrchestratorEvent::PipelineComplete {
            total_latency_ms: 500,
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PipelineComplete");
        assert_eq!(telemetry.latency_ms, Some(500));
    }

    #[test]
    fn test_convert_routing_decided_event() {
        let event = OrchestratorEvent::RoutingDecided {
            stage_name: "asr".to_string(),
            target: "cloud".to_string(),
            reason: "network_optimal".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "RoutingDecided");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("cloud".to_string()));
        assert!(telemetry.data.is_some());
        let data = telemetry.data.unwrap();
        assert!(data.contains("network_optimal"));
    }

    #[test]
    fn test_convert_execution_started_event() {
        let event = OrchestratorEvent::ExecutionStarted {
            stage_name: "asr".to_string(),
            target: "local".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionStarted");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
    }

    #[test]
    fn test_convert_execution_completed_event() {
        let event = OrchestratorEvent::ExecutionCompleted {
            stage_name: "asr".to_string(),
            target: "local".to_string(),
            execution_time_ms: 75,
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionCompleted");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
        assert_eq!(telemetry.latency_ms, Some(75));
    }

    #[test]
    fn test_convert_execution_failed_event() {
        let event = OrchestratorEvent::ExecutionFailed {
            stage_name: "tts".to_string(),
            target: "cloud".to_string(),
            error: "Timeout".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionFailed");
        assert_eq!(telemetry.stage_name, Some("tts".to_string()));
        assert_eq!(telemetry.target, Some("cloud".to_string()));
        assert_eq!(telemetry.error, Some("Timeout".to_string()));
    }

    #[test]
    fn test_convert_policy_evaluated_allowed() {
        let event = OrchestratorEvent::PolicyEvaluated {
            stage_name: "asr".to_string(),
            allowed: true,
            reason: Some("All conditions met".to_string()),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PolicyEvaluated");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert!(telemetry.error.is_none()); // No error when allowed
        assert!(telemetry.data.is_some());
    }

    #[test]
    fn test_convert_policy_evaluated_denied() {
        let event = OrchestratorEvent::PolicyEvaluated {
            stage_name: "llm".to_string(),
            allowed: false,
            reason: Some("Privacy policy violation".to_string()),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PolicyEvaluated");
        assert_eq!(telemetry.stage_name, Some("llm".to_string()));
        assert_eq!(telemetry.error, Some("Privacy policy violation".to_string()));
    }

    #[test]
    fn test_telemetry_event_serialization() {
        let event = TelemetryEvent {
            event_type: "StageStart".to_string(),
            stage_name: Some("asr".to_string()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms: 1234567890,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("StageStart"));
        assert!(json.contains("asr"));

        let deserialized: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_type, "StageStart");
        assert_eq!(deserialized.stage_name, Some("asr".to_string()));
    }

    #[test]
    fn test_register_and_publish() {
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let event = TelemetryEvent {
            event_type: "TestEvent".to_string(),
            stage_name: Some("test".to_string()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms: 0,
        };

        publish_telemetry_event(event.clone());

        // Should receive the event
        let received = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(received.is_ok());
        let received_event = received.unwrap();
        assert_eq!(received_event.event_type, "TestEvent");
    }

    #[test]
    fn test_telemetry_config_defaults() {
        let config = TelemetryConfig::default();
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.flush_interval_secs, 5);
        assert_eq!(config.max_retries, 3);
        assert!(config.enable_retry_queue);
    }

    #[test]
    fn test_http_exporter_circuit_breaker_initial_state() {
        let config = TelemetryConfig::new("https://example.com", "test-key");
        let exporter = HttpTelemetryExporter::new(config);
        assert!(!exporter.is_circuit_open());
    }

    #[test]
    fn test_http_exporter_circuit_breaker_reset() {
        let config = TelemetryConfig::new("https://example.com", "test-key");
        let exporter = HttpTelemetryExporter::new(config);

        // Manually trigger failures to open the circuit
        for _ in 0..3 {
            exporter.circuit.record_failure();
        }
        assert!(exporter.is_circuit_open());

        // Reset should close it
        exporter.reset_circuit();
        assert!(!exporter.is_circuit_open());
    }

    #[test]
    fn test_http_exporter_failed_queue_initial_empty() {
        let config = TelemetryConfig::new("https://example.com", "test-key");
        let exporter = HttpTelemetryExporter::new(config);
        assert_eq!(exporter.failed_queue_size(), 0);
        assert_eq!(exporter.dropped_count(), 0);
    }

    #[test]
    fn test_queue_failed_events() {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let dropped = Arc::new(AtomicU32::new(0));

        let events = vec![
            PlatformEvent {
                session_id: Uuid::new_v4(),
                event_type: "Test".to_string(),
                payload: serde_json::json!({}),
                device_id: None,
                platform: None,
                app_version: None,
                timestamp: None,
                pipeline_id: None,
                trace_id: None,
                stages: None,
            },
        ];

        queue_failed_events(events, &queue, &dropped);
        assert_eq!(queue.lock().unwrap().len(), 1);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_is_retryable_status() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(502));
        assert!(is_retryable_status(503));
        assert!(is_retryable_status(504));
        assert!(!is_retryable_status(200));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(404));
    }
}

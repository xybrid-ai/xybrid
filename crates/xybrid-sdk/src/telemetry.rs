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
pub use xybrid_core::device::DeviceProfile;
use xybrid_core::event_bus::OrchestratorEvent;
use xybrid_core::execution::listener::{self as execution_listener, ExecutionEvent};
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

/// Default telemetry ingest URL
pub const DEFAULT_INGEST_URL: &str = "https://ingest.xybrid.dev";

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

    /// Human-friendly device label (e.g. "Sami's MacBook Pro"). Shown in the
    /// dashboard alongside the stable `device_id`.
    pub device_label: Option<String>,
    /// Full `DeviceProfile` override. When `Some`, fields win over both the
    /// auto-detected profile and any partial patch.
    pub device_profile_override: Option<DeviceProfile>,
    /// Partial hardware overrides. Merged onto the auto-detected profile.
    pub device_profile_patch: DeviceProfile,
    /// When `true` (default), the exporter probes local hardware at init and
    /// populates the `device` substructure on every event.
    pub auto_hardware_detection: bool,
    /// When `true`, the exporter includes the machine's hostname in the
    /// `device` substructure. Off by default because hostnames are PII.
    pub capture_hostname: bool,
    /// Internal: set to `true` only when the caller supplied `device_id`
    /// explicitly via `with_device(...)`. Distinguishes caller-supplied
    /// identifiers from the auto-wired default so the opt-out path can
    /// suppress the latter without dropping the former.
    #[doc(hidden)]
    pub device_id_explicit: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        let device = crate::device::Device::current();
        Self {
            endpoint: String::new(),
            api_key: String::new(),
            session_id: Uuid::new_v4(),
            device_id: Some(device.id.clone()),
            platform: Some(device.platform.clone()),
            app_version: None,
            batch_size: 10,
            flush_interval_secs: 5,
            max_retries: 3,
            enable_retry_queue: true,
            device_label: None,
            device_profile_override: None,
            device_profile_patch: DeviceProfile::default(),
            auto_hardware_detection: true,
            capture_hostname: false,
            device_id_explicit: false,
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

    /// Set device metadata.
    ///
    /// Sets both the `device_id` (stable identifier) and `platform` (OS family
    /// string). Independent of the hardware profile; auto-detection keeps
    /// running unless you opt out via `with_auto_hardware_detection(false)`.
    pub fn with_device(
        mut self,
        device_id: impl Into<String>,
        platform: impl Into<String>,
    ) -> Self {
        self.device_id = Some(device_id.into());
        self.platform = Some(platform.into());
        self.device_id_explicit = true;
        self
    }

    /// Override only the platform string (device ID remains auto-detected).
    pub fn with_platform(mut self, platform: impl Into<String>) -> Self {
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

    /// Set a human-friendly label for this device. Example: `"Sami's MacBook Pro"`.
    ///
    /// The label is shown in the console alongside the stable `device_id`.
    pub fn with_device_label(mut self, label: impl Into<String>) -> Self {
        self.device_label = Some(label.into());
        self
    }

    /// Supply the complete `DeviceProfile` emitted on the wire. Disables
    /// automatic hardware detection so any field left as `None` stays `None`
    /// — callers wanting to opt out of leaking OS / chip / RAM just omit
    /// those fields. For partial overlays onto auto-detected values, use the
    /// `with_hardware_*` field-specific builders instead.
    pub fn with_hardware(mut self, profile: DeviceProfile) -> Self {
        self.device_profile_override = Some(profile);
        self.auto_hardware_detection = false;
        self
    }

    /// Override the detected chip family / CPU brand.
    pub fn with_hardware_chip(mut self, chip: impl Into<String>) -> Self {
        self.device_profile_patch.chip_family = Some(chip.into());
        self
    }

    /// Override the detected RAM (gigabytes).
    pub fn with_hardware_ram_gb(mut self, gb: u32) -> Self {
        self.device_profile_patch.ram_gb = Some(gb);
        self
    }

    /// Override the detected OS name and version.
    pub fn with_hardware_os(mut self, os: impl Into<String>, version: impl Into<String>) -> Self {
        self.device_profile_patch.os = Some(os.into());
        self.device_profile_patch.os_version = Some(version.into());
        self
    }

    /// Override the detected CPU architecture (e.g. `"arm64"`, `"x86_64"`).
    pub fn with_hardware_arch(mut self, arch: impl Into<String>) -> Self {
        self.device_profile_patch.arch = Some(arch.into());
        self
    }

    /// Add an arbitrary app-provided attribute, stored under `device.custom`
    /// on the wire event.
    pub fn with_device_attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.device_profile_patch
            .custom
            .insert(key.into(), value.into());
        self
    }

    /// Opt out of all hardware auto-detection. When `false`, the `device`
    /// substructure only contains fields the app supplies explicitly.
    pub fn with_auto_hardware_detection(mut self, enabled: bool) -> Self {
        self.auto_hardware_detection = enabled;
        self
    }

    /// Opt into hostname capture. Off by default because hostnames like
    /// `Samis-MacBook` are effectively PII and make the payload identify
    /// a person rather than a piece of hardware.
    pub fn with_hostname_capture(mut self, enabled: bool) -> Self {
        self.capture_hostname = enabled;
        self
    }
}

/// Event payload for platform API (matches IngestTelemetryEvent)
#[derive(Debug, Clone, Serialize)]
struct PlatformEvent {
    session_id: Uuid,
    event_type: String,
    payload: serde_json::Value,
    // `device_id` honors the opt-out contract: when the SDK clears it
    // because the caller opted out of hardware detection without supplying
    // an explicit id, the wire event omits the field entirely rather than
    // emitting `"device_id": null`. Some ingest schemas treat absent vs
    // null-but-present differently, so this matters.
    #[serde(skip_serializing_if = "Option::is_none")]
    device_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_label: Option<String>,
    platform: Option<String>,
    app_version: Option<String>,
    /// Hardware + OS snapshot. `None` when the app has opted out of
    /// auto-detection and supplied no explicit overrides.
    #[serde(skip_serializing_if = "Option::is_none")]
    device: Option<DeviceProfile>,
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
    device_profile: Option<DeviceProfile>,
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
    /// Create a new HTTP exporter with the given configuration.
    pub fn new(mut config: TelemetryConfig) -> Self {
        let device_profile = resolve_device_profile(&config);
        // Privacy opt-out contract: when the caller disabled hardware
        // auto-detection and did not explicitly supply an identifier via
        // `with_device(...)`, suppress the `device_id` that
        // `TelemetryConfig::default()` auto-wired from `Device::current()`.
        // Explicit non-hardware context (labels, attributes, hostname capture)
        // no longer re-enables the default identifier — the caller must opt
        // back in via `with_device(...)`. `platform` is kept because it's an
        // OS family string, not PII.
        if !config.auto_hardware_detection && !config.device_id_explicit {
            config.device_id = None;
        }

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
            device_profile,
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
    /// - `XYBRID_INGEST_URL` - Ingest endpoint (default: https://ingest.xybrid.dev)
    /// - `XYBRID_PLATFORM_URL` - Legacy fallback (deprecated)
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("XYBRID_API_KEY").ok()?;
        // Try new env var first, then legacy, then default
        let endpoint = std::env::var("XYBRID_INGEST_URL")
            .or_else(|_| std::env::var("XYBRID_PLATFORM_URL"))
            .unwrap_or_else(|_| DEFAULT_INGEST_URL.to_string());

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
        let device_profile = self.device_profile.clone();
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
                    retry_failed_events(&failed_queue, &config, &agent, &circuit, &retry_policy);
                }

                // Then flush the current buffer
                flush_buffer_with_retry(
                    &buffer,
                    &config,
                    device_profile.as_ref(),
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
            self.device_profile.as_ref(),
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
                self.device_profile.as_ref(),
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
            self.device_profile.as_ref(),
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
        let device_profile = self.device_profile.clone();
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
                        device_profile.as_ref(),
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
    device_profile: Option<&DeviceProfile>,
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
            device_profile,
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
    device_profile: Option<&DeviceProfile>,
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
                .map(|e| convert_to_platform_event(e, config, device_profile, pid, tid))
                .collect();
            queue_failed_events(platform_events, failed_queue, dropped_count);
        }
        return;
    }

    let pid = pipeline_id.read().ok().and_then(|g| *g);
    let tid = trace_id.read().ok().and_then(|g| *g);

    let platform_events: Vec<PlatformEvent> = events
        .iter()
        .map(|e| convert_to_platform_event(e, config, device_profile, pid, tid))
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

    let url = format!("{}/v1/events/batch", config.endpoint.trim_end_matches('/'));

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
    device_profile: Option<&DeviceProfile>,
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
        // Hoist LLM token counts from the span metadata into the outer
        // payload so analytics backends that read `tokens_in` /
        // `tokens_out` at the top of the event see them without having
        // to descend into the span tree. No-op when no `llm_inference*`
        // span is present.
        if let Some((tokens_in, tokens_out)) = extract_llm_token_counts(&spans) {
            if let Some(n) = tokens_in {
                payload["tokens_in"] = serde_json::json!(n);
            }
            if let Some(n) = tokens_out {
                payload["tokens_out"] = serde_json::json!(n);
            }
        }
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
        device_label: config.device_label.clone(),
        platform: config.platform.clone(),
        app_version: config.app_version.clone(),
        device: device_profile.cloned(),
        timestamp,
        pipeline_id,
        trace_id,
        stages,
    }
}

/// Resolve the effective device profile for a config: auto-detect (if on),
/// then apply the per-field patch, then apply the full override. Hostname is
/// only filled when `config.capture_hostname` is true.
fn resolve_device_profile(config: &TelemetryConfig) -> Option<DeviceProfile> {
    let mut profile = if config.auto_hardware_detection {
        DeviceProfile::detect()
    } else {
        DeviceProfile::default()
    };
    profile = profile.merged_with(config.device_profile_patch.clone());
    if let Some(override_) = config.device_profile_override.clone() {
        profile = profile.merged_with(override_);
    }
    if config.capture_hostname && profile.hostname.is_none() {
        profile.hostname = detect_hostname();
    }
    if profile.is_empty() {
        None
    } else {
        Some(profile)
    }
}

fn detect_hostname() -> Option<String> {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .ok()
        .map(|hostname| hostname.trim().to_string())
        .filter(|hostname| !hostname.is_empty())
}

/// Walk a `stages` JSON (shape: `{"spans":[{"name","metadata":{...}}]}`) and
/// return `(tokens_in, tokens_out)` read from the first span that either
/// carries LLM-style metadata (`ttft_ms`, `tokens_generated`, `tokens_out`,
/// `completion_tokens`) or is named with a known LLM prefix
/// (`llm_inference*` or `inference:*`).
///
/// Looks at both canonical keys (`tokens_in` / `tokens_out`) and OpenAI-ish
/// keys (`prompt_tokens` / `completion_tokens` + `tokens_generated`) so the
/// hoist works regardless of which LLM adapter emitted the span or how the
/// enclosing orchestrator named it.
fn extract_llm_token_counts(stages: &serde_json::Value) -> Option<(Option<u64>, Option<u64>)> {
    let spans = stages.get("spans")?.as_array()?;
    let read = |meta: Option<&serde_json::Value>, keys: &[&str]| -> Option<u64> {
        for k in keys {
            let Some(v) = meta.and_then(|m| m.get(*k)) else {
                continue;
            };
            if let Some(n) = v.as_u64() {
                return Some(n);
            }
            if let Some(s) = v.as_str() {
                if let Ok(n) = s.parse::<u64>() {
                    return Some(n);
                }
            }
        }
        None
    };
    // Token accounting invariant: the authoritative counts live on the LAST
    // LLM span in the trace. A streaming run emits a timing-only span first
    // (ttft_ms, no counts yet) and a final accounting span with the totals;
    // a retried run emits one span per attempt and we want the final attempt.
    // Earlier-span values only win as a fallback when no later span carries
    // the corresponding key at all.
    let mut saw_llm_span = false;
    let mut tokens_in: Option<u64> = None;
    let mut tokens_out: Option<u64> = None;
    for span in spans {
        let name = span.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let meta = span.get("metadata");
        let is_llm_span = name.starts_with("llm_inference")
            || name.starts_with("inference:")
            || meta
                .map(|m| {
                    m.get("ttft_ms").is_some()
                        || m.get("tokens_generated").is_some()
                        || m.get("tokens_out").is_some()
                        || m.get("completion_tokens").is_some()
                })
                .unwrap_or(false);
        if !is_llm_span {
            continue;
        }
        saw_llm_span = true;
        if let Some(v) = read(meta, &["tokens_in", "prompt_tokens"]) {
            tokens_in = Some(v);
        }
        if let Some(v) = read(
            meta,
            &["tokens_out", "completion_tokens", "tokens_generated"],
        ) {
            tokens_out = Some(v);
        }
    }
    if saw_llm_span {
        Some((tokens_in, tokens_out))
    } else {
        None
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
/// ```rust,ignore
/// use xybrid_sdk::telemetry::{init_platform_telemetry, TelemetryConfig};
///
/// let config = TelemetryConfig::new("https://ingest.xybrid.dev", "your-api-key")
///     .with_device("device-123", "ios")
///     .with_app_version("1.0.0");
///
/// init_platform_telemetry(config);
/// ```
pub fn init_platform_telemetry(config: TelemetryConfig) {
    // Enable span tracing in xybrid-core for execution profiling
    core_tracing::init_tracing(true);

    // Register automatic execution listener so TemplateExecutor emits
    // ExecutionStarted / ExecutionCompleted / ExecutionFailed events
    register_execution_listener();

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

        // Register automatic execution listener
        register_execution_listener();

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

/// Register the execution listener that converts `ExecutionEvent`s from
/// xybrid-core's `TemplateExecutor` into `TelemetryEvent`s and publishes them.
fn register_execution_listener() {
    execution_listener::set_execution_listener(|event| {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let telemetry_event = match event {
            ExecutionEvent::Started { model_id, method } => TelemetryEvent {
                event_type: "ExecutionStarted".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: None,
                error: None,
                data: Some(format!(r#"{{"model":"{}"}}"#, model_id)),
                timestamp_ms,
            },
            ExecutionEvent::Completed {
                model_id,
                method,
                latency_ms,
            } => TelemetryEvent {
                event_type: "ExecutionCompleted".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: Some(latency_ms as u32),
                error: None,
                data: Some(format!(r#"{{"model":"{}"}}"#, model_id)),
                timestamp_ms,
            },
            ExecutionEvent::Failed {
                model_id,
                method,
                latency_ms,
                error,
            } => TelemetryEvent {
                event_type: "ExecutionFailed".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: Some(latency_ms as u32),
                error: Some(error),
                data: Some(format!(r#"{{"model":"{}"}}"#, model_id)),
                timestamp_ms,
            },
        };

        publish_telemetry_event(telemetry_event);
    });
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

    // Remove automatic execution listener
    execution_listener::clear_execution_listener();

    if let Ok(mut exporter) = PLATFORM_EXPORTER.write() {
        if let Some(exp) = exporter.take() {
            exp.stop();
        }
    }
}

/// Register a telemetry event sender
pub fn register_telemetry_sender(sender: TelemetrySender) {
    // Use if-let to gracefully handle poisoned mutex
    if let Ok(mut senders) = TELEMETRY_SENDERS.lock() {
        senders.push(sender);
    }
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
    // Use unwrap_or_else to recover from poisoned mutex - this prevents
    // a panic in one component from permanently breaking telemetry
    let Ok(senders) = TELEMETRY_SENDERS.lock() else {
        // Mutex is poisoned, silently skip telemetry rather than crash
        return;
    };
    let mut dead_senders = Vec::new();

    for (idx, sender) in senders.iter().enumerate() {
        if sender.send(event.clone()).is_err() {
            dead_senders.push(idx);
        }
    }

    // Remove dead senders
    drop(senders);
    if !dead_senders.is_empty() {
        if let Ok(mut senders) = TELEMETRY_SENDERS.lock() {
            for idx in dead_senders.iter().rev() {
                senders.remove(*idx);
            }
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
        assert_eq!(
            telemetry.error,
            Some("Privacy policy violation".to_string())
        );
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
        assert!(config.auto_hardware_detection);
        assert!(!config.capture_hostname);
    }

    #[test]
    fn test_http_exporter_circuit_breaker_initial_state() {
        let config = TelemetryConfig::new("https://example.com", "test-key")
            .with_device("test-device", "test-platform");
        let exporter = HttpTelemetryExporter::new(config);
        assert!(!exporter.is_circuit_open());
    }

    #[test]
    fn test_http_exporter_circuit_breaker_reset() {
        let config = TelemetryConfig::new("https://example.com", "test-key")
            .with_device("test-device", "test-platform");
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
        let config = TelemetryConfig::new("https://example.com", "test-key")
            .with_device("test-device", "test-platform");
        let exporter = HttpTelemetryExporter::new(config);
        assert_eq!(exporter.failed_queue_size(), 0);
        assert_eq!(exporter.dropped_count(), 0);
    }

    #[test]
    fn test_queue_failed_events() {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let dropped = Arc::new(AtomicU32::new(0));

        let events = vec![PlatformEvent {
            session_id: Uuid::new_v4(),
            event_type: "Test".to_string(),
            payload: serde_json::json!({}),
            device_id: None,
            device_label: None,
            platform: None,
            app_version: None,
            device: None,
            timestamp: None,
            pipeline_id: None,
            trace_id: None,
            stages: None,
        }];

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

    // ------------------------------------------------------------------
    // Device telemetry — opt-out contract + random-UUID device_id
    //
    // These tests cover the adversarial-review findings: auto-detection
    // opt-out must omit both the `device` object and the auto-generated
    // `device_id`; the default device_id must not be a hardware fingerprint.
    // ------------------------------------------------------------------

    fn sample_event() -> TelemetryEvent {
        TelemetryEvent {
            event_type: "TestEvent".to_string(),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms: 0,
        }
    }

    #[test]
    fn opt_out_emits_no_device_and_no_auto_id() {
        // Start with a default config (which auto-wires `device_id` from
        // `Device::current()`), then opt out of hardware detection without
        // providing a label or overrides. This mirrors the `HttpTelemetryExporter::new`
        // merge path where the strict opt-out contract clears the
        // default-injected id.
        let mut config =
            TelemetryConfig::new("http://example.invalid", "k").with_auto_hardware_detection(false);
        let profile = resolve_device_profile(&config);
        assert!(profile.is_none(), "strict opt-out must yield None profile");

        // Replicate the exporter's strict-opt-out clearing so the test covers
        // the full emission path, not just `convert_to_platform_event`.
        if !config.auto_hardware_detection && !config.device_id_explicit {
            config.device_id = None;
        }

        let event =
            convert_to_platform_event(&sample_event(), &config, profile.as_ref(), None, None);
        let json = serde_json::to_value(&event).unwrap();
        assert!(
            json.get("device").is_none(),
            "no `device` key on strict opt-out, got: {json}"
        );
        assert!(
            json.get("device_id").is_none(),
            "strict opt-out must omit `device_id` entirely, not emit null. got: {json}"
        );
    }

    #[test]
    fn extract_llm_token_counts_reads_canonical_keys() {
        let stages = serde_json::json!({
            "spans": [
                { "name": "execute:preprocessing", "metadata": {} },
                {
                    "name": "llm_inference_with_messages",
                    "metadata": {
                        "tokens_in": "128",
                        "tokens_out": "42",
                        "tokens_generated": "42"
                    }
                }
            ]
        });
        let (tin, tout) = extract_llm_token_counts(&stages).expect("should find llm span");
        assert_eq!(tin, Some(128));
        assert_eq!(tout, Some(42));
    }

    #[test]
    fn extract_llm_token_counts_falls_back_to_openai_style_keys() {
        let stages = serde_json::json!({
            "spans": [{
                "name": "llm_inference_streaming",
                "metadata": {
                    "prompt_tokens": "16",
                    "completion_tokens": "64"
                }
            }]
        });
        let (tin, tout) = extract_llm_token_counts(&stages).expect("should find llm span");
        assert_eq!(tin, Some(16));
        assert_eq!(tout, Some(64));
    }

    #[test]
    fn extract_llm_token_counts_returns_none_without_llm_span() {
        let stages = serde_json::json!({
            "spans": [
                { "name": "execute:asr.whisper-tiny", "metadata": {} }
            ]
        });
        assert!(extract_llm_token_counts(&stages).is_none());
    }

    #[test]
    fn extract_llm_token_counts_scans_across_llm_spans() {
        let stages = serde_json::json!({
            "spans": [
                {
                    "name": "llm_inference_streaming",
                    "metadata": { "ttft_ms": 120 }
                },
                {
                    "name": "inference:qwen2.5-0.5b",
                    "metadata": { "tokens_in": 32, "tokens_out": 96 }
                }
            ]
        });
        let (tin, tout) = extract_llm_token_counts(&stages).expect("should find llm spans");
        assert_eq!(tin, Some(32));
        assert_eq!(tout, Some(96));
    }

    #[test]
    fn extract_llm_token_counts_prefers_last_authoritative_span() {
        // Retry-like trace: three spans, each with progressively larger
        // token counts. The authoritative totals live on the final attempt.
        // We must NOT keep the first-seen values from span #1.
        let stages = serde_json::json!({
            "spans": [
                {
                    "name": "llm_inference_streaming",
                    "metadata": { "tokens_in": 10, "tokens_out": 5 }
                },
                {
                    "name": "llm_inference_streaming",
                    "metadata": { "ttft_ms": 50 }
                },
                {
                    "name": "llm_inference_streaming",
                    "metadata": { "tokens_in": 128, "tokens_out": 42 }
                }
            ]
        });
        let (tin, tout) = extract_llm_token_counts(&stages).expect("should find llm spans");
        assert_eq!(tin, Some(128), "must take last-span tokens_in, not first");
        assert_eq!(tout, Some(42), "must take last-span tokens_out, not first");
    }

    #[test]
    fn extract_llm_token_counts_falls_back_across_partial_spans() {
        // Streaming-with-partial: the final span has no tokens_in but an
        // earlier span did. The fallback must carry the earlier value
        // through rather than drop it, since nothing later overrides it.
        let stages = serde_json::json!({
            "spans": [
                {
                    "name": "llm_inference_streaming",
                    "metadata": { "tokens_in": 64 }
                },
                {
                    "name": "llm_inference_streaming",
                    "metadata": { "tokens_out": 256 }
                }
            ]
        });
        let (tin, tout) = extract_llm_token_counts(&stages).expect("should find llm spans");
        assert_eq!(tin, Some(64));
        assert_eq!(tout, Some(256));
    }

    #[test]
    fn opt_out_with_explicit_attribute_suppresses_device_id() {
        // Privacy contract: opting out of hardware detection must suppress
        // the auto-wired `device_id` even when explicit non-hardware context
        // (labels, attributes, hostname) puts the profile back in "has
        // context" mode. Callers who want a stable id must opt back in via
        // `with_device(...)`.
        let config = TelemetryConfig::new("http://example.invalid", "k")
            .with_auto_hardware_detection(false)
            .with_device_attribute("tailnet", "production");
        let exporter = HttpTelemetryExporter::new(config);
        assert!(
            exporter.config.device_id.is_none(),
            "opt-out + explicit attribute must suppress auto-wired device_id, got: {:?}",
            exporter.config.device_id
        );
    }

    #[test]
    fn opt_out_with_explicit_with_device_preserves_id() {
        // Inverse of the suppression test: when the caller explicitly opts
        // back in via `with_device(...)`, the identifier must survive the
        // opt-out clear.
        let config = TelemetryConfig::new("http://example.invalid", "k")
            .with_auto_hardware_detection(false)
            .with_device("caller-supplied-id", "linux");
        let exporter = HttpTelemetryExporter::new(config);
        assert_eq!(
            exporter.config.device_id.as_deref(),
            Some("caller-supplied-id"),
            "explicit with_device must survive opt-out clear"
        );
    }

    #[test]
    fn with_hardware_disables_auto_detection() {
        let profile = DeviceProfile {
            chip_family: Some("supplied-chip".into()),
            ram_gb: Some(16),
            ..Default::default()
        };
        let config = TelemetryConfig::new("http://example.invalid", "k").with_hardware(profile);
        assert!(
            !config.auto_hardware_detection,
            "with_hardware must disable auto-detection"
        );
        let resolved = resolve_device_profile(&config).expect("profile present");
        assert_eq!(resolved.chip_family.as_deref(), Some("supplied-chip"));
        assert_eq!(resolved.ram_gb, Some(16));
        assert!(resolved.os.is_none(), "os must stay None (opt-out honored)");
        assert!(
            resolved.arch.is_none(),
            "arch must stay None (opt-out honored)"
        );
    }

    #[test]
    fn opt_out_with_explicit_attribute_still_emits_device() {
        let config = TelemetryConfig::new("http://example.invalid", "k")
            .with_auto_hardware_detection(false)
            .with_device_attribute("tailnet", "production");
        let profile = resolve_device_profile(&config);
        let profile = profile.expect("explicit attribute must surface a profile");
        assert!(profile.chip_family.is_none());
        assert!(profile.ram_gb.is_none());
        assert_eq!(
            profile.custom.get("tailnet").map(String::as_str),
            Some("production")
        );
    }
}

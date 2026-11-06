//! Execution stage tracing.
//!
//! Execution code marks stages with [`SpanGuard`] and annotates them with
//! [`add_metadata`]. Those calls go to the [`TraceRecorder`] installed on the
//! current thread for the duration of one run — never to a process-wide
//! collector — so concurrent runs cannot see or reset each other's spans.
//! With no recorder installed every call is a cheap no-op.
//!
//! The SDK installs a recorder that turns stages into OpenTelemetry child
//! spans of the run's logical request. Tests and local tooling use
//! [`CollectingRecorder`] / [`capture`] to inspect the stage tree.
//!
//! ```
//! use xybrid_core::tracing::{add_metadata, capture, SpanGuard};
//!
//! let trace = capture();
//! {
//!     let _stage = SpanGuard::new("inference");
//!     add_metadata("model", "whisper");
//! }
//! assert_eq!(trace.stages_json()["spans"][0]["name"], "inference");
//! ```
//!
//! Work that hops threads inside a run carries the recorder explicitly with
//! [`current_recorder`] and [`RecorderScope::install`].

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A span representing a unit of work with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Name of the span (e.g., "preprocessing:AudioDecode")
    pub name: String,
    /// Duration in milliseconds (set when span ends)
    pub duration_ms: Option<u64>,
    /// Parent span ID (None for root spans)
    pub parent_id: Option<usize>,
    /// Additional metadata (e.g., model_id, step_type)
    pub metadata: HashMap<String, String>,
    /// Span ID (index in collector)
    #[serde(skip)]
    pub id: usize,
    /// Start time (not serialized - internal use)
    #[serde(skip)]
    start: Option<Instant>,
}

impl Span {
    fn new(name: String, parent_id: Option<usize>, id: usize) -> Self {
        Self {
            name,
            duration_ms: None,
            parent_id,
            metadata: HashMap::new(),
            id,
            start: Some(Instant::now()),
        }
    }

    fn end(&mut self) {
        if let Some(start) = self.start.take() {
            self.duration_ms = Some(start.elapsed().as_millis() as u64);
        }
    }
}

/// An in-memory stage tree for one trace.
#[derive(Debug, Default)]
pub struct SpanCollector {
    spans: Vec<Span>,
    stack: Vec<usize>,
    enabled: bool,
}

impl SpanCollector {
    /// Create a new enabled SpanCollector
    pub fn new() -> Self {
        Self {
            spans: Vec::new(),
            stack: Vec::new(),
            enabled: true,
        }
    }

    /// Create a new SpanCollector with enabled/disabled state
    pub fn with_enabled(enabled: bool) -> Self {
        Self {
            spans: Vec::new(),
            stack: Vec::new(),
            enabled,
        }
    }

    /// Check if tracing is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable tracing
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Start a new span under the innermost open span, returns span ID
    pub fn start_span(&mut self, name: impl Into<String>) -> usize {
        let parent_id = self.stack.last().copied();
        self.start_span_with_parent(name, parent_id)
    }

    /// Start a new span under an explicit parent, returns span ID
    pub fn start_span_with_parent(
        &mut self,
        name: impl Into<String>,
        parent_id: Option<usize>,
    ) -> usize {
        if !self.enabled {
            return 0;
        }
        let span_id = self.spans.len();
        self.spans.push(Span::new(name.into(), parent_id, span_id));
        self.stack.push(span_id);
        span_id
    }

    /// End the current span (top of stack)
    pub fn end_span(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(span_id) = self.stack.pop() {
            if let Some(span) = self.spans.get_mut(span_id) {
                span.end();
            }
        }
    }

    /// End a specific span by ID
    pub fn end_span_by_id(&mut self, span_id: usize) {
        if !self.enabled {
            return;
        }
        if let Some(span) = self.spans.get_mut(span_id) {
            span.end();
        }
        self.stack.retain(|&id| id != span_id);
    }

    /// Add metadata to the current span
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        if let Some(&span_id) = self.stack.last() {
            self.add_metadata_to_span(span_id, key, value);
        }
    }

    /// Add metadata to a specific span
    pub fn add_metadata_to_span(
        &mut self,
        span_id: usize,
        key: impl Into<String>,
        value: impl Into<String>,
    ) {
        if !self.enabled {
            return;
        }
        if let Some(span) = self.spans.get_mut(span_id) {
            span.metadata.insert(key.into(), value.into());
        }
    }

    /// Get all collected spans
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Get spans as a serializable structure
    pub fn to_stages_json(&self) -> serde_json::Value {
        let spans: Vec<serde_json::Value> = self
            .spans
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "parent_id": s.parent_id,
                    "metadata": s.metadata,
                })
            })
            .collect();

        serde_json::json!({ "spans": spans })
    }

    /// Get total duration (sum of root spans)
    pub fn total_duration_ms(&self) -> u64 {
        self.spans
            .iter()
            .filter(|s| s.parent_id.is_none())
            .filter_map(|s| s.duration_ms)
            .sum()
    }

    /// Clear the collector
    pub fn reset(&mut self) {
        self.spans.clear();
        self.stack.clear();
    }

    /// Check if any spans have been collected
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// Receives the stages of one run.
///
/// Span ids are recorder-defined; `parent` is the id of the innermost stage
/// open on the calling thread when the span started (`None` for the run's
/// first stage). Implementations must be cheap: they run on the inference
/// thread.
pub trait TraceRecorder: Send + Sync {
    /// Starts a stage and returns its id.
    fn start_span(&self, name: &str, parent: Option<u64>) -> u64;
    /// Ends a stage.
    fn end_span(&self, id: u64);
    /// Annotates a stage.
    fn add_metadata(&self, id: u64, key: &str, value: &str);
    /// W3C trace headers (`traceparent`, optional `tracestate`) for outbound
    /// requests made inside stage `id` (or the run itself when `None`).
    /// Recorders without a trace return `None`.
    fn trace_headers(&self, _id: Option<u64>) -> Option<TraceHeaders> {
        None
    }
}

/// W3C Trace Context headers for an outbound request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceHeaders {
    /// `traceparent` header value.
    pub traceparent: String,
    /// `tracestate` header value, if any.
    pub tracestate: Option<String>,
}

struct ThreadTrace {
    recorder: Arc<dyn TraceRecorder>,
    stack: Vec<u64>,
}

thread_local! {
    static CURRENT: RefCell<Vec<ThreadTrace>> = const { RefCell::new(Vec::new()) };
}

/// Installs a recorder on the current thread until dropped; restores the
/// previous one (scopes nest).
#[must_use = "the recorder is uninstalled when the scope drops"]
pub struct RecorderScope {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl RecorderScope {
    /// Installs `recorder` on this thread.
    pub fn install(recorder: Arc<dyn TraceRecorder>) -> Self {
        CURRENT.with(|c| {
            c.borrow_mut().push(ThreadTrace {
                recorder,
                stack: Vec::new(),
            })
        });
        Self {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for RecorderScope {
    fn drop(&mut self) {
        CURRENT.with(|c| {
            c.borrow_mut().pop();
        });
    }
}

impl fmt::Debug for RecorderScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RecorderScope")
    }
}

/// The recorder installed on this thread, to carry into another thread.
pub fn current_recorder() -> Option<Arc<dyn TraceRecorder>> {
    CURRENT.with(|c| c.borrow().last().map(|t| t.recorder.clone()))
}

/// Whether a recorder is installed on this thread.
pub fn is_tracing_enabled() -> bool {
    CURRENT.with(|c| !c.borrow().is_empty())
}

/// Start a stage on this thread's recorder. Returns its id (0 when no
/// recorder is installed).
pub fn start_span(name: impl Into<String>) -> u64 {
    CURRENT.with(|c| {
        let mut c = c.borrow_mut();
        let Some(trace) = c.last_mut() else {
            return 0;
        };
        let name = name.into();
        let id = trace
            .recorder
            .start_span(&name, trace.stack.last().copied());
        trace.stack.push(id);
        id
    })
}

/// End the innermost stage on this thread.
pub fn end_span() {
    CURRENT.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(trace) = c.last_mut() {
            if let Some(id) = trace.stack.pop() {
                trace.recorder.end_span(id);
            }
        }
    });
}

/// End a specific stage on this thread.
pub fn end_span_by_id(id: u64) {
    CURRENT.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(trace) = c.last_mut() {
            if trace.stack.contains(&id) {
                trace.stack.retain(|s| *s != id);
                trace.recorder.end_span(id);
            }
        }
    });
}

/// Annotate the innermost open stage on this thread.
pub fn add_metadata(key: impl Into<String>, value: impl Into<String>) {
    CURRENT.with(|c| {
        let c = c.borrow();
        if let Some(trace) = c.last() {
            if let Some(&id) = trace.stack.last() {
                trace.recorder.add_metadata(id, &key.into(), &value.into());
            }
        }
    });
}

/// W3C trace headers for an outbound request made from the innermost open
/// stage on this thread, so remote spans join the run's trace.
pub fn current_trace_headers() -> Option<TraceHeaders> {
    CURRENT.with(|c| {
        let c = c.borrow();
        let trace = c.last()?;
        trace.recorder.trace_headers(trace.stack.last().copied())
    })
}

/// The `traceparent` part of [`current_trace_headers`].
pub fn current_traceparent() -> Option<String> {
    current_trace_headers().map(|h| h.traceparent)
}

/// Adds `traceparent`/`tracestate` to an outbound request when the current
/// run is traced.
pub fn inject_trace_headers(request: ureq::Request) -> ureq::Request {
    match current_trace_headers() {
        Some(headers) => {
            let request = request.set("traceparent", &headers.traceparent);
            match headers.tracestate {
                Some(state) => request.set("tracestate", &state),
                None => request,
            }
        }
        None => request,
    }
}

/// RAII guard for a stage.
pub struct SpanGuard {
    id: u64,
    active: bool,
}

impl SpanGuard {
    /// Starts a stage on this thread's recorder.
    pub fn new(name: impl Into<String>) -> Self {
        let active = is_tracing_enabled();
        let id = if active { start_span(name) } else { 0 };
        Self { id, active }
    }

    /// The stage id.
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if self.active {
            end_span_by_id(self.id);
        }
    }
}

impl fmt::Debug for SpanGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanGuard").field("id", &self.id).finish()
    }
}

/// A recorder that keeps the stage tree in memory (tests, local tooling).
#[derive(Debug, Default)]
pub struct CollectingRecorder {
    collector: Mutex<SpanCollector>,
}

impl CollectingRecorder {
    /// A new, empty recorder.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            collector: Mutex::new(SpanCollector::new()),
        })
    }

    /// The stages recorded so far, as `{"spans": [...]}`.
    pub fn stages_json(&self) -> serde_json::Value {
        self.collector
            .lock()
            .map(|c| c.to_stages_json())
            .unwrap_or_else(|_| serde_json::json!({ "spans": [] }))
    }

    /// A copy of the recorded spans.
    pub fn spans(&self) -> Vec<Span> {
        self.collector
            .lock()
            .map(|c| c.spans().to_vec())
            .unwrap_or_default()
    }
}

impl TraceRecorder for CollectingRecorder {
    fn start_span(&self, name: &str, parent: Option<u64>) -> u64 {
        self.collector
            .lock()
            .map(|mut c| c.start_span_with_parent(name, parent.map(|p| p as usize)) as u64)
            .unwrap_or(0)
    }

    fn end_span(&self, id: u64) {
        if let Ok(mut c) = self.collector.lock() {
            c.end_span_by_id(id as usize);
        }
    }

    fn add_metadata(&self, id: u64, key: &str, value: &str) {
        if let Ok(mut c) = self.collector.lock() {
            c.add_metadata_to_span(id as usize, key, value);
        }
    }
}

/// Collects the stages run on this thread until dropped.
#[must_use = "stages are only collected while the capture is alive"]
pub struct TraceCapture {
    recorder: Arc<CollectingRecorder>,
    _scope: RecorderScope,
}

impl TraceCapture {
    /// The stages recorded so far, as `{"spans": [...]}`.
    pub fn stages_json(&self) -> serde_json::Value {
        self.recorder.stages_json()
    }

    /// The recorder, to install on helper threads.
    pub fn recorder(&self) -> Arc<CollectingRecorder> {
        self.recorder.clone()
    }
}

impl fmt::Debug for TraceCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TraceCapture").finish_non_exhaustive()
    }
}

/// Starts collecting the stages run on this thread.
pub fn capture() -> TraceCapture {
    let recorder = CollectingRecorder::new();
    let scope = RecorderScope::install(recorder.clone());
    TraceCapture {
        recorder,
        _scope: scope,
    }
}

/// Macro for easy span creation with automatic scope-based ending
#[macro_export]
macro_rules! trace_span {
    ($name:expr) => {
        let _guard = $crate::tracing::SpanGuard::new($name);
    };
    ($name:expr, $($key:expr => $value:expr),+) => {
        let _guard = $crate::tracing::SpanGuard::new($name);
        $( $crate::tracing::add_metadata($key, $value); )+
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_collector() {
        let mut collector = SpanCollector::new();

        let _span1 = collector.start_span("parent");
        collector.add_metadata("key1", "value1");

        let _span2 = collector.start_span("child");
        collector.add_metadata("key2", "value2");
        collector.end_span(); // end child

        collector.end_span(); // end parent

        assert_eq!(collector.spans().len(), 2);
        assert!(collector.spans()[0].duration_ms.is_some());
        assert!(collector.spans()[1].duration_ms.is_some());
        assert_eq!(collector.spans()[1].parent_id, Some(0));
    }

    #[test]
    fn test_disabled_collector() {
        let mut collector = SpanCollector::with_enabled(false);

        collector.start_span("test");
        collector.add_metadata("key", "value");
        collector.end_span();

        assert!(collector.spans().is_empty());
    }

    #[test]
    fn test_to_stages_json() {
        let mut collector = SpanCollector::new();

        collector.start_span("test");
        collector.add_metadata("model", "whisper");
        std::thread::sleep(std::time::Duration::from_millis(10));
        collector.end_span();

        let json = collector.to_stages_json();
        let spans = json["spans"].as_array().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0]["name"], "test");
        assert!(spans[0]["duration_ms"].as_u64().unwrap() >= 10);
    }

    #[test]
    fn no_recorder_means_no_op() {
        let _g = SpanGuard::new("ignored");
        add_metadata("k", "v");
        assert!(!is_tracing_enabled());
        assert!(current_traceparent().is_none());
    }

    #[test]
    fn captures_nested_stages_on_this_thread_only() {
        let trace = capture();
        {
            let _outer = SpanGuard::new("outer");
            add_metadata("a", "1");
            {
                let _inner = SpanGuard::new("inner");
                add_metadata("b", "2");
            }
            std::thread::spawn(|| {
                // Another thread has no recorder: nothing leaks in.
                let _g = SpanGuard::new("other-thread");
            })
            .join()
            .unwrap();
        }
        let spans = trace.recorder().spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[1].parent_id, Some(0));
        assert_eq!(spans[0].metadata["a"], "1");
        assert_eq!(spans[1].metadata["b"], "2");
    }

    #[test]
    fn concurrent_captures_are_isolated() {
        let handles: Vec<_> = (0..4)
            .map(|n| {
                std::thread::spawn(move || {
                    let trace = capture();
                    for _ in 0..=n {
                        let _g = SpanGuard::new(format!("t{n}"));
                    }
                    trace.recorder().spans().len()
                })
            })
            .collect();
        let counts: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(counts, vec![1, 2, 3, 4]);
    }

    #[test]
    fn recorder_can_follow_work_across_threads() {
        let trace = capture();
        let _outer = SpanGuard::new("outer");
        let recorder = current_recorder().unwrap();
        std::thread::spawn(move || {
            let _scope = RecorderScope::install(recorder);
            let _g = SpanGuard::new("helper");
        })
        .join()
        .unwrap();
        drop(_outer);
        let names: Vec<String> = trace
            .recorder()
            .spans()
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, vec!["outer", "helper"]);
    }

    #[test]
    fn scopes_nest_and_restore() {
        let outer = capture();
        {
            let inner = capture();
            let _g = SpanGuard::new("inner-only");
            drop(_g);
            assert_eq!(inner.recorder().spans().len(), 1);
        }
        let _g = SpanGuard::new("outer-only");
        drop(_g);
        assert_eq!(outer.recorder().spans().len(), 1);
    }
}

//! Tracing module - Span collection for observability.
//!
//! This module provides span-based tracing for TemplateExecutor and other components.
//! Spans represent units of work with timing and can be nested to form a tree structure.
//!
//! # Usage
//!
//! ```ignore
//! use xybrid_core::tracing::{SpanCollector, SpanGuard};
//!
//! let mut collector = SpanCollector::new();
//! let span_id = collector.start_span("preprocessing");
//! collector.add_metadata("step", "AudioDecode");
//! // ... do work ...
//! collector.end_span();
//!
//! // Or use RAII guard:
//! {
//!     let _guard = SpanGuard::new(&mut collector, "inference");
//!     // span automatically ends when guard drops
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

/// Collector for tracing spans during execution
#[derive(Debug, Default)]
pub struct SpanCollector {
    spans: Vec<Span>,
    stack: Vec<usize>, // Stack of active span IDs
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

    /// Start a new span, returns span ID
    pub fn start_span(&mut self, name: impl Into<String>) -> usize {
        if !self.enabled {
            return 0;
        }

        let parent_id = self.stack.last().copied();
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
        // Remove from stack if present
        self.stack.retain(|&id| id != span_id);
    }

    /// Add metadata to the current span
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        if !self.enabled {
            return;
        }

        if let Some(&span_id) = self.stack.last() {
            if let Some(span) = self.spans.get_mut(span_id) {
                span.metadata.insert(key.into(), value.into());
            }
        }
    }

    /// Get all collected spans
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Get spans as a serializable structure for API transmission
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

    /// Reset the collector for a new trace
    pub fn reset(&mut self) {
        self.spans.clear();
        self.stack.clear();
    }

    /// Check if any spans have been collected
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

// Thread-safe global span collector for use across the crate
lazy_static::lazy_static! {
    static ref GLOBAL_COLLECTOR: Arc<Mutex<SpanCollector>> = Arc::new(Mutex::new(SpanCollector::with_enabled(false)));
}

/// Initialize the global tracing collector
pub fn init_tracing(enabled: bool) {
    if let Ok(mut collector) = GLOBAL_COLLECTOR.lock() {
        collector.set_enabled(enabled);
        collector.reset();
    }
}

/// Start a span in the global collector
pub fn start_span(name: impl Into<String>) -> usize {
    GLOBAL_COLLECTOR
        .lock()
        .map(|mut c| c.start_span(name))
        .unwrap_or(0)
}

/// End the current span in the global collector
pub fn end_span() {
    if let Ok(mut collector) = GLOBAL_COLLECTOR.lock() {
        collector.end_span();
    }
}

/// End a specific span by ID
pub fn end_span_by_id(id: usize) {
    if let Ok(mut collector) = GLOBAL_COLLECTOR.lock() {
        collector.end_span_by_id(id);
    }
}

/// Add metadata to the current span
pub fn add_metadata(key: impl Into<String>, value: impl Into<String>) {
    if let Ok(mut collector) = GLOBAL_COLLECTOR.lock() {
        collector.add_metadata(key, value);
    }
}

/// Get the stages JSON from the global collector
pub fn get_stages_json() -> serde_json::Value {
    GLOBAL_COLLECTOR
        .lock()
        .map(|c| c.to_stages_json())
        .unwrap_or(serde_json::json!({ "spans": [] }))
}

/// Reset the global collector
pub fn reset_tracing() {
    if let Ok(mut collector) = GLOBAL_COLLECTOR.lock() {
        collector.reset();
    }
}

/// Check if global tracing is enabled
pub fn is_tracing_enabled() -> bool {
    GLOBAL_COLLECTOR
        .lock()
        .map(|c| c.is_enabled())
        .unwrap_or(false)
}

/// RAII guard for automatic span ending
pub struct SpanGuard {
    id: usize,
    use_global: bool,
}

impl SpanGuard {
    /// Create a new guard that uses the global collector
    pub fn new(name: impl Into<String>) -> Self {
        let id = start_span(name);
        Self {
            id,
            use_global: true,
        }
    }

    /// Get the span ID
    pub fn id(&self) -> usize {
        self.id
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if self.use_global {
            end_span();
        }
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

        let span1 = collector.start_span("parent");
        collector.add_metadata("key1", "value1");

        let span2 = collector.start_span("child");
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
}

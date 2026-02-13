//! Tracing visualization utilities for flame graph-like output
//!
//! This module provides ASCII flame graph visualization of pipeline execution traces.

use colored::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A span representing a unit of work with timing
#[derive(Debug, Clone)]
pub struct Span {
    pub name: String,
    pub start: Instant,
    pub end: Option<Instant>,
    pub parent_id: Option<usize>,
    pub metadata: HashMap<String, String>,
}

impl Span {
    pub fn duration(&self) -> Option<Duration> {
        self.end.map(|e| e.duration_since(self.start))
    }

    pub fn duration_ms(&self) -> Option<u64> {
        self.duration().map(|d| d.as_millis() as u64)
    }
}

/// Tree node for building hierarchical span visualization
#[derive(Debug)]
struct SpanNode {
    span_id: usize,
    children: Vec<SpanNode>,
}

/// Collector for tracing spans during pipeline execution
#[derive(Debug, Default)]
pub struct SpanCollector {
    spans: Vec<Span>,
    stack: Vec<usize>, // Stack of active span IDs
}

impl SpanCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new span, returns span ID
    pub fn start_span(&mut self, name: impl Into<String>) -> usize {
        let parent_id = self.stack.last().copied();
        let span_id = self.spans.len();

        self.spans.push(Span {
            name: name.into(),
            start: Instant::now(),
            end: None,
            parent_id,
            metadata: HashMap::new(),
        });

        self.stack.push(span_id);
        span_id
    }

    /// End the current span
    pub fn end_span(&mut self) {
        if let Some(span_id) = self.stack.pop() {
            if let Some(span) = self.spans.get_mut(span_id) {
                span.end = Some(Instant::now());
            }
        }
    }

    /// End a specific span by ID
    pub fn end_span_by_id(&mut self, span_id: usize) {
        if let Some(span) = self.spans.get_mut(span_id) {
            span.end = Some(Instant::now());
        }
        // Remove from stack if present
        self.stack.retain(|&id| id != span_id);
    }

    /// Add metadata to current span
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
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

    /// Build span tree from root spans
    fn build_tree(&self) -> Vec<SpanNode> {
        // Find root spans (no parent)
        let root_ids: Vec<usize> = self
            .spans
            .iter()
            .enumerate()
            .filter(|(_, span)| span.parent_id.is_none())
            .map(|(id, _)| id)
            .collect();

        root_ids.into_iter().map(|id| self.build_node(id)).collect()
    }

    fn build_node(&self, span_id: usize) -> SpanNode {
        let children: Vec<SpanNode> = self
            .spans
            .iter()
            .enumerate()
            .filter(|(_, span)| span.parent_id == Some(span_id))
            .map(|(id, _)| self.build_node(id))
            .collect();

        SpanNode { span_id, children }
    }

    /// Generate ASCII flame graph visualization
    pub fn render_flame_graph(&self) -> String {
        let mut output = String::new();
        let tree = self.build_tree();

        // Calculate total duration for scaling
        let total_duration: u64 = self
            .spans
            .iter()
            .filter(|s| s.parent_id.is_none())
            .filter_map(|s| s.duration_ms())
            .sum();

        if total_duration == 0 {
            return "No spans recorded".to_string();
        }

        output.push_str(&format!(
            "\n{}\n",
            "═══════════════════════════════════════════════════════════════════════════════"
                .bright_cyan()
        ));
        output.push_str(&format!(
            "{}\n",
            "                         PIPELINE EXECUTION TRACE".bold()
        ));
        output.push_str(&format!(
            "{}\n\n",
            "═══════════════════════════════════════════════════════════════════════════════"
                .bright_cyan()
        ));

        for node in &tree {
            self.render_node(&mut output, node, 0, total_duration);
        }

        output.push_str(&format!(
            "\n{}\n",
            "───────────────────────────────────────────────────────────────────────────────"
                .bright_black()
        ));

        // Summary
        output.push_str(&format!(
            "Total Duration: {}\n",
            format!("{}ms", total_duration).bright_yellow()
        ));

        let span_count = self.spans.len();
        output.push_str(&format!(
            "Total Spans: {}\n",
            span_count.to_string().bright_cyan()
        ));

        output
    }

    fn render_node(&self, output: &mut String, node: &SpanNode, depth: usize, total_ms: u64) {
        let span = &self.spans[node.span_id];
        let duration_ms = span.duration_ms().unwrap_or(0);

        // Calculate bar width (max 50 chars)
        let bar_width = if total_ms > 0 {
            ((duration_ms as f64 / total_ms as f64) * 50.0) as usize
        } else {
            0
        };
        let bar_width = bar_width.clamp(1, 50);

        // Indentation
        let indent = "  ".repeat(depth);
        let connector = if depth > 0 { "├─ " } else { "" };

        // Color code based on duration
        let (bar_char, color) = if duration_ms < 50 {
            ('█', "green")
        } else if duration_ms < 200 {
            ('█', "yellow")
        } else {
            ('█', "red")
        };

        let bar: String = std::iter::repeat_n(bar_char, bar_width).collect();
        let colored_bar = match color {
            "green" => bar.bright_green(),
            "yellow" => bar.bright_yellow(),
            "red" => bar.bright_red(),
            _ => bar.white(),
        };

        // Format name with metadata
        let mut display_name = span.name.clone();
        if let Some(target) = span.metadata.get("target") {
            display_name = format!("{} [{}]", display_name, target);
        }

        // Duration string
        let duration_str = format!("{}ms", duration_ms);
        let colored_duration = match color {
            "green" => duration_str.bright_green(),
            "yellow" => duration_str.bright_yellow(),
            "red" => duration_str.bright_red(),
            _ => duration_str.white(),
        };

        output.push_str(&format!(
            "{}{}{} {} {}\n",
            indent,
            connector.bright_black(),
            colored_bar,
            display_name.cyan(),
            colored_duration
        ));

        // Render children
        for child in &node.children {
            self.render_node(output, child, depth + 1, total_ms);
        }
    }

    /// Generate JSON trace output (compatible with Chrome trace format)
    pub fn to_chrome_trace_json(&self) -> String {
        let events: Vec<serde_json::Value> = self
            .spans
            .iter()
            .flat_map(|span| {
                let mut events = vec![];
                let name = &span.name;
                let ts = 0; // Would need real timestamps

                // Begin event
                events.push(serde_json::json!({
                    "name": name,
                    "cat": "pipeline",
                    "ph": "B",
                    "ts": ts,
                    "pid": 1,
                    "tid": 1,
                    "args": span.metadata
                }));

                // End event
                if let Some(duration) = span.duration() {
                    events.push(serde_json::json!({
                        "name": name,
                        "cat": "pipeline",
                        "ph": "E",
                        "ts": ts + duration.as_micros() as u64,
                        "pid": 1,
                        "tid": 1
                    }));
                }

                events
            })
            .collect();

        serde_json::json!({ "traceEvents": events }).to_string()
    }
}

// Thread-safe global span collector
lazy_static::lazy_static! {
    pub static ref GLOBAL_COLLECTOR: Arc<Mutex<SpanCollector>> = Arc::new(Mutex::new(SpanCollector::new()));
}

/// Global functions for easy tracing
pub fn start_span(name: impl Into<String>) -> usize {
    GLOBAL_COLLECTOR.lock().unwrap().start_span(name)
}

pub fn end_span() {
    GLOBAL_COLLECTOR.lock().unwrap().end_span()
}

pub fn end_span_by_id(id: usize) {
    GLOBAL_COLLECTOR.lock().unwrap().end_span_by_id(id)
}

pub fn add_metadata(key: impl Into<String>, value: impl Into<String>) {
    GLOBAL_COLLECTOR.lock().unwrap().add_metadata(key, value)
}

pub fn render_trace() -> String {
    GLOBAL_COLLECTOR.lock().unwrap().render_flame_graph()
}

pub fn reset_collector() {
    *GLOBAL_COLLECTOR.lock().unwrap() = SpanCollector::new();
}

/// RAII guard for automatic span ending
pub struct SpanGuard {
    _id: usize,
}

impl SpanGuard {
    pub fn new(name: impl Into<String>) -> Self {
        let id = start_span(name);
        Self { _id: id }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        end_span();
    }
}

/// Macro for easy span creation
#[macro_export]
macro_rules! trace_span {
    ($name:expr) => {
        let _guard = $crate::tracing_viz::SpanGuard::new($name);
    };
    ($name:expr, $($key:expr => $value:expr),+) => {
        let _guard = $crate::tracing_viz::SpanGuard::new($name);
        $( $crate::tracing_viz::add_metadata($key, $value); )+
    };
}

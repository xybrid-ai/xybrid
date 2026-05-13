//! Execution listener — global hook for observing TemplateExecutor events.
//!
//! This module provides a lightweight global listener pattern (similar to
//! `crate::tracing`) that downstream crates (e.g., xybrid-sdk) can use to
//! receive execution lifecycle events without coupling xybrid-core to the
//! telemetry pipeline.
//!
//! # Usage
//!
//! ```rust,ignore
//! use xybrid_core::execution::listener::{set_execution_listener, ExecutionEvent};
//!
//! set_execution_listener(|event| {
//!     match event {
//!         ExecutionEvent::Started { model_id, method } => { /* ... */ }
//!         ExecutionEvent::Completed { model_id, method, latency_ms } => { /* ... */ }
//!         ExecutionEvent::Failed { model_id, method, latency_ms, error } => { /* ... */ }
//!     }
//! });
//! ```

use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Events emitted by TemplateExecutor during execution.
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    /// Emitted when an execute method begins.
    Started {
        model_id: String,
        /// Which method was called (e.g., "execute", "execute_streaming").
        method: String,
    },
    /// Emitted when an execute method completes successfully.
    Completed {
        model_id: String,
        method: String,
        latency_ms: u64,
    },
    /// Emitted when an execute method fails.
    Failed {
        model_id: String,
        method: String,
        latency_ms: u64,
        error: String,
    },
}

type ListenerFn = Box<dyn Fn(ExecutionEvent) + Send + Sync>;

lazy_static::lazy_static! {
    static ref EXECUTION_LISTENER: Arc<Mutex<Option<ListenerFn>>> = Arc::new(Mutex::new(None));
}

/// Register a global execution listener.
///
/// Only one listener can be active at a time; calling this replaces any
/// previously registered listener.
pub fn set_execution_listener(listener: impl Fn(ExecutionEvent) + Send + Sync + 'static) {
    if let Ok(mut l) = EXECUTION_LISTENER.lock() {
        *l = Some(Box::new(listener));
    }
}

/// Remove the currently registered execution listener.
pub fn clear_execution_listener() {
    if let Ok(mut l) = EXECUTION_LISTENER.lock() {
        *l = None;
    }
}

/// Emit an execution event to the registered listener (if any).
pub(crate) fn emit(event: ExecutionEvent) {
    if let Ok(l) = EXECUTION_LISTENER.lock() {
        if let Some(listener) = l.as_ref() {
            listener(event);
        }
    }
}

/// RAII guard that emits `Started` on creation and a terminal event on drop.
///
/// Call [`set_failed`](ExecutionGuard::set_failed) before dropping to emit
/// `Failed`; call [`set_controlled_abort`](ExecutionGuard::set_controlled_abort)
/// for aborts that are handled by a higher-level orchestrator event.
pub(crate) struct ExecutionGuard {
    model_id: String,
    method: String,
    start: Instant,
    terminal: Mutex<ExecutionTerminal>,
}

enum ExecutionTerminal {
    Completed,
    Failed(String),
    Suppressed,
}

impl ExecutionGuard {
    /// Create a new guard and immediately emit `ExecutionEvent::Started`.
    pub(crate) fn new(model_id: impl Into<String>, method: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let method = method.into();
        emit(ExecutionEvent::Started {
            model_id: model_id.clone(),
            method: method.clone(),
        });
        Self {
            model_id,
            method,
            start: Instant::now(),
            terminal: Mutex::new(ExecutionTerminal::Completed),
        }
    }

    /// Create a guard that never emits `Started` or `Completed`, only
    /// `Failed` when [`set_failed`](ExecutionGuard::set_failed) is
    /// called.
    ///
    /// Used by outer-span methods whose user-facing telemetry is
    /// already carried by an inner SDK-level event (e.g. `ModelComplete`
    /// from `XybridModel::run_streaming_with_context`). Emitting a
    /// `Started` / `Completed` pair from those paths produces
    /// duplicate "phantom" rows in the Traces dashboard that don't
    /// correspond to a user-facing operation. Error reporting is
    /// preserved: `set_failed` still flips the terminal state so the
    /// guard's `Drop` emits `Failed`.
    pub(crate) fn new_silent(model_id: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            method: method.into(),
            start: Instant::now(),
            terminal: Mutex::new(ExecutionTerminal::Suppressed),
        }
    }

    /// Mark this execution as failed. The error message will be included
    /// in the `Failed` event emitted on drop.
    pub(crate) fn set_failed(&self, error: impl Into<String>) {
        if let Ok(mut terminal) = self.terminal.lock() {
            *terminal = ExecutionTerminal::Failed(error.into());
        }
    }

    /// Suppress the terminal event when a controlled abort is represented by a
    /// richer orchestrator event such as `LocalAborted`.
    pub(crate) fn set_controlled_abort(&self) {
        if let Ok(mut terminal) = self.terminal.lock() {
            *terminal = ExecutionTerminal::Suppressed;
        }
    }
}

impl Drop for ExecutionGuard {
    fn drop(&mut self) {
        let latency_ms = self.start.elapsed().as_millis() as u64;
        let terminal = self
            .terminal
            .lock()
            .map(|mut terminal| std::mem::replace(&mut *terminal, ExecutionTerminal::Suppressed))
            .unwrap_or(ExecutionTerminal::Completed);
        match terminal {
            ExecutionTerminal::Failed(err) => emit(ExecutionEvent::Failed {
                model_id: self.model_id.clone(),
                method: self.method.clone(),
                latency_ms,
                error: err,
            }),
            ExecutionTerminal::Completed => emit(ExecutionEvent::Completed {
                model_id: self.model_id.clone(),
                method: self.method.clone(),
                latency_ms,
            }),
            ExecutionTerminal::Suppressed => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // The execution listener and the events it fires are global, so the
    // tests below must run serially — otherwise one test's listener
    // captures another test's guard emits. `cargo test` parallelises by
    // default; this lock pins them to one-at-a-time without pulling in
    // the `serial_test` crate.
    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn controlled_abort_suppresses_terminal_execution_event() {
        let _serial = test_lock();
        clear_execution_listener();
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_listener = events.clone();

        set_execution_listener(move |event| {
            events_for_listener.lock().unwrap().push(event);
        });

        {
            let guard = ExecutionGuard::new("local-model", "execute_streaming");
            guard.set_controlled_abort();
        }

        clear_execution_listener();
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ExecutionEvent::Started { model_id, method }
                if model_id == "local-model" && method == "execute_streaming"
        ));
    }

    #[test]
    fn silent_guard_emits_nothing_on_success() {
        let _serial = test_lock();
        clear_execution_listener();
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_listener = events.clone();

        set_execution_listener(move |event| {
            events_for_listener.lock().unwrap().push(event);
        });

        {
            // Successful path: silent guard goes through new → drop with
            // no `set_failed` call. Neither `Started` nor `Completed`
            // should surface to the listener.
            let _guard =
                ExecutionGuard::new_silent("local-model", "execute_streaming_with_context");
        }

        clear_execution_listener();
        let events = events.lock().unwrap();
        assert!(
            events.is_empty(),
            "silent guard on a success path must not emit any events; got: {:?}",
            events
        );
    }

    #[test]
    fn silent_guard_still_emits_failed_on_error() {
        let _serial = test_lock();
        clear_execution_listener();
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_listener = events.clone();

        set_execution_listener(move |event| {
            events_for_listener.lock().unwrap().push(event);
        });

        {
            // Error path: `set_failed` flips the terminal state and the
            // guard's `Drop` must emit `Failed` so error attribution is
            // preserved even when `Started`/`Completed` are suppressed.
            let guard = ExecutionGuard::new_silent("local-model", "execute_streaming_with_context");
            guard.set_failed("boom");
        }

        clear_execution_listener();
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ExecutionEvent::Failed { model_id, method, error, .. }
                if model_id == "local-model"
                    && method == "execute_streaming_with_context"
                    && error == "boom"
        ));
    }
}

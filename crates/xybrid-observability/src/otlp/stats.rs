//! Per-signal delivery counters (CONTRACT.md "Lifecycle and delivery").

/// Delivery counters of one signal. Every snapshot satisfies
/// `accepted == exported + dropped + failed + pending`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignalStats {
    /// Every item offered, including items refused at admission.
    pub accepted: u64,
    /// Items the endpoint acknowledged (2xx minus partial-success rejections).
    pub exported: u64,
    /// Items discarded by local policy without a deciding delivery attempt:
    /// admission refused (item or byte budget), offered after shutdown began,
    /// oversized single item, still queued when shutdown cancelled, pending
    /// or offered after the destination's response hook disabled it.
    pub dropped: u64,
    /// Items whose export ended without acknowledgement: rejection, retries
    /// exhausted, deadline or cancellation in flight, transport error, lost
    /// acknowledgement (delivery unknown), partial-success rejection, encoding
    /// failure, oversized metrics collection.
    pub failed: u64,
    /// Accepted items not yet terminal (queued, in flight, or in backoff).
    pub pending: u64,
    /// Library-owned bytes held for delivery now: the conservative per-item
    /// estimate computed at admission, summed over the pending items (queued
    /// and in flight; for metrics, the collection in flight). It is the
    /// running total that `max_buffered_bytes` (32 MiB) admits against, not
    /// process memory, and it is zero once nothing is pending (always after
    /// shutdown).
    pub buffered_bytes: u64,
}

impl SignalStats {
    /// Whether `accepted == exported + dropped + failed + pending`.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.accepted == self.exported + self.dropped + self.failed + self.pending
    }

    pub(crate) fn settle(&mut self, outcome: Outcome) {
        debug_assert!(outcome.total() <= self.pending, "settled more than pending");
        self.pending = self.pending.saturating_sub(outcome.total());
        self.exported += outcome.exported;
        self.failed += outcome.failed;
        self.dropped += outcome.dropped;
    }
}

/// Delivery counters of a pipeline, per signal (zeros when not configured).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    /// Sampled spans that ended.
    pub traces: SignalStats,
    /// Log records emitted to the pipeline's logger provider.
    pub logs: SignalStats,
    /// Metric data points of each collection handed to the exporter.
    pub metrics: SignalStats,
}

/// Terminal outcomes of a group of items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Outcome {
    pub exported: u64,
    pub failed: u64,
    pub dropped: u64,
}

impl Outcome {
    pub(crate) fn failed(n: u64) -> Self {
        Self {
            failed: n,
            ..Self::default()
        }
    }

    pub(crate) fn dropped(n: u64) -> Self {
        Self {
            dropped: n,
            ..Self::default()
        }
    }

    pub(crate) fn total(&self) -> u64 {
        self.exported + self.failed + self.dropped
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.exported += other.exported;
        self.failed += other.failed;
        self.dropped += other.dropped;
    }
}

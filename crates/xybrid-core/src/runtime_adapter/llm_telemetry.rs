//! Shared telemetry helpers for LLM backends.
//!
//! Both `mistral` and `llama_cpp` backends derive streaming-chunk
//! latency statistics the same way (mean / p95 of inter-chunk gaps,
//! TTFT, decode/prefill TPS). Keeping the implementation here avoids
//! duplication and keeps the metric semantics identical across
//! backends — important because the platform ingest expects a single
//! contract regardless of which backend produced the numbers.
//!
//! Two entry points are offered:
//! - `StreamingTelemetry`: ergonomic recorder for backends that
//!   don't need synthetic-timestamp injection (e.g. `llama_cpp`).
//! - `compute_streaming_fields`: pure function for backends that
//!   own their own timestamp vec — used by `mistral` so its
//!   `handle_response` tests can drive the state machine with
//!   injected `Instant`s.

use std::time::Instant;

/// Compute (mean, p95) of inter-chunk latencies (ms). Returns `(None, None)`
/// when the input is empty — callers should treat that as "only one chunk
/// was emitted, latency summaries are not meaningful".
///
/// p95 uses nearest-rank on a sorted copy:
///   `sorted[((len - 1) as f32 * 0.95).round() as usize]`
pub(crate) fn itl_stats(xs: &[u32]) -> (Option<f32>, Option<u32>) {
    if xs.is_empty() {
        return (None, None);
    }
    let sum: u64 = xs.iter().map(|&x| x as u64).sum();
    let mean = sum as f32 / xs.len() as f32;

    let mut sorted: Vec<u32> = xs.to_vec();
    sorted.sort_unstable();
    let idx = (((sorted.len() - 1) as f32) * 0.95).round() as usize;
    let p95 = sorted[idx];

    (Some(mean), Some(p95))
}

/// Telemetry fields derived from a streaming LLM run.
///
/// Merged into `GenerationOutput` at the end of generation. Engines
/// that report `decode_tps` / `prefill_tps` directly (e.g. mistralrs
/// `Usage.avg_*_tok_per_sec`) should prefer their values via
/// `engine_reported.or(fields.decode_tps)` on the caller side —
/// reported values are more accurate than our TTFT/ITL estimates.
#[derive(Debug, Clone)]
pub(crate) struct StreamingTelemetryFields {
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    pub ttft_ms: Option<u64>,
    pub mean_itl_ms: Option<f32>,
    pub p95_itl_ms: Option<u32>,
    pub emitted_chunks: Option<u32>,
    pub inter_chunk_ms: Vec<u32>,
    pub decode_tps: Option<f32>,
    pub prefill_tps: Option<f32>,
}

/// Compute telemetry fields from raw streaming observations.
///
/// - `ttft_ms`: first chunk timestamp − `start`. Captures prefill +
///   one token of sampling.
/// - `mean_itl_ms` / `p95_itl_ms`: gaps between consecutive chunk
///   timestamps (see [`itl_stats`]).
/// - `decode_tps`: `1000 / mean_itl_ms`. Inverse of the mean
///   inter-chunk gap — steady-state decode throughput, excludes
///   prefill. `None` if no ITL data or mean is 0.
/// - `prefill_tps`: `prompt_token_count * 1000 / ttft_ms`.
///   Approximates prefill throughput on prompts long enough for
///   prefill to dominate. Pass `prompt_token_count = 0` when the
///   caller has an engine-reported value to override this with
///   (mistralrs) — the derivation returns `None` in that case.
pub(crate) fn compute_streaming_fields(
    start: Instant,
    chunk_timestamps: &[Instant],
    prompt_token_count: usize,
    tokens_generated: usize,
) -> StreamingTelemetryFields {
    let elapsed = start.elapsed();

    let ttft_ms = chunk_timestamps
        .first()
        .map(|t0| t0.duration_since(start).as_millis() as u64);

    let inter_chunk_ms: Vec<u32> = chunk_timestamps
        .windows(2)
        .map(|w| w[1].duration_since(w[0]).as_millis() as u32)
        .collect();

    let (mean_itl_ms, p95_itl_ms) = itl_stats(&inter_chunk_ms);

    let decode_tps = mean_itl_ms.and_then(|m| if m > 0.0 { Some(1000.0 / m) } else { None });

    let prefill_tps = ttft_ms.and_then(|t| {
        if t > 0 && prompt_token_count > 0 {
            Some(prompt_token_count as f32 * 1000.0 / t as f32)
        } else {
            None
        }
    });

    let tokens_per_second = if elapsed.as_secs_f32() > 0.0 {
        tokens_generated as f32 / elapsed.as_secs_f32()
    } else {
        0.0
    };

    StreamingTelemetryFields {
        generation_time_ms: elapsed.as_millis() as u64,
        tokens_per_second,
        ttft_ms,
        mean_itl_ms,
        p95_itl_ms,
        emitted_chunks: Some(chunk_timestamps.len() as u32),
        inter_chunk_ms,
        decode_tps,
        prefill_tps,
    }
}

/// Ergonomic streaming-telemetry recorder.
///
/// Used by backends that don't need their own timestamp vec for
/// test injection (currently `llama_cpp`). Call [`Self::new`]
/// immediately before the generation loop, [`Self::record_chunk`]
/// from inside the streaming callback on every C-layer invocation
/// (before any stop-pattern filtering — the stream itself is being
/// measured, not user-visible output), then [`Self::finalize`] with
/// the canonical `tokens_generated` count.
///
/// Backends with pre-existing timestamp-owning state (e.g.
/// `mistral::StreamState`) should call [`compute_streaming_fields`]
/// directly instead.
pub(crate) struct StreamingTelemetry {
    start: Instant,
    chunk_timestamps: Vec<Instant>,
    prompt_token_count: usize,
}

impl StreamingTelemetry {
    /// Start the clock and record the prompt token count.
    ///
    /// Call as late as possible before the generation loop (after
    /// tokenization / KV-cache reset / context window checks) so
    /// TTFT captures only the prefill + first-decode window.
    pub fn new(prompt_token_count: usize) -> Self {
        Self {
            start: Instant::now(),
            chunk_timestamps: Vec::new(),
            prompt_token_count,
        }
    }

    /// Record a chunk observation at `Instant::now()`.
    pub fn record_chunk(&mut self) {
        self.chunk_timestamps.push(Instant::now());
    }

    /// Finalize into telemetry fields.
    ///
    /// `tokens_generated` is the canonical count reflected in
    /// `tokens_per_second`. Pass in the backend's authoritative
    /// count (e.g. `output_tokens.len()`) — not derived from
    /// `chunk_timestamps.len()` because engines may coalesce
    /// multiple tokens into a single chunk.
    pub fn finalize(&self, tokens_generated: usize) -> StreamingTelemetryFields {
        compute_streaming_fields(
            self.start,
            &self.chunk_timestamps,
            self.prompt_token_count,
            tokens_generated,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_streaming_fields, itl_stats};
    use std::time::{Duration, Instant};

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(itl_stats(&[]), (None, None));
    }

    #[test]
    fn single_value() {
        let (mean, p95) = itl_stats(&[10]);
        assert_eq!(mean, Some(10.0));
        assert_eq!(p95, Some(10));
    }

    #[test]
    fn multiple_values_sorted() {
        let (mean, p95) = itl_stats(&[10, 20, 30, 40]);
        assert_eq!(mean, Some(25.0));
        // len=4 → idx = round(3 * 0.95) = round(2.85) = 3 → sorted[3] = 40.
        assert_eq!(p95, Some(40));
    }

    #[test]
    fn multiple_values_unsorted() {
        let (mean, p95) = itl_stats(&[30, 10, 40, 20]);
        assert_eq!(mean, Some(25.0));
        assert_eq!(p95, Some(40));
    }

    #[test]
    fn streaming_fields_empty_stream() {
        let start = Instant::now();
        let fields = compute_streaming_fields(start, &[], 0, 0);
        assert_eq!(fields.ttft_ms, None);
        assert_eq!(fields.mean_itl_ms, None);
        assert_eq!(fields.p95_itl_ms, None);
        assert_eq!(fields.emitted_chunks, Some(0));
        assert!(fields.inter_chunk_ms.is_empty());
        assert_eq!(fields.decode_tps, None);
        assert_eq!(fields.prefill_tps, None);
        assert_eq!(fields.tokens_per_second, 0.0);
    }

    #[test]
    fn streaming_fields_derives_ttft_itl_decode_tps() {
        let start = Instant::now();
        let chunks = [
            start + Duration::from_millis(40),
            start + Duration::from_millis(60),
            start + Duration::from_millis(80),
            start + Duration::from_millis(100),
        ];
        let fields = compute_streaming_fields(start, &chunks, 16, 4);

        assert_eq!(fields.ttft_ms, Some(40));
        // gaps: 20, 20, 20
        assert_eq!(fields.mean_itl_ms, Some(20.0));
        assert_eq!(fields.p95_itl_ms, Some(20));
        // 1000 / 20 = 50 tok/s
        assert_eq!(fields.decode_tps, Some(50.0));
        // 16 prompt tokens over 40ms TTFT = 400 tok/s
        assert_eq!(fields.prefill_tps, Some(400.0));
        assert_eq!(fields.emitted_chunks, Some(4));
        assert_eq!(fields.inter_chunk_ms, vec![20, 20, 20]);
    }

    #[test]
    fn streaming_fields_zero_prompt_suppresses_prefill_tps() {
        // Caller signals "override me, engine reports prefill directly"
        // by passing prompt_token_count = 0.
        let start = Instant::now();
        let chunks = [start + Duration::from_millis(50)];
        let fields = compute_streaming_fields(start, &chunks, 0, 1);
        assert_eq!(fields.ttft_ms, Some(50));
        assert_eq!(fields.prefill_tps, None);
    }
}

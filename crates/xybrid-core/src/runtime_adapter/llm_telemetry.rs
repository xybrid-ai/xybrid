//! Shared telemetry helpers for LLM backends.
//!
//! Both `mistral` and `llama_cpp` backends derive streaming-chunk
//! latency statistics the same way (mean / p95 of inter-chunk gaps).
//! Keeping the implementation here avoids duplication and keeps the
//! metric semantics identical across backends — important because
//! the platform ingest expects a single contract regardless of which
//! backend produced the numbers.

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

#[cfg(test)]
mod tests {
    use super::itl_stats;

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
}

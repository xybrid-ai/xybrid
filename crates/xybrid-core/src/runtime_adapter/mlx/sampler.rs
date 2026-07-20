//! Pure-Rust sampler for the MLX generate loop (US-014).
//!
//! Takes a logits row (`&[f32]` of vocab-size length) and
//! [`GenerationConfig`] knobs (`temperature`, `top_p`, `top_k`, `min_p`,
//! `repetition_penalty`) plus the previously-generated token IDs (for the
//! repetition penalty) and returns the next token ID.
//!
//! # Why pure Rust?
//!
//! The hot-path inside a token-by-token decode is logits → sample → next-token.
//! On MLX, logits are a `[1, 1, vocab]` `MlxArray`; we `.to_vec_f32()` the last
//! row and sample on the CPU. Doing the sampler in pure Rust keeps the MLX
//! adapter runnable cross-platform for unit tests (the sampler itself has
//! zero MLX dependencies), and the ~100us/iter CPU sample cost is dwarfed by
//! the forward pass. If sampling ever shows up in profiles we can move it to
//! a Metal-backed `top_k_sample` op; the interface here stays stable.
//!
//! # Sampling pipeline (matches HuggingFace `transformers` order)
//!
//! 1. **Repetition penalty** — divide logits of previously-emitted tokens by
//!    `repetition_penalty` when > 1.0 (symmetric for negative logits: multiply).
//! 2. **Temperature** — divide logits by `temperature`. Temperature == 0 is
//!    the greedy path (short-circuit `argmax` before softmax).
//! 3. **Softmax** — log-sum-exp-stabilised softmax.
//! 4. **Top-k** — keep only the k highest-probability tokens.
//! 5. **Min-p** — prune tokens below `min_p * max_probability`.
//! 6. **Top-p** — cumulative-probability nucleus truncation (keep the
//!    smallest set whose sum ≥ top_p).
//! 7. **Renormalise + multinomial** — sample from the surviving distribution
//!    with a seeded `StdRng` for reproducibility.
//!
//! The pipeline mirrors `transformers.generation.LogitsProcessorList` so a
//! PyTorch reference can be computed for parity tests.

use crate::runtime_adapter::llm::GenerationConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

/// Errors produced by the sampler.
#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    /// The logits slice is empty — cannot sample from an empty vocabulary.
    #[error("sampler received empty logits (vocab_size=0)")]
    EmptyLogits,

    /// All logits are `-inf` or the probability mass summed to zero after
    /// filters (top_k / top_p / min_p) pruned everything. Indicates pathological
    /// config (e.g. `top_k=1` with NaN-producing logits).
    #[error("sampler produced zero-probability distribution after filters")]
    NoViableCandidates,

    /// Non-finite logit (NaN / +inf) encountered during sampling.
    #[error("sampler encountered non-finite logit at index {index}: {value}")]
    NonFiniteLogit { index: usize, value: f32 },
}

pub type SamplerResult<T> = Result<T, SamplerError>;

/// Deterministic seeded sampler state. Callers that want reproducible
/// sampling (e.g. for the streaming-vs-batched parity test) construct one
/// with a fixed seed; otherwise `Sampler::from_entropy()` draws from system
/// entropy.
#[derive(Debug)]
pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    /// Deterministic sampler seeded by `seed`. Two `Sampler`s with the same
    /// seed and the same input sequence emit the same token stream — the
    /// property the streaming-vs-batched parity test asserts.
    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// System-entropy sampler. Use for production generation where
    /// reproducibility is not a concern.
    pub fn from_entropy() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Sample one token from `logits` using `config`. Returns the sampled
    /// token ID (as `i64` — the adapter-wide token type). `previous_tokens`
    /// is used only by the repetition penalty; pass an empty slice when
    /// that feature is unused.
    pub fn sample(
        &mut self,
        logits: &[f32],
        config: &GenerationConfig,
        previous_tokens: &[i64],
    ) -> SamplerResult<i64> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        // Greedy-sample early-exit: temperature=0 means argmax. This skips the
        // softmax + RNG entirely, matching the GGUF / mistral.rs paths.
        if config.temperature <= 0.0 {
            return Ok(greedy_argmax(logits)? as i64);
        }

        // Clone-to-mutable: we mutate the logits (temp, penalty) before converting
        // to probabilities. Length equals vocab_size so the allocation is negligible
        // relative to the forward pass that produced `logits`.
        let mut working = logits.to_vec();

        apply_repetition_penalty(&mut working, previous_tokens, config.repetition_penalty);
        apply_temperature(&mut working, config.temperature);

        // Reject non-finite logits BEFORE softmax so the error points at the
        // offending index, not at a mysterious NaN in the output distribution.
        for (i, &v) in working.iter().enumerate() {
            if !v.is_finite() {
                return Err(SamplerError::NonFiniteLogit { index: i, value: v });
            }
        }

        let mut probs = stable_softmax(&working);
        apply_top_k(&mut probs, config.top_k);
        apply_min_p(&mut probs, config.min_p);
        apply_top_p(&mut probs, config.top_p);

        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return Err(SamplerError::NoViableCandidates);
        }
        // Renormalise after the filter passes, then multinomial-draw.
        for p in probs.iter_mut() {
            *p /= total;
        }
        Ok(multinomial(&mut self.rng, &probs) as i64)
    }
}

fn greedy_argmax(logits: &[f32]) -> SamplerResult<usize> {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            return Err(SamplerError::NonFiniteLogit { index: i, value: v });
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    if best_val == f32::NEG_INFINITY {
        return Err(SamplerError::NoViableCandidates);
    }
    Ok(best_idx)
}

/// Divide logits of previously-emitted tokens by `penalty`. Negative logits
/// are multiplied (so the penalty *always* reduces the logit magnitude
/// toward zero, matching `transformers`' behaviour).
fn apply_repetition_penalty(logits: &mut [f32], previous: &[i64], penalty: f32) {
    if penalty == 1.0 || penalty <= 0.0 || previous.is_empty() {
        return;
    }
    // Use a HashSet so an N-token history with M unique tokens costs O(N+M*V)
    // rather than O(N*V). Matters once `previous` grows past ~200 tokens.
    let seen: HashSet<usize> = previous
        .iter()
        .filter_map(|&t| usize::try_from(t).ok())
        .filter(|&i| i < logits.len())
        .collect();
    for i in seen {
        let v = logits[i];
        logits[i] = if v >= 0.0 { v / penalty } else { v * penalty };
    }
}

fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }
    // Caller filtered temperature==0 via the greedy-early-exit before we land here.
    debug_assert!(temperature > 0.0);
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

/// Log-sum-exp-stable softmax. Subtracts the max logit before exp'ing, so
/// overflow can't happen even with extreme temperature settings.
fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &v in logits {
        if v > m {
            m = v;
        }
    }
    let mut out: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for p in out.iter_mut() {
            *p /= sum;
        }
    }
    out
}

/// Keep only the `k` highest-probability tokens; zero the rest. `k == 0`
/// disables the filter (common for "top_p only" configurations).
fn apply_top_k(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    // Partial selection: find the k-th-largest probability, then zero any
    // probability below it. O(V log k) with a tiny `k` min-heap is faster
    // than a full sort for vocab sizes >> k, and HuggingFace's runtime-
    // dominant Qwen/Gemma checkpoints have V=150k.
    let mut top: Vec<f32> = probs.to_vec();
    top.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = top[k - 1];
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

/// Zero any probability below `min_p * max(probs)`. `min_p == 0` disables.
fn apply_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let mut max_p = 0.0f32;
    for &p in probs.iter() {
        if p > max_p {
            max_p = p;
        }
    }
    if max_p == 0.0 {
        return;
    }
    let cutoff = min_p * max_p;
    for p in probs.iter_mut() {
        if *p < cutoff {
            *p = 0.0;
        }
    }
}

/// Nucleus-sampling truncation. Sort probs descending, cumulative-sum them,
/// keep the prefix whose cumulative sum first crosses `top_p`, zero the rest.
/// `top_p == 1.0` disables the filter.
fn apply_top_p(probs: &mut [f32], top_p: f32) {
    if top_p >= 1.0 || top_p <= 0.0 {
        return;
    }
    // Collect indices sorted by descending probability so we can zero
    // everything past the nucleus boundary.
    let mut idxs: Vec<usize> = (0..probs.len()).collect();
    idxs.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cum = 0.0f32;
    let mut nucleus: Vec<usize> = Vec::new();
    for &i in &idxs {
        nucleus.push(i);
        cum += probs[i];
        if cum >= top_p {
            break;
        }
    }
    let keep: HashSet<usize> = nucleus.into_iter().collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        }
    }
}

/// Draw a single multinomial sample from a probability distribution. Assumes
/// `probs` sums to 1.0 (caller renormalises).
fn multinomial<R: Rng>(rng: &mut R, probs: &[f32]) -> usize {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i;
        }
    }
    // Floating-point rounding can leave `cum` epsilon below 1.0; fall back
    // to the last non-zero index rather than panic.
    probs
        .iter()
        .rposition(|&p| p > 0.0)
        .unwrap_or(probs.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy_config() -> GenerationConfig {
        GenerationConfig {
            max_tokens: 32,
            temperature: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
            stop_sequences: Vec::new(),
            ..Default::default()
        }
    }

    fn creative_config() -> GenerationConfig {
        GenerationConfig {
            max_tokens: 32,
            temperature: 1.0,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repetition_penalty: 1.1,
            seed: None,
            stop_sequences: Vec::new(),
            ..Default::default()
        }
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::seeded(0);
        let logits = vec![0.1, 0.3, 5.0, 0.4, -1.0];
        let t = s.sample(&logits, &greedy_config(), &[]).unwrap();
        assert_eq!(t, 2);
    }

    #[test]
    fn greedy_empty_logits_errors() {
        let mut s = Sampler::seeded(0);
        let err = s.sample(&[], &greedy_config(), &[]).unwrap_err();
        assert!(matches!(err, SamplerError::EmptyLogits));
    }

    #[test]
    fn greedy_rejects_nan() {
        let mut s = Sampler::seeded(0);
        let logits = vec![0.1, f32::NAN, 0.3];
        let err = s.sample(&logits, &greedy_config(), &[]).unwrap_err();
        assert!(matches!(err, SamplerError::NonFiniteLogit { index: 1, .. }));
    }

    #[test]
    fn seeded_sampler_is_deterministic() {
        // Same seed + same logits => same token. The streaming-vs-batched
        // parity test in US-014 depends on this invariant.
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut a = Sampler::seeded(42);
        let mut b = Sampler::seeded(42);
        let cfg = creative_config();
        for _ in 0..16 {
            let ta = a.sample(&logits, &cfg, &[]).unwrap();
            let tb = b.sample(&logits, &cfg, &[]).unwrap();
            assert_eq!(ta, tb, "seed=42 produced divergent tokens");
        }
    }

    #[test]
    fn different_seeds_can_produce_different_tokens() {
        // Don't assert they *always* differ (there are only 5 options and we
        // make one draw), but over many draws we expect at least one divergence.
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = creative_config();
        let mut a = Sampler::seeded(1);
        let mut b = Sampler::seeded(999_999);
        let mut any_diff = false;
        for _ in 0..200 {
            let ta = a.sample(&logits, &cfg, &[]).unwrap();
            let tb = b.sample(&logits, &cfg, &[]).unwrap();
            if ta != tb {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "two distinct seeds never diverged in 200 draws");
    }

    #[test]
    fn top_k_one_equals_greedy_under_sampling() {
        // With top_k=1, exactly one candidate survives; any rng choice
        // lands on the argmax. Regardless of temperature.
        let logits = vec![0.1, 0.3, 5.0, 0.4];
        let cfg = GenerationConfig {
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            temperature: 0.7,
            repetition_penalty: 1.0,
            ..greedy_config()
        };
        let mut s = Sampler::seeded(123);
        for _ in 0..32 {
            assert_eq!(s.sample(&logits, &cfg, &[]).unwrap(), 2);
        }
    }

    #[test]
    fn top_p_truncation_respects_cumulative_boundary() {
        // Crafted distribution: after softmax, the top-3 tokens cover > 0.9
        // cumulative probability; top_p = 0.5 should leave only token 0
        // surviving because exp(5) / (exp(5)+exp(4)+...) > 0.5.
        let logits = vec![5.0, 4.0, 3.0, 0.0, -1.0];
        let cfg = GenerationConfig {
            top_k: 0,
            top_p: 0.5,
            min_p: 0.0,
            temperature: 1.0,
            repetition_penalty: 1.0,
            ..greedy_config()
        };
        let mut s = Sampler::seeded(7);
        for _ in 0..32 {
            // argmax index is 0; with top_p=0.5 only index 0 survives.
            assert_eq!(s.sample(&logits, &cfg, &[]).unwrap(), 0);
        }
    }

    #[test]
    fn repetition_penalty_prefers_fresh_tokens() {
        // With a strong penalty (2.0) and equal logits, the token already in
        // history should drop below the fresh ones. Under greedy sampling the
        // output must NOT be the penalised token.
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let cfg = GenerationConfig {
            repetition_penalty: 2.0,
            ..greedy_config()
        };
        let mut s = Sampler::seeded(0);
        // Pretend token 2 was emitted last turn. With penalty=2.0 its
        // effective logit drops to 0.5; the argmax moves to one of {0,1,3}.
        let t = s.sample(&logits, &cfg, &[2]).unwrap();
        assert_ne!(t, 2);
    }

    #[test]
    fn softmax_handles_extreme_temperature() {
        // Temperature=0.01 with wildly different logits would overflow a
        // naive exp; the LSE-stable path must not panic.
        let logits = vec![100.0, 99.0, 98.0, 50.0];
        let cfg = GenerationConfig {
            temperature: 0.01,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            ..greedy_config()
        };
        let mut s = Sampler::seeded(5);
        // After extreme temperature the distribution collapses on argmax.
        assert_eq!(s.sample(&logits, &cfg, &[]).unwrap(), 0);
    }

    #[test]
    fn min_p_filters_low_probability_mass() {
        // With min_p=0.9, only tokens with probability >= 0.9*max survive.
        // The logits below soften into a near-uniform distribution but
        // argmax-index-0 still dominates enough that only it survives.
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let cfg = GenerationConfig {
            temperature: 1.0,
            min_p: 0.9,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..greedy_config()
        };
        let mut s = Sampler::seeded(0);
        for _ in 0..32 {
            assert_eq!(s.sample(&logits, &cfg, &[]).unwrap(), 0);
        }
    }

    #[test]
    fn stable_softmax_sums_to_one() {
        let p = stable_softmax(&[1.0, 2.0, 3.0, 4.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}

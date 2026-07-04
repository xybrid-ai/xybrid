//! Statistical trust: deterministic confidence intervals + the gate policy.
//!
//! The gate is **not** a raw-number compare (trust layer → statistical
//! trust). It applies a minimum case count, a non-inferiority margin, a
//! bootstrap confidence interval, and can return **`inconclusive`** as a
//! first-class verdict — so `eval gate` / `eval compare` never ship or block on
//! noise. Candidates that score erratically across repeats are flagged flaky.
//!
//! Everything here is deterministic given a seed: the PRNG is a self-contained
//! SplitMix64 (no `rand` dependency, no entropy source), resampling draws are
//! bias-free (multiply-high, never modulo), and summation order is fixed.

use serde::{Deserialize, Serialize};

// ============================================================================
// Deterministic PRNG (SplitMix64)
// ============================================================================

/// A small, self-contained, fully deterministic PRNG (SplitMix64). Used for
/// bootstrap resampling so a confidence interval is reproducible from a seed on
/// any platform. Not cryptographic.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit output (canonical SplitMix64).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform index in `[0, n)` with no modulo bias (Lemire multiply-high).
    /// Guarantees `result < n` for any `n >= 1`.
    pub fn index(&mut self, n: usize) -> usize {
        debug_assert!(n >= 1, "index requires n >= 1");
        (((self.next_u64() as u128) * (n as u128)) >> 64) as usize
    }
}

// ============================================================================
// Summary statistics (fixed-order, deterministic)
// ============================================================================

/// Deterministic mean of a sample (fixed left-fold order → reproducible across
/// architectures; no FMA contraction on separate adds). Empty ⇒ 0.0.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &v in values {
        sum += v;
    }
    sum / values.len() as f64
}

/// Population standard deviation (fixed order, ÷N). Fewer than 2 values ⇒ 0.0.
pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let mut acc = 0.0;
    for &v in values {
        acc += (v - m) * (v - m);
    }
    (acc / values.len() as f64).sqrt()
}

/// Sample standard deviation (fixed order, ÷(N−1) — Bessel-corrected). Repeats
/// are a *sample* of a candidate's behavior, so flakiness uses this, not the
/// population estimator. Fewer than 2 values ⇒ 0.0.
pub fn sample_std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let mut acc = 0.0;
    for &v in values {
        acc += (v - m) * (v - m);
    }
    (acc / (values.len() - 1) as f64).sqrt()
}

/// Nearest-rank percentile (`p` in `0..=100`). Returns `None` for an empty
/// sample (never underflows). Sorts a copy.
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(percentile_sorted(&sorted, p))
}

/// Nearest-rank percentile over an already-sorted, non-empty slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    let idx = rank.min(sorted.len()) - 1;
    sorted[idx]
}

// ============================================================================
// Bootstrap confidence interval
// ============================================================================

/// Default bootstrap resamples.
pub const DEFAULT_BOOTSTRAP_ITERATIONS: usize = 1000;
/// Default confidence level.
pub const DEFAULT_CONFIDENCE: f64 = 0.95;
/// Default seed (greedy/seed=42 is pinned for reproducibility).
pub const DEFAULT_SEED: u64 = 42;
/// CPU budget for bootstrap resampling: total draws (`iterations × n`) are
/// clamped to this so a large sample (n can be ~500k) can't spin the gate.
const MAX_BOOTSTRAP_DRAWS: usize = 20_000_000;

/// A confidence interval on an aggregate quality score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound.
    pub low: f64,
    /// Upper bound.
    pub high: f64,
    /// Sample size.
    pub n: usize,
    /// Repeat count used to produce the scores (1 = single run).
    pub repeats: u32,
    /// Bootstrap iterations actually used after CPU-budget clamping.
    #[serde(default)]
    pub effective_iterations: u32,
}

/// Bootstrap a `confidence`-level CI for the mean of `scores`, deterministically
/// from `seed`. Returns `None` for an empty sample.
pub fn bootstrap_ci(
    scores: &[f64],
    seed: u64,
    iterations: usize,
    confidence: f64,
) -> Option<ConfidenceInterval> {
    let n = scores.len();
    if n == 0 {
        return None;
    }
    // Clamp iterations so total resample draws (iterations × n) stay within the
    // CPU budget; keeps a large-n gate fast while remaining deterministic.
    let iterations = iterations
        .max(1)
        .min((MAX_BOOTSTRAP_DRAWS / n.max(1)).max(1));
    let mut rng = Rng::new(seed);
    let mut means = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += scores[rng.index(n)];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = (1.0 - confidence) / 2.0;
    Some(ConfidenceInterval {
        low: percentile_sorted(&means, alpha * 100.0),
        high: percentile_sorted(&means, (1.0 - alpha) * 100.0),
        n,
        repeats: 1,
        effective_iterations: iterations.min(u32::MAX as usize) as u32,
    })
}

// ============================================================================
// Gate policy
// ============================================================================

/// Tolerance for threshold comparisons. Plain-summation bootstrap can land a
/// true `0.9` at `0.89999999999999913`; without slack the gate would spuriously
/// `Fail` a candidate that met the bar exactly.
const GATE_EPS: f64 = 1e-9;

/// The statistical gate policy. Defaults are permissive (so a small explicit
/// run isn't forced to `inconclusive`); production ramps raise `min_cases` and
/// set `non_inferiority_margin` via the evalset manifest.
#[derive(Debug, Clone)]
pub struct GatePolicy {
    /// Minimum scorable cases; below this ⇒ `Inconclusive`.
    pub min_cases: usize,
    /// Absolute minimum quality (0..1). When set, the CI is compared to it.
    pub min_quality: Option<f64>,
    /// Maximum allowed p95 latency (ms). A hard SLO; over budget ⇒ `Fail`.
    pub max_p95_latency_ms: Option<f64>,
    /// Non-inferiority margin (compare mode): |Δ| within this ⇒ tie/inconclusive.
    pub non_inferiority_margin: f64,
    /// Cross-repeat std above which a candidate is flaky (quarantined).
    pub flaky_std_threshold: f64,
    /// Bootstrap seed.
    pub seed: u64,
    /// Bootstrap resamples.
    pub bootstrap_iterations: usize,
    /// Confidence level.
    pub confidence: f64,
}

impl Default for GatePolicy {
    fn default() -> Self {
        Self {
            min_cases: 1,
            min_quality: None,
            max_p95_latency_ms: None,
            non_inferiority_margin: 0.0,
            flaky_std_threshold: 0.1,
            seed: DEFAULT_SEED,
            bootstrap_iterations: DEFAULT_BOOTSTRAP_ITERATIONS,
            confidence: DEFAULT_CONFIDENCE,
        }
    }
}

/// A gate verdict — `Inconclusive` is first-class (CI-neutral: neither promote
/// nor block).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateVerdict {
    /// Meets the bar.
    Pass,
    /// Below the bar.
    Fail,
    /// Not enough signal to decide (too few cases, within margin, CI straddles).
    Inconclusive,
}

/// The full gate decision, with the evidence behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateDecision {
    /// The verdict.
    pub verdict: GateVerdict,
    /// Aggregate quality (mean score).
    pub quality: f64,
    /// Confidence interval, when computable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci: Option<ConfidenceInterval>,
    /// Whether the candidate was flagged flaky (excluded from ranking).
    pub flaky: bool,
    /// Human-readable explanation of the decision.
    pub reason: String,
}

/// Latency evidence available to a gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyStats {
    /// P95 latency over measured scorable cases.
    pub p95_ms: Option<f64>,
    /// Scorable cases with measured latency.
    pub measured_cases: usize,
    /// Total scorable cases that should have latency.
    pub scorable_cases: usize,
}

impl LatencyStats {
    /// Build latency evidence from a legacy p95-only value.
    pub fn from_p95(p95_ms: Option<f64>, scorable_cases: usize) -> Self {
        Self {
            p95_ms,
            measured_cases: p95_ms.map_or(0, |_| scorable_cases),
            scorable_cases,
        }
    }
}

impl GatePolicy {
    /// Evaluate a candidate's per-case `scores` (Unblessed cases already
    /// excluded by the caller) against this policy.
    ///
    /// `p95_latency_ms` enables the latency SLO; `baseline_quality` enables
    /// non-inferiority (compare mode). Evaluation order is fixed:
    /// **min-N → latency SLO → quality-vs-CI → non-inferiority**.
    pub fn evaluate(
        &self,
        scores: &[f64],
        p95_latency_ms: Option<f64>,
        baseline_quality: Option<f64>,
    ) -> GateDecision {
        self.evaluate_with_latency(
            scores,
            LatencyStats::from_p95(p95_latency_ms, scores.len()),
            baseline_quality,
        )
    }

    /// Evaluate with latency coverage, not only the measured percentile.
    pub fn evaluate_with_latency(
        &self,
        scores: &[f64],
        latency: LatencyStats,
        baseline_quality: Option<f64>,
    ) -> GateDecision {
        self.evaluate_with_latency_and_repeats(scores, latency, baseline_quality, None)
    }

    /// Evaluate with optional repeat-quality evidence for flakiness.
    pub fn evaluate_with_latency_and_repeats(
        &self,
        scores: &[f64],
        latency: LatencyStats,
        baseline_quality: Option<f64>,
        repeat_qualities: Option<&[f64]>,
    ) -> GateDecision {
        let n = scores.len();

        let quality = mean(scores);
        let mut ci = bootstrap_ci(
            scores,
            self.seed,
            self.bootstrap_iterations,
            self.confidence,
        );
        let mut verdict = GateVerdict::Pass;
        let mut reasons = Vec::new();

        if n < self.min_cases.max(1) {
            verdict = stricter(verdict, GateVerdict::Inconclusive);
            reasons.push(format!(
                "minimum case count inconclusive ({n} < {})",
                self.min_cases.max(1)
            ));
        }

        if let Some(max) = self.max_p95_latency_ms {
            // An observed p95 over budget is a Fail even under partial coverage:
            // unmeasured cases can only add violations, never retract observed ones.
            if let Some(p95) = latency.p95_ms {
                if p95 > max {
                    verdict = stricter(verdict, GateVerdict::Fail);
                    reasons.push(format!("p95 latency {p95:.0}ms over budget {max:.0}ms"));
                }
            }
            if latency.measured_cases < latency.scorable_cases {
                verdict = stricter(verdict, GateVerdict::Inconclusive);
                reasons.push(format!(
                    "latency measured for {}/{} cases",
                    latency.measured_cases, latency.scorable_cases
                ));
            } else if latency.p95_ms.is_none() {
                verdict = stricter(verdict, GateVerdict::Inconclusive);
                reasons.push("p95 latency not measured".to_string());
            }
        }

        if let Some(min_q) = self.min_quality {
            match &ci {
                Some(ci) if ci.high < min_q - GATE_EPS => {
                    verdict = stricter(verdict, GateVerdict::Fail);
                    reasons.push(format!(
                        "CI [{:.3},{:.3}] entirely below {min_q:.3}",
                        ci.low, ci.high
                    ));
                }
                Some(ci) if ci.low >= min_q - GATE_EPS => {}
                Some(ci) => {
                    verdict = stricter(verdict, GateVerdict::Inconclusive);
                    reasons.push(format!(
                        "CI [{:.3},{:.3}] straddles threshold {min_q:.3}",
                        ci.low, ci.high
                    ));
                }
                None if n == 0 => {
                    verdict = stricter(verdict, GateVerdict::Inconclusive);
                    reasons.push(format!(
                        "quality CI not computable for threshold {min_q:.3}"
                    ));
                }
                None => {
                    if quality < min_q - GATE_EPS {
                        verdict = stricter(verdict, GateVerdict::Fail);
                        reasons.push(format!("quality {quality:.3} below threshold {min_q:.3}"));
                    }
                }
            }
        }

        if let Some(base) = baseline_quality {
            let delta = quality - base;
            if delta.abs() <= self.non_inferiority_margin {
                verdict = stricter(verdict, GateVerdict::Inconclusive);
                reasons.push(format!("Δ {delta:+.3} within non-inferiority margin"));
            } else if delta > 0.0 {
            } else {
                verdict = stricter(verdict, GateVerdict::Fail);
                reasons.push(format!("Δ {delta:+.3} under baseline"));
            }
        }

        if self.min_quality.is_none()
            && self.max_p95_latency_ms.is_none()
            && baseline_quality.is_none()
        {
            verdict = stricter(verdict, GateVerdict::Inconclusive);
            reasons.push("no gate criteria configured".to_string());
        }

        let flaky = repeat_qualities.is_some_and(|qualities| {
            if let Some(ci) = &mut ci {
                ci.repeats = qualities.len().max(1).min(u32::MAX as usize) as u32;
            }
            let std = sample_std_dev(qualities);
            let flaky = is_flaky(qualities, self.flaky_std_threshold);
            if flaky {
                verdict = stricter(verdict, GateVerdict::Inconclusive);
                reasons.push(format!(
                    "flaky: repeat std {std:.3} over threshold {:.3}",
                    self.flaky_std_threshold
                ));
            }
            flaky
        });

        GateDecision {
            verdict,
            quality,
            ci,
            flaky,
            reason: if reasons.is_empty() {
                "all configured gate criteria passed".to_string()
            } else {
                reasons.join("; ")
            },
        }
    }
}

fn stricter(current: GateVerdict, next: GateVerdict) -> GateVerdict {
    match (current, next) {
        (GateVerdict::Fail, _) | (_, GateVerdict::Fail) => GateVerdict::Fail,
        (GateVerdict::Inconclusive, _) | (_, GateVerdict::Inconclusive) => {
            GateVerdict::Inconclusive
        }
        (GateVerdict::Pass, GateVerdict::Pass) => GateVerdict::Pass,
    }
}

/// Whether a candidate is flaky given its per-repeat quality means: fewer than 2
/// repeats can't be flaky; otherwise the cross-repeat std must stay within
/// `threshold`.
pub fn is_flaky(repeat_qualities: &[f64], threshold: f64) -> bool {
    repeat_qualities.len() >= 2 && sample_std_dev(repeat_qualities) > threshold
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PRNG ----

    #[test]
    fn splitmix64_matches_reference_vector() {
        // Canonical SplitMix64 outputs for seed 0 (widely published reference
        // values used to seed xoshiro). Locks the algorithm, not just self-
        // consistency.
        let mut rng = Rng::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }

    #[test]
    fn rng_same_seed_is_reproducible() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_different_seed_diverges() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn index_is_in_range_and_never_n() {
        let mut rng = Rng::new(7);
        for n in [1usize, 2, 3, 50, 97, 1000] {
            for _ in 0..2000 {
                let i = rng.index(n);
                assert!(i < n, "index {i} >= n {n}");
            }
        }
    }

    #[test]
    fn index_sequence_is_deterministic() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        let seq_a: Vec<usize> = (0..20).map(|_| a.index(50)).collect();
        let seq_b: Vec<usize> = (0..20).map(|_| b.index(50)).collect();
        assert_eq!(seq_a, seq_b);
    }

    // ---- percentile ----

    #[test]
    fn percentile_empty_is_none() {
        assert_eq!(percentile(&[], 50.0), None);
    }

    #[test]
    fn percentile_exact_indices() {
        // 1..=20; p95 nearest-rank = ceil(0.95*20)=19 → 19th value = 19.0
        let v: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        assert_eq!(percentile(&v, 95.0), Some(19.0));
        // 1..=50; p95 = ceil(47.5)=48 → 48th value = 48.0
        let v: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        assert_eq!(percentile(&v, 95.0), Some(48.0));
    }

    #[test]
    fn percentile_tiny_samples_dont_panic() {
        assert_eq!(percentile(&[5.0], 95.0), Some(5.0));
        assert_eq!(percentile(&[1.0, 2.0], 50.0), Some(1.0));
        assert_eq!(percentile(&[1.0, 2.0], 100.0), Some(2.0));
    }

    // ---- bootstrap CI ----

    #[test]
    fn ci_all_pass_is_degenerate_one() {
        let scores = vec![1.0; 30];
        let ci = bootstrap_ci(&scores, DEFAULT_SEED, 500, 0.95).unwrap();
        assert_eq!(ci.low, 1.0);
        assert_eq!(ci.high, 1.0);
    }

    #[test]
    fn ci_all_fail_is_degenerate_zero() {
        let scores = vec![0.0; 30];
        let ci = bootstrap_ci(&scores, DEFAULT_SEED, 500, 0.95).unwrap();
        assert_eq!(ci.low, 0.0);
        assert_eq!(ci.high, 0.0);
    }

    #[test]
    fn ci_is_deterministic_under_seed() {
        let scores: Vec<f64> = (0..50)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let a = bootstrap_ci(&scores, DEFAULT_SEED, 500, 0.95).unwrap();
        let b = bootstrap_ci(&scores, DEFAULT_SEED, 500, 0.95).unwrap();
        assert_eq!(a, b);
        // Brackets the true mean 0.5, with a plausible width (tolerance band,
        // not exact f64 equality).
        assert!(a.low < 0.5 && a.high > 0.5, "CI {:?} should bracket 0.5", a);
        assert!(a.high - a.low < 0.5, "CI too wide: {:?}", a);
    }

    #[test]
    fn ci_empty_is_none() {
        assert_eq!(bootstrap_ci(&[], DEFAULT_SEED, 100, 0.95), None);
    }

    #[test]
    fn ci_large_n_is_bounded_and_deterministic() {
        // S3: a large sample (default 1000 iters × 500k draws = 500M) must be
        // clamped so the gate can't spin; result stays deterministic per seed.
        let scores: Vec<f64> = (0..500_000)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let start = std::time::Instant::now();
        let a = bootstrap_ci(&scores, DEFAULT_SEED, DEFAULT_BOOTSTRAP_ITERATIONS, 0.95).unwrap();
        let elapsed = start.elapsed();
        // Effective iterations are clamped: 20M / 500k = 40, far below the 1000
        // requested, so this returns quickly rather than running 500M draws.
        assert!(
            elapsed.as_secs() < 5,
            "bootstrap on large n took too long: {elapsed:?}"
        );
        let b = bootstrap_ci(&scores, DEFAULT_SEED, DEFAULT_BOOTSTRAP_ITERATIONS, 0.95).unwrap();
        assert_eq!(a, b, "clamped bootstrap must stay deterministic");
        assert_eq!(a.effective_iterations, 40);
        assert!(a.low < 0.5 && a.high > 0.5, "CI {a:?} should bracket 0.5");
    }

    // ---- gate verdict matrix ----

    fn policy(min_cases: usize, min_quality: Option<f64>) -> GatePolicy {
        GatePolicy {
            min_cases,
            min_quality,
            bootstrap_iterations: 500,
            ..GatePolicy::default()
        }
    }

    #[test]
    fn gate_below_min_n_is_inconclusive() {
        let p = policy(30, Some(0.9));
        let d = p.evaluate(&vec![1.0; 29], None, None);
        assert_eq!(d.verdict, GateVerdict::Inconclusive);
        assert!(d.reason.contains("minimum case count"));
    }

    #[test]
    fn gate_min_n_evaluated_before_ci() {
        // n=2 all-pass would otherwise produce CI [1,1] ≥ 0.9 (a Pass); min-N
        // must win.
        let p = policy(30, Some(0.9));
        let d = p.evaluate(&[1.0, 1.0], None, None);
        assert_eq!(d.verdict, GateVerdict::Inconclusive);
    }

    #[test]
    fn gate_zero_cases_inconclusive_no_panic() {
        let p = policy(1, Some(0.9));
        let d = p.evaluate(&[], None, None);
        assert_eq!(d.verdict, GateVerdict::Inconclusive);
    }

    #[test]
    fn gate_all_pass_above_threshold_passes() {
        let p = policy(10, Some(0.9));
        let d = p.evaluate(&vec![1.0; 50], None, None);
        assert_eq!(d.verdict, GateVerdict::Pass);
    }

    #[test]
    fn gate_all_fail_below_threshold_fails() {
        let p = policy(10, Some(0.9));
        let d = p.evaluate(&vec![0.0; 50], None, None);
        assert_eq!(d.verdict, GateVerdict::Fail);
    }

    #[test]
    fn gate_ci_straddling_threshold_is_inconclusive() {
        // mean ~0.9 with spread → CI straddles a 0.9 threshold.
        let scores: Vec<f64> = (0..50).map(|i| if i < 45 { 1.0 } else { 0.0 }).collect();
        let p = policy(10, Some(0.9));
        let d = p.evaluate(&scores, None, None);
        assert_eq!(d.verdict, GateVerdict::Inconclusive, "{}", d.reason);
    }

    #[test]
    fn gate_latency_over_budget_fails() {
        let mut p = policy(10, Some(0.5));
        p.max_p95_latency_ms = Some(800.0);
        let d = p.evaluate(&vec![1.0; 50], Some(900.0), None);
        assert_eq!(d.verdict, GateVerdict::Fail);
        assert!(d.reason.contains("latency"));
    }

    #[test]
    fn gate_exactly_at_threshold_passes_not_fails() {
        // C1: scores that sum to exactly the threshold mean (0.9) can come back
        // as 0.8999999… from plain-summation bootstrap. With the epsilon, the
        // CI sits at/above the bar → Pass, never a false-negative Fail.
        let p = policy(10, Some(0.9));
        let d = p.evaluate(&vec![0.9; 50], None, None);
        assert_eq!(d.verdict, GateVerdict::Pass, "{}", d.reason);
    }

    #[test]
    fn gate_latency_budget_set_but_p95_unmeasured_is_inconclusive() {
        // C5: a budget with no measured p95 must NOT silently pass.
        let mut p = policy(10, Some(0.5));
        p.max_p95_latency_ms = Some(800.0);
        let d = p.evaluate(&vec![1.0; 50], None, None);
        assert_eq!(d.verdict, GateVerdict::Inconclusive, "{}", d.reason);
        assert!(d.reason.contains("latency measured for 0/50 cases"));
    }

    #[test]
    fn gate_latency_partial_coverage_is_inconclusive() {
        let mut p = policy(1, Some(0.5));
        p.max_p95_latency_ms = Some(800.0);
        let d = p.evaluate_with_latency(
            &vec![1.0; 100],
            LatencyStats {
                p95_ms: Some(100.0),
                measured_cases: 1,
                scorable_cases: 100,
            },
            None,
        );
        assert_eq!(d.verdict, GateVerdict::Inconclusive, "{}", d.reason);
        assert!(d.reason.contains("latency measured for 1/100 cases"));
    }

    #[test]
    fn gate_latency_over_budget_fails_even_with_partial_coverage() {
        // An observed over-budget p95 is a Fail; under-coverage cannot soften it
        // to Inconclusive because unmeasured cases can only add violations.
        let mut p = policy(1, Some(0.5));
        p.max_p95_latency_ms = Some(800.0);
        let d = p.evaluate_with_latency(
            &vec![1.0; 100],
            LatencyStats {
                p95_ms: Some(2000.0),
                measured_cases: 1,
                scorable_cases: 100,
            },
            None,
        );
        assert_eq!(d.verdict, GateVerdict::Fail, "{}", d.reason);
        assert!(d.reason.contains("over budget"));
        assert!(d.reason.contains("latency measured for 1/100 cases"));
    }

    #[test]
    fn gate_combines_min_quality_and_non_inferiority_verdicts() {
        let mut p = policy(10, Some(0.9));
        p.non_inferiority_margin = 0.05;
        let d = p.evaluate(&vec![1.0; 50], None, Some(1.2));
        assert_eq!(d.verdict, GateVerdict::Fail, "{}", d.reason);
        assert!(d.reason.contains("under baseline"));
    }

    #[test]
    fn gate_non_inferiority_tie_is_inconclusive() {
        let mut p = policy(10, None);
        p.non_inferiority_margin = 0.05;
        // candidate 0.82 vs baseline 0.80 → Δ 0.02 within margin → tie.
        let scores: Vec<f64> = (0..50).map(|i| if i < 41 { 1.0 } else { 0.0 }).collect();
        let d = p.evaluate(&scores, None, Some(0.80));
        assert_eq!(d.verdict, GateVerdict::Inconclusive, "{}", d.reason);
    }

    #[test]
    fn gate_non_inferiority_clear_win_passes() {
        let mut p = policy(10, None);
        p.non_inferiority_margin = 0.05;
        let d = p.evaluate(&vec![1.0; 50], None, Some(0.80));
        assert_eq!(d.verdict, GateVerdict::Pass);
    }

    #[test]
    fn gate_non_inferiority_clear_loss_fails() {
        let mut p = policy(10, None);
        p.non_inferiority_margin = 0.05;
        let d = p.evaluate(&vec![0.0; 50], None, Some(0.80));
        assert_eq!(d.verdict, GateVerdict::Fail);
    }

    #[test]
    fn gate_no_criteria_is_inconclusive() {
        let p = policy(1, None);
        let d = p.evaluate(&[1.0, 0.0, 1.0], None, None);
        assert_eq!(d.verdict, GateVerdict::Inconclusive);
        assert!(d.reason.contains("no gate criteria"));
    }

    // ---- flaky ----

    #[test]
    fn flaky_requires_two_repeats() {
        assert!(!is_flaky(&[0.9], 0.1));
    }

    #[test]
    fn flaky_high_variance_is_flagged() {
        assert!(is_flaky(&[0.9, 0.2, 0.95, 0.1], 0.1));
    }

    #[test]
    fn flaky_stable_repeats_not_flagged() {
        assert!(!is_flaky(&[0.90, 0.91, 0.89, 0.90], 0.1));
    }

    #[test]
    fn flaky_uses_sample_std_not_population() {
        // C7: two repeats [0.90, 0.72]. Population std ≈ 0.09 (misses the 0.1
        // bar); sample std ≈ 0.127 (flags it). Flakiness must use the sample
        // estimator so a borderline-noisy 2-repeat set isn't waved through.
        let repeats = [0.90, 0.72];
        assert!(
            std_dev(&repeats) < 0.1,
            "population std should be below the threshold here"
        );
        assert!(
            sample_std_dev(&repeats) > 0.1,
            "sample std should exceed the threshold here"
        );
        assert!(is_flaky(&repeats, 0.1));
    }

    #[test]
    fn gate_repeats_make_flaky_pass_inconclusive() {
        let p = policy(1, Some(0.0));
        let d = p.evaluate_with_latency_and_repeats(
            &[1.0, 1.0, 0.0, 0.0],
            LatencyStats::from_p95(Some(100.0), 4),
            None,
            Some(&[1.0, 0.0]),
        );
        assert!(d.flaky);
        assert_eq!(d.verdict, GateVerdict::Inconclusive);
        assert_eq!(d.ci.unwrap().repeats, 2);
        assert!(d.reason.contains("flaky: repeat std"));
    }
}

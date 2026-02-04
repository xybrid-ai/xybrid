//! Benchmark types for execution provider performance comparison.
//!
//! This module provides types for benchmarking model inference,
//! comparing CPU vs hardware-accelerated execution providers (CoreML, CUDA, etc.).
//!
//! All platform bindings (Flutter, Kotlin, Swift) should use these SDK types
//! to ensure consistent statistics calculation across platforms.
//!
//! ## Comparing Execution Providers
//!
//! Use `compare_benchmarks()` to compare performance between different providers:
//!
//! ```rust
//! use xybrid_sdk::{BenchmarkResult, compare_benchmarks};
//!
//! let cpu_result = BenchmarkResult::from_times(
//!     "model".to_string(), "cpu".to_string(), vec![100.0, 110.0], 210.0
//! );
//! let coreml_result = BenchmarkResult::from_times(
//!     "model".to_string(), "coreml-ane".to_string(), vec![20.0, 22.0], 42.0
//! );
//!
//! let comparison = compare_benchmarks(&cpu_result, &coreml_result);
//! println!("{}", comparison);
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Benchmark Result Types
// ============================================================================

/// Statistics from a benchmark run.
///
/// Contains timing statistics calculated from individual iteration times,
/// including mean, median, percentiles, and standard deviation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Model identifier
    pub model_id: String,
    /// Execution provider used (cpu, coreml-ane, cuda, etc.)
    pub execution_provider: String,
    /// Number of iterations
    pub iterations: u32,
    /// Mean latency in milliseconds
    pub mean_ms: f64,
    /// Minimum latency in milliseconds
    pub min_ms: f64,
    /// Maximum latency in milliseconds
    pub max_ms: f64,
    /// Median latency in milliseconds
    pub median_ms: f64,
    /// 95th percentile latency in milliseconds
    pub p95_ms: f64,
    /// Standard deviation in milliseconds
    pub std_dev_ms: f64,
    /// Total time including warmup
    pub total_time_ms: f64,
    /// Individual iteration times (for detailed analysis)
    pub iteration_times_ms: Vec<f64>,
}

impl BenchmarkResult {
    /// Create a new BenchmarkResult from iteration times.
    ///
    /// Calculates all statistics (mean, median, p95, std dev) from the provided
    /// timing data. Returns a result with zeroed statistics if times is empty.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier of the benchmarked model
    /// * `execution_provider` - Name of the execution provider (e.g., "cpu", "coreml-ane")
    /// * `times_ms` - Vector of individual iteration times in milliseconds
    /// * `total_time_ms` - Total elapsed time including warmup
    ///
    /// # Example
    ///
    /// ```
    /// use xybrid_sdk::BenchmarkResult;
    ///
    /// let times = vec![10.0, 12.0, 11.0, 15.0, 9.0];
    /// let result = BenchmarkResult::from_times(
    ///     "my-model".to_string(),
    ///     "cpu".to_string(),
    ///     times,
    ///     100.0,
    /// );
    /// assert_eq!(result.iterations, 5);
    /// assert_eq!(result.min_ms, 9.0);
    /// ```
    pub fn from_times(
        model_id: String,
        execution_provider: String,
        times_ms: Vec<f64>,
        total_time_ms: f64,
    ) -> Self {
        let iterations = times_ms.len() as u32;

        if times_ms.is_empty() {
            return Self {
                model_id,
                execution_provider,
                iterations: 0,
                mean_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                median_ms: 0.0,
                p95_ms: 0.0,
                std_dev_ms: 0.0,
                total_time_ms,
                iteration_times_ms: vec![],
            };
        }

        let mut sorted = times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);

        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let p95_idx = ((sorted.len() as f64) * 0.95).ceil() as usize - 1;
        let p95 = sorted.get(p95_idx).copied().unwrap_or(max);

        let variance =
            sorted.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
        let std_dev = variance.sqrt();

        Self {
            model_id,
            execution_provider,
            iterations,
            mean_ms: mean,
            min_ms: min,
            max_ms: max,
            median_ms: median,
            p95_ms: p95,
            std_dev_ms: std_dev,
            total_time_ms,
            iteration_times_ms: times_ms,
        }
    }

    /// Format as a human-readable summary.
    ///
    /// Returns a multi-line string with key statistics formatted for display.
    ///
    /// # Example
    ///
    /// ```
    /// use xybrid_sdk::BenchmarkResult;
    ///
    /// let times = vec![10.0, 12.0, 11.0];
    /// let result = BenchmarkResult::from_times(
    ///     "my-model".to_string(),
    ///     "cpu".to_string(),
    ///     times,
    ///     50.0,
    /// );
    /// let summary = result.summary();
    /// assert!(summary.contains("my-model"));
    /// assert!(summary.contains("cpu"));
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "{} ({}):\n  Mean: {:.2}ms | Median: {:.2}ms | Std Dev: {:.2}ms\n  Min: {:.2}ms | Max: {:.2}ms | P95: {:.2}ms\n  Iterations: {}",
            self.model_id,
            self.execution_provider,
            self.mean_ms,
            self.median_ms,
            self.std_dev_ms,
            self.min_ms,
            self.max_ms,
            self.p95_ms,
            self.iterations
        )
    }
}

// ============================================================================
// Execution Provider Info
// ============================================================================

/// Information about the current execution provider configuration.
///
/// Provides compile-time platform detection to determine which hardware
/// acceleration is available (CoreML on Apple, CUDA on NVIDIA, etc.).
///
/// # Example
///
/// ```
/// use xybrid_sdk::ExecutionProviderInfo;
///
/// let info = ExecutionProviderInfo::current();
/// println!("Running on {} with {}", info.platform, info.name);
/// if info.coreml_available {
///     println!("CoreML acceleration available!");
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProviderInfo {
    /// Name of the execution provider (cpu, coreml-ane, cuda, etc.)
    pub name: String,
    /// Detailed description of the execution provider
    pub description: String,
    /// Whether CoreML execution provider is available
    pub coreml_available: bool,
    /// Whether running on Apple Silicon (aarch64 macOS/iOS)
    pub apple_silicon: bool,
    /// Platform identifier (ios, macos, android, linux, windows, other)
    pub platform: String,
}

impl ExecutionProviderInfo {
    /// Create execution provider info for the current platform.
    ///
    /// Uses compile-time platform detection to determine:
    /// - Target OS (iOS, macOS, Android, Linux, Windows)
    /// - Architecture (aarch64 for Apple Silicon detection)
    /// - Feature flags (coreml-ep for CoreML availability)
    ///
    /// # Example
    ///
    /// ```
    /// use xybrid_sdk::ExecutionProviderInfo;
    ///
    /// let info = ExecutionProviderInfo::current();
    /// assert!(!info.name.is_empty());
    /// assert!(!info.platform.is_empty());
    /// ```
    pub fn current() -> Self {
        let platform = if cfg!(target_os = "ios") {
            "ios"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_os = "android") {
            "android"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "windows"
        } else {
            "other"
        };

        let coreml_available = cfg!(feature = "ort-coreml");
        let apple_silicon = cfg!(any(target_os = "ios", target_os = "macos"))
            && cfg!(target_arch = "aarch64");

        let (name, description) = if coreml_available && apple_silicon {
            (
                "coreml-ane".to_string(),
                "CoreML with Neural Engine acceleration (Apple Silicon)".to_string(),
            )
        } else if coreml_available {
            (
                "coreml".to_string(),
                "CoreML execution provider available".to_string(),
            )
        } else {
            (
                "cpu".to_string(),
                "CPU execution (no hardware acceleration)".to_string(),
            )
        };

        Self {
            name,
            description,
            coreml_available,
            apple_silicon,
            platform: platform.to_string(),
        }
    }
}

// ============================================================================
// Benchmark Comparison
// ============================================================================

/// Compare benchmark results between two runs.
///
/// Calculates speedup factor, time difference, and percentage improvement.
/// Useful for comparing CPU vs hardware-accelerated (CoreML, CUDA) performance.
///
/// # Arguments
///
/// * `baseline` - The baseline benchmark result (typically CPU)
/// * `comparison` - The comparison benchmark result (typically hardware-accelerated)
///
/// # Returns
///
/// A formatted string showing the comparison including speedup factor.
///
/// # Example
///
/// ```
/// use xybrid_sdk::{BenchmarkResult, compare_benchmarks};
///
/// let cpu = BenchmarkResult::from_times(
///     "model".to_string(),
///     "cpu".to_string(),
///     vec![100.0, 110.0, 105.0],
///     315.0,
/// );
///
/// let coreml = BenchmarkResult::from_times(
///     "model".to_string(),
///     "coreml-ane".to_string(),
///     vec![20.0, 22.0, 21.0],
///     63.0,
/// );
///
/// let comparison = compare_benchmarks(&cpu, &coreml);
/// assert!(comparison.contains("Speedup"));
/// assert!(comparison.contains("cpu"));
/// assert!(comparison.contains("coreml-ane"));
/// ```
pub fn compare_benchmarks(baseline: &BenchmarkResult, comparison: &BenchmarkResult) -> String {
    let speedup = if comparison.mean_ms > 0.0 {
        baseline.mean_ms / comparison.mean_ms
    } else {
        0.0
    };

    let diff_ms = baseline.mean_ms - comparison.mean_ms;
    let diff_pct = if baseline.mean_ms > 0.0 {
        (diff_ms / baseline.mean_ms) * 100.0
    } else {
        0.0
    };

    format!(
        "Benchmark Comparison\n\
         ====================\n\
         Baseline ({}):\n\
           Mean: {:.2}ms | Median: {:.2}ms\n\
         Comparison ({}):\n\
           Mean: {:.2}ms | Median: {:.2}ms\n\
         \n\
         Speedup: {:.2}x\n\
         Difference: {:.2}ms ({:.1}%)",
        baseline.execution_provider,
        baseline.mean_ms,
        baseline.median_ms,
        comparison.execution_provider,
        comparison.mean_ms,
        comparison.median_ms,
        speedup,
        diff_ms,
        diff_pct
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_from_times() {
        let times = vec![10.0, 12.0, 11.0, 15.0, 9.0];
        let result =
            BenchmarkResult::from_times("test-model".to_string(), "cpu".to_string(), times, 100.0);

        assert_eq!(result.iterations, 5);
        assert_eq!(result.min_ms, 9.0);
        assert_eq!(result.max_ms, 15.0);
        // Mean = (10+12+11+15+9)/5 = 57/5 = 11.4
        assert!((result.mean_ms - 11.4).abs() < 0.01);
        // Median of sorted [9,10,11,12,15] = 11
        assert_eq!(result.median_ms, 11.0);
    }

    #[test]
    fn test_benchmark_result_empty() {
        let result =
            BenchmarkResult::from_times("test".to_string(), "cpu".to_string(), vec![], 0.0);

        assert_eq!(result.iterations, 0);
        assert_eq!(result.mean_ms, 0.0);
        assert_eq!(result.min_ms, 0.0);
        assert_eq!(result.max_ms, 0.0);
    }

    #[test]
    fn test_benchmark_result_single_value() {
        let result = BenchmarkResult::from_times(
            "test".to_string(),
            "cpu".to_string(),
            vec![42.0],
            42.0,
        );

        assert_eq!(result.iterations, 1);
        assert_eq!(result.mean_ms, 42.0);
        assert_eq!(result.median_ms, 42.0);
        assert_eq!(result.min_ms, 42.0);
        assert_eq!(result.max_ms, 42.0);
        assert_eq!(result.std_dev_ms, 0.0);
    }

    #[test]
    fn test_benchmark_result_even_count_median() {
        // For even count, median is average of two middle values
        let times = vec![10.0, 20.0, 30.0, 40.0];
        let result =
            BenchmarkResult::from_times("test".to_string(), "cpu".to_string(), times, 100.0);

        // Sorted: [10, 20, 30, 40], median = (20 + 30) / 2 = 25
        assert_eq!(result.median_ms, 25.0);
    }

    #[test]
    fn test_benchmark_summary_format() {
        let result = BenchmarkResult {
            model_id: "test-model".to_string(),
            execution_provider: "coreml-ane".to_string(),
            iterations: 10,
            mean_ms: 15.5,
            min_ms: 10.0,
            max_ms: 20.0,
            median_ms: 15.0,
            p95_ms: 19.0,
            std_dev_ms: 2.5,
            total_time_ms: 200.0,
            iteration_times_ms: vec![],
        };

        let summary = result.summary();
        assert!(summary.contains("test-model"));
        assert!(summary.contains("coreml-ane"));
        assert!(summary.contains("15.50ms"));
        assert!(summary.contains("Iterations: 10"));
    }

    #[test]
    fn test_execution_provider_info_current() {
        let info = ExecutionProviderInfo::current();

        // Name should never be empty
        assert!(!info.name.is_empty());
        // Platform should never be empty
        assert!(!info.platform.is_empty());
        // Description should never be empty
        assert!(!info.description.is_empty());

        // Platform should be one of the known values
        assert!(["ios", "macos", "android", "linux", "windows", "other"]
            .contains(&info.platform.as_str()));
    }

    #[test]
    fn test_compare_benchmarks_speedup() {
        let baseline = BenchmarkResult {
            model_id: "test".to_string(),
            execution_provider: "cpu".to_string(),
            iterations: 10,
            mean_ms: 100.0,
            min_ms: 90.0,
            max_ms: 110.0,
            median_ms: 100.0,
            p95_ms: 108.0,
            std_dev_ms: 5.0,
            total_time_ms: 1000.0,
            iteration_times_ms: vec![],
        };

        let comparison = BenchmarkResult {
            model_id: "test".to_string(),
            execution_provider: "coreml-ane".to_string(),
            iterations: 10,
            mean_ms: 20.0,
            min_ms: 18.0,
            max_ms: 22.0,
            median_ms: 20.0,
            p95_ms: 21.0,
            std_dev_ms: 1.0,
            total_time_ms: 200.0,
            iteration_times_ms: vec![],
        };

        let comparison_str = compare_benchmarks(&baseline, &comparison);

        // Should show 5x speedup (100ms / 20ms = 5.0x)
        assert!(comparison_str.contains("5.00x"));
        // Should contain both provider names
        assert!(comparison_str.contains("cpu"));
        assert!(comparison_str.contains("coreml-ane"));
        // Should contain the comparison header
        assert!(comparison_str.contains("Benchmark Comparison"));
        // Should show the difference (80ms improvement)
        assert!(comparison_str.contains("80.00ms"));
    }

    #[test]
    fn test_compare_benchmarks_zero_comparison_mean() {
        let baseline = BenchmarkResult {
            model_id: "test".to_string(),
            execution_provider: "cpu".to_string(),
            iterations: 10,
            mean_ms: 100.0,
            min_ms: 90.0,
            max_ms: 110.0,
            median_ms: 100.0,
            p95_ms: 108.0,
            std_dev_ms: 5.0,
            total_time_ms: 1000.0,
            iteration_times_ms: vec![],
        };

        let comparison = BenchmarkResult {
            model_id: "test".to_string(),
            execution_provider: "broken".to_string(),
            iterations: 0,
            mean_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            median_ms: 0.0,
            p95_ms: 0.0,
            std_dev_ms: 0.0,
            total_time_ms: 0.0,
            iteration_times_ms: vec![],
        };

        let comparison_str = compare_benchmarks(&baseline, &comparison);

        // Should handle zero gracefully and show 0.00x speedup
        assert!(comparison_str.contains("0.00x"));
    }

    #[test]
    fn test_compare_benchmarks_zero_baseline_mean() {
        let baseline = BenchmarkResult {
            model_id: "test".to_string(),
            execution_provider: "broken".to_string(),
            iterations: 0,
            mean_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            median_ms: 0.0,
            p95_ms: 0.0,
            std_dev_ms: 0.0,
            total_time_ms: 0.0,
            iteration_times_ms: vec![],
        };

        let comparison = BenchmarkResult {
            model_id: "test".to_string(),
            execution_provider: "cpu".to_string(),
            iterations: 10,
            mean_ms: 100.0,
            min_ms: 90.0,
            max_ms: 110.0,
            median_ms: 100.0,
            p95_ms: 108.0,
            std_dev_ms: 5.0,
            total_time_ms: 1000.0,
            iteration_times_ms: vec![],
        };

        let comparison_str = compare_benchmarks(&baseline, &comparison);

        // Should handle zero baseline gracefully
        assert!(comparison_str.contains("0.0%"));
    }
}

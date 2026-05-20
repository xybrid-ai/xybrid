/// LLM streaming types for Xybrid.
///
/// This module provides types for streaming token generation.
library;

import 'result.dart';

/// Typed reason for a local stream abort that is eligible for cloud fallback.
enum CloudFallbackReason {
  userCancelled('user_cancelled'),
  stressThrottle('stress_throttle'),
  stressMemory('stress_memory'),
  stressThermal('stress_thermal'),
  stressCpuSustained('stress_cpu_sustained'),
  budgetExceeded('budget_exceeded');

  const CloudFallbackReason(this.wireName);

  /// Stable snake_case value used by the Rust SDK and telemetry.
  final String wireName;
}

/// A single token emitted during streaming generation.
class StreamToken {
  /// The generated token text.
  final String token;

  /// Zero-based index of this token in the generation sequence.
  final int index;

  /// Cumulative text generated so far.
  final String cumulativeText;

  /// Whether this is the final token.
  final bool isFinal;

  /// Finish reason if this is the final token (e.g., "stop", "length", "error").
  final String? finishReason;

  /// Final inference metrics. Present only on the completion token.
  final XybridInferenceMetrics? metrics;

  /// Reason for a local abort that crossed the stream as a cloud-fallback marker.
  final CloudFallbackReason? cloudFallbackReason;

  StreamToken({
    required this.token,
    required this.index,
    required this.cumulativeText,
    required this.isFinal,
    this.finishReason,
    this.metrics,
    this.cloudFallbackReason,
  });

  /// Check if this token marks a typed local-to-cloud fallback abort.
  bool get isCloudFallbackAbort => cloudFallbackReason != null;

  /// Check if this token represents an error.
  bool get isError =>
      finishReason != null && finishReason!.startsWith('error:');

  /// Get the error message if this is an error token.
  String? get errorMessage {
    if (isError && finishReason != null) {
      return finishReason!.replaceFirst('error: ', '');
    }
    return null;
  }
}

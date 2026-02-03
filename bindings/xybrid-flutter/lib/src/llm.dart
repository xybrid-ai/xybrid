/// LLM streaming types for Xybrid.
///
/// This module provides types for streaming token generation.
library;

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

  StreamToken({
    required this.token,
    required this.index,
    required this.cumulativeText,
    required this.isFinal,
    this.finishReason,
  });

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

/// LLM streaming types for Xybrid.
///
/// This module provides types for streaming token generation.
library;

import 'result.dart';
import 'tools.dart';

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

  /// Finish reason if this is the final token (e.g., "stop", "length",
  /// "tool_calls", "error").
  final String? finishReason;

  /// Final inference metrics. Present only on the completion token.
  final XybridInferenceMetrics? metrics;

  /// Tool calls the model asked for this turn — final token only.
  ///
  /// Tool-call blocks are suppressed from the streamed text, so there is
  /// nothing in [token] to parse: stop at this token, run the calls, then
  /// continue the turn by streaming a [XybridEnvelope.toolResults] envelope
  /// through the same method. Empty on every other token, and on turns that
  /// asked for no tool.
  final List<ToolCall> toolCalls;

  /// Whether the model asked to call at least one tool.
  bool get hasToolCalls => toolCalls.isNotEmpty;

  /// The completed turn's raw output text, tool-call block included — pass it
  /// as [XybridEnvelope.toolResults]'s `priorAssistantText` to continue the
  /// turn. Null unless [hasToolCalls] is true.
  ///
  /// Deliberately not [cumulativeText]: that is the text your UI painted,
  /// with the tool-call blocks suppressed.
  final String? rawText;

  StreamToken({
    required this.token,
    required this.index,
    required this.cumulativeText,
    required this.isFinal,
    this.finishReason,
    this.metrics,
    this.toolCalls = const [],
    this.rawText,
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

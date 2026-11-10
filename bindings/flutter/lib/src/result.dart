/// Inference result types for Xybrid.
///
/// This class wraps the FRB-generated [FfiRunResult] with a clean,
/// idiomatic Dart API.
library;

import 'dart:typed_data';

import 'rust/api/result.dart';
import 'tools.dart';
import 'observability.dart';
import 'decisions.dart';
import 'utils/audio.dart';

/// Per-stage latency entry for pipeline runs.
///
/// One entry per executed stage; the [stageId] matches the stage name in
/// the pipeline definition.
class XybridStageLatency {
  XybridStageLatency.fromFfi(FfiStageLatency inner)
      : stageId = inner.stageId,
        latencyMs = inner.latencyMs,
        observation = inner.observation;

  final String stageId;
  final int latencyMs;

  /// Inference reference of this pipeline stage, when tracing is available.
  final ObservationRef? observation;
}

/// Typed inference metrics surfaced on every [XybridResult].
///
/// LLM-specific fields ([ttftMs], [tokensPerSecond], [prefillTps],
/// [decodeTps], [tokensOut]) are `null` for ASR/TTS/embedding runs. For
/// pipeline runs they are parsed from the **final** stage envelope only,
/// so they are also `null` when the final stage isn't the LLM (e.g. an
/// `ASR → LLM → TTS` pipeline).
///
/// [stageLatenciesMs] is empty for `model.run()` and populated for
/// `pipeline.run()`.
class XybridInferenceMetrics {
  XybridInferenceMetrics.fromFfi(FfiInferenceMetrics inner)
      : totalMs = inner.totalMs,
        ttftMs = inner.ttftMs,
        tokensPerSecond = inner.tokensPerSecond,
        prefillTps = inner.prefillTps,
        decodeTps = inner.decodeTps,
        tokensOut = inner.tokensOut,
        imagePreprocessMs = inner.imagePreprocessMs,
        stageLatenciesMs = inner.stageLatenciesMs
            .map(XybridStageLatency.fromFfi)
            .toList(growable: false);

  /// Wall-clock latency in ms (mirrors [XybridResult.latencyMs]).
  final int totalMs;

  /// Time to first token, ms. LLM streaming only.
  final int? ttftMs;

  /// Generation throughput, tokens/sec. LLM only.
  final double? tokensPerSecond;

  /// Prefill phase tok/s. LLM only.
  final double? prefillTps;

  /// Decode phase tok/s. LLM only.
  final double? decodeTps;

  /// Completion tokens produced. LLM only.
  final int? tokensOut;

  /// Image preprocessing latency in ms. Vision-language runs only.
  final int? imagePreprocessMs;

  /// Per-stage wall-clock latencies. Empty for single-model runs.
  final List<XybridStageLatency> stageLatenciesMs;
}

/// Result of a model inference operation.
///
/// Access the output using the appropriate getter based on model type:
/// - [text] for ASR (speech-to-text) results
/// - [audioBytes] for TTS (text-to-speech) results
/// - [embedding] for embedding model results
/// - [decisions] for typed text/JSON decision answers
class XybridResult {
  /// The underlying FRB result.
  final FfiRunResult _inner;

  FfiResult? get _legacy => switch (_inner) {
        FfiRunResult_Legacy(:final field0) => field0,
        FfiRunResult_Decision() => null,
      };

  FfiDecisionResult? get _decision => switch (_inner) {
        FfiRunResult_Legacy() => null,
        FfiRunResult_Decision(:final field0) => field0,
      };

  /// Internal constructor from FRB result.
  /// @nodoc
  XybridResult.fromFfi(this._inner);

  /// Whether the inference completed successfully.
  bool get success => _legacy?.success ?? true;

  /// Trace reference for feedback/evaluations; null when untraced or suppressed.
  ObservationRef? get observation =>
      _legacy?.observation ?? _decision?.observation;

  /// Whether this answer came from the device or the cloud gateway.
  ///
  /// Cloud fallback — speculative or reactive — keeps the model id identical on
  /// both legs by design, so this is the only way to tell them apart.
  FfiExecutionTarget get executionTarget =>
      _legacy?.executionTarget ?? _decision!.executionTarget;

  /// Text output (for ASR models).
  ///
  /// Returns null if the model doesn't produce text output.
  String? get text => _legacy?.text;

  /// The model's chain-of-thought / reasoning, if it emitted any.
  ///
  /// Reasoning models wrap their scratchpad in `<think>` blocks. [text]
  /// always excludes those; this getter is where they surface, so you can
  /// show or hide the reasoning independently of the answer.
  ///
  /// Returns null when the model emitted no reasoning, or the backend
  /// doesn't surface one.
  String? get reasoningContent => _legacy?.reasoningContent;

  /// Audio bytes output (for TTS models).
  ///
  /// Returns raw PCM audio bytes (16-bit signed, little-endian).
  /// Use [audioAsWav] for playback-ready WAV format.
  ///
  /// Returns null if the model doesn't produce audio output.
  Uint8List? get audioBytes => _legacy?.audioBytes;

  /// Audio output as WAV format (for TTS models).
  ///
  /// Wraps the raw PCM bytes in a WAV header for easy playback.
  /// Default format is 24kHz mono (Kokoro TTS output).
  ///
  /// Example:
  /// ```dart
  /// final result = await model.run(envelope);
  /// final wavBytes = result.audioAsWav();
  /// // Play with just_audio or save as .wav file
  /// ```
  ///
  /// Returns null if the model doesn't produce audio output.
  Uint8List? audioAsWav({int sampleRate = 24000, int channels = 1}) {
    final bytes = audioBytes;
    if (bytes == null) return null;
    return wrapInWavHeader(bytes, sampleRate: sampleRate, channels: channels);
  }

  /// Embedding vector output (for embedding models).
  ///
  /// Returns null if the model doesn't produce embeddings.
  List<double>? get embedding => _legacy?.embedding?.toList();

  /// Tool calls the model asked for this turn.
  ///
  /// Empty unless the request offered tools via [GenerationConfig.tools]. Run
  /// each call yourself, then feed the outcomes back with
  /// [XybridEnvelope.toolResults] — one run is one model turn.
  ///
  /// The raw tool-call block stays in [text] untouched, and malformed model
  /// output yields an empty list rather than an error.
  late final List<ToolCall> toolCalls =
      _legacy?.toolCalls.map(ToolCall.fromFfi).toList(growable: false) ??
          const [];

  /// Ordered typed decisions, or null for legacy outputs.
  late final Decisions? decisions =
      _decision == null ? null : Decisions.fromFfi(_decision!);

  /// Total evaluated input tokens for a decision request.
  int? get tokensIn => _decision?.inputTokens.toInt();

  /// Decision models produce zero generated tokens.
  int? get tokensOut => _decision != null ? 0 : metrics.tokensOut;

  /// Whether the model asked to call at least one tool.
  bool get hasToolCalls => toolCalls.isNotEmpty;

  /// Inference latency in milliseconds.
  int get latencyMs => _legacy?.latencyMs ?? _decision!.latencyMs;

  /// Typed metrics for this run (TTFT, tok/s, per-stage latencies, etc.).
  ///
  /// LLM-specific fields are null for ASR/TTS/embedding runs;
  /// `stageLatenciesMs` is empty for single-model runs.
  late final XybridInferenceMetrics metrics = XybridInferenceMetrics.fromFfi(
    _legacy?.metrics ?? _decision!.metrics,
  );
}

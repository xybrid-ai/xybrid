/// Configuration for LLM text generation parameters.
///
/// All fields are optional. When `null`, the model's default value is used.
///
/// ## Presets
///
/// ```dart
/// // Deterministic output (temperature = 0)
/// GenerationConfig.greedy()
///
/// // Creative output (higher temperature and top_p)
/// GenerationConfig.creative()
///
/// // Custom
/// GenerationConfig(maxTokens: 512, temperature: 0.5)
/// ```
library;

import 'rust/api/model.dart';

/// Generation parameters for LLM inference.
///
/// Controls how the model generates text. All fields are optional —
/// when `null`, the model's built-in default is used.
class GenerationConfig {
  /// Maximum tokens to generate. Default (when null): 2048.
  final int? maxTokens;

  /// Sampling temperature. Default (when null): 0.7.
  /// 0.0 = deterministic, higher = more random.
  final double? temperature;

  /// Top-p (nucleus) sampling threshold. Default (when null): 0.9.
  final double? topP;

  /// Min-p sampling threshold. Default (when null): 0.05.
  /// Prunes tokens with probability below `minP * maxProbability`.
  final double? minP;

  /// Top-k sampling (0 = disabled). Default (when null): 40.
  final int? topK;

  /// Repetition penalty (1.0 = disabled). Default (when null): 1.1.
  final double? repetitionPenalty;

  /// Stop sequences. When null or empty, only EOS token stops generation.
  final List<String>? stopSequences;

  /// Create a custom generation config.
  ///
  /// Only set fields you want to override — `null` fields use model defaults.
  const GenerationConfig({
    this.maxTokens,
    this.temperature,
    this.topP,
    this.minP,
    this.topK,
    this.repetitionPenalty,
    this.stopSequences,
  });

  /// Greedy decoding preset (deterministic, temperature=0).
  ///
  /// Produces the same output every time for the same input.
  const GenerationConfig.greedy()
      : maxTokens = null,
        temperature = 0.0,
        topP = 1.0,
        minP = null,
        topK = 0,
        repetitionPenalty = null,
        stopSequences = null;

  /// Creative generation preset (higher temperature).
  ///
  /// Produces more varied and creative output.
  const GenerationConfig.creative()
      : maxTokens = null,
        temperature = 0.9,
        topP = 0.95,
        minP = null,
        topK = 50,
        repetitionPenalty = null,
        stopSequences = null;

  /// Convert to the FRB-generated FFI type.
  FfiGenerationConfig toFfi() {
    return FfiGenerationConfig(
      maxTokens: maxTokens,
      temperature: temperature,
      topP: topP,
      minP: minP,
      topK: topK,
      repetitionPenalty: repetitionPenalty,
      stopSequences: stopSequences,
    );
  }
}

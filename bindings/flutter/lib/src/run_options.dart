library;

import 'rust/api/model.dart';
import 'runtime_config.dart';
import 'observability.dart';

/// Resource or cancellation signals that can stop a local run.
enum AbortSignal {
  memoryPressureCritical,
  thermalCritical,
}

/// Per-run abort behavior exposed to Flutter callers.
class AbortPolicy {
  final Set<AbortSignal> stopOn;
  final bool fallbackToCloud;
  final int? maxGraceTokens;

  const AbortPolicy({
    this.stopOn = const {
      AbortSignal.memoryPressureCritical,
      AbortSignal.thermalCritical,
    },
    this.fallbackToCloud = false,
    this.maxGraceTokens,
  });

  const AbortPolicy.cloudFallback({
    this.stopOn = const {
      AbortSignal.memoryPressureCritical,
      AbortSignal.thermalCritical,
    },
    this.maxGraceTokens,
  }) : fallbackToCloud = true;
}

/// Per-run execution and observation controls. Cloud provider/model/gateway
/// overrides and fallback permission apply to `runStreamingWithFallback`.
class RunOptions {
  final AbortPolicy abortPolicy;
  final ObservationOptions? observation;
  final String? cloudProvider;
  final String? cloudModel;
  final String? cloudGatewayUrl;
  final String? correlationId;
  final int? maxGraceTokens;

  const RunOptions({
    this.abortPolicy = const AbortPolicy(),
    this.observation,
    this.cloudProvider,
    this.cloudModel,
    this.cloudGatewayUrl,
    this.correlationId,
    this.maxGraceTokens,
  });

  /// Enable local streaming abort with authenticated cloud fallback.
  const RunOptions.cloudFallback({
    this.abortPolicy = const AbortPolicy.cloudFallback(),
    this.observation,
    this.cloudProvider,
    this.cloudModel,
    this.cloudGatewayUrl,
    this.correlationId,
    this.maxGraceTokens,
  });

  FfiRunOptions toFfi() {
    final graceTokens = abortPolicy.maxGraceTokens ?? maxGraceTokens;
    return FfiRunOptions(
      traceparent: observation?.traceparent,
      tracestate: observation?.tracestate,
      conversationId: observation?.conversationId,
      suppressTrace: observation?.suppressTrace ?? false,
      cloudProvider: cloudProvider,
      cloudModel: cloudModel,
      cloudGatewayUrl: cloudGatewayUrl ?? XybridRuntimeConfig.gatewayUrl,
      correlationId: correlationId,
      abortOnMemoryPressureCritical:
          abortPolicy.stopOn.contains(AbortSignal.memoryPressureCritical),
      abortOnThermalCritical:
          abortPolicy.stopOn.contains(AbortSignal.thermalCritical),
      fallbackToCloud: abortPolicy.fallbackToCloud,
      maxGraceTokens: graceTokens,
    );
  }
}

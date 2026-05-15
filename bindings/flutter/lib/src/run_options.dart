library;

import 'rust/api/model.dart';
import 'runtime_config.dart';

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

/// Per-run controls for SDK execution.
class RunOptions {
  final AbortPolicy abortPolicy;
  final String? cloudProvider;
  final String? cloudModel;
  final String? cloudGatewayUrl;
  final String? correlationId;
  final int? maxGraceTokens;

  const RunOptions({
    this.abortPolicy = const AbortPolicy(),
    this.cloudProvider,
    this.cloudModel,
    this.cloudGatewayUrl,
    this.correlationId,
    this.maxGraceTokens,
  });

  /// Enable local streaming abort with authenticated cloud fallback.
  const RunOptions.cloudFallback({
    this.abortPolicy = const AbortPolicy.cloudFallback(),
    this.cloudProvider,
    this.cloudModel,
    this.cloudGatewayUrl,
    this.correlationId,
    this.maxGraceTokens,
  });

  FfiRunOptions toFfi() {
    final graceTokens = abortPolicy.maxGraceTokens ?? maxGraceTokens;
    return FfiRunOptions(
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

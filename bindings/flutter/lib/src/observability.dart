/// Explicit OpenTelemetry export and run correlation.
library;

import 'rust/api/observability.dart' as ffi;
import 'rust/api/manual_observation.dart' as manual;

export 'rust/api/observability.dart' show FfiRedactionRule;

/// Trace and span identifiers returned by a traced inference.
typedef ObservationRef = ffi.FfiObservationRef;

/// Manual native handles. Dispose unfinished handles to cancel without flushing.
typedef WorkflowObservation = manual.FfiWorkflowObservation;
typedef ToolObservation = manual.FfiToolObservation;
typedef InferenceObservation = manual.FfiInferenceObservation;
typedef ModelLoadObservation = manual.FfiModelLoadObservation;
typedef ObservationContext = manual.FfiObservationContext;
typedef InferenceRequest = manual.FfiInferenceRequest;
typedef ToolCallInfo = manual.FfiToolCallInfo;
typedef ModelLoadInfo = manual.FfiModelLoadInfo;
typedef ObservationUsage = manual.FfiUsage;
typedef ExecutionLocation = manual.FfiExecutionLocation;

/// Native exporter status and delivery counters.
typedef ObservabilityStats = ffi.FfiObservabilityStats;

/// Export configuration. Xybrid sends traces only; a Collector receives
/// traces, metrics and logs. Prompts/outputs are captured by default unless
/// [metadataOnly] is true (or disabled through the native environment policy).
class ObservabilityConfig {
  final String? apiKey;
  final String? endpoint;
  final String? tracesEndpoint;
  final Map<String, String> headers;
  final bool jsonProtocol;
  final String? serviceName;
  final String? serviceVersion;
  final Map<String, String> resourceAttributes;
  final bool metadataOnly;
  final int? maxContentBytes;
  final int? liveTraceIntervalMs;
  final String? deviceId;
  final List<ffi.FfiRedactionRule> redactionRules;
  final List<String> providerOrigins;

  const ObservabilityConfig.xybrid({
    required String this.apiKey,
    this.tracesEndpoint,
    this.headers = const {},
    this.jsonProtocol = false,
    this.serviceName,
    this.serviceVersion,
    this.resourceAttributes = const {},
    this.metadataOnly = false,
    this.maxContentBytes,
    this.liveTraceIntervalMs,
    this.deviceId,
    this.redactionRules = const [],
    this.providerOrigins = const [],
  }) : endpoint = null;

  const ObservabilityConfig.otlp({
    required String this.endpoint,
    this.tracesEndpoint,
    this.headers = const {},
    this.jsonProtocol = false,
    this.serviceName,
    this.serviceVersion,
    this.resourceAttributes = const {},
    this.metadataOnly = false,
    this.maxContentBytes,
    this.liveTraceIntervalMs,
    this.deviceId,
    this.redactionRules = const [],
    this.providerOrigins = const [],
  }) : apiKey = null;

  /// @nodoc
  ffi.FfiObservabilityConfig toFfi() => ffi.FfiObservabilityConfig(
        apiKey: apiKey,
        endpoint: endpoint,
        tracesEndpoint: tracesEndpoint,
        headers: headers.entries.map((e) => (e.key, e.value)).toList(),
        jsonProtocol: jsonProtocol,
        serviceName: serviceName,
        serviceVersion: serviceVersion,
        resourceAttributes:
            resourceAttributes.entries.map((e) => (e.key, e.value)).toList(),
        metadataOnly: metadataOnly,
        maxContentBytes: maxContentBytes,
        liveTraceIntervalMs: liveTraceIntervalMs,
        deviceId: deviceId,
        redactionRules: redactionRules,
        providerOrigins: providerOrigins,
      );
}

/// Observation controls. Invalid explicit [traceparent] values are rejected.
/// Suppressing the detailed
/// trace still records aggregate metrics.
class ObservationOptions {
  final String? traceparent;
  final String? tracestate;
  final String? conversationId;
  final bool suppressTrace;

  const ObservationOptions({
    this.traceparent,
    this.tracestate,
    this.conversationId,
    this.suppressTrace = false,
  });

  /// @nodoc
  ffi.FfiObservationOptions toFfi() => ffi.FfiObservationOptions(
        traceparent: traceparent,
        tracestate: tracestate,
        conversationId: conversationId,
        suppressTrace: suppressTrace,
      );
}

/// Native observer control plane. Call after `Xybrid.init()`. All export runs
/// in Rust; Dart supplies configuration and W3C context, never span batches.
class XybridObservability {
  XybridObservability._();

  static Future<WorkflowObservation> startWorkflow(String name,
          {ObservationOptions? options}) =>
      WorkflowObservation.newInstance(name: name, options: options?.toFfi());

  static Future<ToolObservation> startTool(ToolCallInfo call,
          {ObservationOptions? options}) =>
      ToolObservation.newInstance(call: call, options: options?.toFfi());

  static Future<InferenceObservation> startInference(InferenceRequest request,
          {ObservationOptions? options}) =>
      InferenceObservation.newInstance(
          request: request, options: options?.toFfi());

  static Future<ModelLoadObservation> startModelLoad(ModelLoadInfo info,
          {ObservationOptions? options}) =>
      ModelLoadObservation.newInstance(info: info, options: options?.toFfi());

  /// Start exporting or replace the active configuration. Errors propagate;
  /// a corrected configuration may be submitted after an unsuccessful init.
  static void init(ObservabilityConfig config) =>
      ffi.initObservability(config: config.toFfi());

  /// Delivery counters: live while export runs; after [shutdown], the final
  /// ones (what the shutdown failed or dropped included) until the next
  /// [init]; all zero before any.
  static ObservabilityStats get stats => ffi.observabilityStats();

  /// Export queued records without blocking the Dart UI isolate.
  static Future<void> flush({int timeoutMs = 5000}) =>
      ffi.flushObservability(timeoutMs: timeoutMs);

  /// Flush and stop exporting. Repeated shutdown is safe; init can restart it.
  static Future<void> shutdown({int timeoutMs = 5000}) =>
      ffi.shutdownObservability(timeoutMs: timeoutMs);

  /// Returns false when logs are disabled (including Xybrid's traces preset).
  static bool recordFeedback(ObservationRef target,
          {double? score, String? label, String? comment}) =>
      ffi.recordFeedback(
          target: target, score: score, label: label, comment: comment);

  /// Returns false when logs are disabled (including Xybrid's traces preset).
  static bool recordEvaluation(ObservationRef target,
          {required String name,
          double? scoreValue,
          String? scoreLabel,
          String? explanation}) =>
      ffi.recordEvaluation(
          target: target,
          name: name,
          scoreValue: scoreValue,
          scoreLabel: scoreLabel,
          explanation: explanation);
}

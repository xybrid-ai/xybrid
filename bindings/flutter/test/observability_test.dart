import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/observability.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

class _ObserverApi implements XybridRustLibApi {
  FfiObservabilityConfig? config;
  String? apiKey;
  String? legacyIngest;
  String? resourceMode;
  bool enabled = false;
  bool reject = false;
  int? timeout;
  Completer<void>? flushPending;
  FfiObservationRef? target;
  String? evaluation;

  @override
  void crateApiObservabilityInitObservability(
      {required FfiObservabilityConfig config}) {
    if (reject) throw StateError('rejected config');
    this.config = config;
    enabled = true;
  }

  @override
  FfiObservabilityStats crateApiObservabilityObservabilityStats() =>
      FfiObservabilityStats(
        enabled: enabled,
        spansExported: BigInt.zero,
        spansDropped: BigInt.zero,
        exportFailures: BigInt.zero,
      );

  @override
  Future<void> crateApiObservabilityFlushObservability(
      {required int timeoutMs}) {
    timeout = timeoutMs;
    return flushPending?.future ?? Future.value();
  }

  @override
  Future<void> crateApiObservabilityShutdownObservability(
      {required int timeoutMs}) async {
    timeout = timeoutMs;
    enabled = false;
  }

  @override
  bool crateApiObservabilityRecordFeedback(
      {required FfiObservationRef target,
      double? score,
      String? label,
      String? comment}) {
    this.target = target;
    return false; // traces-only backend: logs unavailable
  }

  @override
  bool crateApiObservabilityRecordEvaluation(
      {required FfiObservationRef target,
      required String name,
      double? scoreValue,
      String? scoreLabel,
      String? explanation}) {
    this.target = target;
    evaluation = name;
    return true;
  }

  @override
  void crateApiSdkClientXybridSdkClientSetApiKey({required String apiKey}) =>
      this.apiKey = apiKey;

  @override
  void crateApiSdkClientXybridSdkClientConfigurePlatformTelemetry(
      {required String apiKey, String? ingestUrl, String? resourceTelemetry}) {
    this.apiKey = apiKey;
    legacyIngest = ingestUrl;
    enabled = true;
  }

  @override
  void crateApiSdkClientXybridSdkClientValidateResourceTelemetry(
      {String? mode}) {
    if (mode == 'invalid') throw ArgumentError('invalid resource mode');
  }

  @override
  void crateApiSdkClientXybridSdkClientConfigureResourceTelemetry(
          {String? mode}) =>
      resourceMode = mode;

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  final api = _ObserverApi();
  setUpAll(() => XybridRustLib.initMock(api: api));
  tearDownAll(XybridRustLib.dispose);
  setUp(() {
    api.config = null;
    api.enabled = false;
    api.reject = false;
    api.legacyIngest = null;
    api.resourceMode = null;
    api.flushPending = null;
  });

  test(
      'API key authenticates without starting export; ingest is explicit opt-in',
      () async {
    await Xybrid.init(apiKey: 'xy_auth');
    expect(api.apiKey, 'xy_auth');
    expect(api.config, isNull);
    expect(api.enabled, isFalse);
    await Xybrid.init(apiKey: 'xy_auth', ingestUrl: 'http://collector:4318');
    expect(api.legacyIngest, 'http://collector:4318');
    expect(api.enabled, isTrue);
  });

  test('resource sampling is independent of export', () async {
    await Xybrid.init(resourceTelemetry: 'boundary');
    expect(api.resourceMode, 'boundary');
    expect(api.enabled, isFalse);
  });

  test('invalid resource mode cannot enable export', () async {
    await expectLater(
        Xybrid.init(
            resourceTelemetry: 'invalid',
            observability: const ObservabilityConfig.xybrid(apiKey: 'xy_test')),
        throwsArgumentError);
    expect(api.config, isNull);
    expect(api.enabled, isFalse);
    await expectLater(
        Xybrid.init(
            resourceTelemetry: 'invalid',
            apiKey: 'xy_test',
            ingestUrl: 'http://collector:4318'),
        throwsArgumentError);
    expect(api.legacyIngest, isNull);
    expect(api.enabled, isFalse);
  });

  test('config forwards collector headers, content policy and resource fields',
      () {
    XybridObservability.init(const ObservabilityConfig.otlp(
      endpoint: 'http://collector:4318',
      tracesEndpoint: 'http://traces/custom',
      headers: {'Authorization': 'Bearer example'},
      jsonProtocol: true,
      serviceName: 'chat',
      serviceVersion: '1.0',
      resourceAttributes: {'deployment.environment': 'test'},
      metadataOnly: true,
      maxContentBytes: 123,
      liveTraceIntervalMs: 456,
    ));
    final config = api.config!;
    expect(config.apiKey, isNull);
    expect(config.endpoint, 'http://collector:4318');
    expect(config.tracesEndpoint, 'http://traces/custom');
    expect(config.headers.single, ('Authorization', 'Bearer example'));
    expect(
        config.resourceAttributes.single, ('deployment.environment', 'test'));
    expect(config.serviceName, 'chat');
    expect(config.serviceVersion, '1.0');
    expect(config.jsonProtocol, isTrue);
    expect(config.metadataOnly, isTrue);
    expect(config.maxContentBytes, 123);
    expect(config.liveTraceIntervalMs, 456);
  });

  test(
      'failed Xybrid init config remains retryable and shutdown allows restart',
      () async {
    api.reject = true;
    await expectLater(
        Xybrid.init(
            observability: const ObservabilityConfig.xybrid(apiKey: 'xy_test')),
        throwsStateError);
    api.reject = false;
    await Xybrid.init(
        observability: const ObservabilityConfig.xybrid(
            apiKey: 'xy_test', metadataOnly: true));
    expect(api.config!.apiKey, 'xy_test');
    expect(XybridObservability.stats.enabled, isTrue);
    await XybridObservability.shutdown(timeoutMs: 321);
    expect(api.timeout, 321);
    expect(XybridObservability.stats.enabled, isFalse);
    XybridObservability.init(
        const ObservabilityConfig.otlp(endpoint: 'http://collector:4318'));
    expect(XybridObservability.stats.enabled, isTrue);
  });

  test('flush awaits native completion and propagates its timeout', () async {
    api.flushPending = Completer<void>();
    var complete = false;
    final flush = XybridObservability.flush(timeoutMs: 234);
    unawaited(flush.then((_) => complete = true));
    await Future<void>.delayed(Duration.zero);
    expect(api.timeout, 234);
    expect(complete, isFalse);
    api.flushPending!.complete();
    await flush;
  });

  test('feedback and evaluation retain the target and native acceptance result',
      () {
    const ref = ObservationRef(
        traceId: '0af7651916cd43dd8448eb211c80319c',
        spanId: 'b7ad6b7169203331');
    expect(XybridObservability.recordFeedback(ref, score: 1.0), isFalse);
    expect(api.target, ref);
    expect(
        XybridObservability.recordEvaluation(ref,
            name: 'quality', scoreValue: 0.8),
        isTrue);
    expect(api.evaluation, 'quality');
  });
}

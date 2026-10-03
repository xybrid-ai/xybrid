import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/envelope.dart';
import 'package:xybrid_flutter/src/rust/api/model.dart';
import 'package:xybrid_flutter/src/rust/api/result.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

/// What the native side sends when a run's output has no `FfiResult` field.
const _unsupported = 'Unsupported model capability: a multi-part message '
    'cannot be returned through the Flutter bindings yet';

/// Stands in for the native library: a registry load yields a [_Model] whose
/// streaming runs replay [events].
class _StreamApi implements XybridRustLibApi {
  /// The native events every streaming run replays.
  List<FfiStreamEvent> events = const [];

  @override
  FfiModelLoader crateApiModelFfiModelLoaderFromRegistry({
    required String modelId,
  }) =>
      _Loader(_Model(this));

  @override
  FfiEnvelope crateApiEnvelopeFfiEnvelopeText({
    required String text,
    String? voiceId,
    double? speed,
  }) =>
      _Envelope();

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Loader implements FfiModelLoader {
  _Loader(this._model);

  final FfiModel _model;

  @override
  Future<FfiModel> load() async => _model;

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Model implements FfiModel {
  _Model(this._api);

  final _StreamApi _api;

  @override
  Stream<FfiStreamEvent> runStream({
    required FfiEnvelope envelope,
    FfiGenerationConfig? config,
    FfiCancellationToken? cancellationToken,
    required bool preempt,
    String? frameSessionId,
  }) =>
      Stream.fromIterable(_api.events);

  @override
  Stream<FfiStreamEvent> runStreamWithFallback({
    required FfiEnvelope envelope,
    required FfiRunOptions options,
    FfiGenerationConfig? config,
    FfiCancellationToken? cancellationToken,
  }) =>
      Stream.fromIterable(_api.events);

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Envelope implements FfiEnvelope {
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

FfiStreamEvent _finalToken(String text) => FfiStreamEvent.token(
      FfiStreamToken(
        token: text,
        index: 0,
        cumulativeText: text,
        finishReason: 'stop',
        toolCalls: const [],
      ),
    );

FfiStreamEvent _complete(String text) => FfiStreamEvent.complete(
      FfiResult(
        success: true,
        text: text,
        latencyMs: 10,
        executionTarget: FfiExecutionTarget.local,
        metrics: const FfiInferenceMetrics(totalMs: 10, stageLatenciesMs: []),
        toolCalls: const [],
      ),
    );

void main() {
  final api = _StreamApi();
  late XybridModel model;

  setUpAll(() async {
    XybridRustLib.initMock(api: api);
    model = await XybridModelLoader.fromRegistry('test-model').load();
  });
  tearDownAll(XybridRustLib.dispose);

  final runs = <String, Stream<StreamToken> Function()>{
    'runStreaming': () => model.runStreaming(XybridEnvelope.text('Hi')),
    'runStreamingWithFallback': () => model.runStreamingWithFallback(
          XybridEnvelope.text('Hi'),
          options: const RunOptions(),
        ),
  };

  for (final MapEntry(key: method, value: run) in runs.entries) {
    group(method, () {
      test('an error after the final token still ends the stream', () async {
        api.events = [
          _finalToken('Hello'),
          const FfiStreamEvent.error(_unsupported),
        ];

        final tokens = await run().toList();

        expect(
          tokens.map((token) => token.finishReason),
          ['stop', 'error: $_unsupported'],
        );
        expect(tokens.last.isError, isTrue);
        expect(tokens.last.errorMessage, _unsupported);
      });

      test('a completion after the final token adds no token', () async {
        api.events = [_finalToken('Hello'), _complete('Hello')];

        final tokens = await run().toList();

        expect(tokens.single.finishReason, 'stop');
        expect(tokens.single.cumulativeText, 'Hello');
      });
    });
  }
}

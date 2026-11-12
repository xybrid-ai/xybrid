import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/context.dart';
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
  Object? streamError;
  FfiRunOptions? options;

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
  FfiConversationContext crateApiContextFfiConversationContextCreate() =>
      _Context();

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

  Stream<FfiStreamEvent> _stream() async* {
    yield* Stream.fromIterable(_api.events);
    final error = _api.streamError;
    if (error != null) throw error;
  }

  @override
  Future<FfiRunResult> run(
      {required FfiEnvelope envelope,
      FfiGenerationConfig? config,
      FfiCancellationToken? cancellationToken,
      FfiRunOptions? options}) async {
    _api.options = options;
    return FfiRunResult.legacy(FfiResult(
        success: true,
        text: 'Hello',
        latencyMs: 10,
        executionTarget: FfiExecutionTarget.local,
        metrics: const FfiInferenceMetrics(totalMs: 10, stageLatenciesMs: []),
        toolCalls: const []));
  }

  @override
  Future<FfiRunResult> runWithContext(
          {required FfiEnvelope envelope,
          required FfiConversationContext context,
          FfiGenerationConfig? config,
          FfiRunOptions? options}) =>
      run(envelope: envelope, config: config, options: options);

  @override
  Stream<FfiTtsStreamEvent> runTtsStream(
      {required FfiEnvelope envelope,
      FfiGenerationConfig? config,
      FfiCancellationToken? cancellationToken,
      FfiRunOptions? options}) {
    _api.options = options;
    return const Stream.empty();
  }

  @override
  Stream<FfiStreamEvent> runStream({
    required FfiEnvelope envelope,
    FfiGenerationConfig? config,
    FfiCancellationToken? cancellationToken,
    required bool preempt,
    String? frameSessionId,
    FfiRunOptions? options,
  }) {
    _api.options = options;
    return _stream();
  }

  @override
  Stream<FfiStreamEvent> runStreamWithContext({
    required FfiEnvelope envelope,
    required FfiConversationContext context,
    FfiGenerationConfig? config,
    FfiCancellationToken? cancellationToken,
    required bool preempt,
    String? frameSessionId,
    FfiRunOptions? options,
  }) {
    _api.options = options;
    return _stream();
  }

  @override
  Stream<FfiStreamEvent> runStreamWithFallback({
    required FfiEnvelope envelope,
    required FfiRunOptions options,
    FfiGenerationConfig? config,
    FfiCancellationToken? cancellationToken,
  }) {
    _api.options = options;
    return _stream();
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Envelope implements FfiEnvelope {
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Context implements FfiConversationContext {
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

FfiStreamEvent _complete(String text,
        {ObservationRef? observation,
        List<FfiToolCall> toolCalls = const []}) =>
    FfiStreamEvent.complete(
      FfiResult(
        success: true,
        observation: observation,
        text: text,
        latencyMs: 10,
        executionTarget: FfiExecutionTarget.local,
        metrics: const FfiInferenceMetrics(totalMs: 10, stageLatenciesMs: []),
        toolCalls: toolCalls,
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

  setUp(() {
    api.events = const [];
    api.streamError = null;
  });

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

      test('traced completion retains its reference after a native final token',
          () async {
        const reference = ObservationRef(
            traceId: '0af7651916cd43dd8448eb211c80319c',
            spanId: 'b7ad6b7169203331');
        const call =
            FfiToolCall(id: 'call-1', name: 'weather', argumentsJson: '{}');
        api.events = [
          const FfiStreamEvent.token(FfiStreamToken(
              token: '',
              index: 0,
              cumulativeText: '',
              finishReason: 'tool_calls',
              toolCalls: [call])),
          _complete('', observation: reference, toolCalls: [call]),
        ];
        final tokens = await run().toList();
        expect(tokens, hasLength(1));
        expect(tokens.single.finishReason, 'tool_calls');
        expect(tokens.last.observation, reference);
        expect(tokens.last.metrics, isNotNull);
        expect(tokens.expand((token) => token.toolCalls), hasLength(1));
      });

      test(
          'traced length completion enriches exactly one native terminal token',
          () async {
        const reference = ObservationRef(
            traceId: '0af7651916cd43dd8448eb211c80319c',
            spanId: 'b7ad6b7169203331');
        api.events = [
          const FfiStreamEvent.token(FfiStreamToken(
              token: 'Hello',
              index: 4,
              cumulativeText: 'Hello',
              toolCalls: [])),
          const FfiStreamEvent.token(FfiStreamToken(
              token: '!',
              index: 5,
              cumulativeText: 'Hello!',
              finishReason: 'length',
              toolCalls: [])),
          _complete('Hello!', observation: reference),
        ];
        final tokens = await run().toList();
        expect(tokens, hasLength(2));
        expect(tokens.where((token) => token.isFinal), hasLength(1));
        final terminal = tokens.last;
        expect(terminal.finishReason, 'length');
        expect(terminal.index, 5);
        expect(terminal.token, '!');
        expect(terminal.cumulativeText, 'Hello!');
        expect(terminal.observation, reference);
        expect(terminal.metrics!.totalMs, 10);
      });

      test('terminal token survives a close without a completion event',
          () async {
        api.events = [_finalToken('Hello')];
        expect((await run().toList()).single.finishReason, 'stop');
      });

      test('bridge error after a buffered terminal token is surfaced',
          () async {
        api.events = [_finalToken('Hello')];
        api.streamError = _unsupported;
        final tokens = await run().toList();
        expect(tokens.map((token) => token.finishReason),
            ['stop', 'error: $_unsupported']);
      });

      test('bridge stream errors become error tokens', () async {
        api.streamError = _unsupported;

        final tokens = await run().toList();

        expect(tokens.single.isFinal, isTrue);
        expect(tokens.single.isError, isTrue);
        expect(tokens.single.errorMessage, _unsupported);
      });
    });
  }

  test(
      'chunked TTS forwards observation options without changing its audio shape',
      () async {
    const options = RunOptions(
        observation:
            ObservationOptions(conversationId: 'tts-1', suppressTrace: true));
    final chunks = await model
        .runTtsStreaming(XybridEnvelope.text('Hi'), options: options)
        .toList();
    expect(chunks, isEmpty);
    expect(api.options!.conversationId, 'tts-1');
    expect(api.options!.suppressTrace, isTrue);
  });

  test('batch and context batch forward observation options', () async {
    const options = RunOptions(
        observation: ObservationOptions(
            conversationId: 'conversation-1', suppressTrace: true));
    await model.run(XybridEnvelope.text('Hi'), options: options);
    expect(api.options!.conversationId, 'conversation-1');
    expect(api.options!.suppressTrace, isTrue);
    await model.runWithContext(XybridEnvelope.text('Hi'), ConversationContext(),
        options: options);
    expect(api.options!.conversationId, 'conversation-1');
  });

  test('normal and context streaming forward observation options', () async {
    const options = RunOptions(
        observation: ObservationOptions(
            conversationId: 'conversation-1', suppressTrace: true));
    api.events = [_complete('Hello')];
    await model
        .runStreaming(XybridEnvelope.text('Hi'), options: options)
        .toList();
    expect(api.options!.conversationId, 'conversation-1');
    expect(api.options!.suppressTrace, isTrue);
    await model
        .runStreamingWithContext(
            XybridEnvelope.text('Hi'), ConversationContext(),
            options: options)
        .toList();
    expect(api.options!.conversationId, 'conversation-1');
  });

  group('runStreamingWithContext', () {
    Stream<StreamToken> run() => model.runStreamingWithContext(
          XybridEnvelope.text('Hi'),
          ConversationContext(),
        );

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

    test('bridge stream errors become error tokens', () async {
      api.streamError = _unsupported;

      final tokens = await run().toList();

      expect(tokens.single.isFinal, isTrue);
      expect(tokens.single.isError, isTrue);
      expect(tokens.single.errorMessage, _unsupported);
    });
  });
}

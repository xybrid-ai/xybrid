import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/context.dart';
import 'package:xybrid_flutter/src/rust/api/envelope.dart';
import 'package:xybrid_flutter/src/rust/api/model.dart';
import 'package:xybrid_flutter/src/rust/api/result.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _loadDiagnostic = 'Failed to load model: failed to create ONNX session: '
    'model.onnx: unsupported opset version 23';
const _runDiagnostic =
    'Inference failed: adapter error: unsupported audio input';
const _contextDiagnostic =
    'Inference with context failed: adapter error: context window exceeded';

/// Replaces the native library with handles that replay native failures.
class _ErrorApi implements XybridRustLibApi {
  Object? loadError;
  Object runError = _runDiagnostic;
  List<FfiLoadEvent> loadEvents = const [];

  @override
  FfiModelLoader crateApiModelFfiModelLoaderFromRegistry({
    required String modelId,
  }) =>
      _Loader(this);

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
  _Loader(this._api);

  final _ErrorApi _api;

  @override
  Future<FfiModel> load() async {
    final error = _api.loadError;
    if (error != null) throw error;
    return _Model(_api);
  }

  @override
  Stream<FfiLoadEvent> loadWithProgress() =>
      Stream.fromIterable(_api.loadEvents);

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Model implements FfiModel {
  _Model(this._api);

  final _ErrorApi _api;

  @override
  Future<FfiResult> run({
    required FfiEnvelope envelope,
    FfiGenerationConfig? config,
  }) async =>
      throw _api.runError;

  @override
  Future<FfiResult> runWithContext({
    required FfiEnvelope envelope,
    required FfiConversationContext context,
    FfiGenerationConfig? config,
  }) async =>
      throw _api.runError;

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

void main() {
  final api = _ErrorApi();
  late XybridModelLoader loader;
  late XybridModel model;

  setUpAll(() => XybridRustLib.initMock(api: api));
  tearDownAll(XybridRustLib.dispose);

  setUp(() async {
    api.loadError = null;
    api.runError = _runDiagnostic;
    api.loadEvents = const [];
    loader = XybridModelLoader.fromRegistry('test-model');
    model = await loader.load();
  });

  group('load', () {
    test('wraps the native diagnostic in XybridException', () async {
      // FRB decodes Result<T, String> errors as Dart strings.
      api.loadError = _loadDiagnostic;

      await expectLater(
        loader.load(),
        throwsA(isA<XybridException>().having(
          (error) => error.message,
          'message',
          _loadDiagnostic,
        )),
      );
    });

    test('preserves an existing XybridException', () async {
      final error = XybridException(_loadDiagnostic);
      api.loadError = error;

      await expectLater(loader.load(), throwsA(same(error)));
    });
  });

  final runs = <String, Future<XybridResult> Function()>{
    'run': () => model.run(XybridEnvelope.text('Hi')),
    'runWithContext': () => model.runWithContext(
          XybridEnvelope.text('Hi'),
          ConversationContext(),
        ),
  };

  for (final MapEntry(key: method, value: run) in runs.entries) {
    group(method, () {
      test('preserves the native diagnostic without duplicate prefixes',
          () async {
        final diagnostic =
            method == 'run' ? _runDiagnostic : _contextDiagnostic;
        api.runError = diagnostic;

        await expectLater(
          run(),
          throwsA(isA<XybridException>().having(
            (error) => error.message,
            'message',
            diagnostic,
          )),
        );
      });

      test('preserves an existing XybridException', () async {
        final error = XybridException(_runDiagnostic);
        api.runError = error;

        await expectLater(run(), throwsA(same(error)));
      });
    });
  }

  test('loadWithProgress reports the native diagnostic as a LoadError event',
      () async {
    api.loadEvents = const [FfiLoadEvent.error(_loadDiagnostic)];

    final events = await loader.loadWithProgress().toList();

    expect(
      events.single,
      isA<LoadError>().having(
        (event) => event.message,
        'message',
        _loadDiagnostic,
      ),
    );
  });
}

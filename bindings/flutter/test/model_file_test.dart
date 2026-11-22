import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/model.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _path = '/data/models/model.gguf';
const _missing = 'Invalid configuration: GGUF file not found: $_path';

/// Replaces the native library with a model-file loader that records the path
/// it was given and replays a scripted load.
class _ModelFileApi implements XybridRustLibApi {
  String? path;
  Object? loadError;
  List<FfiLoadEvent> loadEvents = const [];

  @override
  FfiModelLoader crateApiModelFfiModelLoaderFromModelFile({
    required String path,
  }) {
    this.path = path;
    return _Loader(this);
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Loader implements FfiModelLoader {
  _Loader(this._api);

  final _ModelFileApi _api;

  @override
  Future<FfiModel> load() async {
    final error = _api.loadError;
    if (error != null) throw error;
    return _Model();
  }

  @override
  Stream<FfiLoadEvent> loadWithProgress() =>
      Stream.fromIterable(_api.loadEvents);

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Model implements FfiModel {
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  final api = _ModelFileApi();

  setUpAll(() => XybridRustLib.initMock(api: api));
  tearDownAll(XybridRustLib.dispose);

  setUp(() {
    api.path = null;
    api.loadError = null;
    api.loadEvents = const [];
  });

  test('fromModelFile passes the path through unchanged and loads', () async {
    final loader = XybridModelLoader.fromModelFile(_path);

    expect(api.path, _path);
    expect(await loader.load(), isA<XybridModel>());
  });

  test('a missing file surfaces as XybridException with the native message',
      () async {
    // FRB decodes Result<T, String> errors as Dart strings.
    api.loadError = _missing;

    await expectLater(
      XybridModelLoader.fromModelFile(_path).load(),
      throwsA(isA<XybridException>().having(
        (error) => error.message,
        'message',
        contains('GGUF file not found: $_path'),
      )),
    );
  });

  test('loadWithProgress maps the lone completion event', () async {
    api.loadEvents = const [FfiLoadEvent.complete()];

    final events = await XybridModelLoader.fromModelFile(_path)
        .loadWithProgress()
        .toList();

    expect(events.single, isA<LoadComplete>());
  });

  test('loadWithProgress maps the lone error event', () async {
    api.loadEvents = const [FfiLoadEvent.error(_missing)];

    final events = await XybridModelLoader.fromModelFile(_path)
        .loadWithProgress()
        .toList();

    expect(
      events.single,
      isA<LoadError>().having((event) => event.message, 'message', _missing),
    );
  });
}

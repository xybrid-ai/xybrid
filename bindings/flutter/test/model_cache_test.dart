import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

class _CacheApi implements XybridRustLibApi {
  final entries = Completer<List<CacheEntry>>();
  final removal = Completer<int>();
  String? removedModel;

  @override
  Future<List<CacheEntry>> crateApiSdkClientXybridSdkClientCacheEntries() =>
      entries.future;

  @override
  Future<int> crateApiSdkClientXybridSdkClientRemoveCachedModel({
    required String modelId,
  }) {
    removedModel = modelId;
    return removal.future;
  }

  @override
  Future<int> crateApiSdkClientXybridSdkClientCleanExpiredCache() =>
      Future.error(StateError('persistent retention is unavailable'));

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  late _CacheApi api;
  setUpAll(() {
    api = _CacheApi();
    XybridRustLib.initMock(api: api);
  });
  tearDownAll(XybridRustLib.dispose);

  test('cache listing returns a Future while native work is pending', () async {
    final pending = Xybrid.modelCacheEntries();
    var completed = false;
    unawaited(pending.then((_) => completed = true));
    await Future<void>.delayed(Duration.zero);
    expect(completed, isFalse);
    api.entries.complete([]);
    expect(await pending, isEmpty);
  });

  test('eviction forwards the model ID and waits for its count', () async {
    final pending = Xybrid.removeCachedModel('owner/repo');
    expect(api.removedModel, 'owner/repo');
    api.removal.complete(2);
    expect(await pending, 2);
  });

  test('unsupported expiry is propagated as an error', () async {
    await expectLater(Xybrid.cleanExpiredModelCache(), throwsStateError);
  });
}

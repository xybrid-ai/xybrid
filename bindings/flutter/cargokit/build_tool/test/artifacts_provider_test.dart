// Drives the real ArtifactProvider against a local HTTP server that plays the
// part of the GitHub release, so the whole consumer path is exercised: request
// order, signature checks, decompression, the per-app copy and the shared
// cache.
import 'dart:io';
import 'dart:typed_data';

import 'package:build_tool/src/artifact_compression.dart';
import 'package:build_tool/src/artifacts_provider.dart';
import 'package:build_tool/src/builder.dart';
import 'package:build_tool/src/cargo.dart';
import 'package:build_tool/src/crate_hash.dart';
import 'package:build_tool/src/options.dart';
import 'package:build_tool/src/shared_artifact_cache.dart';
import 'package:build_tool/src/target.dart';
import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:path/path.dart' as path;
import 'package:test/test.dart';

const rawName = 'aarch64-linux-android_libfake_crate.so';
const rawSig = '$rawName.sig';
const gzName = '$rawName.gz';
const gzSig = '$gzName.sig';

void main() {
  late Directory temp;
  late HttpServer server;
  late KeyPair keys;
  late String crateHash;
  late String appDir;
  late SharedArtifactCache sharedCache;
  late Map<String, List<int>> published;
  late List<String> requests;
  late Uint8List payload;

  final target = Target.forRustTriple('aarch64-linux-android')!;

  setUp(() async {
    temp = Directory.systemTemp.createTempSync('artifacts_provider_test');
    final manifestDir = Directory(path.join(temp.path, 'rust'))..createSync();
    Directory(path.join(manifestDir.path, 'src')).createSync();
    File(path.join(manifestDir.path, 'src', 'lib.rs')).writeAsStringSync('');
    // `workspace = true` with no workspace root above: like the published
    // package, this crate cannot be built from source, so a test that finds no
    // binary fails fast instead of reaching for rustup.
    File(path.join(manifestDir.path, 'Cargo.toml')).writeAsStringSync(
        '[package]\nname = "fake_crate"\nedition.workspace = true\n');
    crateHash = CrateHash.compute(manifestDir.path);

    appDir = path.join(temp.path, 'app_build');
    Directory(appDir).createSync();
    sharedCache = SharedArtifactCache(path.join(temp.path, 'shared'));
    keys = generateKey();
    payload = Uint8List.fromList(List.generate(200000, (i) => (i ~/ 7) % 251));
    published = {};
    requests = [];

    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((request) {
      final name = request.uri.pathSegments.last;
      requests.add(name);
      final body = request.uri.pathSegments.first == 'precompiled_$crateHash'
          ? published[name]
          : null;
      if (body == null) {
        request.response.statusCode = HttpStatus.notFound;
      } else {
        request.response.contentLength = body.length;
        request.response.add(body);
      }
      request.response.close();
    });
  });

  tearDown(() async {
    await server.close(force: true);
    temp.deleteSync(recursive: true);
  });

  void publish(String name, List<int> bytes, {String? signatureName}) {
    published[name] = bytes;
    published[signatureName ?? '$name.sig'] =
        sign(keys.privateKey, Uint8List.fromList(bytes));
  }

  ArtifactProvider provider() => ArtifactProvider(
        environment: BuildEnvironment(
          configuration: BuildConfiguration.release,
          crateOptions: CargokitCrateOptions(
            precompiledBinaries: PrecompiledBinaries(
              uriPrefix: 'http://127.0.0.1:${server.port}/precompiled_',
              publicKey: keys.publicKey,
            ),
          ),
          targetTempDir: appDir,
          manifestDir: path.join(temp.path, 'rust'),
          crateInfo: CrateInfo(packageName: 'fake_crate'),
          isAndroid: false,
        ),
        userOptions: CargokitUserOptions(
          usePrecompiledBinaries: true,
          verboseLogging: false,
        ),
        resolveSharedCache: () => sharedCache,
      );

  Future<Uint8List> fetch() async {
    final artifacts = await provider().getArtifacts([target]);
    return File(artifacts[target]!.single.path).readAsBytesSync();
  }

  Set<String> sharedEntries() {
    final dir = Directory(path.join(sharedCache.rootDir, crateHash));
    if (!dir.existsSync()) {
      return {};
    }
    return dir.listSync().map((e) => path.basename(e.path)).toSet()
      ..remove('.last-used');
  }

  test('downloads only the compressed form when it is published', () async {
    publish(rawName, payload);
    publish(gzName, compressArtifact(payload));

    expect(await fetch(), payload);
    expect(requests, [gzSig, gzName]);
    expect(sharedEntries(), {gzName, gzSig});
  });

  test('falls back to the uncompressed form for an older release', () async {
    publish(rawName, payload);

    expect(await fetch(), payload);
    expect(requests, [gzSig, rawSig, rawName]);
    expect(sharedEntries(), {rawName, rawSig});
  });

  test('falls back when the compressed asset fails verification', () async {
    publish(rawName, payload);
    published[gzName] = compressArtifact(payload);
    published[gzSig] = sign(keys.privateKey, Uint8List.fromList([1, 2, 3]));

    expect(await fetch(), payload);
    expect(requests, [gzSig, gzName, rawSig, rawName]);
    expect(sharedEntries(), {rawName, rawSig});
  });

  test('falls back when the compressed signature is malformed', () async {
    publish(rawName, payload);
    published[gzName] = compressArtifact(payload);
    published[gzSig] = [1, 2, 3];

    expect(await fetch(), payload);
    expect(requests, [gzSig, gzName, rawSig, rawName]);
    expect(sharedEntries(), {rawName, rawSig});
  });

  test('falls back when a signed compressed asset is not valid gzip', () async {
    publish(rawName, payload);
    publish(gzName, [0, 1, 2, 3, 4, 5, 6, 7]);

    expect(await fetch(), payload);
    expect(sharedEntries(), {rawName, rawSig});
  });

  test('restores from the compressed shared entry without any request',
      () async {
    publish(gzName, compressArtifact(payload));
    await fetch();
    Directory(path.join(appDir, 'precompiled')).deleteSync(recursive: true);
    requests.clear();

    expect(await fetch(), payload);
    expect(requests, isEmpty);
  });

  test('uses the per-app copy on a rebuild', () async {
    publish(gzName, compressArtifact(payload));
    await fetch();
    requests.clear();
    Directory(sharedCache.rootDir).deleteSync(recursive: true);

    expect(await fetch(), payload);
    expect(requests, isEmpty);
  });

  test('refuses a source build when nothing is published', () async {
    expect(fetch, throwsA(isA<SourceBuildUnavailableException>()));
  });
}

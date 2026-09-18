import 'dart:io';
import 'dart:typed_data';

import 'package:build_tool/src/shared_artifact_cache.dart';
import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:path/path.dart' as path;
import 'package:test/test.dart';

const hash = '0123456789abcdef0123456789abcdef';
const otherHash = 'fedcba9876543210fedcba9876543210';
const fileName = 'aarch64-apple-ios_libxybrid_flutter_ffi.a';
const signatureFileName = '$fileName.sig';

void main() {
  late Directory temp;
  late SharedArtifactCache cache;
  late KeyPair keys;
  late Uint8List payload;
  late Uint8List signature;
  late String destination;

  setUp(() {
    temp = Directory.systemTemp.createTempSync('shared_artifact_cache_test');
    cache = SharedArtifactCache(path.join(temp.path, 'shared'));
    keys = generateKey();
    payload = Uint8List.fromList(List.generate(4096, (i) => i % 251));
    signature = sign(keys.privateKey, payload);
    destination = path.join(temp.path, 'app', fileName);
    Directory(path.dirname(destination)).createSync(recursive: true);
  });

  tearDown(() => temp.deleteSync(recursive: true));

  void storePayload({String crateHash = hash}) => cache.store(
        crateHash: crateHash,
        fileName: fileName,
        bytes: payload,
        signatureFileName: signatureFileName,
        signatureBytes: signature,
      );

  bool restorePayload({PublicKey? publicKey}) => cache.restore(
        crateHash: hash,
        fileName: fileName,
        signatureFileName: signatureFileName,
        publicKey: publicKey ?? keys.publicKey,
        destinationPath: destination,
      );

  group('fromEnvironment', () {
    test('defaults to ~/.xybrid/cache/precompiled', () {
      final resolved = SharedArtifactCache.fromEnvironment(
          environment: {'HOME': '/home/u'}, isWindows: false);

      expect(resolved!.rootDir, '/home/u/.xybrid/cache/precompiled');
    });

    test('uses USERPROFILE on Windows', () {
      final resolved = SharedArtifactCache.fromEnvironment(
          environment: {'USERPROFILE': r'C:\Users\u'}, isWindows: true);

      expect(resolved!.rootDir, r'C:\Users\u\.xybrid\cache\precompiled');
    });

    test('ignores the POSIX-style HOME that Git Bash sets on Windows', () {
      final resolved = SharedArtifactCache.fromEnvironment(
        environment: {'HOME': '/c/Users/u', 'USERPROFILE': r'C:\Users\u'},
        isWindows: true,
      );

      expect(resolved!.rootDir, r'C:\Users\u\.xybrid\cache\precompiled');
    });

    test('is disabled on Windows when only a POSIX-style HOME exists', () {
      expect(
          SharedArtifactCache.fromEnvironment(
              environment: {'HOME': '/c/Users/u'}, isWindows: true),
          isNull);
    });

    test('ignores USERPROFILE off Windows', () {
      final resolved = SharedArtifactCache.fromEnvironment(
        environment: {'HOME': '/home/u', 'USERPROFILE': r'C:\Users\u'},
        isWindows: false,
      );

      expect(resolved!.rootDir, '/home/u/.xybrid/cache/precompiled');
    });

    test('honours the override variable', () {
      final resolved = SharedArtifactCache.fromEnvironment(environment: {
        'HOME': '/home/u',
        SharedArtifactCache.overrideVariable: '/ci/cache',
      }, isWindows: false);

      expect(resolved!.rootDir, '/ci/cache');
    });

    test('is disabled by an empty override or a missing home', () {
      expect(
          SharedArtifactCache.fromEnvironment(environment: {
            'HOME': '/home/u',
            SharedArtifactCache.overrideVariable: '',
          }, isWindows: false),
          isNull);
      expect(
          SharedArtifactCache.fromEnvironment(
              environment: {}, isWindows: false),
          isNull);
    });
  });

  group('restore', () {
    test('misses when nothing was stored', () {
      expect(restorePayload(), isFalse);
      expect(File(destination).existsSync(), isFalse);
    });

    test('copies a stored entry whose signature verifies', () {
      storePayload();

      expect(restorePayload(), isTrue);
      expect(File(destination).readAsBytesSync(), payload);
    });

    test('rejects and deletes an entry that was tampered with', () {
      storePayload();
      final entry = File(path.join(cache.rootDir, hash, fileName));
      entry.writeAsBytesSync([...payload.sublist(1), 0]);

      expect(restorePayload(), isFalse);
      expect(File(destination).existsSync(), isFalse);
      expect(entry.existsSync(), isFalse);
    });

    test('rejects an entry signed by a different key', () {
      storePayload();

      expect(restorePayload(publicKey: generateKey().publicKey), isFalse);
      expect(File(destination).existsSync(), isFalse);
    });

    test('treats a malformed signature file as a failed verification', () {
      storePayload();
      File(path.join(cache.rootDir, hash, signatureFileName))
          .writeAsBytesSync([1, 2, 3]);

      expect(restorePayload(), isFalse);
    });

    test('misses when the signature file is absent', () {
      storePayload();
      File(path.join(cache.rootDir, hash, signatureFileName)).deleteSync();

      expect(restorePayload(), isFalse);
    });
  });

  group('store', () {
    test('leaves no temporary files behind', () {
      storePayload();

      final names = Directory(path.join(cache.rootDir, hash))
          .listSync()
          .map((e) => path.basename(e.path))
          .toSet();
      expect(names, {fileName, signatureFileName, '.last-used'});
    });

    test('does not throw when the cache root cannot be created', () {
      final blocker = File(path.join(temp.path, 'not-a-directory'))
        ..writeAsStringSync('x');
      final broken = SharedArtifactCache(path.join(blocker.path, 'shared'));

      expect(
          () => broken.store(
                crateHash: hash,
                fileName: fileName,
                bytes: payload,
                signatureFileName: signatureFileName,
                signatureBytes: signature,
              ),
          returnsNormally);
    });
  });

  group('pruneStale', () {
    final now = DateTime(2026, 9, 18);

    void age(String crateHash, Duration age) =>
        File(path.join(cache.rootDir, crateHash, '.last-used'))
            .setLastModifiedSync(now.subtract(age));

    test('removes entries unused for longer than the maximum age', () {
      storePayload();
      storePayload(crateHash: otherHash);
      age(otherHash, const Duration(days: 120));

      cache.pruneStale(keepHash: hash, now: now);

      expect(
          Directory(path.join(cache.rootDir, otherHash)).existsSync(), isFalse);
      expect(Directory(path.join(cache.rootDir, hash)).existsSync(), isTrue);
    });

    test('keeps recent entries and always keeps the current hash', () {
      storePayload();
      storePayload(crateHash: otherHash);
      age(otherHash, const Duration(days: 30));
      age(hash, const Duration(days: 400));

      cache.pruneStale(keepHash: hash, now: now);

      expect(
          Directory(path.join(cache.rootDir, otherHash)).existsSync(), isTrue);
      expect(Directory(path.join(cache.rootDir, hash)).existsSync(), isTrue);
    });

    test('never touches directories that are not crate hashes', () {
      final unrelated = Directory(path.join(cache.rootDir, 'my-other-stuff'))
        ..createSync(recursive: true);
      File(path.join(unrelated.path, '.last-used'))
        ..createSync()
        ..setLastModifiedSync(now.subtract(const Duration(days: 999)));

      cache.pruneStale(keepHash: hash, now: now);

      expect(unrelated.existsSync(), isTrue);
    });
  });
}

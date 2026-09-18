import 'dart:typed_data';

import 'package:build_tool/src/artifact_compression.dart';
import 'package:build_tool/src/precompile_binaries.dart';
import 'package:build_tool/src/target.dart';
import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:test/test.dart';

void main() {
  final target = Target.forRustTriple('aarch64-apple-ios')!;
  const name = 'libxybrid_flutter_ffi.a';
  final data = Uint8List.fromList(List.generate(50000, (i) => (i ~/ 5) % 199));

  test('publishes the binary and its gzip form, each with a signature', () {
    final keys = generateKey();

    final assets = PrecompileBinaries.buildReleaseAssets(
      target: target,
      name: name,
      data: data,
      privateKey: keys.privateKey,
    );

    expect(assets.map((a) => a.name), [
      'aarch64-apple-ios_libxybrid_flutter_ffi.a',
      'aarch64-apple-ios_libxybrid_flutter_ffi.a.sig',
      'aarch64-apple-ios_libxybrid_flutter_ffi.a.gz',
      'aarch64-apple-ios_libxybrid_flutter_ffi.a.gz.sig',
    ]);
    expect(assets.map((a) => a.name),
        PrecompileBinaries.remoteAssetNames(target, name));
    expect(assets[0].assetData, data);
    expect(decompressArtifact(assets[2].assetData), data);
    expect(assets[2].assetData.length, lessThan(data.length));
  });

  group('planArtifactUpload', () {
    const raw = 'aarch64-apple-ios_libxybrid_flutter_ffi.a';
    const rawSig = '$raw.sig';
    const gz = '$raw.gz';
    const gzSig = '$gz.sig';

    ArtifactUploadPlan plan(Set<String> published) =>
        PrecompileBinaries.planArtifactUpload(
            target: target, name: name, published: published);

    test('publishes everything from the local build on a fresh release', () {
      final result = plan({});

      expect(result.source, CanonicalSource.localBuild);
      expect(result.orphans, isEmpty);
      expect(result.uploads, [raw, rawSig, gz, gzSig]);
    });

    test('has nothing to do when both forms are published', () {
      expect(plan({raw, rawSig, gz, gzSig}).isComplete, isTrue);
    });

    test('derives the compressed form from the published binary', () {
      // A release cut before compressed assets, or a run that died after the
      // uncompressed pair: the .gz must compress what consumers already get.
      final result = plan({raw, rawSig});

      expect(result.source, CanonicalSource.publishedUncompressed);
      expect(result.orphans, isEmpty);
      expect(result.uploads, [gz, gzSig]);
    });

    test('derives the uncompressed form from a published compressed pair', () {
      final result = plan({gz, gzSig});

      expect(result.source, CanonicalSource.publishedCompressed);
      expect(result.uploads, [raw, rawSig]);
    });

    test('never signs a published binary that has no signature', () {
      // The run died between the binary and its .sig. Signing this run's
      // rebuilt bytes would not match the published file, and signing the
      // published file would make the run a signing oracle.
      final result = plan({raw});

      expect(result.source, CanonicalSource.localBuild);
      expect(result.orphans, [raw]);
      expect(result.uploads, [raw, rawSig, gz, gzSig]);
    });

    test('replaces an orphaned compressed binary, keeping the published pair',
        () {
      final result = plan({raw, rawSig, gz});

      expect(result.source, CanonicalSource.publishedUncompressed);
      expect(result.orphans, [gz]);
      expect(result.uploads, [gz, gzSig]);
    });

    test('treats lone signatures as orphans too', () {
      final result = plan({rawSig, gzSig});

      expect(result.source, CanonicalSource.localBuild);
      expect(result.orphans, [rawSig, gzSig]);
      expect(result.uploads, [raw, rawSig, gz, gzSig]);
    });

    test('ignores assets that belong to other targets', () {
      final result = plan({'x86_64-linux-android_libxybrid_flutter_ffi.so'});

      expect(result.source, CanonicalSource.localBuild);
      expect(result.orphans, isEmpty);
    });
  });

  test('signs exactly the bytes that are served', () {
    final keys = generateKey();

    final assets = PrecompileBinaries.buildReleaseAssets(
      target: target,
      name: name,
      data: data,
      privateKey: keys.privateKey,
    );

    expect(verify(keys.publicKey, assets[0].assetData, assets[1].assetData),
        isTrue);
    expect(verify(keys.publicKey, assets[2].assetData, assets[3].assetData),
        isTrue);
    // The compressed signature does not cover the uncompressed bytes.
    expect(verify(keys.publicKey, assets[0].assetData, assets[3].assetData),
        isFalse);
  });
}

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

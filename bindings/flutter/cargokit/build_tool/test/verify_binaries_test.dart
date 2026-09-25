import 'dart:typed_data';

import 'package:build_tool/src/artifact_compression.dart';
import 'package:build_tool/src/verify_binaries.dart';
import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:test/test.dart';

void main() {
  late KeyPair keys;
  late Uint8List artifact;
  late Uint8List compressed;
  late Uint8List signature;

  setUp(() {
    keys = generateKey();
    artifact = Uint8List.fromList(List.generate(4096, (i) => i % 251));
    compressed = compressArtifact(artifact);
    signature = sign(keys.privateKey, compressed);
  });

  CompressedArtifactVerification verifyCompressed({
    int compressedStatusCode = 200,
    int signatureStatusCode = 200,
    Uint8List? compressedBytes,
    Uint8List? signatureBytes,
    Uint8List? uncompressedBytes,
  }) =>
      verifyCompressedArtifact(
        publicKey: keys.publicKey,
        compressedStatusCode: compressedStatusCode,
        compressedBytes: compressedBytes ?? compressed,
        signatureStatusCode: signatureStatusCode,
        signatureBytes: signatureBytes ?? signature,
        uncompressedBytes: uncompressedBytes ?? artifact,
      );

  test('accepts an older release when both compressed assets are absent', () {
    expect(
      verifyCompressed(compressedStatusCode: 404, signatureStatusCode: 404),
      CompressedArtifactVerification.notPublished,
    );
  });

  test('rejects a compressed binary published without its signature', () {
    expect(
      verifyCompressed(signatureStatusCode: 404),
      CompressedArtifactVerification.missingSignature,
    );
  });

  test('rejects a compressed signature published without its binary', () {
    expect(
      verifyCompressed(compressedStatusCode: 404),
      CompressedArtifactVerification.missingBinary,
    );
  });

  test('reports a malformed signature without throwing', () {
    expect(
      verifyCompressed(signatureBytes: Uint8List.fromList([1, 2, 3])),
      CompressedArtifactVerification.invalidSignature,
    );
  });

  test('reports signed non-gzip bytes without throwing', () {
    final notGzip = Uint8List.fromList([1, 2, 3]);

    expect(
      verifyCompressed(
        compressedBytes: notGzip,
        signatureBytes: sign(keys.privateKey, notGzip),
      ),
      CompressedArtifactVerification.invalidGzip,
    );
  });

  test('rejects a compressed artifact that expands to different bytes', () {
    expect(
      verifyCompressed(uncompressedBytes: Uint8List.fromList([4, 5, 6])),
      CompressedArtifactVerification.mismatch,
    );
  });

  test('accepts a signed compressed copy of the uncompressed artifact', () {
    expect(verifyCompressed(), CompressedArtifactVerification.valid);
  });
}

/// This is copied from Cargokit (which is the official way to use it currently)
/// Details: https://fzyzcjy.github.io/flutter_rust_bridge/manual/integrate/builtin

import 'dart:io';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:github/github.dart';
import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;

import 'artifact_compression.dart';
import 'artifacts_provider.dart';
import 'builder.dart';
import 'cargo.dart';
import 'crate_hash.dart';
import 'download_progress.dart';
import 'options.dart';
import 'rustup.dart';
import 'target.dart';

final _log = Logger('precompile_binaries');

class PrecompileBinaries {
  PrecompileBinaries({
    required this.privateKey,
    required this.githubToken,
    required this.repositorySlug,
    required this.manifestDir,
    required this.targets,
    this.androidSdkLocation,
    this.androidNdkVersion,
    this.androidMinSdkVersion,
    this.tempDir,
    this.prebuiltArtifacts = const {},
  });

  final PrivateKey privateKey;
  final String githubToken;
  final RepositorySlug repositorySlug;
  final String manifestDir;
  final List<Target> targets;
  final String? androidSdkLocation;
  final String? androidNdkVersion;
  final int? androidMinSdkVersion;
  final String? tempDir;

  /// Local addition (not upstream Cargokit): rust triple -> directory holding
  /// an already-built artifact for that target (e.g. a Bazel output). Targets
  /// in this map skip the cargo build; the crate hash, asset naming, ed25519
  /// signing, and upload are unchanged, so the consumer-side contract is
  /// byte-identical to a cargo-built asset.
  final Map<String, String> prebuiltArtifacts;

  static String fileName(Target target, String name) {
    return '${target.rust}_$name';
  }

  static String signatureFileName(Target target, String name) {
    return '${target.rust}_$name.sig';
  }

  /// The gzip form of [fileName] (xybrid addition, see
  /// `artifact_compression.dart`).
  static String compressedFileName(Target target, String name) {
    return '${fileName(target, name)}$compressedExtension';
  }

  /// Signature of the *compressed* bytes, so a consumer can verify a download
  /// before handing it to the decompressor.
  static String compressedSignatureFileName(Target target, String name) {
    return '${compressedFileName(target, name)}.sig';
  }

  /// Every asset name published for one artifact.
  static List<String> remoteAssetNames(Target target, String name) => [
        fileName(target, name),
        signatureFileName(target, name),
        compressedFileName(target, name),
        compressedSignatureFileName(target, name),
      ];

  /// Builds the release assets for one artifact: the binary and its gzip form,
  /// each with a detached signature over exactly the bytes that are served.
  ///
  /// The uncompressed pair is still published on purpose. It is what cargokit
  /// releases before this change download, and it is the consumer's automatic
  /// fallback if the compressed asset is missing or fails verification.
  static List<CreateReleaseAsset> buildReleaseAssets({
    required Target target,
    required String name,
    required Uint8List data,
    required PrivateKey privateKey,
  }) {
    final compressed = compressArtifact(data);
    // Round-trip with the codec consumers use before anything is published.
    if (!const ListEquality<int>()
        .equals(decompressArtifact(compressed), data)) {
      throw Exception('Compressed artifact does not round-trip: $name');
    }

    final publicKey = public(privateKey);
    final assets = <CreateReleaseAsset>[];
    for (final (assetName, bytes) in [
      (fileName(target, name), data),
      (compressedFileName(target, name), compressed),
    ]) {
      final signature = sign(privateKey, bytes);
      if (!verify(publicKey, bytes, signature)) {
        throw Exception('Signature verification failed');
      }
      assets.add(CreateReleaseAsset(
        name: assetName,
        contentType: "application/octet-stream",
        assetData: bytes,
      ));
      assets.add(CreateReleaseAsset(
        name: '$assetName.sig',
        contentType: "application/octet-stream",
        assetData: signature,
      ));
    }
    return assets;
  }

  Future<void> run() async {
    final crateInfo = CrateInfo.load(manifestDir);

    final targets = List.of(this.targets);
    if (targets.isEmpty) {
      targets.addAll([
        ...Target.buildableTargets(),
        if (androidSdkLocation != null) ...Target.androidTargets(),
      ]);
    }

    _log.info('Precompiling binaries for $targets');

    final hash = CrateHash.compute(manifestDir);
    _log.info('Computed crate hash: $hash');

    final String tagName = 'precompiled_$hash';

    final github = GitHub(auth: Authentication.withToken(githubToken));
    final repo = github.repositories;
    final release = await _getOrCreateRelease(
      repo: repo,
      tagName: tagName,
      packageName: crateInfo.packageName,
      hash: hash,
    );

    final tempDir = this.tempDir != null
        ? Directory(this.tempDir!)
        : Directory.systemTemp.createTempSync('precompiled_');

    tempDir.createSync(recursive: true);

    final crateOptions = CargokitCrateOptions.load(
      manifestDir: manifestDir,
    );

    final buildEnvironment = BuildEnvironment(
      configuration: BuildConfiguration.release,
      crateOptions: crateOptions,
      targetTempDir: tempDir.path,
      manifestDir: manifestDir,
      crateInfo: crateInfo,
      isAndroid: androidSdkLocation != null,
      androidSdkPath: androidSdkLocation,
      androidNdkVersion: androidNdkVersion,
      androidMinSdkVersion: androidMinSdkVersion,
    );

    // Lazy: only touch rustup when a target actually needs a cargo build, so
    // a run where every target is prebuilt works on a runner without Rust.
    Rustup? rustup;

    final uploaded = {
      for (final asset in release.assets ?? <ReleaseAsset>[]) asset.name
    };

    for (final target in targets) {
      final artifactNames = getArtifactNames(
        target: target,
        libraryName: crateInfo.libName,
        remote: true,
      );

      if (artifactNames.every((name) =>
          PrecompileBinaries.remoteAssetNames(target, name)
              .every(uploaded.contains))) {
        _log.info("All artifacts for $target already exist - skipping");
        continue;
      }

      final String res;
      final prebuiltDir = prebuiltArtifacts[target.rust];
      if (prebuiltDir != null) {
        _log.info('Using prebuilt artifacts for $target from $prebuiltDir');
        res = prebuiltDir;
      } else {
        _log.info('Building for $target');

        rustup ??= Rustup();
        final builder =
            RustBuilder(target: target, environment: buildEnvironment);
        builder.prepare(rustup);
        res = await builder.build();
      }

      final assets = <CreateReleaseAsset>[];
      for (final name in artifactNames) {
        final file = File(path.join(res, name));
        if (!file.existsSync()) {
          throw Exception('Missing artifact: ${file.path}');
        }

        final data = file.readAsBytesSync();
        final built = PrecompileBinaries.buildReleaseAssets(
          target: target,
          name: name,
          data: data,
          privateKey: privateKey,
        );
        _log.info('$name: ${formatByteSize(data.length)}, '
            '${formatByteSize(built[2].assetData.length)} compressed');
        assets.addAll(built);
      }
      // A release that already holds some of these (a re-run, or one created
      // before compressed assets existed) only gets what it is missing;
      // re-uploading an existing name is rejected by GitHub.
      assets.removeWhere((asset) => uploaded.contains(asset.name));
      _log.info('Uploading assets: ${assets.map((e) => e.name)}');
      for (final asset in assets) {
        // This seems to be failing on CI so do it one by one
        int retryCount = 0;
        while (true) {
          try {
            await repo.uploadReleaseAssets(release, [asset]);
            break;
          } on Exception catch (e) {
            if (retryCount == 10) {
              rethrow;
            }
            ++retryCount;
            _log.shout(
                'Upload failed (attempt $retryCount, will retry): ${e.toString()}');
            await Future.delayed(Duration(seconds: 2));
          }
        }
      }
    }

    _log.info('Cleaning up');
    tempDir.deleteSync(recursive: true);
  }

  Future<Release> _getOrCreateRelease({
    required RepositoriesService repo,
    required String tagName,
    required String packageName,
    required String hash,
  }) async {
    Release release;
    try {
      _log.info('Fetching release $tagName');
      release = await repo.getReleaseByTagName(repositorySlug, tagName);
    } on ReleaseNotFound {
      _log.info('Release not found - creating release $tagName');
      release = await repo.createRelease(
          repositorySlug,
          CreateRelease.from(
            tagName: tagName,
            name: 'Precompiled binaries ${hash.substring(0, 8)}',
            targetCommitish: null,
            isDraft: false,
            isPrerelease: false,
            body: 'Precompiled binaries for crate $packageName, '
                'crate hash $hash.',
          ));
    }
    return release;
  }
}

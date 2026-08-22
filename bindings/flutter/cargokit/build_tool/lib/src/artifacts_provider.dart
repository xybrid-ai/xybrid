/// This is copied from Cargokit (which is the official way to use it currently)
/// Details: https://fzyzcjy.github.io/flutter_rust_bridge/manual/integrate/builtin

import 'dart:io';

import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:http/http.dart';
import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;

import 'builder.dart';
import 'crate_hash.dart';
import 'options.dart';
import 'precompile_binaries.dart';
import 'rustup.dart';
import 'target.dart';

class Artifact {
  /// File system location of the artifact.
  final String path;

  /// Actual file name that the artifact should have in destination folder.
  final String finalFileName;

  AritifactType get type {
    if (finalFileName.endsWith('.dll') ||
        finalFileName.endsWith('.dll.lib') ||
        finalFileName.endsWith('.pdb') ||
        finalFileName.endsWith('.so') ||
        finalFileName.endsWith('.dylib')) {
      return AritifactType.dylib;
    } else if (finalFileName.endsWith('.lib') || finalFileName.endsWith('.a')) {
      return AritifactType.staticlib;
    } else {
      throw Exception('Unknown artifact type for $finalFileName');
    }
  }

  Artifact({
    required this.path,
    required this.finalFileName,
  });
}

final _log = Logger('artifacts_provider');

class ArtifactProvider {
  ArtifactProvider({
    required this.environment,
    required this.userOptions,
  });

  final BuildEnvironment environment;
  final CargokitUserOptions userOptions;

  /// Whether precompiled binaries should be used for this crate.
  ///
  /// xybrid deviates from upstream cargokit here. Upstream's default is "build
  /// from source whenever Rustup is installed", which silently disabled
  /// precompiled binaries for every consumer who had ever installed Rust and
  /// dropped them into a build the published package cannot perform (#338).
  ///
  /// The default is resolved per crate location instead:
  ///
  /// * Published package — no workspace root, no sibling crates. Precompiled
  ///   binaries are the only viable path, so they are used regardless of the
  ///   local toolchain.
  /// * Monorepo checkout — upstream's rule applies. Source builds matter here
  ///   because [CrateHash] only covers `rust/`, so edits to `xybrid-core`,
  ///   `xybrid-sdk` or `xybrid-ffi-facade` do not change the artifact key and a
  ///   precompiled binary would silently ship stale code.
  ///
  /// An explicit `use_precompiled_binaries` in `cargokit_options.yaml` always
  /// wins.
  late final bool usePrecompiledBinaries = userOptions.usePrecompiledBinaries ??
      (_sourceBuildBlocker() != null || Rustup.executablePath() == null);

  Future<Map<Target, List<Artifact>>> getArtifacts(List<Target> targets) async {
    final result = await _getPrecompiledArtifacts(targets);

    final pendingTargets = List.of(targets);
    pendingTargets.removeWhere((element) => result.containsKey(element));

    if (pendingTargets.isEmpty) {
      return result;
    }

    // xybrid addition. Upstream cargokit falls straight through to a source
    // build here. The published `xybrid_flutter` package cannot be built from
    // source — see `_sourceBuildBlocker` — so that fallback used to surface as
    // an unrelated cargo error (#338). Stop with an actionable message instead.
    final blocker = _sourceBuildBlocker();
    if (blocker != null) {
      throw SourceBuildUnavailableException(
        targets: pendingTargets,
        blocker: blocker,
        precompiledEnabled: usePrecompiledBinaries &&
            environment.crateOptions.precompiledBinaries != null,
      );
    }

    final rustup = Rustup();
    for (final target in targets) {
      final builder = RustBuilder(target: target, environment: environment);
      builder.prepare(rustup);
      _log.info('Building ${environment.crateInfo.packageName} for $target');
      final targetDir = await builder.build();
      // For local build accept both static and dynamic libraries.
      final artifactNames = <String>{
        ...getArtifactNames(
          target: target,
          libraryName: environment.crateInfo.libName,
          aritifactType: AritifactType.dylib,
          remote: false,
        ),
        ...getArtifactNames(
          target: target,
          libraryName: environment.crateInfo.libName,
          aritifactType: AritifactType.staticlib,
          remote: false,
        )
      };
      final artifacts = artifactNames
          .map((artifactName) => Artifact(
                path: path.join(targetDir, artifactName),
                finalFileName: artifactName,
              ))
          .where((element) => File(element.path).existsSync())
          .toList();
      result[target] = artifacts;
    }
    return result;
  }

  /// Returns a description of why this crate cannot be built from source in
  /// its current location, or `null` if a source build is viable.
  ///
  /// Both checks below hold inside the monorepo and fail in the package
  /// published to pub.dev, which ships `rust/` without the workspace root it
  /// inherits from and without the sibling crates its path dependencies point
  /// at.
  String? _sourceBuildBlocker() {
    final manifestFile = File(path.join(environment.manifestDir, 'Cargo.toml'));
    if (!manifestFile.existsSync()) {
      return 'no Cargo.toml in ${environment.manifestDir}';
    }
    final manifest = manifestFile.readAsStringSync();

    if (RegExp(r'workspace\s*=\s*true').hasMatch(manifest) &&
        _workspaceRootDir() == null) {
      return 'Cargo.toml inherits fields from a workspace root '
          '(`workspace = true`) but no workspace root exists above '
          '${environment.manifestDir}';
    }

    final missingPathDeps = RegExp(r'path\s*=\s*"([^"]+)"')
        .allMatches(manifest)
        .map((m) => m.group(1)!)
        .where((dep) =>
            !Directory(path.normalize(path.join(environment.manifestDir, dep)))
                .existsSync())
        .toSet();
    if (missingPathDeps.isNotEmpty) {
      return 'Cargo.toml has path dependencies that are not present: '
          '${missingPathDeps.join(', ')}';
    }

    return null;
  }

  /// Nearest ancestor directory holding a `Cargo.toml` with a `[workspace]`
  /// table, or `null` if there is none.
  String? _workspaceRootDir() {
    var dir = Directory(environment.manifestDir).absolute;
    while (true) {
      final manifest = File(path.join(dir.path, 'Cargo.toml'));
      if (manifest.existsSync() &&
          RegExp(r'^\s*\[workspace[\].]', multiLine: true)
              .hasMatch(manifest.readAsStringSync())) {
        return dir.path;
      }
      if (dir.parent.path == dir.path) {
        return null;
      }
      dir = dir.parent;
    }
  }

  Future<Map<Target, List<Artifact>>> _getPrecompiledArtifacts(
      List<Target> targets) async {
    if (!usePrecompiledBinaries) {
      _log.info('Precompiled binaries are disabled');
      return {};
    }
    if (environment.crateOptions.precompiledBinaries == null) {
      _log.fine('Precompiled binaries not enabled for this crate');
      return {};
    }

    final start = Stopwatch()..start();
    final crateHash = CrateHash.compute(environment.manifestDir,
        tempStorage: environment.targetTempDir);
    _log.fine(
        'Computed crate hash $crateHash in ${start.elapsedMilliseconds}ms');

    final downloadedArtifactsDir =
        path.join(environment.targetTempDir, 'precompiled', crateHash);
    Directory(downloadedArtifactsDir).createSync(recursive: true);

    final res = <Target, List<Artifact>>{};

    for (final target in targets) {
      final requiredArtifacts = getArtifactNames(
        target: target,
        libraryName: environment.crateInfo.libName,
        remote: true,
      );
      final artifactsForTarget = <Artifact>[];

      for (final artifact in requiredArtifacts) {
        final fileName = PrecompileBinaries.fileName(target, artifact);
        final downloadedPath = path.join(downloadedArtifactsDir, fileName);
        if (!File(downloadedPath).existsSync()) {
          final signatureFileName =
              PrecompileBinaries.signatureFileName(target, artifact);
          await _tryDownloadArtifacts(
            crateHash: crateHash,
            fileName: fileName,
            signatureFileName: signatureFileName,
            finalPath: downloadedPath,
          );
        }
        if (File(downloadedPath).existsSync()) {
          artifactsForTarget.add(Artifact(
            path: downloadedPath,
            finalFileName: artifact,
          ));
        } else {
          break;
        }
      }

      // Only provide complete set of artifacts.
      if (artifactsForTarget.length == requiredArtifacts.length) {
        _log.fine('Found precompiled artifacts for $target');
        res[target] = artifactsForTarget;
      }
    }

    return res;
  }

  static Future<Response> _get(Uri url, {Map<String, String>? headers}) async {
    int attempt = 0;
    const maxAttempts = 10;
    while (true) {
      try {
        return await get(url, headers: headers);
      } on SocketException catch (e) {
        // Try to detect reset by peer error and retry.
        if (attempt++ < maxAttempts &&
            (e.osError?.errorCode == 54 || e.osError?.errorCode == 10054)) {
          _log.severe(
              'Failed to download $url: $e, attempt $attempt of $maxAttempts, will retry...');
          await Future.delayed(Duration(seconds: 1));
          continue;
        } else {
          rethrow;
        }
      } on ClientException catch (e) {
        // The release host sometimes drops the connection mid-response
        // ("Connection closed before full header was received"). That is a
        // transport hiccup, not a verdict on whether the artifact exists,
        // so retry instead of failing the build on it.
        if (attempt++ < maxAttempts) {
          _log.severe(
              'Failed to download $url: $e, attempt $attempt of $maxAttempts, will retry...');
          await Future.delayed(Duration(seconds: 1));
          continue;
        } else {
          rethrow;
        }
      }
    }
  }

  Future<void> _tryDownloadArtifacts({
    required String crateHash,
    required String fileName,
    required String signatureFileName,
    required String finalPath,
  }) async {
    final precompiledBinaries = environment.crateOptions.precompiledBinaries!;
    final prefix = precompiledBinaries.uriPrefix;
    final url = Uri.parse('$prefix$crateHash/$fileName');
    final signatureUrl = Uri.parse('$prefix$crateHash/$signatureFileName');
    _log.fine('Downloading signature from $signatureUrl');
    final signature = await _get(signatureUrl);
    if (signature.statusCode == 404) {
      _log.warning(
          'Precompiled binaries not available for crate hash $crateHash ($fileName)');
      return;
    }
    if (signature.statusCode != 200) {
      _log.severe(
          'Failed to download signature $signatureUrl: status ${signature.statusCode}');
      return;
    }
    _log.fine('Downloading binary from $url');
    final res = await _get(url);
    if (res.statusCode != 200) {
      _log.severe('Failed to download binary $url: status ${res.statusCode}');
      return;
    }
    if (verify(
        precompiledBinaries.publicKey, res.bodyBytes, signature.bodyBytes)) {
      File(finalPath).writeAsBytesSync(res.bodyBytes);
    } else {
      _log.shout('Signature verification failed! Ignoring binary.');
    }
  }
}

/// Thrown when precompiled binaries did not cover every target and the crate
/// cannot be built from source either.
class SourceBuildUnavailableException implements Exception {
  SourceBuildUnavailableException({
    required this.targets,
    required this.blocker,
    required this.precompiledEnabled,
  });

  /// Targets left without an artifact.
  final List<Target> targets;

  /// Why the source-build fallback cannot succeed.
  final String blocker;

  /// Whether precompiled binaries were attempted before this fallback.
  final bool precompiledEnabled;

  @override
  String toString() {
    return [
      ' ',
      'No native library available for: ${targets.join(', ')}.',
      ' ',
      if (precompiledEnabled)
        'This package ships precompiled binaries, but none matched. '
            'Building from source is not possible here:'
      else
        'Precompiled binaries are disabled (`use_precompiled_binaries: false`), '
            'and building from source is not possible here:',
      ' ',
      '  $blocker',
      ' ',
      if (!precompiledEnabled)
        'Remove `use_precompiled_binaries: false` from cargokit_options.yaml '
            'to use the precompiled binaries.'
      else
        'Please report this at https://github.com/xybrid-ai/xybrid/issues '
            'with the target above and your Flutter version.',
      ' ',
    ].join('\n');
  }
}

enum AritifactType {
  staticlib,
  dylib,
}

AritifactType artifactTypeForTarget(Target target) {
  if (target.darwinPlatform != null) {
    return AritifactType.staticlib;
  } else {
    return AritifactType.dylib;
  }
}

List<String> getArtifactNames({
  required Target target,
  required String libraryName,
  required bool remote,
  AritifactType? aritifactType,
}) {
  aritifactType ??= artifactTypeForTarget(target);
  if (target.darwinArch != null) {
    if (aritifactType == AritifactType.staticlib) {
      return ['lib$libraryName.a'];
    } else {
      return ['lib$libraryName.dylib'];
    }
  } else if (target.rust.contains('-windows-')) {
    if (aritifactType == AritifactType.staticlib) {
      return ['$libraryName.lib'];
    } else {
      return [
        '$libraryName.dll',
        '$libraryName.dll.lib',
        if (!remote) '$libraryName.pdb'
      ];
    }
  } else if (target.rust.contains('-linux-')) {
    if (aritifactType == AritifactType.staticlib) {
      return ['lib$libraryName.a'];
    } else {
      return ['lib$libraryName.so'];
    }
  } else {
    throw Exception("Unsupported target: ${target.rust}");
  }
}

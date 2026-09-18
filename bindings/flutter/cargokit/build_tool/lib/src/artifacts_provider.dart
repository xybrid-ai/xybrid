/// This is copied from Cargokit (which is the official way to use it currently)
/// Details: https://fzyzcjy.github.io/flutter_rust_bridge/manual/integrate/builtin

import 'dart:io';
import 'dart:typed_data';

import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:http/http.dart';
import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;

import 'artifact_compression.dart';
import 'builder.dart';
import 'crate_hash.dart';
import 'download_progress.dart';
import 'options.dart';
import 'precompile_binaries.dart';
import 'rustup.dart';
import 'shared_artifact_cache.dart';
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

/// One published form of an artifact: as-is, or gzip-compressed (xybrid).
class _RemoteForm {
  const _RemoteForm({
    required this.fileName,
    required this.signatureFileName,
    this.decode,
  });

  final String fileName;

  /// Signs exactly the bytes served as [fileName].
  final String signatureFileName;

  /// Turns the verified download into the artifact; `null` when served as-is.
  final Uint8List Function(List<int> verifiedBytes)? decode;
}

class ArtifactProvider {
  ArtifactProvider({
    required this.environment,
    required this.userOptions,
    SharedArtifactCache? Function()? resolveSharedCache,
  }) : _resolveSharedCache =
            resolveSharedCache ?? SharedArtifactCache.fromEnvironment;

  /// Where the user-level cache lives. A seam for tests, which must not read
  /// or write the real `~/.xybrid`.
  final SharedArtifactCache? Function() _resolveSharedCache;

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

    // xybrid addition: the directory above lives under the app's build output,
    // so `flutter clean` and every new project start from nothing. The shared
    // cache sits behind it and turns those into a local, re-verified copy.
    final sharedCache = _resolveSharedCache();

    final res = <Target, List<Artifact>>{};

    for (final target in targets) {
      final requiredArtifacts = getArtifactNames(
        target: target,
        libraryName: environment.crateInfo.libName,
        remote: true,
      );
      final artifactsForTarget = <Artifact>[];
      var downloadedNow = false;
      var restoredFromSharedCache = false;

      for (final artifact in requiredArtifacts) {
        final fileName = PrecompileBinaries.fileName(target, artifact);
        final downloadedPath = path.join(downloadedArtifactsDir, fileName);
        if (!File(downloadedPath).existsSync()) {
          // Preference order (xybrid): the compressed form is a third of the
          // download. The uncompressed form stays as the fallback — it is all
          // that releases from before compressed assets have, and it covers a
          // compressed asset that is missing or fails verification.
          final forms = [
            _RemoteForm(
              fileName: PrecompileBinaries.compressedFileName(target, artifact),
              signatureFileName: PrecompileBinaries.compressedSignatureFileName(
                  target, artifact),
              decode: decompressArtifact,
            ),
            _RemoteForm(
              fileName: fileName,
              signatureFileName:
                  PrecompileBinaries.signatureFileName(target, artifact),
            ),
          ];

          final restored = sharedCache != null &&
              forms.any((form) => sharedCache.restore(
                    crateHash: crateHash,
                    fileName: form.fileName,
                    signatureFileName: form.signatureFileName,
                    publicKey:
                        environment.crateOptions.precompiledBinaries!.publicKey,
                    destinationPath: downloadedPath,
                    decode: form.decode,
                  ));
          if (restored) {
            restoredFromSharedCache = true;
            final size = formatByteSize(File(downloadedPath).lengthSync());
            _log.info('Reusing $fileName ($size) from the shared cache '
                '${sharedCache.rootDir} (signature verified)');
          } else {
            downloadedNow = true;
            for (final form in forms) {
              final downloaded = await _tryDownloadArtifact(
                crateHash: crateHash,
                form: form,
                hasFallback: !identical(form, forms.last),
                finalPath: downloadedPath,
                sharedCache: sharedCache,
              );
              if (downloaded) {
                break;
              }
            }
          }
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
        // INFO on purpose (xybrid): without this line a consumer's build log
        // never says whether the native library was downloaded, reused from
        // an earlier build, or compiled.
        final source = downloadedNow
            ? 'downloaded'
            : restoredFromSharedCache
                ? 'shared cache'
                : 'cached';
        _log.info('Using precompiled ${environment.crateInfo.packageName} '
            'for $target ($source)');
        res[target] = artifactsForTarget;
      }
    }

    return res;
  }

  static Future<Response> _get(Uri url, {Map<String, String>? headers}) {
    return _withRetry(url, () => get(url, headers: headers));
  }

  /// Streams [url] so its size can be announced before the body arrives and
  /// progress reported while it does (xybrid addition).
  ///
  /// A precompiled static library can exceed 100 MB. Upstream fetched it with
  /// a buffered `get` and logged at FINE, so the build step sat silent for
  /// minutes and read as "the Rust engine is compiling". A non-200 response
  /// comes back with an empty body and is logged by the caller.
  static Future<({int statusCode, Uint8List bodyBytes})> _download(
    Uri url, {
    required String label,
  }) {
    return _withRetry(url, () async {
      final client = Client();
      try {
        final response = await client.send(Request('GET', url));
        if (response.statusCode != 200) {
          await response.stream.drain<void>();
          return (statusCode: response.statusCode, bodyBytes: Uint8List(0));
        }

        final total = response.contentLength;
        final size = total == null ? 'size unknown' : formatByteSize(total);
        _log.info('Downloading precompiled $label ($size) from $url');

        final progress = DownloadProgress(label: label, totalBytes: total);
        final stopwatch = Stopwatch()..start();
        final body = BytesBuilder(copy: false);
        await for (final chunk in response.stream) {
          body.add(chunk);
          final line = progress.add(chunk.length, stopwatch.elapsed);
          if (line != null) {
            _log.info(line);
          }
        }
        _log.info(progress.summary(stopwatch.elapsed));
        return (statusCode: response.statusCode, bodyBytes: body.takeBytes());
      } finally {
        client.close();
      }
    });
  }

  /// Runs [attempt], retrying the transport failures the release host is
  /// known to produce. A retried download starts over from the first byte.
  static Future<T> _withRetry<T>(
    Uri url,
    Future<T> Function() attempt,
  ) async {
    int attempts = 0;
    const maxAttempts = 10;
    while (true) {
      try {
        return await attempt();
      } on SocketException catch (e) {
        // Try to detect reset by peer error and retry.
        if (attempts++ < maxAttempts &&
            (e.osError?.errorCode == 54 || e.osError?.errorCode == 10054)) {
          _log.severe(
              'Failed to download $url: $e, attempt $attempts of $maxAttempts, will retry...');
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
        if (attempts++ < maxAttempts) {
          _log.severe(
              'Failed to download $url: $e, attempt $attempts of $maxAttempts, will retry...');
          await Future.delayed(Duration(seconds: 1));
          continue;
        } else {
          rethrow;
        }
      }
    }
  }

  /// Downloads one published [form] of an artifact into [finalPath].
  ///
  /// Returns whether the artifact is now in place. Every failure is logged and
  /// reported as `false` so the caller can move on to the next form.
  /// [hasFallback] only tunes the wording: a release from before compressed
  /// assets existed answers 404 for them, which is not worth a warning.
  Future<bool> _tryDownloadArtifact({
    required String crateHash,
    required _RemoteForm form,
    required bool hasFallback,
    required String finalPath,
    required SharedArtifactCache? sharedCache,
  }) async {
    final precompiledBinaries = environment.crateOptions.precompiledBinaries!;
    final prefix = precompiledBinaries.uriPrefix;
    final fileName = form.fileName;
    final url = Uri.parse('$prefix$crateHash/$fileName');
    final signatureUrl =
        Uri.parse('$prefix$crateHash/${form.signatureFileName}');
    _log.fine('Downloading signature from $signatureUrl');
    final signature = await _get(signatureUrl);
    if (signature.statusCode == 404) {
      if (hasFallback) {
        _log.fine('$fileName is not published for crate hash $crateHash');
      } else {
        _log.warning(
            'Precompiled binaries not available for crate hash $crateHash ($fileName)');
      }
      return false;
    }
    if (signature.statusCode != 200) {
      _log.severe(
          'Failed to download signature $signatureUrl: status ${signature.statusCode}');
      return false;
    }
    final res = await _download(url, label: fileName);
    if (res.statusCode != 200) {
      _log.severe('Failed to download binary $url: status ${res.statusCode}');
      return false;
    }
    if (!verify(
        precompiledBinaries.publicKey, res.bodyBytes, signature.bodyBytes)) {
      _log.shout('Signature verification failed! Ignoring $fileName.');
      return false;
    }

    // Only verified bytes reach the decompressor.
    final decode = form.decode;
    final Uint8List artifactBytes;
    try {
      artifactBytes = decode == null ? res.bodyBytes : decode(res.bodyBytes);
    } on FormatException catch (e) {
      _log.warning('Could not decompress $fileName: $e');
      return false;
    }
    writeFileAtomically(finalPath, artifactBytes);
    if (decode != null) {
      _log.info(
          'Unpacked $fileName to ${formatByteSize(artifactBytes.length)}');
    }

    // The shared cache keeps the form that was downloaded, with the signature
    // that covers it — for the compressed form that is also a third of the
    // disk space.
    if (sharedCache != null) {
      sharedCache.store(
        crateHash: crateHash,
        fileName: fileName,
        bytes: res.bodyBytes,
        signatureFileName: form.signatureFileName,
        signatureBytes: signature.bodyBytes,
      );
      sharedCache.pruneStale(keepHash: crateHash);
    }
    return true;
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

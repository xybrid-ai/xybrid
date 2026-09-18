/// xybrid addition — not part of upstream cargokit.
///
/// A user-level cache of precompiled binaries, shared by every app and project
/// on the machine.
///
/// Upstream keeps downloads under the consuming app's build directory only, so
/// `flutter clean` — or simply starting a second project — fetches the same
/// signed binary again. For xybrid that is a static library well over 100 MB.
///
/// Trust model: a cache entry is never used on faith. Its detached ed25519
/// signature is stored beside it and re-verified against the public key pinned
/// in the package's `cargokit.yaml` every time an entry is copied into an app.
/// A file that was corrupted or swapped in the shared location therefore fails
/// closed: it is deleted and the binary is downloaded again.
///
/// The cache is an optimisation and must never fail a build. Every file-system
/// error here degrades to "cache miss" or "not stored".
library;

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:ed25519_edwards/ed25519_edwards.dart';
import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;

final _log = Logger('shared_artifact_cache');

/// Writes [bytes] to [finalPath] through a temporary sibling and a rename, so
/// an interrupted build can never leave a truncated file behind under a name
/// that later builds trust.
void writeFileAtomically(String finalPath, List<int> bytes) {
  final suffix = '$pid.${Random().nextInt(1 << 32).toRadixString(16)}.tmp';
  final temp = File('$finalPath.$suffix');
  try {
    temp.writeAsBytesSync(bytes, flush: true);
    temp.renameSync(finalPath);
  } catch (_) {
    if (temp.existsSync()) {
      temp.deleteSync();
    }
    rethrow;
  }
}

class SharedArtifactCache {
  SharedArtifactCache(this.rootDir);

  /// Relocates the cache (for example into a directory a CI system persists
  /// between runs). Set to an empty value to turn the shared cache off.
  static const overrideVariable = 'XYBRID_PRECOMPILED_CACHE_DIR';

  /// Entries untouched for this long are removed the next time something new
  /// is stored. Every SDK release adds a directory of several hundred MB, so
  /// without pruning the cache only ever grows.
  static const defaultMaxAge = Duration(days: 90);

  static const _lastUsedMarker = '.last-used';

  /// Only directories named like a cargokit crate hash are ever pruned, so
  /// pointing [overrideVariable] at a directory that holds other things cannot
  /// delete them.
  static final _crateHashPattern = RegExp(r'^[0-9a-f]{32}$');

  final String rootDir;

  /// Resolves the cache location, or `null` when it is disabled or there is no
  /// home directory to put it in (some sandboxes and CI containers).
  ///
  /// Defaults to `~/.xybrid/cache/precompiled`, next to the SDK's model cache
  /// and the iOS ONNX Runtime cache.
  ///
  /// The home directory is read the way `rustup.dart` reads it: `USERPROFILE`
  /// on Windows, `HOME` everywhere else. On Windows `HOME` is only set by Git
  /// Bash / MSYS, as a POSIX-style path (`/c/Users/name`) that `dart:io` would
  /// resolve against the current drive's root — a cache in the wrong place
  /// that builds started from cmd or PowerShell would never find.
  static SharedArtifactCache? fromEnvironment({
    Map<String, String>? environment,
    bool? isWindows,
  }) {
    final env = environment ?? Platform.environment;
    final override = env[overrideVariable];
    if (override != null) {
      return override.trim().isEmpty ? null : SharedArtifactCache(override);
    }
    final windows = isWindows ?? Platform.isWindows;
    final home = env[windows ? 'USERPROFILE' : 'HOME'];
    if (home == null || home.isEmpty) {
      return null;
    }
    final context =
        path.Context(style: windows ? path.Style.windows : path.Style.posix);
    return SharedArtifactCache(
        context.join(home, '.xybrid', 'cache', 'precompiled'));
  }

  /// Copies the cached [fileName] to [destinationPath] if it is present and its
  /// stored signature verifies against [publicKey].
  ///
  /// Returns `false` on a miss. An entry that fails verification is deleted so
  /// the caller's fallback download replaces it.
  ///
  /// [decode] turns the verified bytes into what belongs at [destinationPath]
  /// — decompression, for an entry stored in its compressed form. It only ever
  /// runs on bytes whose signature has been checked.
  bool restore({
    required String crateHash,
    required String fileName,
    required String signatureFileName,
    required PublicKey publicKey,
    required String destinationPath,
    List<int> Function(Uint8List verifiedBytes)? decode,
  }) {
    final entry = File(path.join(rootDir, crateHash, fileName));
    final signature = File(path.join(rootDir, crateHash, signatureFileName));
    try {
      if (!entry.existsSync() || !signature.existsSync()) {
        return false;
      }
      final bytes = entry.readAsBytesSync();
      if (!_verifies(publicKey, bytes, signature.readAsBytesSync())) {
        _log.warning('Shared cache entry ${entry.path} failed signature '
            'verification; deleting it and downloading a fresh copy.');
        _deleteQuietly(entry);
        _deleteQuietly(signature);
        return false;
      }
      writeFileAtomically(
          destinationPath, decode == null ? bytes : decode(bytes));
      _markUsed(crateHash);
      return true;
    } on FormatException catch (e) {
      _log.warning('Shared cache entry ${entry.path} could not be decoded '
          '($e); deleting it and downloading a fresh copy.');
      _deleteQuietly(entry);
      _deleteQuietly(signature);
      return false;
    } on FileSystemException catch (e) {
      _log.fine('Shared cache unavailable at $rootDir: $e');
      return false;
    }
  }

  /// Stores an already-verified download for later builds and other projects.
  void store({
    required String crateHash,
    required String fileName,
    required List<int> bytes,
    required String signatureFileName,
    required List<int> signatureBytes,
  }) {
    try {
      final dir = Directory(path.join(rootDir, crateHash))
        ..createSync(recursive: true);
      // Signature first: `restore` looks for the binary first, so a reader
      // that finds the binary always finds its signature too.
      writeFileAtomically(
          path.join(dir.path, signatureFileName), signatureBytes);
      writeFileAtomically(path.join(dir.path, fileName), bytes);
      _markUsed(crateHash);
    } on FileSystemException catch (e) {
      _log.info('Could not store $fileName in the shared cache $rootDir '
          '(the build is unaffected): $e');
    }
  }

  /// Removes crate-hash directories, other than [keepHash], that have not been
  /// used for [maxAge].
  void pruneStale({
    required String keepHash,
    Duration maxAge = defaultMaxAge,
    DateTime? now,
  }) {
    final reference = now ?? DateTime.now();
    try {
      final root = Directory(rootDir);
      if (!root.existsSync()) {
        return;
      }
      for (final dir in root.listSync().whereType<Directory>()) {
        final name = path.basename(dir.path);
        if (name == keepHash || !_crateHashPattern.hasMatch(name)) {
          continue;
        }
        final marker = File(path.join(dir.path, _lastUsedMarker));
        final lastUsed = marker.existsSync()
            ? marker.lastModifiedSync()
            : dir.statSync().modified;
        if (reference.difference(lastUsed) > maxAge) {
          _log.fine('Pruning shared cache entry $name, last used $lastUsed');
          dir.deleteSync(recursive: true);
        }
      }
    } on FileSystemException catch (e) {
      _log.fine('Could not prune the shared cache at $rootDir: $e');
    }
  }

  void _markUsed(String crateHash) {
    final marker = File(path.join(rootDir, crateHash, _lastUsedMarker));
    try {
      if (!marker.existsSync()) {
        marker.createSync(recursive: true);
      }
      marker.setLastModifiedSync(DateTime.now());
    } on FileSystemException catch (e) {
      _log.fine('Could not update ${marker.path}: $e');
    }
  }

  static bool _verifies(PublicKey key, Uint8List bytes, Uint8List signature) {
    try {
      return verify(key, bytes, signature);
    } catch (_) {
      // A malformed signature file (wrong length) makes `verify` throw.
      return false;
    }
  }

  static void _deleteQuietly(File file) {
    try {
      if (file.existsSync()) {
        file.deleteSync();
      }
    } on FileSystemException catch (e) {
      _log.fine('Could not delete ${file.path}: $e');
    }
  }
}

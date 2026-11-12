// The step 6 staging scenario on the host (lane flutter-mac): `flutter test`
// on the Dart VM, against the compiled native library.
//
// Runs observability_staging_scenario.dart (the README's steps, shared with
// the on-device lane) with the config run_staging.py wrote, and writes
// expectations.json. Skipped unless XYBRID_STAGING_CONFIG and
// XYBRID_STAGING_EXPECTATIONS are set, so `flutter test` stays native-free;
// run_staging.py sets both:
//
//   cargo build --release -p xybrid_flutter --features platform-macos
//   python3 tools/scripts/observability_staging/run_staging.py run \
//     --lane flutter-mac -- bash -c 'cd {repo}/bindings/flutter && \
//     XYBRID_FLUTTER_NATIVE_LIB={repo}/target/release/libxybrid_flutter_ffi.dylib \
//     flutter test --no-pub test/observability_staging_test.dart'
//
// The config holds the staging key: it is deleted as soon as it is read.
// XYBRID_FLUTTER_NATIVE_LIB names the compiled libxybrid_flutter_ffi, as for
// observability_scenario_test.dart.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    show ExternalLibrary;
import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

import 'observability_staging_scenario.dart';

/// Loads the native library from XYBRID_FLUTTER_NATIVE_LIB when given, then
/// initializes the SDK as an application would.
Future<void> _initialize(Map<String, String> env) async {
  final library = env['XYBRID_FLUTTER_NATIVE_LIB'];
  if (library != null && library.isNotEmpty) {
    if (!File(library).existsSync()) {
      throw StateError('XYBRID_FLUTTER_NATIVE_LIB=$library does not exist');
    }
    await XybridRustLib.init(externalLibrary: ExternalLibrary.open(library));
  }
  await Xybrid.init();
}

String _packageVersion() {
  final pubspec = File('pubspec.yaml').readAsStringSync();
  return RegExp(r'^version:\s*(\S+)', multiLine: true)
          .firstMatch(pubspec)
          ?.group(1) ??
      'unknown';
}

void main() {
  final env = Platform.environment;
  final configPath = env['XYBRID_STAGING_CONFIG'] ?? '';
  final expectationsPath = env['XYBRID_STAGING_EXPECTATIONS'] ?? '';

  test(
    'staging scenario exports a registry model run two ways',
    () async {
      final configFile = File(configPath);
      final text = configFile.readAsStringSync();
      // It holds the staging key: delete it before anything else runs.
      configFile.deleteSync();
      final config = jsonDecode(text) as Map<String, Object?>;

      final expectations = await runStagingScenario(
        config,
        initialize: () => _initialize(env),
        sdkVersion: _packageVersion(),
      );
      config.clear();

      final expectationsFile = File(expectationsPath);
      expectationsFile.parent.createSync(recursive: true);
      final temporary = File('${expectationsFile.path}.tmp')
        ..writeAsStringSync(
            '${const JsonEncoder.withIndent('  ').convert(expectations)}\n');
      temporary.renameSync(expectationsFile.path);
      if (expectations['scenario_error'] != null) {
        fail('scenario step failed: ${expectations['scenario_error']}');
      }
    },
    skip: configPath.isEmpty || expectationsPath.isEmpty
        ? 'set XYBRID_STAGING_CONFIG and XYBRID_STAGING_EXPECTATIONS '
            '(run_staging.py does) to run the staging scenario'
        : false,
    timeout: const Timeout(Duration(minutes: 30)),
  );
}

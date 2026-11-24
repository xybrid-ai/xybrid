// The step 6 staging scenario on a phone (lanes flutter-pixel and
// flutter-iphone), through `flutter drive`.
//
// The scenario itself is bindings/flutter/test/observability_staging_scenario.dart,
// imported by a relative path so the host lane and this one run the same
// code. Its config holds the staging key, so it never goes through the build
// or the command line: once the app is running, the lane copies it to
// <documents>/xybrid-staging/config.json (path_provider's documents
// directory: app_flutter/ in the app's data directory on Android, Documents/
// in the app container on iOS). This test polls for it every second for up
// to ten minutes, deletes it as soon as it is read, and hands
// expectations.json back through reportData; test_driver/integration_test.dart
// writes it on the host.
//
//   flutter drive --no-pub --keep-app-running -d <device> \
//     --driver test_driver/integration_test.dart \
//     --target integration_test/observability_staging_test.dart \
//     --dart-define=XYBRID_STAGING_RUN=<run>
//
// Skipped unless XYBRID_STAGING_RUN is defined. A config for another run (left
// by an earlier launch) is deleted and ignored.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';
import 'package:xybrid_flutter/xybrid.dart';

import '../../../bindings/flutter/test/observability_staging_scenario.dart';

const _run = String.fromEnvironment('XYBRID_STAGING_RUN');

/// Waits for this run's config, deleting every config it reads. A file that
/// does not parse yet is still being copied.
Future<Map<String, Object?>> _awaitConfig(File file) async {
  final deadline = DateTime.now().add(const Duration(minutes: 10));
  while (DateTime.now().isBefore(deadline)) {
    if (file.existsSync()) {
      Map<String, Object?>? config;
      try {
        config = jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      } on FormatException {
        config = null;
      }
      if (config != null) {
        file.deleteSync(); // it holds the staging key; nothing reads it again
        if (config['run'] == _run) return config;
        config.clear();
      }
    }
    await Future<void>.delayed(const Duration(seconds: 1));
  }
  if (file.existsSync()) file.deleteSync();
  throw StateError('no config for run $_run arrived at ${file.path}');
}

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'staging scenario exports a registry model run two ways',
    (tester) async {
      final documents = await getApplicationDocumentsDirectory();
      final directory = Directory('${documents.path}/xybrid-staging')
        ..createSync(recursive: true);
      final config = await _awaitConfig(File('${directory.path}/config.json'));

      final expectations =
          await runStagingScenario(config, initialize: Xybrid.init);
      config.clear();

      binding.reportData = {'expectations': expectations};
      if (expectations['scenario_error'] != null) {
        fail('scenario step failed: ${expectations['scenario_error']}');
      }
    },
    skip: _run.isEmpty,
    timeout: const Timeout(Duration(minutes: 30)),
  );
}

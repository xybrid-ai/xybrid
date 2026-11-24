// Host side of `flutter drive` for the integration tests, and the end of the
// step 6 staging lanes (integration_test/observability_staging_test.dart).
//
// With XYBRID_STAGING_EXPECTATIONS set (run_staging.py sets it; the staging
// key never passes through here), writes the expectations the app reported
// there, also when the test failed. Otherwise writes the response data where
// the stock driver does.
import 'dart:convert';
import 'dart:io';

import 'package:integration_test/integration_test_driver.dart';

/// The xybrid_flutter version the app was built from (its path dependency):
/// the binding has no runtime version call.
String? _packageVersion() {
  final pubspec = File.fromUri(
      Platform.script.resolve('../../../bindings/flutter/pubspec.yaml'));
  if (!pubspec.existsSync()) return null;
  return RegExp(r'^version:\s*(\S+)', multiLine: true)
      .firstMatch(pubspec.readAsStringSync())
      ?.group(1);
}

Future<void> main() => integrationDriver(
      timeout: const Duration(minutes: 40),
      writeResponseOnFailure: true,
      responseDataCallback: (data) async {
        final path = Platform.environment['XYBRID_STAGING_EXPECTATIONS'] ?? '';
        if (path.isEmpty) return writeResponseData(data);
        final expectations = data?['expectations'];
        if (expectations is! Map<String, dynamic>) return;
        expectations['sdk_version'] ??= _packageVersion();
        final temporary = File('$path.tmp');
        await temporary.parent.create(recursive: true);
        await temporary.writeAsString(
            '${const JsonEncoder.withIndent('  ').convert(expectations)}\n');
        await temporary.rename(path);
      },
    );

// Temporary on-device telemetry smoke test.
//
// Verifies the Android metrics pipeline end-to-end: Xybrid.init with an API
// key starts the exporter, a real inference produces telemetry events, and
// the exporter POSTs them to the (adb-reversed) local ingest server.
// Failures must show up in `adb logcat -s xybrid` via the new native logging.
//
// The MNIST model ships as a bundled asset (assets/mnist/) because the test
// network blocks outbound TCP 443 — the registry is unreachable on-device.
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:xybrid_flutter/xybrid.dart';

Future<String> materializeMnist() async {
  final dir = Directory('${Directory.systemTemp.path}/mnist');
  await dir.create(recursive: true);
  for (final name in ['model.onnx', 'model_metadata.json']) {
    final data = await rootBundle.load('assets/mnist/$name');
    await File('${dir.path}/$name').writeAsBytes(data.buffer.asUint8List());
  }
  return dir.path;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('telemetry exports after a real inference', (tester) async {
    await Xybrid.init(
      apiKey: 'sk_test_dummy_telemetry_smoke',
      ingestUrl: 'http://localhost:8787',
    );

    final modelDir = await materializeMnist();
    final model = await XybridModelLoader.fromDirectory(modelDir).load();

    // Rough "1" digit: 28x28 grayscale as a flat f32 vector, matching
    // mnist_input_envelope() in xybrid-sdk's telemetry_integration.rs.
    final pixels = List<double>.filled(28 * 28, 0.0);
    for (var row = 4; row < 24; row++) {
      for (var col = 13; col < 15; col++) {
        pixels[row * 28 + col] = 255.0;
      }
    }
    final result = await model.run(XybridEnvelope.embedding(pixels));
    expect(result, isNotNull);

    // Exporter flush interval is 5s — leave time for at least two flushes
    // plus retries so the POST (or its failure log) is observable.
    await Future<void>.delayed(const Duration(seconds: 12));
  }, timeout: const Timeout(Duration(minutes: 10)));
}

// Integration tests for Xybrid SDK on macOS.
//
// These tests run in a real Flutter environment with the native library loaded.
// Run with: flutter test integration_test/xybrid_integration_test.dart -d macos

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:xybrid/xybrid.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Xybrid SDK Integration Tests', () {
    setUpAll(() async {
      // Initialize the SDK before running tests
      await Xybrid.init();
    });

    test('can create text envelope', () {
      final envelope = XybridEnvelope.text('hello');
      expect(envelope, isNotNull);
    });

    test('can create text envelope with options', () {
      final envelope = XybridEnvelope.text(
        'hello',
        voiceId: 'default',
        speed: 1.0,
      );
      expect(envelope, isNotNull);
    });

    test('can create audio envelope', () {
      final envelope = XybridEnvelope.audio(
        bytes: [0, 0, 0, 0],
        sampleRate: 16000,
        channels: 1,
      );
      expect(envelope, isNotNull);
    });

    test('can create embedding envelope', () {
      final envelope = XybridEnvelope.embedding([0.1, 0.2, 0.3]);
      expect(envelope, isNotNull);
    });
  });
}

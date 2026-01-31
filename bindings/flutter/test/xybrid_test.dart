// Minimal Xybrid SDK tests for macOS.
//
// These tests verify the Dart API can be imported and types are accessible.
// Full integration tests with envelope creation require a Flutter integration
// test environment where the native library can be loaded.
//
// To run integration tests that call native code:
//   flutter test integration_test/xybrid_integration_test.dart
// (See example/ for a runnable app with native library)

import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid/xybrid.dart';

void main() {
  group('Library imports', () {
    test('imports library without error', () {
      // This test verifies the library can be imported successfully.
      // If this test runs, the import at the top of this file succeeded.
      expect(true, isTrue);
    });
  });

  group('Library exports', () {
    test('exports XybridEnvelope', () {
      // Verify the type is exported and accessible
      expect(XybridEnvelope, isNotNull);
    });

    test('exports XybridModelLoader', () {
      expect(XybridModelLoader, isNotNull);
    });

    test('exports XybridModel', () {
      expect(XybridModel, isNotNull);
    });

    test('exports XybridResult', () {
      expect(XybridResult, isNotNull);
    });

    test('exports XybridPipeline', () {
      expect(XybridPipeline, isNotNull);
    });

    test('exports XybridException', () {
      expect(XybridException, isNotNull);
    });

    test('exports Xybrid', () {
      expect(Xybrid, isNotNull);
    });
  });

  group('XybridEnvelope construction', () {
    // Note: Envelope construction requires FRB initialization, which requires
    // loading the native library. In unit tests, the native library isn't
    // loaded, so these tests verify the expected behavior.
    //
    // For full integration tests that can create envelopes, use:
    //   flutter test integration_test/ -d macos

    test('text envelope requires initialization', () {
      // Verify that calling XybridEnvelope.text() throws StateError
      // when FRB is not initialized (expected behavior in unit tests)
      expect(
        () => XybridEnvelope.text('hello'),
        throwsA(isA<StateError>()),
      );
    });

    test('audio envelope requires initialization', () {
      expect(
        () => XybridEnvelope.audio(
          bytes: [0, 0, 0, 0],
          sampleRate: 16000,
          channels: 1,
        ),
        throwsA(isA<StateError>()),
      );
    });

    test('embedding envelope requires initialization', () {
      expect(
        () => XybridEnvelope.embedding([0.1, 0.2, 0.3]),
        throwsA(isA<StateError>()),
      );
    });
  });
}

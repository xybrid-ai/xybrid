/// SDK initialization for Xybrid.
///
/// This provides the main entry point for initializing the Xybrid runtime.
library;

import 'dart:async';

import 'rust/frb_generated.dart';

/// Main entry point for the Xybrid SDK.
///
/// Call [Xybrid.init] once before using any other Xybrid functionality.
///
/// ```dart
/// void main() async {
///   await Xybrid.init();
///
///   // Now you can use Xybrid
///   final loader = XybridModelLoader.fromRegistry('kokoro-82m');
///   final model = await loader.load();
///   // ...
/// }
/// ```
class Xybrid {
  static bool _initialized = false;
  static final Completer<void> _initCompleter = Completer<void>();
  static bool _initializing = false;

  /// Private constructor to prevent instantiation.
  Xybrid._();

  /// Initialize the Xybrid runtime.
  ///
  /// This must be called once before using any Xybrid functionality.
  /// It is safe to call this multiple times - subsequent calls are no-ops.
  ///
  /// Example:
  /// ```dart
  /// void main() async {
  ///   await Xybrid.init();
  ///   // SDK is ready to use
  /// }
  /// ```
  ///
  /// Throws an exception if initialization fails (e.g., native library not found).
  static Future<void> init() async {
    // Fast path: already initialized
    if (_initialized) {
      return;
    }

    // Handle concurrent initialization attempts
    if (_initializing) {
      return _initCompleter.future;
    }

    _initializing = true;

    try {
      await XybridRustLib.init();
      _initialized = true;
      _initCompleter.complete();
    } catch (e) {
      _initCompleter.completeError(e);
      _initializing = false;
      rethrow;
    }
  }

  /// Check if the SDK has been initialized.
  ///
  /// Returns `true` if [init] has been called successfully.
  static bool get isInitialized => _initialized;
}

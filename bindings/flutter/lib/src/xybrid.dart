/// SDK initialization for Xybrid.
///
/// This provides the main entry point for initializing the Xybrid runtime.
library;

import 'dart:async';
import 'dart:io' show Platform;

import 'package:battery_plus/battery_plus.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:xybrid_flutter/src/rust/api/device.dart';
import 'package:xybrid_flutter/src/rust/api/sdk_client.dart';
import 'package:path_provider/path_provider.dart';
import 'rust/frb_generated.dart';

import '../xybrid.dart';
import 'device_snapshot.dart';

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

  /// Battery state subscription retained for the process lifetime so
  /// the underlying platform stream isn't garbage-collected. `null`
  /// outside iOS / Android — desktop platforms use xybrid-core's
  /// in-process pollers and don't subscribe here.
  ///
  /// The two lints below are intentional: the field exists solely
  /// to anchor the subscription against GC, and the SDK is a
  /// process-lifetime singleton (matches the Kotlin
  /// `Xybrid.init(context)` and Swift `Xybrid.initialize()` shapes —
  /// neither of which exposes a teardown either).
  // ignore: cancel_subscriptions, unused_field
  static StreamSubscription<BatteryState>? _batterySubscription;

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
      // On iOS and macOS, we use static linking with -force_load.
      // The Rust symbols are linked directly into the main executable,
      // so we must use DynamicLibrary.process() to look them up.
      ExternalLibrary? externalLibrary;
      if (Platform.isIOS || Platform.isMacOS) {
        externalLibrary = ExternalLibrary.process(iKnowHowToUseIt: true);
      }

      await XybridRustLib.init(externalLibrary: externalLibrary);

      if (Platform.isAndroid) {
        final appDir = await getApplicationSupportDirectory();
        final cacheDir = '${appDir.path}/xybrid/models';
        XybridSdkClient.initSdkCacheDir(cacheDir: cacheDir);
      }

      await _registerPlatformObservers();

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

  static void setApiKey(String apiKey) {
    XybridSdkClient.setApiKey(apiKey: apiKey);
  }

  /// Check if a model is cached locally (extracted and ready to use).
  ///
  /// This is a pure filesystem check — no network access required.
  /// Returns `true` if the model has been previously downloaded and extracted.
  ///
  /// Use this to check model availability without triggering a download:
  /// ```dart
  /// if (Xybrid.isModelCached('kokoro-82m')) {
  ///   // Model is ready, can load instantly
  ///   final model = await Xybrid.model('kokoro-82m').load();
  /// } else {
  ///   // Model needs to be downloaded first
  /// }
  /// ```
  static bool isModelCached(String modelId) {
    return XybridSdkClient.isModelCached(modelId: modelId);
  }

  static void initTelemetry() {
    // TODO - Implement telemetry
    // XybridSdkClient.enableTelemetry();
    throw UnimplementedError();
  }

  /// Read the routing-engine's current view of device state.
  ///
  /// Returns the same [DeviceSnapshot] the engine reads internally on
  /// each routing decision — battery + thermal from the platform
  /// pollers / host pushes, CPU + memory from sysinfo. Force-refreshes
  /// on every call so a diagnostics surface that polls at ~1 Hz sees
  /// fresh data each tick. The refresh cost is bounded at `< 1 ms` on
  /// a warm monitor.
  ///
  /// Intended for app-side diagnostics views ("what does the routing
  /// engine see on this device right now?"). Production code should
  /// not poll this — the engine reads it internally.
  static DeviceSnapshot currentDeviceSnapshot() {
    return DeviceSnapshot.fromFfi(XybridDevice.currentSnapshot());
  }

  /// Create a ModelLoader for the specified model.
  ///
  /// This is the entry point for the **Loader → Model → Run** pattern.
  ///
  /// ## From Registry (recommended for production)
  /// ```dart
  /// final loader = Xybrid.model(modelId: 'whisper-tiny');
  /// final model = await loader.load();
  /// ```
  static XybridModelLoader model(String modelId) =>
      XybridModelLoader.fromRegistry(modelId);

  /// Create a PipelineRef for multi-stage inference pipelines.
  ///
  /// Pipelines orchestrate multiple models in sequence (e.g., ASR → LLM → TTS).
  ///
  /// ## From YAML Content
  /// ```dart
  /// final yaml = '''
  /// name: "Voice Assistant"
  /// stages:
  ///   - whisper-tiny
  ///   - llm-stage
  ///   - kokoro-tts
  /// ''';
  /// final ref = Xybrid.pipeline(yaml: yaml);
  /// final pipeline = await ref.load();
  /// ```
  ///
  /// ## From File
  /// ```dart
  /// final ref = Xybrid.pipeline(filePath: 'pipelines/voice-assistant.yaml');
  /// ```
  static XybridPipeline pipeline({String? yaml, String? filePath}) {
    final hasYaml = yaml != null;
    final hasFile = filePath != null;

    if (!hasYaml && !hasFile) {
      throw ArgumentError('Must provide either yaml or filePath');
    }

    if (hasYaml && hasFile) {
      throw ArgumentError('Only one source can be specified: yaml or filePath');
    }

    if (hasYaml) {
      return XybridPipeline.fromYaml(yaml);
    }

    return XybridPipeline.fromFile(filePath!);
  }

  /// Subscribe to OS-level battery state on mobile platforms and
  /// forward each reading into the routing engine via the FRB
  /// push-state surface. Desktop platforms (macOS / Linux / Windows)
  /// use xybrid-core's in-process pollers, and Flutter on iOS gets
  /// thermal in-Rust via NSProcessInfo, so this only runs on iOS and
  /// Android.
  ///
  /// `battery_plus` exposes a snapshot getter (`batteryLevel`) and a
  /// stream of high-level state changes (`onBatteryStateChanged`) but
  /// no continuous level stream. The level is therefore re-read on
  /// each state-change event (plug/unplug, charge full, etc.) plus
  /// once at init so the first cache miss isn't blind. Routing
  /// decisions are bucket-based (low / mid / high), so the resulting
  /// granularity is sufficient — the cache TTL inside `ResourceMonitor`
  /// is the limiting factor on freshness regardless.
  ///
  /// Thermal on Flutter Android is currently a gap: Dart has no
  /// cross-platform package wrapping `PowerManager.OnThermalStatusChangedListener`,
  /// and adding a Kotlin plugin layer is out of scope here. Consumer
  /// apps that need it can call `XybridDevice.setThermalState(...)`
  /// directly from their own platform channel.
  static Future<void> _registerPlatformObservers() async {
    if (!(Platform.isAndroid || Platform.isIOS)) {
      return;
    }
    final battery = Battery();
    await _pushBatteryLevel(battery);
    _batterySubscription = battery.onBatteryStateChanged.listen((_) {
      // Re-read the level on each state transition; battery_plus
      // doesn't expose a level stream so this is the canonical
      // refresh point.
      unawaited(_pushBatteryLevel(battery));
    });
  }

  static Future<void> _pushBatteryLevel(Battery battery) async {
    try {
      final level = await battery.batteryLevel;
      // battery_plus returns 0..=100 already; clamp defensively in case
      // a future platform impl reports out-of-range values rather than
      // throwing.
      final pct = level.clamp(0, 100);
      XybridDevice.setBatteryLevel(percent: pct);
    } catch (_) {
      // Sensor unavailable (emulator, sandboxed context, etc.) —
      // surface as "unknown" rather than a fake reading. The routing
      // engine treats `None` as "no signal" and falls back to other
      // evidence.
      XybridDevice.clearBatteryLevel();
    }
  }
}

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

import 'device_snapshot.dart';
import 'model_loader.dart';
import 'pipeline.dart';
import 'runtime_config.dart';

/// Main entry point for the Xybrid SDK.
///
/// Call [Xybrid.init] once before using any other Xybrid functionality.
/// Inference runs locally whether or not you pass an API key; supplying
/// one starts the telemetry exporter so your runs show up on the
/// dashboard. Get a free key at <https://dashboard.xybrid.dev>.
///
/// ```dart
/// void main() async {
///   // Anonymous — local inference, telemetry disabled
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
  /// All parameters are optional. Without an `apiKey`, the SDK runs fully
  /// on-device and telemetry is disabled — the first inference logs a
  /// one-shot hint pointing at the dashboard (suppress with the
  /// `XYBRID_QUIET=1` environment variable). Pass `apiKey` to start the
  /// platform telemetry exporter automatically; `ingestUrl` overrides the
  /// destination for a self-hosted dashboard, and `resourceTelemetry`
  /// enables CPU/memory sampling.
  ///
  /// ```dart
  /// void main() async {
  ///   // Anonymous — local inference only
  ///   await Xybrid.init();
  ///
  ///   // Authenticated — telemetry flows to the dashboard
  ///   await Xybrid.init(
  ///     apiKey: const String.fromEnvironment('XYBRID_API_KEY'),
  ///   );
  /// }
  /// ```
  ///
  /// Get a free key at <https://dashboard.xybrid.dev>.
  ///
  /// Throws an exception if initialization fails (e.g., native library not found).
  static Future<void> init({
    String? apiKey,
    String? gatewayUrl,
    String? ingestUrl,
    String? resourceTelemetry,
  }) async {
    // Fast path: already initialized
    if (_initialized) {
      _applyRuntimeConfig(
        apiKey: apiKey,
        gatewayUrl: gatewayUrl,
        ingestUrl: ingestUrl,
        resourceTelemetry: resourceTelemetry,
      );
      return;
    }

    // Handle concurrent initialization attempts
    if (_initializing) {
      await _initCompleter.future;
      _applyRuntimeConfig(
        apiKey: apiKey,
        gatewayUrl: gatewayUrl,
        ingestUrl: ingestUrl,
        resourceTelemetry: resourceTelemetry,
      );
      return;
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
      _applyRuntimeConfig(
        apiKey: apiKey,
        gatewayUrl: gatewayUrl,
        ingestUrl: ingestUrl,
        resourceTelemetry: resourceTelemetry,
      );
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

  static void setGatewayUrl(String gatewayUrl) {
    XybridRuntimeConfig.gatewayUrl = gatewayUrl;
    XybridSdkClient.setGatewayUrl(gatewayUrl: gatewayUrl);
  }

  static void applyDebugMemoryPressure() {
    XybridDevice.applyDebugMemoryPressure();
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

  /// Legacy telemetry entry point. Prefer passing `apiKey` (and, for a
  /// self-hosted dashboard, `ingestUrl`) to [init] — that starts the
  /// exporter as part of normal initialization.
  ///
  /// Retained for backward compatibility and advanced callers that must
  /// start telemetry after [init] has already run. Behaves identically to
  /// the bundled path: the Rust layer holds a process-wide once-guard, so
  /// whichever route fires first wins and later calls are safe no-ops.
  /// There is no reconfigure path — changing `endpoint` or `apiKey`
  /// requires restarting the process.
  ///
  /// `endpoint` is the platform ingest URL (e.g. `https://ingest.xybrid.dev`
  /// in production, or `http://192.168.1.78:8081` for a local dashboard on
  /// the host machine). `apiKey` authenticates the sender. Must be called
  /// after [init] has completed.
  static void initTelemetry(
      {required String endpoint, required String apiKey}) {
    if (!_initialized) {
      throw StateError('Xybrid.init() must complete before initTelemetry()');
    }
    XybridSdkClient.initTelemetry(endpoint: endpoint, apiKey: apiKey);
  }

  /// Whether the process-wide telemetry exporter is running.
  ///
  /// Reads from the Rust once-flag, so this stays correct across Flutter
  /// hot-restart (which would reset any Dart-side state) and second
  /// isolates that didn't themselves call [initTelemetry].
  static bool get isTelemetryInitialized =>
      XybridSdkClient.isTelemetryInitialized();

  /// Runtime features compiled into the native xybrid library.
  static List<String> runtimeFeatures() => XybridSdkClient.runtimeFeatures();

  /// Whether a named runtime feature was compiled into the native library.
  static bool supportsRuntimeFeature(String feature) =>
      runtimeFeatures().contains(feature);

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

  static void _applyRuntimeConfig({
    String? apiKey,
    String? gatewayUrl,
    String? ingestUrl,
    String? resourceTelemetry,
  }) {
    final key = _nonEmpty(apiKey);
    final gateway = _nonEmpty(gatewayUrl);
    final ingest = _nonEmpty(ingestUrl);
    final resourceMode = _nonEmpty(resourceTelemetry);

    if (key != null) {
      setApiKey(key);
    }
    if (gateway != null) {
      setGatewayUrl(gateway);
    }
    // An API key alone starts the exporter; the Rust layer defaults the
    // ingest URL to the production endpoint when `ingest` is null, so
    // `Xybrid.init(apiKey: ...)` lights up the dashboard without the caller
    // needing to know the ingest URL.
    if (key != null) {
      XybridSdkClient.configurePlatformTelemetry(
        apiKey: key,
        ingestUrl: ingest,
        resourceTelemetry: resourceMode,
      );
    }
  }

  static String? _nonEmpty(String? value) {
    final trimmed = value?.trim();
    return trimmed == null || trimmed.isEmpty ? null : trimmed;
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

/// Public Dart types for the routing-engine's view of device state.
///
/// Mirrors `xybrid_sdk::ResourceSnapshot`. Optional fields are `null`
/// when the underlying sensor isn't available on the running platform —
/// a `null` here is what the routing engine reads as "no signal," and
/// downstream gates treat that as "do not penalize." The FRB-generated
/// types live in `src/rust/api/device.dart`; this file is the
/// host-facing surface so apps don't depend on the generated symbols.
library;

import 'rust/api/device.dart' as ffi;

/// Derived memory-pressure classification from `available / total`.
///
/// `unknown` means the runtime couldn't compute the ratio — either
/// sysinfo refused to answer or the host hasn't pushed a memory-warning
/// observer. The routing engine treats `unknown` as "no penalty," not
/// "memory is fine."
enum MemoryPressure { unknown, normal, warn, critical }

/// Thermal pressure classification matching the desktop pollers'
/// Celsius bands (Normal < 60, Warm 60–70, Hot 70–80, Critical >= 80).
enum ThermalState { normal, warm, hot, critical }

/// Snapshot of the routing-engine's current view of device state.
///
/// Read via [Xybrid.currentDeviceSnapshot]. Intended for diagnostics
/// surfaces only — production code should not poll this. The engine
/// reads it internally on each routing decision.
class DeviceSnapshot {
  /// Global CPU usage at sample time, 0–100. `null` if sysinfo couldn't
  /// produce a reading on the running platform.
  final double? cpuPct;

  /// Resident set size of the current process in MB. `null` when the
  /// runtime couldn't resolve the current PID.
  final int? processRssMb;

  /// Available system memory in MB at sample time.
  final int? availableMemMb;

  /// Total system memory in MB.
  final int? totalMemMb;

  /// Derived memory pressure. See [MemoryPressure] for thresholds.
  final MemoryPressure memoryPressure;

  /// Current thermal state.
  final ThermalState thermalState;

  /// Battery charge percent, 0–100. `null` when no battery is present
  /// or the host hasn't pushed a level yet.
  final int? batteryPct;

  /// Monotonic-ish capture timestamp (epoch millis). Useful for
  /// detecting whether a UI is reading stale cached snapshots.
  final DateTime capturedAt;

  const DeviceSnapshot({
    this.cpuPct,
    this.processRssMb,
    this.availableMemMb,
    this.totalMemMb,
    required this.memoryPressure,
    required this.thermalState,
    this.batteryPct,
    required this.capturedAt,
  });

  /// Adapter from the FRB-generated snapshot struct. Intentionally
  /// package-private — apps reach for [Xybrid.currentDeviceSnapshot]
  /// which calls this internally.
  factory DeviceSnapshot.fromFfi(ffi.FfiResourceSnapshot s) {
    return DeviceSnapshot(
      cpuPct: s.cpuPct,
      processRssMb: s.processRssMb,
      availableMemMb: s.availableMemMb,
      totalMemMb: s.totalMemMb,
      memoryPressure: _memoryPressureFromFfi(s.memoryPressure),
      thermalState: _thermalStateFromFfi(s.thermalState),
      batteryPct: s.batteryPct,
      capturedAt: DateTime.fromMillisecondsSinceEpoch(s.capturedAtMs.toInt()),
    );
  }
}

MemoryPressure _memoryPressureFromFfi(ffi.FfiMemoryPressure value) {
  switch (value) {
    case ffi.FfiMemoryPressure.unknown:
      return MemoryPressure.unknown;
    case ffi.FfiMemoryPressure.normal:
      return MemoryPressure.normal;
    case ffi.FfiMemoryPressure.warn:
      return MemoryPressure.warn;
    case ffi.FfiMemoryPressure.critical:
      return MemoryPressure.critical;
  }
}

ThermalState _thermalStateFromFfi(ffi.FfiThermalState value) {
  switch (value) {
    case ffi.FfiThermalState.normal:
      return ThermalState.normal;
    case ffi.FfiThermalState.warm:
      return ThermalState.warm;
    case ffi.FfiThermalState.hot:
      return ThermalState.hot;
    case ffi.FfiThermalState.critical:
      return ThermalState.critical;
  }
}

/// xybrid addition — not part of upstream cargokit.
///
/// Wording and pacing for precompiled-binary download logs. Kept free of I/O
/// so the rules are unit-testable; `ArtifactProvider` does the actual logging.
library;

/// Formats [bytes] in decimal units (1 MB = 1,000,000 bytes), the convention
/// GitHub uses when listing release assets, so a logged size can be compared
/// with the release page at a glance.
String formatByteSize(int bytes) {
  if (bytes < 1000) {
    return '$bytes B';
  }
  if (bytes < 1000 * 1000) {
    return '${(bytes / 1000).toStringAsFixed(1)} kB';
  }
  if (bytes < 1000 * 1000 * 1000) {
    return '${(bytes / (1000 * 1000)).toStringAsFixed(1)} MB';
  }
  return '${(bytes / (1000 * 1000 * 1000)).toStringAsFixed(2)} GB';
}

/// Tracks one download and decides when it is worth a progress line.
///
/// A precompiled static library can exceed 100 MB. Without periodic output the
/// build step that fetches it sits silent for minutes, which reads as "the
/// Rust engine is compiling" to anyone watching the log.
class DownloadProgress {
  DownloadProgress({
    required this.label,
    required this.totalBytes,
    this.interval = const Duration(seconds: 10),
  });

  /// What is being downloaded, as shown in every line (the asset file name).
  final String label;

  /// Size announced by the server, or `null` when it sent no Content-Length.
  final int? totalBytes;

  /// Minimum time between two progress lines.
  final Duration interval;

  int _receivedBytes = 0;
  Duration _lastReport = Duration.zero;

  int get receivedBytes => _receivedBytes;

  /// Records [chunkLength] more bytes received at [elapsed] since the start.
  ///
  /// Returns a progress line when [interval] has passed since the last one,
  /// otherwise `null`. Fast downloads therefore produce no progress lines at
  /// all — only the announcement and the summary.
  String? add(int chunkLength, Duration elapsed) {
    _receivedBytes += chunkLength;
    if (elapsed - _lastReport < interval) {
      return null;
    }
    _lastReport = elapsed;
    final total = totalBytes;
    if (total == null || total <= 0) {
      return '$label: ${formatByteSize(_receivedBytes)} so far';
    }
    final percent = (_receivedBytes * 100 / total).floor().clamp(0, 100);
    return '$label: ${formatByteSize(_receivedBytes)} of '
        '${formatByteSize(total)} ($percent%)';
  }

  /// The completion line: size, wall time and average throughput.
  String summary(Duration elapsed) {
    final seconds = elapsed.inMilliseconds / 1000;
    final took =
        seconds < 10 ? '${seconds.toStringAsFixed(1)}s' : '${seconds.round()}s';
    final rate = elapsed.inMilliseconds > 0
        ? ' (${formatByteSize((_receivedBytes / seconds).round())}/s)'
        : '';
    return 'Downloaded $label: ${formatByteSize(_receivedBytes)} in $took$rate';
  }
}

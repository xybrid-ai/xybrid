// On-device check for registry downloads (#599).
//
// `lfm2.5-350m` is the model from the OfflineGPT report: its registry entry
// declares `size_bytes: 0`, so before #599 the bar sat at 0% until the whole
// 229 MB had landed. This test downloads it fresh and checks that:
//
// - a 0-byte frame arrives first (the app can show "connecting"),
// - the total comes from the server, so the percentage is real,
// - progress never moves backwards,
// - the file arrives whole: the model loads and generates.
//
// No API key is passed: native logs (`adb logcat -s xybrid`, or the
// `dev.xybrid.sdk` subsystem on iOS) must still appear.
//
//   flutter test integration_test/download_progress_test.dart -d <device>
//
// Lines prefixed `DLCHECK` are the report. They also go to `dlcheck.txt` on
// the device, for when the console detaches from it.
// Pass `--dart-define=MODEL=<id>` to try another registry model.
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:xybrid_flutter/xybrid.dart';

const modelId = String.fromEnvironment('MODEL', defaultValue: 'lfm2.5-350m');

// iOS may empty an app's tmp/ once it is backgrounded, so the report goes to
// Documents there (pull it with `xcrun devicectl device copy from`).
final _reportFile = File(
  Platform.isIOS
      ? '${Platform.environment['HOME']}/Documents/dlcheck.txt'
      : '${Directory.systemTemp.path}/dlcheck.txt',
);

void report(String line) {
  print('DLCHECK $line');
  _reportFile.writeAsStringSync('$line\n', mode: FileMode.append, flush: true);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'registry download reports real progress and arrives whole',
    (tester) async {
      if (_reportFile.existsSync()) _reportFile.deleteSync();
      await Xybrid.init();
      final removed = await Xybrid.removeCachedModel(modelId);
      report('model=$modelId removed_cache_entries=$removed');

      final clock = Stopwatch()..start();
      final frames = <LoadProgress>[];
      String? error;
      var completed = false;
      double? totalKnownAt;
      var printedAt = -1.0;

      await for (final event in XybridModelLoader.fromRegistry(
        modelId,
      ).loadWithProgress()) {
        switch (event) {
          case LoadProgress():
            frames.add(event);
            final seconds = clock.elapsedMilliseconds / 1000;
            if (event.totalBytes != null && totalKnownAt == null) {
              totalKnownAt = seconds;
            }
            // First frames, then roughly every 10%, keeps the log readable.
            if (frames.length <= 3 || event.progress - printedAt >= 0.1) {
              printedAt = event.progress;
              report(
                't=${seconds.toStringAsFixed(1)}s '
                'progress=${(event.progress * 100).toStringAsFixed(1)}% '
                'bytes=${event.downloadedBytes} total=${event.totalBytes}',
              );
            }
          case LoadComplete():
            completed = true;
          case LoadError(:final message):
            error = message;
        }
      }
      final downloadSeconds = clock.elapsedMilliseconds / 1000;
      report(
        'done in ${downloadSeconds.toStringAsFixed(1)}s '
        'frames=${frames.length} total_known_at=${totalKnownAt}s '
        'completed=$completed error=$error',
      );

      report('assertions start');
      expect(error, isNull);
      expect(completed, isTrue);
      expect(frames, isNotEmpty);
      expect(frames.first.downloadedBytes, 0, reason: 'start frame first');
      expect(totalKnownAt, isNotNull, reason: 'server size never adopted');
      expect(
        frames.any((f) => f.progress > 0.05 && f.progress < 0.95),
        isTrue,
        reason: 'no mid-download percentage was reported',
      );
      for (var i = 1; i < frames.length; i++) {
        expect(
          frames[i].progress >= frames[i - 1].progress &&
              frames[i].downloadedBytes >= frames[i - 1].downloadedBytes,
          isTrue,
          reason: 'progress rewound at frame $i',
        );
      }

      final model = await XybridModelLoader.fromRegistry(modelId).load();
      final result = await model.run(
        XybridEnvelope.text('Reply with one short sentence: hello.'),
        config: const GenerationConfig(maxTokens: 16),
      );
      report('generated="${result.text}"');
      expect(result.text, isNotNull);
      expect(result.text!.trim(), isNotEmpty);
      report('PASS');
    },
    timeout: const Timeout(Duration(minutes: 30)),
  );
}

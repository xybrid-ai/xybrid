import 'package:build_tool/src/download_progress.dart';
import 'package:test/test.dart';

void main() {
  group('formatByteSize', () {
    test('uses decimal units like the GitHub release page', () {
      expect(formatByteSize(0), '0 B');
      expect(formatByteSize(999), '999 B');
      expect(formatByteSize(64174), '64.2 kB');
      expect(formatByteSize(26893720), '26.9 MB');
      expect(formatByteSize(180639448), '180.6 MB');
      expect(formatByteSize(2500000000), '2.50 GB');
    });
  });

  group('DownloadProgress', () {
    test('stays quiet until the interval has passed', () {
      final progress = DownloadProgress(label: 'lib.a', totalBytes: 100000000);

      expect(progress.add(10000000, const Duration(seconds: 3)), isNull);
      expect(progress.add(10000000, const Duration(seconds: 9)), isNull);
      expect(progress.receivedBytes, 20000000);
    });

    test('reports size, total and percent once the interval has passed', () {
      final progress = DownloadProgress(label: 'lib.a', totalBytes: 180639448);

      progress.add(50000000, const Duration(seconds: 4));
      final line = progress.add(22400000, const Duration(seconds: 10));

      expect(line, 'lib.a: 72.4 MB of 180.6 MB (40%)');
    });

    test('waits a full interval between two reports', () {
      final progress = DownloadProgress(label: 'lib.a', totalBytes: 100000000);

      expect(progress.add(1000000, const Duration(seconds: 10)), isNotNull);
      expect(progress.add(1000000, const Duration(seconds: 15)), isNull);
      expect(progress.add(1000000, const Duration(seconds: 20)), isNotNull);
    });

    test('reports bytes so far when the server sent no length', () {
      final progress = DownloadProgress(label: 'lib.a', totalBytes: null);

      final line = progress.add(5000000, const Duration(seconds: 10));

      expect(line, 'lib.a: 5.0 MB so far');
    });

    test('never reports more than 100 percent', () {
      final progress = DownloadProgress(label: 'lib.a', totalBytes: 1000);

      final line = progress.add(5000, const Duration(seconds: 10));

      expect(line, contains('(100%)'));
    });

    test('summarises size, duration and throughput', () {
      final progress = DownloadProgress(label: 'lib.a', totalBytes: 180639448);
      progress.add(180639448, const Duration(seconds: 1));

      expect(progress.summary(const Duration(seconds: 42)),
          'Downloaded lib.a: 180.6 MB in 42s (4.3 MB/s)');
    });

    test('summary keeps a decimal for short downloads and survives zero time',
        () {
      final progress = DownloadProgress(label: 'lib.so', totalBytes: 26893720);
      progress.add(26893720, Duration.zero);

      expect(progress.summary(const Duration(milliseconds: 2500)),
          'Downloaded lib.so: 26.9 MB in 2.5s (10.8 MB/s)');
      expect(progress.summary(Duration.zero),
          'Downloaded lib.so: 26.9 MB in 0.0s');
    });
  });
}

// `source_build_possible.sh` is the shell twin of `_sourceBuildBlocker` in
// artifacts_provider.dart: native build scripts ask it whether work that only
// a source build needs can be skipped. These layouts are the ones that rule
// distinguishes. CI runs this under dash, consumers under macOS /bin/sh.
import 'dart:io';

import 'package:path/path.dart' as path;
import 'package:test/test.dart';

void main() {
  late Directory temp;
  final script = path.normalize(
      path.join(Directory.current.path, '..', 'source_build_possible.sh'));

  setUp(() {
    temp = Directory.systemTemp.createTempSync('source_build_possible_test');
  });

  tearDown(() => temp.deleteSync(recursive: true));

  String crate(String relativeDir, String manifest) {
    final dir = Directory(path.join(temp.path, relativeDir))
      ..createSync(recursive: true);
    File(path.join(dir.path, 'Cargo.toml')).writeAsStringSync(manifest);
    return dir.path;
  }

  ProcessResult run(String manifestDir) =>
      Process.runSync('sh', [script, manifestDir]);

  test('the script exists next to build_pod.sh', () {
    expect(File(script).existsSync(), isTrue);
  });

  test('a self-contained crate can be built from source', () {
    final dir = crate('plain', '[package]\nname = "plain"\n');

    expect(run(dir).exitCode, 0);
  });

  test('a workspace member under its workspace root can be built', () {
    crate('repo', '[workspace]\nmembers = ["bindings/rust"]\n');
    crate('repo/crates/sdk', '[package]\nname = "sdk"\n');
    final dir = crate(
        'repo/bindings/rust',
        '[package]\nname = "member"\nedition.workspace = true\n'
            '[dependencies]\nsdk = { path = "../../crates/sdk" }\n');

    expect(run(dir).exitCode, 0);
  });

  test('the published layout is refused: no workspace root above', () {
    final dir = crate(
        'pkg/rust', '[package]\nname = "member"\nedition.workspace = true\n');

    final result = run(dir);

    expect(result.exitCode, 1);
    expect(result.stdout, contains('workspace root'));
  });

  test('the published layout is refused: path dependencies are missing', () {
    final dir = crate(
        'pkg/rust',
        '[package]\nname = "member"\n[dependencies]\n'
            'sdk = { path = "../../../crates/xybrid-sdk" }\n');

    final result = run(dir);

    expect(result.exitCode, 1);
    expect(result.stdout, contains('../../../crates/xybrid-sdk'));
  });

  test('a directory without a manifest is refused', () {
    final dir = Directory(path.join(temp.path, 'empty'))..createSync();

    expect(run(dir.path).exitCode, 1);
  });

  test('agrees with the real monorepo crate', () {
    final rustDir =
        path.normalize(path.join(Directory.current.path, '..', '..', 'rust'));

    expect(run(rustDir).exitCode, 0);
  });
}

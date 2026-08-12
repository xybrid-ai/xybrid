import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/result.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _metrics = FfiInferenceMetrics(totalMs: 10, stageLatenciesMs: []);

FfiResult _ffiResult({String? text, String? reasoningContent}) => FfiResult(
      success: true,
      text: text,
      reasoningContent: reasoningContent,
      latencyMs: 10,
      executionTarget: FfiExecutionTarget.local,
      metrics: _metrics,
    );

void main() {
  test('reasoning content surfaces separately from the answer', () {
    final result = XybridResult.fromFfi(
      _ffiResult(text: 'Paris', reasoningContent: 'The capital of France is…'),
    );

    expect(result.text, 'Paris');
    expect(result.reasoningContent, 'The capital of France is…');
  });

  test('reasoning content is null when the model emitted none', () {
    final result = XybridResult.fromFfi(_ffiResult(text: 'Paris'));

    expect(result.text, 'Paris');
    expect(result.reasoningContent, isNull);
  });
}

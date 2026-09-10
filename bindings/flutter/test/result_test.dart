import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/model.dart';
import 'package:xybrid_flutter/src/rust/api/result.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _metrics = FfiInferenceMetrics(totalMs: 10, stageLatenciesMs: []);

FfiResult _ffiResult({
  String? text,
  String? reasoningContent,
  List<FfiToolCall> toolCalls = const [],
}) => FfiResult(
      success: true,
      text: text,
      reasoningContent: reasoningContent,
      latencyMs: 10,
      executionTarget: FfiExecutionTarget.local,
      metrics: _metrics,
      toolCalls: toolCalls,
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

  test('tool calls surface as the public Dart shape', () {
    final result = XybridResult.fromFfi(
      _ffiResult(
        text: 'calling a tool',
        toolCalls: const [
          FfiToolCall(
            id: 'call_0',
            name: 'get_weather',
            argumentsJson: '{"city":"Paris"}',
          ),
        ],
      ),
    );

    expect(result.hasToolCalls, isTrue);
    expect(result.toolCalls, hasLength(1));
    expect(result.toolCalls.single.id, 'call_0');
    expect(result.toolCalls.single.name, 'get_weather');
    expect(result.toolCalls.single.argumentsJson, '{"city":"Paris"}');
    // The raw block stays in the text — parsing is additive.
    expect(result.text, 'calling a tool');
  });

  test('a response without tool calls exposes an empty list', () {
    final result = XybridResult.fromFfi(_ffiResult(text: 'Paris'));

    expect(result.hasToolCalls, isFalse);
    expect(result.toolCalls, isEmpty);
  });
}

import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

/// The raw output of a tools-bearing turn: display text plus the protocol
/// block. Only `rawText` carries it — `cumulativeText` is what the UI painted,
/// with the block suppressed.
const _rawTurn =
    'checking<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>';

void main() {
  test('a mid-stream token carries no tool calls', () {
    final token = StreamToken(
      token: 'chec',
      index: 0,
      cumulativeText: 'chec',
      isFinal: false,
    );

    expect(token.hasToolCalls, isFalse);
    expect(token.toolCalls, isEmpty);
    expect(token.rawText, isNull);
  });

  test('the terminal token hands back typed calls without parsing text', () {
    final token = StreamToken(
      token: '',
      index: 4,
      cumulativeText: 'checking',
      isFinal: true,
      finishReason: 'tool_calls',
      toolCalls: const [
        ToolCall(
          id: 'call_0',
          name: 'get_weather',
          argumentsJson: '{"city":"Paris"}',
        ),
      ],
      rawText: _rawTurn,
    );

    expect(token.hasToolCalls, isTrue);
    expect(token.finishReason, 'tool_calls');
    expect(token.toolCalls.single.name, 'get_weather');
    expect(token.toolCalls.single.argumentsJson, '{"city":"Paris"}');

    // The streamed text never contained the protocol block...
    expect(token.cumulativeText, isNot(contains('<|tool_call_start|>')));
    // ...but the continuation needs it verbatim, which is what rawText is for.
    expect(token.rawText, _rawTurn);
  });

  test('rawText is what the continuation envelope replays', () {
    final token = StreamToken(
      token: '',
      index: 4,
      cumulativeText: 'checking',
      isFinal: true,
      finishReason: 'tool_calls',
      toolCalls: const [
        ToolCall(
          id: 'call_0',
          name: 'get_weather',
          argumentsJson: '{"city":"Paris"}',
        ),
      ],
      rawText: _rawTurn,
    );

    // Building the turn-2 envelope is a straight read off the terminal token:
    // no text parsing, no second source of truth.
    final results = [
      for (final call in token.toolCalls)
        ToolResult(
          callId: call.id,
          name: call.name,
          contentJson: '{"tempC":21}',
        ),
    ];

    expect(results.single.callId, 'call_0');
    expect(token.rawText, contains('<|tool_call_start|>'));
  });
}

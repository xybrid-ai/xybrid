import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _grammar = 'root ::= "{" ws "\\"city\\"" ws ":" ws string ws "}"';

void main() {
  test('grammar defaults to null so generation stays unconstrained', () {
    const config = GenerationConfig();

    expect(config.grammar, isNull);
    expect(config.toFfi().grammar, isNull);
  });

  test('grammar reaches the FFI config', () {
    const config = GenerationConfig(grammar: _grammar);

    expect(config.toFfi().grammar, _grammar);
  });

  test('greedy preset accepts a grammar without losing its sampling values',
      () {
    const config = GenerationConfig.greedy(grammar: _grammar);
    final ffi = config.toFfi();

    expect(ffi.grammar, _grammar);
    expect(ffi.temperature, 0.0);
    expect(ffi.topP, 1.0);
    expect(ffi.topK, 0);
  });

  test('creative preset accepts a grammar', () {
    const config = GenerationConfig.creative(grammar: _grammar);
    final ffi = config.toFfi();

    expect(ffi.grammar, _grammar);
    expect(ffi.temperature, 0.9);
  });

  test('presets leave grammar null when none is given', () {
    expect(const GenerationConfig.greedy().toFfi().grammar, isNull);
    expect(const GenerationConfig.creative().toFfi().grammar, isNull);
  });

  test('every other field still crosses unchanged', () {
    const config = GenerationConfig(
      maxTokens: 512,
      temperature: 0.5,
      topP: 0.8,
      minP: 0.02,
      topK: 20,
      repetitionPenalty: 1.2,
      stopSequences: ['<|end|>'],
      grammar: _grammar,
    );
    final ffi = config.toFfi();

    expect(ffi.maxTokens, 512);
    expect(ffi.temperature, 0.5);
    expect(ffi.topP, 0.8);
    expect(ffi.minP, 0.02);
    expect(ffi.topK, 20);
    expect(ffi.repetitionPenalty, 1.2);
    expect(ffi.stopSequences, ['<|end|>']);
    expect(ffi.grammar, _grammar);
  });

  test('jsonSchemaToGbnf is exported from the public surface', () {
    expect(jsonSchemaToGbnf, isA<Function>());
  });

  group('tools', () {
    const tool = ToolDefinition(
      name: 'get_weather',
      description: 'Current weather for a city.',
      parametersJson: '{"type":"object","properties":{"city":{"type":"string"}}}',
    );

    test('tools default to null so nothing changes for non-tool runs', () {
      expect(const GenerationConfig().tools, isNull);
      expect(const GenerationConfig().toFfi().tools, isNull);
    });

    test('tools reach the FFI config', () {
      final ffi = const GenerationConfig(tools: [tool]).toFfi();

      expect(ffi.tools, hasLength(1));
      expect(ffi.tools!.single.name, 'get_weather');
      expect(ffi.tools!.single.description, 'Current weather for a city.');
      expect(ffi.tools!.single.parametersJson, contains('"city"'));
    });

    test('greedy preset carries tools — the usual tool-calling shape', () {
      final ffi = const GenerationConfig.greedy(tools: [tool]).toFfi();

      expect(ffi.tools, hasLength(1));
      expect(ffi.temperature, 0.0);
    });

    test('creative preset carries tools', () {
      expect(
        const GenerationConfig.creative(tools: [tool]).toFfi().tools,
        hasLength(1),
      );
    });

    test('grammar and tools coexist on one config', () {
      final ffi = const GenerationConfig.greedy(
        grammar: _grammar,
        tools: [tool],
      ).toFfi();

      expect(ffi.grammar, _grammar);
      expect(ffi.tools, hasLength(1));
    });

    test('ToolResult maps onto the FFI shape', () {
      const result = ToolResult(
        callId: 'call_0',
        name: 'get_weather',
        contentJson: '{"temperature_c":17.5}',
      );
      final ffi = result.toFfi();

      expect(ffi.callId, 'call_0');
      expect(ffi.name, 'get_weather');
      expect(ffi.contentJson, '{"temperature_c":17.5}');
    });
  });
}

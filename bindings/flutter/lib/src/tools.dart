/// Tool (function) calling for LLMs.
///
/// One `run` is one model turn, so the loop lives in your code:
///
/// ```dart
/// final tools = [
///   ToolDefinition(
///     name: 'get_weather',
///     description: 'Current weather for a city.',
///     parametersJson: '{"type":"object","properties":{"city":{"type":"string"}}}',
///   ),
/// ];
///
/// final first = await model.run(
///   XybridEnvelope.text('weather in Paris?'),
///   generationConfig: GenerationConfig.greedy(tools: tools),
/// );
///
/// final results = [
///   for (final call in first.toolCalls)
///     ToolResult(
///       callId: call.id,
///       name: call.name,
///       contentJson: await runMyTool(call.name, call.argumentsJson),
///     ),
/// ];
///
/// final second = await model.run(
///   XybridEnvelope.toolResults(
///     userText: 'weather in Paris?',
///     priorAssistantText: first.text ?? '',
///     results: results,
///   ),
///   generationConfig: GenerationConfig.greedy(tools: tools),
/// );
/// ```
///
/// Tool calling is llama.cpp-only today. Unsupported paths — a model with no
/// embedded chat template, the mistralrs backend, the cloud fallback leg —
/// reject a tool-bearing request rather than quietly generating without the
/// tools. Continuation runs on the non-streaming text path only.
library;

import 'rust/api/model.dart';

/// A tool the model may ask to call.
class ToolDefinition {
  /// Function name the model will emit, e.g. `get_weather`.
  final String name;

  /// What the tool does. The model reads this to decide when to call it.
  final String description;

  /// JSON Schema for the arguments, as a JSON string. Pass
  /// `{"type":"object","properties":{}}` for a tool that takes none.
  ///
  /// Invalid JSON here fails the `run` call rather than silently dropping the
  /// tool, so a typo surfaces where you can see it.
  final String parametersJson;

  const ToolDefinition({
    required this.name,
    required this.description,
    required this.parametersJson,
  });

  /// Convert to the FRB-generated FFI type.
  FfiToolDefinition toFfi() => FfiToolDefinition(
    name: name,
    description: description,
    parametersJson: parametersJson,
  );
}

/// One tool call the model emitted this turn.
class ToolCall {
  /// Correlation id for this call, e.g. `call_0`. Echo it back as
  /// [ToolResult.callId].
  final String id;

  /// Which tool the model wants to run.
  final String name;

  /// Arguments as a JSON object string.
  final String argumentsJson;

  const ToolCall({
    required this.id,
    required this.name,
    required this.argumentsJson,
  });

  /// @nodoc
  ToolCall.fromFfi(FfiToolCall inner)
    : id = inner.id,
      name = inner.name,
      argumentsJson = inner.argumentsJson;
}

/// The outcome of running one tool, fed back to the model next turn.
class ToolResult {
  /// The [ToolCall.id] this answers.
  final String callId;

  /// The tool that was invoked.
  final String name;

  /// The tool's output as a JSON string. Wrap plain values — `42`,
  /// `"sunny"` — so the whole field parses as JSON.
  final String contentJson;

  const ToolResult({
    required this.callId,
    required this.name,
    required this.contentJson,
  });

  /// Convert to the FRB-generated FFI type.
  FfiToolResult toFfi() =>
      FfiToolResult(callId: callId, name: name, contentJson: contentJson);
}

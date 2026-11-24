import Foundation

/// Strongly retains one generated handle. Registry removal drops the last
/// owner after in-flight calls return, cancelling unfinished work without flush.
final class XybridObservationEntry {
  private let getContext: () -> XybridObservationContext?
  private let getReference: () -> XybridObservationRef?
  let complete: () -> Void
  let fail: (String, String?) -> Void
  let cancel: () -> Void
  let isFinished: () -> Bool
  private let update: (String, JSObject) throws -> Void

  private init(context: @escaping () -> XybridObservationContext?,
               reference: @escaping () -> XybridObservationRef?,
               complete: @escaping () -> Void,
               fail: @escaping (String, String?) -> Void,
               cancel: @escaping () -> Void,
               isFinished: @escaping () -> Bool,
               update: @escaping (String, JSObject) throws -> Void) {
    self.getContext = context; self.getReference = reference
    self.complete = complete; self.fail = fail; self.cancel = cancel
    self.isFinished = isFinished; self.update = update
  }

  func context() -> JSObject? {
    getContext().map { ["traceparent": $0.traceparent, "tracestate": $0.tracestate as Any? ?? NSNull()] }
  }
  func reference() -> JSObject? { getReference().map(XybridCodec.encodeObservationRef) }
  func record(_ field: String, _ value: JSObject) throws {
    if isFinished() { return }
    try update(field, value)
  }

  private static let what = "manual observation"
  // JavaScript's exact integer range, enforced before converting foreign numbers.
  private static func integer(_ o: JSObject, _ key: String) throws -> Double? {
    guard let value = try XybridCodec.optionalDouble(o, key, what) else { return nil }
    guard value >= 0, value <= 9007199254740991, value.rounded(.towardZero) == value else {
      throw BridgeError.invalidArgument("observation integer must be a nonnegative safe integer")
    }
    return value
  }
  private static func location(_ o: JSObject) throws -> XybridExecutionLocation {
    switch try XybridCodec.optionalString(o, "location", what) ?? "local" {
    case "local": return .local
    case "remote": return .remote
    default: throw BridgeError.invalidArgument("unknown execution location")
    }
  }
  private static func decodeToolCallInfo(_ o: JSObject) throws -> XybridToolCallInfo {
    XybridToolCallInfo(
      name: try XybridCodec.string(o, "name", what),
      callId: try XybridCodec.optionalString(o, "callId", what),
      toolType: try XybridCodec.optionalString(o, "toolType", what),
      description: try XybridCodec.optionalString(o, "description", what),
      argumentsJson: try XybridCodec.optionalString(o, "argumentsJson", what)
    )
  }
  private static func decodeInferenceRequest(_ o: JSObject) throws -> XybridInferenceRequest {
    XybridInferenceRequest(
      operation: try XybridCodec.string(o, "operation", what),
      provider: try XybridCodec.string(o, "provider", what),
      model: try XybridCodec.optionalString(o, "model", what),
      location: try location(o),
      runtime: try XybridCodec.optionalString(o, "runtime", what),
      task: try XybridCodec.optionalString(o, "task", what),
      stream: try XybridCodec.optionalBool(o, "stream", what) ?? false,
      maxTokens: try integer(o, "maxTokens").map { Int64($0) },
      temperature: try XybridCodec.optionalDouble(o, "temperature", what),
      topP: try XybridCodec.optionalDouble(o, "topP", what),
      serverAddress: try XybridCodec.optionalString(o, "serverAddress", what),
      serverPort: try integer(o, "serverPort").map { Int64($0) }
    )
  }
  private static func decodeModelLoadInfo(_ o: JSObject) throws -> XybridModelLoadInfo {
    XybridModelLoadInfo(
      model: try XybridCodec.string(o, "model", what),
      provider: try XybridCodec.optionalString(o, "provider", what),
      runtime: try XybridCodec.optionalString(o, "runtime", what),
      format: try XybridCodec.optionalString(o, "format", what),
      sizeBytes: try integer(o, "sizeBytes").map { UInt64($0) }
    )
  }
  private static func decodeUsage(_ o: JSObject) throws -> XybridUsage {
    XybridUsage(
      inputTokens: try integer(o, "inputTokens").map { UInt64($0) },
      outputTokens: try integer(o, "outputTokens").map { UInt64($0) },
      cacheReadInputTokens: try integer(o, "cacheReadInputTokens").map { UInt64($0) },
      cacheWriteInputTokens: try integer(o, "cacheWriteInputTokens").map { UInt64($0) },
      reasoningOutputTokens: try integer(o, "reasoningOutputTokens").map { UInt64($0) }
    )
  }

  static func start(_ kind: String, _ o: JSObject, _ options: XybridObservationOptions?) throws -> XybridObservationEntry {
    switch kind {
    case "workflow":
      let op = try XybridWorkflowObservation(name: try XybridCodec.string(o, "name", what), options: options)
      return XybridObservationEntry(context: op.context, reference: op.observationRef, complete: op.complete, fail: op.fail, cancel: op.cancel, isFinished: op.isFinished) { field, value in
        switch field {
        default: throw BridgeError.invalidArgument("unsupported observation field")
        }
      }
    case "tool":
      let op = try XybridToolObservation(call: try decodeToolCallInfo(o), options: options)
      return XybridObservationEntry(context: op.context, reference: op.observationRef, complete: op.complete, fail: op.fail, cancel: op.cancel, isFinished: op.isFinished) { field, value in
        switch field {
        case "result": try op.recordResult(resultJson: try XybridCodec.string(value, "json", what))
        default: throw BridgeError.invalidArgument("unsupported observation field")
        }
      }
    case "inference":
      let op = try XybridInferenceObservation(request: try decodeInferenceRequest(o), options: options)
      return XybridObservationEntry(context: op.context, reference: op.observationRef, complete: op.complete, fail: op.fail, cancel: op.cancel, isFinished: op.isFinished) { field, value in
        switch field {
        case "systemInstructions": try op.recordSystemInstructions(partsJson: try XybridCodec.string(value, "json", what))
        case "inputMessages": try op.recordInputMessages(messagesJson: try XybridCodec.string(value, "json", what))
        case "outputMessages": try op.recordOutputMessages(messagesJson: try XybridCodec.string(value, "json", what))
        case "toolDefinitions": try op.recordToolDefinitions(definitionsJson: try XybridCodec.string(value, "json", what))
        case "chunk": op.recordChunk()
        case "firstToken": try op.recordFirstToken(atEpochMs: try integer(value, "atEpochMs").map { UInt64($0) })
        case "usage": op.recordUsage(usage: try decodeUsage(value))
        case "finishReasons": op.recordFinishReasons(reasons: try XybridCodec.stringArray(value, "reasons", what))
        case "response": op.recordResponse(responseId: try XybridCodec.optionalString(value, "responseId", what), model: try XybridCodec.optionalString(value, "model", what))
        default: throw BridgeError.invalidArgument("unsupported observation field")
        }
      }
    case "modelLoad":
      let op = try XybridModelLoadObservation(info: try decodeModelLoadInfo(o), options: options)
      return XybridObservationEntry(context: op.context, reference: op.observationRef, complete: op.complete, fail: op.fail, cancel: op.cancel, isFinished: op.isFinished) { field, value in
        switch field {
        case "cacheHit": op.recordCacheHit(hit: try XybridCodec.optionalBool(value, "hit", what) ?? false)
        default: throw BridgeError.invalidArgument("unsupported observation field")
        }
      }
    default: throw BridgeError.invalidArgument("unknown observation kind")
    }
  }
}

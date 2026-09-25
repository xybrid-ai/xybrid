import Foundation

// Translation between the bolt Swift types (xybrid_bolt.swift) and the
// Foundation values React Native passes across the TurboModule boundary.
//
// Pure functions only — no SDK calls, no module state — so
// tests/swift (scripts/test-ios.sh) can exercise them against the real bolt
// records on a Mac host. The wire shapes are defined once, in src/wire.ts;
// every convenience field (voiceId, speed, sampleRate…) was already folded
// into envelope metadata on the JS side, so envelopes here are just a payload
// kind plus string metadata, exactly like `XybridEnvelope`.
//
// Names avoid every type alias in Xybrid.swift (`Envelope`, `Result`,
// `Model`, …), which compiles into the same module.

/// A JS object as React Native hands it over.
typealias JSObject = [String: Any]

/// A failure raised by the bridge itself rather than the SDK.
enum BridgeError: Error {
  /// A JS argument could not be decoded. → `xybrid_invalid_argument`
  case invalidArgument(String)
  /// Unknown, released, or wrong-kind handle. → `xybrid_handle`
  case handle(String)
  /// A configuration conflict. → `xybrid_config_error`
  case config(String)
}

enum XybridCodec {
  // MARK: - Reading JS values

  /// The value for `key`, treating `NSNull` (JS `null`) as absent.
  static func value(_ object: JSObject, _ key: String) -> Any? {
    guard let value = object[key], !(value is NSNull) else { return nil }
    return value
  }

  static func object(_ any: Any?, _ what: String) throws -> JSObject {
    guard let object = any as? JSObject else {
      throw BridgeError.invalidArgument("\(what) must be an object")
    }
    return object
  }

  static func optionalObject(_ object: JSObject, _ key: String, _ what: String) throws -> JSObject? {
    guard let raw = value(object, key) else { return nil }
    return try self.object(raw, what)
  }

  static func string(_ object: JSObject, _ key: String, _ what: String) throws -> String {
    guard let string = value(object, key) as? String else {
      throw BridgeError.invalidArgument("\(what): '\(key)' must be a string")
    }
    return string
  }

  static func optionalString(_ object: JSObject, _ key: String, _ what: String) throws -> String? {
    guard let raw = value(object, key) else { return nil }
    guard let string = raw as? String else {
      throw BridgeError.invalidArgument("\(what): '\(key)' must be a string")
    }
    return string
  }

  static func optionalBool(_ object: JSObject, _ key: String, _ what: String) throws -> Bool? {
    guard let raw = value(object, key) else { return nil }
    // JS booleans arrive as NSNumber-backed `Bool`; accept only real booleans.
    guard let number = raw as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
      throw BridgeError.invalidArgument("\(what): '\(key)' must be a boolean")
    }
    return number.boolValue
  }

  static func optionalDouble(_ object: JSObject, _ key: String, _ what: String) throws -> Double? {
    guard let raw = value(object, key) else { return nil }
    guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
          number.doubleValue.isFinite else {
      throw BridgeError.invalidArgument("\(what): '\(key)' must be a finite number")
    }
    return number.doubleValue
  }

  static func optionalFloat(_ object: JSObject, _ key: String, _ what: String) throws -> Float? {
    try optionalDouble(object, key, what).map { Float($0) }
  }

  /// A non-negative integer that fits in `UInt32`; fractions truncate.
  static func optionalUInt32(_ object: JSObject, _ key: String, _ what: String) throws -> UInt32? {
    guard let double = try optionalDouble(object, key, what) else { return nil }
    return try uint32(double, "\(what): '\(key)'")
  }

  static func uint32(_ double: Double, _ what: String) throws -> UInt32 {
    guard double >= 0, double <= Double(UInt32.max) else {
      throw BridgeError.invalidArgument("\(what) must be between 0 and \(UInt32.max)")
    }
    return UInt32(double)
  }

  static func stringArray(_ object: JSObject, _ key: String, _ what: String) throws -> [String] {
    guard let raw = value(object, key) else { return [] }
    guard let strings = raw as? [String] else {
      throw BridgeError.invalidArgument("\(what): '\(key)' must be an array of strings")
    }
    return strings
  }

  static func base64(_ object: JSObject, _ key: String, _ what: String) throws -> Data {
    guard let data = decodeBase64(try string(object, key, what)) else {
      throw BridgeError.invalidArgument("\(what): '\(key)' is not valid base64")
    }
    return data
  }

  /// Strict standard base64 (whitespace ignored), or `nil` — the same rule
  /// the Android codec applies, so garbage fails the same way everywhere.
  static func decodeBase64(_ encoded: String) -> Data? {
    Data(base64Encoded: String(encoded.filter { !$0.isWhitespace }))
  }

  /// Accept plain paths and `file://` URLs (what Expo's file APIs hand out).
  static func filePath(_ raw: String) -> String {
    guard raw.hasPrefix("file://"), let url = URL(string: raw), url.isFileURL else { return raw }
    return url.path
  }

  /// Little-endian Float32 PCM from base64.
  static func float32Samples(_ encoded: String) throws -> [Float] {
    guard let data = decodeBase64(encoded) else {
      throw BridgeError.invalidArgument("samples are not valid base64")
    }
    guard data.count % MemoryLayout<Float>.size == 0 else {
      throw BridgeError.invalidArgument("samples must be whole Float32 values (\(data.count) bytes)")
    }
    var samples = [Float](repeating: 0, count: data.count / MemoryLayout<Float>.size)
    _ = samples.withUnsafeMutableBytes { data.copyBytes(to: $0) }
    return samples.map { Float(bitPattern: UInt32(littleEndian: $0.bitPattern)) }
  }

  // MARK: - Envelopes

  static func decodeEnvelope(_ any: Any?) throws -> XybridEnvelope {
    let object = try self.object(any, "envelope")
    let kind = try string(object, "kind", "envelope")
    let what = "\(kind) envelope"
    let payload: XybridEnvelopeKind
    switch kind {
    case "text":
      payload = .text(text: try string(object, "text", what))
    case "audio":
      payload = .audio(bytes: try base64(object, "bytesBase64", what))
    case "embedding":
      guard let numbers = value(object, "data") as? [NSNumber] else {
        throw BridgeError.invalidArgument("\(what): 'data' must be an array of numbers")
      }
      payload = .embedding(values: numbers.map { $0.floatValue })
    case "image":
      payload = .image(bytes: try base64(object, "bytesBase64", what),
                       format: try string(object, "format", what))
    case "multipart":
      guard let parts = value(object, "parts") as? [Any] else {
        throw BridgeError.invalidArgument("\(what): 'parts' must be an array")
      }
      payload = .multiPart(parts: try parts.map(decodeEnvelope))
    default:
      throw BridgeError.invalidArgument("unknown envelope kind '\(kind)'")
    }
    return XybridEnvelope(kind: payload, metadata: try decodeMetadata(object["metadata"]))
  }

  static func decodeMetadata(_ any: Any?) throws -> [XybridMetadataEntry] {
    guard let any, !(any is NSNull) else { return [] }
    guard let object = any as? JSObject else {
      throw BridgeError.invalidArgument("envelope metadata must be an object of strings")
    }
    return try object.keys.sorted().map { key in
      guard let value = object[key] as? String else {
        throw BridgeError.invalidArgument("envelope metadata '\(key)' must be a string")
      }
      return XybridMetadataEntry(key: key, value: value)
    }
  }

  static func encodeEnvelope(_ envelope: XybridEnvelope) -> JSObject {
    var out: JSObject = [
      "metadata": Dictionary(envelope.metadata.map { ($0.key, $0.value) },
                             uniquingKeysWith: { _, last in last }),
    ]
    switch envelope.kind {
    case .text(let text):
      out["kind"] = "text"
      out["text"] = text
    case .audio(let bytes):
      out["kind"] = "audio"
      out["bytesBase64"] = bytes.base64EncodedString()
    case .embedding(let values):
      out["kind"] = "embedding"
      out["data"] = values.map { Double($0) }
    case .image(let bytes, let format):
      out["kind"] = "image"
      out["bytesBase64"] = bytes.base64EncodedString()
      out["format"] = format
    case .multiPart(let parts):
      out["kind"] = "multipart"
      out["parts"] = parts.map(encodeEnvelope)
    }
    return out
  }

  // MARK: - Generation and run options

  static func decodeToolDefinition(_ any: Any) throws -> XybridToolDefinition {
    let object = try self.object(any, "tool")
    return XybridToolDefinition(
      name: try string(object, "name", "tool"),
      description: try string(object, "description", "tool"),
      parametersJson: try string(object, "parametersJson", "tool")
    )
  }

  static func decodeGenerationConfig(_ object: JSObject) throws -> XybridGenerationConfig {
    let what = "generationConfig"
    let tools = try (value(object, "tools") as? [Any] ?? []).map(decodeToolDefinition)
    return XybridGenerationConfig(
      maxTokens: try optionalUInt32(object, "maxTokens", what),
      temperature: try optionalFloat(object, "temperature", what),
      topP: try optionalFloat(object, "topP", what),
      minP: try optionalFloat(object, "minP", what),
      topK: try optionalUInt32(object, "topK", what),
      repetitionPenalty: try optionalFloat(object, "repetitionPenalty", what),
      stopSequences: try stringArray(object, "stopSequences", what),
      grammar: try optionalString(object, "grammar", what),
      tools: tools
    )
  }

  static func encodeGenerationConfig(_ config: XybridGenerationConfig) -> JSObject {
    var out: JSObject = [
      "stopSequences": config.stopSequences,
      "tools": config.tools.map {
        ["name": $0.name, "description": $0.description, "parametersJson": $0.parametersJson]
      },
    ]
    if let value = config.maxTokens { out["maxTokens"] = value }
    if let value = config.temperature { out["temperature"] = Double(value) }
    if let value = config.topP { out["topP"] = Double(value) }
    if let value = config.minP { out["minP"] = Double(value) }
    if let value = config.topK { out["topK"] = value }
    if let value = config.repetitionPenalty { out["repetitionPenalty"] = Double(value) }
    if let value = config.grammar { out["grammar"] = value }
    return out
  }

  static func decodeAbortSignal(_ raw: String) throws -> XybridAbortSignal {
    switch raw {
    case "memoryPressureWarn": return .memoryPressureWarn
    case "memoryPressureCritical": return .memoryPressureCritical
    case "thermalHot": return .thermalHot
    case "thermalCritical": return .thermalCritical
    default: throw BridgeError.invalidArgument("unknown abort signal '\(raw)'")
    }
  }

  /// Run options plus the two handles that ride along with them.
  struct RunRequest {
    var options: XybridRunOptions?
    var context: String?
    var cancel: String?
  }

  static func decodeRunRequest(_ any: Any?) throws -> RunRequest {
    guard let any, !(any is NSNull) else { return RunRequest() }
    let object = try self.object(any, "run options")
    let what = "run options"
    let options = XybridRunOptions(
      generationConfig: try optionalObject(object, "generationConfig", what).map(decodeGenerationConfig),
      abortOn: try stringArray(object, "abortOn", what).map(decodeAbortSignal),
      fallbackToCloud: try optionalBool(object, "fallbackToCloud", what) ?? false,
      maxGraceTokens: try optionalUInt32(object, "maxGraceTokens", what) ?? 0,
      correlationId: try optionalString(object, "correlationId", what)
    )
    return RunRequest(
      options: options,
      context: try optionalString(object, "context", what),
      cancel: try optionalString(object, "cancel", what)
    )
  }

  static func decodeToolResult(_ any: Any) throws -> XybridToolResult {
    let object = try self.object(any, "tool result")
    return XybridToolResult(
      callId: try string(object, "callId", "tool result"),
      name: try string(object, "name", "tool result"),
      contentJson: try string(object, "contentJson", "tool result")
    )
  }

  // MARK: - Results

  static func encodeOutputType(_ type: XybridOutputType) -> String {
    switch type {
    case .text: return "text"
    case .audio: return "audio"
    case .embedding: return "embedding"
    case .unknown: return "unknown"
    }
  }

  static func encodeExecutionTarget(_ target: XybridExecutionTarget) -> String {
    target == .cloud ? "cloud" : "local"
  }

  /// Metrics in native stage order; LLM-only values only when reported.
  static func encodeMetrics(_ metrics: XybridInferenceMetrics) -> JSObject {
    var out: JSObject = [
      "totalMs": metrics.totalMs,
      "stageLatenciesMs": metrics.stageLatenciesMs.map {
        ["stageId": $0.stageId, "latencyMs": $0.latencyMs]
      },
    ]
    if let value = metrics.ttftMs { out["ttftMs"] = value }
    if let value = metrics.tokensPerSecond { out["tokensPerSecond"] = Double(value) }
    if let value = metrics.prefillTps { out["prefillTps"] = Double(value) }
    if let value = metrics.decodeTps { out["decodeTps"] = Double(value) }
    if let value = metrics.tokensOut { out["tokensOut"] = value }
    return out
  }

  static func encodeToolCall(_ call: XybridToolCall) -> JSObject {
    ["id": call.id, "name": call.name, "argumentsJson": call.argumentsJson]
  }

  static func encodeResult(_ result: XybridResult) -> JSObject {
    var out: JSObject = [
      "envelope": encodeEnvelope(result.envelope),
      "outputType": encodeOutputType(result.outputType),
      "modelId": result.modelId,
      "latencyMs": result.latencyMs,
      "executionTarget": encodeExecutionTarget(result.executionTarget),
      "metrics": encodeMetrics(result.metrics),
      "toolCalls": result.toolCalls.map(encodeToolCall),
    ]
    if let reasoning = result.reasoningContent { out["reasoningContent"] = reasoning }
    return out
  }

  static func encodeStreamToken(_ token: XybridStreamToken) -> JSObject {
    var out: JSObject = [
      "token": token.token,
      // u64 → Double: JS numbers are exact to 2^53; Int() could trap.
      "index": Double(token.index),
      "cumulativeText": token.cumulativeText,
      "toolCalls": token.toolCalls.map(encodeToolCall),
    ]
    if let id = token.tokenId { out["tokenId"] = Double(id) }
    if let reason = token.finishReason { out["finishReason"] = reason }
    if let raw = token.rawText { out["rawText"] = raw }
    return out
  }

  static func encodeDownloadStatus(_ status: XybridDownloadStatus) -> JSObject {
    let state: String
    switch status.state {
    case .downloading: state = "downloading"
    case .ready: state = "ready"
    case .failed: state = "failed"
    case .cancelled: state = "cancelled"
    }
    var out: JSObject = [
      "state": state,
      "progress": Double(status.progress),
      "downloadedBytes": Double(status.downloadedBytes),
    ]
    // Absent, not 0, when the source declares no size.
    if let total = status.totalBytes { out["totalBytes"] = Double(total) }
    return out
  }

  static func encodeVoice(_ voice: XybridVoiceInfo) -> JSObject {
    var out: JSObject = ["id": voice.id, "name": voice.name]
    if let gender = voice.gender { out["gender"] = gender }
    if let language = voice.language { out["language"] = language }
    if let style = voice.style { out["style"] = style }
    return out
  }

  static func encodeCacheStatus(_ status: XybridCacheStatus) -> JSObject {
    [
      "totalSizeBytes": Double(status.totalSizeBytes),
      "entryCount": status.entryCount,
      "modelCount": status.modelCount,
      "extractedModelCount": status.extractedModelCount,
      "cacheRoot": status.cacheRoot,
    ]
  }

  static func encodeCacheEntry(_ entry: XybridCacheEntry) -> JSObject {
    let location: String
    switch entry.location {
    case .registry: location = "registry"
    case .extracted: location = "extracted"
    case .huggingFace: location = "huggingFace"
    case .huggingFaceHub: location = "huggingFaceHub"
    }
    return [
      "modelId": entry.modelId,
      "location": location,
      "path": entry.path,
      "sizeBytes": Double(entry.sizeBytes),
    ]
  }

  static func encodeStageResult(_ stage: XybridStageResult) -> JSObject {
    [
      "stageId": stage.stageId,
      "envelope": encodeEnvelope(stage.envelope),
      "outputType": encodeOutputType(stage.outputType),
      "latencyMs": stage.latencyMs,
      "executionTarget": encodeExecutionTarget(stage.executionTarget),
      "metrics": encodeMetrics(stage.metrics),
    ]
  }

  static func encodePipelineResult(_ result: XybridPipelineResult) -> JSObject {
    [
      "envelope": encodeEnvelope(result.envelope),
      "outputType": encodeOutputType(result.outputType),
      "latencyMs": result.latencyMs,
      "stages": result.stages.map(encodeStageResult),
    ]
  }

  static func encodePartial(_ partial: XybridPartialResult) -> JSObject {
    [
      "text": partial.text,
      "isStable": partial.isStable,
      "chunkSequence": Double(partial.chunkSequence),
      "audioDurationMs": Double(partial.audioDurationMs),
    ]
  }

  // MARK: - Live ASR

  static func decodeStreamingConfig(_ any: Any?) throws -> XybridStreamingConfig {
    let object: JSObject = (any == nil || any is NSNull) ? [:] : try self.object(any, "streaming config")
    let what = "streaming config"
    let vad: XybridVadMode
    if let vadObject = try optionalObject(object, "vad", what) {
      vad = .enabled(modelDir: filePath(try string(vadObject, "modelDir", "streaming config vad")))
    } else {
      vad = .off
    }
    return XybridStreamingConfig(
      sampleRate: try optionalUInt32(object, "sampleRate", what) ?? 16_000,
      vad: vad,
      vadThreshold: try optionalFloat(object, "vadThreshold", what) ?? 0.5,
      language: try optionalString(object, "language", what),
      audioCtx: try optionalUInt32(object, "audioCtx", what)
    )
  }

  // MARK: - Errors

  /// The stable `xybrid_*` code for an SDK error (see src/errors.ts).
  static func code(for error: XybridError) -> String {
    switch error {
    case .modelNotFound: return "xybrid_model_not_found"
    case .directoryNotFound: return "xybrid_directory_not_found"
    case .metadataNotFound: return "xybrid_metadata_not_found"
    case .metadataInvalid: return "xybrid_metadata_invalid"
    case .loadError: return "xybrid_load_error"
    case .inferenceError: return "xybrid_inference_error"
    case .abortedForCloudFallback: return "xybrid_aborted_cloud_fallback"
    case .streamingNotSupported: return "xybrid_streaming_unsupported"
    case .notLoaded: return "xybrid_not_loaded"
    case .configError: return "xybrid_config_error"
    case .networkError: return "xybrid_network_error"
    case .offline: return "xybrid_offline"
    case .ioError: return "xybrid_io_error"
    case .cacheError: return "xybrid_cache_error"
    case .pipelineError: return "xybrid_pipeline_error"
    case .circuitOpen: return "xybrid_circuit_open"
    case .rateLimited: return "xybrid_rate_limited"
    case .timeout: return "xybrid_timeout"
    case .missingArtifact: return "xybrid_missing_artifact"
    case .unsupportedModelCapability: return "xybrid_unsupported_model_capability"
    case .unsupportedBackendCapability: return "xybrid_unsupported_backend_capability"
    case .invalidImage: return "xybrid_invalid_image"
    case .cancelled: return "xybrid_cancelled"
    }
  }

  /// `(code, message)` for anything a bridged call can throw.
  static func rejection(for error: Error) -> (code: String, message: String) {
    switch error {
    case let error as XybridError:
      return (code(for: error), error.errorDescription ?? "Xybrid error")
    case BridgeError.invalidArgument(let message):
      return ("xybrid_invalid_argument", message)
    case BridgeError.handle(let message):
      return ("xybrid_handle", message)
    case BridgeError.config(let message):
      return ("xybrid_config_error", message)
    default:
      return ("xybrid_unknown", error.localizedDescription)
    }
  }
}

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
    case "decisionRequest":
      if let json = value(object, "requestJson") {
        guard value(object, "request") == nil, let json = json as? String else {
          throw BridgeError.invalidArgument("invalid decision request")
        }
        payload = try decisionRequestJsonEnvelope(json: json).kind
      } else {
        payload = try decisionRequestEnvelope(request: decodeDecisionRequest(object["request"])).kind
      }
    case "decisions":
      throw BridgeError.invalidArgument("decision results cannot be used as model input")
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
    case .decisionRequest(let request):
      out["kind"] = "decisionRequest"
      out["request"] = encodeDecisionRequest(request)
    case .decisions(let decisions):
      out["kind"] = "decisions"
      out["decisions"] = encodeDecisions(decisions)
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

  // MARK: - Ordered decisions

  static func decodeDecisionContent(_ any: Any?) throws -> XybridDecisionContent? {
    guard let any, !(any is NSNull) else { return nil }
    let o = try object(any, "decision content")
    switch try string(o, "kind", "decision content") {
    case "text": return .text(text: try string(o, "text", "decision content"))
    case "json": return .json(jsonText: try string(o, "jsonText", "decision content"))
    default: throw BridgeError.invalidArgument("unsupported decision content")
    }
  }

  static func decodeDecisionRequest(_ any: Any?) throws -> XybridDecisionRequest {
    let o = try object(any, "decision request")
    guard let questions = value(o, "questions") as? [Any], questions.count <= 16 else {
      throw BridgeError.invalidArgument("invalid decision question count")
    }
    return XybridDecisionRequest(state: try decodeDecisionContent(o["state"]), questions: try questions.map { any in
      let q = try object(any, "decision question")
      let k = try object(q["kind"], "decision question kind")
      let kind: XybridDecisionQuestionKind
      switch try string(k, "kind", "decision question kind") {
      case "choice":
        guard let options = value(k, "options") as? [Any], options.count <= 255 else {
          throw BridgeError.invalidArgument("invalid decision option count")
        }
        kind = .choice(options: try options.map { any in
          let option = try object(any, "decision option")
          return XybridDecisionOption(key: try string(option, "key", "decision option"),
            description: try decodeDecisionContent(option["description"]))
        })
      case "yesNo":
        kind = .yesNo(yes: try decodeDecisionContent(k["yes"]), no: try decodeDecisionContent(k["no"]))
      case "score":
        guard let levels = value(k, "levels") as? [Any], levels.count <= 255 else {
          throw BridgeError.invalidArgument("invalid decision level count")
        }
        kind = .score(levels: try levels.map { any in
          let level = try object(any, "decision level")
          return XybridDecisionLevel(description: try decodeDecisionContent(level["description"]))
        })
      default: throw BridgeError.invalidArgument("unsupported decision question kind")
      }
      return XybridDecisionQuestion(id: try string(q, "id", "decision question"),
        instructions: try decodeDecisionContent(q["instructions"]), kind: kind)
    })
  }

  static func encodeDecisionContent(_ content: XybridDecisionContent) -> JSObject {
    switch content {
    case .text(let text): return ["kind": "text", "text": text]
    case .json(let json): return ["kind": "json", "jsonText": json]
    }
  }

  static func encodeDecisionRequest(_ request: XybridDecisionRequest) -> JSObject {
    var out: JSObject = ["questions": request.questions.map { q -> JSObject in
      var kind: JSObject
      switch q.kind {
      case .choice(let options):
        kind = ["kind": "choice", "options": options.map { option -> JSObject in
          var out: JSObject = ["key": option.key]
          if let description = option.description { out["description"] = encodeDecisionContent(description) }
          return out
        }]
      case .yesNo(let yes, let no):
        kind = ["kind": "yesNo"]
        if let yes { kind["yes"] = encodeDecisionContent(yes) }
        if let no { kind["no"] = encodeDecisionContent(no) }
      case .score(let levels):
        kind = ["kind": "score", "levels": levels.map { level -> JSObject in
          if let description = level.description { return ["description": encodeDecisionContent(description)] }
          return [:]
        }]
      }
      var out: JSObject = ["id": q.id, "kind": kind]
      if let instructions = q.instructions { out["instructions"] = encodeDecisionContent(instructions) }
      return out
    }]
    if let state = request.state { out["state"] = encodeDecisionContent(state) }
    return out
  }

  static func encodeDecisions(_ decisions: XybridDecisions) -> JSObject {
    ["passes": decisions.passes, "inputTokens": decisions.inputTokens,
     "answers": decisions.answers.map { answer -> JSObject in
      let value: JSObject
      switch answer.value {
      case .choice(let key): value = ["kind": "choice", "key": key]
      case .yesNo(let pYes): value = ["kind": "yesNo", "pYes": pYes]
      case .score(let score, let statistic):
        value = ["kind": "score", "value": score, "statistic": statistic == .median ? "median" : "expected"]
      }
      var out: JSObject = ["questionId": answer.questionId, "value": value,
        "distribution": answer.distribution.map { entry -> JSObject in
          ["key": entry.key, "rawScore": entry.rawScore == -.infinity ? "-Infinity" : Double(entry.rawScore) as Any,
           "probability": entry.probability, "origin": entry.origin == .caller ? "caller" : "fixed"]
        }]
      if let confidence = answer.confidence { out["confidence"] = confidence }
      return out
    }]
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
      correlationId: try optionalString(object, "correlationId", what),
      traceparent: try optionalString(object, "traceparent", what),
      tracestate: try optionalString(object, "tracestate", what),
      conversationId: try optionalString(object, "conversationId", what),
      suppressTrace: try optionalBool(object, "suppressTrace", what) ?? false
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
    case .decisions: return "decisions"
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
    if let observation = result.observation { out["observation"] = encodeObservationRef(observation) }
    return out
  }

  // MARK: - Observability

  static func encodeObservationRef(_ ref: XybridObservationRef) -> JSObject {
    // The portable reference shape shared with the JavaScript packages.
    ["trace_id": ref.traceId, "span_id": ref.spanId]
  }

  static func decodeObservationRef(_ any: Any?) throws -> XybridObservationRef {
    let object = try self.object(any, "observation")
    return XybridObservationRef(traceId: try string(object, "trace_id", "observation"),
                                spanId: try string(object, "span_id", "observation"))
  }

  static func encodeSignalStats(_ stats: XybridSignalStats) -> JSObject {
    ["accepted": Double(stats.accepted), "exported": Double(stats.exported),
     "dropped": Double(stats.dropped), "failed": Double(stats.failed), "pending": Double(stats.pending)]
  }

  static func encodeObservabilityStats(_ stats: XybridObservabilityStats) -> JSObject {
    [
      "enabled": stats.enabled,
      "traces": encodeSignalStats(stats.traces),
      "logs": encodeSignalStats(stats.logs),
      "metrics": encodeSignalStats(stats.metrics),
    ]
  }

  /// An object of strings, in key order; absent means empty.
  static func stringMap(_ object: JSObject, _ key: String, _ what: String) throws -> [(String, String)] {
    guard let map = try optionalObject(object, key, "\(what): '\(key)'") else { return [] }
    return try map.keys.sorted().map { name in
      guard let value = map[name] as? String else {
        throw BridgeError.invalidArgument("\(what): '\(key).\(name)' must be a string")
      }
      return (name, value)
    }
  }

  static func encodeTtsAudio(_ packet: XybridTtsAudioChunk) -> JSObject {
    ["pcmBase64": Data(packet.pcm).base64EncodedString(),
     "sampleRate": Double(packet.sampleRate), "channels": Double(packet.channels),
     "firstSample": Double(packet.firstSample)]
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
      "observation": stage.observation.map(encodeObservationRef) ?? NSNull(),
    ]
  }

  static func encodePipelineResult(_ result: XybridPipelineResult) -> JSObject {
    [
      "envelope": encodeEnvelope(result.envelope),
      "outputType": encodeOutputType(result.outputType),
      "latencyMs": result.latencyMs,
      "stages": result.stages.map(encodeStageResult),
      "observation": result.observation.map { encodeObservationRef($0) as Any } ?? NSNull(),
    ]
  }

  static func encodePartial(_ partial: XybridPartialResult) -> JSObject {
    [
      "text": partial.text,
      "isStable": partial.isStable,
      "chunkSequence": Double(partial.chunkSequence),
      "audioDurationMs": Double(partial.audioDurationMs),
      "decoderObservation": partial.decoderObservation.map { encodeObservationRef($0) as Any } ?? NSNull(),
    ]
  }

  static func encodeTranscription(_ result: XybridTranscriptionResult) -> JSObject {
    ["text": result.text, "durationMs": Double(result.durationMs),
     "chunksProcessed": Double(result.chunksProcessed),
     "observation": result.observation.map { encodeObservationRef($0) as Any } ?? NSNull()]
  }

  static func decodeObservation(_ any: Any?) throws -> XybridObservationOptions? {
    guard let any, !(any is NSNull) else { return nil }
    let o = try object(any, "observation options")
    return XybridObservationOptions(
      traceparent: try optionalString(o, "traceparent", "observation options"),
      tracestate: try optionalString(o, "tracestate", "observation options"),
      conversationId: try optionalString(o, "conversationId", "observation options"),
      suppressTrace: try optionalBool(o, "suppressTrace", "observation options") ?? false)
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
      audioCtx: try optionalUInt32(object, "audioCtx", what),
      observation: try decodeObservation(object["observation"])
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
    case .telemetryDelivery: return "xybrid_telemetry_delivery"
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

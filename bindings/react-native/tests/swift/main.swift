// Runs ios/XybridCodec.swift against the real bolt records (xybrid_bolt.swift)
// on a Mac host. Nothing here calls into Rust — the codec is pure — so the
// test links with undefined FFI symbols (see scripts/test-ios.sh).

import Foundation

var failures = 0

func check(_ condition: @autoclosure () -> Bool, _ message: String, line: Int = #line) {
  if !condition() {
    failures += 1
    print("FAIL line \(line): \(message)")
  }
}

func expectInvalidArgument(_ message: String, line: Int = #line, _ body: () throws -> Void) {
  do {
    try body()
    failures += 1
    print("FAIL line \(line): \(message) — did not throw")
  } catch BridgeError.invalidArgument {
    // expected
  } catch {
    failures += 1
    print("FAIL line \(line): \(message) — threw \(error)")
  }
}

/// Round-trip through JSONSerialization, as React Native hands values over.
func js(_ value: Any) -> Any {
  let data = try! JSONSerialization.data(withJSONObject: value)
  return try! JSONSerialization.jsonObject(with: data)
}

// MARK: - Envelopes

do {
  let text = try XybridCodec.decodeEnvelope(js([
    "kind": "text", "text": "hi", "metadata": ["voice_id": "af", "speed": "1.5"],
  ]))
  check(text.kind == .text(text: "hi"), "text payload")
  check(text.metadata.map(\.key) == ["speed", "voice_id"], "metadata sorted by key")

  let audio = try XybridCodec.decodeEnvelope(js(["kind": "audio", "bytesBase64": "AQID", "metadata": [:]]))
  check(audio.kind == .audio(bytes: Data([1, 2, 3])), "audio bytes decoded")

  let embedding = try XybridCodec.decodeEnvelope(js(["kind": "embedding", "data": [0.5, 1, -2]]))
  check(embedding.kind == .embedding(values: [0.5, 1, -2]), "embedding values")

  let multi = try XybridCodec.decodeEnvelope(js([
    "kind": "multipart",
    "metadata": ["xybrid.role": "user"],
    "parts": [
      ["kind": "text", "text": "what?"],
      ["kind": "image", "bytesBase64": "iVBO", "format": "png", "metadata": NSNull()],
    ],
  ]))
  if case .multiPart(let parts) = multi.kind {
    check(parts.count == 2, "two parts")
    check(parts[1].kind == .image(bytes: Data(base64Encoded: "iVBO")!, format: "png"), "image part")
  } else {
    check(false, "multipart kind")
  }

  // Encoding is the exact inverse.
  for envelope in [text, audio, embedding, multi] {
    let decoded = try XybridCodec.decodeEnvelope(js(XybridCodec.encodeEnvelope(envelope)))
    check(decoded == envelope, "round trip \(envelope.kind)")
  }
} catch {
  check(false, "envelope decoding threw \(error)")
}

expectInvalidArgument("unknown kind") { _ = try XybridCodec.decodeEnvelope(js(["kind": "video"])) }
expectInvalidArgument("missing text") { _ = try XybridCodec.decodeEnvelope(js(["kind": "text"])) }
expectInvalidArgument("bad base64") {
  _ = try XybridCodec.decodeEnvelope(js(["kind": "audio", "bytesBase64": "%%%"]))
}
expectInvalidArgument("unpadded base64") {
  _ = try XybridCodec.decodeEnvelope(js(["kind": "audio", "bytesBase64": "AQI"]))
}
check(XybridCodec.decodeBase64("AQ\nID") == Data([1, 2, 3]), "whitespace is ignored")
expectInvalidArgument("non-string metadata") {
  _ = try XybridCodec.decodeEnvelope(js(["kind": "text", "text": "x", "metadata": ["n": 1]]))
}
expectInvalidArgument("not an object") { _ = try XybridCodec.decodeEnvelope("text") }

// MARK: - Run options

do {
  let request = try XybridCodec.decodeRunRequest(js([
    "generationConfig": [
      "maxTokens": 64, "temperature": 0.2, "topK": 40, "stopSequences": ["\n"], "grammar": "root ::= x",
      "tools": [["name": "f", "description": "d", "parametersJson": "{}"]],
    ],
    "abortOn": ["thermalHot", "memoryPressureCritical"],
    "fallbackToCloud": true,
    "maxGraceTokens": 8,
    "correlationId": "req",
    "cloudProvider": "openai",
    "cloudModel": "gpt-4o-mini",
    "cloudGatewayUrl": "https://api.xybrid.dev/v1",
    "context": "context:1",
    "cancel": "cancel:1",
  ]))
  let options = request.options!
  check(options.generationConfig?.maxTokens == 64, "maxTokens")
  check(options.generationConfig?.temperature == Float(0.2), "temperature")
  check(options.generationConfig?.topK == 40, "topK")
  check(options.generationConfig?.stopSequences == ["\n"], "stop sequences")
  check(options.generationConfig?.grammar == "root ::= x", "grammar")
  check(options.generationConfig?.tools == [XybridToolDefinition(name: "f", description: "d", parametersJson: "{}")], "tools")
  check(options.abortOn == [.thermalHot, .memoryPressureCritical], "abort signals")
  check(options.fallbackToCloud && options.maxGraceTokens == 8, "platform knobs")
  check(options.correlationId == "req", "correlation id")
  check(options.cloudProvider == "openai", "cloud provider")
  check(options.cloudModel == "gpt-4o-mini", "cloud model")
  check(options.cloudGatewayUrl == "https://api.xybrid.dev/v1", "cloud gateway")
  let cloudOnly = try XybridCodec.decodeRunRequest(js(["fallbackToCloud": false, "cloudProvider": ""]))
  check(cloudOnly.options?.fallbackToCloud == false && cloudOnly.options?.cloudProvider == "", "disabled fallback and blank provider")
  check(cloudOnly.options?.cloudModel == nil && cloudOnly.options?.cloudGatewayUrl == nil, "omitted cloud fields")
  check(request.context == "context:1" && request.cancel == "cancel:1", "handles")

  let empty = try XybridCodec.decodeRunRequest(nil)
  check(empty.options == nil && empty.context == nil, "null options")
} catch {
  check(false, "run options threw \(error)")
}

expectInvalidArgument("negative maxTokens") {
  _ = try XybridCodec.decodeRunRequest(js(["generationConfig": ["maxTokens": -1]]))
}
expectInvalidArgument("unknown abort signal") { _ = try XybridCodec.decodeRunRequest(js(["abortOn": ["hot"]])) }
expectInvalidArgument("string where a boolean belongs") {
  _ = try XybridCodec.decodeRunRequest(js(["fallbackToCloud": "yes"]))
}
expectInvalidArgument("number where a boolean belongs") {
  _ = try XybridCodec.decodeRunRequest(js(["fallbackToCloud": 1]))
}
expectInvalidArgument("number where a cloud string belongs") {
  _ = try XybridCodec.decodeRunRequest(js(["cloudProvider": 1]))
}

// MARK: - Results

let metrics = XybridInferenceMetrics(
  totalMs: 120, ttftMs: 18, tokensPerSecond: 42.5, prefillTps: nil, decodeTps: 39.75, tokensOut: 24,
  stageLatenciesMs: [XybridStageLatency(stageId: "a", latencyMs: 7), XybridStageLatency(stageId: "b", latencyMs: 108)]
)
let encodedMetrics = XybridCodec.encodeMetrics(metrics)
check(encodedMetrics["totalMs"] as? UInt32 == 120, "totalMs")
check(encodedMetrics["tokensPerSecond"] as? Double == 42.5, "tokens per second")
check(encodedMetrics["prefillTps"] == nil, "absent optional metric stays absent")
check((encodedMetrics["stageLatenciesMs"] as? [[String: Any]])?.map { $0["stageId"] as? String } == ["a", "b"],
      "stage order preserved")

let result = XybridResult(
  envelope: XybridEnvelope(kind: .text(text: "hello"), metadata: [XybridMetadataEntry(key: "xybrid.role", value: "assistant")]),
  outputType: .text, modelId: "qwen", latencyMs: 120, executionTarget: .cloud, metrics: metrics,
  toolCalls: [XybridToolCall(id: "1", name: "f", argumentsJson: "{\"a\":1}")],
  reasoningContent: "because"
)
let encodedResult = XybridCodec.encodeResult(result)
check(encodedResult["outputType"] as? String == "text", "output type")
check(encodedResult["executionTarget"] as? String == "cloud", "execution target")
check(encodedResult["reasoningContent"] as? String == "because", "reasoning")
check((encodedResult["toolCalls"] as? [[String: Any]])?.first?["argumentsJson"] as? String == "{\"a\":1}", "tool calls")
check(JSONSerialization.isValidJSONObject(encodedResult), "result is JSON-safe")

let token = XybridStreamToken(token: "lo", tokenId: 7, index: 1, cumulativeText: "Hello", finishReason: "tool_calls",
                              toolCalls: [], rawText: "<tool_call>")
let encodedToken = XybridCodec.encodeStreamToken(token)
check(encodedToken["index"] as? Double == 1 && encodedToken["tokenId"] as? Double == 7, "token numbers")
check(encodedToken["rawText"] as? String == "<tool_call>", "raw text")

let unknownSize = XybridCodec.encodeDownloadStatus(
  XybridDownloadStatus(state: .downloading, progress: 0.25, downloadedBytes: 5_000_000_000, totalBytes: nil))
check(unknownSize["totalBytes"] == nil, "unknown total is absent")
check(unknownSize["downloadedBytes"] as? Double == 5_000_000_000, "u64 bytes as Double")
check(unknownSize["state"] as? String == "downloading", "state tag")

// MARK: - Live ASR

do {
  let defaults = try XybridCodec.decodeStreamingConfig(nil)
  check(defaults.sampleRate == 16_000 && defaults.vad == .off && defaults.vadThreshold == 0.5, "streaming defaults")
  let vad = try XybridCodec.decodeStreamingConfig(js(["vad": ["modelDir": "file:///tmp/silero"], "language": "en"]))
  check(vad.vad == .enabled(modelDir: "/tmp/silero"), "vad dir from a file URL")
  check(vad.language == "en", "language")

  var bytes = Data()
  for sample: Float in [0, 1, -0.5] {
    withUnsafeBytes(of: sample.bitPattern.littleEndian) { bytes.append(contentsOf: $0) }
  }
  let samples = try XybridCodec.float32Samples(bytes.base64EncodedString())
  check(samples == [0, 1, -0.5], "little-endian PCM")
} catch {
  check(false, "streaming config threw \(error)")
}
expectInvalidArgument("partial float") { _ = try XybridCodec.float32Samples(Data([1, 2, 3]).base64EncodedString()) }

// MARK: - Errors

check(XybridCodec.rejection(for: XybridError.rateLimited(retryAfterSecs: 3)).code == "xybrid_rate_limited", "sdk code")
check(XybridCodec.rejection(for: BridgeError.handle("x")).code == "xybrid_handle", "handle code")
check(XybridCodec.rejection(for: BridgeError.invalidArgument("x")).code == "xybrid_invalid_argument", "argument code")
check(XybridCodec.rejection(for: CocoaError(.fileNoSuchFile)).code == "xybrid_unknown", "foreign error code")

if failures > 0 {
  print("\(failures) codec check(s) failed")
  exit(1)
}
print("iOS codec: all checks passed")

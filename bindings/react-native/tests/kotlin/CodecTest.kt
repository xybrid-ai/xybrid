// Runs android/.../XybridCodec.kt against the real bolt records
// (bindings/kotlin/.../XybridBolt.kt) on a plain JVM. Nothing here calls into
// Rust — the codec is pure — so no native library is loaded.

import ai.xybrid.XybridAbortSignal
import ai.xybrid.XybridDownloadState
import ai.xybrid.XybridDownloadStatus
import ai.xybrid.XybridEnvelope
import ai.xybrid.XybridEnvelopeKind
import ai.xybrid.XybridError
import ai.xybrid.XybridExecutionTarget
import ai.xybrid.XybridInferenceMetrics
import ai.xybrid.XybridMetadataEntry
import ai.xybrid.XybridObservabilityStats
import ai.xybrid.XybridObservationRef
import ai.xybrid.XybridOutputType
import ai.xybrid.XybridPartialResult
import ai.xybrid.XybridResult
import ai.xybrid.XybridSignalStats
import ai.xybrid.XybridStageLatency
import ai.xybrid.XybridStreamToken
import ai.xybrid.XybridToolCall
import ai.xybrid.XybridTranscriptionResult
import ai.xybrid.XybridVadMode
import ai.xybrid.reactnative.BridgeException
import ai.xybrid.reactnative.XybridCodec
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.system.exitProcess

private var failures = 0

private fun check(condition: Boolean, message: String) {
  if (!condition) {
    failures++
    println("FAIL: $message")
  }
}

private fun expectInvalidArgument(message: String, body: () -> Unit) {
  try {
    body()
    failures++
    println("FAIL: $message — did not throw")
  } catch (e: BridgeException.InvalidArgument) {
    // expected
  } catch (t: Throwable) {
    failures++
    println("FAIL: $message — threw $t")
  }
}

/** What `ReadableMap.toHashMap()` produces: numbers are Doubles. */
private fun js(vararg pairs: Pair<String, Any?>): Map<String, Any?> = hashMapOf(*pairs)

/** Envelopes compare structurally except for their array payloads. */
private fun sameEnvelope(a: XybridEnvelope, b: XybridEnvelope): Boolean {
  if (a.metadata != b.metadata) return false
  val x = a.kind
  val y = b.kind
  return when {
    x is XybridEnvelopeKind.Text && y is XybridEnvelopeKind.Text -> x.text == y.text
    x is XybridEnvelopeKind.Audio && y is XybridEnvelopeKind.Audio -> x.bytes.contentEquals(y.bytes)
    x is XybridEnvelopeKind.Embedding && y is XybridEnvelopeKind.Embedding -> x.values.contentEquals(y.values)
    x is XybridEnvelopeKind.Image && y is XybridEnvelopeKind.Image ->
      x.bytes.contentEquals(y.bytes) && x.format == y.format
    x is XybridEnvelopeKind.MultiPart && y is XybridEnvelopeKind.MultiPart ->
      x.parts.size == y.parts.size && x.parts.indices.all { sameEnvelope(x.parts[it], y.parts[it]) }
    else -> false
  }
}

fun main() {
  // -- Envelopes
  val text = XybridCodec.decodeEnvelope(js("kind" to "text", "text" to "hi", "metadata" to js("voice_id" to "af", "speed" to "1.5")))
  check((text.kind as? XybridEnvelopeKind.Text)?.text == "hi", "text payload")
  check(text.metadata.map { it.key } == listOf("speed", "voice_id"), "metadata sorted by key")

  val audio = XybridCodec.decodeEnvelope(js("kind" to "audio", "bytesBase64" to "AQID", "metadata" to null))
  check((audio.kind as? XybridEnvelopeKind.Audio)?.bytes?.contentEquals(byteArrayOf(1, 2, 3)) == true, "audio bytes")

  val embedding = XybridCodec.decodeEnvelope(js("kind" to "embedding", "data" to listOf(0.5, 1.0, -2.0)))
  check((embedding.kind as? XybridEnvelopeKind.Embedding)?.values?.contentEquals(floatArrayOf(0.5f, 1f, -2f)) == true, "embedding")

  val multi = XybridCodec.decodeEnvelope(
    js(
      "kind" to "multipart",
      "metadata" to js("xybrid.role" to "user"),
      "parts" to listOf(js("kind" to "text", "text" to "what?"), js("kind" to "image", "bytesBase64" to "iVBO", "format" to "png")),
    ),
  )
  check((multi.kind as? XybridEnvelopeKind.MultiPart)?.parts?.size == 2, "multipart parts")

  for (envelope in listOf(text, audio, embedding, multi)) {
    val roundTrip = XybridCodec.decodeEnvelope(XybridCodec.encodeEnvelope(envelope))
    check(sameEnvelope(roundTrip, envelope), "round trip ${envelope.kind::class.simpleName}")
  }

  expectInvalidArgument("unknown kind") { XybridCodec.decodeEnvelope(js("kind" to "video")) }
  expectInvalidArgument("missing text") { XybridCodec.decodeEnvelope(js("kind" to "text")) }
  expectInvalidArgument("bad base64") { XybridCodec.decodeEnvelope(js("kind" to "audio", "bytesBase64" to "%%%")) }
  expectInvalidArgument("unpadded base64") { XybridCodec.decodeEnvelope(js("kind" to "audio", "bytesBase64" to "AQI")) }
  check(XybridCodec.decodeBase64("AQ\nID")?.contentEquals(byteArrayOf(1, 2, 3)) == true, "whitespace is ignored")
  expectInvalidArgument("non-string metadata") {
    XybridCodec.decodeEnvelope(js("kind" to "text", "text" to "x", "metadata" to js("n" to 1.0)))
  }
  expectInvalidArgument("not an object") { XybridCodec.decodeEnvelope("text") }

  // -- Run options
  val request = XybridCodec.decodeRunRequest(
    js(
      "generationConfig" to js(
        "maxTokens" to 64.0,
        "temperature" to 0.2,
        "topK" to 40.0,
        "stopSequences" to listOf("\n"),
        "grammar" to "root ::= x",
        "tools" to listOf(js("name" to "f", "description" to "d", "parametersJson" to "{}")),
      ),
      "abortOn" to listOf("thermalHot", "memoryPressureCritical"),
      "fallbackToCloud" to true,
      "maxGraceTokens" to 8.0,
      "correlationId" to "req",
      "traceparent" to "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
      "tracestate" to "vendor=1",
      "conversationId" to "conv-1",
      "suppressTrace" to true,
      "frameSessionId" to "camera-7",
      "context" to "context:1",
      "cancel" to "cancel:1",
    ),
  )
  val options = request.options!!
  check(options.generationConfig?.maxTokens == 64u, "maxTokens")
  check(options.generationConfig?.temperature == 0.2f, "temperature")
  check(options.generationConfig?.topK == 40u, "topK")
  check(options.generationConfig?.grammar == "root ::= x", "grammar")
  check(options.generationConfig?.tools?.single()?.parametersJson == "{}", "tools")
  check(options.abortOn == listOf(XybridAbortSignal.THERMAL_HOT, XybridAbortSignal.MEMORY_PRESSURE_CRITICAL), "abort signals")
  check(options.fallbackToCloud && options.maxGraceTokens == 8u, "platform knobs")
  check(options.traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01", "traceparent")
  check(options.tracestate == "vendor=1" && options.conversationId == "conv-1", "trace context")
  check(options.suppressTrace, "suppress trace")
  check(options.frameSessionId == "camera-7", "frame session")
  check(XybridCodec.decodeRunRequest(js()).options?.suppressTrace == false, "suppressTrace defaults off")
  check(XybridCodec.decodeRunRequest(js()).options?.frameSessionId == null, "no frame session by default")
  check(request.context == "context:1" && request.cancel == "cancel:1", "handles")
  check(XybridCodec.decodeRunRequest(null).options == null, "null options")

  expectInvalidArgument("negative maxTokens") {
    XybridCodec.decodeRunRequest(js("generationConfig" to js("maxTokens" to -1.0)))
  }
  expectInvalidArgument("unknown abort signal") { XybridCodec.decodeRunRequest(js("abortOn" to listOf("hot"))) }
  expectInvalidArgument("string where a boolean belongs") { XybridCodec.decodeRunRequest(js("fallbackToCloud" to "yes")) }

  // -- Results
  val metrics = XybridInferenceMetrics(
    totalMs = 120u,
    ttftMs = 18u,
    tokensPerSecond = 42.5f,
    prefillTps = null,
    decodeTps = 39.75f,
    tokensOut = 24u,
    stageLatenciesMs = listOf(XybridStageLatency("a", 7u), XybridStageLatency("b", 108u)),
  )
  val encodedMetrics = XybridCodec.encodeMetrics(metrics)
  check(encodedMetrics["totalMs"] == 120.0, "totalMs as Double")
  check(encodedMetrics["tokensPerSecond"] == 42.5, "tokens per second")
  check("prefillTps" !in encodedMetrics, "absent optional metric stays absent")
  @Suppress("UNCHECKED_CAST")
  val stages = encodedMetrics["stageLatenciesMs"] as List<Map<String, Any?>>
  check(stages.map { it["stageId"] } == listOf("a", "b"), "stage order preserved")

  val result = XybridResult(
    envelope = XybridEnvelope(XybridEnvelopeKind.Text("hello"), listOf(XybridMetadataEntry("xybrid.role", "assistant"))),
    outputType = XybridOutputType.TEXT,
    modelId = "qwen",
    latencyMs = 120u,
    executionTarget = XybridExecutionTarget.CLOUD,
    metrics = metrics,
    toolCalls = listOf(XybridToolCall("1", "f", "{\"a\":1}")),
    reasoningContent = "because",
    observation = XybridObservationRef("a".repeat(32), "b".repeat(16)),
  )
  val encodedResult = XybridCodec.encodeResult(result)
  check(encodedResult["observation"] == mapOf("trace_id" to "a".repeat(32), "span_id" to "b".repeat(16)), "observation")
  check("observation" !in XybridCodec.encodeResult(result.copy(observation = null)), "untraced run has no observation")
  check(encodedResult["outputType"] == "text" && encodedResult["executionTarget"] == "cloud", "tags")
  check(encodedResult["latencyMs"] == 120.0, "latency as Double")
  check(encodedResult["reasoningContent"] == "because", "reasoning")
  check(encodedResult.values.none { it is UInt || it is ULong }, "no unsigned values reach Arguments")

  val token = XybridCodec.encodeStreamToken(
    XybridStreamToken("lo", 7L, 1uL, "Hello", "tool_calls", emptyList(), "<tool_call>"),
  )
  check(token["index"] == 1.0 && token["tokenId"] == 7.0, "token numbers")
  check(token["rawText"] == "<tool_call>", "raw text")

  val unknownSize = XybridCodec.encodeDownloadStatus(
    XybridDownloadStatus(XybridDownloadState.DOWNLOADING, 0.25f, 5_000_000_000uL, null),
  )
  check("totalBytes" !in unknownSize, "unknown total is absent")
  check(unknownSize["downloadedBytes"] == 5_000_000_000.0, "u64 bytes as Double")

  // -- Observability
  check(XybridCodec.decodeObservationRef(js("trace_id" to "t", "span_id" to "s")) == XybridObservationRef("t", "s"), "ref")
  expectInvalidArgument("observation without a span id") { XybridCodec.decodeObservationRef(js("trace_id" to "t")) }
  val idle = XybridSignalStats(0uL, 0uL, 0uL, 0uL, 0uL)
  val stats = XybridCodec.encodeObservabilityStats(XybridObservabilityStats(
    true,
    XybridSignalStats(5_000_000_001uL, 5_000_000_000uL, 1uL, 0uL, 0uL),
    idle,
    XybridSignalStats(2uL, 1uL, 0uL, 1uL, 0uL),
  ))
  val traces = stats["traces"] as Map<*, *>
  check(stats["enabled"] == true && traces["exported"] == 5_000_000_000.0, "per-signal stats")
  check(traces["accepted"] == 5_000_000_001.0 && traces["dropped"] == 1.0, "trace counters")
  check((stats["metrics"] as Map<*, *>)["failed"] == 1.0, "metric counters")
  check((stats["logs"] as Map<*, *>)["pending"] == 0.0, "log counters")
  check("spansExported" !in stats, "no legacy span counters")
  check(
    listOf("traces", "logs", "metrics").all { signal -> (stats[signal] as Map<*, *>).values.none { it is ULong } },
    "no unsigned stats reach Arguments",
  )

  val observationOptions = XybridCodec.decodeObservation(js(
    "traceparent" to "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    "conversationId" to "chat-42",
  ))
  check(observationOptions?.traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01", "observation parent")
  check(observationOptions?.conversationId == "chat-42" && observationOptions?.suppressTrace == false, "observation defaults")
  check(XybridCodec.decodeObservation(null) == null, "absent observation")
  val observedConfig = XybridCodec.decodeStreamingConfig(js("observation" to js("suppressTrace" to true)))
  check(observedConfig.observation?.suppressTrace == true, "ASR config observation")

  val ref = XybridObservationRef("t", "s")
  val transcript = XybridCodec.encodeTranscription(XybridTranscriptionResult("hello", 200uL, 2uL, ref))
  check(transcript["text"] == "hello" && transcript["chunksProcessed"] == 2.0, "transcript")
  check((transcript["observation"] as Map<*, *>)["span_id"] == "s", "transcript reference")
  val partial = XybridCodec.encodePartial(XybridPartialResult("he", false, 0uL, 100uL, null))
  check(partial.containsKey("decoderObservation") && partial["decoderObservation"] == null, "partial without a decoder run")
  val headers = XybridCodec.stringMap(js("headers" to js("b" to "2", "a" to "1")), "headers", "config")
  check(headers == listOf("a" to "1", "b" to "2"), "string map in key order")
  expectInvalidArgument("non-string header") { XybridCodec.stringMap(js("headers" to js("a" to 1.0)), "headers", "config") }

  // -- Live ASR
  val defaults = XybridCodec.decodeStreamingConfig(null)
  check(defaults.sampleRate == 16_000u && defaults.vad == XybridVadMode.Off && defaults.vadThreshold == 0.5f, "defaults")
  val vad = XybridCodec.decodeStreamingConfig(js("vad" to js("modelDir" to "file:///tmp/silero"), "language" to "en"))
  check(vad.vad == XybridVadMode.Enabled("/tmp/silero"), "vad dir from a file URL")

  val pcm = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN).putFloat(0f).putFloat(1f).putFloat(-0.5f).array()
  val samples = XybridCodec.float32Samples(java.util.Base64.getEncoder().encodeToString(pcm))
  check(samples.contentEquals(floatArrayOf(0f, 1f, -0.5f)), "little-endian PCM")
  expectInvalidArgument("partial float") {
    XybridCodec.float32Samples(java.util.Base64.getEncoder().encodeToString(byteArrayOf(1, 2, 3)))
  }

  // -- Errors
  check(XybridCodec.rejection(XybridError.RateLimited(3uL)).first == "xybrid_rate_limited", "sdk code")
  check(XybridCodec.rejection(BridgeException.Handle("x")).first == "xybrid_handle", "handle code")
  check(XybridCodec.rejection(IllegalStateException("x")).first == "xybrid_unknown", "foreign error code")

  if (failures > 0) {
    println("$failures codec check(s) failed")
    exitProcess(1)
  }
  println("Android codec: all checks passed")
}

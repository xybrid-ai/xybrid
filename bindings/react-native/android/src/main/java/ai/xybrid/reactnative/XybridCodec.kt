package ai.xybrid.reactnative

import ai.xybrid.XybridAbortSignal
import ai.xybrid.XybridCacheEntry
import ai.xybrid.XybridCacheEntryLocation
import ai.xybrid.XybridCacheStatus
import ai.xybrid.XybridDownloadState
import ai.xybrid.XybridDownloadStatus
import ai.xybrid.XybridEnvelope
import ai.xybrid.XybridEnvelopeKind
import ai.xybrid.XybridError
import ai.xybrid.XybridExecutionTarget
import ai.xybrid.XybridGenerationConfig
import ai.xybrid.XybridInferenceMetrics
import ai.xybrid.XybridMetadataEntry
import ai.xybrid.XybridOutputType
import ai.xybrid.XybridPartialResult
import ai.xybrid.XybridPipelineResult
import ai.xybrid.XybridResult
import ai.xybrid.XybridRunOptions
import ai.xybrid.XybridStageResult
import ai.xybrid.XybridStreamToken
import ai.xybrid.XybridStreamingConfig
import ai.xybrid.XybridToolCall
import ai.xybrid.XybridToolDefinition
import ai.xybrid.XybridToolResult
import ai.xybrid.XybridVadMode
import ai.xybrid.XybridVoiceInfo
import android.util.Base64
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Translation between the bolt Kotlin types and plain Kotlin maps and lists.
//
// Pure functions only — no SDK calls, no React Native types — so
// tests/kotlin runs them on a plain JVM against the real bolt records.
// XybridModule converts at the edge: `ReadableMap.toHashMap()` in,
// `Arguments.makeNativeMap()` out. The wire shapes are defined once, in
// src/wire.ts; envelopes are a payload kind plus string metadata, exactly
// like `XybridEnvelope`, because every convenience field was already folded
// into metadata on the JS side.
//
// Every number crosses as a Double (JS numbers), and unsigned values are
// converted explicitly: `Arguments` cannot marshal Kotlin's UInt/ULong.

/** A JS object as `ReadableMap.toHashMap()` hands it over. */
internal typealias JsObject = Map<String, Any?>

/** A failure raised by the bridge itself rather than the SDK. */
internal sealed class BridgeException(message: String) : RuntimeException(message) {
  /** A JS argument could not be decoded. → `xybrid_invalid_argument` */
  class InvalidArgument(message: String) : BridgeException(message)

  /** Unknown, released, or wrong-kind handle. → `xybrid_handle` */
  class Handle(message: String) : BridgeException(message)

  /** A configuration conflict. → `xybrid_config_error` */
  class Config(message: String) : BridgeException(message)
}

internal object XybridCodec {
  // -- Reading JS values --------------------------------------------------------

  fun obj(any: Any?, what: String): JsObject {
    @Suppress("UNCHECKED_CAST")
    return any as? Map<String, Any?> ?: throw BridgeException.InvalidArgument("$what must be an object")
  }

  fun optionalObj(o: JsObject, key: String, what: String): JsObject? = o[key]?.let { obj(it, "$what: '$key'") }

  fun string(o: JsObject, key: String, what: String): String =
    o[key] as? String ?: throw BridgeException.InvalidArgument("$what: '$key' must be a string")

  fun optionalString(o: JsObject, key: String, what: String): String? {
    val raw = o[key] ?: return null
    return raw as? String ?: throw BridgeException.InvalidArgument("$what: '$key' must be a string")
  }

  fun optionalBoolean(o: JsObject, key: String, what: String): Boolean? {
    val raw = o[key] ?: return null
    return raw as? Boolean ?: throw BridgeException.InvalidArgument("$what: '$key' must be a boolean")
  }

  fun optionalDouble(o: JsObject, key: String, what: String): Double? {
    val raw = o[key] ?: return null
    val number = (raw as? Number)?.toDouble()
    if (number == null || !number.isFinite()) {
      throw BridgeException.InvalidArgument("$what: '$key' must be a finite number")
    }
    return number
  }

  fun optionalFloat(o: JsObject, key: String, what: String): Float? = optionalDouble(o, key, what)?.toFloat()

  /** A non-negative integer that fits in `UInt`; fractions truncate. */
  fun optionalUInt(o: JsObject, key: String, what: String): UInt? =
    optionalDouble(o, key, what)?.let { uint(it, "$what: '$key'") }

  fun uint(value: Double, what: String): UInt {
    if (!value.isFinite() || value < 0 || value > UInt.MAX_VALUE.toDouble()) {
      throw BridgeException.InvalidArgument("$what must be between 0 and ${UInt.MAX_VALUE}")
    }
    return value.toLong().toUInt()
  }

  fun stringList(o: JsObject, key: String, what: String): List<String> {
    val raw = o[key] ?: return emptyList()
    val list = raw as? List<*> ?: throw BridgeException.InvalidArgument("$what: '$key' must be an array of strings")
    return list.map { it as? String ?: throw BridgeException.InvalidArgument("$what: '$key' must be an array of strings") }
  }

  fun base64(o: JsObject, key: String, what: String): ByteArray =
    decodeBase64(string(o, key, what)) ?: throw BridgeException.InvalidArgument("$what: '$key' is not valid base64")

  private val base64Shape = Regex("^[A-Za-z0-9+/]*={0,2}$")

  /**
   * Strict standard base64 (whitespace ignored), or `null`. Android's own
   * decoder silently skips characters outside the alphabet, which would turn
   * garbage into an empty payload instead of an error — and disagree with iOS.
   */
  fun decodeBase64(encoded: String): ByteArray? {
    val compact = encoded.filterNot(Char::isWhitespace)
    if (compact.length % 4 != 0 || !base64Shape.matches(compact)) return null
    return try {
      Base64.decode(compact, Base64.NO_WRAP)
    } catch (e: IllegalArgumentException) {
      null
    }
  }

  /** Accept plain paths and `file://` URLs (what Expo's file APIs hand out). */
  fun filePath(raw: String): String =
    if (raw.startsWith("file://")) java.net.URI(raw).path ?: raw else raw

  /** Little-endian Float32 PCM from base64. */
  fun float32Samples(encoded: String): FloatArray {
    val bytes = decodeBase64(encoded) ?: throw BridgeException.InvalidArgument("samples are not valid base64")
    if (bytes.size % 4 != 0) {
      throw BridgeException.InvalidArgument("samples must be whole Float32 values (${bytes.size} bytes)")
    }
    val floats = FloatArray(bytes.size / 4)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
    return floats
  }

  // -- Envelopes ------------------------------------------------------------------

  fun decodeEnvelope(any: Any?): XybridEnvelope {
    val o = obj(any, "envelope")
    val kind = string(o, "kind", "envelope")
    val what = "$kind envelope"
    val payload = when (kind) {
      "text" -> XybridEnvelopeKind.Text(string(o, "text", what))
      "audio" -> XybridEnvelopeKind.Audio(base64(o, "bytesBase64", what))
      "embedding" -> {
        val numbers = o["data"] as? List<*>
          ?: throw BridgeException.InvalidArgument("$what: 'data' must be an array of numbers")
        XybridEnvelopeKind.Embedding(
          FloatArray(numbers.size) { i ->
            (numbers[i] as? Number)?.toFloat()
              ?: throw BridgeException.InvalidArgument("$what: 'data' must be an array of numbers")
          },
        )
      }
      "image" -> XybridEnvelopeKind.Image(base64(o, "bytesBase64", what), string(o, "format", what))
      "multipart" -> {
        val parts = o["parts"] as? List<*>
          ?: throw BridgeException.InvalidArgument("$what: 'parts' must be an array")
        XybridEnvelopeKind.MultiPart(parts.map(::decodeEnvelope))
      }
      else -> throw BridgeException.InvalidArgument("unknown envelope kind '$kind'")
    }
    return XybridEnvelope(payload, decodeMetadata(o["metadata"]))
  }

  fun decodeMetadata(any: Any?): List<XybridMetadataEntry> {
    if (any == null) return emptyList()
    val o = any as? Map<*, *>
      ?: throw BridgeException.InvalidArgument("envelope metadata must be an object of strings")
    return o.keys.map { it.toString() }.sorted().map { key ->
      val value = o[key] as? String
        ?: throw BridgeException.InvalidArgument("envelope metadata '$key' must be a string")
      XybridMetadataEntry(key, value)
    }
  }

  fun encodeEnvelope(envelope: XybridEnvelope): Map<String, Any?> {
    val out = linkedMapOf<String, Any?>()
    when (val kind = envelope.kind) {
      is XybridEnvelopeKind.Text -> {
        out["kind"] = "text"
        out["text"] = kind.text
      }
      is XybridEnvelopeKind.Audio -> {
        out["kind"] = "audio"
        out["bytesBase64"] = Base64.encodeToString(kind.bytes, Base64.NO_WRAP)
      }
      is XybridEnvelopeKind.Embedding -> {
        out["kind"] = "embedding"
        out["data"] = kind.values.map { it.toDouble() }
      }
      is XybridEnvelopeKind.Image -> {
        out["kind"] = "image"
        out["bytesBase64"] = Base64.encodeToString(kind.bytes, Base64.NO_WRAP)
        out["format"] = kind.format
      }
      is XybridEnvelopeKind.MultiPart -> {
        out["kind"] = "multipart"
        out["parts"] = kind.parts.map(::encodeEnvelope)
      }
    }
    out["metadata"] = envelope.metadata.associate { it.key to it.value }
    return out
  }

  // -- Generation and run options -----------------------------------------------

  fun decodeToolDefinition(any: Any?): XybridToolDefinition {
    val o = obj(any, "tool")
    return XybridToolDefinition(
      name = string(o, "name", "tool"),
      description = string(o, "description", "tool"),
      parametersJson = string(o, "parametersJson", "tool"),
    )
  }

  fun decodeGenerationConfig(o: JsObject): XybridGenerationConfig {
    val what = "generationConfig"
    val tools = (o["tools"] as? List<*>).orEmpty().map(::decodeToolDefinition)
    return XybridGenerationConfig(
      maxTokens = optionalUInt(o, "maxTokens", what),
      temperature = optionalFloat(o, "temperature", what),
      topP = optionalFloat(o, "topP", what),
      minP = optionalFloat(o, "minP", what),
      topK = optionalUInt(o, "topK", what),
      repetitionPenalty = optionalFloat(o, "repetitionPenalty", what),
      stopSequences = stringList(o, "stopSequences", what),
      grammar = optionalString(o, "grammar", what),
      tools = tools,
    )
  }

  fun encodeGenerationConfig(config: XybridGenerationConfig): Map<String, Any?> {
    val out = linkedMapOf<String, Any?>(
      "stopSequences" to config.stopSequences,
      "tools" to config.tools.map {
        mapOf("name" to it.name, "description" to it.description, "parametersJson" to it.parametersJson)
      },
    )
    config.maxTokens?.let { out["maxTokens"] = it.toDouble() }
    config.temperature?.let { out["temperature"] = it.toDouble() }
    config.topP?.let { out["topP"] = it.toDouble() }
    config.minP?.let { out["minP"] = it.toDouble() }
    config.topK?.let { out["topK"] = it.toDouble() }
    config.repetitionPenalty?.let { out["repetitionPenalty"] = it.toDouble() }
    config.grammar?.let { out["grammar"] = it }
    return out
  }

  fun decodeAbortSignal(raw: String): XybridAbortSignal = when (raw) {
    "memoryPressureWarn" -> XybridAbortSignal.MEMORY_PRESSURE_WARN
    "memoryPressureCritical" -> XybridAbortSignal.MEMORY_PRESSURE_CRITICAL
    "thermalHot" -> XybridAbortSignal.THERMAL_HOT
    "thermalCritical" -> XybridAbortSignal.THERMAL_CRITICAL
    else -> throw BridgeException.InvalidArgument("unknown abort signal '$raw'")
  }

  /** Run options plus the two handles that ride along with them. */
  data class RunRequest(
    val options: XybridRunOptions? = null,
    val context: String? = null,
    val cancel: String? = null,
  )

  fun decodeRunRequest(any: Any?): RunRequest {
    if (any == null) return RunRequest()
    val what = "run options"
    val o = obj(any, what)
    val options = XybridRunOptions(
      generationConfig = optionalObj(o, "generationConfig", what)?.let(::decodeGenerationConfig),
      abortOn = stringList(o, "abortOn", what).map(::decodeAbortSignal),
      fallbackToCloud = optionalBoolean(o, "fallbackToCloud", what) ?: false,
      maxGraceTokens = optionalUInt(o, "maxGraceTokens", what) ?: 0u,
      correlationId = optionalString(o, "correlationId", what),
      cloudProvider = optionalString(o, "cloudProvider", what),
      cloudModel = optionalString(o, "cloudModel", what),
      cloudGatewayUrl = optionalString(o, "cloudGatewayUrl", what),
    )
    return RunRequest(
      options = options,
      context = optionalString(o, "context", what),
      cancel = optionalString(o, "cancel", what),
    )
  }

  fun decodeToolResult(any: Any?): XybridToolResult {
    val o = obj(any, "tool result")
    return XybridToolResult(
      callId = string(o, "callId", "tool result"),
      name = string(o, "name", "tool result"),
      contentJson = string(o, "contentJson", "tool result"),
    )
  }

  // -- Results --------------------------------------------------------------------

  fun encodeOutputType(type: XybridOutputType): String = when (type) {
    XybridOutputType.TEXT -> "text"
    XybridOutputType.AUDIO -> "audio"
    XybridOutputType.EMBEDDING -> "embedding"
    XybridOutputType.UNKNOWN -> "unknown"
  }

  fun encodeExecutionTarget(target: XybridExecutionTarget): String =
    if (target == XybridExecutionTarget.CLOUD) "cloud" else "local"

  /** Metrics in native stage order; LLM-only values only when reported. */
  fun encodeMetrics(metrics: XybridInferenceMetrics): Map<String, Any?> {
    val out = linkedMapOf<String, Any?>("totalMs" to metrics.totalMs.toDouble())
    metrics.ttftMs?.let { out["ttftMs"] = it.toDouble() }
    metrics.tokensPerSecond?.let { out["tokensPerSecond"] = it.toDouble() }
    metrics.prefillTps?.let { out["prefillTps"] = it.toDouble() }
    metrics.decodeTps?.let { out["decodeTps"] = it.toDouble() }
    metrics.tokensOut?.let { out["tokensOut"] = it.toDouble() }
    out["stageLatenciesMs"] = metrics.stageLatenciesMs.map {
      mapOf("stageId" to it.stageId, "latencyMs" to it.latencyMs.toDouble())
    }
    return out
  }

  fun encodeToolCall(call: XybridToolCall): Map<String, Any?> =
    mapOf("id" to call.id, "name" to call.name, "argumentsJson" to call.argumentsJson)

  fun encodeResult(result: XybridResult): Map<String, Any?> {
    val out = linkedMapOf<String, Any?>(
      "envelope" to encodeEnvelope(result.envelope),
      "outputType" to encodeOutputType(result.outputType),
      "modelId" to result.modelId,
      "latencyMs" to result.latencyMs.toDouble(),
      "executionTarget" to encodeExecutionTarget(result.executionTarget),
      "metrics" to encodeMetrics(result.metrics),
      "toolCalls" to result.toolCalls.map(::encodeToolCall),
    )
    result.reasoningContent?.let { out["reasoningContent"] = it }
    return out
  }

  fun encodeStreamToken(token: XybridStreamToken): Map<String, Any?> {
    val out = linkedMapOf<String, Any?>(
      "token" to token.token,
      // u64 → Double: JS numbers are exact to 2^53.
      "index" to token.index.toDouble(),
      "cumulativeText" to token.cumulativeText,
      "toolCalls" to token.toolCalls.map(::encodeToolCall),
    )
    token.tokenId?.let { out["tokenId"] = it.toDouble() }
    token.finishReason?.let { out["finishReason"] = it }
    token.rawText?.let { out["rawText"] = it }
    return out
  }

  fun encodeDownloadStatus(status: XybridDownloadStatus): Map<String, Any?> {
    val out = linkedMapOf<String, Any?>(
      "state" to when (status.state) {
        XybridDownloadState.DOWNLOADING -> "downloading"
        XybridDownloadState.READY -> "ready"
        XybridDownloadState.FAILED -> "failed"
        XybridDownloadState.CANCELLED -> "cancelled"
      },
      "progress" to status.progress.toDouble(),
      "downloadedBytes" to status.downloadedBytes.toDouble(),
    )
    // Absent, not 0, when the source declares no size.
    status.totalBytes?.let { out["totalBytes"] = it.toDouble() }
    return out
  }

  fun encodeVoice(voice: XybridVoiceInfo): Map<String, Any?> {
    val out = linkedMapOf<String, Any?>("id" to voice.id, "name" to voice.name)
    voice.gender?.let { out["gender"] = it }
    voice.language?.let { out["language"] = it }
    voice.style?.let { out["style"] = it }
    return out
  }

  fun encodeCacheStatus(status: XybridCacheStatus): Map<String, Any?> = mapOf(
    "totalSizeBytes" to status.totalSizeBytes.toDouble(),
    "entryCount" to status.entryCount.toDouble(),
    "modelCount" to status.modelCount.toDouble(),
    "extractedModelCount" to status.extractedModelCount.toDouble(),
    "cacheRoot" to status.cacheRoot,
  )

  fun encodeCacheEntry(entry: XybridCacheEntry): Map<String, Any?> = mapOf(
    "modelId" to entry.modelId,
    "location" to when (entry.location) {
      XybridCacheEntryLocation.REGISTRY -> "registry"
      XybridCacheEntryLocation.EXTRACTED -> "extracted"
      XybridCacheEntryLocation.HUGGING_FACE -> "huggingFace"
      XybridCacheEntryLocation.HUGGING_FACE_HUB -> "huggingFaceHub"
    },
    "path" to entry.path,
    "sizeBytes" to entry.sizeBytes.toDouble(),
  )

  fun encodeStageResult(stage: XybridStageResult): Map<String, Any?> = mapOf(
    "stageId" to stage.stageId,
    "envelope" to encodeEnvelope(stage.envelope),
    "outputType" to encodeOutputType(stage.outputType),
    "latencyMs" to stage.latencyMs.toDouble(),
    "executionTarget" to encodeExecutionTarget(stage.executionTarget),
    "metrics" to encodeMetrics(stage.metrics),
  )

  fun encodePipelineResult(result: XybridPipelineResult): Map<String, Any?> = mapOf(
    "envelope" to encodeEnvelope(result.envelope),
    "outputType" to encodeOutputType(result.outputType),
    "latencyMs" to result.latencyMs.toDouble(),
    "stages" to result.stages.map(::encodeStageResult),
  )

  fun encodePartial(partial: XybridPartialResult): Map<String, Any?> = mapOf(
    "text" to partial.text,
    "isStable" to partial.isStable,
    "chunkSequence" to partial.chunkSequence.toDouble(),
    "audioDurationMs" to partial.audioDurationMs.toDouble(),
  )

  // -- Live ASR -------------------------------------------------------------------

  fun decodeStreamingConfig(any: Any?): XybridStreamingConfig {
    val what = "streaming config"
    val o = if (any == null) emptyMap() else obj(any, what)
    val vad = optionalObj(o, "vad", what)
      ?.let { XybridVadMode.Enabled(filePath(string(it, "modelDir", "streaming config vad"))) }
      ?: XybridVadMode.Off
    return XybridStreamingConfig(
      sampleRate = optionalUInt(o, "sampleRate", what) ?: 16_000u,
      vad = vad,
      vadThreshold = optionalFloat(o, "vadThreshold", what) ?: 0.5f,
      language = optionalString(o, "language", what),
      audioCtx = optionalUInt(o, "audioCtx", what),
    )
  }

  // -- Errors ---------------------------------------------------------------------

  /** The stable `xybrid_*` code for an SDK error (see src/errors.ts). */
  fun code(error: XybridError): String = when (error) {
    is XybridError.ModelNotFound -> "xybrid_model_not_found"
    is XybridError.DirectoryNotFound -> "xybrid_directory_not_found"
    is XybridError.MetadataNotFound -> "xybrid_metadata_not_found"
    is XybridError.MetadataInvalid -> "xybrid_metadata_invalid"
    is XybridError.LoadError -> "xybrid_load_error"
    is XybridError.InferenceError -> "xybrid_inference_error"
    is XybridError.AbortedForCloudFallback -> "xybrid_aborted_cloud_fallback"
    is XybridError.StreamingNotSupported -> "xybrid_streaming_unsupported"
    is XybridError.NotLoaded -> "xybrid_not_loaded"
    is XybridError.ConfigError -> "xybrid_config_error"
    is XybridError.NetworkError -> "xybrid_network_error"
    is XybridError.Offline -> "xybrid_offline"
    is XybridError.IoError -> "xybrid_io_error"
    is XybridError.CacheError -> "xybrid_cache_error"
    is XybridError.PipelineError -> "xybrid_pipeline_error"
    is XybridError.CircuitOpen -> "xybrid_circuit_open"
    is XybridError.RateLimited -> "xybrid_rate_limited"
    is XybridError.Timeout -> "xybrid_timeout"
    is XybridError.MissingArtifact -> "xybrid_missing_artifact"
    is XybridError.UnsupportedModelCapability -> "xybrid_unsupported_model_capability"
    is XybridError.UnsupportedBackendCapability -> "xybrid_unsupported_backend_capability"
    is XybridError.InvalidImage -> "xybrid_invalid_image"
    is XybridError.Cancelled -> "xybrid_cancelled"
  }

  /** `(code, message)` for anything a bridged call can throw. */
  fun rejection(error: Throwable): Pair<String, String> = when (error) {
    is XybridError -> code(error) to (error.message ?: "Xybrid error")
    is BridgeException.InvalidArgument -> "xybrid_invalid_argument" to (error.message ?: "invalid argument")
    is BridgeException.Handle -> "xybrid_handle" to (error.message ?: "unknown handle")
    is BridgeException.Config -> "xybrid_config_error" to (error.message ?: "configuration conflict")
    else -> "xybrid_unknown" to (error.message ?: error.toString())
  }
}

package ai.xybrid.reactnative

// TurboModule implementation. Forwards every JS call into the Kotlin
// wrapper that ships at `bindings/kotlin/src/main/kotlin/ai/xybrid/Xybrid.kt`,
// which is itself a thin layer over the BoltFFI-generated bindings.
//
// Model handles are opaque string IDs (UUIDs). The native side keeps a
// concurrent map of `id -> XybridModel`; `releaseModel` drops the entry and
// closes the handle so the underlying Rust `Arc<XybridModel>` decrements and
// frees.

import ai.xybrid.Envelope
import ai.xybrid.Xybrid
import ai.xybrid.XybridAbortSignal
import ai.xybrid.XybridEnvelope
import ai.xybrid.XybridDownloadStatus
import ai.xybrid.XybridError
import ai.xybrid.XybridExecutionTarget
import ai.xybrid.XybridGenerationConfig
import ai.xybrid.XybridModel
import ai.xybrid.XybridResult
import ai.xybrid.XybridRunOptions
import ai.xybrid.XybridStreamEventKind
import ai.xybrid.XybridStreamToken
import ai.xybrid.XybridThermalState
import ai.xybrid.XybridVoiceInfo
import ai.xybrid.audioBytes
import ai.xybrid.clearBatteryLevel
import ai.xybrid.clearThermalState
import ai.xybrid.embedding
import ai.xybrid.initSdkCacheDir
import ai.xybrid.isSpeculativeCloudEnabled
import ai.xybrid.jsonSchemaToGbnf
import ai.xybrid.reasoningContent
import ai.xybrid.setBatteryLevel
import ai.xybrid.setBinding
import ai.xybrid.setThermalState
import ai.xybrid.success
import ai.xybrid.text
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

class XybridModule(reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext) {

  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val models = ConcurrentHashMap<String, XybridModel>()
  private val streams = ConcurrentHashMap<String, StreamEntry>()

  override fun getName(): String = NAME

  // Released when the RN module is torn down (fast refresh, bundle reload,
  // host teardown). Native model weights are hundreds of MB, so failing to
  // close them promptly OOMs the device — cancel in-flight work and free
  // every handle here. Streams are closed too: a live generation keeps a
  // worker thread (and the model) alive until it is aborted.
  override fun invalidate() {
    super.invalidate()
    scope.cancel()
    // Close streaming sessions before their models: streamClose needs the
    // still-alive model handle, and closing it unwinds the generation thread.
    streams.values.forEach { it.model.streamClose(it.streamId) }
    streams.clear()
    models.values.forEach { it.close() }
    models.clear()
  }

  // -- Lifecycle --

  @ReactMethod
  fun initialize(cacheDir: String?, promise: Promise) {
    try {
      // Register react-native as the binding identity *before* invoking
      // Xybrid.init, which would otherwise lock in "kotlin" via its own
      // setBinding call. set_binding is OnceLock-guarded so the first call
      // wins; this ordering pins the registry header to react-native.
      setBinding("react-native")
      Xybrid.init(reactApplicationContext)

      // Override the cache dir if the JS side supplied one. Otherwise
      // Xybrid.init has already pointed it at <filesDir>/xybrid/models.
      if (!cacheDir.isNullOrEmpty()) {
        File(cacheDir).mkdirs()
        initSdkCacheDir(cacheDir)
      }
      promise.resolve(null)
    } catch (t: Throwable) {
      promise.reject("xybrid_init", t.message, t)
    }
  }

  // -- Loaders --
  //
  // Bolt collapsed `XybridModelLoader.fromX(...).load()` into the
  // `XybridModel` factories: the primary constructor loads from the registry,
  // and `fromBundle` / `fromDirectory` / `fromHuggingface` are companion
  // factories. Each loads eagerly (there is no separate `.load()` step).

  @ReactMethod
  fun loadFromRegistry(modelId: String, promise: Promise) {
    runLoad(promise) { XybridModel(modelId) }
  }

  // Serves from the cloud gateway while the weights download in the
  // background, so this resolves without waiting on the download.
  @ReactMethod
  fun loadFromRegistrySpeculative(modelId: String, promise: Promise) {
    runLoad(promise) { XybridModel.fromRegistrySpeculative(modelId) }
  }

  @ReactMethod
  fun loadFromBundle(path: String, promise: Promise) {
    runLoad(promise) { XybridModel.fromBundle(path) }
  }

  @ReactMethod
  fun loadFromDirectory(path: String, promise: Promise) {
    runLoad(promise) { XybridModel.fromDirectory(path) }
  }

  @ReactMethod
  fun loadFromHuggingface(repo: String, promise: Promise) {
    runLoad(promise) { XybridModel.fromHuggingface(repo) }
  }

  @ReactMethod
  fun releaseModel(handle: String, promise: Promise) {
    // Abort + drop any live streams started from this model first (streamClose
    // needs the still-alive model handle), so releasing the model unwinds
    // their generation instead of orphaning a worker thread.
    val iter = streams.entries.iterator()
    while (iter.hasNext()) {
      val entry = iter.next()
      if (entry.value.modelHandle == handle) {
        entry.value.model.streamClose(entry.value.streamId)
        iter.remove()
      }
    }
    models.remove(handle)?.close()
    promise.resolve(null)
  }

  // -- Model lifecycle --

  @ReactMethod
  fun warmup(handle: String, promise: Promise) {
    runVoid(handle, promise) { it.warmup() }
  }

  @ReactMethod
  fun unload(handle: String, promise: Promise) {
    runVoid(handle, promise) { it.unload() }
  }

  // -- Inference --

  @ReactMethod
  fun run(handle: String, envelope: ReadableMap, config: ReadableMap?, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    val env = try {
      decodeEnvelope(envelope)
    } catch (e: IllegalArgumentException) {
      promise.reject("xybrid_envelope", e.message, e)
      return
    }
    val opts = config?.let(::decodeRunOptions)

    scope.launch {
      try {
        val result = model.run(env, opts)
        promise.resolve(encodeResult(result))
      } catch (e: XybridError) {
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
        // Don't swallow coroutine cancellation (e.g. scope.cancel() on
        // module invalidation) — let it propagate so the machinery unwinds.
        if (t is CancellationException) throw t
        promise.reject("xybrid", t.message, t)
      }
    }
  }

  // -- Streaming --

  @ReactMethod
  fun streamStart(handle: String, envelope: ReadableMap, options: ReadableMap?, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    val env = try {
      decodeEnvelope(envelope)
    } catch (e: IllegalArgumentException) {
      promise.reject("xybrid_envelope", e.message, e)
      return
    }
    val opts = options?.let(::decodeRunOptions)

    scope.launch {
      try {
        val streamId = model.runStream(env, opts)
        val id = UUID.randomUUID().toString()
        streams[id] = StreamEntry(model, streamId, handle)
        promise.resolve(id)
      } catch (e: XybridError) {
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
        if (t is CancellationException) throw t
        promise.reject("xybrid", t.message, t)
      }
    }
  }

  @ReactMethod
  fun streamNext(streamHandle: String, promise: Promise) {
    val entry = streams[streamHandle]
    if (entry == null) {
      // Released/unknown stream → treat as exhausted, not an error.
      promise.resolve(null)
      return
    }
    scope.launch {
      try {
        // Blocks until the next event is ready; runs on Dispatchers.IO like run.
        val event = entry.model.streamNext(entry.streamId)
        when (event.kind) {
          XybridStreamEventKind.TOKEN -> {
            val token = event.token
            if (token == null) promise.resolve(null) else promise.resolve(encodeTokenEvent(token))
          }
          XybridStreamEventKind.COMPLETE -> {
            // `streamResult` also closes the bolt-side session; drop our
            // bookkeeping entry so later calls resolve null (exhausted).
            val result = entry.model.streamResult(entry.streamId)
            streams.remove(streamHandle)
            val out = Arguments.createMap()
            out.putString("kind", "complete")
            out.putMap("result", encodeResult(result))
            promise.resolve(out)
          }
        }
      } catch (e: XybridError) {
        // A failed streamNext already closed the session bolt-side; mirror
        // that here, then reject with the same typed codes as `run`.
        streams.remove(streamHandle)
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
        if (t is CancellationException) throw t
        streams.remove(streamHandle)
        promise.reject("xybrid", t.message, t)
      }
    }
  }

  @ReactMethod
  fun streamRelease(streamHandle: String, promise: Promise) {
    // Closing the bolt session aborts the underlying generation run (its
    // receiver drops, unwinding the backend). Idempotent if the session
    // already finished or errored.
    streams.remove(streamHandle)?.let { it.model.streamClose(it.streamId) }
    promise.resolve(null)
  }

  // -- TTS introspection --

  @ReactMethod
  fun voices(handle: String, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    if (!model.hasVoices()) {
      promise.resolve(null)
      return
    }
    val out = Arguments.createArray()
    model.voices().forEach { out.pushMap(encodeVoice(it)) }
    promise.resolve(out)
  }

  @ReactMethod
  fun defaultVoiceId(handle: String, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    promise.resolve(model.defaultVoice()?.id)
  }

  @ReactMethod
  fun hasVoices(handle: String, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    promise.resolve(model.hasVoices())
  }

  // -- Speculative cloud --

  @ReactMethod
  fun isCloudServing(handle: String, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    promise.resolve(model.isCloudServing())
  }

  @ReactMethod
  fun downloadStatus(handle: String, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    promise.resolve(encodeDownloadStatus(model.downloadStatus()))
  }

  // Blocks natively until the download settles, so it runs on the module scope
  // rather than the RN thread.
  @ReactMethod
  fun awaitDownload(handle: String, timeoutMs: Double, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    scope.launch {
      try {
        val status = model.awaitDownload(timeoutMs.coerceAtLeast(0.0).toULong())
        promise.resolve(encodeDownloadStatus(status))
      } catch (e: XybridError) {
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
        // Same discipline as runLoad: let cancellation unwind the scope, but
        // never leave the JS promise unsettled on a real failure.
        if (t is CancellationException) throw t
        promise.reject("xybrid", t.message, t)
      }
    }
  }

  // -- Cloud gateway configuration --

  @ReactMethod
  fun setPlatformUrl(url: String, promise: Promise) {
    ai.xybrid.setPlatformUrl(url)
    promise.resolve(null)
  }

  @ReactMethod
  fun setSpeculativeCloud(enabled: Boolean, promise: Promise) {
    ai.xybrid.setSpeculativeCloud(enabled)
    promise.resolve(null)
  }

  @ReactMethod
  fun isSpeculativeCloudEnabled(promise: Promise) {
    promise.resolve(isSpeculativeCloudEnabled())
  }

  // -- Platform-state push --

  @ReactMethod
  fun setBatteryLevel(percent: Double, promise: Promise) {
    val bounded = percent.coerceIn(0.0, 100.0).toInt()
    setBatteryLevel(bounded.toUByte())
    promise.resolve(null)
  }

  @ReactMethod
  fun clearBatteryLevel(promise: Promise) {
    clearBatteryLevel()
    promise.resolve(null)
  }

  @ReactMethod
  fun setThermalState(state: String, promise: Promise) {
    val mapped = when (state.lowercase(java.util.Locale.ROOT)) {
      "normal" -> XybridThermalState.NORMAL
      "warm" -> XybridThermalState.WARM
      "hot" -> XybridThermalState.HOT
      "critical" -> XybridThermalState.CRITICAL
      else -> {
        promise.reject("xybrid_thermal", "Unknown thermal state: $state")
        return
      }
    }
    setThermalState(mapped)
    promise.resolve(null)
  }

  @ReactMethod
  fun clearThermalState(promise: Promise) {
    clearThermalState()
    promise.resolve(null)
  }

  // -- Utilities --

  @ReactMethod
  fun jsonSchemaToGbnf(schemaJson: String, promise: Promise) {
    try {
      // Shared JSON-Schema→GBNF converter from the bolt bindings. Fast (pure
      // string transform), so no coroutine hop is needed.
      promise.resolve(jsonSchemaToGbnf(schemaJson))
    } catch (e: XybridError) {
      rejectXybrid(promise, e)
    } catch (t: Throwable) {
      promise.reject("xybrid", t.message, t)
    }
  }

  // MARK: - Helpers

  private fun runLoad(promise: Promise, factory: suspend () -> XybridModel) {
    scope.launch {
      try {
        val model = factory()
        val id = UUID.randomUUID().toString()
        models[id] = model
        promise.resolve(id)
      } catch (e: XybridError) {
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
        // Don't swallow coroutine cancellation (e.g. scope.cancel() on
        // module invalidation) — let it propagate so the machinery unwinds.
        if (t is CancellationException) throw t
        promise.reject("xybrid", t.message, t)
      }
    }
  }

  // Run a void-returning model op (warmup / unload) off the RN thread,
  // resolving on success and mapping XybridError on failure.
  private fun runVoid(handle: String, promise: Promise, op: suspend (XybridModel) -> Unit) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    scope.launch {
      try {
        op(model)
        promise.resolve(null)
      } catch (e: XybridError) {
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
        if (t is CancellationException) throw t
        promise.reject("xybrid", t.message, t)
      }
    }
  }

  // Build a bolt [XybridEnvelope] via the `Envelope` factories, which fold the
  // well-known TTS / ASR options (sample_rate, channels, voice_id, speed) into
  // envelope metadata entries — the bolt `XybridEnvelopeKind` variants
  // themselves only carry the raw payload.
  private fun decodeEnvelope(map: ReadableMap): XybridEnvelope {
    val kind = map.getString("kind") ?: throw IllegalArgumentException("envelope missing 'kind'")
    return when (kind) {
      "audio" -> {
        val b64 = map.getString("bytesBase64")
          ?: throw IllegalArgumentException("audio envelope: 'bytesBase64' missing")
        val bytes = Base64.decode(b64, Base64.DEFAULT)
        val sampleRate = if (map.hasKey("sampleRate") && !map.isNull("sampleRate")) map.getInt("sampleRate") else 16000
        val channels = if (map.hasKey("channels") && !map.isNull("channels")) map.getInt("channels") else 1
        Envelope.audio(bytes, sampleRate.toUInt(), channels.toUInt())
      }
      "text" -> {
        val text = map.getString("text")
          ?: throw IllegalArgumentException("text envelope: 'text' missing")
        val voiceId = if (map.hasKey("voiceId") && !map.isNull("voiceId")) map.getString("voiceId") else null
        val speed = if (map.hasKey("speed") && !map.isNull("speed")) map.getDouble("speed") else null
        if (voiceId != null) {
          Envelope.text(text, voiceId, speed ?: 1.0)
        } else {
          Envelope.text(text)
        }
      }
      "embedding" -> {
        val arr = map.getArray("data")
          ?: throw IllegalArgumentException("embedding envelope: 'data' missing")
        Envelope.embedding(arr.toFloatArray())
      }
      else -> throw IllegalArgumentException("Unknown envelope kind: $kind")
    }
  }

  private fun ReadableArray.toFloatArray(): FloatArray {
    val out = FloatArray(size())
    for (i in 0 until size()) out[i] = getDouble(i).toFloat()
    return out
  }

  // Map the JS `RunOptions` payload onto bolt's XybridRunOptions. The JS facade
  // normalizes its argument to `{ generationConfig, abortOn, fallbackToCloud,
  // maxGraceTokens, correlationId }`, so every field the Apple/Kotlin SDKs
  // expose is reachable from React Native.
  private fun decodeRunOptions(map: ReadableMap): XybridRunOptions {
    val gc = if (map.hasKey("generationConfig") && !map.isNull("generationConfig")) {
      map.getMap("generationConfig")?.let(::decodeGenerationConfig)
    } else {
      null
    }
    val abortOn = if (map.hasKey("abortOn") && !map.isNull("abortOn")) {
      val arr = map.getArray("abortOn")!!
      val out = ArrayList<XybridAbortSignal>(arr.size())
      for (i in 0 until arr.size()) {
        decodeAbortSignal(arr.getString(i))?.let(out::add)
      }
      out
    } else {
      emptyList()
    }
    val maxGrace = if (map.hasKey("maxGraceTokens") && !map.isNull("maxGraceTokens")) {
      map.getInt("maxGraceTokens").coerceAtLeast(0).toUInt()
    } else {
      0u
    }
    return XybridRunOptions(
      generationConfig = gc,
      abortOn = abortOn,
      fallbackToCloud = map.hasKey("fallbackToCloud") && !map.isNull("fallbackToCloud") &&
        map.getBoolean("fallbackToCloud"),
      maxGraceTokens = maxGrace,
      correlationId = if (map.hasKey("correlationId") && !map.isNull("correlationId")) {
        map.getString("correlationId")
      } else {
        null
      },
    )
  }

  private fun decodeGenerationConfig(map: ReadableMap): XybridGenerationConfig {
    fun uintOrNull(key: String): UInt? {
      if (!map.hasKey(key) || map.isNull(key)) return null
      // Guard against negative JS values wrapping around to a huge UInt.
      val value = map.getInt(key)
      return if (value >= 0) value.toUInt() else null
    }
    fun floatOrNull(key: String) =
      if (map.hasKey(key) && !map.isNull(key)) map.getDouble(key).toFloat() else null
    val stops = if (map.hasKey("stopSequences") && !map.isNull("stopSequences")) {
      val arr = map.getArray("stopSequences")!!
      val out = ArrayList<String>(arr.size())
      for (i in 0 until arr.size()) out.add(arr.getString(i) ?: "")
      out
    } else {
      emptyList()
    }
    return XybridGenerationConfig(
      maxTokens = uintOrNull("maxTokens"),
      temperature = floatOrNull("temperature"),
      topP = floatOrNull("topP"),
      minP = floatOrNull("minP"),
      topK = uintOrNull("topK"),
      repetitionPenalty = floatOrNull("repetitionPenalty"),
      stopSequences = stops,
      grammar = if (map.hasKey("grammar") && !map.isNull("grammar")) map.getString("grammar") else null,
    )
  }

  private fun decodeAbortSignal(raw: String?): XybridAbortSignal? = when (raw) {
    "memoryPressureWarn" -> XybridAbortSignal.MEMORY_PRESSURE_WARN
    "memoryPressureCritical" -> XybridAbortSignal.MEMORY_PRESSURE_CRITICAL
    "thermalHot" -> XybridAbortSignal.THERMAL_HOT
    "thermalCritical" -> XybridAbortSignal.THERMAL_CRITICAL
    else -> null
  }

  private fun encodeResult(r: XybridResult): WritableMap {
    val out = Arguments.createMap()
    out.putBoolean("success", r.success)
    out.putInt("latencyMs", r.latencyMs.toInt())
    out.putString(
      "executionTarget",
      if (r.executionTarget == XybridExecutionTarget.CLOUD) "cloud" else "local",
    )
    r.text?.let { out.putString("text", it) }
    r.reasoningContent?.let { out.putString("reasoningContent", it) }
    r.audioBytes?.let { out.putString("audioBytesBase64", Base64.encodeToString(it, Base64.NO_WRAP)) }
    r.embedding?.let {
      val arr = Arguments.createArray()
      it.forEach { f -> arr.pushDouble(f.toDouble()) }
      out.putArray("embedding", arr)
    }
    return out
  }

  // Encode the download snapshot as the `DownloadStatus` object the JS facade
  // expects; the state is a lowercase string tag matching `DownloadState`.
  private fun encodeDownloadStatus(s: XybridDownloadStatus): WritableMap {
    val out = Arguments.createMap()
    out.putString(
      "state",
      when (s.state) {
        ai.xybrid.XybridDownloadState.DOWNLOADING -> "downloading"
        ai.xybrid.XybridDownloadState.READY -> "ready"
        ai.xybrid.XybridDownloadState.FAILED -> "failed"
      },
    )
    out.putDouble("progress", s.progress.toDouble())
    return out
  }

  // Encode a bolt `XybridStreamToken` as the discriminated `token` event the
  // JS facade narrows by `kind`. The terminal `complete` event is built at the
  // call site (it pairs `streamResult` with `encodeResult` so the generator's
  // return value matches `run`).
  private fun encodeTokenEvent(t: XybridStreamToken): WritableMap {
    val out = Arguments.createMap()
    out.putString("kind", "token")
    val token = Arguments.createMap()
    token.putString("token", t.token)
    // Double, not Int: index is u64 (ULong) and RN numbers are doubles (exact
    // to 2^53) — toInt() would truncate a large index.
    token.putDouble("index", t.index.toDouble())
    token.putString("cumulativeText", t.cumulativeText)
    t.tokenId?.let { token.putDouble("tokenId", it.toDouble()) }
    t.finishReason?.let { token.putString("finishReason", it) }
    out.putMap("token", token)
    return out
  }

  private fun encodeVoice(v: XybridVoiceInfo): WritableMap {
    val out = Arguments.createMap()
    out.putString("id", v.id)
    out.putString("name", v.name)
    v.gender?.let { out.putString("gender", it) }
    v.language?.let { out.putString("language", it) }
    v.style?.let { out.putString("style", it) }
    return out
  }

  private fun rejectXybrid(promise: Promise, e: XybridError) {
    val code = when (e) {
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
      is XybridError.InvalidImage -> "xybrid_invalid_image"
      is XybridError.MissingArtifact -> "xybrid_missing_artifact"
      is XybridError.UnsupportedModelCapability -> "xybrid_unsupported_model_capability"
      is XybridError.UnsupportedBackendCapability -> "xybrid_unsupported_backend_capability"
    }
    promise.reject(code, e.message ?: "Xybrid error", e)
  }

  // A live streaming session: the model it runs on (bolt sessions are
  // model-scoped ids, so streamNext/streamClose are model methods), the
  // session id, and the model's handle string (so releasing a model can drain
  // its streams). No native handle of its own to free: `streamClose` is an
  // idempotent map-remove inside the bolt model, and the session `Arc` is
  // released when the last in-flight `streamNext` returns — so the
  // use-after-free/deferred-close machinery the old stream *handle* needed
  // does not apply here. Abort still takes effect at the next token boundary.
  private data class StreamEntry(
    val model: XybridModel,
    val streamId: ULong,
    val modelHandle: String,
  )

  companion object {
    const val NAME = "RNXybrid"
  }
}

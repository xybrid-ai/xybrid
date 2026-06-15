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
import ai.xybrid.XybridEnvelope
import ai.xybrid.XybridError
import ai.xybrid.XybridModel
import ai.xybrid.XybridResult
import ai.xybrid.XybridThermalState
import ai.xybrid.XybridVoiceInfo
import ai.xybrid.audioBytes
import ai.xybrid.clearBatteryLevel
import ai.xybrid.clearThermalState
import ai.xybrid.embedding
import ai.xybrid.initSdkCacheDir
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

  override fun getName(): String = NAME

  // Released when the RN module is torn down (fast refresh, bundle reload,
  // host teardown). Native model weights are hundreds of MB, so failing to
  // close them promptly OOMs the device — cancel in-flight work and free
  // every handle here.
  override fun invalidate() {
    super.invalidate()
    scope.cancel()
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
    models.remove(handle)?.close()
    promise.resolve(null)
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
    // NOTE: bolt's `XybridModel.run(envelope)` does not yet accept a
    // per-call generation config — config is ignored until the facade/bolt
    // surface threads `GenerationConfig` through `run`. Tracked as a
    // bolt-binding follow-up.

    scope.launch {
      try {
        val result = model.run(env)
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

  private fun encodeResult(r: XybridResult): WritableMap {
    val out = Arguments.createMap()
    out.putBoolean("success", r.success)
    out.putInt("latencyMs", r.latencyMs.toInt())
    r.text?.let { out.putString("text", it) }
    r.audioBytes?.let { out.putString("audioBytesBase64", Base64.encodeToString(it, Base64.NO_WRAP)) }
    r.embedding?.let {
      val arr = Arguments.createArray()
      it.forEach { f -> arr.pushDouble(f.toDouble()) }
      out.putArray("embedding", arr)
    }
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
    }
    promise.reject(code, e.message ?: "Xybrid error", e)
  }

  companion object {
    const val NAME = "RNXybrid"
  }
}

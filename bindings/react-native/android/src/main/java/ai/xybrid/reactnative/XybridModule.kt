package ai.xybrid.reactnative

// TurboModule implementation. Forwards every JS call into the Kotlin
// wrapper that ships at `bindings/kotlin/src/main/kotlin/ai/xybrid/Xybrid.kt`,
// which is itself a thin layer over the UniFFI-generated bindings.
//
// Model handles are opaque string IDs (UUIDs). The native side keeps a
// concurrent map of `id -> XybridModel`; `releaseModel` drops the entry so
// the underlying Rust `Arc<XybridModel>` decrements and frees.

import ai.xybrid.Xybrid
import ai.xybrid.XybridEnvelope
import ai.xybrid.XybridException
import ai.xybrid.XybridGenerationConfig
import ai.xybrid.XybridModelLoader
import ai.xybrid.XybridModel
import ai.xybrid.XybridResult
import ai.xybrid.XybridThermalState
import ai.xybrid.XybridVoiceInfo
import ai.xybrid.clearBatteryLevel
import ai.xybrid.clearThermalState
import ai.xybrid.initSdkCacheDir
import ai.xybrid.setBatteryLevel
import ai.xybrid.setBinding
import ai.xybrid.setThermalState
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

class XybridModule(reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext) {

  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val models = ConcurrentHashMap<String, XybridModel>()

  override fun getName(): String = NAME

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

  @ReactMethod
  fun loadFromRegistry(modelId: String, promise: Promise) {
    runLoad(promise) { XybridModelLoader.fromRegistry(modelId).load() }
  }

  @ReactMethod
  fun loadFromBundle(path: String, promise: Promise) {
    runLoad(promise) { XybridModelLoader.fromBundle(path).load() }
  }

  @ReactMethod
  fun loadFromDirectory(path: String, promise: Promise) {
    runLoad(promise) { XybridModelLoader.fromDirectory(path).load() }
  }

  @ReactMethod
  fun loadFromHuggingface(repo: String, promise: Promise) {
    runLoad(promise) { XybridModelLoader.fromHuggingface(repo).load() }
  }

  @ReactMethod
  fun releaseModel(handle: String, promise: Promise) {
    models.remove(handle)?.destroy()
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
    val cfg = config?.let(::decodeConfig)

    scope.launch {
      try {
        val result = model.run(env, cfg)
        promise.resolve(encodeResult(result))
      } catch (e: XybridException) {
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
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
    val list = model.voices()
    if (list == null) {
      promise.resolve(null)
      return
    }
    val out = Arguments.createArray()
    list.forEach { out.pushMap(encodeVoice(it)) }
    promise.resolve(out)
  }

  @ReactMethod
  fun defaultVoiceId(handle: String, promise: Promise) {
    val model = models[handle]
    if (model == null) {
      promise.reject("xybrid_handle", "Unknown model handle: $handle")
      return
    }
    promise.resolve(model.defaultVoiceId())
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
    val mapped = when (state.lowercase()) {
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
      } catch (e: XybridException) {
        rejectXybrid(promise, e)
      } catch (t: Throwable) {
        promise.reject("xybrid", t.message, t)
      }
    }
  }

  private fun decodeEnvelope(map: ReadableMap): XybridEnvelope {
    val kind = map.getString("kind") ?: throw IllegalArgumentException("envelope missing 'kind'")
    return when (kind) {
      "audio" -> {
        val b64 = map.getString("bytesBase64")
          ?: throw IllegalArgumentException("audio envelope: 'bytesBase64' missing")
        val bytes = Base64.decode(b64, Base64.DEFAULT)
        val sampleRate = if (map.hasKey("sampleRate")) map.getInt("sampleRate") else 16000
        val channels = if (map.hasKey("channels")) map.getInt("channels") else 1
        XybridEnvelope.Audio(bytes, sampleRate.toUInt(), channels.toUInt())
      }
      "text" -> {
        val text = map.getString("text")
          ?: throw IllegalArgumentException("text envelope: 'text' missing")
        val voiceId = if (map.hasKey("voiceId") && !map.isNull("voiceId")) map.getString("voiceId") else null
        val speed = if (map.hasKey("speed") && !map.isNull("speed")) map.getDouble("speed") else null
        XybridEnvelope.Text(text, voiceId, speed)
      }
      "embedding" -> {
        val arr = map.getArray("data")
          ?: throw IllegalArgumentException("embedding envelope: 'data' missing")
        XybridEnvelope.Embedding(arr.toFloatList())
      }
      else -> throw IllegalArgumentException("Unknown envelope kind: $kind")
    }
  }

  private fun ReadableArray.toFloatList(): List<Float> {
    val out = ArrayList<Float>(size())
    for (i in 0 until size()) out.add(getDouble(i).toFloat())
    return out
  }

  private fun decodeConfig(map: ReadableMap): XybridGenerationConfig {
    fun uintOrNull(key: String) =
      if (map.hasKey(key) && !map.isNull(key)) map.getInt(key).toUInt() else null
    fun floatOrNull(key: String) =
      if (map.hasKey(key) && !map.isNull(key)) map.getDouble(key).toFloat() else null
    val stops = if (map.hasKey("stopSequences") && !map.isNull("stopSequences")) {
      val arr = map.getArray("stopSequences")!!
      val out = ArrayList<String>(arr.size())
      for (i in 0 until arr.size()) out.add(arr.getString(i) ?: "")
      out
    } else null
    return XybridGenerationConfig(
      maxTokens = uintOrNull("maxTokens"),
      temperature = floatOrNull("temperature"),
      topP = floatOrNull("topP"),
      minP = floatOrNull("minP"),
      topK = uintOrNull("topK"),
      repetitionPenalty = floatOrNull("repetitionPenalty"),
      stopSequences = stops,
    )
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

  private fun rejectXybrid(promise: Promise, e: XybridException) {
    val code = when (e) {
      is XybridException.ModelNotFound -> "xybrid_model_not_found"
      is XybridException.DirectoryNotFound -> "xybrid_directory_not_found"
      is XybridException.MetadataNotFound -> "xybrid_metadata_not_found"
      is XybridException.MetadataInvalid -> "xybrid_metadata_invalid"
      is XybridException.LoadError -> "xybrid_load_error"
      is XybridException.InferenceError -> "xybrid_inference_error"
      is XybridException.StreamingNotSupported -> "xybrid_streaming_unsupported"
      is XybridException.NotLoaded -> "xybrid_not_loaded"
      is XybridException.ConfigError -> "xybrid_config_error"
      is XybridException.NetworkError -> "xybrid_network_error"
      is XybridException.IoError -> "xybrid_io_error"
      is XybridException.CacheError -> "xybrid_cache_error"
      is XybridException.PipelineError -> "xybrid_pipeline_error"
      is XybridException.CircuitOpen -> "xybrid_circuit_open"
      is XybridException.RateLimited -> "xybrid_rate_limited"
      is XybridException.Timeout -> "xybrid_timeout"
    }
    promise.reject(code, e.message ?: "Xybrid error", e)
  }

  companion object {
    const val NAME = "RNXybrid"
  }
}

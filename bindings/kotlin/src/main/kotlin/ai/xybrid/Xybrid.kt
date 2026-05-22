/**
 * Xybrid SDK for Android
 *
 * Hand-written wrappers + compatibility shims over the BoltFFI-generated
 * bindings in `XybridBolt.kt`. Both files live in the same `ai.xybrid`
 * package so consumers `import ai.xybrid.…` and see a single surface.
 *
 * For full API documentation, see https://docs.xybrid.dev/sdk/kotlin
 */
@file:Suppress("unused")

package ai.xybrid

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.io.File

// -- SDK Initialization --

/**
 * Main entry point for the Xybrid SDK.
 *
 * Call [Xybrid.init] once before using any other Xybrid functionality.
 *
 * ```kotlin
 * class MyApplication : Application() {
 *     override fun onCreate() {
 *         super.onCreate()
 *         Xybrid.init(this)
 *     }
 * }
 * ```
 */
object Xybrid {
    @Volatile
    private var initialized = false

    /**
     * Initialize the Xybrid runtime.
     *
     * Idempotent and thread-safe — subsequent calls after a successful
     * initialization are no-ops.
     *
     * Subscribes to OS-level battery (`ACTION_BATTERY_CHANGED` sticky
     * broadcast) and thermal (`PowerManager.OnThermalStatusChangedListener`,
     * API 29+) notifications so the routing engine has live telemetry
     * without consumer apps writing boilerplate. Receivers register
     * against the application context so they survive Activity rotation.
     */
    @JvmStatic
    fun init(context: Context) {
        if (initialized) return
        synchronized(this) {
            if (initialized) return
            setBinding("kotlin")
            val cacheDir = File(context.filesDir, "xybrid/models")
            initSdkCacheDir(cacheDir.absolutePath)
            registerPlatformObservers(context.applicationContext)
            initialized = true
        }
    }

    /** Returns `true` if [init] has been called successfully. */
    @JvmStatic
    val isInitialized: Boolean get() = initialized

    private fun registerPlatformObservers(appContext: Context) {
        val batteryReceiver = object : BroadcastReceiver() {
            override fun onReceive(received: Context, intent: Intent) {
                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                if (level < 0 || scale <= 0) {
                    clearBatteryLevel()
                    return
                }
                val pct = ((level * 100) / scale).coerceIn(0, 100)
                setBatteryLevel(pct.toUByte())
            }
        }
        appContext.registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val pm = appContext.getSystemService(Context.POWER_SERVICE) as PowerManager
            setThermalState(thermalStatusToXybrid(pm.currentThermalStatus))
            pm.addThermalStatusListener { status ->
                setThermalState(thermalStatusToXybrid(status))
            }
        }
    }

    /**
     * Map Android's 7-bucket thermal status to xybrid's 4-band
     * [`XybridThermalState`]. The last three bands (CRITICAL, EMERGENCY,
     * SHUTDOWN) all collapse to `CRITICAL` — the routing engine should
     * treat them identically (pause heavy work).
     */
    private fun thermalStatusToXybrid(status: Int): XybridThermalState = when (status) {
        PowerManager.THERMAL_STATUS_NONE,
        PowerManager.THERMAL_STATUS_LIGHT,
        -> XybridThermalState.NORMAL
        PowerManager.THERMAL_STATUS_MODERATE -> XybridThermalState.WARM
        PowerManager.THERMAL_STATUS_SEVERE -> XybridThermalState.HOT
        else -> XybridThermalState.CRITICAL
    }
}

// -- Public Type Aliases --
//
// Bolt collapsed `XybridModelLoader.fromRegistry(id).load()` into the
// `XybridModel.fromRegistry(id)` companion-object factory — there is no
// loader type anymore. The Model / Result / Envelope / VoiceInfo /
// GenerationConfig aliases stay for convenience.

/** A loaded model ready for inference. */
typealias Model = XybridModel

/** The result of a model inference operation. */
typealias Result = XybridResult

/**
 * Errors that can occur during model loading or inference.
 *
 * Backwards-compat alias — the previous uniffi binding emitted
 * `XybridException`. Bolt emits the same hierarchy under the
 * `XybridError` name; this alias keeps `catch (e: XybridException)`
 * compiling against the bolt surface.
 */
typealias XybridException = XybridError

/** Voice metadata for TTS models. */
typealias VoiceInfo = XybridVoiceInfo

/** LLM generation parameters (temperature, top-p, max tokens, etc.). */
typealias GenerationConfig = XybridGenerationConfig

// -- GenerationConfig Presets --

/** Preset factory methods for [GenerationConfig]. */
object GenerationConfigs {
    /** Greedy decoding preset (deterministic, temperature=0). */
    @JvmStatic
    fun greedy() = XybridGenerationConfig(
        maxTokens = null,
        temperature = 0.0f,
        topP = 1.0f,
        minP = null,
        topK = 0u,
        repetitionPenalty = null,
        stopSequences = emptyList(),
    )

    /** Creative generation preset (higher temperature). */
    @JvmStatic
    fun creative() = XybridGenerationConfig(
        maxTokens = null,
        temperature = 0.9f,
        topP = 0.95f,
        minP = null,
        topK = 50u,
        repetitionPenalty = null,
        stopSequences = emptyList(),
    )
}

// -- XybridResult compatibility shim --
//
// The bolt-generated `XybridResult` carries an `envelope` whose `kind` is
// a sealed-class hierarchy (`Text`, `Audio`, `Embedding`). The previous
// uniffi-generated `XybridResult` flattened these into nullable fields
// (`text`, `audioBytes`, `embedding`) plus a `success` flag. Consumers
// (the Android example, anything downstream) read those flat fields, so
// we mirror them as extension properties on the bolt type.

/** `true` for any result returned from [XybridModel.run]. */
val XybridResult.success: Boolean get() = true

/** `true` if the result carries no output (`OutputType.UNKNOWN`). */
val XybridResult.isFailure: Boolean get() = outputType == XybridOutputType.UNKNOWN

/** Text payload, if the result is `.Text`. `null` otherwise. */
val XybridResult.text: String?
    get() = (envelope.kind as? XybridEnvelopeKind.Text)?.text

/** Audio bytes, if the result is `.Audio`. `null` otherwise. */
val XybridResult.audioBytes: ByteArray?
    get() = (envelope.kind as? XybridEnvelopeKind.Audio)?.bytes

/** Embedding vector, if the result is `.Embedding`. `null` otherwise. */
val XybridResult.embedding: FloatArray?
    get() = (envelope.kind as? XybridEnvelopeKind.Embedding)?.values

/** The latency in seconds as a Double. */
val XybridResult.latencySeconds: Double get() = latencyMs.toDouble() / 1000.0

// -- XybridEnvelope Factory Methods --
//
// Bolt's `XybridEnvelope` is a flat struct with `kind: XybridEnvelopeKind`
// and `metadata: List<XybridMetadataEntry>`. The previous uniffi factories
// (`XybridEnvelope.Text(...)`, `.Audio(...)`) were enum-variant
// constructors. Reproduce those factories here, folding the well-known
// TTS / ASR metadata keys into entries.

/** Factory methods for creating [XybridEnvelope] instances. */
object Envelope {
    /**
     * Creates an audio envelope from raw PCM data.
     * @param bytes Raw PCM audio bytes.
     * @param sampleRate Sample rate in Hz (default 16000).
     * @param channels Number of channels (default 1).
     *
     * Drop `@JvmOverloads` here: `UInt` is an inline value class and the
     * `@JvmOverloads` annotation can't be applied to functions that the
     * value-class-mangling rules touch. Kotlin callers still get the
     * default-argument ergonomics; Java callers would need to pass all
     * three explicitly anyway (UInt isn't a first-class Java type).
     */
    @JvmStatic
    fun audio(bytes: ByteArray, sampleRate: UInt = 16000u, channels: UInt = 1u): XybridEnvelope =
        XybridEnvelope(
            kind = XybridEnvelopeKind.Audio(bytes),
            metadata = listOf(
                XybridMetadataEntry("sample_rate", sampleRate.toString()),
                XybridMetadataEntry("channels", channels.toString()),
            ),
        )

    /** Creates a text envelope for TTS with default voice. */
    @JvmStatic
    fun text(text: String): XybridEnvelope =
        XybridEnvelope(kind = XybridEnvelopeKind.Text(text), metadata = emptyList())

    /**
     * Creates a text envelope for TTS with voice and speed options.
     * @param voiceId Voice ID (e.g. "af_heart" for Kokoro).
     * @param speed Speed multiplier (1.0 = normal, default).
     */
    @JvmStatic
    @JvmOverloads
    fun text(text: String, voiceId: String, speed: Double = 1.0): XybridEnvelope {
        val metadata = mutableListOf<XybridMetadataEntry>()
        metadata.add(XybridMetadataEntry("voice_id", voiceId))
        metadata.add(XybridMetadataEntry("speed", speed.toString()))
        return XybridEnvelope(kind = XybridEnvelopeKind.Text(text), metadata = metadata)
    }

    /** Creates an embedding envelope from raw vector data. */
    @JvmStatic
    fun embedding(data: FloatArray): XybridEnvelope =
        XybridEnvelope(kind = XybridEnvelopeKind.Embedding(data), metadata = emptyList())
}

// -- XybridVoiceInfo Extensions --

/** Returns `true` if the voice gender is male. */
val XybridVoiceInfo.isMale: Boolean get() = gender == "male"

/** Returns `true` if the voice gender is female. */
val XybridVoiceInfo.isFemale: Boolean get() = gender == "female"

// -- XybridError Extensions --

/** User-friendly error message for display. Falls back to a category
 * label when the variant has no embedded message. */
val XybridError.displayMessage: String
    get() = message ?: when (this) {
        is XybridError.ModelNotFound -> "Model not found: $id"
        is XybridError.DirectoryNotFound -> "Directory not found: $path"
        is XybridError.MetadataNotFound -> "Model metadata not found at $path"
        is XybridError.MetadataInvalid -> message
        is XybridError.LoadError -> message
        is XybridError.InferenceError -> message
        is XybridError.AbortedForCloudFallback -> "Aborted for cloud fallback: $reason"
        is XybridError.StreamingNotSupported -> "Streaming is not supported by this model"
        is XybridError.NotLoaded -> "Model not loaded"
        is XybridError.ConfigError -> message
        is XybridError.NetworkError -> message
        is XybridError.Offline -> message
        is XybridError.IoError -> message
        is XybridError.CacheError -> message
        is XybridError.PipelineError -> message
        is XybridError.CircuitOpen -> message
        is XybridError.RateLimited -> "Rate limited, retry after $retryAfterSecs seconds"
        is XybridError.Timeout -> "Request timeout after $timeoutMs ms"
    }

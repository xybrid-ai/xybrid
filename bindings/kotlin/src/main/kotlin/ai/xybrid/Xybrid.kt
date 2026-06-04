/**
 * Xybrid SDK for Android
 * Convenience wrappers and extensions for the UniFFI-generated bindings.
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
 * Inference runs on-device whether or not you authenticate; pass an
 * `apiKey` to start the telemetry exporter and see your runs on the
 * dashboard. Get a free key at https://dashboard.xybrid.dev.
 *
 * ```kotlin
 * class MyApplication : Application() {
 *     override fun onCreate() {
 *         super.onCreate()
 *         // Anonymous — local inference, telemetry disabled
 *         Xybrid.init(this)
 *
 *         // Authenticated — telemetry flows to the dashboard
 *         Xybrid.init(this, apiKey = BuildConfig.XYBRID_API_KEY)
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
     * This must be called once before using any Xybrid functionality.
     * It is safe to call multiple times — subsequent calls are no-ops.
     *
     * Typically called from `Application.onCreate()` or `Activity.onCreate()`.
     *
     * All parameters except [context] are optional. Without an [apiKey], the
     * SDK runs fully on-device and telemetry is disabled — the first
     * inference logs a one-shot hint pointing at the dashboard (suppress
     * with the `XYBRID_QUIET=1` environment variable). Pass [apiKey] to
     * start the platform telemetry exporter; [ingestUrl] overrides the
     * destination for a self-hosted dashboard, and [gatewayUrl] overrides
     * the LLM gateway. Configuration is applied on the first call; because
     * `init` is idempotent, a later call with different arguments is a no-op.
     *
     * Also subscribes to OS-level battery and thermal notifications and
     * forwards each value through the SDK's push-state surface so the
     * routing engine has live telemetry without consumer apps writing
     * boilerplate. Receivers/listeners are registered against the
     * application context so they survive Activity rotation. Battery
     * monitoring uses the sticky `ACTION_BATTERY_CHANGED` broadcast,
     * which delivers the current value immediately on registration —
     * no separate seed call is needed. Thermal monitoring requires
     * API 29+ ([`PowerManager.OnThermalStatusChangedListener`]); on
     * older devices the routing engine sees `thermal_state = None`
     * (treated as "no signal" rather than an optimistic default).
     *
     * @param context Android context (application or activity).
     * @param apiKey Xybrid API key. When set, starts the telemetry exporter.
     * @param gatewayUrl Optional override for the LLM gateway URL.
     * @param ingestUrl Optional override for the telemetry ingest URL.
     */
    @JvmStatic
    @JvmOverloads
    fun init(
        context: Context,
        apiKey: String? = null,
        gatewayUrl: String? = null,
        ingestUrl: String? = null,
    ) {
        if (initialized) return
        synchronized(this) {
            if (initialized) return
            setBinding("kotlin")
            val cacheDir = File(context.filesDir, "xybrid/models")
            initSdkCacheDir(cacheDir.absolutePath)
            configureRuntime(apiKey = apiKey, gatewayUrl = gatewayUrl, ingestUrl = ingestUrl)
            registerPlatformObservers(context.applicationContext)
            initialized = true
        }
    }

    /** Returns `true` if [init] has been called successfully. */
    @JvmStatic
    val isInitialized: Boolean get() = initialized

    private fun registerPlatformObservers(appContext: Context) {
        // Sticky-broadcast registration delivers the latest battery state
        // synchronously, so the receiver's first onReceive seeds the value
        // without a separate read.
        val batteryReceiver = object : BroadcastReceiver() {
            override fun onReceive(received: Context, intent: Intent) {
                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                if (level < 0 || scale <= 0) {
                    clearBatteryLevel()
                    return
                }
                // Scale to 0..=100; clamp so a brief sensor blip can't
                // overflow the UByte the FFI expects.
                val pct = ((level * 100) / scale).coerceIn(0, 100)
                setBatteryLevel(pct.toUByte())
            }
        }
        appContext.registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

        // Thermal status listener is API 29+ — older devices simply
        // don't get a thermal signal, which the routing engine treats
        // as `thermal_state = None`.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val pm = appContext.getSystemService(Context.POWER_SERVICE) as PowerManager
            // Push the current status immediately so first cache miss isn't
            // blind, then keep updating on every transition.
            setThermalState(thermalStatusToXybrid(pm.currentThermalStatus))
            pm.addThermalStatusListener { status ->
                setThermalState(thermalStatusToXybrid(status))
            }
        }
    }

    /**
     * Map Android's 7-bucket thermal status to xybrid's 4-band
     * [`XybridThermalState`].
     *
     * Android (API 29+) reports `THERMAL_STATUS_NONE`/`LIGHT`/`MODERATE`/
     * `SEVERE`/`CRITICAL`/`EMERGENCY`/`SHUTDOWN`. The last three all
     * indicate the device is on the verge of forced throttling or shutdown,
     * so they collapse to `Critical` — the routing engine should treat
     * them identically (pause heavy work).
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

/** Loads ML models from the registry or local bundles. */
typealias ModelLoader = XybridModelLoader

/** A loaded model ready for inference. */
typealias Model = XybridModel

/** The result of a model inference operation. */
typealias Result = XybridResult

/** Errors that can occur during model loading or inference. */
typealias XybridError = XybridException

/** Voice metadata for TTS models. */
typealias VoiceInfo = XybridVoiceInfo

/** LLM generation parameters (temperature, top-p, max tokens, etc.). */
typealias GenerationConfig = XybridGenerationConfig

// -- GenerationConfig Presets --

/** Preset factory methods for [GenerationConfig]. */
object GenerationConfigs {
    /**
     * Greedy decoding preset (deterministic, temperature=0).
     *
     * Produces the same output every time for the same input.
     */
    @JvmStatic
    fun greedy() = XybridGenerationConfig(
        maxTokens = null,
        temperature = 0.0f,
        topP = 1.0f,
        minP = null,
        topK = 0u,
        repetitionPenalty = null,
        stopSequences = null,
    )

    /**
     * Creative generation preset (higher temperature).
     *
     * Produces more varied and creative output.
     */
    @JvmStatic
    fun creative() = XybridGenerationConfig(
        maxTokens = null,
        temperature = 0.9f,
        topP = 0.95f,
        minP = null,
        topK = 50u,
        repetitionPenalty = null,
        stopSequences = null,
    )
}

// -- XybridResult Extensions --

/** Returns `true` if inference failed. */
val XybridResult.isFailure: Boolean get() = !success

/** The latency in seconds as a Double. */
val XybridResult.latencySeconds: Double get() = latencyMs.toDouble() / 1000.0

// -- XybridEnvelope Factory Methods --

/** Factory methods for creating [XybridEnvelope] instances. */
object Envelope {
    /**
     * Creates an audio envelope from raw PCM data.
     * @param bytes Raw PCM audio bytes
     * @param sampleRate Sample rate in Hz (default: 16000)
     * @param channels Number of audio channels (default: 1)
     */
    @JvmStatic
    fun audio(bytes: ByteArray, sampleRate: UInt = 16000u, channels: UInt = 1u): XybridEnvelope =
        XybridEnvelope.Audio(bytes, sampleRate, channels)

    /** Creates a text envelope for TTS with default voice. */
    @JvmStatic
    fun text(text: String): XybridEnvelope = XybridEnvelope.Text(text, null, null)

    /**
     * Creates a text envelope for TTS with voice and speed options.
     * @param voiceId Voice ID (e.g., "af_heart" for Kokoro)
     * @param speed Speed multiplier (1.0 = normal, default)
     */
    @JvmStatic
    @JvmOverloads
    fun text(text: String, voiceId: String, speed: Double = 1.0): XybridEnvelope =
        XybridEnvelope.Text(text, voiceId, speed)

    /** Creates an embedding envelope from raw vector data. */
    @JvmStatic
    fun embedding(data: List<Float>): XybridEnvelope = XybridEnvelope.Embedding(data)
}

// -- XybridVoiceInfo Extensions --

/** Returns `true` if the voice gender is male. */
val XybridVoiceInfo.isMale: Boolean get() = gender == "male"

/** Returns `true` if the voice gender is female. */
val XybridVoiceInfo.isFemale: Boolean get() = gender == "female"

// -- XybridException Extensions --

/** User-friendly error message for display. */
val XybridException.displayMessage: String
    get() = message ?: "Unknown error"

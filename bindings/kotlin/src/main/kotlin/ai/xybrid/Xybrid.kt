/**
 * Xybrid SDK for Android
 * Convenience wrappers and extensions for the UniFFI-generated bindings.
 * For full API documentation, see https://docs.xybrid.dev/sdk/kotlin
 */
@file:Suppress("unused")

package ai.xybrid

import android.content.Context
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
     * This must be called once before using any Xybrid functionality.
     * It is safe to call multiple times — subsequent calls are no-ops.
     *
     * Typically called from `Application.onCreate()` or `Activity.onCreate()`.
     *
     * @param context Android context (application or activity).
     */
    @JvmStatic
    fun init(context: Context) {
        if (initialized) return
        synchronized(this) {
            if (initialized) return
            val cacheDir = File(context.filesDir, "xybrid/models")
            initSdkCacheDir(cacheDir.absolutePath)
            initialized = true
        }
    }

    /** Returns `true` if [init] has been called successfully. */
    @JvmStatic
    val isInitialized: Boolean get() = initialized
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

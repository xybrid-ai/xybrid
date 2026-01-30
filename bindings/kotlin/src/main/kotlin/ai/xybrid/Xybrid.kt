/**
 * Xybrid SDK for Android
 * Convenience wrappers and extensions for the UniFFI-generated bindings.
 * For full API documentation, see https://docs.xybrid.dev/sdk/kotlin
 */
@file:Suppress("unused")

package ai.xybrid

// -- Public Type Aliases --

/** Loads ML models from the registry or local bundles. */
typealias ModelLoader = XybridModelLoader

/** A loaded model ready for inference. */
typealias Model = XybridModel

/** Input data for model inference. Use [Envelope] factory methods. */
typealias Envelope = XybridEnvelope

/** The result of a model inference operation. */
typealias Result = XybridResult

/** Errors that can occur during model loading or inference. */
typealias XybridError = XybridException

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
    @JvmOverloads
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

// -- XybridException Extensions --

/** User-friendly error message for display. */
val XybridException.displayMessage: String
    get() = when (this) {
        is XybridException.ModelNotFound -> "Model not found: $modelId"
        is XybridException.InferenceFailed -> "Inference failed: $message"
        is XybridException.InvalidInput -> "Invalid input: $message"
        is XybridException.IoException -> "I/O error: $message"
    }

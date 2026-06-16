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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

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
     * Idempotent and thread-safe — subsequent calls after a successful
     * initialization are no-ops.
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

/**
 * Run inference with the model's default options.
 *
 * Convenience over the generated [XybridModel.run] (which takes an
 * `XybridRunOptions?`) so simple call sites stay one-argument. Forwards
 * `null` options. Use the two-arg `run(envelope, options)` to override
 * generation config, abort signals, or cloud-fallback behaviour.
 */
fun XybridModel.run(envelope: XybridEnvelope): XybridResult = this.run(envelope, null)

// -- Async (suspend) conveniences --
//
// bolt's load/run are synchronous + blocking. These suspend wrappers restore the
// pre-migration suspend API shape: each runs the blocking call on
// `Dispatchers.IO`, so coroutine callers `suspend` without blocking the calling
// thread (e.g. the main/UI thread).
//
// (boltffi *can* export `async fn` natively, but the SDK's async path uses tokio
// `spawn_blocking`, which needs an ambient tokio runtime context that boltffi's
// future driver does not establish. Wrapping the synchronous call on a worker
// dispatcher is therefore the correct, low-risk way to surface suspend today.)

/** Load a model from the xybrid registry off the caller's thread. */
suspend fun XybridModel.Companion.fromRegistryAsync(id: String): XybridModel =
    withContext(Dispatchers.IO) { XybridModel(id) }

/** Load a model from a local directory off the caller's thread. */
suspend fun XybridModel.Companion.fromDirectoryAsync(path: String): XybridModel =
    withContext(Dispatchers.IO) { fromDirectory(path) }

/** Load a model from a local `.xyb` bundle off the caller's thread. */
suspend fun XybridModel.Companion.fromBundleAsync(path: String): XybridModel =
    withContext(Dispatchers.IO) { fromBundle(path) }

/** Resolve and load a model from a HuggingFace repo off the caller's thread. */
suspend fun XybridModel.Companion.fromHuggingfaceAsync(repo: String): XybridModel =
    withContext(Dispatchers.IO) { fromHuggingface(repo) }

/** Run inference off the caller's thread (on [Dispatchers.IO]). */
suspend fun XybridModel.runAsync(
    envelope: XybridEnvelope,
    options: XybridRunOptions? = null,
): XybridResult = withContext(Dispatchers.IO) { this@runAsync.run(envelope, options) }

/** Warm up the model off the caller's thread (on [Dispatchers.IO]). */
suspend fun XybridModel.warmupAsync() = withContext(Dispatchers.IO) { this@warmupAsync.warmup() }

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

    /**
     * Creates an encoded image envelope for vision-language models. The format
     * hint is normalized and validated up front (`jpg` -> `jpeg`; unsupported
     * formats throw [XybridError.ConfigError], mirroring the Swift binding);
     * the bytes themselves are decode-validated on the Rust side at run time
     * (surfacing as a [XybridError.InvalidImage] for bad or oversized input).
     * @param bytes Encoded PNG, JPEG, or WebP bytes.
     * @param format Image format hint (`png`, `jpeg`, `jpg`, or `webp`).
     */
    @JvmStatic
    fun image(bytes: ByteArray, format: String): XybridEnvelope =
        XybridEnvelope(
            kind = XybridEnvelopeKind.Image(bytes, normalizeImageFormat(format)),
            metadata = emptyList(),
        )

    /**
     * Creates a multimodal user message: prompt text plus image attachments,
     * tagged with the `User` role.
     * @param text User prompt text.
     * @param images Image envelopes created by [image].
     */
    @JvmStatic
    @JvmOverloads
    fun userMessage(text: String, images: List<XybridEnvelope> = emptyList()): XybridEnvelope {
        if (!images.all { it.kind is XybridEnvelopeKind.Image }) {
            throw XybridError.ConfigError("Envelope.userMessage accepts only image envelopes")
        }
        val parts = mutableListOf(
            XybridEnvelope(kind = XybridEnvelopeKind.Text(text), metadata = emptyList()),
        )
        parts.addAll(images)
        return XybridEnvelope(
            kind = XybridEnvelopeKind.MultiPart(parts),
            metadata = listOf(XybridMetadataEntry("xybrid.role", "user")),
        )
    }

    /**
     * Normalizes an image format hint to the canonical lowercase form the
     * Rust core expects (`jpg` -> `jpeg`), rejecting unsupported formats early
     * with [XybridError.ConfigError] rather than deferring to a run-time
     * [XybridError.InvalidImage]. Mirrors the Swift binding's
     * `normalizeImageFormat`.
     */
    private fun normalizeImageFormat(format: String): String =
        when (val normalized = format.trim().lowercase()) {
            "jpg" -> "jpeg"
            "jpeg", "png", "webp" -> normalized
            else -> throw XybridError.ConfigError(
                "Unsupported image format '$format'. Supported formats: png, jpeg, jpg, webp",
            )
        }
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

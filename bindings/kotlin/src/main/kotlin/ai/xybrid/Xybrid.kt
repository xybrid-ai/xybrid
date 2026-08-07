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
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
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

    /**
     * Describe a registry model without resolving, downloading, or loading it.
     *
     * Call [XybridModelLoader.load] from a coroutine to perform the load, or
     * [XybridModelLoader.loadBlocking] from an existing worker thread.
     */
    @JvmStatic
    fun model(id: String): ModelLoader = model(ModelSource.registry(id))

    /**
     * Describe a model source without resolving, downloading, or loading it.
     */
    @JvmStatic
    fun model(source: ModelSource): ModelLoader = XybridModelLoader.from(source)

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

// -- Model loading --

/**
 * A model location that can be prepared without performing I/O.
 *
 * Construct a loader with [Xybrid.model] or [XybridModelLoader.from]. Loading
 * is always a separate, explicit operation.
 */
sealed interface ModelSource {
    /** A model resolved through the Xybrid registry. */
    data class Registry(val id: String) : ModelSource

    /** A local `.xyb` model bundle. */
    data class Bundle(val path: String) : ModelSource

    /** A local directory containing `model_metadata.json`. */
    data class Directory(val path: String) : ModelSource

    /** A Hugging Face repository (`org/repo` or `org/repo:variant`). */
    data class HuggingFace(val repo: String) : ModelSource

    /**
     * A registry model served from the cloud gateway while its weights
     * download in the background. See
     * [XybridModelLoader.fromRegistrySpeculative].
     */
    data class RegistrySpeculative(val id: String) : ModelSource

    companion object {
        /** Describe a registry model. */
        @JvmStatic
        fun registry(id: String): ModelSource = Registry(id)

        /** Describe a local `.xyb` bundle. */
        @JvmStatic
        fun bundle(path: String): ModelSource = Bundle(path)

        /** Describe a local model directory. */
        @JvmStatic
        fun directory(path: String): ModelSource = Directory(path)

        /** Describe a Hugging Face repository. */
        @JvmStatic
        fun huggingFace(repo: String): ModelSource = HuggingFace(repo)

        /** Describe a registry model to be served from the cloud while it downloads. */
        @JvmStatic
        fun registrySpeculative(id: String): ModelSource = RegistrySpeculative(id)
    }
}

/**
 * A cheap model reference that defers all expensive work until [load].
 *
 * Creating a loader never performs network, disk, or native-runtime work.
 */
class XybridModelLoader private constructor(
    /** The source this loader will resolve. */
    val source: ModelSource,
) {
    /** Load the model without blocking the calling coroutine's thread. */
    suspend fun load(): XybridModel = withContext(Dispatchers.IO) { loadBlocking() }

    /**
     * Load the model synchronously.
     *
     * This may resolve registry metadata, download files, access disk, and
     * initialize the inference runtime. Do not call it from Android's main
     * thread.
     */
    fun loadBlocking(): XybridModel = when (val current = source) {
        is ModelSource.Registry -> XybridModel(current.id)
        is ModelSource.Bundle -> XybridModel.fromBundle(current.path)
        is ModelSource.Directory -> XybridModel.fromDirectory(current.path)
        is ModelSource.HuggingFace -> XybridModel.fromHuggingface(current.repo)
        is ModelSource.RegistrySpeculative ->
            XybridModel.fromRegistrySpeculative(current.id)
    }

    /**
     * Whether [load] would actually speculate: speculation is possible for this
     * source, an API key resolves, and the model is not already cached.
     *
     * Always `false` for non-speculative sources. Never touches the network.
     */
    val willSpeculate: Boolean
        get() = when (val current = source) {
            is ModelSource.RegistrySpeculative -> willSpeculateForModel(current.id)
            else -> false
        }

    companion object {
        /** Create a loader for an already-described source. */
        @JvmStatic
        fun from(source: ModelSource): XybridModelLoader = XybridModelLoader(source)

        /** Create a loader for a registry model. */
        @JvmStatic
        fun fromRegistry(id: String): XybridModelLoader = from(ModelSource.registry(id))

        /**
         * Create a loader that answers from the cloud gateway while the
         * registry weights download in the background, instead of blocking on
         * the download.
         *
         * [load] returns almost immediately with a cloud-backed model that
         * switches to on-device by itself once the download lands. Requires an
         * API key and an uncached model — otherwise it behaves exactly like
         * [fromRegistry], which [willSpeculate] reports up front. LLM/chat
         * models only.
         */
        @JvmStatic
        fun fromRegistrySpeculative(id: String): XybridModelLoader =
            from(ModelSource.registrySpeculative(id))

        /** Create a loader for a local `.xyb` bundle. */
        @JvmStatic
        fun fromBundle(path: String): XybridModelLoader = from(ModelSource.bundle(path))

        /** Create a loader for a local model directory. */
        @JvmStatic
        fun fromDirectory(path: String): XybridModelLoader = from(ModelSource.directory(path))

        /** Create a loader for a Hugging Face repository. */
        @JvmStatic
        fun fromHuggingFace(repo: String): XybridModelLoader =
            from(ModelSource.huggingFace(repo))

        /** Compatibility spelling retained for existing Kotlin callers. */
        @JvmStatic
        fun fromHuggingface(repo: String): XybridModelLoader = fromHuggingFace(repo)
    }
}

/** Idiomatic short name for the high-level model loader. */
typealias ModelLoader = XybridModelLoader

// -- Public Type Aliases --

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
@Deprecated(
    message = "Use Xybrid.model(id).load().",
    replaceWith = ReplaceWith("Xybrid.model(id).load()"),
)
suspend fun XybridModel.Companion.fromRegistryAsync(id: String): XybridModel =
    Xybrid.model(id).load()

/** Load a model from a local directory off the caller's thread. */
@Deprecated(
    message = "Use Xybrid.model(ModelSource.directory(path)).load().",
    replaceWith = ReplaceWith("Xybrid.model(ModelSource.directory(path)).load()"),
)
suspend fun XybridModel.Companion.fromDirectoryAsync(path: String): XybridModel =
    Xybrid.model(ModelSource.directory(path)).load()

/** Load a model from a local `.xyb` bundle off the caller's thread. */
@Deprecated(
    message = "Use Xybrid.model(ModelSource.bundle(path)).load().",
    replaceWith = ReplaceWith("Xybrid.model(ModelSource.bundle(path)).load()"),
)
suspend fun XybridModel.Companion.fromBundleAsync(path: String): XybridModel =
    Xybrid.model(ModelSource.bundle(path)).load()

/** Resolve and load a model from a HuggingFace repo off the caller's thread. */
@Deprecated(
    message = "Use Xybrid.model(ModelSource.huggingFace(repo)).load().",
    replaceWith = ReplaceWith("Xybrid.model(ModelSource.huggingFace(repo)).load()"),
)
suspend fun XybridModel.Companion.fromHuggingfaceAsync(repo: String): XybridModel =
    Xybrid.model(ModelSource.huggingFace(repo)).load()

/** Run inference off the caller's thread (on [Dispatchers.IO]). */
suspend fun XybridModel.runAsync(
    envelope: XybridEnvelope,
    options: XybridRunOptions? = null,
): XybridResult = withContext(Dispatchers.IO) { this@runAsync.run(envelope, options) }

/** Warm up the model off the caller's thread (on [Dispatchers.IO]). */
suspend fun XybridModel.warmupAsync() = withContext(Dispatchers.IO) { this@warmupAsync.warmup() }

/** Unload the model, freeing its memory, off the caller's thread (on [Dispatchers.IO]). */
suspend fun XybridModel.unloadAsync() = withContext(Dispatchers.IO) { this@unloadAsync.unload() }

/**
 * Stream inference token-by-token as a cold [Flow].
 *
 * Emits each [XybridStreamToken] as it is generated and completes when
 * generation finishes; throws [XybridError] if the run fails mid-stream.
 * Cancelling the collecting coroutine aborts generation at the next token
 * boundary — the session is closed, which unwinds the backend instead of
 * running to `max_tokens`. Ergonomic wrapper over the pull-based session API
 * ([XybridModel.runStream] / [XybridModel.streamNext] /
 * [XybridModel.streamClose]).
 *
 * ```kotlin
 * model.streamTokens(envelope).collect { token -> print(token.token) }
 * ```
 */
fun XybridModel.streamTokens(
    envelope: XybridEnvelope,
    options: XybridRunOptions? = null,
): Flow<XybridStreamToken> = flow {
    val streamId = runStream(envelope, options)
    try {
        while (true) {
            // Cooperative cancellation: collecting coroutine cancelled -> throws
            // here at the next token boundary, the finally closes the session.
            currentCoroutineContext().ensureActive()
            val event = streamNext(streamId)
            when (event.kind) {
                XybridStreamEventKind.TOKEN -> event.token?.let { emit(it) }
                XybridStreamEventKind.COMPLETE -> break
            }
        }
    } finally {
        // Idempotent (the session may already be gone after an error), and
        // aborts an in-flight run when collection stops early.
        streamClose(streamId)
    }
}.flowOn(Dispatchers.IO)

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

/**
 * The model's chain-of-thought / reasoning text (LLM `<think>` blocks),
 * surfaced separately from [text], which always excludes it. `null` when the
 * model emitted no reasoning or the backend doesn't surface one.
 *
 * Carried on the envelope's `reasoning_content` metadata rather than the
 * payload `kind`, so it reads from `metadata` rather than the enum.
 */
val XybridResult.reasoningContent: String?
    get() = envelope.metadata.firstOrNull { it.key == "reasoning_content" }?.value

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
    get() = when (this) {
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
        is XybridError.MissingArtifact -> message
        is XybridError.UnsupportedModelCapability -> message
        is XybridError.UnsupportedBackendCapability -> message
        is XybridError.InvalidImage -> message
    }

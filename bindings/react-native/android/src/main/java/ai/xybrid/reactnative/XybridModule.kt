package ai.xybrid.reactnative

import ai.xybrid.Xybrid
import ai.xybrid.XybridCancellationToken
import ai.xybrid.XybridConversationContext
import ai.xybrid.XybridDownload
import ai.xybrid.XybridError
import ai.xybrid.XybridModel
import ai.xybrid.XybridPartialResult
import ai.xybrid.XybridPipeline
import ai.xybrid.XybridStreamEventKind
import ai.xybrid.XybridStreamingSession
import ai.xybrid.XybridThermalState
import ai.xybrid.partials
import android.content.Context
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import java.io.File
import kotlin.math.roundToInt
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

// The React Native TurboModule. Extending the Codegen-generated
// `NativeXybridSpec` makes the compiler hold this class to
// src/NativeXybrid.ts; each override decodes its arguments (XybridCodec),
// calls the bolt Kotlin SDK from the `ai.xybrid:xybrid-kotlin` AAR, and
// settles the promise with a typed `xybrid_*` code on failure.
//
// Blocking SDK calls (load, run, stream pulls, downloads, disk walks) run on
// Dispatchers.IO; cheap getters settle inline. Every call that touches a
// native object holds a lease on it (see XybridHandles), so disposing a
// handle mid-call can never free memory the call is using.

class XybridModule(reactContext: ReactApplicationContext) : NativeXybridSpec(reactContext) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val handles = XybridHandles()

  // Module teardown (reload, host shutdown): stop in-flight work, then free
  // every object as soon as the calls using it return.
  override fun invalidate() {
    handles.forEachLive(XybridCancellationToken::class.java) { it.cancel() }
    handles.disposeAll()
    scope.cancel()
    super.invalidate()
  }

  // -- Settling promises ------------------------------------------------------

  private inline fun now(promise: Promise, block: () -> Any?) {
    try {
      SdkSetup.ensureBase(reactApplicationContext)
      promise.resolve(toJs(block()))
    } catch (t: Throwable) {
      reject(promise, t)
    }
  }

  private fun background(promise: Promise, block: suspend () -> Any?) {
    scope.launch {
      try {
        SdkSetup.ensureBase(reactApplicationContext)
        promise.resolve(toJs(block()))
      } catch (e: CancellationException) {
        promise.reject("xybrid_cancelled", "The Xybrid module was torn down", e)
        throw e
      } catch (t: Throwable) {
        reject(promise, t)
      }
    }
  }

  /**
   * The SDK reports a run stopped by its token as an inference error
   * ("user_cancelled"); JS gets the dedicated `xybrid_cancelled` code.
   */
  private fun cancellationAware(error: Throwable, token: XybridCancellationToken): Throwable =
    if (error !is BridgeException && error !is CancellationException && token.isCancelled()) {
      XybridError.Cancelled("The run was cancelled")
    } else {
      error
    }

  private fun reject(promise: Promise, error: Throwable) {
    val (code, message) = XybridCodec.rejection(error)
    promise.reject(code, message, error)
  }

  /** Codec output (maps, lists, Doubles) → what `Promise.resolve` accepts. */
  @Suppress("UNCHECKED_CAST")
  private fun toJs(value: Any?): Any? = when (value) {
    null, is Unit -> null
    is Map<*, *> -> Arguments.makeNativeMap(value as Map<String, Any?>)
    is List<*> -> Arguments.makeNativeArray(value)
    is Int -> value.toDouble()
    is Long -> value.toDouble()
    is Float -> value.toDouble()
    else -> value
  }

  // -- SDK configuration ------------------------------------------------------

  override fun initialize(options: ReadableMap?, promise: Promise) {
    try {
      val what = "initialize options"
      val o = options?.toHashMap() ?: emptyMap()
      val cacheDir = XybridCodec.optionalString(o, "cacheDir", what).nonBlank()
      SdkSetup.ensureBase(reactApplicationContext, cacheDir?.let(XybridCodec::filePath))
      SdkSetup.configure(
        SdkSetup.Runtime(
          apiKey = XybridCodec.optionalString(o, "apiKey", what).nonBlank(),
          gatewayUrl = XybridCodec.optionalString(o, "gatewayUrl", what).nonBlank(),
          ingestUrl = XybridCodec.optionalString(o, "ingestUrl", what).nonBlank(),
        ),
      )
      promise.resolve(null)
    } catch (t: Throwable) {
      reject(promise, t)
    }
  }

  override fun sdkVersion(promise: Promise) = now(promise) { ai.xybrid.version() }

  override fun hasApiKey(promise: Promise) = now(promise) { ai.xybrid.hasApiKey() }

  override fun setProviderApiKey(provider: String, apiKey: String, promise: Promise) =
    now(promise) { ai.xybrid.setProviderApiKey(provider, apiKey) }

  override fun setPlatformUrl(url: String, promise: Promise) =
    now(promise) { ai.xybrid.setPlatformUrl(url) }

  override fun setSpeculativeCloud(enabled: Boolean, promise: Promise) =
    now(promise) { ai.xybrid.setSpeculativeCloud(enabled) }

  override fun isSpeculativeCloudEnabled(promise: Promise) =
    now(promise) { ai.xybrid.isSpeculativeCloudEnabled() }

  override fun willSpeculate(modelId: String, promise: Promise) =
    background(promise) { ai.xybrid.willSpeculateForModel(modelId) }

  override fun releaseMemory(promise: Promise) =
    background(promise) { ai.xybrid.releaseMemory().toDouble() }

  override fun setAutoRelease(enabled: Boolean, promise: Promise) =
    now(promise) { ai.xybrid.setAutoRelease(enabled) }

  override fun isAutoReleaseEnabled(promise: Promise) =
    now(promise) { ai.xybrid.isAutoReleaseEnabled() }

  // -- Device state push ------------------------------------------------------

  override fun setBatteryLevel(percent: Double, promise: Promise) = now(promise) {
    if (!percent.isFinite()) throw BridgeException.InvalidArgument("battery level must be a finite number")
    ai.xybrid.setBatteryLevel(percent.roundToInt().coerceIn(0, 100).toUByte())
  }

  override fun clearBatteryLevel(promise: Promise) = now(promise) { ai.xybrid.clearBatteryLevel() }

  override fun setThermalState(state: String, promise: Promise) = now(promise) {
    val mapped = when (state) {
      "normal" -> XybridThermalState.NORMAL
      "warm" -> XybridThermalState.WARM
      "hot" -> XybridThermalState.HOT
      "critical" -> XybridThermalState.CRITICAL
      else -> throw BridgeException.InvalidArgument("unknown thermal state '$state'")
    }
    ai.xybrid.setThermalState(mapped)
  }

  override fun clearThermalState(promise: Promise) = now(promise) { ai.xybrid.clearThermalState() }

  // -- Model cache (disk walks: off the calling thread) ----------------------

  override fun cacheStatus(promise: Promise) =
    background(promise) { XybridCodec.encodeCacheStatus(ai.xybrid.cacheStatus()) }

  override fun cacheEntries(promise: Promise) =
    background(promise) { ai.xybrid.cacheEntries().map(XybridCodec::encodeCacheEntry) }

  override fun cacheIsModelCached(modelId: String, promise: Promise) =
    background(promise) { ai.xybrid.cacheIsModelCached(modelId) }

  override fun cacheModelPath(modelId: String, promise: Promise) =
    background(promise) { ai.xybrid.cacheModelPath(modelId) }

  override fun cacheExtractedModelIds(promise: Promise) =
    background(promise) { ai.xybrid.cacheListExtractedModelIds() }

  override fun cacheRemoveModel(modelId: String, promise: Promise) =
    background(promise) { ai.xybrid.cacheRemoveModel(modelId).toDouble() }

  override fun cacheClear(promise: Promise) = background(promise) { ai.xybrid.cacheClear().toDouble() }

  // -- Stateless helpers ------------------------------------------------------

  override fun jsonSchemaToGbnf(schemaJson: String, promise: Promise) =
    now(promise) { ai.xybrid.jsonSchemaToGbnf(schemaJson) }

  override fun toolResultsEnvelope(
    userText: String,
    priorAssistantText: String,
    results: ReadableArray,
    promise: Promise,
  ) = now(promise) {
    val decoded = results.toArrayList().map(XybridCodec::decodeToolResult)
    XybridCodec.encodeEnvelope(ai.xybrid.toolResultsEnvelope(userText, priorAssistantText, decoded))
  }

  // -- Handles ----------------------------------------------------------------

  override fun dispose(handle: String, promise: Promise) {
    handles.dispose(handle)
    promise.resolve(null)
  }

  // -- Models -----------------------------------------------------------------

  override fun loadModel(source: ReadableMap, promise: Promise) {
    val raw = source.toHashMap()
    background(promise) {
      val what = "model source"
      val o = XybridCodec.obj(raw, what)
      val kind = XybridCodec.string(o, "kind", what)
      val value = XybridCodec.string(o, "value", what)
      val model = when (kind) {
        "registry" -> XybridModel.fromRegistry(value)
        "registrySpeculative" -> XybridModel.fromRegistrySpeculative(value)
        "bundle" -> XybridModel.fromBundle(XybridCodec.filePath(value))
        "directory" -> XybridModel.fromDirectory(XybridCodec.filePath(value))
        "huggingFace" -> XybridCodec.optionalString(o, "revision", what)
          ?.let { XybridModel.fromHuggingfaceWithRevision(value, it) }
          ?: XybridModel.fromHuggingface(value)
        "modelFile" -> XybridModel.fromModelFile(XybridCodec.filePath(value))
        else -> throw BridgeException.InvalidArgument("unknown model source kind '$kind'")
      }
      handles.insert(model, "model")
    }
  }

  override fun modelInfo(model: String, promise: Promise) = now(promise) {
    handles.use(model, XybridModel::class.java) { m ->
      mapOf(
        "modelId" to m.modelId(),
        "version" to m.version(),
        "outputType" to XybridCodec.encodeOutputType(m.outputType()),
        "isLlm" to m.isLlm(),
        "supportsStreaming" to m.supportsStreaming(),
        "supportsTokenStreaming" to m.supportsTokenStreaming(),
        "supportsToolCalling" to m.supportsToolCalling(),
        "hasVoices" to m.hasVoices(),
        "defaultGenerationConfig" to XybridCodec.encodeGenerationConfig(m.defaultGenerationConfig()),
      )
    }
  }

  override fun isLoaded(model: String, promise: Promise) =
    now(promise) { handles.use(model, XybridModel::class.java) { it.isLoaded() } }

  override fun warmup(model: String, promise: Promise) =
    background(promise) { handles.use(model, XybridModel::class.java) { it.warmup() } }

  override fun unload(model: String, promise: Promise) =
    background(promise) { handles.use(model, XybridModel::class.java) { it.unload() } }

  override fun isCloudServing(model: String, promise: Promise) =
    now(promise) { handles.use(model, XybridModel::class.java) { it.isCloudServing() } }

  override fun downloadStatus(model: String, promise: Promise) = now(promise) {
    handles.use(model, XybridModel::class.java) { XybridCodec.encodeDownloadStatus(it.downloadStatus()) }
  }

  override fun awaitDownload(model: String, timeoutMs: Double, promise: Promise) = background(promise) {
    // NaN and negatives clamp to 0 (a non-blocking read); huge values saturate.
    val timeout = if (timeoutMs.isFinite()) timeoutMs.coerceAtLeast(0.0).toULong() else 0uL
    handles.use(model, XybridModel::class.java) { XybridCodec.encodeDownloadStatus(it.awaitDownload(timeout)) }
  }

  override fun voices(model: String, promise: Promise) = now(promise) {
    handles.use(model, XybridModel::class.java) { m -> m.voices().map(XybridCodec::encodeVoice) }
  }

  override fun defaultVoice(model: String, promise: Promise) = now(promise) {
    handles.use(model, XybridModel::class.java) { m -> m.defaultVoice()?.let(XybridCodec::encodeVoice) }
  }

  override fun voice(model: String, voiceId: String, promise: Promise) = now(promise) {
    handles.use(model, XybridModel::class.java) { m -> m.voice(voiceId)?.let(XybridCodec::encodeVoice) }
  }

  // -- Inference --------------------------------------------------------------

  /**
   * Everything one run needs, leased for its duration: the model, the
   * optional conversation context, and a stop button — the caller's token,
   * or one of our own registered under the model so disposing the model (or
   * tearing the module down) stops the run.
   */
  private inner class RunScope(
    private val modelHandle: String,
    rawOptions: Any?,
  ) : AutoCloseable {
    val request = XybridCodec.decodeRunRequest(rawOptions)
    private val held = mutableListOf<AutoCloseable>()
    private var ownToken: String? = null

    val model: XybridModel
    val context: XybridConversationContext?
    val cancel: XybridCancellationToken

    init {
      try {
        model = hold(handles.lease(modelHandle, XybridModel::class.java))
        context = request.context?.let { hold(handles.lease(it, XybridConversationContext::class.java)) }
        val tokenHandle = request.cancel ?: XybridCancellationToken().let { token ->
          handles.insert(token, "cancel", owner = modelHandle, onDispose = token::cancel)
            .also { ownToken = it }
        }
        cancel = hold(handles.lease(tokenHandle, XybridCancellationToken::class.java))
      } catch (t: Throwable) {
        close()
        throw t
      }
    }

    private fun <T : Any> hold(lease: XybridHandles.Lease<T>): T {
      held.add(lease)
      return lease.value
    }

    /**
     * Hand every lease — and our own token, if we made one — over to a
     * stream entry, which releases them (last first) when it is closed.
     */
    fun transferToStream(): List<AutoCloseable> {
      val transferred = held.toMutableList()
      ownToken?.let { handle -> transferred.add(AutoCloseable { handles.dispose(handle) }) }
      held.clear()
      ownToken = null
      return transferred
    }

    override fun close() {
      held.asReversed().forEach(AutoCloseable::close)
      held.clear()
      ownToken?.let(handles::dispose)
    }
  }

  override fun run(model: String, envelope: ReadableMap, options: ReadableMap?, promise: Promise) {
    val rawEnvelope = envelope.toHashMap()
    val rawOptions = options?.toHashMap()
    background(promise) {
      val input = XybridCodec.decodeEnvelope(rawEnvelope)
      RunScope(model, rawOptions).use { run ->
        val context = run.context
        val result = try {
          if (context != null) {
            run.model.runWithContext(input, context, run.request.options, run.cancel)
          } else {
            run.model.run(input, run.request.options, run.cancel)
          }
        } catch (t: Throwable) {
          throw cancellationAware(t, run.cancel)
        }
        XybridCodec.encodeResult(result)
      }
    }
  }

  /** A pull-based token stream plus the leases that keep its model and token alive. */
  private class TokenStream(
    val model: XybridModel,
    val streamId: ULong,
    val cancel: XybridCancellationToken,
    private val resources: List<AutoCloseable>,
  ) {
    /** Abort generation now; safe while a pull is in flight (see XybridHandles). */
    fun abort() {
      cancel.cancel()
      model.streamClose(streamId)
    }

    fun release() = resources.asReversed().forEach(AutoCloseable::close)
  }

  override fun streamStart(model: String, envelope: ReadableMap, options: ReadableMap?, promise: Promise) {
    val rawEnvelope = envelope.toHashMap()
    val rawOptions = options?.toHashMap()
    background(promise) {
      val input = XybridCodec.decodeEnvelope(rawEnvelope)
      RunScope(model, rawOptions).use { run ->
        val context = run.context
        val streamId = if (context != null) {
          run.model.runStreamWithContext(input, context, run.request.options, run.cancel)
        } else {
          run.model.runStream(input, run.request.options, run.cancel)
        }
        val stream = TokenStream(run.model, streamId, run.cancel, run.transferToStream())
        handles.insert(stream, "stream", owner = model, onDispose = stream::abort, onClose = stream::release)
      }
    }
  }

  override fun streamNext(stream: String, promise: Promise) = background(promise) {
    // A disposed stream reads as exhausted, not as an error.
    val lease = handles.leaseOrNull(stream, TokenStream::class.java) ?: return@background null
    lease.use {
      val s = it.value
      try {
        val event = s.model.streamNext(s.streamId)
        when (event.kind) {
          XybridStreamEventKind.TOKEN -> {
            val token = event.token
              ?: throw XybridError.InferenceError("stream returned a token event without a token")
            mapOf("kind" to "token", "token" to XybridCodec.encodeStreamToken(token))
          }
          XybridStreamEventKind.COMPLETE -> {
            // `streamResult` closes the bolt session; drop our entry with it.
            val result = s.model.streamResult(s.streamId)
            handles.dispose(stream)
            mapOf("kind" to "complete", "result" to XybridCodec.encodeResult(result))
          }
        }
      } catch (t: Throwable) {
        // A failed pull already closed the bolt session.
        handles.dispose(stream)
        throw cancellationAware(t, s.cancel)
      }
    }
  }

  // -- Cancellation tokens ----------------------------------------------------

  override fun createCancelToken(promise: Promise) =
    now(promise) { handles.insert(XybridCancellationToken(), "cancel") }

  override fun cancel(token: String, promise: Promise) =
    now(promise) { handles.use(token, XybridCancellationToken::class.java) { it.cancel() } }

  // -- Conversation contexts ----------------------------------------------------

  override fun createContext(contextId: String?, promise: Promise) = now(promise) {
    val context = contextId?.let { XybridConversationContext.withId(it) } ?: XybridConversationContext()
    handles.insert(context, "context")
  }

  override fun contextPush(context: String, envelope: ReadableMap, promise: Promise) {
    val raw = envelope.toHashMap()
    // Image turns are decode-validated natively, so this can take a moment.
    background(promise) {
      handles.use(context, XybridConversationContext::class.java) { it.push(XybridCodec.decodeEnvelope(raw)) }
    }
  }

  override fun contextSetSystem(context: String, envelope: ReadableMap, promise: Promise) {
    val raw = envelope.toHashMap()
    background(promise) {
      handles.use(context, XybridConversationContext::class.java) { it.setSystem(XybridCodec.decodeEnvelope(raw)) }
    }
  }

  override fun contextClear(context: String, promise: Promise) =
    now(promise) { handles.use(context, XybridConversationContext::class.java) { it.clear() } }

  override fun contextInfo(context: String, promise: Promise) = now(promise) {
    handles.use(context, XybridConversationContext::class.java) { c ->
      mapOf("id" to c.id(), "historyLength" to c.historyLen().toDouble(), "hasSystem" to c.hasSystem())
    }
  }

  override fun contextHistory(context: String, promise: Promise) = now(promise) {
    handles.use(context, XybridConversationContext::class.java) { c -> c.history().map(XybridCodec::encodeEnvelope) }
  }

  override fun contextSetMaxHistoryLength(context: String, length: Double, promise: Promise) = now(promise) {
    val max = XybridCodec.uint(length, "max history length")
    handles.use(context, XybridConversationContext::class.java) { it.setMaxHistoryLen(max) }
  }

  // -- Standalone downloads ---------------------------------------------------

  override fun startDownload(modelId: String, platform: String?, promise: Promise) = background(promise) {
    val download = platform?.let { XybridDownload.fromRegistryWithPlatform(modelId, it) }
      ?: XybridDownload.fromRegistry(modelId)
    handles.insert(download, "download")
  }

  override fun downloadHandleStatus(download: String, promise: Promise) = now(promise) {
    handles.use(download, XybridDownload::class.java) { XybridCodec.encodeDownloadStatus(it.status()) }
  }

  override fun downloadHandleError(download: String, promise: Promise) =
    now(promise) { handles.use(download, XybridDownload::class.java) { it.error() } }

  override fun cancelDownload(download: String, promise: Promise) =
    now(promise) { handles.use(download, XybridDownload::class.java) { it.cancel() } }

  // -- Pipelines ----------------------------------------------------------------

  override fun loadPipeline(source: ReadableMap, promise: Promise) {
    val raw = source.toHashMap()
    background(promise) {
      val what = "pipeline source"
      val o = XybridCodec.obj(raw, what)
      val kind = XybridCodec.string(o, "kind", what)
      val value = XybridCodec.string(o, "value", what)
      val pipeline = when (kind) {
        "yaml" -> XybridPipeline.fromYaml(value)
        "file" -> XybridPipeline.fromFile(XybridCodec.filePath(value))
        "bundle" -> XybridPipeline.fromBundle(XybridCodec.filePath(value))
        else -> throw BridgeException.InvalidArgument("unknown pipeline source kind '$kind'")
      }
      handles.insert(pipeline, "pipeline")
    }
  }

  override fun pipelineInfo(pipeline: String, promise: Promise) = now(promise) {
    handles.use(pipeline, XybridPipeline::class.java) { p ->
      mapOf("name" to p.name(), "stageNames" to p.stageNames(), "stageCount" to p.stageCount().toDouble())
    }
  }

  override fun runPipeline(pipeline: String, envelope: ReadableMap, options: ReadableMap?, promise: Promise) {
    val rawEnvelope = envelope.toHashMap()
    val rawOptions = options?.toHashMap()
    background(promise) {
      val input = XybridCodec.decodeEnvelope(rawEnvelope)
      val request = XybridCodec.decodeRunRequest(rawOptions)
      handles.use(pipeline, XybridPipeline::class.java) { p ->
        XybridCodec.encodePipelineResult(p.run(input, request.options))
      }
    }
  }

  // -- Live ASR sessions ------------------------------------------------------

  /** A live-ASR session plus a buffer of its partial transcripts for pull reads. */
  private class SessionEntry(val session: XybridStreamingSession) {
    // Partials are cumulative — each supersedes the last — so a reader that
    // falls behind only loses stale text.
    val partials = Channel<XybridPartialResult>(capacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST)
  }

  override fun openStreamingSession(model: String, config: ReadableMap?, promise: Promise) {
    val rawConfig = config?.toHashMap()
    // Opening warms the weights, so it runs off the calling thread.
    background(promise) {
      val streamingConfig = XybridCodec.decodeStreamingConfig(rawConfig)
      val session = handles.use(model, XybridModel::class.java) { XybridStreamingSession.forModel(it, streamingConfig) }
      val entry = SessionEntry(session)
      val handle = handles.insert(entry, "session", onDispose = session::cancel, onClose = session::close)
      // The collector holds a lease for its whole life, so the session is
      // only closed after it has unsubscribed; cancelling the session (on
      // dispose) is what ends the partial stream.
      val lease = handles.lease(handle, SessionEntry::class.java)
      scope.launch(start = CoroutineStart.UNDISPATCHED) {
        try {
          session.partials().collect { entry.partials.send(it) }
        } finally {
          entry.partials.close()
          lease.close()
        }
      }
      handle
    }
  }

  override fun sessionFeed(session: String, samplesBase64: String, promise: Promise) = background(promise) {
    val samples = XybridCodec.float32Samples(samplesBase64)
    // Blocks only when the worker's queue is full (back-pressure).
    handles.use(session, SessionEntry::class.java) { it.session.feed(samples) }
  }

  override fun sessionNextPartial(session: String, promise: Promise) = background(promise) {
    val lease = handles.leaseOrNull(session, SessionEntry::class.java) ?: return@background null
    lease.use { it.value.partials.receiveCatching().getOrNull()?.let(XybridCodec::encodePartial) }
  }

  override fun sessionFlush(session: String, promise: Promise) =
    background(promise) { handles.use(session, SessionEntry::class.java) { it.session.flush() } }

  override fun sessionReset(session: String, promise: Promise) =
    background(promise) { handles.use(session, SessionEntry::class.java) { it.session.reset() } }

  override fun sessionCancel(session: String, promise: Promise) =
    now(promise) { handles.use(session, SessionEntry::class.java) { it.session.cancel() } }

  override fun sessionIsRunning(session: String, promise: Promise) =
    now(promise) { handles.use(session, SessionEntry::class.java) { it.session.isRunning() } }

  companion object {
    const val NAME = NativeXybridSpec.NAME
  }
}

/** Blank means absent for configuration strings. */
private fun String?.nonBlank(): String? = this?.trim()?.takeIf { it.isNotEmpty() }

/**
 * Process-wide SDK setup. The Rust SDK's configuration is process-global
 * (first-set-wins) and outlives this module: a JS reload builds a new module
 * instance but not a new process. So what has been applied is tracked
 * process-wide too, which is what keeps a reload from starting a second
 * telemetry exporter.
 */
private object SdkSetup {
  data class Runtime(val apiKey: String?, val gatewayUrl: String?, val ingestUrl: String?) {
    val isEmpty get() = apiKey == null && gatewayUrl == null && ingestUrl == null
  }

  private var appliedCacheDir: String? = null
  private var appliedRuntime: Runtime? = null

  /**
   * Register the binding, device observers and the cache directory, once.
   * Every bridged call runs this first, so local inference needs no
   * `initialize()` at all.
   */
  @Synchronized
  fun ensureBase(context: Context, requested: String? = null) {
    val applied = appliedCacheDir
    if (applied != null) {
      if (requested != null && requested != applied) {
        throw BridgeException.Config(
          "the model cache is already at $applied; cacheDir only applies to the first Xybrid call",
        )
      }
      return
    }
    // First-set-wins in the SDK: claim the binding and the cache directory
    // before Xybrid.init() would register "kotlin" and <filesDir>/xybrid/models.
    // Its own configureRuntime(null, null, null) is a no-op; what it adds is
    // the battery and thermal observers.
    ai.xybrid.setBinding("react-native")
    val directory = requested ?: File(context.filesDir, "xybrid/models").absolutePath
    File(directory).mkdirs()
    ai.xybrid.initSdkCacheDir(directory)
    Xybrid.init(context)
    appliedCacheDir = directory
  }

  /** Apply the API key and URL overrides — once per process. */
  @Synchronized
  fun configure(runtime: Runtime) {
    if (runtime.isEmpty) return
    val applied = appliedRuntime
    if (applied != null) {
      if (applied == runtime) return
      throw BridgeException.Config(
        "Xybrid is already initialized with different options; they apply once per app process",
      )
    }
    ai.xybrid.configureRuntime(runtime.apiKey, runtime.gatewayUrl, runtime.ingestUrl)
    appliedRuntime = runtime
  }
}

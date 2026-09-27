// Xybrid SDK - Client
// Main entry point for the Xybrid SDK.

using System;

namespace Xybrid
{
    /// <summary>
    /// Main entry point for the Xybrid SDK.
    /// </summary>
    /// <remarks>
    /// Call <see cref="Initialize"/> once at startup before using any other SDK features.
    /// </remarks>
    public static class XybridClient
    {
        private static bool _initialized;
        private static bool _telemetryInitialized;
        private static readonly object _lock = new object();

        /// <summary>
        /// Gets whether the SDK has been initialized.
        /// </summary>
        public static bool IsInitialized
        {
            get
            {
                lock (_lock)
                {
                    return _initialized;
                }
            }
        }

        /// <summary>
        /// Gets the SDK version string.
        /// </summary>
        public static string Version => XybridBolt.XybridBolt.Version();

        /// <summary>
        /// Releases every idle loaded model's memory; returns how many were released.
        /// </summary>
        /// <remarks>
        /// Wire this to the platform's low-memory signal, e.g. from
        /// <c>Application.lowMemory</c>. Models with a run in flight are skipped,
        /// and a released model reloads itself the next time it is used &#x2014; there
        /// is no new error to handle and nothing to reload by hand.
        /// </remarks>
        public static uint ReleaseMemory() => XybridBolt.XybridBolt.ReleaseMemory();

        /// <summary>
        /// Enables or disables automatic model release for subsequent loads.
        /// </summary>
        /// <remarks>
        /// When enabled, loading a model while the device reports memory pressure
        /// first releases least-recently-used idle models. Off by default;
        /// <see cref="ReleaseMemory"/> works either way.
        /// </remarks>
        public static void SetAutoRelease(bool enabled) =>
            XybridBolt.XybridBolt.SetAutoRelease(enabled);

        /// <summary>
        /// Gets whether automatic model release is enabled process-wide.
        /// </summary>
        public static bool IsAutoReleaseEnabled => XybridBolt.XybridBolt.IsAutoReleaseEnabled();

        /// <summary>
        /// Enables or disables speculative cloud serving for subsequent loads.
        /// </summary>
        /// <remarks>
        /// This is the process-wide <em>default</em>: it applies to registry loads
        /// that do not opt in per-load. <see cref="ModelLoader.FromRegistrySpeculative"/>
        /// opts in explicitly and is unaffected by this toggle. Off by default.
        /// Either way, speculation also needs a resolvable API key and a model that
        /// is not already cached &#x2014; <see cref="ModelLoader.WillSpeculate"/>
        /// reports the combined answer for a specific loader.
        /// </remarks>
        public static void SetSpeculativeCloud(bool enabled) =>
            XybridBolt.XybridBolt.SetSpeculativeCloud(enabled);

        /// <summary>
        /// Gets whether the global speculative-cloud default is on.
        /// </summary>
        public static bool IsSpeculativeCloudEnabled =>
            XybridBolt.XybridBolt.IsSpeculativeCloudEnabled();

        /// <summary>
        /// Gets whether a Xybrid gateway API key is resolvable, from either
        /// <see cref="Initialize"/> or the environment.
        /// </summary>
        /// <remarks>
        /// Inference runs on-device without one; this reports whether the optional
        /// platform features (cloud routing, telemetry) can engage.
        /// </remarks>
        public static bool HasApiKey => XybridBolt.XybridBolt.HasApiKey();

        /// <summary>
        /// Sets the API key for a specific cloud provider.
        /// </summary>
        /// <remarks>
        /// Separate from <see cref="Initialize"/>'s <c>apiKey</c>, which sets the
        /// Xybrid platform key. Use this when routing to a provider the gateway
        /// forwards to.
        /// </remarks>
        public static void SetProviderApiKey(string provider, string apiKey) =>
            XybridBolt.XybridBolt.SetProviderApiKey(provider, apiKey);

        /// <summary>
        /// Converts a JSON Schema into a GBNF grammar for
        /// <see cref="GenerationConfig.SetGrammar"/>.
        /// </summary>
        /// <remarks>
        /// C# has no top-level functions, so this lives here rather than being a
        /// free function the way Swift, Kotlin and Dart expose it.
        /// Grammar is llama.cpp-only &#x2014; other backends ignore it.
        /// </remarks>
        /// <param name="schemaJson">The JSON Schema, as a JSON string.</param>
        /// <exception cref="XybridException">
        /// Thrown on invalid JSON, or schema constructs outside the supported subset.
        /// </exception>
        public static string JsonSchemaToGbnf(string schemaJson)
        {
            try
            {
                return XybridBolt.XybridBolt.JsonSchemaToGbnf(schemaJson);
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        /// <summary>
        /// Initializes the Xybrid SDK.
        /// </summary>
        /// <param name="apiKey">
        /// Optional Xybrid API key. When provided, the platform telemetry exporter
        /// starts automatically and your inference runs show up on the dashboard.
        /// Omit it to run anonymously — inference still runs fully on-device, and
        /// the first inference logs a one-shot hint pointing at the dashboard
        /// (suppress with the <c>XYBRID_QUIET=1</c> environment variable). Get a
        /// free key at https://dashboard.xybrid.dev.
        /// </param>
        /// <param name="ingestUrl">
        /// Optional override for the telemetry ingest URL (for a self-hosted
        /// dashboard). Ignored when <paramref name="apiKey"/> is null or blank.
        /// </param>
        /// <param name="gatewayUrl">
        /// Optional override for the LLM gateway URL. Declared last so existing
        /// positional <c>Initialize(apiKey, ingestUrl)</c> call sites keep compiling;
        /// Swift and Kotlin order it before <paramref name="ingestUrl"/>.
        /// </param>
        /// <remarks>
        /// This method should be called once at application startup, before using
        /// any other SDK features. It is safe to call multiple times - subsequent
        /// calls are no-ops, so configuration is applied on the first call only.
        /// </remarks>
        /// <exception cref="XybridException">Thrown if initialization fails.</exception>
        public static void Initialize(
            string apiKey = null,
            string ingestUrl = null,
            string gatewayUrl = null)
        {
            lock (_lock)
            {
                if (_initialized)
                {
                    return;
                }

                // Runtime init runs on bolt: set the binding tag (used for
                // telemetry attribution). The pre-bolt xybrid_init() was a no-op.
                // Telemetry now runs entirely through bolt (A2.2), so there is no
                // longer a second C-ABI binding state to keep in sync.
                XybridBolt.XybridBolt.SetBinding("unity");

                // One call configures the whole runtime, exactly as the Swift
                // `Xybrid.initialize(apiKey:gatewayUrl:ingestUrl:)` and Kotlin
                // `Xybrid.init(context, apiKey, …)` wrappers do. Unity used to
                // build a TelemetryConfig here instead, which started the
                // exporter but never registered the key with the runtime — so
                // `HasApiKey` stayed false and cloud routing (including
                // ModelLoader.FromRegistrySpeculative) silently fell back to the
                // local path unless XYBRID_API_KEY happened to be set in the
                // environment. ConfigureRuntime gates both, from one key.
                //
                // A non-blank apiKey makes ConfigureRuntime start the telemetry
                // exporter, so do NOT also call InitializeTelemetry here: the
                // two paths are mutually exclusive and the second would throw.
                XybridBolt.XybridBolt.ConfigureRuntime(apiKey, gatewayUrl, ingestUrl);

                _initialized = true;

                // Kept inside the lock so a concurrent caller that observes
                // _initialized == true (and returns) is guaranteed the exporter
                // is already running. Recording it here keeps FlushTelemetry and
                // ShutdownTelemetry working after the apiKey-only init path, and
                // makes a later InitializeTelemetry(config) fail loudly rather
                // than double-starting the exporter.
                if (!string.IsNullOrWhiteSpace(apiKey))
                {
                    _telemetryInitialized = true;
                }
            }
        }

        /// <summary>
        /// Ensures the SDK is initialized, throwing if not.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown if SDK is not initialized.</exception>
        internal static void EnsureInitialized()
        {
            if (!IsInitialized)
            {
                throw new InvalidOperationException(
                    "Xybrid SDK is not initialized. Call XybridClient.Initialize() first.");
            }
        }

        /// <summary>Gets aggregate storage usage across all managed model-cache areas.</summary>
        public static XybridBolt.XybridCacheStatus ModelCacheStatus()
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheStatus());
        }

        /// <summary>
        /// Lists physical entries across registry, extraction, and Hugging Face caches.
        /// A model can appear more than once when several managed copies exist.
        /// </summary>
        public static XybridBolt.XybridCacheEntry[] ModelCacheEntries()
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheEntries());
        }

        /// <summary>Returns whether a model occupies any managed cache entry.</summary>
        public static bool HasCachedModelData(string modelId)
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheIsModelCached(modelId));
        }

        /// <summary>
        /// Returns a preferred local path for a model, or null when absent.
        /// Presence does not necessarily mean the model is extracted and ready.
        /// </summary>
        public static string CachedModelPath(string modelId)
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheModelPath(modelId));
        }

        /// <summary>Lists model IDs extracted, validated, and ready to run offline.</summary>
        public static string[] ExtractedModelIds()
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheListExtractedModelIds());
        }

        /// <summary>
        /// Removes every managed cache entry for one model. Do not call this while
        /// the same model is loading.
        /// </summary>
        public static uint RemoveCachedModel(string modelId)
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheRemoveModel(modelId));
        }

        /// <summary>
        /// Clears all managed model-cache storage. Do not call this while any model
        /// is loading.
        /// </summary>
        public static uint ClearModelCache()
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheClear());
        }

        private static T CacheCall<T>(Func<T> call)
        {
            EnsureInitialized();
            try
            {
                return call();
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        /// <summary>
        /// Convenience method to load a model from the registry.
        /// </summary>
        /// <param name="modelId">The model ID (e.g., "kokoro-82m").</param>
        /// <returns>A loaded model ready for inference.</returns>
        /// <exception cref="XybridException">Thrown if loading fails.</exception>
        /// <remarks>
        /// This is equivalent to:
        /// <code>
        /// using (var loader = ModelLoader.FromRegistry(modelId))
        /// {
        ///     return loader.Load();
        /// }
        /// </code>
        /// </remarks>
        public static Model LoadModel(string modelId)
        {
            using (var loader = ModelLoader.FromRegistry(modelId))
            {
                return loader.Load();
            }
        }

        /// <summary>
        /// Convenience method to load a model from a local bundle.
        /// </summary>
        /// <param name="path">Path to the model bundle.</param>
        /// <returns>A loaded model ready for inference.</returns>
        /// <exception cref="XybridException">Thrown if loading fails.</exception>
        public static Model LoadModelFromBundle(string path)
        {
            using (var loader = ModelLoader.FromBundle(path))
            {
                return loader.Load();
            }
        }

        /// <summary>
        /// Convenience method to load a model from a raw GGUF file
        /// (auto-generates metadata from the GGUF header).
        /// </summary>
        /// <remarks>
        /// On load, metadata is generated from the GGUF header and written as a
        /// <c>model_metadata.json</c> sidecar next to the file if one isn't already
        /// present, then the containing directory is loaded.
        /// </remarks>
        /// <param name="filePath">Path to the GGUF model file.</param>
        /// <returns>A loaded model ready for inference.</returns>
        /// <exception cref="XybridException">Thrown if loading fails.</exception>
        public static Model LoadModelFromFile(string filePath)
        {
            using (var loader = ModelLoader.FromModelFile(filePath))
            {
                return loader.Load();
            }
        }

        /// <summary>
        /// Initializes the Xybrid telemetry sender from a prepared configuration.
        /// </summary>
        /// <param name="config">
        /// The telemetry configuration. Ownership of the underlying native handle is
        /// transferred: on both success and failure, <paramref name="config"/> is
        /// detached and must not be reused. Disposing it afterwards is a safe no-op.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="config"/> is null.</exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown if the SDK has not been initialized (call <see cref="Initialize"/> first),
        /// or if telemetry has already been initialized without an intervening
        /// <see cref="ShutdownTelemetry"/>.
        /// </exception>
        /// <exception cref="XybridException">Thrown if native telemetry initialization fails.</exception>
        /// <remarks>
        /// Advanced entry point. For the common case, pass an <c>apiKey</c> to
        /// <see cref="Initialize(string, string)"/> instead — that starts the
        /// exporter as part of SDK init. Use this overload only when you need the
        /// extra knobs on <see cref="TelemetryConfig"/> (batch size, flush
        /// interval, device label/attributes); both paths share the same
        /// process-wide once-guard.
        /// Thread-safe: serialized via the SDK's initialization lock. Call
        /// <see cref="ShutdownTelemetry"/> before re-initializing.
        /// </remarks>
        public static void InitializeTelemetry(TelemetryConfig config)
        {
            if (config == null)
            {
                throw new ArgumentNullException(nameof(config));
            }

            EnsureInitialized();

            lock (_lock)
            {
                if (_telemetryInitialized)
                {
                    throw new InvalidOperationException(
                        "Xybrid telemetry is already initialized. Call XybridClient.ShutdownTelemetry() before re-initializing.");
                }

                // Ownership transfers here: DetachHandle neutralizes `config` so a
                // later Dispose() is a no-op. Init() consumes the config's inner
                // state; we dispose the (now-empty) bolt handle on every path.
                XybridBolt.XybridTelemetryConfig bolt = config.DetachHandle();
                try
                {
                    bolt.Init();
                }
                catch (Exception ex) when (
                    ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
                {
                    throw BoltErrors.Translate(ex);
                }
                finally
                {
                    bolt.Dispose();
                }

                _telemetryInitialized = true;
            }
        }

        /// <summary>
        /// Flushes any pending telemetry events to the collector.
        /// </summary>
        /// <remarks>
        /// Thread-safe. No-op if telemetry has never been initialized or has been
        /// shut down. Safe to call from lifecycle hooks such as
        /// <c>OnApplicationPause(true)</c>.
        /// </remarks>
        public static void FlushTelemetry()
        {
            lock (_lock)
            {
                if (!_telemetryInitialized)
                {
                    return;
                }

                XybridBolt.XybridBolt.TelemetryFlush();
            }
        }

        /// <summary>
        /// Shuts down the telemetry sender, releasing its background worker.
        /// </summary>
        /// <remarks>
        /// Thread-safe and idempotent: the first call stops the sender, subsequent
        /// calls are no-ops. Fire-and-forget semantics &#x2014; this method does not
        /// block on a final flush. Call <see cref="FlushTelemetry"/> first if you
        /// need pending events delivered before shutdown.
        /// </remarks>
        public static void ShutdownTelemetry()
        {
            lock (_lock)
            {
                if (!_telemetryInitialized)
                {
                    return;
                }

                _telemetryInitialized = false;
                XybridBolt.XybridBolt.TelemetryShutdown();
            }
        }
    }
}

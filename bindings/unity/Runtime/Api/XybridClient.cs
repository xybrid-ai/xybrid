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
        /// <remarks>
        /// This method should be called once at application startup, before using
        /// any other SDK features. It is safe to call multiple times - subsequent
        /// calls are no-ops, so configuration is applied on the first call only.
        /// </remarks>
        /// <exception cref="XybridException">Thrown if initialization fails.</exception>
        public static void Initialize(string apiKey = null, string ingestUrl = null)
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

                _initialized = true;

                // Fold telemetry into init: a non-blank API key starts the
                // exporter, mirroring the Swift initialize(apiKey:) / Kotlin
                // init(apiKey =) surfaces. The standalone
                // InitializeTelemetry(TelemetryConfig) path remains available for
                // advanced configuration (batch size, device attributes, flush
                // interval). TelemetryConfig defaults the endpoint to the
                // production ingest URL, so apiKey alone is enough.
                //
                // Kept inside the lock so a concurrent caller that observes
                // _initialized == true (and returns) is guaranteed the exporter
                // is already running — and so the _telemetryInitialized read
                // here has the same visibility as InitializeTelemetry's write.
                // C# locks are reentrant, so InitializeTelemetry re-taking _lock
                // is safe.
                if (!string.IsNullOrWhiteSpace(apiKey) && !_telemetryInitialized)
                {
                    var config = new TelemetryConfig(apiKey);
                    if (!string.IsNullOrWhiteSpace(ingestUrl))
                    {
                        config.WithEndpoint(ingestUrl);
                    }

                    InitializeTelemetry(config);
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

        /// <summary>Throws until persistent retention is supported. Use per-model eviction.</summary>
        public static uint CleanExpiredModelCache()
        {
            return CacheCall(() => XybridBolt.XybridBolt.CacheCleanExpired());
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

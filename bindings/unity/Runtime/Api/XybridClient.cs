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
        private static readonly TimeSpan DefaultObservabilityTimeout = TimeSpan.FromSeconds(5);

        private static bool _initialized;
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
        /// checks both for a specific loader &#x2014; and the load asks the
        /// registry first, so only chat models are served this way.
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
        /// platform features (cloud routing) can engage. A key does not enable
        /// observability export; see <see cref="InitializeObservability"/>.
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
        /// Optional Xybrid API key for cloud routing and the LLM gateway. It does
        /// not enable observability export &#x2014; build an
        /// <see cref="ObservabilityConfig"/> and call
        /// <see cref="InitializeObservability"/> for that. Omit it to run
        /// anonymously; inference still runs fully on-device. Get a key at
        /// https://console.xybrid.dev.
        /// </param>
        /// <param name="gatewayUrl">Optional override for the LLM gateway URL.</param>
        /// <remarks>
        /// This method should be called once at application startup, before using
        /// any other SDK features. It is safe to call multiple times - subsequent
        /// calls are no-ops, so configuration is applied on the first call only.
        /// </remarks>
        /// <exception cref="XybridException">Thrown if initialization fails.</exception>
        public static void Initialize(string apiKey = null, string gatewayUrl = null)
        {
            lock (_lock)
            {
                if (_initialized)
                {
                    return;
                }

                // Runtime init runs on bolt: set the binding tag (used for
                // registry and observability attribution).
                XybridBolt.XybridBolt.SetBinding("unity");

                // One call registers the key and gateway with the runtime, exactly
                // as the Swift `Xybrid.initialize(apiKey:gatewayUrl:)` and Kotlin
                // `Xybrid.init(context, apiKey, gatewayUrl)` wrappers do, so
                // `HasApiKey` and cloud routing (including
                // ModelLoader.FromRegistrySpeculative) see it.
                XybridBolt.XybridBolt.ConfigureRuntime(apiKey, gatewayUrl);

                _initialized = true;
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
            XybridBolt.AndroidCacheFolder.ApplyDefault();
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
        /// Starts OpenTelemetry export with a prepared configuration, replacing
        /// any previous one.
        /// </summary>
        /// <param name="config">
        /// The observability configuration. Ownership of the underlying native
        /// handle is transferred: on both success and failure,
        /// <paramref name="config"/> is detached and must not be reused.
        /// Disposing it afterwards is a safe no-op.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="config"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">
        /// Thrown if <paramref name="config"/> was already disposed or handed off.
        /// </exception>
        /// <exception cref="XybridException">Thrown if the configuration is rejected.</exception>
        /// <remarks>
        /// Export is off until this is called; an API key passed to
        /// <see cref="Initialize"/> does not enable it. It does not depend on
        /// <see cref="Initialize"/> and may run before or after it. Call
        /// <see cref="ShutdownObservability"/> before the app exits so queued
        /// spans are delivered.
        /// </remarks>
        public static void InitializeObservability(ObservabilityConfig config)
        {
            if (config == null)
            {
                throw new ArgumentNullException(nameof(config));
            }

            // Ownership transfers here: DetachHandle neutralizes `config` so a
            // later Dispose() is a no-op. Init() consumes the config's inner
            // state; we dispose the (now-empty) bolt handle on every path.
            XybridBolt.XybridObservabilityConfig bolt = config.DetachHandle();
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
        }

        /// <summary>
        /// Gets whether observability export is running.
        /// </summary>
        public static bool IsObservabilityEnabled => ObservabilityStats().Enabled;

        /// <summary>
        /// Returns the export pipeline's delivery counters (all zero when
        /// observability is off).
        /// </summary>
        public static XybridBolt.XybridObservabilityStats ObservabilityStats() =>
            XybridBolt.XybridBolt.ObservabilityStats();

        /// <summary>
        /// Exports queued spans, metrics, and logs, waiting at most
        /// <paramref name="timeout"/>.
        /// </summary>
        /// <param name="timeout">Upper bound on the wait. Defaults to 5 seconds.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown if <paramref name="timeout"/> is negative or exceeds <see cref="uint.MaxValue"/> milliseconds.
        /// </exception>
        /// <exception cref="XybridException">Thrown if the bound is exceeded.</exception>
        /// <remarks>
        /// A no-op when observability is off. Call it from
        /// <c>OnApplicationPause(true)</c>: mobile OSes can kill a backgrounded
        /// app without delivering <c>OnApplicationQuit</c>.
        /// </remarks>
        public static void FlushObservability(TimeSpan? timeout = null)
        {
            uint ms = ObservabilityConfig.ToMilliseconds(
                timeout ?? DefaultObservabilityTimeout, nameof(timeout));
            ObservabilityCall(() => XybridBolt.XybridBolt.ObservabilityFlush(ms));
        }

        /// <summary>
        /// Flushes and stops export, waiting at most <paramref name="timeout"/>.
        /// </summary>
        /// <param name="timeout">Upper bound on the wait. Defaults to 5 seconds.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown if <paramref name="timeout"/> is negative or exceeds <see cref="uint.MaxValue"/> milliseconds.
        /// </exception>
        /// <exception cref="XybridException">Thrown if the bound is exceeded.</exception>
        /// <remarks>
        /// Idempotent, and a no-op when observability is off. Export stays off
        /// until the next <see cref="InitializeObservability"/>.
        /// </remarks>
        public static void ShutdownObservability(TimeSpan? timeout = null)
        {
            uint ms = ObservabilityConfig.ToMilliseconds(
                timeout ?? DefaultObservabilityTimeout, nameof(timeout));
            ObservabilityCall(() => XybridBolt.XybridBolt.ObservabilityShutdown(ms));
        }

        /// <summary>
        /// Records end-user feedback on a run.
        /// </summary>
        /// <param name="target">The run's <see cref="InferenceResult.Observation"/>.</param>
        /// <param name="score">Optional numeric score (for example 1 for thumbs-up, 0 for thumbs-down).</param>
        /// <param name="label">Optional label (for example <c>"helpful"</c>).</param>
        /// <param name="comment">Optional free-text comment.</param>
        /// <returns>
        /// False when no logs signal is configured (the <see cref="ObservabilityConfig.Xybrid"/>
        /// preset sends traces only) or observability is off.
        /// </returns>
        /// <exception cref="XybridException">
        /// Thrown if <paramref name="target"/> is not a valid trace/span id pair.
        /// </exception>
        public static bool RecordFeedback(
            XybridBolt.XybridObservationRef target,
            double? score = null,
            string label = null,
            string comment = null)
        {
            return ObservabilityCall(() =>
                XybridBolt.XybridBolt.ObservabilityRecordFeedback(target, score, label, comment));
        }

        /// <summary>
        /// Records an evaluation result (for example an LLM-as-judge score) on a run.
        /// </summary>
        /// <param name="target">The run's <see cref="InferenceResult.Observation"/>.</param>
        /// <param name="name">Evaluation name (for example <c>"relevance"</c>).</param>
        /// <param name="scoreValue">Optional numeric score.</param>
        /// <param name="scoreLabel">Optional categorical score.</param>
        /// <param name="explanation">Optional explanation of the score.</param>
        /// <returns>
        /// False when no logs signal is configured (the <see cref="ObservabilityConfig.Xybrid"/>
        /// preset sends traces only) or observability is off.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="name"/> is null, empty, or whitespace.
        /// </exception>
        /// <exception cref="XybridException">
        /// Thrown if <paramref name="target"/> is not a valid trace/span id pair.
        /// </exception>
        public static bool RecordEvaluation(
            XybridBolt.XybridObservationRef target,
            string name,
            double? scoreValue = null,
            string scoreLabel = null,
            string explanation = null)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("name must be a non-empty string.", nameof(name));
            }

            return ObservabilityCall(() => XybridBolt.XybridBolt.ObservabilityRecordEvaluation(
                target, name, scoreValue, scoreLabel, explanation));
        }

        private static void ObservabilityCall(Action call)
        {
            ObservabilityCall(() =>
            {
                call();
                return true;
            });
        }

        private static T ObservabilityCall<T>(Func<T> call)
        {
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
    }
}

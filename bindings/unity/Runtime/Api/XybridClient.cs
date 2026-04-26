// Xybrid SDK - Client
// Main entry point for the Xybrid SDK.

using System;
using Xybrid.Native;

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
        public static unsafe string Version
        {
            get
            {
                byte* versionPtr = NativeMethods.xybrid_version();
                return NativeHelpers.FromUtf8Ptr(versionPtr) ?? "unknown";
            }
        }

        /// <summary>
        /// Initializes the Xybrid SDK.
        /// </summary>
        /// <remarks>
        /// This method should be called once at application startup, before using
        /// any other SDK features. It is safe to call multiple times - subsequent
        /// calls are no-ops.
        /// </remarks>
        /// <exception cref="XybridException">Thrown if initialization fails.</exception>
        public static unsafe void Initialize()
        {
            lock (_lock)
            {
                if (_initialized)
                {
                    return;
                }

                byte[] bindingBytes = NativeHelpers.ToUtf8Bytes("unity");
                fixed (byte* bindingPtr = bindingBytes)
                {
                    NativeMethods.xybrid_set_binding(bindingPtr);
                }

                int result = NativeMethods.xybrid_init();
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to initialize Xybrid SDK");
                }

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
        /// Convenience method to load a model from a raw GGUF file.
        /// Auto-generates metadata from the GGUF binary header.
        /// </summary>
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
        /// Thread-safe: serialized via the SDK's initialization lock. Call
        /// <see cref="ShutdownTelemetry"/> before re-initializing.
        /// </remarks>
        public static unsafe void InitializeTelemetry(TelemetryConfig config)
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

                IntPtr raw = config.DetachHandle();
                var handle = (XybridTelemetryConfigHandle*)raw.ToPointer();
                int result = NativeMethods.xybrid_telemetry_init(handle);
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to initialize Xybrid telemetry");
                }

                _telemetryInitialized = true;
            }
        }

        /// <summary>
        /// Flushes any pending telemetry events to the collector.
        /// </summary>
        /// <exception cref="XybridException">Thrown if the native flush fails.</exception>
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

                int result = NativeMethods.xybrid_telemetry_flush();
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to flush Xybrid telemetry");
                }
            }
        }

        /// <summary>
        /// Shuts down the telemetry sender, releasing its background worker.
        /// </summary>
        /// <exception cref="XybridException">Thrown if the native shutdown fails.</exception>
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
                int result = NativeMethods.xybrid_telemetry_shutdown();
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to shut down Xybrid telemetry");
                }
            }
        }
    }
}

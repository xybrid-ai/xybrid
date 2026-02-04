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
        public static void Initialize()
        {
            lock (_lock)
            {
                if (_initialized)
                {
                    return;
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
    }
}

// Xybrid SDK - Model Loader
// Factory for loading models from registry or local bundles.

using System;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// Loads models from the xybrid registry, local bundles, or directories.
    /// </summary>
    /// <remarks>
    /// Use the static factory methods to create a loader, then call <see cref="Load"/>
    /// to get a ready-to-use <see cref="Model"/>.
    /// </remarks>
    public sealed class ModelLoader : IDisposable
    {
        private unsafe XybridModelLoaderHandle* _handle;
        private bool _disposed;

        /// <summary>
        /// Gets whether this loader has been disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        private unsafe ModelLoader(XybridModelLoaderHandle* handle)
        {
            _handle = handle;
        }

        /// <summary>
        /// Creates a model loader that will fetch from the xybrid registry.
        /// </summary>
        /// <param name="modelId">The model ID (e.g., "kokoro-82m", "whisper-tiny").</param>
        /// <returns>A new ModelLoader configured to load from the registry.</returns>
        /// <exception cref="ArgumentNullException">Thrown if modelId is null.</exception>
        /// <exception cref="XybridException">Thrown if loader creation fails.</exception>
        /// <remarks>
        /// The model will be downloaded from the registry if not already cached locally.
        /// </remarks>
        public static unsafe ModelLoader FromRegistry(string modelId)
        {
            if (modelId == null)
            {
                throw new ArgumentNullException(nameof(modelId));
            }

            byte[] modelIdBytes = NativeHelpers.ToUtf8Bytes(modelId);

            fixed (byte* modelIdPtr = modelIdBytes)
            {
                XybridModelLoaderHandle* handle = NativeMethods.xybrid_model_loader_from_registry(modelIdPtr);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError($"Failed to create loader for model '{modelId}'");
                }

                return new ModelLoader(handle);
            }
        }

        /// <summary>
        /// Creates a model loader that will load from a local bundle path.
        /// </summary>
        /// <param name="path">The file path to the model bundle (.xyb file or directory).</param>
        /// <returns>A new ModelLoader configured to load from the local bundle.</returns>
        /// <exception cref="ArgumentNullException">Thrown if path is null.</exception>
        /// <exception cref="XybridException">Thrown if loader creation fails.</exception>
        public static unsafe ModelLoader FromBundle(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path));
            }

            byte[] pathBytes = NativeHelpers.ToUtf8Bytes(path);

            fixed (byte* pathPtr = pathBytes)
            {
                XybridModelLoaderHandle* handle = NativeMethods.xybrid_model_loader_from_bundle(pathPtr);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError($"Failed to create loader for bundle at '{path}'");
                }

                return new ModelLoader(handle);
            }
        }

        /// <summary>
        /// Creates a model loader from a local directory containing model files
        /// and a <c>model_metadata.json</c>.
        /// </summary>
        /// <param name="directoryPath">Path to the directory containing model files and model_metadata.json.</param>
        /// <returns>A new ModelLoader configured to load from the directory.</returns>
        /// <exception cref="ArgumentNullException">Thrown if directoryPath is null.</exception>
        /// <exception cref="XybridException">Thrown if the directory does not exist, or the metadata is missing or invalid.</exception>
        public static unsafe ModelLoader FromDirectory(string directoryPath)
        {
            if (directoryPath == null)
            {
                throw new ArgumentNullException(nameof(directoryPath));
            }

            byte[] pathBytes = NativeHelpers.ToUtf8Bytes(directoryPath);

            fixed (byte* pathPtr = pathBytes)
            {
                XybridModelLoaderHandle* handle = NativeMethods.xybrid_model_loader_from_directory(pathPtr);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError($"Failed to create loader from directory '{directoryPath}'");
                }

                return new ModelLoader(handle);
            }
        }

        /// <summary>
        /// Creates a model loader from a HuggingFace Hub repository.
        /// Downloads model files from HuggingFace and caches them locally.
        /// Model metadata is auto-generated if not present in the repository.
        /// </summary>
        /// <param name="repo">The HuggingFace repository ID (e.g., "xybrid-ai/kokoro-82m").</param>
        /// <returns>A new ModelLoader configured to download from HuggingFace.</returns>
        /// <exception cref="ArgumentNullException">Thrown if repo is null.</exception>
        /// <exception cref="XybridException">Thrown if loader creation fails.</exception>
        /// <remarks>
        /// Requires the <c>huggingface</c> feature flag to be enabled at compile time.
        /// </remarks>
        public static unsafe ModelLoader FromHuggingFace(string repo)
        {
            if (repo == null)
            {
                throw new ArgumentNullException(nameof(repo));
            }

            byte[] repoBytes = NativeHelpers.ToUtf8Bytes(repo);

            fixed (byte* repoPtr = repoBytes)
            {
                XybridModelLoaderHandle* handle = NativeMethods.xybrid_model_loader_from_huggingface(repoPtr);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError($"Failed to create loader for HuggingFace repo '{repo}'");
                }

                return new ModelLoader(handle);
            }
        }

        /// <summary>
        /// Loads the model and prepares it for inference.
        /// </summary>
        /// <returns>A loaded <see cref="Model"/> ready for inference.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this loader is disposed.</exception>
        /// <exception cref="XybridException">Thrown if model loading fails.</exception>
        /// <exception cref="ModelNotFoundException">Thrown if the model cannot be found.</exception>
        /// <remarks>
        /// For registry models, this may download the model if not already cached.
        /// The loader can be disposed after loading - the model is independent.
        /// </remarks>
        public unsafe Model Load()
        {
            ThrowIfDisposed();

            XybridModelHandle* modelHandle = NativeMethods.xybrid_model_loader_load(_handle);
            if (modelHandle == null)
            {
                NativeHelpers.ThrowLastError("Failed to load model");
            }

            return new Model(modelHandle);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(ModelLoader));
            }
        }

        /// <summary>
        /// Releases the native resources used by this loader.
        /// </summary>
        public unsafe void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != null)
                {
                    NativeMethods.xybrid_model_loader_free(_handle);
                    _handle = null;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released.
        /// </summary>
        ~ModelLoader()
        {
            Dispose();
        }
    }
}

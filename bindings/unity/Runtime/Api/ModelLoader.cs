// Xybrid SDK - Model Loader
// Factory for loading models from registry or local bundles.

using System;

namespace Xybrid
{
    /// <summary>
    /// Loads models from the xybrid registry, local bundles, or directories.
    /// </summary>
    /// <remarks>
    /// Use the static factory methods to create a loader, then call <see cref="Load"/>
    /// to get a ready-to-use <see cref="Model"/>. The loader records the source and
    /// defers the actual load (and any download) to <see cref="Load"/>.
    /// </remarks>
    public sealed class ModelLoader : IDisposable
    {
        private enum Source
        {
            Registry,
            Bundle,
            Directory,
            HuggingFace,
        }

        private readonly Source _source;
        private readonly string _value;
        private bool _disposed;

        /// <summary>Gets whether this loader has been disposed.</summary>
        public bool IsDisposed => _disposed;

        private ModelLoader(Source source, string value)
        {
            _source = source;
            _value = value;
        }

        /// <summary>
        /// Creates a model loader that will fetch from the xybrid registry.
        /// </summary>
        /// <param name="modelId">The model ID (e.g., "kokoro-82m", "whisper-tiny").</param>
        /// <exception cref="ArgumentNullException">Thrown if modelId is null.</exception>
        public static ModelLoader FromRegistry(string modelId)
        {
            if (modelId == null)
            {
                throw new ArgumentNullException(nameof(modelId));
            }
            return new ModelLoader(Source.Registry, modelId);
        }

        /// <summary>
        /// Creates a model loader that will load from a local bundle path.
        /// </summary>
        /// <param name="path">The file path to the model bundle (.xyb file or directory).</param>
        /// <exception cref="ArgumentNullException">Thrown if path is null.</exception>
        public static ModelLoader FromBundle(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path));
            }
            return new ModelLoader(Source.Bundle, path);
        }

        /// <summary>
        /// Creates a model loader from a local directory containing model files
        /// and a <c>model_metadata.json</c>.
        /// </summary>
        /// <param name="directoryPath">Path to the directory containing model files and model_metadata.json.</param>
        /// <exception cref="ArgumentNullException">Thrown if directoryPath is null.</exception>
        public static ModelLoader FromDirectory(string directoryPath)
        {
            if (directoryPath == null)
            {
                throw new ArgumentNullException(nameof(directoryPath));
            }
            return new ModelLoader(Source.Directory, directoryPath);
        }

        /// <summary>
        /// Creates a model loader from a raw GGUF model file.
        /// </summary>
        /// <param name="filePath">Path to the GGUF model file.</param>
        /// <exception cref="ArgumentNullException">Thrown if filePath is null.</exception>
        /// <exception cref="NotSupportedException">
        /// GGUF auto-metadata loading is not yet available on the bolt backend.
        /// </exception>
        public static ModelLoader FromModelFile(string filePath)
        {
            if (filePath == null)
            {
                throw new ArgumentNullException(nameof(filePath));
            }
            throw new NotSupportedException(
                "ModelLoader.FromModelFile (GGUF auto-metadata) is not yet available on the " +
                "bolt backend. Use FromDirectory with a directory containing model_metadata.json.");
        }

        /// <summary>
        /// Creates a model loader from a HuggingFace Hub repository.
        /// </summary>
        /// <param name="repo">The HuggingFace repository ID (e.g., "xybrid-ai/kokoro-82m").</param>
        /// <exception cref="ArgumentNullException">Thrown if repo is null.</exception>
        public static ModelLoader FromHuggingFace(string repo)
        {
            if (repo == null)
            {
                throw new ArgumentNullException(nameof(repo));
            }
            return new ModelLoader(Source.HuggingFace, repo);
        }

        /// <summary>
        /// Loads the model and prepares it for inference.
        /// </summary>
        /// <returns>A loaded <see cref="Model"/> ready for inference.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this loader is disposed.</exception>
        /// <exception cref="XybridException">Thrown if model loading fails.</exception>
        /// <exception cref="ModelNotFoundException">Thrown if the model cannot be found.</exception>
        public Model Load()
        {
            ThrowIfDisposed();

            try
            {
                XybridBolt.XybridModel bolt;
                switch (_source)
                {
                    case Source.Registry:
                        bolt = XybridBolt.XybridModel.FromRegistry(_value);
                        break;
                    case Source.Bundle:
                        bolt = XybridBolt.XybridModel.FromBundle(_value);
                        break;
                    case Source.Directory:
                        bolt = XybridBolt.XybridModel.FromDirectory(_value);
                        break;
                    case Source.HuggingFace:
                        bolt = XybridBolt.XybridModel.FromHuggingface(_value);
                        break;
                    default:
                        throw new InvalidOperationException("Unknown model source.");
                }

                return new Model(bolt);
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(ModelLoader));
            }
        }

        /// <summary>
        /// No-op: the loader holds no native resources until <see cref="Load"/>.
        /// Retained so existing <c>using</c> call sites keep compiling.
        /// </summary>
        public void Dispose()
        {
            _disposed = true;
        }
    }
}

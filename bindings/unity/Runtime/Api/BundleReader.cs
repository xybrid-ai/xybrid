// Xybrid SDK - BundleReader
// Reads .xyb bundle files (tar + zstd) via the native library.

using System;
using System.Runtime.InteropServices;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// Reads and inspects .xyb model bundle files.
    /// </summary>
    /// <remarks>
    /// <para>
    /// .xyb files are tar + zstd compressed archives containing model files and metadata.
    /// This class uses the native Rust library to decompress and parse them, avoiding
    /// the need for managed zstd support.
    /// </para>
    /// <para>
    /// This class is useful for:
    /// <list type="bullet">
    /// <item>Inspecting bundle contents without extracting (Editor tooling, ScriptedImporters)</item>
    /// <item>Extracting bundles to StreamingAssets for runtime loading</item>
    /// <item>Reading model metadata for custom asset workflows</item>
    /// </list>
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// using (var bundle = BundleReader.Open("/path/to/model.xyb"))
    /// {
    ///     Debug.Log($"Model: {bundle.ModelId} v{bundle.Version}");
    ///     Debug.Log($"Files: {bundle.FileCount}");
    ///
    ///     string metadata = bundle.GetMetadataJson();
    ///     if (metadata != null)
    ///         Debug.Log($"Metadata: {metadata}");
    ///
    ///     bundle.ExtractTo(Application.streamingAssetsPath + "/Models/" + bundle.ModelId);
    /// }
    /// </code>
    /// </example>
    public sealed class BundleReader : IDisposable
    {
        private unsafe XybridBundleHandle* _handle;
        private bool _disposed;

        // Cached manifest fields
        private readonly string _modelId;
        private readonly string _version;
        private readonly string _target;
        private readonly string _hash;
        private readonly bool _hasMetadata;
        private readonly uint _fileCount;

        /// <summary>
        /// Gets the model identifier from the bundle manifest.
        /// </summary>
        public string ModelId => _modelId;

        /// <summary>
        /// Gets the version string from the bundle manifest.
        /// </summary>
        public string Version => _version;

        /// <summary>
        /// Gets the target platform from the bundle manifest (e.g., "universal", "macos-arm64").
        /// </summary>
        public string Target => _target;

        /// <summary>
        /// Gets the SHA-256 hash from the bundle manifest.
        /// </summary>
        public string Hash => _hash;

        /// <summary>
        /// Gets whether the bundle contains a model_metadata.json file.
        /// </summary>
        public bool HasMetadata => _hasMetadata;

        /// <summary>
        /// Gets the number of files in the bundle (excluding manifest.json).
        /// </summary>
        public uint FileCount => _fileCount;

        /// <summary>
        /// Gets whether this reader has been disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        private unsafe BundleReader(XybridBundleHandle* handle)
        {
            _handle = handle;

            // Cache all manifest fields so they survive handle disposal
            byte* ptr;

            ptr = NativeMethods.xybrid_bundle_model_id(handle);
            _modelId = NativeHelpers.FromUtf8Ptr(ptr) ?? "unknown";

            ptr = NativeMethods.xybrid_bundle_version(handle);
            _version = NativeHelpers.FromUtf8Ptr(ptr) ?? "0.0.0";

            ptr = NativeMethods.xybrid_bundle_target(handle);
            _target = NativeHelpers.FromUtf8Ptr(ptr) ?? "unknown";

            ptr = NativeMethods.xybrid_bundle_hash(handle);
            _hash = NativeHelpers.FromUtf8Ptr(ptr) ?? "";

            _hasMetadata = NativeMethods.xybrid_bundle_has_metadata(handle) != 0;
            _fileCount = NativeMethods.xybrid_bundle_file_count(handle);
        }

        /// <summary>
        /// Opens a .xyb bundle file for reading.
        /// </summary>
        /// <param name="path">Absolute path to the .xyb file.</param>
        /// <returns>A BundleReader for inspecting the bundle.</returns>
        /// <exception cref="ArgumentNullException">Thrown if path is null.</exception>
        /// <exception cref="XybridException">Thrown if the bundle cannot be opened.</exception>
        public static unsafe BundleReader Open(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            fixed (byte* pathBytes = NativeHelpers.ToUtf8Bytes(path))
            {
                XybridBundleHandle* handle = NativeMethods.xybrid_bundle_open(pathBytes);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to open bundle");
                }

                return new BundleReader(handle);
            }
        }

        /// <summary>
        /// Gets the model_metadata.json content from the bundle.
        /// </summary>
        /// <returns>The metadata JSON string, or null if the bundle has no model_metadata.json.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this reader is disposed.</exception>
        /// <exception cref="XybridException">Thrown if reading metadata fails.</exception>
        public unsafe string GetMetadataJson()
        {
            ThrowIfDisposed();

            byte* ptr = NativeMethods.xybrid_bundle_metadata_json(_handle);
            if (ptr == null)
            {
                // Distinguish "not present" from error
                string error = NativeHelpers.GetLastError();
                if (error != null && error != "Unknown error")
                {
                    throw new XybridException($"Failed to read metadata: {error}");
                }
                return null;
            }

            string json = NativeHelpers.FromUtf8Ptr(ptr);
            NativeMethods.xybrid_free_string(ptr);
            return json;
        }

        /// <summary>
        /// Gets the full manifest as a JSON string.
        /// </summary>
        /// <returns>The manifest JSON string.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this reader is disposed.</exception>
        /// <exception cref="XybridException">Thrown if reading manifest fails.</exception>
        public unsafe string GetManifestJson()
        {
            ThrowIfDisposed();

            byte* ptr = NativeMethods.xybrid_bundle_manifest_json(_handle);
            if (ptr == null)
            {
                NativeHelpers.ThrowLastError("Failed to read manifest");
            }

            string json = NativeHelpers.FromUtf8Ptr(ptr);
            NativeMethods.xybrid_free_string(ptr);
            return json;
        }

        /// <summary>
        /// Gets the filename at the specified index in the bundle's file list.
        /// </summary>
        /// <param name="index">Zero-based index into the file list.</param>
        /// <returns>The filename, or null if index is out of bounds.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this reader is disposed.</exception>
        public unsafe string GetFileName(uint index)
        {
            ThrowIfDisposed();

            byte* ptr = NativeMethods.xybrid_bundle_file_name(_handle, index);
            return NativeHelpers.FromUtf8Ptr(ptr);
        }

        /// <summary>
        /// Gets all filenames in the bundle.
        /// </summary>
        /// <returns>Array of filenames.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this reader is disposed.</exception>
        public string[] GetFileNames()
        {
            ThrowIfDisposed();

            var names = new string[_fileCount];
            for (uint i = 0; i < _fileCount; i++)
            {
                names[i] = GetFileName(i);
            }
            return names;
        }

        /// <summary>
        /// Extracts all bundle contents to the specified directory.
        /// </summary>
        /// <param name="outputDir">Directory to extract files into. Created if it doesn't exist.</param>
        /// <exception cref="ArgumentNullException">Thrown if outputDir is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this reader is disposed.</exception>
        /// <exception cref="XybridException">Thrown if extraction fails.</exception>
        public unsafe void ExtractTo(string outputDir)
        {
            ThrowIfDisposed();

            if (outputDir == null)
                throw new ArgumentNullException(nameof(outputDir));

            fixed (byte* dirBytes = NativeHelpers.ToUtf8Bytes(outputDir))
            {
                int result = NativeMethods.xybrid_bundle_extract(_handle, dirBytes);
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to extract bundle");
                }
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(BundleReader));
            }
        }

        /// <summary>
        /// Releases the native resources used by this reader.
        /// </summary>
        public unsafe void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != null)
                {
                    NativeMethods.xybrid_bundle_free(_handle);
                    _handle = null;
                }
                _disposed = true;
            }
        }

        ~BundleReader()
        {
            Dispose();
        }

        /// <summary>
        /// Returns a string representation of the bundle.
        /// </summary>
        public override string ToString()
        {
            return $"Bundle({_modelId} v{_version}, {_fileCount} files)";
        }
    }
}

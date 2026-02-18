// Xybrid SDK - GenerationConfig
// LLM generation parameters for controlling inference behavior.

using System;
using System.Runtime.InteropServices;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// LLM generation parameters for controlling inference behavior.
    /// </summary>
    /// <remarks>
    /// Use this class to configure generation parameters like temperature, top-p,
    /// and max tokens. All fields start unset — the model's defaults are used for
    /// any field you don't explicitly set.
    ///
    /// This class must be disposed when no longer needed to release native resources.
    /// </remarks>
    /// <example>
    /// <code>
    /// // Use a preset
    /// using var config = GenerationConfig.Greedy();
    ///
    /// // Or customize
    /// using var config = new GenerationConfig();
    /// config.SetMaxTokens(512);
    /// config.SetTemperature(0.3f);
    ///
    /// using var result = model.Run(envelope, config);
    /// </code>
    /// </example>
    public sealed class GenerationConfig : IDisposable
    {
        private unsafe XybridGenerationConfigHandle* _handle;
        private bool _disposed;

        /// <summary>
        /// Gets whether this config has been disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        /// <summary>
        /// Gets the native handle for passing to native methods.
        /// </summary>
        internal unsafe XybridGenerationConfigHandle* Handle
        {
            get
            {
                ThrowIfDisposed();
                return _handle;
            }
        }

        /// <summary>
        /// Creates a new generation config with all fields unset (model defaults).
        /// </summary>
        /// <remarks>
        /// Call setter methods to override specific fields.
        /// </remarks>
        public unsafe GenerationConfig()
        {
            _handle = NativeMethods.xybrid_generation_config_new();
        }

        private unsafe GenerationConfig(XybridGenerationConfigHandle* handle)
        {
            _handle = handle;
        }

        /// <summary>
        /// Creates a greedy decoding config (deterministic, temperature=0).
        /// </summary>
        /// <remarks>
        /// Produces the same output every time for the same input.
        /// </remarks>
        public static unsafe GenerationConfig Greedy()
        {
            return new GenerationConfig(NativeMethods.xybrid_generation_config_greedy());
        }

        /// <summary>
        /// Creates a creative generation config (higher temperature).
        /// </summary>
        /// <remarks>
        /// Produces more varied and creative output.
        /// </remarks>
        public static unsafe GenerationConfig Creative()
        {
            return new GenerationConfig(NativeMethods.xybrid_generation_config_creative());
        }

        /// <summary>
        /// Set the maximum number of tokens to generate.
        /// </summary>
        /// <param name="maxTokens">Maximum tokens (e.g., 512, 2048).</param>
        public unsafe void SetMaxTokens(int maxTokens)
        {
            ThrowIfDisposed();
            NativeMethods.xybrid_generation_config_set_max_tokens(_handle, (uint)maxTokens);
        }

        /// <summary>
        /// Set the sampling temperature.
        /// </summary>
        /// <param name="temperature">
        /// Temperature value. 0.0 = deterministic, higher = more random.
        /// Typical range: 0.0 to 2.0.
        /// </param>
        public unsafe void SetTemperature(float temperature)
        {
            ThrowIfDisposed();
            NativeMethods.xybrid_generation_config_set_temperature(_handle, temperature);
        }

        /// <summary>
        /// Set the top-p (nucleus) sampling threshold.
        /// </summary>
        /// <param name="topP">Top-p value (0.0 to 1.0). Default: 0.9.</param>
        public unsafe void SetTopP(float topP)
        {
            ThrowIfDisposed();
            NativeMethods.xybrid_generation_config_set_top_p(_handle, topP);
        }

        /// <summary>
        /// Set the min-p sampling threshold.
        /// </summary>
        /// <param name="minP">Min-p value (0.0 to 1.0). Default: 0.05.</param>
        public unsafe void SetMinP(float minP)
        {
            ThrowIfDisposed();
            NativeMethods.xybrid_generation_config_set_min_p(_handle, minP);
        }

        /// <summary>
        /// Set top-k sampling (0 = disabled).
        /// </summary>
        /// <param name="topK">Top-k value. 0 disables top-k filtering. Default: 40.</param>
        public unsafe void SetTopK(int topK)
        {
            ThrowIfDisposed();
            NativeMethods.xybrid_generation_config_set_top_k(_handle, (uint)topK);
        }

        /// <summary>
        /// Set the repetition penalty.
        /// </summary>
        /// <param name="penalty">Penalty value. 1.0 = disabled. Default: 1.1.</param>
        public unsafe void SetRepetitionPenalty(float penalty)
        {
            ThrowIfDisposed();
            NativeMethods.xybrid_generation_config_set_repetition_penalty(_handle, penalty);
        }

        /// <summary>
        /// Add a stop sequence. Can be called multiple times.
        /// </summary>
        /// <param name="stop">The stop sequence string.</param>
        public unsafe void AddStop(string stop)
        {
            ThrowIfDisposed();
            if (stop == null)
                throw new ArgumentNullException(nameof(stop));

            var bytes = System.Text.Encoding.UTF8.GetBytes(stop + "\0");
            fixed (byte* ptr = bytes)
            {
                NativeMethods.xybrid_generation_config_add_stop(_handle, ptr);
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(GenerationConfig));
            }
        }

        /// <summary>
        /// Releases the native resources used by this config.
        /// </summary>
        public unsafe void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != null)
                {
                    NativeMethods.xybrid_generation_config_free(_handle);
                    _handle = null;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released.
        /// </summary>
        ~GenerationConfig()
        {
            Dispose();
        }
    }
}

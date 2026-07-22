// Xybrid SDK - GenerationConfig
// LLM generation parameters for controlling inference behavior.

using System;
using System.Collections.Generic;

namespace Xybrid
{
    /// <summary>
    /// LLM generation parameters for controlling inference behavior.
    /// </summary>
    /// <remarks>
    /// Use this class to configure generation parameters like temperature, top-p,
    /// and max tokens. All fields start unset — the model's defaults are used for
    /// any field you don't explicitly set.
    /// </remarks>
    /// <example>
    /// <code>
    /// // Use a preset
    /// var config = GenerationConfig.Greedy();
    ///
    /// // Or customize
    /// var config = new GenerationConfig();
    /// config.SetMaxTokens(512);
    /// config.SetTemperature(0.3f);
    ///
    /// using var result = model.Run(envelope, config);
    /// </code>
    /// </example>
    public sealed class GenerationConfig : IDisposable
    {
        private uint? _maxTokens;
        private float? _temperature;
        private float? _topP;
        private float? _minP;
        private uint? _topK;
        private float? _repetitionPenalty;
        private readonly List<string> _stopSequences = new List<string>();

        /// <summary>
        /// Gets whether this config has been disposed. Retained for source
        /// compatibility; the config now holds no native resources.
        /// </summary>
        public bool IsDisposed { get; private set; }

        /// <summary>
        /// Creates a new generation config with all fields unset (model defaults).
        /// </summary>
        public GenerationConfig()
        {
        }

        /// <summary>
        /// Creates a greedy decoding config (deterministic, temperature=0).
        /// </summary>
        public static GenerationConfig Greedy()
        {
            var config = new GenerationConfig();
            config._temperature = 0.0f;
            config._topP = 1.0f;
            config._topK = 0;
            return config;
        }

        /// <summary>
        /// Creates a creative generation config (higher temperature).
        /// </summary>
        public static GenerationConfig Creative()
        {
            var config = new GenerationConfig();
            config._temperature = 0.9f;
            config._topP = 0.95f;
            config._topK = 50;
            return config;
        }

        /// <summary>
        /// Set the maximum number of tokens to generate.
        /// </summary>
        /// <param name="maxTokens">Maximum tokens (e.g., 512, 2048).</param>
        public void SetMaxTokens(int maxTokens)
        {
            ThrowIfDisposed();
            _maxTokens = (uint)maxTokens;
        }

        /// <summary>
        /// Set the sampling temperature.
        /// </summary>
        /// <param name="temperature">
        /// Temperature value. 0.0 = deterministic, higher = more random.
        /// Typical range: 0.0 to 2.0.
        /// </param>
        public void SetTemperature(float temperature)
        {
            ThrowIfDisposed();
            _temperature = temperature;
        }

        /// <summary>
        /// Set the top-p (nucleus) sampling threshold.
        /// </summary>
        /// <param name="topP">Top-p value (0.0 to 1.0). Default: 0.9.</param>
        public void SetTopP(float topP)
        {
            ThrowIfDisposed();
            _topP = topP;
        }

        /// <summary>
        /// Set the min-p sampling threshold.
        /// </summary>
        /// <param name="minP">Min-p value (0.0 to 1.0). Default: 0.05.</param>
        public void SetMinP(float minP)
        {
            ThrowIfDisposed();
            _minP = minP;
        }

        /// <summary>
        /// Set top-k sampling (0 = disabled).
        /// </summary>
        /// <param name="topK">Top-k value. 0 disables top-k filtering. Default: 40.</param>
        public void SetTopK(int topK)
        {
            ThrowIfDisposed();
            _topK = (uint)topK;
        }

        /// <summary>
        /// Set the repetition penalty.
        /// </summary>
        /// <param name="penalty">Penalty value. 1.0 = disabled. Default: 1.1.</param>
        public void SetRepetitionPenalty(float penalty)
        {
            ThrowIfDisposed();
            _repetitionPenalty = penalty;
        }

        /// <summary>
        /// Add a stop sequence. Can be called multiple times.
        /// </summary>
        /// <param name="stop">The stop sequence string.</param>
        public void AddStop(string stop)
        {
            ThrowIfDisposed();
            if (stop == null)
                throw new ArgumentNullException(nameof(stop));
            _stopSequences.Add(stop);
        }

        /// <summary>
        /// Snapshot the current values as the bolt wire type consumed by
        /// <see cref="Model"/>. Grammar-constrained decoding is not exposed by
        /// the Unity API, so it is always null.
        /// </summary>
        internal XybridBolt.XybridGenerationConfig ToBolt() =>
            new XybridBolt.XybridGenerationConfig(
                _maxTokens,
                _temperature,
                _topP,
                _minP,
                _topK,
                _repetitionPenalty,
                _stopSequences.ToArray(),
                null);

        private void ThrowIfDisposed()
        {
            if (IsDisposed)
            {
                throw new ObjectDisposedException(nameof(GenerationConfig));
            }
        }

        /// <summary>
        /// No-op: the config holds no native resources. Retained so existing
        /// <c>using</c> call sites keep compiling.
        /// </summary>
        public void Dispose()
        {
            IsDisposed = true;
        }
    }
}

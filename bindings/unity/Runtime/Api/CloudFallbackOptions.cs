// Xybrid SDK - optional cloud fallback destination for a model run.

namespace Xybrid
{
    /// <summary>
    /// Selects a cloud destination for a model run. Setting a destination does
    /// not enable fallback; set <see cref="FallbackToCloud"/> explicitly.
    /// Omitted values preserve the existing SDK behavior.
    /// </summary>
    public sealed class CloudFallbackOptions
    {
        /// <summary>Allows the existing local-to-cloud fallback path.</summary>
        public bool FallbackToCloud { get; set; }

        /// <summary>Maximum grace tokens before the fallback transition.</summary>
        public uint MaxGraceTokens { get; set; }

        /// <summary>Optional provider selected by the caller.</summary>
        public string CloudProvider { get; set; }

        /// <summary>Optional cloud model selected by the caller.</summary>
        public string CloudModel { get; set; }

        /// <summary>
        /// Optional gateway base URL. The shared facade validates its scheme,
        /// host, and versioned /v1 path before executing the run.
        /// </summary>
        public string CloudGatewayUrl { get; set; }
    }
}

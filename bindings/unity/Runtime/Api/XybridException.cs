// Xybrid SDK - Exception Types
// High-level exception for Xybrid SDK errors.

using System;

namespace Xybrid
{
    /// <summary>
    /// Exception thrown when a Xybrid SDK operation fails.
    /// </summary>
    public class XybridException : Exception
    {
        /// <summary>
        /// Creates a new XybridException with the specified message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public XybridException(string message) : base(message)
        {
        }

        /// <summary>
        /// Creates a new XybridException with the specified message and inner exception.
        /// </summary>
        /// <param name="message">The error message.</param>
        /// <param name="innerException">The inner exception that caused this error.</param>
        public XybridException(string message, Exception innerException) : base(message, innerException)
        {
        }
    }

    /// <summary>
    /// Exception thrown when a requested model is not found.
    /// </summary>
    public class ModelNotFoundException : XybridException
    {
        /// <summary>
        /// The model ID that was not found.
        /// </summary>
        public string ModelId { get; }

        /// <summary>
        /// Creates a new ModelNotFoundException.
        /// </summary>
        /// <param name="modelId">The model ID that was not found.</param>
        public ModelNotFoundException(string modelId)
            : base($"Model not found: {modelId}")
        {
            ModelId = modelId;
        }
    }

    /// <summary>
    /// Exception thrown when inference execution fails.
    /// </summary>
    public class InferenceException : XybridException
    {
        /// <summary>
        /// Creates a new InferenceException.
        /// </summary>
        /// <param name="message">The error message from the inference engine.</param>
        public InferenceException(string message)
            : base($"Inference failed: {message}")
        {
        }
    }

    /// <summary>
    /// Translates bolt-layer exceptions into the public Xybrid exception
    /// taxonomy. Bolt surfaces failures as <see cref="XybridBolt.XybridErrorException"/>
    /// (typed variants) or <see cref="XybridBolt.BoltException"/> (loader /
    /// last-error), which the public API maps to
    /// <see cref="ModelNotFoundException"/> / <see cref="InferenceException"/> /
    /// <see cref="XybridException"/>.
    /// </summary>
    internal static class BoltErrors
    {
        /// <summary>Translate a bolt exception to the public taxonomy.</summary>
        public static XybridException Translate(Exception ex)
        {
            switch (ex)
            {
                case XybridBolt.XybridErrorException typed:
                    return TranslateError(typed.Error);
                case XybridBolt.BoltException bolt:
                    return new XybridException(bolt.Message, bolt);
                default:
                    return new XybridException(ex.Message, ex);
            }
        }

        private static XybridException TranslateError(XybridBolt.XybridError error)
        {
            switch (error)
            {
                case XybridBolt.XybridError.ModelNotFound modelNotFound:
                    return new ModelNotFoundException(modelNotFound.Id);
                case XybridBolt.XybridError.InferenceError inferenceError:
                    return new InferenceException(inferenceError.Message);
                default:
                    return new XybridException(Describe(error));
            }
        }

        /// <summary>
        /// A human-readable message for a typed error — the inner text, not the
        /// record's default <c>ToString()</c>. Used for a failed
        /// <see cref="InferenceResult.Error"/> so it reads like the pre-bolt
        /// "Inference failed: ..." messages rather than "NotLoaded { }".
        /// </summary>
        internal static string Describe(XybridBolt.XybridError error)
        {
            switch (error)
            {
                case XybridBolt.XybridError.ModelNotFound e: return $"model not found: {e.Id}";
                case XybridBolt.XybridError.DirectoryNotFound e: return $"directory not found: {e.Path}";
                case XybridBolt.XybridError.MetadataNotFound e: return $"metadata not found: {e.Path}";
                case XybridBolt.XybridError.MetadataInvalid e: return e.Message;
                case XybridBolt.XybridError.LoadError e: return e.Message;
                case XybridBolt.XybridError.InferenceError e: return e.Message;
                case XybridBolt.XybridError.AbortedForCloudFallback e: return e.Reason;
                case XybridBolt.XybridError.StreamingNotSupported _: return "streaming not supported";
                case XybridBolt.XybridError.NotLoaded _: return "model not loaded";
                case XybridBolt.XybridError.ConfigError e: return e.Message;
                case XybridBolt.XybridError.NetworkError e: return e.Message;
                case XybridBolt.XybridError.Offline e: return e.Message;
                case XybridBolt.XybridError.IoError e: return e.Message;
                case XybridBolt.XybridError.CacheError e: return e.Message;
                case XybridBolt.XybridError.PipelineError e: return e.Message;
                case XybridBolt.XybridError.CircuitOpen e: return e.Message;
                case XybridBolt.XybridError.RateLimited e: return $"rate limited (retry after {e.RetryAfterSecs}s)";
                case XybridBolt.XybridError.Timeout e: return $"timed out after {e.TimeoutMs}ms";
                case XybridBolt.XybridError.MissingArtifact e: return e.Message;
                case XybridBolt.XybridError.UnsupportedModelCapability e: return e.Message;
                case XybridBolt.XybridError.UnsupportedBackendCapability e: return e.Message;
                case XybridBolt.XybridError.InvalidImage e: return e.Message;
                default: return error.ToString();
            }
        }
    }
}

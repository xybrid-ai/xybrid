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
}

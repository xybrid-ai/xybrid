// Xybrid SDK - ConversationContext
// Manages multi-turn conversation history for LLM interactions.

using System;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// Manages multi-turn conversation history for LLM models.
    /// </summary>
    /// <remarks>
    /// ConversationContext stores conversation turns with proper message roles
    /// (System, User, Assistant). It supports:
    /// <list type="bullet">
    ///   <item>System prompts that persist across Clear() calls</item>
    ///   <item>Automatic FIFO pruning when history exceeds max length</item>
    ///   <item>Chat template formatting for different LLM formats</item>
    /// </list>
    ///
    /// This class must be disposed when no longer needed to release native resources.
    /// </remarks>
    /// <example>
    /// <code>
    /// using var context = new ConversationContext();
    /// context.SetSystem("You are a helpful assistant.");
    ///
    /// var input = Envelope.Text("Hello!", MessageRole.User);
    /// context.Push(input);
    ///
    /// using var result = model.Run(input, context);
    /// context.Push(Envelope.Text(result.Text, MessageRole.Assistant));
    /// </code>
    /// </example>
    public sealed class ConversationContext : IDisposable
    {
        private unsafe XybridContextHandle* _handle;
        private bool _disposed;

        /// <summary>
        /// Gets whether this context has been disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        /// <summary>
        /// Gets the conversation context ID.
        /// </summary>
        public unsafe string Id
        {
            get
            {
                ThrowIfDisposed();
                byte* idPtr = NativeMethods.xybrid_context_id(_handle);
                if (idPtr == null)
                    return string.Empty;
                return NativeHelpers.FromUtf8Ptr(idPtr);
            }
        }

        /// <summary>
        /// Gets the current history length (excluding system prompt).
        /// </summary>
        public unsafe uint HistoryLength
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.xybrid_context_history_len(_handle);
            }
        }

        /// <summary>
        /// Gets whether a system prompt is set.
        /// </summary>
        public unsafe bool HasSystem
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.xybrid_context_has_system(_handle) != 0;
            }
        }

        /// <summary>
        /// Gets the internal native handle. For internal use only.
        /// </summary>
        internal unsafe XybridContextHandle* Handle
        {
            get
            {
                ThrowIfDisposed();
                return _handle;
            }
        }

        /// <summary>
        /// Creates a new conversation context with a generated UUID.
        /// </summary>
        /// <exception cref="XybridException">Thrown if context creation fails.</exception>
        public unsafe ConversationContext()
        {
            _handle = NativeMethods.xybrid_context_new();
            if (_handle == null)
            {
                NativeHelpers.ThrowLastError("Failed to create conversation context");
            }
        }

        /// <summary>
        /// Creates a new conversation context with a specific ID.
        /// </summary>
        /// <param name="id">The context identifier.</param>
        /// <exception cref="ArgumentNullException">Thrown if id is null.</exception>
        /// <exception cref="XybridException">Thrown if context creation fails.</exception>
        public unsafe ConversationContext(string id)
        {
            if (id == null)
            {
                throw new ArgumentNullException(nameof(id));
            }

            byte[] idBytes = NativeHelpers.ToUtf8Bytes(id);
            fixed (byte* idPtr = idBytes)
            {
                _handle = NativeMethods.xybrid_context_with_id(idPtr);
                if (_handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create conversation context with ID");
                }
            }
        }

        /// <summary>
        /// Sets the system prompt for this conversation.
        /// </summary>
        /// <param name="systemPrompt">The system prompt text.</param>
        /// <remarks>
        /// The system prompt defines the assistant's behavior and persists
        /// across Clear() calls.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown if systemPrompt is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if setting system prompt fails.</exception>
        public unsafe void SetSystem(string systemPrompt)
        {
            ThrowIfDisposed();

            if (systemPrompt == null)
            {
                throw new ArgumentNullException(nameof(systemPrompt));
            }

            byte[] textBytes = NativeHelpers.ToUtf8Bytes(systemPrompt);
            fixed (byte* textPtr = textBytes)
            {
                int result = NativeMethods.xybrid_context_set_system(_handle, textPtr);
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to set system prompt");
                }
            }
        }

        /// <summary>
        /// Sets the maximum history length before FIFO pruning.
        /// </summary>
        /// <param name="maxLength">Maximum number of history entries (default is 50).</param>
        /// <remarks>
        /// When the history exceeds this limit, the oldest messages are dropped.
        /// </remarks>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if setting max length fails.</exception>
        public unsafe void SetMaxHistoryLength(uint maxLength)
        {
            ThrowIfDisposed();

            int result = NativeMethods.xybrid_context_set_max_history_len(_handle, maxLength);
            if (result != 0)
            {
                NativeHelpers.ThrowLastError("Failed to set max history length");
            }
        }

        /// <summary>
        /// Pushes a text message with the specified role to the conversation history.
        /// </summary>
        /// <param name="text">The message text.</param>
        /// <param name="role">The message role.</param>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if pushing the message fails.</exception>
        public unsafe void Push(string text, MessageRole role)
        {
            ThrowIfDisposed();

            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            byte[] textBytes = NativeHelpers.ToUtf8Bytes(text);
            fixed (byte* textPtr = textBytes)
            {
                XybridEnvelopeHandle* envelope = NativeMethods.xybrid_envelope_text_with_role(textPtr, (int)role);
                if (envelope == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create envelope for context push");
                }

                try
                {
                    int result = NativeMethods.xybrid_context_push(_handle, envelope);
                    if (result != 0)
                    {
                        NativeHelpers.ThrowLastError("Failed to push message to context");
                    }
                }
                finally
                {
                    NativeMethods.xybrid_envelope_free(envelope);
                }
            }
        }

        /// <summary>
        /// Clears the conversation history but preserves the system prompt and ID.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if clearing fails.</exception>
        public unsafe void Clear()
        {
            ThrowIfDisposed();

            int result = NativeMethods.xybrid_context_clear(_handle);
            if (result != 0)
            {
                NativeHelpers.ThrowLastError("Failed to clear conversation context");
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(ConversationContext));
            }
        }

        /// <summary>
        /// Releases the native resources used by this context.
        /// </summary>
        public unsafe void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != null)
                {
                    NativeMethods.xybrid_context_free(_handle);
                    _handle = null;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released.
        /// </summary>
        ~ConversationContext()
        {
            Dispose();
        }

        /// <summary>
        /// Returns a string representation of the context.
        /// </summary>
        public override string ToString()
        {
            if (_disposed)
                return "ConversationContext(disposed)";
            return $"ConversationContext(Id={Id}, History={HistoryLength})";
        }
    }
}

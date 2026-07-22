// Xybrid SDK - ConversationContext
// Manages multi-turn conversation history for LLM interactions.

using System;

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
    /// context.Push("Hello!", MessageRole.User);
    /// using var result = model.Run(Envelope.Text("Hello!"), context);
    /// context.Push(result.Text, MessageRole.Assistant);
    /// </code>
    /// </example>
    public sealed class ConversationContext : IDisposable
    {
        private readonly XybridBolt.XybridConversationContext _bolt;
        private bool _disposed;

        /// <summary>The bolt handle backing this context. For internal use.</summary>
        internal XybridBolt.XybridConversationContext Bolt
        {
            get
            {
                ThrowIfDisposed();
                return _bolt;
            }
        }

        /// <summary>Gets whether this context has been disposed.</summary>
        public bool IsDisposed => _disposed;

        /// <summary>Gets the conversation context ID.</summary>
        public string Id
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.Id();
            }
        }

        /// <summary>Gets the current history length (excluding system prompt).</summary>
        public uint HistoryLength
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.HistoryLen();
            }
        }

        /// <summary>Gets whether a system prompt is set.</summary>
        public bool HasSystem
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.HasSystem();
            }
        }

        /// <summary>Creates a new conversation context with a generated UUID.</summary>
        public ConversationContext()
        {
            _bolt = new XybridBolt.XybridConversationContext();
        }

        /// <summary>Creates a new conversation context with a specific ID.</summary>
        /// <param name="id">The context identifier.</param>
        /// <exception cref="ArgumentNullException">Thrown if id is null.</exception>
        public ConversationContext(string id)
        {
            if (id == null)
            {
                throw new ArgumentNullException(nameof(id));
            }
            _bolt = XybridBolt.XybridConversationContext.WithId(id);
        }

        /// <summary>
        /// Sets the system prompt for this conversation. Persists across Clear().
        /// </summary>
        /// <param name="systemPrompt">The system prompt text.</param>
        /// <exception cref="ArgumentNullException">Thrown if systemPrompt is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if setting the system prompt fails.</exception>
        public void SetSystem(string systemPrompt)
        {
            ThrowIfDisposed();
            if (systemPrompt == null)
            {
                throw new ArgumentNullException(nameof(systemPrompt));
            }

            try
            {
                _bolt.SetSystem(Envelope.Text(systemPrompt, MessageRole.System).Bolt);
            }
            catch (Exception ex) when (IsBoltError(ex))
            {
                throw BoltErrors.Translate(ex);
            }
        }

        /// <summary>
        /// Sets the maximum history length before FIFO pruning.
        /// </summary>
        /// <param name="maxLength">Maximum number of history entries (default is 50).</param>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        public void SetMaxHistoryLength(uint maxLength)
        {
            ThrowIfDisposed();
            _bolt.SetMaxHistoryLen(maxLength);
        }

        /// <summary>
        /// Pushes a text message with the specified role to the conversation history.
        /// </summary>
        /// <param name="text">The message text.</param>
        /// <param name="role">The message role.</param>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if pushing the message fails.</exception>
        public void Push(string text, MessageRole role)
        {
            ThrowIfDisposed();
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            try
            {
                _bolt.Push(Envelope.Text(text, role).Bolt);
            }
            catch (Exception ex) when (IsBoltError(ex))
            {
                throw BoltErrors.Translate(ex);
            }
        }

        /// <summary>
        /// Clears the conversation history but preserves the system prompt and ID.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown if this context is disposed.</exception>
        public void Clear()
        {
            ThrowIfDisposed();
            _bolt.Clear();
        }

        private static bool IsBoltError(Exception ex) =>
            ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException;

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(ConversationContext));
            }
        }

        /// <summary>Releases the native resources used by this context.</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _bolt.Dispose();
                _disposed = true;
            }
        }

        /// <summary>Returns a string representation of the context.</summary>
        public override string ToString()
        {
            if (_disposed)
            {
                return "ConversationContext(disposed)";
            }
            return $"ConversationContext(Id={Id}, History={HistoryLength})";
        }
    }
}

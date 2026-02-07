// Xybrid SDK - MessageRole
// Defines message roles for conversation context.

namespace Xybrid
{
    /// <summary>
    /// Message role for conversation context.
    /// Used to tag messages in a conversation history.
    /// </summary>
    public enum MessageRole
    {
        /// <summary>
        /// System prompt that defines assistant behavior.
        /// Persists across context.Clear() calls.
        /// </summary>
        System = 0,

        /// <summary>
        /// User message.
        /// </summary>
        User = 1,

        /// <summary>
        /// Assistant response.
        /// </summary>
        Assistant = 2
    }
}

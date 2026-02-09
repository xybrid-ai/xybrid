// Xybrid SDK - StreamToken
// Data class for tokens received during streaming inference.

namespace Xybrid
{
    /// <summary>
    /// Represents a single token emitted during streaming inference.
    /// </summary>
    /// <remarks>
    /// For LLM models, this is called for each generated token.
    /// For non-LLM models, a single StreamToken is emitted with the complete result.
    /// </remarks>
    public sealed class StreamToken
    {
        /// <summary>
        /// The generated token text (may be partial for multi-byte characters).
        /// </summary>
        public string Token { get; }

        /// <summary>
        /// The raw token ID, or null if not available.
        /// </summary>
        public long? TokenId { get; }

        /// <summary>
        /// Zero-based index of this token in the generation sequence.
        /// </summary>
        public uint Index { get; }

        /// <summary>
        /// Cumulative text generated so far (all tokens concatenated).
        /// </summary>
        public string CumulativeText { get; }

        /// <summary>
        /// Reason for stopping, or null if generation is still in progress.
        /// Values: "stop" (hit stop sequence/EOS), "length" (hit max_tokens).
        /// </summary>
        public string FinishReason { get; }

        /// <summary>
        /// Whether this is the final token in the sequence.
        /// </summary>
        public bool IsFinal => FinishReason != null;

        internal StreamToken(string token, long? tokenId, uint index,
                           string cumulativeText, string finishReason)
        {
            Token = token;
            TokenId = tokenId;
            Index = index;
            CumulativeText = cumulativeText;
            FinishReason = finishReason;
        }

        /// <summary>
        /// Returns a string representation of the token.
        /// </summary>
        public override string ToString()
        {
            return IsFinal
                ? $"StreamToken(Index={Index}, Token=\"{Token}\", Finish=\"{FinishReason}\")"
                : $"StreamToken(Index={Index}, Token=\"{Token}\")";
        }
    }
}

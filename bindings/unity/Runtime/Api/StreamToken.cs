// Xybrid SDK - StreamToken
// Data class for tokens received during streaming inference.

using System.Collections.Generic;

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
        /// Values: "stop" (hit stop sequence/EOS), "length" (hit max_tokens),
        /// "tool_calls" (the turn ended on a parseable tool-call block).
        /// </summary>
        public string FinishReason { get; }

        /// <summary>
        /// Whether this is the final token in the sequence.
        /// </summary>
        public bool IsFinal => FinishReason != null;

        /// <summary>
        /// Tool calls the model emitted this turn, on the final token only.
        /// </summary>
        /// <remarks>
        /// Tool-call blocks are suppressed from the streamed text, so there is
        /// nothing in <see cref="Token"/> to parse: halt here, run the tools,
        /// then continue the turn by streaming a tool-results envelope through
        /// the same call. Empty on every other token.
        /// </remarks>
        public IReadOnlyList<XybridBolt.XybridToolCall> ToolCalls { get; }

        /// <summary>
        /// Whether this token carries tool calls to execute.
        /// </summary>
        public bool HasToolCalls => ToolCalls != null && ToolCalls.Count > 0;

        /// <summary>
        /// The completed turn's raw output text, tool-call block included.
        /// Pass it as the prior assistant text when building the tool-results
        /// envelope. Null unless <see cref="HasToolCalls"/> is true.
        /// </summary>
        /// <remarks>
        /// Not the same as <see cref="CumulativeText"/>, which reports the
        /// streamed text with the protocol blocks suppressed.
        /// </remarks>
        public string RawText { get; }

        internal StreamToken(string token, long? tokenId, uint index,
                           string cumulativeText, string finishReason,
                           IReadOnlyList<XybridBolt.XybridToolCall> toolCalls = null,
                           string rawText = null)
        {
            Token = token;
            TokenId = tokenId;
            Index = index;
            CumulativeText = cumulativeText;
            FinishReason = finishReason;
            ToolCalls = toolCalls ?? System.Array.Empty<XybridBolt.XybridToolCall>();
            RawText = rawText;
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

// Unity-only additions to the generated BoltFFI bindings.
//
// This file used to hand-port the entire inference path — XybridEnvelope,
// XybridEnvelopeKind, XybridResult, their wire codecs, hand-written P/Invoke,
// and Run/RunWithContext — because boltffi 0.25.3's C# lowering dropped every
// signature touching a data-carrying enum. boltffi 0.29 emits all of it, so the
// hand-port is gone: keeping it would redefine the generated types and fail to
// compile.
//
// What remains is genuinely additive: callback-style streaming. The generated
// API is pull-based (RunStream -> StreamNext* -> StreamResult -> StreamClose),
// and Unity callers want to hand over an Action<XybridStreamToken> instead of
// driving that loop themselves.

#nullable enable

using System;

namespace XybridBolt
{
    public sealed partial class XybridModel
    {
        /// <summary>
        /// Streaming inference that invokes <paramref name="onToken"/> for each
        /// token and returns the final result.
        /// </summary>
        /// <exception cref="ArgumentNullException">If <paramref name="onToken"/> is null.</exception>
        /// <exception cref="XybridErrorException">If the model returns a typed error.</exception>
        /// <exception cref="ObjectDisposedException">If the model has been disposed.</exception>
        public XybridResult RunStreaming(
            XybridEnvelope envelope,
            Action<XybridStreamToken> onToken,
            XybridRunOptions? options = null,
            XybridCancellationToken? cancellation = null)
        {
            if (onToken is null) throw new ArgumentNullException(nameof(onToken));
            return WithCancellation(cancellation, token =>
                DrainStream(RunStream(envelope, options, token), onToken));
        }

        /// <summary>
        /// Context-aware <see cref="RunStreaming"/>: seeds inference with
        /// <paramref name="context"/>'s conversation history for multi-turn chat.
        /// </summary>
        /// <exception cref="ArgumentNullException">If <paramref name="onToken"/> is null.</exception>
        /// <exception cref="XybridErrorException">If the model returns a typed error.</exception>
        /// <exception cref="ObjectDisposedException">If the model has been disposed.</exception>
        public XybridResult RunStreamingWithContext(
            XybridEnvelope envelope,
            Action<XybridStreamToken> onToken,
            XybridConversationContext context,
            XybridRunOptions? options = null,
            XybridCancellationToken? cancellation = null)
        {
            if (onToken is null) throw new ArgumentNullException(nameof(onToken));
            return WithCancellation(cancellation, token =>
                DrainStream(RunStreamWithContext(envelope, context, options, token), onToken));
        }

        private static T WithCancellation<T>(
            XybridCancellationToken? cancellation,
            Func<XybridCancellationToken, T> run)
        {
            if (cancellation is not null) return run(cancellation);

            using var owned = new XybridCancellationToken();
            return run(owned);
        }

        /// <summary>
        /// Pump a started stream to completion, forwarding each token.
        /// </summary>
        /// <remarks>
        /// Closes the stream on every exit path, so an exception mid-stream
        /// cannot leak the native stream slot.
        /// </remarks>
        private XybridResult DrainStream(ulong streamId, Action<XybridStreamToken> onToken)
        {
            try
            {
                while (true)
                {
                    XybridStreamEvent streamEvent = StreamNext(streamId);
                    if (streamEvent.Kind != XybridStreamEventKind.Token)
                    {
                        break;
                    }

                    if (streamEvent.Token is XybridStreamToken token)
                    {
                        onToken(token);
                    }
                }

                return StreamResult(streamId);
            }
            finally
            {
                StreamClose(streamId);
            }
        }
    }
}

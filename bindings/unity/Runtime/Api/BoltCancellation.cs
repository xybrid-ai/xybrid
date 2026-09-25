// Xybrid SDK - Cancellation bridge
// Links a .NET CancellationToken to the native bolt stop button.

using System;
using System.Threading;

namespace Xybrid
{
    /// <summary>
    /// Bridges a .NET <see cref="CancellationToken"/> to the native
    /// <c>XybridBolt.XybridCancellationToken</c> that every bolt run requires.
    /// </summary>
    /// <remarks>
    /// The generated bolt surface takes the stop button as a <em>required</em>
    /// argument: BoltFFI 0.30.1 cannot express an optional handle parameter
    /// (both <c>Option&lt;&amp;Handle&gt;</c> and <c>Option&lt;Arc&lt;Handle&gt;&gt;</c>
    /// are rejected when lowering to the binding IR). So a run that is never
    /// cancelled still needs a token, and this type manufactures one rather
    /// than pushing that detail onto app code.
    ///
    /// When the caller passes <see cref="CancellationToken.None"/> the native
    /// token is created but never signalled, which costs one handle per run.
    /// When the caller passes a real token, cancelling it drives
    /// <c>Cancel()</c> on the native side, which stops generation at the next
    /// token boundary.
    ///
    /// Always dispose — the registration must be released before the native
    /// token, or the callback could fire against a freed handle.
    /// </remarks>
    internal readonly struct BoltCancellation : IDisposable
    {
        private readonly XybridBolt.XybridCancellationToken _token;
        private readonly CancellationTokenRegistration _registration;

        private BoltCancellation(
            XybridBolt.XybridCancellationToken token,
            CancellationTokenRegistration registration)
        {
            _token = token;
            _registration = registration;
        }

        /// <summary>The native token to hand to a bolt run method.</summary>
        internal XybridBolt.XybridCancellationToken Token => _token;

        /// <summary>
        /// Creates a native token, wired to <paramref name="cancellationToken"/>
        /// when that token can actually be cancelled.
        /// </summary>
        internal static BoltCancellation From(CancellationToken cancellationToken)
        {
            var token = new XybridBolt.XybridCancellationToken();

            if (!cancellationToken.CanBeCanceled)
            {
                return new BoltCancellation(token, default);
            }

            // Already cancelled: signal up front so the run fails fast rather
            // than generating a first token and only then noticing.
            if (cancellationToken.IsCancellationRequested)
            {
                token.Cancel();
                return new BoltCancellation(token, default);
            }

            // The callback runs on whichever thread cancels. Cancel() is
            // documented as safe from any thread, and Dispose() below releases
            // this registration before the handle, so the callback cannot
            // outlive the token it captures.
            CancellationTokenRegistration registration =
                cancellationToken.Register(static state =>
                    ((XybridBolt.XybridCancellationToken)state).Cancel(), token);

            return new BoltCancellation(token, registration);
        }

        public void Dispose()
        {
            // Order matters: unregister first, then free the native handle.
            _registration.Dispose();
            _token?.Dispose();
        }
    }
}

// Hand-written supplement to the generated BoltFFI C# bindings.
//
// boltffi 0.25.3's C# generator silently drops every method that takes an
// envelope (a data-carrying enum used as a function INPUT the 0.25.3 C#
// lowering can't express) plus the types they need -- XybridEnvelope,
// XybridEnvelopeKind, XybridResult. That covers the whole inference surface:
// `run` / `run_stream` / `run_with_context` / `run_stream_with_context` on the
// model, and `push` / `set_system` on the conversation context. Swift/Kotlin
// generate them fine; C# does not. Rather than block Unity on inference, this
// file hand-ports that path, exactly as bindings/python/xybrid/_bolt.py does
// for the Python SDK. It is pinned to the boltffi 0.25.3 wire ABI.
//
// The wire format mirrors the generated Swift (bindings/apple/.../xybrid_bolt.swift,
// which is device-validated) and reuses the generated C# WireReader/WireWriter
// primitives in Runtime/Bolt/XybridBolt.cs. This layer is NOT emitted by
// tools/scripts/gen_unity_bolt_csharp.py; that script only marks XybridModel and
// XybridConversationContext `partial` so the methods below can extend them. When
// boltffi >= 0.26 lands (which is expected to express this path), delete this
// file and let the generator emit inference natively.

#nullable enable

using System;
using System.Runtime.InteropServices;

namespace XybridBolt
{
    /// <summary>
    /// A unit of input or output crossing the FFI boundary: a payload
    /// (<see cref="XybridEnvelopeKind"/>) plus free-form metadata.
    /// </summary>
    public sealed class XybridEnvelope
    {
        public XybridEnvelopeKind Kind { get; }
        public XybridMetadataEntry[] Metadata { get; }

        public XybridEnvelope(XybridEnvelopeKind kind, XybridMetadataEntry[]? metadata = null)
        {
            Kind = kind ?? throw new ArgumentNullException(nameof(kind));
            Metadata = metadata ?? Array.Empty<XybridMetadataEntry>();
        }

        internal void WireEncodeTo(WireWriter wire)
        {
            Kind.WireEncodeTo(wire);
            wire.WriteI32(Metadata.Length);
            foreach (XybridMetadataEntry entry in Metadata)
            {
                entry.WireEncodeTo(wire);
            }
        }

        internal static XybridEnvelope Decode(WireReader reader) =>
            new XybridEnvelope(
                XybridEnvelopeKind.Decode(reader),
                reader.ReadEncodedArray<XybridMetadataEntry>(r => XybridMetadataEntry.Decode(r)));
    }

    /// <summary>
    /// The payload variants an <see cref="XybridEnvelope"/> can carry. The wire
    /// tags (0..4) and field order are part of the contract and must stay in
    /// lockstep with <c>XybridEnvelopeKind</c> in crates/xybrid-bolt/src/lib.rs.
    /// </summary>
    public abstract class XybridEnvelopeKind
    {
        // Private ctor seals the hierarchy to the nested variants below (nested
        // types can reach the enclosing type's private members).
        private XybridEnvelopeKind() { }

        internal abstract void WireEncodeTo(WireWriter wire);

        public sealed class Text : XybridEnvelopeKind
        {
            public string Value { get; }
            public Text(string value) { Value = value ?? throw new ArgumentNullException(nameof(value)); }
            internal override void WireEncodeTo(WireWriter wire)
            {
                wire.WriteI32(0);
                wire.WriteString(Value);
            }
        }

        public sealed class Audio : XybridEnvelopeKind
        {
            public byte[] Bytes { get; }
            public Audio(byte[] bytes) { Bytes = bytes ?? throw new ArgumentNullException(nameof(bytes)); }
            internal override void WireEncodeTo(WireWriter wire)
            {
                wire.WriteI32(1);
                wire.WriteBytes(Bytes);
            }
        }

        public sealed class Embedding : XybridEnvelopeKind
        {
            public float[] Values { get; }
            public Embedding(float[] values) { Values = values ?? throw new ArgumentNullException(nameof(values)); }
            internal override void WireEncodeTo(WireWriter wire)
            {
                wire.WriteI32(2);
                wire.WriteBlittableArray(Values);
            }
        }

        public sealed class Image : XybridEnvelopeKind
        {
            public byte[] Bytes { get; }
            public string Format { get; }
            public Image(byte[] bytes, string format)
            {
                Bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
                Format = format ?? throw new ArgumentNullException(nameof(format));
            }
            internal override void WireEncodeTo(WireWriter wire)
            {
                wire.WriteI32(3);
                wire.WriteBytes(Bytes);
                wire.WriteString(Format);
            }
        }

        public sealed class MultiPart : XybridEnvelopeKind
        {
            public XybridEnvelope[] Parts { get; }
            public MultiPart(XybridEnvelope[] parts) { Parts = parts ?? throw new ArgumentNullException(nameof(parts)); }
            internal override void WireEncodeTo(WireWriter wire)
            {
                wire.WriteI32(4);
                wire.WriteI32(Parts.Length);
                foreach (XybridEnvelope part in Parts)
                {
                    part.WireEncodeTo(wire);
                }
            }
        }

        internal static XybridEnvelopeKind Decode(WireReader reader)
        {
            int tag = reader.ReadI32();
            switch (tag)
            {
                case 0:
                    return new Text(reader.ReadString());
                case 1:
                    return new Audio(reader.ReadBytes());
                case 2:
                    return new Embedding(reader.ReadLengthPrefixedBlittableArray<float>());
                case 3:
                    return new Image(reader.ReadBytes(), reader.ReadString());
                case 4:
                    return new MultiPart(
                        reader.ReadEncodedArray<XybridEnvelope>(r => XybridEnvelope.Decode(r)));
                default:
                    throw new InvalidOperationException($"Unknown XybridEnvelopeKind wire tag: {tag}");
            }
        }
    }

    /// <summary>
    /// The output of <see cref="XybridModel.Run"/>: the result envelope plus
    /// the model id, output type, latency, execution target, and inference
    /// metrics.
    /// </summary>
    public sealed class XybridResult
    {
        public XybridEnvelope Envelope { get; }
        public XybridOutputType OutputType { get; }
        public string ModelId { get; }
        public uint LatencyMs { get; }

        /// <summary>
        /// Whether this answer came from the device or the cloud gateway.
        /// Cloud fallback keeps the model id identical on both legs, so this is
        /// the only way to tell them apart.
        /// </summary>
        public XybridExecutionTarget ExecutionTarget { get; }

        public XybridInferenceMetrics Metrics { get; }

        internal XybridResult(
            XybridEnvelope envelope,
            XybridOutputType outputType,
            string modelId,
            uint latencyMs,
            XybridExecutionTarget executionTarget,
            XybridInferenceMetrics metrics)
        {
            Envelope = envelope;
            OutputType = outputType;
            ModelId = modelId;
            LatencyMs = latencyMs;
            ExecutionTarget = executionTarget;
            Metrics = metrics;
        }

        // Field order must track the Rust `XybridResult` exactly:
        // execution_target is encoded between latency and metrics, so reading
        // metrics straight after latency would misalign every field that
        // follows.
        internal static XybridResult Decode(WireReader reader) =>
            new XybridResult(
                XybridEnvelope.Decode(reader),
                XybridOutputTypeWire.Decode(reader),
                reader.ReadString(),
                reader.ReadU32(),
                XybridExecutionTargetWire.Decode(reader),
                XybridInferenceMetrics.Decode(reader));
    }

    internal static class BoltInferenceNative
    {
        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_model_run")]
        internal static extern FfiBuf XybridModelRun(
            IntPtr self,
            byte[] envelope,
            UIntPtr envelopeLen,
            byte[] options,
            UIntPtr optionsLen);

        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_model_run_stream")]
        internal static extern FfiBuf XybridModelRunStream(
            IntPtr self,
            byte[] envelope,
            UIntPtr envelopeLen,
            byte[] options,
            UIntPtr optionsLen);

        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_model_stream_result")]
        internal static extern FfiBuf XybridModelStreamResult(IntPtr self, ulong streamId);

        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_model_run_with_context")]
        internal static extern FfiBuf XybridModelRunWithContext(
            IntPtr self,
            byte[] envelope,
            UIntPtr envelopeLen,
            IntPtr context,
            byte[] options,
            UIntPtr optionsLen);

        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_model_run_stream_with_context")]
        internal static extern FfiBuf XybridModelRunStreamWithContext(
            IntPtr self,
            byte[] envelope,
            UIntPtr envelopeLen,
            IntPtr context,
            byte[] options,
            UIntPtr optionsLen);

        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_conversation_context_push")]
        internal static extern FfiBuf XybridConversationContextPush(
            IntPtr self,
            byte[] envelope,
            UIntPtr envelopeLen);

        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_conversation_context_set_system")]
        internal static extern FfiBuf XybridConversationContextSetSystem(
            IntPtr self,
            byte[] envelope,
            UIntPtr envelopeLen);
    }

    public sealed partial class XybridModel
    {
        /// <summary>
        /// Run inference on <paramref name="envelope"/>, optionally overriding
        /// defaults with <paramref name="options"/>. Blocks until complete.
        /// </summary>
        /// <exception cref="ArgumentNullException">If <paramref name="envelope"/> is null.</exception>
        /// <exception cref="XybridErrorException">If the model returns a typed error.</exception>
        /// <exception cref="ObjectDisposedException">If the model has been disposed.</exception>
        public XybridResult Run(XybridEnvelope envelope, XybridRunOptions? options = null)
        {
            if (envelope is null) throw new ArgumentNullException(nameof(envelope));
            ThrowIfDisposed();

            byte[] envelopeBytes = EncodeEnvelope(envelope);
            byte[] optionsBytes = EncodeOptions(options);

            FfiBuf buf = BoltInferenceNative.XybridModelRun(
                _handle,
                envelopeBytes,
                (UIntPtr)envelopeBytes.Length,
                optionsBytes,
                (UIntPtr)optionsBytes.Length);
            try
            {
                var reader = new WireReader(buf);
                if (reader.ReadU8() != 0) throw new XybridErrorException(XybridError.Decode(reader));
                return XybridResult.Decode(reader);
            }
            finally
            {
                NativeMethods.FreeBuf(buf);
                // Run blocks for the whole inference. Once _handle is read into
                // the native call above, the JIT may treat `this` as dead; if it
                // is otherwise unreferenced, a GC here could run ~XybridModel and
                // free the handle while Rust is still using it (use-after-free).
                // Keep the wrapper alive across the call.
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// Run inference while delivering each token to
        /// <paramref name="onToken"/> on the calling thread. Blocks until the
        /// final result is available.
        /// </summary>
        /// <remarks>
        /// All model types support this shape. Check
        /// <see cref="SupportsTokenStreaming"/> to distinguish true
        /// token-by-token LLM output from models that emit one result token.
        /// If the callback throws, streaming is closed and that exception is
        /// propagated to the caller.
        /// </remarks>
        /// <exception cref="ArgumentNullException">
        /// If <paramref name="envelope"/> or <paramref name="onToken"/> is null.
        /// </exception>
        /// <exception cref="XybridErrorException">If the model returns a typed error.</exception>
        /// <exception cref="ObjectDisposedException">If the model has been disposed.</exception>
        public XybridResult RunStreaming(
            XybridEnvelope envelope,
            Action<XybridStreamToken> onToken,
            XybridRunOptions? options = null)
        {
            if (envelope is null) throw new ArgumentNullException(nameof(envelope));
            if (onToken is null) throw new ArgumentNullException(nameof(onToken));
            ThrowIfDisposed();

            byte[] envelopeBytes = EncodeEnvelope(envelope);
            byte[] optionsBytes = EncodeOptions(options);
            ulong streamId = StartStream(BoltInferenceNative.XybridModelRunStream(
                _handle,
                envelopeBytes,
                (UIntPtr)envelopeBytes.Length,
                optionsBytes,
                (UIntPtr)optionsBytes.Length));
            return DrainStream(streamId, onToken);
        }

        /// <summary>
        /// Context-aware variant of <see cref="Run"/>: seeds inference with
        /// <paramref name="context"/>'s conversation history for multi-turn chat.
        /// Only the generation config from <paramref name="options"/> is applied.
        /// </summary>
        /// <exception cref="ArgumentNullException">If any argument is null.</exception>
        /// <exception cref="XybridErrorException">If the model returns a typed error.</exception>
        /// <exception cref="ObjectDisposedException">If the model has been disposed.</exception>
        public XybridResult RunWithContext(
            XybridEnvelope envelope,
            XybridConversationContext context,
            XybridRunOptions? options = null)
        {
            if (envelope is null) throw new ArgumentNullException(nameof(envelope));
            if (context is null) throw new ArgumentNullException(nameof(context));
            ThrowIfDisposed();

            byte[] envelopeBytes = EncodeEnvelope(envelope);
            byte[] optionsBytes = EncodeOptions(options);

            FfiBuf buf = BoltInferenceNative.XybridModelRunWithContext(
                _handle,
                envelopeBytes,
                (UIntPtr)envelopeBytes.Length,
                context.RawHandle,
                optionsBytes,
                (UIntPtr)optionsBytes.Length);
            try
            {
                var reader = new WireReader(buf);
                if (reader.ReadU8() != 0) throw new XybridErrorException(XybridError.Decode(reader));
                return XybridResult.Decode(reader);
            }
            finally
            {
                NativeMethods.FreeBuf(buf);
                // Blocking call: keep both the model and the context alive so a
                // GC can't finalize either and free a handle mid-run.
                GC.KeepAlive(this);
                GC.KeepAlive(context);
            }
        }

        /// <summary>
        /// Context-aware variant of <see cref="RunStreaming"/>: streams tokens
        /// while seeding inference with <paramref name="context"/>'s history.
        /// </summary>
        /// <exception cref="ArgumentNullException">If any argument is null.</exception>
        /// <exception cref="XybridErrorException">If the model returns a typed error.</exception>
        /// <exception cref="ObjectDisposedException">If the model has been disposed.</exception>
        public XybridResult RunStreamingWithContext(
            XybridEnvelope envelope,
            Action<XybridStreamToken> onToken,
            XybridConversationContext context,
            XybridRunOptions? options = null)
        {
            if (envelope is null) throw new ArgumentNullException(nameof(envelope));
            if (onToken is null) throw new ArgumentNullException(nameof(onToken));
            if (context is null) throw new ArgumentNullException(nameof(context));
            ThrowIfDisposed();

            byte[] envelopeBytes = EncodeEnvelope(envelope);
            byte[] optionsBytes = EncodeOptions(options);

            ulong streamId;
            try
            {
                streamId = StartStream(BoltInferenceNative.XybridModelRunStreamWithContext(
                    _handle,
                    envelopeBytes,
                    (UIntPtr)envelopeBytes.Length,
                    context.RawHandle,
                    optionsBytes,
                    (UIntPtr)optionsBytes.Length));
            }
            finally
            {
                // The worker snapshots the context before returning the id, so
                // the context only needs to survive the start call.
                GC.KeepAlive(context);
            }
            return DrainStream(streamId, onToken);
        }

        /// <summary>Decode a run_stream* start buffer into a session id (or throw).</summary>
        private ulong StartStream(FfiBuf startBuf)
        {
            try
            {
                var reader = new WireReader(startBuf);
                if (reader.ReadU8() != 0) throw new XybridErrorException(XybridError.Decode(reader));
                return reader.ReadU64();
            }
            finally
            {
                NativeMethods.FreeBuf(startBuf);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// Pull events for <paramref name="streamId"/>, invoking
        /// <paramref name="onToken"/> per token until the terminal event, then
        /// return the final result. Closes the session on exit.
        /// </summary>
        private XybridResult DrainStream(ulong streamId, Action<XybridStreamToken> onToken)
        {
            try
            {
                while (true)
                {
                    XybridStreamEvent streamEvent = StreamNext(streamId);
                    switch (streamEvent.Kind)
                    {
                        case XybridStreamEventKind.Token:
                            if (streamEvent.Token is not { } token)
                            {
                                throw new InvalidOperationException(
                                    "Bolt streaming returned a token event without a token payload.");
                            }
                            onToken(token);
                            break;
                        case XybridStreamEventKind.Complete:
                            return TakeStreamResult(streamId);
                        default:
                            throw new InvalidOperationException(
                                $"Unknown Bolt stream event kind: {streamEvent.Kind}");
                    }
                }
            }
            finally
            {
                // Closing drops the bounded Rust receiver, which stops the
                // producer at its next token if the callback exits early.
                if (_handle != IntPtr.Zero)
                {
                    StreamClose(streamId);
                }
                GC.KeepAlive(this);
            }
        }

        private static byte[] EncodeEnvelope(XybridEnvelope envelope)
        {
            using var wire = new WireWriter(64);
            envelope.WireEncodeTo(wire);
            return wire.ToArray();
        }

        private static byte[] EncodeOptions(XybridRunOptions? options)
        {
            // Options cross as an Optional<XybridRunOptions>: a u8 presence tag
            // followed by the encoded options when present (matching the Swift
            // `writer.writeOptional(options)`).
            using var wire = new WireWriter(16);
            if (options is { } opts)
            {
                wire.WriteU8(1);
                opts.WireEncodeTo(wire);
            }
            else
            {
                wire.WriteU8(0);
            }
            return wire.ToArray();
        }

        private XybridResult TakeStreamResult(ulong streamId)
        {
            FfiBuf buf = BoltInferenceNative.XybridModelStreamResult(_handle, streamId);
            try
            {
                var reader = new WireReader(buf);
                if (reader.ReadU8() != 0) throw new XybridErrorException(XybridError.Decode(reader));
                return XybridResult.Decode(reader);
            }
            finally
            {
                NativeMethods.FreeBuf(buf);
            }
        }
    }

    public sealed partial class XybridConversationContext
    {
        /// <summary>
        /// Append a turn — typically a user or assistant message envelope — to
        /// the conversation history.
        /// </summary>
        /// <exception cref="ArgumentNullException">If <paramref name="envelope"/> is null.</exception>
        /// <exception cref="XybridErrorException">If the envelope fails validation.</exception>
        /// <exception cref="ObjectDisposedException">If the context has been disposed.</exception>
        public void Push(XybridEnvelope envelope)
        {
            if (envelope is null) throw new ArgumentNullException(nameof(envelope));
            ThrowIfDisposed();
            SendEnvelope(BoltInferenceNative.XybridConversationContextPush, envelope);
        }

        /// <summary>
        /// Set the persistent system-prompt envelope (survives <see cref="Clear"/>).
        /// </summary>
        /// <exception cref="ArgumentNullException">If <paramref name="envelope"/> is null.</exception>
        /// <exception cref="XybridErrorException">If the envelope fails validation.</exception>
        /// <exception cref="ObjectDisposedException">If the context has been disposed.</exception>
        public void SetSystem(XybridEnvelope envelope)
        {
            if (envelope is null) throw new ArgumentNullException(nameof(envelope));
            ThrowIfDisposed();
            SendEnvelope(BoltInferenceNative.XybridConversationContextSetSystem, envelope);
        }

        // Encode `envelope`, call `native(handle, bytes, len)`, and check the
        // returned Result tag (a `Result<(), XybridError>` crosses as a 1-byte
        // tag followed by the error when non-zero).
        private void SendEnvelope(
            Func<IntPtr, byte[], UIntPtr, FfiBuf> native,
            XybridEnvelope envelope)
        {
            using var wire = new WireWriter(64);
            envelope.WireEncodeTo(wire);
            byte[] bytes = wire.ToArray();

            FfiBuf buf = native(_handle, bytes, (UIntPtr)bytes.Length);
            try
            {
                var reader = new WireReader(buf);
                if (reader.ReadU8() != 0) throw new XybridErrorException(XybridError.Decode(reader));
            }
            finally
            {
                NativeMethods.FreeBuf(buf);
                GC.KeepAlive(this);
            }
        }
    }
}

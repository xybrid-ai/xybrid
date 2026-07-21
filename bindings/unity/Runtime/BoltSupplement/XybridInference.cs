// Hand-written supplement to the generated BoltFFI C# bindings.
//
// boltffi 0.25.3's C# generator silently drops the entire inference path: no
// `run` method, and none of the types it needs -- XybridEnvelope,
// XybridEnvelopeKind (a data-carrying enum used as a function INPUT, which the
// 0.25.3 C# lowering can't express), or XybridResult. Swift/Kotlin generate
// them fine; C# does not. Rather than block Unity on inference, this file
// hand-ports that path, exactly as bindings/python/xybrid/_bolt.py does for the
// Python SDK. It is pinned to the boltffi 0.25.3 wire ABI.
//
// The wire format mirrors the generated Swift (bindings/apple/.../xybrid_bolt.swift,
// which is device-validated) and reuses the generated C# WireReader/WireWriter
// primitives in Runtime/Bolt/XybridBolt.cs. This layer is NOT emitted by
// tools/scripts/gen_unity_bolt_csharp.py; that script only marks XybridModel
// `partial` so the Run() method below can extend it. When boltffi >= 0.26 lands
// (which is expected to express this path), delete this file and let the
// generator emit run() natively.

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
    /// the model id, output type, latency, and inference metrics.
    /// </summary>
    public sealed class XybridResult
    {
        public XybridEnvelope Envelope { get; }
        public XybridOutputType OutputType { get; }
        public string ModelId { get; }
        public uint LatencyMs { get; }
        public XybridInferenceMetrics Metrics { get; }

        internal XybridResult(
            XybridEnvelope envelope,
            XybridOutputType outputType,
            string modelId,
            uint latencyMs,
            XybridInferenceMetrics metrics)
        {
            Envelope = envelope;
            OutputType = outputType;
            ModelId = modelId;
            LatencyMs = latencyMs;
            Metrics = metrics;
        }

        internal static XybridResult Decode(WireReader reader) =>
            new XybridResult(
                XybridEnvelope.Decode(reader),
                XybridOutputTypeWire.Decode(reader),
                reader.ReadString(),
                reader.ReadU32(),
                XybridInferenceMetrics.Decode(reader));
    }

    internal static class BoltRunNative
    {
        [DllImport(NativeMethods.LibName, EntryPoint = "boltffi_xybrid_model_run")]
        internal static extern FfiBuf XybridModelRun(
            IntPtr self,
            byte[] envelope,
            UIntPtr envelopeLen,
            byte[] options,
            UIntPtr optionsLen);
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

            using var envelopeWire = new WireWriter(64);
            envelope.WireEncodeTo(envelopeWire);
            byte[] envelopeBytes = envelopeWire.ToArray();

            // Options cross as an Optional<XybridRunOptions>: a u8 presence tag
            // followed by the encoded options when present (matching the Swift
            // `writer.writeOptional(options)`).
            using var optionsWire = new WireWriter(16);
            if (options is { } opts)
            {
                optionsWire.WriteU8(1);
                opts.WireEncodeTo(optionsWire);
            }
            else
            {
                optionsWire.WriteU8(0);
            }
            byte[] optionsBytes = optionsWire.ToArray();

            FfiBuf buf = BoltRunNative.XybridModelRun(
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
            }
        }
    }
}

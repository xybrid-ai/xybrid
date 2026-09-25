#nullable enable

using System;
using System.Runtime.InteropServices;
using NUnit.Framework;
using XybridBolt;

namespace Xybrid.Tests.Editor
{
    [TestFixture]
    public class ResultWireCompatibilityTests
    {
        [Test]
        public void ResultDecoderAcceptsToolCallingWireWithoutReasoningTail()
        {
            XybridResult toolCalling = Decode(ResultWire());
            Assert.AreEqual(new[] { "call-1" }, Array.ConvertAll(toolCalling.ToolCalls, call => call.Id));
            Assert.AreEqual("metadata reasoning", toolCalling.ReasoningContent);

            XybridResult current = Decode(ResultWire(typedReasoning: "typed reasoning"));
            Assert.AreEqual(new[] { "call-1" }, Array.ConvertAll(current.ToolCalls, call => call.Id));
            Assert.AreEqual("typed reasoning", current.ReasoningContent);
        }

        private static XybridResult Decode(byte[] bytes)
        {
            GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            try
            {
                var reader = new WireReader(handle.AddrOfPinnedObject(), checked((nuint)bytes.Length));
                return XybridResult.Decode(reader);
            }
            finally
            {
                handle.Free();
            }
        }

        private static byte[] ResultWire(string? typedReasoning = null)
        {
            var envelope = new XybridEnvelope(
                new XybridEnvelopeKind.Text("answer"),
                new[] { new XybridMetadataEntry("reasoning_content", "metadata reasoning") }
            );
            var metrics = new XybridInferenceMetrics(
                7,
                null,
                null,
                null,
                null,
                null,
                Array.Empty<XybridStageLatency>()
            );
            var writer = new WireWriter();
            envelope.Encode(writer);
            writer.WriteI32((int)XybridOutputType.Text);
            writer.WriteString("model");
            writer.WriteU32(9);
            writer.WriteI32((int)XybridExecutionTarget.Local);
            metrics.Encode(writer);
            var toolCalls = new[] { new XybridToolCall("call-1", "lookup", "{}") };
            writer.WriteU32(checked((uint)toolCalls.Length));
            foreach (XybridToolCall call in toolCalls) call.Encode(writer);
            if (typedReasoning is not null)
            {
                writer.WriteU8(1);
                writer.WriteString(typedReasoning);
            }
            return writer.ToArray();
        }
    }
}

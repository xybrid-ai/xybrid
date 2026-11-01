#nullable enable

using System;
using System.Collections.Generic;
using System.Globalization;
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

        [Test]
        public void FromBoltParsesPhonemeTimingsFromMetadata()
        {
            // Through the wire, so the non-ASCII phoneme survives UTF-8 too.
            XybridResult wire = Decode(Encode(AudioResult(
                new XybridMetadataEntry("sample_rate", "24000"),
                new XybridMetadataEntry(
                    "phoneme_timings",
                    "[{\"phoneme\":\"h\",\"start\":0.0125,\"end\":0.05},"
                    + "{\"phoneme\":\"ə\",\"start\":0.05,\"end\":0.1}]"))));

            using InferenceResult result = InferenceResult.FromBolt(wire);

            Assert.IsTrue(result.Success);
            Assert.IsTrue(result.HasAudio);
            Assert.IsTrue(result.HasPhonemeTimings);
            AssertTimings(
                result.PhonemeTimings,
                ("h", 0.0125f, 0.05f),
                ("ə", 0.05f, 0.1f));
        }

        [Test]
        public void MissingOrMalformedPhonemeTimingsYieldEmptyList()
        {
            using (InferenceResult absent = InferenceResult.FromBolt(AudioResult()))
            {
                Assert.IsNotNull(absent.PhonemeTimings);
                Assert.AreEqual(0, absent.PhonemeTimings.Count);
                Assert.IsFalse(absent.HasPhonemeTimings);
                Assert.IsTrue(absent.HasAudio);
            }

            using (InferenceResult failed = InferenceResult.Failed("boom"))
            {
                Assert.IsNotNull(failed.PhonemeTimings);
                Assert.IsFalse(failed.HasPhonemeTimings);
            }

            using (InferenceResult empty = InferenceResult.FromBolt(
                       AudioResult(new XybridMetadataEntry("phoneme_timings", "[]"))))
            {
                Assert.IsNotNull(empty.PhonemeTimings);
                Assert.IsFalse(empty.HasPhonemeTimings);
            }

            string deep = new string('[', 1000) + new string(']', 1000);
            string[] malformed =
            {
                "",
                "   ",
                "null",
                "{}",
                "not json",
                "[",
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1}",
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1},]",
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1}] trailing",
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1}][]",
                "[{\"phoneme\":\"h\",\"start\":0}]",
                "[{\"start\":0,\"end\":1}]",
                "[{\"phoneme\":1,\"start\":0,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":\"0\",\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":null,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1,\"start\":2}]",
                "[{\"phoneme\":\"h\",\"start\":0,5,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":01,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":.5,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":1.,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":1e,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":+1,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":NaN,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":1e999,\"end\":1}]",
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1,\"x\":tru}]",
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1,\"x\":" + deep + "}]",
                "[{\"phoneme\":\"\\x\",\"start\":0,\"end\":1}]",
                "[{\"phoneme\":\"\\u12\",\"start\":0,\"end\":1}]",
                "[{\"phoneme\":\"\\ud83d\",\"start\":0,\"end\":1}]",
                "[{\"phoneme\":\"\\ude00\",\"start\":0,\"end\":1}]",
                "[{\"phoneme\":\"a\tb\",\"start\":0,\"end\":1}]",
                "[{\"phoneme\":\"h,\"start\":0,\"end\":1}]",
                "[{'phoneme':'h','start':0,'end':1}]",
                // A malformed entry anywhere empties the whole list.
                "[{\"phoneme\":\"h\",\"start\":0,\"end\":1},{\"phoneme\":\"i\"}]",
            };
            foreach (string json in malformed)
            {
                using InferenceResult result = InferenceResult.FromBolt(
                    AudioResult(new XybridMetadataEntry("phoneme_timings", json)));
                Assert.IsNotNull(result.PhonemeTimings, json);
                Assert.AreEqual(0, result.PhonemeTimings.Count, json);
                Assert.IsTrue(result.Success, json);
                Assert.IsTrue(result.HasAudio, json);
            }
        }

        [Test]
        public void PhonemeTimingParserHandlesEscapesExponentsAndCommaLocale()
        {
            const string json =
                " [\n"
                + "  {\"phoneme\": \"\\\"\", \"start\": 1.25e-2, \"end\": 5E-2},\n"
                + "  {\"end\": 1E-1, \"phoneme\": \"a\\\\b\\/\", \"start\": 0.05},\n"
                + "  {\"phoneme\": \"\\u0259\\u02C8\", \"start\": 0.1, \"end\": 0.125,"
                + " \"extra\": {\"nested\": [1, -2.5e+3, true, false, null, \"x\\\"y\"]}},\n"
                + "  {\"phoneme\": \"\\ud83d\\ude00\", \"start\": 1.5E+0, \"end\": 2}\n"
                + "]\r\n";

            CultureInfo saved = CultureInfo.CurrentCulture;
            try
            {
                CultureInfo.CurrentCulture = new CultureInfo("de-DE");
                // Guard the premise: this locale really does write one half as "0,5".
                Assume.That(CultureInfo.CurrentCulture.NumberFormat.NumberDecimalSeparator, Is.EqualTo(","));

                AssertTimings(
                    PhonemeTimingParser.Parse(json),
                    ("\"", 0.0125f, 0.05f),
                    ("a\\b/", 0.05f, 0.1f),
                    ("əˈ", 0.1f, 0.125f),
                    ("\U0001F600", 1.5f, 2f));
            }
            finally
            {
                CultureInfo.CurrentCulture = saved;
            }
        }

        private static void AssertTimings(
            IReadOnlyList<PhonemeTiming> actual,
            params (string Phoneme, float Start, float End)[] expected)
        {
            Assert.IsNotNull(actual);
            Assert.AreEqual(expected.Length, actual.Count);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i].Phoneme, actual[i].Phoneme, "phoneme " + i);
                Assert.AreEqual(expected[i].Start, actual[i].Start, "start " + i);
                Assert.AreEqual(expected[i].End, actual[i].End, "end " + i);
            }
        }

        private static XybridResult AudioResult(params XybridMetadataEntry[] metadata) =>
            new XybridResult(
                new XybridEnvelope(new XybridEnvelopeKind.Audio(new byte[] { 0, 0, 1, 0 }), metadata),
                XybridOutputType.Audio,
                "kokoro-82m-timed",
                12,
                XybridExecutionTarget.Local,
                new XybridInferenceMetrics(12, null, null, null, null, null, Array.Empty<XybridStageLatency>()),
                Array.Empty<XybridToolCall>());

        private static byte[] Encode(XybridResult result)
        {
            var writer = new WireWriter();
            result.Encode(writer);
            return writer.ToArray();
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

package ai.xybrid

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultWireCompatibilityTest {
    @Test
    fun resultDecoderAcceptsToolCallingWireWithoutReasoningTail() {
        val toolCalling = XybridResult.fromByteArray(resultWire())
        assertEquals(listOf("call-1"), toolCalling.toolCalls.map { it.id })
        assertEquals("metadata reasoning", toolCalling.reasoningContent)

        val current = XybridResult.fromByteArray(
            resultWire(typedReasoning = "typed reasoning")
        )
        assertEquals(listOf("call-1"), current.toolCalls.map { it.id })
        assertEquals("typed reasoning", current.reasoningContent)
        assertNull(current.observation)
    }

    @Test
    fun resultDecoderReadsObservationTail() {
        val observation = XybridObservationRef(
            traceId = "0af7651916cd43dd8448eb211c80319c",
            spanId = "b7ad6b7169203331",
        )

        val traced = XybridResult.fromByteArray(
            resultWire(typedReasoning = "typed reasoning", observation = observation)
        )

        assertEquals("typed reasoning", traced.reasoningContent)
        assertEquals(observation, traced.observation)
    }

    @Test
    fun runOptionsDefaultsEveryUnsetField() {
        val options = runOptions(conversationId = "conv-1", suppressTrace = true)

        assertNull(options.generationConfig)
        assertEquals(emptyList<XybridAbortSignal>(), options.abortOn)
        assertFalse(options.fallbackToCloud)
        assertEquals(0u, options.maxGraceTokens)
        assertNull(options.correlationId)
        assertNull(options.traceparent)
        assertNull(options.tracestate)
        assertEquals("conv-1", options.conversationId)
        assertTrue(options.suppressTrace)
    }

    private fun resultWire(
        typedReasoning: String? = null,
        observation: XybridObservationRef? = null,
    ): ByteArray {
        val envelope = XybridEnvelope(
            XybridEnvelopeKind.Text("answer"),
            listOf(XybridMetadataEntry("reasoning_content", "metadata reasoning")),
        )
        val metrics = XybridInferenceMetrics(
            totalMs = 7u,
            ttftMs = null,
            tokensPerSecond = null,
            prefillTps = null,
            decodeTps = null,
            tokensOut = null,
            stageLatenciesMs = emptyList(),
        )
        val writer = WireWriter(256)
        envelope.writeTo(writer)
        writer.writeI32(XybridOutputType.TEXT.value)
        writer.writeString("model")
        writer.writeU32(9u)
        writer.writeI32(XybridExecutionTarget.LOCAL.value)
        metrics.writeTo(writer)
        val toolCalls = listOf(XybridToolCall("call-1", "lookup", "{}"))
        writer.writeSequence(toolCalls, toolCalls.size) { output, call -> call.writeTo(output) }
        if (typedReasoning != null || observation != null) {
            writer.writeOptionalValue(typedReasoning) { output, reasoning ->
                output.writeString(reasoning)
            }
        }
        if (observation != null) {
            writer.writeOptionalValue(observation) { output, ref -> ref.writeTo(output) }
        }
        return writer.toByteArray()
    }
}

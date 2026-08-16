package ai.xybrid

import org.junit.Assert.assertEquals
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
    }

    private fun resultWire(
        typedReasoning: String? = null,
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
        if (typedReasoning != null) {
            writer.writeOptionalValue(typedReasoning) { output, reasoning ->
                output.writeString(reasoning)
            }
        }
        return writer.toByteArray()
    }
}

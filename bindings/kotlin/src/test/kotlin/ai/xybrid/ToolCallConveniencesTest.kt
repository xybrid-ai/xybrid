package ai.xybrid

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the `hasToolCalls` extensions. They exist so the branch a tool loop
 * actually writes reads the same in Kotlin as it does in Dart, Swift and C#.
 */
class ToolCallConveniencesTest {
    private val call = XybridToolCall("call_0", "get_weather", """{"city":"Paris"}""")

    @Test
    fun resultReportsWhetherTheTurnAskedForTools() {
        assertFalse(result(emptyList()).hasToolCalls)
        assertTrue(result(listOf(call)).hasToolCalls)
    }

    @Test
    fun onlyTheTerminalStreamTokenCarriesCalls() {
        val midStream = XybridStreamToken(
            token = "check",
            tokenId = null,
            index = 0uL,
            cumulativeText = "check",
            finishReason = null,
            toolCalls = emptyList(),
            rawText = null,
        )
        val terminal = XybridStreamToken(
            token = "",
            tokenId = null,
            index = 1uL,
            cumulativeText = "checking",
            finishReason = "tool_calls",
            toolCalls = listOf(call),
            rawText = "checking<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>",
        )

        assertFalse(midStream.hasToolCalls)
        assertTrue(terminal.hasToolCalls)
        assertEquals("tool_calls", terminal.finishReason)

        // rawText, not cumulativeText, is what the continuation replays: call
        // blocks are suppressed from the emitted text.
        assertTrue(terminal.rawText!!.contains("tool_call_start"))
        assertFalse(terminal.cumulativeText.contains("tool_call_start"))
    }

    private fun result(toolCalls: List<XybridToolCall>) = XybridResult(
        envelope = XybridEnvelope(XybridEnvelopeKind.Text("answer"), emptyList()),
        outputType = XybridOutputType.TEXT,
        modelId = "model",
        latencyMs = 0u,
        executionTarget = XybridExecutionTarget.LOCAL,
        metrics = XybridInferenceMetrics(
            totalMs = 0u,
            ttftMs = null,
            tokensPerSecond = null,
            prefillTps = null,
            decodeTps = null,
            tokensOut = null,
            stageLatenciesMs = emptyList(),
        ),
        toolCalls = toolCalls,
        reasoningContent = null,
    )
}

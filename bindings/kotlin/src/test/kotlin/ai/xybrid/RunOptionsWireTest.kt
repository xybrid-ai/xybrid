package ai.xybrid

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class RunOptionsWireTest {
    @Test
    fun legacyConstructorKeepsCloudTargetOmitted() {
        val options = XybridRunOptions(null, emptyList(), false, 0u, null)
        val decoded = XybridRunOptions.fromByteArray(options.toByteArray())

        assertFalse(decoded.fallbackToCloud)
        assertNull(decoded.cloudProvider)
        assertNull(decoded.cloudModel)
        assertNull(decoded.cloudGatewayUrl)
    }

    @Test
    fun explicitCloudTargetSurvivesWireRoundTripWithoutEnablingFallback() {
        val options = XybridRunOptions(
            generationConfig = null,
            abortOn = emptyList(),
            fallbackToCloud = false,
            maxGraceTokens = 0u,
            correlationId = "trace",
            cloudProvider = "openai",
            cloudModel = "gpt-4o-mini",
            cloudGatewayUrl = "https://api.xybrid.dev/v1",
        )
        val decoded = XybridRunOptions.fromByteArray(options.toByteArray())

        assertFalse(decoded.fallbackToCloud)
        assertEquals("trace", decoded.correlationId)
        assertEquals("openai", decoded.cloudProvider)
        assertEquals("gpt-4o-mini", decoded.cloudModel)
        assertEquals("https://api.xybrid.dev/v1", decoded.cloudGatewayUrl)
    }
}

package ai.xybrid

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ObservationTest {
    private val parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    @Test
    fun telemetryDeliveryHasADisplayMessage() {
        val error: XybridError = XybridError.TelemetryDelivery("2 spans dropped")

        assertEquals("Telemetry delivery failed: 2 spans dropped", error.displayMessage)
    }

    @Test
    fun streamingConfigsCarryObservationControls() {
        val options = observationOptions(traceparent = parent, conversationId = "chat-42")

        assertNull(defaultStreamingConfig().observation)
        assertNull(streamingConfigWithVad("/vad").observation)
        assertEquals(options, streamingConfigWithVad("/vad", observation = options).observation)
    }

    @Test
    fun contextChildOptionsKeepOnlyTheParent() {
        val context = XybridObservationContext(parent, "vendor=1")

        assertEquals(XybridObservationOptions(parent, "vendor=1", null, false), context.childOptions)
    }
}

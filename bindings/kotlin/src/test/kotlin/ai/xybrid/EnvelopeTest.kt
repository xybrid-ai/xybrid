package ai.xybrid

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class EnvelopeTest {
    @Test
    fun imageNormalizesJpegAlias() {
        val bytes = byteArrayOf(1, 2, 3)

        val envelope = Envelope.image(bytes, "JPG")

        assertTrue(envelope is XybridEnvelope.Image)
        val image = envelope as XybridEnvelope.Image
        assertArrayEquals(bytes, image.bytes)
        assertEquals("jpeg", image.format)
    }

    @Test
    fun imageRejectsUnsupportedFormat() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            Envelope.image(byteArrayOf(1), "gif")
        }

        assertTrue(error.message!!.contains("Unsupported image format"))
    }

    @Test
    fun userMessageAcceptsImageAttachments() {
        val image = Envelope.image(byteArrayOf(1), "png")

        val envelope = Envelope.userMessage("describe this", listOf(image))

        assertTrue(envelope is XybridEnvelope.UserMessage)
        val message = envelope as XybridEnvelope.UserMessage
        assertEquals("describe this", message.text)
        assertEquals(listOf(image), message.images)
    }

    @Test
    fun userMessageRejectsNonImageAttachments() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            Envelope.userMessage("describe this", listOf(Envelope.text("not an image")))
        }

        assertTrue(error.message!!.contains("only image envelopes"))
    }
}

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

        val image = envelope.kind as XybridEnvelopeKind.Image
        assertArrayEquals(bytes, image.bytes)
        assertEquals("jpeg", image.format)
    }

    @Test
    fun imageRejectsUnsupportedFormat() {
        val error = assertThrows(XybridError.ConfigError::class.java) {
            Envelope.image(byteArrayOf(1), "gif")
        }

        assertTrue(error.message.contains("Unsupported image format"))
    }

    @Test
    fun userMessageAcceptsImageAttachments() {
        val image = Envelope.image(byteArrayOf(1), "png")

        val envelope = Envelope.userMessage("describe this", listOf(image))

        val multipart = envelope.kind as XybridEnvelopeKind.MultiPart
        assertEquals(2, multipart.parts.size)
        assertEquals("describe this", (multipart.parts[0].kind as XybridEnvelopeKind.Text).text)
        assertEquals(image, multipart.parts[1])
        assertTrue(envelope.metadata.any { it.key == "xybrid.role" && it.value == "user" })
    }

    @Test
    fun userMessageRejectsNonImageAttachments() {
        val error = assertThrows(XybridError.ConfigError::class.java) {
            Envelope.userMessage("describe this", listOf(Envelope.text("not an image")))
        }

        assertTrue(error.message.contains("only image envelopes"))
    }
}

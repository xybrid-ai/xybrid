package ai.xybrid.smoke

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.xybrid.Envelope
import ai.xybrid.Xybrid
import ai.xybrid.XybridConversationContext
import ai.xybrid.version
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Executes the packaged JNI on Android without a model, credentials, or network. */
@RunWith(AndroidJUnit4::class)
class NativeBindingTest {
    @Test
    fun testPackagedNativeCalls() {
        Xybrid.init(InstrumentationRegistry.getInstrumentation().targetContext)
        assertTrue(Xybrid.isInitialized)
        assertTrue(version().isNotBlank())

        XybridConversationContext.withId("jni-smoke").use { context ->
            assertEquals("jni-smoke", context.id())
            assertEquals(0u, context.historyLen())
            val input = Envelope.text("JNI round trip: ação 日本語")
            context.push(input)
            assertEquals(1u, context.historyLen())
            assertEquals(input.kind, context.history().single().kind)
            context.clear()
            assertEquals(0u, context.historyLen())
        }
    }
}

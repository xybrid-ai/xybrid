package ai.xybrid

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test

class CancellationLeaseTest {
    @Test
    fun releaseWaitsForAnInFlightNativeCancel() {
        val enteredCancel = CountDownLatch(1)
        val leaveCancel = CountDownLatch(1)
        val enteredFinish = CountDownLatch(1)
        val released = CountDownLatch(1)
        val lease = CancellationLease(
            cancelNative = {
                enteredCancel.countDown()
                check(leaveCancel.await(5, TimeUnit.SECONDS))
            },
            releaseNative = { released.countDown() },
        )
        val cancelling = Thread { lease.cancel() }
        val finishing = Thread {
            enteredFinish.countDown()
            lease.finish()
        }
        cancelling.start()
        try {
            assertTrue(enteredCancel.await(2, TimeUnit.SECONDS))
            finishing.start()
            assertTrue(enteredFinish.await(2, TimeUnit.SECONDS))
            assertFalse("released a handle while cancel was using it", released.await(100, TimeUnit.MILLISECONDS))
        } finally {
            leaveCancel.countDown()
            cancelling.join(2000)
            finishing.join(2000)
        }
        assertFalse(cancelling.isAlive)
        assertFalse(finishing.isAlive)
        assertEquals(0L, released.count)
    }

    @Test
    fun lateCancellationNeverCallsAReleasedHandle() {
        val cancels = AtomicInteger()
        val releases = AtomicInteger()
        val lease = CancellationLease(
            cancelNative = { cancels.incrementAndGet() },
            releaseNative = { releases.incrementAndGet() },
        )
        lease.finish()
        lease.cancel()
        lease.finish()
        assertEquals(0, cancels.get())
        assertEquals(1, releases.get())
    }
}

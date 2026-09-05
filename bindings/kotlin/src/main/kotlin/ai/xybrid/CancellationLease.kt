package ai.xybrid

/**
 * Serializes cancellation with release of a wrapper-owned native handle.
 * The generated open-handle check and JNI call are not one atomic operation.
 * Never hold this lock during inference: cancellation must be able to enter.
 */
internal class CancellationLease(
    private val cancelNative: () -> Unit,
    private val releaseNative: () -> Unit,
) {
    private val lock = Any()
    private var finished = false

    fun cancel() = synchronized(lock) {
        if (!finished) cancelNative()
    }

    fun finish() = synchronized(lock) {
        if (!finished) {
            finished = true
            releaseNative()
        }
    }
}

package ai.xybrid.reactnative

import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Every native object JS holds — models, conversation contexts, cancellation
 * tokens, token streams, downloads, pipelines, live-ASR sessions — keyed by an
 * opaque `"<kind>:<uuid>"` string.
 *
 * A BoltFFI handle is a raw owned pointer with no reference count: closing it
 * while another thread is inside a call on it frees memory that call is still
 * using. So every call holds a [Lease] on the entries it touches, and
 * disposing a handle defers `onClose` until the last lease ends. `onDispose`
 * runs at once, for side effects that must not wait (aborting a stream,
 * stopping a session); entries owned by a disposed handle go with it.
 */
internal class XybridHandles {
  private val entries = ConcurrentHashMap<String, Entry>()

  private class Entry(
    val value: Any,
    val owner: String?,
    private val onDispose: (() -> Unit)?,
    private val onClose: (() -> Unit)?,
  ) {
    private val leases = AtomicInteger(0)
    private val closed = AtomicBoolean(false)

    @Volatile
    private var disposed = false

    /** Take a lease; `false` once the entry is disposed. */
    fun acquire(): Boolean {
      leases.incrementAndGet()
      if (disposed) {
        release()
        return false
      }
      return true
    }

    fun release() {
      if (leases.decrementAndGet() == 0 && disposed) close()
    }

    fun dispose() {
      disposed = true
      onDispose?.invoke()
      if (leases.get() == 0) close()
    }

    private fun close() {
      if (closed.compareAndSet(false, true)) onClose?.invoke()
    }
  }

  /** A claim on a live entry. Close it exactly once; extra closes are ignored. */
  class Lease<T : Any> internal constructor(val value: T, private val onRelease: () -> Unit) : AutoCloseable {
    private val released = AtomicBoolean(false)

    override fun close() {
      if (released.compareAndSet(false, true)) onRelease()
    }
  }

  /**
   * Register [value] and return its handle. [onClose] defaults to closing an
   * [AutoCloseable] (every BoltFFI handle is one).
   */
  fun insert(
    value: Any,
    kind: String,
    owner: String? = null,
    onDispose: (() -> Unit)? = null,
    onClose: (() -> Unit)? = (value as? AutoCloseable)?.let { closeable -> { closeable.close() } },
  ): String {
    val handle = "$kind:${UUID.randomUUID()}"
    entries[handle] = Entry(value, owner, onDispose, onClose)
    return handle
  }

  /**
   * Lease the object behind [handle]; a `xybrid_handle` error when it is
   * unknown, disposed, or another kind of object.
   */
  fun <T : Any> lease(handle: String, type: Class<T>): Lease<T> {
    val entry = entries[handle]
    val value = entry?.value
    if (entry == null || !type.isInstance(value) || !entry.acquire()) {
      throw BridgeException.Handle("unknown or released handle: $handle")
    }
    return Lease(type.cast(value)!!, entry::release)
  }

  /** Like [lease], but `null` instead of an error. */
  fun <T : Any> leaseOrNull(handle: String, type: Class<T>): Lease<T>? =
    try {
      lease(handle, type)
    } catch (e: BridgeException.Handle) {
      null
    }

  /** Run [block] on the object behind [handle] under a lease. */
  inline fun <T : Any, R> use(handle: String, type: Class<T>, block: (T) -> R): R =
    lease(handle, type).use { block(it.value) }

  /** Dispose [handle] and every entry it owns. Idempotent. */
  fun dispose(handle: String) {
    val removed = mutableListOf<Entry>()
    entries.remove(handle)?.let(removed::add)
    val owned = entries.entries.filter { it.value.owner == handle }
    for ((key, entry) in owned) {
      if (entries.remove(key, entry)) removed.add(entry)
    }
    removed.forEach(Entry::dispose)
  }

  /** Dispose everything (module invalidation: reload, teardown). */
  fun disposeAll() {
    val removed = entries.keys.toList().mapNotNull { entries.remove(it) }
    removed.forEach(Entry::dispose)
  }

  /** Run [action] on every live object of type [T], each under a lease. */
  fun <T : Any> forEachLive(type: Class<T>, action: (T) -> Unit) {
    for (entry in entries.values) {
      val value = entry.value
      if (!type.isInstance(value) || !entry.acquire()) continue
      try {
        action(type.cast(value)!!)
      } finally {
        entry.release()
      }
    }
  }
}

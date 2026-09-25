import Foundation

/// Every native object JS holds — models, conversation contexts, cancellation
/// tokens, token streams, downloads, pipelines, live-ASR sessions — keyed by
/// an opaque `"<kind>:<uuid>"` string.
///
/// Lifetimes are ARC's: in-flight calls hold their own strong references, so
/// disposing a handle while a call uses it only drops the registry's
/// reference and the object is freed when that call returns. `onDispose`
/// runs the side effects that must happen immediately (aborting a stream,
/// stopping a session) and entries owned by a disposed handle go with it.
final class XybridHandles {
  private struct Entry {
    let object: AnyObject
    let owner: String?
    let onDispose: (() -> Void)?
  }

  private let lock = NSLock()
  private var entries: [String: Entry] = [:]

  /// Register `object` and return its new handle.
  func insert(_ object: AnyObject,
              kind: String,
              owner: String? = nil,
              onDispose: (() -> Void)? = nil) -> String {
    let handle = "\(kind):\(UUID().uuidString)"
    lock.lock()
    entries[handle] = Entry(object: object, owner: owner, onDispose: onDispose)
    lock.unlock()
    return handle
  }

  /// The live object behind `handle`, or a `xybrid_handle` error when the
  /// handle is unknown, disposed, or names another kind of object.
  func get<T>(_ handle: String, as type: T.Type) throws -> T {
    lock.lock()
    let object = entries[handle]?.object
    lock.unlock()
    guard let typed = object as? T else {
      throw BridgeError.handle("unknown or released handle: \(handle)")
    }
    return typed
  }

  /// Like `get`, but `nil` instead of an error.
  func find<T>(_ handle: String, as type: T.Type) -> T? {
    try? get(handle, as: type)
  }

  /// Dispose `handle` and every entry it owns. Idempotent.
  func dispose(_ handle: String) {
    lock.lock()
    var removed: [Entry] = []
    if let entry = entries.removeValue(forKey: handle) { removed.append(entry) }
    for (key, entry) in entries where entry.owner == handle {
      entries.removeValue(forKey: key)
      removed.append(entry)
    }
    lock.unlock()
    // Side effects outside the lock: they call into the SDK.
    removed.forEach { $0.onDispose?() }
  }

  /// Dispose everything (module invalidation: reload, teardown).
  func disposeAll() {
    lock.lock()
    let removed = Array(entries.values)
    entries.removeAll()
    lock.unlock()
    removed.forEach { $0.onDispose?() }
  }

  /// Every live object of type `T` (e.g. to cancel all tokens on teardown).
  func all<T>(_ type: T.Type) -> [T] {
    lock.lock()
    defer { lock.unlock() }
    return entries.values.compactMap { $0.object as? T }
  }
}

/// A pull-based token stream: the model it runs on (bolt stream ids are
/// model-scoped) and the stop button that aborts it.
final class XybridTokenStreamEntry {
  let model: XybridModel
  let streamId: UInt64
  let cancel: XybridCancellationToken

  init(model: XybridModel, streamId: UInt64, cancel: XybridCancellationToken) {
    self.model = model
    self.streamId = streamId
    self.cancel = cancel
  }

  /// Abort generation now. Safe while a `streamNext` is in flight: closing
  /// only forgets the session id, and the in-flight pull keeps its own
  /// reference until it returns.
  func abort() {
    cancel.cancel()
    model.streamClose(streamId: streamId)
  }
}

/// A live-ASR session plus a pump that buffers its partial transcripts for
/// pull-based reads from JS.
final class XybridSessionEntry {
  let session: XybridStreamingSession
  private let partials: PartialQueue

  init(session: XybridStreamingSession) {
    self.session = session
    self.partials = PartialQueue(stream: session.partials())
  }

  deinit {
    partials.stop()
  }

  /// The next partial transcript, or `nil` once the session has finished.
  func nextPartial() async -> XybridPartialResult? {
    await partials.next()
  }
}

/// Drains an `AsyncStream` on its own task so JS can pull one element per
/// call. (An `AsyncStream` iterator must never be awaited from two tasks at
/// once, which overlapping bridge calls could otherwise do.)
private final class PartialQueue: @unchecked Sendable {
  /// Partials are cumulative — each supersedes the last — so a reader that
  /// falls this far behind only loses stale text.
  private static let capacity = 64

  private let lock = NSLock()
  private var buffer: [XybridPartialResult] = []
  private var waiters: [CheckedContinuation<XybridPartialResult?, Never>] = []
  private var finished = false
  private var pump: Task<Void, Never>?

  init(stream: AsyncStream<XybridPartialResult>) {
    pump = Task.detached { [weak self] in
      for await partial in stream {
        self?.push(partial)
      }
      self?.finish()
    }
  }

  func next() async -> XybridPartialResult? {
    await withCheckedContinuation { continuation in
      lock.lock()
      if !buffer.isEmpty {
        let partial = buffer.removeFirst()
        lock.unlock()
        continuation.resume(returning: partial)
      } else if finished {
        lock.unlock()
        continuation.resume(returning: nil)
      } else {
        waiters.append(continuation)
        lock.unlock()
      }
    }
  }

  func stop() {
    pump?.cancel()
    finish()
  }

  private func push(_ partial: XybridPartialResult) {
    lock.lock()
    if !waiters.isEmpty {
      let waiter = waiters.removeFirst()
      lock.unlock()
      waiter.resume(returning: partial)
      return
    }
    buffer.append(partial)
    if buffer.count > Self.capacity { buffer.removeFirst(buffer.count - Self.capacity) }
    lock.unlock()
  }

  private func finish() {
    lock.lock()
    finished = true
    let pending = waiters
    waiters.removeAll()
    lock.unlock()
    pending.forEach { $0.resume(returning: nil) }
  }
}

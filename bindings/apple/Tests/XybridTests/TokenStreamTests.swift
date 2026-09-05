import Foundation
import XCTest
@testable import Xybrid

final class TokenStreamTests: XCTestCase {
    func testPullsExactlyOnceForEachRequestedElement() async throws {
        let probe = StreamProbe(events: [
            .token("one", index: 0),
            .token("two", index: 1),
            .complete,
        ])
        let stream = probe.stream()
        var iterator = stream.makeAsyncIterator()

        XCTAssertEqual(probe.startCount, 0)
        XCTAssertEqual(probe.pullCount, 0)

        let first = try await iterator.next()
        XCTAssertEqual(first?.token, "one")
        XCTAssertEqual(probe.startCount, 1)
        XCTAssertEqual(probe.pullCount, 1)

        await Task.yield()
        XCTAssertEqual(probe.pullCount, 1, "the stream must not prefetch")

        let second = try await iterator.next()
        XCTAssertEqual(second?.token, "two")
        XCTAssertEqual(probe.pullCount, 2)
        let end = try await iterator.next()
        XCTAssertNil(end)
        XCTAssertEqual(probe.pullCount, 3)
        XCTAssertTrue(probe.waitUntilClosed(timeout: 2))
        XCTAssertEqual(probe.closeCount, 1)
    }

    func testErrorTerminatesIterator() async throws {
        let probe = StreamProbe(events: [])
        var iterator = probe.stream().makeAsyncIterator()

        do {
            _ = try await iterator.next()
            XCTFail("the first pull should throw")
        } catch {
            // The iterator must surface the native error exactly once.
        }

        let firstAfterError = try await iterator.next()
        let secondAfterError = try await iterator.next()
        XCTAssertNil(firstAfterError)
        XCTAssertNil(secondAfterError)
        XCTAssertEqual(probe.startCount, 1)
        XCTAssertEqual(probe.pullCount, 1)
        XCTAssertTrue(probe.waitUntilClosed(timeout: 2))
        XCTAssertEqual(probe.closeCount, 1)
    }

    func testStreamAllowsOnlyOneInferenceRun() async throws {
        let probe = StreamProbe(events: [
            .token("one", index: 0),
            .complete,
        ])
        let stream = probe.stream()
        let streamCopy = stream
        var firstIterator = stream.makeAsyncIterator()
        var secondIterator = streamCopy.makeAsyncIterator()

        let firstFromSecondIterator = try await secondIterator.next()
        let secondFromSecondIterator = try await secondIterator.next()
        XCTAssertNil(firstFromSecondIterator)
        XCTAssertNil(secondFromSecondIterator)
        XCTAssertEqual(probe.startCount, 0)

        let token = try await firstIterator.next()
        let end = try await firstIterator.next()
        XCTAssertEqual(token?.token, "one")
        XCTAssertNil(end)
        XCTAssertEqual(probe.startCount, 1)
        XCTAssertEqual(probe.pullCount, 2)
        XCTAssertTrue(probe.waitUntilClosed(timeout: 2))
        XCTAssertEqual(probe.closeCount, 1)
    }

    func testBreakingIterationClosesTheNativeSession() async throws {
        let probe = StreamProbe(events: [
            .token("one", index: 0),
            .token("two", index: 1),
        ])

        try await consumeOne(probe.stream())

        XCTAssertEqual(probe.pullCount, 1)
        XCTAssertTrue(probe.waitUntilClosed(timeout: 2))
        XCTAssertEqual(probe.closeCount, 1)
    }

    func testCancellationClosesAnInFlightPull() async throws {
        let probe = StreamProbe(events: [], blockPullUntilClosed: true)
        let stream = probe.stream()
        let task = Task { () throws -> XybridStreamToken? in
            var iterator = stream.makeAsyncIterator()
            return try await iterator.next()
        }

        XCTAssertTrue(probe.waitUntilPullStarts(timeout: 2))
        task.cancel()

        let result = try await task.value
        XCTAssertNil(result)
        XCTAssertEqual(probe.closeCount, 1)
    }

    func testCancellationDoesNotWaitForNativeCleanup() async throws {
        let probe = StreamProbe(events: [], blockPullUntilClosed: true, blockClose: true)
        let task = Task { () throws -> XybridStreamToken? in
            var iterator = probe.stream().makeAsyncIterator()
            return try await iterator.next()
        }
        XCTAssertTrue(probe.waitUntilPullStarts(timeout: 2))
        let cancellationReturned = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            task.cancel()
            cancellationReturned.signal()
        }
        let closeStarted = probe.waitUntilClosed(timeout: 2)
        let returned = await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                continuation.resume(returning: cancellationReturned.wait(timeout: .now() + 2) == .success)
            }
        }
        // Always release the mock worker, including on a failed assertion.
        probe.releaseClose()
        XCTAssertTrue(closeStarted)
        XCTAssertTrue(returned, "cancelling a task must not drain native inference inline")
        let result = try await task.value
        XCTAssertNil(result)
        XCTAssertEqual(probe.closeCount, 1)
    }

    private func consumeOne(_ stream: XybridTokenStream) async throws {
        for try await token in stream {
            XCTAssertEqual(token.token, "one")
            break
        }
    }
}

private final class StreamProbe: @unchecked Sendable {
    private let condition = NSCondition()
    private var events: [XybridStreamEvent]
    private let blockPullUntilClosed: Bool
    private let blockClose: Bool
    private var starts = 0
    private var pulls = 0
    private var closes = 0
    private var pullStarted = false
    private var closed = false
    private var closeReleased = false

    init(events: [XybridStreamEvent], blockPullUntilClosed: Bool = false, blockClose: Bool = false) {
        self.events = events
        self.blockPullUntilClosed = blockPullUntilClosed
        self.blockClose = blockClose
    }

    var startCount: Int { read { starts } }
    var pullCount: Int { read { pulls } }
    var closeCount: Int { read { closes } }

    func stream() -> XybridTokenStream {
        XybridTokenStream(
            start: { self.start() },
            next: { try self.next(streamId: $0) },
            close: { self.close(streamId: $0) }
        )
    }

    func waitUntilPullStarts(timeout: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(timeout)
        while !pullStarted {
            if !condition.wait(until: deadline) { return false }
        }
        return true
    }

    func waitUntilClosed(timeout: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(timeout)
        while !closed {
            if !condition.wait(until: deadline) { return false }
        }
        return true
    }

    func releaseClose() {
        condition.lock()
        closeReleased = true
        condition.broadcast()
        condition.unlock()
    }

    private func start() -> UInt64 {
        condition.lock()
        starts += 1
        condition.unlock()
        return 7
    }

    private func next(streamId: UInt64) throws -> XybridStreamEvent {
        guard streamId == 7 else {
            throw XybridError.inferenceError(message: "unexpected stream id")
        }
        condition.lock()
        pulls += 1
        pullStarted = true
        condition.broadcast()
        if blockPullUntilClosed {
            while !closed {
                condition.wait()
            }
            condition.unlock()
            throw XybridError.inferenceError(message: "stream closed")
        }
        guard !events.isEmpty else {
            condition.unlock()
            throw XybridError.inferenceError(message: "unexpected pull")
        }
        let event = events.removeFirst()
        condition.unlock()
        return event
    }

    private func close(streamId: UInt64) {
        guard streamId == 7 else { return }
        condition.lock()
        if !closed {
            closes += 1
            closed = true
        }
        condition.broadcast()
        while blockClose && !closeReleased {
            condition.wait()
        }
        condition.unlock()
    }

    private func read<T>(_ value: () -> T) -> T {
        condition.lock()
        defer { condition.unlock() }
        return value()
    }
}

private extension XybridStreamEvent {
    static func token(_ text: String, index: UInt64) -> Self {
        Self(
            kind: .token,
            token: XybridStreamToken(
                token: text,
                tokenId: nil,
                index: index,
                cumulativeText: text,
                finishReason: nil,
                toolCalls: [],
                rawText: nil
            )
        )
    }

    static var complete: Self {
        Self(kind: .complete, token: nil)
    }
}

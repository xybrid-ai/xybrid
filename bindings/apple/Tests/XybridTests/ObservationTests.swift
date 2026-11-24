import Foundation
import XCTest
@testable import Xybrid

final class ObservationTests: XCTestCase {
    func testTelemetryDeliveryHasADescription() {
        let error = XybridError.telemetryDelivery(message: "2 spans dropped")

        XCTAssertEqual(error.errorDescription, "Telemetry delivery failed: 2 spans dropped")
    }

    func testStreamingConfigsCarryObservationControls() {
        let options = XybridObservationOptions.make(
            traceparent: "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            conversationId: "chat-42"
        )

        XCTAssertNil(XybridStreamingConfig.default.observation)
        XCTAssertEqual(XybridStreamingConfig.default.observed(by: options).observation, options)
        XCTAssertEqual(
            XybridStreamingConfig.voiceActivity(modelDir: "/vad", observation: options).observation,
            options
        )
    }

    func testContextChildOptionsKeepOnlyTheParent() {
        let context = XybridObservationContext(
            traceparent: "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00",
            tracestate: "vendor=1"
        )

        XCTAssertEqual(
            context.childOptions,
            XybridObservationOptions(
                traceparent: context.traceparent,
                tracestate: "vendor=1",
                conversationId: nil,
                suppressTrace: false
            )
        )
    }
}

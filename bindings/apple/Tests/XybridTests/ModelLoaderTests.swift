import Foundation
import XCTest
@testable import Xybrid

final class ModelLoaderTests: XCTestCase {
    func testRegistryShortcutCreatesUnloadedRegistryReference() {
        let loader = Xybrid.model("kokoro-82m")

        XCTAssertEqual(loader.source, .registry("kokoro-82m"))
    }

    func testTypedSourceCreatesUnloadedBundleReference() {
        let url = URL(fileURLWithPath: "/models/kokoro.xyb")
        let source = ModelSource.bundle(url)

        let loader = Xybrid.model(source)

        XCTAssertEqual(loader.source, source)
    }

    func testResultDecodesReleasedNativeWireWithoutReasoningTail() {
        let envelope = XybridEnvelope(
            kind: .text(text: "answer"),
            metadata: [XybridMetadataEntry(key: "reasoning_content", value: "thinking")]
        )
        let metrics = XybridInferenceMetrics(
            totalMs: 1,
            ttftMs: nil,
            tokensPerSecond: nil,
            prefillTps: nil,
            decodeTps: nil,
            tokensOut: nil,
            stageLatenciesMs: []
        )
        let sourceCompatible = XybridResult(
            envelope: envelope,
            outputType: .text,
            modelId: "model",
            latencyMs: 1,
            executionTarget: .local,
            metrics: metrics
        )
        var writer = WireWriter()
        envelope.encode(to: &writer)
        writer.writeI32(XybridOutputType.text.rawValue)
        writer.writeString("model")
        writer.writeU32(1)
        writer.writeI32(XybridExecutionTarget.local.rawValue)
        metrics.encode(to: &writer)

        var reader = WireReader(data: writer.data)
        let result = XybridResult.decode(from: &reader)

        XCTAssertEqual(result.reasoningContent, "thinking")
        XCTAssertNil(sourceCompatible.reasoningContent)
        XCTAssertEqual(reader.position, writer.data.count)
    }
}

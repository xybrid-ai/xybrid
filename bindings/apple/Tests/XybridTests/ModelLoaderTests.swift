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

    func testResultDecodesToolCallingWireWithoutReasoningTail() {
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
            metrics: metrics,
            toolCalls: []
        )
        var writer = WireWriter()
        envelope.encode(to: &writer)
        writer.writeI32(XybridOutputType.text.rawValue)
        writer.writeString("model")
        writer.writeU32(1)
        writer.writeI32(XybridExecutionTarget.local.rawValue)
        metrics.encode(to: &writer)
        writer.writeArray([] as [XybridToolCall]) { writer, call in call.encode(to: &writer) }

        var reader = WireReader(data: writer.data)
        let result = XybridResult.decode(from: &reader)

        XCTAssertEqual(result.reasoningContent, "thinking")
        XCTAssertNil(sourceCompatible.reasoningContent)
        XCTAssertEqual(reader.position, writer.data.count)
    }

    /// Guards the `hasToolCalls` conveniences. They exist so the branch a
    /// tool loop actually writes reads the same in Swift as it does in Dart,
    /// Kotlin and C#.
    func testToolCallConveniencesOnResultAndStreamToken() {
        let call = XybridToolCall(
            id: "call_0",
            name: "get_weather",
            argumentsJson: #"{"city":"Paris"}"#
        )
        let metrics = XybridInferenceMetrics(
            totalMs: 0,
            ttftMs: nil,
            tokensPerSecond: nil,
            prefillTps: nil,
            decodeTps: nil,
            tokensOut: nil,
            stageLatenciesMs: []
        )
        func result(_ toolCalls: [XybridToolCall]) -> XybridResult {
            XybridResult(
                envelope: XybridEnvelope(kind: .text(text: "answer"), metadata: []),
                outputType: .text,
                modelId: "model",
                latencyMs: 0,
                executionTarget: .local,
                metrics: metrics,
                toolCalls: toolCalls
            )
        }

        XCTAssertFalse(result([]).hasToolCalls)
        XCTAssertTrue(result([call]).hasToolCalls)

        let midStream = XybridStreamToken(
            token: "check",
            tokenId: nil,
            index: 0,
            cumulativeText: "check",
            finishReason: nil,
            toolCalls: [],
            rawText: nil
        )
        let terminal = XybridStreamToken(
            token: "",
            tokenId: nil,
            index: 1,
            cumulativeText: "checking",
            finishReason: "tool_calls",
            toolCalls: [call],
            rawText: #"checking<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>"#
        )

        XCTAssertFalse(midStream.hasToolCalls)
        XCTAssertTrue(terminal.hasToolCalls)

        // rawText, not cumulativeText, is what the continuation replays: call
        // blocks are suppressed from the emitted text.
        XCTAssertTrue(terminal.rawText?.contains("tool_call_start") == true)
        XCTAssertFalse(terminal.cumulativeText.contains("tool_call_start"))
    }
}

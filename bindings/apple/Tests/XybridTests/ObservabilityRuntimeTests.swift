import Foundation
import XCTest
import Xybrid

/// The shared observability runtime scenario, run through the public Swift API.
///
/// Implements `tools/scripts/observability_capture/scenario.md`: every binding
/// performs the same steps against `capture_server.py` and writes the same
/// `expectations.json`, which `check_capture.py` then compares with what the
/// receiver recorded. No model weights are involved.
///
/// The test runs only when `XYBRID_SCENARIO_ENDPOINT` and
/// `XYBRID_SCENARIO_EXPECTATIONS` are set and skips otherwise, so ordinary
/// `swift test` / `xcodebuild test` runs are unaffected. `run_scenario.py`
/// sets them (plus `XYBRID_SCENARIO_WORKDIR`); `xcodebuild` forwards a
/// variable to the test runner on a simulator only under the
/// `TEST_RUNNER_` prefix, so the CI lane in `.github/workflows/build-apple.yml`
/// passes `TEST_RUNNER_XYBRID_SCENARIO_*`. `XYBRID_SCENARIO_SPEC` overrides
/// the path of `scenario.json`, which defaults to the copy in this checkout.
final class ObservabilityRuntimeTests: XCTestCase {
    func testSharedScenarioExportsWhatItReports() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let endpoint = nonEmpty(environment["XYBRID_SCENARIO_ENDPOINT"]),
              let expectationsPath = nonEmpty(environment["XYBRID_SCENARIO_EXPECTATIONS"])
        else {
            throw XCTSkip(
                "Set XYBRID_SCENARIO_ENDPOINT and XYBRID_SCENARIO_EXPECTATIONS (run_scenario.py does) "
                    + "to run the observability runtime scenario"
            )
        }
        let expectations = URL(fileURLWithPath: expectationsPath)
        let workdir = nonEmpty(environment["XYBRID_SCENARIO_WORKDIR"])
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? expectations.deletingLastPathComponent().appendingPathComponent("work", isDirectory: true)
        let spec = nonEmpty(environment["XYBRID_SCENARIO_SPEC"])
            .map { URL(fileURLWithPath: $0) }
            ?? defaultSpecURL()

        let scenario = ObservabilityScenario(endpoint: endpoint, workdir: workdir)
        let error = scenario.run(spec: spec)
        try scenario.write(to: expectations)
        if let error {
            XCTFail("The observability scenario stopped: \(error)")
        }
    }
}

// MARK: - Scenario

private let scenarioBinding = "swift"
private let scenarioSchema = "xybrid-observability-scenario/1"

private func nonEmpty(_ value: String?) -> String? {
    guard let value, !value.isEmpty else { return nil }
    return value
}

/// `tools/scripts/observability_capture/scenario.json` in the checkout this
/// file was compiled from (a simulator can read host paths).
private func defaultSpecURL() -> URL {
    var root = URL(fileURLWithPath: #filePath)
    // ObservabilityRuntimeTests.swift, XybridTests, Tests, apple, bindings
    for _ in 0..<5 {
        root.deleteLastPathComponent()
    }
    return root.appendingPathComponent("tools/scripts/observability_capture/scenario.json")
}

/// T and P: hex of the UTF-8 name, right-padded with zeros (scenario.md, Identifiers).
private func scenarioIdentifiers(_ binding: String) -> (traceId: String, parentSpanId: String) {
    func hex(_ text: String, digits: Int) -> String {
        let encoded = text.utf8.map { byte -> String in
            let digit = String(byte, radix: 16)
            return digit.count == 1 ? "0" + digit : digit
        }.joined()
        return String((encoded + String(repeating: "0", count: digits)).prefix(digits))
    }
    return (hex("xy:" + binding, digits: 32), hex(binding, digits: 16))
}

/// Stops the scenario with a message recorded as `scenario_error`.
private struct ScenarioFailure: Error, CustomStringConvertible {
    let description: String
}

private func describe(_ error: Error) -> String {
    "\(type(of: error)): \(error)"
}

/// Runs steps (a) to (e) and records what the binding reported.
private final class ObservabilityScenario {
    private let endpoint: String
    private let workdir: URL
    private let traceId: String
    private let parentSpanId: String
    private var references: [String: Any] = Dictionary(
        uniqueKeysWithValues: ["workflow", "tool", "inference", "inference_failed", "inference_released", "load", "run"]
            .map { ($0, NSNull() as Any) }
    )
    private var automatic: [String: Any] = [:]
    private var stats: [String: Any] = [:]
    private var lifecycle: [String: Any] = [:]
    private var scenarioError: String?

    init(endpoint: String, workdir: URL) {
        self.endpoint = endpoint
        self.workdir = workdir
        let ids = scenarioIdentifiers(scenarioBinding)
        traceId = ids.traceId
        parentSpanId = ids.parentSpanId
    }

    /// Runs the scenario; returns the error that stopped it, if any.
    func run(spec specURL: URL) -> String? {
        do {
            let spec = try ScenarioSpec.load(from: specURL)
            // What every app calls first; it registers `binding=swift`.
            Xybrid.initialize()
            try configure(spec)
            try runOperations(spec)
            try runLifecycle(spec)
        } catch {
            scenarioError = describe(error)
            // Leave nothing exporting to a receiver that is about to stop.
            try? Xybrid.Observability.shutdown(timeoutMs: 5_000)
        }
        return scenarioError
    }

    /// (f) Writes `expectations.json`, also after a failed step.
    func write(to url: URL) throws {
        let document: [String: Any] = [
            "schema": scenarioSchema,
            "binding": scenarioBinding,
            // The generated free function: `Xybrid.version()` would name the enum.
            "sdk_version": version(),
            "trace_id": traceId,
            "parent_span_id": parentSpanId,
            "references": references,
            "automatic": automatic,
            "stats": stats,
            "lifecycle": lifecycle,
            "scenario_error": scenarioError ?? NSNull(),
        ]
        let data = try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try (data + Data("\n".utf8)).write(to: url, options: .atomic)
    }

    // (a) OTLP/JSON export to the receiver, with content rules and a device id.
    private func configure(_ spec: ScenarioSpec) throws {
        try Xybrid.Observability.otlp(endpoint: endpoint)
            .jsonProtocol()
            .service(spec.forBinding(\.serviceName), version: spec.perBinding.serviceVersion)
            .deviceId(spec.forBinding(\.deviceId))
            .redactLiteral(spec.secret, replacement: spec.replacement)
            .omitContent(spec.omittedAttribute)
            .start()
    }

    /// Steps (b) to (d). Every handle is released when this returns.
    private func runOperations(_ spec: ScenarioSpec) throws {
        // (b) Parent-in: the workflow continues the caller's trace T under span P.
        let workflow = try Xybrid.Observability.startWorkflow(
            spec.workflowName, options: .make(traceparent: "00-\(traceId)-\(parentSpanId)-01")
        )
        references["workflow"] = reference(workflow.observationRef())
        guard let context = workflow.context() else {
            throw ScenarioFailure(description: "the workflow started under 00-\(traceId)-\(parentSpanId)-01 has no context")
        }
        let child = context.childOptions

        // (c) Manual children: a tool, a completed inference, a failed one and
        // one released unfinished.
        let tool = try Xybrid.Observability.startTool(
            XybridToolCallInfo(
                name: spec.tool.name,
                callId: spec.tool.callId,
                toolType: spec.tool.toolType,
                description: spec.tool.description,
                argumentsJson: spec.tool.argumentsJson
            ),
            options: child
        )
        try tool.recordResult(resultJson: spec.tool.resultJson)
        references["tool"] = reference(tool.observationRef())
        tool.complete()

        let request = try spec.inferenceRequest.request()
        let inference = try Xybrid.Observability.startInference(request, options: child)
        try inference.recordSystemInstructions(partsJson: spec.inference.systemInstructionsJson)
        try inference.recordInputMessages(messagesJson: spec.inference.inputMessagesJson)
        try inference.recordOutputMessages(messagesJson: spec.inference.outputMessagesJson)
        references["inference"] = reference(inference.observationRef())
        inference.complete()

        let failed = try Xybrid.Observability.startInference(request, options: child)
        references["inference_failed"] = reference(failed.observationRef())
        failed.fail(errorType: spec.failedInference.errorType, message: spec.failedInference.message)

        var released: XybridInferenceObservation? = try Xybrid.Observability.startInference(request, options: child)
        try released?.recordInputMessages(messagesJson: spec.releasedInference.inputMessagesJson)
        references["inference_released"] = reference(released?.observationRef())
        // Keep the handle alive across the first read whatever the optimizer does.
        stats["before_release"] = withExtendedLifetime(released) { snapshot() }
        released = nil  // the last owner, never ended: cancels, does not flush
        stats["after_release"] = snapshot()

        // (d) The automatic path: a weightless model loads lazily and fails to
        // run, both under the workflow with a conversation id.
        let modelDirectory = workdir.appendingPathComponent(spec.model.directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        try Data(spec.model.metadataJson.utf8)
            .write(to: modelDirectory.appendingPathComponent("model_metadata.json"), options: .atomic)
        let conversation = spec.forBinding(\.conversationId)
        let automaticParent = XybridObservationOptions.make(
            traceparent: context.traceparent, tracestate: context.tracestate, conversationId: conversation
        )
        var model: XybridModel?
        do {
            model = try XybridModel(load: .directory(path: modelDirectory.path), observation: automaticParent)
        } catch {
            automatic["load"] = ["outcome": "error", "error": describe(error)]
            automatic["run"] = ["outcome": "skipped", "error": NSNull()]
        }
        if let loaded = model {
            automatic["load"] = ["outcome": "ok", "error": NSNull()]
            references["load"] = reference(loaded.loadObservationRef())
            let options = XybridRunOptions.make(
                traceparent: context.traceparent, tracestate: context.tracestate, conversationId: conversation
            )
            do {
                let result = try loaded.run(envelope: .text(spec.model.runInputText), options: options)
                automatic["run"] = ["outcome": "ok", "error": NSNull()]
                references["run"] = reference(result.observation)
            } catch {
                automatic["run"] = ["outcome": "error", "error": describe(error)]
            }
        }
        model = nil  // release the model before the workflow ends

        workflow.complete()
    }

    // (e) Lifecycle: flush, stats, shutdown twice, then a call that must stay local.
    private func runLifecycle(_ spec: ScenarioSpec) throws {
        let timeout = spec.timeoutMs
        lifecycle["flush"] = outcome { try Xybrid.Observability.flush(timeoutMs: timeout) }
        stats["after_flush"] = snapshot()
        lifecycle["shutdown"] = (0..<2).map { _ in
            outcome { try Xybrid.Observability.shutdown(timeoutMs: timeout) }
        }
        stats["after_shutdown"] = snapshot()

        var after: XybridWorkflowObservation? = try Xybrid.Observability.startWorkflow(spec.afterShutdownWorkflowName)
        let afterReference = after?.observationRef()
        let afterContext = after?.context()
        after?.complete()
        after = nil
        lifecycle["after_shutdown"] = [
            "workflow_ref": reference(afterReference),
            "workflow_context": afterContext?.traceparent ?? NSNull(),
            "flush": outcome { try Xybrid.Observability.flush(timeoutMs: timeout) },
        ] as [String: Any]
    }

    private func reference(_ value: XybridObservationRef?) -> Any {
        guard let value else { return NSNull() }
        return ["trace_id": value.traceId, "span_id": value.spanId]
    }

    private func snapshot() -> [String: Any] {
        let current = Xybrid.Observability.stats
        func counters(_ signal: XybridSignalStats) -> [String: Any] {
            [
                "accepted": Int(clamping: signal.accepted),
                "exported": Int(clamping: signal.exported),
                "dropped": Int(clamping: signal.dropped),
                "failed": Int(clamping: signal.failed),
                "pending": Int(clamping: signal.pending),
            ]
        }
        return [
            "enabled": current.enabled,
            "traces": counters(current.traces),
            "logs": counters(current.logs),
            "metrics": counters(current.metrics),
        ]
    }

    private func outcome(_ body: () throws -> Void) -> String {
        do {
            try body()
            return "ok"
        } catch {
            return "error: \(describe(error))"
        }
    }
}

// MARK: - scenario.json

/// The constants the scenario needs from `scenario.json` (snake_case keys).
private struct ScenarioSpec: Decodable {
    struct PerBinding: Decodable {
        let serviceName: String
        let serviceVersion: String
        let deviceId: String
        let conversationId: String
    }

    struct Tool: Decodable {
        let name: String
        let callId: String?
        let toolType: String?
        let description: String?
        let argumentsJson: String?
        let resultJson: String
    }

    struct InferenceRequest: Decodable {
        let operation: String
        let provider: String
        let model: String?
        let location: String
        let runtime: String?
        let task: String?
        let stream: Bool
        let maxTokens: Int64?
        let temperature: Double?
        let topP: Double?
        let serverAddress: String?
        let serverPort: Int64?

        func request() throws -> XybridInferenceRequest {
            let executionLocation: XybridExecutionLocation
            switch location {
            case "local": executionLocation = .local
            case "remote": executionLocation = .remote
            default: throw ScenarioFailure(description: "scenario.json has an unknown location \(location)")
            }
            return XybridInferenceRequest(
                operation: operation,
                provider: provider,
                model: model,
                location: executionLocation,
                runtime: runtime,
                task: task,
                stream: stream,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                serverAddress: serverAddress,
                serverPort: serverPort
            )
        }
    }

    struct Inference: Decodable {
        let systemInstructionsJson: String
        let inputMessagesJson: String
        let outputMessagesJson: String
    }

    struct FailedInference: Decodable {
        let errorType: String
        let message: String
    }

    struct ReleasedInference: Decodable {
        let inputMessagesJson: String
    }

    struct Model: Decodable {
        let directoryName: String
        let metadataJson: String
        let runInputText: String
    }

    let schema: String
    let perBinding: PerBinding
    let secret: String
    let replacement: String
    let omittedAttribute: String
    let workflowName: String
    let afterShutdownWorkflowName: String
    let tool: Tool
    let inferenceRequest: InferenceRequest
    let inference: Inference
    let failedInference: FailedInference
    let releasedInference: ReleasedInference
    let model: Model
    let timeoutMs: UInt32

    static func load(from url: URL) throws -> ScenarioSpec {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ScenarioFailure(description: "cannot read \(url.path) (set XYBRID_SCENARIO_SPEC): \(error)")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let spec = try decoder.decode(ScenarioSpec.self, from: data)
        guard spec.schema == scenarioSchema else {
            throw ScenarioFailure(description: "\(url.path) is schema \(spec.schema), expected \(scenarioSchema)")
        }
        return spec
    }

    /// A per-binding template with `{binding}` filled in.
    func forBinding(_ template: KeyPath<PerBinding, String>) -> String {
        perBinding[keyPath: template].replacingOccurrences(of: "{binding}", with: scenarioBinding)
    }
}

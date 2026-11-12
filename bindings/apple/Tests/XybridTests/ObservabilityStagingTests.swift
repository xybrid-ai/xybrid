import Foundation
import XCTest
import Xybrid

/// The step 6 staging scenario, run through the public Swift API.
///
/// Implements `tools/scripts/observability_staging/README.md`: two sessions in
/// one process, `direct` (the Xybrid preset, metadata only) and `collector`
/// (OTLP to the pinned Collector, content captured with one literal
/// redaction), each loading a real registry model under a remote parent,
/// stopping one stream after three chunks, streaming a second to the end and
/// recording feedback and an evaluation on the run reference. The Python
/// reference is `bindings/python/tests/observability_staging.py`.
///
/// The test runs only when `XYBRID_STAGING_CONFIG` and
/// `XYBRID_STAGING_EXPECTATIONS` are set and skips otherwise, so ordinary
/// `xcodebuild test` runs are unaffected. `run_staging.py` writes the config
/// (it holds the staging key; the scenario deletes it once read) and fills in
/// both paths; on a simulator `xcodebuild` forwards them to the test runner
/// only as `TEST_RUNNER_XYBRID_STAGING_CONFIG` and
/// `TEST_RUNNER_XYBRID_STAGING_EXPECTATIONS`. A package test cannot run on a
/// physical iPhone: `bindings/apple/scripts/observability-scenario-ios-device.sh`
/// (`build-staging`, `run-staging`) runs the same scenario inside the example app.
final class ObservabilityStagingTests: XCTestCase {
    func testStagingScenarioReportsWhatItExported() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let configPath = environment["XYBRID_STAGING_CONFIG"], !configPath.isEmpty,
              let expectationsPath = environment["XYBRID_STAGING_EXPECTATIONS"], !expectationsPath.isEmpty
        else {
            throw XCTSkip(
                "Set XYBRID_STAGING_CONFIG and XYBRID_STAGING_EXPECTATIONS (run_staging.py does) "
                    + "to run the staging scenario"
            )
        }
        let scenario = try ObservabilityStagingScenario(configAt: URL(fileURLWithPath: configPath))
        let error = scenario.run()
        let expectations = URL(fileURLWithPath: expectationsPath)
        try FileManager.default.createDirectory(
            at: expectations.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try scenario.expectations().write(to: expectations, options: .atomic)
        if let error {
            XCTFail("The staging scenario stopped: \(error)")
        }
    }
}

// MARK: - Staging scenario

/// Runs both sessions and records what the binding reported as
/// `expectations.json`. Foundation and Xybrid only: the device script appends
/// everything from the line above to the example app.
private final class ObservabilityStagingScenario {
    private static let schema = "xybrid-staging-scenario/1"

    private let config: Config
    private var sessions: [String: Any] = [:]
    private var scenarioError: String?

    /// Reads `config.json` and deletes it straight away: it holds the staging
    /// key, which only the `direct` session's exporter receives.
    init(configAt url: URL) throws {
        let data = try Data(contentsOf: url)
        try FileManager.default.removeItem(at: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        config = try decoder.decode(Config.self, from: data)
        guard config.schema == Self.schema else {
            throw Failure(description: "config.json is schema \(config.schema), expected \(Self.schema)")
        }
    }

    /// Runs `direct` then `collector`; returns the error that stopped the
    /// scenario, if any.
    func run() -> String? {
        do {
            // What every app calls first; it registers `binding=swift`.
            Xybrid.initialize()
            for session in ["direct", "collector"] {
                var record: [String: Any] = [:]
                defer { sessions[session] = record }
                try runSession(session, into: &record)
            }
        } catch {
            scenarioError = "\(type(of: error)): \(error)"
            // Leave nothing exporting to a Collector that is about to stop.
            try? Xybrid.Observability.shutdown(timeoutMs: config.constants.timeoutMs)
        }
        return scenarioError
    }

    /// `expectations.json`, also after a failed step. Never holds the key.
    func expectations() throws -> Data {
        let document: [String: Any] = [
            "schema": Self.schema,
            "binding": config.binding,
            "lane": config.lane,
            "run": config.run,
            // The generated free function: `Xybrid.version()` would name the enum.
            "sdk_version": version(),
            "model_id": config.modelId,
            "device": Self.device(),
            "sessions": sessions,
            "scenario_error": orNull(scenarioError),
        ]
        let data = try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
        return data + Data("\n".utf8)
    }

    /// Steps 1 to 7 for one session (README, Steps).
    private func runSession(_ session: String, into record: inout [String: Any]) throws {
        let constants = config.constants
        guard let values = config.sessions[session] else {
            throw Failure(description: "config.json has no \(session) session")
        }
        let conversation = constants.perLane.conversationId
            .replacingOccurrences(of: "{run}", with: config.run)
            .replacingOccurrences(of: "{session}", with: session)
        var references: [String: Any] = ["workflow": NSNull(), "load": NSNull(), "run": NSNull()]
        record = ["trace_id": values.traceId, "parent_span_id": values.parentSpanId]
        defer { record["references"] = references }
        let initOutcome = outcome { try configure(session, values) }
        record["init"] = initOutcome
        guard initOutcome == "ok" else {
            throw Failure(description: "\(session) init failed: \(initOutcome)")
        }

        do {
            // Step 2: the workflow continues the driver's trace T under span P.
            let workflow = try Xybrid.Observability.startWorkflow(
                constants.workflowName, options: .make(traceparent: "00-\(values.traceId)-\(values.parentSpanId)-01")
            )
            references["workflow"] = reference(workflow.observationRef())
            guard let context = workflow.context() else {
                throw Failure(description: "the workflow started under the driver's parent has no context")
            }

            do {
                // Step 3: a real registry model, loaded under the workflow.
                let model = try XybridModel(
                    load: .registry(id: config.modelId, speculative: false),
                    observation: XybridObservationOptions.make(
                        traceparent: context.traceparent, tracestate: context.tracestate, conversationId: conversation
                    )
                )
                references["load"] = reference(model.loadObservationRef())

                // Step 4: stopped after three chunks by closing the stream.
                let stopped = try stream(
                    model, constants.stopped.prompt, runOptions(constants.stopped.maxTokens, context, conversation),
                    stopAfter: constants.stopped.stopAfterChunks
                )
                record["stopped"] = ["outcome": stopped.outcome, "chunks_consumed": stopped.chunks, "delivery": "pull"]

                // Step 5: streamed to the end; the result carries the run reference.
                let completed = try stream(
                    model, constants.completed.prompt,
                    runOptions(constants.completed.maxTokens, context, conversation), stopAfter: nil
                )
                guard let final = completed.result else {
                    throw Failure(description: "the completed stream returned no result")
                }
                record["completed"] = [
                    "outcome": completed.outcome,
                    "chunks": completed.chunks,
                    "tokens_out": orNull(final.metrics.tokensOut.map { Int($0) }),
                    "text_chars": (final.text ?? "").unicodeScalars.count,
                    "finish_reason": orNull(completed.finishReason),
                ]
                references["run"] = reference(final.observation)
                guard let runRef = final.observation else {
                    throw Failure(description: "the completed run returned no observation reference")
                }

                // Step 6: quality results on the run reference.
                let feedback = constants.feedback, evaluation = constants.evaluation
                let feedbackRecorded = try Xybrid.Observability.recordFeedback(
                    for: runRef, score: feedback.score, label: feedback.label, comment: feedback.comment
                )
                let evaluationRecorded = try Xybrid.Observability.recordEvaluation(
                    for: runRef, name: evaluation.name, score: evaluation.scoreValue, label: evaluation.scoreLabel,
                    explanation: evaluation.explanation
                )
                record["quality"] = ["feedback": feedbackRecorded, "evaluation": evaluationRecorded]
            }  // the model is released by here, before the workflow ends
            workflow.complete()
        }  // and the workflow here

        // Step 7: lifecycle.
        let timeout = constants.timeoutMs
        let flush = outcome { try Xybrid.Observability.flush(timeoutMs: timeout) }
        let afterFlush = snapshot()
        let shutdown = outcome { try Xybrid.Observability.shutdown(timeoutMs: timeout) }
        record["lifecycle"] = ["flush": flush, "shutdown": shutdown]
        record["stats"] = ["after_flush": afterFlush, "after_shutdown": snapshot()]
    }

    /// Step 1: the session's export, identity and content rules, then start.
    private func configure(_ session: String, _ values: Config.Session) throws {
        let constants = config.constants
        let observability: XybridObservabilityConfig
        if session == "direct" {
            let apiKey = try Self.field(values.apiKey, "sessions.direct.api_key")
            let endpoint = try Self.field(values.tracesEndpoint, "sessions.direct.traces_endpoint")
            observability = Xybrid.Observability.xybrid(apiKey: apiKey).tracesEndpoint(endpoint).metadataOnly()
        } else {
            let endpoint = try Self.field(values.endpoint, "sessions.\(session).endpoint")
            observability = try Xybrid.Observability.otlp(endpoint: endpoint)
                .redactLiteral(constants.secret, replacement: constants.replacement)
        }
        try observability
            .service(
                constants.perLane.serviceName.replacingOccurrences(of: "{binding}", with: config.binding),
                version: constants.perLane.serviceVersion
            )
            .deviceId(constants.perLane.deviceId.replacingOccurrences(of: "{lane}", with: config.lane))
            .resourceAttribute("xybrid.acceptance.run", config.run)
            .resourceAttribute("xybrid.acceptance.lane", config.lane)
            .resourceAttribute("xybrid.acceptance.commit", config.commit)
            .resourceAttribute("xybrid.acceptance.session", session)
            .start()
    }

    /// Steps 4 and 5: a pull stream, closed after `stopAfter` chunks or read
    /// to the end.
    private func stream(
        _ model: XybridModel, _ prompt: String, _ options: XybridRunOptions, stopAfter: Int?
    ) throws -> Streamed {
        let streamId = try model.runStream(envelope: .text(prompt), options: options)
        var open = true
        defer {
            if open { model.streamClose(streamId: streamId) }
        }
        var chunks = 0
        var finish: String?
        while true {
            let event = try model.streamNext(streamId: streamId)
            if event.kind == .complete {
                let result = try model.streamResult(streamId: streamId)  // also forgets the session
                open = false
                return Streamed(
                    outcome: stopAfter == nil ? "ok" : "completed", chunks: chunks, result: result, finishReason: finish
                )
            }
            guard event.kind == .token, let token = event.token else { continue }
            chunks += 1
            finish = token.finishReason ?? finish
            if let stopAfter, chunks >= stopAfter {
                // Closing before the terminal event cancels the run at its next
                // token (a user cancellation, error.type cancelled). The native
                // session buffers up to 32 tokens ahead of this loop: delivery
                // "pull".
                model.streamClose(streamId: streamId)
                open = false
                return Streamed(outcome: "cancelled", chunks: chunks, result: nil, finishReason: nil)
            }
        }
    }

    private func runOptions(
        _ maxTokens: UInt32, _ context: XybridObservationContext, _ conversation: String
    ) -> XybridRunOptions {
        .make(
            generationConfig: .make(maxTokens: maxTokens), traceparent: context.traceparent,
            tracestate: context.tracestate, conversationId: conversation
        )
    }

    private func reference(_ value: XybridObservationRef?) -> Any {
        guard let value else { return NSNull() }
        return ["trace_id": value.traceId, "span_id": value.spanId]
    }

    /// `value`, or JSON `null` when it is absent.
    private func orNull(_ value: Any?) -> Any {
        value ?? NSNull()
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
            return "error: \(type(of: error)): \(error)"
        }
    }

    /// A value only one session's config carries.
    private static func field(_ value: String?, _ name: String) throws -> String {
        guard let value, !value.isEmpty else { throw Failure(description: "config.json has no \(name)") }
        return value
    }

    /// What the Python reference reads from `platform`, from uname(3): the
    /// machine is the model identifier on an iPhone and `arm64` on a simulator.
    private static func device() -> [String: Any] {
        var name = utsname()
        uname(&name)
        func text<T>(_ field: T) -> String {
            withUnsafeBytes(of: field) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        }
        return [
            "system": text(name.sysname),
            "release": text(name.release),
            "machine": text(name.machine),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    /// Stops the scenario with a message recorded as `scenario_error`.
    private struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    /// What one stream delivered.
    private struct Streamed {
        let outcome: String
        let chunks: Int
        let result: XybridResult?
        let finishReason: String?
    }

    /// The parts of `config.json` the scenario reads (snake_case keys).
    private struct Config: Decodable {
        struct Session: Decodable {
            let traceId: String
            let parentSpanId: String
            // `direct` only; the key is never logged or written.
            let tracesEndpoint: String?
            let apiKey: String?
            // `collector` only.
            let endpoint: String?
        }

        struct Constants: Decodable {
            struct PerLane: Decodable {
                let serviceName: String
                let serviceVersion: String
                let deviceId: String
                let conversationId: String
            }

            struct Stopped: Decodable {
                let prompt: String
                let maxTokens: UInt32
                let stopAfterChunks: Int
            }

            struct Completed: Decodable {
                let prompt: String
                let maxTokens: UInt32
            }

            struct Feedback: Decodable {
                let score: Double
                let label: String
                let comment: String
            }

            struct Evaluation: Decodable {
                let name: String
                let scoreValue: Double
                let scoreLabel: String
                let explanation: String
            }

            let perLane: PerLane
            let workflowName: String
            let stopped: Stopped
            let completed: Completed
            let secret: String
            let replacement: String
            let feedback: Feedback
            let evaluation: Evaluation
            let timeoutMs: UInt32
        }

        let schema: String
        let binding: String
        let lane: String
        let run: String
        let commit: String
        let modelId: String
        let sessions: [String: Session]
        let constants: Constants
    }
}

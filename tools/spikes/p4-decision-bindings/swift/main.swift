// P4 spike: decision records round-trip through the committed generated Swift
// bindings (bindings/apple/Sources/Xybrid/xybrid_bolt.swift) and the real
// cdylib. Runs on Linux with a swift.org toolchain; see ../README.md.
// Mirrors ../python/roundtrip.py check for check.

import Foundation

let stateJSON =
    "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null}," +
    "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808," +
    "9223372036854775808,18446744073709551616,-0]," +
    "\"\u{e9}\":\"\\u00e9\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}"
let stateJSONCanonical =
    "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null}," +
    "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808," +
    "9223372036854775808,1.8446744073709552e19,0]," +
    "\"\u{e9}\":\"\u{e9}\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}"

var failures = 0

func check(_ name: String, _ condition: Bool, _ detail: String = "") {
    if condition { print("ok   \(name)") } else { failures += 1; print("FAIL \(name): \(detail)") }
}

func optionKeys() -> [String] { (0..<30).map { String(($0 * 7) % 30) } }

func optionDescription(_ i: Int) -> XybridDecisionContent? {
    switch i % 4 {
    case 0: return .text(value: "option \(i)")
    case 1: return .json(jsonText: "{\"z\":\(i),\"a\":[\(i).0,\"x\"]}")
    case 2: return nil
    default: return .json(jsonText: "\"a JSON string becomes text\"")
    }
}

func request(state: String = stateJSON) -> XybridDecisionRequest {
    XybridDecisionRequest(
        state: .json(jsonText: state),
        questions: [
            XybridDecisionQuestion(
                id: "10",
                instructions: .content(value: .json(jsonText: "{\"task\":\"pick\",\"why\":[3,1,2]}")),
                kind: .choice(options: optionKeys().enumerated().map {
                    XybridDecisionOption(key: $0.element, description: optionDescription($0.offset))
                })
            ),
            XybridDecisionQuestion(
                id: "2",
                instructions: .content(value: .text(value: "Is it urgent?")),
                kind: .yesNo(yes: .json(jsonText: "{\"means\":\"urgent\",\"rank\":1}"), no: nil)
            ),
            XybridDecisionQuestion(
                id: "1",
                instructions: nil,
                kind: .score(levels: [
                    nil, .text(value: "low"), .json(jsonText: "[1,2.5]"), .json(jsonText: "\"mid\""), .json(jsonText: "null"),
                ].map { XybridDecisionLevel(description: $0) })
            ),
        ]
    )
}

func expectedOutline() -> [String] {
    var lines = ["state=json:\(stateJSONCanonical)"]
    lines.append("q0:id=10:choice:instructions=json:{\"task\":\"pick\",\"why\":[3,1,2]}")
    for (i, key) in optionKeys().enumerated() {
        let description: String
        switch i % 4 {
        case 0: description = "text:option \(i)"
        case 1: description = "json:{\"z\":\(i),\"a\":[\(i).0,\"x\"]}"
        case 2: description = "none"
        default: description = "text:a JSON string becomes text"
        }
        lines.append("q0.o\(i):key=\(key):\(description)")
    }
    lines += [
        "q1:id=2:yes_no:instructions=text:Is it urgent?",
        "q1.yes:json:{\"means\":\"urgent\",\"rank\":1}",
        "q1.no:none",
        "q2:id=1:score:instructions=none",
        "q2.l0:none", "q2.l1:text:low", "q2.l2:json:[1,2.5]", "q2.l3:text:mid", "q2.l4:none",
    ]
    return lines
}

func bits32(_ value: Float) -> String { String(format: "%08x", value.bitPattern) }
func bits64(_ value: Double) -> String { String(format: "%016llx", value.bitPattern) }

func decodedBits(_ decisions: XybridDecisions) -> [String] {
    var lines: [String] = []
    for (a, answer) in decisions.answers.enumerated() {
        for (s, score) in answer.distribution.enumerated() {
            lines.append("a\(a).s\(s).raw=\(bits32(score.rawScore))")
            lines.append("a\(a).s\(s).p=\(bits64(score.probability))")
        }
        switch answer.value {
        case let .choice(key): lines.append("a\(a).value=choice:\(key)")
        case let .yesNo(pYes): lines.append("a\(a).value=yes_no:\(bits64(pYes))")
        case let .score(value, statistic):
            lines.append("a\(a).value=score:\(bits64(value)):\(statistic == .median ? "Median" : "Expected")")
        }
        lines.append(answer.confidence.map { "a\(a).confidence=\(bits64($0))" } ?? "a\(a).confidence=none")
    }
    return lines
}

func firstDifference(_ got: [String], _ expected: [String]) -> String {
    zip(got, expected).first { $0 != $1 }.map { "\($0) != \($1)" } ?? "\(got.count) vs \(expected.count) lines"
}

do {
    let req = request()

    let outline = try decisionSpikeRequestOutline(request: req)
    check("Rust saw the state, 3 questions and 30 options in order", outline == expectedOutline(),
          firstDifference(outline, expectedOutline()))

    var normalized = request(state: stateJSONCanonical)
    if case let .choice(options) = normalized.questions[0].kind {
        normalized.questions[0].kind = .choice(options: options.enumerated().map {
            $0.offset % 4 == 3 ? XybridDecisionOption(key: $0.element.key, description: .text(value: "a JSON string becomes text")) : $0.element
        })
    }
    if case var .score(levels) = normalized.questions[2].kind {
        levels[3] = XybridDecisionLevel(description: .text(value: "mid"))
        levels[4] = XybridDecisionLevel(description: nil)
        normalized.questions[2].kind = .score(levels: levels)
    }
    check("echo returns the normalized request", try decisionSpikeEchoRequest(request: req) == normalized)

    let decisions = try decisionSpikeDecide(request: req)
    check("answers in request order", decisions.answers.map(\.questionId) == ["10", "2", "1"])
    check("choice distribution in option order", decisions.answers[0].distribution.map(\.key) == optionKeys())
    check("yes/no order is true, false", decisions.answers[1].distribution.map(\.key) == ["true", "false"])
    check("yes/no has no confidence", decisions.answers[1].confidence == nil)
    check("score level 0 raw score is -inf", decisions.answers[2].distribution[0].rawScore == -.infinity)
    check("u64 usage crosses", decisions.usage.evaluatedTokens == 1000 + 1017 + 1034)

    let expectedBits = try decisionSpikeDecideBits(request: req)
    let gotBits = decodedBits(decisions)
    check("\(expectedBits.count) float bit patterns decode exactly", gotBits == expectedBits, firstDifference(gotBits, expectedBits))

    check("Swift re-encodes the result bit for bit", try decisionSpikeVerifyDecisions(request: req, decisions: decisions) == nil)
    var nudged = decisions
    nudged.answers[0].distribution[0].probability = decisions.answers[0].distribution[0].probability.nextUp
    check("a one-ULP change is detected",
          try decisionSpikeVerifyDecisions(request: req, decisions: nudged) == "answers[0].distribution[0]")

    do {
        _ = try decisionSpikeDecide(request: request(state: "{\"secret\":1,\"secret\":2}"))
        check("invalid JSON is refused", false)
    } catch let XybridError.inferenceError(message) {
        check("invalid JSON is refused as a typed, content-free error",
              message.contains("state") && !message.contains("secret"), message)
    }

    let probe = XybridProbeOptionalList(levels: [nil, .text(value: "x"), .json(jsonText: "[1]"), nil], labels: [nil, "a", nil])
    check("Vec<Option<data enum>> survives", decisionSpikeProbeOptionalList(probe: probe) == probe)

    let nanPayload = Double(bitPattern: 0x7FF8_0000_0000_1234)
    let specials: [Double] = [-0.0, 5e-324, .greatestFiniteMagnitude, -.infinity, nanPayload]
    let nesting = XybridProbeNesting(
        rawScores: [[1.5, -.infinity], [], [-0.0]],
        probabilities: [specials, [0.1 + 0.2]],
        usage: XybridDecisionUsage(passes: .max, evaluatedTokens: .max)
    )
    let back = decisionSpikeProbeNesting(probe: nesting)
    check("nested lists keep f64 bits (-0.0, subnormal, max, -inf, NaN payload)",
          back.probabilities[0].map(bits64) == specials.map(bits64) && bits64(back.probabilities[1][0]) == bits64(0.1 + 0.2))
    check("nested f32 rows and Option<record> survive",
          back.rawScores.map { $0.map(bits32) } == nesting.rawScores.map { $0.map(bits32) } && back.usage == nesting.usage)

    let tree = try decisionSpikeProbeJson(jsonText: "{\"2\":1,\"10\":[1.0,-0,18446744073709551615],\"1\":{\"z\":null,\"a\":\"s\"}}")
    if case let .obj(entries) = tree {
        check("JSON tree keeps integer-like key order", entries.map(\.key) == ["2", "10", "1"])
        if case let .seq(items) = entries[1].value, case .float = items[0], case .int = items[1], case .uint = items[2] {
            check("JSON tree keeps number identity", true)
        } else {
            check("JSON tree keeps number identity", false)
        }
    } else {
        check("JSON tree is an object", false)
    }
    check("recursive JSON tree echoes", decisionSpikeProbeJsonEcho(value: tree) == tree)

    if case let .decisionResult(result) = decisionSpikeProbePayload(payload: .decisionResult(decisions: decisions)) {
        check("a result inside a data-enum variant survives",
              try decisionSpikeVerifyDecisions(request: req, decisions: result) == nil)
    } else {
        check("a result inside a data-enum variant survives", false)
    }
    let requestPayload = XybridProbePayload.decisionRequest(request: req)
    check("a request inside a data-enum variant survives", decisionSpikeProbePayload(payload: requestPayload) == requestPayload)

    print("note String(describing:) of a request shows caller content: \(String(describing: req).contains("urgent"))")
} catch {
    failures += 1
    print("FAIL unexpected error: \(error)")
}
print(failures == 0 ? "SWIFT ROUND TRIP: ALL CHECKS PASSED" : "SWIFT ROUND TRIP: \(failures) FAILED")
exit(failures == 0 ? 0 : 1)

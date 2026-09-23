import Foundation

struct XybridStageLatency {
  let stageId: String
  let latencyMs: UInt32
}

struct XybridInferenceMetrics {
  let totalMs: UInt32
  let ttftMs: UInt32?
  let tokensPerSecond: Float?
  let prefillTps: Float?
  let decodeTps: Float?
  let tokensOut: UInt32?
  let stageLatenciesMs: [XybridStageLatency]
}

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
  guard condition() else {
    FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8))
    exit(1)
  }
}

let populated = encodeInferenceMetrics(
  XybridInferenceMetrics(
    totalMs: 120,
    ttftMs: 18,
    tokensPerSecond: 42.5,
    prefillTps: 95.25,
    decodeTps: 39.75,
    tokensOut: 24,
    stageLatenciesMs: [
      XybridStageLatency(stageId: "preprocess", latencyMs: 7),
      XybridStageLatency(stageId: "inference", latencyMs: 108),
      XybridStageLatency(stageId: "postprocess", latencyMs: 5),
    ]
  )
)

require(populated["totalMs"] as? UInt32 == 120, "totalMs changed")
require(populated["ttftMs"] as? UInt32 == 18, "ttftMs changed")
require(populated["tokensPerSecond"] as? Float == 42.5, "tokensPerSecond changed")
require(populated["prefillTps"] as? Float == 95.25, "prefillTps changed")
require(populated["decodeTps"] as? Float == 39.75, "decodeTps changed")
require(populated["tokensOut"] as? UInt32 == 24, "tokensOut changed")
let stages = populated["stageLatenciesMs"] as? [[String: Any]]
require(stages?.count == 3, "stage count changed")
require(stages?[0]["stageId"] as? String == "preprocess", "first stage reordered")
require(stages?[0]["latencyMs"] as? UInt32 == 7, "first stage latency changed")
require(stages?[1]["stageId"] as? String == "inference", "second stage reordered")
require(stages?[2]["stageId"] as? String == "postprocess", "third stage reordered")

let absent = encodeInferenceMetrics(
  XybridInferenceMetrics(
    totalMs: 33,
    ttftMs: nil,
    tokensPerSecond: nil,
    prefillTps: nil,
    decodeTps: nil,
    tokensOut: nil,
    stageLatenciesMs: []
  )
)
for key in ["ttftMs", "tokensPerSecond", "prefillTps", "decodeTps", "tokensOut"] {
  require(absent[key] == nil, "\(key) should be absent")
}

let zeros = encodeInferenceMetrics(
  XybridInferenceMetrics(
    totalMs: 0,
    ttftMs: 0,
    tokensPerSecond: 0,
    prefillTps: 0,
    decodeTps: 0,
    tokensOut: 0,
    stageLatenciesMs: [XybridStageLatency(stageId: "zero", latencyMs: 0)]
  )
)
require(zeros["ttftMs"] != nil, "reported zero ttftMs became absent")
require(zeros["tokensPerSecond"] != nil, "reported zero throughput became absent")
require(zeros["tokensOut"] != nil, "reported zero tokensOut became absent")

print("Swift metrics conversion fixtures passed")

package ai.xybrid

// Synthetic canonical records, like tests/swift/main.swift. The production
// encoder is compiled unchanged; no Bolt/JNI library is loaded by this fixture.
data class XybridStageLatency(
  val stageId: String,
  val latencyMs: UInt,
)

data class XybridInferenceMetrics(
  val totalMs: UInt,
  val ttftMs: UInt?,
  val tokensPerSecond: Float?,
  val prefillTps: Float?,
  val decodeTps: Float?,
  val tokensOut: UInt?,
  val stageLatenciesMs: List<XybridStageLatency>,
)

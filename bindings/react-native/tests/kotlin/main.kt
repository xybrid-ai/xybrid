package ai.xybrid.reactnative

import ai.xybrid.XybridInferenceMetrics
import ai.xybrid.XybridStageLatency
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap

private fun expectNumber(map: WritableMap, key: String, expected: Double) {
  val actual = map.values[key]
  check(actual is Double && actual == expected) {
    "$key: expected Double $expected, got $actual"
  }
}

private fun expectStages(map: WritableMap, expected: List<Pair<String, Double>>) {
  val stages = map.values["stageLatenciesMs"] as? WritableArray
    ?: error("stageLatenciesMs must be present as an array")
  val actual = stages.values.map { it?.values }
  val wanted = expected.map { (id, latency) -> mapOf("stageId" to id, "latencyMs" to latency) }
  check(actual == wanted) { "stageLatenciesMs: expected $wanted, got $actual" }
}

fun main() {
  val optionalKeys = setOf("ttftMs", "tokensPerSecond", "prefillTps", "decodeTps", "tokensOut")
  val requiredKeys = setOf("totalMs", "stageLatenciesMs")
  val metrics = XybridInferenceMetrics(
    totalMs = 120u,
    ttftMs = 18u,
    tokensPerSecond = 42.5f,
    prefillTps = 95.25f,
    decodeTps = 39.75f,
    tokensOut = 24u,
    stageLatenciesMs = listOf(
      XybridStageLatency("preprocess", 7u),
      XybridStageLatency("inference", 108u),
      XybridStageLatency("postprocess", 5u),
    ),
  )
  val populated = encodeInferenceMetrics(metrics)
  check(populated.values.keys == requiredKeys + optionalKeys) { "populated metrics keys changed" }
  expectNumber(populated, "totalMs", 120.0)
  expectNumber(populated, "ttftMs", 18.0)
  expectNumber(populated, "tokensPerSecond", 42.5)
  expectNumber(populated, "prefillTps", 95.25)
  expectNumber(populated, "decodeTps", 39.75)
  expectNumber(populated, "tokensOut", 24.0)
  expectStages(populated, listOf("preprocess" to 7.0, "inference" to 108.0, "postprocess" to 5.0))

  val noLlmMetrics = XybridInferenceMetrics(33u, null, null, null, null, null, emptyList())
  val absent = encodeInferenceMetrics(noLlmMetrics)
  check(absent.values.keys == requiredKeys) { "unreported LLM fields must be absent, not null or zero" }
  expectNumber(absent, "totalMs", 33.0)
  expectStages(absent, emptyList())

  val zeros = encodeInferenceMetrics(XybridInferenceMetrics(
    totalMs = 0u,
    ttftMs = 0u,
    tokensPerSecond = 0f,
    prefillTps = 0f,
    decodeTps = 0f,
    tokensOut = 0u,
    stageLatenciesMs = listOf(XybridStageLatency("zero", 0u)),
  ))
  check(zeros.values.keys == requiredKeys + optionalKeys) { "reported zero fields must remain present" }
  for (key in optionalKeys + "totalMs") expectNumber(zeros, key, 0.0)
  expectStages(zeros, listOf("zero" to 0.0))

  // Each optional must be gated by its own presence, not by another field.
  val independentFields = listOf(
    Triple("ttftMs", noLlmMetrics.copy(ttftMs = 18u), 18.0),
    Triple("tokensPerSecond", noLlmMetrics.copy(tokensPerSecond = 42.5f), 42.5),
    Triple("prefillTps", noLlmMetrics.copy(prefillTps = 95.25f), 95.25),
    Triple("decodeTps", noLlmMetrics.copy(decodeTps = 39.75f), 39.75),
    Triple("tokensOut", noLlmMetrics.copy(tokensOut = 24u), 24.0),
  )
  for ((key, input, expected) in independentFields) {
    val encoded = encodeInferenceMetrics(input)
    check(encoded.values.keys == requiredKeys + key) { "$key must be independent of other LLM fields" }
    expectNumber(encoded, "totalMs", 33.0)
    expectNumber(encoded, key, expected)
    expectStages(encoded, emptyList())
  }

  // u32 measurements must not overflow through an intermediate signed Int.
  val large = encodeInferenceMetrics(noLlmMetrics.copy(
    totalMs = UInt.MAX_VALUE,
    ttftMs = UInt.MAX_VALUE,
    tokensOut = UInt.MAX_VALUE,
    stageLatenciesMs = listOf(XybridStageLatency("large", UInt.MAX_VALUE)),
  ))
  check(large.values.keys == requiredKeys + setOf("ttftMs", "tokensOut"))
  for (key in listOf("totalMs", "ttftMs", "tokensOut")) expectNumber(large, key, 4294967295.0)
  expectStages(large, listOf("large" to 4294967295.0))

  println("Kotlin metrics conversion fixtures passed (9 cases)")
}

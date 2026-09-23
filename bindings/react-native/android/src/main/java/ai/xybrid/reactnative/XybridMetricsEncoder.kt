package ai.xybrid.reactnative

import ai.xybrid.XybridInferenceMetrics
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.WritableMap

// Shared by batch and streaming results. Keep the converter separate from the
// Android module so its actual implementation can run in the JVM fixture.
internal fun encodeInferenceMetrics(m: XybridInferenceMetrics): WritableMap {
  val out = Arguments.createMap()
  out.putDouble("totalMs", m.totalMs.toDouble())
  m.ttftMs?.let { out.putDouble("ttftMs", it.toDouble()) }
  m.tokensPerSecond?.let { out.putDouble("tokensPerSecond", it.toDouble()) }
  m.prefillTps?.let { out.putDouble("prefillTps", it.toDouble()) }
  m.decodeTps?.let { out.putDouble("decodeTps", it.toDouble()) }
  m.tokensOut?.let { out.putDouble("tokensOut", it.toDouble()) }
  val stages = Arguments.createArray()
  m.stageLatenciesMs.forEach { stage ->
    val encoded = Arguments.createMap()
    encoded.putString("stageId", stage.stageId)
    encoded.putDouble("latencyMs", stage.latencyMs.toDouble())
    stages.pushMap(encoded)
  }
  out.putArray("stageLatenciesMs", stages)
  return out
}

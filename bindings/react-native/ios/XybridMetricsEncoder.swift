import Foundation

/// Convert the canonical Bolt metrics record into React Native's plain-object
/// representation. Iterating `stageLatenciesMs` preserves native stage order,
/// and optional LLM measurements are inserted only when the backend reports
/// them.
func encodeInferenceMetrics(_ metrics: XybridInferenceMetrics) -> [String: Any] {
  var encoded: [String: Any] = [
    "totalMs": metrics.totalMs,
    "stageLatenciesMs": metrics.stageLatenciesMs.map {
      ["stageId": $0.stageId, "latencyMs": $0.latencyMs]
    },
  ]
  if let ttft = metrics.ttftMs { encoded["ttftMs"] = ttft }
  if let rate = metrics.tokensPerSecond { encoded["tokensPerSecond"] = rate }
  if let rate = metrics.prefillTps { encoded["prefillTps"] = rate }
  if let rate = metrics.decodeTps { encoded["decodeTps"] = rate }
  if let count = metrics.tokensOut { encoded["tokensOut"] = count }
  return encoded
}

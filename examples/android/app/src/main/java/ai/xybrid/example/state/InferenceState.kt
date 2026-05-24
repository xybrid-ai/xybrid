package ai.xybrid.example.state

import ai.xybrid.XybridInferenceMetrics
import ai.xybrid.example.data.ModelTask

/**
 * State for inference operations
 */
sealed class InferenceState {
    object Idle : InferenceState()
    object Running : InferenceState()
    data class Completed(
        val task: ModelTask,
        val text: String?,
        val audioBytes: ByteArray?,
        val latencyMs: Long,
        val metrics: XybridInferenceMetrics? = null
    ) : InferenceState()
    data class Error(val message: String) : InferenceState()
}

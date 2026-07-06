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
        /** Chain-of-thought from a thinking model (`reasoning: true`), or null. */
        val reasoningContent: String? = null,
        val audioBytes: ByteArray?,
        val latencyMs: Long,
        val metrics: XybridInferenceMetrics? = null
    ) : InferenceState()
    data class Error(val message: String) : InferenceState()
}

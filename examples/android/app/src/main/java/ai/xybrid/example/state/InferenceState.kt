package ai.xybrid.example.state

/**
 * State for inference operations
 */
sealed class InferenceState {
    object Idle : InferenceState()
    object Running : InferenceState()
    data class Completed(val text: String?, val audioSize: Int?, val latencyMs: Long) : InferenceState()
    data class Error(val message: String) : InferenceState()
}

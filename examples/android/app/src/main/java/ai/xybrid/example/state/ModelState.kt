package ai.xybrid.example.state

import ai.xybrid.XybridModel

/**
 * State for model operations
 */
sealed class ModelState {
    object NotLoaded : ModelState()
    object Loading : ModelState()
    data class Loaded(val model: XybridModel) : ModelState()
    data class Error(val message: String) : ModelState()
}

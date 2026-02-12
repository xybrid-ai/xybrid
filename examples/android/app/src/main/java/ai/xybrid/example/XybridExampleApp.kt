package ai.xybrid.example

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// Xybrid SDK imports
import ai.xybrid.XybridModelLoader
import ai.xybrid.XybridException
import ai.xybrid.Envelope
import ai.xybrid.displayMessage

// State and component imports
import ai.xybrid.example.data.CatalogModel
import ai.xybrid.example.state.ModelState
import ai.xybrid.example.state.InferenceState
import ai.xybrid.example.components.ModelLoadingCard
import ai.xybrid.example.components.InferenceCard
import ai.xybrid.example.components.AboutCard

/**
 * Main Xybrid Example App composable.
 * Demonstrates registry-based model loading and inference.
 */
@Composable
fun XybridExampleApp() {
    var modelState by remember { mutableStateOf<ModelState>(ModelState.NotLoaded) }
    var inferenceState by remember { mutableStateOf<InferenceState>(InferenceState.Idle) }
    var selectedModel by remember { mutableStateOf<CatalogModel?>(null) }
    var inputText by remember { mutableStateOf("") }

    val coroutineScope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(scrollState),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Title
        Text(
            text = "Xybrid SDK Example",
            style = MaterialTheme.typography.headlineMedium
        )

        // Step 1: Select & Load Model
        ModelLoadingCard(
            modelState = modelState,
            selectedModel = selectedModel,
            onModelSelected = { model ->
                selectedModel = model
                inputText = model.defaultInput
            },
            onLoadModel = {
                val model = selectedModel ?: return@ModelLoadingCard
                modelState = ModelState.Loading
                coroutineScope.launch {
                    try {
                        val loaded = withContext(Dispatchers.IO) {
                            val loader = XybridModelLoader.fromRegistry(model.id)
                            loader.load()
                        }
                        modelState = ModelState.Loaded(loaded)
                    } catch (e: XybridException) {
                        modelState = ModelState.Error(e.displayMessage)
                    } catch (e: Exception) {
                        modelState = ModelState.Error(
                            e.message ?: "Failed to load model"
                        )
                    }
                }
            },
            onUnloadModel = {
                modelState = ModelState.NotLoaded
                inferenceState = InferenceState.Idle
            },
            onRetry = { modelState = ModelState.NotLoaded }
        )

        // Step 2: Run Inference
        InferenceCard(
            inferenceState = inferenceState,
            modelState = modelState,
            inputText = inputText,
            onInputTextChange = { inputText = it },
            onRunInference = {
                val model = (modelState as? ModelState.Loaded)?.model ?: return@InferenceCard
                inferenceState = InferenceState.Running
                coroutineScope.launch {
                    try {
                        val result = withContext(Dispatchers.IO) {
                            val envelope = Envelope.text(inputText)
                            model.run(envelope)
                        }

                        if (result.success) {
                            inferenceState = InferenceState.Completed(
                                text = result.text,
                                audioSize = result.audioBytes?.size,
                                latencyMs = result.latencyMs.toLong()
                            )
                        } else {
                            inferenceState = InferenceState.Error(
                                result.toString()
                            )
                        }
                    } catch (e: XybridException) {
                        inferenceState = InferenceState.Error(e.displayMessage)
                    } catch (e: Exception) {
                        inferenceState = InferenceState.Error(
                            e.message ?: "Inference failed"
                        )
                    }
                }
            },
            onRetry = { inferenceState = InferenceState.Idle }
        )

        // Info section
        AboutCard()
    }
}

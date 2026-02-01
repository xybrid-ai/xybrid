package ai.xybrid.example

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// Xybrid SDK imports
// Note: These will work once the native libraries are built.
// See README.md for build instructions.
// import ai.xybrid.XybridModelLoader
// import ai.xybrid.XybridEnvelope
// import ai.xybrid.XybridModel
// import ai.xybrid.XybridResult
// import ai.xybrid.XybridException
// import ai.xybrid.Envelope
// import ai.xybrid.displayMessage

/**
 * Application state for SDK initialization
 */
sealed class AppState {
    object NotInitialized : AppState()
    object Initializing : AppState()
    object Ready : AppState()
    data class Error(val message: String) : AppState()
}

/**
 * State for model operations
 */
sealed class ModelState {
    object NotLoaded : ModelState()
    object Loading : ModelState()
    data class Loaded(val modelId: String) : ModelState()
    data class Error(val message: String) : ModelState()
}

/**
 * State for inference operations
 */
sealed class InferenceState {
    object Idle : InferenceState()
    object Running : InferenceState()
    data class Completed(val text: String?, val latencyMs: Long) : InferenceState()
    data class Error(val message: String) : InferenceState()
}

/**
 * Main Xybrid Example App composable.
 * Demonstrates SDK initialization, model loading, and inference.
 */
@Composable
fun XybridExampleApp() {
    var appState by remember { mutableStateOf<AppState>(AppState.NotInitialized) }
    var modelState by remember { mutableStateOf<ModelState>(ModelState.NotLoaded) }
    var inferenceState by remember { mutableStateOf<InferenceState>(InferenceState.Idle) }

    var modelId by remember { mutableStateOf("kokoro-82m") }
    var inputText by remember { mutableStateOf("Hello, welcome to Xybrid!") }

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

        // Step 1: SDK Initialization
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Step 1: Initialize SDK",
                    style = MaterialTheme.typography.titleMedium
                )

                when (val state = appState) {
                    is AppState.NotInitialized -> {
                        Button(
                            onClick = {
                                appState = AppState.Initializing
                                coroutineScope.launch {
                                    try {
                                        // Simulated SDK initialization
                                        // In real app: Xybrid.init()
                                        withContext(Dispatchers.IO) {
                                            // Simulate network/init delay
                                            kotlinx.coroutines.delay(500)
                                        }
                                        appState = AppState.Ready
                                    } catch (e: Exception) {
                                        appState = AppState.Error(e.message ?: "Unknown error")
                                    }
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Initialize SDK")
                        }
                    }
                    is AppState.Initializing -> {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(24.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Initializing...")
                        }
                    }
                    is AppState.Ready -> {
                        Text(
                            text = "SDK Ready",
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    is AppState.Error -> {
                        Text(
                            text = "Error: ${state.message}",
                            color = MaterialTheme.colorScheme.error
                        )
                        Button(
                            onClick = { appState = AppState.NotInitialized },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Retry")
                        }
                    }
                }
            }
        }

        // Step 2: Model Loading
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Step 2: Load Model",
                    style = MaterialTheme.typography.titleMedium
                )

                OutlinedTextField(
                    value = modelId,
                    onValueChange = { modelId = it },
                    label = { Text("Model ID") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = appState is AppState.Ready && modelState !is ModelState.Loading,
                    singleLine = true
                )

                when (val state = modelState) {
                    is ModelState.NotLoaded -> {
                        Button(
                            onClick = {
                                modelState = ModelState.Loading
                                coroutineScope.launch {
                                    try {
                                        // Simulated model loading
                                        // In real app:
                                        // val loader = XybridModelLoader.fromRegistry(modelId)
                                        // val model = loader.load()
                                        withContext(Dispatchers.IO) {
                                            kotlinx.coroutines.delay(1000)
                                        }
                                        modelState = ModelState.Loaded(modelId)
                                    } catch (e: Exception) {
                                        modelState = ModelState.Error(
                                            e.message ?: "Failed to load model"
                                        )
                                    }
                                }
                            },
                            enabled = appState is AppState.Ready,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Load Model")
                        }
                    }
                    is ModelState.Loading -> {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(24.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Loading model...")
                        }
                    }
                    is ModelState.Loaded -> {
                        Text(
                            text = "Model loaded: ${state.modelId}",
                            color = MaterialTheme.colorScheme.primary
                        )
                        OutlinedButton(
                            onClick = {
                                modelState = ModelState.NotLoaded
                                inferenceState = InferenceState.Idle
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Unload Model")
                        }
                    }
                    is ModelState.Error -> {
                        Text(
                            text = "Error: ${state.message}",
                            color = MaterialTheme.colorScheme.error
                        )
                        Button(
                            onClick = { modelState = ModelState.NotLoaded },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Retry")
                        }
                    }
                }
            }
        }

        // Step 3: Run Inference
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Step 3: Run Inference (TTS)",
                    style = MaterialTheme.typography.titleMedium
                )

                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    label = { Text("Text to synthesize") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = modelState is ModelState.Loaded && inferenceState !is InferenceState.Running,
                    minLines = 2,
                    maxLines = 4
                )

                when (val state = inferenceState) {
                    is InferenceState.Idle -> {
                        Button(
                            onClick = {
                                inferenceState = InferenceState.Running
                                coroutineScope.launch {
                                    try {
                                        val startTime = System.currentTimeMillis()
                                        // Simulated inference
                                        // In real app:
                                        // val envelope = Envelope.text(inputText)
                                        // val result = model.run(envelope)
                                        // val audioBytes = result.audioBytes
                                        withContext(Dispatchers.IO) {
                                            kotlinx.coroutines.delay(800)
                                        }
                                        val latency = System.currentTimeMillis() - startTime
                                        inferenceState = InferenceState.Completed(
                                            text = "Audio generated (${inputText.length} chars)",
                                            latencyMs = latency
                                        )
                                    } catch (e: Exception) {
                                        inferenceState = InferenceState.Error(
                                            e.message ?: "Inference failed"
                                        )
                                    }
                                }
                            },
                            enabled = modelState is ModelState.Loaded && inputText.isNotBlank(),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Run Inference")
                        }
                    }
                    is InferenceState.Running -> {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(24.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Running inference...")
                        }
                    }
                    is InferenceState.Completed -> {
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.primaryContainer
                            )
                        ) {
                            Column(
                                modifier = Modifier.padding(12.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "Result",
                                    style = MaterialTheme.typography.labelMedium
                                )
                                state.text?.let {
                                    Text(text = it)
                                }
                                Text(
                                    text = "Latency: ${state.latencyMs} ms",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            OutlinedButton(
                                onClick = { inferenceState = InferenceState.Idle },
                                modifier = Modifier.weight(1f)
                            ) {
                                Text("Reset")
                            }
                            Button(
                                onClick = {
                                    // In real app: play audio using MediaPlayer or ExoPlayer
                                    // val audioBytes = result.audioBytes
                                    // mediaPlayer.setDataSource(...)
                                },
                                modifier = Modifier.weight(1f)
                            ) {
                                Text("Play Audio")
                            }
                        }
                    }
                    is InferenceState.Error -> {
                        Text(
                            text = "Error: ${state.message}",
                            color = MaterialTheme.colorScheme.error
                        )
                        Button(
                            onClick = { inferenceState = InferenceState.Idle },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Retry")
                        }
                    }
                }
            }
        }

        // Info section
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "About This Example",
                    style = MaterialTheme.typography.titleSmall
                )
                Text(
                    text = "This app demonstrates the Xybrid SDK integration pattern for Android. " +
                           "It shows SDK initialization, model loading from registry, and running TTS inference.",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = "See README.md for native library build instructions.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

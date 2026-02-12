package ai.xybrid.example.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ai.xybrid.example.state.InferenceState
import ai.xybrid.example.state.ModelState

@Composable
fun InferenceCard(
    inferenceState: InferenceState,
    modelState: ModelState,
    inputText: String,
    onInputTextChange: (String) -> Unit,
    onRunInference: () -> Unit,
    onRetry: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Step 2: Run Inference",
                style = MaterialTheme.typography.titleMedium
            )

            OutlinedTextField(
                value = inputText,
                onValueChange = onInputTextChange,
                label = { Text("Text to synthesize") },
                modifier = Modifier.fillMaxWidth(),
                enabled = modelState is ModelState.Loaded && inferenceState !is InferenceState.Running,
                minLines = 2,
                maxLines = 4
            )

            when (val state = inferenceState) {
                is InferenceState.Idle -> {
                    Button(
                        onClick = onRunInference,
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
                            state.audioSize?.let { size ->
                                Text(
                                    text = "Audio output: ${size / 1024} KB",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                            Text(
                                text = "Latency: ${state.latencyMs} ms",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }

                    Button(
                        onClick = onRetry,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Run Again")
                    }
                }
                is InferenceState.Error -> {
                    Text(
                        text = "Error: ${state.message}",
                        color = MaterialTheme.colorScheme.error
                    )
                    Button(
                        onClick = onRetry,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Retry")
                    }
                }
            }
        }
    }
}

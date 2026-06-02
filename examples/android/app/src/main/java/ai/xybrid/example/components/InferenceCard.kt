package ai.xybrid.example.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import ai.xybrid.example.audio.PcmPlayer
import ai.xybrid.example.data.CatalogModel
import ai.xybrid.example.data.ModelTask
import ai.xybrid.example.data.acceptsTextInput
import ai.xybrid.example.data.inputLabel
import ai.xybrid.example.state.InferenceState
import ai.xybrid.example.state.ModelState
import androidx.compose.foundation.text.selection.SelectionContainer

@Composable
fun InferenceCard(
    inferenceState: InferenceState,
    modelState: ModelState,
    selectedModel: CatalogModel?,
    inputText: String,
    selectedVoiceId: String?,
    pcmPlayer: PcmPlayer,
    recordedAudio: ByteArray?,
    isRecording: Boolean,
    hasAudioPermission: Boolean,
    onInputTextChange: (String) -> Unit,
    onRequestAudioPermission: () -> Unit,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit,
    onClearRecording: () -> Unit,
    onRunInference: () -> Unit,
    onRetry: () -> Unit
) {
    val task = selectedModel?.task

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

            // Task-specific input area
            if (task != null && task.acceptsTextInput()) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = onInputTextChange,
                    label = { Text(task.inputLabel()) },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = modelState is ModelState.Loaded && inferenceState !is InferenceState.Running,
                    minLines = if (task == ModelTask.LLM) 3 else 2,
                    maxLines = if (task == ModelTask.LLM) 6 else 4
                )

                // Show voice hint for TTS
                if (task == ModelTask.TTS && selectedVoiceId != null) {
                    Text(
                        text = "Voice: $selectedVoiceId",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            } else if (task == ModelTask.ASR) {
                AudioRecordingInput(
                    recordedAudio = recordedAudio,
                    isRecording = isRecording,
                    hasPermission = hasAudioPermission,
                    enabled = modelState is ModelState.Loaded && inferenceState !is InferenceState.Running,
                    onRequestPermission = onRequestAudioPermission,
                    onStartRecording = onStartRecording,
                    onStopRecording = onStopRecording,
                    onClearRecording = onClearRecording
                )
            }

            when (val state = inferenceState) {
                is InferenceState.Idle -> {
                    val canRun = modelState is ModelState.Loaded
                            && task != null
                            && when (task) {
                                ModelTask.TTS, ModelTask.LLM -> inputText.isNotBlank()
                                ModelTask.ASR -> recordedAudio != null && recordedAudio.isNotEmpty()
                            }
                    Button(
                        onClick = onRunInference,
                        enabled = canRun,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            when (task) {
                                ModelTask.TTS -> "Synthesize Speech"
                                ModelTask.LLM -> "Generate"
                                ModelTask.ASR -> "Transcribe"
                                null -> "Run Inference"
                            }
                        )
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
                        Text(
                            when (task) {
                                ModelTask.TTS -> "Synthesizing..."
                                ModelTask.LLM -> "Generating..."
                                ModelTask.ASR -> "Transcribing..."
                                null -> "Running..."
                            }
                        )
                    }
                }
                is InferenceState.Completed -> {
                    CompletedResult(state, pcmPlayer)

                    Button(
                        onClick = {
                            pcmPlayer.stop()
                            onRetry()
                        },
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

@Composable
private fun AudioRecordingInput(
    recordedAudio: ByteArray?,
    isRecording: Boolean,
    hasPermission: Boolean,
    enabled: Boolean,
    onRequestPermission: () -> Unit,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit,
    onClearRecording: () -> Unit
) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.small,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Audio Input",
                style = MaterialTheme.typography.labelMedium
            )

            if (!hasPermission) {
                Text(
                    text = "Microphone permission is required for speech recognition.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Button(
                    onClick = onRequestPermission,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Grant Microphone Permission")
                }
            } else if (isRecording) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.error
                    )
                    Text(
                        text = "Recording...",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
                Button(
                    onClick = onStopRecording,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Stop Recording")
                }
            } else if (recordedAudio != null && recordedAudio.isNotEmpty()) {
                val durationSec = recordedAudio.size.toFloat() / 2 / 16000
                val sizeKb = recordedAudio.size / 1024
                Text(
                    text = "Recorded: ${"%.1f".format(durationSec)}s  |  ${sizeKb} KB  |  16kHz mono",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = onClearRecording,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Discard")
                    }
                    Button(
                        onClick = onStartRecording,
                        enabled = enabled,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Re-record")
                    }
                }
            } else {
                Text(
                    text = "Tap below to record audio for transcription.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Button(
                    onClick = onStartRecording,
                    enabled = enabled,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Start Recording")
                }
            }
        }
    }
}

@Composable
private fun CompletedResult(state: InferenceState.Completed, pcmPlayer: PcmPlayer) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Result",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold
                )
                TaskBadge(state.task)
            }

            when (state.task) {
                ModelTask.TTS -> TtsResult(state, pcmPlayer)
                ModelTask.LLM -> LlmResult(state)
                ModelTask.ASR -> AsrResult(state)
            }

            // Latency always shown
            Text(
                text = "Latency: ${state.latencyMs} ms",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
            )

            // Typed metrics section — populated from result.metrics.
            // LLM-specific fields are null for TTS/ASR; stage latencies are
            // empty for single-model runs.
            state.metrics?.let { MetricsSection(it) }
        }
    }
}

@Composable
private fun MetricsSection(metrics: ai.xybrid.XybridInferenceMetrics) {
    val rows = buildList<Pair<String, String>> {
        metrics.ttftMs?.let { add("TTFT" to "$it ms") }
        metrics.tokensPerSecond?.let { add("Throughput" to "%.1f tok/s".format(it)) }
        metrics.prefillTps?.let { add("Prefill" to "%.1f tok/s".format(it)) }
        metrics.decodeTps?.let { add("Decode" to "%.1f tok/s".format(it)) }
        metrics.tokensOut?.let { add("Tokens out" to it.toString()) }
        if (metrics.stageLatenciesMs.isNotEmpty()) {
            val stages = metrics.stageLatenciesMs.joinToString(", ") { "${it.stageId}=${it.latencyMs}ms" }
            add("Stages" to stages)
        }
    }
    if (rows.isEmpty()) return
    Spacer(modifier = Modifier.height(4.dp))
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.small,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(8.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            Text(
                text = "Metrics",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold
            )
            rows.forEach { (label, value) ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(text = label, style = MaterialTheme.typography.bodySmall)
                    Text(
                        text = value,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

@Composable
private fun TtsResult(state: InferenceState.Completed, pcmPlayer: PcmPlayer) {
    val audioBytes = state.audioBytes
    if (audioBytes != null && audioBytes.isNotEmpty()) {
        var isPlaying by remember { mutableStateOf(false) }
        val durationSec = pcmPlayer.estimateDurationSec(audioBytes)
        val sizeKb = audioBytes.size / 1024

        Surface(
            color = MaterialTheme.colorScheme.tertiaryContainer,
            shape = MaterialTheme.shapes.small,
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Audio generated",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                        Text(
                            text = "${sizeKb} KB  |  ${"%.1f".format(durationSec)}s  |  24kHz mono",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                    }
                }

                // Play / Stop button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (isPlaying) {
                        OutlinedButton(
                            onClick = {
                                pcmPlayer.stop()
                                isPlaying = false
                            },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Stop")
                        }
                    } else {
                        Button(
                            onClick = {
                                isPlaying = true
                                pcmPlayer.play(audioBytes) {
                                    isPlaying = false
                                }
                            },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Play Audio")
                        }
                    }
                }
            }
        }
    } else {
        Text(
            text = "No audio output received",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.5f)
        )
    }

    // TTS can also return text (e.g., phonemes) as debug info
    state.text?.let {
        if (it.isNotBlank()) {
            Text(
                text = it,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onPrimaryContainer
            )
        }
    }
}

@Composable
private fun LlmResult(state: InferenceState.Completed) {
    val clipboardManager = LocalClipboardManager.current

    state.text?.let { text ->
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            SelectionContainer {
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
            TextButton(
                onClick = { clipboardManager.setText(AnnotatedString(text)) },
                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
            ) {
                Text(
                    text = "Copy to clipboard",
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }
    } ?: Text(
        text = "No text output",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.5f)
    )
}

@Composable
private fun AsrResult(state: InferenceState.Completed) {
    val clipboardManager = LocalClipboardManager.current

    state.text?.let { text ->
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            SelectionContainer {
                Text(
                    text = "\"$text\"",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
            TextButton(
                onClick = { clipboardManager.setText(AnnotatedString(text)) },
                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
            ) {
                Text(
                    text = "Copy to clipboard",
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }
    } ?: Text(
        text = "No transcription output",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.5f)
    )
}

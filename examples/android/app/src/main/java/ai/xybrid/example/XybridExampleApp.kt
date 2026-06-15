package ai.xybrid.example

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// Xybrid SDK imports
import ai.xybrid.XybridModel
import ai.xybrid.XybridError
import ai.xybrid.Envelope
import ai.xybrid.displayMessage
// XybridResult compatibility extensions — bolt's struct exposes
// envelope.kind rather than the flat `success`/`text`/`audioBytes`
// fields uniffi emitted; the extensions in Xybrid.kt restore that shape.
import ai.xybrid.success
import ai.xybrid.text
import ai.xybrid.audioBytes

// State and component imports
import ai.xybrid.example.audio.AudioRecorder
import ai.xybrid.example.audio.PcmPlayer
import ai.xybrid.example.data.CatalogModel
import ai.xybrid.example.data.ModelTask
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
    var selectedVoiceId by remember { mutableStateOf<String?>(null) }
    var recordedAudio by remember { mutableStateOf<ByteArray?>(null) }
    var isRecording by remember { mutableStateOf(false) }

    val pcmPlayer = remember { PcmPlayer() }
    val audioRecorder = remember { AudioRecorder() }
    DisposableEffect(Unit) {
        onDispose {
            pcmPlayer.release()
            audioRecorder.release()
        }
    }

    val context = LocalContext.current
    var hasAudioPermission by remember {
        mutableStateOf(audioRecorder.hasPermission(context))
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasAudioPermission = granted }

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
                selectedVoiceId = null
                recordedAudio = null
            },
            onLoadModel = {
                val model = selectedModel ?: return@ModelLoadingCard
                modelState = ModelState.Loading
                coroutineScope.launch {
                    try {
                        val loaded = withContext(Dispatchers.IO) {
                            // Bolt collapsed the loader-then-load 2-step into a
                            // single XybridModel constructor. (The Kotlin
                            // emitter exposes from_registry as the primary
                            // secondary constructor; from_directory /
                            // from_bundle / from_huggingface remain on the
                            // companion object.)
                            XybridModel(model.id)
                        }
                        modelState = ModelState.Loaded(loaded)

                        // Pick default voice for TTS models. Bolt returns the
                        // full VoiceInfo; extract `.id` for the picker state.
                        if (model.task == ModelTask.TTS) {
                            selectedVoiceId = loaded.defaultVoice()?.id
                        }
                    } catch (e: XybridError) {
                        modelState = ModelState.Error(e.displayMessage)
                    } catch (e: Exception) {
                        modelState = ModelState.Error(
                            e.message ?: "Failed to load model"
                        )
                    }
                }
            },
            onUnloadModel = {
                pcmPlayer.stop()
                modelState = ModelState.NotLoaded
                inferenceState = InferenceState.Idle
                selectedVoiceId = null
                recordedAudio = null
            },
            onRetry = { modelState = ModelState.NotLoaded }
        )

        // Step 2: Run Inference
        InferenceCard(
            inferenceState = inferenceState,
            modelState = modelState,
            selectedModel = selectedModel,
            inputText = inputText,
            selectedVoiceId = selectedVoiceId,
            pcmPlayer = pcmPlayer,
            recordedAudio = recordedAudio,
            isRecording = isRecording,
            hasAudioPermission = hasAudioPermission,
            onInputTextChange = { inputText = it },
            onRequestAudioPermission = {
                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            },
            onStartRecording = {
                recordedAudio = null
                if (audioRecorder.start(context)) {
                    isRecording = true
                }
            },
            onStopRecording = {
                isRecording = false
                recordedAudio = audioRecorder.stop()
            },
            onClearRecording = {
                isRecording = false
                audioRecorder.release()
                recordedAudio = null
            },
            onRunInference = {
                val model = (modelState as? ModelState.Loaded)?.model ?: return@InferenceCard
                val task = selectedModel?.task ?: return@InferenceCard
                inferenceState = InferenceState.Running
                coroutineScope.launch {
                    try {
                        val result = withContext(Dispatchers.IO) {
                            val envelope = when (task) {
                                ModelTask.TTS -> {
                                    val voiceId = selectedVoiceId
                                    if (voiceId != null) {
                                        Envelope.text(inputText, voiceId)
                                    } else {
                                        Envelope.text(inputText)
                                    }
                                }
                                ModelTask.LLM -> Envelope.text(inputText)
                                ModelTask.ASR -> {
                                    val audio = recordedAudio
                                        ?: error("No recorded audio available")
                                    Envelope.audio(audio, 16000u, 1u)
                                }
                            }
                            // Bolt's `run` is single-arg (no separate config
                            // parameter in the current scaffold); the optional
                            // GenerationConfig is folded into the envelope or a
                            // future `runWithOptions` overload.
                            model.run(envelope)
                        }

                        if (result.success) {
                            inferenceState = InferenceState.Completed(
                                task = task,
                                text = result.text,
                                audioBytes = result.audioBytes,
                                latencyMs = result.latencyMs.toLong(),
                                metrics = result.metrics
                            )
                        } else {
                            inferenceState = InferenceState.Error(
                                result.text ?: "Inference returned unsuccessful result"
                            )
                        }
                    } catch (e: XybridError) {
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

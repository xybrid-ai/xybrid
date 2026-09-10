package ai.xybrid.example.data

enum class ModelTask(val label: String) {
    TTS("Text-to-Speech"),
    ASR("Speech Recognition"),
    LLM("Text Generation");
}

data class CatalogModel(
    val id: String,
    val displayName: String,
    val task: ModelTask,
    val description: String,
    val defaultInput: String,
    val parameterCount: String? = null
)

/** Input field label for a given task. */
fun ModelTask.inputLabel(): String = when (this) {
    ModelTask.TTS -> "Text to synthesize"
    ModelTask.ASR -> "Audio input required"
    ModelTask.LLM -> "Prompt"
}

/** Whether the task accepts text input from the user. */
fun ModelTask.acceptsTextInput(): Boolean = when (this) {
    ModelTask.TTS, ModelTask.LLM -> true
    ModelTask.ASR -> false
}

val MODEL_CATALOG = listOf(
    CatalogModel(
        id = "gemma-3-1b",
        displayName = "Gemma 3 1B",
        task = ModelTask.LLM,
        description = "Google's compact language model for text generation",
        defaultInput = "Hello, welcome to Xybrid!",
        parameterCount = "1B"
    ),
    CatalogModel(
        id = "lfm2.5-350m",
        displayName = "LFM2.5 350M",
        task = ModelTask.LLM,
        description = "Liquid AI's hybrid conv+attention LLM, 9 languages, tool calling",
        defaultInput = "Hello, welcome to Xybrid!",
        parameterCount = "354M"
    ),
    CatalogModel(
        id = "lfm2.5-1.2b-thinking",
        displayName = "LFM2.5 1.2B Thinking",
        task = ModelTask.LLM,
        description = "Liquid AI reasoning model — emits a chain-of-thought before its answer. Try the result's \"Reasoning\" section.",
        defaultInput = "Is 97 a prime number? Reason step by step, then answer.",
        parameterCount = "1.2B"
    ),
    CatalogModel(
        id = "kokoro-82m",
        displayName = "Kokoro 82M",
        task = ModelTask.TTS,
        description = "High-quality English TTS with multiple voices",
        defaultInput = "Hello, welcome to Xybrid!",
        parameterCount = "82M"
    ),
    CatalogModel(
        id = "kitten-tts-nano-0.8",
        displayName = "KittenTTS Nano 0.8",
        task = ModelTask.TTS,
        description = "Lightweight English TTS model",
        defaultInput = "Hello, welcome to Xybrid!",
        parameterCount = "Nano"
    ),
    CatalogModel(
        id = "whisper-tiny-ggml",
        displayName = "Whisper Tiny",
        task = ModelTask.ASR,
        description = "OpenAI Whisper tiny model for speech-to-text",
        defaultInput = ""
    ),
    CatalogModel(
        id = "wav2vec2-base-960h",
        displayName = "Wav2Vec2 Base",
        task = ModelTask.ASR,
        description = "Facebook Wav2Vec2 trained on 960h LibriSpeech",
        defaultInput = "",
        parameterCount = "95M"
    )
)

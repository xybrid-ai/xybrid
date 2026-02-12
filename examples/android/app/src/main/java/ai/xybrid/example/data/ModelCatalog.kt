package ai.xybrid.example.data

data class CatalogModel(
    val id: String,
    val displayName: String,
    val task: String,
    val description: String,
    val defaultInput: String
)

val MODEL_CATALOG = listOf(
    CatalogModel(
        id = "gemma-3-1b",
        displayName = "Gemma 3 1B",
        task = "LLM",
        description = "High-quality English TTS model (82M parameters)",
        defaultInput = "Hello, welcome to Xybrid!"
    ),
    CatalogModel(
        id = "kokoro-82m",
        displayName = "Kokoro 82M",
        task = "Text-to-Speech",
        description = "High-quality English TTS model (82M parameters)",
        defaultInput = "Hello, welcome to Xybrid!"
    ),
    CatalogModel(
        id = "whisper-tiny",
        displayName = "Whisper Tiny",
        task = "Speech Recognition",
        description = "OpenAI Whisper tiny model for speech-to-text",
        defaultInput = "Provide audio input for transcription"
    ),
    CatalogModel(
        id = "wav2vec2-base-960h",
        displayName = "Wav2Vec2 Base",
        task = "Speech Recognition",
        description = "Facebook Wav2Vec2 base model trained on 960h LibriSpeech",
        defaultInput = "Provide audio input for transcription"
    )
)

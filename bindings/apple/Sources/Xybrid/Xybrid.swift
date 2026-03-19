//
//  Xybrid.swift
//  Xybrid SDK for iOS/macOS
//
//  Convenience wrappers and extensions for the UniFFI-generated bindings.
//  For full API documentation, see https://docs.xybrid.dev/sdk/swift
//

import Foundation

// MARK: - Public Type Re-exports

/// Loads ML models from the registry or local bundles.
/// Use `fromRegistry(modelId:)` for cloud models or `fromBundle(path:)` for local models.
public typealias ModelLoader = XybridModelLoader

/// A loaded model ready for inference.
/// Call `run(envelope:config:)` to execute inference on input data.
public typealias Model = XybridModel

/// Input data for model inference.
/// Use `.audio(bytes:sampleRate:channels:)`, `.text(text:voiceId:speed:)`, or `.embedding(data:)`.
public typealias Envelope = XybridEnvelope

/// The result of a model inference operation.
/// Check `success` and access output via `text`, `audioBytes`, or `embedding` properties.
public typealias Result = XybridResult

/// Errors that can occur during model loading or inference.
public typealias XybridSDKError = XybridError

/// Voice metadata for TTS models.
/// Describes a single voice available in a TTS model's voice catalog.
public typealias VoiceInfo = XybridVoiceInfo

/// Generation parameters for LLM inference (temperature, top_p, max_tokens, etc.).
public typealias GenerationConfig = XybridGenerationConfig

// MARK: - XybridResult Extensions

public extension XybridResult {
    /// Returns `true` if inference failed.
    var isFailure: Bool { !success }

    /// The latency as a `TimeInterval` in seconds.
    var latency: TimeInterval { TimeInterval(latencyMs) / 1000.0 }
}

// MARK: - XybridEnvelope Extensions

public extension XybridEnvelope {
    /// Creates an audio envelope from raw PCM data.
    /// - Parameters:
    ///   - pcmData: Raw PCM audio bytes
    ///   - sampleRate: Sample rate in Hz (e.g., 16000 for ASR)
    ///   - channels: Number of audio channels (typically 1 for mono)
    static func audio(pcmData: Data, sampleRate: UInt32 = 16000, channels: UInt32 = 1) -> XybridEnvelope {
        return .audio(bytes: pcmData, sampleRate: sampleRate, channels: channels)
    }

    /// Creates a text envelope for TTS with default voice.
    /// - Parameter content: The text to synthesize
    static func text(_ content: String) -> XybridEnvelope {
        return .text(text: content, voiceId: nil, speed: nil)
    }

    /// Creates a text envelope for TTS with voice and speed options.
    /// - Parameters:
    ///   - content: The text to synthesize
    ///   - voice: Voice ID (e.g., "af_heart" for Kokoro)
    ///   - speed: Speed multiplier (1.0 = normal, 0.5 = slower, 2.0 = faster)
    static func text(_ content: String, voice: String, speed: Double = 1.0) -> XybridEnvelope {
        return .text(text: content, voiceId: voice, speed: speed)
    }
}

// MARK: - XybridVoiceInfo Extensions

public extension XybridVoiceInfo {
    /// Returns `true` if the voice gender is male.
    var isMale: Bool { gender == "male" }

    /// Returns `true` if the voice gender is female.
    var isFemale: Bool { gender == "female" }
}

// MARK: - XybridError Extensions

extension XybridError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .ModelNotFound(let message):
            return "Model not found: \(message)"
        case .DirectoryNotFound(let message):
            return "Directory not found: \(message)"
        case .MetadataNotFound(let message):
            return "model_metadata.json not found: \(message)"
        case .MetadataInvalid(let message):
            return "model_metadata.json is invalid: \(message)"
        case .LoadError(let message):
            return "Load error: \(message)"
        case .InferenceError(let message):
            return "Inference failed: \(message)"
        case .StreamingNotSupported:
            return "Streaming is not supported by this model"
        case .NotLoaded:
            return "Model not loaded"
        case .ConfigError(let message):
            return "Invalid configuration: \(message)"
        case .NetworkError(let message):
            return "Network error: \(message)"
        case .IoError(let message):
            return "I/O error: \(message)"
        case .CacheError(let message):
            return "Cache error: \(message)"
        case .PipelineError(let message):
            return "Pipeline error: \(message)"
        case .CircuitOpen(let message):
            return "Circuit breaker open: \(message)"
        case .RateLimited(let retryAfterSecs):
            return "Rate limited, retry after \(retryAfterSecs) seconds"
        case .Timeout(let timeoutMs):
            return "Request timeout after \(timeoutMs)ms"
        }
    }
}

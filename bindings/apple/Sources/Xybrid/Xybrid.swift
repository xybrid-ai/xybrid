//
//  Xybrid.swift
//  Xybrid SDK for iOS/macOS
//
//  Hand-written wrappers + a compatibility shim that smooths over the
//  uniffi → bolt migration. The BoltFFI-generated bindings live in
//  `xybrid_bolt.swift` alongside this file; both compile into the same
//  Swift module so consumers `import Xybrid` and see one surface.
//
//  For full API documentation, see https://docs.xybrid.dev/sdk/swift
//

import Foundation
#if os(iOS)
import UIKit
#endif

// MARK: - SDK Initialization

/// Main entry point for the Xybrid SDK on iOS/macOS.
///
/// Call `Xybrid.initialize()` once before using any other Xybrid functionality.
/// It registers the binding identifier so registry calls are attributed to the
/// Swift SDK, and is safe to call multiple times — subsequent calls are no-ops.
///
/// Inference runs on-device whether or not you authenticate. Pass an `apiKey`
/// to start the telemetry exporter and see your runs on the dashboard — get a
/// free key at https://dashboard.xybrid.dev.
///
/// ```swift
/// // Anonymous — local inference, telemetry disabled
/// Xybrid.initialize()
///
/// // Authenticated — telemetry flows to the dashboard
/// Xybrid.initialize(apiKey: ProcessInfo.processInfo.environment["XYBRID_API_KEY"])
/// ```
public enum Xybrid {
    private static let initLock = NSLock()
    nonisolated(unsafe) private static var initialized = false
    #if os(iOS)
    // Retained so the NotificationCenter block-based observer keeps
    // firing for the process lifetime. NotificationCenter holds the
    // observer weakly through this token; losing the reference would
    // silently drop battery updates.
    nonisolated(unsafe) private static var batteryObserver: NSObjectProtocol?
    #endif

    /// Initialize the Xybrid runtime.
    ///
    /// Registers the Swift binding identifier with the SDK so that the
    /// `X-Xybrid-Client` header on registry HTTP calls reports
    /// `binding=swift`. Idempotent and thread-safe.
    ///
    /// All parameters are optional. Without an `apiKey`, the SDK runs fully
    /// on-device and telemetry is disabled — the first inference logs a
    /// one-shot hint pointing at the dashboard (suppress with the
    /// `XYBRID_QUIET=1` environment variable). Pass `apiKey` to start the
    /// platform telemetry exporter; `ingestUrl` overrides the destination
    /// for a self-hosted dashboard, and `gatewayUrl` overrides the LLM
    /// gateway. Get a free key at https://dashboard.xybrid.dev.
    ///
    /// Configuration is applied on the first call; because `initialize()` is
    /// idempotent, a later call with different arguments is a no-op.
    ///
    /// On iOS, also enables `UIDevice` battery monitoring and subscribes
    /// to `UIDevice.batteryLevelDidChangeNotification`, forwarding each
    /// reading through the SDK's push-state surface so the routing
    /// engine has live battery telemetry. Thermal state on Apple
    /// platforms is sourced from `NSProcessInfo.thermalState` directly
    /// in `xybrid-core` (no host wiring needed). On macOS, both
    /// battery (IOKit) and thermal (NSProcessInfo) are in-Rust, so
    /// nothing extra is registered here.
    ///
    /// - Parameters:
    ///   - apiKey: Xybrid API key. When set, starts the telemetry exporter.
    ///   - gatewayUrl: Optional override for the LLM gateway URL.
    ///   - ingestUrl: Optional override for the telemetry ingest URL.
    public static func initialize(
        apiKey: String? = nil,
        gatewayUrl: String? = nil,
        ingestUrl: String? = nil
    ) {
        initLock.lock()
        defer { initLock.unlock() }
        if initialized { return }
        setBinding(binding: "swift")
        configureRuntime(apiKey: apiKey, gatewayUrl: gatewayUrl, ingestUrl: ingestUrl)
        // `registerPlatformObservers()` touches UIKit (`UIDevice.current`,
        // `isBatteryMonitoringEnabled`) on iOS, which is main-thread-only.
        // `initialize()` is documented as callable from any thread (apps
        // commonly call it inside a `Task`), so hop to main when needed.
        // The `initialized` guard ensures only one caller ever reaches here,
        // so the deferred registration runs exactly once.
        if Thread.isMainThread {
            registerPlatformObservers()
        } else {
            DispatchQueue.main.async { registerPlatformObservers() }
        }
        initialized = true
    }

    /// Returns `true` if `initialize()` has been called.
    public static var isInitialized: Bool {
        initLock.lock()
        defer { initLock.unlock() }
        return initialized
    }

    private static func registerPlatformObservers() {
        #if os(iOS)
        let device = UIDevice.current
        // Battery monitoring is opt-in on iOS — without this flag,
        // `batteryLevel` returns -1.0 and the notification never fires.
        device.isBatteryMonitoringEnabled = true

        // Push the current value immediately so the first cache miss
        // isn't blind, then keep updating on every notification.
        pushBatteryLevel(device.batteryLevel)

        batteryObserver = NotificationCenter.default.addObserver(
            forName: UIDevice.batteryLevelDidChangeNotification,
            object: nil,
            queue: nil
        ) { _ in
            pushBatteryLevel(UIDevice.current.batteryLevel)
        }
        #endif
    }

    #if os(iOS)
    /// Convert `UIDevice.batteryLevel` (Float in 0.0...1.0, or -1.0 for
    /// unknown) into the SDK's `UInt8` 0..=100 representation. Negative
    /// or non-finite values surface as "unknown" via [`clearBatteryLevel`]
    /// so the routing engine doesn't see a fake 0% reading.
    private static func pushBatteryLevel(_ level: Float) {
        guard level.isFinite, level >= 0 else {
            clearBatteryLevel()
            return
        }
        let pct = (level * 100).rounded()
        // Clamp defensively; iOS has been observed to report 1.01 briefly
        // near a full charge during recalibration.
        let bounded = max(0, min(100, Int(pct)))
        setBatteryLevel(percent: UInt8(bounded))
    }
    #endif
}

// MARK: - Public Type Re-exports

/// A loaded model ready for inference.
/// Call `run(envelope:)` to execute inference on input data.
public typealias Model = XybridModel

// The bolt handle wraps a thread-safe, `Arc`-backed Rust model (the facade's
// types are `Send + Sync`), so the handle is safe to move across threads and
// actors — e.g. loading or running on a `Task.detached` background executor,
// which is the recommended pattern since bolt's `load`/`run` are blocking.
// boltffi does not emit `Sendable` on generated handle types yet, so declare it
// here in the hand-written wrapper (regen-safe — never overwritten by
// `boltffi generate`, unlike `xybrid_bolt.swift`).
extension XybridModel: @unchecked Sendable {}

public extension XybridModel {
    /// Run inference with the model's default options.
    ///
    /// Convenience over `run(envelope:options:)` so simple call sites stay
    /// one-argument; forwards `nil` options. Use the two-arg form to override
    /// generation config, abort signals, or cloud-fallback behaviour.
    func run(envelope: XybridEnvelope) throws -> XybridResult {
        try run(envelope: envelope, options: nil)
    }
}

// MARK: - Async conveniences
//
// bolt's `load` and `run` are synchronous + blocking. These wrappers restore the
// pre-migration `async` API shape: each runs the blocking call on a detached
// background executor (`Task.detached`), so callers `await` without blocking the
// calling thread or actor — exactly what UI code wants for model load / inference.
//
// (boltffi *can* export `async fn` natively, but the SDK's async path uses tokio
// `spawn_blocking`, which panics without an ambient tokio runtime context that
// boltffi's own future driver does not establish. Wrapping the synchronous call
// off-thread is therefore the correct, low-risk way to surface async today.)
public extension XybridModel {
    /// Load a model from the xybrid registry without blocking the caller.
    static func fromRegistryAsync(_ id: String) async throws -> XybridModel {
        try await Task.detached { try XybridModel(fromRegistry: id) }.value
    }

    /// Load a model from a local directory without blocking the caller.
    static func fromDirectoryAsync(_ path: String) async throws -> XybridModel {
        try await Task.detached { try XybridModel(fromDirectory: path) }.value
    }

    /// Load a model from a local `.xyb` bundle without blocking the caller.
    static func fromBundleAsync(_ path: String) async throws -> XybridModel {
        try await Task.detached { try XybridModel(fromBundle: path) }.value
    }

    /// Resolve and load a model from a HuggingFace repo without blocking the caller.
    static func fromHuggingfaceAsync(_ repo: String) async throws -> XybridModel {
        try await Task.detached { try XybridModel(fromHuggingface: repo) }.value
    }

    /// Run inference without blocking the calling thread or actor.
    func runAsync(
        envelope: XybridEnvelope,
        options: XybridRunOptions? = nil
    ) async throws -> XybridResult {
        try await Task.detached { try self.run(envelope: envelope, options: options) }.value
    }

    /// Warm up the model without blocking the calling thread or actor.
    func warmupAsync() async throws {
        try await Task.detached { try self.warmup() }.value
    }

    /// Unload the model, freeing its memory, without blocking the calling
    /// thread or actor.
    func unloadAsync() async throws {
        try await Task.detached { try self.unload() }.value
    }
}

/// Input data for model inference.
/// Use `.audio(pcmData:sampleRate:channels:)` or `.text(_:voice:speed:)`.
public typealias Envelope = XybridEnvelope

/// The result of a model inference operation.
public typealias Result = XybridResult

/// Errors that can occur during model loading or inference.
public typealias XybridSDKError = XybridError

/// Voice metadata for TTS models.
public typealias VoiceInfo = XybridVoiceInfo

/// Generation parameters for LLM inference (temperature, top_p, max_tokens, etc.).
public typealias GenerationConfig = XybridGenerationConfig

// MARK: - XybridResult compatibility shim
//
// The bolt-generated `XybridResult` carries an `envelope` whose `kind` is
// a tagged enum (`.text(text:)`, `.audio(bytes:)`, `.embedding(values:)`).
// The previous uniffi-generated `XybridResult` flattened these into
// optional fields (`text`, `audioBytes`, `embedding`) plus a `success`
// flag. Consumer code (notably the iOS example) reads those flat
// fields, so we mirror them as computed properties on the bolt type.

public extension XybridResult {
    /// `true` for any result returned from `XybridModel.run`. Bolt
    /// surfaces failures as `throws` rather than a `success` flag; this
    /// flag stays at `true` for shape compatibility with the previous
    /// uniffi shape.
    var success: Bool { true }

    /// `true` if the result carries no output (`OutputType.unknown`).
    var isFailure: Bool { outputType == .unknown }

    /// Text payload, if the result is `.text`. `nil` otherwise.
    var text: String? {
        if case .text(let text) = envelope.kind { return text }
        return nil
    }

    /// Audio bytes, if the result is `.audio`. `nil` otherwise.
    ///
    /// Returns `Data` (not `[UInt8]`) because that's what BoltFFI emits
    /// for `Vec<u8>` — gives Swift consumers the standard `Data` API
    /// (slicing, hashing, AVAudioPlayer init) without a copy.
    var audioBytes: Data? {
        if case .audio(let bytes) = envelope.kind { return bytes }
        return nil
    }

    /// Embedding vector, if the result is `.embedding`. `nil` otherwise.
    var embedding: [Float]? {
        if case .embedding(let values) = envelope.kind { return values }
        return nil
    }

    /// The latency as a `TimeInterval` in seconds.
    var latency: TimeInterval { TimeInterval(latencyMs) / 1000.0 }
}

// MARK: - XybridEnvelope compatibility factories
//
// Mirrors the previous uniffi factories that constructed envelopes with
// inline metadata. Bolt's struct uses `kind` + `metadata` arrays; these
// helpers fold the well-known TTS / ASR metadata keys into entries.

public extension XybridEnvelope {
    /// Creates an audio envelope with format metadata.
    /// - Parameters:
    ///   - pcmData: Raw PCM / WAV bytes.
    ///   - sampleRate: Sample rate in Hz (e.g. 16000 for ASR).
    ///   - channels: Number of channels (typically 1 for mono).
    static func audio(pcmData: Data, sampleRate: UInt32 = 16000, channels: UInt32 = 1) -> XybridEnvelope {
        return XybridEnvelope(
            kind: .audio(bytes: pcmData),
            metadata: [
                XybridMetadataEntry(key: "sample_rate", value: String(sampleRate)),
                XybridMetadataEntry(key: "channels", value: String(channels)),
            ]
        )
    }

    /// Creates a text envelope for TTS with default voice.
    static func text(_ content: String) -> XybridEnvelope {
        return XybridEnvelope(kind: .text(text: content), metadata: [])
    }

    /// Creates a text envelope for TTS with explicit voice + speed.
    static func text(_ content: String, voice: String, speed: Double = 1.0) -> XybridEnvelope {
        return text(text: content, voiceId: voice, speed: speed)
    }

    /// Mirrors the previous uniffi factory signature. Voice / speed fold
    /// into metadata entries (the executor reads them by key).
    static func text(text: String, voiceId: String?, speed: Double?) -> XybridEnvelope {
        var metadata: [XybridMetadataEntry] = []
        if let v = voiceId {
            metadata.append(XybridMetadataEntry(key: "voice_id", value: v))
        }
        if let s = speed {
            metadata.append(XybridMetadataEntry(key: "speed", value: String(s)))
        }
        return XybridEnvelope(kind: .text(text: text), metadata: metadata)
    }

    /// Creates an embedding envelope from a float vector.
    static func embedding(data: [Float]) -> XybridEnvelope {
        return XybridEnvelope(kind: .embedding(values: data), metadata: [])
    }

    /// Creates an encoded image envelope for vision-language models. The bytes
    /// are decode-validated on the Rust side at run time (surfacing as a
    /// `XybridError.invalidImage` for bad/oversized/unsupported input).
    /// - Parameters:
    ///   - bytes: Encoded PNG, JPEG, or WebP data
    ///   - format: Image format hint (`png`, `jpeg`, `jpg`, or `webp`)
    static func image(_ bytes: Data, format: String) throws -> XybridEnvelope {
        return XybridEnvelope(
            kind: .image(bytes: bytes, format: try normalizeImageFormat(format)),
            metadata: []
        )
    }

    /// Creates a multi-part user message with text and image attachments,
    /// tagged with the `User` role.
    /// - Parameters:
    ///   - text: User prompt text
    ///   - images: Image envelopes created by `image(_:format:)`
    static func userMessage(_ text: String, images: [XybridEnvelope] = []) throws -> XybridEnvelope {
        guard images.allSatisfy({ envelope in
            if case .image = envelope.kind { return true }
            return false
        }) else {
            throw XybridError.configError(message: "Envelope.userMessage accepts only image envelopes")
        }
        var parts = [XybridEnvelope(kind: .text(text: text), metadata: [])]
        parts.append(contentsOf: images)
        return XybridEnvelope(
            kind: .multiPart(parts: parts),
            metadata: [XybridMetadataEntry(key: "xybrid.role", value: "user")]
        )
    }

    private static func normalizeImageFormat(_ format: String) throws -> String {
        let normalized = format.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "jpg":
            return "jpeg"
        case "jpeg", "png", "webp":
            return normalized
        default:
            throw XybridError.configError(
                message: "Unsupported image format '\(format)'. Supported formats: png, jpeg, jpg, webp"
            )
        }
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
        // BoltFFI emits enum cases in lowerCamelCase (Swift idiom) with
        // named associated values from the Rust variant fields.
        switch self {
        case .modelNotFound(let id):
            return "Model not found: \(id)"
        case .directoryNotFound(let path):
            return "Directory not found: \(path)"
        case .metadataNotFound(let path):
            return "model_metadata.json not found: \(path)"
        case .metadataInvalid(let message):
            return "model_metadata.json is invalid: \(message)"
        case .loadError(let message):
            return "Load error: \(message)"
        case .inferenceError(let message):
            return "Inference failed: \(message)"
        case .abortedForCloudFallback(let reason):
            return "Aborted for cloud fallback: \(reason)"
        case .streamingNotSupported:
            return "Streaming is not supported by this model"
        case .notLoaded:
            return "Model not loaded"
        case .configError(let message):
            return "Invalid configuration: \(message)"
        case .networkError(let message):
            return "Network error: \(message)"
        case .offline(let message):
            return "Registry unreachable: \(message)"
        case .ioError(let message):
            return "I/O error: \(message)"
        case .cacheError(let message):
            return "Cache error: \(message)"
        case .pipelineError(let message):
            return "Pipeline error: \(message)"
        case .circuitOpen(let message):
            return "Circuit breaker open: \(message)"
        case .rateLimited(let retryAfterSecs):
            return "Rate limited, retry after \(retryAfterSecs) seconds"
        case .timeout(let timeoutMs):
            return "Request timeout after \(timeoutMs)ms"
        case .missingArtifact(let message):
            return "Missing artifact: \(message)"
        case .unsupportedModelCapability(let message):
            return "Unsupported model capability: \(message)"
        case .unsupportedBackendCapability(let message):
            return "Unsupported backend capability: \(message)"
        case .invalidImage(let message):
            return "Invalid image: \(message)"
        }
    }
}

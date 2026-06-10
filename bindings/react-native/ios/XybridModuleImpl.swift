import Foundation
import React

// `XybridModuleImpl` does the actual work of every TurboModule call. It
// holds the `id -> XybridModel` map (model handles are opaque strings on
// the JS side) and translates between RN's NSDictionary payloads and the
// Swift-native UniFFI types in the bundled `Xybrid.swift` wrapper.
//
// All long-running operations (load, run) hop onto a detached Task so the
// React Native module thread isn't blocked. Errors map to NSError with the
// underlying XybridError's `errorDescription` as the message.

@objc(XybridModuleImpl)
public final class XybridModuleImpl: NSObject {
  private let modelsLock = NSLock()
  private var models: [String: XybridModel] = [:]

  // -- Lifecycle --

  @objc public func initializeWithCacheDir(_ cacheDir: String?,
                                           resolve: @escaping RCTPromiseResolveBlock,
                                           reject: @escaping RCTPromiseRejectBlock) {
    // The Swift Xybrid.initialize() registers the binding identifier and
    // wires up UIDevice battery observers. We override the binding right
    // after to "react-native" — Xybrid.initialize() registers "swift" by
    // default, but the registry's first-set-wins OnceLock means we have to
    // call set_binding *first* if we want a different value. So: do that
    // here directly and skip the convenience initializer's binding step
    // by relying on its idempotency guard.
    setBinding(binding: "react-native")
    Xybrid.initialize()

    if let dir = cacheDir, !dir.isEmpty {
      initSdkCacheDir(cacheDir: dir)
    } else {
      // Default cache root: <Library>/Caches/xybrid/models
      let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
      let xybridCache = caches.appendingPathComponent("xybrid/models", isDirectory: true)
      try? FileManager.default.createDirectory(at: xybridCache, withIntermediateDirectories: true)
      initSdkCacheDir(cacheDir: xybridCache.path)
    }
    resolve(nil)
  }

  // -- Loaders --

  @objc public func loadFromRegistry(_ modelId: String,
                                     resolve: @escaping RCTPromiseResolveBlock,
                                     reject: @escaping RCTPromiseRejectBlock) {
    let loader = XybridModelLoader.fromRegistry(modelId: modelId)
    runAsyncLoad(loader: loader, resolve: resolve, reject: reject)
  }

  @objc public func loadFromBundle(_ path: String,
                                   resolve: @escaping RCTPromiseResolveBlock,
                                   reject: @escaping RCTPromiseRejectBlock) {
    let loader = XybridModelLoader.fromBundle(path: path)
    runAsyncLoad(loader: loader, resolve: resolve, reject: reject)
  }

  @objc public func loadFromDirectory(_ path: String,
                                      resolve: @escaping RCTPromiseResolveBlock,
                                      reject: @escaping RCTPromiseRejectBlock) {
    do {
      let loader = try XybridModelLoader.fromDirectory(path: path)
      runAsyncLoad(loader: loader, resolve: resolve, reject: reject)
    } catch let error as XybridError {
      rejectXybrid(error, reject)
    } catch {
      reject("xybrid", error.localizedDescription, error)
    }
  }

  @objc public func loadFromHuggingface(_ repo: String,
                                        resolve: @escaping RCTPromiseResolveBlock,
                                        reject: @escaping RCTPromiseRejectBlock) {
    let loader = XybridModelLoader.fromHuggingface(repo: repo)
    runAsyncLoad(loader: loader, resolve: resolve, reject: reject)
  }

  @objc public func releaseModel(_ handle: String,
                                 resolve: @escaping RCTPromiseResolveBlock,
                                 reject: @escaping RCTPromiseRejectBlock) {
    modelsLock.lock()
    models.removeValue(forKey: handle)
    modelsLock.unlock()
    resolve(nil)
  }

  // -- Inference --

  @objc public func run(_ handle: String,
                        envelope: NSDictionary,
                        config: NSDictionary?,
                        resolve: @escaping RCTPromiseResolveBlock,
                        reject: @escaping RCTPromiseRejectBlock) {
    guard let model = lookup(handle) else {
      reject("xybrid_handle", "Unknown model handle: \(handle)", nil)
      return
    }
    let envelopeOrError = decodeEnvelope(envelope)
    let cfg = config.flatMap(decodeGenerationConfig)

    Task.detached {
      switch envelopeOrError {
      case .failure(let err):
        reject("xybrid_envelope", err, nil)
      case .success(let env):
        do {
          let result = try await model.run(envelope: env, config: cfg)
          resolve(self.encodeResult(result))
        } catch let error as XybridError {
          self.rejectXybrid(error, reject)
        } catch {
          reject("xybrid", error.localizedDescription, error)
        }
      }
    }
  }

  // -- TTS introspection --

  @objc public func voices(_ handle: String,
                           resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    guard let model = lookup(handle) else {
      reject("xybrid_handle", "Unknown model handle: \(handle)", nil)
      return
    }
    let voices = model.voices()?.map { encodeVoice($0) }
    resolve(voices as Any)
  }

  @objc public func defaultVoiceId(_ handle: String,
                                   resolve: @escaping RCTPromiseResolveBlock,
                                   reject: @escaping RCTPromiseRejectBlock) {
    guard let model = lookup(handle) else {
      reject("xybrid_handle", "Unknown model handle: \(handle)", nil)
      return
    }
    resolve(model.defaultVoiceId() as Any)
  }

  @objc public func hasVoices(_ handle: String,
                              resolve: @escaping RCTPromiseResolveBlock,
                              reject: @escaping RCTPromiseRejectBlock) {
    guard let model = lookup(handle) else {
      reject("xybrid_handle", "Unknown model handle: \(handle)", nil)
      return
    }
    resolve(model.hasVoices())
  }

  // -- Platform-state push --

  @objc public func setBatteryLevel(_ percent: Double,
                                    resolve: @escaping RCTPromiseResolveBlock,
                                    reject: @escaping RCTPromiseRejectBlock) {
    let bounded = max(0, min(100, Int(percent.rounded())))
    // Free function from xybrid_uniffi.swift; overload resolution distinguishes
    // it from the @objc member above by the `percent:` UInt8 label.
    setBatteryLevel(percent: UInt8(bounded))
    resolve(nil)
  }

  @objc public func clearBatteryLevel(_ resolve: @escaping RCTPromiseResolveBlock,
                                      reject: @escaping RCTPromiseRejectBlock) {
    clearBatteryLevel()
    resolve(nil)
  }

  @objc public func setThermalState(_ state: String,
                                    resolve: @escaping RCTPromiseResolveBlock,
                                    reject: @escaping RCTPromiseRejectBlock) {
    let mapped: XybridThermalState
    switch state.lowercased() {
    case "normal": mapped = .normal
    case "warm": mapped = .warm
    case "hot": mapped = .hot
    case "critical": mapped = .critical
    default:
      reject("xybrid_thermal", "Unknown thermal state: \(state)", nil)
      return
    }
    setThermalState(state: mapped)
    resolve(nil)
  }

  @objc public func clearThermalState(_ resolve: @escaping RCTPromiseResolveBlock,
                                      reject: @escaping RCTPromiseRejectBlock) {
    clearThermalState()
    resolve(nil)
  }

  // MARK: - Helpers

  private func lookup(_ handle: String) -> XybridModel? {
    modelsLock.lock()
    defer { modelsLock.unlock() }
    return models[handle]
  }

  private func store(_ model: XybridModel) -> String {
    let id = UUID().uuidString
    modelsLock.lock()
    models[id] = model
    modelsLock.unlock()
    return id
  }

  private func runAsyncLoad(loader: XybridModelLoader,
                            resolve: @escaping RCTPromiseResolveBlock,
                            reject: @escaping RCTPromiseRejectBlock) {
    Task.detached {
      do {
        let model = try await loader.load()
        let id = self.store(model)
        resolve(id)
      } catch let error as XybridError {
        self.rejectXybrid(error, reject)
      } catch {
        reject("xybrid", error.localizedDescription, error)
      }
    }
  }

  private func decodeEnvelope(_ dict: NSDictionary) -> Result<XybridEnvelope, String> {
    guard let kind = dict["kind"] as? String else {
      return .failure("Envelope missing 'kind' field")
    }
    switch kind {
    case "audio":
      guard let b64 = dict["bytesBase64"] as? String,
            let bytes = Data(base64Encoded: b64) else {
        return .failure("audio envelope: 'bytesBase64' missing or invalid")
      }
      let sampleRate = (dict["sampleRate"] as? NSNumber)?.uint32Value ?? 16000
      let channels = (dict["channels"] as? NSNumber)?.uint32Value ?? 1
      return .success(.audio(bytes: bytes, sampleRate: sampleRate, channels: channels))
    case "text":
      guard let text = dict["text"] as? String else {
        return .failure("text envelope: 'text' missing")
      }
      return .success(.text(text: text,
                            voiceId: dict["voiceId"] as? String,
                            speed: (dict["speed"] as? NSNumber)?.doubleValue))
    case "embedding":
      guard let raw = dict["data"] as? [NSNumber] else {
        return .failure("embedding envelope: 'data' must be a number array")
      }
      return .success(.embedding(data: raw.map { $0.floatValue }))
    default:
      return .failure("Unknown envelope kind: \(kind)")
    }
  }

  private func decodeGenerationConfig(_ dict: NSDictionary) -> XybridGenerationConfig {
    return XybridGenerationConfig(
      maxTokens: (dict["maxTokens"] as? NSNumber)?.uint32Value,
      temperature: (dict["temperature"] as? NSNumber)?.floatValue,
      topP: (dict["topP"] as? NSNumber)?.floatValue,
      minP: (dict["minP"] as? NSNumber)?.floatValue,
      topK: (dict["topK"] as? NSNumber)?.uint32Value,
      repetitionPenalty: (dict["repetitionPenalty"] as? NSNumber)?.floatValue,
      stopSequences: dict["stopSequences"] as? [String]
    )
  }

  private func encodeResult(_ r: XybridResult) -> [String: Any] {
    var out: [String: Any] = [
      "success": r.success,
      "latencyMs": r.latencyMs,
    ]
    if let text = r.text { out["text"] = text }
    if let bytes = r.audioBytes {
      out["audioBytesBase64"] = bytes.base64EncodedString()
    }
    if let emb = r.embedding { out["embedding"] = emb }
    return out
  }

  private func encodeVoice(_ v: XybridVoiceInfo) -> [String: Any] {
    var out: [String: Any] = ["id": v.id, "name": v.name]
    if let g = v.gender { out["gender"] = g }
    if let l = v.language { out["language"] = l }
    if let s = v.style { out["style"] = s }
    return out
  }

  private func rejectXybrid(_ error: XybridError, _ reject: RCTPromiseRejectBlock) {
    let code: String
    switch error {
    case .ModelNotFound: code = "xybrid_model_not_found"
    case .DirectoryNotFound: code = "xybrid_directory_not_found"
    case .MetadataNotFound: code = "xybrid_metadata_not_found"
    case .MetadataInvalid: code = "xybrid_metadata_invalid"
    case .LoadError: code = "xybrid_load_error"
    case .InferenceError: code = "xybrid_inference_error"
    case .StreamingNotSupported: code = "xybrid_streaming_unsupported"
    case .NotLoaded: code = "xybrid_not_loaded"
    case .ConfigError: code = "xybrid_config_error"
    case .NetworkError: code = "xybrid_network_error"
    case .IoError: code = "xybrid_io_error"
    case .CacheError: code = "xybrid_cache_error"
    case .PipelineError: code = "xybrid_pipeline_error"
    case .CircuitOpen: code = "xybrid_circuit_open"
    case .RateLimited: code = "xybrid_rate_limited"
    case .Timeout: code = "xybrid_timeout"
    }
    reject(code, error.errorDescription ?? "Xybrid error", error)
  }
}


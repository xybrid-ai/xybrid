import Foundation
import React

// The Swift half of the TurboModule. XybridModule.mm conforms to the
// Codegen protocol and forwards every selector here unchanged; each method
// decodes its arguments (XybridCodec), calls the bolt Swift SDK compiled into
// this pod (Xybrid.swift + xybrid_bolt.swift), and settles the promise with a
// typed `xybrid_*` code on failure.
//
// Blocking SDK calls (load, run, stream pulls, downloads, disk walks) run on
// a detached task so the module's method queue stays free; cheap getters
// settle inline.

// File-scope aliases for the generated bolt free functions. Inside the class,
// a member named like a global (`setBatteryLevel`, `cacheStatus`, …) shadows
// it — "use of 'X' refers to instance method rather than global function" —
// and the module name can't qualify it portably. At file scope no member is
// in scope, so these bind to the globals. Annotated ones pin the overload,
// should a wrapper in Xybrid.swift ever reuse the name.
private let ffiSetBinding = setBinding
private let ffiInitSdkCacheDir = initSdkCacheDir
private let ffiConfigureRuntime: (String?, String?) -> Void = configureRuntime
private let ffiVersion = version
private let ffiHasApiKey = hasApiKey
private let ffiSetProviderApiKey = setProviderApiKey
private let ffiSetPlatformUrl = setPlatformUrl
private let ffiSetSpeculativeCloud = setSpeculativeCloud
private let ffiIsSpeculativeCloudEnabled = isSpeculativeCloudEnabled
private let ffiWillSpeculateForModel = willSpeculateForModel
private let ffiReleaseMemory = releaseMemory
private let ffiSetAutoRelease = setAutoRelease
private let ffiIsAutoReleaseEnabled = isAutoReleaseEnabled
private let ffiSetBatteryLevel = setBatteryLevel
private let ffiClearBatteryLevel = clearBatteryLevel
private let ffiSetThermalState = setThermalState
private let ffiClearThermalState = clearThermalState
private let ffiCacheStatus = cacheStatus
private let ffiCacheEntries = cacheEntries
private let ffiCacheIsModelCached = cacheIsModelCached
private let ffiCacheModelPath = cacheModelPath
private let ffiCacheListExtractedModelIds = cacheListExtractedModelIds
private let ffiCacheRemoveModel = cacheRemoveModel
private let ffiCacheClear = cacheClear
private let ffiJsonSchemaToGbnf = jsonSchemaToGbnf
private let ffiToolResultsEnvelope = toolResultsEnvelope
private let ffiObservabilityDefaultTracesEndpoint: () -> String = observabilityDefaultTracesEndpoint
private let ffiObservabilityStats: () -> XybridObservabilityStats = observabilityStats
private let ffiObservabilityFlush: (UInt32) throws -> Void = observabilityFlush
private let ffiObservabilityShutdown: (UInt32) throws -> Void = observabilityShutdown
private let ffiObservabilityRecordFeedback:
  (XybridObservationRef, Double?, String?, String?) throws -> Bool = observabilityRecordFeedback
private let ffiObservabilityRecordEvaluation:
  (XybridObservationRef, String, Double?, String?, String?) throws -> Bool = observabilityRecordEvaluation

/// Process-wide SDK setup. The Rust SDK's configuration is process-global
/// (first-set-wins), and it outlives this module: a JS reload builds a new
/// module instance but not a new process. So what has been applied is
/// tracked process-wide too, which is what lets a reload repeat
/// `initialize` with the same options.
private enum SdkSetup {
  struct Runtime: Equatable {
    var apiKey: String?
    var gatewayUrl: String?

    var isEmpty: Bool { apiKey == nil && gatewayUrl == nil }
  }

  private static let lock = NSLock()
  nonisolated(unsafe) private static var appliedCacheDir: String?
  nonisolated(unsafe) private static var appliedRuntime: Runtime?

  /// Register the binding, battery observers and the cache directory, once.
  /// Every bridged call runs this first, so local inference needs no
  /// `initialize()` at all.
  static func ensureBase(cacheDir requested: String? = nil) throws {
    lock.lock()
    defer { lock.unlock() }
    if let applied = appliedCacheDir {
      if let requested, requested != applied {
        throw BridgeError.config(
          "the model cache is already at \(applied); cacheDir only applies to the first Xybrid call")
      }
      return
    }
    // First-set-wins in the SDK: claim the binding before Xybrid.initialize()
    // registers "swift". Its own configureRuntime(nil, nil) is a no-op;
    // what it adds is the UIDevice battery observer.
    ffiSetBinding("react-native")
    Xybrid.initialize(apiKey: nil, gatewayUrl: nil)
    let directory = try requested ?? defaultCacheDirectory()
    try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
    ffiInitSdkCacheDir(directory)
    appliedCacheDir = directory
  }

  /// Apply the API key and URL overrides — once per process.
  static func configure(_ runtime: Runtime) throws {
    guard !runtime.isEmpty else { return }
    lock.lock()
    defer { lock.unlock() }
    if let applied = appliedRuntime {
      if applied == runtime { return }
      throw BridgeError.config(
        "Xybrid is already initialized with different options; they apply once per app process")
    }
    ffiConfigureRuntime(runtime.apiKey, runtime.gatewayUrl)
    appliedRuntime = runtime
  }

  /// A bolt observability config from the object src/observability.ts sends.
  static func observabilityConfig(_ any: Any?) throws -> XybridObservabilityConfig {
    let what = "observability config"
    let object = try XybridCodec.object(any, what)
    let kind = try XybridCodec.string(object, "kind", what)
    let config: XybridObservabilityConfig
    switch kind {
    case "xybrid": config = XybridObservabilityConfig(xybrid: try XybridCodec.string(object, "apiKey", what))
    case "otlp": config = XybridObservabilityConfig(otlp: try XybridCodec.string(object, "endpoint", what))
    default: throw BridgeError.invalidArgument("unknown observability preset '\(kind)'")
    }
    if let endpoint = try XybridCodec.optionalString(object, "tracesEndpoint", what) {
      config.setTracesEndpoint(endpoint: endpoint)
    }
    for (name, value) in try XybridCodec.stringMap(object, "headers", what) {
      config.setHeader(name: name, value: value)
    }
    if let json = try XybridCodec.optionalBool(object, "jsonProtocol", what) {
      config.setJsonProtocol(json: json)
    }
    if let service = try XybridCodec.optionalObject(object, "service", "\(what): 'service'") {
      config.setService(name: try XybridCodec.string(service, "name", "\(what) service"),
                        version: try XybridCodec.optionalString(service, "version", "\(what) service"))
    }
    for (key, value) in try XybridCodec.stringMap(object, "resourceAttributes", what) {
      config.setResourceAttribute(key: key, value: value)
    }
    if let raw = XybridCodec.value(object, "providerOrigins") {
      guard let origins = raw as? [String] else { throw BridgeError.invalidArgument("providerOrigins must be strings") }
      for origin in origins { try config.allowProviderContext(origin: origin) }
    }
    if let id = try XybridCodec.optionalString(object, "deviceId", what) {
      config.setDeviceId(id: id)
    }
    if let rawRules = XybridCodec.value(object, "redactionRules") {
      guard let rules = rawRules as? [Any] else { throw BridgeError.invalidArgument("redactionRules must be an array") }
      for raw in rules {
        let rule = try XybridCodec.object(raw, "redaction rule")
        switch try XybridCodec.string(rule, "kind", "redaction rule") {
        case "replaceLiteral": try config.addLiteralRedaction(find: XybridCodec.string(rule, "find", "redaction rule"), replacement: XybridCodec.string(rule, "replacement", "redaction rule"))
        case "omitAttribute": try config.omitContentAttribute(key: XybridCodec.string(rule, "key", "redaction rule"))
        default: throw BridgeError.invalidArgument("unknown redaction rule")
        }
      }
    }
    if let metadataOnly = try XybridCodec.optionalBool(object, "metadataOnly", what) {
      config.setMetadataOnly(metadataOnly: metadataOnly)
    }
    if let bytes = try XybridCodec.optionalUInt32(object, "maxContentBytes", what) {
      config.setMaxContentBytes(bytes: bytes)
    }
    if let ms = try XybridCodec.optionalUInt32(object, "liveTraceIntervalMs", what) {
      config.setLiveTraceIntervalMs(ms: ms)
    }
    return config
  }

  private static func defaultCacheDirectory() throws -> String {
    guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
      throw BridgeError.config("could not resolve the caches directory")
    }
    return caches.appendingPathComponent("xybrid/models", isDirectory: true).path
  }
}

/// Clamp a JS millisecond timeout into `UInt64` without trapping on NaN,
/// infinity, negatives, or values at or above 2^63 (`UInt64.max` itself is
/// not exactly representable as a `Double`).
private func clampTimeoutMs(_ raw: Double) -> UInt64 {
  guard raw.isFinite, raw > 0 else { return 0 }
  let ceiling = Double(UInt64(1) << 63)
  return raw >= ceiling ? UInt64(1) << 63 : UInt64(raw)
}

/// Blank means absent for configuration strings.
private func nonBlank(_ value: String?) -> String? {
  guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
    return nil
  }
  return trimmed
}

@objc(XybridModuleImpl)
public final class XybridModuleImpl: NSObject {
  private let handles = XybridHandles()

  // MARK: - Settling promises

  /// Settle inline — for calls that return immediately.
  private func now(_ resolve: RCTPromiseResolveBlock,
                   _ reject: RCTPromiseRejectBlock,
                   _ work: () throws -> Any?) {
    do {
      try SdkSetup.ensureBase()
      resolve(try work())
    } catch {
      Self.reject(reject, error)
    }
  }

  /// Settle from a detached task — for calls that block or suspend.
  private func background(_ resolve: @escaping RCTPromiseResolveBlock,
                          _ reject: @escaping RCTPromiseRejectBlock,
                          _ work: @escaping () async throws -> Any?) {
    Task.detached {
      do {
        try SdkSetup.ensureBase()
        resolve(try await work())
      } catch {
        Self.reject(reject, error)
      }
    }
  }

  /// The SDK reports a run stopped by its token as an inference error
  /// ("user_cancelled"); JS gets the dedicated `xybrid_cancelled` code.
  private static func cancellationAware(_ error: Error, _ token: XybridCancellationToken) -> Error {
    guard token.isCancelled(), !(error is BridgeError) else { return error }
    return XybridError.cancelled(message: "The run was cancelled")
  }

  private static func reject(_ reject: RCTPromiseRejectBlock, _ error: Error) {
    let (code, message) = XybridCodec.rejection(for: error)
    reject(code, message, NSError(domain: "Xybrid", code: 0,
                                  userInfo: [NSLocalizedDescriptionKey: message]))
  }

  // MARK: - Lifecycle

  /// Module teardown (reload, host shutdown): stop in-flight work, free all.
  @objc public func invalidate() {
    handles.all(XybridCancellationToken.self).forEach { $0.cancel() }
    handles.disposeAll()
  }

  // MARK: - SDK configuration

  @objc(initialize:resolve:reject:)
  public func initialize(_ options: NSDictionary?,
                         resolve: @escaping RCTPromiseResolveBlock,
                         reject: @escaping RCTPromiseRejectBlock) {
    do {
      let object = (options as? JSObject) ?? [:]
      let what = "initialize options"
      let cacheDir = try nonBlank(XybridCodec.optionalString(object, "cacheDir", what))
      try SdkSetup.ensureBase(cacheDir: cacheDir.map(XybridCodec.filePath))
      try SdkSetup.configure(SdkSetup.Runtime(
        apiKey: nonBlank(try XybridCodec.optionalString(object, "apiKey", what)),
        gatewayUrl: nonBlank(try XybridCodec.optionalString(object, "gatewayUrl", what))
      ))
      resolve(nil)
    } catch {
      Self.reject(reject, error)
    }
  }

  @objc(sdkVersion:reject:)
  public func sdkVersion(_ resolve: @escaping RCTPromiseResolveBlock,
                         reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiVersion() }
  }

  @objc(hasApiKey:reject:)
  public func hasApiKey(_ resolve: @escaping RCTPromiseResolveBlock,
                        reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiHasApiKey() }
  }

  @objc(setProviderApiKey:apiKey:resolve:reject:)
  public func setProviderApiKey(_ provider: String,
                                apiKey: String,
                                resolve: @escaping RCTPromiseResolveBlock,
                                reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiSetProviderApiKey(provider, apiKey); return nil }
  }

  @objc(setPlatformUrl:resolve:reject:)
  public func setPlatformUrl(_ url: String,
                             resolve: @escaping RCTPromiseResolveBlock,
                             reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiSetPlatformUrl(url); return nil }
  }

  @objc(setSpeculativeCloud:resolve:reject:)
  public func setSpeculativeCloud(_ enabled: Bool,
                                  resolve: @escaping RCTPromiseResolveBlock,
                                  reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiSetSpeculativeCloud(enabled); return nil }
  }

  @objc(isSpeculativeCloudEnabled:reject:)
  public func isSpeculativeCloudEnabled(_ resolve: @escaping RCTPromiseResolveBlock,
                                        reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiIsSpeculativeCloudEnabled() }
  }

  @objc(willSpeculate:resolve:reject:)
  public func willSpeculate(_ modelId: String,
                            resolve: @escaping RCTPromiseResolveBlock,
                            reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { ffiWillSpeculateForModel(modelId) }
  }

  @objc(releaseMemory:reject:)
  public func releaseMemory(_ resolve: @escaping RCTPromiseResolveBlock,
                            reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { ffiReleaseMemory() }
  }

  @objc(setAutoRelease:resolve:reject:)
  public func setAutoRelease(_ enabled: Bool,
                             resolve: @escaping RCTPromiseResolveBlock,
                             reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiSetAutoRelease(enabled); return nil }
  }

  @objc(isAutoReleaseEnabled:reject:)
  public func isAutoReleaseEnabled(_ resolve: @escaping RCTPromiseResolveBlock,
                                   reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiIsAutoReleaseEnabled() }
  }

  // MARK: - Observability

  @objc(observabilityInit:resolve:reject:)
  public func observabilityInit(_ config: NSDictionary,
                                resolve: @escaping RCTPromiseResolveBlock,
                                reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) {
      try SdkSetup.observabilityConfig(config).`init`()
      return nil
    }
  }

  @objc(observabilityDefaultTracesEndpoint:reject:)
  public func observabilityDefaultTracesEndpoint(_ resolve: @escaping RCTPromiseResolveBlock,
                                                 reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiObservabilityDefaultTracesEndpoint() }
  }

  @objc(observabilityStats:reject:)
  public func observabilityStats(_ resolve: @escaping RCTPromiseResolveBlock,
                                 reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { XybridCodec.encodeObservabilityStats(ffiObservabilityStats()) }
  }

  @objc(observabilityFlush:resolve:reject:)
  public func observabilityFlush(_ timeoutMs: Double,
                                 resolve: @escaping RCTPromiseResolveBlock,
                                 reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) {
      try ffiObservabilityFlush(try XybridCodec.uint32(timeoutMs, "flush timeout"))
      return nil
    }
  }

  @objc(observabilityShutdown:resolve:reject:)
  public func observabilityShutdown(_ timeoutMs: Double,
                                    resolve: @escaping RCTPromiseResolveBlock,
                                    reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) {
      try ffiObservabilityShutdown(try XybridCodec.uint32(timeoutMs, "shutdown timeout"))
      return nil
    }
  }

  @objc(observabilityRecordFeedback:feedback:resolve:reject:)
  public func observabilityRecordFeedback(_ target: NSDictionary,
                                          feedback: NSDictionary,
                                          resolve: @escaping RCTPromiseResolveBlock,
                                          reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let what = "feedback"
      let object = try XybridCodec.object(feedback, what)
      return try ffiObservabilityRecordFeedback(
        try XybridCodec.decodeObservationRef(target),
        try XybridCodec.optionalDouble(object, "score", what),
        try XybridCodec.optionalString(object, "label", what),
        try XybridCodec.optionalString(object, "comment", what))
    }
  }

  @objc(observabilityRecordEvaluation:evaluation:resolve:reject:)
  public func observabilityRecordEvaluation(_ target: NSDictionary,
                                            evaluation: NSDictionary,
                                            resolve: @escaping RCTPromiseResolveBlock,
                                            reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let what = "evaluation"
      let object = try XybridCodec.object(evaluation, what)
      return try ffiObservabilityRecordEvaluation(
        try XybridCodec.decodeObservationRef(target),
        try XybridCodec.string(object, "name", what),
        try XybridCodec.optionalDouble(object, "scoreValue", what),
        try XybridCodec.optionalString(object, "scoreLabel", what),
        try XybridCodec.optionalString(object, "explanation", what))
    }
  }

  @objc(observationStart:descriptor:options:resolve:reject:)
  public func observationStart(_ kind: String, descriptor: NSDictionary, options: NSDictionary?, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let entry = try XybridObservationEntry.start(kind, try XybridCodec.object(descriptor, "observation"), try XybridCodec.decodeObservation(options))
      return self.handles.insert(entry, kind: "observation")
    }
  }

  @objc(observationContext:resolve:reject:)
  public func observationContext(_ handle: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      return try self.handles.get(handle, as: XybridObservationEntry.self).context()
    }
  }

  @objc(observationReference:resolve:reject:)
  public func observationReference(_ handle: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      return try self.handles.get(handle, as: XybridObservationEntry.self).reference()
    }
  }

  @objc(observationComplete:resolve:reject:)
  public func observationComplete(_ handle: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try self.handles.get(handle, as: XybridObservationEntry.self).complete(); return nil
    }
  }

  @objc(observationFail:errorType:message:resolve:reject:)
  public func observationFail(_ handle: String, errorType: String, message: String?, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try self.handles.get(handle, as: XybridObservationEntry.self).fail(errorType, message); return nil
    }
  }

  @objc(observationCancel:resolve:reject:)
  public func observationCancel(_ handle: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try self.handles.get(handle, as: XybridObservationEntry.self).cancel(); return nil
    }
  }

  @objc(observationIsFinished:resolve:reject:)
  public func observationIsFinished(_ handle: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      return try self.handles.get(handle, as: XybridObservationEntry.self).isFinished()
    }
  }

  @objc(observationRecord:field:value:resolve:reject:)
  public func observationRecord(_ handle: String, field: String, value: NSDictionary, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try self.handles.get(handle, as: XybridObservationEntry.self).record(field, try XybridCodec.object(value, "observation value")); return nil
    }
  }

  // MARK: - Device state push

  @objc(setBatteryLevel:resolve:reject:)
  public func setBatteryLevel(_ percent: Double,
                              resolve: @escaping RCTPromiseResolveBlock,
                              reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      guard percent.isFinite else {
        throw BridgeError.invalidArgument("battery level must be a finite number")
      }
      ffiSetBatteryLevel(UInt8(max(0, min(100, percent.rounded()))))
      return nil
    }
  }

  @objc(clearBatteryLevel:reject:)
  public func clearBatteryLevel(_ resolve: @escaping RCTPromiseResolveBlock,
                                reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiClearBatteryLevel(); return nil }
  }

  @objc(setThermalState:resolve:reject:)
  public func setThermalState(_ state: String,
                              resolve: @escaping RCTPromiseResolveBlock,
                              reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let mapped: XybridThermalState
      switch state {
      case "normal": mapped = .normal
      case "warm": mapped = .warm
      case "hot": mapped = .hot
      case "critical": mapped = .critical
      default: throw BridgeError.invalidArgument("unknown thermal state '\(state)'")
      }
      ffiSetThermalState(mapped)
      return nil
    }
  }

  @objc(clearThermalState:reject:)
  public func clearThermalState(_ resolve: @escaping RCTPromiseResolveBlock,
                                reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { ffiClearThermalState(); return nil }
  }

  // MARK: - Model cache (disk walks: off the method queue)

  @objc(cacheStatus:reject:)
  public func cacheStatus(_ resolve: @escaping RCTPromiseResolveBlock,
                          reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { XybridCodec.encodeCacheStatus(try ffiCacheStatus()) }
  }

  @objc(cacheEntries:reject:)
  public func cacheEntries(_ resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { try ffiCacheEntries().map(XybridCodec.encodeCacheEntry) }
  }

  @objc(cacheIsModelCached:resolve:reject:)
  public func cacheIsModelCached(_ modelId: String,
                                 resolve: @escaping RCTPromiseResolveBlock,
                                 reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { try ffiCacheIsModelCached(modelId) }
  }

  @objc(cacheModelPath:resolve:reject:)
  public func cacheModelPath(_ modelId: String,
                             resolve: @escaping RCTPromiseResolveBlock,
                             reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { try ffiCacheModelPath(modelId) }
  }

  @objc(cacheExtractedModelIds:reject:)
  public func cacheExtractedModelIds(_ resolve: @escaping RCTPromiseResolveBlock,
                                     reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { try ffiCacheListExtractedModelIds() }
  }

  @objc(cacheRemoveModel:resolve:reject:)
  public func cacheRemoveModel(_ modelId: String,
                               resolve: @escaping RCTPromiseResolveBlock,
                               reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { try ffiCacheRemoveModel(modelId) }
  }

  @objc(cacheClear:reject:)
  public func cacheClear(_ resolve: @escaping RCTPromiseResolveBlock,
                         reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { try ffiCacheClear() }
  }

  // MARK: - Stateless helpers

  @objc(jsonSchemaToGbnf:resolve:reject:)
  public func jsonSchemaToGbnf(_ schemaJson: String,
                               resolve: @escaping RCTPromiseResolveBlock,
                               reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { try ffiJsonSchemaToGbnf(schemaJson) }
  }

  @objc(toolResultsEnvelope:priorAssistantText:results:resolve:reject:)
  public func toolResultsEnvelope(_ userText: String,
                                  priorAssistantText: String,
                                  results: NSArray,
                                  resolve: @escaping RCTPromiseResolveBlock,
                                  reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let decoded = try (results as? [Any] ?? []).map(XybridCodec.decodeToolResult)
      return try XybridCodec.encodeEnvelope(ffiToolResultsEnvelope(userText, priorAssistantText, decoded))
    }
  }

  // MARK: - Handles

  @objc(dispose:resolve:reject:)
  public func dispose(_ handle: String,
                      resolve: @escaping RCTPromiseResolveBlock,
                      reject: @escaping RCTPromiseRejectBlock) {
    handles.dispose(handle)
    resolve(nil)
  }

  // MARK: - Models

  @objc(loadModel:resolve:reject:)
  public func loadModel(_ source: NSDictionary,
                        resolve: @escaping RCTPromiseResolveBlock,
                        reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      let object = try XybridCodec.object(source, "model source")
      let what = "model source"
      let kind = try XybridCodec.string(object, "kind", what)
      let value = try XybridCodec.string(object, "value", what)
      let modelSource: XybridModelSource
      switch kind {
      case "registry": modelSource = .registry(id: value, speculative: false)
      case "registrySpeculative": modelSource = .registry(id: value, speculative: true)
      case "bundle": modelSource = .bundle(path: XybridCodec.filePath(value))
      case "directory": modelSource = .directory(path: XybridCodec.filePath(value))
      case "huggingFace": modelSource = .huggingFace(repo: value, revision: try XybridCodec.optionalString(object, "revision", what))
      case "modelFile": modelSource = .modelFile(path: XybridCodec.filePath(value))
      default: throw BridgeError.invalidArgument("unknown model source kind '\(kind)'")
      }
      let model = try XybridModel(load: modelSource, observation: XybridCodec.decodeObservation(object["observation"]))
      return handles.insert(model, kind: "model")
    }
  }

  @objc(modelInfo:resolve:reject:)
  public func modelInfo(_ model: String,
                        resolve: @escaping RCTPromiseResolveBlock,
                        reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let m = try handles.get(model, as: XybridModel.self)
      let info: JSObject = [
        "modelId": m.modelId(),
        "loadObservationRef": m.loadObservationRef().map { XybridCodec.encodeObservationRef($0) as Any } ?? NSNull(),
        "version": m.version(),
        "outputType": XybridCodec.encodeOutputType(m.outputType()),
        "isLlm": m.isLlm(),
        "supportsStreaming": m.supportsStreaming(),
        "supportsTokenStreaming": m.supportsTokenStreaming(),
        "supportsToolCalling": m.supportsToolCalling().map { $0 as Any } ?? NSNull(),
        "hasVoices": m.hasVoices(),
        "defaultGenerationConfig": XybridCodec.encodeGenerationConfig(m.defaultGenerationConfig()),
      ]
      return info
    }
  }

  @objc(isLoaded:resolve:reject:)
  public func isLoaded(_ model: String,
                       resolve: @escaping RCTPromiseResolveBlock,
                       reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { try handles.get(model, as: XybridModel.self).isLoaded() }
  }

  @objc(warmup:resolve:reject:)
  public func warmup(_ model: String,
                     resolve: @escaping RCTPromiseResolveBlock,
                     reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      try handles.get(model, as: XybridModel.self).warmup()
      return nil
    }
  }

  @objc(unload:resolve:reject:)
  public func unload(_ model: String,
                     resolve: @escaping RCTPromiseResolveBlock,
                     reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      try handles.get(model, as: XybridModel.self).unload()
      return nil
    }
  }

  @objc(isCloudServing:resolve:reject:)
  public func isCloudServing(_ model: String,
                             resolve: @escaping RCTPromiseResolveBlock,
                             reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { try handles.get(model, as: XybridModel.self).isCloudServing() }
  }

  @objc(downloadStatus:resolve:reject:)
  public func downloadStatus(_ model: String,
                             resolve: @escaping RCTPromiseResolveBlock,
                             reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      XybridCodec.encodeDownloadStatus(try handles.get(model, as: XybridModel.self).downloadStatus())
    }
  }

  @objc(awaitDownload:timeoutMs:resolve:reject:)
  public func awaitDownload(_ model: String,
                            timeoutMs: Double,
                            resolve: @escaping RCTPromiseResolveBlock,
                            reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      let m = try handles.get(model, as: XybridModel.self)
      return XybridCodec.encodeDownloadStatus(m.awaitDownload(timeoutMs: clampTimeoutMs(timeoutMs)))
    }
  }

  @objc(voices:resolve:reject:)
  public func voices(_ model: String,
                     resolve: @escaping RCTPromiseResolveBlock,
                     reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(model, as: XybridModel.self).voices().map(XybridCodec.encodeVoice)
    }
  }

  @objc(defaultVoice:resolve:reject:)
  public func defaultVoice(_ model: String,
                           resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(model, as: XybridModel.self).defaultVoice().map(XybridCodec.encodeVoice)
    }
  }

  @objc(voice:voiceId:resolve:reject:)
  public func voice(_ model: String,
                    voiceId: String,
                    resolve: @escaping RCTPromiseResolveBlock,
                    reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(model, as: XybridModel.self).voice(voiceId: voiceId).map(XybridCodec.encodeVoice)
    }
  }

  // MARK: - Inference

  /// Resolve the handles a run request carries. Without a caller token a
  /// batch run still gets one, registered under the model so disposing the
  /// model (or tearing the module down) stops it; a stream's own entry plays
  /// that role instead, so streams skip the registration.
  private func prepareRun(_ modelHandle: String, _ options: NSDictionary?, registerToken: Bool) throws
    -> (model: XybridModel, request: XybridCodec.RunRequest,
        context: XybridConversationContext?, cancel: XybridCancellationToken, ownToken: String?)
  {
    let model = try handles.get(modelHandle, as: XybridModel.self)
    let request = try XybridCodec.decodeRunRequest(options)
    let context = try request.context.map { try handles.get($0, as: XybridConversationContext.self) }
    if let token = request.cancel {
      return (model, request, context, try handles.get(token, as: XybridCancellationToken.self), nil)
    }
    let token = XybridCancellationToken()
    let handle = registerToken
      ? handles.insert(token, kind: "cancel", owner: modelHandle, onDispose: { token.cancel() })
      : nil
    return (model, request, context, token, handle)
  }

  @objc(run:envelope:options:resolve:reject:)
  public func run(_ model: String,
                  envelope: NSDictionary,
                  options: NSDictionary?,
                  resolve: @escaping RCTPromiseResolveBlock,
                  reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      let input = try XybridCodec.decodeEnvelope(envelope)
      let run = try self.prepareRun(model, options, registerToken: true)
      defer { if let own = run.ownToken { handles.dispose(own) } }
      do {
        let result: XybridResult
        if let context = run.context {
          result = try run.model.runWithContext(envelope: input, context: context,
                                                options: run.request.options, cancel: run.cancel)
        } else {
          result = try run.model.run(envelope: input, options: run.request.options, cancel: run.cancel)
        }
        return try XybridCodec.encodeResult(result)
      } catch {
        throw Self.cancellationAware(error, run.cancel)
      }
    }
  }

  @objc(streamStart:envelope:options:resolve:reject:)
  public func streamStart(_ model: String,
                          envelope: NSDictionary,
                          options: NSDictionary?,
                          resolve: @escaping RCTPromiseResolveBlock,
                          reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      let input = try XybridCodec.decodeEnvelope(envelope)
      let run = try self.prepareRun(model, options, registerToken: false)
      let streamId: UInt64
      if let context = run.context {
        streamId = try run.model.runStreamWithContext(envelope: input, context: context,
                                                      options: run.request.options, cancel: run.cancel)
      } else {
        streamId = try run.model.runStream(envelope: input, options: run.request.options, cancel: run.cancel)
      }
      let entry = XybridTokenStreamEntry(model: run.model, streamId: streamId, cancel: run.cancel)
      return handles.insert(entry, kind: "stream", owner: model, onDispose: { entry.abort() })
    }
  }

  @objc(streamNext:resolve:reject:)
  public func streamNext(_ stream: String,
                         resolve: @escaping RCTPromiseResolveBlock,
                         reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      // A disposed stream reads as exhausted, not as an error.
      guard let entry = handles.find(stream, as: XybridTokenStreamEntry.self) else { return nil }
      do {
        let event = try entry.model.streamNext(streamId: entry.streamId)
        switch event.kind {
        case .token:
          guard let token = event.token else {
            throw XybridError.inferenceError(message: "stream returned a token event without a token")
          }
          return ["kind": "token", "token": XybridCodec.encodeStreamToken(token)] as JSObject
        case .complete:
          // `streamResult` closes the bolt session; drop our entry with it.
          let result = try entry.model.streamResult(streamId: entry.streamId)
          handles.dispose(stream)
          return ["kind": "complete", "result": try XybridCodec.encodeResult(result)] as JSObject
        case .audio:
          guard let packet = event.audio else {
            throw XybridError.inferenceError(message: "stream returned an audio event without audio")
          }
          return ["kind": "audio", "audio": XybridCodec.encodeTtsAudio(packet)] as JSObject
        }
      } catch {
        // Dispose also aborts a stream rejected for an unsupported event.
        handles.dispose(stream)
        throw Self.cancellationAware(error, entry.cancel)
      }
    }
  }

  // MARK: - Cancellation tokens

  @objc(createCancelToken:reject:)
  public func createCancelToken(_ resolve: @escaping RCTPromiseResolveBlock,
                                reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { handles.insert(XybridCancellationToken(), kind: "cancel") }
  }

  @objc(cancel:resolve:reject:)
  public func cancel(_ token: String,
                     resolve: @escaping RCTPromiseResolveBlock,
                     reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(token, as: XybridCancellationToken.self).cancel()
      return nil
    }
  }

  // MARK: - Conversation contexts

  @objc(createContext:resolve:reject:)
  public func createContext(_ contextId: String?,
                            resolve: @escaping RCTPromiseResolveBlock,
                            reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let context = contextId.map { XybridConversationContext(withId: $0) } ?? XybridConversationContext()
      return handles.insert(context, kind: "context")
    }
  }

  @objc(contextPush:envelope:resolve:reject:)
  public func contextPush(_ context: String,
                          envelope: NSDictionary,
                          resolve: @escaping RCTPromiseResolveBlock,
                          reject: @escaping RCTPromiseRejectBlock) {
    // Image turns are decode-validated natively, so this can take a moment.
    background(resolve, reject) { [handles] in
      try handles.get(context, as: XybridConversationContext.self)
        .push(envelope: try XybridCodec.decodeEnvelope(envelope))
      return nil
    }
  }

  @objc(contextSetSystem:envelope:resolve:reject:)
  public func contextSetSystem(_ context: String,
                               envelope: NSDictionary,
                               resolve: @escaping RCTPromiseResolveBlock,
                               reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      try handles.get(context, as: XybridConversationContext.self)
        .setSystem(envelope: try XybridCodec.decodeEnvelope(envelope))
      return nil
    }
  }

  @objc(contextClear:resolve:reject:)
  public func contextClear(_ context: String,
                           resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(context, as: XybridConversationContext.self).clear()
      return nil
    }
  }

  @objc(contextInfo:resolve:reject:)
  public func contextInfo(_ context: String,
                          resolve: @escaping RCTPromiseResolveBlock,
                          reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let c = try handles.get(context, as: XybridConversationContext.self)
      return ["id": c.id(), "historyLength": c.historyLen(), "hasSystem": c.hasSystem()] as JSObject
    }
  }

  @objc(contextHistory:resolve:reject:)
  public func contextHistory(_ context: String,
                             resolve: @escaping RCTPromiseResolveBlock,
                             reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(context, as: XybridConversationContext.self).history().map(XybridCodec.encodeEnvelope)
    }
  }

  @objc(contextSetMaxHistoryLength:length:resolve:reject:)
  public func contextSetMaxHistoryLength(_ context: String,
                                         length: Double,
                                         resolve: @escaping RCTPromiseResolveBlock,
                                         reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let c = try handles.get(context, as: XybridConversationContext.self)
      c.setMaxHistoryLen(len: try XybridCodec.uint32(length, "max history length"))
      return nil
    }
  }

  // MARK: - Standalone downloads

  @objc(startDownload:platform:resolve:reject:)
  public func startDownload(_ modelId: String,
                            platform: String?,
                            resolve: @escaping RCTPromiseResolveBlock,
                            reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      let download = platform.map { XybridDownload(fromRegistryWithPlatform: modelId, platform: $0) }
        ?? XybridDownload(fromRegistry: modelId)
      return handles.insert(download, kind: "download")
    }
  }

  @objc(downloadHandleStatus:resolve:reject:)
  public func downloadHandleStatus(_ download: String,
                                   resolve: @escaping RCTPromiseResolveBlock,
                                   reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      XybridCodec.encodeDownloadStatus(try handles.get(download, as: XybridDownload.self).status())
    }
  }

  @objc(downloadHandleError:resolve:reject:)
  public func downloadHandleError(_ download: String,
                                  resolve: @escaping RCTPromiseResolveBlock,
                                  reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { try handles.get(download, as: XybridDownload.self).error() }
  }

  @objc(cancelDownload:resolve:reject:)
  public func cancelDownload(_ download: String,
                             resolve: @escaping RCTPromiseResolveBlock,
                             reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(download, as: XybridDownload.self).cancel()
      return nil
    }
  }

  // MARK: - Pipelines

  @objc(loadPipeline:resolve:reject:)
  public func loadPipeline(_ source: NSDictionary,
                           resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      let object = try XybridCodec.object(source, "pipeline source")
      let kind = try XybridCodec.string(object, "kind", "pipeline source")
      let value = try XybridCodec.string(object, "value", "pipeline source")
      let pipeline: XybridPipeline
      switch kind {
      case "yaml": pipeline = try XybridPipeline(fromYaml: value)
      case "file": pipeline = try XybridPipeline(fromFile: XybridCodec.filePath(value))
      case "bundle": pipeline = try XybridPipeline(fromBundle: XybridCodec.filePath(value))
      default: throw BridgeError.invalidArgument("unknown pipeline source kind '\(kind)'")
      }
      return handles.insert(pipeline, kind: "pipeline")
    }
  }

  @objc(pipelineInfo:resolve:reject:)
  public func pipelineInfo(_ pipeline: String,
                           resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      let p = try handles.get(pipeline, as: XybridPipeline.self)
      return [
        "name": p.name().map { $0 as Any } ?? NSNull(),
        "stageNames": p.stageNames(),
        "stageCount": p.stageCount(),
      ] as JSObject
    }
  }

  @objc(runPipeline:envelope:options:resolve:reject:)
  public func runPipeline(_ pipeline: String,
                          envelope: NSDictionary,
                          options: NSDictionary?,
                          resolve: @escaping RCTPromiseResolveBlock,
                          reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      let p = try handles.get(pipeline, as: XybridPipeline.self)
      let result = try p.run(envelope: try XybridCodec.decodeEnvelope(envelope),
                             options: try XybridCodec.decodeRunRequest(options).options)
      return try XybridCodec.encodePipelineResult(result)
    }
  }

  // MARK: - Live ASR sessions

  @objc(openStreamingSession:config:resolve:reject:)
  public func openStreamingSession(_ model: String,
                                   config: NSDictionary?,
                                   resolve: @escaping RCTPromiseResolveBlock,
                                   reject: @escaping RCTPromiseRejectBlock) {
    // Opening warms the weights, so it runs off the method queue.
    background(resolve, reject) { [handles] in
      let m = try handles.get(model, as: XybridModel.self)
      let session = try XybridStreamingSession(forModel: m, config: try XybridCodec.decodeStreamingConfig(config))
      let entry = XybridSessionEntry(session: session)
      return handles.insert(entry, kind: "session", onDispose: { session.cancel() })
    }
  }

  @objc(sessionFeed:samplesBase64:resolve:reject:)
  public func sessionFeed(_ session: String,
                          samplesBase64: String,
                          resolve: @escaping RCTPromiseResolveBlock,
                          reject: @escaping RCTPromiseRejectBlock) {
    // `feed` blocks only when the worker's queue is full (back-pressure).
    background(resolve, reject) { [handles] in
      let entry = try handles.get(session, as: XybridSessionEntry.self)
      try entry.session.feed(samples: try XybridCodec.float32Samples(samplesBase64))
      return nil
    }
  }

  @objc(sessionNextPartial:resolve:reject:)
  public func sessionNextPartial(_ session: String,
                                 resolve: @escaping RCTPromiseResolveBlock,
                                 reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      guard let entry = handles.find(session, as: XybridSessionEntry.self) else { return nil }
      return await entry.nextPartial().map(XybridCodec.encodePartial)
    }
  }

  @objc(sessionFlush:resolve:reject:)
  public func sessionFlush(_ session: String,
                           resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      try XybridCodec.encodeTranscription(handles.get(session, as: XybridSessionEntry.self).session.flush())
    }
  }

  @objc(sessionReset:resolve:reject:)
  public func sessionReset(_ session: String,
                           resolve: @escaping RCTPromiseResolveBlock,
                           reject: @escaping RCTPromiseRejectBlock) {
    background(resolve, reject) { [handles] in
      try handles.get(session, as: XybridSessionEntry.self).session.reset()
      return nil
    }
  }

  @objc(sessionCancel:resolve:reject:)
  public func sessionCancel(_ session: String,
                            resolve: @escaping RCTPromiseResolveBlock,
                            reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) {
      try handles.get(session, as: XybridSessionEntry.self).session.cancel()
      return nil
    }
  }

  @objc(sessionIsRunning:resolve:reject:)
  public func sessionIsRunning(_ session: String,
                               resolve: @escaping RCTPromiseResolveBlock,
                               reject: @escaping RCTPromiseRejectBlock) {
    now(resolve, reject) { try handles.get(session, as: XybridSessionEntry.self).session.isRunning() }
  }
}

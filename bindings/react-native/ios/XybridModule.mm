#import "XybridModule.h"
#import "react_native_xybrid-Swift.h"

// ObjC++ shim that forwards every TurboModule call to the Swift
// `XybridModuleImpl` actor. The split exists because:
//   1. RCT_EXPORT_MODULE / codegen need an ObjC class on the registration path.
//   2. The xybrid-uniffi APIs are Swift-native (async functions, `Arc<T>`
//      surfaces as Swift class instances) — calling them from ObjC++ is
//      possible but loses the `async`/`throws` ergonomics. Keeping the bridge
//      thin in ObjC and the work in Swift mirrors how the Apple SDK wrapper
//      is structured (see bindings/apple/Sources/Xybrid/Xybrid.swift).

@implementation XybridModule {
  XybridModuleImpl *_impl;
}

RCT_EXPORT_MODULE(RNXybrid)

+ (BOOL)requiresMainQueueSetup { return NO; }

- (instancetype)init {
  if ((self = [super init])) {
    _impl = [XybridModuleImpl new];
  }
  return self;
}

#pragma mark - Lifecycle

RCT_REMAP_METHOD(initialize,
                 initializeWithCacheDir:(NSString * _Nullable)cacheDir
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl initializeWithCacheDir:cacheDir resolve:resolve reject:reject];
}

#pragma mark - Loaders

RCT_REMAP_METHOD(loadFromRegistry,
                 loadFromRegistry:(NSString *)modelId
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl loadFromRegistry:modelId resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(loadFromBundle,
                 loadFromBundle:(NSString *)path
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl loadFromBundle:path resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(loadFromDirectory,
                 loadFromDirectory:(NSString *)path
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl loadFromDirectory:path resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(loadFromHuggingface,
                 loadFromHuggingface:(NSString *)repo
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl loadFromHuggingface:repo resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(releaseModel,
                 releaseModel:(NSString *)handle
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl releaseModel:handle resolve:resolve reject:reject];
}

#pragma mark - Inference

RCT_REMAP_METHOD(run,
                 run:(NSString *)handle
                 envelope:(NSDictionary *)envelope
                 config:(NSDictionary * _Nullable)config
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl run:handle envelope:envelope config:config resolve:resolve reject:reject];
}

#pragma mark - TTS introspection

RCT_REMAP_METHOD(voices,
                 voices:(NSString *)handle
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl voices:handle resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(defaultVoiceId,
                 defaultVoiceId:(NSString *)handle
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl defaultVoiceId:handle resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(hasVoices,
                 hasVoices:(NSString *)handle
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl hasVoices:handle resolve:resolve reject:reject];
}

#pragma mark - Platform-state push

RCT_REMAP_METHOD(setBatteryLevel,
                 setBatteryLevel:(double)percent
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl setBatteryLevel:percent resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(clearBatteryLevel,
                 clearBatteryLevelWithResolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl clearBatteryLevel:resolve reject:reject];
}

RCT_REMAP_METHOD(setThermalState,
                 setThermalState:(NSString *)state
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl setThermalState:state resolve:resolve reject:reject];
}

RCT_REMAP_METHOD(clearThermalState,
                 clearThermalStateWithResolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject) {
  [_impl clearThermalState:resolve reject:reject];
}

#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params {
  return std::make_shared<facebook::react::NativeRNXybridSpecJSI>(params);
}
#endif

@end

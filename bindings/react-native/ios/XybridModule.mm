#import <RNXybridSpec/RNXybridSpec.h>
#import "react_native_xybrid-Swift.h"

// Thin ObjC++ forwarder: Codegen dispatches JS calls by the selectors of the
// generated `NativeXybridSpec` protocol, and each one below hands its
// arguments unchanged to the Swift `XybridModuleImpl`, which has the same
// selector. The split exists because the protocol header is ObjC++ (Swift
// cannot import it) and `getTurboModule:` returns a C++ type.
//
// Every selector must match the generated protocol exactly — a mismatch
// compiles (as a warning) but crashes at the first call, since the TurboModule
// builds an NSInvocation from the protocol's selector. The podspec makes that
// warning an error and tests/spec-conformance.test.mjs checks the list
// against src/NativeXybrid.ts. New Architecture only.
//
// There is deliberately no public header: the Codegen header is ObjC++, and
// a public header importing it would land in the pod's umbrella header, which
// Swift then fails to import ("This file must be compiled as Obj-C++").

@interface XybridModule : NSObject <NativeXybridSpec>
@end

@implementation XybridModule {
  XybridModuleImpl *_impl;
}

// React Native 0.77+ finds the module through `codegenConfig.ios.modulesProvider`
// in package.json; RCT_EXPORT_MODULE keeps 0.76 working.
RCT_EXPORT_MODULE(RNXybrid)

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

- (instancetype)init
{
  if ((self = [super init])) {
    _impl = [XybridModuleImpl new];
  }
  return self;
}

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<facebook::react::NativeXybridSpecJSI>(params);
}

// Reload / teardown: stop in-flight work and free every native object.
- (void)invalidate
{
  [_impl invalidate];
}

- (void)initialize:(NSDictionary * _Nullable)options
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject
{
  [_impl initialize:options resolve:resolve reject:reject];
}

- (void)sdkVersion:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject
{
  [_impl sdkVersion:resolve reject:reject];
}

- (void)hasApiKey:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject
{
  [_impl hasApiKey:resolve reject:reject];
}

- (void)setProviderApiKey:(NSString *)provider
                   apiKey:(NSString *)apiKey
                  resolve:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject
{
  [_impl setProviderApiKey:provider apiKey:apiKey resolve:resolve reject:reject];
}

- (void)setPlatformUrl:(NSString *)url
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject
{
  [_impl setPlatformUrl:url resolve:resolve reject:reject];
}

- (void)setSpeculativeCloud:(BOOL)enabled
                    resolve:(RCTPromiseResolveBlock)resolve
                     reject:(RCTPromiseRejectBlock)reject
{
  [_impl setSpeculativeCloud:enabled resolve:resolve reject:reject];
}

- (void)isSpeculativeCloudEnabled:(RCTPromiseResolveBlock)resolve
                           reject:(RCTPromiseRejectBlock)reject
{
  [_impl isSpeculativeCloudEnabled:resolve reject:reject];
}

- (void)willSpeculate:(NSString *)modelId
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_impl willSpeculate:modelId resolve:resolve reject:reject];
}

- (void)releaseMemory:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_impl releaseMemory:resolve reject:reject];
}

- (void)setAutoRelease:(BOOL)enabled
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject
{
  [_impl setAutoRelease:enabled resolve:resolve reject:reject];
}

- (void)isAutoReleaseEnabled:(RCTPromiseResolveBlock)resolve
                      reject:(RCTPromiseRejectBlock)reject
{
  [_impl isAutoReleaseEnabled:resolve reject:reject];
}

- (void)setBatteryLevel:(double)percent
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject
{
  [_impl setBatteryLevel:percent resolve:resolve reject:reject];
}

- (void)clearBatteryLevel:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject
{
  [_impl clearBatteryLevel:resolve reject:reject];
}

- (void)setThermalState:(NSString *)state
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject
{
  [_impl setThermalState:state resolve:resolve reject:reject];
}

- (void)clearThermalState:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject
{
  [_impl clearThermalState:resolve reject:reject];
}

- (void)cacheStatus:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject
{
  [_impl cacheStatus:resolve reject:reject];
}

- (void)cacheEntries:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject
{
  [_impl cacheEntries:resolve reject:reject];
}

- (void)cacheIsModelCached:(NSString *)modelId
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject
{
  [_impl cacheIsModelCached:modelId resolve:resolve reject:reject];
}

- (void)cacheModelPath:(NSString *)modelId
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject
{
  [_impl cacheModelPath:modelId resolve:resolve reject:reject];
}

- (void)cacheExtractedModelIds:(RCTPromiseResolveBlock)resolve
                        reject:(RCTPromiseRejectBlock)reject
{
  [_impl cacheExtractedModelIds:resolve reject:reject];
}

- (void)cacheRemoveModel:(NSString *)modelId
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject
{
  [_impl cacheRemoveModel:modelId resolve:resolve reject:reject];
}

- (void)cacheClear:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject
{
  [_impl cacheClear:resolve reject:reject];
}

- (void)jsonSchemaToGbnf:(NSString *)schemaJson
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject
{
  [_impl jsonSchemaToGbnf:schemaJson resolve:resolve reject:reject];
}

- (void)toolResultsEnvelope:(NSString *)userText
         priorAssistantText:(NSString *)priorAssistantText
                    results:(NSArray *)results
                    resolve:(RCTPromiseResolveBlock)resolve
                     reject:(RCTPromiseRejectBlock)reject
{
  [_impl toolResultsEnvelope:userText priorAssistantText:priorAssistantText results:results resolve:resolve reject:reject];
}

- (void)dispose:(NSString *)handle
        resolve:(RCTPromiseResolveBlock)resolve
         reject:(RCTPromiseRejectBlock)reject
{
  [_impl dispose:handle resolve:resolve reject:reject];
}

- (void)loadModel:(NSDictionary *)source
          resolve:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject
{
  [_impl loadModel:source resolve:resolve reject:reject];
}

- (void)modelInfo:(NSString *)model
          resolve:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject
{
  [_impl modelInfo:model resolve:resolve reject:reject];
}

- (void)isLoaded:(NSString *)model
         resolve:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject
{
  [_impl isLoaded:model resolve:resolve reject:reject];
}

- (void)warmup:(NSString *)model
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject
{
  [_impl warmup:model resolve:resolve reject:reject];
}

- (void)unload:(NSString *)model
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject
{
  [_impl unload:model resolve:resolve reject:reject];
}

- (void)isCloudServing:(NSString *)model
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject
{
  [_impl isCloudServing:model resolve:resolve reject:reject];
}

- (void)downloadStatus:(NSString *)model
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject
{
  [_impl downloadStatus:model resolve:resolve reject:reject];
}

- (void)awaitDownload:(NSString *)model
            timeoutMs:(double)timeoutMs
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_impl awaitDownload:model timeoutMs:timeoutMs resolve:resolve reject:reject];
}

- (void)voices:(NSString *)model
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject
{
  [_impl voices:model resolve:resolve reject:reject];
}

- (void)defaultVoice:(NSString *)model
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject
{
  [_impl defaultVoice:model resolve:resolve reject:reject];
}

- (void)voice:(NSString *)model
      voiceId:(NSString *)voiceId
      resolve:(RCTPromiseResolveBlock)resolve
       reject:(RCTPromiseRejectBlock)reject
{
  [_impl voice:model voiceId:voiceId resolve:resolve reject:reject];
}

- (void)run:(NSString *)model
   envelope:(NSDictionary *)envelope
    options:(NSDictionary * _Nullable)options
    resolve:(RCTPromiseResolveBlock)resolve
     reject:(RCTPromiseRejectBlock)reject
{
  [_impl run:model envelope:envelope options:options resolve:resolve reject:reject];
}

- (void)streamStart:(NSString *)model
           envelope:(NSDictionary *)envelope
            options:(NSDictionary * _Nullable)options
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject
{
  [_impl streamStart:model envelope:envelope options:options resolve:resolve reject:reject];
}

- (void)streamNext:(NSString *)stream
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject
{
  [_impl streamNext:stream resolve:resolve reject:reject];
}

- (void)createCancelToken:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject
{
  [_impl createCancelToken:resolve reject:reject];
}

- (void)cancel:(NSString *)token
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject
{
  [_impl cancel:token resolve:resolve reject:reject];
}

- (void)createContext:(NSString * _Nullable)contextId
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_impl createContext:contextId resolve:resolve reject:reject];
}

- (void)contextPush:(NSString *)context
           envelope:(NSDictionary *)envelope
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject
{
  [_impl contextPush:context envelope:envelope resolve:resolve reject:reject];
}

- (void)contextSetSystem:(NSString *)context
                envelope:(NSDictionary *)envelope
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject
{
  [_impl contextSetSystem:context envelope:envelope resolve:resolve reject:reject];
}

- (void)contextClear:(NSString *)context
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject
{
  [_impl contextClear:context resolve:resolve reject:reject];
}

- (void)contextInfo:(NSString *)context
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject
{
  [_impl contextInfo:context resolve:resolve reject:reject];
}

- (void)contextHistory:(NSString *)context
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject
{
  [_impl contextHistory:context resolve:resolve reject:reject];
}

- (void)contextSetMaxHistoryLength:(NSString *)context
                            length:(double)length
                           resolve:(RCTPromiseResolveBlock)resolve
                            reject:(RCTPromiseRejectBlock)reject
{
  [_impl contextSetMaxHistoryLength:context length:length resolve:resolve reject:reject];
}

- (void)startDownload:(NSString *)modelId
             platform:(NSString * _Nullable)platform
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_impl startDownload:modelId platform:platform resolve:resolve reject:reject];
}

- (void)downloadHandleStatus:(NSString *)download
                     resolve:(RCTPromiseResolveBlock)resolve
                      reject:(RCTPromiseRejectBlock)reject
{
  [_impl downloadHandleStatus:download resolve:resolve reject:reject];
}

- (void)downloadHandleError:(NSString *)download
                    resolve:(RCTPromiseResolveBlock)resolve
                     reject:(RCTPromiseRejectBlock)reject
{
  [_impl downloadHandleError:download resolve:resolve reject:reject];
}

- (void)cancelDownload:(NSString *)download
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject
{
  [_impl cancelDownload:download resolve:resolve reject:reject];
}

- (void)loadPipeline:(NSDictionary *)source
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject
{
  [_impl loadPipeline:source resolve:resolve reject:reject];
}

- (void)pipelineInfo:(NSString *)pipeline
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject
{
  [_impl pipelineInfo:pipeline resolve:resolve reject:reject];
}

- (void)runPipeline:(NSString *)pipeline
           envelope:(NSDictionary *)envelope
            options:(NSDictionary * _Nullable)options
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject
{
  [_impl runPipeline:pipeline envelope:envelope options:options resolve:resolve reject:reject];
}

- (void)openStreamingSession:(NSString *)model
                      config:(NSDictionary * _Nullable)config
                     resolve:(RCTPromiseResolveBlock)resolve
                      reject:(RCTPromiseRejectBlock)reject
{
  [_impl openStreamingSession:model config:config resolve:resolve reject:reject];
}

- (void)sessionFeed:(NSString *)session
      samplesBase64:(NSString *)samplesBase64
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject
{
  [_impl sessionFeed:session samplesBase64:samplesBase64 resolve:resolve reject:reject];
}

- (void)sessionNextPartial:(NSString *)session
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject
{
  [_impl sessionNextPartial:session resolve:resolve reject:reject];
}

- (void)sessionFlush:(NSString *)session
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject
{
  [_impl sessionFlush:session resolve:resolve reject:reject];
}

- (void)sessionReset:(NSString *)session
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject
{
  [_impl sessionReset:session resolve:resolve reject:reject];
}

- (void)sessionCancel:(NSString *)session
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_impl sessionCancel:session resolve:resolve reject:reject];
}

- (void)sessionIsRunning:(NSString *)session
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject
{
  [_impl sessionIsRunning:session resolve:resolve reject:reject];
}
@end

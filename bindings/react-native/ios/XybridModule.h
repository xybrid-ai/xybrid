#import <React/RCTBridgeModule.h>

#ifdef RCT_NEW_ARCH_ENABLED
#import <RNXybridSpec/RNXybridSpec.h>

@interface XybridModule : NSObject <NativeRNXybridSpec>
@end

#else

@interface XybridModule : NSObject <RCTBridgeModule>
@end

#endif

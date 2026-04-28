#ifndef outer_loop_layerhost
#define outer_loop_layerhost

// PrivateCALayerBridge.h  (add to "Objective-C Bridging Header")

#import <QuartzCore/QuartzCore.h>
#import <dlfcn.h>          // for CGSMainConnectionID()


#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t CGSConnectionID;
CGSConnectionID CGSMainConnectionID(void);   // from SkyLight/CoreGraphicsServices

#ifdef __cplusplus
}
#endif

typedef uint32_t CAContextID;
//
//@interface CAContext : NSObject
//  + (id)contextWithCGSConnection:(CAContextID)contextId options:(NSDictionary*)optionsDict;
//  @property(readonly) CAContextID contextId;
//  @property(retain) CALayer *layer;
//@end

@interface CAContext : NSObject
+ (instancetype)contextWithCGSConnection:(CGSConnectionID)conn
                                 options:(NSDictionary*)opts;
@property(readonly) CAContextID contextId;
@property(retain)  CALayer     *layer;
@end

@interface CALayerHost : CALayer
@property uint32_t contextId;
@property BOOL inheritsSecurity;
@property BOOL preservesFlip;
@end



#endif

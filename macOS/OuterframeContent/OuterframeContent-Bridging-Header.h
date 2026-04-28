#import <QuartzCore/QuartzCore.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t CGSConnectionID;
CGSConnectionID CGSMainConnectionID(void);

#ifdef __cplusplus
}
#endif

typedef uint32_t CAContextID;

@interface CAContext : NSObject
+ (instancetype)contextWithCGSConnection:(CGSConnectionID)conn
                                 options:(NSDictionary *)opts;
@property (readonly) CAContextID contextId;
@property (retain) CALayer *layer;
@end

@interface CALayerHost : CALayer
@property uint32_t contextId;
@property BOOL inheritsSecurity;
@property BOOL preservesFlip;
@end

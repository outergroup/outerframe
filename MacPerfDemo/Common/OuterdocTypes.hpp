//
//  OuterdocTypes.hpp
//


#ifndef outerdoc_types_hpp
#define outerdoc_types_hpp

#import <Foundation/NSUUID.h>
#import <Foundation/NSRange.h>

/// A return type that plugins can use.
@interface ElementInfo : NSObject
@property (nonatomic, strong) NSUUID *elementID;
@property (nonatomic, assign) NSRange range;
@end

@interface CanvasInfo : NSObject
@property (nonatomic, strong) NSUUID *canvasID;
@property (nonatomic, assign) int location;
@property (nonatomic, assign) int width;
@property (nonatomic, assign) int height;
@end

#endif

//
//  ContentCommunication.hpp
//

#ifndef content_communication_hpp
#define content_communication_hpp

#ifdef __OBJC__
@class NSAttributedString;
#else
typedef void NSAttributedString;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#include "OuterdocTypes.hpp"
#import <Foundation/NSArray.h>


// Message handling typedefs for Outerdoc messages
typedef void (^TextMessageHandler)(NSString* _Nonnull text);
typedef void (^CreateDynamicSubstringsHandler)(NSArray<NSUUID*>* _Nonnull substringIds, NSArray<NSString*>* _Nonnull fileTypes);
typedef void (^CreateDynamicSubstringHandler)(NSUUID* _Nonnull substringId, NSString* _Nonnull fileType);
typedef void (^DisposeSubstringHandler)(NSUUID* _Nonnull substringId);

// Process an OuterdocMessage using handlers
void processOuterdocMessage(NSData* _Nonnull data,
                          TextMessageHandler _Nullable textHandler,
                          CreateDynamicSubstringsHandler _Nullable createDynamicSubstringsHandler,
                          CreateDynamicSubstringHandler _Nullable createDynamicSubstringHandler,
                          DisposeSubstringHandler _Nullable disposeSubstringHandler);

// Message handling typedefs for OuterContent messages
typedef void (^CreateDynamicSubstringsResponseHandler)(NSArray<NSUUID*>* _Nonnull substringIds,
                                              NSString* _Nonnull status,
                                              NSArray<NSAttributedString*>* _Nonnull attributedStrings,
                                              NSArray<NSArray<ElementInfo*>*>* _Nonnull elementsArrays,
                                              NSArray<NSArray<CanvasInfo*>*>* _Nonnull canvasesArrays,
                                              NSArray<NSArray<NSString*>*>* _Nonnull surfaceServiceNames);
typedef void (^CreateDynamicSubstringResponseHandler)(NSUUID* _Nonnull substringId,
                                            NSAttributedString* _Nonnull attributedString,
                                            NSArray<ElementInfo*>* _Nullable elements,
                                            NSArray<CanvasInfo*>* _Nullable canvases,
                                            NSArray<NSString*>* _Nonnull surfaceServiceNames);
typedef void (^UpdateElementsHandler)(NSUUID* _Nonnull substringId,
                                    NSArray<NSUUID*>* _Nonnull elementIds,
                                    NSArray<NSData*>* _Nonnull newContents);

// Process an OuterContentMessage using handlers
void processOuterContentMessage(NSData* _Nonnull data,
                              CreateDynamicSubstringsResponseHandler _Nullable createDynamicSubstringsResponseHandler,
                              CreateDynamicSubstringResponseHandler _Nullable createDynamicSubstringResponseHandler,
                              UpdateElementsHandler _Nullable updateElementsHandler);

// Serialization functions for sending messages
NSData* _Nullable serializeTextMessage(NSString* _Nonnull text);
NSData* _Nullable serializeCreateDynamicSubstringsRequest(NSArray<NSUUID*>* _Nonnull substringIds, NSArray<NSString*>* _Nonnull fileTypes);
NSData* _Nullable serializeCreateDynamicSubstringRequest(NSUUID* _Nonnull substringId, NSString* _Nonnull fileType);
NSData* _Nullable serializeDisposeSubstringRequest(NSUUID* _Nonnull substringId);

NSData* _Nullable serializeCreateDynamicSubstringsResponse(NSArray<NSUUID*>* _Nonnull substringIds,
                                                  NSString* _Nonnull status,
                                                  NSArray<NSAttributedString*>* _Nonnull attributedStrings,
                                                  NSArray<NSArray<ElementInfo*>*>* _Nonnull elementsArrays,
                                                  NSArray<NSArray<CanvasInfo*>*>* _Nonnull canvasesArrays,
                                                  NSArray<NSArray<NSString*>*>* _Nonnull surfaceServiceNames);
NSData* _Nullable serializeCreateDynamicSubstringResponse(NSUUID* _Nonnull substringId,
                                                NSAttributedString* _Nonnull attributedString,
                                                NSArray<ElementInfo*>* _Nullable elements,
                                                NSArray<CanvasInfo*>* _Nullable canvases,
                                                NSArray<NSString*>* _Nonnull surfaceServiceNames);
NSData* _Nullable serializeUpdateElementsMessage(NSUUID* _Nonnull substringId,
                                               NSArray<NSUUID*>* _Nonnull elementIds,
                                               NSArray<NSData*>* _Nonnull newContents);



#endif

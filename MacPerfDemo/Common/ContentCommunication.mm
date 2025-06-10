//
//  ContentCommunication.mm
//

#include "ContentCommunication.hpp"
#include "Communication.capnp.h"

#include <capnp/serialize-packed.h>

#include <Foundation/Foundation.h>


// Helper function to convert UUID between Cap'n Proto and Foundation
static void setUUID(UUID::Builder builder, NSUUID* uuid) {
    uuid_t bytes;
    [uuid getUUIDBytes:bytes];
    uint64_t high, low;
    memcpy(&high, bytes, 8);
    memcpy(&low, bytes + 8, 8);
    builder.setHigh(high);
    builder.setLow(low);
}

static NSUUID* getUUID(UUID::Reader reader) {
    uuid_t bytes;
    uint64_t high = reader.getHigh();
    uint64_t low = reader.getLow();
    memcpy(bytes, &high, 8);
    memcpy(bytes + 8, &low, 8);
    return [[NSUUID alloc] initWithUUIDBytes:bytes];
}

// Helper function to deserialize NSKeyedArchiver data to NSAttributedString
static NSAttributedString* deserializeArchivedData(NSData* archivedData) {
    if (archivedData && archivedData.length > 0) {
        NSError* error = nil;
        NSAttributedString* attrString = [NSKeyedUnarchiver unarchivedObjectOfClass:[NSAttributedString class]
                                                                           fromData:archivedData
                                                                              error:&error];
        if (error) {
            NSLog(@"Error deserializing archived data: %@", error);
            return [[NSAttributedString alloc] init];
        }
        return attrString ?: [[NSAttributedString alloc] init];
    }
    return [[NSAttributedString alloc] init];
}

// Helper function to deserialize elements
static NSArray<ElementInfo*>* deserializeElements(::capnp::List<RangeIDPair>::Reader elements) {
    NSMutableArray<ElementInfo*>* result = [NSMutableArray array];
    for (auto element : elements) {
        ElementInfo* info = [[ElementInfo alloc] init];
        info.range = NSMakeRange(element.getLocation(), element.getLength());
        info.elementID = getUUID(element.getId());
        [result addObject:info];
    }
    return result;
}

// Helper function to deserialize canvases
static NSArray<CanvasInfo*>* deserializeCanvases(::capnp::List<LocationCanvasPair>::Reader canvases) {
    NSMutableArray<CanvasInfo*>* result = [NSMutableArray array];
    for (auto canvas : canvases) {
        CanvasInfo* info = [[CanvasInfo alloc] init];
        info.canvasID = getUUID(canvas.getId());
        info.location = static_cast<int>(canvas.getLocation());
        info.width = canvas.getWidth();
        info.height = canvas.getHeight();
        [result addObject:info];
    }
    return result;
}


// Process OuterdocMessage
void processOuterdocMessage(NSData* _Nonnull data,
                          TextMessageHandler _Nullable textHandler,
                          CreateDynamicSubstringsHandler _Nullable createDynamicSubstringsHandler,
                          CreateDynamicSubstringHandler _Nullable createDynamicSubstringHandler,
                          DisposeSubstringHandler _Nullable disposeSubstringHandler) {
    try {
        const capnp::word* words = reinterpret_cast<const capnp::word*>([data bytes]);
        size_t wordCount = [data length] / sizeof(capnp::word);

        ::capnp::FlatArrayMessageReader reader(kj::ArrayPtr<const capnp::word>(words, wordCount));
        OuterdocMessage::Reader message = reader.getRoot<OuterdocMessage>();

        switch (message.which()) {
            case OuterdocMessage::TEXT_MESSAGE:
                if (textHandler) {
                    NSString* text = [NSString stringWithUTF8String:message.getTextMessage().cStr()];
                    textHandler(text);
                }
                break;

            case OuterdocMessage::CREATE_DYNAMIC_SUBSTRINGS: {
                if (createDynamicSubstringsHandler) {
                    auto request = message.getCreateDynamicSubstrings();
                    NSMutableArray<NSUUID*>* substringIds = [NSMutableArray array];
                    NSMutableArray<NSString*>* fileTypes = [NSMutableArray array];

                    for (auto uuid : request.getSubstringIds()) {
                        [substringIds addObject:getUUID(uuid)];
                    }

                    for (auto fileType : request.getFileTypes()) {
                        [fileTypes addObject:[NSString stringWithUTF8String:fileType.cStr()]];
                    }

                    createDynamicSubstringsHandler(substringIds, fileTypes);
                }
                break;
            }

            case OuterdocMessage::CREATE_DYNAMIC_SUBSTRING: {
                if (createDynamicSubstringHandler) {
                    auto request = message.getCreateDynamicSubstring();
                    NSUUID* substringId = getUUID(request.getSubstringId());
                    NSString* fileType = [NSString stringWithUTF8String:request.getFileType().cStr()];
                    createDynamicSubstringHandler(substringId, fileType);
                }
                break;
            }

            case OuterdocMessage::DISPOSE_SUBSTRING:
                if (disposeSubstringHandler) {
                    auto request = message.getDisposeSubstring();
                    NSUUID* substringId = getUUID(request.getSubstringId());
                    disposeSubstringHandler(substringId);
                }
                break;
        }
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in processOuterdocMessage: %s", e.getDescription().cStr());
    }
}

// Process OuterContentMessage
void processOuterContentMessage(NSData* _Nonnull data,
                              CreateDynamicSubstringsResponseHandler _Nullable createDynamicSubstringsResponseHandler,
                              CreateDynamicSubstringResponseHandler _Nullable createDynamicSubstringResponseHandler,
                              UpdateElementsHandler _Nullable updateElementsHandler) {
    try {
        const capnp::word* words = reinterpret_cast<const capnp::word*>([data bytes]);
        size_t wordCount = [data length] / sizeof(capnp::word);

        ::capnp::FlatArrayMessageReader reader(kj::ArrayPtr<const capnp::word>(words, wordCount));
        OuterContentMessage::Reader message = reader.getRoot<OuterContentMessage>();

        switch (message.which()) {
            case OuterContentMessage::CREATE_DYNAMIC_SUBSTRINGS_RESPONSE: {
                if (createDynamicSubstringsResponseHandler) {
                    auto response = message.getCreateDynamicSubstringsResponse();
                    NSMutableArray<NSUUID*>* substringIds = [NSMutableArray array];
                    NSMutableArray<NSAttributedString*>* attributedStrings = [NSMutableArray array];
                    NSMutableArray<NSArray*>* elementsArrays = [NSMutableArray array];
                    NSMutableArray<NSArray*>* canvasesArrays = [NSMutableArray array];
                    NSMutableArray<NSArray<NSString*>*>* surfaceServiceNames = [NSMutableArray array];

                    for (auto uuid : response.getSubstringIds()) {
                        [substringIds addObject:getUUID(uuid)];
                    }

                    NSString* status = [NSString stringWithUTF8String:response.getStatus().cStr()];

                    for (auto result : response.getResults()) {
                        // Deserialize RTFD data
                        auto archivedList = result.getAttributedString();
                        NSData* archivedData = nil;
                        if (archivedList.size() > 0) {
                            archivedData = [NSData dataWithBytes:archivedList[0].begin() length:archivedList[0].size()];
                        }
                        NSAttributedString* attrString = deserializeArchivedData(archivedData);
                        [attributedStrings addObject:attrString];
                        
                        NSArray<ElementInfo*>* elements = deserializeElements(result.getElements());
                        [elementsArrays addObject:elements];
                        
                        NSArray<CanvasInfo*>* canvases = deserializeCanvases(result.getCanvases());
                        [canvasesArrays addObject:canvases];

                        // Extract service names from canvases
                        NSMutableArray<NSString*>* serviceNames = [NSMutableArray array];
                        for (auto canvas : result.getCanvases()) {
                            [serviceNames addObject:[NSString stringWithUTF8String:canvas.getServiceName().cStr()]];
                        }
                        [surfaceServiceNames addObject:serviceNames];
                    }

                    createDynamicSubstringsResponseHandler(substringIds, status, attributedStrings, elementsArrays, canvasesArrays, surfaceServiceNames);
                }
                break;
            }

            case OuterContentMessage::CREATE_DYNAMIC_SUBSTRING_RESPONSE: {
                if (createDynamicSubstringResponseHandler) {
                    auto response = message.getCreateDynamicSubstringResponse();
                    NSUUID* substringId = getUUID(response.getSubstringId());
                    
                    // Deserialize archived data
                    auto archivedData = response.getAttributedString();
                    NSData* archivedNSData = [NSData dataWithBytes:archivedData.begin() length:archivedData.size()];
                    NSAttributedString* attrString = deserializeArchivedData(archivedNSData);
                    NSArray<ElementInfo*>* elements = deserializeElements(response.getElements());
                    NSArray<CanvasInfo*>* canvases = deserializeCanvases(response.getCanvases());

                    // Extract service names from canvases
                    NSMutableArray<NSString*>* serviceNames = [NSMutableArray array];
                    for (auto canvas : response.getCanvases()) {
                        [serviceNames addObject:[NSString stringWithUTF8String:canvas.getServiceName().cStr()]];
                    }

                    createDynamicSubstringResponseHandler(substringId, attrString, elements, canvases, serviceNames);
                }
                break;
            }

            case OuterContentMessage::UPDATE_ELEMENTS: {
                if (updateElementsHandler) {
                    auto update = message.getUpdateElements();
                    NSUUID* substringId = getUUID(update.getSubstringId());
                    NSMutableArray<NSUUID*>* elementIds = [NSMutableArray array];
                    NSMutableArray<NSData*>* newContents = [NSMutableArray array];

                    for (auto uuid : update.getElementIds()) {
                        [elementIds addObject:getUUID(uuid)];
                    }

                    for (auto content : update.getNewContents()) {
                        [newContents addObject:[NSData dataWithBytes:content.begin() length:content.size()]];
                    }

                    updateElementsHandler(substringId, elementIds, newContents);
                }
                break;
            }
        }
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in processOuterContentMessage: %s", e.getDescription().cStr());
    }
}

// Serialization functions for Outerdoc messages
NSData* _Nullable serializeTextMessage(NSString* _Nonnull text) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterdocMessage::Builder builder = message.initRoot<OuterdocMessage>();
        builder.setTextMessage([text UTF8String]);

        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        return [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeTextMessage: %s", e.getDescription().cStr());
        return nil;
    }
}

NSData* _Nullable serializeCreateDynamicSubstringsRequest(NSArray<NSUUID*>* _Nonnull substringIds, NSArray<NSString*>* _Nonnull fileTypes) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterdocMessage::Builder builder = message.initRoot<OuterdocMessage>();
        auto request = builder.initCreateDynamicSubstrings();

        auto idsBuilder = request.initSubstringIds(static_cast<unsigned int>([substringIds count]));
        for (unsigned int i = 0; i < [substringIds count]; i++) {
            setUUID(idsBuilder[i], substringIds[i]);
        }

        auto typesBuilder = request.initFileTypes(static_cast<unsigned int>([fileTypes count]));
        for (unsigned int i = 0; i < [fileTypes count]; i++) {
            typesBuilder.set(i, [fileTypes[i] UTF8String]);
        }

        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        return [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeCreateDynamicSubstringsRequest: %s", e.getDescription().cStr());
        return nil;
    }
}

NSData* _Nullable serializeCreateDynamicSubstringRequest(NSUUID* _Nonnull substringId, NSString* _Nonnull fileType) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterdocMessage::Builder builder = message.initRoot<OuterdocMessage>();
        auto request = builder.initCreateDynamicSubstring();
        setUUID(request.initSubstringId(), substringId);
        request.setFileType([fileType UTF8String]);

        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        return [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeCreateDynamicSubstringRequest: %s", e.getDescription().cStr());
        return nil;
    }
}

NSData* _Nullable serializeDisposeSubstringRequest(NSUUID* _Nonnull substringId) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterdocMessage::Builder builder = message.initRoot<OuterdocMessage>();
        auto request = builder.initDisposeSubstring();
        setUUID(request.initSubstringId(), substringId);

        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        return [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeDisposeSubstringRequest: %s", e.getDescription().cStr());
        return nil;
    }
}

// Serialization functions for OuterContent messages
NSData* _Nullable serializeCreateDynamicSubstringsResponse(NSArray<NSUUID*>* _Nonnull substringIds,
                                                  NSString* _Nonnull status,
                                                  NSArray<NSAttributedString*>* _Nonnull attributedStrings,
                                                  NSArray<NSArray<ElementInfo*>*>* _Nonnull elementsArrays,
                                                  NSArray<NSArray<CanvasInfo*>*>* _Nonnull canvasesArrays,
                                                  NSArray<NSArray<NSString*>*>* _Nonnull surfaceServiceNames) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterContentMessage::Builder builder = message.initRoot<OuterContentMessage>();
        auto response = builder.initCreateDynamicSubstringsResponse();

        auto idsBuilder = response.initSubstringIds(static_cast<unsigned int>([substringIds count]));
        for (unsigned int i = 0; i < [substringIds count]; i++) {
            setUUID(idsBuilder[i], substringIds[i]);
        }

        response.setStatus([status UTF8String]);

        auto resultsBuilder = response.initResults(static_cast<unsigned int>([attributedStrings count]));
        for (unsigned int i = 0; i < [attributedStrings count]; i++) {
            auto result = resultsBuilder[i];
            
            NSAttributedString* attributedString = attributedStrings[i];
            NSArray<ElementInfo*>* elements = elementsArrays[i];
            NSArray<CanvasInfo*>* canvases = canvasesArrays[i];
            
            // Serialize to NSKeyedArchiver data
            NSError* error = nil;
            NSData* archivedData = [NSKeyedArchiver archivedDataWithRootObject:attributedString
                                                         requiringSecureCoding:YES
                                                                         error:&error];
            if (archivedData && !error) {
                auto archivedBuilder = result.initAttributedString(1);
                archivedBuilder.set(0, kj::arrayPtr(reinterpret_cast<const kj::byte*>([archivedData bytes]), [archivedData length]));
            } else {
                if (error) {
                    NSLog(@"Error archiving NSAttributedString: %@", error);
                }
                result.initAttributedString(0);
            }
            
            // Serialize elements
            if (elements && [elements count] > 0) {
                auto elementsBuilder = result.initElements(static_cast<unsigned int>([elements count]));
                for (unsigned int j = 0; j < [elements count]; j++) {
                    ElementInfo *element = [elements objectAtIndex:j];
                    auto elementBuilder = elementsBuilder[j];
                    
                    elementBuilder.setLocation(element.range.location);
                    elementBuilder.setLength(element.range.length);
                    
                    auto uuidBuilder = elementBuilder.initId();
                    uuid_t bytes;
                    [element.elementID getUUIDBytes:bytes];
                    uint64_t high, low;
                    memcpy(&high, bytes, 8);
                    memcpy(&low, bytes + 8, 8);
                    uuidBuilder.setHigh(high);
                    uuidBuilder.setLow(low);
                }
            } else {
                result.initElements(0);
            }
            
            // Serialize canvases with service names
            NSArray<NSString*>* serviceNames = surfaceServiceNames[i];
            if (canvases && [canvases count] > 0) {
                auto canvasesBuilder = result.initCanvases(static_cast<unsigned int>([canvases count]));
                for (unsigned int j = 0; j < [canvases count]; j++) {
                    CanvasInfo *canvas = [canvases objectAtIndex:j];
                    auto canvasBuilder = canvasesBuilder[j];
                    
                    canvasBuilder.setLocation(canvas.location);
                    canvasBuilder.setWidth(canvas.width);
                    canvasBuilder.setHeight(canvas.height);
                    
                    auto uuidBuilder = canvasBuilder.initId();
                    uuid_t bytes;
                    [canvas.canvasID getUUIDBytes:bytes];
                    uint64_t high, low;
                    memcpy(&high, bytes, 8);
                    memcpy(&low, bytes + 8, 8);
                    uuidBuilder.setHigh(high);
                    uuidBuilder.setLow(low);
                    
                    // Set the service name for this canvas
                    if (j < [serviceNames count]) {
                        canvasBuilder.setServiceName([serviceNames[j] UTF8String]);
                    } else {
                        canvasBuilder.setServiceName("");
                    }
                }
            } else {
                result.initCanvases(0);
            }
        }

        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        return [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeCreateDynamicSubstringsResponse: %s", e.getDescription().cStr());
        return nil;
    }
}

NSData* _Nullable serializeCreateDynamicSubstringResponse(NSUUID* _Nonnull substringId,
                                                NSAttributedString* _Nonnull attributedString,
                                                NSArray<ElementInfo*>* _Nullable elements,
                                                NSArray<CanvasInfo*>* _Nullable canvases,
                                                NSArray<NSString*>* _Nonnull surfaceServiceNames) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterContentMessage::Builder builder = message.initRoot<OuterContentMessage>();
        auto response = builder.initCreateDynamicSubstringResponse();

        setUUID(response.initSubstringId(), substringId);
        
        // Serialize to NSKeyedArchiver data
        NSError* error = nil;
        NSData* archivedData = [NSKeyedArchiver archivedDataWithRootObject:attributedString
                                                     requiringSecureCoding:YES
                                                                     error:&error];
        if (archivedData && !error) {
            response.setAttributedString(kj::arrayPtr(reinterpret_cast<const kj::byte*>([archivedData bytes]), [archivedData length]));
        } else {
            if (error) {
                NSLog(@"Error archiving NSAttributedString: %@", error);
            }
        }
        
        // Serialize elements
        if (elements && [elements count] > 0) {
            auto elementsBuilder = response.initElements(static_cast<unsigned int>([elements count]));
            for (unsigned int i = 0; i < [elements count]; i++) {
                ElementInfo *element = [elements objectAtIndex:i];
                auto elementBuilder = elementsBuilder[i];
                
                elementBuilder.setLocation(element.range.location);
                elementBuilder.setLength(element.range.length);
                
                auto uuidBuilder = elementBuilder.initId();
                uuid_t bytes;
                [element.elementID getUUIDBytes:bytes];
                uint64_t high, low;
                memcpy(&high, bytes, 8);
                memcpy(&low, bytes + 8, 8);
                uuidBuilder.setHigh(high);
                uuidBuilder.setLow(low);
            }
        } else {
            response.initElements(0);
        }
        
        // Serialize canvases with service names
        if (canvases && [canvases count] > 0) {
            auto canvasesBuilder = response.initCanvases(static_cast<unsigned int>([canvases count]));
            for (unsigned int i = 0; i < [canvases count]; i++) {
                CanvasInfo *canvas = [canvases objectAtIndex:i];
                auto canvasBuilder = canvasesBuilder[i];
                
                canvasBuilder.setLocation(canvas.location);
                canvasBuilder.setWidth(canvas.width);
                canvasBuilder.setHeight(canvas.height);
                
                auto uuidBuilder = canvasBuilder.initId();
                uuid_t bytes;
                [canvas.canvasID getUUIDBytes:bytes];
                uint64_t high, low;
                memcpy(&high, bytes, 8);
                memcpy(&low, bytes + 8, 8);
                uuidBuilder.setHigh(high);
                uuidBuilder.setLow(low);
                
                // Set the service name for this canvas
                if (i < [surfaceServiceNames count]) {
                    canvasBuilder.setServiceName([surfaceServiceNames[i] UTF8String]);
                } else {
                    canvasBuilder.setServiceName("");
                }
            }
        } else {
            response.initCanvases(0);
        }

        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        return [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeCreateDynamicSubstringResponse: %s", e.getDescription().cStr());
        return nil;
    }
}

NSData* _Nullable serializeUpdateElementsMessage(NSUUID* _Nonnull substringId,
                                               NSArray<NSUUID*>* _Nonnull elementIds,
                                               NSArray<NSData*>* _Nonnull newContents) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterContentMessage::Builder builder = message.initRoot<OuterContentMessage>();
        auto update = builder.initUpdateElements();

        setUUID(update.initSubstringId(), substringId);

        auto idsBuilder = update.initElementIds(static_cast<unsigned int>([elementIds count]));
        for (unsigned int i = 0; i < [elementIds count]; i++) {
            setUUID(idsBuilder[i], elementIds[i]);
        }

        auto contentsBuilder = update.initNewContents(static_cast<unsigned int>([newContents count]));
        for (unsigned int i = 0; i < [newContents count]; i++) {
            NSData* content = newContents[i];
            contentsBuilder.set(i, kj::arrayPtr(reinterpret_cast<const kj::byte*>([content bytes]), [content length]));
        }

        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        return [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeUpdateElementsMessage: %s", e.getDescription().cStr());
        return nil;
    }
}

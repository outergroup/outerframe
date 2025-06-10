//
//  CrossPlatformSerialization.h
//

#ifndef cross_platform_serialization_hpp
#define cross_platform_serialization_hpp


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

#import <Foundation/NSAttributedString.h>
#import <Foundation/NSArray.h>

// Parses an outerdoc serialized document from a file descriptor in one pass.
// On success, writes the parsed attributedString to the provided out parameters and returns true.
// On failure, returns false and out parameters are unchanged.
bool readOuterdocFromPackedFile(
    int fd,
    NSAttributedString **outAttributedString
);

// Parses an outerdoc serialized document buffer in one pass, assuming it's in packed format.
// On success, writes the parsed attributedString to the provided out parameters and returns true.
// On failure, returns false and out parameters are unchanged.
bool readOuterdocFromPackedBuffer(
    NSData *data,
    NSAttributedString **outAttributedString
);

// Parses an outerdoc serialized document buffer in one pass, assuming it's in unpacked format.
// On success, writes the parsed attributedString to the provided out parameters and returns true.
// On failure, returns false and out parameters are unchanged.
bool readOuterdocFromBuffer(
    NSData *data,
    NSAttributedString **outAttributedString
);

// Write to file descriptor
bool writeOuterdocAttributedString(int fd,
                                   NSAttributedString* attributedString);

// Serializes an OuterdocAttributedString to data
NSData* serializeOuterdocAttributedString(
    NSAttributedString *attributedString
);

#endif

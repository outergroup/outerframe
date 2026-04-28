//
//  CrossPlatformSerialization.cpp
//

#import <AppKit/AppKit.h>
#import "CrossPlatformSerialization.hpp"
#import <Foundation/Foundation.h>
#include <unistd.h>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <kj/array.h>

#include "Document.capnp.h"

#include "OuterdocViewer-Swift.h"
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

void serializeAttributes(NSAttributedString *attributedString,
                         OuterdocAttributedString::Builder &attributedStringBuilder);

void deserializeAttributes(OuterdocAttributedString::Reader &contentReader, NSMutableAttributedString *attrString);



// Common helper function to process an OuterdocAttributedString::Reader
NSAttributedString* buildAttributedString(OuterdocAttributedString::Reader attributedStringProto) {
    NSString *nsString = [NSString stringWithUTF8String:attributedStringProto.getString().cStr()];
    NSMutableAttributedString *result = [[NSMutableAttributedString alloc] initWithString:nsString];

    // Use the helper function to deserialize attributes
    deserializeAttributes(attributedStringProto, result);

    // Process inline attachments: system UTTypes become NSTextAttachment, others become placeholder attachments
    if (attributedStringProto.hasAttachments()) {
        auto attachmentsList = attributedStringProto.getAttachments();
        for (auto attProto : attachmentsList) {
            NSString *origType = [NSString stringWithUTF8String:attProto.getFileType().cStr()];
            auto locationDataList = attProto.getLocations();
            for (auto locData : locationDataList) {
                NSUInteger location = locData.getLocation();
                NSRange range = NSMakeRange(location, 1);
                NSTextAttachment *attachment = nil;
                // Determine attachment type and set payload wrapper with appropriate extension
                UTType *uti = [UTType typeWithIdentifier:origType];
                if (uti != nil) {
                    // System-recognized UTType => plain NSTextAttachment
                    NSTextAttachment *textAttachment = [[NSTextAttachment alloc] init];
                    textAttachment.fileType = origType;
                    // Copy payload if present
                    auto dataBuf = locData.getData();
                    size_t byteCount = dataBuf.size() * sizeof(capnp::word);
                    if (byteCount > 0) {
                        NSData *payload = [NSData dataWithBytes:dataBuf.begin() length:byteCount];
                        NSFileWrapper *wrapper = [[NSFileWrapper alloc] initRegularFileWithContents:payload];
                        // Name the file using the UTI's preferred filename extension
                        NSString *filename = @"attachment";
                        NSString *ext = uti.preferredFilenameExtension;
                        if (ext.length > 0) {
                            filename = [filename stringByAppendingPathExtension:ext];
                        }
                        wrapper.preferredFilename = filename;
                        textAttachment.fileWrapper = wrapper;
                    }
                    attachment = textAttachment;
                } else {
                    // Placeholder for a dynamic substring
                    SubstringPlaceholderAttachment *ov = [[SubstringPlaceholderAttachment alloc] init];
                    ov.contentType = origType;
                    // Copy payload if present
                    auto dataBuf = locData.getData();
                    size_t byteCount = dataBuf.size() * sizeof(capnp::word);
                    if (byteCount > 0) {
                        NSData *payload = [NSData dataWithBytes:dataBuf.begin() length:byteCount];
                        NSFileWrapper *wrapper = [[NSFileWrapper alloc] initRegularFileWithContents:payload];
                        wrapper.preferredFilename = @"attachment.outerdocsubstring";
                        ov.fileWrapper = wrapper;
                    }
                    attachment = ov;
                }
                // Insert the attachment into the attributed string
                if (attachment && location < [result length]) {
                    [result addAttribute:NSAttachmentAttributeName value:attachment range:range];
                }
            }
        }
    }

    return result;
}

// Helper function to serialize attachments
static void serializeAttachments(
    NSAttributedString *attributedString,
    OuterdocAttributedString::Builder &attributedStringBuilder
) {
    // Map fileType to array of (range, data)
    NSMutableDictionary<NSString*, NSMutableArray*> *groups = [NSMutableDictionary dictionary];
    [attributedString enumerateAttribute:NSAttachmentAttributeName
                                inRange:NSMakeRange(0, attributedString.length)
                                options:0
                             usingBlock:^(id value, NSRange range, BOOL *stop) {
        NSTextAttachment *attachment = (NSTextAttachment *)value;
        // Determine fileType: for placeholders, use its content type, else use attachment.fileType
        NSString *fileType = nil;
        if ([attachment isKindOfClass:[SubstringPlaceholderAttachment class]]) {
            SubstringPlaceholderAttachment *ov = (SubstringPlaceholderAttachment *)attachment;
            fileType = ov.contentType;
            if (!fileType) {
                fileType = [attachment fileType];
            }
        } else {
            fileType = [attachment fileType];
        }
        if (fileType) {
            NSMutableArray *arr = groups[fileType];
            if (!arr) {
                arr = [NSMutableArray array];
                groups[fileType] = arr;
            }
            // Extract payload data if present
            NSData *payload = attachment.fileWrapper.regularFileContents ?: [NSData data];
            [arr addObject:@{ @"range": [NSValue valueWithRange:range], @"data": payload }];
        }
    }];

    // Serialize attachment groups
    if (groups.count > 0) {
        // Explicitly cast number of attachment groups to unsigned int
        unsigned int attachmentGroupCount = static_cast<unsigned int>(groups.count);
        auto attachmentsList = attributedStringBuilder.initAttachments(attachmentGroupCount);
        int idx = 0;
        for (NSString *fileType in groups) {
            auto multiBuilder = attachmentsList[idx++];
            multiBuilder.setFileType([fileType UTF8String]);
            NSArray *entries = groups[fileType];
            // Explicitly cast number of entries to unsigned int
            unsigned int entryCount = static_cast<unsigned int>(entries.count);
            auto locList = multiBuilder.initLocations(entryCount);
            // Iterate over entries using unsigned int count
            for (unsigned int j = 0; j < entryCount; ++j) {
                NSDictionary *info = entries[j];
                NSRange range = [info[@"range"] rangeValue];
                auto pairBuilder = locList[j];
                pairBuilder.setLocation(range.location);
                NSData *data = info[@"data"];
                size_t len = data.length;
                if (len > 0) {
                    // Explicitly cast payload length to unsigned int to avoid precision warnings
                    unsigned int dataLen = static_cast<unsigned int>(len);
                    auto blob = pairBuilder.initData(dataLen);
                    // Copy payload bytes into Cap'n Proto Data field
                    const capnp::byte *bytes = reinterpret_cast<const capnp::byte *>(data.bytes);
                    memcpy(blob.begin(), (void*)bytes, len);
                } else {
                    pairBuilder.initData(0);
                }
            }
        }
    } else {
        attributedStringBuilder.initAttachments(0);
    }
}

// MARK: - Unified document parsing result

static bool readOuterdocFromProto(
    OuterdocAttributedString::Reader attributedStringProto,
    NSAttributedString **outAttributedString
) {
    @try {
        // Parse attributed string content
        NSAttributedString *parsedString = buildAttributedString(attributedStringProto);

        // Output results
        if (outAttributedString) {
            *outAttributedString = parsedString;
        }
        return true;
    } @catch (NSException *ex) {
        NSLog(@"Exception in readOuterdocFromProto: %@", ex);
        return false;
    }
}

bool readOuterdocFromBuffer(
    NSData *data,
    NSAttributedString **outAttributedString
) {
    @try {
        if (data.length == 0) {
            NSLog(@"Error: Empty data in readOuterdocFromBuffer");
            return false;
        }

        try {
            // Process in non-packed format
            const capnp::word* wordPtr = reinterpret_cast<const capnp::word*>([data bytes]);
            size_t wordCount = [data length] / sizeof(capnp::word);

            if (wordCount > 0) {
                kj::ArrayPtr<const capnp::word> wordArray(wordPtr, wordCount);
                ::capnp::FlatArrayMessageReader message(wordArray);
                auto proto = message.getRoot<OuterdocAttributedString>();
                return readOuterdocFromProto(proto, outAttributedString);
            } else {
                NSLog(@"Invalid word count for non-packed format in readOuterdocFromBuffer");
                return false;
            }
        } catch (const kj::Exception& e) {
            NSLog(@"kj::Exception in readOuterdocFromBuffer: %s", e.getDescription().cStr());
            return false;
        } catch (const std::exception& e) {
            NSLog(@"std::exception in readOuterdocFromBuffer: %s", e.what());
            return false;
        } catch (...) {
            NSLog(@"Unknown C++ exception in readOuterdocFromBuffer");
            return false;
        }
    } @catch (NSException *ex) {
        NSLog(@"NSException in readOuterdocFromBuffer: %@", ex);
        return false;
    }
}

bool readOuterdocFromPackedBuffer(
    NSData *data,
    NSAttributedString **outAttributedString
) {
    @try {
        if (data.length == 0) {
            NSLog(@"Error: Empty data in readOuterdocFromPackedBuffer");
            return false;
        }

        try {
            // Load buffer into Cap'n Proto message reader
            kj::ArrayPtr<const capnp::byte> words(
                reinterpret_cast<const capnp::byte*>([data bytes]), [data length]);
            kj::ArrayInputStream inputStream(words);

            try {
                ::capnp::PackedMessageReader message(inputStream);
                auto proto = message.getRoot<OuterdocAttributedString>();
                return readOuterdocFromProto(proto, outAttributedString);
            } catch (const kj::Exception& e) {
                NSLog(@"kj::Exception in PackedMessageReader: %s", e.getDescription().cStr());
                return false;
            } catch (const std::exception& e) {
                NSLog(@"std::exception in capnp deserialization: %s", e.what());
                return false;
            } catch (...) {
                NSLog(@"Unknown C++ exception in capnp deserialization");
                return false;
            }
        } catch (const kj::Exception& e) {
            NSLog(@"kj::Exception in readOuterdocFromPackedBuffer: %s", e.getDescription().cStr());
            return false;
        } catch (const std::exception& e) {
            NSLog(@"std::exception in readOuterdocFromPackedBuffer: %s", e.what());
            return false;
        } catch (...) {
            NSLog(@"Unknown C++ exception in readOuterdocFromPackedBuffer");
            return false;
        }
    } @catch (NSException *ex) {
        NSLog(@"NSException in readOuterdocFromPackedBuffer: %@", ex);
        return false;
    }
}

bool readOuterdocFromPackedFile(
    int fd,
    NSAttributedString **outAttributedString
) {
    @try {
        ::capnp::PackedFdMessageReader message(fd);
        auto proto = message.getRoot<OuterdocAttributedString>();
        return readOuterdocFromProto(proto, outAttributedString);
    } @catch (NSException *ex) {
        NSLog(@"Exception in readOuterdocFromPackedFile: %@", ex);
        return false;
    }
}


NSData* serializeOuterdocAttributedString(
    NSAttributedString *attributedString
) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterdocAttributedString::Builder attributedStringBuilder = message.initRoot<OuterdocAttributedString>();

        // Use the helper function to serialize attributes
        serializeAttributes(attributedString, attributedStringBuilder);

        // Use the helper function to serialize attachments
        serializeAttachments(attributedString, attributedStringBuilder);

        // Serialize to data
        kj::Array<capnp::word> wordArray = capnp::messageToFlatArray(message);
        NSData *serializedData = [NSData dataWithBytes:wordArray.begin() length:wordArray.size() * sizeof(capnp::word)];
        return serializedData;
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in serializeOuterdocAttributedString: %s", e.getDescription().cStr());
        NSLog(@"WARNING: Returning empty data due to exception!");
        return [NSData data];
    }
}

bool writeOuterdocAttributedString(int fd,
                                     NSAttributedString* attributedString) {
    try {
        ::capnp::MallocMessageBuilder message;
        OuterdocAttributedString::Builder attributedStringProto = message.initRoot<OuterdocAttributedString>();

        // Use the helper function to serialize attributes
        serializeAttributes(attributedString, attributedStringProto);

        // Use the helper function to serialize attachments
        serializeAttachments(attributedString, attributedStringProto);

        // Write the message to the file descriptor
        ::capnp::writePackedMessageToFd(fd, message);
        return true;
    } catch (const kj::Exception& e) {
        NSLog(@"Exception in writeOuterdocAttributedString: %s", e.getDescription().cStr());
        return false;
    }
}

// Helper function to serialize NSAttributedString attributes to capnproto format
// Used in writeOuterdocAttributedString
void serializeAttributes(
    NSAttributedString *attributedString,
    OuterdocAttributedString::Builder &attributedStringBuilder
) {
    // Set the string text
    attributedStringBuilder.setString([[attributedString string] UTF8String]);

    // Collect attribute ranges
    __block NSMutableArray<NSValue *> *ranges = [NSMutableArray array];
    __block NSMutableArray<NSDictionary *> *attributesArray = [NSMutableArray array];

    [attributedString enumerateAttributesInRange:NSMakeRange(0, [attributedString length])
                                        options:0
                                     usingBlock:^(NSDictionary *attrs, NSRange range, BOOL *stop) {
        [ranges addObject:[NSValue valueWithRange:range]];
        [attributesArray addObject:attrs];
    }];

    // Create attributes list, casting count to unsigned int
    unsigned int attrCount = static_cast<unsigned int>(ranges.count);
    auto attributesList = attributedStringBuilder.initAttributes(attrCount);

    // Process each range's attributes
    for (unsigned int i = 0; i < attrCount; ++i) {
        NSRange range = [[ranges objectAtIndex:i] rangeValue];
        NSDictionary *attrs = [attributesArray objectAtIndex:i];

        auto rangeAttributesPair = attributesList[i];
        rangeAttributesPair.setLocation(range.location);
        rangeAttributesPair.setLength(range.length);

        // Count how many attributes we'll need to create
        int attributeCount = 0;
        if (attrs[NSFontAttributeName]) attributeCount++;
        if (attrs[NSForegroundColorAttributeName]) attributeCount++;
        if (attrs[NSBackgroundColorAttributeName]) attributeCount++;
        if (attrs[NSUnderlineStyleAttributeName]) attributeCount++;
        if (attrs[NSStrikethroughStyleAttributeName]) attributeCount++;
        if (attrs[NSSuperscriptAttributeName] && [attrs[NSSuperscriptAttributeName] intValue] > 0) attributeCount++;
        if (attrs[NSSuperscriptAttributeName] && [attrs[NSSuperscriptAttributeName] intValue] < 0) attributeCount++;
        if (attrs[NSLinkAttributeName]) attributeCount++;
        if (attrs[NSParagraphStyleAttributeName]) attributeCount++;
        if (attrs[NSBaselineOffsetAttributeName]) attributeCount++;

        // Initialize attributes list for this range
        auto attrsList = rangeAttributesPair.initAttributes(attributeCount);

        // Process each attribute type
        int currentAttrIndex = 0;

        // Handle font attributes
        NSFont *font = attrs[NSFontAttributeName];
        if (font) {
            auto attr = attrsList[currentAttrIndex++];
            auto fontBuilder = attr.initFont();

            // Set font properties
            fontBuilder.setFamily([[font familyName] UTF8String]);
            fontBuilder.setSize(font.pointSize);

            // Get font weight
            NSFontDescriptor *descriptor = [font fontDescriptor];
            NSDictionary *traits = [descriptor objectForKey:NSFontTraitsAttribute];
            NSNumber *weightValue = traits[NSFontWeightTrait];
            unsigned int weight = 400; // Default weight

            if (weightValue != nil) {
                // Convert NSFontWeightTrait value to an integer weight (100-900)
                CGFloat floatWeight = [weightValue floatValue];
                weight = (unsigned int)(((floatWeight + 1.0) / 2.0) * 800.0 + 100.0);
            }
            fontBuilder.setWeight(weight);

            // Check if italic
            BOOL isItalic = (([descriptor symbolicTraits] & NSFontDescriptorTraitItalic) != 0);
            fontBuilder.setItalic(isItalic);
        }

        // Handle foreground color
        NSColor *foregroundColor = attrs[NSForegroundColorAttributeName];
        if (foregroundColor) {
            // Try to map NSColor to one of our enum values
            Color colorEnum = Color::LABEL_COLOR;  // Default
            BOOL mappedSuccessfully = NO;

            if ([foregroundColor isEqual:[NSColor labelColor]]) {
                colorEnum = Color::LABEL_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor secondaryLabelColor]]) {
                colorEnum = Color::SECONDARY_LABEL_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor tertiaryLabelColor]]) {
                colorEnum = Color::TERTIARY_LABEL_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor quaternaryLabelColor]]) {
                colorEnum = Color::QUATERNARY_LABEL_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor systemRedColor]]) {
                colorEnum = Color::SYSTEM_RED_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor systemGreenColor]]) {
                colorEnum = Color::SYSTEM_GREEN_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor systemBlueColor]]) {
                colorEnum = Color::SYSTEM_BLUE_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor systemOrangeColor]]) {
                colorEnum = Color::SYSTEM_ORANGE_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor systemYellowColor]]) {
                colorEnum = Color::SYSTEM_YELLOW_COLOR;
                mappedSuccessfully = YES;
            } else if ([foregroundColor isEqual:[NSColor systemBrownColor]]) {
                colorEnum = Color::SYSTEM_BROWN_COLOR;
                mappedSuccessfully = YES;
            }

            // Only save if we could map to an enum value
            if (mappedSuccessfully) {
                auto attr = attrsList[currentAttrIndex++];
                attr.setForegroundColor(colorEnum);
            }
        }

        // Handle background color
        NSColor *backgroundColor = attrs[NSBackgroundColorAttributeName];
        if (backgroundColor) {
            auto attr = attrsList[currentAttrIndex++];
            auto colorBuilder = attr.initBackgroundColor();

            // Convert to RGB colorspace if needed
            NSColor *rgbColor = [backgroundColor colorUsingColorSpace:[NSColorSpace sRGBColorSpace]];
            if (rgbColor) {
                colorBuilder.setRed(rgbColor.redComponent * 255);
                colorBuilder.setGreen(rgbColor.greenComponent * 255);
                colorBuilder.setBlue(rgbColor.blueComponent * 255);
                colorBuilder.setAlpha(rgbColor.alphaComponent * 255);
            }
        }

        // Handle underline
        NSNumber *underlineStyle = attrs[NSUnderlineStyleAttributeName];
        if (underlineStyle && [underlineStyle intValue] > 0) {
            auto attr = attrsList[currentAttrIndex++];
            attr.setUnderline(true);
        }

        // Handle strikethrough
        NSNumber *strikethroughStyle = attrs[NSStrikethroughStyleAttributeName];
        if (strikethroughStyle && [strikethroughStyle intValue] > 0) {
            auto attr = attrsList[currentAttrIndex++];
            attr.setStrikethrough(true);
        }

        // Handle superscript/subscript
        NSNumber *superscriptValue = attrs[NSSuperscriptAttributeName];
        if (superscriptValue != nil) {
            int value = [superscriptValue intValue];
            if (value > 0) {
                auto attr = attrsList[currentAttrIndex++];
                attr.setSuperscript(true);
            } else if (value < 0) {
                auto attr = attrsList[currentAttrIndex++];
                attr.setSubscript(true);
            }
        }

        // Handle link
        NSURL *linkURL = attrs[NSLinkAttributeName];
        if (linkURL) {
            auto attr = attrsList[currentAttrIndex++];
            attr.setLink([[linkURL absoluteString] UTF8String]);
        }

        // Handle paragraph style
        NSParagraphStyle *paragraphStyle = attrs[NSParagraphStyleAttributeName];
        if (paragraphStyle) {
            auto attr = attrsList[currentAttrIndex++];
            auto paragraphBuilder = attr.initParagraph();

            // Set alignment
            uint8_t alignment;
            switch (paragraphStyle.alignment) {
                case NSTextAlignmentLeft: alignment = 0; break;
                case NSTextAlignmentRight: alignment = 1; break;
                case NSTextAlignmentCenter: alignment = 2; break;
                case NSTextAlignmentJustified: alignment = 3; break;
                default: alignment = 0;
            }
            paragraphBuilder.setAlignment(alignment);

            paragraphBuilder.setLineSpacing(paragraphStyle.lineSpacing);
            paragraphBuilder.setFirstLineHeadIndent(paragraphStyle.firstLineHeadIndent);
            paragraphBuilder.setHeadIndent(paragraphStyle.headIndent);
            paragraphBuilder.setTailIndent(paragraphStyle.tailIndent);
            paragraphBuilder.setLineHeightMultiple(paragraphStyle.lineHeightMultiple);
            paragraphBuilder.setMinimumLineHeight(paragraphStyle.minimumLineHeight);
            paragraphBuilder.setMaximumLineHeight(paragraphStyle.maximumLineHeight);
        }

        // Handle baseline offset
        NSNumber *baselineOffset = attrs[NSBaselineOffsetAttributeName];
        if (baselineOffset) {
            auto attr = attrsList[currentAttrIndex++];
            attr.setBaselineOffset([baselineOffset floatValue]);
        }
    }
}

// Helper function to deserialize attributes from capnproto to NSAttributedString
void deserializeAttributes(
    OuterdocAttributedString::Reader &contentReader,
    NSMutableAttributedString *attrString
) {
    auto attributesList = contentReader.getAttributes();
    for (auto rangeAttrs : attributesList) {
        NSRange range = NSMakeRange(rangeAttrs.getLocation(), rangeAttrs.getLength());

        // Process each attribute in this range
        auto attributes = rangeAttrs.getAttributes();
        NSMutableDictionary *nsAttributes = [NSMutableDictionary dictionary];

        for (auto attr : attributes) {
            switch (attr.which()) {
                case Attribute::FONT: {
                    auto fontReader = attr.getFont();
                    NSString *fontFamily = [NSString stringWithUTF8String:fontReader.getFamily().cStr()];
                    CGFloat fontSize = fontReader.getSize();
                    CGFloat fontWeight = (CGFloat)(fontReader.getWeight() - 100) / 800.0 * 2.0 - 1.0;
                    BOOL isItalic = fontReader.getItalic();

                    // Create font descriptor
                    NSFontDescriptor *descriptor = [NSFontDescriptor fontDescriptorWithFontAttributes:@{
                        NSFontFamilyAttribute: fontFamily,
                        NSFontTraitsAttribute: @{
                            NSFontWeightTrait: @(fontWeight),
                            NSFontSlantTrait: isItalic ? @(1.0) : @(0.0)
                        }
                    }];

                    // Create font and apply
                    NSFont *font = [NSFont fontWithDescriptor:descriptor size:fontSize];
                    if (font) {
                        [nsAttributes setObject:font forKey:NSFontAttributeName];
                    }
                    break;
                }
                case Attribute::FOREGROUND_COLOR: {
                    Color colorEnum = attr.getForegroundColor();
                    NSColor *color = nil;

                    // Map enum value back to NSColor
                    switch (colorEnum) {
                        case Color::LABEL_COLOR: color = [NSColor labelColor]; break;
                        case Color::SECONDARY_LABEL_COLOR: color = [NSColor secondaryLabelColor]; break;
                        case Color::TERTIARY_LABEL_COLOR: color = [NSColor tertiaryLabelColor]; break;
                        case Color::QUATERNARY_LABEL_COLOR: color = [NSColor quaternaryLabelColor]; break;
                        case Color::SYSTEM_RED_COLOR: color = [NSColor systemRedColor]; break;
                        case Color::SYSTEM_GREEN_COLOR: color = [NSColor systemGreenColor]; break;
                        case Color::SYSTEM_BLUE_COLOR: color = [NSColor systemBlueColor]; break;
                        case Color::SYSTEM_ORANGE_COLOR: color = [NSColor systemOrangeColor]; break;
                        case Color::SYSTEM_YELLOW_COLOR: color = [NSColor systemYellowColor]; break;
                        case Color::SYSTEM_BROWN_COLOR: color = [NSColor systemBrownColor]; break;
                        default: color = [NSColor labelColor]; break;
                    }

                    if (color) {
                        [nsAttributes setObject:color forKey:NSForegroundColorAttributeName];
                    }
                    break;
                }
                case Attribute::BACKGROUND_COLOR: {
                    auto colorReader = attr.getBackgroundColor();
                    float r = colorReader.getRed() / 255.0;
                    float g = colorReader.getGreen() / 255.0;
                    float b = colorReader.getBlue() / 255.0;
                    float a = colorReader.getAlpha() / 255.0;

                    NSColor *color = [NSColor colorWithSRGBRed:r green:g blue:b alpha:a];
                    if (color) {
                        [nsAttributes setObject:color forKey:NSBackgroundColorAttributeName];
                    }
                    break;
                }
                case Attribute::UNDERLINE: {
                    if (attr.getUnderline()) {
                        [nsAttributes setObject:@(NSUnderlineStyleSingle)
                                          forKey:NSUnderlineStyleAttributeName];
                    }
                    break;
                }
                case Attribute::STRIKETHROUGH: {
                    if (attr.getStrikethrough()) {
                        [nsAttributes setObject:@(NSUnderlineStyleSingle)
                                          forKey:NSStrikethroughStyleAttributeName];
                    }
                    break;
                }
                case Attribute::SUPERSCRIPT: {
                    if (attr.getSuperscript()) {
                        [nsAttributes setObject:@(1)
                                          forKey:NSSuperscriptAttributeName];
                    }
                    break;
                }
                case Attribute::SUBSCRIPT: {
                    if (attr.getSubscript()) {
                        [nsAttributes setObject:@(-1)
                                          forKey:NSSuperscriptAttributeName];
                    }
                    break;
                }
                case Attribute::LINK: {
                    NSString *urlString = [NSString stringWithUTF8String:attr.getLink().cStr()];
                    NSURL *url = [NSURL URLWithString:urlString];
                    if (url) {
                        [nsAttributes setObject:url
                                          forKey:NSLinkAttributeName];
                    }
                    break;
                }
                case Attribute::PARAGRAPH: {
                    auto paragraphReader = attr.getParagraph();
                    NSMutableParagraphStyle *paragraphStyle = [[NSMutableParagraphStyle alloc] init];

                    // Set alignment
                    uint8_t alignmentValue = paragraphReader.getAlignment();
                    NSTextAlignment alignment;
                    switch (alignmentValue) {
                        case 0: alignment = NSTextAlignmentLeft; break;
                        case 1: alignment = NSTextAlignmentRight; break;
                        case 2: alignment = NSTextAlignmentCenter; break;
                        case 3: alignment = NSTextAlignmentJustified; break;
                        default: alignment = NSTextAlignmentLeft; break;
                    }
                    paragraphStyle.alignment = alignment;

                    // Set other paragraph properties
                    paragraphStyle.lineSpacing = paragraphReader.getLineSpacing();
                    paragraphStyle.firstLineHeadIndent = paragraphReader.getFirstLineHeadIndent();
                    paragraphStyle.headIndent = paragraphReader.getHeadIndent();
                    paragraphStyle.tailIndent = paragraphReader.getTailIndent();
                    paragraphStyle.lineHeightMultiple = paragraphReader.getLineHeightMultiple();
                    paragraphStyle.minimumLineHeight = paragraphReader.getMinimumLineHeight();
                    paragraphStyle.maximumLineHeight = paragraphReader.getMaximumLineHeight();

                    [nsAttributes setObject:paragraphStyle
                                      forKey:NSParagraphStyleAttributeName];
                    break;
                }
                case Attribute::BASELINE_OFFSET: {
                    float baselineOffset = attr.getBaselineOffset();
                    [nsAttributes setObject:@(baselineOffset)
                                      forKey:NSBaselineOffsetAttributeName];
                    break;
                }
                default:
                    // Unknown attribute type
                    break;
            }
        }

        // Apply all attributes to this range
        if ([nsAttributes count] > 0) {
            [attrString setAttributes:nsAttributes range:range];
        }
    }
}


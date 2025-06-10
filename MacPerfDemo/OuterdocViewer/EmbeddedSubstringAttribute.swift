//
//  EmbeddedSubstringAttribute.swift
//

import Foundation

/// Attribute value representing an embedded dynamic substring.
public class EmbeddedSubstringAttribute: NSObject, NSCopying {
    public let contentType: String
    public let attributedString: NSAttributedString
    public let substringId: UUID

    public init(contentType: String,
                attributedString: NSAttributedString,
                substringId: UUID) {
        self.contentType = contentType
        self.attributedString = attributedString
        self.substringId = substringId
    }

    public func copy(with zone: NSZone? = nil) -> Any {
        // Create a new instance with the same values
        return EmbeddedSubstringAttribute(contentType: self.contentType,
                                          attributedString: self.attributedString,
                                          substringId: self.substringId)
    }
}

/// Custom attribute key for dynamic substrings
public extension NSAttributedString.Key {
    static let embeddedSubstring = NSAttributedString.Key("dev.outergroup.embeddedSubstring")
    static let element = NSAttributedString.Key("dev.outergroup.element")
}

/// Attribute value representing a dynamic element within an embedded dynamic substring.
public class ElementAttribute: NSObject, NSCopying {
    public let elementID: UUID

    public init(elementID: UUID) {
        self.elementID = elementID
    }

    public func copy(with zone: NSZone? = nil) -> Any {
        // Create a new instance with the same elementID
        return ElementAttribute(elementID: self.elementID)
    }
}

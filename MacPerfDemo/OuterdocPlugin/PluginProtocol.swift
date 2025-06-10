//
//  PluginProtocol.swift
//

import Foundation
import IOSurface


/// Protocol for animation schedulers that manage animation frames for figures
@objc public protocol AnimationScheduler {
    /// Request the next animation frame for a figure
    func requestAnimationFrame()
}


/// Protocol that outerdoc plugins conform to in order to declare which file types they handle.
@objc public protocol OuterdocPlugin {
    /// Returns the list of outerdoc-substring-based content types that this plugin supports.
    @objc static func suppertedOuterdocContentTypes() -> [String]

    /// Creates an instance of a dynamic substring of the given content type.
    /// The generator provides both the initial outerdoc data and animation functionality.
    /// - Parameter fileType: A text-based (outerdoc) file type identifier supported by this plugin.
    /// - Returns: An instance, or nil on failure.
    @objc static func createDynamicSubstring(contentType: String) -> OuterdocDynamicSubstring?
}

/// Container class for element updates
@objc public class OuterdocElementUpdates: NSObject {
    @objc public let elementIds: [UUID]
    @objc public let newStrings: [NSAttributedString]

    @objc public init(elementIds: [UUID], newStrings: [NSAttributedString]) {
        self.elementIds = elementIds
        self.newStrings = newStrings
        super.init()
    }
}

/// Container class for initial outerdoc substring data
@objcMembers public class OuterdocSubstring: NSObject {
    public let attributedString: NSAttributedString
    // If the below are instead defined as [ElementInfo] and [CanvasInfo],
    // they can't be accessed by Swift callers of this type, for some reason.
    public let elements: NSArray
    public let canvases: NSArray
    public let canvasSurfaces: [IOSurface]

    @objc public init(attributedString: NSAttributedString,
                           elements: [ElementInfo] = [],
                           canvases: [CanvasInfo] = [],
                           canvasSurfaces: [IOSurface] = []) {
        self.attributedString = attributedString
        self.elements = elements as NSArray
        self.canvases = canvases as NSArray
        self.canvasSurfaces = canvasSurfaces
        super.init()
    }
}

/// Protocol for dynamic outerdoc substrings
@objc public protocol OuterdocDynamicSubstring {
    /// Generate the next animation frame for the outerdoc substring
    /// - Parameter targetTimestamp: The intended presentation time for this frame
    /// - Returns: Object containing arrays of element IDs and their corresponding serialized content
    @objc func generateNextFrame(targetTimestamp: CFTimeInterval) -> OuterdocElementUpdates?

    /// Set the animation scheduler
    /// - Parameter scheduler: The animation scheduler to use for this substring
    @objc func setAnimationScheduler(scheduler: AnimationScheduler?)

    /// Get the initial outerdoc substring for this substring
    /// - Returns: Structured outerdoc substring data including IOSurfaces
    @objc func getInitialSubstring() -> OuterdocSubstring

    /// Update canvases using the provided timestamp
    /// - Parameter targetTimestamp: The intended presentation time for this frame
    @objc func updateCanvases(targetTimestamp: CFTimeInterval)
}

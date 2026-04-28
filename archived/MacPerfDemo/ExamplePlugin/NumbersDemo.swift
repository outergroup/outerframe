//
//  NumbersDemo.swift
//

import Foundation
import OuterdocPlugin
import AppKit

// Configuration
fileprivate let N_NUMBERS = 20

class NumbersDemo: OuterdocDynamicSubstring {
    
    private let instanceID = UUID()
    private var elementIds: [String: UUID]
    private var elementRanges: [String: NSRange] = [:]
    private var cachedElementContents: [UUID: NSAttributedString] = [:]
    
    // Animation properties
    private weak var animationScheduler: AnimationScheduler?
    private var animationStartTime: Double = 0
    
    init() {
        var elementIds = [String: UUID]()
        for i in 0..<N_NUMBERS {
            elementIds["number\(i)"] = UUID()
        }
        self.elementIds = elementIds
        
        let fragmentContainer = generateFragmentContainer()
        self.elementRanges = fragmentContainer.elementRanges
    }
    
    func getInitialSubstring() -> OuterdocSubstring {
        let fragmentContainer = generateFragmentContainer()
        let container = fragmentContainer.attributedString
        
        var elements: [ElementInfo] = []
        
        // Add element info for each number
        for i in 0..<N_NUMBERS {
            let numberID = elementIds["number\(i)"]!
            
            if let numberRange = fragmentContainer.elementRanges["number\(i)"] {
                let elementInfo = ElementInfo()
                elementInfo.elementID = numberID
                elementInfo.range = numberRange
                elements.append(elementInfo)
            }
        }
        
        return OuterdocSubstring(
            attributedString: container,
            elements: elements,
            canvases: [],  // No canvases
            canvasSurfaces: []  // No canvas surfaces
        )
    }
    
    func generateNextFrame(targetTimestamp: CFTimeInterval) -> OuterdocElementUpdates? {
        var updates: [UUID: NSAttributedString] = [:]
        
        // Update each number element
        for (elementType, elementID) in elementIds {
            if elementType.hasPrefix("number") {
                // Extract index
                let indexString = elementType.dropFirst("number".count)
                if let index = Int(indexString) {
                    // Generate the element content
                    let content = generateNumberContent(index: index, targetTimestamp: targetTimestamp)
                    
                    // Check if content has changed
                    let cached = cachedElementContents[elementID]
                    if cached?.string != content.string {
                        cachedElementContents[elementID] = content
                        updates[elementID] = content
                    }
                }
            }
        }
        
        var ret: OuterdocElementUpdates? = nil
        if !updates.isEmpty {
            var elementIds = [UUID]()
            var newStrings = [NSAttributedString]()
            for (elementId, newString) in updates {
                elementIds.append(elementId)
                newStrings.append(newString)
            }
            ret = OuterdocElementUpdates(elementIds: elementIds, newStrings: newStrings)
        }
        
        // Request next animation frame
        animationScheduler?.requestAnimationFrame()
        
        return ret
    }
    
    func updateCanvases(targetTimestamp: CFTimeInterval) {
        // No canvases to update
    }
    
    func setAnimationScheduler(scheduler: AnimationScheduler?) {
        self.animationScheduler = scheduler
        self.animationStartTime = CACurrentMediaTime()
        
        // Request first animation frame
        scheduler?.requestAnimationFrame()
    }
    
    // Generate animated number content
    private func generateNumberContent(index: Int, targetTimestamp: CFTimeInterval) -> NSAttributedString {
        let elapsedTime = targetTimestamp - animationStartTime
        
        // Simple sine wave with different frequencies
        let frequency = 0.5 + Double(index) * 0.1
        let value = (sin(elapsedTime * frequency) + 1.0) / 2.0
        
        // Format the number with two decimal places
        let text = String(format: "%4.2f", value)
        
        let monospacedFont = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: monospacedFont,
            .foregroundColor: NSColor.labelColor
        ]
        
        return NSAttributedString(string: text, attributes: attributes)
    }
}

private struct FragmentContainer {
    var attributedString: NSAttributedString
    var elementRanges: [String: NSRange]
}

private func generateFragmentContainer() -> FragmentContainer {
    let monospacedFont = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
    let attributedString = NSMutableAttributedString()
    var elementRanges = [String: NSRange]()
    
    // Create centered paragraph style
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.alignment = .center
    
    for i in 0..<N_NUMBERS {
        // Create label
        let labelText = String(format: "Parameter %2d: ", i + 1)
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: monospacedFont,
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraphStyle
        ]
        
        // Create initial number value
        let numberText = "0.50"
        let numberAttributes: [NSAttributedString.Key: Any] = [
            .font: monospacedFont,
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraphStyle
        ]
        
        // Build the line
        let lineString = NSMutableAttributedString()
        lineString.append(NSAttributedString(string: labelText, attributes: labelAttributes))
        
        let numberStart = lineString.length
        lineString.append(NSAttributedString(string: numberText, attributes: numberAttributes))
        let numberLength = numberText.count
        
        // Store element range
        let startPosition = attributedString.length
        attributedString.append(lineString)
        
        // Add newline
        attributedString.append(NSAttributedString(string: "\n", attributes: [
            .font: monospacedFont,
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraphStyle
        ]))
        
        // Adjust range for position in full string
        elementRanges["number\(i)"] = NSRange(location: startPosition + numberStart, length: numberLength)
    }
    
    return FragmentContainer(attributedString: attributedString, elementRanges: elementRanges)
}

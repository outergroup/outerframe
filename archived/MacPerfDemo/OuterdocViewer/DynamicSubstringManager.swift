//
//  DynamicSubstringManager.swift
//

import Foundation
import AppKit

/// Class to manage a single display link for all dynamic substrings
class DynamicSubstringDisplayLinks {
    static let shared = DynamicSubstringDisplayLinks()

    private let lock = NSLock()
    private var displayLink: CADisplayLink?
    private var hasPendingRequest = false
    
    // Shared memory integration
    private var requestListenerThread: Thread?
    private var completionListenerThread: Thread?
    private var isListening = false

    @objc func displayLinkFired(_ displayLink: CADisplayLink) {
        var shouldProcessFrame = false

        lock.lock()
        if hasPendingRequest {
            hasPendingRequest = false
            shouldProcessFrame = true
        }
        lock.unlock()

        // If we have a pending request, generate frames for all awaiting substrings
        if shouldProcessFrame {
            DynamicSubstringManager.shared.generateFrame(targetTimestamp: displayLink.targetTimestamp)
        }
    }

    func ensureDisplayLinkExists() {
        lock.lock()
        if displayLink == nil {
            if let screen = NSScreen.main {
                displayLink = screen.displayLink(target: self, selector: #selector(displayLinkFired(_:)))
                displayLink?.add(to: .main, forMode: .common)
            }
        }
        
        // Start listening for shared memory animation requests if not already started
        if !isListening {
            startSharedMemoryListener()
        }
        lock.unlock()
    }
    
    private func startSharedMemoryListener() {
        guard !isListening else { return }
        
        isListening = true
        
        // Start thread to listen for animation frame requests
        requestListenerThread = Thread {
            self.sharedMemoryRequestLoop()
        }
        requestListenerThread?.name = "AnimationFrameRequestListener"
        requestListenerThread?.start()
        
        // Start thread to listen for render completion
        completionListenerThread = Thread {
            self.sharedMemoryCompletionLoop()
        }
        completionListenerThread?.name = "RenderCompletionListener"
        completionListenerThread?.start()
    }
    
    private func sharedMemoryRequestLoop() {
        // Try to connect to OuterContent's shared memory
        guard let sharedMemory = OuterContentConnection.shared.getSharedMemory() else {
            print("DynamicSubstringDisplayLinks: No shared memory available, using message-based system")
            return
        }
        
        while isListening {
            // Wait for animation frame requests from OuterContent
            if sharedMemory.waitForAnimationFrameRequest() {
                // Check if we're still listening (in case we were woken up for cleanup)
                if isListening {
                    // Trigger the display link by calling the existing requestFrame method
                    self.requestFrame()
                }
            }
        }
    }
    
    private func sharedMemoryCompletionLoop() {
        // Try to connect to OuterContent's shared memory
        guard let sharedMemory = OuterContentConnection.shared.getSharedMemory() else {
            print("DynamicSubstringDisplayLinks: No shared memory available for completion listening")
            return
        }

        while isListening {
            // Wait for render completion from OuterContent
            guard sharedMemory.waitForRenderComplete() else {
                continue
            }

            // Check if we're still listening (in case we were woken up for cleanup)
            guard isListening else {
                break
            }

            // Atomically get and clear substring updates, then process them
            let substringUpdates = sharedMemory.getAndClearSubstringUpdates()
            guard !substringUpdates.isEmpty else {
                continue
            }

            DispatchQueue.main.async {
                CATransaction.begin()
                CATransaction.setDisableActions(true)
                for substringId in substringUpdates {
                    if let canvasIDs = GlobalCanvasRegistry.shared.canvasesForSubstring[substringId] {
                        for canvasID in canvasIDs {
                            guard let canvasView = GlobalCanvasRegistry.shared.getView(for: canvasID) else {
                                print("DynamicSubstringDisplayLinks: No canvas view found for canvasID: \(canvasID)")
                                continue
                            }

                            canvasView.onIOSurfaceUpdate()
                        }
                    }
                }
                CATransaction.commit()
            }
        }
    }

    func requestFrame() {
        ensureDisplayLinkExists()
        
        lock.lock()
        hasPendingRequest = true
        lock.unlock()
    }

    func cleanup() {
        var linkToInvalidate: CADisplayLink? = nil

        lock.lock()
        linkToInvalidate = displayLink
        displayLink = nil
        hasPendingRequest = false
        
        // Stop the shared memory listener thread
        isListening = false
        lock.unlock()

        // Wake up both listener threads if they're waiting on semaphores
        if let sharedMemory = OuterContentConnection.shared.getSharedMemory() {
            sharedMemory.requestAnimationFrame() // Wake up request listener thread
            sharedMemory.signalRenderComplete() // Wake up completion listener thread
        }

        // Invalidate outside the lock to prevent deadlocks
        linkToInvalidate?.invalidate()
        
        // Wait for both listener threads to finish
        requestListenerThread?.cancel()
        requestListenerThread = nil
        completionListenerThread?.cancel()
        completionListenerThread = nil
    }

    deinit {
        cleanup()
    }
}

/// Manager class for animated document substrings
class DynamicSubstringManager {
    /// Shared singleton instance
    static let shared = DynamicSubstringManager()

    // Map of substring IDs to their handlers for element updates
    private var substringElementHandlers: [UUID: ([UUID: NSAttributedString]) -> Void] = [:]

    private let substringLock = NSLock()


    // Private initializer to enforce singleton pattern
    private init() {}

    // Map of substring IDs to their file types
    var substringFileTypes: [UUID: String] = [:]

    // Using ObjectIdentifier as key instead of NSTextStorage to avoid using mutable objects as dictionary keys
    // This prevents issues when the NSTextStorage object changes and its hash value changes
    private var textStorageToSubstrings: [ObjectIdentifier: Set<UUID>] = [:]
    private var weakTextStorages: [ObjectIdentifier: WeakRef<NSTextStorage>] = [:]
    private let mapLock = NSLock()

    // Wrapper class to hold weak references
    private class WeakRef<T: AnyObject> {
        weak var value: T?

        init(_ value: T) {
            self.value = value
        }
    }

    func register(substringId: UUID, fileType: String, textStorage: NSTextStorage) {
        // Register substring type
        substringFileTypes[substringId] = fileType

        // Ensure display link exists (shared for all substrings)
        DynamicSubstringDisplayLinks.shared.ensureDisplayLinkExists()

        // Get a stable identifier for the text storage
        let textStorageID = ObjectIdentifier(textStorage)

        // Track this substring with its text storage for cleanup
        mapLock.lock()
        // Store a weak reference to the text storage
        weakTextStorages[textStorageID] = WeakRef(textStorage)

        // Create or update the set of substrings for this text storage
        if textStorageToSubstrings[textStorageID] == nil {
            textStorageToSubstrings[textStorageID] = []
        }
        textStorageToSubstrings[textStorageID]?.insert(substringId)
        mapLock.unlock()

        // Set up the frame handler
        setupFrameHandler(substringId: substringId, textStorage: textStorage)
    }

    /// Clean up all dynamic substrings associated with a text storage
    /// - Parameter textStorage: The text storage being closed or deallocated
    func cleanupSubstrings(for textStorage: NSTextStorage) {
        // Get a stable identifier for the text storage
        let textStorageID = ObjectIdentifier(textStorage)

        // Get the list of substrings to clean up
        mapLock.lock()
        let substringsToCleanup = textStorageToSubstrings[textStorageID] ?? []

        // Remove this text storage from our tracking maps
        textStorageToSubstrings.removeValue(forKey: textStorageID)
        weakTextStorages.removeValue(forKey: textStorageID)
        mapLock.unlock()

        // Clean up each substring
        for substringId in substringsToCleanup {
            print("DynamicSubstringManager: Cleaning up substring \(substringId) during document close")

            // Remove from substring file types
            substringFileTypes.removeValue(forKey: substringId)

            // Clean up shared display link if no substrings remain
            if substringFileTypes.isEmpty {
                DynamicSubstringDisplayLinks.shared.cleanup()
            }

            // Remove handler from UIProcessHandler
            removeDocsubstringHandler(substringId: substringId)

            // Dispose the substring in the XPC service
            OuterContentConnection.shared.disposeSubstring(substringId: substringId)
            print("Disposed dynamic substring with ID \(substringId)")
        }
    }

    /// Set up a handler to process new animation frames
    /// - Parameters:
    ///   - substringId: The unique identifier for the substring to animate
    ///   - textStorage: The text storage containing the substring
    func setupFrameHandler(substringId: UUID, textStorage: NSTextStorage) {
        // Register an element handler for surgical updates to specific parts of the substring
        registerSubstringElementHandler(substringId: substringId) { [weak self] elementUpdates in
            DispatchQueue.main.async {
                // Find the substring in the text storage
                var substringRange = NSRange(location: 0, length: 0)
                var substringAttribute: EmbeddedSubstringAttribute? = nil
                textStorage.enumerateAttribute(.embeddedSubstring,
                                             in: NSRange(location: 0, length: textStorage.length),
                                             options: []) { value, range, stop in
                    if let attr = value as? EmbeddedSubstringAttribute, attr.substringId == substringId {
                        substringAttribute = attr
                        substringRange = range
                        stop.pointee = true
                    }
                }

                guard let substringAttribute else {
                    print("substring with ID \(substringId) no longer found in document - stopping animation")
                    self?.cleanupsubstring(substringId: substringId, textStorage: textStorage)
                    return
                }

                // Start batch editing
                textStorage.beginEditing()

                // Find all .element attributes within this substring
                // This is the proper way to locate elements by their UUID
                for (elementID, newContent) in elementUpdates {
                    var elementUpdated = false
                    textStorage.enumerateAttribute(.element,
                                                   in: NSMakeRange(0, textStorage.length), //  substringRange,
                                                 options: []) { value, elementRange, stop in
                        if let elementAttr = value as? ElementAttribute,
                           elementAttr.elementID == elementID {
                            substringRange.length += newContent.length - elementRange.length

                            textStorage.replaceCharacters(in: elementRange, with: newContent)

                            // Reapply the element attribute (same range since length didn't change)
                            textStorage.addAttribute(.element,
                                                     value: ElementAttribute(elementID: elementID),
                                                     range: NSRange(location: elementRange.location, length: newContent.length))
                            textStorage.addAttribute(.embeddedSubstring,
                                                     value: substringAttribute,
                                                     range: NSRange(location: elementRange.location, length: newContent.length))

                            elementUpdated = true

                            // Stop looking after finding the matching element
                            stop.pointee = true
                        }
                    }

                    if !elementUpdated {
                        print("Warning: Element with ID \(elementID) not found in document")
                    }
                }

                textStorage.endEditing()
            }
        }
    }

    /// Clean up a single substring when it's no longer in the document
    /// - Parameters:
    ///   - substringId: substring ID to clean up
    ///   - textStorage: Associated text storage
    private func cleanupsubstring(substringId: UUID, textStorage: NSTextStorage) {
        // Remove from our tracking maps
        substringFileTypes.removeValue(forKey: substringId)

        // Clean up shared display link if no substrings remain
        if substringFileTypes.isEmpty {
            DynamicSubstringDisplayLinks.shared.cleanup()
        }

        // Get a stable identifier for the text storage
        let textStorageID = ObjectIdentifier(textStorage)

        mapLock.lock()
        textStorageToSubstrings[textStorageID]?.remove(substringId)

        // If this was the last substring for this text storage, remove the weak reference too
        if let substrings = textStorageToSubstrings[textStorageID], substrings.isEmpty {
            textStorageToSubstrings.removeValue(forKey: textStorageID)
            weakTextStorages.removeValue(forKey: textStorageID)
        }
        mapLock.unlock()
        removeDocsubstringHandler(substringId: substringId)

        // Dispose in content process
        OuterContentConnection.shared.disposeSubstring(substringId: substringId)
    }

    /// Removes any stale references from the tracking dictionaries
    /// This should be called periodically to clean up references to deallocated text storages
    func cleanupStaleReferences() {
        mapLock.lock()
        defer { mapLock.unlock() }

        // Find keys where the weak reference has been deallocated
        let staleKeys = weakTextStorages.compactMap { (key, weakRef) -> ObjectIdentifier? in
            if weakRef.value == nil {
                return key
            }
            return nil
        }

        // Remove stale references
        for key in staleKeys {
            textStorageToSubstrings.removeValue(forKey: key)
            weakTextStorages.removeValue(forKey: key)
        }

        if !staleKeys.isEmpty {
            print("DynamicSubstringManager: Cleaned up \(staleKeys.count) stale text storage references")
        }
    }

    func substringFrameReady(substringId: UUID, elementIds: [UUID], newStrings: [NSAttributedString]) {
        substringLock.lock()
        defer { substringLock.unlock() }

        if let handler = substringElementHandlers[substringId] {
            var updateDict = [UUID: NSAttributedString]()
            for i in 0..<min(elementIds.count, newStrings.count) {
                let elementId = elementIds[i]
                let newString = newStrings[i]
                updateDict[elementId] = newString
            }

            if !updateDict.isEmpty {
                handler(updateDict)
            } else {
                print("UIProcessHandler: No valid element updates could be deserialized")
            }
        } else {
            print("UIProcessHandler: No handlers registered (neither element nor substring) for ID: \(substringId)")
        }
    }

    // Called when the display link fires and any substrings need frames
    func generateFrame(targetTimestamp: CFTimeInterval = CACurrentMediaTime()) {
        OuterContentConnection.shared.generateFrameOptimized(targetTimestamp: targetTimestamp)
    }

    // Register a handler to receive element updates for a doc substring
    func registerSubstringElementHandler(substringId: UUID, handler: @escaping ([UUID: NSAttributedString]) -> Void) {
        substringLock.lock()
        defer { substringLock.unlock() }

        print("registerSubstringElementHandler: Registering handler for substring ID: \(substringId)")

        // Check if replacing an existing handler
        if substringElementHandlers[substringId] != nil {
            print("Note: Replacing existing element handler")
        }

        substringElementHandlers[substringId] = handler
        print("Element handler registered successfully")
    }

    // Remove all handlers for a doc substring
    func removeDocsubstringHandler(substringId: UUID) {
        substringLock.lock()
        defer { substringLock.unlock() }

        print("removeDocsubstringHandler: Removing handlers for substring ID: \(substringId)")

        // Check what we're removing
        let hadElementHandler = substringElementHandlers[substringId] != nil

        // Remove all handlers and pending animation
        substringElementHandlers.removeValue(forKey: substringId)

        // Report what was removed
        print("Handlers removed: element=\(hadElementHandler)")
    }
}

//
//  GlobalCanvasRegistry.swift
//

import AppKit

/// A shared registry for mapping canvas identifiers to their views.
/// This facilitates communication between the main app and the content process.
public class GlobalCanvasRegistry {
    public static let shared = GlobalCanvasRegistry()
    
    private init() {
        // Private initializer to enforce singleton pattern
    }

    // Thread safety for attachment view operations
    let lock = NSLock()

    // Map of canvas IDs to their views
    var views: [UUID: CanvasAttachmentView] = [:]

    // Ordered list of canvases in each substring
    var canvasesForSubstring: [UUID: [UUID]] = [:]

    // Map of canvas IDs to their substring IDs
    var substringMap: [UUID: UUID] = [:]

    func registerView(canvasID: UUID, substringId: UUID, view: NSView) {
        print("GlobalCanvasRegistry.registerView: Begin with canvasID: \(canvasID)")
        
        lock.lock()
        
        print("GlobalCanvasRegistry.registerView: State BEFORE registration - views count: \(views.count), substringMap count: \(substringMap.count)")
        
        // Log view type info
        print("GlobalCanvasRegistry.registerView: View type: \(type(of: view)), isKind of CanvasAttachmentView: \(view is CanvasAttachmentView)")
        
        if let canvasView = view as? CanvasAttachmentView {
            views[canvasID] = canvasView
            substringMap[canvasID] = substringId
            print("GlobalCanvasRegistry.registerView: Successfully registered view for canvasID: \(canvasID), substringId: \(substringId)")
        } else {
            print("GlobalCanvasRegistry.registerView: Failed to register view - not a CanvasAttachmentView")
        }
        
        print("GlobalCanvasRegistry.registerView: State AFTER registration - views count: \(views.count), substringMap count: \(substringMap.count)")
        
        lock.unlock()
    }

    func getView(for canvasID: UUID) -> CanvasAttachmentView? {
        lock.lock()
        defer { lock.unlock() }

        return views[canvasID]
    }

    func getSubstringId(for canvasID: UUID) -> UUID? {
        lock.lock()
        defer { lock.unlock() }

        return substringMap[canvasID]
    }

    func removeCanvas(canvasID: UUID) {
        lock.lock()
        defer { lock.unlock() }

        views.removeValue(forKey: canvasID)
        substringMap.removeValue(forKey: canvasID)
    }

    // Register just the mapping between canvas and substring, without a view
    // This is useful for early registration before views are created
    func registerCanvasMapping(canvasID: UUID, substringId: UUID) {
        print("GlobalCanvasRegistry.registerCanvasMapping: Begin with canvasID: \(canvasID), substringId: \(substringId)")
        
        lock.lock()
        
        // Store the canvas-to-substring mapping
        substringMap[canvasID] = substringId

        if let current = canvasesForSubstring[substringId] {
            canvasesForSubstring[substringId] = current + [canvasID]
        } else {
            canvasesForSubstring[substringId] = [canvasID]
        }

        print("GlobalCanvasRegistry.registerCanvasMapping: Registered mapping, substringMap count: \(substringMap.count)")
        
        lock.unlock()
    }
    
    // Get all canvas views for a specific substring
    func getViewsForFragment(substringId: UUID) -> [UUID: CanvasAttachmentView] {
        lock.lock()
        defer { lock.unlock() }

        var result = [UUID: CanvasAttachmentView]()

        // Get all canvas views from the registry that match this substring ID
        for (canvasID, substringId_) in substringMap {
            if substringId_ == substringId,
               let view = views[canvasID] {
                result[canvasID] = view
            }
        }

        return result
    }
}

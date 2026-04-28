//
//  SharedRenderMemory.swift
//  Created for high-performance render loop optimization.
//

import Foundation
import QuartzCore.CAMediaTiming

/// Layout of shared memory region for render loop optimization
public struct SharedRenderMemory {
    static let size = 4096 // 4KB should be sufficient for control data
    
    /// Header containing metadata and synchronization info
    struct Header {
        var version: UInt32 = 1
        var processID: pid_t = 0
        var frameNumber: UInt64 = 0
        var targetTimestamp: CFTimeInterval = 0
        var substringUpdateCount: UInt32 = 0
        var padding: (UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0) // Ensure 8-byte alignment
    }
    
    
    /// Fragment update information - just needs substringId since IOSurface is already set up
    struct UpdateSubstringCanvasesNotification {
        var substringId: uuid_t = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        var padding: (UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0) // Maintain alignment
    }
    
    static let maxSubstringUpdates = 32
}

/// Swift wrapper for shared memory operations  
public class RenderLoopSynchronization {
    private let memoryName: String
    private var memoryPtr: UnsafeMutableRawPointer?
    private var fileDescriptor: Int32 = -1
    private let isOwner: Bool
    
    // Semaphores for synchronization
    private var animationFrameRequestSemaphore: Int32 = -1  // OuterContent signals when it wants frame
    private var renderFrameSemaphore: Int32 = -1            // Main app signals when it's time to render
    private var renderCompleteSemaphore: Int32 = -1         // OuterContent signals when render is done
    
    public init?(name: String, create: Bool = true) {
        self.memoryName = name
        self.isOwner = create
        
        if create {
            // OuterContent creates shared memory and semaphores
            var fd: Int32 = 0
            self.memoryPtr = create_shared_memory(name, SharedRenderMemory.size, &fd)
            self.fileDescriptor = fd
            
            guard memoryPtr != nil else {
                return nil
            }
            
            // Initialize memory with zero
            memset(memoryPtr!, 0, SharedRenderMemory.size)
            
            // Initialize header
            let header = memoryPtr!.assumingMemoryBound(to: SharedRenderMemory.Header.self)
            header.pointee.version = 1
            header.pointee.processID = getpid()
            
            // Create semaphores
            if create_mach_semaphore(&animationFrameRequestSemaphore, 0) != 0 {
                return nil
            }
            if create_mach_semaphore(&renderFrameSemaphore, 0) != 0 {
                destroy_mach_semaphore(animationFrameRequestSemaphore)
                return nil
            }
            if create_mach_semaphore(&renderCompleteSemaphore, 0) != 0 {
                destroy_mach_semaphore(animationFrameRequestSemaphore)
                destroy_mach_semaphore(renderFrameSemaphore)
                return nil
            }
            
            // Register semaphores with bootstrap service so main app can find them
            let pid = getpid()
            let requestSemaphoreName = "outercontent.semaphore.request.\(pid)"
            let renderSemaphoreName = "outercontent.semaphore.render.\(pid)"
            let completeSemaphoreName = "outercontent.semaphore.complete.\(pid)"
            
            if bootstrap_register_wrapper(requestSemaphoreName, animationFrameRequestSemaphore) != 0 {
                destroy_mach_semaphore(animationFrameRequestSemaphore)
                destroy_mach_semaphore(renderFrameSemaphore)
                destroy_mach_semaphore(renderCompleteSemaphore)
                return nil
            }
            if bootstrap_register_wrapper(renderSemaphoreName, renderFrameSemaphore) != 0 {
                destroy_mach_semaphore(animationFrameRequestSemaphore)
                destroy_mach_semaphore(renderFrameSemaphore)
                destroy_mach_semaphore(renderCompleteSemaphore)
                return nil
            }
            if bootstrap_register_wrapper(completeSemaphoreName, renderCompleteSemaphore) != 0 {
                destroy_mach_semaphore(animationFrameRequestSemaphore)
                destroy_mach_semaphore(renderFrameSemaphore)
                destroy_mach_semaphore(renderCompleteSemaphore)
                return nil
            }
        } else {
            // Main app opens existing shared memory created by OuterContent
            self.memoryPtr = open_shared_memory(name, SharedRenderMemory.size)
            guard memoryPtr != nil else {
                return nil
            }
            
            // Get OuterContent's process ID from the header to find semaphores
            let header = memoryPtr!.assumingMemoryBound(to: SharedRenderMemory.Header.self)
            let outerContentPid = header.pointee.processID
            
            // Look up semaphores that OuterContent registered with bootstrap
            let requestSemaphoreName = "outercontent.semaphore.request.\(outerContentPid)"
            let renderSemaphoreName = "outercontent.semaphore.render.\(outerContentPid)"
            let completeSemaphoreName = "outercontent.semaphore.complete.\(outerContentPid)"

            if bootstrap_look_up_wrapper(requestSemaphoreName, &animationFrameRequestSemaphore) != 0 {
                return nil
            }
            if bootstrap_look_up_wrapper(renderSemaphoreName, &renderFrameSemaphore) != 0 {
                return nil
            }
            if bootstrap_look_up_wrapper(completeSemaphoreName, &renderCompleteSemaphore) != 0 {
                return nil
            }
        }
    }
    
    deinit {
        cleanup()
    }
    
    public func cleanup() {
        if animationFrameRequestSemaphore != -1 {
            destroy_mach_semaphore(animationFrameRequestSemaphore)
            animationFrameRequestSemaphore = -1
        }
        if renderFrameSemaphore != -1 {
            destroy_mach_semaphore(renderFrameSemaphore)
            renderFrameSemaphore = -1
        }
        if renderCompleteSemaphore != -1 {
            destroy_mach_semaphore(renderCompleteSemaphore)
            renderCompleteSemaphore = -1
        }
        
        if let ptr = memoryPtr {
            if isOwner {
                close_shared_memory(ptr, SharedRenderMemory.size, fileDescriptor)
                _ = memoryName.withCString { namePtr in
                    shm_unlink(namePtr)
                }
            } else {
                munmap(ptr, SharedRenderMemory.size)
            }
            memoryPtr = nil
        }
    }
    
    // MARK: - Synchronization
    
    /// OuterContent calls this when it wants an animation frame
    public func requestAnimationFrame() {
        guard animationFrameRequestSemaphore != -1 else { return }
        signal_mach_semaphore(animationFrameRequestSemaphore)
    }
    
    /// Main app calls this to wait for animation frame requests from OuterContent
    public func waitForAnimationFrameRequest() -> Bool {
        guard animationFrameRequestSemaphore != -1 else { return false }
        return wait_mach_semaphore(animationFrameRequestSemaphore) == 0
    }

    /// Main app calls this to signal OuterContent to render at display link timing
    public func signalRenderFrame() {
        guard renderFrameSemaphore != -1 else { return }
        signal_mach_semaphore(renderFrameSemaphore)
    }
    
    /// OuterContent calls this to wait for render timing from main app
    public func waitForRenderFrame() -> Bool {
        guard renderFrameSemaphore != -1 else { return false }
        return wait_mach_semaphore(renderFrameSemaphore) == 0
    }
    
    /// OuterContent calls this when rendering is complete
    public func signalRenderComplete() {
        guard renderCompleteSemaphore != -1 else { return }
        signal_mach_semaphore(renderCompleteSemaphore)
    }
    
    /// Main app calls this to wait for render completion
    public func waitForRenderComplete() -> Bool {
        guard renderCompleteSemaphore != -1 else { return false }
        return wait_mach_semaphore(renderCompleteSemaphore) == 0
    }
    
    // MARK: - Memory Access
    
    public func updateHeader(frameNumber: UInt64, targetTimestamp: CFTimeInterval) {
        guard let ptr = memoryPtr else { return }
        let header = ptr.assumingMemoryBound(to: SharedRenderMemory.Header.self)
        header.pointee.frameNumber = frameNumber
        header.pointee.targetTimestamp = targetTimestamp
    }
    
    public func getTargetTimestamp() -> CFTimeInterval {
        guard let ptr = memoryPtr else { return CACurrentMediaTime() }
        let header = ptr.assumingMemoryBound(to: SharedRenderMemory.Header.self)
        return header.pointee.targetTimestamp
    }
    
    
    public func addFragmentUpdate(_ substringId: UUID) {
        guard let ptr = memoryPtr else { return }
        let header = ptr.assumingMemoryBound(to: SharedRenderMemory.Header.self)
        
        let currentCount = Int(header.pointee.substringUpdateCount)
        guard currentCount < SharedRenderMemory.maxSubstringUpdates else { return }
        
        let updateOffset = MemoryLayout<SharedRenderMemory.Header>.size
        let updatePtr = (ptr + updateOffset).assumingMemoryBound(to: SharedRenderMemory.UpdateSubstringCanvasesNotification.self)
        
        var uuidBytes = uuid_t(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        _ = withUnsafeBytes(of: substringId.uuid) { bytes in
            memcpy(&uuidBytes, bytes.baseAddress!, MemoryLayout<uuid_t>.size)
        }
        
        updatePtr[currentCount].substringId = uuidBytes
        header.pointee.substringUpdateCount = UInt32(currentCount + 1)
    }
    
    public func getSubstringUpdates() -> [UUID] {
        guard let ptr = memoryPtr else { return [] }
        let header = ptr.assumingMemoryBound(to: SharedRenderMemory.Header.self)
        
        let updateCount = Int(header.pointee.substringUpdateCount)
        guard updateCount > 0 else { return [] }
        
        let updateOffset = MemoryLayout<SharedRenderMemory.Header>.size
        let updatePtr = (ptr + updateOffset).assumingMemoryBound(to: SharedRenderMemory.UpdateSubstringCanvasesNotification.self)
        
        var updates: [UUID] = []
        for i in 0..<updateCount {
            let uuidBytes = updatePtr[i].substringId
            let uuid = UUID(uuid: uuidBytes)
            updates.append(uuid)
        }
        
        return updates
    }
    
    /// Atomically get and clear substring updates to avoid race conditions
    public func getAndClearSubstringUpdates() -> [UUID] {
        guard let ptr = memoryPtr else { return [] }
        let header = ptr.assumingMemoryBound(to: SharedRenderMemory.Header.self)
        
        let updateCount = Int(header.pointee.substringUpdateCount)
        guard updateCount > 0 else { return [] }
        
        let updateOffset = MemoryLayout<SharedRenderMemory.Header>.size
        let updatePtr = (ptr + updateOffset).assumingMemoryBound(to: SharedRenderMemory.UpdateSubstringCanvasesNotification.self)
        
        var updates: [UUID] = []
        for i in 0..<updateCount {
            let uuidBytes = updatePtr[i].substringId
            let uuid = UUID(uuid: uuidBytes)
            updates.append(uuid)
        }
        
        // Clear immediately after reading to make it atomic
        header.pointee.substringUpdateCount = 0
        
        return updates
    }
    
    public func clearSubstringUpdates() {
        guard let ptr = memoryPtr else { return }
        let header = ptr.assumingMemoryBound(to: SharedRenderMemory.Header.self)
        header.pointee.substringUpdateCount = 0
    }
}

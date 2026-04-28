//
//  OuterContentConnection.swift
//

import IOSurface
import Foundation
import Darwin
import QuartzCore
import AppKit.NSAttributedString

// Connection to the XPC service and OuterContent process
public class OuterContentConnection: NSObject {
    @objc public static let shared = OuterContentConnection()

    // OuterContent process management
    private var outerContentProcess: Process?
    private var clientSocket: Int32 = -1
    private let socketPath: String = {
        let tempDir = NSTemporaryDirectory()
        return tempDir.appending("outercontent.sock")
    }()

    // Structure to hold deserialized RuntimeOuterdoc data
    public struct RuntimeOuterdocData {
        public let attributedString: NSAttributedString
        public let elements: [ElementInfo]
        public let canvases: [CanvasInfo]
    }
    
    // Completion handlers for OuterContent responses
    private var pendingCreateDynamicSubstringCompletions: [UUID: (RuntimeOuterdocData, [IOSurface]) -> Void] = [:]
    private var pendingCreateDynamicSubstringsCompletions: [([RuntimeOuterdocData], [[IOSurface]]) -> Void] = []
    private var pendingAnimationFrameCompletions: [UUID: () -> Void] = [:]
    
    // Shared memory for optimized render loop
    private var sharedMemory: RenderLoopSynchronization?

    private override init() {
        super.init()
        launchOuterContentProcess()
    }

    func disposeSubstring(substringId: UUID) {
        guard clientSocket != -1 else {
            print("OuterContentConnection: OuterContent process not available for disposal")
            return
        }

        sendDisposeSubstringMessage(substringId: substringId)
    }

    /// Creates an outerdoc substring for the given file type via OuterContent process.
    /// - Parameters:
    ///   - substringId: The stable substring identifier (UUID) to use for this substring
    ///   - fileType: Identifier for the registered text-based (outerdoc) figure type
    ///   - completion: Called when creation is complete, returns serialized outerdoc data and canvas surfaces
    public func createDynamicSubstring(substringId: UUID, contentType: String, completion: @escaping (RuntimeOuterdocData, [IOSurface]) -> Void) {
        guard clientSocket != -1 else {
            print("OuterContentConnection: OuterContent process not available for createDynamicSubstring")
            completion(RuntimeOuterdocData(attributedString: NSAttributedString(), elements: [], canvases: []), [])
            return
        }

        // Store completion handler for response
        pendingCreateDynamicSubstringCompletions[substringId] = completion

        // Send createDynamicSubstring message to OuterContent process
        sendCreateDynamicSubstringMessage(substringId: substringId, fileType: contentType)
    }

    /// Creates outerdoc substrings for multiple file types via OuterContent process.
    /// - Parameters:
    ///   - substringIds: Array of stable substring identifiers (UUIDs) to use for these substrings
    ///   - fileTypes: Array of registered text-based (outerdoc) figure types (must match length of substringIds)
    ///   - completion: Called when generation is complete, returns arrays of content data and canvas surfaces
    public func createDynamicSubstrings(substringIds: [UUID], contentTypes: [String], completion: @escaping ([RuntimeOuterdocData], [[IOSurface]]) -> Void) {
        guard clientSocket != -1 else {
            print("OuterContentConnection: OuterContent process not available for createDynamicSubstrings")
            let emptyData = RuntimeOuterdocData(attributedString: NSAttributedString(), elements: [], canvases: [])
            completion(Array(repeating: emptyData, count: contentTypes.count),
                       Array(repeating: [], count: contentTypes.count))
            return
        }

        // Store completion handler for response
        pendingCreateDynamicSubstringsCompletions.append(completion)

        // Send createDynamicSubstrings message to OuterContent process
        sendCreateDynamicSubstringsMessage(substringIds: substringIds, fileTypes: contentTypes)
    }

    // MARK: - OuterContent Process Management

    private func launchOuterContentProcess() {
        guard let executablePath = getOuterContentExecutablePath() else {
            print("Could not find OuterContent executable")
            return
        }

        // Clean up any existing socket file
        try? FileManager.default.removeItem(atPath: socketPath)

        outerContentProcess = Process()
        outerContentProcess?.executableURL = URL(fileURLWithPath: executablePath)
        outerContentProcess?.arguments = [socketPath]
//        outerContentProcess?.arguments = [socketPath, "--wait-for-debugger"]

        outerContentProcess?.terminationHandler = { process in
            print("OuterContent process terminated with exit code: \(process.terminationStatus)")
        }

        do {
            try outerContentProcess?.run()
            let childPid = outerContentProcess!.processIdentifier
            print("Launched OuterContent process \(childPid)")

            // Give the process a moment to start listening, then connect
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.connectToOuterContent()
            }
        } catch {
            print("Failed to launch OuterContent process: \(error)")
        }
    }

    private func getOuterContentExecutablePath() -> String? {
        let bundle = Bundle.main.bundleURL
        let executablePath = bundle
            .appendingPathComponent("Contents")
            .appendingPathComponent("MacOS")
            .appendingPathComponent("OuterContent")
            .path

        if FileManager.default.fileExists(atPath: executablePath) {
            return executablePath
        }

        // Fallback for development builds
        let devPath = bundle
            .deletingLastPathComponent()
            .appendingPathComponent("OuterContent")
            .path

        if FileManager.default.fileExists(atPath: devPath) {
            return devPath
        }

        return nil
    }

    private func connectToOuterContent() {
        clientSocket = socket(AF_UNIX, SOCK_STREAM, 0)
        if clientSocket == -1 {
            print("Failed to create client socket: \(String(cString: strerror(errno)))")
            return
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)

        // Copy path to sun_path
        let pathBytes = socketPath.utf8CString
        let pathSize = min(pathBytes.count, MemoryLayout.size(ofValue: addr.sun_path) - 1)
        withUnsafeMutableBytes(of: &addr.sun_path) { sunPathPtr in
            pathBytes.withUnsafeBytes { pathPtr in
                sunPathPtr.copyMemory(from: UnsafeRawBufferPointer(start: pathPtr.baseAddress, count: pathSize))
            }
        }

        let connectResult = connect(clientSocket, withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { $0 }
        }, socklen_t(MemoryLayout<sockaddr_un>.size))

        if connectResult == -1 {
            print("Failed to connect to OuterContent: \(String(cString: strerror(errno)))")
            close(clientSocket)
            clientSocket = -1

            // Retry connection after a delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.connectToOuterContent()
            }
            return
        }

        print("Connected to OuterContent")
        sendTestMessage()
        startReceiving()
        
        // Initialize shared memory for optimized render loop
        setupSharedMemory()
    }

    private func startReceiving() {
        DispatchQueue.global().async {
            while self.clientSocket != -1 {
                if let messageData = self.receiveBinaryMessage() {
                    self.handleOuterContentBinaryResponse(messageData)
                }
            }
        }
    }

    private func receiveBinaryMessage() -> Data? {
        // Read length prefix (4 bytes)
        var lengthBytes = [UInt8](repeating: 0, count: 4)
        let lengthBytesRead = recv(clientSocket, &lengthBytes, 4, MSG_WAITALL)
        if lengthBytesRead != 4 {
            return nil
        }

        let length = lengthBytes.withUnsafeBytes { bytes in
            bytes.load(as: UInt32.self).littleEndian
        }

        // Read payload
        var payload = Data(count: Int(length))
        let payloadBytesRead = payload.withUnsafeMutableBytes { bytes in
            recv(clientSocket, bytes.baseAddress, Int(length), MSG_WAITALL)
        }

        if payloadBytesRead != Int(length) {
            return nil
        }

        return payload
    }

    private func sendTestMessage() {
        // Send a simple test message
        guard let data = serializeTextMessage("Hello from OuterContentConnection!") else {
            print("Failed to serialize test message")
            return
        }
        sendBinaryMessage(data)
    }

    public func sendToOuterContent(_ message: String) {
        guard let data = serializeTextMessage(message) else {
            print("Failed to serialize text message")
            return
        }
        sendBinaryMessage(data)
    }

    private func sendBinaryMessage(_ data: Data) {
        guard clientSocket != -1 else { return }

        // Send length prefix (4 bytes, little-endian)
        var length = UInt32(data.count).littleEndian
        let lengthSent = withUnsafeBytes(of: &length) { bytes in
            send(clientSocket, bytes.baseAddress, 4, 0)
        }

        if lengthSent != 4 {
            print("Failed to send message length")
            return
        }

        // Send payload
        data.withUnsafeBytes { bytes in
            let bytesPtr = bytes.bindMemory(to: UInt8.self)
            send(clientSocket, bytesPtr.baseAddress, data.count, 0)
        }
    }

    private func sendCreateDynamicSubstringsMessage(substringIds: [UUID], fileTypes: [String]) {
        guard let data = serializeCreateDynamicSubstringsRequest(substringIds, fileTypes) else {
            print("Failed to serialize createDynamicSubstrings request")
            return
        }
        sendBinaryMessage(data)
    }

    private func sendCreateDynamicSubstringMessage(substringId: UUID, fileType: String) {
        guard let data = serializeCreateDynamicSubstringRequest(substringId, fileType) else {
            print("Failed to serialize createDynamicSubstring request")
            return
        }
        sendBinaryMessage(data)
    }

    private func sendDisposeSubstringMessage(substringId: UUID) {
        guard let data = serializeDisposeSubstringRequest(substringId) else {
            print("Failed to serialize disposeSubstring request")
            return
        }
        sendBinaryMessage(data)
    }

    private func handleOuterContentBinaryResponse(_ data: Data) {
        guard !data.isEmpty else { return }

        processOuterContentMessage(
            data,
            { substringIds, status, attributedStrings, elementsArrays, canvasesArrays, surfaceServiceNames in  // createDynamicSubstringsResponse handler
                self.handleCreateDynamicSubstringsResponse(
                    substringIds: substringIds,
                    status: status,
                    attributedStrings: attributedStrings,
                    elementsArrays: elementsArrays,
                    canvasesArrays: canvasesArrays,
                    surfaceServiceNames: surfaceServiceNames)
            },
            { substringId, attributedString, elements, canvases, surfaceServiceNames in  // createDynamicSubstringResponse handler
                self.handleCreateDynamicSubstringResponse(substringId: substringId, attributedString: attributedString, elements: elements, canvases: canvases, surfaceServiceNames: surfaceServiceNames)
            },
            { substringId, elementIds, newContents in  // updateElements handler
                self.handleUpdateElements(substringId: substringId, elementIds: elementIds, newContents: newContents)
            }
        )
    }

    private func handleCreateDynamicSubstringsResponse(substringIds: [UUID], status: String, attributedStrings: [NSAttributedString], elementsArrays: [[ElementInfo]], canvasesArrays: [[CanvasInfo]], surfaceServiceNames: [[String]]) {
        // Process surface service names to get IOSurfaces
        var resultsSurfaces: [[IOSurface]] = []
        var runtimeOuterdocDatas: [RuntimeOuterdocData] = []

        for (i, serviceNames) in surfaceServiceNames.enumerated() {
            var substringSurfaces: [IOSurface] = []
            for serviceName in serviceNames {
                // Lookup IOSurface from bootstrap service
                var port: Int32 = 0
                let lookupResult = bootstrap_look_up_wrapper(serviceName, &port)
                if lookupResult == 0 {
                    if let surface = IOSurfaceLookupFromMachPort(mach_port_t(port)) {
                        substringSurfaces.append(surface)
                        
                        // Clean up mach port immediately after successful transfer
                        mach_port_deallocate(mach_task_self_, mach_port_t(port))
                    } else {
                        print("Failed to reconstruct IOSurface from bootstrap service '\(serviceName)'")
                        
                        // Clean up mach port even on failure
                        mach_port_deallocate(mach_task_self_, mach_port_t(port))
                    }
                } else {
                    print("Failed to lookup bootstrap service '\(serviceName)': \(lookupResult)")
                }
            }
            resultsSurfaces.append(substringSurfaces)
            
            // Create RuntimeOuterdocData for this substring
            if i < attributedStrings.count {
                let runtimeOuterdocData = RuntimeOuterdocData(
                    attributedString: attributedStrings[i],
                    elements: i < elementsArrays.count ? elementsArrays[i] : [],
                    canvases: i < canvasesArrays.count ? canvasesArrays[i] : []
                )
                runtimeOuterdocDatas.append(runtimeOuterdocData)
            } else {
                // Add empty RuntimeOuterdocData
                runtimeOuterdocDatas.append(RuntimeOuterdocData(
                    attributedString: NSAttributedString(),
                    elements: [],
                    canvases: []
                ))
            }
        }

        // Call completion handler
        if !pendingCreateDynamicSubstringsCompletions.isEmpty {
            let completion = pendingCreateDynamicSubstringsCompletions.removeFirst()
            completion(runtimeOuterdocDatas, resultsSurfaces)
        }
    }

    private func handleCreateDynamicSubstringResponse(substringId: UUID, attributedString: NSAttributedString, elements: [ElementInfo]?, canvases: [CanvasInfo]?, surfaceServiceNames: [String]) {
        // Process surface service names to get IOSurfaces
        var surfaces: [IOSurface] = []
        for serviceName in surfaceServiceNames {
            // Lookup IOSurface from bootstrap service
            var port: Int32 = 0
            let lookupResult = bootstrap_look_up_wrapper(serviceName, &port)
            if lookupResult == 0 {
                if let surface = IOSurfaceLookupFromMachPort(mach_port_t(port)) {
                    surfaces.append(surface)
                    
                    // Clean up mach port immediately after successful transfer
                    mach_port_deallocate(mach_task_self_, mach_port_t(port))
                } else {
                    print("Failed to reconstruct IOSurface from bootstrap service '\(serviceName)'")
                    
                    // Clean up mach port even on failure
                    mach_port_deallocate(mach_task_self_, mach_port_t(port))
                }
            } else {
                print("Failed to lookup bootstrap service '\(serviceName)': \(lookupResult)")
            }
        }

        // Call completion handler
        if let completion = pendingCreateDynamicSubstringCompletions.removeValue(forKey: substringId) {
            let runtimeOuterdocData = RuntimeOuterdocData(
                attributedString: attributedString,
                elements: elements ?? [],
                canvases: canvases ?? []
            )
            completion(runtimeOuterdocData, surfaces)
        }
    }

    private func handleUpdateElements(substringId: UUID, elementIds: [UUID], newContents: [Data]) {
        // Convert NSKeyedArchiver data back to NSAttributedStrings
        var newStrings: [NSAttributedString] = []
        for elementData in newContents {
            do {
                if let newString = try NSKeyedUnarchiver.unarchivedObject(ofClass: NSAttributedString.self, from: elementData) {
                    newStrings.append(newString)
                } else {
                    print("Invalid updateElements: couldn't unarchive NSAttributedString")
                    newStrings.append(NSAttributedString(string: ""))
                }
            } catch {
                print("Invalid updateElements: error unarchiving NSAttributedString: \(error)")
                newStrings.append(NSAttributedString(string: ""))
            }
        }

        DynamicSubstringManager.shared.substringFrameReady(substringId: substringId, elementIds: elementIds, newStrings: newStrings)
    }

    public func cleanup() {
        if clientSocket != -1 {
            close(clientSocket)
            clientSocket = -1
        }


        outerContentProcess?.terminate()
        outerContentProcess?.waitUntilExit()
        try? FileManager.default.removeItem(atPath: socketPath)

        sharedMemory?.cleanup()
    }

    deinit {
        cleanup()
    }
    
    /// Expose shared memory for DynamicSubstringDisplayLinks listener thread
    func getSharedMemory() -> RenderLoopSynchronization? {
        return sharedMemory
    }
    
    // MARK: - Shared Memory Optimization
    
    private func setupSharedMemory() {
        guard let process = outerContentProcess else { return }
        
        let memoryName = "/outercontent_render_\(process.processIdentifier)"
        sharedMemory = RenderLoopSynchronization(name: memoryName, create: false)
        
        if sharedMemory != nil {
            print("OuterContentConnection: Shared memory render loop connected")
        } else {
            print("OuterContentConnection: Failed to connect to shared memory")
        }
    }
    
    /// Optimized animation frame generation using shared memory and semaphores
    func generateFrameOptimized(targetTimestamp: CFTimeInterval) {
        let memory = sharedMemory!
        // We're already here because the listener thread detected an animation frame request
        // Set timing information and signal OuterContent to render
        memory.updateHeader(frameNumber: CACurrentMediaTime().bitPattern, targetTimestamp: targetTimestamp)
        memory.signalRenderFrame()
    }
}

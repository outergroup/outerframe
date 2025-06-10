//
//  main.swift
//

import Foundation
import Darwin
import OuterdocPlugin
import IOSurface
import AppKit.NSAttributedString

func debuggerIsAttached() -> Bool {
    var info = kinfo_proc()
    var size = MemoryLayout<kinfo_proc>.stride
    var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
    return sysctl(&mib, u_int(mib.count), &info, &size, nil, 0) == 0 && (info.kp_proc.p_flag & P_TRACED) != 0
}

func waitForDebugger() {
    print("Waiting for debugger to attach (PID: \(getpid()))...")
    while !debuggerIsAttached() {
        usleep(100000) // 100ms
    }
    print("Debugger attached, continuing...")
}

/// Scheduler that connects a dynamic substring to its host for animation frame requests
fileprivate class FragmentAnimationScheduler: AnimationScheduler {
    private weak var host: OuterContentServer?
    private let substringId: UUID

    init(substringId: UUID, host: OuterContentServer) {
        self.substringId = substringId
        self.host = host
    }

    func requestAnimationFrame() {
        host!.requestAnimationFrameOptimized(substringId: substringId)
    }
}

class OuterContentServer {
    private let socketPath: String
    private var serverSocket: Int32 = -1
    private var clientSocket: Int32 = -1
    private var isRunning = false
    
    // Plugin management
    private var loadedBundles: [URL: Bundle] = [:]
    private var pluginsByContentType: [String: OuterdocPlugin.Type] = [:]
    private var substringMap: [UUID: OuterdocDynamicSubstring] = [:]
    
    // Animation frame tracking
    private var substringSchedulerMap: [UUID: FragmentAnimationScheduler] = [:]
    private var substringsAwaitingAnimationFrame: [UUID] = []

    // Shared memory for high-performance render loop
    private var sharedMemory: RenderLoopSynchronization?
    private var renderThread: Thread?
    
    init(socketPath: String) {
        self.socketPath = socketPath
    }
    
    func loadBundledPlugin() {
        // Load ExamplePlugin.bundle from the app's Resources folder
        let examplePluginPath = Bundle.main.bundlePath + "/Contents/Resources/ExamplePlugin.bundle"
        let examplePluginURL = URL(fileURLWithPath: examplePluginPath)

        if FileManager.default.fileExists(atPath: examplePluginPath) {
            print("Loading bundled ExamplePlugin from: \(examplePluginPath)")

            if let substringTypes = loadAndRegisterPlugin(bundleURL: examplePluginURL) {
                print("Successfully loaded bundled ExamplePlugin with types: \(substringTypes)")
            } else {
                print("Failed to load bundled ExamplePlugin")
            }
        } else {
            print("Bundled ExamplePlugin.bundle not found at: \(examplePluginPath)")
        }
    }

    func start() {
        serverSocket = socket(AF_UNIX, SOCK_STREAM, 0)
        if serverSocket == -1 {
            print("Failed to create socket: \(String(cString: strerror(errno)))")
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
        
        // Remove existing socket file
        unlink(socketPath)
        
        let bindResult = bind(serverSocket, withUnsafePointer(to: &addr) { 
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { $0 } 
        }, socklen_t(MemoryLayout<sockaddr_un>.size))
        
        if bindResult == -1 {
            print("Failed to bind socket: \(String(cString: strerror(errno)))")
            close(serverSocket)
            return
        }
        
        if listen(serverSocket, 1) == -1 {
            print("Failed to listen on socket: \(String(cString: strerror(errno)))")
            close(serverSocket)
            return
        }
        
        print("Server listening on \(socketPath)")
        isRunning = true
        
        // Initialize shared memory for render loop optimization
        setupSharedMemory()
        
        // Accept connections in background
        DispatchQueue.global(qos: .userInitiated).async {
            self.acceptLoop()
        }
    }
    
    private func acceptLoop() {
        while isRunning {
            clientSocket = accept(serverSocket, nil, nil)
            if clientSocket == -1 {
                if isRunning {
                    print("Failed to accept connection: \(String(cString: strerror(errno)))")
                }
                continue
            }
            
            print("Client connected")
            handleClient()
            close(clientSocket)
            print("Client disconnected")
        }
    }
    
    private func handleClient() {
        while isRunning {
            autoreleasepool {
                if let messageData = receiveBinaryMessage() {
                    DispatchQueue.main.async {
                        self.handleBinaryMessage(messageData)
                    }
                }
            }
        }
    }
    
    private func receiveBinaryMessage() -> Data? {
        // Read length prefix (4 bytes)
        var lengthBytes = [UInt8](repeating: 0, count: 4)
        let lengthBytesRead = recv(clientSocket, &lengthBytes, 4, MSG_WAITALL)

        guard lengthBytesRead != 0 else {
            print("recv returned 0 bytes. We are assuming the client exited.")
            stop()
            exit(0)
        }

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
    
    private func handleBinaryMessage(_ data: Data) {
        guard !data.isEmpty else { return }
        
        processOuterdocMessage(
            data,
            { text in  // textMessage handler
                print("Received text: \(text)")
            },
            { substringIds, fileTypes in  // createDynamicSubstrings handler
                self.handleCreateDynamicSubstrings(substringIds: substringIds, fileTypes: fileTypes)
            },
            { substringId, fileType in  // createDynamicSubstring handler
                self.handleCreateDynamicSubstring(substringId: substringId, fileType: fileType)
            },
            { substringId in  // disposeSubstring handler
                self.handleDisposeSubstring(substringId: substringId)
            }
        )
    }
    
    private func handleCreateDynamicSubstrings(substringIds: [UUID], fileTypes: [String]) {
        // Actually process the createDynamicSubstrings request
        let results = processCreateDynamicSubstrings(substringIds: substringIds, fileTypes: fileTypes)
        
        // Send binary response back
        sendCreateDynamicSubstringsResponse(substringIds: substringIds, status: "processed", results: results)
    }
    
    private func handleCreateDynamicSubstring(substringId: UUID, fileType: String) {
        // Process the createDynamicSubstring request
        let (outerdocSubstring, surfaces) = processCreateDynamicSubstring(substringId: substringId, fileType: fileType)
        
        // Send response
        sendCreateDynamicSubstringResponse(substringId: substringId, outerdocSubstring: outerdocSubstring, surfaces: surfaces)
    }
    
    private func handleDisposeSubstring(substringId: UUID) {
        // Process the disposeSubstring request
        processDisposeSubstring(substringId: substringId)
    }

    /// Load a plugin bundle and ask it to register its supported file types with the UI process.
    /// - Returns: List of outerdoc-based types, or nil on failure
    private func loadAndRegisterPlugin(bundleURL: URL) -> [String]? {
        // Avoid reloading
        if loadedBundles[bundleURL] == nil {
            // Load the bundle
            guard let bundle = Bundle(path: bundleURL.path) else {
                print("registerPlugin: failed to create bundle at path: \(bundleURL.path)")
                return nil
            }
            guard bundle.load() else {
                print("registerPlugin: failed to load bundle executable")
                return nil
            }
            loadedBundles[bundleURL] = bundle
        } else {
            print("Plugin already loaded: \(bundleURL.path)")
        }
        // Invoke plugin code to declare its outerdoc file types
        guard let pluginType = loadedBundles[bundleURL]?.principalClass as? OuterdocPlugin.Type else {
            print("registerPlugin: principalClass does not conform to OuterdocPlugin for bundle: \(bundleURL.path)")
            return nil
        }
        let odTypes = pluginType.suppertedOuterdocContentTypes()
        // Register plugin class for each supported content type
        for fileType in odTypes {
            pluginsByContentType[fileType] = pluginType
        }
        return odTypes
    }

    private func processCreateDynamicSubstring(substringId: UUID, fileType: String) -> (OuterdocSubstring?, [IOSurface]) {
        // Get plugin class for this content type
        guard let pluginClass = pluginsByContentType[fileType] else {
            print("No plugin found for content type: \(fileType)")
            return (nil, [])
        }
        
        // Create the fragment
        guard let fragment = pluginClass.createDynamicSubstring(contentType: fileType) else {
            print("Failed to create fragment for type: \(fileType)")
            return (nil, [])
        }
        
        // Create a scheduler for this fragment
        let scheduler = FragmentAnimationScheduler(substringId: substringId, host: self)
        fragment.setAnimationScheduler(scheduler: scheduler)
        
        // Store the fragment generator and scheduler
        substringMap[substringId] = fragment
        substringSchedulerMap[substringId] = scheduler

        // Get initial substring with canvasSurfaces
        let initialSubstring = fragment.getInitialSubstring()
        let canvasSurfaces = initialSubstring.canvasSurfaces
        
        return (initialSubstring, canvasSurfaces)
    }
    
    private func processDisposeSubstring(substringId: UUID) {
        substringMap.removeValue(forKey: substringId)
        substringSchedulerMap.removeValue(forKey: substringId)

        // Note: We intentionally don't close bundle handles here since other fragments might be using them
        // Bundle handles will be cleaned up when the OuterContent process terminates
    }

    private func processCreateDynamicSubstrings(substringIds: [UUID], fileTypes: [String]) -> [(OuterdocSubstring?, [IOSurface])] {
        var results: [(OuterdocSubstring?, [IOSurface])] = []
        
        for (index, substringId) in substringIds.enumerated() {
            guard index < fileTypes.count else {
                print("Mismatched substringIds and fileTypes arrays")
                results.append((nil, []))
                continue
            }
            
            let fileType = fileTypes[index]
            
            // Get plugin class for this content type
            guard let pluginClass = pluginsByContentType[fileType] else {
                print("No plugin found for content type: \(fileType)")
                results.append((nil, []))
                continue
            }
            
            // Create the fragment
            guard let fragment = pluginClass.createDynamicSubstring(contentType: fileType) else {
                print("Failed to create fragment for type: \(fileType)")
                results.append((nil, []))
                continue
            }
            
            // Create a scheduler for this fragment
            let scheduler = FragmentAnimationScheduler(substringId: substringId, host: self)
            fragment.setAnimationScheduler(scheduler: scheduler)
            
            // Store the fragment generator and scheduler for future animation frames
            substringMap[substringId] = fragment
            substringSchedulerMap[substringId] = scheduler
            
            // Get initial substring with canvasSurfaces
            let initialSubstring = fragment.getInitialSubstring()
            let canvasSurfaces = initialSubstring.canvasSurfaces
            
            results.append((initialSubstring, canvasSurfaces))
            
            print("Created fragment \(substringId) for type \(fileType)")
            print("  Canvas surfaces: \(canvasSurfaces.count)")
        }
        
        return results
    }
    
    private func sendBinaryMessage(_ data: Data) {
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
    
    private func sendCreateDynamicSubstringsResponse(substringIds: [UUID], status: String, results: [(OuterdocSubstring?, [IOSurface])]) {
        // Convert results to arrays for Cap'n Proto serialization
        var attributedStrings: [NSAttributedString] = []
        var elementsArrays: [[ElementInfo]] = []
        var canvasesArrays: [[CanvasInfo]] = []
        var surfaceServiceNames: [[String]] = []
        
        for (i, (outerdocSubstring, surfaces)) in results.enumerated() {
            if let outerdocSubstring = outerdocSubstring {
                attributedStrings.append(outerdocSubstring.attributedString)
                elementsArrays.append(outerdocSubstring.elements as? [ElementInfo] ?? [])
                canvasesArrays.append(outerdocSubstring.canvases as? [CanvasInfo] ?? [])
            } else {
                // Add empty values for nil substrings
                attributedStrings.append(NSAttributedString())
                elementsArrays.append([])
                canvasesArrays.append([])
            }
            
            var serviceNames: [String] = []
            for (surfaceIndex, surface) in surfaces.enumerated() {
                let serviceName = registerIOSurfaceWithBootstrap(surface, substringId: substringIds[i], surfaceIndex: surfaceIndex)
                serviceNames.append(serviceName)
            }
            surfaceServiceNames.append(serviceNames)
        }

        guard let data = serializeCreateDynamicSubstringsResponse(
            substringIds,
            status,
            attributedStrings,
            elementsArrays,
            canvasesArrays,
            surfaceServiceNames
        ) else {
            print("Failed to serialize createDynamicSubstringsResponse")
            return
        }
        
        sendBinaryMessage(data)
    }
    
    private func sendCreateDynamicSubstringResponse(substringId: UUID, outerdocSubstring: OuterdocSubstring?, surfaces: [IOSurface]) {
        // Register IOSurfaces and collect service names
        var serviceNames: [String] = []
        for (index, surface) in surfaces.enumerated() {
            let serviceName = registerIOSurfaceWithBootstrap(surface, substringId: substringId, surfaceIndex: index)
            serviceNames.append(serviceName)
        }
        
        if let outerdocSubstring = outerdocSubstring {
            guard let data = serializeCreateDynamicSubstringResponse(
                substringId,
                outerdocSubstring.attributedString,
                outerdocSubstring.elements as? [ElementInfo],
                outerdocSubstring.canvases as? [CanvasInfo],
                serviceNames
            ) else {
                print("Failed to serialize createDynamicSubstringResponse")
                return
            }
            sendBinaryMessage(data)
        } else {
            // Send empty response
            guard let data = serializeCreateDynamicSubstringResponse(
                substringId,
                NSAttributedString(),
                nil,
                nil,
                serviceNames
            ) else {
                print("Failed to serialize createDynamicSubstringResponse")
                return
            }
            sendBinaryMessage(data)
        }
    }
    
    private func sendUpdateElementsMessage(substringId: UUID, elementIds: [UUID], newStrings: [NSAttributedString]) {
        // Convert NSAttributedStrings to Data (NSKeyedArchiver format)
        var newContents: [Data] = []
        for newString in newStrings {
            do {
                let archivedData = try NSKeyedArchiver.archivedData(withRootObject: newString, requiringSecureCoding: true)
                newContents.append(archivedData)
            } catch {
                print("Failed to archive NSAttributedString: \(error)")
                newContents.append(Data())
            }
        }
        
        guard let data = serializeUpdateElementsMessage(substringId, elementIds, newContents) else {
            print("Failed to serialize updateElementsMessage")
            return
        }
        
        sendBinaryMessage(data)
    }
    
    private func registerIOSurfaceWithBootstrap(_ surface: IOSurface, substringId: UUID, surfaceIndex: Int) -> String {
        // Create unique service name
        let serviceName = "outercontent.surface.\(substringId.uuidString).\(surfaceIndex).\(getpid())"
        
        // Get mach port from IOSurface
        let surfacePort = IOSurfaceCreateMachPort(surface)
        
        // Register with bootstrap service
        let result = bootstrap_register_wrapper(serviceName, Int32(surfacePort))
        
        if result != 0 {
            print("⚠️ Failed to register IOSurface bootstrap service '\(serviceName)': \(result)")
        }
        
        return serviceName
    }
    
    func stop() {
        isRunning = false
        if clientSocket != -1 {
            close(clientSocket)
        }
        if serverSocket != -1 {
            close(serverSocket)
        }
        unlink(socketPath)
        
        // Cleanup shared memory
        sharedMemory?.cleanup()
        sharedMemory = nil
    }
    
    // MARK: - Shared Memory Render Loop
    
    private func setupSharedMemory() {
        let memoryName = "/outercontent_render_\(getpid())"
        sharedMemory = RenderLoopSynchronization(name: memoryName, create: true)
        
        if sharedMemory != nil {
            print("OuterContent: Shared memory render loop initialized")
            startRenderThread()
        } else {
            print("OuterContent: Failed to initialize shared memory, falling back to message-based rendering")
        }
    }
    
    private func startRenderThread() {
        renderThread = Thread {
            self.renderLoop()
        }
        renderThread?.start()
    }
    
    private func renderLoop() {
        Thread.current.name = "OuterContent-RenderLoop"
        
        while isRunning {
            // Wait for render timing signal from main app's display link
            guard let memory = sharedMemory, memory.waitForRenderFrame() else {
                continue
            }

            // Get target timestamp from shared memory and local fragment requests
            let targetTimestamp = memory.getTargetTimestamp()
            let fragmentRequests = self.substringsAwaitingAnimationFrame
            self.substringsAwaitingAnimationFrame.removeAll(keepingCapacity: true)

            if !fragmentRequests.isEmpty {
                // Process on main thread, in case the main thread (which may have been responsible for queueing these fragmentRequests) is still initializing.
                DispatchQueue.main.async {
                    // Process animation frames for all requesting fragments
                    self.processAnimationFrames(substringIds: Array(fragmentRequests), targetTimestamp: targetTimestamp)

                    // Signal completion so main app can process canvas updates
                    memory.signalRenderComplete()
                }
            }
        }
    }
    
    private func processAnimationFrames(substringIds: [UUID], targetTimestamp: CFTimeInterval) {
        guard let memory = sharedMemory else { return }

        for substringId in substringIds {
            guard let fragment = substringMap[substringId] else {
                print("OuterContent: Failed to find fragment with ID: \(substringId)")
                continue
            }

            // Generate element updates directly without messages
            if let elementUpdates = fragment.generateNextFrame(targetTimestamp: targetTimestamp) {
                // TODO: Store element updates in shared memory for main process to read
                sendUpdateElementsMessage(substringId: substringId, elementIds: elementUpdates.elementIds, newStrings: elementUpdates.newStrings)
            }

            // Update canvases - IOSurfaces are already set up from createDynamicSubstrings
            fragment.updateCanvases(targetTimestamp: targetTimestamp)

            // Just notify that this fragment has been updated
            memory.addFragmentUpdate(substringId)
        }
    }

    // Updated requestAnimationFrame to use shared memory when available
    fileprivate func requestAnimationFrameOptimized(substringId: UUID) {
        substringsAwaitingAnimationFrame.append(substringId)
        sharedMemory!.requestAnimationFrame()
    }
}

// Parse command line arguments
var socketPath = NSTemporaryDirectory().appending("outercontent.sock")
print("socketPath \(socketPath)")
var waitForDebuggerFlag = false

var i = 1
while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    if arg == "--wait-for-debugger" {
        waitForDebuggerFlag = true
    } else {
        socketPath = arg
    }
    i += 1
}

// Wait for debugger if requested
if waitForDebuggerFlag {
    waitForDebugger()
}

let server = OuterContentServer(socketPath: socketPath)

// Handle termination signals
signal(SIGINT) { _ in
    server.stop()
    exit(0)
}

signal(SIGTERM) { _ in
    server.stop()
    exit(0)
}

server.start()

// Load bundled plugins after starting the server
DispatchQueue.main.async {
    server.loadBundledPlugin()
}

// Keep the main thread alive
RunLoop.main.run()


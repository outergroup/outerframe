import Foundation
import AppKit
import Darwin
import QuartzCore
import Security

enum OuterframeContentConnectionError: Error {
    case executableNotFound
    case pluginLoadFailed(String)
    case sendFailed(Error)
    case disconnected
    case posixFailure(code: Int32, context: String)
    case pluginUnloadTimedOut
    case randomCredentialGenerationFailed(OSStatus)
}

extension OuterframeContentConnectionError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .executableNotFound:
            return "OuterframeContent executable not found."
        case .pluginLoadFailed(let message):
            return "Plugin load failed: \(message)"
        case .sendFailed(let underlyingError):
            return "Failed to send message to OuterframeContent: \(underlyingError.localizedDescription)"
        case .disconnected:
            return "OuterframeContent connection was disconnected."
        case .posixFailure(let code, let context):
            let description = String(cString: strerror(code))
            return "\(context) failed: \(description)"
        case .pluginUnloadTimedOut:
            return "Timed out waiting for plugin to unload."
        case .randomCredentialGenerationFailed(let status):
            return "Failed to generate SOCKS proxy credentials (Security status \(status))."
        }
    }
}

private func secureProxyCredential(byteCount: Int = 16) throws -> String {
    var bytes = [UInt8](repeating: 0, count: byteCount)
    let status = bytes.withUnsafeMutableBytes { buffer in
        SecRandomCopyBytes(kSecRandomDefault, byteCount, buffer.baseAddress!)
    }
    guard status == errSecSuccess else {
        throw OuterframeContentConnectionError.randomCredentialGenerationFailed(status)
    }

    return bytes.map { String(format: "%02x", $0) }.joined()
}

/// Cursor types that plugins can request
enum PluginCursorType: Int {
    case arrow = 0
    case iBeam = 1
    case crosshair = 2
    case openHand = 3
    case closedHand = 4
    case pointingHand = 5
    case resizeLeft = 6
    case resizeRight = 7
    case resizeLeftRight = 8
    case resizeUp = 9
    case resizeDown = 10
    case resizeUpDown = 11
}

enum OuterframeHapticFeedbackStyle: Int {
    case generic = 0
    case alignment = 1
    case levelChange = 2
}

/// Input modes that plugins can request. Represented as a bitmask so modes can be combined.
struct OuterframeContentInputMode: OptionSet, Sendable {
    let rawValue: UInt8

    init(rawValue: UInt8) {
        self.rawValue = rawValue
    }

    static let textInput = OuterframeContentInputMode(rawValue: 1 << 0)   // Keyboard events interpreted as text
    static let rawKeys = OuterframeContentInputMode(rawValue: 1 << 1)     // Raw key events forwarded to the plugin
    static let none: OuterframeContentInputMode = []

    var allowsTextInput: Bool { contains(.textInput) }
    var allowsRawKeys: Bool { contains(.rawKeys) }
}

/// Describes whether the plugin can currently satisfy copy/paste commands.
struct OuterframeContentEditingCapabilities: Sendable {
    var canCopy: Bool
    var canCut: Bool
    var acceptablePasteboardTypeIdentifiers: [String]

    init(canCopy: Bool,
                canCut: Bool,
                acceptablePasteboardTypeIdentifiers: [String]) {
        self.canCopy = canCopy
        self.canCut = canCut
        self.acceptablePasteboardTypeIdentifiers = acceptablePasteboardTypeIdentifiers
    }
}


protocol OuterframeContentConnectionDelegate: AnyObject {
    func registerDisplayLinkCallback() -> UUID
    func unregisterDisplayLinkCallback(_ callbackId: UUID)
    func handlePluginLoaded(contextId: CAContextID)
    func updateCursor(_ cursorType: PluginCursorType)
    func setInputMode(_ inputMode: OuterframeContentInputMode)
    func showPluginRequestedContextMenu(attributedText: NSAttributedString, at location: CGPoint)
    func showPluginRequestedDefinition(attributedText: NSAttributedString, at location: CGPoint)
    func handleTextCursorUpdate(cursors: [OuterframeContentTextCursorSnapshot])
    func handleAccessibilityTreeChanged(notificationMask: UInt8)
    func handlePluginOpenNewWindow(urlString: String, displayString: String?, preferredSize: CGSize?)
    func setPasteboardCapabilities(canCopy: Bool, canCut: Bool, pasteboardTypeIdentifiers: [String])
    func outerframeContentConnectionDebuggerAttached(_ connection: OuterframeContentConnection)
    func outerframeContentConnectionDidTimeoutWaitingForPluginLoaded(_ connection: OuterframeContentConnection)
    func outerframeContentConnection(_ connection: OuterframeContentConnection, didReceiveStdout text: String)
    func outerframeContentConnection(_ connection: OuterframeContentConnection, didReceiveStderr text: String)
    func performHapticFeedback(_ style: OuterframeHapticFeedbackStyle)
}

/// Manages the connection to OuterframeContent process
@MainActor
final class OuterframeContentConnection: NSObject {
    private nonisolated(unsafe) static let outerframeProcessesConnection: NSXPCConnection = {
        let connection = NSXPCConnection(serviceName: OuterframeConfiguration.processesXPCServiceName)
        connection.remoteObjectInterface = NSXPCInterface(with: OuterframeProcessesProtocol.self)
        connection.resume()
        return connection
    }()

    override init() {
        super.init()
    }

    /// Socket for infrastructure messages (loadPlugin, pluginLoaded, etc.)
    private let infrastructureSocket = OuterframeContentSocket()
    /// Socket for plugin messages (mouse, keyboard, display link, etc.)
    private let pluginSocket = OuterframeContentSocket()
    private var stdoutHandle: FileHandle?
    private var stderrHandle: FileHandle?
    private var outerframeContentPID: pid_t?
    private var outerframeContentInstanceId: UUID?
    private var outerframeContentTerminationSource: DispatchSourceProcess?
    private var socketReady = false
    private var awaitingPluginLoaded = false
    private var pluginLoadedTimeoutTask: Task<Void, Never>?

    private var networkProxyPort: UInt16?
    private var proxyUsername: String?
    private var proxyPassword: String?

    weak var delegate: OuterframeContentConnectionDelegate?

    // Store load plugin continuations
    private var loadPluginContinuations: [UUID: CheckedContinuation<Void, Error>] = [:]

    // Store pending copy requests waiting on plugin response
    private var pendingCopyRequests: [UUID: ([OuterframeContentPasteboardItem]?) -> Void] = [:]
    private var pendingAccessibilitySnapshotRequests: [UUID: (OuterframeAccessibilitySnapshot?) -> Void] = [:]
    private var unloadPluginRequestInFlight = false
    private var unloadPluginContinuations: [CheckedContinuation<Void, Error>] = []
    private var unloadPluginTimeoutTask: Task<Void, Never>?

    private var lastSystemAppearancePayload: Data?

    var processIdentifier: pid_t? {
        outerframeContentPID
    }

    var currentNetworkProxyPort: UInt16? {
        networkProxyPort
    }

    private func failPendingLoadPluginRequests(with error: Error) {
        failPendingCopyRequests()
        failPendingAccessibilityRequests()
        guard !loadPluginContinuations.isEmpty else { return }
        let continuations = loadPluginContinuations
        loadPluginContinuations.removeAll()
        for continuation in continuations.values {
            continuation.resume(throwing: error)
        }
    }

    private func failPendingCopyRequests() {
        guard !pendingCopyRequests.isEmpty else { return }
        let requests = pendingCopyRequests
        pendingCopyRequests.removeAll()
        for completion in requests.values {
            completion(nil)
        }
    }

    private func failPendingAccessibilityRequests() {
        guard !pendingAccessibilitySnapshotRequests.isEmpty else { return }
        let requests = pendingAccessibilitySnapshotRequests
        pendingAccessibilitySnapshotRequests.removeAll()
        for completion in requests.values {
            completion(nil)
        }
    }

    private func startOutputMonitoring(stdout: FileHandle, stderr: FileHandle) {
        stdoutHandle = stdout
        stderrHandle = stderr

        stdout.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            Task { @MainActor in
                self?.handleProcessOutput(data, isError: false)
            }
        }

        stderr.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            Task { @MainActor in
                self?.handleProcessOutput(data, isError: true)
            }
        }
    }

    private func stopOutputMonitoring() {
        stdoutHandle?.readabilityHandler = nil
        stderrHandle?.readabilityHandler = nil
        stdoutHandle?.closeFile()
        stderrHandle?.closeFile()
        stdoutHandle = nil
        stderrHandle = nil
    }

    private func handleProcessOutput(_ data: Data, isError: Bool) {
        let text = String(decoding: data, as: UTF8.self)
        if isError {
            delegate?.outerframeContentConnection(self, didReceiveStderr: text)
        } else {
            delegate?.outerframeContentConnection(self, didReceiveStdout: text)
        }
    }

    private func schedulePluginLoadedTimeout() {
        cancelPluginLoadedTimeout()
        awaitingPluginLoaded = true
        pluginLoadedTimeoutTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: 2_000_000_000)
            } catch {
                return
            }
            await MainActor.run {
                guard let self else { return }
                guard self.awaitingPluginLoaded else { return }
                self.awaitingPluginLoaded = false
                self.delegate?.outerframeContentConnectionDidTimeoutWaitingForPluginLoaded(self)
            }
        }
    }

    private func cancelPluginLoadedTimeout() {
        awaitingPluginLoaded = false
        pluginLoadedTimeoutTask?.cancel()
        pluginLoadedTimeoutTask = nil
    }

    private func cancelUnloadPluginTimeout() {
        let timeoutTask = unloadPluginTimeoutTask
        unloadPluginTimeoutTask = nil
        timeoutTask?.cancel()
    }

    private func completeUnloadPluginRequest() {
        guard unloadPluginRequestInFlight else { return }
        unloadPluginRequestInFlight = false
        cancelUnloadPluginTimeout()
        guard !unloadPluginContinuations.isEmpty else { return }
        let continuations = unloadPluginContinuations
        unloadPluginContinuations.removeAll(keepingCapacity: false)
        for continuation in continuations {
            continuation.resume(returning: ())
        }
    }

    private func failPendingUnloadPluginRequests(with error: Error) {
        guard unloadPluginRequestInFlight || !unloadPluginContinuations.isEmpty else {
            cancelUnloadPluginTimeout()
            return
        }
        unloadPluginRequestInFlight = false
        cancelUnloadPluginTimeout()
        guard !unloadPluginContinuations.isEmpty else { return }
        let continuations = unloadPluginContinuations
        unloadPluginContinuations.removeAll(keepingCapacity: false)
        for continuation in continuations {
            continuation.resume(throwing: error)
        }
    }

    private func startUnloadPluginTimeoutTimer(nanoseconds: UInt64) {
        cancelUnloadPluginTimeout()
        guard nanoseconds > 0 else { return }
        unloadPluginTimeoutTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: nanoseconds)
            } catch {
                return
            }
            await MainActor.run {
                guard let self else { return }
                guard self.unloadPluginRequestInFlight else { return }
                self.failPendingUnloadPluginRequests(with: OuterframeContentConnectionError.pluginUnloadTimedOut)
            }
        }
    }

    func start(networkProxyEndpoint: OuterframeNetworkProxyEndpoint) async throws {
        let proxyPort = networkProxyEndpoint.port
        networkProxyPort = proxyPort
        proxyUsername = networkProxyEndpoint.username
        proxyPassword = networkProxyEndpoint.password

        do {
            let (infraHandle, pluginHandle, stdoutHandle, stderrHandle, pid, instanceId) = try await Self.launchOuterframeContentViaXPC(
                networkProxyPort: proxyPort
            )
            let infraFD = dup(infraHandle.fileDescriptor)
            let pluginFD = dup(pluginHandle.fileDescriptor)
            guard infraFD >= 0 else {
                throw OuterframeContentConnectionError.posixFailure(code: errno, context: "dup infra socket from XPC")
            }
            guard pluginFD >= 0 else {
                close(infraFD)
                throw OuterframeContentConnectionError.posixFailure(code: errno, context: "dup plugin socket from XPC")
            }
            let stdoutFD = dup(stdoutHandle.fileDescriptor)
            guard stdoutFD >= 0 else {
                close(infraFD)
                close(pluginFD)
                throw OuterframeContentConnectionError.posixFailure(code: errno, context: "dup stdout from XPC")
            }
            let stderrFD = dup(stderrHandle.fileDescriptor)
            guard stderrFD >= 0 else {
                close(infraFD)
                close(pluginFD)
                close(stdoutFD)
                throw OuterframeContentConnectionError.posixFailure(code: errno, context: "dup stderr from XPC")
            }
            let stdoutReadHandle = FileHandle(fileDescriptor: stdoutFD, closeOnDealloc: true)
            let stderrReadHandle = FileHandle(fileDescriptor: stderrFD, closeOnDealloc: true)

            outerframeContentPID = pid
            outerframeContentInstanceId = instanceId
            monitorOuterframeContentTermination(pid: pid)

            await infrastructureSocket.setDelegate(self)
            await infrastructureSocket.start(withFileDescriptor: infraFD)

            await pluginSocket.setDelegate(self)
            await pluginSocket.start(withFileDescriptor: pluginFD)

            startOutputMonitoring(stdout: stdoutReadHandle, stderr: stderrReadHandle)
            socketReady = true
            print("Launched OuterframeContent process \(pid) via XPC")
        } catch {
            networkProxyPort = nil
            proxyUsername = nil
            proxyPassword = nil
            await infrastructureSocket.stop()
            await pluginSocket.stop()
            stopOutputMonitoring()
            outerframeContentPID = nil
            socketReady = false
            failPendingLoadPluginRequests(with: error)
            failPendingCopyRequests()
            failPendingAccessibilityRequests()
            failPendingUnloadPluginRequests(with: error)
            throw error
        }
    }

    private nonisolated static func launchOuterframeContentViaXPC(networkProxyPort: UInt16) async throws -> (FileHandle, FileHandle, FileHandle, FileHandle, pid_t, UUID) {
        try await withCheckedThrowingContinuation { continuation in
            guard let proxy = outerframeProcessesConnection.remoteObjectProxy as? OuterframeProcessesProtocol else {
                continuation.resume(throwing: OuterframeContentConnectionError.posixFailure(code: -1, context: "XPC proxy creation"))
                return
            }

            proxy.launchOuterframeContent(networkProxyPort: networkProxyPort) { infraHandle, pluginHandle, stdoutHandle, stderrHandle, pid, instanceId, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let infraHandle,
                          let pluginHandle,
                          let stdoutHandle,
                          let stderrHandle,
                          let instanceId {
                    continuation.resume(returning: (infraHandle, pluginHandle, stdoutHandle, stderrHandle, pid, instanceId))
                } else {
                    continuation.resume(throwing: OuterframeContentConnectionError.posixFailure(code: -1, context: "XPC returned nil"))
                }
            }
        }
    }

    /// Phase 1 of external attachment: Start the network proxy and return its port.
    /// Call this before sending the handshake response, then call attachExternalSocket after.
    func prepareExternalConnection() async throws -> UInt16 {
        let proxyPort = try await OuterframeNetworkProxyConnection.shared.start()
        networkProxyPort = proxyPort
        proxyUsername = try secureProxyCredential()
        proxyPassword = try secureProxyCredential()
        return proxyPort
    }

    /// Phase 2 of external attachment: Attach the sockets after handshake response is sent.
    /// Must call prepareExternalConnection first.
    /// The infrastructureSocketFD carries loadPlugin/pluginLoaded etc., while pluginSocketFD carries
    /// mouse/keyboard/display link messages.
    func attachExternalSocket(infrastructureSocketFD: Int32, pluginSocketFD: Int32) async {
        // Configure socket options for infrastructure socket
        let infraFlags = fcntl(infrastructureSocketFD, F_GETFL)
        if infraFlags != -1 {
            _ = fcntl(infrastructureSocketFD, F_SETFL, infraFlags | O_NONBLOCK)
        }
        var noSigPipe: Int32 = 1
        setsockopt(infrastructureSocketFD, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))

        // Configure socket options for plugin socket
        let pluginFlags = fcntl(pluginSocketFD, F_GETFL)
        if pluginFlags != -1 {
            _ = fcntl(pluginSocketFD, F_SETFL, pluginFlags | O_NONBLOCK)
        }
        setsockopt(pluginSocketFD, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))

        // Start socket communication
        await infrastructureSocket.setDelegate(self)
        await infrastructureSocket.start(withFileDescriptor: infrastructureSocketFD)

        await pluginSocket.setDelegate(self)
        await pluginSocket.start(withFileDescriptor: pluginSocketFD)

        socketReady = true

        print("Browser: Attached external OuterframeContent sockets, proxy port: \(networkProxyPort ?? 0)")
    }

    private func monitorOuterframeContentTermination(pid: pid_t) {
        outerframeContentTerminationSource?.cancel()

        let source = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: .main)
        source.setEventHandler { [weak self] in
            guard let self else { return }
            var status: Int32 = 0
            waitpid(pid, &status, 0)
            let infraSocket = self.infrastructureSocket
            let plugSocket = self.pluginSocket
            Task {
                await infraSocket.stop()
                await plugSocket.stop()
            }
            self.outerframeContentPID = nil
            self.outerframeContentTerminationSource?.cancel()
        }
        source.setCancelHandler { [weak self] in
            self?.outerframeContentTerminationSource = nil
        }
        outerframeContentTerminationSource = source
        source.resume()
    }

    private func decodeAttributedString(from data: Data) -> NSAttributedString? {
        if let attributed = try? NSKeyedUnarchiver.unarchivedObject(ofClass: NSAttributedString.self, from: data) {
            return attributed
        }

        if let attributed = try? NSAttributedString(data: data,
                                                    options: [.documentType: NSAttributedString.DocumentType.rtf],
                                                    documentAttributes: nil) {
            return attributed
        }

        print("Browser: Failed to decode attributed text from data of length \(data.count)")
        return nil
    }

    func loadPlugin(url: URL,
                    data: Data,
                    outerURLString: String,
                    bundleURLString: String,
                    width: CGFloat,
                    height: CGFloat) async throws {
        let requestId = UUID()

        let proxyHost: String?
        if let port = networkProxyPort, port > 0 {
            proxyHost = "127.0.0.1"
        } else {
            proxyHost = nil
        }

        let infraMessage = BrowserToContentInfraMessage.loadPlugin(requestId: requestId,
                                                                  pluginURL: url.absoluteString)

        var proxy: InitializeContentProxy?
        if let proxyHost, let proxyPort = networkProxyPort {
            proxy = InitializeContentProxy(host: proxyHost,
                                           port: proxyPort,
                                           username: proxyUsername,
                                           password: proxyPassword)
        }

        let initializeContentArguments = InitializeContentArguments(
            data: data,
            contentWidth: width,
            contentHeight: height,
            appearance: NSApp.effectiveAppearance,
            proxy: proxy,
            url: outerURLString,
            bundleUrl: bundleURLString,
            windowIsActive: true
        )
        let initializeContentMessage = BrowserToContentMessage.initializeContent(args: initializeContentArguments)

        try await withCheckedThrowingContinuation { continuation in
            loadPluginContinuations[requestId] = continuation

            Task { @MainActor in
                do {
                    // Send infrastructure message to tell OuterframeContent which bundle to load
                    try await self.infrastructureSocket.send(infraMessage.encode())

                    // Send initializeContent directly to the plugin over the plugin socket
                    try await self.pluginSocket.send(initializeContentMessage.encode())
                    self.schedulePluginLoadedTimeout()
                } catch {
                    print("Browser: Failed to send loadPlugin messages: \(error)")

                    if let continuation = loadPluginContinuations.removeValue(forKey: requestId) {
                        continuation.resume(throwing: OuterframeContentConnectionError.sendFailed(error))
                    }
                }
            }
        }
    }

    func unloadPlugin(timeout: TimeInterval) async throws {
        let timeoutNanoseconds = timeout > 0 ? UInt64(timeout * 1_000_000_000) : nil

        try await withCheckedThrowingContinuation { continuation in
            unloadPluginContinuations.append(continuation)

            guard !unloadPluginRequestInFlight else { return }
            unloadPluginRequestInFlight = true

            if let timeoutNanoseconds {
                startUnloadPluginTimeoutTimer(nanoseconds: timeoutNanoseconds)
            } else {
                cancelUnloadPluginTimeout()
            }

            let message = BrowserToContentInfraMessage.unloadPlugin
            Task { @MainActor in
                do {
                    try await self.infrastructureSocket.send(message.encode())
                } catch {
                    print("Browser: Failed to send unloadPlugin message to OuterframeContent: \(error)")
                    self.failPendingUnloadPluginRequests(with: OuterframeContentConnectionError.sendFailed(error))
                }
            }
        }
    }

    func resizeContent(width: CGFloat, height: CGFloat) {
        let message = BrowserToContentMessage.resizeContent(width: width, height: height)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send resizeContent message to OuterframeContent: \(error)")
            }
        }
    }

    enum MouseEventType {
        case down
        case dragged
        case up
        case moved
        case rightDown
        case rightUp
    }

    func sendMouseEvent(type: MouseEventType, point: CGPoint, modifierFlags: NSEvent.ModifierFlags, clickCount: Int = 0) {
        let x = Float32(point.x)
        let y = Float32(point.y)
        let flags = UInt64(modifierFlags.rawValue)
        let clicks = UInt32(clickCount)

        let kind: OuterframeContentMouseEventKind
        switch type {
        case .down:
            kind = .mouseDown
        case .dragged:
            kind = .mouseDragged
        case .up:
            kind = .mouseUp
        case .moved:
            kind = .mouseMoved
        case .rightDown:
            kind = .rightMouseDown
        case .rightUp:
            kind = .rightMouseUp
        }

        let message = BrowserToContentMessage.mouseEvent(kind: kind,
                                                              x: x,
                                                              y: y,
                                                              modifierFlags: flags,
                                                              clickCount: clicks)

        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send mouse event to OuterframeContent: \(error)")
            }
        }
    }

    func sendScrollWheelEvent(point: CGPoint,
                              delta: CGPoint,
                              modifierFlags: NSEvent.ModifierFlags,
                              phase: NSEvent.Phase,
                              momentumPhase: NSEvent.Phase,
                              isMomentum: Bool,
                              isPrecise: Bool) {
        let x = Float32(point.x)
        let y = Float32(point.y)
        let deltaX = Float32(delta.x)
        let deltaY = Float32(delta.y)
        let flags = UInt64(modifierFlags.rawValue)
        let phaseRaw = UInt32(truncatingIfNeeded: phase.rawValue)
        let momentumPhaseRaw = UInt32(truncatingIfNeeded: momentumPhase.rawValue)

        let message = BrowserToContentMessage.scrollWheelEvent(x: x,
                                                                    y: y,
                                                                    deltaX: deltaX,
                                                                    deltaY: deltaY,
                                                                    modifierFlags: flags,
                                                                    phase: phaseRaw,
                                                                    momentumPhase: momentumPhaseRaw,
                                                                    isMomentum: isMomentum,
                                                                    isPrecise: isPrecise)

        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send scroll wheel event to OuterframeContent: \(error)")
            }
        }
    }

    func sendQuickLookEvent(point: CGPoint) {
        let message = BrowserToContentMessage.quickLook(x: Float32(point.x),
                                                             y: Float32(point.y))
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send quickLook message to OuterframeContent: \(error)")
            }
        }
    }

    func sendMagnification(surfaceId: Int,
                           magnification delta: CGFloat,
                           location: CGPoint,
                           scrollOffset: CGPoint) {
        precondition(surfaceId >= 0 && surfaceId <= Int(UInt32.max), "Surface identifier out of range")
        let message = BrowserToContentMessage.magnification(surfaceId: UInt32(surfaceId),
                                                                 magnification: Float32(delta),
                                                                 x: Float32(location.x),
                                                                 y: Float32(location.y),
                                                                 scrollX: Float32(scrollOffset.x),
                                                                 scrollY: Float32(scrollOffset.y))

        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send magnification message: \(error)")
            }
        }
    }

    func sendMagnificationEnded(surfaceId: Int,
                                magnification delta: CGFloat,
                                location: CGPoint,
                                scrollOffset: CGPoint) {
        precondition(surfaceId >= 0 && surfaceId <= Int(UInt32.max), "Surface identifier out of range")
        let message = BrowserToContentMessage.magnificationEnded(surfaceId: UInt32(surfaceId),
                                                                      magnification: Float32(delta),
                                                                      x: Float32(location.x),
                                                                      y: Float32(location.y),
                                                                      scrollX: Float32(scrollOffset.x),
                                                                      scrollY: Float32(scrollOffset.y))

        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send magnificationEnded message: \(error)")
            }
        }
    }

    func sendDisplayLinkFrame(frameNumber: UInt64, targetTimestamp: CFTimeInterval) {
        let pluginSocket = self.pluginSocket
        Task {
            do {
                let message = BrowserToContentMessage.displayLinkFired(frameNumber: frameNumber, targetTimestamp: targetTimestamp)
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send display link frame message: \(error)")
            }
        }
    }

    // MARK: - System Colors

    func sendSystemAppearanceUpdate(force: Bool = false) {
        let appearance = NSApp.effectiveAppearance

        // Cache serialized data to detect changes
        let data: Data
        do {
            data = try NSKeyedArchiver.archivedData(withRootObject: appearance, requiringSecureCoding: true)
        } catch {
            print("Browser: Failed to serialize system appearance: \(error)")
            return
        }

        if !force, let last = lastSystemAppearancePayload, last == data {
            return
        }

        lastSystemAppearancePayload = data

        let message = BrowserToContentMessage.systemAppearanceUpdate(appearance: appearance)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send systemAppearanceUpdate message: \(error)")
            }
        }
    }

    func sendWindowActiveState(isActive: Bool) {
        let message = BrowserToContentMessage.windowActiveUpdate(isActive: isActive)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send windowActiveUpdate message: \(error)")
            }
        }
    }

    func sendViewFocusChanged(isFocused: Bool) {
        let message = BrowserToContentMessage.viewFocusChanged(isFocused: isFocused)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send viewFocusChanged message: \(error)")
            }
        }
    }

    func setDebuggerAttachmentMonitoring(isEnabled: Bool) {
        let message = BrowserToContentInfraMessage.setDebuggerAttachmentMonitoring(isEnabled: isEnabled)
        let infrastructureSocket = self.infrastructureSocket
        Task {
            do {
                try await infrastructureSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send setDebuggerAttachmentMonitoring message: \(error)")
            }
        }
    }

    // MARK: - Text Input Support

    func sendTextInput(text: String, replacementRange: NSRange) {
        let hasRange = replacementRange.location != NSNotFound
        let locationValue = hasRange ? UInt64(clamping: replacementRange.location) : 0
        let lengthValue = hasRange ? UInt64(clamping: replacementRange.length) : 0
        let message = BrowserToContentMessage.textInput(text: text,
                                                             hasReplacementRange: hasRange,
                                                             replacementLocation: locationValue,
                                                             replacementLength: lengthValue)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send textInput message: \(error)")
            }
        }
    }

    func sendSetMarkedText(text: String, selectedRange: NSRange, replacementRange: NSRange) {
        let hasRange = replacementRange.location != NSNotFound
        let selectedLocation = UInt64(clamping: selectedRange.location)
        let selectedLength = UInt64(clamping: selectedRange.length)
        let replacementLocation = hasRange ? UInt64(clamping: replacementRange.location) : 0
        let replacementLength = hasRange ? UInt64(clamping: replacementRange.length) : 0
        let message = BrowserToContentMessage.setMarkedText(text: text,
                                                                 selectedLocation: selectedLocation,
                                                                 selectedLength: selectedLength,
                                                                 hasReplacementRange: hasRange,
                                                                 replacementLocation: replacementLocation,
                                                                 replacementLength: replacementLength)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send setMarkedText message: \(error)")
            }
        }
    }

    func sendUnmarkText() {
        let message = BrowserToContentMessage.unmarkText
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send unmarkText message: \(error)")
            }
        }
    }

    func sendTextInputFocus(fieldID: UUID, hasFocus: Bool) {
        let message = BrowserToContentMessage.textInputFocus(fieldID: fieldID,
                                                                  hasFocus: hasFocus)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send textInputFocus message: \(error)")
            }
        }
    }

    func sendTextCommand(command: String) {
        let message = BrowserToContentMessage.textCommand(command: command)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send textCommand message: \(error)")
            }
        }
    }

    func requestPasteboardItemsForCopy(completion: @escaping ([OuterframeContentPasteboardItem]?) -> Void) {
        guard socketReady else {
            completion(nil)
            return
        }

        let requestId = UUID()
        pendingCopyRequests[requestId] = completion
        let message = BrowserToContentMessage.copySelectedPasteboardRequest(requestId: requestId)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send copySelectedPasteboardRequest message: \(error)")
                if let pending = self.pendingCopyRequests.removeValue(forKey: requestId) {
                    pending(nil)
                }
            }
        }
    }

    func requestAccessibilitySnapshot(completion: @escaping (OuterframeAccessibilitySnapshot?) -> Void) {
        guard socketReady else {
            completion(nil)
            return
        }

        let requestId = UUID()
        pendingAccessibilitySnapshotRequests[requestId] = completion
        let message = BrowserToContentMessage.accessibilitySnapshotRequest(requestId: requestId)
        let pluginSocket = self.pluginSocket
        Task { [weak self] in
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send accessibilitySnapshotRequest message: \(error)")
                await MainActor.run {
                    guard let self else { return }
                    if let pending = self.pendingAccessibilitySnapshotRequests.removeValue(forKey: requestId) {
                        pending(nil)
                    }
                }
            }
        }
    }

    func sendPasteboardItemsForPaste(_ items: [OuterframeContentPasteboardItem]) {
        let message = BrowserToContentMessage.pasteboardContentDelivered(items: items)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send pasteboardContentDelivered message: \(error)")
            }
        }
    }

    func sendSetCursorPosition(fieldID: UUID, position: UInt64, modifySelection: Bool) {
        let message = BrowserToContentMessage.setCursorPosition(fieldID: fieldID,
                                                                     position: position,
                                                                     modifySelection: modifySelection)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send setCursorPosition message: \(error)")
            }
        }
    }

    func sendKeyDown(keyCode: UInt16, characters: String, charactersIgnoringModifiers: String, modifierFlags: NSEvent.ModifierFlags, isRepeat: Bool) {
        let message = BrowserToContentMessage.keyDown(keyCode: keyCode,
                                                           characters: characters,
                                                           charactersIgnoringModifiers: charactersIgnoringModifiers,
                                                           modifierFlags: UInt64(modifierFlags.rawValue),
                                                           isRepeat: isRepeat)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send keyDown message to OuterframeContent: \(error)")
            }
        }
    }

    func sendKeyUp(keyCode: UInt16, characters: String, charactersIgnoringModifiers: String, modifierFlags: NSEvent.ModifierFlags, isRepeat: Bool) {
        let message = BrowserToContentMessage.keyUp(keyCode: keyCode,
                                                         characters: characters,
                                                         charactersIgnoringModifiers: charactersIgnoringModifiers,
                                                         modifierFlags: UInt64(modifierFlags.rawValue),
                                                         isRepeat: isRepeat)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send keyUp message to OuterframeContent: \(error)")
            }
        }
    }

    private func handleGetImageWithSystemSymbolName(requestId: UUID, symbolName: String, pointSize: Float, weight: String, scale: Float, tintRed: Float, tintGreen: Float, tintBlue: Float, tintAlpha: Float) {
        // Create and configure the SF Symbol image
        guard let image = NSImage(systemSymbolName: symbolName, accessibilityDescription: nil) else {
            self.sendImageWithSystemSymbolNameResponse(requestId: requestId, imageData: nil, width: 0, height: 0, success: false, errorMessage: "Symbol '\(symbolName)' not found")
            return
        }

        // Configure the symbol
        var symbolWeight: NSFont.Weight = .regular
        switch weight.lowercased() {
        case "ultralight": symbolWeight = .ultraLight
        case "thin": symbolWeight = .thin
        case "light": symbolWeight = .light
        case "regular": symbolWeight = .regular
        case "medium": symbolWeight = .medium
        case "semibold": symbolWeight = .semibold
        case "bold": symbolWeight = .bold
        case "heavy": symbolWeight = .heavy
        case "black": symbolWeight = .black
        default: symbolWeight = .regular
        }

        let config = NSImage.SymbolConfiguration(pointSize: CGFloat(pointSize), weight: symbolWeight)
        guard let configuredImage = image.withSymbolConfiguration(config) else {
            self.sendImageWithSystemSymbolNameResponse(requestId: requestId, imageData: nil, width: 0, height: 0, success: false, errorMessage: "Failed to configure symbol")
            return
        }

        // Create a bitmap representation with the tint color
        let imageSize = CGSize(width: configuredImage.size.width * CGFloat(scale),
                               height: configuredImage.size.height * CGFloat(scale))

        guard let bitmapRep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(imageSize.width),
            pixelsHigh: Int(imageSize.height),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ) else {
            self.sendImageWithSystemSymbolNameResponse(requestId: requestId, imageData: nil, width: 0, height: 0, success: false, errorMessage: "Failed to create bitmap")
            return
        }

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmapRep)

        // Clear background
        NSColor.clear.set()
        NSRect(origin: .zero, size: imageSize).fill()

        // Draw the symbol with tint
        let tintColor = NSColor(red: CGFloat(tintRed), green: CGFloat(tintGreen), blue: CGFloat(tintBlue), alpha: CGFloat(tintAlpha))
        configuredImage.draw(in: NSRect(origin: .zero, size: imageSize), from: .zero, operation: .sourceOver, fraction: 1.0)

        // Apply tint using source-in compositing
        tintColor.set()
        NSRect(origin: .zero, size: imageSize).fill(using: .sourceIn)

        NSGraphicsContext.restoreGraphicsState()

        // Convert to PNG data
        guard let pngData = bitmapRep.representation(using: .png, properties: [:]) else {
            self.sendImageWithSystemSymbolNameResponse(requestId: requestId, imageData: nil, width: 0, height: 0, success: false, errorMessage: "Failed to create PNG data")
            return
        }

        // Send response back to plugin
        self.sendImageWithSystemSymbolNameResponse(requestId: requestId, imageData: pngData, width: UInt32(imageSize.width), height: UInt32(imageSize.height), success: true, errorMessage: nil)
    }

    private func sendImageWithSystemSymbolNameResponse(requestId: UUID, imageData: Data?, width: UInt32, height: UInt32, success: Bool, errorMessage: String?) {
        let message = BrowserToContentMessage.imageWithSystemSymbolName(requestId: requestId,
                                                                             imageData: imageData,
                                                                             width: width,
                                                                             height: height,
                                                                             success: success,
                                                                             errorMessage: errorMessage)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send imageWithSystemSymbolName message: \(error)")
            }
        }
    }

    func cleanup() async {
        cancelPluginLoadedTimeout()
        failPendingUnloadPluginRequests(with: OuterframeContentConnectionError.disconnected)

        // Send shutdown messages so plugin and OuterframeContent can clean up
        if outerframeContentPID != nil {
            // First, send shutdown to plugin so it can clean up
            do {
                let pluginShutdownMessage = BrowserToContentMessage.shutdown
                try await pluginSocket.send(pluginShutdownMessage.encode())
            } catch {
                print("Browser: Failed to send shutdown message to plugin: \(error)")
            }

            // Then, send shutdown to OuterframeContent infrastructure
            do {
                let infraShutdownMessage = BrowserToContentInfraMessage.shutdown
                try await infrastructureSocket.send(infraShutdownMessage.encode())
            } catch {
                print("Browser: Failed to send shutdown message to OuterframeContent: \(error)")
            }
        }

        // Ask XPC service to ensure the process exits (with 2 second timeout)
        if let instanceId = outerframeContentInstanceId {
            await Self.ensureOuterframeContentExitsViaXPC(instanceId: instanceId, timeout: 2.0)
        }

        outerframeContentPID = nil
        outerframeContentInstanceId = nil
        outerframeContentTerminationSource?.cancel()
        outerframeContentTerminationSource = nil

        failPendingLoadPluginRequests(with: OuterframeContentConnectionError.disconnected)
        failPendingCopyRequests()
        failPendingAccessibilityRequests()

        lastSystemAppearancePayload = nil

        networkProxyPort = nil
        proxyUsername = nil
        proxyPassword = nil

        await infrastructureSocket.stop()
        await pluginSocket.stop()
        stopOutputMonitoring()
    }

    private nonisolated static func ensureOuterframeContentExitsViaXPC(instanceId: UUID, timeout: TimeInterval) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            guard let proxy = outerframeProcessesConnection.remoteObjectProxyWithErrorHandler({ error in
                print("Browser: XPC error during ensureOuterframeContentExits: \(error)")
                continuation.resume()
            }) as? OuterframeProcessesProtocol else {
                print("Browser: Failed to get XPC proxy for ensureOuterframeContentExits")
                continuation.resume()
                return
            }

            proxy.ensureOuterframeContentExits(instanceId: instanceId, timeout: timeout) { error in
                if let error = error {
                    print("Browser: ensureOuterframeContentExits error: \(error)")
                }
                continuation.resume()
            }
        }
    }
}

extension OuterframeContentConnection: OuterframeContentSocketDelegate {
    @MainActor func outerframeContentSocket(_ socket: OuterframeContentSocket, didReceiveMessageType typeRaw: UInt16, payload: Data) {
        if !socketReady {
            socketReady = true
        }

        // Route to appropriate handler based on which socket received the message
        if socket === infrastructureSocket {
            handleInfrastructureMessage(typeRaw: typeRaw, payload: payload)
        } else {
            handlePluginMessage(typeRaw: typeRaw, payload: payload)
        }
    }

    private func handleInfrastructureMessage(typeRaw: UInt16, payload: Data) {
        let message: ContentToBrowserInfraMessage
        do {
            message = try ContentToBrowserInfraMessage.decode(typeRaw: typeRaw, payload: payload)
        } catch {
            print("Browser: Failed to decode infrastructure message (type \(typeRaw)): \(error)")
            return
        }

        switch message {
        case .loadPluginSuccess(let requestId):
            guard let continuation = loadPluginContinuations.removeValue(forKey: requestId) else {
                print("Browser: Received loadPluginSuccess for unknown request id \(requestId)")
                return
            }
            continuation.resume(returning: ())

        case .loadPluginFailure(let requestId, let errorMessage):
            cancelPluginLoadedTimeout()
            guard let continuation = loadPluginContinuations.removeValue(forKey: requestId) else {
                print("Browser: Received loadPluginFailure for unknown request id \(requestId): \(errorMessage)")
                return
            }
            continuation.resume(throwing: OuterframeContentConnectionError.pluginLoadFailed(errorMessage))

        case .pluginLoaded(let contextId, let success):
            cancelPluginLoadedTimeout()
            print("Browser: Received pluginLoaded message - contextId: \(contextId), success: \(success)")
            guard success else { return }
            guard let delegate = delegate else { return }
            delegate.handlePluginLoaded(contextId: contextId)

        case .pluginUnloaded:
            cancelPluginLoadedTimeout()
            completeUnloadPluginRequest()

        case .debuggerAttached:
            delegate?.outerframeContentConnectionDebuggerAttached(self)
        }
    }

    private func handlePluginMessage(typeRaw: UInt16, payload: Data) {
        let message: ContentToBrowserMessage
        do {
            message = try ContentToBrowserMessage.decode(typeRaw: typeRaw, payload: payload)
        } catch {
            print("Browser: Failed to decode plugin message (type \(typeRaw)): \(error)")
            return
        }

        switch message {
        case .startDisplayLink(let callbackId):
            guard let delegate = delegate else { return }
            let browserCallbackId = delegate.registerDisplayLinkCallback()
            let ack = BrowserToContentMessage.displayLinkCallbackRegistered(callbackId: callbackId, browserCallbackId: browserCallbackId)
            Task {
                do {
                    try await self.pluginSocket.send(ack.encode())
                } catch {
                    print("Browser: Failed to send displayLinkCallbackRegistered message: \(error)")
                }
            }

        case .stopDisplayLink(let browserCallbackId):
            delegate?.unregisterDisplayLinkCallback(browserCallbackId)

        case .cursorUpdate(let cursorType):
            guard let delegate,
                  let type = PluginCursorType(rawValue: Int(cursorType)) else { return }
            delegate.updateCursor(type)

        case .inputModeUpdate(let inputMode):
            guard let delegate else { return }
            let mode = OuterframeContentInputMode(rawValue: inputMode)
            delegate.setInputMode(mode)

        case .showContextMenu(let attributedTextData, let locationX, let locationY):
            guard let delegate = delegate else { return }
            let location = CGPoint(x: CGFloat(locationX), y: CGFloat(locationY))
            guard let attributedText = decodeAttributedString(from: attributedTextData) else {
                print("Browser: Failed to decode attributed text for context menu")
                return
            }
            delegate.showPluginRequestedContextMenu(attributedText: attributedText, at: location)

        case .showDefinition(let attributedTextData, let locationX, let locationY):
            guard let delegate = delegate else { return }
            let location = CGPoint(x: CGFloat(locationX), y: CGFloat(locationY))
            guard let attributedText = decodeAttributedString(from: attributedTextData) else {
                print("Browser: Failed to decode attributed text for definition")
                return
            }
            delegate.showPluginRequestedDefinition(attributedText: attributedText, at: location)

        case .hapticFeedback(let style):
            guard let delegate,
                  let feedbackStyle = OuterframeHapticFeedbackStyle(rawValue: Int(style)) else {
                return
            }
            delegate.performHapticFeedback(feedbackStyle)

        case .getImageWithSystemSymbolName(let requestId,
                                           let symbolName,
                                           let pointSize,
                                           let weight,
                                           let scale,
                                           let tintRed,
                                           let tintGreen,
                                           let tintBlue,
                                           let tintAlpha):
            handleGetImageWithSystemSymbolName(requestId: requestId,
                                  symbolName: symbolName,
                                  pointSize: pointSize,
                                  weight: weight,
                                  scale: scale,
                                  tintRed: tintRed,
                                  tintGreen: tintGreen,
                                  tintBlue: tintBlue,
                                  tintAlpha: tintAlpha)

        case .textCursorUpdate(let cursors):
            guard let delegate = delegate else { return }
            delegate.handleTextCursorUpdate(cursors: cursors)

        case .copySelectedPasteboardResponse(let requestId, let items):
            if let completion = pendingCopyRequests.removeValue(forKey: requestId) {
                completion(items)
            } else {
                print("Browser: Received copySelectedPasteboardResponse for unknown request id \(requestId)")
            }

        case .accessibilitySnapshotResponse(let requestId, let snapshotData):
            if let completion = pendingAccessibilitySnapshotRequests.removeValue(forKey: requestId) {
                let snapshot = snapshotData.flatMap { OuterframeAccessibilitySnapshot.deserialize(from: $0) }
                completion(snapshot)
            } else {
                print("Browser: Received accessibilitySnapshotResponse for unknown request id \(requestId)")
            }

        case .accessibilityTreeChanged(let notificationMask):
            delegate?.handleAccessibilityTreeChanged(notificationMask: notificationMask)

        case .openNewWindow(let urlString, let displayString, let preferredWidth, let preferredHeight):
            let size: CGSize?
            if let preferredWidth, let preferredHeight {
                size = CGSize(width: CGFloat(preferredWidth), height: CGFloat(preferredHeight))
            } else {
                size = nil
            }
            delegate?.handlePluginOpenNewWindow(urlString: urlString,
                                                displayString: displayString,
                                                preferredSize: size)

        case .setPasteboardCapabilities(let canCopy, let canCut, let pasteboardTypes):
            delegate?.setPasteboardCapabilities(canCopy: canCopy,
                                                canCut: canCut,
                                                pasteboardTypeIdentifiers: pasteboardTypes)
        }
    }

    @MainActor func outerframeContentSocketDidClose(_ socket: OuterframeContentSocket) {
        print("Browser: OuterframeContent socket closed")
        socketReady = false
        failPendingUnloadPluginRequests(with: OuterframeContentConnectionError.disconnected)
        failPendingLoadPluginRequests(with: OuterframeContentConnectionError.disconnected)
        failPendingCopyRequests()
        failPendingAccessibilityRequests()
    }
}

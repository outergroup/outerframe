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
    func unregisterDisplayLinkCallback(_ callbackID: UUID)
    func handlePluginLoaded(contextID: CAContextID)
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
    let pluginSocket = OuterframeContentSocket()
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
        let requestID = UUID()

        let proxyHost: String?
        if let port = networkProxyPort, port > 0 {
            proxyHost = "127.0.0.1"
        } else {
            proxyHost = nil
        }

        let infraMessage = BrowserToContentInfraMessage.loadPlugin(requestID: requestID,
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
            contentSize: CGSize(width: width, height: height),
            appearance: NSApp.effectiveAppearance,
            proxy: proxy,
            url: outerURLString,
            bundleUrl: bundleURLString,
            windowIsActive: true
        )
        let initializeContentMessage = BrowserToContentMessage.initializeContent(args: initializeContentArguments)

        try await withCheckedThrowingContinuation { continuation in
            loadPluginContinuations[requestID] = continuation

            Task { @MainActor in
                do {
                    // Send infrastructure message to tell OuterframeContent which bundle to load
                    try await self.infrastructureSocket.send(infraMessage.encode())

                    // Send initializeContent directly to the plugin over the plugin socket
                    try await self.pluginSocket.send(initializeContentMessage.encode())
                    self.schedulePluginLoadedTimeout()
                } catch {
                    print("Browser: Failed to send loadPlugin messages: \(error)")

                    if let continuation = loadPluginContinuations.removeValue(forKey: requestID) {
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

    func resizeContent(size: CGSize) {
        sendToOuterframeContent(.resizeContent(size: size))
    }

    func sendMagnification(surfaceID: Int,
                           magnification delta: CGFloat,
                           location: CGPoint,
                           scrollOffset: CGPoint) {
        precondition(surfaceID >= 0 && surfaceID <= Int(UInt32.max), "Surface identifier out of range")
        sendToOuterframeContent(.magnification(surfaceID: UInt32(surfaceID),
                                               magnification: delta,
                                               location: location,
                                               scrollOffset: scrollOffset))
    }

    func sendMagnificationEnded(surfaceID: Int,
                                magnification delta: CGFloat,
                                location: CGPoint,
                                scrollOffset: CGPoint) {
        precondition(surfaceID >= 0 && surfaceID <= Int(UInt32.max), "Surface identifier out of range")
        sendToOuterframeContent(.magnificationEnded(surfaceID: UInt32(surfaceID),
                                                    magnification: delta,
                                                    location: location,
                                                    scrollOffset: scrollOffset))
    }

    func sendDisplayLinkFrame(frameNumber: UInt64, targetTimestamp: CFTimeInterval) {
        sendToOuterframeContent(.displayLinkFired(frameNumber: frameNumber, targetTimestamp: targetTimestamp))
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

        sendToOuterframeContent(.systemAppearanceUpdate(appearance: appearance))
    }

    func sendWindowActiveState(isActive: Bool) {
        sendToOuterframeContent(.windowActiveUpdate(isActive: isActive))
    }

    func sendViewFocusChanged(isFocused: Bool) {
        sendToOuterframeContent(.viewFocusChanged(isFocused: isFocused))
    }

    func setDebuggerAttachmentMonitoring(isEnabled: Bool) {
        sendToInfrastructure(.setDebuggerAttachmentMonitoring(isEnabled: isEnabled))
    }

    // MARK: - Text Input Support

    func sendTextInput(text: String, replacementRange: NSRange) {
        let hasRange = replacementRange.location != NSNotFound
        let locationValue = hasRange ? UInt64(clamping: replacementRange.location) : 0
        let lengthValue = hasRange ? UInt64(clamping: replacementRange.length) : 0
        sendToOuterframeContent(.textInput(text: text,
                                           hasReplacementRange: hasRange,
                                           replacementLocation: locationValue,
                                           replacementLength: lengthValue))
    }

    func sendSetMarkedText(text: String, selectedRange: NSRange, replacementRange: NSRange) {
        let hasRange = replacementRange.location != NSNotFound
        let selectedLocation = UInt64(clamping: selectedRange.location)
        let selectedLength = UInt64(clamping: selectedRange.length)
        let replacementLocation = hasRange ? UInt64(clamping: replacementRange.location) : 0
        let replacementLength = hasRange ? UInt64(clamping: replacementRange.length) : 0
        sendToOuterframeContent(.setMarkedText(text: text,
                                               selectedLocation: selectedLocation,
                                               selectedLength: selectedLength,
                                               hasReplacementRange: hasRange,
                                               replacementLocation: replacementLocation,
                                               replacementLength: replacementLength))
    }

    func sendUnmarkText() {
        sendToOuterframeContent(.unmarkText)
    }

    func sendTextInputFocus(fieldID: UUID, hasFocus: Bool) {
        sendToOuterframeContent(.textInputFocus(fieldID: fieldID, hasFocus: hasFocus))
    }

    func sendTextCommand(command: String) {
        sendToOuterframeContent(.textCommand(command: command))
    }

    func requestPasteboardItemsForCopy(completion: @escaping ([OuterframeContentPasteboardItem]?) -> Void) {
        guard socketReady else {
            completion(nil)
            return
        }

        let requestID = UUID()
        pendingCopyRequests[requestID] = completion
        let message = BrowserToContentMessage.copySelectedPasteboardRequest(requestID: requestID)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send copySelectedPasteboardRequest message: \(error)")
                if let pending = self.pendingCopyRequests.removeValue(forKey: requestID) {
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

        let requestID = UUID()
        pendingAccessibilitySnapshotRequests[requestID] = completion
        let message = BrowserToContentMessage.accessibilitySnapshotRequest(requestID: requestID)
        let pluginSocket = self.pluginSocket
        Task { [weak self] in
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send accessibilitySnapshotRequest message: \(error)")
                await MainActor.run {
                    guard let self else { return }
                    if let pending = self.pendingAccessibilitySnapshotRequests.removeValue(forKey: requestID) {
                        pending(nil)
                    }
                }
            }
        }
    }

    func sendPasteboardItemsForPaste(_ items: [OuterframeContentPasteboardItem]) {
        sendToOuterframeContent(.pasteboardContentDelivered(items: items))
    }

    func sendSetCursorPosition(fieldID: UUID, position: UInt64, modifySelection: Bool) {
        sendToOuterframeContent(.setCursorPosition(fieldID: fieldID,
                                                   position: position,
                                                   modifySelection: modifySelection))
    }

    private func handleGetImageWithSystemSymbolName(requestID: UUID, symbolName: String, pointSize: CGFloat, weight: Double, scale: CGFloat) {
        guard let image = NSImage(systemSymbolName: symbolName, accessibilityDescription: nil) else {
            self.sendImageWithSystemSymbolNameResponse(requestID: requestID, alphaMaskData: nil, width: 0, height: 0, bytesPerRow: 0, success: false, errorMessage: "Symbol '\(symbolName)' not found")
            return
        }

        let symbolWeight = NSFont.Weight(CGFloat(weight))
        let config = NSImage.SymbolConfiguration(pointSize: pointSize, weight: symbolWeight)
        guard let configuredImage = image.withSymbolConfiguration(config) else {
            self.sendImageWithSystemSymbolNameResponse(requestID: requestID, alphaMaskData: nil, width: 0, height: 0, bytesPerRow: 0, success: false, errorMessage: "Failed to configure symbol")
            return
        }

        let pixelWidth = max(Int(ceil(configuredImage.size.width * scale)), 1)
        let pixelHeight = max(Int(ceil(configuredImage.size.height * scale)), 1)
        let rgbaBytesPerRow = pixelWidth * 4
        var rgbaData = Data(count: rgbaBytesPerRow * pixelHeight)

        let rendered = rgbaData.withUnsafeMutableBytes { bytes -> Bool in
            guard let baseAddress = bytes.baseAddress,
                  let context = CGContext(data: baseAddress,
                                          width: pixelWidth,
                                          height: pixelHeight,
                                          bitsPerComponent: 8,
                                          bytesPerRow: rgbaBytesPerRow,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                return false
            }

            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
            NSColor.clear.set()
            NSRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight).fill()
            configuredImage.draw(in: NSRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight), from: .zero, operation: .sourceOver, fraction: 1.0)
            NSGraphicsContext.restoreGraphicsState()
            return true
        }

        guard rendered else {
            self.sendImageWithSystemSymbolNameResponse(requestID: requestID, alphaMaskData: nil, width: 0, height: 0, bytesPerRow: 0, success: false, errorMessage: "Failed to create bitmap")
            return
        }

        var alphaMaskData = Data(count: pixelWidth * pixelHeight)
        rgbaData.withUnsafeBytes { rgbaBytes in
            alphaMaskData.withUnsafeMutableBytes { alphaBytes in
                guard let rgbaBaseAddress = rgbaBytes.baseAddress,
                      let alphaBaseAddress = alphaBytes.baseAddress else {
                    return
                }
                let rgba = rgbaBaseAddress.assumingMemoryBound(to: UInt8.self)
                let alpha = alphaBaseAddress.assumingMemoryBound(to: UInt8.self)
                for y in 0..<pixelHeight {
                    for x in 0..<pixelWidth {
                        alpha[y * pixelWidth + x] = rgba[y * rgbaBytesPerRow + x * 4 + 3]
                    }
                }
            }
        }

        self.sendImageWithSystemSymbolNameResponse(requestID: requestID, alphaMaskData: alphaMaskData, width: UInt32(pixelWidth), height: UInt32(pixelHeight), bytesPerRow: UInt32(pixelWidth), success: true, errorMessage: nil)
    }

    private func sendImageWithSystemSymbolNameResponse(requestID: UUID, alphaMaskData: Data?, width: UInt32, height: UInt32, bytesPerRow: UInt32, success: Bool, errorMessage: String?) {
        sendToOuterframeContent(.imageWithSystemSymbolName(requestID: requestID,
                                                           alphaMaskData: alphaMaskData,
                                                           width: width,
                                                           height: height,
                                                           bytesPerRow: bytesPerRow,
                                                           success: success,
                                                           errorMessage: errorMessage))
    }

    func sendToOuterframeContent(_ message: BrowserToContentMessage) {
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send message to OuterframeContent: \(error)")
            }
        }
    }

    private func sendToInfrastructure(_ message: BrowserToContentInfraMessage) {
        let infrastructureSocket = self.infrastructureSocket
        Task {
            do {
                try await infrastructureSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send infrastructure message to OuterframeContent: \(error)")
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
    @MainActor func outerframeContentSocket(_ socket: OuterframeContentSocket, didReceiveMessage message: Data) {
        if !socketReady {
            socketReady = true
        }

        if socket === infrastructureSocket {
            handleInfrastructureMessage(messageData: message)
        } else {
            handlePluginMessage(messageData: message)
        }
    }

    private func handleInfrastructureMessage(messageData: Data) {
        let message: ContentToBrowserInfraMessage
        do {
            message = try ContentToBrowserInfraMessage.decode(message: messageData)
        } catch {
            print("Browser: Failed to decode infrastructure message: \(error)")
            return
        }

        switch message {
        case .loadPluginSuccess(let requestID):
            guard let continuation = loadPluginContinuations.removeValue(forKey: requestID) else {
                print("Browser: Received loadPluginSuccess for unknown request ID \(requestID)")
                return
            }
            continuation.resume(returning: ())

        case .loadPluginFailure(let requestID, let errorMessage):
            cancelPluginLoadedTimeout()
            guard let continuation = loadPluginContinuations.removeValue(forKey: requestID) else {
                print("Browser: Received loadPluginFailure for unknown request ID \(requestID): \(errorMessage)")
                return
            }
            continuation.resume(throwing: OuterframeContentConnectionError.pluginLoadFailed(errorMessage))

        case .pluginLoaded(let contextID, let success):
            cancelPluginLoadedTimeout()
            print("Browser: Received pluginLoaded message - contextID: \(contextID), success: \(success)")
            guard success else { return }
            guard let delegate = delegate else { return }
            delegate.handlePluginLoaded(contextID: contextID)

        case .pluginUnloaded:
            cancelPluginLoadedTimeout()
            completeUnloadPluginRequest()

        case .debuggerAttached:
            delegate?.outerframeContentConnectionDebuggerAttached(self)
        }
    }

    private func handlePluginMessage(messageData: Data) {
        let message: ContentToBrowserMessage
        do {
            message = try ContentToBrowserMessage.decode(message: messageData)
        } catch {
            print("Browser: Failed to decode plugin message: \(error)")
            return
        }

        switch message {
        case .startDisplayLink(let callbackID):
            guard let delegate = delegate else { return }
            let browserCallbackID = delegate.registerDisplayLinkCallback()
            let ack = BrowserToContentMessage.displayLinkCallbackRegistered(callbackID: callbackID, browserCallbackID: browserCallbackID)
            Task {
                do {
                    try await self.pluginSocket.send(ack.encode())
                } catch {
                    print("Browser: Failed to send displayLinkCallbackRegistered message: \(error)")
                }
            }

        case .stopDisplayLink(let browserCallbackID):
            delegate?.unregisterDisplayLinkCallback(browserCallbackID)

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
            let location = CGPoint(x: locationX, y: locationY)
            guard let attributedText = decodeAttributedString(from: attributedTextData) else {
                print("Browser: Failed to decode attributed text for context menu")
                return
            }
            delegate.showPluginRequestedContextMenu(attributedText: attributedText, at: location)

        case .showDefinition(let attributedTextData, let locationX, let locationY):
            guard let delegate = delegate else { return }
            let location = CGPoint(x: locationX, y: locationY)
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

        case .getImageWithSystemSymbolName(let requestID,
                                           let symbolName,
                                           let pointSize,
                                           let weight,
                                           let scale):
            handleGetImageWithSystemSymbolName(requestID: requestID,
                                               symbolName: symbolName,
                                               pointSize: pointSize,
                                               weight: weight,
                                  scale: scale)

        case .textCursorUpdate(let cursors):
            guard let delegate = delegate else { return }
            delegate.handleTextCursorUpdate(cursors: cursors)

        case .copySelectedPasteboardResponse(let requestID, let items):
            if let completion = pendingCopyRequests.removeValue(forKey: requestID) {
                completion(items)
            } else {
                print("Browser: Received copySelectedPasteboardResponse for unknown request ID \(requestID)")
            }

        case .accessibilitySnapshotResponse(let requestID, let snapshotData):
            if let completion = pendingAccessibilitySnapshotRequests.removeValue(forKey: requestID) {
                let snapshot = snapshotData.flatMap { OuterframeAccessibilitySnapshot.deserialize(from: $0) }
                completion(snapshot)
            } else {
                print("Browser: Received accessibilitySnapshotResponse for unknown request ID \(requestID)")
            }

        case .accessibilityTreeChanged(let notificationMask):
            delegate?.handleAccessibilityTreeChanged(notificationMask: notificationMask)

        case .openNewWindow(let urlString, let displayString, let preferredSize):
            delegate?.handlePluginOpenNewWindow(urlString: urlString,
                                                displayString: displayString,
                                                preferredSize: preferredSize)

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

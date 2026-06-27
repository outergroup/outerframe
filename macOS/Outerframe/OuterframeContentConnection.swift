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

private final class OuterframeSynchronousReplyBox<T>: @unchecked Sendable {
    private let condition = NSCondition()
    private var value: T?

    func resume(_ value: T) {
        condition.lock()
        self.value = value
        condition.broadcast()
        condition.unlock()
    }

    func wait(until deadline: Date) -> T? {
        condition.lock()
        defer { condition.unlock() }

        while value == nil {
            if !condition.wait(until: deadline) {
                break
            }
        }
        return value
    }
}

private final class OuterframeSynchronousReplyRegistry<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var boxes: [UUID: OuterframeSynchronousReplyBox<T>] = [:]

    func register(_ requestID: UUID) -> OuterframeSynchronousReplyBox<T> {
        let box = OuterframeSynchronousReplyBox<T>()
        lock.lock()
        boxes[requestID] = box
        lock.unlock()
        return box
    }

    func remove(_ requestID: UUID) {
        lock.lock()
        boxes.removeValue(forKey: requestID)
        lock.unlock()
    }

    func resume(requestID: UUID, value: T) -> Bool {
        lock.lock()
        let box = boxes.removeValue(forKey: requestID)
        lock.unlock()

        guard let box else { return false }
        box.resume(value)
        return true
    }
}

private struct OuterframeAccessibilitySnapshotReply: Sendable {
    let snapshotData: Data?
}


protocol OuterframeContentConnectionDelegate: AnyObject {
    func registerDisplayLinkCallback() -> UUID
    func unregisterDisplayLinkCallback(_ callbackID: UUID)
    func handlePluginLoaded(contextID: CAContextID)
    func updateCursor(_ cursorType: PluginCursorType)
    func setInputMode(_ inputMode: OuterframeContentInputMode)
    func showPluginRequestedContextMenu(attributedText: NSAttributedString, at location: CGPoint)
    func showPluginRequestedContextMenuItems(menuID: UUID,
                                             attributedText: NSAttributedString?,
                                             items: [OuterframeContextMenuItem],
                                             at location: CGPoint)
    func showPluginRequestedDefinition(attributedText: NSAttributedString, at location: CGPoint)
    func handleTextInputGeometryUpdate(_ geometry: OuterframeContentTextInputGeometry?)
    func handleAccessibilityTreeChanged(notificationMask: UInt8)
    func handlePluginOpenNewWindow(urlString: String, displayString: String?, preferredSize: CGSize?)
    func handlePluginNavigate(urlString: String)
    func handlePluginOpenNewTab(urlString: String, displayString: String?)
    func handlePluginHistoryPushEntry(entryID: UUID, urlString: String?)
    func handlePluginHistoryReplaceEntry(entryID: UUID, urlString: String?)
    func handlePluginHistoryGo(delta: Int32)
    func handlePluginSetTitle(_ title: String?)
    func handlePluginSetIcon(_ icon: OuterframePresentationIcon)
    func setPasteboardDropBehaviorUniform(_ pasteboardTypeIdentifiers: [String])
    func setAcceptedPasteboardPasteTypes(_ pasteboardTypeIdentifiers: [String])
    func setPasteboardDropBehaviorHitTest()
    func handlePasteboardAccessRequest(requestID: UUID,
                                       operation: OuterframePasteboardAccessOperation,
                                       pasteboardTypeIdentifiers: [String],
                                       items: [OuterframeContentPasteboardItem])
    func beginDraggingPasteboardItems(_ items: [OuterframeContentDraggingItem], operationMask: UInt32)
    func releaseDroppedFileAccess(accessID: UUID)
    func outerframeContentConnectionDebuggerAttached(_ connection: OuterframeContentConnection)
    func outerframeContentConnectionDidTimeoutWaitingForPluginLoaded(_ connection: OuterframeContentConnection)
    func outerframeContentConnection(_ connection: OuterframeContentConnection, didReceiveStdout text: String)
    func outerframeContentConnection(_ connection: OuterframeContentConnection, didReceiveStderr text: String)
    func performHapticFeedback(_ style: OuterframeHapticFeedbackStyle)
}

struct OuterframeFilePromiseWriteResult: Sendable {
    let promiseID: UUID
    let localPath: String
    let deleteWhenDone: Bool
}

/// Manages the connection to OuterframeContent process
@MainActor
final class OuterframeContentConnection: NSObject {
    private nonisolated static let outerframeProcessesConnectionLock = NSLock()
    private nonisolated(unsafe) static var outerframeProcessesConnectionStorage: NSXPCConnection?

    private nonisolated(unsafe) static var outerframeProcessesConnection: NSXPCConnection {
        outerframeProcessesConnectionLock.lock()
        defer { outerframeProcessesConnectionLock.unlock() }

        if let outerframeProcessesConnectionStorage {
            return outerframeProcessesConnectionStorage
        }

        let connection = makeOuterframeProcessesConnection()
        outerframeProcessesConnectionStorage = connection
        return connection
    }

    private nonisolated static func makeOuterframeProcessesConnection() -> NSXPCConnection {
        let connection = NSXPCConnection(serviceName: OuterframeConfiguration.processesXPCServiceName)
        connection.remoteObjectInterface = NSXPCInterface(with: OuterframeProcessesProtocol.self)
        connection.interruptionHandler = {
            NSLog("OuterframeContentConnection: OuterframeProcesses XPC connection interrupted")
        }
        connection.invalidationHandler = {
            NSLog("OuterframeContentConnection: OuterframeProcesses XPC connection invalidated")
            outerframeProcessesConnectionLock.lock()
            if outerframeProcessesConnectionStorage === connection {
                outerframeProcessesConnectionStorage = nil
            }
            outerframeProcessesConnectionLock.unlock()
        }
        connection.resume()
        return connection
    }

    override init() {
        super.init()
        pluginSocket.immediateMessageHandler = { [weak self] messageData in
            self?.handleImmediatePluginMessage(messageData) ?? false
        }
    }

    /// Socket for infrastructure messages (loadPlugin, pluginLoaded, etc.)
    private let infrastructureSocket = OuterframeContentSocket(label: "org.outerframe.outerframecontent.socket.infrastructure")
    /// Socket for plugin messages (mouse, keyboard, display link, drag validation, etc.)
    let pluginSocket = OuterframeContentSocket(label: "org.outerframe.outerframecontent.socket.plugin",
                                               qos: .userInteractive)
    private var stdoutHandle: FileHandle?
    private var stderrHandle: FileHandle?
    private var outerframeContentPID: pid_t?
    private var outerframeContentInstanceId: UUID?
    private var outerframeContentTerminationSource: DispatchSourceProcess?
    private var stagedFileDirectoryURL: URL?
    var currentStagedFileDirectoryURL: URL? {
        stagedFileDirectoryURL
    }
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
    private var pendingFilePromiseWriteRequests: [UUID: @Sendable (Result<OuterframeFilePromiseWriteResult, Error>) -> Void] = [:]
    private let editCommandValidationReplies = OuterframeSynchronousReplyRegistry<OuterframeEditCommandSet>()
    private let pasteboardDropHitTestReplies = OuterframeSynchronousReplyRegistry<UInt32>()
    private let accessibilitySnapshotReplies = OuterframeSynchronousReplyRegistry<OuterframeAccessibilitySnapshotReply>()
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
        failPendingFilePromiseWriteRequests(with: error)
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

    private func failPendingFilePromiseWriteRequests(with error: Error) {
        guard !pendingFilePromiseWriteRequests.isEmpty else { return }
        let requests = pendingFilePromiseWriteRequests
        pendingFilePromiseWriteRequests.removeAll()
        for completion in requests.values {
            completion(.failure(error))
        }
    }

    nonisolated private func handleImmediatePluginMessage(_ messageData: Data) -> Bool {
        guard let message = try? ContentToBrowserMessage.decode(message: messageData) else {
            return false
        }
        switch message {
        case .editCommandValidationResponse(let requestID, let enabledCommands):
            return editCommandValidationReplies.resume(requestID: requestID, value: enabledCommands)
        case .pasteboardDropHitTestResponse(let requestID, let operationMask):
            return pasteboardDropHitTestReplies.resume(requestID: requestID, value: operationMask)
        case .accessibilitySnapshotResponse(let requestID, let snapshotData):
            let reply = OuterframeAccessibilitySnapshotReply(snapshotData: snapshotData)
            return accessibilitySnapshotReplies.resume(requestID: requestID, value: reply)
        default:
            return false
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

    func start(networkProxyEndpoint: OuterframeNetworkProxyEndpoint,
               storageContext: OuterframeStorageContext) async throws {
        let proxyPort = networkProxyEndpoint.port
        let stagedFileDirectoryURL = try Self.createStagedFileDirectory(storageContext: storageContext)
        networkProxyPort = proxyPort
        proxyUsername = networkProxyEndpoint.username
        proxyPassword = networkProxyEndpoint.password
        self.stagedFileDirectoryURL = stagedFileDirectoryURL

        do {
            let (infraHandle, pluginHandle, stdoutHandle, stderrHandle, pid, instanceId) = try await Self.launchOuterframeContentViaXPC(
                networkProxyPort: proxyPort,
                stagedFileDirectoryPath: stagedFileDirectoryURL.path
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
            self.stagedFileDirectoryURL = nil
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
            failPendingFilePromiseWriteRequests(with: error)
            failPendingUnloadPluginRequests(with: error)
            throw error
        }
    }

    private nonisolated static func launchOuterframeContentViaXPC(networkProxyPort: UInt16,
                                                                  stagedFileDirectoryPath: String) async throws -> (FileHandle, FileHandle, FileHandle, FileHandle, pid_t, UUID) {
        try await withCheckedThrowingContinuation { continuation in
            guard let proxy = outerframeProcessesConnection.remoteObjectProxy as? OuterframeProcessesProtocol else {
                continuation.resume(throwing: OuterframeContentConnectionError.posixFailure(code: -1, context: "XPC proxy creation"))
                return
            }

            proxy.launchOuterframeContent(networkProxyPort: networkProxyPort,
                                          stagedFileDirectoryPath: stagedFileDirectoryPath) { infraHandle, pluginHandle, stdoutHandle, stderrHandle, pid, instanceId, error in
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

    private nonisolated static func createStagedFileDirectory(storageContext: OuterframeStorageContext) throws -> URL {
        let directoryURL = storageContext.serverTemporaryDirectoryURL
            .appendingPathComponent("origins", isDirectory: true)
            .appendingPathComponent(directoryName(forOriginIdentifier: storageContext.originIdentifier), isDirectory: true)
            .appendingPathComponent("org.outerframe", isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        return canonicalFileURL(for: directoryURL)
    }

    private nonisolated static func directoryName(forOriginIdentifier originIdentifier: String) -> String {
        let trimmed = originIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        let labelSource = trimmed.isEmpty ? "origin" : trimmed.lowercased()
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        var label = ""
        var previousWasSeparator = false

        for scalar in labelSource.unicodeScalars {
            if allowed.contains(scalar) {
                label.unicodeScalars.append(scalar)
                previousWasSeparator = false
            } else if !previousWasSeparator {
                label.append("-")
                previousWasSeparator = true
            }
            if label.count >= 48 {
                break
            }
        }

        label = label.trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
        if label.isEmpty {
            label = "origin"
        }

        return "\(label)-h-\(fnv1a64Hex(trimmed))"
    }

    private nonisolated static func fnv1a64Hex(_ string: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        return String(format: "%016llx", hash)
    }

    private nonisolated static func canonicalFileURL(for url: URL) -> URL {
        let path = url.path
        return path.withCString { pointer -> URL in
            var resolved = [CChar](repeating: 0, count: Int(PATH_MAX))
            if realpath(pointer, &resolved) != nil {
                return resolved.withUnsafeBufferPointer { buffer in
                    URL(fileURLWithPath: String(cString: buffer.baseAddress!), isDirectory: true)
                }
            }
            return url
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
                    height: CGFloat,
                    historyEntryID: UUID? = nil) async throws {
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
            windowIsActive: true,
            historyEntryID: historyEntryID
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
        let message = BrowserToContentMessage.selectionToPasteboardCopyRequest(requestID: requestID)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send selectionToPasteboardCopyRequest message: \(error)")
                if let pending = self.pendingCopyRequests.removeValue(forKey: requestID) {
                    pending(nil)
                }
            }
        }
    }

    func requestPasteboardItemsForCut(completion: @escaping ([OuterframeContentPasteboardItem]?) -> Void) {
        guard socketReady else {
            completion(nil)
            return
        }

        let requestID = UUID()
        pendingCopyRequests[requestID] = completion
        let message = BrowserToContentMessage.selectionToPasteboardCutRequest(requestID: requestID)
        let pluginSocket = self.pluginSocket
        Task {
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send selectionToPasteboardCutRequest message: \(error)")
                if let pending = self.pendingCopyRequests.removeValue(forKey: requestID) {
                    pending(nil)
                }
            }
        }
    }

    func requestFilePromiseWrite(promiseID: UUID,
                                 completion: @escaping @Sendable (Result<OuterframeFilePromiseWriteResult, Error>) -> Void) {
        guard socketReady else {
            completion(.failure(OuterframeContentConnectionError.disconnected))
            return
        }

        let requestID = UUID()
        pendingFilePromiseWriteRequests[requestID] = completion
        let message = BrowserToContentMessage.filePromiseWriteRequest(requestID: requestID, promiseID: promiseID)
        let pluginSocket = self.pluginSocket
        Task { [weak self] in
            do {
                try await pluginSocket.send(message.encode())
            } catch {
                print("Browser: Failed to send filePromiseWriteRequest message: \(error)")
                await MainActor.run {
                    guard let self else { return }
                    if let pending = self.pendingFilePromiseWriteRequests.removeValue(forKey: requestID) {
                        pending(.failure(error))
                    }
                }
            }
        }
    }

    func sendHistoryEntryAccepted(entryID: UUID, urlString: String) {
        sendToOuterframeContent(.historyEntryAccepted(entryID: entryID, url: urlString))
    }

    func sendHistoryEntryRejected(entryID: UUID, errorMessage: String) {
        sendToOuterframeContent(.historyEntryRejected(entryID: entryID, errorMessage: errorMessage))
    }

    func sendHistoryTraversal(entryID: UUID, urlString: String) {
        sendToOuterframeContent(.historyTraversal(entryID: entryID, url: urlString))
    }

    func sendHistoryContextUpdate(currentEntryID: UUID, urlString: String, length: UInt32, canGoBack: Bool, canGoForward: Bool) {
        sendToOuterframeContent(.historyContextUpdate(currentEntryID: currentEntryID,
                                                      url: urlString,
                                                      length: length,
                                                      canGoBack: canGoBack,
                                                      canGoForward: canGoForward))
    }

    func sendPasteboardItemsForPaste(_ items: [OuterframeContentPasteboardItem]) {
        sendToOuterframeContent(.pasteboardContentPasted(items: items))
    }

    func sendPasteboardItemsForDrop(_ items: [OuterframeContentPasteboardItem], at point: CGPoint) {
        sendToOuterframeContent(.pasteboardContentDropped(point: point, items: items))
    }

    func validateEditCommandsSynchronously(_ commands: OuterframeEditCommandSet,
                                           timeoutMilliseconds: UInt32) -> OuterframeEditCommandSet? {
        guard socketReady else { return nil }

        let requestID = UUID()
        let replyBox = editCommandValidationReplies.register(requestID)
        let message = BrowserToContentMessage.editCommandValidationRequest(requestID: requestID,
                                                                           commands: commands)
        do {
            try pluginSocket.sendBlocking(message.encode())
        } catch {
            editCommandValidationReplies.remove(requestID)
            print("Browser: Failed to send editCommandValidationRequest message: \(error)")
            return nil
        }

        let timeout = max(TimeInterval(timeoutMilliseconds) / 1000, 0.001)
        guard let result = replyBox.wait(until: Date(timeIntervalSinceNow: timeout)) else {
            editCommandValidationReplies.remove(requestID)
            return nil
        }
        return result
    }

    func validatePasteboardDropSynchronously(point: CGPoint,
                                             pasteboardTypes: [String],
                                             operationMask: UInt32,
                                             modifierFlags: UInt64,
                                             timeoutMilliseconds: UInt32) -> UInt32? {
        guard socketReady else { return nil }

        let requestID = UUID()
        let replyBox = pasteboardDropHitTestReplies.register(requestID)
        let message = BrowserToContentMessage.pasteboardDropHitTestRequest(requestID: requestID,
                                                                           point: point,
                                                                           pasteboardTypes: pasteboardTypes,
                                                                           operationMask: operationMask,
                                                                           modifierFlags: modifierFlags)
        do {
            try pluginSocket.sendBlocking(message.encode())
        } catch {
            pasteboardDropHitTestReplies.remove(requestID)
            print("Browser: Failed to send pasteboardDropHitTestRequest message: \(error)")
            return nil
        }

        let timeout = max(TimeInterval(timeoutMilliseconds) / 1000, 0.001)
        guard let result = replyBox.wait(until: Date(timeIntervalSinceNow: timeout)) else {
            pasteboardDropHitTestReplies.remove(requestID)
            return nil
        }
        return result
    }

    func requestAccessibilitySnapshotSynchronously(timeoutMilliseconds: UInt32) -> OuterframeAccessibilitySnapshot? {
        guard socketReady else { return nil }

        let requestID = UUID()
        let replyBox = accessibilitySnapshotReplies.register(requestID)
        let message = BrowserToContentMessage.accessibilitySnapshotRequest(requestID: requestID)
        do {
            try pluginSocket.sendBlocking(message.encode())
        } catch {
            accessibilitySnapshotReplies.remove(requestID)
            print("Browser: Failed to send accessibilitySnapshotRequest message: \(error)")
            return nil
        }

        let timeout = max(TimeInterval(timeoutMilliseconds) / 1000, 0.001)
        guard let reply = replyBox.wait(until: Date(timeIntervalSinceNow: timeout)) else {
            accessibilitySnapshotReplies.remove(requestID)
            return nil
        }
        return reply.snapshotData.flatMap { OuterframeAccessibilitySnapshot.deserialize(from: $0) }
    }

    func sendPasteboardAccessResponse(requestID: UUID,
                                      granted: Bool,
                                      items: [OuterframeContentPasteboardItem]) {
        sendToOuterframeContent(.pasteboardAccessResponse(requestID: requestID,
                                                          granted: granted,
                                                          items: items))
    }

    func sendSetCursorPosition(fieldID: UUID, position: UInt64, modifySelection: Bool) {
        sendToOuterframeContent(.setCursorPosition(fieldID: fieldID,
                                                   position: position,
                                                   modifySelection: modifySelection))
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

        let pid = outerframeContentPID

        // Send shutdown messages so plugin and OuterframeContent can clean up
        if pid != nil {
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

        var xpcHandledTermination = false
        if let instanceId = outerframeContentInstanceId {
            xpcHandledTermination = await Self.ensureOuterframeContentExitsViaXPC(instanceId: instanceId, timeout: 2.0)
        }
        if !xpcHandledTermination, let pid {
            await Self.ensureOuterframeContentExitsLocally(pid: pid, timeout: 2.0)
        }

        outerframeContentPID = nil
        outerframeContentInstanceId = nil
        outerframeContentTerminationSource?.cancel()
        outerframeContentTerminationSource = nil

        failPendingLoadPluginRequests(with: OuterframeContentConnectionError.disconnected)
        failPendingCopyRequests()
        failPendingFilePromiseWriteRequests(with: OuterframeContentConnectionError.disconnected)

        lastSystemAppearancePayload = nil

        networkProxyPort = nil
        proxyUsername = nil
        proxyPassword = nil

        stagedFileDirectoryURL = nil

        await infrastructureSocket.stop()
        await pluginSocket.stop()
        stopOutputMonitoring()
    }

    private nonisolated static func ensureOuterframeContentExitsViaXPC(instanceId: UUID, timeout: TimeInterval) async -> Bool {
        await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            guard let proxy = outerframeProcessesConnection.remoteObjectProxyWithErrorHandler({ error in
                print("Browser: XPC error during ensureOuterframeContentExits: \(error)")
                continuation.resume(returning: false)
            }) as? OuterframeProcessesProtocol else {
                print("Browser: Failed to get XPC proxy for ensureOuterframeContentExits")
                continuation.resume(returning: false)
                return
            }

            proxy.ensureOuterframeContentExits(instanceId: instanceId, timeout: timeout) { didFindInstance, error in
                if let error = error {
                    print("Browser: ensureOuterframeContentExits error: \(error)")
                }
                continuation.resume(returning: didFindInstance && error == nil)
            }
        }
    }

    private nonisolated static func ensureOuterframeContentExitsLocally(pid: pid_t, timeout: TimeInterval) async {
        guard pid > 1 else { return }
        await Task.detached(priority: .utility) {
            let deadline = Date(timeIntervalSinceNow: timeout)
            while Date() < deadline {
                if !Self.processExists(pid: pid) {
                    return
                }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }

            guard Self.processExists(pid: pid) else { return }
            if kill(pid, SIGKILL) != 0 && errno != ESRCH {
                print("Browser: Failed to kill OuterframeContent process \(pid) locally (errno \(errno))")
            }
        }.value
    }

    private nonisolated static func processExists(pid: pid_t) -> Bool {
        if kill(pid, 0) == 0 {
            return true
        }
        return errno == EPERM
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

        case .showContextMenuItems(let menuID, let locationX, let locationY, let attributedTextData, let items):
            guard let delegate = delegate else { return }
            let attributedText: NSAttributedString?
            if let attributedTextData {
                attributedText = decodeAttributedString(from: attributedTextData)
                if attributedText == nil {
                    print("Browser: Failed to decode attributed text for context menu items")
                }
            } else {
                attributedText = nil
            }
            delegate.showPluginRequestedContextMenuItems(menuID: menuID,
                                                         attributedText: attributedText,
                                                         items: items,
                                                         at: CGPoint(x: locationX, y: locationY))

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

        case .textInputGeometryUpdate(let geometry):
            delegate?.handleTextInputGeometryUpdate(geometry)

        case .selectionToPasteboardResponse(let requestID, let items):
            if let completion = pendingCopyRequests.removeValue(forKey: requestID) {
                completion(items)
            } else {
                print("Browser: Received selectionToPasteboardResponse for unknown request ID \(requestID)")
            }

        case .pasteboardAccessRequest(let requestID, let operation, let pasteboardTypes, let items):
            delegate?.handlePasteboardAccessRequest(requestID: requestID,
                                                    operation: operation,
                                                    pasteboardTypeIdentifiers: pasteboardTypes,
                                                    items: items)

        case .beginDraggingPasteboardItems(let items, let operationMask):
            delegate?.beginDraggingPasteboardItems(items, operationMask: operationMask)

        case .releaseDroppedFileAccess(let accessID):
            delegate?.releaseDroppedFileAccess(accessID: accessID)

        case .filePromiseWriteResponse(let requestID, let promiseID, let success, let localPath, let deleteWhenDone, let errorMessage):
            guard let completion = pendingFilePromiseWriteRequests.removeValue(forKey: requestID) else {
                print("Browser: Received filePromiseWriteResponse for unknown request ID \(requestID)")
                return
            }
            if success, let localPath {
                completion(.success(OuterframeFilePromiseWriteResult(promiseID: promiseID,
                                                                     localPath: localPath,
                                                                     deleteWhenDone: deleteWhenDone)))
            } else {
                completion(.failure(NSError(domain: NSCocoaErrorDomain,
                                            code: NSFileWriteUnknownError,
                                            userInfo: [NSLocalizedDescriptionKey: errorMessage ?? "Content failed to write promised file."])))
            }

        case .accessibilitySnapshotResponse(let requestID, _):
            print("Browser: Received accessibilitySnapshotResponse outside synchronous wait for request ID \(requestID)")

        case .accessibilityTreeChanged(let notificationMask):
            delegate?.handleAccessibilityTreeChanged(notificationMask: notificationMask)

        case .openNewWindow(let urlString, let displayString, let preferredSize):
            delegate?.handlePluginOpenNewWindow(urlString: urlString,
                                                displayString: displayString,
                                                preferredSize: preferredSize)

        case .navigate(let urlString):
            delegate?.handlePluginNavigate(urlString: urlString)

        case .openNewTab(let urlString, let displayString):
            delegate?.handlePluginOpenNewTab(urlString: urlString,
                                             displayString: displayString)

        case .historyPushEntry(let entryID, let urlString):
            delegate?.handlePluginHistoryPushEntry(entryID: entryID, urlString: urlString)

        case .historyReplaceEntry(let entryID, let urlString):
            delegate?.handlePluginHistoryReplaceEntry(entryID: entryID, urlString: urlString)

        case .historyGo(let delta):
            delegate?.handlePluginHistoryGo(delta: delta)

        case .setTitle(let title):
            delegate?.handlePluginSetTitle(title)

        case .setIcon(let icon):
            delegate?.handlePluginSetIcon(icon)

        case .editCommandValidationResponse:
            break

        case .setPasteboardDropBehaviorUniform(let pasteboardTypes):
            delegate?.setPasteboardDropBehaviorUniform(pasteboardTypes)

        case .setAcceptedPasteboardPasteTypes(let pasteboardTypes):
            delegate?.setAcceptedPasteboardPasteTypes(pasteboardTypes)

        case .setPasteboardDropBehaviorHitTest:
            delegate?.setPasteboardDropBehaviorHitTest()

        case .pasteboardDropHitTestResponse:
            break
        }
    }

    @MainActor func outerframeContentSocketDidClose(_ socket: OuterframeContentSocket) {
        print("Browser: OuterframeContent socket closed")
        socketReady = false
        failPendingUnloadPluginRequests(with: OuterframeContentConnectionError.disconnected)
        failPendingLoadPluginRequests(with: OuterframeContentConnectionError.disconnected)
        failPendingCopyRequests()
        failPendingFilePromiseWriteRequests(with: OuterframeContentConnectionError.disconnected)
    }
}

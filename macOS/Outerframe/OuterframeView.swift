import AppKit
import Network

enum OuterframeHostIdentityError: LocalizedError {
    case missingBundleIdentifier

    var errorDescription: String? {
        switch self {
        case .missingBundleIdentifier:
            return "Host app bundle identifier is missing."
        }
    }
}

enum OuterframeHostIdentity {
    static func bundleIdentifier() throws -> String {
        guard let bundleIdentifier = Bundle.main.bundleIdentifier, !bundleIdentifier.isEmpty else {
            throw OuterframeHostIdentityError.missingBundleIdentifier
        }

        return bundleIdentifier
    }
}

public struct OuterframeCacheContext {
    public let serverID: UUID?
    public let host: String
    public let port: Int32

    public init(serverID: UUID?, host: String, port: Int32) {
        self.serverID = serverID
        self.host = host
        self.port = port
    }
}

public struct OuterframeNetworkProxyEndpoint: Sendable, Hashable {
    public let host: String
    public let port: UInt16
    public let username: String?
    public let password: String?

    public init(host: String, port: UInt16, username: String? = nil, password: String? = nil) {
        self.host = host
        self.port = port
        self.username = username
        self.password = password
    }

}

public enum OuterframeViewLoadResult {
    case loaded
    case notOuterframe
}

private enum OuterframeLoadError: Error {
    case invalidOuterframeData
    case invalidPathLength
    case invalidRemotePath
    case invalidDownloadURL
    case missingBundle
    case archiveByteStreamOpenFailed
    case archiveDecompressionStreamOpenFailed
    case archiveDecodeStreamOpenFailed
    case archiveExtractStreamOpenFailed
    case archiveExtractionFailed(underlying: Error)
    case unexpectedHTTPStatus(Int)
    case unrecognizedReturnCode
    case localBundleNotFound(String)
    case xpcConnectionFailed
}

extension OuterframeLoadError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .invalidOuterframeData:
            return "Outerframe payload is invalid."
        case .invalidPathLength:
            return "Outerframe payload path length is inconsistent."
        case .invalidRemotePath:
            return "Outerframe payload contains an invalid remote path."
        case .invalidDownloadURL:
            return "Failed to construct plugin download URL."
        case .missingBundle:
            return "No plugin bundle found after extracting archive."
        case .archiveByteStreamOpenFailed:
            return "Failed to open archive byte stream."
        case .archiveDecompressionStreamOpenFailed:
            return "Failed to create archive decompression stream."
        case .archiveDecodeStreamOpenFailed:
            return "Failed to create archive decode stream."
        case .archiveExtractStreamOpenFailed:
            return "Failed to create archive extraction stream."
        case .archiveExtractionFailed(let underlying):
            return "Failed to extract archive: \(underlying.localizedDescription)"
        case .unexpectedHTTPStatus(let status):
            return "Unexpected HTTP status code: \(status)."
        case .unrecognizedReturnCode:
            return "Unexpected loadPlugin return code"
        case .localBundleNotFound(let path):
            return "Local bundle not found at: \(path)"
        case .xpcConnectionFailed:
            return "Failed to connect to OuterframeProcesses XPC service."
        }
    }
}

private enum OuterframeBundleCache {
    struct Context {
        let serverID: UUID?
        let host: String
        let port: Int32
        let remotePath: String

        init(serverID: UUID?, host: String, port: Int32, remotePath: String) {
            self.serverID = serverID
            self.host = OuterframeBundleCache.normalizeHost(host)
            self.port = port > 0 ? port : 0
            self.remotePath = remotePath
        }
    }

    private nonisolated(unsafe) static var cachedBaseDirectory: URL?

    static func existingBundleURL(for context: Context) throws -> URL? {
        let entryDirectory = try entryDirectoryPath(for: context)
        if FileManager.default.fileExists(atPath: entryDirectory.path) {
            return firstBundle(in: entryDirectory)
        }
        return nil
    }

    static func prepareEntryDirectory(for context: Context) throws -> URL {
        try ensureBaseDirectoryExists()

        let fm = FileManager.default
        let endpointDirectory = try endpointDirectory(for: context, createIfNeeded: true)
        let entryDirectory = endpointDirectory.appendingPathComponent(resourceDirectoryName(for: context.remotePath),
                                                                      isDirectory: true)

        if fm.fileExists(atPath: entryDirectory.path) {
            try fm.removeItem(at: entryDirectory)
        }

        try fm.createDirectory(at: entryDirectory, withIntermediateDirectories: true, attributes: nil)
        return entryDirectory
    }

    static func entryDirectoryPath(for context: Context) throws -> URL {
        let base = try baseDirectoryPath()

        let serverDir = base.appendingPathComponent(serverDirectoryName(for: context.serverID), isDirectory: true)
        let endpointDir = serverDir.appendingPathComponent(endpointDirectoryName(host: context.host, port: context.port), isDirectory: true)
        return endpointDir.appendingPathComponent(resourceDirectoryName(for: context.remotePath), isDirectory: true)
    }

    static func removeBundles(forHost host: String, port: Int32, serverID: UUID? = nil) {
        do {
            let normalizedHost = normalizeHost(host)
            let fm = FileManager.default
            let endpointName = endpointDirectoryName(host: normalizedHost, port: port)

            let serverDirectories: [URL]
            if let serverID {
                let serverDir = try serverDirectory(for: serverID, createIfNeeded: false)
                serverDirectories = [serverDir]
            } else {
                let base = try baseDirectoryPath()
                guard fm.fileExists(atPath: base.path) else { return }

                let contents = try fm.contentsOfDirectory(at: base,
                                                          includingPropertiesForKeys: nil,
                                                          options: [.skipsHiddenFiles])
                serverDirectories = contents.filter { url in
                    var isDirectory: ObjCBool = false
                    return fm.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
                }
            }

            for serverDir in serverDirectories {
                let endpointDir = serverDir.appendingPathComponent(endpointName, isDirectory: true)
                if fm.fileExists(atPath: endpointDir.path) {
                    try fm.removeItem(at: endpointDir)
                }
            }
        } catch {
            print("OuterframeBundleCache: Failed to remove bundles for \(host):\(port) - \(error)")
        }
    }

    static func removeAllBundles(forServerID serverID: UUID) {
        do {
            let fm = FileManager.default
            let serverDir = try serverDirectory(for: serverID, createIfNeeded: false)
            if fm.fileExists(atPath: serverDir.path) {
                try fm.removeItem(at: serverDir)
            }
        } catch {
            print("OuterframeBundleCache: Failed to remove bundles for server \(serverID) - \(error)")
        }
    }

    private static func endpointDirectory(for context: Context, createIfNeeded: Bool) throws -> URL {
        let fm = FileManager.default
        let serverDirectory = try serverDirectory(for: context.serverID, createIfNeeded: createIfNeeded)
        let endpointDirectory = serverDirectory.appendingPathComponent(endpointDirectoryName(host: context.host,
                                                                                             port: context.port),
                                                                       isDirectory: true)

        if createIfNeeded && !fm.fileExists(atPath: endpointDirectory.path) {
            try fm.createDirectory(at: endpointDirectory, withIntermediateDirectories: true, attributes: nil)
        }

        return endpointDirectory
    }

    private static func serverDirectory(for serverID: UUID?, createIfNeeded: Bool) throws -> URL {
        let fm = FileManager.default
        let base = try baseDirectoryPath()

        if createIfNeeded {
            try ensureBaseDirectoryExists()
        } else if !fm.fileExists(atPath: base.path) {
            return base.appendingPathComponent(serverDirectoryName(for: serverID), isDirectory: true)
        }

        let directory = base.appendingPathComponent(serverDirectoryName(for: serverID), isDirectory: true)

        if createIfNeeded && !fm.fileExists(atPath: directory.path) {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
        }

        return directory
    }

    private static func baseDirectoryPath() throws -> URL {
        if let cachedBaseDirectory {
            return cachedBaseDirectory
        }

        let base = try fallbackBaseDirectoryPath()
        cachedBaseDirectory = base
        return base
    }

    private static func fallbackBaseDirectoryPath() throws -> URL {
        let bundleIdentifier = try OuterframeHostIdentity.bundleIdentifier()
        let libraryDirectory = FileManager.default.urls(for: .libraryDirectory,
                                                        in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library", isDirectory: true)

        return libraryDirectory
            .appendingPathComponent(bundleIdentifier, isDirectory: true)
            .appendingPathComponent("Cache", isDirectory: true)
            .appendingPathComponent("OuterframeBundleCache", isDirectory: true)
    }

    private static func ensureBaseDirectoryExists() throws {
        let fm = FileManager.default
        let base = try baseDirectoryPath()
        if !fm.fileExists(atPath: base.path) {
            try fm.createDirectory(at: base, withIntermediateDirectories: true, attributes: nil)
        }
    }

    private static func serverDirectoryName(for serverID: UUID?) -> String {
        guard let serverID else { return "global" }
        return "server-\(serverID.uuidString.lowercased())"
    }

    private static func endpointDirectoryName(host: String, port: Int32) -> String {
        let sanitizedHost = sanitizedFileName(from: host)
        return "\(sanitizedHost)_\(port)"
    }

    private static func resourceDirectoryName(for remotePath: String) -> String {
        let normalizedPath = remotePath.isEmpty ? "root" : remotePath
        let sanitized = sanitizedFileName(from: normalizedPath)
        let hash = fnv1a64Hash(normalizedPath)
        return "\(sanitized)_\(String(format: "%016llx", hash))"
    }

    private static func firstBundle(in directory: URL) -> URL? {
        let enumerator = FileManager.default.enumerator(at: directory,
                                                        includingPropertiesForKeys: nil,
                                                        options: [.skipsHiddenFiles])
        while let candidate = enumerator?.nextObject() as? URL {
            if candidate.pathExtension.lowercased() == "bundle" {
                return candidate
            }
        }
        return nil
    }

    private static func sanitizedFileName(from string: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        var result = ""
        for scalar in string.unicodeScalars {
            if allowed.contains(scalar) {
                result.append(String(scalar))
            } else {
                result.append("_")
            }
        }
        return result
    }

    private static func normalizeHost(_ host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "localhost"
        }
        return trimmed.lowercased()
    }

    private static func fnv1a64Hash(_ string: String) -> UInt64 {
        let offset: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x100000001b3

        var hash = offset
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        return hash
    }
}

public enum OuterframeCacheManager {
    public static func removeBundles(forHost host: String, port: Int32, serverID: UUID? = nil) {
        OuterframeBundleCache.removeBundles(forHost: host, port: port, serverID: serverID)
    }

    public static func removeAllBundles(forServerID serverID: UUID) {
        OuterframeBundleCache.removeAllBundles(forServerID: serverID)
    }
}

@MainActor
public final class OuterframeView: NSView, NSMenuItemValidation, NSServicesMenuRequestor, NSTextInputClient {
    public weak var delegate: OuterframeViewDelegate?

    private var layerHost: CALayerHost?
    private var trackingArea: NSTrackingArea?
    private var currentSelectedText: String?
    private var currentSelectedAttributedText: NSAttributedString?
    private var rightClickLocation: NSPoint = .zero
    private var backgroundEffectView: NSVisualEffectView?

    // Current loaded plugin
    private var pluginIsReady = false

    private var displayLink: CADisplayLink?
    private var outerframeContentConnection: OuterframeContentConnection?
    private var pendingOuterframeContentCleanupTask: Task<Void, Never>?
    private var pendingOuterframeContentCleanupToken: UUID?
    private var debugOverlayView: OuterframeDebugOverlayView?
    private var pendingDebugLoad: PendingDebugLoad?
    private var debuggerAttachedForCurrentConnection = false
    private var hasExternalConnectionReady = false
    private var localBundleOverridePath: String?
    private var debugModeObserver: NSObjectProtocol?
    private var autoResumeObserver: NSObjectProtocol?
    private var accessibilityDisplayOptionsObserver: NSObjectProtocol?

    // Track active display link callbacks
    private var activeDisplayLinkCallbacks = [UUID: Bool]() // UUID -> isActive
    private let displayLinkLock = NSLock()

    // Track window visibility state
    private var isWindowVisible = true
    private var shouldRunDisplayLink = false
    private var windowIsActive = false

    var currentInputMode: OuterframeContentInputMode = .rawKeys

    public static let outerframeMimeType = "application/vnd.outerframe"
    public static let topLevelNavigationCapabilityHeaderField = "Outerframe-Accept"
    public static let topLevelNavigationCapabilityHeaderValue = OuterframeView.outerframeMimeType

    private nonisolated(unsafe) static let outerframeProcessesConnection: NSXPCConnection = {
        let connection = NSXPCConnection(serviceName: OuterframeConfiguration.processesXPCServiceName)
        connection.remoteObjectInterface = NSXPCInterface(with: OuterframeProcessesProtocol.self)
        connection.resume()
        return connection
    }()

    // Text input state
    private var currentTextFieldId: String?
    private var currentText: String = ""
    private var textSelectedRange: NSRange = NSRange(location: 0, length: 0)
    private var markedText: NSAttributedString?
    private var textMarkedRange: NSRange = NSRange(location: NSNotFound, length: 0)

    // Text cursor indicators
    private var textCursorIndicators: [String: NSTextInsertionIndicator] = [:]
    private var overlayScrollIndicators: [String: NSScroller] = [:]

    private let manualMagnificationSurfaceId = Int(UInt32.max)

    // Accessibility caching
    private static let accessibilityActiveInterval: TimeInterval = 30
    private var accessibilitySnapshot: OuterframeAccessibilitySnapshot?
    private var accessibilityElements: [OuterframeAccessibilityElement] = []
    private var accessibilityRequestInFlight = false
    private var accessibilitySnapshotIsStale = false
    private var lastAccessibilityQueryUptime: TimeInterval?
    private var cachedVoiceOverEnabled = false
    private var pendingAccessibilityNotifications: OuterframeAccessibilityNotification = []
    private var needsInitialAccessibilityAnnouncement = true

    private struct EditingCapabilities {
        let canCopy: Bool
        let canCut: Bool
        let acceptablePasteboardTypes: Set<NSPasteboard.PasteboardType>

        func allowsPaste(from pasteboard: NSPasteboard) -> Bool {
            guard !acceptablePasteboardTypes.isEmpty else { return false }
            guard let availableTypes = pasteboard.types, !availableTypes.isEmpty else { return false }
            for type in availableTypes where acceptablePasteboardTypes.contains(type) {
                return true
            }
            return false
        }
    }

    private var editingCapabilitiesOverride: EditingCapabilities?

    private func setOuterframeContentConnection(_ connection: OuterframeContentConnection?) {
        if outerframeContentConnection === connection {
            return
        }

        outerframeContentConnection?.delegate = nil

        if let connection {
            connection.delegate = self
            outerframeContentConnection = connection
            delegate?.outerframeViewDidResetOuterframeContentOutput(self)
        } else {
            outerframeContentConnection = nil
            debuggerAttachedForCurrentConnection = false
            cancelPendingDebugLoadIfNeeded(error: OuterframeContentConnectionError.disconnected)
        }
    }

    public override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        commonInit()
    }

    required public init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        wantsLayer = true
        installMaterialBackgroundIfNeeded()
        updateTrackingAreas()
        cachedVoiceOverEnabled = NSWorkspace.shared.isVoiceOverEnabled

        if debugModeObserver == nil {
            debugModeObserver = NotificationCenter.default.addObserver(forName: OuterframeDebugSettings.debugModeDidChangeNotification,
                                                                        object: nil,
                                                                        queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.handleDebugModePreferenceChange()
                }
            }
        }

        if autoResumeObserver == nil {
            autoResumeObserver = NotificationCenter.default.addObserver(forName: OuterframeDebugSettings.autoResumeDidChangeNotification,
                                                                         object: nil,
                                                                         queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.updateDebugOverlayAutoResumeState()
                }
            }
        }

        if accessibilityDisplayOptionsObserver == nil {
            accessibilityDisplayOptionsObserver = NotificationCenter.default.addObserver(forName: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification,
                                                                                         object: nil,
                                                                                         queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.cachedVoiceOverEnabled = NSWorkspace.shared.isVoiceOverEnabled
                }
            }
        }
    }

    private func handleDebugModePreferenceChange() {
        guard !OuterframeDebugSettings.isDebugModeEnabled else { return }
        guard pendingDebugLoad != nil else { return }
        resumePendingDebugLoad()
    }

    private func updateDebugOverlayAutoResumeState() {
        let shouldAutoResume = OuterframeDebugSettings.shouldAutoResumeOnDebuggerAttach
        debugOverlayView?.setAutoResumeSelected(shouldAutoResume)

        guard shouldAutoResume,
              let pending = pendingDebugLoad,
              pending.debuggerAttached else { return }
        resumePendingDebugLoad()
    }

    public override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        outerframeContentConnection?.sendSystemAppearanceUpdate(force: false)
    }

    public override var isOpaque: Bool {
        return false
    }

    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        installMaterialBackgroundIfNeeded()

        if window == nil {
            if let observer = debugModeObserver {
                NotificationCenter.default.removeObserver(observer)
            }
            if let observer = autoResumeObserver {
                NotificationCenter.default.removeObserver(observer)
            }
            if let observer = accessibilityDisplayOptionsObserver {
                NotificationCenter.default.removeObserver(observer)
            }
            cancelPendingDebugLoadIfNeeded(error: OuterframeContentConnectionError.disconnected)
        }
    }

    private func installMaterialBackgroundIfNeeded() {
        guard backgroundEffectView == nil else { return }
        let effectView = OuterframeBackgroundVisualEffectView()
        effectView.material = .headerView
        effectView.blendingMode = .withinWindow
        effectView.state = .followsWindowActiveState
        effectView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(effectView, positioned: .below, relativeTo: nil)
        NSLayoutConstraint.activate([
            effectView.leadingAnchor.constraint(equalTo: leadingAnchor),
            effectView.trailingAnchor.constraint(equalTo: trailingAnchor),
            effectView.topAnchor.constraint(equalTo: topAnchor),
            effectView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
        backgroundEffectView = effectView
    }

    public override func updateTrackingAreas() {
        // Remove existing tracking area
        if let existingTrackingArea = trackingArea {
            removeTrackingArea(existingTrackingArea)
        }

        // Create new tracking area
        let options: NSTrackingArea.Options = [
            .mouseMoved,
            .mouseEnteredAndExited,
            .activeInKeyWindow,
            .inVisibleRect
        ]

        trackingArea = NSTrackingArea(
            rect: bounds,
            options: options,
            owner: self,
            userInfo: nil
        )

        if let trackingArea = trackingArea {
            addTrackingArea(trackingArea)
        }
    }

    public override func resizeSubviews(withOldSize oldSize: NSSize) {
        super.resizeSubviews(withOldSize: oldSize)
    }

    public override func layout() {
        super.layout()
        layerHost?.frame = bounds
    }

    public override var acceptsFirstResponder: Bool {
        return true
    }

    public override var inputContext: NSTextInputContext? {
        return currentInputMode.allowsTextInput ? super.inputContext : nil
    }

    public override func resignFirstResponder() -> Bool {
        // Notify the content that this view lost first responder
        outerframeContentConnection?.sendViewFocusChanged(isFocused: false)
        return super.resignFirstResponder()
    }

    public override func becomeFirstResponder() -> Bool {
        let result = super.becomeFirstResponder()
        // Notify the content that this view gained first responder
        outerframeContentConnection?.sendViewFocusChanged(isFocused: true)
        return result
    }

    // MARK: - Text Cursor Management

    func updateTextCursors(_ cursors: [[String: Any]]) {
        // Remove existing cursor indicators that are not in the new list
        var newFieldIds = Set<String>()
        var firstVisibleFieldId: String?
        for cursor in cursors {
            if let fieldId = cursor["fieldId"] as? String {
                newFieldIds.insert(fieldId)
            }
        }

        // Remove old indicators
        for (fieldId, indicator) in textCursorIndicators {
            if !newFieldIds.contains(fieldId) {
                indicator.removeFromSuperview()
                textCursorIndicators.removeValue(forKey: fieldId)
            }
        }

        // Update or create indicators
        for cursor in cursors {
            guard let fieldId = cursor["fieldId"] as? String,
                  let x = cursor["x"] as? CGFloat,
                  let y = cursor["y"] as? CGFloat,
                  let width = cursor["width"] as? CGFloat,
                  let height = cursor["height"] as? CGFloat,
                  let visible = cursor["visible"] as? Bool else { continue }

            let indicator: NSTextInsertionIndicator
            if let existingIndicator = textCursorIndicators[fieldId] {
                indicator = existingIndicator
            } else {
                indicator = NSTextInsertionIndicator(frame: NSRect(x: x, y: y, width: width, height: height))
                indicator.displayMode = .automatic
                addSubview(indicator)
                textCursorIndicators[fieldId] = indicator
            }

            // Update position and visibility
            let flippedY = bounds.height - y - height  // Convert from top-left to bottom-left coordinate system
            indicator.frame = NSRect(x: x, y: flippedY, width: width, height: height)
            indicator.isHidden = !visible
            if visible && firstVisibleFieldId == nil {
                firstVisibleFieldId = fieldId
            }
        }

        currentTextFieldId = firstVisibleFieldId
        inputContext?.invalidateCharacterCoordinates()
    }

    // MARK: - NSTextInputClient

    nonisolated public func insertText(_ string: Any, replacementRange: NSRange) {
        let text = (string as? String) ?? (string as? NSAttributedString)?.string ?? ""
        MainActor.assumeIsolated {

            // Clear any marked text first
            if hasMarkedText() {
                unmarkText()
            }

            // Send text input to plugin via IPC
            sendTextInput(text, replacementRange: replacementRange)
        }
    }

    nonisolated public func setMarkedText(_ string: Any, selectedRange: NSRange, replacementRange: NSRange) {
        nonisolated(unsafe) let attributedString = string as? NSAttributedString ?? NSAttributedString(string: string as? String ?? "")

        MainActor.assumeIsolated {
            markedText = attributedString
            textMarkedRange = NSRange(location: replacementRange.location != NSNotFound ? replacementRange.location : selectedRange.location,
                                      length: attributedString.length)

            // Send marked text to plugin
            sendSetMarkedText(attributedString.string, selectedRange: selectedRange, replacementRange: replacementRange)
        }
    }

    nonisolated public func unmarkText() {
        MainActor.assumeIsolated {
            markedText = nil
            textMarkedRange = NSRange(location: NSNotFound, length: 0)

            // Notify plugin
            sendUnmarkText()
        }
    }

    nonisolated public func selectedRange() -> NSRange {
        MainActor.assumeIsolated {
            return textSelectedRange
        }
    }

    nonisolated public func markedRange() -> NSRange {
        MainActor.assumeIsolated {
            return textMarkedRange
        }
    }

    nonisolated public func hasMarkedText() -> Bool {
        MainActor.assumeIsolated {
            return markedText != nil && textMarkedRange.location != NSNotFound
        }
    }

    nonisolated public func attributedSubstring(forProposedRange range: NSRange, actualRange: NSRangePointer?) -> NSAttributedString? {
        // Return the substring from our current text
        guard range.location != NSNotFound else { return nil }

        let fullText = MainActor.assumeIsolated {
            currentText
        } as NSString

        guard range.location + range.length <= fullText.length else { return nil }

        actualRange?.pointee = range
        let substring = fullText.substring(with: range)
        return NSAttributedString(string: substring)
    }

    nonisolated public func validAttributesForMarkedText() -> [NSAttributedString.Key] {
        MainActor.assumeIsolated {
            // Return supported attributes for marked text
            return [.font, .foregroundColor, .backgroundColor, .underlineStyle]
        }
    }

    nonisolated public func firstRect(forCharacterRange range: NSRange, actualRange: NSRangePointer?) -> NSRect {
        nonisolated(unsafe) let actualRange2 = actualRange

        return MainActor.assumeIsolated {
            actualRange2?.pointee = range

            // Use the cursor indicator position if available
            var viewRect: NSRect
            if let fieldId = currentTextFieldId,
               let indicator = textCursorIndicators[fieldId],
               !indicator.isHidden {
                // Position IME just above the cursor indicator
                viewRect = indicator.frame
            } else if let firstVisibleIndicator = textCursorIndicators.values.first(where: { !$0.isHidden }) {
                viewRect = firstVisibleIndicator.frame
            } else {
                // Fallback to a default position
                viewRect = NSRect(x: 20, y: self.bounds.height - 30, width: 2, height: 20)
            }

            // Convert view coordinates to screen coordinates
            let windowRect = self.convert(viewRect, to: nil)
            let screenRect = self.window?.convertToScreen(windowRect) ?? windowRect

            return screenRect
        }
    }

    nonisolated public func characterIndex(for point: NSPoint) -> Int {
        MainActor.assumeIsolated {
            // Return the character index at the given point
            // For now, return the end of the text
            return currentText.count
        }
    }

    public func setPasteboardCapabilities(canCopy: Bool,
                                          canCut: Bool,
                                          pasteboardTypeIdentifiers: [String]) {
        let convertedTypes = pasteboardTypeIdentifiers
            .filter { !$0.isEmpty }
            .map { NSPasteboard.PasteboardType($0) }
        editingCapabilitiesOverride = EditingCapabilities(canCopy: canCopy,
                                                          canCut: canCut,
                                                          acceptablePasteboardTypes: Set(convertedTypes))
    }

    func resetEditingCapabilities() {
        editingCapabilitiesOverride = nil
    }

    private func writePasteboardItems(_ items: [OuterframeContentPasteboardItem]) -> Bool {
        guard !items.isEmpty else { return false }

        var preparedItems: [(NSPasteboard.PasteboardType, Data)] = []
        preparedItems.reserveCapacity(items.count)
        for item in items where !item.typeIdentifier.isEmpty {
            let type = NSPasteboard.PasteboardType(item.typeIdentifier)
            preparedItems.append((type, item.data))
        }

        guard !preparedItems.isEmpty else { return false }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()

        var wroteAny = false
        for (type, data) in preparedItems {
            if pasteboard.setData(data, forType: type) {
                wroteAny = true
            }
        }

        return wroteAny
    }

    public func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
        if let capabilities = editingCapabilitiesOverride {
            switch menuItem.action {
            case #selector(copy(_:)):
                return capabilities.canCopy
            case #selector(cut(_:)):
                return capabilities.canCut
            case #selector(paste(_:)):
                return capabilities.allowsPaste(from: NSPasteboard.general)
            default:
                break
            }
        }

        switch menuItem.action {
        case #selector(copy(_:)):
            if currentInputMode.allowsTextInput {
                return true
            }
            return currentSelectedText?.isEmpty == false
        case #selector(cut(_:)):
            if currentInputMode.allowsTextInput {
                return true
            }
            return false
        case #selector(selectAll(_:)):
            return currentInputMode.allowsTextInput
        case #selector(paste(_:)):
            if currentInputMode.allowsTextInput {
                return NSPasteboard.general.string(forType: .string) != nil
            }
            return false
        default:
            return true
        }
    }

    @objc func copy(_ sender: Any?) {
        if let capabilities = editingCapabilitiesOverride, !capabilities.canCopy {
            NSSound.beep()
            return
        }

        if currentInputMode.allowsTextInput {
            requestPasteboardItemsForCopy { [weak self] items in
                Task { @MainActor in
                    guard let self else { return }
                    if let items, self.writePasteboardItems(items) {
                        return
                    }
                    if let fallback = self.currentSelectedText, !fallback.isEmpty {
                        let pasteboard = NSPasteboard.general
                        pasteboard.clearContents()
                        pasteboard.setString(fallback, forType: .string)
                    }
                }
            }
            return
        }

        if let selectedText = currentSelectedText, !selectedText.isEmpty {
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            pasteboard.setString(selectedText, forType: .string)
        }
    }

    @objc func cut(_ sender: Any?) {
        if let capabilities = editingCapabilitiesOverride, !capabilities.canCut {
            NSSound.beep()
            return
        }

        guard currentInputMode.allowsTextInput else {
            NSSound.beep()
            return
        }

        requestPasteboardItemsForCopy { [weak self] items in
            Task { @MainActor in
                guard let self else { return }
                guard let items, self.writePasteboardItems(items) else {
                    NSSound.beep()
                    return
                }
                self.sendTextCommand("deleteBackward")
            }
        }
    }

    public override func selectAll(_ sender: Any?) {
        guard currentInputMode.allowsTextInput else {
            super.selectAll(sender)
            return
        }
        sendTextCommand("selectAll")
    }

    @objc func paste(_ sender: Any?) {
        let pasteboard = NSPasteboard.general

        if let capabilities = editingCapabilitiesOverride,
           !capabilities.allowsPaste(from: pasteboard) {
            NSSound.beep()
            return
        }

        if currentInputMode.allowsTextInput {
            if let capabilities = editingCapabilitiesOverride {
                guard let firstItem = pasteboard.pasteboardItems?.first else {
                    NSSound.beep()
                    return
                }

                var payload: [OuterframeContentPasteboardItem] = []
                payload.reserveCapacity(firstItem.types.count)
                let acceptedTypes = capabilities.acceptablePasteboardTypes
                for type in firstItem.types {
                    if acceptedTypes.contains(type),
                       let data = firstItem.data(forType: type) {
                        payload.append(OuterframeContentPasteboardItem(typeIdentifier: type.rawValue, data: data))
                    }
                }

                guard !payload.isEmpty else {
                    NSSound.beep()
                    return
                }

                sendPasteboardItemsForPaste(payload)
            } else {
                guard let text = pasteboard.string(forType: .string) else { return }
                sendTextInput(text, replacementRange: NSRange(location: NSNotFound, length: 0))
            }
            return
        }
        NSApp.sendAction(#selector(NSText.paste(_:)), to: nil, from: sender)
    }

    @objc func lookUp(_ sender: Any?) {
        // This method should only be called from plugin-requested menus
        // where we've already stored the necessary information
        guard let attributedText = currentSelectedAttributedText else { return }

        // Use the stored right-click location for the popover
        self.showDefinition(for: attributedText, at: rightClickLocation)
    }

    public override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        handleMouseDown(at: point, modifierFlags: event.modifierFlags, clickCount: event.clickCount)
        super.mouseDown(with: event)
    }

    public override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        handleMouseDragged(to: point, modifierFlags: event.modifierFlags)
    }

    public override func mouseUp(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        handleMouseUp(at: point, modifierFlags: event.modifierFlags)
    }

    public override func mouseMoved(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        handleMouseMoved(to: point, modifierFlags: event.modifierFlags)
    }

    public override func mouseEntered(with event: NSEvent) {
        // Reset cursor when entering the view
        NSCursor.arrow.set()
    }

    public override func mouseExited(with event: NSEvent) {
        // Reset cursor when leaving the view
        NSCursor.arrow.set()
    }

    public override func scrollWheel(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        handleScrollWheel(at: point, with: event)
    }

    public override func magnify(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        handleMagnify(at: point, with: event)
    }

    // MARK: - Keyboard Handling

    public override func keyDown(with event: NSEvent) {
        if currentInputMode.allowsRawKeys {
            handleKeyDown(event)
        }

        if currentInputMode.allowsTextInput {
            // This will call our insertText, doCommandBySelector, etc. methods
            interpretKeyEvents([event])
        }

        if currentInputMode.isEmpty {
            super.keyDown(with: event)
        }
    }

    public override func keyUp(with event: NSEvent) {
        if currentInputMode.allowsRawKeys {
            handleKeyUp(event)
        } else {
            super.keyUp(with: event)
        }
    }

    // Handle text navigation and editing commands
    public override nonisolated func doCommand(by selector: Selector) {
        MainActor.assumeIsolated {
            // Map selectors to command types that we'll send to the plugin
            let commandMap: [Selector: String] = [
                // Cursor movement
                #selector(moveLeft(_:)): "moveLeft",
                #selector(moveRight(_:)): "moveRight",
                #selector(moveUp(_:)): "moveUp",
                #selector(moveDown(_:)): "moveDown",
                #selector(moveForward(_:)): "moveRight",  // Ctrl+F
                #selector(moveBackward(_:)): "moveLeft",   // Ctrl+B
                #selector(moveWordLeft(_:)): "moveWordLeft",
                #selector(moveWordRight(_:)): "moveWordRight",
                #selector(moveWordForward(_:)): "moveWordRight",  // Alt+F
                #selector(moveWordBackward(_:)): "moveWordLeft",   // Alt+B
                #selector(moveToBeginningOfLine(_:)): "moveToBeginningOfLine",
                #selector(moveToEndOfLine(_:)): "moveToEndOfLine",
                #selector(moveToLeftEndOfLine(_:)): "moveToLeftEndOfLine", // Cmd-left
                #selector(moveToRightEndOfLine(_:)): "moveToRightEndOfLine", // Cmd-right
                #selector(moveToBeginningOfDocument(_:)): "moveToBeginningOfDocument",
                #selector(moveToEndOfDocument(_:)): "moveToEndOfDocument",
                #selector(moveToBeginningOfParagraph(_:)): "moveToBeginningOfParagraph",
                #selector(moveToEndOfParagraph(_:)): "moveToEndOfParagraph",

                // Selection movement
                #selector(moveLeftAndModifySelection(_:)): "moveLeftAndModifySelection",
                #selector(moveRightAndModifySelection(_:)): "moveRightAndModifySelection",
                #selector(moveForwardAndModifySelection(_:)): "moveRightAndModifySelection",  // Shift+Ctrl+F
                #selector(moveBackwardAndModifySelection(_:)): "moveLeftAndModifySelection",   // Shift+Ctrl+B
                #selector(moveWordLeftAndModifySelection(_:)): "moveWordLeftAndModifySelection",
                #selector(moveWordRightAndModifySelection(_:)): "moveWordRightAndModifySelection",
                #selector(moveWordForwardAndModifySelection(_:)): "moveWordRightAndModifySelection",
                #selector(moveWordBackwardAndModifySelection(_:)): "moveWordLeftAndModifySelection",
                #selector(moveToBeginningOfLineAndModifySelection(_:)): "moveToBeginningOfLineAndModifySelection",
                #selector(moveToEndOfLineAndModifySelection(_:)): "moveToEndOfLineAndModifySelection",
                #selector(moveToLeftEndOfLineAndModifySelection(_:)): "moveToLeftEndOfLineAndModifySelection", // Shift+Cmd-left
                #selector(moveToRightEndOfLineAndModifySelection(_:)): "moveToRightEndOfLineAndModifySelection", // Shift+Cmd-right
                #selector(moveToBeginningOfDocumentAndModifySelection(_:)): "moveToBeginningOfDocumentAndModifySelection",
                #selector(moveToEndOfDocumentAndModifySelection(_:)): "moveToEndOfDocumentAndModifySelection",
                #selector(moveUpAndModifySelection(_:)): "moveUpAndModifySelection",
                #selector(moveDownAndModifySelection(_:)): "moveDownAndModifySelection",
                #selector(moveToBeginningOfParagraphAndModifySelection(_:)): "moveToBeginningOfParagraphAndModifySelection",
                #selector(moveToEndOfParagraphAndModifySelection(_:)): "moveToEndOfParagraphAndModifySelection",
                #selector(selectAll(_:)): "selectAll",
                #selector(selectWord(_:)): "selectWord",
                #selector(selectLine(_:)): "selectLine",
                #selector(selectParagraph(_:)): "selectParagraph",

                // Deletion
                #selector(deleteBackward(_:)): "deleteBackward",
                #selector(deleteForward(_:)): "deleteForward",
                #selector(deleteWordBackward(_:)): "deleteWordBackward",
                #selector(deleteWordForward(_:)): "deleteWordForward",
                #selector(deleteBackwardByDecomposingPreviousCharacter(_:)): "deleteBackward",  // Another delete variant
                #selector(deleteToBeginningOfLine(_:)): "deleteToBeginningOfLine",
                #selector(deleteToEndOfLine(_:)): "deleteToEndOfLine",
                #selector(deleteToBeginningOfParagraph(_:)): "deleteToBeginningOfParagraph",
                #selector(deleteToEndOfParagraph(_:)): "deleteToEndOfParagraph",
                #selector(deleteToMark(_:)): "deleteToMark",  // Ctrl+W (Emacs delete to mark)
                #selector(yank(_:)): "yank",  // Ctrl+Y (Emacs paste)

                // Other actions
                #selector(insertNewline(_:)): "insertNewline",
                #selector(insertTab(_:)): "insertTab",
                #selector(insertBacktab(_:)): "insertBacktab",
                #selector(cancelOperation(_:)): "cancelOperation"
            ]

            if let command = commandMap[selector] {
                // Debug: Log the command being sent
                print("Sending command '\(command)' for selector: \(selector)")
                // Send command to plugin
                sendTextCommand(command)
            } else {
                // For unhandled selectors, call super or log
                print("Unhandled selector: \(selector)")
                super.doCommand(by: selector)
            }
        }
    }

    public override func rightMouseDown(with event: NSEvent) {
        // Pass right-click to plugin as a separate event type
        let point = convert(event.locationInWindow, from: nil)
        handleRightMouseDown(at: point, modifierFlags: event.modifierFlags, clickCount: event.clickCount)
    }

    public override func rightMouseUp(with event: NSEvent) {
        // Pass right mouse up to plugin
        let point = convert(event.locationInWindow, from: nil)
        handleRightMouseUp(at: point, modifierFlags: event.modifierFlags)
    }

    public override func quickLookPreviewItems(_ sender: Any?) {   // keyboard fallback?
        print("quickLookPreviewItems? This doesn't currently work.")
    }

    public override func quickLook(with event: NSEvent) {
        print("quickLook? This doesn't currently work.")
        // Handle Force Touch click
        let point = convert(event.locationInWindow, from: nil)
        handleQuickLook(at: point)

        // Don't call super to prevent default Quick Look behavior
    }

    public override func menu(for event: NSEvent) -> NSMenu? {
        // Don't show any menu - let the plugin decide
        return nil
    }

    public func showPluginRequestedContextMenu(attributedText: NSAttributedString, at location: CGPoint) {
        let selectedText = attributedText.string.isEmpty ? nil : attributedText.string
        currentSelectedAttributedText = selectedText == nil ? nil : attributedText
        currentSelectedText = selectedText
        rightClickLocation = location

        let menu = NSMenu()

        if let selectedText {
            let lookupItem = NSMenuItem(title: "Look Up \"\(selectedText)\"", action: #selector(lookUp(_:)), keyEquivalent: "")
            lookupItem.target = self
            menu.addItem(lookupItem)

            menu.addItem(NSMenuItem.separator())
        }

        let cutItem = NSMenuItem(title: "Cut", action: #selector(cut(_:)), keyEquivalent: "x")
        cutItem.target = self
        menu.addItem(cutItem)

        let copyItem = NSMenuItem(title: "Copy", action: #selector(copy(_:)), keyEquivalent: "c")
        copyItem.target = self
        menu.addItem(copyItem)

        let pasteItem = NSMenuItem(title: "Paste", action: #selector(paste(_:)), keyEquivalent: "v")
        pasteItem.target = self
        menu.addItem(pasteItem)

        let selectAllItem = NSMenuItem(title: "Select All", action: #selector(selectAll(_:)), keyEquivalent: "a")
        selectAllItem.target = self
        menu.addItem(selectAllItem)

        if selectedText != nil {
            menu.addItem(NSMenuItem.separator())

            let servicesItem = NSMenuItem(title: "Services", action: nil, keyEquivalent: "")
            let servicesMenu = NSMenu()
            servicesItem.submenu = servicesMenu
            menu.addItem(servicesItem)
            NSApp.servicesMenu = servicesMenu
        }

        menu.allowsContextMenuPlugIns = true

        menu.popUp(positioning: nil, at: location, in: self)
    }

    public func showPluginRequestedDefinition(attributedText: NSAttributedString, at location: CGPoint) {
        // Use the location directly - plugin should send correct coordinates
        self.showDefinition(for: attributedText, at: location)
    }

    func updateWithContextId(_ contextId: CAContextID) {
        print("CanvasView: updateWithContextId called with contextId: \(contextId)")

        // Ensure we have a layer
        resetAccessibilitySnapshot()

        if layer == nil {
            wantsLayer = true
        }

        layerHost?.removeFromSuperlayer()


        let layerHost2 = CALayerHost()
        layerHost = layerHost2

        layerHost2.contextId = contextId
        layerHost2.frame = bounds
        layerHost2.zPosition = 1
//        layerHost2.autoresizingMask = [CAAutoresizingMask.layerWidthSizable, CAAutoresizingMask.layerHeightSizable]

        layer?.addSublayer(layerHost2)
        layerHost = layerHost2
    }

    func clearLayer() {
        layerHost?.removeFromSuperlayer()
        layerHost = nil
        resetAccessibilitySnapshot()
    }

    public override func setFrameSize(_ newSize: NSSize) {
        let oldSize = frame.size
        super.setFrameSize(newSize)

        if oldSize != newSize {
            layerHost?.frame = bounds
            notifySizeChanged()
        }
    }

    private func notifySizeChanged() {
        let size = bounds.size

        withActivePluginConnection { connection in
            connection.resizeContent(width: size.width, height: size.height)
        }
    }

    // MARK: - NSServicesMenuRequestor

    public override func validRequestor(forSendType sendType: NSPasteboard.PasteboardType?, returnType: NSPasteboard.PasteboardType?) -> Any? {
        // We can provide string data if we have selected text
        if let sendType = sendType, sendType == .string, currentSelectedText != nil {
            return self
        }
        return super.validRequestor(forSendType: sendType, returnType: returnType)
    }

    nonisolated public func writeSelection(to pboard: NSPasteboard, types: [NSPasteboard.PasteboardType]) -> Bool {
        let text = MainActor.assumeIsolated {
            currentSelectedText
        }

        guard let text else {
            return false
        }

        // Clear the pasteboard and write our selected text
        pboard.clearContents()
        pboard.setString(text, forType: .string)
        return true
    }

    // MARK: - Accessibility

    public override func isAccessibilityElement() -> Bool {
        return false
    }

    @MainActor
    public override func accessibilityChildren() -> [Any]? {
        recordAccessibilityQuery()
        requestAccessibilitySnapshotIfNeeded(force: false)
        return accessibilityElements.isEmpty ? nil : accessibilityElements
    }

    public override func accessibilityHitTest(_ point: NSPoint) -> Any? {
        let hitElement: OuterframeAccessibilityElement? = MainActor.assumeIsolated {
            recordAccessibilityQuery()
            requestAccessibilitySnapshotIfNeeded(force: false)
            return hitTestAccessibilityElementsOnMain(point: point, elements: accessibilityElements)
        }
        if let element = hitElement {
            return element
        }
        return super.accessibilityHitTest(point)
    }

    func resetAccessibilitySnapshot() {
        accessibilitySnapshot = nil
        accessibilityElements = []
        accessibilityRequestInFlight = false
        accessibilitySnapshotIsStale = false
        lastAccessibilityQueryUptime = nil
        pendingAccessibilityNotifications = []
        needsInitialAccessibilityAnnouncement = true
    }

    func requestAccessibilitySnapshot(force: Bool) {
        requestAccessibilitySnapshotIfNeeded(force: force)
    }

    func handleAccessibilityTreeChanged(notifications: OuterframeAccessibilityNotification) {
        pendingAccessibilityNotifications.formUnion(notifications)
        accessibilitySnapshotIsStale = true
        if shouldMaintainLiveAccessibilitySnapshot {
            requestAccessibilitySnapshotIfNeeded(force: true)
        }
    }

    private func requestAccessibilitySnapshotIfNeeded(force: Bool) {
        guard let connection = outerframeContentConnection else { return }
        if accessibilityRequestInFlight {
            return
        }
        if !force, accessibilitySnapshot != nil, !accessibilitySnapshotIsStale {
            return
        }
        accessibilityRequestInFlight = true
        connection.requestAccessibilitySnapshot { [weak self] snapshot in
            Task { @MainActor in
                guard let self else { return }
                self.accessibilityRequestInFlight = false
                self.updateAccessibilitySnapshot(snapshot)
            }
        }
    }

    private func updateAccessibilitySnapshot(_ snapshot: OuterframeAccessibilitySnapshot?) {
        let resolvedSnapshot = snapshot ?? OuterframeAccessibilitySnapshot.notImplementedSnapshot()
        accessibilitySnapshot = resolvedSnapshot
        accessibilityElements = buildAccessibilityElements(from: resolvedSnapshot)
        accessibilitySnapshotIsStale = false

        var notificationsToPost = pendingAccessibilityNotifications
        pendingAccessibilityNotifications = []
        if notificationsToPost.isEmpty || needsInitialAccessibilityAnnouncement {
            notificationsToPost.insert(.layoutChanged)
        }
        needsInitialAccessibilityAnnouncement = false
        postAccessibilityNotifications(for: notificationsToPost)
    }

    private func recordAccessibilityQuery() {
        lastAccessibilityQueryUptime = ProcessInfo.processInfo.systemUptime
    }

    private var shouldMaintainLiveAccessibilitySnapshot: Bool {
        if cachedVoiceOverEnabled {
            return true
        }
        guard let lastAccessibilityQueryUptime else {
            return false
        }
        return ProcessInfo.processInfo.systemUptime - lastAccessibilityQueryUptime <= Self.accessibilityActiveInterval
    }

    private func buildAccessibilityElements(from snapshot: OuterframeAccessibilitySnapshot) -> [OuterframeAccessibilityElement] {
        snapshot.rootNodes.map { buildAccessibilityElement(for: $0, parent: nil) }
    }

    private func buildAccessibilityElement(for node: OuterframeAccessibilityNode,
                                           parent: OuterframeAccessibilityElement?) -> OuterframeAccessibilityElement {
        let element = OuterframeAccessibilityElement(owner: self, node: node, parent: parent)
        element.childElements = node.children.map { buildAccessibilityElement(for: $0, parent: element) }
        return element
    }

    @MainActor
    private func hitTestAccessibilityElementsOnMain(point: NSPoint,
                                                    elements: [OuterframeAccessibilityElement]) -> OuterframeAccessibilityElement? {
        for element in elements {
            let frame = element.accessibilityFrame()
            if frame.contains(point) {
                if let childHit = hitTestAccessibilityElementsOnMain(point: point, elements: element.childElements) {
                    return childHit
                }
                return element
            }
        }
        return nil
    }

    private func postAccessibilityNotifications(for notifications: OuterframeAccessibilityNotification) {
        if notifications.contains(.layoutChanged) {
            NSAccessibility.post(element: self, notification: .layoutChanged)
        }
        if notifications.contains(.selectedChildrenChanged) {
            NSAccessibility.post(element: self, notification: .selectedChildrenChanged)
        }
        if notifications.contains(.focusedElementChanged) {
            NSAccessibility.post(element: self, notification: .focusedUIElementChanged)
        }
    }

    private nonisolated func accessibilityRole(for role: OuterframeAccessibilityRole) -> NSAccessibility.Role {
        switch role {
        case .container:
            return .group
        case .staticText:
            return .staticText
        case .button:
            return .button
        case .image:
            return .image
        case .table:
            return .table
        case .row:
            return .row
        case .cell:
            return .cell
        case .textField:
            return .textField
        @unknown default:
            return .unknown
        }
    }

    private nonisolated func accessibilityFrame(for rect: CGRect) -> NSRect {
        MainActor.assumeIsolated {
            var frame = NSRect(x: rect.origin.x, y: rect.origin.y, width: rect.size.width, height: rect.size.height)
            frame = convert(frame, to: nil)
            if let window = window {
                frame = window.convertToScreen(frame)
            }
            return frame
        }
    }

    private final class OuterframeAccessibilityElement: NSAccessibilityElement, @unchecked Sendable {
        weak var owner: OuterframeView?
        weak var parentElement: OuterframeAccessibilityElement?
        let node: OuterframeAccessibilityNode
        var childElements: [OuterframeAccessibilityElement] = []

        init(owner: OuterframeView, node: OuterframeAccessibilityNode, parent: OuterframeAccessibilityElement?) {
            self.owner = owner
            self.node = node
            self.parentElement = parent
            super.init()
        }

        override func accessibilityRole() -> NSAccessibility.Role? {
            owner?.accessibilityRole(for: node.role)
        }

        override func accessibilityLabel() -> String? {
            node.label
        }

        override func accessibilityValue() -> Any? {
            node.value
        }

        override func accessibilityHelp() -> String? {
            node.hint
        }

        override func accessibilityIdentifier() -> String? {
            "outerframe-node-\(node.identifier)"
        }

        override func accessibilityFrame() -> NSRect {
            guard let owner = owner else { return .zero }
            return owner.accessibilityFrame(for: node.frame)
        }

        override func accessibilityChildren() -> [Any]? {
            childElements.isEmpty ? nil : childElements
        }

        override func accessibilityParent() -> Any? {
            parentElement ?? owner
        }

        override func isAccessibilityElement() -> Bool {
            return true
        }

        override func isAccessibilityEnabled() -> Bool {
            node.isEnabled
        }
    }

    // MARK: - Mouse Event Handling

    public func handleMouseDown(at point: CGPoint, modifierFlags: NSEvent.ModifierFlags, clickCount: Int) {
        withActivePluginConnection { connection in
            connection.sendMouseEvent(type: .down, point: point, modifierFlags: modifierFlags, clickCount: clickCount)
        }
    }

    public func handleMouseDragged(to point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendMouseEvent(type: .dragged, point: point, modifierFlags: modifierFlags)
        }
    }

    public func handleMouseUp(at point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendMouseEvent(type: .up, point: point, modifierFlags: modifierFlags)
        }
    }

    public func handleMouseMoved(to point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendMouseEvent(type: .moved, point: point, modifierFlags: modifierFlags)
        }
    }

    public func handleScrollWheel(at point: CGPoint, with event: NSEvent) {
        withActivePluginConnection { connection in
            let delta = CGPoint(x: event.scrollingDeltaX, y: event.scrollingDeltaY)
            let isMomentum = !event.momentumPhase.isEmpty
            connection.sendScrollWheelEvent(point: point,
                                            delta: delta,
                                            modifierFlags: event.modifierFlags,
                                            phase: event.phase,
                                            momentumPhase: event.momentumPhase,
                                            isMomentum: isMomentum,
                                            isPrecise: event.hasPreciseScrollingDeltas)
        }
    }

    public func handleMagnify(at point: CGPoint, with event: NSEvent) {
        withActivePluginConnection { connection in
            let delta = event.magnification
            if delta != 0 {
                connection.sendMagnification(surfaceId: manualMagnificationSurfaceId,
                                             magnification: delta,
                                             location: point,
                                             scrollOffset: .zero)
            }

            if event.phase == .ended || event.phase == .cancelled {
                connection.sendMagnificationEnded(surfaceId: manualMagnificationSurfaceId,
                                                  magnification: 0,
                                                  location: point,
                                                  scrollOffset: .zero)
            }
        }
    }

    func handleRightMouseDown(at point: CGPoint, modifierFlags: NSEvent.ModifierFlags, clickCount: Int) {
        withActivePluginConnection { connection in
            connection.sendMouseEvent(type: .rightDown, point: point, modifierFlags: modifierFlags, clickCount: clickCount)
        }
    }

    public func handleRightMouseUp(at point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendMouseEvent(type: .rightUp, point: point, modifierFlags: modifierFlags)
        }
    }


    public func handleQuickLook(at point: CGPoint) {
        withActivePluginConnection { connection in
            connection.sendQuickLookEvent(point: point)
        }
    }

    public func handleKeyDown(_ event: NSEvent) {
        guard currentInputMode.allowsRawKeys else { return }
        withActivePluginConnection { connection in
            let characters = event.characters ?? ""
            let charactersIgnoringModifiers = event.charactersIgnoringModifiers ?? ""
            connection.sendKeyDown(keyCode: event.keyCode, characters: characters, charactersIgnoringModifiers: charactersIgnoringModifiers, modifierFlags: event.modifierFlags, isRepeat: event.isARepeat)
        }
    }

    public func handleKeyUp(_ event: NSEvent) {
        guard currentInputMode.allowsRawKeys else { return }
        withActivePluginConnection { connection in
            let characters = event.characters ?? ""
            let charactersIgnoringModifiers = event.charactersIgnoringModifiers ?? ""
            connection.sendKeyUp(keyCode: event.keyCode, characters: characters, charactersIgnoringModifiers: charactersIgnoringModifiers, modifierFlags: event.modifierFlags, isRepeat: event.isARepeat)
        }
    }

    func setInputMode(_ inputMode: OuterframeContentInputMode) {
        var labels: [String] = []
        if inputMode.allowsTextInput { labels.append("textInput") }
        if inputMode.allowsRawKeys { labels.append("rawKeys") }
        if labels.isEmpty { labels.append("none") }
        let description = labels.joined(separator: "|")
        print("OuterframeView: Updating input mode to \(description)")
        currentInputMode = inputMode
    }

    func updateCursor(_ cursorType: PluginCursorType) {
        // Update cursor based on plugin's request
        switch cursorType {
        case .arrow:
            NSCursor.arrow.set()
        case .iBeam:
            NSCursor.iBeam.set()
        case .crosshair:
            NSCursor.crosshair.set()
        case .openHand:
            NSCursor.openHand.set()
        case .closedHand:
            NSCursor.closedHand.set()
        case .pointingHand:
            NSCursor.pointingHand.set()
        case .resizeLeft:
            NSCursor.resizeLeft.set()
        case .resizeRight:
            NSCursor.resizeRight.set()
        case .resizeLeftRight:
            NSCursor.resizeLeftRight.set()
        case .resizeUp:
            NSCursor.resizeUp.set()
        case .resizeDown:
            NSCursor.resizeDown.set()
        case .resizeUpDown:
            NSCursor.resizeUpDown.set()
        @unknown default:
            NSCursor.arrow.set()
        }
    }

    func performHapticFeedback(_ style: OuterframeHapticFeedbackStyle) {
        let manager = NSHapticFeedbackManager.defaultPerformer
        let feedbackStyle: NSHapticFeedbackManager.FeedbackPattern
        switch style {
        case .alignment:
            feedbackStyle = .alignment
        case .levelChange:
            feedbackStyle = .levelChange
        case .generic:
            feedbackStyle = .generic
        @unknown default:
            feedbackStyle = .generic
        }
        manager.perform(feedbackStyle, performanceTime: .default)
    }

    @discardableResult
    private func cleanupCurrentPlugin() -> Task<Void, Never>? {
        guard let connection = outerframeContentConnection else {
            // Already called cleanup
            return pendingOuterframeContentCleanupTask
        }

        pluginIsReady = false
        stopDisplayLink()

        setOuterframeContentConnection(nil)
        let cleanupToken = UUID()
        pendingOuterframeContentCleanupToken = cleanupToken
        let cleanupTask = Task { [weak self] in
            do {
                try await connection.unloadPlugin(timeout: 30)
            } catch OuterframeContentConnectionError.pluginUnloadTimedOut {
                print("Browser: Plugin unload timed out; terminating OuterframeContent process")
            } catch {
                print("Browser: Plugin unload failed: \(error)")
            }
            await connection.cleanup()
            await MainActor.run {
                guard let self else { return }
                guard self.pendingOuterframeContentCleanupToken == cleanupToken else { return }
                self.pendingOuterframeContentCleanupTask = nil
                self.pendingOuterframeContentCleanupToken = nil
            }
        }
        pendingOuterframeContentCleanupTask = cleanupTask
        return cleanupTask
    }

    public func cleanupPlugin() {
        _ = cleanupCurrentPlugin()
    }

    public func shutdownOuterframeContent() async {
        let cleanupTask = cleanupCurrentPlugin()
        await cleanupTask?.value
    }

    public func loadOuterframeContent(from url: URL,
                                      networkProxyEndpoint: OuterframeNetworkProxyEndpoint,
                                      cacheContext: OuterframeCacheContext?,
                                      bypassCache: Bool = false) async throws -> OuterframeViewLoadResult {
        // Check for local bundle override (used for external OuterframeContent debugging)
        if let localPath = localBundleOverridePath {
            let localBundleURL = URL(fileURLWithPath: localPath)

            // Verify the local bundle exists
            guard FileManager.default.fileExists(atPath: localPath) else {
                throw OuterframeLoadError.localBundleNotFound(localPath)
            }

            // Still fetch the resource to get pluginData from the server
            switch try await fetchOuterframeResource(from: url,
                                                     networkProxyEndpoint: networkProxyEndpoint,
                                                     bypassCache: bypassCache) {
            case .notOuterframe:
                return .notOuterframe
            case .resource(_, let pluginData, let originalURL, let downloadURL):
                // Use local bundle instead of downloading
                try await loadPlugin(bundleURL: localBundleURL,
                                     data: pluginData,
                                     originalURL: originalURL,
                                     downloadURL: downloadURL,
                                     networkProxyEndpoint: networkProxyEndpoint)
                return .loaded
            }
        }

        // Normal flow: download bundle from server
        switch try await fetchOuterframeResource(from: url,
                                                 networkProxyEndpoint: networkProxyEndpoint,
                                                 bypassCache: bypassCache) {
        case .notOuterframe:
            return .notOuterframe
        case .resource(let remotePath, let pluginData, let originalURL, let downloadURL):
            let bundleContext = makeBundleCacheContext(from: cacheContext,
                                                       remotePath: remotePath,
                                                       fallbackURL: downloadURL)
            let bundleURL = try await downloadPluginBundle(remotePath: remotePath,
                                                           downloadURL: downloadURL,
                                                           networkProxyEndpoint: networkProxyEndpoint,
                                                           cacheContext: bundleContext,
                                                           bypassCache: bypassCache)

            try await loadPlugin(bundleURL: bundleURL,
                                 data: pluginData,
                                 originalURL: originalURL,
                                 downloadURL: downloadURL,
                                 networkProxyEndpoint: networkProxyEndpoint)
            return .loaded
        }
    }

    private enum OuterframeFetchOutcome {
        case notOuterframe
        case resource(remotePath: String, pluginData: Data, originalURL: URL, downloadURL: URL)
    }

    public func loadPlugin(bundleURL: URL,
                           data: Data,
                           originalURL: URL,
                           downloadURL: URL,
                           networkProxyEndpoint: OuterframeNetworkProxyEndpoint) async throws {
        let pluginURLs = makePluginLoadURLStrings(originalURL: originalURL,
                                                  downloadURL: downloadURL)
        try await loadPlugin(bundleURL: bundleURL,
                             data: data,
                             outerURLString: pluginURLs.outerURLString,
                             bundleURLString: pluginURLs.bundleURLString,
                             networkProxyEndpoint: networkProxyEndpoint)
    }

    func loadPlugin(bundleURL: URL,
                    data: Data,
                    outerURLString: String,
                    bundleURLString: String,
                    networkProxyEndpoint: OuterframeNetworkProxyEndpoint) async throws {
        let connection: OuterframeContentConnection

        // Check if an external connection is already attached and ready BEFORE cleanup
        if hasExternalConnectionReady, let existingConnection = outerframeContentConnection {
            // Reuse the external connection instead of spawning a new process
            connection = existingConnection
            hasExternalConnectionReady = false
            // Skip cleanupCurrentPlugin - we're reusing this connection
        } else {
            // Normal flow: cleanup any existing plugin and spawn a new process
            cleanupCurrentPlugin()

            connection = OuterframeContentConnection()
            debuggerAttachedForCurrentConnection = false
            setOuterframeContentConnection(connection)

            do {
                try await connection.start(networkProxyEndpoint: networkProxyEndpoint)
            } catch {
                await connection.cleanup()
                setOuterframeContentConnection(nil)
                throw error
            }
        }

        let viewportSize = bounds.size
        let width = viewportSize.width
        let height = viewportSize.height

        do {
            try await loadPluginWithDebugHandling(connection: connection,
                                                  bundleURL: bundleURL,
                                                  data: data,
                                                  outerURLString: outerURLString,
                                                  bundleURLString: bundleURLString,
                                                  width: width,
                                                  height: height)
        } catch {
            await connection.cleanup()
            setOuterframeContentConnection(nil)
            throw error
        }
    }

    /// Phase 1: Prepare for external OuterframeContent connection.
    /// Creates the connection and starts the network proxy.
    /// Returns the proxy port to include in the handshake response.
    /// After sending the handshake response, call completeExternalOuterframeContent(socketFD:).
    public func prepareExternalOuterframeContent(localBundlePath: String?) async throws -> UInt16 {
        cleanupCurrentPlugin()

        if let existing = outerframeContentConnection {
            await existing.cleanup()
            setOuterframeContentConnection(nil)
        }

        // Store local bundle override path for use during plugin loading
        localBundleOverridePath = localBundlePath

        let connection = OuterframeContentConnection()
        debuggerAttachedForCurrentConnection = false
        setOuterframeContentConnection(connection)

        do {
            let proxyPort = try await connection.prepareExternalConnection()
            return proxyPort
        } catch {
            await connection.cleanup()
            setOuterframeContentConnection(nil)
            localBundleOverridePath = nil
            throw error
        }
    }

    /// Phase 2: Complete external OuterframeContent connection after handshake response is sent.
    /// Must call prepareExternalOuterframeContent() first.
    /// - Parameters:
    ///   - infrastructureSocketFD: Socket for infrastructure messages (loadPlugin, pluginLoaded, etc.)
    ///   - pluginSocketFD: Socket for plugin messages (mouse, keyboard, display link, etc.)
    public func completeExternalOuterframeContent(infrastructureSocketFD: Int32, pluginSocketFD: Int32) async {
        guard let connection = outerframeContentConnection else {
            print("OuterframeView: No prepared connection for completeExternalOuterframeContent")
            return
        }

        await connection.attachExternalSocket(infrastructureSocketFD: infrastructureSocketFD,
                                              pluginSocketFD: pluginSocketFD)
        connection.sendWindowActiveState(isActive: windowIsActive)
        connection.sendSystemAppearanceUpdate(force: true)

        // Mark that an external connection is ready - loadPlugin should reuse it
        hasExternalConnectionReady = true
    }

    private func loadPluginWithDebugHandling(connection: OuterframeContentConnection,
                                             bundleURL: URL,
                                             data: Data,
                                             outerURLString: String,
                                             bundleURLString: String,
                                             width: CGFloat,
                                             height: CGFloat) async throws {
        if OuterframeDebugSettings.isDebugModeEnabled {
            try await waitForDebuggerBeforeLoadingPlugin(connection: connection,
                                                         bundleURL: bundleURL,
                                                         data: data,
                                                         outerURLString: outerURLString,
                                                         bundleURLString: bundleURLString,
                                                         width: width,
                                                         height: height)
        } else {
            try await connection.loadPlugin(url: bundleURL,
                                            data: data,
                                            outerURLString: outerURLString,
                                            bundleURLString: bundleURLString,
                                            width: width,
                                            height: height)
        }
    }

    private func waitForDebuggerBeforeLoadingPlugin(connection: OuterframeContentConnection,
                                                    bundleURL: URL,
                                                    data: Data,
                                                    outerURLString: String,
                                                    bundleURLString: String,
                                                    width: CGFloat,
                                                    height: CGFloat) async throws {
        try await withCheckedThrowingContinuation { continuation in
            let pendingLoad = PendingDebugLoad(connection: connection,
                                               bundleURL: bundleURL,
                                               pluginData: data,
                                               outerURLString: outerURLString,
                                               bundleURLString: bundleURLString,
                                               width: width,
                                               height: height,
                                               debuggerAttached: debuggerAttachedForCurrentConnection,
                                               continuation: continuation)
            pendingDebugLoad = pendingLoad
            connection.setDebuggerAttachmentMonitoring(isEnabled: true)
            showDebugPauseOverlay(debuggerAttached: pendingLoad.debuggerAttached)

            if pendingLoad.debuggerAttached,
               OuterframeDebugSettings.shouldAutoResumeOnDebuggerAttach {
                resumePendingDebugLoad()
            }
        }
    }

    private func fetchOuterframeResource(from url: URL,
                                         networkProxyEndpoint: OuterframeNetworkProxyEndpoint,
                                         bypassCache: Bool) async throws -> OuterframeFetchOutcome {
        // First, do a HEAD request to check Content-Type without consuming the response body.
        // This avoids issues with URLs that have single-use tokens (e.g., Jupyter Notebook).
        let headSession = makeProxyAwareSession(networkProxyEndpoint: networkProxyEndpoint)
        defer { headSession.finishTasksAndInvalidate() }

        var headRequest = URLRequest(url: url)
        headRequest.httpMethod = "HEAD"
        headRequest.setValue(Self.topLevelNavigationCapabilityHeaderValue,
                             forHTTPHeaderField: Self.topLevelNavigationCapabilityHeaderField)
        if bypassCache {
            headRequest.cachePolicy = .reloadIgnoringLocalCacheData
        }
        let headResponse: URLResponse
        do {
            let responsePair = try await headSession.data(for: headRequest)
            headResponse = responsePair.1
        } catch {
            return .notOuterframe
        }

        guard let headHttpResponse = headResponse as? HTTPURLResponse else {
            return .notOuterframe
        }

        guard (200...299).contains(headHttpResponse.statusCode) else {
            return .notOuterframe
        }

        // Check if this is Outerframe content based on Content-Type header
        guard let mime = headHttpResponse.mimeType?.lowercased(),
              mime == Self.outerframeMimeType else {
            return .notOuterframe
        }

        // It's Outerframe content - now do the full GET request to fetch the data
        let getSession = makeProxyAwareSession(networkProxyEndpoint: networkProxyEndpoint)
        defer { getSession.finishTasksAndInvalidate() }

        var request = URLRequest(url: url)
        request.setValue(Self.topLevelNavigationCapabilityHeaderValue,
                         forHTTPHeaderField: Self.topLevelNavigationCapabilityHeaderField)
        if bypassCache {
            request.cachePolicy = .reloadIgnoringLocalCacheData
        }
        let (data, response) = try await getSession.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            return .notOuterframe
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw OuterframeLoadError.unexpectedHTTPStatus(httpResponse.statusCode)
        }

        guard data.count >= 3 else {
            throw OuterframeLoadError.invalidOuterframeData
        }

        let formatVersion = data[0]
        guard formatVersion == 0 else {
            throw OuterframeLoadError.invalidOuterframeData
        }

        let pathLength = UInt16(data[1]) << 8 | UInt16(data[2])
        let contentBlobStart = 3 + Int(pathLength)

        guard data.count >= contentBlobStart else {
            throw OuterframeLoadError.invalidPathLength
        }

        let pathData = data.subdata(in: 3..<contentBlobStart)
        guard let remotePath = String(data: pathData, encoding: .utf8) else {
            throw OuterframeLoadError.invalidRemotePath
        }

        let pluginData = data.subdata(in: contentBlobStart..<data.count)

        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            throw OuterframeLoadError.invalidDownloadURL
        }
        components.query = nil
        components.fragment = nil
        components.path = remotePath + "/" + Self.currentMacOSPlatformPathComponent
        guard let downloadURL = components.url else {
            throw OuterframeLoadError.invalidDownloadURL
        }

        return .resource(remotePath: remotePath,
                         pluginData: pluginData,
                         originalURL: url,
                         downloadURL: downloadURL)
    }

    private static var currentMacOSPlatformPathComponent: String {
        #if arch(arm64)
        return "macos-arm"
        #elseif arch(x86_64)
        return "macos-x86"
        #else
        #error("Unsupported macOS architecture")
        #endif
    }

    private func makeBundleCacheContext(from cacheContext: OuterframeCacheContext?,
                                        remotePath: String,
                                        fallbackURL: URL) -> OuterframeBundleCache.Context {
        let serverID = cacheContext?.serverID
        let host: String
        let port: Int32

        if let cacheContext {
            host = cacheContext.host
            port = cacheContext.port
        } else {
            host = (fallbackURL.host?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "localhost")
            port = Int32(fallbackURL.port ?? 0)
        }

        return OuterframeBundleCache.Context(serverID: serverID,
                                             host: host,
                                             port: port,
                                             remotePath: remotePath)
    }

    private func downloadPluginBundle(remotePath: String,
                                      downloadURL: URL,
                                      networkProxyEndpoint: OuterframeNetworkProxyEndpoint,
                                      cacheContext: OuterframeBundleCache.Context,
                                      bypassCache: Bool) async throws -> URL {
        let entryDirectory = try OuterframeBundleCache.entryDirectoryPath(for: cacheContext)

        var pluginFileName = URL(fileURLWithPath: remotePath).lastPathComponent
        if pluginFileName.isEmpty {
            pluginFileName = "OuterframePlugin-\(UUID().uuidString)"
        }
        pluginFileName += ".aar"

        let bundleURL = try await downloadAndExtractBundleWithoutQuarantineViaXPC(
            downloadURL: downloadURL.absoluteString,
            proxyHost: networkProxyEndpoint.host,
            proxyPort: networkProxyEndpoint.port,
            proxyUsername: networkProxyEndpoint.username,
            proxyPassword: networkProxyEndpoint.password,
            destinationDirectoryURL: entryDirectory,
            archiveFilename: pluginFileName,
            bypassCache: bypassCache
        )

        return bundleURL
    }

    private nonisolated func downloadAndExtractBundleWithoutQuarantineViaXPC(downloadURL: String,
                                                                             proxyHost: String?,
                                                                             proxyPort: UInt16,
                                                                             proxyUsername: String?,
                                                                             proxyPassword: String?,
                                                                             destinationDirectoryURL: URL,
                                                                             archiveFilename: String,
                                                                             bypassCache: Bool) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            guard let proxy = Self.outerframeProcessesConnection.remoteObjectProxyWithErrorHandler({ error in
                continuation.resume(throwing: error)
            }) as? OuterframeProcessesProtocol else {
                continuation.resume(throwing: OuterframeLoadError.xpcConnectionFailed)
                return
            }

            proxy.downloadAndExtractBundleWithoutQuarantine(downloadURL: downloadURL,
                                                            proxyHost: proxyHost,
                                                            proxyPort: proxyPort,
                                                            proxyUsername: proxyUsername,
                                                            proxyPassword: proxyPassword,
                                                            destinationDirectoryURL: destinationDirectoryURL,
                                                            archiveFilename: archiveFilename,
                                                            bypassCache: bypassCache) { bundleURL, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let bundleURL = bundleURL else {
                    continuation.resume(throwing: OuterframeLoadError.missingBundle)
                    return
                }

                continuation.resume(returning: bundleURL)
            }
        }
    }

    private func makePluginLoadURLStrings(originalURL: URL,
                                          downloadURL: URL) -> (outerURLString: String, bundleURLString: String) {
        let presentationOriginalURL = makeSyntheticLoopbackURL(from: originalURL) ?? originalURL
        let presentationDownloadURL = makeSyntheticLoopbackURL(from: downloadURL) ?? downloadURL

        return (
            outerURLString: presentationOriginalURL.absoluteString,
            bundleURLString: presentationDownloadURL.absoluteString
        )
    }

    private func makeSyntheticLoopbackURL(from url: URL) -> URL? {
        guard let host = url.host, isLoopbackHost(host) else { return nil }
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
        components.host = "outerframe.local"
        return components.url
    }

    private func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized == "localhost"
            || normalized == "127.0.0.1"
            || normalized == "::1"
            || normalized == "0:0:0:0:0:0:0:1"
            || normalized == "0000:0000:0000:0000:0000:0000:0000:0001"
    }

    private func makeProxyAwareSession(networkProxyEndpoint: OuterframeNetworkProxyEndpoint) -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        if !networkProxyEndpoint.host.isEmpty,
           let endpointPort = NWEndpoint.Port(rawValue: networkProxyEndpoint.port) {
            let endpoint = NWEndpoint.hostPort(host: NWEndpoint.Host(networkProxyEndpoint.host),
                                               port: endpointPort)
            var proxy = ProxyConfiguration(socksv5Proxy: endpoint)
            if let username = networkProxyEndpoint.username,
               let password = networkProxyEndpoint.password,
               (!username.isEmpty || !password.isEmpty) {
                proxy.applyCredential(username: username, password: password)
            }
            proxy.allowFailover = false
            configuration.proxyConfigurations = [proxy]
        }
        return URLSession(configuration: configuration)
    }

    private func showDebugPauseOverlay(debuggerAttached: Bool) {
        let overlay = ensureDebugOverlayView()
        overlay.isHidden = false
        overlay.alphaValue = 1.0
        overlay.updateDebuggerState(isAttached: debuggerAttached,
                                    processIdentifier: currentOuterframeContentProcessIdentifier())
        overlay.setAutoResumeSelected(OuterframeDebugSettings.shouldAutoResumeOnDebuggerAttach)
    }

    private func ensureDebugOverlayView() -> OuterframeDebugOverlayView {
        if let overlay = debugOverlayView {
            return overlay
        }

        let overlay = OuterframeDebugOverlayView()
        overlay.translatesAutoresizingMaskIntoConstraints = false
        overlay.resumeHandler = { [weak self] in
            self?.resumePendingDebugLoad()
        }
        overlay.autoResumeChangedHandler = { isSelected in
            OuterframeDebugSettings.shouldAutoResumeOnDebuggerAttach = isSelected
        }
        addSubview(overlay, positioned: .above, relativeTo: nil)
        NSLayoutConstraint.activate([
            overlay.leadingAnchor.constraint(equalTo: leadingAnchor),
            overlay.trailingAnchor.constraint(equalTo: trailingAnchor),
            overlay.topAnchor.constraint(equalTo: topAnchor),
            overlay.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
        debugOverlayView = overlay
        return overlay
    }

    private func hideDebugPauseOverlay() {
        debugOverlayView?.isHidden = true
    }

    private func currentOuterframeContentProcessIdentifier() -> pid_t? {
        if let pending = pendingDebugLoad {
            return pending.connection.processIdentifier
        }
        return outerframeContentConnection?.processIdentifier
    }

    private func resumePendingDebugLoad() {
        guard let pending = pendingDebugLoad else { return }
        pendingDebugLoad = nil
        hideDebugPauseOverlay()
        pending.connection.setDebuggerAttachmentMonitoring(isEnabled: false)

        let connection = pending.connection
        let bundleURL = pending.bundleURL
        let pluginData = pending.pluginData
        let outerURLString = pending.outerURLString
        let bundleURLString = pending.bundleURLString
        let width = pending.width
        let height = pending.height
        let continuation = pending.continuation
        pending.continuation = nil

        Task { @MainActor in
            do {
                try await connection.loadPlugin(url: bundleURL,
                                                data: pluginData,
                                                outerURLString: outerURLString,
                                                bundleURLString: bundleURLString,
                                                width: width,
                                                height: height)
                continuation?.resume(returning: ())
            } catch {
                continuation?.resume(throwing: error)
            }
        }
    }

    private func cancelPendingDebugLoadIfNeeded(error: Error) {
        guard let pending = pendingDebugLoad else { return }
        pendingDebugLoad = nil
        hideDebugPauseOverlay()
        pending.connection.setDebuggerAttachmentMonitoring(isEnabled: false)
        if let continuation = pending.continuation {
            pending.continuation = nil
            continuation.resume(throwing: error)
        }
    }


    public func registerDisplayLinkCallback() -> UUID {
        let callbackId = UUID()

        displayLinkLock.lock()
        activeDisplayLinkCallbacks[callbackId] = true
        displayLinkLock.unlock()

        // Start the display link if it's not already running
        startDisplayLink()

        return callbackId
    }

    public func unregisterDisplayLinkCallback(_ callbackId: UUID) {
        displayLinkLock.lock()
        activeDisplayLinkCallbacks.removeValue(forKey: callbackId)
        let isEmpty = activeDisplayLinkCallbacks.isEmpty
        displayLinkLock.unlock()

        // Stop display link if no more callbacks
        if isEmpty {
            stopDisplayLink()
        }
    }

    private func startDisplayLink() {
        guard displayLink == nil else { return }

        // Only actually start if window is visible
        guard isWindowVisible else {
            shouldRunDisplayLink = true  // Remember to start when window becomes visible
            return
        }

        if let screen = NSScreen.main {
            displayLink = screen.displayLink(target: self, selector: #selector(displayLinkCallback))
            displayLink?.add(to: .main, forMode: .common)
            shouldRunDisplayLink = true
        }
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
        shouldRunDisplayLink = false
    }

    public func updateWindowVisibility(_ isVisible: Bool) {
        guard isWindowVisible != isVisible else { return }
        isWindowVisible = isVisible
        if isVisible {
            if shouldRunDisplayLink && displayLink == nil {
                startDisplayLink()
            }
        } else {
            displayLink?.invalidate()
            displayLink = nil
        }
    }

    public func updateWindowActivationState(_ isActive: Bool) {
        if windowIsActive == isActive {
            return
        }
        windowIsActive = isActive
        outerframeContentConnection?.sendWindowActiveState(isActive: windowIsActive)
    }

    @objc private func displayLinkCallback(_ displayLink: CADisplayLink) {
        renderFrame(targetTimestamp: displayLink.targetTimestamp.bitPattern)
    }

    private func renderFrame(targetTimestamp: UInt64) {
        // Check if we have any active display link callbacks
        displayLinkLock.lock()
        let hasActiveCallbacks = !activeDisplayLinkCallbacks.isEmpty
        displayLinkLock.unlock()

        if hasActiveCallbacks {
            // Convert to CFTimeInterval
            let timestamp = CACurrentMediaTime()

            // Notify OuterframeContent via socket
            outerframeContentConnection?.sendDisplayLinkFrame(frameNumber: targetTimestamp, targetTimestamp: timestamp)
        } else {
            // No active callbacks, stop the display link
            DispatchQueue.main.async { [weak self] in
                self?.stopDisplayLink()
            }
        }
    }

    public func handlePluginLoaded(contextId: CAContextID) {
        pluginIsReady = true

        // Update canvas with the plugin's context ID
        resetEditingCapabilities()
        resetAccessibilitySnapshot()
        updateWithContextId(contextId)

        // Make sure the canvas has focus for keyboard input
        window?.makeFirstResponder(self)

        if let outerframeContentConnection {
            outerframeContentConnection.sendSystemAppearanceUpdate(force: true)
            outerframeContentConnection.sendWindowActiveState(isActive: windowIsActive)
        }

        requestAccessibilitySnapshot(force: true)
    }

    public func handleAccessibilityTreeChanged(notificationMask: UInt8) {
        let notifications = OuterframeAccessibilityNotification(rawValue: notificationMask)
        handleAccessibilityTreeChanged(notifications: notifications)
    }

    public func handleTextCursorUpdate(cursors: [[String: Any]]) {
        // Forward cursor updates to the canvas view
        updateTextCursors(cursors)
    }

    public func handlePluginPageMetadataUpdate(title: String?, iconPNGData: Data?, iconWidth: UInt32, iconHeight: UInt32) {
        var shouldUpdateIcon = false
        var decodedIcon: NSImage? = nil

        if let iconPNGData {
            shouldUpdateIcon = true
            if let image = NSImage(data: iconPNGData) {
                if iconWidth > 0 && iconHeight > 0 {
                    image.size = NSSize(width: CGFloat(iconWidth), height: CGFloat(iconHeight))
                }
                decodedIcon = image
            }
        }

        if shouldUpdateIcon {
            delegate?.outerframeView(self, didUpdateFavicon: decodedIcon)
        }

        delegate?.outerframeView(self, didUpdatePageTitle: title)
    }

    public func handlePluginStartPageMetadataUpdate(title: String?, iconPNGData: Data?, iconWidth: UInt32, iconHeight: UInt32) {
        var renderedIcon: NSImage? = nil
        var resolvedSize: CGSize? = nil

        if let iconPNGData,
           let image = NSImage(data: iconPNGData) {
            if iconWidth > 0 && iconHeight > 0 {
                let size = NSSize(width: CGFloat(iconWidth), height: CGFloat(iconHeight))
                image.size = size
                resolvedSize = size
            } else {
                resolvedSize = image.size
            }
            renderedIcon = image
        }

        let metadata = OuterframeStartPageMetadata(title: title,
                                                   icon: renderedIcon,
                                                   iconSize: resolvedSize ?? renderedIcon?.size)
        delegate?.outerframeView(self, didUpdateStartPageMetadata: metadata)
    }

    public func handlePluginOpenNewWindow(urlString: String, displayString: String?, preferredSize: CGSize?) {
        delegate?.outerframeView(self,
                                 didRequestOpenWindowWithURLString: urlString,
                                 displayString: displayString,
                                 preferredSize: preferredSize)
    }

    func outerframeContentConnectionDebuggerAttached(_ connection: OuterframeContentConnection) {
        guard connection === outerframeContentConnection else { return }
        debuggerAttachedForCurrentConnection = true

        guard let pending = pendingDebugLoad else { return }
        pending.debuggerAttached = true
        debugOverlayView?.updateDebuggerState(isAttached: true,
                                              processIdentifier: connection.processIdentifier)

        if OuterframeDebugSettings.shouldAutoResumeOnDebuggerAttach {
            resumePendingDebugLoad()
        }
    }

    func outerframeContentConnectionDidTimeoutWaitingForPluginLoaded(_ connection: OuterframeContentConnection) {
        guard connection === outerframeContentConnection else { return }
        delegate?.outerframeViewDidTimeoutWaitingForPluginLoaded(self)
    }

    func outerframeContentConnection(_ connection: OuterframeContentConnection, didReceiveStdout text: String) {
        guard connection === outerframeContentConnection else { return }
        delegate?.outerframeView(self, didReceiveOuterframeContentStdout: text)
    }

    func outerframeContentConnection(_ connection: OuterframeContentConnection, didReceiveStderr text: String) {
        guard connection === outerframeContentConnection else { return }
        delegate?.outerframeView(self, didReceiveOuterframeContentStderr: text)
    }




    // MARK: - Text Input Handling

    func sendTextInput(_ text: String, replacementRange: NSRange) {
        withActivePluginConnection { connection in
            connection.sendTextInput(text: text, replacementRange: replacementRange)
        }
    }

    func sendSetMarkedText(_ text: String, selectedRange: NSRange, replacementRange: NSRange) {
        withActivePluginConnection { connection in
            connection.sendSetMarkedText(text: text, selectedRange: selectedRange, replacementRange: replacementRange)
        }
    }

    func sendUnmarkText() {
        withActivePluginConnection { connection in
            connection.sendUnmarkText()
        }
    }

    func sendTextInputFocus(fieldId: String, hasFocus: Bool) {
        withActivePluginConnection { connection in
            connection.sendTextInputFocus(fieldId: fieldId, hasFocus: hasFocus)
        }
    }

    func sendTextCommand(_ command: String) {
        withActivePluginConnection { connection in
            connection.sendTextCommand(command: command)
        }
    }

    func requestPasteboardItemsForCopy(completion: @escaping ([OuterframeContentPasteboardItem]?) -> Void) {
        guard pluginIsReady, let connection = outerframeContentConnection else {
            completion(nil)
            return
        }
        connection.requestPasteboardItemsForCopy(completion: completion)
    }

    func requestAccessibilitySnapshot(completion: @escaping (OuterframeAccessibilitySnapshot?) -> Void) {
        guard pluginIsReady, let connection = outerframeContentConnection else {
            completion(nil)
            return
        }
        connection.requestAccessibilitySnapshot(completion: completion)
    }

    func sendPasteboardItemsForPaste(_ items: [OuterframeContentPasteboardItem]) {
        withActivePluginConnection { connection in
            connection.sendPasteboardItemsForPaste(items)
        }
    }

    func sendSetCursorPosition(fieldId: String, position: Int, modifySelection: Bool) {
        withActivePluginConnection { connection in
            connection.sendSetCursorPosition(fieldId: fieldId, position: UInt64(position), modifySelection: modifySelection)
        }
    }



    func withActivePluginConnection(_ action: (OuterframeContentConnection) -> Void) {
        guard pluginIsReady, let connection = outerframeContentConnection else {
            return
        }

        action(connection)
    }

}


@MainActor
private final class PendingDebugLoad {
    let connection: OuterframeContentConnection
    let bundleURL: URL
    let pluginData: Data
    let outerURLString: String
    let bundleURLString: String
    let width: CGFloat
    let height: CGFloat
    var debuggerAttached: Bool
    var continuation: CheckedContinuation<Void, Error>?

    init(connection: OuterframeContentConnection,
         bundleURL: URL,
         pluginData: Data,
         outerURLString: String,
         bundleURLString: String,
         width: CGFloat,
         height: CGFloat,
         debuggerAttached: Bool,
         continuation: CheckedContinuation<Void, Error>) {
        self.connection = connection
        self.bundleURL = bundleURL
        self.pluginData = pluginData
        self.outerURLString = outerURLString
        self.bundleURLString = bundleURLString
        self.width = width
        self.height = height
        self.debuggerAttached = debuggerAttached
        self.continuation = continuation
    }
}

private final class OuterframeDebugOverlayView: NSView {
    var resumeHandler: (() -> Void)?
    var autoResumeChangedHandler: ((Bool) -> Void)?

    private let cardView = NSVisualEffectView()
    private let statusLabel = NSTextField(labelWithString: "Waiting for debugger to attach…")
    private let autoResumeButton = NSButton(checkboxWithTitle: "Automatically resume when debugger attaches.", target: nil, action: nil)
    private let resumeButton = NSButton(title: "Resume", target: nil, action: nil)
    private let stackView = NSStackView()
    private var currentProcessIdentifier: pid_t?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.withAlphaComponent(0.25).cgColor

        cardView.translatesAutoresizingMaskIntoConstraints = false
        cardView.material = .hudWindow
        cardView.state = .active
        cardView.wantsLayer = true
        cardView.layer?.cornerRadius = 16
        addSubview(cardView)

        statusLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        statusLabel.alignment = .center
        statusLabel.lineBreakMode = .byWordWrapping
        statusLabel.maximumNumberOfLines = 0

        autoResumeButton.target = self
        autoResumeButton.action = #selector(handleAutoResumeChanged(_:))

        resumeButton.target = self
        resumeButton.action = #selector(handleResumeTapped(_:))
        resumeButton.bezelStyle = .rounded

        stackView.orientation = .vertical
        stackView.alignment = .centerX
        stackView.spacing = 12
        stackView.translatesAutoresizingMaskIntoConstraints = false
        stackView.addArrangedSubview(statusLabel)
        stackView.addArrangedSubview(autoResumeButton)
        stackView.addArrangedSubview(resumeButton)
        cardView.addSubview(stackView)

        NSLayoutConstraint.activate([
            cardView.centerXAnchor.constraint(equalTo: centerXAnchor),
            cardView.centerYAnchor.constraint(equalTo: centerYAnchor),
            cardView.leadingAnchor.constraint(greaterThanOrEqualTo: leadingAnchor, constant: 40),
            cardView.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -40),
            cardView.widthAnchor.constraint(lessThanOrEqualToConstant: 800)
        ])

        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: cardView.topAnchor, constant: 24),
            stackView.bottomAnchor.constraint(equalTo: cardView.bottomAnchor, constant: -24),
            stackView.leadingAnchor.constraint(equalTo: cardView.leadingAnchor, constant: 28),
            stackView.trailingAnchor.constraint(equalTo: cardView.trailingAnchor, constant: -28)
        ])
    }

    func updateDebuggerState(isAttached: Bool, processIdentifier: pid_t?) {
        currentProcessIdentifier = processIdentifier
        statusLabel.stringValue = statusMessage(isAttached: isAttached)
    }

    private func statusMessage(isAttached: Bool) -> String {
        if let pid = currentProcessIdentifier {
            if isAttached {
                return "Debugger attached. Resume when ready."
            } else {
                return "Waiting for debugger to attach to OuterframeContent (PID \(pid))"
            }
        }
        return isAttached ? "Debugger attached. Resume when ready." : "Waiting for debugger to attach…"
    }

    func setAutoResumeSelected(_ isSelected: Bool) {
        autoResumeButton.state = isSelected ? .on : .off
    }

    @objc private func handleResumeTapped(_ sender: Any?) {
        resumeHandler?()
    }

    @objc private func handleAutoResumeChanged(_ sender: NSButton) {
        autoResumeChangedHandler?(sender.state == .on)
    }
}


private final class OuterframeBackgroundVisualEffectView: NSVisualEffectView {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}


extension OuterframeView: OuterframeContentConnectionDelegate {}

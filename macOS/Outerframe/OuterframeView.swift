import AppKit
import Darwin
import Network
import UniformTypeIdentifiers

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

public struct OuterframeStorageContext {
    public let serverTemporaryDirectoryURL: URL
    public let originIdentifier: String

    public init(serverTemporaryDirectoryURL: URL, originIdentifier: String) {
        self.serverTemporaryDirectoryURL = serverTemporaryDirectoryURL
        self.originIdentifier = originIdentifier
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
        return URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Caches", isDirectory: true)
            .appendingPathComponent(bundleIdentifier, isDirectory: true)
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

private final class PluginContextMenuAction: NSObject {
    let menuID: UUID
    let itemID: String

    init(menuID: UUID, itemID: String) {
        self.menuID = menuID
        self.itemID = itemID
    }
}

private final class PluginContextMenuLabelView: NSView {
    private let label: NSTextField

    init(title: String, style: OuterframeContextMenuItemStyle) {
        label = NSTextField(labelWithString: title)
        let height = CGFloat(style.height > 0 ? style.height : 23)
        super.init(frame: NSRect(x: 0, y: 0, width: 220, height: height))
        wantsLayer = true
        let fontSize = CGFloat(style.fontSize > 0 ? style.fontSize : 11)
        let fontWeight = style.fontWeight != 0 ? NSFont.Weight(rawValue: CGFloat(style.fontWeight)) : .semibold
        label.font = NSFont.systemFont(ofSize: fontSize, weight: fontWeight)
        label.textColor = NSColor(outerframeMenuRGBA: style.textColorRGBA) ?? .secondaryLabelColor
        label.lineBreakMode = .byTruncatingTail
        switch style.alignment {
        case .natural:
            label.alignment = .natural
        case .left:
            label.alignment = .left
        case .center:
            label.alignment = .center
        case .right:
            label.alignment = .right
        }
        let leftInset = CGFloat(style.leftInset)
        let rightInset = CGFloat(style.rightInset)
        let topInset = CGFloat(style.topInset)
        let bottomInset = CGFloat(style.bottomInset)
        label.frame = NSRect(x: leftInset,
                             y: bottomInset,
                             width: max(0, 220 - leftInset - rightInset),
                             height: max(0, height - topInset - bottomInset))
        label.autoresizingMask = [.width]
        addSubview(label)
    }

    required init?(coder: NSCoder) {
        return nil
    }
}

private extension NSColor {
    convenience init?(outerframeMenuRGBA rgba: UInt32) {
        guard rgba != 0 else { return nil }
        let red = CGFloat((rgba >> 24) & 0xff) / 255
        let green = CGFloat((rgba >> 16) & 0xff) / 255
        let blue = CGFloat((rgba >> 8) & 0xff) / 255
        let alpha = CGFloat(rgba & 0xff) / 255
        self.init(red: red, green: green, blue: blue, alpha: alpha)
    }
}

private struct OuterframeFilePromisePayload {
    let id: UUID
    let name: String
    let fileSize: UInt64?
    let fileType: String?
}

private struct OuterframeDroppedFileAccessPayload: Sendable {
    let id: UUID
    let name: String
    let localPath: String
    let fileSize: UInt64?
    let fileType: String?
    let isDirectory: Bool
}

private struct OuterframeDroppedFileAccess: Sendable {
    let id: UUID
    let stagedURL: URL
    let cleanupURL: URL
}

private enum OuterframePasteboardPayload {
    private static let version: UInt32 = 1
    private static let droppedFileAccessMissingSize = UInt64.max
    private static let droppedFileAccessDirectoryFlag: UInt32 = 1 << 0
    private static let filePromiseMissingSize = UInt64.max

    static func decodeFilePromise(_ data: Data) -> OuterframeFilePromisePayload? {
        var cursor = BinaryPayloadCursor(data)
        guard cursor.readUInt32() == version,
              cursor.readUInt32() != nil,
              let id = cursor.readUUID(),
              let encodedFileSize = cursor.readUInt64(),
              let name = cursor.readStringReference(),
              !name.isEmpty,
              let fileType = cursor.readStringReference() else {
            return nil
        }

        return OuterframeFilePromisePayload(id: id,
                                            name: name,
                                            fileSize: encodedFileSize == filePromiseMissingSize ? nil : encodedFileSize,
                                            fileType: fileType.isEmpty ? nil : fileType)
    }

    static func encodeDroppedFileAccess(_ payload: OuterframeDroppedFileAccessPayload) -> Data? {
        var builder = BinaryPayloadBuilder(referenceBaseOffset: 0)
        builder.append(uint32: version)
        builder.append(uint32: payload.isDirectory ? droppedFileAccessDirectoryFlag : 0)
        builder.append(uuid: payload.id)
        builder.append(uint64: payload.fileSize ?? droppedFileAccessMissingSize)
        guard builder.append(stringReference: payload.name),
              builder.append(stringReference: payload.fileType ?? ""),
              builder.append(stringReference: payload.localPath) else {
            return nil
        }
        return builder.finalize()
    }
}

private struct BinaryPayloadBuilder {
    private struct Reference {
        let patchOffset: Int
        let variableOffset: Int
        let length: Int
    }

    private var fixed = Data()
    private var variable = Data()
    private var references: [Reference] = []
    private let referenceBaseOffset: Int

    init(referenceBaseOffset: Int) {
        self.referenceBaseOffset = referenceBaseOffset
    }

    mutating func append(uint32 value: UInt32) {
        fixed.appendLittleEndian(value)
    }

    mutating func append(uint64 value: UInt64) {
        fixed.appendLittleEndian(value)
    }

    mutating func append(uuid value: UUID) {
        fixed.append(uuid: value)
    }

    mutating func append(stringReference string: String) -> Bool {
        guard let data = string.data(using: .utf8),
              data.count <= Int(UInt32.max) else {
            return false
        }
        let patchOffset = fixed.count
        fixed.appendLittleEndian(UInt32(0))
        fixed.appendLittleEndian(UInt32(data.count))
        references.append(Reference(patchOffset: patchOffset,
                                    variableOffset: variable.count,
                                    length: data.count))
        variable.append(data)
        return true
    }

    mutating func finalize() -> Data? {
        guard fixed.count <= Int(UInt32.max),
              variable.count <= Int(UInt32.max),
              variable.count <= Int(UInt32.max) - fixed.count else {
            return nil
        }

        for reference in references {
            let offset = referenceBaseOffset + fixed.count + reference.variableOffset
            guard offset <= Int(UInt32.max),
                  reference.length <= Int(UInt32.max) else {
                return nil
            }
            fixed.replaceLittleEndianUInt32(at: reference.patchOffset, with: UInt32(offset))
            fixed.replaceLittleEndianUInt32(at: reference.patchOffset + 4, with: UInt32(reference.length))
        }

        var payload = Data(capacity: fixed.count + variable.count)
        payload.append(fixed)
        payload.append(variable)
        return payload
    }
}

private struct BinaryPayloadCursor {
    private let data: Data
    private var offset = 0

    init(_ data: Data) {
        self.data = data
    }

    mutating func readUInt32() -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        var value: UInt32 = 0
        for index in 0..<4 {
            value |= UInt32(data[offset + index]) << UInt32(index * 8)
        }
        offset += 4
        return value
    }

    mutating func readUInt64() -> UInt64? {
        guard offset + 8 <= data.count else { return nil }
        var value: UInt64 = 0
        for index in 0..<8 {
            value |= UInt64(data[offset + index]) << UInt64(index * 8)
        }
        offset += 8
        return value
    }

    mutating func readUUID() -> UUID? {
        guard offset + 16 <= data.count else { return nil }
        let bytes = Array(data[offset..<(offset + 16)])
        offset += 16
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }

    mutating func readStringReference() -> String? {
        guard let data = readDataReference() else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private mutating func readDataReference() -> Data? {
        guard let offsetValue = readUInt32(),
              let lengthValue = readUInt32() else {
            return nil
        }

        let start = Int(offsetValue)
        let length = Int(lengthValue)
        guard start <= data.count,
              length <= data.count - start else {
            return nil
        }
        return data.subdata(in: start..<(start + length))
    }
}

private extension Data {
    mutating func appendLittleEndian(_ value: UInt32) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of: &value) { append(contentsOf: $0) }
    }

    mutating func appendLittleEndian(_ value: UInt64) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of: &value) { append(contentsOf: $0) }
    }

    mutating func append(uuid value: UUID) {
        var uuidValue = value.uuid
        Swift.withUnsafeBytes(of: &uuidValue) { append(contentsOf: $0) }
    }

    mutating func replaceLittleEndianUInt32(at offset: Int, with value: UInt32) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of: &value) {
            replaceSubrange(offset..<(offset + 4), with: $0)
        }
    }
}

private func copyPromisedFileContents(from sourceURL: URL, to targetURL: URL) throws {
    let fileManager = FileManager.default
    let sourceValues = try sourceURL.resourceValues(forKeys: [.isDirectoryKey])
    var isDirectory: ObjCBool = false
    if fileManager.fileExists(atPath: targetURL.path, isDirectory: &isDirectory) {
        if sourceValues.isDirectory == true || isDirectory.boolValue {
            throw NSError(domain: NSCocoaErrorDomain,
                          code: NSFileWriteFileExistsError,
                          userInfo: [NSLocalizedDescriptionKey: "Destination already exists: \(targetURL.path)"])
        }
        try fileManager.removeItem(at: targetURL)
    }

    if sourceValues.isDirectory == true {
        try fileManager.copyItem(at: sourceURL, to: targetURL)
        return
    }

    try copyPromisedFileBytes(from: sourceURL, to: targetURL)
}

private func copyPromisedFileBytes(from sourceURL: URL, to targetURL: URL) throws {
    let input = try FileHandle(forReadingFrom: sourceURL)
    defer { try? input.close() }

    guard FileManager.default.createFile(atPath: targetURL.path, contents: nil) else {
        throw NSError(domain: NSCocoaErrorDomain,
                      code: NSFileWriteUnknownError,
                      userInfo: [NSLocalizedDescriptionKey: "Could not create promised file at \(targetURL.path)"])
    }

    let output = try FileHandle(forWritingTo: targetURL)
    defer { try? output.close() }

    do {
        while let chunk = try input.read(upToCount: 1024 * 1024),
              !chunk.isEmpty {
            try output.write(contentsOf: chunk)
        }
    } catch {
        try? FileManager.default.removeItem(at: targetURL)
        throw error
    }
}

private final class OuterframeFilePromiseWriter: NSObject, NSFilePromiseProviderDelegate, @unchecked Sendable {
    let promiseID: UUID
    let fileName: String
    var completion: ((OuterframeFilePromiseWriter) -> Void)?
    private let allowedStagingDirectoryURL: URL?
    private let requestStagedFile: @Sendable (UUID, @escaping @Sendable (Result<OuterframeFilePromiseWriteResult, Error>) -> Void) -> Void
    private let filePromiseQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "org.outerframe.file-promise"
        queue.maxConcurrentOperationCount = 2
        return queue
    }()

    init(promiseID: UUID,
         fileName: String,
         allowedStagingDirectoryURL: URL?,
         requestStagedFile: @escaping @Sendable (UUID, @escaping @Sendable (Result<OuterframeFilePromiseWriteResult, Error>) -> Void) -> Void) {
        self.promiseID = promiseID
        self.fileName = Self.safePromisedFileName(fileName)
        self.allowedStagingDirectoryURL = allowedStagingDirectoryURL
        self.requestStagedFile = requestStagedFile
    }

    nonisolated func filePromiseProvider(_ filePromiseProvider: NSFilePromiseProvider,
                                         fileNameForType fileType: String) -> String {
        return fileName
    }

    nonisolated func operationQueue(for filePromiseProvider: NSFilePromiseProvider) -> OperationQueue {
        filePromiseQueue
    }

    nonisolated func filePromiseProvider(_ filePromiseProvider: NSFilePromiseProvider,
                                         writePromiseTo url: URL,
                                         completionHandler: @escaping @Sendable (Error?) -> Void) {
        let targetURL = targetURL(forPromisedURL: url)

        requestStagedFile(promiseID) { [weak self] result in
            guard let self else { return }
            defer {
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.completion?(self)
                }
            }

            do {
                let writeResult = try result.get()
                try self.copyWriteResult(writeResult, to: targetURL)
                completionHandler(nil)
            } catch {
                NSLog("Outerframe file promise: write failed for '%@' to %@: %@",
                      self.fileName,
                      targetURL.path,
                      String(describing: error))
                completionHandler(error)
            }
        }
    }

    private func copyWriteResult(_ writeResult: OuterframeFilePromiseWriteResult, to targetURL: URL) throws {
        guard writeResult.promiseID == promiseID else {
            throw NSError(domain: NSCocoaErrorDomain,
                          code: NSFileWriteUnknownError,
                          userInfo: [NSLocalizedDescriptionKey: "Content responded with the wrong file promise ID."])
        }
        let stagedURL = try validatedStagedFileURL(for: writeResult)
        try copyPromisedFileContents(from: stagedURL, to: targetURL)
        if writeResult.deleteWhenDone {
            try? FileManager.default.removeItem(at: stagedURL)
        }
    }

    private func targetURL(forPromisedURL url: URL) -> URL {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
           isDirectory.boolValue {
            return url.appendingPathComponent(fileName)
        }

        if url.hasDirectoryPath {
            return url.appendingPathComponent(fileName)
        }

        if let values = try? url.resourceValues(forKeys: [.isDirectoryKey]),
           values.isDirectory == true {
            return url.appendingPathComponent(fileName)
        }

        return url
    }

    private func validatedStagedFileURL(for writeResult: OuterframeFilePromiseWriteResult) throws -> URL {
        let stagedURL = URL(fileURLWithPath: writeResult.localPath)
        guard stagedURL.isFileURL else {
            throw NSError(domain: NSCocoaErrorDomain,
                          code: NSFileReadUnsupportedSchemeError,
                          userInfo: [NSLocalizedDescriptionKey: "Promised file staging URL is not a file URL."])
        }
        if let allowedStagingDirectoryURL {
            let stagedPath = stagedURL.standardizedFileURL.path
            let allowedPath = allowedStagingDirectoryURL.standardizedFileURL.path
            guard stagedPath == allowedPath || stagedPath.hasPrefix(allowedPath + "/") else {
                throw NSError(domain: NSCocoaErrorDomain,
                              code: NSFileReadNoPermissionError,
                              userInfo: [NSLocalizedDescriptionKey: "Promised file was staged outside OUTERFRAME_STAGING_DIR."])
            }
        }
        let values = try stagedURL.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
        guard values.isRegularFile == true || values.isDirectory == true else {
            throw NSError(domain: NSCocoaErrorDomain,
                          code: NSFileReadNoSuchFileError,
                          userInfo: [NSLocalizedDescriptionKey: "Promised staged item is not a regular file or directory."])
        }
        return stagedURL
    }

    private static func safePromisedFileName(_ name: String) -> String {
        var sanitized = name
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: ":", with: "_")
            .replacingOccurrences(of: "\0", with: "")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        sanitized = URL(fileURLWithPath: sanitized).lastPathComponent

        if sanitized.isEmpty || sanitized == "." || sanitized == ".." {
            return "Untitled"
        }
        return sanitized
    }
}

@MainActor
public final class OuterframeView: NSView, NSMenuItemValidation, NSServicesMenuRequestor, NSTextInputClient, NSDraggingSource {
    public weak var delegate: OuterframeViewDelegate?

    private var layerHost: CALayerHost?
    private var trackingArea: NSTrackingArea?
    private var currentSelectedText: String?
    private var currentSelectedAttributedText: NSAttributedString?
    private var rightClickLocation: NSPoint = .zero
    private var backgroundEffectView: NSVisualEffectView?
    private var lastMouseDownEvent: NSEvent?
    private var lastMouseDraggedEvent: NSEvent?
    private var activeFilePromiseWriters: [AnyObject] = []
    private var activePasteboardFilePromiseProviders: [AnyObject] = []
    private var droppedFileAccesses: [UUID: OuterframeDroppedFileAccess] = [:]
    private var currentDraggingOperationMask: NSDragOperation = .copy

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
    private var presentationBundleURL: URL?
    private var presentationStagingDirectoryURL: URL?
    private var debugModeObserver: NSObjectProtocol?
    private var autoResumeObserver: NSObjectProtocol?

    // Track active display link callbacks
    private var activeDisplayLinkCallbacks = [UUID: Bool]() // UUID -> isActive
    private let displayLinkLock = NSLock()

    // Track window visibility state
    private var isWindowVisible = true
    private var shouldRunDisplayLink = false
    private var windowIsActive = false

    var currentInputMode: OuterframeContentInputMode = .rawKeys

    public nonisolated static let outerframeMimeType = "application/vnd.outerframe"
    public nonisolated static let filePromisePasteboardTypeIdentifier = "org.outerframe.file-promise"
    public nonisolated static let droppedFileAccessPasteboardTypeIdentifier = "org.outerframe.dropped-file-access"
    public nonisolated static let topLevelNavigationCapabilityHeaderField = "Outerframe-Accept"
    public nonisolated static let topLevelNavigationCapabilityHeaderValue = OuterframeView.outerframeMimeType

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
            NSLog("OuterframeView: OuterframeProcesses XPC connection interrupted")
        }
        connection.invalidationHandler = {
            NSLog("OuterframeView: OuterframeProcesses XPC connection invalidated")
            outerframeProcessesConnectionLock.lock()
            if outerframeProcessesConnectionStorage === connection {
                outerframeProcessesConnectionStorage = nil
            }
            outerframeProcessesConnectionLock.unlock()
        }
        connection.resume()
        return connection
    }

    // Text input state
    private var currentText: String = ""
    private var textSelectedRange: NSRange = NSRange(location: 0, length: 0)
    private var markedText: NSAttributedString?
    private var textMarkedRange: NSRange = NSRange(location: NSNotFound, length: 0)
    private var currentTextInputGeometry: OuterframeContentTextInputGeometry?
    private var overlayScrollIndicators: [String: NSScroller] = [:]

    private let manualMagnificationSurfaceId = Int(UInt32.max)

    private static let accessibilitySnapshotTimeoutMilliseconds: UInt32 = 250
    private var accessibilityElements: [OuterframeAccessibilityElement] = []

    private var acceptedPasteboardPasteTypes: [NSPasteboard.PasteboardType] = []
    private var acceptedPasteboardDropTypes: [NSPasteboard.PasteboardType] = []
    private var pasteboardDropHitTestEnabled = false
    private var editCommandValidationSnapshot: EditCommandValidationSnapshot?
    private static let editCommandValidationTimeoutMilliseconds: UInt32 = 25
    private static let pasteboardDropHitTestTimeoutMilliseconds: UInt32 = 5

    private struct EditCommandValidationSnapshot {
        let enabledCommands: OuterframeEditCommandSet?
    }

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
        registerForDraggedTypes([
            .fileURL,
            .URL,
            .string,
            NSPasteboard.PasteboardType(OuterframeView.filePromisePasteboardTypeIdentifier),
            NSPasteboard.PasteboardType(OuterframeView.droppedFileAccessPasteboardTypeIdentifier)
        ])
        updateTrackingAreas()

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

    // MARK: - Text Input Geometry

    func updateTextInputGeometry(_ geometry: OuterframeContentTextInputGeometry?) {
        currentTextInputGeometry = geometry
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

            var viewRect: NSRect
            if let geometry = currentTextInputGeometry {
                let rect = geometry.rect
                viewRect = NSRect(x: rect.origin.x,
                                  y: bounds.height - rect.origin.y - rect.size.height,
                                  width: rect.size.width,
                                  height: rect.size.height)
            } else {
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

    public func setPasteboardDropBehaviorUniform(_ pasteboardTypeIdentifiers: [String]) {
        acceptedPasteboardDropTypes = pasteboardTypeIdentifiers
            .filter { !$0.isEmpty }
            .map { NSPasteboard.PasteboardType($0) }
        pasteboardDropHitTestEnabled = false
    }

    public func setAcceptedPasteboardPasteTypes(_ pasteboardTypeIdentifiers: [String]) {
        acceptedPasteboardPasteTypes = pasteboardTypeIdentifiers
            .filter { !$0.isEmpty }
            .map { NSPasteboard.PasteboardType($0) }
    }

    public func setPasteboardDropBehaviorHitTest() {
        pasteboardDropHitTestEnabled = true
    }

    func resetPasteboardInteractionState() {
        acceptedPasteboardPasteTypes = []
        acceptedPasteboardDropTypes = []
        pasteboardDropHitTestEnabled = false
        editCommandValidationSnapshot = nil
    }

    private func currentEditCommandValidationSnapshot() -> EditCommandValidationSnapshot {
        if let snapshot = editCommandValidationSnapshot {
            return snapshot
        }

        let enabledCommands = outerframeContentConnection?.validateEditCommandsSynchronously(
            .standard,
            timeoutMilliseconds: Self.editCommandValidationTimeoutMilliseconds
        )
        let snapshot = EditCommandValidationSnapshot(enabledCommands: enabledCommands)
        editCommandValidationSnapshot = snapshot

        DispatchQueue.main.async { [weak self] in
            self?.editCommandValidationSnapshot = nil
        }

        return snapshot
    }

    private func contentValidation(for command: OuterframeEditCommandSet) -> Bool? {
        currentEditCommandValidationSnapshot().enabledCommands?.contains(command)
    }

    private func writePasteboardItems(_ items: [OuterframeContentPasteboardItem]) -> Bool {
        guard !items.isEmpty else { return false }

        var nativeItems: [NSPasteboardWriting] = []
        var filePromiseProviders: [AnyObject] = []
        nativeItems.reserveCapacity(items.count)
        for item in items {
            if let fileURL = fileURLPasteboardWriter(for: item) {
                nativeItems.append(fileURL)
                continue
            }

            if let (payload, writer) = filePromiseWriter(for: item) {
                let providerType = filePromiseProviderType(payload.name, fileType: payload.fileType)
                let provider = NSFilePromiseProvider(fileType: providerType, delegate: writer)
                provider.userInfo = writer
                writer.completion = { [weak self, weak provider] _ in
                    guard let provider else { return }
                    self?.activePasteboardFilePromiseProviders.removeAll { $0 === provider }
                }
                nativeItems.append(provider)
                filePromiseProviders.append(provider)
                continue
            }

            let nativeItem = NSPasteboardItem()
            var wroteRepresentation = false
            for representation in item.representations where !representation.typeIdentifier.isEmpty {
                let type = NSPasteboard.PasteboardType(representation.typeIdentifier)
                if nativeItem.setData(representation.data, forType: type) {
                    wroteRepresentation = true
                }
            }
            if wroteRepresentation {
                nativeItems.append(nativeItem)
            }
        }

        guard !nativeItems.isEmpty else { return false }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        activePasteboardFilePromiseProviders.removeAll()
        let wrote = pasteboard.writeObjects(nativeItems)
        if wrote {
            activePasteboardFilePromiseProviders.append(contentsOf: filePromiseProviders)
        }
        return wrote
    }

    private func fileURLPasteboardWriter(for item: OuterframeContentPasteboardItem) -> NSURL? {
        guard let representation = item.representations.first(where: {
            $0.typeIdentifier == NSPasteboard.PasteboardType.fileURL.rawValue
        }),
              let urlString = String(data: representation.data, encoding: .utf8),
              let url = URL(string: urlString),
              url.isFileURL else {
            return nil
        }
        return url as NSURL
    }

    private func pasteboardItems(from pasteboard: NSPasteboard,
                                 acceptedTypes: [NSPasteboard.PasteboardType]) -> [OuterframeContentPasteboardItem] {
        guard let pasteboardItems = pasteboard.pasteboardItems else { return [] }
        return pasteboardItems.compactMap { nativeItem in
            let candidateTypes = acceptedTypes.isEmpty
                ? nativeItem.types
                : nativeItem.types.filter { acceptedTypes.contains($0) }
            let representations = candidateTypes.compactMap { type -> OuterframeContentPasteboardRepresentation? in
                guard let data = nativeItem.data(forType: type) else { return nil }
                return OuterframeContentPasteboardRepresentation(typeIdentifier: type.rawValue, data: data)
            }
            guard !representations.isEmpty else { return nil }
            return OuterframeContentPasteboardItem(representations: representations)
        }
    }

    private func completePasteboardAccessRequest(requestID: UUID,
                                                 operation: OuterframePasteboardAccessOperation,
                                                 pasteboardTypeIdentifiers: [String],
                                                 items: [OuterframeContentPasteboardItem],
                                                 allowed: Bool) {
        guard allowed else {
            outerframeContentConnection?.sendPasteboardAccessResponse(requestID: requestID,
                                                                      granted: false,
                                                                      items: [])
            return
        }

        switch operation {
        case .read:
            let pasteboardTypes = pasteboardTypeIdentifiers.filter { !$0.isEmpty }.map { NSPasteboard.PasteboardType($0) }
            let responseItems = pasteboardItems(from: NSPasteboard.general, acceptedTypes: pasteboardTypes)
            outerframeContentConnection?.sendPasteboardAccessResponse(requestID: requestID,
                                                                      granted: !responseItems.isEmpty,
                                                                      items: responseItems)
        case .write:
            let wrote = writePasteboardItems(items)
            outerframeContentConnection?.sendPasteboardAccessResponse(requestID: requestID,
                                                                      granted: wrote,
                                                                      items: [])
        }
    }

    public func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
        switch menuItem.action {
        case #selector(copy(_:)):
            return contentValidation(for: .copy) == true

        case #selector(cut(_:)):
            return contentValidation(for: .cut) == true

        case #selector(paste(_:)) where !acceptedPasteboardPasteTypes.isEmpty:
            guard contentValidation(for: .paste) == true else {
                return false
            }
            return pasteboardMatchesAcceptedTypes(NSPasteboard.general, acceptedTypes: acceptedPasteboardPasteTypes)

        case #selector(selectAll(_:)):
            return contentValidation(for: .selectAll) == true
        case #selector(performFindPanelAction(_:)):
            return validateFindMenuAction(menuItem)
        case #selector(paste(_:)):
            return contentValidation(for: .paste) == true && NSPasteboard.general.string(forType: .string) != nil
        case #selector(lookUp(_:)):
            return currentSelectedAttributedText != nil
        default:
            return true
        }
    }

    private func validateFindMenuAction(_ menuItem: NSMenuItem) -> Bool {
        switch NSFindPanelAction(rawValue: UInt(menuItem.tag)) {
        case .showFindPanel:
            return contentValidation(for: .find) == true
        case .next:
            return contentValidation(for: .findNext) == true
        case .previous:
            return contentValidation(for: .findPrevious) == true
        case .setFindString:
            return contentValidation(for: .copy) == true
        default:
            return false
        }
    }

    @objc func copy(_ sender: Any?) {
        guard contentValidation(for: .copy) == true else {
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
            }
        }
    }

    @objc func cut(_ sender: Any?) {
        guard contentValidation(for: .cut) == true else {
            NSSound.beep()
            return
        }

        requestPasteboardItemsForCut { [weak self] items in
            Task { @MainActor in
                guard let self else { return }
                guard let items, self.writePasteboardItems(items) else {
                    NSSound.beep()
                    return
                }
            }
        }
    }

    public override func selectAll(_ sender: Any?) {
        guard contentValidation(for: .selectAll) == true else {
            NSSound.beep()
            return
        }

        sendTextCommand("selectAll")
    }

    @objc func paste(_ sender: Any?) {
        let pasteboard = NSPasteboard.general

        guard contentValidation(for: .paste) == true else {
            NSSound.beep()
            return
        }

        if !acceptedPasteboardPasteTypes.isEmpty {
            if acceptsFileAccess(acceptedTypes: acceptedPasteboardPasteTypes),
               let fileURLs = pasteboard.readObjects(forClasses: [NSURL.self],
                                                     options: [.urlReadingFileURLsOnly: true]) as? [URL],
               !fileURLs.isEmpty {
                pasteFileURLsFromPasteboard(fileURLs)
                return
            }

            guard pasteboardMatchesAcceptedTypes(pasteboard, acceptedTypes: acceptedPasteboardPasteTypes) else {
                NSSound.beep()
                return
            }
            let payload = pasteboardItems(from: pasteboard,
                                          acceptedTypes: acceptedPasteboardPasteTypes)
            guard !payload.isEmpty else {
                NSSound.beep()
                return
            }
            sendPasteboardItemsForPaste(payload)
            return
        }

        if currentInputMode.allowsTextInput {
            guard let text = pasteboard.string(forType: .string) else { return }
            sendTextInput(text, replacementRange: NSRange(location: NSNotFound, length: 0))
            return
        }

        NSSound.beep()
    }

    @objc public func performFindPanelAction(_ sender: Any?) {
        guard let action = findPanelAction(from: sender) else {
            sendTextCommand("find")
            return
        }

        switch action {
        case .showFindPanel:
            sendTextCommand("find")
        case .next:
            sendTextCommand("findNext")
        case .previous:
            sendTextCommand("findPrevious")
        case .setFindString:
            sendTextCommand("useSelectionForFind")
        default:
            NSSound.beep()
        }
    }

    private func findPanelAction(from sender: Any?) -> NSFindPanelAction? {
        if let menuItem = sender as? NSMenuItem {
            return NSFindPanelAction(rawValue: UInt(menuItem.tag))
        }

        if let number = sender as? NSNumber {
            return NSFindPanelAction(rawValue: number.uintValue)
        }

        if let intValue = sender as? Int {
            return NSFindPanelAction(rawValue: UInt(intValue))
        }

        return nil
    }

    private func pasteFileURLsFromPasteboard(_ fileURLs: [URL]) {
        guard let stagingDirectoryURL = outerframeContentConnection?.currentStagedFileDirectoryURL else {
            NSSound.beep()
            return
        }
        let connection = outerframeContentConnection

        Task.detached(priority: .userInitiated) { [weak self] in
            var accesses: [OuterframeDroppedFileAccess] = []
            var items: [OuterframeContentPasteboardItem] = []

            for url in fileURLs {
                guard url.isFileURL else { continue }
                guard let resourceValues = try? url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .fileSizeKey, .contentTypeKey]),
                      resourceValues.isRegularFile == true || resourceValues.isDirectory == true else {
                    continue
                }

                let accessID = UUID()
                let fileName = Self.sanitizedStagedFileName(url.lastPathComponent)
                let accessDirectory = stagingDirectoryURL
                    .appendingPathComponent("dropped-file-access", isDirectory: true)
                    .appendingPathComponent(accessID.uuidString, isDirectory: true)
                let stagedURL = accessDirectory.appendingPathComponent(fileName,
                                                                       isDirectory: resourceValues.isDirectory == true)

                do {
                    try FileManager.default.createDirectory(at: accessDirectory, withIntermediateDirectories: true)
                    if FileManager.default.fileExists(atPath: stagedURL.path) {
                        try FileManager.default.removeItem(at: stagedURL)
                    }
                    try Self.stageDroppedFile(from: url, to: stagedURL, isDirectory: resourceValues.isDirectory == true)
                } catch {
                    try? FileManager.default.removeItem(at: accessDirectory)
                    print("Outerframe pasted file access: failed to stage '\(url.path)': \(error)")
                    continue
                }

                let stagedValues = try? stagedURL.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey, .contentTypeKey])
                let isDirectory = stagedValues?.isDirectory == true
                let payload = OuterframeDroppedFileAccessPayload(
                    id: accessID,
                    name: fileName,
                    localPath: stagedURL.path,
                    fileSize: isDirectory ? nil : stagedValues?.fileSize.map { UInt64(max($0, 0)) },
                    fileType: stagedValues?.contentType?.identifier ?? resourceValues.contentType?.identifier,
                    isDirectory: isDirectory
                )
                guard let data = OuterframePasteboardPayload.encodeDroppedFileAccess(payload) else {
                    try? FileManager.default.removeItem(at: accessDirectory)
                    continue
                }

                accesses.append(OuterframeDroppedFileAccess(id: accessID,
                                                            stagedURL: stagedURL,
                                                            cleanupURL: accessDirectory))
                items.append(OuterframeContentPasteboardItem(representations: [
                    OuterframeContentPasteboardRepresentation(typeIdentifier: Self.droppedFileAccessPasteboardTypeIdentifier,
                                                              data: data)
                ]))
            }

            await MainActor.run {
                guard let self, !items.isEmpty else {
                    NSSound.beep()
                    return
                }
                for access in accesses {
                    self.droppedFileAccesses[access.id] = access
                }
                connection?.sendPasteboardItemsForPaste(items)
            }
        }
    }

    @objc func lookUp(_ sender: Any?) {
        // This method should only be called from plugin-requested menus
        // where we've already stored the necessary information
        guard let attributedText = currentSelectedAttributedText else { return }

        // Use the stored right-click location for the popover
        self.showDefinition(for: attributedText, at: rightClickLocation)
    }

    @objc func pluginContextMenuItemSelected(_ sender: NSMenuItem) {
        guard let action = sender.representedObject as? PluginContextMenuAction else { return }
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.contextMenuItemSelected(menuID: action.menuID,
                                                                        itemID: action.itemID))
        }
    }

    @objc func copyPluginContextMenuSelection(_ sender: NSMenuItem) {
        requestPasteboardItemsForCopy { [weak self] items in
            Task { @MainActor in
                guard let self else { return }
                if let items, self.writePasteboardItems(items) {
                    return
                }
                NSSound.beep()
            }
        }
    }

    public override func mouseDown(with event: NSEvent) {
        lastMouseDownEvent = event
        lastMouseDraggedEvent = nil
        let point = convert(event.locationInWindow, from: nil)
        handleMouseDown(at: point, modifierFlags: event.modifierFlags, clickCount: event.clickCount)
        super.mouseDown(with: event)
    }

    public override func mouseDragged(with event: NSEvent) {
        lastMouseDraggedEvent = event
        let point = convert(event.locationInWindow, from: nil)
        handleMouseDragged(to: point, modifierFlags: event.modifierFlags)
    }

    public override func mouseUp(with event: NSEvent) {
        lastMouseDraggedEvent = nil
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
                #selector(cancelOperation(_:)): "cancelOperation",
                NSSelectorFromString("saveDocument:"): "saveDocument"
            ]

            if selector == NSSelectorFromString("noop:") {
                return
            }

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
        let selectedText = attributedText.string
        var items: [OuterframeContextMenuItem] = []
        if !selectedText.isEmpty {
            items.append(OuterframeContextMenuItem(id: "lookup",
                                                   title: "Look Up \"\(selectedText)\"",
                                                   action: .standardLookUp))
            items.append(OuterframeContextMenuItem(id: "lookup-separator",
                                                   title: "",
                                                   kind: .separator,
                                                   isEnabled: false))
        }
        items.append(OuterframeContextMenuItem(id: "cut", title: "Cut", action: .standardCut))
        items.append(OuterframeContextMenuItem(id: "copy", title: "Copy", action: .standardCopy))
        items.append(OuterframeContextMenuItem(id: "paste", title: "Paste", action: .standardPaste))
        items.append(OuterframeContextMenuItem(id: "select-all", title: "Select All", action: .standardSelectAll))
        if !selectedText.isEmpty {
            items.append(OuterframeContextMenuItem(id: "services-separator",
                                                   title: "",
                                                   kind: .separator,
                                                   isEnabled: false))
            items.append(OuterframeContextMenuItem(id: "services", title: "Services", action: .standardServices))
        }
        showPluginRequestedContextMenuItems(menuID: UUID(),
                                            attributedText: attributedText,
                                            items: items,
                                            at: location)
    }

    public func showPluginRequestedContextMenuItems(menuID: UUID,
                                                    attributedText: NSAttributedString?,
                                                    items: [OuterframeContextMenuItem],
                                                    at location: CGPoint) {
        guard !items.isEmpty else { return }

        let selectedText = attributedText?.string.isEmpty == false ? attributedText?.string : nil
        currentSelectedAttributedText = selectedText == nil ? nil : attributedText
        currentSelectedText = selectedText
        rightClickLocation = location

        let menu = NSMenu()
        menu.autoenablesItems = false
        func append(_ snapshots: [OuterframeContextMenuItem], to menu: NSMenu) {
            for item in snapshots {
                switch item.kind {
                case .separator:
                    menu.addItem(NSMenuItem.separator())

                case .label:
                    let menuItem = NSMenuItem(title: item.title, action: nil, keyEquivalent: "")
                    menuItem.isEnabled = false
                    menuItem.view = PluginContextMenuLabelView(title: item.title, style: item.style)
                    menu.addItem(menuItem)

                case .submenu:
                    let menuItem = NSMenuItem(title: item.title, action: nil, keyEquivalent: "")
                    configurePluginContextMenuItem(menuItem, from: item, menuID: menuID)
                    let submenu = NSMenu()
                    append(item.children, to: submenu)
                    menuItem.submenu = submenu
                    menu.addItem(menuItem)

                case .command:
                    if item.action == .standardServices {
                        let servicesItem = NSMenuItem(title: item.title, action: nil, keyEquivalent: "")
                        configurePluginContextMenuItem(servicesItem, from: item, menuID: menuID)
                        let servicesMenu = NSMenu()
                        servicesItem.submenu = servicesMenu
                        servicesItem.isEnabled = item.isEnabled && selectedText != nil
                        menu.addItem(servicesItem)
                        NSApp.servicesMenu = servicesMenu
                        continue
                    }

                    let selector: Selector
                    let fallbackKeyEquivalent: String
                    switch item.action {
                    case .contentCommand:
                        selector = #selector(pluginContextMenuItemSelected(_:))
                        fallbackKeyEquivalent = ""
                    case .standardCopy:
                        selector = #selector(copy(_:))
                        fallbackKeyEquivalent = "c"
                    case .standardPaste:
                        selector = #selector(paste(_:))
                        fallbackKeyEquivalent = "v"
                    case .standardCut:
                        selector = #selector(cut(_:))
                        fallbackKeyEquivalent = "x"
                    case .standardSelectAll:
                        selector = #selector(selectAll(_:))
                        fallbackKeyEquivalent = "a"
                    case .standardLookUp:
                        selector = #selector(lookUp(_:))
                        fallbackKeyEquivalent = ""
                    case .standardServices:
                        continue
                    }
                    let keyEquivalent = item.keyEquivalent.isEmpty ? fallbackKeyEquivalent : item.keyEquivalent
                    let menuItem = NSMenuItem(title: item.title,
                                              action: selector,
                                              keyEquivalent: keyEquivalent)
                    menuItem.target = self
                    configurePluginContextMenuItem(menuItem, from: item, menuID: menuID)
                    menu.addItem(menuItem)
                }
            }
        }
        append(items, to: menu)

        menu.allowsContextMenuPlugIns = true
        menu.popUp(positioning: nil, at: location, in: self)
    }

    private func configurePluginContextMenuItem(_ menuItem: NSMenuItem,
                                                from snapshot: OuterframeContextMenuItem,
                                                menuID: UUID) {
        menuItem.isEnabled = snapshot.isEnabled
        menuItem.representedObject = PluginContextMenuAction(menuID: menuID, itemID: snapshot.id)
        menuItem.indentationLevel = Int(snapshot.indentationLevel)
        switch snapshot.state {
        case .off:
            menuItem.state = .off
        case .on:
            menuItem.state = .on
        case .mixed:
            menuItem.state = .mixed
        }
        if snapshot.keyEquivalentModifierMask != 0 {
            menuItem.keyEquivalentModifierMask = NSEvent.ModifierFlags(rawValue: UInt(snapshot.keyEquivalentModifierMask))
        }
        if !snapshot.systemImageName.isEmpty,
           let image = NSImage(systemSymbolName: snapshot.systemImageName, accessibilityDescription: snapshot.title) {
            menuItem.image = image
        }
        if let attributedTitle = attributedTitle(for: snapshot.title, style: snapshot.style) {
            menuItem.attributedTitle = attributedTitle
        }
    }

    private func attributedTitle(for title: String,
                                 style: OuterframeContextMenuItemStyle) -> NSAttributedString? {
        var attributes: [NSAttributedString.Key: Any] = [:]
        if style.fontSize > 0 || style.fontWeight != 0 {
            let fontSize = CGFloat(style.fontSize > 0 ? style.fontSize : Float32(NSFont.systemFontSize))
            let fontWeight = style.fontWeight != 0 ? NSFont.Weight(rawValue: CGFloat(style.fontWeight)) : .regular
            attributes[.font] = NSFont.systemFont(ofSize: fontSize, weight: fontWeight)
        }
        if let color = NSColor(outerframeMenuRGBA: style.textColorRGBA) {
            attributes[.foregroundColor] = color
        }
        guard !attributes.isEmpty else { return nil }
        return NSAttributedString(string: title, attributes: attributes)
    }

    public func showPluginRequestedDefinition(attributedText: NSAttributedString, at location: CGPoint) {
        // Use the location directly - plugin should send correct coordinates
        self.showDefinition(for: attributedText, at: location)
    }

    func updateWithContextID(_ contextID: CAContextID) {
        print("CanvasView: updateWithContextID called with contextID: \(contextID)")

        // Ensure we have a layer
        resetAccessibilitySnapshot()

        if layer == nil {
            wantsLayer = true
        }

        layerHost?.removeFromSuperlayer()


        let layerHost2 = CALayerHost()
        layerHost = layerHost2

        layerHost2.contextId = contextID
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
            connection.resizeContent(size: size)
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

    // MARK: - Drag and Drop

    private func draggingOperation(from rawMask: UInt32) -> NSDragOperation {
        var operation: NSDragOperation = []
        if rawMask & UInt32(NSDragOperation.copy.rawValue) != 0 {
            operation.insert(.copy)
        }
        if rawMask & UInt32(NSDragOperation.move.rawValue) != 0 {
            operation.insert(.move)
        }
        if rawMask & UInt32(NSDragOperation.link.rawValue) != 0 {
            operation.insert(.link)
        }
        return operation.isEmpty ? .copy : operation
    }

    private func dragPreview(for item: OuterframeContentDraggingItem,
                             fallbackImage: NSImage,
                             fallbackSize: NSSize,
                             dragStartPoint: NSPoint) -> (NSRect, NSImage) {
        if let previewImageData = item.previewImageData,
           let previewImage = NSImage(data: previewImageData) {
            let size = item.previewSize ?? previewImage.size
            let frame: NSRect
            if let previewFrameOrigin = item.previewFrameOrigin {
                frame = NSRect(origin: previewFrameOrigin, size: size)
            } else {
                frame = dragFrame(size: size, around: dragStartPoint)
            }
            return (frame, previewImage)
        }
        return (dragFrame(size: fallbackSize, around: dragStartPoint), fallbackImage)
    }

    private func dragFrame(size: NSSize, around point: NSPoint) -> NSRect {
        NSRect(x: point.x + 14,
               y: point.y - size.height - 10,
               width: size.width,
               height: size.height)
    }

    private func fallbackFileIcon(fileName: String, fileType: String?) -> NSImage {
        if let fileType,
           let contentType = UTType(fileType) {
            return NSWorkspace.shared.icon(for: contentType)
        }

        let fileExtension = URL(fileURLWithPath: fileName).pathExtension
        if !fileExtension.isEmpty,
           let contentType = UTType(filenameExtension: fileExtension) {
            return NSWorkspace.shared.icon(for: contentType)
        }

        return NSWorkspace.shared.icon(for: UTType.data)
    }

    private func filePromiseProviderType(_ fileName: String, fileType: String?) -> String {
        if let fileType,
           let contentType = UTType(fileType),
           contentType.conforms(to: .data) || contentType.conforms(to: .directory) {
            return contentType.identifier
        }

        let fileExtension = URL(fileURLWithPath: fileName).pathExtension
        if !fileExtension.isEmpty,
           let contentType = UTType(filenameExtension: fileExtension),
           contentType.conforms(to: .data) || contentType.conforms(to: .directory) {
            return contentType.identifier
        }

        return UTType.data.identifier
    }

    private func filePromiseWriter(for item: OuterframeContentPasteboardItem) -> (OuterframeFilePromisePayload, OuterframeFilePromiseWriter)? {
        guard let filePromiseRepresentation = item.representations.first(where: {
            $0.typeIdentifier == Self.filePromisePasteboardTypeIdentifier
        }),
              let payload = OuterframePasteboardPayload.decodeFilePromise(filePromiseRepresentation.data) else {
            return nil
        }

        let connection = outerframeContentConnection
        let writer = OuterframeFilePromiseWriter(
            promiseID: payload.id,
            fileName: payload.name,
            allowedStagingDirectoryURL: outerframeContentConnection?.currentStagedFileDirectoryURL,
            requestStagedFile: { promiseID, completion in
                let request: @Sendable () -> Void = {
                    guard let connection else {
                        completion(.failure(OuterframeContentConnectionError.disconnected))
                        return
                    }
                    MainActor.assumeIsolated {
                        connection.requestFilePromiseWrite(promiseID: promiseID, completion: completion)
                    }
                }
                if Thread.isMainThread {
                    request()
                } else {
                    DispatchQueue.main.async(execute: request)
                }
            }
        )
        return (payload, writer)
    }

    private func filePromiseWriter(for item: OuterframeContentDraggingItem) -> (OuterframeFilePromisePayload, OuterframeFilePromiseWriter)? {
        filePromiseWriter(for: item.pasteboardItem)
    }

    private func makeDragItem(for item: OuterframeContentDraggingItem,
                              dragStartPoint: NSPoint) -> (NSDraggingItem, AnyObject?)? {
        if let (payload, writer) = filePromiseWriter(for: item) {
            let providerType = filePromiseProviderType(payload.name, fileType: payload.fileType)
            let provider = NSFilePromiseProvider(fileType: providerType, delegate: writer)
            provider.userInfo = writer
            writer.completion = { [weak self, weak provider] _ in
                guard let provider else { return }
                self?.activeFilePromiseWriters.removeAll { $0 === provider }
            }
            let draggingItem = NSDraggingItem(pasteboardWriter: provider)
            let (frame, contents) = dragPreview(
                for: item,
                fallbackImage: fallbackFileIcon(fileName: payload.name, fileType: payload.fileType),
                fallbackSize: NSSize(width: 96, height: 96),
                dragStartPoint: dragStartPoint
            )
            draggingItem.setDraggingFrame(frame, contents: contents)
            return (draggingItem, provider)
        }

        let pasteboardItem = NSPasteboardItem()
        var wroteRepresentation = false
        for representation in item.pasteboardItem.representations where !representation.typeIdentifier.isEmpty {
            let pasteboardType = NSPasteboard.PasteboardType(representation.typeIdentifier)
            if pasteboardItem.setData(representation.data, forType: pasteboardType) {
                wroteRepresentation = true
            }
        }
        guard wroteRepresentation else { return nil }
        let draggingItem = NSDraggingItem(pasteboardWriter: pasteboardItem)
        let (frame, contents) = dragPreview(
            for: item,
            fallbackImage: NSWorkspace.shared.icon(for: UTType.data),
            fallbackSize: NSSize(width: 96, height: 48),
            dragStartPoint: dragStartPoint
        )
        draggingItem.setDraggingFrame(frame, contents: contents)
        return (draggingItem, nil)
    }

    func beginDraggingPasteboardItems(_ items: [OuterframeContentDraggingItem], operationMask: UInt32) {
        guard let event = lastMouseDraggedEvent ?? lastMouseDownEvent ?? NSApp.currentEvent else { return }
        let dragStartPoint = convert(event.locationInWindow, from: nil)

        var dragItems: [NSDraggingItem] = []
        var filePromiseWriters: [AnyObject] = []

        for item in items {
            guard let (draggingItem, filePromiseWriter) = makeDragItem(for: item, dragStartPoint: dragStartPoint) else { continue }
            dragItems.append(draggingItem)
            if let filePromiseWriter {
                filePromiseWriters.append(filePromiseWriter)
            }
        }

        guard !dragItems.isEmpty else { return }
        currentDraggingOperationMask = draggingOperation(from: operationMask)
        activeFilePromiseWriters.append(contentsOf: filePromiseWriters)
        beginDraggingSession(with: dragItems, event: event, source: self)
    }

    public func draggingSession(_ session: NSDraggingSession,
                                sourceOperationMaskFor context: NSDraggingContext) -> NSDragOperation {
        currentDraggingOperationMask
    }

    public func draggingSession(_ session: NSDraggingSession, endedAt screenPoint: NSPoint, operation: NSDragOperation) {
        currentDraggingOperationMask = .copy
    }

    public func ignoreModifierKeys(for session: NSDraggingSession) -> Bool {
        false
    }

    public override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        acceptableDropOperation(for: sender)
    }

    public override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        acceptableDropOperation(for: sender)
    }

    public override func draggingExited(_ sender: NSDraggingInfo?) {
        super.draggingExited(sender)
    }

    public override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard acceptableDropOperation(for: sender) != [] else { return false }
        let pasteboard = sender.draggingPasteboard
        let point = convert(sender.draggingLocation, from: nil)

        if let fileURLs = pasteboard.readObjects(forClasses: [NSURL.self],
                                                 options: [.urlReadingFileURLsOnly: true]) as? [URL],
           !fileURLs.isEmpty,
           acceptsDroppedFilesForAccess() {
            guard let stagingDirectoryURL = outerframeContentConnection?.currentStagedFileDirectoryURL else {
                return false
            }
            let connection = outerframeContentConnection
            Task.detached(priority: .userInitiated) { [weak self] in
                var accesses: [OuterframeDroppedFileAccess] = []
                var items: [OuterframeContentPasteboardItem] = []

                for url in fileURLs {
                    guard url.isFileURL else { continue }
                    guard let resourceValues = try? url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .fileSizeKey, .contentTypeKey]),
                          resourceValues.isRegularFile == true || resourceValues.isDirectory == true else {
                        continue
                    }

                    let accessID = UUID()
                    let fileName = Self.sanitizedStagedFileName(url.lastPathComponent)
                    let accessDirectory = stagingDirectoryURL
                        .appendingPathComponent("dropped-file-access", isDirectory: true)
                        .appendingPathComponent(accessID.uuidString, isDirectory: true)
                    let stagedURL = accessDirectory.appendingPathComponent(fileName,
                                                                           isDirectory: resourceValues.isDirectory == true)

                    do {
                        try FileManager.default.createDirectory(at: accessDirectory, withIntermediateDirectories: true)
                        if FileManager.default.fileExists(atPath: stagedURL.path) {
                            try FileManager.default.removeItem(at: stagedURL)
                        }
                        try Self.stageDroppedFile(from: url, to: stagedURL, isDirectory: resourceValues.isDirectory == true)
                    } catch {
                        try? FileManager.default.removeItem(at: accessDirectory)
                        print("Outerframe dropped file access: failed to stage '\(url.path)': \(error)")
                        continue
                    }

                    let stagedValues = try? stagedURL.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey, .contentTypeKey])
                    let isDirectory = stagedValues?.isDirectory == true
                    let payload = OuterframeDroppedFileAccessPayload(
                        id: accessID,
                        name: fileName,
                        localPath: stagedURL.path,
                        fileSize: isDirectory ? nil : stagedValues?.fileSize.map { UInt64(max($0, 0)) },
                        fileType: stagedValues?.contentType?.identifier ?? resourceValues.contentType?.identifier,
                        isDirectory: isDirectory
                    )
                    guard let data = OuterframePasteboardPayload.encodeDroppedFileAccess(payload) else {
                        try? FileManager.default.removeItem(at: accessDirectory)
                        continue
                    }

                    accesses.append(OuterframeDroppedFileAccess(id: accessID,
                                                                stagedURL: stagedURL,
                                                                cleanupURL: accessDirectory))
                    items.append(OuterframeContentPasteboardItem(representations: [
                        OuterframeContentPasteboardRepresentation(typeIdentifier: Self.droppedFileAccessPasteboardTypeIdentifier,
                                                                  data: data)
                    ]))
                }

                await MainActor.run {
                    guard let self, !items.isEmpty else { return }
                    for access in accesses {
                        self.droppedFileAccesses[access.id] = access
                    }
                    connection?.sendPasteboardItemsForDrop(items, at: point)
                }
            }
            return true
        }

        let items = pasteboardItemsForDrop(from: pasteboard)
        guard !items.isEmpty else { return false }
        withActivePluginConnection { connection in
            connection.sendPasteboardItemsForDrop(items, at: point)
        }
        return true
    }

    private func acceptableDropOperation(for sender: NSDraggingInfo) -> NSDragOperation {
        guard !acceptedPasteboardDropTypes.isEmpty else { return [] }
        let pasteboard = sender.draggingPasteboard

        if !pasteboardDropHitTestEnabled {
            guard pasteboardMatchesAcceptedTypes(pasteboard, acceptedTypes: acceptedPasteboardDropTypes) else { return [] }
            return sender.draggingSourceOperationMask.contains(.copy) ? .copy : []
        }

        guard let outerframeContentConnection else { return [] }
        let requestedOperationMask = UInt32(truncatingIfNeeded: sender.draggingSourceOperationMask.rawValue)
        let responseMask = outerframeContentConnection.validatePasteboardDropSynchronously(
            point: convert(sender.draggingLocation, from: nil),
            pasteboardTypes: pasteboardTypeIdentifiersForValidation(from: pasteboard),
            operationMask: requestedOperationMask,
            modifierFlags: UInt64(NSApp.currentEvent?.modifierFlags.rawValue ?? 0),
            timeoutMilliseconds: Self.pasteboardDropHitTestTimeoutMilliseconds
        ) ?? 0
        let allowed = NSDragOperation(rawValue: UInt(responseMask)).intersection(sender.draggingSourceOperationMask)
        return allowed.contains(.copy) ? .copy : []
    }

    private func pasteboardMatchesAcceptedTypes(_ pasteboard: NSPasteboard,
                                                acceptedTypes: [NSPasteboard.PasteboardType]) -> Bool {
        guard !acceptedTypes.isEmpty else { return false }
        if let availableTypes = pasteboard.types {
            for type in availableTypes where acceptedTypes.contains(type) {
                return true
            }
        }

        let acceptsLocalFiles = acceptedTypes.contains(NSPasteboard.PasteboardType(Self.droppedFileAccessPasteboardTypeIdentifier))
            || acceptedTypes.contains(.fileURL)
        return acceptsLocalFiles && pasteboard.canReadObject(forClasses: [NSURL.self],
                                                             options: [.urlReadingFileURLsOnly: true])
    }

    private func acceptsDroppedFilesForAccess() -> Bool {
        acceptsFileAccess(acceptedTypes: acceptedPasteboardDropTypes)
    }

    private func acceptsFileAccess(acceptedTypes: [NSPasteboard.PasteboardType]) -> Bool {
        acceptedTypes.contains(NSPasteboard.PasteboardType(Self.droppedFileAccessPasteboardTypeIdentifier))
            || acceptedTypes.contains(.fileURL)
    }

    private func pasteboardTypeIdentifiersForValidation(from pasteboard: NSPasteboard) -> [String] {
        var identifiers = Set((pasteboard.types ?? []).map(\.rawValue))
        if pasteboard.canReadObject(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) {
            identifiers.insert(NSPasteboard.PasteboardType.fileURL.rawValue)
            identifiers.insert(Self.droppedFileAccessPasteboardTypeIdentifier)
        }
        return identifiers.sorted()
    }

    private func pasteboardItemsForDrop(from pasteboard: NSPasteboard) -> [OuterframeContentPasteboardItem] {
        guard !acceptedPasteboardDropTypes.isEmpty else { return [] }

        if pasteboard.canReadObject(forClasses: [NSURL.self],
                                     options: [.urlReadingFileURLsOnly: true]) {
            return []
        }

        guard pasteboard.pasteboardItems?.first != nil else { return [] }
        return pasteboardItems(from: pasteboard, acceptedTypes: acceptedPasteboardDropTypes)
    }

    nonisolated private static func stageDroppedFile(from sourceURL: URL, to destinationURL: URL, isDirectory: Bool) throws {
        if !isDirectory && clonefile(sourceURL.path, destinationURL.path, 0) == 0 {
            return
        }
        try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
    }

    // MARK: - Accessibility

    public override func isAccessibilityElement() -> Bool {
        return false
    }

    @MainActor
    public override func accessibilityChildren() -> [Any]? {
        refreshAccessibilityElements()
        return accessibilityElements.isEmpty ? nil : accessibilityElements
    }

    public override func accessibilityHitTest(_ point: NSPoint) -> Any? {
        let hitElement: OuterframeAccessibilityElement? = MainActor.assumeIsolated {
            refreshAccessibilityElements()
            return hitTestAccessibilityElementsOnMain(point: point, elements: accessibilityElements)
        }
        if let element = hitElement {
            return element
        }
        return super.accessibilityHitTest(point)
    }

    func resetAccessibilitySnapshot() {
        accessibilityElements = []
    }

    func handleAccessibilityTreeChanged(notifications: OuterframeAccessibilityNotification) {
        accessibilityElements = []
        postAccessibilityNotifications(for: notifications)
    }

    @discardableResult
    private func refreshAccessibilityElements() -> OuterframeAccessibilitySnapshot? {
        guard let connection = outerframeContentConnection else {
            accessibilityElements = []
            return nil
        }
        let snapshot = connection.requestAccessibilitySnapshotSynchronously(
            timeoutMilliseconds: Self.accessibilitySnapshotTimeoutMilliseconds
        )
        let resolvedSnapshot = snapshot ?? OuterframeAccessibilitySnapshot.notImplementedSnapshot()
        accessibilityElements = buildAccessibilityElements(from: resolvedSnapshot)
        return resolvedSnapshot
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
            connection.sendToOuterframeContent(.mouseDown(point: point,
                                                          modifierFlags: modifierFlags,
                                                          clickCount: clickCount))
        }
    }

    public func handleMouseDragged(to point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.mouseDragged(point: point,
                                                             modifierFlags: modifierFlags))
        }
    }

    public func handleMouseUp(at point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.mouseUp(point: point,
                                                        modifierFlags: modifierFlags))
        }
    }

    public func handleMouseMoved(to point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.mouseMoved(point: point,
                                                           modifierFlags: modifierFlags))
        }
    }

    public func handleScrollWheel(at point: CGPoint, with event: NSEvent) {
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.scrollWheelEvent(point: point,
                                                                 delta: CGPoint(x: event.scrollingDeltaX, y: event.scrollingDeltaY),
                                                                 modifierFlags: event.modifierFlags,
                                                                 phase: event.phase,
                                                                 momentumPhase: event.momentumPhase,
                                                                 hasPreciseScrollingDeltas: event.hasPreciseScrollingDeltas))
        }
    }

    public func handleMagnify(at point: CGPoint, with event: NSEvent) {
        withActivePluginConnection { connection in
            let delta = event.magnification
            if delta != 0 {
                connection.sendMagnification(surfaceID: manualMagnificationSurfaceId,
                                             magnification: delta,
                                             location: point,
                                             scrollOffset: .zero)
            }

            if event.phase == .ended || event.phase == .cancelled {
                connection.sendMagnificationEnded(surfaceID: manualMagnificationSurfaceId,
                                                  magnification: 0,
                                                  location: point,
                                                  scrollOffset: .zero)
            }
        }
    }

    func handleRightMouseDown(at point: CGPoint, modifierFlags: NSEvent.ModifierFlags, clickCount: Int) {
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.rightMouseDown(point: point,
                                                               modifierFlags: modifierFlags,
                                                               clickCount: clickCount))
        }
    }

    public func handleRightMouseUp(at point: CGPoint, modifierFlags: NSEvent.ModifierFlags) {
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.rightMouseUp(point: point,
                                                             modifierFlags: modifierFlags))
        }
    }


    public func handleQuickLook(at point: CGPoint) {
        withActivePluginConnection { connection in
            connection.sendToOuterframeContent(.quickLook(point: point))
        }
    }

    public func handleKeyDown(_ event: NSEvent) {
        guard currentInputMode.allowsRawKeys else { return }
        withActivePluginConnection { connection in
            let characters = event.characters ?? ""
            let charactersIgnoringModifiers = event.charactersIgnoringModifiers ?? ""
            connection.sendToOuterframeContent(.keyDown(keyCode: event.keyCode,
                                                        characters: characters,
                                                        charactersIgnoringModifiers: charactersIgnoringModifiers,
                                                        modifierFlags: event.modifierFlags,
                                                        isARepeat: event.isARepeat))
        }
    }

    public func handleKeyUp(_ event: NSEvent) {
        guard currentInputMode.allowsRawKeys else { return }
        withActivePluginConnection { connection in
            let characters = event.characters ?? ""
            let charactersIgnoringModifiers = event.charactersIgnoringModifiers ?? ""
            connection.sendToOuterframeContent(.keyUp(keyCode: event.keyCode,
                                                      characters: characters,
                                                      charactersIgnoringModifiers: charactersIgnoringModifiers,
                                                      modifierFlags: event.modifierFlags,
                                                      isARepeat: event.isARepeat))
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
        presentationBundleURL = nil
        presentationStagingDirectoryURL = nil
        stopDisplayLink()
        activeFilePromiseWriters.removeAll()
        releaseAllDroppedFileAccesses()

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
                                      storageContext: OuterframeStorageContext,
                                      cacheContext: OuterframeCacheContext?,
                                      bypassCache: Bool = false,
                                      historyEntryID: UUID? = nil) async throws -> OuterframeViewLoadResult {
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
                                     networkProxyEndpoint: networkProxyEndpoint,
                                     storageContext: storageContext,
                                     historyEntryID: historyEntryID)
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
                                 networkProxyEndpoint: networkProxyEndpoint,
                                 storageContext: storageContext,
                                 historyEntryID: historyEntryID)
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
                           networkProxyEndpoint: OuterframeNetworkProxyEndpoint,
                           storageContext: OuterframeStorageContext,
                           historyEntryID: UUID? = nil) async throws {
        let pluginURLs = makePluginLoadURLStrings(originalURL: originalURL,
                                                  downloadURL: downloadURL)
        try await loadPlugin(bundleURL: bundleURL,
                             data: data,
                             outerURLString: pluginURLs.outerURLString,
                             bundleURLString: pluginURLs.bundleURLString,
                             networkProxyEndpoint: networkProxyEndpoint,
                             storageContext: storageContext,
                             historyEntryID: historyEntryID)
    }

    func loadPlugin(bundleURL: URL,
                    data: Data,
                    outerURLString: String,
                    bundleURLString: String,
                    networkProxyEndpoint: OuterframeNetworkProxyEndpoint,
                    storageContext: OuterframeStorageContext,
                    historyEntryID: UUID? = nil) async throws {
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
                try await connection.start(networkProxyEndpoint: networkProxyEndpoint,
                                           storageContext: storageContext)
            } catch {
                await connection.cleanup()
                setOuterframeContentConnection(nil)
                throw error
            }
        }

        let viewportSize = bounds.size
        let width = viewportSize.width
        let height = viewportSize.height
        presentationBundleURL = bundleURL.resolvingSymlinksInPath()
        presentationStagingDirectoryURL = connection.currentStagedFileDirectoryURL?.resolvingSymlinksInPath()

        do {
            try await loadPluginWithDebugHandling(connection: connection,
                                                  bundleURL: bundleURL,
                                                  data: data,
                                                  outerURLString: outerURLString,
                                                  bundleURLString: bundleURLString,
                                                  width: width,
                                                  height: height,
                                                  historyEntryID: historyEntryID)
        } catch {
            await connection.cleanup()
            setOuterframeContentConnection(nil)
            presentationBundleURL = nil
            presentationStagingDirectoryURL = nil
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
                                             height: CGFloat,
                                             historyEntryID: UUID?) async throws {
        if OuterframeDebugSettings.isDebugModeEnabled {
            try await waitForDebuggerBeforeLoadingPlugin(connection: connection,
                                                         bundleURL: bundleURL,
                                                         data: data,
                                                         outerURLString: outerURLString,
                                                         bundleURLString: bundleURLString,
                                                         width: width,
                                                         height: height,
                                                         historyEntryID: historyEntryID)
        } else {
            try await connection.loadPlugin(url: bundleURL,
                                            data: data,
                                            outerURLString: outerURLString,
                                            bundleURLString: bundleURLString,
                                            width: width,
                                            height: height,
                                            historyEntryID: historyEntryID)
        }
    }

    private func waitForDebuggerBeforeLoadingPlugin(connection: OuterframeContentConnection,
                                                    bundleURL: URL,
                                                    data: Data,
                                                    outerURLString: String,
                                                    bundleURLString: String,
                                                    width: CGFloat,
                                                    height: CGFloat,
                                                    historyEntryID: UUID?) async throws {
        try await withCheckedThrowingContinuation { continuation in
            let pendingLoad = PendingDebugLoad(connection: connection,
                                               bundleURL: bundleURL,
                                               pluginData: data,
                                               outerURLString: outerURLString,
                                               bundleURLString: bundleURLString,
                                               width: width,
                                               height: height,
                                               historyEntryID: historyEntryID,
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

        let outerframeMagic = Data("OUTR".utf8)
        let headerLength = 40
        guard data.count >= headerLength else {
            throw OuterframeLoadError.invalidOuterframeData
        }

        guard data.prefix(outerframeMagic.count) == outerframeMagic else {
            throw OuterframeLoadError.invalidOuterframeData
        }

        let formatVersion = readOuterframeUInt32(data, at: 4)
        guard formatVersion == 1 else {
            throw OuterframeLoadError.invalidOuterframeData
        }

        let pathOffsetValue = readOuterframeUInt64(data, at: 8)
        let pathLengthValue = readOuterframeUInt64(data, at: 16)
        let pluginDataOffsetValue = readOuterframeUInt64(data, at: 24)
        let pluginDataLengthValue = readOuterframeUInt64(data, at: 32)

        guard pathOffsetValue <= UInt64(Int.max),
              pathLengthValue <= UInt64(Int.max),
              pluginDataOffsetValue <= UInt64(Int.max),
              pluginDataLengthValue <= UInt64(Int.max) else {
            throw OuterframeLoadError.invalidOuterframeData
        }

        let pathOffset = Int(pathOffsetValue)
        let pathLength = Int(pathLengthValue)
        let pluginDataOffset = Int(pluginDataOffsetValue)
        let pluginDataLength = Int(pluginDataLengthValue)

        guard pathOffset >= headerLength,
              pathOffset <= data.count,
              pathLength <= data.count - pathOffset else {
            throw OuterframeLoadError.invalidPathLength
        }

        guard pluginDataOffset >= headerLength,
              pluginDataOffset <= data.count,
              pluginDataLength <= data.count - pluginDataOffset else {
            throw OuterframeLoadError.invalidOuterframeData
        }

        let pathData = data.subdata(in: pathOffset..<(pathOffset + pathLength))
        guard let remotePath = String(data: pathData, encoding: .utf8) else {
            throw OuterframeLoadError.invalidRemotePath
        }

        let pluginData = data.subdata(in: pluginDataOffset..<(pluginDataOffset + pluginDataLength))

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
        let historyEntryID = pending.historyEntryID
        let continuation = pending.continuation
        pending.continuation = nil

        Task { @MainActor in
            do {
                try await connection.loadPlugin(url: bundleURL,
                                                data: pluginData,
                                                outerURLString: outerURLString,
                                                bundleURLString: bundleURLString,
                                                width: width,
                                                height: height,
                                                historyEntryID: historyEntryID)
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
        let callbackID = UUID()

        displayLinkLock.lock()
        activeDisplayLinkCallbacks[callbackID] = true
        displayLinkLock.unlock()

        // Start the display link if it's not already running
        startDisplayLink()

        return callbackID
    }

    public func unregisterDisplayLinkCallback(_ callbackID: UUID) {
        displayLinkLock.lock()
        activeDisplayLinkCallbacks.removeValue(forKey: callbackID)
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

    public func handlePluginLoaded(contextID: CAContextID) {
        pluginIsReady = true

        // Update canvas with the plugin's context ID
        resetPasteboardInteractionState()
        resetAccessibilitySnapshot()
        updateWithContextID(contextID)

        // Make sure the canvas has focus for keyboard input
        window?.makeFirstResponder(self)

        if let outerframeContentConnection {
            outerframeContentConnection.sendSystemAppearanceUpdate(force: true)
            outerframeContentConnection.sendWindowActiveState(isActive: windowIsActive)
        }

    }

    func handlePasteboardAccessRequest(requestID: UUID,
                                       operation: OuterframePasteboardAccessOperation,
                                       pasteboardTypeIdentifiers: [String],
                                       items: [OuterframeContentPasteboardItem]) {
        if operation == .write {
            completePasteboardAccessRequest(requestID: requestID,
                                            operation: operation,
                                            pasteboardTypeIdentifiers: pasteboardTypeIdentifiers,
                                            items: items,
                                            allowed: true)
            return
        }

        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Allow Outerframe content to read the clipboard?"
        alert.informativeText = "The content requested programmatic clipboard read access."
        alert.addButton(withTitle: "Allow")
        alert.addButton(withTitle: "Deny")

        let finish: (Bool) -> Void = { [weak self] allowed in
            self?.completePasteboardAccessRequest(requestID: requestID,
                                                  operation: operation,
                                                  pasteboardTypeIdentifiers: pasteboardTypeIdentifiers,
                                                  items: items,
                                                  allowed: allowed)
        }

        if let window {
            alert.beginSheetModal(for: window) { response in
                finish(response == .alertFirstButtonReturn)
            }
        } else {
            finish(alert.runModal() == .alertFirstButtonReturn)
        }
    }

    func releaseDroppedFileAccess(accessID: UUID) {
        guard let access = droppedFileAccesses.removeValue(forKey: accessID) else { return }
        try? FileManager.default.removeItem(at: access.cleanupURL)
    }

    private func releaseAllDroppedFileAccesses() {
        let accesses = Array(droppedFileAccesses.values)
        droppedFileAccesses.removeAll()
        for access in accesses {
            try? FileManager.default.removeItem(at: access.cleanupURL)
        }
    }

    nonisolated private static func sanitizedStagedFileName(_ name: String) -> String {
        let fallbackName = "Dropped File"
        let baseName = URL(fileURLWithPath: name).lastPathComponent
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let nonEmptyName = baseName.isEmpty ? fallbackName : baseName
        let sanitized = nonEmptyName
            .components(separatedBy: CharacterSet(charactersIn: "/\0"))
            .filter { !$0.isEmpty }
            .joined(separator: "_")
        if sanitized.isEmpty || sanitized == "." || sanitized == ".." {
            return fallbackName
        }
        return sanitized
    }

    public func handleAccessibilityTreeChanged(notificationMask: UInt8) {
        let notifications = OuterframeAccessibilityNotification(rawValue: notificationMask)
        handleAccessibilityTreeChanged(notifications: notifications)
    }

    func handleTextInputGeometryUpdate(_ geometry: OuterframeContentTextInputGeometry?) {
        updateTextInputGeometry(geometry)
    }

    public func handlePluginOpenNewWindow(urlString: String, displayString: String?, preferredSize: CGSize?) {
        delegate?.outerframeView(self,
                                 didRequestOpenWindowWithURLString: urlString,
                                 displayString: displayString,
                                 preferredSize: preferredSize)
    }

    public func handlePluginNavigate(urlString: String) {
        delegate?.outerframeView(self, didRequestNavigateToURLString: urlString)
    }

    public func handlePluginOpenNewTab(urlString: String, displayString: String?) {
        delegate?.outerframeView(self,
                                 didRequestOpenTabWithURLString: urlString,
                                 displayString: displayString)
    }

    public func handlePluginHistoryPushEntry(entryID: UUID, urlString: String?) {
        delegate?.outerframeView(self, didRequestHistoryPushEntryWithID: entryID, urlString: urlString)
    }

    public func handlePluginHistoryReplaceEntry(entryID: UUID, urlString: String?) {
        delegate?.outerframeView(self, didRequestHistoryReplaceEntryWithID: entryID, urlString: urlString)
    }

    public func handlePluginHistoryGo(delta: Int32) {
        delegate?.outerframeView(self, didRequestHistoryGo: delta)
    }

    func handlePluginSetTitle(_ title: String?) {
        delegate?.outerframeView(self, didSetTitle: title)
    }

    func handlePluginSetIcon(_ icon: OuterframePresentationIcon) {
        delegate?.outerframeView(self, didSetIcon: resolvePresentationIcon(icon))
    }

    private func resolvePresentationIcon(_ icon: OuterframePresentationIcon) -> NSImage? {
        switch icon {
        case .none:
            return nil
        case .bundleResource(let path):
            guard let presentationBundleURL,
                  let url = resolvedPresentationURL(path: path,
                                                    relativeTo: presentationBundleURL,
                                                    allowAbsolutePath: false) else {
                return nil
            }
            return NSImage(contentsOf: url)
        case .stagedFile(let path):
            guard let presentationStagingDirectoryURL,
                  let url = resolvedPresentationURL(path: path,
                                                    relativeTo: presentationStagingDirectoryURL,
                                                    allowAbsolutePath: true) else {
                return nil
            }
            return NSImage(contentsOf: url)
        }
    }

    private func resolvedPresentationURL(path: String,
                                         relativeTo rootURL: URL,
                                         allowAbsolutePath: Bool) -> URL? {
        let trimmedPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty,
              !trimmedPath.contains("\0") else {
            return nil
        }

        let candidateURL: URL
        if trimmedPath.hasPrefix("/") {
            guard allowAbsolutePath else { return nil }
            candidateURL = URL(fileURLWithPath: trimmedPath)
        } else {
            candidateURL = rootURL.appendingPathComponent(trimmedPath)
        }

        let resolvedRoot = rootURL.resolvingSymlinksInPath().standardizedFileURL
        let resolvedCandidate = candidateURL.resolvingSymlinksInPath().standardizedFileURL
        let rootPath = resolvedRoot.path
        let candidatePath = resolvedCandidate.path
        guard candidatePath == rootPath || candidatePath.hasPrefix(rootPath + "/") else {
            return nil
        }
        return resolvedCandidate
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

    func sendTextInputFocus(fieldID: UUID, hasFocus: Bool) {
        withActivePluginConnection { connection in
            connection.sendTextInputFocus(fieldID: fieldID, hasFocus: hasFocus)
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

    func requestPasteboardItemsForCut(completion: @escaping ([OuterframeContentPasteboardItem]?) -> Void) {
        guard pluginIsReady, let connection = outerframeContentConnection else {
            completion(nil)
            return
        }
        connection.requestPasteboardItemsForCut(completion: completion)
    }

    func sendPasteboardItemsForPaste(_ items: [OuterframeContentPasteboardItem]) {
        withActivePluginConnection { connection in
            connection.sendPasteboardItemsForPaste(items)
        }
    }

    public func acceptHistoryEntry(entryID: UUID, url: URL) {
        withActivePluginConnection { connection in
            connection.sendHistoryEntryAccepted(entryID: entryID, urlString: url.absoluteString)
        }
    }

    public func rejectHistoryEntry(entryID: UUID, errorMessage: String) {
        withActivePluginConnection { connection in
            connection.sendHistoryEntryRejected(entryID: entryID, errorMessage: errorMessage)
        }
    }

    public func sendHistoryTraversal(entryID: UUID, url: URL) {
        withActivePluginConnection { connection in
            connection.sendHistoryTraversal(entryID: entryID, urlString: url.absoluteString)
        }
    }

    public func sendHistoryContextUpdate(currentEntryID: UUID, url: URL, length: Int, canGoBack: Bool, canGoForward: Bool) {
        withActivePluginConnection { connection in
            let clampedLength = UInt32(min(max(length, 0), Int(UInt32.max)))
            connection.sendHistoryContextUpdate(currentEntryID: currentEntryID,
                                                urlString: url.absoluteString,
                                                length: clampedLength,
                                                canGoBack: canGoBack,
                                                canGoForward: canGoForward)
        }
    }

    func sendSetCursorPosition(fieldID: UUID, position: Int, modifySelection: Bool) {
        withActivePluginConnection { connection in
            connection.sendSetCursorPosition(fieldID: fieldID, position: UInt64(position), modifySelection: modifySelection)
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
    let historyEntryID: UUID?
    var debuggerAttached: Bool
    var continuation: CheckedContinuation<Void, Error>?

    init(connection: OuterframeContentConnection,
         bundleURL: URL,
         pluginData: Data,
         outerURLString: String,
         bundleURLString: String,
         width: CGFloat,
         height: CGFloat,
         historyEntryID: UUID?,
         debuggerAttached: Bool,
         continuation: CheckedContinuation<Void, Error>) {
        self.connection = connection
        self.bundleURL = bundleURL
        self.pluginData = pluginData
        self.outerURLString = outerURLString
        self.bundleURLString = bundleURLString
        self.width = width
        self.height = height
        self.historyEntryID = historyEntryID
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

private func readOuterframeUInt32(_ data: Data, at offset: Int) -> UInt32 {
    UInt32(data[offset])
        | (UInt32(data[offset + 1]) << 8)
        | (UInt32(data[offset + 2]) << 16)
        | (UInt32(data[offset + 3]) << 24)
}

private func readOuterframeUInt64(_ data: Data, at offset: Int) -> UInt64 {
    UInt64(data[offset])
        | (UInt64(data[offset + 1]) << 8)
        | (UInt64(data[offset + 2]) << 16)
        | (UInt64(data[offset + 3]) << 24)
        | (UInt64(data[offset + 4]) << 32)
        | (UInt64(data[offset + 5]) << 40)
        | (UInt64(data[offset + 6]) << 48)
        | (UInt64(data[offset + 7]) << 56)
}

extension OuterframeView: OuterframeContentConnectionDelegate {}

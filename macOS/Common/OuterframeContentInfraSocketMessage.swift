import Foundation
import AppKit.NSAppearance

let OuterframeContentInfraSocketHeaderLength = MemoryLayout<UInt8>.size + MemoryLayout<UInt32>.size

// MARK: - Infrastructure Messages (Browser ↔ OuterframeContent)

/// Messages from Browser to OuterframeContent on the infrastructure socket
enum BrowserToContentInfraMessage {
    case loadPlugin(requestId: UUID, pluginURL: String)
    case unloadPlugin
    case setDebuggerAttachmentMonitoring(isEnabled: Bool)
    case shutdown

    func encode() throws -> Data {
        switch self {
        case .loadPlugin(let requestId, let pluginURL):
            var payload = Data()
            payload.append(uuid: requestId)
            try payload.append(lengthPrefixedUTF8_32: pluginURL)
            return makeBrowserToContentInfraFrame(type: .loadPluginRequest, payload: payload)

        case .unloadPlugin:
            return makeBrowserToContentInfraFrame(type: .unloadPluginRequest, payload: Data())

        case .setDebuggerAttachmentMonitoring(let isEnabled):
            var payload = Data(capacity: 1)
            payload.append(uint8: isEnabled ? 1 : 0)
            return makeBrowserToContentInfraFrame(type: .setDebuggerAttachmentMonitoring, payload: payload)

        case .shutdown:
            return makeBrowserToContentInfraFrame(type: .shutdown, payload: Data())
        }
    }

    static func decode(typeRaw: UInt8, payload: Data) throws -> BrowserToContentInfraMessage {
        guard let type = BrowserToContentInfraMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentInfraSocketMessageError.unknownType(typeRaw)
        }

        var cursor = DataCursor(payload)

        switch type {
        case .loadPluginRequest:
            guard let requestId = cursor.readUUID(),
                  let url = cursor.readString32() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .loadPlugin(requestId: requestId, pluginURL: url)

        case .unloadPluginRequest:
            return .unloadPlugin

        case .setDebuggerAttachmentMonitoring:
            guard let raw = cursor.readUInt8() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .setDebuggerAttachmentMonitoring(isEnabled: raw != 0)

        case .shutdown:
            return .shutdown
        }
    }
}

/// Messages from OuterframeContent to Browser on the infrastructure socket
enum ContentToBrowserInfraMessage {
    case loadPluginSuccess(requestId: UUID)
    case loadPluginFailure(requestId: UUID, errorMessage: String)
    case pluginLoaded(contextId: UInt32, success: Bool)
    case pluginUnloaded
    case debuggerAttached

    func encode() throws -> Data {
        switch self {
        case .loadPluginSuccess(let requestId):
            var payload = Data(capacity: 16)
            payload.append(uuid: requestId)
            return makeContentToBrowserInfraFrame(type: .loadPluginSuccess, payload: payload)

        case .loadPluginFailure(let requestId, let errorMessage):
            var payload = Data()
            payload.append(uuid: requestId)
            try payload.append(lengthPrefixedUTF8_32: errorMessage)
            return makeContentToBrowserInfraFrame(type: .loadPluginFailure, payload: payload)

        case .pluginLoaded(let contextId, let success):
            var payload = Data(capacity: 4 + 1)
            payload.append(uint32: contextId)
            payload.append(uint8: success ? 1 : 0)
            return makeContentToBrowserInfraFrame(type: .pluginLoaded, payload: payload)

        case .pluginUnloaded:
            return makeContentToBrowserInfraFrame(type: .pluginUnloaded, payload: Data())

        case .debuggerAttached:
            return makeContentToBrowserInfraFrame(type: .debuggerAttached, payload: Data())
        }
    }

    static func decode(typeRaw: UInt8, payload: Data) throws -> ContentToBrowserInfraMessage {
        guard let type = ContentToBrowserInfraMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentInfraSocketMessageError.unknownType(typeRaw)
        }

        var cursor = DataCursor(payload)

        switch type {
        case .loadPluginSuccess:
            guard let requestId = cursor.readUUID() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .loadPluginSuccess(requestId: requestId)

        case .loadPluginFailure:
            guard let requestId = cursor.readUUID(),
                  let message = cursor.readString32() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .loadPluginFailure(requestId: requestId, errorMessage: message)

        case .pluginLoaded:
            guard let contextId = cursor.readUInt32(),
                  let successRaw = cursor.readUInt8() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .pluginLoaded(contextId: contextId, success: successRaw != 0)

        case .pluginUnloaded:
            return .pluginUnloaded

        case .debuggerAttached:
            return .debuggerAttached
        }
    }
}

enum OuterframeContentInfraSocketMessageError: Error {
    case unknownType(UInt8)
    case truncatedPayload
    case encodingFailure(String)
}

// MARK: - Message Kind Enums

private enum BrowserToContentInfraMessageKind: UInt8 {
    case loadPluginRequest = 3
    case unloadPluginRequest = 6
    case setDebuggerAttachmentMonitoring = 48
    case shutdown = 49
}

private enum ContentToBrowserInfraMessageKind: UInt8 {
    case loadPluginSuccess = 4
    case loadPluginFailure = 5
    case pluginLoaded = 19
    case pluginUnloaded = 20
    case debuggerAttached = 47
}


// MARK: - Frame Helpers

private func makeBrowserToContentInfraFrame(type: BrowserToContentInfraMessageKind, payload: Data) -> Data {
    var frame = Data(capacity: OuterframeContentInfraSocketHeaderLength + payload.count)
    frame.append(type.rawValue)
    frame.append(uint32: UInt32(payload.count))
    frame.append(payload)
    return frame
}

private func makeContentToBrowserInfraFrame(type: ContentToBrowserInfraMessageKind, payload: Data) -> Data {
    var frame = Data(capacity: OuterframeContentInfraSocketHeaderLength + payload.count)
    frame.append(type.rawValue)
    frame.append(uint32: UInt32(payload.count))
    frame.append(payload)
    return frame
}


// MARK: - Data Cursor

private struct DataCursor {
    private let data: Data
    private var offset: Int = 0

    init(_ data: Data) {
        self.data = data
    }

    mutating func readUInt32() -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        let value = data[offset..<(offset + 4)].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        offset += 4
        return value
    }

    mutating func readUInt16() -> UInt16? {
        guard offset + 2 <= data.count else { return nil }
        let value = data[offset..<(offset + 2)].reduce(UInt16(0)) { ($0 << 8) | UInt16($1) }
        offset += 2
        return value
    }

    mutating func readUInt8() -> UInt8? {
        guard offset + 1 <= data.count else { return nil }
        let value = data[offset]
        offset += 1
        return value
    }

    mutating func readUInt64() -> UInt64? {
        guard offset + 8 <= data.count else { return nil }
        let value = data[offset..<(offset + 8)].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
        offset += 8
        return value
    }

    mutating func readFloat32() -> Float32? {
        guard let bits = readUInt32() else { return nil }
        return Float32(bitPattern: bits)
    }

    mutating func readFloat64() -> Float64? {
        guard let bits = readUInt64() else { return nil }
        return Float64(bitPattern: bits)
    }

    mutating func readData(_ length: Int) -> Data? {
        guard offset + length <= data.count else { return nil }
        let range = offset..<(offset + length)
        offset += length
        return data.subdata(in: range)
    }

    mutating func readData32() -> Data? {
        guard let length = readUInt32() else { return nil }
        return readData(Int(length))
    }

    mutating func readString32() -> String? {
        guard let data = readData32() else { return nil }
        return String(data: data, encoding: .utf8)
    }

    mutating func readUUID() -> UUID? {
        guard let bytes = readData(16) else { return nil }
        return bytes.withUnsafeBytes { raw -> UUID? in
            guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return nil }
            return NSUUID(uuidBytes: base) as UUID
        }
    }
}

// MARK: - Data Extensions

fileprivate extension Data {
    mutating func append(uint32 value: UInt32) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { append(contentsOf: $0) }
    }

    mutating func append(int32 value: Int32) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { append(contentsOf: $0) }
    }

    mutating func append(uint16 value: UInt16) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { append(contentsOf: $0) }
    }

    mutating func append(uint8 value: UInt8) {
        append(value)
    }

    mutating func append(uint64 value: UInt64) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { append(contentsOf: $0) }
    }

    mutating func append(float64 value: Double) {
        append(uint64: value.bitPattern)
    }

    mutating func append(float32 value: Float32) {
        append(int32: Int32(bitPattern: value.bitPattern))
    }

    mutating func append(uuid: UUID) {
        var uuidValue = uuid.uuid
        Swift.withUnsafeBytes(of: &uuidValue) { append(contentsOf: $0) }
    }

    mutating func append(lengthPrefixedUTF8_32 string: String) throws {
        guard let data = string.data(using: .utf8) else {
            throw OuterframeContentInfraSocketMessageError.encodingFailure("Invalid UTF-8 string")
        }
        try append(lengthPrefixedData32: data)
    }

    mutating func append(lengthPrefixedData32 data: Data) throws {
        guard data.count <= UInt32.max else {
            throw OuterframeContentInfraSocketMessageError.encodingFailure("Data too long")
        }
        append(uint32: UInt32(data.count))
        append(data)
    }
}

import Foundation
import AppKit.NSAppearance

let OuterframeContentInfraSocketHeaderLength = MemoryLayout<UInt32>.size
let OuterframeContentInfraSocketMessageTypeLength = MemoryLayout<UInt16>.size

// MARK: - Infrastructure Messages (Browser ↔ OuterframeContent)

/// Messages from Browser to OuterframeContent on the infrastructure socket
enum BrowserToContentInfraMessage {
    case loadPlugin(requestID: UUID, pluginURL: String)
    case unloadPlugin
    case setDebuggerAttachmentMonitoring(isEnabled: Bool)
    case shutdown

    func encode() throws -> Data {
        switch self {
        case .loadPlugin(let requestID, let pluginURL):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            try payload.append(stringReference: pluginURL)
            return makeBrowserToContentInfraFrame(type: .loadPluginRequest, payload: try payload.finalize())

        case .unloadPlugin:
            return makeBrowserToContentInfraFrame(type: .unloadPluginRequest, payload: Data())

        case .setDebuggerAttachmentMonitoring(let isEnabled):
            var payload = Data(capacity: 1)
            payload.append(uint8: isEnabled ? 1 << 0 : 0)
            return makeBrowserToContentInfraFrame(type: .setDebuggerAttachmentMonitoring, payload: payload)

        case .shutdown:
            return makeBrowserToContentInfraFrame(type: .shutdown, payload: Data())
        }
    }

    static func decode(message: Data) throws -> BrowserToContentInfraMessage {
        var cursor = DataCursor(message)
        guard let typeRaw = cursor.readUInt16() else {
            throw OuterframeContentInfraSocketMessageError.truncatedPayload
        }
        guard let type = BrowserToContentInfraMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentInfraSocketMessageError.unknownType(typeRaw)
        }

        switch type {
        case .loadPluginRequest:
            guard let requestID = cursor.readUUID(),
                  let url = cursor.readStringReference() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .loadPlugin(requestID: requestID, pluginURL: url)

        case .unloadPluginRequest:
            return .unloadPlugin

        case .setDebuggerAttachmentMonitoring:
            guard let raw = cursor.readUInt8() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .setDebuggerAttachmentMonitoring(isEnabled: raw & (1 << 0) != 0)

        case .shutdown:
            return .shutdown
        }
    }
}

/// Messages from OuterframeContent to Browser on the infrastructure socket
enum ContentToBrowserInfraMessage {
    case loadPluginSuccess(requestID: UUID)
    case loadPluginFailure(requestID: UUID, errorMessage: String)
    case pluginLoaded(contextID: UInt32, success: Bool)
    case pluginUnloaded
    case debuggerAttached

    func encode() throws -> Data {
        switch self {
        case .loadPluginSuccess(let requestID):
            var payload = Data(capacity: 16)
            payload.append(uuid: requestID)
            return makeContentToBrowserInfraFrame(type: .loadPluginSuccess, payload: payload)

        case .loadPluginFailure(let requestID, let errorMessage):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            try payload.append(stringReference: errorMessage)
            return makeContentToBrowserInfraFrame(type: .loadPluginFailure, payload: try payload.finalize())

        case .pluginLoaded(let contextID, let success):
            var payload = Data(capacity: 4 + 1)
            payload.append(uint32: contextID)
            payload.append(uint8: success ? 1 << 0 : 0)
            return makeContentToBrowserInfraFrame(type: .pluginLoaded, payload: payload)

        case .pluginUnloaded:
            return makeContentToBrowserInfraFrame(type: .pluginUnloaded, payload: Data())

        case .debuggerAttached:
            return makeContentToBrowserInfraFrame(type: .debuggerAttached, payload: Data())
        }
    }

    static func decode(message: Data) throws -> ContentToBrowserInfraMessage {
        var cursor = DataCursor(message)
        guard let typeRaw = cursor.readUInt16() else {
            throw OuterframeContentInfraSocketMessageError.truncatedPayload
        }
        guard let type = ContentToBrowserInfraMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentInfraSocketMessageError.unknownType(typeRaw)
        }

        switch type {
        case .loadPluginSuccess:
            guard let requestID = cursor.readUUID() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .loadPluginSuccess(requestID: requestID)

        case .loadPluginFailure:
            guard let requestID = cursor.readUUID(),
                  let message = cursor.readStringReference() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .loadPluginFailure(requestID: requestID, errorMessage: message)

        case .pluginLoaded:
            guard let contextID = cursor.readUInt32(),
                  let flags = cursor.readUInt8() else {
                throw OuterframeContentInfraSocketMessageError.truncatedPayload
            }
            return .pluginLoaded(contextID: contextID, success: flags & (1 << 0) != 0)

        case .pluginUnloaded:
            return .pluginUnloaded

        case .debuggerAttached:
            return .debuggerAttached
        }
    }
}

enum OuterframeContentInfraSocketMessageError: Error {
    case unknownType(UInt16)
    case truncatedPayload
    case encodingFailure(String)
}

// MARK: - Message Kind Enums

private enum BrowserToContentInfraMessageKind: UInt16 {
    case loadPluginRequest = 3000
    case unloadPluginRequest = 3001
    case shutdown = 3002
    case setDebuggerAttachmentMonitoring = 3003
}

private enum ContentToBrowserInfraMessageKind: UInt16 {
    case loadPluginSuccess = 4000
    case loadPluginFailure = 4001
    case pluginLoaded = 4002
    case pluginUnloaded = 4003
    case debuggerAttached = 4004
}


// MARK: - Frame Helpers

private func makeBrowserToContentInfraFrame(type: BrowserToContentInfraMessageKind, payload: Data) -> Data {
    let messageLength = OuterframeContentInfraSocketMessageTypeLength + payload.count
    var frame = Data(capacity: OuterframeContentInfraSocketHeaderLength + messageLength)
    frame.append(uint32: UInt32(messageLength))
    frame.append(uint16: type.rawValue)
    frame.append(payload)
    return frame
}

private func makeContentToBrowserInfraFrame(type: ContentToBrowserInfraMessageKind, payload: Data) -> Data {
    let messageLength = OuterframeContentInfraSocketMessageTypeLength + payload.count
    var frame = Data(capacity: OuterframeContentInfraSocketHeaderLength + messageLength)
    frame.append(uint32: UInt32(messageLength))
    frame.append(uint16: type.rawValue)
    frame.append(payload)
    return frame
}


// MARK: - Data Cursor

private struct OffsetPayloadBuilder {
    private struct Reference {
        let patchOffset: Int
        let variableOffset: Int
        let length: Int
    }

    private var fixed = Data()
    private var variable = Data()
    private var references: [Reference] = []
    private let referenceBaseOffset: Int

    init(referenceBaseOffset: Int = OuterframeContentInfraSocketMessageTypeLength) {
        self.referenceBaseOffset = referenceBaseOffset
    }

    mutating func append(uint8 value: UInt8) {
        fixed.append(uint8: value)
    }

    mutating func append(uint32 value: UInt32) {
        fixed.append(uint32: value)
    }

    mutating func append(uuid: UUID) {
        fixed.append(uuid: uuid)
    }

    mutating func append(stringReference string: String) throws {
        guard let data = string.data(using: .utf8) else {
            throw OuterframeContentInfraSocketMessageError.encodingFailure("Invalid UTF-8 string")
        }
        try append(dataReference: data)
    }

    mutating func append(dataReference data: Data) throws {
        guard data.count <= UInt32.max else {
            throw OuterframeContentInfraSocketMessageError.encodingFailure("Data too long")
        }
        let patchOffset = fixed.count
        fixed.append(uint32: 0)
        fixed.append(uint32: UInt32(data.count))
        references.append(Reference(patchOffset: patchOffset,
                                    variableOffset: variable.count,
                                    length: data.count))
        variable.append(data)
    }

    mutating func finalize() throws -> Data {
        guard fixed.count <= UInt32.max,
              variable.count <= UInt32.max,
              variable.count <= Int(UInt32.max) - fixed.count else {
            throw OuterframeContentInfraSocketMessageError.encodingFailure("Payload too long")
        }

        for reference in references {
            let offset = referenceBaseOffset + fixed.count + reference.variableOffset
            guard offset <= UInt32.max,
                  reference.length <= UInt32.max else {
                throw OuterframeContentInfraSocketMessageError.encodingFailure("Payload too long")
            }
            fixed.replaceUInt32(at: reference.patchOffset, with: UInt32(offset))
            fixed.replaceUInt32(at: reference.patchOffset + 4, with: UInt32(reference.length))
        }

        var payload = Data(capacity: fixed.count + variable.count)
        payload.append(fixed)
        payload.append(variable)
        return payload
    }
}

private struct DataCursor {
    private let data: Data
    private var offset: Int = 0

    init(_ data: Data) {
        self.data = data
    }

    mutating func readUInt32() -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        let value = data[offset..<(offset + 4)].enumerated().reduce(UInt32(0)) {
            $0 | (UInt32($1.element) << (8 * $1.offset))
        }
        offset += 4
        return value
    }

    mutating func readUInt16() -> UInt16? {
        guard offset + 2 <= data.count else { return nil }
        let value = data[offset..<(offset + 2)].enumerated().reduce(UInt16(0)) {
            $0 | (UInt16($1.element) << (8 * $1.offset))
        }
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
        let value = data[offset..<(offset + 8)].enumerated().reduce(UInt64(0)) {
            $0 | (UInt64($1.element) << (8 * $1.offset))
        }
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
        guard length >= 0,
              length <= data.count - offset else { return nil }
        let range = offset..<(offset + length)
        offset += length
        return data.subdata(in: range)
    }

    mutating func readDataReference() -> Data? {
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

    mutating func readStringReference() -> String? {
        guard let data = readDataReference() else { return nil }
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
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }

    mutating func append(int32 value: Int32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }

    mutating func append(uint16 value: UInt16) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }

    mutating func append(uint8 value: UInt8) {
        append(value)
    }

    mutating func append(uint64 value: UInt64) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
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

    mutating func replaceUInt32(at offset: Int, with value: UInt32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) {
            replaceSubrange(offset..<(offset + 4), with: $0)
        }
    }

}

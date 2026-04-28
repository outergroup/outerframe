import Foundation
import CoreGraphics

/// Represents the role of an accessibility element exposed by outerframe content.
public enum OuterframeAccessibilityRole: UInt8, Sendable {
    case container = 0
    case staticText = 1
    case button = 2
    case image = 3
    case table = 4
    case row = 5
    case cell = 6
    case textField = 7
}

/// Represents notifications that the outerframe content can request the host to post.
public struct OuterframeAccessibilityNotification: OptionSet, Sendable {
    public let rawValue: UInt8

    public init(rawValue: UInt8) {
        self.rawValue = rawValue
    }

    public static let layoutChanged = OuterframeAccessibilityNotification(rawValue: 1 << 0)
    public static let selectedChildrenChanged = OuterframeAccessibilityNotification(rawValue: 1 << 1)
    public static let focusedElementChanged = OuterframeAccessibilityNotification(rawValue: 1 << 2)
}

/// A node within the content-provided accessibility tree.
public struct OuterframeAccessibilityNode: Sendable {
    public var identifier: UInt32
    public var role: OuterframeAccessibilityRole
    public var frame: CGRect
    public var label: String?
    public var value: String?
    public var hint: String?
    public var children: [OuterframeAccessibilityNode]
    /// For table roles: the total number of rows (may exceed children count for virtualized tables)
    public var rowCount: Int?
    /// For table roles: the number of columns
    public var columnCount: Int?
    /// Whether the element is enabled (default true)
    public var isEnabled: Bool

    public init(identifier: UInt32,
                role: OuterframeAccessibilityRole,
                frame: CGRect,
                label: String? = nil,
                value: String? = nil,
                hint: String? = nil,
                children: [OuterframeAccessibilityNode] = [],
                rowCount: Int? = nil,
                columnCount: Int? = nil,
                isEnabled: Bool = true) {
        self.identifier = identifier
        self.role = role
        self.frame = frame
        self.label = label
        self.value = value
        self.hint = hint
        self.children = children
        self.rowCount = rowCount
        self.columnCount = columnCount
        self.isEnabled = isEnabled
    }
}

/// A snapshot describing all accessibility elements exposed by the outerframe content.
public struct OuterframeAccessibilitySnapshot: Sendable {
    public var rootNodes: [OuterframeAccessibilityNode]

    public init(rootNodes: [OuterframeAccessibilityNode]) {
        self.rootNodes = rootNodes
    }

    /// Convenience helper for content that has not implemented accessibility yet.
    public static func notImplementedSnapshot(message: String = "Accessibility not implemented") -> OuterframeAccessibilitySnapshot {
        let child = OuterframeAccessibilityNode(identifier: 1,
                                            role: .staticText,
                                            frame: .zero,
                                            label: message)
        let root = OuterframeAccessibilityNode(identifier: 0,
                                           role: .container,
                                           frame: .zero,
                                           children: [child])
        return OuterframeAccessibilitySnapshot(rootNodes: [root])
    }

    /// Serialises the snapshot to the binary format shared with the host.
    public func serializedData() -> Data {
        var data = Data()
        data.appendUInt8(1) // format version
        let clampedCount = UInt16(max(0, min(rootNodes.count, Int(UInt16.max))))
        data.appendUInt16(clampedCount)
        for node in rootNodes.prefix(Int(clampedCount)) {
            encode(node: node, into: &data)
        }
        return data
    }

    /// Deserialises a snapshot from binary payload produced by `serializedData`.
    public static func deserialize(from data: Data) -> OuterframeAccessibilitySnapshot? {
        var cursor = AccessibilityDataCursor(data: data)
        guard let version = cursor.readUInt8(), version == 1 else {
            return nil
        }
        guard let rootCount = cursor.readUInt16() else {
            return nil
        }

        var nodes: [OuterframeAccessibilityNode] = []
        nodes.reserveCapacity(Int(rootCount))

        for _ in 0..<rootCount {
            guard let node = decodeNode(from: &cursor) else {
                return nil
            }
            nodes.append(node)
        }

        return OuterframeAccessibilitySnapshot(rootNodes: nodes)
    }

    private func encode(node: OuterframeAccessibilityNode, into data: inout Data) {
        data.appendUInt32(node.identifier)
        data.appendUInt8(node.role.rawValue)
        data.appendFloat32(Float32(node.frame.origin.x))
        data.appendFloat32(Float32(node.frame.origin.y))
        data.appendFloat32(Float32(node.frame.size.width))
        data.appendFloat32(Float32(node.frame.size.height))
        data.appendOptionalString(node.label)
        data.appendOptionalString(node.value)
        data.appendOptionalString(node.hint)
        data.appendOptionalInt32(node.rowCount)
        data.appendOptionalInt32(node.columnCount)
        data.appendUInt8(node.isEnabled ? 1 : 0)

        let clampedChildren = UInt16(max(0, min(node.children.count, Int(UInt16.max))))
        data.appendUInt16(clampedChildren)
        for child in node.children.prefix(Int(clampedChildren)) {
            encode(node: child, into: &data)
        }
    }

    private static func decodeNode(from cursor: inout AccessibilityDataCursor) -> OuterframeAccessibilityNode? {
        guard let identifier = cursor.readUInt32(),
              let roleRaw = cursor.readUInt8(),
              let role = OuterframeAccessibilityRole(rawValue: roleRaw),
              let originX = cursor.readFloat32(),
              let originY = cursor.readFloat32(),
              let width = cursor.readFloat32(),
              let height = cursor.readFloat32(),
              let label = cursor.readOptionalString(),
              let value = cursor.readOptionalString(),
              let hint = cursor.readOptionalString(),
              let rowCount = cursor.readOptionalInt32(),
              let columnCount = cursor.readOptionalInt32(),
              let isEnabledRaw = cursor.readUInt8(),
              let childCount = cursor.readUInt16() else {
            return nil
        }

        let isEnabled = isEnabledRaw != 0

        var children: [OuterframeAccessibilityNode] = []
        children.reserveCapacity(Int(childCount))

        for _ in 0..<childCount {
            guard let child = decodeNode(from: &cursor) else {
                return nil
            }
            children.append(child)
        }

        let frame = CGRect(x: CGFloat(originX),
                           y: CGFloat(originY),
                           width: CGFloat(width),
                           height: CGFloat(height))

        return OuterframeAccessibilityNode(identifier: identifier,
                                       role: role,
                                       frame: frame,
                                       label: label,
                                       value: value,
                                       hint: hint,
                                       children: children,
                                       rowCount: rowCount,
                                       columnCount: columnCount,
                                       isEnabled: isEnabled)
    }
}

// MARK: - Binary helpers

private struct AccessibilityDataCursor {
    private let data: Data
    private var offset: Int = 0

    init(data: Data) {
        self.data = data
    }

    mutating func readUInt8() -> UInt8? {
        guard offset < data.count else { return nil }
        let value = data[offset]
        offset += 1
        return value
    }

    mutating func readUInt16() -> UInt16? {
        guard offset + 2 <= data.count else { return nil }
        let byte0 = UInt16(data[offset])
        let byte1 = UInt16(data[offset + 1])
        offset += 2
        return (byte0 << 8) | byte1
    }

    mutating func readUInt32() -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        let b0 = UInt32(data[offset])
        let b1 = UInt32(data[offset + 1])
        let b2 = UInt32(data[offset + 2])
        let b3 = UInt32(data[offset + 3])
        offset += 4
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    }

    mutating func readFloat32() -> Float32? {
        guard let bits = readUInt32() else { return nil }
        return Float32(bitPattern: bits)
    }

    mutating func readOptionalString() -> Optional<String>? {
        guard let hasValue = readUInt8() else { return nil }
        if hasValue == 0 {
            return .some(nil)
        }
        guard let length = readUInt32() else { return nil }
        let intLength = Int(length)
        guard intLength >= 0, offset + intLength <= data.count else { return nil }
        let range = offset..<(offset + intLength)
        offset += intLength
        let slice = data[range]
        return String(data: slice, encoding: .utf8)
    }

    mutating func readOptionalInt32() -> Optional<Int>? {
        guard let hasValue = readUInt8() else { return nil }
        if hasValue == 0 {
            return .some(nil)
        }
        guard let value = readUInt32() else { return nil }
        return .some(Int(Int32(bitPattern: value)))
    }
}

private extension Data {
    mutating func appendUInt8(_ value: UInt8) {
        append(value)
    }

    mutating func appendUInt16(_ value: UInt16) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { append(contentsOf: $0) }
    }

    mutating func appendUInt32(_ value: UInt32) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { append(contentsOf: $0) }
    }

    mutating func appendFloat32(_ value: Float32) {
        appendUInt32(value.bitPattern)
    }

    mutating func appendOptionalString(_ string: String?) {
        guard let string else {
            appendUInt8(0)
            return
        }
        appendUInt8(1)
        if let data = string.data(using: .utf8) {
            appendUInt32(UInt32(data.count))
            append(data)
        } else {
            appendUInt32(0)
        }
    }

    mutating func appendOptionalInt32(_ value: Int?) {
        guard let value else {
            appendUInt8(0)
            return
        }
        appendUInt8(1)
        appendUInt32(UInt32(bitPattern: Int32(clamping: value)))
    }
}

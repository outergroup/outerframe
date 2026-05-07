import Foundation
import AppKit

let OuterframeContentSocketHeaderLength = MemoryLayout<UInt32>.size
let OuterframeContentSocketMessageTypeLength = MemoryLayout<UInt16>.size

// MARK: - Content Messages (Browser ↔ Content)

struct InitializeContentProxy {
    var host: String
    var port: UInt16
    var username: String?
    var password: String?
}

struct InitializeContentArguments {
    var data: Data?
    var contentSize: CGSize?
    var appearance: NSAppearance?
    var proxy: InitializeContentProxy?
    var url: String?
    var bundleUrl: String?
    var windowIsActive: Bool?

    init(data: Data? = nil,
         contentSize: CGSize? = nil,
         appearance: NSAppearance? = nil,
         proxy: InitializeContentProxy? = nil,
         url: String? = nil,
         bundleUrl: String? = nil,
         windowIsActive: Bool? = nil) {
        self.data = data
        self.contentSize = contentSize
        self.appearance = appearance
        self.proxy = proxy
        self.url = url
        self.bundleUrl = bundleUrl
        self.windowIsActive = windowIsActive
    }
}

fileprivate enum InitArgKind: UInt8 {
    case data = 1
    case contentSize = 2
    case appearance = 3
    case proxy = 4
    case proxyAuth = 5
    case url = 6
    case bundleUrl = 7
    case windowIsActive = 8
}

/// Messages from Browser to Content on the content socket
enum BrowserToContentMessage {
    case initializeContent(args: InitializeContentArguments)
    case displayLinkFired(frameNumber: UInt64, targetTimestamp: Double)
    case displayLinkCallbackRegistered(callbackID: UUID, browserCallbackID: UUID)
    case resizeContent(size: CGSize)
    case mouseDown(point: CGPoint, modifierFlags: NSEvent.ModifierFlags, clickCount: Int)
    case mouseDragged(point: CGPoint, modifierFlags: NSEvent.ModifierFlags)
    case mouseUp(point: CGPoint, modifierFlags: NSEvent.ModifierFlags)
    case mouseMoved(point: CGPoint, modifierFlags: NSEvent.ModifierFlags)
    case rightMouseDown(point: CGPoint, modifierFlags: NSEvent.ModifierFlags, clickCount: Int)
    case rightMouseUp(point: CGPoint, modifierFlags: NSEvent.ModifierFlags)
    case scrollWheelEvent(point: CGPoint,
                          delta: CGPoint,
                          modifierFlags: NSEvent.ModifierFlags,
                          phase: NSEvent.Phase,
                          momentumPhase: NSEvent.Phase,
                          hasPreciseScrollingDeltas: Bool)
    case keyDown(keyCode: UInt16,
                 characters: String,
                 charactersIgnoringModifiers: String,
                 modifierFlags: NSEvent.ModifierFlags,
                 isARepeat: Bool)
    case keyUp(keyCode: UInt16,
               characters: String,
               charactersIgnoringModifiers: String,
               modifierFlags: NSEvent.ModifierFlags,
               isARepeat: Bool)
    case magnification(surfaceID: UInt32, magnification: CGFloat, location: CGPoint, scrollOffset: CGPoint)
    case magnificationEnded(surfaceID: UInt32, magnification: CGFloat, location: CGPoint, scrollOffset: CGPoint)
    case quickLook(point: CGPoint)
    case imageWithSystemSymbolName(requestID: UUID,
                                   alphaMaskData: Data?,
                                   width: UInt32,
                                   height: UInt32,
                                   bytesPerRow: UInt32,
                                   success: Bool,
                                   errorMessage: String?)
    case textInput(text: String,
                   hasReplacementRange: Bool,
                   replacementLocation: UInt64,
                   replacementLength: UInt64)
    case setMarkedText(text: String,
                       selectedLocation: UInt64,
                       selectedLength: UInt64,
                       hasReplacementRange: Bool,
                       replacementLocation: UInt64,
                       replacementLength: UInt64)
    case unmarkText
    case textInputFocus(fieldID: UUID, hasFocus: Bool)
    case textCommand(command: String)
    case setCursorPosition(fieldID: UUID, position: UInt64, modifySelection: Bool)
    case systemAppearanceUpdate(appearance: NSAppearance)
    case windowActiveUpdate(isActive: Bool)
    case viewFocusChanged(isFocused: Bool)
    case copySelectedPasteboardRequest(requestID: UUID)
    case pasteboardContentDelivered(items: [OuterframeContentPasteboardItem])
    case accessibilitySnapshotRequest(requestID: UUID)
    case shutdown

    func encode() throws -> Data {
        switch self {
        case .initializeContent(let arguments):
            var encodedArguments: [Data] = []

            if let data = arguments.data {
                var argPayload = OffsetPayloadBuilder(referenceBaseOffset: 0)
                argPayload.append(uint8: InitArgKind.data.rawValue)
                try argPayload.append(dataReference: data)
                encodedArguments.append(try argPayload.finalize())
            }

            if let contentSize = arguments.contentSize {
                var argPayload = Data(capacity: 1 + 16)
                argPayload.append(uint8: InitArgKind.contentSize.rawValue)
                argPayload.append(float64: contentSize.width)
                argPayload.append(float64: contentSize.height)
                encodedArguments.append(argPayload)
            }

            if let appearance = arguments.appearance {
                var argPayload = OffsetPayloadBuilder(referenceBaseOffset: 0)
                argPayload.append(uint8: InitArgKind.appearance.rawValue)
                let appearanceData = try NSKeyedArchiver.archivedData(withRootObject: appearance, requiringSecureCoding: true)
                try argPayload.append(dataReference: appearanceData)
                encodedArguments.append(try argPayload.finalize())
            }

            if let proxy = arguments.proxy {
                var argPayload = OffsetPayloadBuilder(referenceBaseOffset: 0)
                argPayload.append(uint8: InitArgKind.proxy.rawValue)
                try argPayload.append(stringReference: proxy.host)
                argPayload.append(uint16: proxy.port)
                encodedArguments.append(try argPayload.finalize())

                if proxy.username != nil || proxy.password != nil {
                    var authPayload = OffsetPayloadBuilder(referenceBaseOffset: 0)
                    authPayload.append(uint8: InitArgKind.proxyAuth.rawValue)
                    var flags: UInt8 = 0
                    if let username = proxy.username {
                        flags |= 1 << 0
                    }
                    if let password = proxy.password {
                        flags |= 1 << 1
                    }
                    authPayload.append(uint8: flags)
                    try authPayload.append(stringReference: proxy.username ?? "")
                    try authPayload.append(stringReference: proxy.password ?? "")
                    encodedArguments.append(try authPayload.finalize())
                }
            }

            if let url = arguments.url {
                var argPayload = OffsetPayloadBuilder(referenceBaseOffset: 0)
                argPayload.append(uint8: InitArgKind.url.rawValue)
                try argPayload.append(stringReference: url)
                encodedArguments.append(try argPayload.finalize())
            }

            if let bundleUrl = arguments.bundleUrl {
                var argPayload = OffsetPayloadBuilder(referenceBaseOffset: 0)
                argPayload.append(uint8: InitArgKind.bundleUrl.rawValue)
                try argPayload.append(stringReference: bundleUrl)
                encodedArguments.append(try argPayload.finalize())
            }

            if let windowIsActive = arguments.windowIsActive {
                var argPayload = Data(capacity: 2)
                argPayload.append(uint8: InitArgKind.windowIsActive.rawValue)
                argPayload.append(uint8: windowIsActive ? 1 << 0 : 0)
                encodedArguments.append(argPayload)
            }

            var payload = OffsetPayloadBuilder()
            payload.append(uint16: UInt16(min(encodedArguments.count, Int(UInt16.max))))

            for encodedArgument in encodedArguments {
                try payload.append(dataReference: encodedArgument)
            }

            return makeBrowserToContentFrame(type: .initializeContent, payload: try payload.finalize())

        case .displayLinkFired(let frameNumber, let targetTimestamp):
            var payload = Data(capacity: 16)
            payload.append(uint64: frameNumber)
            payload.append(float64: targetTimestamp)
            return makeBrowserToContentFrame(type: .displayLinkFired, payload: payload)

        case .displayLinkCallbackRegistered(let callbackID, let browserCallbackID):
            var payload = Data(capacity: 16 * 2)
            payload.append(uuid: callbackID)
            payload.append(uuid: browserCallbackID)
            return makeBrowserToContentFrame(type: .displayLinkCallbackRegistered, payload: payload)

        case .resizeContent(let size):
            var payload = Data(capacity: 8 + 8)
            payload.append(float64: size.width)
            payload.append(float64: size.height)
            return makeBrowserToContentFrame(type: .resizeContent, payload: payload)

        case .mouseDown(let point, let modifierFlags, let clickCount):
            return makeMouseEventFrame(type: .mouseDown, point: point,
                                       modifierFlags: modifierFlags, clickCount: clickCount)

        case .mouseDragged(let point, let modifierFlags):
            return makeMouseEventFrame(type: .mouseDragged, point: point, modifierFlags: modifierFlags)

        case .mouseUp(let point, let modifierFlags):
            return makeMouseEventFrame(type: .mouseUp, point: point, modifierFlags: modifierFlags)

        case .mouseMoved(let point, let modifierFlags):
            return makeMouseEventFrame(type: .mouseMoved, point: point, modifierFlags: modifierFlags)

        case .rightMouseDown(let point, let modifierFlags, let clickCount):
            return makeMouseEventFrame(type: .rightMouseDown, point: point,
                                       modifierFlags: modifierFlags, clickCount: clickCount)

        case .rightMouseUp(let point, let modifierFlags):
            return makeMouseEventFrame(type: .rightMouseUp, point: point, modifierFlags: modifierFlags)

        case .scrollWheelEvent(let point,
                               let delta,
                               let modifierFlags,
                               let phase,
                               let momentumPhase,
                               let hasPreciseScrollingDeltas):
            var payload = Data(capacity: 8 * 4 + 8 + 4 + 4 + 1)
            payload.append(float64: point.x)
            payload.append(float64: point.y)
            payload.append(float64: delta.x)
            payload.append(float64: delta.y)
            payload.append(uint64: UInt64(modifierFlags.rawValue))
            payload.append(uint32: UInt32(truncatingIfNeeded: phase.rawValue))
            payload.append(uint32: UInt32(truncatingIfNeeded: momentumPhase.rawValue))
            var flags: UInt8 = 0
            if hasPreciseScrollingDeltas { flags |= 1 << 0 }
            payload.append(uint8: flags)
            return makeBrowserToContentFrame(type: .scrollWheelEvent, payload: payload)

        case .keyDown(let keyCode, let characters, let charactersIgnoringModifiers, let modifierFlags, let isARepeat):
            var payload = OffsetPayloadBuilder()
            payload.append(uint16: keyCode)
            try payload.append(stringReference: characters)
            try payload.append(stringReference: charactersIgnoringModifiers)
            payload.append(uint64: UInt64(modifierFlags.rawValue))
            payload.append(uint8: isARepeat ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .keyDown, payload: try payload.finalize())

        case .keyUp(let keyCode, let characters, let charactersIgnoringModifiers, let modifierFlags, let isARepeat):
            var payload = OffsetPayloadBuilder()
            payload.append(uint16: keyCode)
            try payload.append(stringReference: characters)
            try payload.append(stringReference: charactersIgnoringModifiers)
            payload.append(uint64: UInt64(modifierFlags.rawValue))
            payload.append(uint8: isARepeat ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .keyUp, payload: try payload.finalize())

        case .magnification(let surfaceID, let magnification, let location, let scrollOffset):
            var payload = Data()
            payload.append(uint32: surfaceID)
            payload.append(float64: magnification)
            payload.append(float64: location.x)
            payload.append(float64: location.y)
            payload.append(float64: scrollOffset.x)
            payload.append(float64: scrollOffset.y)
            return makeBrowserToContentFrame(type: .magnification, payload: payload)

        case .magnificationEnded(let surfaceID, let magnification, let location, let scrollOffset):
            var payload = Data()
            payload.append(uint32: surfaceID)
            payload.append(float64: magnification)
            payload.append(float64: location.x)
            payload.append(float64: location.y)
            payload.append(float64: scrollOffset.x)
            payload.append(float64: scrollOffset.y)
            return makeBrowserToContentFrame(type: .magnificationEnded, payload: payload)

        case .quickLook(let point):
            var payload = Data(capacity: 8 + 8)
            payload.append(float64: point.x)
            payload.append(float64: point.y)
            return makeBrowserToContentFrame(type: .quickLook, payload: payload)

        case .imageWithSystemSymbolName(let requestID, let alphaMaskData, let width, let height, let bytesPerRow, let success, let errorMessage):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            payload.append(uint32: width)
            payload.append(uint32: height)
            payload.append(uint32: bytesPerRow)
            var flags: UInt8 = 0
            if success { flags |= 1 << 0 }
            if alphaMaskData != nil { flags |= 1 << 1 }
            if errorMessage != nil { flags |= 1 << 2 }
            payload.append(uint8: flags)
            try payload.append(dataReference: alphaMaskData ?? Data())
            try payload.append(stringReference: errorMessage ?? "")
            return makeBrowserToContentFrame(type: .imageWithSystemSymbolName, payload: try payload.finalize())

        case .textInput(let text, let hasReplacementRange, let replacementLocation, let replacementLength):
            var payload = OffsetPayloadBuilder()
            try payload.append(stringReference: text)
            payload.append(uint8: hasReplacementRange ? 1 << 0 : 0)
            payload.append(uint64: replacementLocation)
            payload.append(uint64: replacementLength)
            return makeBrowserToContentFrame(type: .textInput, payload: try payload.finalize())

        case .setMarkedText(let text, let selectedLocation, let selectedLength, let hasReplacementRange, let replacementLocation, let replacementLength):
            var payload = OffsetPayloadBuilder()
            try payload.append(stringReference: text)
            payload.append(uint64: selectedLocation)
            payload.append(uint64: selectedLength)
            payload.append(uint8: hasReplacementRange ? 1 << 0 : 0)
            payload.append(uint64: replacementLocation)
            payload.append(uint64: replacementLength)
            return makeBrowserToContentFrame(type: .setMarkedText, payload: try payload.finalize())

        case .unmarkText:
            return makeBrowserToContentFrame(type: .unmarkText, payload: Data())

        case .textInputFocus(let fieldID, let hasFocus):
            var payload = Data()
            payload.append(uuid: fieldID)
            payload.append(uint8: hasFocus ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .textInputFocus, payload: payload)

        case .textCommand(let command):
            var payload = OffsetPayloadBuilder()
            try payload.append(stringReference: command)
            return makeBrowserToContentFrame(type: .textCommand, payload: try payload.finalize())

        case .setCursorPosition(let fieldID, let position, let modifySelection):
            var payload = Data()
            payload.append(uuid: fieldID)
            payload.append(uint64: position)
            payload.append(uint8: modifySelection ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .setCursorPosition, payload: payload)

        case .systemAppearanceUpdate(let appearance):
            var payload = OffsetPayloadBuilder()
            let appearanceData = try NSKeyedArchiver.archivedData(withRootObject: appearance, requiringSecureCoding: true)
            try payload.append(dataReference: appearanceData)
            return makeBrowserToContentFrame(type: .systemAppearanceUpdate, payload: try payload.finalize())

        case .windowActiveUpdate(let isActive):
            var payload = Data(capacity: 1)
            payload.append(uint8: isActive ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .windowActiveUpdate, payload: payload)

        case .viewFocusChanged(let isFocused):
            var payload = Data(capacity: 1)
            payload.append(uint8: isFocused ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .viewFocusChanged, payload: payload)

        case .copySelectedPasteboardRequest(let requestID):
            var payload = Data(capacity: 16)
            payload.append(uuid: requestID)
            return makeBrowserToContentFrame(type: .copySelectedPasteboardRequest, payload: payload)

        case .pasteboardContentDelivered(let items):
            var payload = OffsetPayloadBuilder()
            let clampedCount = UInt16(min(items.count, Int(UInt16.max)))
            payload.append(uint16: clampedCount)
            for item in items.prefix(Int(clampedCount)) {
                try payload.append(stringReference: item.typeIdentifier)
                try payload.append(dataReference: item.data)
            }
            return makeBrowserToContentFrame(type: .pasteboardContentDelivered, payload: try payload.finalize())

        case .accessibilitySnapshotRequest(let requestID):
            var payload = Data(capacity: 16)
            payload.append(uuid: requestID)
            return makeBrowserToContentFrame(type: .accessibilitySnapshotRequest, payload: payload)

        case .shutdown:
            return makeBrowserToContentFrame(type: .shutdown, payload: Data())
        }
    }

    static func decode(message: Data) throws -> BrowserToContentMessage {
        var cursor = DataCursor(message)
        guard let typeRaw = cursor.readUInt16() else {
            throw OuterframeContentSocketMessageError.truncatedPayload
        }
        guard let type = BrowserToContentMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentSocketMessageError.unknownType(typeRaw)
        }

        switch type {
        case .initializeContent:
            guard let argCount = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }

            var arguments = InitializeContentArguments()
            var proxyUsername: String?
            var proxyPassword: String?

            for _ in 0..<argCount {
                guard let argData = cursor.readDataReference() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }

                var argCursor = DataCursor(argData)
                guard let kindRaw = argCursor.readUInt8() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }

                guard let kind = InitArgKind(rawValue: kindRaw) else {
                    continue
                }

                switch kind {
                case .data:
                    guard let data = argCursor.readDataReference() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.data = data

                case .contentSize:
                    guard let width = argCursor.readFloat64(),
                          let height = argCursor.readFloat64() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.contentSize = CGSize(width: width, height: height)

                case .appearance:
                    guard let appearanceData = argCursor.readDataReference(),
                          let decoded = try? NSKeyedUnarchiver.unarchivedObject(ofClass: NSAppearance.self, from: appearanceData) else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.appearance = decoded

                case .proxy:
                    guard let proxyHost = argCursor.readStringReference(),
                          let proxyPort = argCursor.readUInt16() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.proxy = InitializeContentProxy(host: proxyHost,
                                                             port: proxyPort,
                                                             username: proxyUsername,
                                                             password: proxyPassword)

                case .proxyAuth:
                    guard let flags = argCursor.readUInt8(),
                          let username = argCursor.readStringReference(),
                          let password = argCursor.readStringReference() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    if flags & (1 << 0) != 0 {
                        proxyUsername = username
                    } else {
                        proxyUsername = nil
                    }

                    if flags & (1 << 1) != 0 {
                        proxyPassword = password
                    } else {
                        proxyPassword = nil
                    }

                    if var proxy = arguments.proxy {
                        proxy.username = proxyUsername
                        proxy.password = proxyPassword
                        arguments.proxy = proxy
                    }

                case .url:
                    guard let url = argCursor.readStringReference() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.url = url

                case .bundleUrl:
                    guard let bundleUrl = argCursor.readStringReference() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.bundleUrl = bundleUrl

                case .windowIsActive:
                    guard let windowIsActiveRaw = argCursor.readUInt8() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.windowIsActive = windowIsActiveRaw & (1 << 0) != 0
                }
            }

            return .initializeContent(args: arguments)

        case .displayLinkFired:
            guard let frameNumber = cursor.readUInt64(),
                  let timestampBits = cursor.readUInt64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let timestamp = Double(bitPattern: timestampBits)
            return .displayLinkFired(frameNumber: frameNumber, targetTimestamp: timestamp)

        case .displayLinkCallbackRegistered:
            guard let callbackID = cursor.readUUID(),
                  let browserCallbackID = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .displayLinkCallbackRegistered(callbackID: callbackID, browserCallbackID: browserCallbackID)

        case .resizeContent:
            guard let width = cursor.readFloat64(),
                  let height = cursor.readFloat64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .resizeContent(size: CGSize(width: width, height: height))

        case .mouseDown:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: true)
            return .mouseDown(point: event.point,
                              modifierFlags: event.modifierFlags, clickCount: event.clickCount)

        case .mouseDragged:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .mouseDragged(point: event.point, modifierFlags: event.modifierFlags)

        case .mouseUp:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .mouseUp(point: event.point, modifierFlags: event.modifierFlags)

        case .mouseMoved:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .mouseMoved(point: event.point, modifierFlags: event.modifierFlags)

        case .rightMouseDown:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: true)
            return .rightMouseDown(point: event.point,
                                   modifierFlags: event.modifierFlags, clickCount: event.clickCount)

        case .rightMouseUp:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .rightMouseUp(point: event.point, modifierFlags: event.modifierFlags)

        case .scrollWheelEvent:
            guard let x = cursor.readFloat64(),
                  let y = cursor.readFloat64(),
                  let deltaX = cursor.readFloat64(),
                  let deltaY = cursor.readFloat64(),
                  let modifierFlags = cursor.readUInt64(),
                  let phaseRaw = cursor.readUInt32(),
                  let momentumPhaseRaw = cursor.readUInt32(),
                  let flags = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .scrollWheelEvent(point: CGPoint(x: x, y: y),
                                     delta: CGPoint(x: deltaX, y: deltaY),
                                     modifierFlags: NSEvent.ModifierFlags(rawValue: UInt(modifierFlags)),
                                     phase: NSEvent.Phase(rawValue: UInt(phaseRaw)),
                                     momentumPhase: NSEvent.Phase(rawValue: UInt(momentumPhaseRaw)),
                                     hasPreciseScrollingDeltas: flags & (1 << 0) != 0)

        case .keyDown:
            guard let keyCode = cursor.readUInt16(),
                  let characters = cursor.readStringReference(),
                  let charactersIgnoringModifiers = cursor.readStringReference(),
                  let modifierFlags = cursor.readUInt64(),
                  let flags = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .keyDown(keyCode: keyCode, characters: characters,
                            charactersIgnoringModifiers: charactersIgnoringModifiers,
                            modifierFlags: NSEvent.ModifierFlags(rawValue: UInt(modifierFlags)),
                            isARepeat: flags & (1 << 0) != 0)

        case .keyUp:
            guard let keyCode = cursor.readUInt16(),
                  let characters = cursor.readStringReference(),
                  let charactersIgnoringModifiers = cursor.readStringReference(),
                  let modifierFlags = cursor.readUInt64(),
                  let flags = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .keyUp(keyCode: keyCode, characters: characters,
                          charactersIgnoringModifiers: charactersIgnoringModifiers,
                          modifierFlags: NSEvent.ModifierFlags(rawValue: UInt(modifierFlags)),
                          isARepeat: flags & (1 << 0) != 0)

        case .magnification:
            guard let surfaceID = cursor.readUInt32(),
                  let magnification = cursor.readFloat64(),
                  let x = cursor.readFloat64(),
                  let y = cursor.readFloat64(),
                  let scrollX = cursor.readFloat64(),
                  let scrollY = cursor.readFloat64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .magnification(surfaceID: surfaceID, magnification: magnification,
                                  location: CGPoint(x: x, y: y),
                                  scrollOffset: CGPoint(x: scrollX, y: scrollY))

        case .magnificationEnded:
            guard let surfaceID = cursor.readUInt32(),
                  let magnification = cursor.readFloat64(),
                  let x = cursor.readFloat64(),
                  let y = cursor.readFloat64(),
                  let scrollX = cursor.readFloat64(),
                  let scrollY = cursor.readFloat64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .magnificationEnded(surfaceID: surfaceID, magnification: magnification,
                                       location: CGPoint(x: x, y: y),
                                       scrollOffset: CGPoint(x: scrollX, y: scrollY))

        case .quickLook:
            guard let x = cursor.readFloat64(),
                  let y = cursor.readFloat64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .quickLook(point: CGPoint(x: x, y: y))

        case .imageWithSystemSymbolName:
            guard let requestID = cursor.readUUID(),
                  let width = cursor.readUInt32(),
                  let height = cursor.readUInt32(),
                  let bytesPerRow = cursor.readUInt32(),
                  let flags = cursor.readUInt8(),
                  let alphaMaskDataReference = cursor.readDataReference(),
                  let errorMessageReference = cursor.readStringReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }

            let alphaMaskData = flags & (1 << 1) != 0 ? alphaMaskDataReference : nil
            let errorMessage = flags & (1 << 2) != 0 ? errorMessageReference : nil

            return .imageWithSystemSymbolName(requestID: requestID, alphaMaskData: alphaMaskData,
                                     width: width, height: height, bytesPerRow: bytesPerRow,
                                     success: flags & (1 << 0) != 0, errorMessage: errorMessage)

        case .textInput:
            guard let text = cursor.readStringReference(),
                  let flags = cursor.readUInt8(),
                  let replacementLocation = cursor.readUInt64(),
                  let replacementLength = cursor.readUInt64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .textInput(text: text, hasReplacementRange: flags & (1 << 0) != 0,
                              replacementLocation: replacementLocation,
                              replacementLength: replacementLength)

        case .setMarkedText:
            guard let text = cursor.readStringReference(),
                  let selectedLocation = cursor.readUInt64(),
                  let selectedLength = cursor.readUInt64(),
                  let flags = cursor.readUInt8(),
                  let replacementLocation = cursor.readUInt64(),
                  let replacementLength = cursor.readUInt64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .setMarkedText(text: text, selectedLocation: selectedLocation,
                                  selectedLength: selectedLength,
                                  hasReplacementRange: flags & (1 << 0) != 0,
                                  replacementLocation: replacementLocation,
                                  replacementLength: replacementLength)

        case .unmarkText:
            return .unmarkText

        case .textInputFocus:
            guard let fieldID = cursor.readUUID(),
                  let flags = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .textInputFocus(fieldID: fieldID, hasFocus: flags & (1 << 0) != 0)

        case .textCommand:
            guard let command = cursor.readStringReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .textCommand(command: command)

        case .setCursorPosition:
            guard let fieldID = cursor.readUUID(),
                  let position = cursor.readUInt64(),
                  let flags = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .setCursorPosition(fieldID: fieldID, position: position,
                                      modifySelection: flags & (1 << 0) != 0)

        case .systemAppearanceUpdate:
            guard let appearanceData = cursor.readDataReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let appearance = (try? NSKeyedUnarchiver.unarchivedObject(ofClass: NSAppearance.self, from: appearanceData))
                ?? NSAppearance.currentDrawing()
            return .systemAppearanceUpdate(appearance: appearance)

        case .windowActiveUpdate:
            guard let raw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .windowActiveUpdate(isActive: raw & (1 << 0) != 0)

        case .viewFocusChanged:
            guard let raw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .viewFocusChanged(isFocused: raw & (1 << 0) != 0)

        case .copySelectedPasteboardRequest:
            guard let requestID = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .copySelectedPasteboardRequest(requestID: requestID)

        case .pasteboardContentDelivered:
            guard let count = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var items: [OuterframeContentPasteboardItem] = []
            items.reserveCapacity(Int(count))
            for _ in 0..<count {
                guard let identifier = cursor.readStringReference(),
                      let data = cursor.readDataReference() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                items.append(OuterframeContentPasteboardItem(typeIdentifier: identifier, data: data))
            }
            return .pasteboardContentDelivered(items: items)

        case .accessibilitySnapshotRequest:
            guard let requestID = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .accessibilitySnapshotRequest(requestID: requestID)

        case .shutdown:
            return .shutdown
        }
    }
}

/// Messages from Content to Browser on the content socket
enum ContentToBrowserMessage {
    case startDisplayLink(callbackID: UUID)
    case stopDisplayLink(browserCallbackID: UUID)
    case cursorUpdate(cursorType: UInt8)
    case inputModeUpdate(inputMode: UInt8)
    case showContextMenu(attributedTextData: Data, locationX: CGFloat, locationY: CGFloat)
    case showDefinition(attributedTextData: Data, locationX: CGFloat, locationY: CGFloat)
    case getImageWithSystemSymbolName(requestID: UUID,
                                      symbolName: String,
                                      pointSize: CGFloat,
                                      weight: Float64,
                                      scale: CGFloat)
    case textCursorUpdate(cursors: [OuterframeContentTextCursorSnapshot])
    case copySelectedPasteboardResponse(requestID: UUID, items: [OuterframeContentPasteboardItem])
    case openNewWindow(url: String, displayString: String?, preferredSize: CGSize?)
    case setPasteboardCapabilities(canCopy: Bool, canCut: Bool, pasteboardTypes: [String])
    case accessibilitySnapshotResponse(requestID: UUID, snapshotData: Data?)
    case accessibilityTreeChanged(notificationMask: UInt8)
    case hapticFeedback(style: UInt8)

    func encode() throws -> Data {
        switch self {
        case .startDisplayLink(let callbackID):
            var payload = Data(capacity: 16)
            payload.append(uuid: callbackID)
            return makeContentToBrowserFrame(type: .startDisplayLink, payload: payload)

        case .stopDisplayLink(let browserCallbackID):
            var payload = Data(capacity: 16)
            payload.append(uuid: browserCallbackID)
            return makeContentToBrowserFrame(type: .stopDisplayLink, payload: payload)

        case .cursorUpdate(let cursorType):
            var payload = Data(capacity: 1)
            payload.append(uint8: cursorType)
            return makeContentToBrowserFrame(type: .cursorUpdate, payload: payload)

        case .inputModeUpdate(let inputMode):
            var payload = Data(capacity: 1)
            payload.append(uint8: inputMode)
            return makeContentToBrowserFrame(type: .inputModeUpdate, payload: payload)

        case .showContextMenu(let attributedTextData, let locationX, let locationY):
            var payload = OffsetPayloadBuilder()
            payload.append(float64: locationX)
            payload.append(float64: locationY)
            try payload.append(dataReference: attributedTextData)
            return makeContentToBrowserFrame(type: .showContextMenu, payload: try payload.finalize())

        case .showDefinition(let attributedTextData, let locationX, let locationY):
            var payload = OffsetPayloadBuilder()
            payload.append(float64: locationX)
            payload.append(float64: locationY)
            try payload.append(dataReference: attributedTextData)
            return makeContentToBrowserFrame(type: .showDefinition, payload: try payload.finalize())

        case .getImageWithSystemSymbolName(let requestID, let symbolName, let pointSize, let weight, let scale):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            try payload.append(stringReference: symbolName)
            payload.append(float64: pointSize)
            payload.append(float64: weight)
            payload.append(float64: scale)
            return makeContentToBrowserFrame(type: .getImageWithSystemSymbolName, payload: try payload.finalize())

        case .textCursorUpdate(let cursors):
            var payload = Data()
            let countValue = UInt32(max(0, min(cursors.count, Int(UInt32.max))))
            payload.append(uint32: countValue)
            for cursor in cursors {
                payload.append(uuid: cursor.fieldID)
                payload.append(float64: cursor.rect.origin.x)
                payload.append(float64: cursor.rect.origin.y)
                payload.append(float64: cursor.rect.size.width)
                payload.append(float64: cursor.rect.size.height)
                payload.append(uint8: cursor.visible ? 1 << 0 : 0)
            }
            return makeContentToBrowserFrame(type: .textCursorUpdate, payload: payload)

        case .copySelectedPasteboardResponse(let requestID, let items):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            let clampedCount = UInt16(min(items.count, Int(UInt16.max)))
            payload.append(uint16: clampedCount)
            for item in items.prefix(Int(clampedCount)) {
                try payload.append(stringReference: item.typeIdentifier)
                try payload.append(dataReference: item.data)
            }
            return makeContentToBrowserFrame(type: .copySelectedPasteboardResponse, payload: try payload.finalize())

        case .openNewWindow(let url, let displayString, let preferredSize):
            var payload = OffsetPayloadBuilder()
            try payload.append(stringReference: url)
            var flags: UInt8 = 0
            if displayString != nil { flags |= 1 << 0 }
            if preferredSize != nil { flags |= 1 << 1 }
            payload.append(uint8: flags)
            try payload.append(stringReference: displayString ?? "")
            payload.append(float64: preferredSize.map { Float64($0.width) } ?? 0)
            payload.append(float64: preferredSize.map { Float64($0.height) } ?? 0)
            return makeContentToBrowserFrame(type: .openNewWindow, payload: try payload.finalize())

        case .setPasteboardCapabilities(let canCopy, let canCut, let pasteboardTypes):
            var payload = OffsetPayloadBuilder()
            var flags: UInt8 = 0
            if canCopy { flags |= 1 << 0 }
            if canCut { flags |= 1 << 1 }
            payload.append(uint8: flags)
            let clampedCount = UInt16(min(pasteboardTypes.count, Int(UInt16.max)))
            payload.append(uint16: clampedCount)
            for identifier in pasteboardTypes.prefix(Int(clampedCount)) {
                try payload.append(stringReference: identifier)
            }
            return makeContentToBrowserFrame(type: .editingCapabilitiesUpdate, payload: try payload.finalize())

        case .accessibilitySnapshotResponse(let requestID, let snapshotData):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            payload.append(uint8: snapshotData != nil ? 1 << 0 : 0)
            try payload.append(dataReference: snapshotData ?? Data())
            return makeContentToBrowserFrame(type: .accessibilitySnapshotResponse, payload: try payload.finalize())

        case .accessibilityTreeChanged(let notificationMask):
            var payload = Data(capacity: 1)
            payload.append(uint8: notificationMask)
            return makeContentToBrowserFrame(type: .accessibilityTreeChanged, payload: payload)

        case .hapticFeedback(let style):
            var payload = Data(capacity: 1)
            payload.append(uint8: style)
            return makeContentToBrowserFrame(type: .hapticFeedback, payload: payload)
        }
    }

    static func decode(message: Data) throws -> ContentToBrowserMessage {
        var cursor = DataCursor(message)
        guard let typeRaw = cursor.readUInt16() else {
            throw OuterframeContentSocketMessageError.truncatedPayload
        }
        guard let type = ContentToBrowserMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentSocketMessageError.unknownType(typeRaw)
        }

        switch type {
        case .startDisplayLink:
            guard let callbackID = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .startDisplayLink(callbackID: callbackID)

        case .stopDisplayLink:
            guard let browserCallbackID = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .stopDisplayLink(browserCallbackID: browserCallbackID)

        case .cursorUpdate:
            guard let cursorType = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .cursorUpdate(cursorType: cursorType)

        case .inputModeUpdate:
            guard let inputMode = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .inputModeUpdate(inputMode: inputMode)

        case .showContextMenu:
            guard let locationX = cursor.readFloat64(),
                  let locationY = cursor.readFloat64(),
                  let attributedTextData = cursor.readDataReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .showContextMenu(attributedTextData: attributedTextData,
                                    locationX: locationX, locationY: locationY)

        case .showDefinition:
            guard let locationX = cursor.readFloat64(),
                  let locationY = cursor.readFloat64(),
                  let attributedTextData = cursor.readDataReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .showDefinition(attributedTextData: attributedTextData,
                                   locationX: locationX, locationY: locationY)

        case .getImageWithSystemSymbolName:
            guard let requestID = cursor.readUUID(),
                  let symbolName = cursor.readStringReference(),
                  let pointSize = cursor.readFloat64(),
                  let weight = cursor.readFloat64(),
                  let scale = cursor.readFloat64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .getImageWithSystemSymbolName(requestID: requestID, symbolName: symbolName,
                                    pointSize: pointSize, weight: weight, scale: scale)

        case .textCursorUpdate:
            guard let cursorCount = cursor.readUInt32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var entries: [OuterframeContentTextCursorSnapshot] = []
            entries.reserveCapacity(Int(cursorCount))
            for _ in 0..<cursorCount {
                guard let fieldID = cursor.readUUID(),
                      let rectX = cursor.readFloat64(),
                      let rectY = cursor.readFloat64(),
                      let rectWidth = cursor.readFloat64(),
                      let rectHeight = cursor.readFloat64(),
                      let flags = cursor.readUInt8() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                entries.append(OuterframeContentTextCursorSnapshot(fieldID: fieldID,
                                                                   rect: CGRect(x: rectX,
                                                                                y: rectY,
                                                                                width: rectWidth,
                                                                                height: rectHeight),
                                                                   visible: flags & (1 << 0) != 0))
            }
            return .textCursorUpdate(cursors: entries)

        case .copySelectedPasteboardResponse:
            guard let requestID = cursor.readUUID(),
                  let count = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var items: [OuterframeContentPasteboardItem] = []
            items.reserveCapacity(Int(count))
            for _ in 0..<count {
                guard let identifier = cursor.readStringReference(),
                      let data = cursor.readDataReference() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                items.append(OuterframeContentPasteboardItem(typeIdentifier: identifier, data: data))
            }
            return .copySelectedPasteboardResponse(requestID: requestID, items: items)

        case .editingCapabilitiesUpdate:
            guard let flags = cursor.readUInt8(),
                  let count = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var identifiers: [String] = []
            identifiers.reserveCapacity(Int(count))
            for _ in 0..<count {
                guard let identifier = cursor.readStringReference() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                identifiers.append(identifier)
            }
            return .setPasteboardCapabilities(canCopy: flags & (1 << 0) != 0,
                                              canCut: flags & (1 << 1) != 0,
                                              pasteboardTypes: identifiers)

        case .accessibilitySnapshotResponse:
            guard let requestID = cursor.readUUID(),
                  let flags = cursor.readUInt8(),
                  let payload = cursor.readDataReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let snapshotData = flags & (1 << 0) != 0 ? payload : nil
            return .accessibilitySnapshotResponse(requestID: requestID, snapshotData: snapshotData)

        case .accessibilityTreeChanged:
            guard let mask = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .accessibilityTreeChanged(notificationMask: mask)

        case .hapticFeedback:
            guard let style = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .hapticFeedback(style: style)

        case .openNewWindow:
            guard let url = cursor.readStringReference(),
                  let flags = cursor.readUInt8(),
                  let displayStringReference = cursor.readStringReference(),
                  let width = cursor.readFloat64(),
                  let height = cursor.readFloat64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let displayString = flags & (1 << 0) != 0 ? displayStringReference : nil
            let preferredSize = flags & (1 << 1) != 0 ? CGSize(width: width, height: height) : nil
            return .openNewWindow(url: url, displayString: displayString,
                                  preferredSize: preferredSize)
        }
    }
}

// MARK: - Supporting Types

struct OuterframeContentTextCursorSnapshot: Sendable {
    let fieldID: UUID
    let rect: CGRect
    let visible: Bool
}

struct OuterframeContentPasteboardItem: Sendable {
    let typeIdentifier: String
    let data: Data

    init(typeIdentifier: String, data: Data) {
        self.typeIdentifier = typeIdentifier
        self.data = data
    }
}

enum OuterframeContentSocketMessageError: Error {
    case unknownType(UInt16)
    case truncatedPayload
    case encodingFailure(String)
}

// MARK: - Message Kind Enums

private enum BrowserToContentMessageKind: UInt16 {
    case initializeContent = 1000
    case resizeContent = 1001
    case shutdown = 1002
    case displayLinkFired = 1003
    case displayLinkCallbackRegistered = 1004
    case systemAppearanceUpdate = 1005
    case windowActiveUpdate = 1006
    case viewFocusChanged = 1007
    case mouseDown = 1008
    case mouseDragged = 1009
    case mouseUp = 1010
    case mouseMoved = 1011
    case rightMouseDown = 1012
    case rightMouseUp = 1013
    case scrollWheelEvent = 1014
    case keyDown = 1015
    case keyUp = 1016
    case magnification = 1017
    case magnificationEnded = 1018
    case quickLook = 1019
    case textInput = 1020
    case setMarkedText = 1021
    case unmarkText = 1022
    case textInputFocus = 1023
    case textCommand = 1024
    case setCursorPosition = 1025
    case imageWithSystemSymbolName = 1026
    case copySelectedPasteboardRequest = 1027
    case pasteboardContentDelivered = 1028
    case accessibilitySnapshotRequest = 1029
}

private enum ContentToBrowserMessageKind: UInt16 {
    case startDisplayLink = 2000
    case stopDisplayLink = 2001
    case cursorUpdate = 2002
    case inputModeUpdate = 2003
    case textCursorUpdate = 2004
    case showContextMenu = 2005
    case showDefinition = 2006
    case getImageWithSystemSymbolName = 2007
    case hapticFeedback = 2008
    case copySelectedPasteboardResponse = 2009
    case editingCapabilitiesUpdate = 2010
    case accessibilitySnapshotResponse = 2011
    case accessibilityTreeChanged = 2012
    case openNewWindow = 2013
}

// MARK: - Frame Helpers

private func makeBrowserToContentFrame(type: BrowserToContentMessageKind, payload: Data) -> Data {
    let messageLength = OuterframeContentSocketMessageTypeLength + payload.count
    var frame = Data(capacity: OuterframeContentSocketHeaderLength + messageLength)
    frame.append(uint32: UInt32(messageLength))
    frame.append(uint16: type.rawValue)
    frame.append(payload)
    return frame
}

private func makeContentToBrowserFrame(type: ContentToBrowserMessageKind, payload: Data) -> Data {
    let messageLength = OuterframeContentSocketMessageTypeLength + payload.count
    var frame = Data(capacity: OuterframeContentSocketHeaderLength + messageLength)
    frame.append(uint32: UInt32(messageLength))
    frame.append(uint16: type.rawValue)
    frame.append(payload)
    return frame
}

private func makeMouseEventFrame(type: BrowserToContentMessageKind,
                                 point: CGPoint,
                                 modifierFlags: NSEvent.ModifierFlags,
                                 clickCount: Int? = nil) -> Data {
    var payload = Data(capacity: clickCount == nil ? 24 : 28)
    payload.append(float64: point.x)
    payload.append(float64: point.y)
    payload.append(uint64: UInt64(modifierFlags.rawValue))
    if let clickCount {
        payload.append(uint32: UInt32(truncatingIfNeeded: clickCount))
    }
    return makeBrowserToContentFrame(type: type, payload: payload)
}

private func readMouseEvent(cursor: inout DataCursor,
                            includesClickCount: Bool) throws -> (point: CGPoint, modifierFlags: NSEvent.ModifierFlags, clickCount: Int) {
    guard let x = cursor.readFloat64(),
          let y = cursor.readFloat64(),
          let modifierFlags = cursor.readUInt64() else {
        throw OuterframeContentSocketMessageError.truncatedPayload
    }
    if includesClickCount {
        guard let clickCount = cursor.readUInt32() else {
            throw OuterframeContentSocketMessageError.truncatedPayload
        }
        return (CGPoint(x: x, y: y), NSEvent.ModifierFlags(rawValue: UInt(modifierFlags)), Int(clickCount))
    }
    return (CGPoint(x: x, y: y), NSEvent.ModifierFlags(rawValue: UInt(modifierFlags)), 0)
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

    init(referenceBaseOffset: Int = OuterframeContentSocketMessageTypeLength) {
        self.referenceBaseOffset = referenceBaseOffset
    }

    mutating func append(uint32 value: UInt32) {
        fixed.append(uint32: value)
    }

    mutating func append(int32 value: Int32) {
        fixed.append(int32: value)
    }

    mutating func append(uint16 value: UInt16) {
        fixed.append(uint16: value)
    }

    mutating func append(uint8 value: UInt8) {
        fixed.append(uint8: value)
    }

    mutating func append(uint64 value: UInt64) {
        fixed.append(uint64: value)
    }

    mutating func append(float32 value: Float32) {
        fixed.append(float32: value)
    }

    mutating func append(float64 value: Double) {
        fixed.append(float64: value)
    }

    mutating func append(uuid: UUID) {
        fixed.append(uuid: uuid)
    }

    mutating func append(stringReference string: String) throws {
        guard let data = string.data(using: .utf8) else {
            throw OuterframeContentSocketMessageError.encodingFailure("Invalid UTF-8 string")
        }
        try append(dataReference: data)
    }

    mutating func append(dataReference data: Data) throws {
        guard data.count <= UInt32.max else {
            throw OuterframeContentSocketMessageError.encodingFailure("Data too long")
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
            throw OuterframeContentSocketMessageError.encodingFailure("Payload too long")
        }

        for reference in references {
            let offset = referenceBaseOffset + fixed.count + reference.variableOffset
            guard offset <= UInt32.max,
                  reference.length <= UInt32.max else {
                throw OuterframeContentSocketMessageError.encodingFailure("Payload too long")
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
        guard offset + length <= data.count else { return nil }
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

import Foundation
import AppKit.NSAppearance

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
    var contentWidth: CGFloat?
    var contentHeight: CGFloat?
    var appearance: NSAppearance?
    var proxy: InitializeContentProxy?
    var url: String?
    var bundleUrl: String?
    var windowIsActive: Bool?

    init(data: Data? = nil,
         contentWidth: CGFloat? = nil,
         contentHeight: CGFloat? = nil,
         appearance: NSAppearance? = nil,
         proxy: InitializeContentProxy? = nil,
         url: String? = nil,
         bundleUrl: String? = nil,
         windowIsActive: Bool? = nil) {
        self.data = data
        self.contentWidth = contentWidth
        self.contentHeight = contentHeight
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
    case resizeContent(width: CGFloat, height: CGFloat)
    case mouseDown(x: Float32, y: Float32, modifierFlags: UInt64, clickCount: UInt32)
    case mouseDragged(x: Float32, y: Float32, modifierFlags: UInt64)
    case mouseUp(x: Float32, y: Float32, modifierFlags: UInt64)
    case mouseMoved(x: Float32, y: Float32, modifierFlags: UInt64)
    case rightMouseDown(x: Float32, y: Float32, modifierFlags: UInt64, clickCount: UInt32)
    case rightMouseUp(x: Float32, y: Float32, modifierFlags: UInt64)
    case scrollWheelEvent(x: Float32,
                          y: Float32,
                          deltaX: Float32,
                          deltaY: Float32,
                          modifierFlags: UInt64,
                          phase: UInt32,
                          momentumPhase: UInt32,
                          hasPreciseScrollingDeltas: Bool)
    case keyDown(keyCode: UInt16,
                 characters: String,
                 charactersIgnoringModifiers: String,
                 modifierFlags: UInt64,
                 isARepeat: Bool)
    case keyUp(keyCode: UInt16,
               characters: String,
               charactersIgnoringModifiers: String,
               modifierFlags: UInt64,
               isARepeat: Bool)
    case magnification(surfaceID: UInt32, magnification: Float32, x: Float32, y: Float32, scrollX: Float32, scrollY: Float32)
    case magnificationEnded(surfaceID: UInt32, magnification: Float32, x: Float32, y: Float32, scrollX: Float32, scrollY: Float32)
    case quickLook(x: Float32, y: Float32)
    case imageWithSystemSymbolName(requestID: UUID,
                                   imageData: Data?,
                                   width: UInt32,
                                   height: UInt32,
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

            if let contentWidth = arguments.contentWidth,
               let contentHeight = arguments.contentHeight {
                var argPayload = Data(capacity: 1 + 16)
                argPayload.append(uint8: InitArgKind.contentSize.rawValue)
                argPayload.append(float64: contentWidth)
                argPayload.append(float64: contentHeight)
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

        case .resizeContent(let width, let height):
            var payload = Data(capacity: 4 + 4)
            payload.append(float64: width)
            payload.append(float64: height)
            return makeBrowserToContentFrame(type: .resizeContent, payload: payload)

        case .mouseDown(let x, let y, let modifierFlags, let clickCount):
            return makeMouseEventFrame(type: .mouseDown, x: x, y: y,
                                       modifierFlags: modifierFlags, clickCount: clickCount)

        case .mouseDragged(let x, let y, let modifierFlags):
            return makeMouseEventFrame(type: .mouseDragged, x: x, y: y, modifierFlags: modifierFlags)

        case .mouseUp(let x, let y, let modifierFlags):
            return makeMouseEventFrame(type: .mouseUp, x: x, y: y, modifierFlags: modifierFlags)

        case .mouseMoved(let x, let y, let modifierFlags):
            return makeMouseEventFrame(type: .mouseMoved, x: x, y: y, modifierFlags: modifierFlags)

        case .rightMouseDown(let x, let y, let modifierFlags, let clickCount):
            return makeMouseEventFrame(type: .rightMouseDown, x: x, y: y,
                                       modifierFlags: modifierFlags, clickCount: clickCount)

        case .rightMouseUp(let x, let y, let modifierFlags):
            return makeMouseEventFrame(type: .rightMouseUp, x: x, y: y, modifierFlags: modifierFlags)

        case .scrollWheelEvent(let x,
                               let y,
                               let deltaX,
                               let deltaY,
                               let modifierFlags,
                               let phaseRaw,
                               let momentumPhaseRaw,
                               let hasPreciseScrollingDeltas):
            var payload = Data(capacity: 4 * 4 + 8 + 4 + 4 + 1)
            payload.append(float32: x)
            payload.append(float32: y)
            payload.append(float32: deltaX)
            payload.append(float32: deltaY)
            payload.append(uint64: modifierFlags)
            payload.append(uint32: phaseRaw)
            payload.append(uint32: momentumPhaseRaw)
            var flags: UInt8 = 0
            if hasPreciseScrollingDeltas { flags |= 1 << 0 }
            payload.append(uint8: flags)
            return makeBrowserToContentFrame(type: .scrollWheelEvent, payload: payload)

        case .keyDown(let keyCode, let characters, let charactersIgnoringModifiers, let modifierFlags, let isARepeat):
            var payload = OffsetPayloadBuilder()
            payload.append(uint16: keyCode)
            try payload.append(stringReference: characters)
            try payload.append(stringReference: charactersIgnoringModifiers)
            payload.append(uint64: modifierFlags)
            payload.append(uint8: isARepeat ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .keyDown, payload: try payload.finalize())

        case .keyUp(let keyCode, let characters, let charactersIgnoringModifiers, let modifierFlags, let isARepeat):
            var payload = OffsetPayloadBuilder()
            payload.append(uint16: keyCode)
            try payload.append(stringReference: characters)
            try payload.append(stringReference: charactersIgnoringModifiers)
            payload.append(uint64: modifierFlags)
            payload.append(uint8: isARepeat ? 1 << 0 : 0)
            return makeBrowserToContentFrame(type: .keyUp, payload: try payload.finalize())

        case .magnification(let surfaceID, let magnification, let x, let y, let scrollX, let scrollY):
            var payload = Data()
            payload.append(uint32: surfaceID)
            payload.append(float32: magnification)
            payload.append(float32: x)
            payload.append(float32: y)
            payload.append(float32: scrollX)
            payload.append(float32: scrollY)
            return makeBrowserToContentFrame(type: .magnification, payload: payload)

        case .magnificationEnded(let surfaceID, let magnification, let x, let y, let scrollX, let scrollY):
            var payload = Data()
            payload.append(uint32: surfaceID)
            payload.append(float32: magnification)
            payload.append(float32: x)
            payload.append(float32: y)
            payload.append(float32: scrollX)
            payload.append(float32: scrollY)
            return makeBrowserToContentFrame(type: .magnificationEnded, payload: payload)

        case .quickLook(let x, let y):
            var payload = Data(capacity: 4 + 4)
            payload.append(float32: x)
            payload.append(float32: y)
            return makeBrowserToContentFrame(type: .quickLook, payload: payload)

        case .imageWithSystemSymbolName(let requestID, let imageData, let width, let height, let success, let errorMessage):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            payload.append(uint32: width)
            payload.append(uint32: height)
            var flags: UInt8 = 0
            if success { flags |= 1 << 0 }
            if imageData != nil { flags |= 1 << 1 }
            if errorMessage != nil { flags |= 1 << 2 }
            payload.append(uint8: flags)
            try payload.append(dataReference: imageData ?? Data())
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
                    arguments.contentWidth = width
                    arguments.contentHeight = height

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
            return .resizeContent(width: width, height: height)

        case .mouseDown:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: true)
            return .mouseDown(x: event.x, y: event.y,
                              modifierFlags: event.modifierFlags, clickCount: event.clickCount)

        case .mouseDragged:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .mouseDragged(x: event.x, y: event.y, modifierFlags: event.modifierFlags)

        case .mouseUp:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .mouseUp(x: event.x, y: event.y, modifierFlags: event.modifierFlags)

        case .mouseMoved:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .mouseMoved(x: event.x, y: event.y, modifierFlags: event.modifierFlags)

        case .rightMouseDown:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: true)
            return .rightMouseDown(x: event.x, y: event.y,
                                   modifierFlags: event.modifierFlags, clickCount: event.clickCount)

        case .rightMouseUp:
            let event = try readMouseEvent(cursor: &cursor, includesClickCount: false)
            return .rightMouseUp(x: event.x, y: event.y, modifierFlags: event.modifierFlags)

        case .scrollWheelEvent:
            guard let x = cursor.readFloat32(),
                  let y = cursor.readFloat32(),
                  let deltaX = cursor.readFloat32(),
                  let deltaY = cursor.readFloat32(),
                  let modifierFlags = cursor.readUInt64(),
                  let phaseRaw = cursor.readUInt32(),
                  let momentumPhaseRaw = cursor.readUInt32(),
                  let flags = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .scrollWheelEvent(x: x, y: y, deltaX: deltaX, deltaY: deltaY,
                                     modifierFlags: modifierFlags, phase: phaseRaw,
                                     momentumPhase: momentumPhaseRaw,
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
                            modifierFlags: modifierFlags, isARepeat: flags & (1 << 0) != 0)

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
                          modifierFlags: modifierFlags, isARepeat: flags & (1 << 0) != 0)

        case .magnification:
            guard let surfaceID = cursor.readUInt32(),
                  let magnification = cursor.readFloat32(),
                  let x = cursor.readFloat32(),
                  let y = cursor.readFloat32(),
                  let scrollX = cursor.readFloat32(),
                  let scrollY = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .magnification(surfaceID: surfaceID, magnification: magnification,
                                  x: x, y: y, scrollX: scrollX, scrollY: scrollY)

        case .magnificationEnded:
            guard let surfaceID = cursor.readUInt32(),
                  let magnification = cursor.readFloat32(),
                  let x = cursor.readFloat32(),
                  let y = cursor.readFloat32(),
                  let scrollX = cursor.readFloat32(),
                  let scrollY = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .magnificationEnded(surfaceID: surfaceID, magnification: magnification,
                                       x: x, y: y, scrollX: scrollX, scrollY: scrollY)

        case .quickLook:
            guard let x = cursor.readFloat32(),
                  let y = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .quickLook(x: x, y: y)

        case .imageWithSystemSymbolName:
            guard let requestID = cursor.readUUID(),
                  let width = cursor.readUInt32(),
                  let height = cursor.readUInt32(),
                  let flags = cursor.readUInt8(),
                  let imageDataReference = cursor.readDataReference(),
                  let errorMessageReference = cursor.readStringReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }

            let imageData = flags & (1 << 1) != 0 ? imageDataReference : nil
            let errorMessage = flags & (1 << 2) != 0 ? errorMessageReference : nil

            return .imageWithSystemSymbolName(requestID: requestID, imageData: imageData,
                                     width: width, height: height,
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
    case showContextMenu(attributedTextData: Data, locationX: Float32, locationY: Float32)
    case showDefinition(attributedTextData: Data, locationX: Float32, locationY: Float32)
    case getImageWithSystemSymbolName(requestID: UUID,
                                      symbolName: String,
                                      pointSize: Float32,
                                      weight: String,
                                      scale: Float32,
                                      tintRed: Float32,
                                      tintGreen: Float32,
                                      tintBlue: Float32,
                                      tintAlpha: Float32)
    case textCursorUpdate(cursors: [OuterframeContentTextCursorSnapshot])
    case copySelectedPasteboardResponse(requestID: UUID, items: [OuterframeContentPasteboardItem])
    case openNewWindow(url: String, displayString: String?, preferredWidth: Float32?, preferredHeight: Float32?)
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
            payload.append(float32: locationX)
            payload.append(float32: locationY)
            try payload.append(dataReference: attributedTextData)
            return makeContentToBrowserFrame(type: .showContextMenu, payload: try payload.finalize())

        case .showDefinition(let attributedTextData, let locationX, let locationY):
            var payload = OffsetPayloadBuilder()
            payload.append(float32: locationX)
            payload.append(float32: locationY)
            try payload.append(dataReference: attributedTextData)
            return makeContentToBrowserFrame(type: .showDefinition, payload: try payload.finalize())

        case .getImageWithSystemSymbolName(let requestID, let symbolName, let pointSize, let weight,
                              let scale, let tintRed, let tintGreen, let tintBlue, let tintAlpha):
            var payload = OffsetPayloadBuilder()
            payload.append(uuid: requestID)
            try payload.append(stringReference: symbolName)
            payload.append(float32: pointSize)
            try payload.append(stringReference: weight)
            payload.append(float32: scale)
            payload.append(float32: tintRed)
            payload.append(float32: tintGreen)
            payload.append(float32: tintBlue)
            payload.append(float32: tintAlpha)
            return makeContentToBrowserFrame(type: .getImageWithSystemSymbolName, payload: try payload.finalize())

        case .textCursorUpdate(let cursors):
            var payload = Data()
            let countValue = UInt32(max(0, min(cursors.count, Int(UInt32.max))))
            payload.append(uint32: countValue)
            for cursor in cursors {
                payload.append(uuid: cursor.fieldID)
                payload.append(float32: cursor.rectX)
                payload.append(float32: cursor.rectY)
                payload.append(float32: cursor.rectWidth)
                payload.append(float32: cursor.rectHeight)
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

        case .openNewWindow(let url, let displayString, let preferredWidth, let preferredHeight):
            var payload = OffsetPayloadBuilder()
            try payload.append(stringReference: url)
            var flags: UInt8 = 0
            if displayString != nil { flags |= 1 << 0 }
            if preferredWidth != nil && preferredHeight != nil { flags |= 1 << 1 }
            payload.append(uint8: flags)
            try payload.append(stringReference: displayString ?? "")
            payload.append(float32: preferredWidth ?? 0)
            payload.append(float32: preferredHeight ?? 0)
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
            guard let locationX = cursor.readFloat32(),
                  let locationY = cursor.readFloat32(),
                  let attributedTextData = cursor.readDataReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .showContextMenu(attributedTextData: attributedTextData,
                                    locationX: locationX, locationY: locationY)

        case .showDefinition:
            guard let locationX = cursor.readFloat32(),
                  let locationY = cursor.readFloat32(),
                  let attributedTextData = cursor.readDataReference() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .showDefinition(attributedTextData: attributedTextData,
                                   locationX: locationX, locationY: locationY)

        case .getImageWithSystemSymbolName:
            guard let requestID = cursor.readUUID(),
                  let symbolName = cursor.readStringReference(),
                  let pointSize = cursor.readFloat32(),
                  let weight = cursor.readStringReference(),
                  let scale = cursor.readFloat32(),
                  let tintRed = cursor.readFloat32(),
                  let tintGreen = cursor.readFloat32(),
                  let tintBlue = cursor.readFloat32(),
                  let tintAlpha = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .getImageWithSystemSymbolName(requestID: requestID, symbolName: symbolName,
                                    pointSize: pointSize, weight: weight, scale: scale,
                                    tintRed: tintRed, tintGreen: tintGreen,
                                    tintBlue: tintBlue, tintAlpha: tintAlpha)

        case .textCursorUpdate:
            guard let cursorCount = cursor.readUInt32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var entries: [OuterframeContentTextCursorSnapshot] = []
            entries.reserveCapacity(Int(cursorCount))
            for _ in 0..<cursorCount {
                guard let fieldID = cursor.readUUID(),
                      let rectX = cursor.readFloat32(),
                      let rectY = cursor.readFloat32(),
                      let rectWidth = cursor.readFloat32(),
                      let rectHeight = cursor.readFloat32(),
                      let flags = cursor.readUInt8() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                entries.append(OuterframeContentTextCursorSnapshot(fieldID: fieldID,
                                                              rectX: rectX, rectY: rectY,
                                                              rectWidth: rectWidth, rectHeight: rectHeight,
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
                  let width = cursor.readFloat32(),
                  let height = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let displayString = flags & (1 << 0) != 0 ? displayStringReference : nil
            let widthValue = flags & (1 << 1) != 0 ? width : nil
            let heightValue = flags & (1 << 1) != 0 ? height : nil
            return .openNewWindow(url: url, displayString: displayString,
                                  preferredWidth: widthValue, preferredHeight: heightValue)
        }
    }
}

// MARK: - Supporting Types

struct OuterframeContentTextCursorSnapshot: Sendable {
    let fieldID: UUID
    let rectX: Float32
    let rectY: Float32
    let rectWidth: Float32
    let rectHeight: Float32
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
                                 x: Float32,
                                 y: Float32,
                                 modifierFlags: UInt64,
                                 clickCount: UInt32? = nil) -> Data {
    var payload = Data(capacity: clickCount == nil ? 16 : 20)
    payload.append(float32: x)
    payload.append(float32: y)
    payload.append(uint64: modifierFlags)
    if let clickCount {
        payload.append(uint32: clickCount)
    }
    return makeBrowserToContentFrame(type: type, payload: payload)
}

private func readMouseEvent(cursor: inout DataCursor,
                            includesClickCount: Bool) throws -> (x: Float32, y: Float32, modifierFlags: UInt64, clickCount: UInt32) {
    guard let x = cursor.readFloat32(),
          let y = cursor.readFloat32(),
          let modifierFlags = cursor.readUInt64() else {
        throw OuterframeContentSocketMessageError.truncatedPayload
    }
    if includesClickCount {
        guard let clickCount = cursor.readUInt32() else {
            throw OuterframeContentSocketMessageError.truncatedPayload
        }
        return (x, y, modifierFlags, clickCount)
    }
    return (x, y, modifierFlags, 0)
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

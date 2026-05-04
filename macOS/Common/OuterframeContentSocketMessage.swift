import Foundation
import AppKit.NSAppearance

let OuterframeContentSocketHeaderLength = MemoryLayout<UInt16>.size + MemoryLayout<UInt32>.size

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
    case displayLinkCallbackRegistered(callbackId: UUID, browserCallbackId: UUID)
    case resizeContent(width: CGFloat, height: CGFloat)
    case mouseEvent(kind: OuterframeContentMouseEventKind,
                    x: Float32,
                    y: Float32,
                    modifierFlags: UInt64,
                    clickCount: UInt32)
    case scrollWheelEvent(x: Float32,
                          y: Float32,
                          deltaX: Float32,
                          deltaY: Float32,
                          modifierFlags: UInt64,
                          phase: UInt32,
                          momentumPhase: UInt32,
                          isMomentum: Bool,
                          isPrecise: Bool)
    case keyDown(keyCode: UInt16,
                 characters: String,
                 charactersIgnoringModifiers: String,
                 modifierFlags: UInt64,
                 isRepeat: Bool)
    case keyUp(keyCode: UInt16,
               characters: String,
               charactersIgnoringModifiers: String,
               modifierFlags: UInt64,
               isRepeat: Bool)
    case magnification(surfaceId: UInt32, magnification: Float32, x: Float32, y: Float32, scrollX: Float32, scrollY: Float32)
    case magnificationEnded(surfaceId: UInt32, magnification: Float32, x: Float32, y: Float32, scrollX: Float32, scrollY: Float32)
    case quickLook(x: Float32, y: Float32)
    case imageWithSystemSymbolName(requestId: UUID,
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
    case textInputFocus(fieldId: String, hasFocus: Bool)
    case textCommand(command: String)
    case setCursorPosition(fieldId: String, position: UInt64, modifySelection: Bool)
    case systemAppearanceUpdate(appearance: NSAppearance)
    case windowActiveUpdate(isActive: Bool)
    case viewFocusChanged(isFocused: Bool)
    case copySelectedPasteboardRequest(requestId: UUID)
    case pasteboardContentDelivered(items: [OuterframeContentPasteboardItem])
    case accessibilitySnapshotRequest(requestId: UUID)
    case shutdown

    func encode() throws -> Data {
        switch self {
        case .initializeContent(let arguments):
            var encodedArguments: [(kind: InitArgKind, payload: Data)] = []

            if let data = arguments.data {
                var argPayload = Data()
                try argPayload.append(lengthPrefixedData32: data)
                encodedArguments.append((kind: .data, payload: argPayload))
            }

            if let contentWidth = arguments.contentWidth,
               let contentHeight = arguments.contentHeight {
                var argPayload = Data()
                argPayload.append(float64: contentWidth)
                argPayload.append(float64: contentHeight)
                encodedArguments.append((kind: .contentSize, payload: argPayload))
            }

            if let appearance = arguments.appearance {
                var argPayload = Data()
                let appearanceData = try NSKeyedArchiver.archivedData(withRootObject: appearance, requiringSecureCoding: true)
                try argPayload.append(lengthPrefixedData32: appearanceData)
                encodedArguments.append((kind: .appearance, payload: argPayload))
            }

            if let proxy = arguments.proxy {
                var argPayload = Data()
                try argPayload.append(lengthPrefixedUTF8_32: proxy.host)
                argPayload.append(uint16: proxy.port)
                encodedArguments.append((kind: .proxy, payload: argPayload))

                if proxy.username != nil || proxy.password != nil {
                    var authPayload = Data()
                    if let username = proxy.username {
                        authPayload.append(uint8: 1)
                        try authPayload.append(lengthPrefixedUTF8_32: username)
                    } else {
                        authPayload.append(uint8: 0)
                    }
                    if let password = proxy.password {
                        authPayload.append(uint8: 1)
                        try authPayload.append(lengthPrefixedUTF8_32: password)
                    } else {
                        authPayload.append(uint8: 0)
                    }
                    encodedArguments.append((kind: .proxyAuth, payload: authPayload))
                }
            }

            if let url = arguments.url {
                var argPayload = Data()
                try argPayload.append(lengthPrefixedUTF8_32: url)
                encodedArguments.append((kind: .url, payload: argPayload))
            }

            if let bundleUrl = arguments.bundleUrl {
                var argPayload = Data()
                try argPayload.append(lengthPrefixedUTF8_32: bundleUrl)
                encodedArguments.append((kind: .bundleUrl, payload: argPayload))
            }

            if let windowIsActive = arguments.windowIsActive {
                var argPayload = Data()
                argPayload.append(uint8: windowIsActive ? 1 : 0)
                encodedArguments.append((kind: .windowIsActive, payload: argPayload))
            }

            var payload = Data()
            payload.append(uint16: UInt16(min(encodedArguments.count, Int(UInt16.max))))

            for encodedArgument in encodedArguments {
                payload.append(uint8: encodedArgument.kind.rawValue)
                let argPayload = encodedArgument.payload
                try payload.append(lengthPrefixedData32: argPayload)
            }

            return makeBrowserToContentFrame(type: .initializeContent, payload: payload)

        case .displayLinkFired(let frameNumber, let targetTimestamp):
            var payload = Data(capacity: 16)
            payload.append(uint64: frameNumber)
            payload.append(float64: targetTimestamp)
            return makeBrowserToContentFrame(type: .displayLinkFired, payload: payload)

        case .displayLinkCallbackRegistered(let callbackId, let browserCallbackId):
            var payload = Data(capacity: 16 * 2)
            payload.append(uuid: callbackId)
            payload.append(uuid: browserCallbackId)
            return makeBrowserToContentFrame(type: .displayLinkCallbackRegistered, payload: payload)

        case .resizeContent(let width, let height):
            var payload = Data(capacity: 4 + 4)
            payload.append(float64: width)
            payload.append(float64: height)
            return makeBrowserToContentFrame(type: .resizeContent, payload: payload)

        case .mouseEvent(let kind, let x, let y, let modifierFlags, let clickCount):
            var payload = Data(capacity: 1 + 4 + 4 + 8 + 4)
            payload.append(uint8: kind.rawValue)
            payload.append(float32: x)
            payload.append(float32: y)
            payload.append(uint64: modifierFlags)
            payload.append(uint32: clickCount)
            return makeBrowserToContentFrame(type: .mouseEvent, payload: payload)

        case .scrollWheelEvent(let x,
                               let y,
                               let deltaX,
                               let deltaY,
                               let modifierFlags,
                               let phaseRaw,
                               let momentumPhaseRaw,
                               let isMomentum,
                               let isPrecise):
            var payload = Data(capacity: 4 * 4 + 8 + 4 + 4 + 1 + 1)
            payload.append(float32: x)
            payload.append(float32: y)
            payload.append(float32: deltaX)
            payload.append(float32: deltaY)
            payload.append(uint64: modifierFlags)
            payload.append(uint32: phaseRaw)
            payload.append(uint32: momentumPhaseRaw)
            payload.append(uint8: isMomentum ? 1 : 0)
            payload.append(uint8: isPrecise ? 1 : 0)
            return makeBrowserToContentFrame(type: .scrollWheelEvent, payload: payload)

        case .keyDown(let keyCode, let characters, let charactersIgnoringModifiers, let modifierFlags, let isRepeat):
            var payload = Data()
            payload.append(uint16: keyCode)
            try payload.append(lengthPrefixedUTF8_32: characters)
            try payload.append(lengthPrefixedUTF8_32: charactersIgnoringModifiers)
            payload.append(uint64: modifierFlags)
            payload.append(uint8: isRepeat ? 1 : 0)
            return makeBrowserToContentFrame(type: .keyDown, payload: payload)

        case .keyUp(let keyCode, let characters, let charactersIgnoringModifiers, let modifierFlags, let isRepeat):
            var payload = Data()
            payload.append(uint16: keyCode)
            try payload.append(lengthPrefixedUTF8_32: characters)
            try payload.append(lengthPrefixedUTF8_32: charactersIgnoringModifiers)
            payload.append(uint64: modifierFlags)
            payload.append(uint8: isRepeat ? 1 : 0)
            return makeBrowserToContentFrame(type: .keyUp, payload: payload)

        case .magnification(let surfaceId, let magnification, let x, let y, let scrollX, let scrollY):
            var payload = Data()
            payload.append(uint32: surfaceId)
            payload.append(float32: magnification)
            payload.append(float32: x)
            payload.append(float32: y)
            payload.append(float32: scrollX)
            payload.append(float32: scrollY)
            return makeBrowserToContentFrame(type: .magnification, payload: payload)

        case .magnificationEnded(let surfaceId, let magnification, let x, let y, let scrollX, let scrollY):
            var payload = Data()
            payload.append(uint32: surfaceId)
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

        case .imageWithSystemSymbolName(let requestId, let imageData, let width, let height, let success, let errorMessage):
            var payload = Data()
            payload.append(uuid: requestId)
            payload.append(uint32: width)
            payload.append(uint32: height)
            payload.append(uint8: success ? 1 : 0)
            if let imageData {
                payload.append(uint8: 1)
                try payload.append(lengthPrefixedData32: imageData)
            } else {
                payload.append(uint8: 0)
            }
            if let errorMessage {
                payload.append(uint8: 1)
                try payload.append(lengthPrefixedUTF8_32: errorMessage)
            } else {
                payload.append(uint8: 0)
            }
            return makeBrowserToContentFrame(type: .imageWithSystemSymbolName, payload: payload)

        case .textInput(let text, let hasReplacementRange, let replacementLocation, let replacementLength):
            var payload = Data()
            try payload.append(lengthPrefixedUTF8_32: text)
            payload.append(uint8: hasReplacementRange ? 1 : 0)
            payload.append(uint64: replacementLocation)
            payload.append(uint64: replacementLength)
            return makeBrowserToContentFrame(type: .textInput, payload: payload)

        case .setMarkedText(let text, let selectedLocation, let selectedLength, let hasReplacementRange, let replacementLocation, let replacementLength):
            var payload = Data()
            try payload.append(lengthPrefixedUTF8_32: text)
            payload.append(uint64: selectedLocation)
            payload.append(uint64: selectedLength)
            payload.append(uint8: hasReplacementRange ? 1 : 0)
            payload.append(uint64: replacementLocation)
            payload.append(uint64: replacementLength)
            return makeBrowserToContentFrame(type: .setMarkedText, payload: payload)

        case .unmarkText:
            return makeBrowserToContentFrame(type: .unmarkText, payload: Data())

        case .textInputFocus(let fieldId, let hasFocus):
            var payload = Data()
            try payload.append(lengthPrefixedUTF8_32: fieldId)
            payload.append(uint8: hasFocus ? 1 : 0)
            return makeBrowserToContentFrame(type: .textInputFocus, payload: payload)

        case .textCommand(let command):
            var payload = Data()
            try payload.append(lengthPrefixedUTF8_32: command)
            return makeBrowserToContentFrame(type: .textCommand, payload: payload)

        case .setCursorPosition(let fieldId, let position, let modifySelection):
            var payload = Data()
            try payload.append(lengthPrefixedUTF8_32: fieldId)
            payload.append(uint64: position)
            payload.append(uint8: modifySelection ? 1 : 0)
            return makeBrowserToContentFrame(type: .setCursorPosition, payload: payload)

        case .systemAppearanceUpdate(let appearance):
            var payload = Data()
            let appearanceData = try NSKeyedArchiver.archivedData(withRootObject: appearance, requiringSecureCoding: true)
            try payload.append(lengthPrefixedData32: appearanceData)
            return makeBrowserToContentFrame(type: .systemAppearanceUpdate, payload: payload)

        case .windowActiveUpdate(let isActive):
            var payload = Data(capacity: 1)
            payload.append(uint8: isActive ? 1 : 0)
            return makeBrowserToContentFrame(type: .windowActiveUpdate, payload: payload)

        case .viewFocusChanged(let isFocused):
            var payload = Data(capacity: 1)
            payload.append(uint8: isFocused ? 1 : 0)
            return makeBrowserToContentFrame(type: .viewFocusChanged, payload: payload)

        case .copySelectedPasteboardRequest(let requestId):
            var payload = Data(capacity: 16)
            payload.append(uuid: requestId)
            return makeBrowserToContentFrame(type: .copySelectedPasteboardRequest, payload: payload)

        case .pasteboardContentDelivered(let items):
            var payload = Data()
            let clampedCount = UInt16(min(items.count, Int(UInt16.max)))
            payload.append(uint16: clampedCount)
            for item in items.prefix(Int(clampedCount)) {
                try payload.append(lengthPrefixedUTF8_32: item.typeIdentifier)
                try payload.append(lengthPrefixedData32: item.data)
            }
            return makeBrowserToContentFrame(type: .pasteboardContentDelivered, payload: payload)

        case .accessibilitySnapshotRequest(let requestId):
            var payload = Data(capacity: 16)
            payload.append(uuid: requestId)
            return makeBrowserToContentFrame(type: .accessibilitySnapshotRequest, payload: payload)

        case .shutdown:
            return makeBrowserToContentFrame(type: .shutdown, payload: Data())
        }
    }

    static func decode(typeRaw: UInt16, payload: Data) throws -> BrowserToContentMessage {
        guard let type = BrowserToContentMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentSocketMessageError.unknownType(typeRaw)
        }

        var cursor = DataCursor(payload)

        switch type {
        case .initializeContent:
            guard let argCount = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }

            var arguments = InitializeContentArguments()
            var proxyUsername: String?
            var proxyPassword: String?

            for _ in 0..<argCount {
                guard let kindRaw = cursor.readUInt8(),
                      let argData = cursor.readData32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }

                guard let kind = InitArgKind(rawValue: kindRaw) else {
                    continue
                }

                var argCursor = DataCursor(argData)

                switch kind {
                case .data:
                    guard let data = argCursor.readData32() else {
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
                    guard let appearanceData = argCursor.readData32(),
                          let decoded = try? NSKeyedUnarchiver.unarchivedObject(ofClass: NSAppearance.self, from: appearanceData) else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.appearance = decoded

                case .proxy:
                    guard let proxyHost = argCursor.readString32(),
                          let proxyPort = argCursor.readUInt16() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.proxy = InitializeContentProxy(host: proxyHost,
                                                             port: proxyPort,
                                                             username: proxyUsername,
                                                             password: proxyPassword)

                case .proxyAuth:
                    guard let usernameIsPresent = argCursor.readUInt8() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    if usernameIsPresent != 0 {
                        guard let username = argCursor.readString32() else {
                            throw OuterframeContentSocketMessageError.truncatedPayload
                        }
                        proxyUsername = username
                    } else {
                        proxyUsername = nil
                    }

                    guard let passwordIsPresent = argCursor.readUInt8() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    if passwordIsPresent != 0 {
                        guard let password = argCursor.readString32() else {
                            throw OuterframeContentSocketMessageError.truncatedPayload
                        }
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
                    guard let url = argCursor.readString32() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.url = url

                case .bundleUrl:
                    guard let bundleUrl = argCursor.readString32() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.bundleUrl = bundleUrl

                case .windowIsActive:
                    guard let windowIsActiveRaw = argCursor.readUInt8() else {
                        throw OuterframeContentSocketMessageError.truncatedPayload
                    }
                    arguments.windowIsActive = windowIsActiveRaw != 0
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
            guard let callbackId = cursor.readUUID(),
                  let browserCallbackId = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .displayLinkCallbackRegistered(callbackId: callbackId, browserCallbackId: browserCallbackId)

        case .resizeContent:
            guard let width = cursor.readFloat64(),
                  let height = cursor.readFloat64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .resizeContent(width: width, height: height)

        case .mouseEvent:
            guard let kindRaw = cursor.readUInt8(),
                  let kind = OuterframeContentMouseEventKind(rawValue: kindRaw),
                  let x = cursor.readFloat32(),
                  let y = cursor.readFloat32(),
                  let modifierFlags = cursor.readUInt64(),
                  let clickCount = cursor.readUInt32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .mouseEvent(kind: kind, x: x, y: y, modifierFlags: modifierFlags, clickCount: clickCount)

        case .scrollWheelEvent:
            guard let x = cursor.readFloat32(),
                  let y = cursor.readFloat32(),
                  let deltaX = cursor.readFloat32(),
                  let deltaY = cursor.readFloat32(),
                  let modifierFlags = cursor.readUInt64(),
                  let phaseRaw = cursor.readUInt32(),
                  let momentumPhaseRaw = cursor.readUInt32(),
                  let isMomentumRaw = cursor.readUInt8(),
                  let isPreciseRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .scrollWheelEvent(x: x, y: y, deltaX: deltaX, deltaY: deltaY,
                                     modifierFlags: modifierFlags, phase: phaseRaw,
                                     momentumPhase: momentumPhaseRaw,
                                     isMomentum: isMomentumRaw != 0, isPrecise: isPreciseRaw != 0)

        case .keyDown:
            guard let keyCode = cursor.readUInt16(),
                  let characters = cursor.readString32(),
                  let charactersIgnoringModifiers = cursor.readString32(),
                  let modifierFlags = cursor.readUInt64(),
                  let repeatRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .keyDown(keyCode: keyCode, characters: characters,
                            charactersIgnoringModifiers: charactersIgnoringModifiers,
                            modifierFlags: modifierFlags, isRepeat: repeatRaw != 0)

        case .keyUp:
            guard let keyCode = cursor.readUInt16(),
                  let characters = cursor.readString32(),
                  let charactersIgnoringModifiers = cursor.readString32(),
                  let modifierFlags = cursor.readUInt64(),
                  let repeatRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .keyUp(keyCode: keyCode, characters: characters,
                          charactersIgnoringModifiers: charactersIgnoringModifiers,
                          modifierFlags: modifierFlags, isRepeat: repeatRaw != 0)

        case .magnification:
            guard let surfaceId = cursor.readUInt32(),
                  let magnification = cursor.readFloat32(),
                  let x = cursor.readFloat32(),
                  let y = cursor.readFloat32(),
                  let scrollX = cursor.readFloat32(),
                  let scrollY = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .magnification(surfaceId: surfaceId, magnification: magnification,
                                  x: x, y: y, scrollX: scrollX, scrollY: scrollY)

        case .magnificationEnded:
            guard let surfaceId = cursor.readUInt32(),
                  let magnification = cursor.readFloat32(),
                  let x = cursor.readFloat32(),
                  let y = cursor.readFloat32(),
                  let scrollX = cursor.readFloat32(),
                  let scrollY = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .magnificationEnded(surfaceId: surfaceId, magnification: magnification,
                                       x: x, y: y, scrollX: scrollX, scrollY: scrollY)

        case .quickLook:
            guard let x = cursor.readFloat32(),
                  let y = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .quickLook(x: x, y: y)

        case .imageWithSystemSymbolName:
            guard let requestId = cursor.readUUID(),
                  let width = cursor.readUInt32(),
                  let height = cursor.readUInt32(),
                  let successRaw = cursor.readUInt8(),
                  let hasImageRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }

            var imageData: Data? = nil
            if hasImageRaw != 0 {
                guard let data = cursor.readData32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                imageData = data
            }

            guard let hasErrorMessageRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }

            var errorMessage: String? = nil
            if hasErrorMessageRaw != 0 {
                guard let message = cursor.readString32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                errorMessage = message
            }

            return .imageWithSystemSymbolName(requestId: requestId, imageData: imageData,
                                     width: width, height: height,
                                     success: successRaw != 0, errorMessage: errorMessage)

        case .textInput:
            guard let text = cursor.readString32(),
                  let hasRangeRaw = cursor.readUInt8(),
                  let replacementLocation = cursor.readUInt64(),
                  let replacementLength = cursor.readUInt64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .textInput(text: text, hasReplacementRange: hasRangeRaw != 0,
                              replacementLocation: replacementLocation,
                              replacementLength: replacementLength)

        case .setMarkedText:
            guard let text = cursor.readString32(),
                  let selectedLocation = cursor.readUInt64(),
                  let selectedLength = cursor.readUInt64(),
                  let hasRangeRaw = cursor.readUInt8(),
                  let replacementLocation = cursor.readUInt64(),
                  let replacementLength = cursor.readUInt64() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .setMarkedText(text: text, selectedLocation: selectedLocation,
                                  selectedLength: selectedLength,
                                  hasReplacementRange: hasRangeRaw != 0,
                                  replacementLocation: replacementLocation,
                                  replacementLength: replacementLength)

        case .unmarkText:
            return .unmarkText

        case .textInputFocus:
            guard let fieldId = cursor.readString32(),
                  let hasFocusRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .textInputFocus(fieldId: fieldId, hasFocus: hasFocusRaw != 0)

        case .textCommand:
            guard let command = cursor.readString32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .textCommand(command: command)

        case .setCursorPosition:
            guard let fieldId = cursor.readString32(),
                  let position = cursor.readUInt64(),
                  let modifySelectionRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .setCursorPosition(fieldId: fieldId, position: position,
                                      modifySelection: modifySelectionRaw != 0)

        case .systemAppearanceUpdate:
            guard let appearanceData = cursor.readData32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let appearance = (try? NSKeyedUnarchiver.unarchivedObject(ofClass: NSAppearance.self, from: appearanceData))
                ?? NSAppearance.currentDrawing()
            return .systemAppearanceUpdate(appearance: appearance)

        case .windowActiveUpdate:
            guard let raw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .windowActiveUpdate(isActive: raw != 0)

        case .viewFocusChanged:
            guard let raw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .viewFocusChanged(isFocused: raw != 0)

        case .copySelectedPasteboardRequest:
            guard let requestId = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .copySelectedPasteboardRequest(requestId: requestId)

        case .pasteboardContentDelivered:
            guard let count = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var items: [OuterframeContentPasteboardItem] = []
            items.reserveCapacity(Int(count))
            for _ in 0..<count {
                guard let identifier = cursor.readString32(),
                      let data = cursor.readData32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                items.append(OuterframeContentPasteboardItem(typeIdentifier: identifier, data: data))
            }
            return .pasteboardContentDelivered(items: items)

        case .accessibilitySnapshotRequest:
            guard let requestId = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .accessibilitySnapshotRequest(requestId: requestId)

        case .shutdown:
            return .shutdown
        }
    }
}

/// Messages from Content to Browser on the content socket
enum ContentToBrowserMessage {
    case startDisplayLink(callbackId: UUID)
    case stopDisplayLink(browserCallbackId: UUID)
    case cursorUpdate(cursorType: UInt8)
    case inputModeUpdate(inputMode: UInt8)
    case showContextMenu(attributedTextData: Data, locationX: Float32, locationY: Float32)
    case showDefinition(attributedTextData: Data, locationX: Float32, locationY: Float32)
    case getImageWithSystemSymbolName(requestId: UUID,
                                      symbolName: String,
                                      pointSize: Float32,
                                      weight: String,
                                      scale: Float32,
                                      tintRed: Float32,
                                      tintGreen: Float32,
                                      tintBlue: Float32,
                                      tintAlpha: Float32)
    case textCursorUpdate(cursors: [OuterframeContentTextCursorSnapshot])
    case copySelectedPasteboardResponse(requestId: UUID, items: [OuterframeContentPasteboardItem])
    case openNewWindow(url: String, displayString: String?, preferredWidth: Float32?, preferredHeight: Float32?)
    case setPasteboardCapabilities(canCopy: Bool, canCut: Bool, pasteboardTypes: [String])
    case accessibilitySnapshotResponse(requestId: UUID, snapshotData: Data?)
    case accessibilityTreeChanged(notificationMask: UInt8)
    case hapticFeedback(style: UInt8)

    func encode() throws -> Data {
        switch self {
        case .startDisplayLink(let callbackId):
            var payload = Data(capacity: 16)
            payload.append(uuid: callbackId)
            return makeContentToBrowserFrame(type: .startDisplayLink, payload: payload)

        case .stopDisplayLink(let browserCallbackId):
            var payload = Data(capacity: 16)
            payload.append(uuid: browserCallbackId)
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
            var payload = Data()
            payload.append(float32: locationX)
            payload.append(float32: locationY)
            try payload.append(lengthPrefixedData32: attributedTextData)
            return makeContentToBrowserFrame(type: .showContextMenu, payload: payload)

        case .showDefinition(let attributedTextData, let locationX, let locationY):
            var payload = Data()
            payload.append(float32: locationX)
            payload.append(float32: locationY)
            try payload.append(lengthPrefixedData32: attributedTextData)
            return makeContentToBrowserFrame(type: .showDefinition, payload: payload)

        case .getImageWithSystemSymbolName(let requestId, let symbolName, let pointSize, let weight,
                              let scale, let tintRed, let tintGreen, let tintBlue, let tintAlpha):
            var payload = Data()
            payload.append(uuid: requestId)
            try payload.append(lengthPrefixedUTF8_32: symbolName)
            payload.append(float32: pointSize)
            try payload.append(lengthPrefixedUTF8_32: weight)
            payload.append(float32: scale)
            payload.append(float32: tintRed)
            payload.append(float32: tintGreen)
            payload.append(float32: tintBlue)
            payload.append(float32: tintAlpha)
            return makeContentToBrowserFrame(type: .getImageWithSystemSymbolName, payload: payload)

        case .textCursorUpdate(let cursors):
            var payload = Data()
            let countValue = UInt32(max(0, min(cursors.count, Int(UInt32.max))))
            payload.append(uint32: countValue)
            for cursor in cursors {
                try payload.append(lengthPrefixedUTF8_32: cursor.fieldId)
                payload.append(float32: cursor.rectX)
                payload.append(float32: cursor.rectY)
                payload.append(float32: cursor.rectWidth)
                payload.append(float32: cursor.rectHeight)
                payload.append(uint8: cursor.visible ? 1 : 0)
            }
            return makeContentToBrowserFrame(type: .textCursorUpdate, payload: payload)

        case .copySelectedPasteboardResponse(let requestId, let items):
            var payload = Data()
            payload.append(uuid: requestId)
            let clampedCount = UInt16(min(items.count, Int(UInt16.max)))
            payload.append(uint16: clampedCount)
            for item in items.prefix(Int(clampedCount)) {
                try payload.append(lengthPrefixedUTF8_32: item.typeIdentifier)
                try payload.append(lengthPrefixedData32: item.data)
            }
            return makeContentToBrowserFrame(type: .copySelectedPasteboardResponse, payload: payload)

        case .openNewWindow(let url, let displayString, let preferredWidth, let preferredHeight):
            var payload = Data()
            try payload.append(lengthPrefixedUTF8_32: url)
            if let displayString {
                payload.append(uint8: 1)
                try payload.append(lengthPrefixedUTF8_32: displayString)
            } else {
                payload.append(uint8: 0)
            }
            if let preferredWidth, let preferredHeight {
                payload.append(uint8: 1)
                payload.append(float32: preferredWidth)
                payload.append(float32: preferredHeight)
            } else {
                payload.append(uint8: 0)
            }
            return makeContentToBrowserFrame(type: .openNewWindow, payload: payload)

        case .setPasteboardCapabilities(let canCopy, let canCut, let pasteboardTypes):
            var payload = Data()
            payload.append(uint8: canCopy ? 1 : 0)
            payload.append(uint8: canCut ? 1 : 0)
            let clampedCount = UInt16(min(pasteboardTypes.count, Int(UInt16.max)))
            payload.append(uint16: clampedCount)
            for identifier in pasteboardTypes.prefix(Int(clampedCount)) {
                try payload.append(lengthPrefixedUTF8_32: identifier)
            }
            return makeContentToBrowserFrame(type: .editingCapabilitiesUpdate, payload: payload)

        case .accessibilitySnapshotResponse(let requestId, let snapshotData):
            var payload = Data()
            payload.append(uuid: requestId)
            if let snapshotData {
                payload.append(uint8: 1)
                try payload.append(lengthPrefixedData32: snapshotData)
            } else {
                payload.append(uint8: 0)
            }
            return makeContentToBrowserFrame(type: .accessibilitySnapshotResponse, payload: payload)

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

    static func decode(typeRaw: UInt16, payload: Data) throws -> ContentToBrowserMessage {
        guard let type = ContentToBrowserMessageKind(rawValue: typeRaw) else {
            throw OuterframeContentSocketMessageError.unknownType(typeRaw)
        }

        var cursor = DataCursor(payload)

        switch type {
        case .startDisplayLink:
            guard let callbackId = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .startDisplayLink(callbackId: callbackId)

        case .stopDisplayLink:
            guard let browserCallbackId = cursor.readUUID() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .stopDisplayLink(browserCallbackId: browserCallbackId)

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
                  let attributedTextData = cursor.readData32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .showContextMenu(attributedTextData: attributedTextData,
                                    locationX: locationX, locationY: locationY)

        case .showDefinition:
            guard let locationX = cursor.readFloat32(),
                  let locationY = cursor.readFloat32(),
                  let attributedTextData = cursor.readData32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .showDefinition(attributedTextData: attributedTextData,
                                   locationX: locationX, locationY: locationY)

        case .getImageWithSystemSymbolName:
            guard let requestId = cursor.readUUID(),
                  let symbolName = cursor.readString32(),
                  let pointSize = cursor.readFloat32(),
                  let weight = cursor.readString32(),
                  let scale = cursor.readFloat32(),
                  let tintRed = cursor.readFloat32(),
                  let tintGreen = cursor.readFloat32(),
                  let tintBlue = cursor.readFloat32(),
                  let tintAlpha = cursor.readFloat32() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            return .getImageWithSystemSymbolName(requestId: requestId, symbolName: symbolName,
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
                guard let fieldId = cursor.readString32(),
                      let rectX = cursor.readFloat32(),
                      let rectY = cursor.readFloat32(),
                      let rectWidth = cursor.readFloat32(),
                      let rectHeight = cursor.readFloat32(),
                      let visibleRaw = cursor.readUInt8() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                entries.append(OuterframeContentTextCursorSnapshot(fieldId: fieldId,
                                                              rectX: rectX, rectY: rectY,
                                                              rectWidth: rectWidth, rectHeight: rectHeight,
                                                              visible: visibleRaw != 0))
            }
            return .textCursorUpdate(cursors: entries)

        case .copySelectedPasteboardResponse:
            guard let requestId = cursor.readUUID(),
                  let count = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var items: [OuterframeContentPasteboardItem] = []
            items.reserveCapacity(Int(count))
            for _ in 0..<count {
                guard let identifier = cursor.readString32(),
                      let data = cursor.readData32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                items.append(OuterframeContentPasteboardItem(typeIdentifier: identifier, data: data))
            }
            return .copySelectedPasteboardResponse(requestId: requestId, items: items)

        case .editingCapabilitiesUpdate:
            guard let canCopyRaw = cursor.readUInt8(),
                  let canCutRaw = cursor.readUInt8(),
                  let count = cursor.readUInt16() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var identifiers: [String] = []
            identifiers.reserveCapacity(Int(count))
            for _ in 0..<count {
                guard let identifier = cursor.readString32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                identifiers.append(identifier)
            }
            return .setPasteboardCapabilities(canCopy: canCopyRaw != 0,
                                              canCut: canCutRaw != 0,
                                              pasteboardTypes: identifiers)

        case .accessibilitySnapshotResponse:
            guard let requestId = cursor.readUUID(),
                  let hasData = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let snapshotData: Data?
            if hasData != 0 {
                guard let payload = cursor.readData32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                snapshotData = payload
            } else {
                snapshotData = nil
            }
            return .accessibilitySnapshotResponse(requestId: requestId, snapshotData: snapshotData)

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
            guard let url = cursor.readString32(),
                  let hasDisplayRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            let displayString: String?
            if hasDisplayRaw != 0 {
                guard let value = cursor.readString32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                displayString = value
            } else {
                displayString = nil
            }
            guard let hasSizeRaw = cursor.readUInt8() else {
                throw OuterframeContentSocketMessageError.truncatedPayload
            }
            var widthValue: Float32? = nil
            var heightValue: Float32? = nil
            if hasSizeRaw != 0 {
                guard let width = cursor.readFloat32(),
                      let height = cursor.readFloat32() else {
                    throw OuterframeContentSocketMessageError.truncatedPayload
                }
                widthValue = width
                heightValue = height
            }
            return .openNewWindow(url: url, displayString: displayString,
                                  preferredWidth: widthValue, preferredHeight: heightValue)
        }
    }
}

// MARK: - Supporting Types

struct OuterframeContentTextCursorSnapshot: Sendable {
    let fieldId: String
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

enum OuterframeContentMouseEventKind: UInt8 {
    case mouseDown = 1
    case mouseDragged = 2
    case mouseUp = 3
    case mouseMoved = 4
    case rightMouseDown = 5
    case rightMouseUp = 6
}

enum OuterframeContentSocketMessageError: Error {
    case unknownType(UInt16)
    case truncatedPayload
    case encodingFailure(String)
}

// MARK: - Message Kind Enums

private enum BrowserToContentMessageKind: UInt16 {
    case initializeContent = 50
    case displayLinkFired = 2
    case displayLinkCallbackRegistered = 15
    case resizeContent = 7
    case mouseEvent = 8
    case scrollWheelEvent = 47
    case keyDown = 9
    case keyUp = 10
    case magnification = 12
    case magnificationEnded = 13
    case quickLook = 20
    case imageWithSystemSymbolName = 21
    case textInput = 22
    case setMarkedText = 23
    case unmarkText = 24
    case textInputFocus = 25
    case textCommand = 26
    case setCursorPosition = 27
    case systemAppearanceUpdate = 38
    case windowActiveUpdate = 39
    case viewFocusChanged = 49
    case copySelectedPasteboardRequest = 40
    case pasteboardContentDelivered = 45
    case accessibilitySnapshotRequest = 46
    case shutdown = 51
}

private enum ContentToBrowserMessageKind: UInt16 {
    case startDisplayLink = 17
    case stopDisplayLink = 18
    case cursorUpdate = 28
    case inputModeUpdate = 29
    case showContextMenu = 34
    case showDefinition = 35
    case getImageWithSystemSymbolName = 36
    case textCursorUpdate = 37
    case copySelectedPasteboardResponse = 40
    case openNewWindow = 41
    case editingCapabilitiesUpdate = 44
    case accessibilitySnapshotResponse = 45
    case accessibilityTreeChanged = 46
    case hapticFeedback = 48
}

// MARK: - Frame Helpers

private func makeBrowserToContentFrame(type: BrowserToContentMessageKind, payload: Data) -> Data {
    var frame = Data(capacity: OuterframeContentSocketHeaderLength + payload.count)
    frame.append(uint16: type.rawValue)
    frame.append(uint32: UInt32(payload.count))
    frame.append(payload)
    return frame
}

private func makeContentToBrowserFrame(type: ContentToBrowserMessageKind, payload: Data) -> Data {
    var frame = Data(capacity: OuterframeContentSocketHeaderLength + payload.count)
    frame.append(uint16: type.rawValue)
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

    mutating func append(lengthPrefixedUTF8_32 string: String) throws {
        guard let data = string.data(using: .utf8) else {
            throw OuterframeContentSocketMessageError.encodingFailure("Invalid UTF-8 string")
        }
        try append(lengthPrefixedData32: data)
    }

    mutating func append(lengthPrefixedData32 data: Data) throws {
        guard data.count <= UInt32.max else {
            throw OuterframeContentSocketMessageError.encodingFailure("Data too long")
        }
        append(uint32: UInt32(data.count))
        append(data)
    }
}

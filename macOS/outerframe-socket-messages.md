# Outerframe Socket Messages

This documents the socket messages between the Browser and outerframe content. Swift serialization/deserialization code is available in `Common/OuterframeContentSocketMessage.swift`.


## Shared frame format

The content socket uses length-prefixed socket frames. The frame length covers the complete message, including `messageType`.

```text
Socket frame:
bytes 0..3:    UInt32 little-endian message length, L
bytes 4..<4+L: message bytes

Message:
bytes 0..1:     UInt16 little-endian messageType
bytes 2..<V:    message-specific fixed-size fields
bytes V..<L:    variable-length region
```

Scalar values are little-endian. `float64` values are IEEE 754 bit patterns. `UUID` is 16 raw UUID bytes. Boolean values are stored as bits in an appropriately named `UInt8 flags` field.

Messages that contain strings or raw data have a fixed-size region followed by a variable-length region. The fixed-size region stores references to strings and data as offset/length pairs. Top-level offsets are relative to the beginning of the message that contains the reference, so message byte 0 is the first byte of `messageType`. Nested payloads, such as `initializeContent` argument payloads, use offsets relative to the beginning of that nested payload. Fields shown as `StringRef32`, `DataRef32`, `StringRef64`, or `DataRef64` are fixed-size offset/length pairs; the referenced bytes live in the variable-length region.

Reusable payload fragments:

```text
StringRef32:
bytes 0..3: UInt32 little-endian offset to UTF-8 bytes, O
bytes 4..7: UInt32 little-endian UTF-8 byte length, L

DataRef32:
bytes 0..3: UInt32 little-endian offset to raw bytes, O
bytes 4..7: UInt32 little-endian data length, L

StringRef64:
bytes 0..7:   UInt64 little-endian offset to UTF-8 bytes, O
bytes 8..15:  UInt64 little-endian UTF-8 byte length, L

DataRef64:
bytes 0..7:   UInt64 little-endian offset to raw bytes, O
bytes 8..15:  UInt64 little-endian data length, L
```

Socket messages use `StringRef32` and `DataRef32` because the socket frame length is `UInt32`. File/container formats such as `.outer` use `StringRef64` and `DataRef64` when the enclosing format can exceed the socket frame limit.

The referenced range is valid only when `offset <= containerLength` and `length <= containerLength - offset`, where `containerLength` is the message length for top-level messages or the equivalent nested payload/file length for nested containers.

## Browser to Content Messages

### `initializeContent` (`messageType = 1000`)

Initial content configuration. It sends a count of tagged arguments so optional fields can be omitted.

Closest macOS/API mirror: a mix of initial `NSView` sizing, `NSAppearance`, current URL/navigation context, window activation state, and proxy configuration. There is no single AppKit API.

```text
bytes 2..3:           UInt16 argument count, N
bytes 4..<4+8*N:      N argument references
argument reference i, B = 4 + 8*i:
  bytes B..B+3:       UInt32 offset to one complete argument payload
  bytes B+4..B+7:     UInt32 length of one complete argument payload
```

Each referenced argument payload starts with its argument kind and may itself have a fixed-size region followed by a variable-length region. Offsets inside an argument payload are relative to argument byte 0, the `UInt8 argument kind` byte.

Argument kinds:

```text
1 data:
  byte 0:    UInt8 argument kind
  bytes 1..8: DataRef32 initial content data

2 contentSize:
  byte 0:       UInt8 argument kind
  bytes 1..8:   float64 contentSize.width
  bytes 9..16:  float64 contentSize.height

3 appearance:
  byte 0:    UInt8 argument kind
  bytes 1..8: DataRef32 macOS NSKeyedArchiver payload for NSAppearance

4 proxy:
  byte 0:     UInt8 argument kind
  bytes 1..8: StringRef32 proxy host
  bytes 9..10: UInt16 proxy port

5 proxyAuth:
  byte 0:    UInt8 argument kind
  byte 1:    UInt8 flags
             bit 0 = hasUsername
             bit 1 = hasPassword
  bytes 2..9: StringRef32 username, empty when absent
  bytes 10..17: StringRef32 password, empty when absent

6 url:
  byte 0:    UInt8 argument kind
  bytes 1..8: StringRef32 URL

7 bundleUrl:
  byte 0:    UInt8 argument kind
  bytes 1..8: StringRef32 bundle URL

8 windowIsActive:
  byte 0: UInt8 argument kind
  byte 1: UInt8 flags
          bit 0 = isActive

9 historyEntryID:
  byte 0: UInt8 argument kind
  bytes 1..16: UUID history entry ID
```

### `displayLinkFired` (`messageType = 1003`)

Notifies content that a registered display-link callback fired.

Closest macOS/API mirror: `CVDisplayLink` or `CADisplayLink` callback timing.

```text
bytes 2..9:   UInt64 frameNumber
bytes 10..17: float64 targetTimestamp
```

### `displayLinkCallbackRegistered` (`messageType = 1004`)

Acknowledges a content display-link registration and returns the browser-side callback ID.

Closest macOS/API mirror: no direct AppKit API; this is the protocol's callback registration handshake around display-link scheduling.

```text
bytes 2..17:  UUID content callbackID
bytes 18..33: UUID browserCallbackID
```

### `resizeContent` (`messageType = 1001`)

Sends the current content viewport size.

Closest macOS/API mirror: `NSView.setFrameSize(_:)`, `NSView.frame`, or resize notifications.

```text
bytes 2..9:   float64 size.width
bytes 10..17: float64 size.height
```

### Mouse events

Sends mouse events in content coordinates.

Closest macOS/API mirror: `NSEvent` mouse events such as `mouseDown`, `mouseDragged`, `mouseUp`, `mouseMoved`, `rightMouseDown`, and `rightMouseUp`.

```text
mouseDown (`messageType = 1008`):
  bytes 2..9:   float64 x
  bytes 10..17: float64 y
  bytes 18..25: UInt64 NSEvent.ModifierFlags rawValue
  bytes 26..29: UInt32 clickCount

mouseDragged (`messageType = 1009`):
  bytes 2..9:   float64 x
  bytes 10..17: float64 y
  bytes 18..25: UInt64 NSEvent.ModifierFlags rawValue

mouseUp (`messageType = 1010`):
  bytes 2..9:   float64 x
  bytes 10..17: float64 y
  bytes 18..25: UInt64 NSEvent.ModifierFlags rawValue

mouseMoved (`messageType = 1011`):
  bytes 2..9:   float64 x
  bytes 10..17: float64 y
  bytes 18..25: UInt64 NSEvent.ModifierFlags rawValue

rightMouseDown (`messageType = 1012`):
  bytes 2..9:   float64 x
  bytes 10..17: float64 y
  bytes 18..25: UInt64 NSEvent.ModifierFlags rawValue
  bytes 26..29: UInt32 clickCount

rightMouseUp (`messageType = 1013`):
  bytes 2..9:   float64 x
  bytes 10..17: float64 y
  bytes 18..25: UInt64 NSEvent.ModifierFlags rawValue
```

### `scrollWheelEvent` (`messageType = 1014`)

Sends a scroll-wheel event in content coordinates.

Closest macOS/API mirror: `NSEvent` scroll-wheel events, including `phase`, `momentumPhase`, and `hasPreciseScrollingDeltas`.

```text
bytes 2..9:   float64 x
bytes 10..17: float64 y
bytes 18..25: float64 deltaX
bytes 26..33: float64 deltaY
bytes 34..41: UInt64 NSEvent.ModifierFlags rawValue
bytes 42..45: UInt32 NSEvent.Phase rawValue
bytes 46..49: UInt32 momentum NSEvent.Phase rawValue
byte 50:      UInt8 flags
              bit 0 = hasPreciseScrollingDeltas
```

### `keyDown` (`messageType = 1015`)

Sends a key-down event.

Closest macOS/API mirror: `NSEvent` key-down events.

```text
bytes 2..3:   UInt16 keyCode
bytes 4..11:  StringRef32 characters
bytes 12..19: StringRef32 charactersIgnoringModifiers
bytes 20..27: UInt64 NSEvent.ModifierFlags rawValue
byte 28:      UInt8 flags
              bit 0 = isARepeat
```

### `keyUp` (`messageType = 1016`)

Sends a key-up event.

Closest macOS/API mirror: `NSEvent` key-up events.

```text
bytes 2..3:   UInt16 keyCode
bytes 4..11:  StringRef32 characters
bytes 12..19: StringRef32 charactersIgnoringModifiers
bytes 20..27: UInt64 NSEvent.ModifierFlags rawValue
byte 28:      UInt8 flags
              bit 0 = isARepeat
```

### `magnification` (`messageType = 1017`)

Sends an in-progress magnification gesture for a surface.

Closest macOS/API mirror: `NSEvent` magnify gesture events or `NSMagnificationGestureRecognizer`, plus custom surface and scroll-offset context.

```text
bytes 2..5:   UInt32 surfaceID
bytes 6..13:  float64 magnification
bytes 14..21: float64 x
bytes 22..29: float64 y
bytes 30..37: float64 scrollX
bytes 38..45: float64 scrollY
```

### `magnificationEnded` (`messageType = 1018`)

Sends the final magnification gesture state for a surface.

Closest macOS/API mirror: `NSEvent` magnify gesture ending or `NSMagnificationGestureRecognizer` ending state, plus custom surface and scroll-offset context.

```text
bytes 2..5:   UInt32 surfaceID
bytes 6..13:  float64 magnification
bytes 14..21: float64 x
bytes 22..29: float64 y
bytes 30..37: float64 scrollX
bytes 38..45: float64 scrollY
```

### `quickLook` (`messageType = 1019`)

Requests a Quick Look-style action at a point.

Closest macOS/API mirror: Quick Look / lookup-style AppKit actions. There is no exact single API in this protocol.

```text
bytes 2..9:   float64 x
bytes 10..17: float64 y
```

### `textInput` (`messageType = 1020`)

Inserts committed text, optionally replacing a range.

Closest macOS/API mirror: `NSTextInputClient.insertText(_:replacementRange:)`.

```text
bytes 2..9:   StringRef32 text
byte 10:      UInt8 flags
              bit 0 = hasReplacementRange
bytes 11..18: UInt64 replacementLocation
bytes 19..26: UInt64 replacementLength
```

### `setMarkedText` (`messageType = 1021`)

Sets marked text during text composition.

Closest macOS/API mirror: `NSTextInputClient.setMarkedText(_:selectedRange:replacementRange:)`.

```text
bytes 2..9:   StringRef32 text
bytes 10..17: UInt64 selectedLocation
bytes 18..25: UInt64 selectedLength
byte 26:      UInt8 flags
              bit 0 = hasReplacementRange
bytes 27..34: UInt64 replacementLocation
bytes 35..42: UInt64 replacementLength
```

### `unmarkText` (`messageType = 1022`)

Clears marked text.

Closest macOS/API mirror: `NSTextInputClient.unmarkText()`.

```text
message-specific bytes: empty
```

### `textInputFocus` (`messageType = 1023`)

Notifies content that a text input field gained or lost focus.

Closest macOS/API mirror: `NSResponder.becomeFirstResponder()` / `resignFirstResponder()`, though this message is field-ID based.

```text
bytes 2..17: UUID fieldID
byte 18:     UInt8 flags
             bit 0 = hasFocus
```

### `textCommand` (`messageType = 1024`)

Sends a text editing command string.

Closest macOS/API mirror: `NSTextInputClient.doCommand(by:)` / `NSResponder` selector-style text commands.

```text
bytes 2..9: StringRef32 command
```

### `setCursorPosition` (`messageType = 1025`)

Moves a content text cursor, optionally modifying the selection.

Closest macOS/API mirror: text selection APIs such as `NSTextView.selectedRange`, though the `fieldID` makes this protocol-specific.

```text
bytes 2..17:  UUID fieldID
bytes 18..25: UInt64 position
byte 26:      UInt8 flags
              bit 0 = modifySelection
```

### `systemAppearanceUpdate` (`messageType = 1005`)

Sends the current effective system appearance.

Closest macOS/API mirror: `NSApp.effectiveAppearance` / `NSAppearance`.

```text
bytes 2..9: DataRef32 macOS NSKeyedArchiver payload for NSAppearance
```

### `windowActiveUpdate` (`messageType = 1006`)

Sends whether the host window is active.

Closest macOS/API mirror: `NSWindow` key/main state and application activation notifications.

```text
byte 2: UInt8 flags
        bit 0 = isActive
```

### `viewFocusChanged` (`messageType = 1007`)

Sends whether the outerframe view has focus.

Closest macOS/API mirror: AppKit first-responder focus changes.

```text
byte 2: UInt8 flags
        bit 0 = isFocused
```

### `copySelectedPasteboardRequest` (`messageType = 1026`)

Requests pasteboard items for the current selection.

Closest macOS/API mirror: `copy(_:)` and `NSPasteboard`, but this is a request/response protocol instead of direct pasteboard access.

```text
bytes 2..17: UUID requestID
```

### `pasteboardContentDelivered` (`messageType = 1027`)

Delivers pasteboard items to content for a paste operation.

Closest macOS/API mirror: `NSPasteboard` and `NSPasteboardItem`.

```text
bytes 2..3:              UInt16 item count, N
bytes 4..<4+16*N:        N pasteboard item records
pasteboard item record i, B = 4 + 16*i:
  bytes B..B+7:          StringRef32 pasteboard type identifier
  bytes B+8..B+15:       DataRef32 item data
```

### `accessibilitySnapshotRequest` (`messageType = 1028`)

Requests an accessibility tree snapshot from content.

Closest macOS/API mirror: `NSAccessibility`.

```text
bytes 2..17: UUID requestID
```

### `historyEntryAccepted` (`messageType = 1029`)

Acknowledges that a browser-side history entry was accepted.

Closest macOS/API mirror: no direct AppKit API. Conceptually close to browser navigation history such as `WKBackForwardList`.

```text
bytes 2..17:  UUID entryID
bytes 18..25: StringRef32 URL
```

### `historyEntryRejected` (`messageType = 1030`)

Reports that a browser-side history entry was rejected.

Closest macOS/API mirror: no direct AppKit API.

```text
bytes 2..17:  UUID entryID
bytes 18..25: StringRef32 errorMessage
```

### `historyTraversal` (`messageType = 1031`)

Notifies content of a committed history traversal.

Closest macOS/API mirror: no direct AppKit API. Conceptually close to browser back/forward navigation state.

```text
bytes 2..17:  UUID entryID
bytes 18..25: StringRef32 URL
```

### `historyContextUpdate` (`messageType = 1032`)

Sends the current browser history context.

Closest macOS/API mirror: no direct AppKit API. Conceptually close to browser back/forward list state.

```text
bytes 2..17:  UUID currentEntryID
bytes 18..25: StringRef32 URL
bytes 26..29: UInt32 history length
byte 30:      UInt8 flags
              bit 0 = canGoBack
              bit 1 = canGoForward
```

### `shutdown` (`messageType = 1002`)

Asks content to shut down.

Closest macOS/API mirror: application/process lifecycle cleanup; no direct AppKit event.

```text
message-specific bytes: empty
```

## Content to Browser Messages

### `startDisplayLink` (`messageType = 2000`)

Requests a browser-managed display-link callback.

Closest macOS/API mirror: `CVDisplayLink` or `CADisplayLink` registration.

```text
bytes 2..17: UUID callbackID
```

### `stopDisplayLink` (`messageType = 2001`)

Stops a browser-managed display-link callback.

Closest macOS/API mirror: invalidating/stopping a `CVDisplayLink` or `CADisplayLink`.

```text
bytes 2..17: UUID browserCallbackID
```

### `cursorUpdate` (`messageType = 2002`)

Requests a host cursor change.

Closest macOS/API mirror: `NSCursor`.

```text
byte 2: UInt8 cursorType
        0 = arrow
        1 = iBeam
        2 = crosshair
        3 = openHand
        4 = closedHand
        5 = pointingHand
        6 = resizeLeft
        7 = resizeRight
        8 = resizeLeftRight
        9 = resizeUp
        10 = resizeDown
        11 = resizeUpDown
```

### `inputModeUpdate` (`messageType = 2003`)

Requests text-input and/or raw-key input handling from the host.

Closest macOS/API mirror: `NSTextInputClient` participation plus raw `NSEvent` keyboard forwarding.

```text
byte 2: UInt8 inputMode bitmask
        bit 0 = textInput
        bit 1 = rawKeys
```

### `showContextMenu` (`messageType = 2005`)

Requests a context menu for attributed text at a content point.

Closest macOS/API mirror: AppKit contextual menus (`NSMenu`) and `NSAttributedString` RTF serialization.

```text
bytes 2..9:   float64 locationX
bytes 10..17: float64 locationY
bytes 18..25: DataRef32 attributed text as RTF data
```

### `showDefinition` (`messageType = 2006`)

Requests the host definition/lookup UI for attributed text at a content point.

Closest macOS/API mirror: AppKit lookup/definition presentation for `NSAttributedString`; payload text is RTF data.

```text
bytes 2..9:   float64 locationX
bytes 10..17: float64 locationY
bytes 18..25: DataRef32 attributed text as RTF data
```

### `textCursorUpdate` (`messageType = 2004`)

Sends text cursor rectangles for host IME/caret UI.

Closest macOS/API mirror: `NSTextInputClient.firstRect(forCharacterRange:actualRange:)` and caret rect tracking.

```text
bytes 2..5:           UInt32 cursor count, N
bytes 6..<6+49*N:     N cursor records
cursor record i, B = 6 + 49*i:
  bytes B..B+15:      UUID fieldID
  bytes B+16..B+23:   float64 rect.origin.x
  bytes B+24..B+31:   float64 rect.origin.y
  bytes B+32..B+39:   float64 rect.size.width
  bytes B+40..B+47:   float64 rect.size.height
  byte B+48:          UInt8 flags
                      bit 0 = visible
```

### `copySelectedPasteboardResponse` (`messageType = 2008`)

Responds to a copy request with pasteboard items.

Closest macOS/API mirror: `NSPasteboard` and `NSPasteboardItem`.

```text
bytes 2..17:             UUID requestID
bytes 18..19:            UInt16 item count, N
bytes 20..<20+16*N:      N pasteboard item records
pasteboard item record i, B = 20 + 16*i:
  bytes B..B+7:          StringRef32 pasteboard type identifier
  bytes B+8..B+15:       DataRef32 item data
```

### `openNewWindow` (`messageType = 2012`)

Requests that the host open a new window for a URL.

Closest macOS/API mirror: `NSWindow` creation and browser-style `WKUIDelegate` new-window handling.

```text
bytes 2..9:   StringRef32 URL
byte 10:      UInt8 flags
              bit 0 = hasDisplayString
              bit 1 = hasPreferredSize
bytes 11..18: StringRef32 displayString, empty when absent
bytes 19..26: float64 preferredSize.width, 0 when absent
bytes 27..34: float64 preferredSize.height, 0 when absent
```

### `setPasteboardCapabilities` (`messageType = 2009`)

Updates whether content can copy/cut and what pasteboard types it accepts.

Closest macOS/API mirror: `NSPasteboard`, `validateUserInterfaceItem(_:)`, and edit-menu enablement.

```text
byte 2:             UInt8 flags
                    bit 0 = canCopy
                    bit 1 = canCut
bytes 3..4:         UInt16 pasteboard type count, N
bytes 5..<5+8*N:    N pasteboard type references
pasteboard type reference i, B = 5 + 8*i:
  bytes B..B+7:     StringRef32 pasteboard type identifier
```

### `accessibilitySnapshotResponse` (`messageType = 2010`)

Responds with a serialized content-provided accessibility tree.

Closest macOS/API mirror: `NSAccessibility`.

```text
bytes 2..17:  UUID requestID
byte 18:      UInt8 flags
              bit 0 = hasSnapshotData
bytes 19..26: DataRef32 OuterframeAccessibilitySnapshot serializedData(), empty when absent
```

The referenced snapshot data has its own fixed-size header and flat node table. Offsets inside the snapshot are relative to snapshot byte 0.

```text
Snapshot:
bytes 0..3:    UInt32 format version, currently 1
bytes 4..7:    UInt32 node record size, currently 74
bytes 8..15:   DataRef32 node records, length must be N * nodeRecordSize
remaining:     N fixed-size node records, then variable-length UTF-8 string data
```

Node records are ordered so a child node's `parentIndex` is always less than the child's own node index. Root nodes use `UInt32.max` as their `parentIndex`.

```text
Node record, currently 74 bytes:
bytes 0..3:    UInt32 identifier
bytes 4..7:    UInt32 parentIndex, or UInt32.max for a root node
bytes 8..15:   float64 frame.origin.x
bytes 16..23:  float64 frame.origin.y
bytes 24..31:  float64 frame.size.width
bytes 32..39:  float64 frame.size.height
bytes 40..47:  StringRef32 label UTF-8 bytes, empty when absent
bytes 48..55:  StringRef32 value UTF-8 bytes, empty when absent
bytes 56..63:  StringRef32 hint UTF-8 bytes, empty when absent
bytes 64..67:  Int32 rowCount, 0 when absent
bytes 68..71:  Int32 columnCount, 0 when absent
byte 72:       UInt8 OuterframeAccessibilityRole rawValue
byte 73:       UInt8 flags
               bit 0 = hasLabel
               bit 1 = hasValue
               bit 2 = hasHint
               bit 3 = hasRowCount
               bit 4 = hasColumnCount
               bit 5 = isEnabled
```

Node record offsets must point within the snapshot data. String offsets must point after the node table and within the snapshot data. Fields marked absent by `flags` must have zero values.

### `accessibilityTreeChanged` (`messageType = 2011`)

Requests that the host post accessibility change notifications.

Closest macOS/API mirror: `NSAccessibility.post(element:notification:)`.

```text
byte 2: UInt8 notificationMask
        bit 0 = layoutChanged
        bit 1 = selectedChildrenChanged
        bit 2 = focusedElementChanged
```

### `hapticFeedback` (`messageType = 2007`)

Requests haptic feedback.

Closest macOS/API mirror: `NSHapticFeedbackManager`.

```text
byte 2: UInt8 style
        0 = generic
        1 = alignment
        2 = levelChange
```

### `historyPushEntry` (`messageType = 2013`)

Requests that the host push a browser history entry.

Closest macOS/API mirror: no direct AppKit API. Conceptually mirrors the web History API `pushState`.

```text
bytes 2..17:  UUID entryID
byte 18:      UInt8 flags
              bit 0 = hasURL
bytes 19..26: StringRef32 URL, empty when absent
```

### `historyReplaceEntry` (`messageType = 2014`)

Requests that the host replace a browser history entry.

Closest macOS/API mirror: no direct AppKit API. Conceptually mirrors the web History API `replaceState`.

```text
bytes 2..17:  UUID entryID
byte 18:      UInt8 flags
              bit 0 = hasURL
bytes 19..26: StringRef32 URL, empty when absent
```

### `historyGo` (`messageType = 2015`)

Requests relative browser history traversal.

Closest macOS/API mirror: no direct AppKit API. Conceptually mirrors the web History API `history.go(delta)`.

```text
bytes 2..5: Int32 delta
```

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

### `selectionToPasteboardCopyRequest` (`messageType = 1026`)

Requests pasteboard items for the current selection.

Closest macOS/API mirror: `copy(_:)` and `NSPasteboard`, but this is a request/response protocol instead of direct pasteboard access.

```text
bytes 2..17: UUID requestID
```

### `selectionToPasteboardCutRequest` (`messageType = 1037`)

Requests pasteboard items for the current selection and asks content to remove that selection as part of the same user-initiated cut operation. Content replies with `selectionToPasteboardResponse`.

Closest macOS/API mirror: `cut(_:)` and `NSPasteboard`, but this is a request/response protocol instead of direct pasteboard access.

```text
bytes 2..17: UUID requestID
```

### `editCommandValidationRequest` (`messageType = 1040`)

Synchronously asks which edit commands content can currently perform.

Closest macOS/API mirror: `validateUserInterfaceItem(_:)` and edit-menu validation.

```text
bytes 2..17: UUID requestID
bytes 18..21: UInt32 requestedCommands bitmask
              bit 0 = copy
              bit 1 = cut
              bit 2 = paste
              bit 3 = selectAll
              bit 4 = find
              bit 5 = findNext
              bit 6 = findPrevious
```

### `pasteboardContentPasted` (`messageType = 1027`)

Delivers pasteboard items to content for a paste operation.

Closest macOS/API mirror: `NSPasteboard` and `NSPasteboardItem`.

```text
bytes 2..3:              UInt16 pasteboard item count, N
then N pasteboard item records:
  UInt16 representation count, R
  then R representation records:
    StringRef32 pasteboard type identifier
    DataRef32 representation data
```

### `pasteboardContentDropped` (`messageType = 1035`)

Delivers pasteboard items to content as the result of a native drop onto the outerframe view. The host reads `NSPasteboard` and converts accepted items into the socket pasteboard record format so sandboxed content never needs direct pasteboard access.

Closest macOS/API mirror: `NSDraggingDestination.performDragOperation(_:)`, `NSPasteboard`, and `NSPasteboardItem`.

```text
bytes 2..9:              float64 locationX
bytes 10..17:            float64 locationY
bytes 18..19:            UInt16 pasteboard item count, N
then N pasteboard item records:
  UInt16 representation count, R
  then R representation records:
    StringRef32 pasteboard type identifier
    DataRef32 representation data
```

### `pasteboardDropHitTestRequest` (`messageType = 1038`)

Synchronously asks content whether a native drag should be accepted at a view-local point. The host sends this only when pasteboard acceptance mode is `hitTest`. The host does not use the accepted pasteboard type list as a validation prefilter in this mode; content receives the visible pasteboard type identifiers and makes the type and location decision.

Closest macOS/API mirror: `NSDraggingDestination.draggingEntered(_:)` and `draggingUpdated(_:)`.

```text
bytes 2..17:             UUID requestID
bytes 18..25:            float64 locationX
bytes 26..33:            float64 locationY
bytes 34..37:            UInt32 source operation mask
bytes 38..45:            UInt64 modifier flags rawValue
bytes 46..47:            UInt16 pasteboard type count, N
bytes 48..<48+8*N:       N pasteboard type references
pasteboard type reference i, B = 48 + 8*i:
  bytes B..B+7:          StringRef32 pasteboard type identifier
```

Content should reply with `pasteboardDropHitTestResponse` immediately. The host may time out and reject the location if content does not answer within the host's validation timeout.

### `pasteboardAccessResponse` (`messageType = 1034`)

Responds to a content-originated programmatic pasteboard access request.

Closest macOS/API mirror: `NSPasteboard`, with host policy gating. Context-menu and keyboard edit commands use host-originated messages such as `selectionToPasteboardCopyRequest` / `pasteboardContentPasted`; this response is only for content-originated access requests.

```text
bytes 2..17:             UUID requestID
byte 18:                 UInt8 flags
                         bit 0 = granted
bytes 19..20:            UInt16 pasteboard item count, N
then N pasteboard item records:
  UInt16 representation count, R
  then R representation records:
    StringRef32 pasteboard type identifier
    DataRef32 representation data
```

### `accessibilitySnapshotRequest` (`messageType = 1028`)

Requests an accessibility tree snapshot from content. The browser sends this while handling a host accessibility query and waits synchronously for the matching `accessibilitySnapshotResponse` until a browser-defined timeout expires. The browser does not return a previously cached snapshot while waiting and does not keep a stale host-side snapshot after content reports a tree change.

Content should compute and send the response promptly on its normal UI state. If content cannot provide a tree, it should still answer with an absent snapshot or an explicit not-implemented snapshot rather than deferring the response to unrelated async work.

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

### `contextMenuItemSelected` (`messageType = 1033`)

Reports that the user selected a content-command item from a content-provided context menu.

Closest macOS/API mirror: `NSMenuItem` action dispatch. The `menuID` is the content-provided menu identifier from `showContextMenuItems`, and `itemID` is the content-provided identifier for the selected item. Items with a standard action, such as `standardCopy`, are handled by the browser and do not send this message.

```text
bytes 2..17:  UUID menuID
bytes 18..25: StringRef32 itemID
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

### `showContextMenuItems` (`messageType = 2016`)

Requests a context menu made from content-provided menu item snapshots at a content point. Each item carries a kind, action, state, optional styling, and optional children. When the user chooses an enabled command item whose action is `contentCommand`, the browser sends `contextMenuItemSelected` back to content with the same `menuID` and that item's `itemID`. Standard actions are handled by the browser. The optional attributed text is the selected text used by standard lookup and Services items.

Closest macOS/API mirror: AppKit contextual menus (`NSMenu` and `NSMenuItem`) with target/action callbacks.

```text
bytes 2..17:             UUID menuID
bytes 18..25:            float64 locationX
bytes 26..33:            float64 locationY
byte 34:                 UInt8 flags
                         bit 0 = has attributed text
bytes 35..36:            UInt16 root item count, N
bytes 37..44:            DataRef32 attributed text as RTF data, or empty data if not present
bytes 45..end:           N recursive menu item records

menu item record:
  byte +0:               UInt8 kind
                         0 = command
                         1 = separator
                         2 = submenu
                         3 = label
  byte +1:               UInt8 action
                         0 = contentCommand
                         1 = standardCopy
                         2 = standardPaste
                         3 = standardCut
                         4 = standardSelectAll
                         5 = standardLookUp
                         6 = standardServices
  byte +2:               UInt8 isEnabled
  byte +3:               UInt8 state
                         0 = off
                         1 = on
                         2 = mixed
  bytes +4..+5:          UInt16 indentationLevel
  bytes +6..+7:          UInt16 child item count, C
  bytes +8..+11:         UInt32 keyEquivalentModifierMask
  bytes +12..+15:        float32 style.height, or 0 for default
  bytes +16..+19:        float32 style.topInset
  bytes +20..+23:        float32 style.leftInset
  bytes +24..+27:        float32 style.bottomInset
  bytes +28..+31:        float32 style.rightInset
  bytes +32..+35:        float32 style.fontSize, or 0 for default
  bytes +36..+39:        float32 style.fontWeight, AppKit NSFont.Weight raw value or 0 for default
  bytes +40..+43:        UInt32 style.textColorRGBA, 0 for default
  byte +44:              UInt8 style.alignment
                         0 = natural
                         1 = left
                         2 = center
                         3 = right
  bytes +45..+47:        reserved, must be 0
  bytes +48..+55:        StringRef32 itemID
  bytes +56..+63:        StringRef32 title
  bytes +64..+71:        StringRef32 keyEquivalent
  bytes +72..+79:        StringRef32 systemImageName
  bytes +80..:           C child item records
```

`command` items render as native `NSMenuItem`s. `separator` items render as native separators. `submenu` items render as native `NSMenuItem`s with a child `NSMenu`. `label` items render as disabled custom rows using the supplied style fields. Empty item arrays are valid on the wire, but the current browser UI does not show a menu for them.

`standardCopy` uses the existing pasteboard request/response path: the browser sends `selectionToPasteboardCopyRequest`, content replies with `selectionToPasteboardResponse`, and the browser writes the returned pasteboard items to `NSPasteboard`.

`standardPaste` uses the browser's normal paste path. The browser reads `NSPasteboard` under host policy and delivers accepted pasteboard items using `pasteboardContentPasted`.

`standardCut`, `standardSelectAll`, `standardLookUp`, and `standardServices` use the browser's normal AppKit edit/lookup/services paths. `standardLookUp` and `standardServices` use the optional attributed text carried by this message.

### `showDefinition` (`messageType = 2006`)

Requests the host definition/lookup UI for attributed text at a content point.

Closest macOS/API mirror: AppKit lookup/definition presentation for `NSAttributedString`; payload text is RTF data.

```text
bytes 2..9:   float64 locationX
bytes 10..17: float64 locationY
bytes 18..25: DataRef32 attributed text as RTF data
```

### `textInputGeometryUpdate` (`messageType = 2004`)

Sends the active text input geometry used by host text input APIs, including IME candidate window placement. Content is responsible for rendering its own caret and selection UI.

Closest macOS/API mirror: `NSTextInputClient.firstRect(forCharacterRange:actualRange:)`.

```text
byte 2:        UInt8 flags
               bit 0 = has geometry
if bit 0 is set:
  bytes 3..18:    UUID fieldID
  bytes 19..26:   float64 rect.origin.x
  bytes 27..34:   float64 rect.origin.y
  bytes 35..42:   float64 rect.size.width
  bytes 43..50:   float64 rect.size.height
```

### `selectionToPasteboardResponse` (`messageType = 2008`)

Responds to a copy request with pasteboard items.

Closest macOS/API mirror: `NSPasteboard` and `NSPasteboardItem`.

```text
bytes 2..17:             UUID requestID
bytes 18..19:            UInt16 pasteboard item count, N
then N pasteboard item records:
  UInt16 representation count, R
  then R representation records:
    StringRef32 pasteboard type identifier
    DataRef32 representation data
```

### `pasteboardAccessRequest` (`messageType = 2017`)

Requests programmatic pasteboard access from the host. The host decides whether to grant the request. The current browser allows writes by default and shows confirmation UI before programmatic reads.

This is intentionally distinct from host-originated edit commands. For example, a context-menu `standardCopy` item is handled by the browser and uses `selectionToPasteboardCopyRequest`, so the host can always allow that user-selected menu operation while still prompting or denying programmatic reads.

```text
bytes 2..17:             UUID requestID
byte 18:                 UInt8 operation
                         0 = read
                         1 = write
bytes 19..20:            UInt16 requested pasteboard type count, T
bytes 21..22:            UInt16 item count, N
bytes 23..<23+8*T:       T requested pasteboard type references
pasteboard type reference i, B = 23 + 8*i:
  bytes B..B+7:          StringRef32 pasteboard type identifier
bytes 23+8*T..<...:      N pasteboard item records
pasteboard item record:
  UInt16 representation count, R
  then R representation records:
    StringRef32 pasteboard type identifier
    DataRef32 representation data
```

### `beginDraggingPasteboardItems` (`messageType = 2018`)

Requests that the host begin a native drag session using content-provided pasteboard items. The host owns the `NSDraggingSession` and writes native pasteboard representations, allowing content to offer files or other drag sources without direct pasteboard access.

Closest macOS/API mirror: `NSDraggingSource`, `NSDraggingItem`, `NSPasteboardItem`, and `NSFilePromiseProvider`.

```text
bytes 2..5:              UInt32 NSDragOperation raw operation mask
bytes 6..7:              UInt16 item count, N
then N dragging item records:
  UInt16 pasteboard representation count, R
  then R representation records:
    StringRef32 pasteboard type identifier
    DataRef32 representation data
  UInt8 flags
                          bit 0 = has drag preview image
                          bit 1 = has explicit drag preview frame origin
  DataRef32 PNG drag preview image data, empty when absent
  float64 drag preview width in source-view points, 0 when absent
  float64 drag preview height in source-view points, 0 when absent
  if bit 1 is set:
    float64 drag preview frame minX in source-view points
    float64 drag preview frame minY in source-view points
```

The drag preview fields affect only source-side drag UI. They are not written to `NSPasteboard` and are not transferable data. Content may provide higher-resolution PNG data than the logical point size for Retina rendering. When content provides an explicit preview frame origin, the host uses that source-view point as the initial drag image origin; otherwise the host places the preview near the drag start point.

The host recognizes these outerframe private pasteboard type identifiers:

```text
org.outerframe.file-promise
  Content-created binary metadata for a file that content can stage later if Finder requests it:
    bytes 0..3:    UInt32 version, currently 1
    bytes 4..7:    UInt32 flags, currently 0
    bytes 8..23:   UUID content-owned file promise ID
    bytes 24..31:  UInt64 file size, UInt64.max when absent
    bytes 32..39:  StringRef32 promised file name
    bytes 40..47:  StringRef32 file type identifier, empty when absent
    bytes 48..<L:  variable-length region
  The host exposes this as an NSFilePromiseProvider. If Finder requests the file, the host sends `filePromiseWriteRequest`; content stages the file and replies with `filePromiseWriteResponse`.

org.outerframe.dropped-file-access
  Host-created binary metadata for a dropped local file that has already been made readable to content under OUTERFRAME_STAGING_DIR.
  This appears as a pasteboard representation inside `pasteboardContentDropped`; the bytes below are that representation's DataRef32 payload:
    bytes 0..3:    UInt32 version, currently 1
    bytes 4..7:    UInt32 flags
                    bit 0 = is directory
    bytes 8..23:   UUID dropped file access ID
    bytes 24..31:  UInt64 file size, UInt64.max when absent
    bytes 32..39:  StringRef32 file name
    bytes 40..47:  StringRef32 file type identifier, empty when absent
    bytes 48..55:  StringRef32 local staged file path
    bytes 56..<L:  variable-length region
  Content should call `releaseDroppedFileAccess` with the access ID when it no longer needs the file.
```

### `filePromiseWriteRequest` (`messageType = 1039`)

Sent by the host to content when a native destination consumes an `org.outerframe.file-promise`.

Closest macOS/API mirror: `NSFilePromiseProviderDelegate.filePromiseProvider(_:writePromiseTo:completionHandler:)`.

```text
bytes 2..17:   UUID requestID
bytes 18..33:  UUID content-owned file promise ID
```

Content should write or download the promised file into `OUTERFRAME_STAGING_DIR`, then reply with `filePromiseWriteResponse`. The host copies the staged file into the native destination.

### `filePromiseWriteResponse` (`messageType = 2027`)

Responds to `filePromiseWriteRequest`.

```text
bytes 2..17:   UUID requestID
bytes 18..33:  UUID content-owned file promise ID
bytes 34:      UInt8 flags
                bit 0 = success
                bit 1 = delete staged file after a successful promise write
bytes 35..42:  StringRef32 local staged file path, empty on failure
bytes 43..50:  StringRef32 error message, empty on success
bytes 51..<L:  variable-length region
```

Successful paths must be under `OUTERFRAME_STAGING_DIR`; the host validates this before copying. The response carries a local staged path, not the file bytes.

### `releaseDroppedFileAccess` (`messageType = 2026`)

Releases a dropped-file access lease previously delivered as `org.outerframe.dropped-file-access`. The host removes temporary staged files for that access ID. The host also releases all remaining access leases when the content process exits or disconnects.

```text
bytes 2..17:             UUID accessID
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

### `navigate` (`messageType = 2028`)

Requests that the host navigate the current browser tab to a URL.

Closest macOS/API mirror: browser-style current-tab navigation, similar to `WKWebView.load(_:)`.

```text
bytes 2..9: StringRef32 URL
```

### `openNewTab` (`messageType = 2029`)

Requests that the host open a new browser tab for a URL.

Closest macOS/API mirror: browser-style new-tab handling. AppKit has no direct equivalent.

```text
bytes 2..9:   StringRef32 URL
byte 10:      UInt8 flags
              bit 0 = hasDisplayString
bytes 11..18: StringRef32 displayString, empty when absent
```

### `setTitle` (`messageType = 2030`)

Updates the host presentation title for the current outerframe content instance. Browser-like hosts may use this for tab titles, address-bar titles, window titles, or other chrome. Hosts where this metadata does not make sense may ignore it.

Closest macOS/API mirror: document/window presentation metadata such as `NSWindow.title`.

```text
byte 2:      UInt8 flags
             bit 0 = hasTitle
bytes 3..10: StringRef32 title, empty when absent
```

When `hasTitle` is not set, the host clears the content-provided title and may fall back to bundle, URL, or host defaults.

### `setIcon` (`messageType = 2031`)

Updates the host presentation icon for the current outerframe content instance. Browser-like hosts may use this for tab icons, address-bar icons, or other chrome. Hosts where this metadata does not make sense may ignore it.

Closest macOS/API mirror: document/window presentation metadata such as represented file icons, plus browser tab icons.

```text
byte 2:      UInt8 iconKind
             0 = none
             1 = bundleResource
             2 = stagedFile
bytes 3..10: StringRef32 iconPath, empty when iconKind = none
```

When `iconKind = none`, the host clears the content-provided icon and may fall back to bundle or host defaults.

`bundleResource` paths are resolved relative to the loaded content bundle. `stagedFile` paths are resolved inside `OUTERFRAME_STAGING_DIR`; hosts may also accept absolute paths that still resolve inside that directory. Hosts must reject paths that resolve outside these already-authorized roots.

### `editCommandValidationResponse` (`messageType = 2009`)

Replies to `editCommandValidationRequest`.

Closest macOS/API mirror: `validateUserInterfaceItem(_:)` and edit-menu enablement.

```text
bytes 2..17: UUID requestID
bytes 18..21: UInt32 enabledCommands bitmask
              bit 0 = copy
              bit 1 = cut
              bit 2 = paste
              bit 3 = selectAll
              bit 4 = find
              bit 5 = findNext
              bit 6 = findPrevious
```

### `setPasteboardDropBehaviorUniform` (`messageType = 2021`)

Sets uniform whole-view drag-and-drop validation, and sets which pasteboard type identifiers the host may accept and serialize for drag-and-drop.

Closest macOS/API mirror: `NSPasteboard.PasteboardType`.

```text
bytes 2..3:       UInt16 pasteboard type count, N
bytes 4..<4+8*N:  N pasteboard type references
pasteboard type reference i, B = 4 + 8*i:
  bytes B..B+7:   StringRef32 pasteboard type identifier
```

An empty list disables drop acceptance. A non-empty list accepts drops across the whole outerframe view when the native pasteboard contains one of these types. This message exits hit-test behavior.

### `setAcceptedPasteboardPasteTypes` (`messageType = 2022`)

Sets which pasteboard type identifiers the host may accept and serialize for user-initiated paste.

Closest macOS/API mirror: `NSPasteboard.PasteboardType`.

```text
bytes 2..3:       UInt16 pasteboard type count, N
bytes 4..<4+8*N:  N pasteboard type references
pasteboard type reference i, B = 4 + 8*i:
  bytes B..B+7:   StringRef32 pasteboard type identifier
```

An empty list disables host-delivered paste. This does not affect drag-and-drop.

### `pasteboardDropHitTestResponse` (`messageType = 2023`)

Responds to `pasteboardDropHitTestRequest`.

Closest macOS/API mirror: `NSDragOperation` returned by `NSDraggingDestination` validation.

```text
bytes 2..17:   UUID requestID
bytes 18..21:  UInt32 accepted operation mask, 0 to reject
```

### `setPasteboardDropBehaviorHitTest` (`messageType = 2024`)

Switches drag-and-drop validation to synchronous content hit-testing.

Closest macOS/API mirror: `NSDraggingDestination` validation.

```text
No payload.
```

When the current drop type list is non-empty, the host validates the current drag location by sending `pasteboardDropHitTestRequest` and waiting briefly for `pasteboardDropHitTestResponse`. The hit-test request includes the native pasteboard types visible to the host; the configured drop type list is still used as the final serialization filter when the drop is delivered. Calling `setPasteboardDropBehaviorUniform` with an empty list disables drops.

### `accessibilitySnapshotResponse` (`messageType = 2010`)

Responds with a serialized content-provided accessibility tree. This is the synchronous reply to `accessibilitySnapshotRequest`; content must echo the request ID. Responses that arrive after the browser timeout may be ignored.

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

Requests that the host post accessibility change notifications. The browser clears any accessibility elements it built from the previous snapshot and posts the requested native notifications. The next host accessibility query causes a new synchronous `accessibilitySnapshotRequest`.

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

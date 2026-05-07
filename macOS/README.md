## How macOS outerframe content works

You provide a dynamically loaded library. On macOS, this is a `dylib` stored inside a bundle.

The core pieces are:
- Your compiled binary exposes a `start` method, which receives a Unix socket file descriptor and a `registerLayer` function pointer
- Your compiled binary does everything via bidirectional messages on the socket
- ...except you'll do a one-time call to `registerLayer` during initialization.

More precisely, to fit within the macOS design, your compiled binary's NSPrincipalClass will implement a ObjC/Swift protocol "OuterframeContentLibrary", which has a `start` method, which receives a socket file descriptor and an "OuterframeAppConnection" instance, which has a `registerLayer` method.

That's the entire core API. There's more to know, but it all exists in the form of socket messages. If you want to use the socket directly, handling your own messaging, you can operate at that level. If you want a more conventional function-based API, the example [Swift](https://github.com/outergroup/hello-outerframe-macOS-swift) and [Objective-C](https://github.com/outergroup/hello-outerframe-macOS-objc) projects both have a "vendored" library that you static-link into your compiled binary. It wraps the socket and provides the asynchronous function-based API that more people expect to use.

The reason the `registerLayer` method exists (and why I didn't implement it as a socket message) that it calls a macOS private API (`CALayerHost`), and I don't want to make outerframe binaries do that. If a future macOS update ever breaks this API, I want to be able to fix it without requiring all outerframe binaries to be updated. I hope Apple someday makes a public API analogous to CALayerHost, as it is vital to having efficient sandboxed UI (Safari and Chrome also use it). The related public API, IOSurface, is less efficient and much less powerful.

The socket messages follow format:

```text
Socket frame:
bytes 0..3:    UInt32 little-endian message length, L
bytes 4..<4+L: message bytes

Message:
bytes 0..1:     UInt16 little-endian messageType
bytes 2..<V:    message-specific fixed-size fields
bytes V..<L:    variable-length region
```

Here's a more specific example: the "mouseEvent" message.

```text
bytes 0..1:   mouseDown (`messageType = 1008`)
bytes 2..9:   float64 x
bytes 10..17: float64 y
bytes 18..25: UInt64 NSEvent.ModifierFlags rawValue
bytes 26..29: UInt32 clickCount
```

Both the browser and the outerframe binary send messages like these. The full set is listed [here](https://github.com/outergroup/outerframe/blob/main/macOS/outerframe-socket-messages.md). This set of messages will evolve over time.

And those are the fundamentals of the outerframe. Everything past this point is just how you structure your code. Sometimes you'll be focused on making scrollable text with mouse-tracking, and appropriate cursor updates and text highlighting. Other times you'll be focused on UI elements. Does your code embrace classes and words like "Controller" and "Delegate"? That's totally up to you; from the outerframe's perspective, you're just someone who registers a CALayer and sends/receives socket messages.

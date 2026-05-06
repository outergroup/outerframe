import Foundation
import Dispatch
import Darwin

@MainActor
protocol OuterframeContentSocketDelegate: AnyObject, Sendable {
    func outerframeContentSocket(_ socket: OuterframeContentSocket, didReceiveMessage message: Data)
    func outerframeContentSocketDidClose(_ socket: OuterframeContentSocket)
}


actor OuterframeContentSocket {
    private let queue: DispatchSerialQueue
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    private nonisolated func assumeIsolatedHack<T>(
           _ block: (isolated OuterframeContentSocket) throws -> T
    ) rethrows -> T where T: Sendable {

        // Before Swift 6, dispatch didn't work well with `Actor.assumeIsolated`.
        // It can report false negatives - where you are actually on the correct queue but 'assumeIsolated'
        // doesn't know it. That was fixed in SE-0424:
        // https://github.com/swiftlang/swift-evolution/blob/main/proposals/0424-custom-isolation-checking-for-serialexecutor.md
        //
        // This feature requires a new version of the Swift runtime, so we need to perform an OS version check
        // and on older systems we need an ugly hack to emulate the new 'assumeIsolated' for Dispatch.

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, *) {
            return try self.assumeIsolated(block)
        } else {
            dispatchPrecondition(condition: .onQueue(self.queue))
            return try withoutActuallyEscaping(block) {
                let isolationStripped = unsafeBitCast($0, to: ((OuterframeContentSocket) throws -> T).self)
                return try isolationStripped(self)
            }
        }
    }

    private var socketFD: Int32 = -1
    private var readSource: SafeDispatchSource?
    private var writeSource: SafeDispatchSource?
    private var pendingWriteBuffer = Data()
    private var incomingBuffer = Data()

    nonisolated(unsafe) weak var delegate: (any OuterframeContentSocketDelegate)?

    private func configureSocketOptions(_ fd: Int32) {
        var one: Int32 = 1
        let result = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout.size(ofValue: one)))
        if result != 0 {
            print("Browser: Failed to set SO_NOSIGPIPE on OuterframeContent socket (errno \(errno))")
        }
    }

    init() {
        queue = DispatchQueue(label: "dev.outergroup.outerframecontent.socket.browser") as! DispatchSerialQueue
    }

    func start(withFileDescriptor fd: Int32) {
        cleanupConnection()

        let flags = fcntl(fd, F_GETFL)
        if flags != -1 {
            _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        }
        configureSocketOptions(fd)

        socketFD = fd

        let rawReadSource = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        rawReadSource.setEventHandler { [weak self] in
            self?.assumeIsolatedHack { actor in
                actor.handleReadable()
            }
        }
        let readSource = SafeDispatchSource(dispatchSource: rawReadSource, isResumed: false)
        readSource.safeResume()
        self.readSource = readSource

        let rawWriteSource = DispatchSource.makeWriteSource(fileDescriptor: fd, queue: queue)
        rawWriteSource.setEventHandler { [weak self] in
            self?.assumeIsolatedHack { actor in
                actor.drainPendingWrites()
            }
        }
        let writeSource = SafeDispatchSource(dispatchSource: rawWriteSource, isResumed: false)
        writeSource.safeSuspend()
        self.writeSource = writeSource
    }

    func stop() {
        cleanupConnection()
        delegate = nil
    }

    func setDelegate(_ delegate: (any OuterframeContentSocketDelegate)?) {
        self.delegate = delegate
    }

    func send(_ data: Data) async throws {
        guard socketFD >= 0 else {
            throw OuterframeContentSocketError.disconnected
        }

        enqueueWrite(data)
    }

    private func handleReadable() {
        guard socketFD >= 0 else { return }

        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let bytesRead = read(socketFD, &buffer, buffer.count)
            if bytesRead > 0 {
                incomingBuffer.append(contentsOf: buffer[0..<bytesRead])
                processIncomingBuffer()
            } else if bytesRead == 0 {
                print("Browser: OuterframeContent socket closed")
                cleanupConnection()
                notifyClosed()
                break
            } else {
                if errno == EWOULDBLOCK || errno == EAGAIN {
                    break
                } else {
                    print("Browser: OuterframeContent socket read error (errno \(errno))")
                    cleanupConnection()
                    notifyClosed()
                    break
                }
            }
        }
    }

    private func drainPendingWrites() {
        guard socketFD >= 0, !pendingWriteBuffer.isEmpty else { return }

        let written = pendingWriteBuffer.withUnsafeBytes { buffer -> Int in
            guard let baseAddress = buffer.baseAddress else { return 0 }
            return write(socketFD, baseAddress, buffer.count)
        }

        if written > 0 {
            pendingWriteBuffer.removeSubrange(0..<written)
        } else if written == -1 && (errno == EWOULDBLOCK || errno == EAGAIN) {
            return
        } else {
            print("Browser: Failed to send OuterframeContent socket message (errno \(errno))")
            cleanupConnection()
            notifyClosed()
            return
        }

        if pendingWriteBuffer.isEmpty {
            writeSource?.safeSuspend()
        }
    }

    private func enqueueWrite(_ data: Data) {
        guard !data.isEmpty else { return }
        pendingWriteBuffer.append(data)
        writeSource?.safeResume()
    }

    private func cleanupConnection() {
        readSource?.cancel()
        readSource = nil
        writeSource?.cancel()
        writeSource = nil
        if socketFD >= 0 {
            close(socketFD)
            socketFD = -1
        }
        pendingWriteBuffer.removeAll(keepingCapacity: false)
        incomingBuffer.removeAll(keepingCapacity: false)
    }

    private func notifyClosed() {
        if let delegate {
            Task { await delegate.outerframeContentSocketDidClose(self) }
        }
    }

    private func processIncomingBuffer() {
        while incomingBuffer.count >= OuterframeContentSocketHeaderLength {
            let lengthRange = incomingBuffer.startIndex..<(incomingBuffer.startIndex + MemoryLayout<UInt32>.size)
            let lengthData = incomingBuffer.subdata(in: lengthRange)
            let messageLength = lengthData.enumerated().reduce(UInt32(0)) {
                $0 | (UInt32($1.element) << (8 * $1.offset))
            }
            let totalLength = OuterframeContentSocketHeaderLength + Int(messageLength)

            guard incomingBuffer.count >= totalLength else { break }

            let messageStart = incomingBuffer.startIndex + OuterframeContentSocketHeaderLength
            let messageEnd = messageStart + Int(messageLength)
            let message = incomingBuffer.subdata(in: messageStart..<messageEnd)

            incomingBuffer.removeSubrange(incomingBuffer.startIndex..<messageEnd)

            if let delegate {
                Task { await delegate.outerframeContentSocket(self, didReceiveMessage: message) }
            }
        }
    }
}

enum OuterframeContentSocketError: Error {
    case disconnected
}

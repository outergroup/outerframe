import Darwin
import Foundation

public struct OuterframeContentRuntime {
    public static func configureOutputBuffering() {
        setvbuf(stdout, nil, _IONBF, 0)
        setvbuf(stderr, nil, _IONBF, 0)
    }

    public static func run(infrastructureSocketFD: Int32,
                           pluginSocketFD: Int32,
                           networkProxyPort: UInt16?,
                           hostBundleIdentifier: String,
                           allowedReadPaths: [String] = [],
                           monitorParentProcess: Bool = true) -> Never {
        print("OuterframeContent starting...")

        let sandboxResult = OuterSandbox.apply(bundleId: hostBundleIdentifier,
                                               allowedProxyPort: networkProxyPort,
                                               allowedReadPaths: allowedReadPaths)
        guard sandboxResult.success else {
            print("OuterframeContent: Failed to apply sandbox: \(sandboxResult.error ?? "Unknown error")")
            exit(0)
        }

        print("OuterframeContent: Outer sandbox applied successfully")
        print("OuterframeContent: Sandbox status: \(OuterSandbox.isSandboxed() ? "sandboxed" : "not sandboxed")")

        Task { @MainActor in
            infraSocketActor = InfraSocket()
            outerlayerHost = OuterlayerHost(socketActor: infraSocketActor,
                                            pluginSocketFD: pluginSocketFD)

            await infraSocketActor.setOuterlayerHost(outerlayerHost)
            await infraSocketActor.start(withFileDescriptor: infrastructureSocketFD)

            signal(SIGINT) { _ in
                outerlayerHost.stop()
                exit(0)
            }

            signal(SIGTERM) { _ in
                outerlayerHost.stop()
                exit(0)
            }
        }

        if monitorParentProcess {
            monitorParentProcessExit()
        }

        RunLoop.main.run()
        fatalError("RunLoop.main.run() returned unexpectedly")
    }
}

nonisolated(unsafe) var infraSocketActor: InfraSocket!
nonisolated(unsafe) var outerlayerHost: OuterlayerHost!
nonisolated(unsafe) var debuggerAttachmentMonitorTask: Task<Void, Never>?

nonisolated(unsafe) private var parentTerminationSource: DispatchSourceProcess?

private func monitorParentProcessExit() {
    let parentPID = getppid()
    guard parentPID > 1 else {
        print("OuterframeContent: Parent already exited, terminating")
        exit(0)
    }

    let source = DispatchSource.makeProcessSource(identifier: pid_t(parentPID),
                                                  eventMask: .exit,
                                                  queue: .main)
    source.setEventHandler {
        print("OuterframeContent: Parent process exited, terminating")
        exit(0)
    }
    source.setCancelHandler {
        print("OuterframeContent: Parent monitor cancelled")
    }
    parentTerminationSource = source
    source.resume()
}

func startDebuggerAttachmentMonitor() {
    debuggerAttachmentMonitorTask?.cancel()
    debuggerAttachmentMonitorTask = Task.detached {
        defer { debuggerAttachmentMonitorTask = nil }
        if debuggerIsAttached() {
            await sendDebuggerAttachedEvent()
            return
        }

        while !Task.isCancelled {
            do {
                try await Task.sleep(nanoseconds: 200_000_000)
            } catch {
                break
            }

            if debuggerIsAttached() {
                await sendDebuggerAttachedEvent()
                break
            }
        }
    }
}

func stopDebuggerAttachmentMonitor() {
    debuggerAttachmentMonitorTask?.cancel()
    debuggerAttachmentMonitorTask = nil
}

private func sendDebuggerAttachedEvent() async {
    guard let infraSocketActor else { return }
    do {
        try await infraSocketActor.send(ContentToBrowserInfraMessage.debuggerAttached.encode())
    } catch {
        print("OuterframeContent: Failed to send debuggerAttached message: \(error)")
    }
}

private func debuggerIsAttached() -> Bool {
    var info = kinfo_proc()
    var size = MemoryLayout<kinfo_proc>.stride
    var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
    return sysctl(&mib, u_int(mib.count), &info, &size, nil, 0) == 0 && (info.kp_proc.p_flag & P_TRACED) != 0
}

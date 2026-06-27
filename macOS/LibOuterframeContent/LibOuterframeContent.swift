import Darwin
import Foundation

private func canonicalURL(for url: URL) -> URL {
    let path = url.path
    let canonicalPath = path.withCString { pointer -> String in
        var resolved = [CChar](repeating: 0, count: Int(PATH_MAX))
        if realpath(pointer, &resolved) != nil {
            return String(cString: resolved)
        }
        return path
    }
    return URL(fileURLWithPath: canonicalPath, isDirectory: true)
}

private func createRuntimeDirectory() throws -> URL {
    let baseURL = canonicalURL(for: FileManager.default.temporaryDirectory)
        .appendingPathComponent("org.outerframe.OuterframeContent", isDirectory: true)
    return baseURL.appendingPathComponent(UUID().uuidString, isDirectory: true)
}

private func configureRuntimeDirectory(_ runtimeURL: URL) throws -> URL {
    try FileManager.default.createDirectory(at: runtimeURL,
                                            withIntermediateDirectories: true)

    let runtimeURL = canonicalURL(for: runtimeURL)
    let cacheURL = runtimeURL.appendingPathComponent("cache", isDirectory: true)

    try FileManager.default.createDirectory(at: cacheURL,
                                            withIntermediateDirectories: true)

    setenv("TMPDIR", runtimeURL.path + "/", 1)
    setenv("XDG_CACHE_HOME", cacheURL.path, 1)

    print("OuterframeContent: Runtime directory: \(runtimeURL.path)")
    return runtimeURL
}

public struct OuterframeContentRuntime {
    public static func configureOutputBuffering() {
        setvbuf(stdout, nil, _IONBF, 0)
        setvbuf(stderr, nil, _IONBF, 0)
    }

    public static func run(infrastructureSocketFD: Int32,
                           pluginSocketFD: Int32,
                           networkProxyPort: UInt16?,
                           hostBundleIdentifier: String,
                           stagedFileDirectoryPath: String? = nil,
                           allowedReadPaths: [String] = []) -> Never {
        print("OuterframeContent starting...")

        let runtimeDirectory: URL
        let effectiveStagedFileDirectoryPath: String?
        do {
            if let providedStagedFileDirectoryPath = stagedFileDirectoryPath,
               !providedStagedFileDirectoryPath.isEmpty {
                let stagedFileDirectoryURL = URL(fileURLWithPath: providedStagedFileDirectoryPath,
                                                 isDirectory: true)
                runtimeDirectory = try configureRuntimeDirectory(stagedFileDirectoryURL)
                effectiveStagedFileDirectoryPath = runtimeDirectory.path
                setenv("OUTERFRAME_STAGING_DIR", runtimeDirectory.path, 1)
                print("OuterframeContent: Staged file directory: \(runtimeDirectory.path)")
            } else {
                runtimeDirectory = try configureRuntimeDirectory(createRuntimeDirectory())
                effectiveStagedFileDirectoryPath = nil
                unsetenv("OUTERFRAME_STAGING_DIR")
            }
        } catch {
            print("OuterframeContent: Failed to create runtime directory: \(error)")
            exit(0)
        }

        let sandboxResult = OuterSandbox.apply(bundleId: hostBundleIdentifier,
                                               allowedProxyPort: networkProxyPort,
                                               allowedReadPaths: allowedReadPaths,
                                               runtimeDirectoryPath: runtimeDirectory.path,
                                               stagedFileDirectoryPath: effectiveStagedFileDirectoryPath)
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

        RunLoop.main.run()
        fatalError("RunLoop.main.run() returned unexpectedly")
    }
}

nonisolated(unsafe) var infraSocketActor: InfraSocket!
nonisolated(unsafe) var outerlayerHost: OuterlayerHost!
nonisolated(unsafe) var debuggerAttachmentMonitorTask: Task<Void, Never>?

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

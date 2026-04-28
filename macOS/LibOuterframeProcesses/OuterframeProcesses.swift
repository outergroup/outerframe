import Foundation
import AppleArchive
import Network
import System

private final class OuterframeBundleDownloadDelegate: NSObject, URLSessionTaskDelegate {
    private let lock = NSLock()
    private var resourceFetchType: URLSessionTaskMetrics.ResourceFetchType?

    var didUseLocalCache: Bool {
        lock.lock()
        defer { lock.unlock() }
        return resourceFetchType == .localCache
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didFinishCollecting metrics: URLSessionTaskMetrics) {
        lock.lock()
        resourceFetchType = metrics.transactionMetrics.last?.resourceFetchType
        lock.unlock()
    }
}

public class OuterframeProcesses: NSObject, OuterframeProcessesProtocol {
    /// Maps instance IDs to PIDs for OuterframeContent processes spawned by this XPC service
    private var outerframeContentInstances: [UUID: pid_t] = [:]
    /// Monitors for process exit to clean up instance mappings
    private var processMonitors: [UUID: DispatchSourceProcess] = [:]
    private let instanceLock = NSLock()

    public func launchOuterframeContent(networkProxyPort: UInt16,
                                        with reply: @escaping (FileHandle?, FileHandle?, FileHandle?, FileHandle?, pid_t, UUID?, NSError?) -> Void) {
        do {
            guard let executablePath = getOuterframeContentExecutablePath() else {
                throw OuterframeProcessesError.executableMissing("OuterframeContent")
            }
            let hostBundleIdentifier = try Self.hostBundleIdentifier()

            func closeDescriptors(_ descriptors: [Int32]) {
                for descriptor in descriptors where descriptor >= 0 {
                    close(descriptor)
                }
            }

            var parentInfraFD: Int32 = -1
            var childInfraFD: Int32 = -1
            var parentPluginFD: Int32 = -1
            var childPluginFD: Int32 = -1
            var stdoutReadFD: Int32 = -1
            var stdoutWriteFD: Int32 = -1
            var stderrReadFD: Int32 = -1
            var stderrWriteFD: Int32 = -1
            var didSucceed = false

            defer {
                if !didSucceed {
                    closeDescriptors([parentInfraFD,
                                      childInfraFD,
                                      parentPluginFD,
                                      childPluginFD,
                                      stdoutReadFD,
                                      stdoutWriteFD,
                                      stderrReadFD,
                                      stderrWriteFD])
                }
            }

            // Create infrastructure socket pair
            var infraSocketPair: [Int32] = [0, 0]
            guard socketpair(AF_UNIX, SOCK_STREAM, 0, &infraSocketPair) == 0 else {
                throw OuterframeProcessesError.posixFailure(code: errno, context: "socketpair (infrastructure)")
            }
            parentInfraFD = infraSocketPair[0]
            childInfraFD = infraSocketPair[1]

            // Create plugin socket pair
            var pluginSocketPair: [Int32] = [0, 0]
            guard socketpair(AF_UNIX, SOCK_STREAM, 0, &pluginSocketPair) == 0 else {
                throw OuterframeProcessesError.posixFailure(code: errno, context: "socketpair (plugin)")
            }
            parentPluginFD = pluginSocketPair[0]
            childPluginFD = pluginSocketPair[1]

            // Create stdout pipe
            var stdoutPipe: [Int32] = [0, 0]
            guard pipe(&stdoutPipe) == 0 else {
                throw OuterframeProcessesError.posixFailure(code: errno, context: "pipe (stdout)")
            }
            stdoutReadFD = stdoutPipe[0]
            stdoutWriteFD = stdoutPipe[1]

            // Create stderr pipe
            var stderrPipe: [Int32] = [0, 0]
            guard pipe(&stderrPipe) == 0 else {
                throw OuterframeProcessesError.posixFailure(code: errno, context: "pipe (stderr)")
            }
            stderrReadFD = stderrPipe[0]
            stderrWriteFD = stderrPipe[1]

            // Clear FD_CLOEXEC on child fds
            let childInfraFlags = fcntl(childInfraFD, F_GETFD)
            if childInfraFlags != -1 {
                _ = fcntl(childInfraFD, F_SETFD, childInfraFlags & ~FD_CLOEXEC)
            }
            let childPluginFlags = fcntl(childPluginFD, F_GETFD)
            if childPluginFlags != -1 {
                _ = fcntl(childPluginFD, F_SETFD, childPluginFlags & ~FD_CLOEXEC)
            }
            let stdoutReadFlags = fcntl(stdoutReadFD, F_GETFD)
            if stdoutReadFlags != -1 {
                _ = fcntl(stdoutReadFD, F_SETFD, stdoutReadFlags | FD_CLOEXEC)
            }
            let stderrReadFlags = fcntl(stderrReadFD, F_GETFD)
            if stderrReadFlags != -1 {
                _ = fcntl(stderrReadFD, F_SETFD, stderrReadFlags | FD_CLOEXEC)
            }

            let pid = try launchOuterframeContentProcess(executablePath: executablePath,
                                                         infrastructureSocketFD: childInfraFD,
                                                         pluginSocketFD: childPluginFD,
                                                         stdoutFD: stdoutWriteFD,
                                                         stderrFD: stderrWriteFD,
                                                         networkProxyPort: networkProxyPort,
                                                         hostBundleIdentifier: hostBundleIdentifier)
            close(childInfraFD)
            childInfraFD = -1
            close(childPluginFD)
            childPluginFD = -1
            close(stdoutWriteFD)
            stdoutWriteFD = -1
            close(stderrWriteFD)
            stderrWriteFD = -1

            let instanceId = UUID()
            instanceLock.lock()
            outerframeContentInstances[instanceId] = pid
            instanceLock.unlock()

            // Monitor process exit to clean up mapping if it exits on its own
            monitorProcessExit(instanceId: instanceId, pid: pid)

            let infraHandle = FileHandle(fileDescriptor: parentInfraFD, closeOnDealloc: true)
            let pluginHandle = FileHandle(fileDescriptor: parentPluginFD, closeOnDealloc: true)
            let stdoutHandle = FileHandle(fileDescriptor: stdoutReadFD, closeOnDealloc: true)
            let stderrHandle = FileHandle(fileDescriptor: stderrReadFD, closeOnDealloc: true)
            didSucceed = true
            reply(infraHandle, pluginHandle, stdoutHandle, stderrHandle, pid, instanceId, nil)
        } catch let error as OuterframeProcessesError {
            reply(nil, nil, nil, nil, 0, nil, error.toNSError())
        } catch {
            reply(nil, nil, nil, nil, 0, nil, error as NSError)
        }
    }

    private func getOuterframeContentExecutablePath() -> String? {
        let bundle = Bundle.main.bundleURL
        let executablePath = bundle
            .appendingPathComponent("Contents")
            .appendingPathComponent("Resources")
            .appendingPathComponent("OuterframeContent")
            .path

        if FileManager.default.fileExists(atPath: executablePath) {
            return executablePath
        }

        return nil
    }

    private func launchOuterframeContentProcess(executablePath: String,
                                                infrastructureSocketFD: Int32,
                                                pluginSocketFD: Int32,
                                                stdoutFD: Int32,
                                                stderrFD: Int32,
                                                networkProxyPort: UInt16,
                                                hostBundleIdentifier: String) throws -> pid_t {
        var pid: pid_t = 0

        var fileActions: posix_spawn_file_actions_t? = nil
        guard posix_spawn_file_actions_init(&fileActions) == 0 else {
            throw OuterframeProcessesError.posixFailure(code: errno, context: "posix_spawn_file_actions_init")
        }

        guard posix_spawn_file_actions_addinherit_np(&fileActions, infrastructureSocketFD) == 0 else {
            posix_spawn_file_actions_destroy(&fileActions)
            throw OuterframeProcessesError.posixFailure(code: errno, context: "posix_spawn_file_actions_addinherit_np (infra)")
        }

        guard posix_spawn_file_actions_addinherit_np(&fileActions, pluginSocketFD) == 0 else {
            posix_spawn_file_actions_destroy(&fileActions)
            throw OuterframeProcessesError.posixFailure(code: errno, context: "posix_spawn_file_actions_addinherit_np (plugin)")
        }

        guard posix_spawn_file_actions_adddup2(&fileActions, stdoutFD, STDOUT_FILENO) == 0 else {
            posix_spawn_file_actions_destroy(&fileActions)
            throw OuterframeProcessesError.posixFailure(code: errno, context: "posix_spawn_file_actions_adddup2 (stdout)")
        }

        guard posix_spawn_file_actions_adddup2(&fileActions, stderrFD, STDERR_FILENO) == 0 else {
            posix_spawn_file_actions_destroy(&fileActions)
            throw OuterframeProcessesError.posixFailure(code: errno, context: "posix_spawn_file_actions_adddup2 (stderr)")
        }

        _ = posix_spawn_file_actions_addclose(&fileActions, stdoutFD)
        _ = posix_spawn_file_actions_addclose(&fileActions, stderrFD)

        var attr: posix_spawnattr_t? = nil
        guard posix_spawnattr_init(&attr) == 0 else {
            posix_spawn_file_actions_destroy(&fileActions)
            throw OuterframeProcessesError.posixFailure(code: errno, context: "posix_spawnattr_init")
        }

        let args = [executablePath,
                    "--infra-socket-fd", "\(infrastructureSocketFD)",
                    "--plugin-socket-fd", "\(pluginSocketFD)",
                    "--network-proxy-port", "\(networkProxyPort)",
                    "--host-bundle-id", hostBundleIdentifier]
        var cArgs: [UnsafeMutablePointer<CChar>?] = args.map { strdup($0) }
        cArgs.append(nil)
        defer {
            for ptr in cArgs where ptr != nil {
                free(ptr)
            }
        }

        let spawnResult = executablePath.withCString { execPath in
            cArgs.withUnsafeMutableBufferPointer { argvBuffer in
                posix_spawn(&pid,
                            execPath,
                            &fileActions,
                            &attr,
                            argvBuffer.baseAddress,
                            nil)
            }
        }

        posix_spawnattr_destroy(&attr)
        posix_spawn_file_actions_destroy(&fileActions)

        if spawnResult != 0 {
            throw OuterframeProcessesError.posixFailure(code: Int32(spawnResult), context: "posix_spawn OuterframeContent")
        }

        return pid
    }

    public func downloadAndExtractBundleWithoutQuarantine(
        downloadURL urlString: String,
        proxyHost: String?,
        proxyPort: UInt16,
        proxyUsername: String?,
        proxyPassword: String?,
        destinationDirectoryURL requestedDestinationURL: URL,
        archiveFilename requestedArchiveFilename: String,
        bypassCache: Bool,
        with reply: @escaping (URL?, NSError?) -> Void) {
        guard let url = URL(string: urlString) else {
            let error = NSError(domain: "OuterframeProcesses",
                                code: 1,
                                userInfo: [NSLocalizedDescriptionKey: "Invalid download URL"])
            reply(nil, error)
            return
        }

        guard requestedDestinationURL.isFileURL else {
            let error = NSError(domain: "OuterframeProcesses",
                                code: 20,
                                userInfo: [NSLocalizedDescriptionKey: "Destination directory must be a file URL"])
            reply(nil, error)
            return
        }

        let cacheBundleIdentifier: String
        do {
            cacheBundleIdentifier = try Self.hostBundleIdentifier()
        } catch let error as OuterframeProcessesError {
            reply(nil, error.toNSError())
            return
        } catch {
            reply(nil, error as NSError)
            return
        }

        let cacheBaseURL = bundleCacheBaseURL(cacheBundleIdentifier: cacheBundleIdentifier).standardizedFileURL
        let destinationURL = requestedDestinationURL.standardizedFileURL
        guard destinationURL.path.hasPrefix(cacheBaseURL.path + "/") else {
            let error = NSError(domain: "OuterframeProcesses",
                                code: 21,
                                userInfo: [NSLocalizedDescriptionKey: "Destination directory is outside the bundle cache"])
            reply(nil, error)
            return
        }
        let archiveFilename = URL(fileURLWithPath: requestedArchiveFilename).lastPathComponent

        var request = URLRequest(url: url)
        request.cachePolicy = bypassCache ? .reloadIgnoringLocalCacheData : .reloadRevalidatingCacheData

        let configuration = URLSessionConfiguration.default
        configuration.requestCachePolicy = request.cachePolicy
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 60
        if let proxyHost, !proxyHost.isEmpty, proxyPort != 0 {
            if let proxyUsername, let proxyPassword {
                guard let endpointPort = NWEndpoint.Port(rawValue: proxyPort) else {
                    let error = NSError(domain: "OuterframeProcesses",
                                        code: 6,
                                        userInfo: [NSLocalizedDescriptionKey: "Invalid SOCKS proxy configuration"])
                    reply(nil, error)
                    return
                }

                let endpoint = NWEndpoint.hostPort(host: NWEndpoint.Host(proxyHost),
                                                   port: endpointPort)
                var proxy = ProxyConfiguration(socksv5Proxy: endpoint)
                proxy.applyCredential(username: proxyUsername, password: proxyPassword)
                proxy.allowFailover = false
                configuration.proxyConfigurations = [proxy]
            }
        }

        let delegate = OuterframeBundleDownloadDelegate()
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        let task = session.dataTask(with: request) { [weak self] data, response, error in
            defer {
                session.finishTasksAndInvalidate()
            }
            if let error = error {
                reply(nil, error as NSError)
                return
            }

            if let httpResponse = response as? HTTPURLResponse,
               !(200...299).contains(httpResponse.statusCode) {
                let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                let error = NSError(domain: "OuterframeProcesses",
                                    code: 5,
                                    userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode) from \(urlString): \(body.trimmingCharacters(in: .whitespacesAndNewlines))"])
                reply(nil, error)
                return
            }

            guard let data = data, !data.isEmpty else {
                let error = NSError(domain: "OuterframeProcesses",
                                    code: 2,
                                    userInfo: [NSLocalizedDescriptionKey: "No data received from \(urlString)"])
                reply(nil, error)
                return
            }

            do {
                let fm = FileManager.default
                if delegate.didUseLocalCache,
                   let bundleURL = self?.findBundle(in: destinationURL) {
                    reply(bundleURL, nil)
                    return
                }

                if fm.fileExists(atPath: destinationURL.path) {
                    try fm.removeItem(at: destinationURL)
                }
                try fm.createDirectory(at: destinationURL, withIntermediateDirectories: true)

                let archiveURL = destinationURL.appendingPathComponent(archiveFilename)
                try data.write(to: archiveURL, options: .atomic)

                let fileExtension = archiveURL.pathExtension.lowercased()

                if fileExtension == "bundle" {
                    reply(archiveURL, nil)
                    return
                }

                let archiveBasedExtensions: Set<String> = ["aar", "aa", "archive", "outerframe"]
                let shouldTreatAsArchive = archiveBasedExtensions.contains(fileExtension) || fileExtension.isEmpty

                if shouldTreatAsArchive {
                    try self?.extractAppleArchive(at: archiveURL, into: destinationURL)
                    if let bundleURL = self?.findBundle(in: destinationURL) {
                        reply(bundleURL, nil)
                        return
                    }
                    let error = NSError(domain: "OuterframeProcesses",
                                        code: 3,
                                        userInfo: [NSLocalizedDescriptionKey: "No bundle found in archive"])
                    reply(nil, error)
                    return
                }

                try fm.removeItem(at: archiveURL)
                let error = NSError(domain: "OuterframeProcesses",
                                    code: 4,
                                    userInfo: [NSLocalizedDescriptionKey: "Unsupported archive format"])
                reply(nil, error)
            } catch {
                reply(nil, error as NSError)
            }
        }
        task.resume()
    }

    private func monitorProcessExit(instanceId: UUID, pid: pid_t) {
        let source = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: .main)
        source.setEventHandler { [weak self] in
            guard let self else { return }
            self.instanceLock.lock()
            self.outerframeContentInstances.removeValue(forKey: instanceId)
            if let monitor = self.processMonitors.removeValue(forKey: instanceId) {
                monitor.cancel()
            }
            self.instanceLock.unlock()
            // Reap the zombie process
            var status: Int32 = 0
            waitpid(pid, &status, WNOHANG)
        }
        source.resume()

        instanceLock.lock()
        processMonitors[instanceId] = source
        instanceLock.unlock()
    }

    public func ensureOuterframeContentExits(instanceId: UUID, timeout: TimeInterval, with reply: @escaping (NSError?) -> Void) {
        instanceLock.lock()
        guard let pid = outerframeContentInstances.removeValue(forKey: instanceId) else {
            instanceLock.unlock()
            // Instance not found - either already exited and cleaned up, or invalid ID
            // Either way, not an error from the caller's perspective
            reply(nil)
            return
        }
        // Cancel the process monitor since we're taking over
        processMonitors.removeValue(forKey: instanceId)?.cancel()
        instanceLock.unlock()

        DispatchQueue.global(qos: .utility).async {
            var status: Int32 = 0
            let deadline = Date(timeIntervalSinceNow: timeout)

            while true {
                let result = waitpid(pid, &status, WNOHANG)
                if result == pid {
                    // Process exited
                    break
                } else if result == 0 {
                    // Process still running
                    if Date() >= deadline {
                        // Timeout expired, force kill
                        kill(pid, SIGKILL)
                        waitpid(pid, &status, 0)
                        break
                    }
                    Thread.sleep(forTimeInterval: 0.1)
                } else if result == -1 && errno == EINTR {
                    continue
                } else {
                    // Error or process doesn't exist
                    break
                }
            }
            reply(nil)
        }
    }

    private func extractAppleArchive(at archiveURL: URL, into destinationDirectory: URL) throws {
        let archivePath = FilePath(archiveURL.path)
        let destinationPath = FilePath(destinationDirectory.path)

        let ignorePermissionFlags = ArchiveFlags(rawValue: 1)

        guard let byteStream = ArchiveByteStream.fileStream(path: archivePath,
                                                            mode: .readOnly,
                                                            options: [],
                                                            permissions: FilePermissions(rawValue: 0o644)) else {
            throw NSError(domain: "OuterframeProcesses",
                          code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to open archive byte stream"])
        }
        defer {
            try? byteStream.close()
        }

        guard let decompressedStream = ArchiveByteStream.decompressionStream(readingFrom: byteStream,
                                                                             flags: ignorePermissionFlags) else {
            throw NSError(domain: "OuterframeProcesses",
                          code: 11,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to open decompression stream"])
        }
        var decompressedStreamNeedsClose = true
        defer {
            if decompressedStreamNeedsClose {
                try? decompressedStream.close()
            }
        }

        guard let decodeStream = ArchiveStream.decodeStream(readingFrom: decompressedStream,
                                                            flags: ignorePermissionFlags) else {
            throw NSError(domain: "OuterframeProcesses",
                          code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to open decode stream"])
        }

        var decodeStreamNeedsClose = true
        defer {
            if decodeStreamNeedsClose {
                try? decodeStream.close()
            }
        }

        guard let extractStream = ArchiveStream.extractStream(extractingTo: destinationPath,
                                                              flags: ignorePermissionFlags) else {
            throw NSError(domain: "OuterframeProcesses",
                          code: 13,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to open extract stream"])
        }

        var extractStreamNeedsClose = true
        defer {
            if extractStreamNeedsClose {
                try? extractStream.close()
            }
        }

        _ = try ArchiveStream.process(readingFrom: decodeStream,
                                      writingTo: extractStream,
                                      flags: ignorePermissionFlags)
        try extractStream.close()
        extractStreamNeedsClose = false
        try decodeStream.close()
        decodeStreamNeedsClose = false
        try decompressedStream.close()
        decompressedStreamNeedsClose = false
    }

    private func findBundle(in directory: URL) -> URL? {
        let enumerator = FileManager.default.enumerator(at: directory,
                                                        includingPropertiesForKeys: nil,
                                                        options: [.skipsHiddenFiles])
        while let candidate = enumerator?.nextObject() as? URL {
            if candidate.pathExtension.lowercased() == "bundle" {
                return candidate
            }
        }
        return nil
    }

    private func bundleCacheBaseURL(cacheBundleIdentifier: String) -> URL {
        let homeDirectory: String
        if let pw = getpwuid(getuid()), let homeDir = pw.pointee.pw_dir {
            homeDirectory = String(cString: homeDir)
        } else {
            homeDirectory = NSHomeDirectory()
        }

        return URL(fileURLWithPath: homeDirectory, isDirectory: true)
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Containers", isDirectory: true)
            .appendingPathComponent(cacheBundleIdentifier, isDirectory: true)
            .appendingPathComponent("Data", isDirectory: true)
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent(cacheBundleIdentifier, isDirectory: true)
            .appendingPathComponent("Cache", isDirectory: true)
            .appendingPathComponent("OuterframeBundleCache", isDirectory: true)
    }

    private static func hostBundleIdentifier() throws -> String {
        var currentURL = Bundle.main.bundleURL

        while currentURL.path != "/" {
            if currentURL.pathExtension == "app" {
                guard let bundleIdentifier = Bundle(url: currentURL)?.bundleIdentifier,
                      !bundleIdentifier.isEmpty else {
                    throw OuterframeProcessesError.hostBundleIdentifierMissing
                }
                return bundleIdentifier
            }

            let parentURL = currentURL.deletingLastPathComponent()
            guard parentURL != currentURL else { break }
            currentURL = parentURL
        }

        throw OuterframeProcessesError.hostBundleIdentifierMissing
    }
}

enum OuterframeProcessesError: Error {
    case executableMissing(_ name: String)
    case hostBundleIdentifierMissing
    case posixFailure(code: Int32, context: String)

    func toNSError() -> NSError {
        switch self {
        case .executableMissing(let name):
            return NSError(domain: "OuterframeProcesses",
                           code: 1,
                           userInfo: [NSLocalizedDescriptionKey: "Executable missing: \(name)"])
        case .hostBundleIdentifierMissing:
            return NSError(domain: "OuterframeProcesses",
                           code: 2,
                           userInfo: [NSLocalizedDescriptionKey: "Host app bundle identifier is missing"])
        case .posixFailure(let code, let context):
            return NSError(domain: "OuterframeProcesses",
                           code: Int(code),
                           userInfo: [NSLocalizedDescriptionKey: "POSIX error in \(context): \(code)"])
        }
    }
}

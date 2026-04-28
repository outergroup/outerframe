import Foundation

@objc public protocol OuterframeProcessesProtocol {
    /// Launches an OuterframeContent process and returns its instance ID, sockets, PID, and stdout/stderr pipes.
    /// The instance ID should be used with ensureOuterframeContentExits to safely terminate the process.
    func launchOuterframeContent(networkProxyPort: UInt16,
                                 with reply: @escaping (FileHandle?, FileHandle?, FileHandle?, FileHandle?, pid_t, UUID?, NSError?) -> Void)

    /// Ensures an OuterframeContent process exits within the given timeout.
    /// Uses the instance ID returned by launchOuterframeContent to identify the process.
    /// Waits for the process to exit, force-killing it if the timeout expires.
    func ensureOuterframeContentExits(instanceId: UUID, timeout: TimeInterval, with reply: @escaping (NSError?) -> Void)

    /// Binaries written by App Sandboxed processes get a quarantine attribute applied to them that prevents them from being loaded. So we must download and extract the files from a non-App Sandboxed process.
    func downloadAndExtractBundleWithoutQuarantine(downloadURL: String,
                                                   proxyHost: String?,
                                                   proxyPort: UInt16,
                                                   proxyUsername: String?,
                                                   proxyPassword: String?,
                                                   destinationDirectoryURL: URL,
                                                   archiveFilename: String,
                                                   bypassCache: Bool,
                                                   with reply: @escaping (URL?, NSError?) -> Void)
}

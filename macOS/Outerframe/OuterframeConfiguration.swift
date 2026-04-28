import Foundation

/// Configuration options for the Outerframe framework.
/// Host apps should set these values before using Outerframe functionality.
public enum OuterframeConfiguration {
    /// The name of the XPC service that implements OuterframeProcessesProtocol.
    /// Defaults to "dev.outergroup.OuterframeProcesses" but host apps can override
    /// this to use their own combined XPC service.
    nonisolated(unsafe) public static var processesXPCServiceName = "dev.outergroup.OuterframeProcesses"

    /// The name of the XPC service that implements Outerframe's origin-scoped
    /// network proxy.
    nonisolated(unsafe) public static var networkProxyXPCServiceName = "dev.outergroup.OuterframeNetworkProxy"

    /// Initializes XPC connections early to avoid latency on first use.
    /// Call this during app startup after setting configuration values.
    public static func warmupXPCConnections() {
        OuterframeNetworkProxyConnection.warmup()
    }
}

import Foundation
import Security

@objc protocol OuterframeNetworkProxyProtocol {
    func ping(with reply: @escaping () -> Void)

    func start(with reply: @escaping (UInt16, NSError?) -> Void)

    func registerOrigin(id: UInt32,
                        username: String,
                        password: String,
                        scheme: String,
                        host: String,
                        port: UInt16,
                        with reply: @escaping (NSError?) -> Void)

    func registerTCPOrigin(id: UInt32,
                           username: String,
                           password: String,
                           scheme: String,
                           host: String,
                           port: UInt16,
                           targetHost: String,
                           targetPort: UInt16,
                           with reply: @escaping (NSError?) -> Void)

    func unregisterOrigin(id: UInt32, with reply: @escaping (NSError?) -> Void)

    func shutdown(with reply: @escaping (NSError?) -> Void)
}

public enum OuterframeNetworkProxyConnectionError: Error {
    case xpcFailure(String)
    case startupFailed(String)
}

@MainActor
public final class OuterframeNetworkProxyConnection: NSObject {
    public static let shared = OuterframeNetworkProxyConnection()

    private nonisolated(unsafe) static let xpcConnection: NSXPCConnection = {
        let connection = NSXPCConnection(serviceName: OuterframeConfiguration.networkProxyXPCServiceName)
        connection.remoteObjectInterface = NSXPCInterface(with: OuterframeNetworkProxyProtocol.self)
        connection.resume()
        return connection
    }()

    private var listeningPort: UInt16?
    private var nextRegistrationID = OuterframeNetworkProxyConnection.makeInitialRegistrationID()
    private var allocatedRegistrationIDs = Set<UInt32>()

    public nonisolated static func warmup() {
        if let proxy = xpcConnection.remoteObjectProxy as? OuterframeNetworkProxyProtocol {
            proxy.ping { }
        }
    }

    public func start() async throws -> UInt16 {
        let port = try await Self.startViaXPC()
        listeningPort = port
        return port
    }

    public func allocateRegistrationID() -> UInt32 {
        while true {
            if nextRegistrationID == 0 {
                nextRegistrationID = 1
            }
            let registrationID = nextRegistrationID
            nextRegistrationID &+= 1
            if allocatedRegistrationIDs.insert(registrationID).inserted {
                return registrationID
            }
        }
    }

    private static func makeInitialRegistrationID() -> UInt32 {
        var value: UInt32 = 0
        let status = withUnsafeMutableBytes(of: &value) { buffer in
            SecRandomCopyBytes(kSecRandomDefault, MemoryLayout<UInt32>.size, buffer.baseAddress!)
        }
        if status == errSecSuccess, value != 0 {
            return value
        }
        return UInt32.random(in: 1...UInt32.max)
    }

    private nonisolated static func startViaXPC() async throws -> UInt16 {
        try await withCheckedThrowingContinuation { continuation in
            guard let proxy = xpcConnection.remoteObjectProxyWithErrorHandler({ error in
                continuation.resume(throwing: OuterframeNetworkProxyConnectionError.xpcFailure(error.localizedDescription))
            }) as? OuterframeNetworkProxyProtocol else {
                continuation.resume(throwing: OuterframeNetworkProxyConnectionError.xpcFailure("Failed to get XPC proxy"))
                return
            }

            proxy.start { port, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else if port > 0 {
                    continuation.resume(returning: port)
                } else {
                    continuation.resume(throwing: OuterframeNetworkProxyConnectionError.startupFailed("Proxy returned invalid port"))
                }
            }
        }
    }

    public func registerOrigin(id: UInt32,
                               username: String,
                               password: String,
                               scheme: String,
                               host: String,
                               port: UInt16) async throws {
        guard listeningPort != nil else {
            throw OuterframeNetworkProxyConnectionError.startupFailed("Proxy not ready")
        }
        try await registerTCPOrigin(id: id,
                                    username: username,
                                    password: password,
                                    scheme: scheme,
                                    host: host,
                                    port: port,
                                    targetHost: host,
                                    targetPort: port)
    }

    public func registerTCPOrigin(id: UInt32,
                                  username: String,
                                  password: String,
                                  scheme: String,
                                  host: String,
                                  port: UInt16,
                                  targetHost: String,
                                  targetPort: UInt16) async throws {
        guard listeningPort != nil else {
            throw OuterframeNetworkProxyConnectionError.startupFailed("Proxy not ready")
        }
        try await Self.registerTCPOriginViaXPC(id: id,
                                               username: username,
                                               password: password,
                                               scheme: scheme,
                                               host: host,
                                               port: port,
                                               targetHost: targetHost,
                                               targetPort: targetPort)
    }

    private nonisolated static func registerTCPOriginViaXPC(id: UInt32,
                                                            username: String,
                                                            password: String,
                                                            scheme: String,
                                                            host: String,
                                                            port: UInt16,
                                                            targetHost: String,
                                                            targetPort: UInt16) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            guard let proxy = xpcConnection.remoteObjectProxyWithErrorHandler({ error in
                continuation.resume(throwing: OuterframeNetworkProxyConnectionError.xpcFailure(error.localizedDescription))
            }) as? OuterframeNetworkProxyProtocol else {
                continuation.resume(throwing: OuterframeNetworkProxyConnectionError.xpcFailure("Failed to get XPC proxy"))
                return
            }

            proxy.registerTCPOrigin(id: id,
                                    username: username,
                                    password: password,
                                    scheme: scheme,
                                    host: host,
                                    port: port,
                                    targetHost: targetHost,
                                    targetPort: targetPort) { error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }

    public func unregisterOrigin(id: UInt32) async {
        guard listeningPort != nil else {
            allocatedRegistrationIDs.remove(id)
            return
        }
        await Self.unregisterOriginViaXPC(id: id)
        allocatedRegistrationIDs.remove(id)
    }

    private nonisolated static func unregisterOriginViaXPC(id: UInt32) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            guard let proxy = xpcConnection.remoteObjectProxyWithErrorHandler({ _ in
                continuation.resume()
            }) as? OuterframeNetworkProxyProtocol else {
                continuation.resume()
                return
            }

            proxy.unregisterOrigin(id: id) { _ in
                continuation.resume()
            }
        }
    }

    public func shutdown() async {
        await Self.shutdownViaXPC()
        listeningPort = nil
        allocatedRegistrationIDs.removeAll(keepingCapacity: false)
        nextRegistrationID = Self.makeInitialRegistrationID()
    }

    private nonisolated static func shutdownViaXPC() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            guard let proxy = xpcConnection.remoteObjectProxyWithErrorHandler({ _ in
                continuation.resume()
            }) as? OuterframeNetworkProxyProtocol else {
                continuation.resume()
                return
            }

            proxy.shutdown { _ in
                continuation.resume()
            }
        }
    }
}

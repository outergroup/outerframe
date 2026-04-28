import Foundation

@objc protocol OpenNetworkProxyProtocol {
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

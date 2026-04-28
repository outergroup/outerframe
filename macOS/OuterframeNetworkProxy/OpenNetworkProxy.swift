import Foundation
import Network

final class OpenNetworkProxy: NSObject, OpenNetworkProxyProtocol {
    private let controller: OpenNetworkProxyController

    init(controller: OpenNetworkProxyController) {
        self.controller = controller
        super.init()
    }

    @objc func ping(with reply: @escaping () -> Void) {
        reply()
    }

    @objc func start(with reply: @escaping (UInt16, NSError?) -> Void) {
        controller.start(reply: reply)
    }

    @objc func registerOrigin(id: UInt32,
                              username: String,
                              password: String,
                              scheme: String,
                              host: String,
                              port: UInt16,
                              with reply: @escaping (NSError?) -> Void) {
        registerTCPOrigin(id: id,
                          username: username,
                          password: password,
                          scheme: scheme,
                          host: host,
                          port: port,
                          targetHost: host,
                          targetPort: port,
                          with: reply)
    }

    @objc func registerTCPOrigin(id: UInt32,
                                 username: String,
                                 password: String,
                                 scheme: String,
                                 host: String,
                                 port: UInt16,
                                 targetHost: String,
                                 targetPort: UInt16,
                                 with reply: @escaping (NSError?) -> Void) {
        controller.registerOrigin(id: id,
                                  username: username,
                                  password: password,
                                  host: host,
                                  port: port,
                                  target: OpenNetworkProxyController.Endpoint(host: targetHost, port: targetPort)) {
            reply(nil)
        }
    }

    @objc func unregisterOrigin(id: UInt32, with reply: @escaping (NSError?) -> Void) {
        controller.unregisterOrigin(id: id)
        reply(nil)
    }

    @objc func shutdown(with reply: @escaping (NSError?) -> Void) {
        controller.shutdown()
        reply(nil)
    }
}

final class OpenNetworkProxyController {
    private let listenerQueue = DispatchQueue(label: "dev.outergroup.outerframe.networkproxy.listener")
    private let stateQueue = DispatchQueue(label: "dev.outergroup.outerframe.networkproxy.state", attributes: .concurrent)

    private var listener: NWListener?
    private var listeningPort: UInt16?
    private var pendingStartReplies: [(UInt16, NSError?) -> Void] = []
    private var isStarting = false
    private var registrationById: [UInt32: OriginRegistration] = [:]
    private var routesByUsername: [String: CredentialRoutes] = [:]
    private var activeConnections: [ObjectIdentifier: ProxyPipeline] = [:]

    func start(reply: @escaping (UInt16, NSError?) -> Void) {
        stateQueue.async(flags: .barrier) { [weak self] in
            guard let self else { return }

            if let port = self.listeningPort {
                reply(port, nil)
                return
            }

            self.pendingStartReplies.append(reply)
            guard !self.isStarting else { return }

            self.isStarting = true
            self.startListener()
        }
    }

    private func startListener() {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)

        do {
            let listener = try NWListener(using: parameters)
            listener.stateUpdateHandler = { [weak self] state in
                self?.handleListenerStateChange(state)
            }
            listener.newConnectionHandler = { [weak self] connection in
                self?.handleNewConnection(connection)
            }
            listener.start(queue: listenerQueue)
            self.listener = listener
        } catch {
            failPendingStarts(message: "Failed to start listener: \(error.localizedDescription)", code: 1)
        }
    }

    private func handleListenerStateChange(_ state: NWListener.State) {
        switch state {
        case .ready:
            guard let port = listener?.port?.rawValue else { return }
            stateQueue.async(flags: .barrier) { [weak self] in
                guard let self, self.listeningPort == nil else { return }
                self.listeningPort = UInt16(port)
                self.isStarting = false
                self.completePendingStarts(port: UInt16(port), error: nil)
            }

        case .failed(let error):
            failPendingStarts(message: "Listener failed: \(error.localizedDescription)", code: 2)
            stopListener()

        case .cancelled:
            stopListener()

        default:
            break
        }
    }

    private func completePendingStarts(port: UInt16, error: NSError?) {
        for pending in pendingStartReplies {
            pending(port, error)
        }
        pendingStartReplies.removeAll()
    }

    private func failPendingStarts(message: String, code: Int) {
        let nsError = NSError(domain: "OuterframeNetworkProxy",
                              code: code,
                              userInfo: [NSLocalizedDescriptionKey: message])
        stateQueue.async(flags: .barrier) { [weak self] in
            guard let self else { return }
            self.isStarting = false
            self.completePendingStarts(port: 0, error: nsError)
        }
    }

    private func stopListener() {
        listener?.stateUpdateHandler = nil
        listener?.newConnectionHandler = nil
        listener?.cancel()
        listener = nil
        stateQueue.async(flags: .barrier) { [weak self] in
            self?.listeningPort = nil
            self?.isStarting = false
        }
    }

    private func handleNewConnection(_ connection: NWConnection) {
        let pipeline = ProxyPipeline(client: connection, controller: self)
        addConnection(pipeline)
        pipeline.start()
    }

    private func addConnection(_ pipeline: ProxyPipeline) {
        let key = ObjectIdentifier(pipeline)
        stateQueue.async(flags: .barrier) { [weak self] in
            self?.activeConnections[key] = pipeline
        }
    }

    fileprivate func removeConnection(_ pipeline: ProxyPipeline) {
        let key = ObjectIdentifier(pipeline)
        stateQueue.async(flags: .barrier) { [weak self] in
            self?.activeConnections.removeValue(forKey: key)
        }
    }

    func registerOrigin(id: UInt32,
                        username: String,
                        password: String,
                        host: String,
                        port: UInt16,
                        target: Endpoint,
                        completion: @escaping () -> Void) {
        let endpoint = Endpoint(host: host, port: port)
        let registration = OriginRegistration(username: username,
                                              password: password,
                                              endpoint: endpoint,
                                              target: target)
        stateQueue.async(flags: .barrier) { [weak self] in
            defer { completion() }
            guard let self else { return }

            if let previous = self.registrationById[id] {
                self.routesByUsername[previous.username]?.routes.removeValue(forKey: previous.endpoint)
                if self.routesByUsername[previous.username]?.routes.isEmpty == true {
                    self.routesByUsername.removeValue(forKey: previous.username)
                }
            }

            self.registrationById[id] = registration
            if self.routesByUsername[username] == nil {
                self.routesByUsername[username] = CredentialRoutes(password: password, routes: [:])
            }
            self.routesByUsername[username]?.password = password
            self.routesByUsername[username]?.routes[endpoint] = target
        }
    }

    func unregisterOrigin(id: UInt32) {
        stateQueue.async(flags: .barrier) { [weak self] in
            guard let self else { return }
            guard let registration = self.registrationById.removeValue(forKey: id) else { return }

            self.routesByUsername[registration.username]?.routes.removeValue(forKey: registration.endpoint)
            if self.routesByUsername[registration.username]?.routes.isEmpty == true {
                self.routesByUsername.removeValue(forKey: registration.username)
            }
        }
    }

    func shutdown() {
        stopListener()
        stateQueue.async(flags: .barrier) { [weak self] in
            guard let self else { return }
            for pipeline in self.activeConnections.values {
                pipeline.cancel()
            }
            self.activeConnections.removeAll()
            self.registrationById.removeAll()
            self.routesByUsername.removeAll()
            self.listeningPort = nil
        }
    }

    func route(username: String, password: String, host: String, port: UInt16) -> Endpoint? {
        let endpoint = Endpoint(host: host, port: port)
        return stateQueue.sync {
            guard let credentialRoutes = routesByUsername[username],
                  credentialRoutes.password == password else {
                return nil
            }
            return credentialRoutes.routes[endpoint]
        }
    }

    func accepts(username: String, password: String) -> Bool {
        stateQueue.sync {
            routesByUsername[username]?.password == password
        }
    }

    struct Endpoint: Hashable {
        let host: String
        let port: UInt16

        init(host: String, port: UInt16) {
            self.host = host.lowercased()
            self.port = port
        }
    }

    private struct OriginRegistration {
        let username: String
        let password: String
        let endpoint: Endpoint
        let target: Endpoint
    }

    private struct CredentialRoutes {
        var password: String
        var routes: [Endpoint: Endpoint]
    }
}

private final class ProxyPipeline {
    private let client: NWConnection
    private let controller: OpenNetworkProxyController
    private let queue = DispatchQueue(label: "dev.outergroup.outerframe.networkproxy.connection", qos: .userInitiated)

    private var server: NWConnection?
    private var pendingClientBuffer = Data()
    private var pendingServerSends: [Data] = []
    private var pendingClientSendCount = 0
    private var shouldCancelAfterClientSends = false
    private var isCancelled = false
    private var serverReady = false
    private var didSendConnectReply = false
    private var stage: Stage = .greeting

    private enum Stage {
        case greeting
        case authentication
        case request(username: String, password: String)
        case relay
    }

    private struct SocksRequest {
        let host: String
        let port: UInt16
        let byteCount: Int
    }

    init(client: NWConnection, controller: OpenNetworkProxyController) {
        self.client = client
        self.controller = controller
    }

    func start() {
        client.stateUpdateHandler = { [weak self] state in
            self?.handleClientStateChange(state)
        }
        client.start(queue: queue)
    }

    func cancel() {
        queue.async { [weak self] in
            guard let self, !self.isCancelled else { return }
            self.isCancelled = true
            self.serverReady = false
            self.client.stateUpdateHandler = nil
            self.server?.stateUpdateHandler = nil
            self.client.cancel()
            self.server?.cancel()
            self.server = nil
            self.controller.removeConnection(self)
        }
    }

    private func handleClientStateChange(_ state: NWConnection.State) {
        switch state {
        case .ready:
            receiveFromClient()
        case .failed, .cancelled:
            cancel()
        default:
            break
        }
    }

    private func receiveFromClient() {
        client.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] content, _, isComplete, error in
            guard let self else { return }

            if let data = content, !data.isEmpty {
                self.handleClientData(data)
            }

            if error != nil || isComplete {
                self.cancel()
                return
            }

            self.receiveFromClient()
        }
    }

    private func handleClientData(_ data: Data) {
        guard !isCancelled else { return }

        switch stage {
        case .relay:
            sendToServer(data)
        default:
            pendingClientBuffer.append(data)
            processBufferedClientData()
        }
    }

    private func processBufferedClientData() {
        while !isCancelled {
            switch stage {
            case .greeting:
                guard pendingClientBuffer.count >= 2 else { return }
                let methodCount = Int(pendingClientBuffer[1])
                guard pendingClientBuffer.count >= 2 + methodCount else { return }
                let version = pendingClientBuffer[0]
                let methods = pendingClientBuffer.subdata(in: 2..<(2 + methodCount))
                pendingClientBuffer.removeSubrange(0..<(2 + methodCount))

                guard version == 0x05, methods.contains(0x02) else {
                    sendToClient(Data([0x05, 0xff]), thenCancel: true)
                    return
                }

                sendToClient(Data([0x05, 0x02]))
                stage = .authentication

            case .authentication:
                guard pendingClientBuffer.count >= 2 else { return }
                let version = pendingClientBuffer[0]
                let usernameLength = Int(pendingClientBuffer[1])
                guard pendingClientBuffer.count >= 2 + usernameLength + 1 else { return }
                let passwordLengthOffset = 2 + usernameLength
                let passwordLength = Int(pendingClientBuffer[passwordLengthOffset])
                let messageLength = passwordLengthOffset + 1 + passwordLength
                guard pendingClientBuffer.count >= messageLength else { return }

                let usernameData = pendingClientBuffer.subdata(in: 2..<passwordLengthOffset)
                let passwordStart = passwordLengthOffset + 1
                let passwordData = pendingClientBuffer.subdata(in: passwordStart..<messageLength)
                pendingClientBuffer.removeSubrange(0..<messageLength)

                guard version == 0x01,
                      let username = String(data: usernameData, encoding: .utf8),
                      let password = String(data: passwordData, encoding: .utf8),
                      controller.accepts(username: username, password: password) else {
                    sendToClient(Data([0x01, 0x01]), thenCancel: true)
                    return
                }

                sendToClient(Data([0x01, 0x00]))
                stage = .request(username: username, password: password)

            case .request(let username, let password):
                guard let request = parseSocksRequest() else { return }
                pendingClientBuffer.removeSubrange(0..<request.byteCount)

                guard let endpoint = controller.route(username: username,
                                                      password: password,
                                                      host: request.host,
                                                      port: request.port) else {
                    sendSocksReply(0x02, thenCancel: true)
                    return
                }

                let remaining = pendingClientBuffer
                pendingClientBuffer.removeAll(keepingCapacity: false)
                createServerConnection(endpoint: endpoint)
                if !remaining.isEmpty {
                    sendToServer(remaining)
                }
                stage = .relay
                return

            case .relay:
                if !pendingClientBuffer.isEmpty {
                    let data = pendingClientBuffer
                    pendingClientBuffer.removeAll(keepingCapacity: false)
                    sendToServer(data)
                }
                return
            }
        }
    }

    private func createServerConnection(endpoint: OpenNetworkProxyController.Endpoint) {
        let endpointHost = NWEndpoint.Host(endpoint.host)
        guard let endpointPort = NWEndpoint.Port(rawValue: endpoint.port) else {
            sendSocksReply(0x01, thenCancel: true)
            return
        }

        let connection = NWConnection(host: endpointHost, port: endpointPort, using: .tcp)
        connection.stateUpdateHandler = { [weak self] state in
            self?.handleServerStateChange(state)
        }
        connection.start(queue: queue)
        server = connection
        serverReady = false
    }

    private func handleServerStateChange(_ state: NWConnection.State) {
        switch state {
        case .ready:
            serverReady = true
            sendSocksReply(0x00)
            flushPendingServerSends()
            receiveFromServer()

        case .failed:
            serverReady = false
            if didSendConnectReply {
                cancelAfterClientSendsComplete()
            } else {
                sendSocksReply(0x05, thenCancel: true)
            }

        case .cancelled:
            serverReady = false
            cancel()

        default:
            break
        }
    }

    private func receiveFromServer() {
        server?.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] content, _, isComplete, error in
            guard let self else { return }

            if let data = content, !data.isEmpty {
                self.sendToClient(data)
            }

            if error != nil {
                self.cancel()
                return
            }

            if isComplete {
                self.cancelAfterClientSendsComplete()
                return
            }

            self.receiveFromServer()
        }
    }

    private func sendToServer(_ data: Data) {
        guard !data.isEmpty else { return }

        if serverReady, let server {
            server.send(content: data, completion: .contentProcessed { [weak self] error in
                if error != nil {
                    self?.cancel()
                }
            })
        } else {
            pendingServerSends.append(data)
        }
    }

    private func flushPendingServerSends() {
        guard serverReady, !pendingServerSends.isEmpty else { return }

        let buffers = pendingServerSends
        pendingServerSends.removeAll(keepingCapacity: false)
        for buffer in buffers {
            sendToServer(buffer)
        }
    }

    private func sendToClient(_ data: Data, thenCancel: Bool = false) {
        pendingClientSendCount += 1
        if thenCancel {
            shouldCancelAfterClientSends = true
        }

        client.send(content: data, completion: .contentProcessed { [weak self] error in
            self?.handleClientSendCompletion(error)
        })
    }

    private func handleClientSendCompletion(_ error: NWError?) {
        if pendingClientSendCount > 0 {
            pendingClientSendCount -= 1
        }

        if error != nil {
            cancel()
            return
        }

        if shouldCancelAfterClientSends && pendingClientSendCount == 0 {
            cancel()
        }
    }

    private func cancelAfterClientSendsComplete() {
        shouldCancelAfterClientSends = true
        if pendingClientSendCount == 0 {
            cancel()
        }
    }

    private func sendSocksReply(_ code: UInt8, thenCancel: Bool = false) {
        didSendConnectReply = true
        sendToClient(Data([0x05, code, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
                     thenCancel: thenCancel)
    }

    private func parseSocksRequest() -> SocksRequest? {
        guard pendingClientBuffer.count >= 4 else { return nil }
        let version = pendingClientBuffer[0]
        let command = pendingClientBuffer[1]
        let addressType = pendingClientBuffer[3]

        guard version == 0x05 else {
            sendSocksReply(0x01, thenCancel: true)
            return nil
        }

        guard command == 0x01 else {
            sendSocksReply(0x07, thenCancel: true)
            return nil
        }

        switch addressType {
        case 0x01:
            let byteCount = 4 + 4 + 2
            guard pendingClientBuffer.count >= byteCount else { return nil }
            let bytes = pendingClientBuffer.subdata(in: 4..<8)
            let host = bytes.map(String.init).joined(separator: ".")
            let port = UInt16(pendingClientBuffer[8]) << 8 | UInt16(pendingClientBuffer[9])
            return SocksRequest(host: host, port: port, byteCount: byteCount)

        case 0x03:
            guard pendingClientBuffer.count >= 5 else { return nil }
            let length = Int(pendingClientBuffer[4])
            let byteCount = 4 + 1 + length + 2
            guard pendingClientBuffer.count >= byteCount else { return nil }
            let hostData = pendingClientBuffer.subdata(in: 5..<(5 + length))
            guard let host = String(data: hostData, encoding: .utf8) else {
                sendSocksReply(0x08, thenCancel: true)
                return nil
            }
            let portOffset = 5 + length
            let port = UInt16(pendingClientBuffer[portOffset]) << 8 | UInt16(pendingClientBuffer[portOffset + 1])
            return SocksRequest(host: host, port: port, byteCount: byteCount)

        case 0x04:
            let byteCount = 4 + 16 + 2
            guard pendingClientBuffer.count >= byteCount else { return nil }
            let bytes = pendingClientBuffer.subdata(in: 4..<20)
            let groups = stride(from: 0, to: 16, by: 2).map { index in
                String(format: "%02x%02x", bytes[bytes.startIndex + index], bytes[bytes.startIndex + index + 1])
            }
            let host = groups.joined(separator: ":")
            let port = UInt16(pendingClientBuffer[20]) << 8 | UInt16(pendingClientBuffer[21])
            return SocksRequest(host: host, port: port, byteCount: byteCount)

        default:
            sendSocksReply(0x08, thenCancel: true)
            return nil
        }
    }
}

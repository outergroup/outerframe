import Foundation
import AppKit

@MainActor
class OuterlayerHost: NSObject {
    private var pluginContext: CAContext?

    /// Infrastructure socket (handles loadPlugin, pluginLoaded, etc.) - stays with OuterlayerHost
    private var socketToBrowser: InfraSocket
    /// Plugin socket fd - stored until we know if we need to keep or release it
    private var pluginSocketFD: Int32

    init(socketActor: InfraSocket, pluginSocketFD: Int32) {
        self.socketToBrowser = socketActor
        self.pluginSocketFD = pluginSocketFD
        super.init()
    }

    private func sendInfraMessage(_ message: ContentToBrowserInfraMessage, context: String) {
        Task {
            do {
                try await socketToBrowser.send(message.encode())
            } catch {
                print("OuterframeContent: Failed to send \(context) message: \(error)")
            }
        }
    }

    func handleBrowserSocketClosed() {
        self.stop()
        exit(0)
    }

    func stop() {
        pluginContext = nil
    }

    func handleLoadPlugin(requestID: UUID, pluginUrl: String) {
        let sendResponse: (@escaping () async throws -> Void) -> Void = { operation in
            Task { @MainActor in
                do {
                    try await operation()
                } catch {
                    print("OuterframeContent: Failed to send loadPlugin response: \(error)")
                }
            }
        }

        let sendFailure: (String) -> Void = { error in
            let message = error.isEmpty ? "Unknown error" : error
            sendResponse {
                try await self.socketToBrowser.send(ContentToBrowserInfraMessage.loadPluginFailure(requestID: requestID, errorMessage: message).encode())
            }
        }

        guard let bundleURL = URL(string: pluginUrl), bundleURL.isFileURL else {
            print("Invalid or non-file plugin URL: \(pluginUrl)")
            sendFailure("Invalid plugin URL")
            return
        }

        guard let bundle = Bundle(url: bundleURL) else {
            print("Failed to create bundle for \(bundleURL)")
            sendFailure("Failed to open plugin bundle \(bundleURL.lastPathComponent)")
            return
        }

        do {
            try bundle.loadAndReturnError()
        } catch {
            let bundleName = bundleURL.lastPathComponent
            let sanitizedReason = sanitizeBundleLoadError(error.localizedDescription, bundleURL: bundleURL)
            let message: String
            if sanitizedReason.isEmpty {
                message = "Failed to load plugin bundle \(bundleName)"
            } else {
                message = "Failed to load plugin bundle \(bundleName): \(sanitizedReason)"
            }
            print("Failed to load plugin bundle from \(bundleURL): \(error)")
            sendFailure(message)
            return
        }

        guard let pluginClass = bundle.principalClass as? NSObject.Type else {
            print("Plugin bundle at \(bundleURL) did not expose a principal class")
            sendFailure("Plugin bundle \(bundleURL.lastPathComponent) does not define a principal class")
            return
        }

        guard let outerlibType = pluginClass as? OuterframeContentLibrary.Type else {
            print("Plugin class does not conform to OuterframeContentLibrary protocol")
            sendFailure("Plugin class does not conform to OuterframeContentLibrary")
            return
        }

        guard let start = outerlibType.start else {
            let error = "API failure: no 'start' method"
            print(error)
            sendFailure(error)
            return
        }

        // Start the plugin - it will read initializeContent from the browser, then call registerLayer
        print("OuterframeContent: Starting plugin with socket fd \(self.pluginSocketFD)")
        let result = start(self.pluginSocketFD, self)

        if result != 0 {
            print("OuterframeContent: Plugin start failed with code \(result)")
            sendFailure("Plugin start failed with code \(result)")
            return
        }

        print("OuterframeContent: Plugin started successfully")
        sendResponse {
            try await self.socketToBrowser.send(ContentToBrowserInfraMessage.loadPluginSuccess(requestID: requestID).encode())
        }
    }

    private func sanitizeBundleLoadError(_ message: String, bundleURL: URL) -> String {
        let bundleName = bundleURL.lastPathComponent
        var sanitized = message.replacingOccurrences(of: bundleURL.path, with: bundleName)
        sanitized = sanitized.replacingOccurrences(of: bundleURL.absoluteString, with: bundleName)
        return sanitized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Message Handlers

    func handleUnloadPlugin() {
        pluginContext = nil

        sendInfraMessage(.pluginUnloaded, context: "pluginUnloaded")
    }

    func handleShutdown() {
        print("OuterframeContent: Received shutdown message, exiting...")
        pluginContext = nil
        exit(0)
    }

    func infraSocket(_ socket: InfraSocket, didReceiveMessage messageData: Data) {
        let message: BrowserToContentInfraMessage
        do {
            message = try BrowserToContentInfraMessage.decode(message: messageData)
        } catch {
            print("OuterframeContent: Failed to decode infrastructure message: \(error)")
            return
        }

        switch message {
        case .loadPlugin(let requestID, let pluginURL):
            handleLoadPlugin(requestID: requestID, pluginUrl: pluginURL)
        case .unloadPlugin:
            handleUnloadPlugin()
        case .setDebuggerAttachmentMonitoring(let isEnabled):
            if isEnabled {
                startDebuggerAttachmentMonitor()
            } else {
                stopDebuggerAttachmentMonitor()
            }
        case .shutdown:
            handleShutdown()
        }
    }

    func infraSocketDidClose(_ socket: InfraSocket) {
        print("OuterframeContent: Browser socket closed")
        self.handleBrowserSocketClosed()
    }
}

extension OuterlayerHost : OuterframeAppConnection {

    /// Called by plugin to announce its CALayer. Creates CAContext and sends pluginLoaded message
    /// on the infrastructure socket.
    func registerLayer(_ layer: CALayer) {
        let conn = CGSMainConnectionID()
        guard let context = CAContext(cgsConnection: conn, options: [:]) else {
            print("OuterframeContent: Failed to create CAContext in registerLayer")
            sendInfraMessage(.pluginLoaded(contextID: 0, success: false), context: "pluginLoaded (failure)")
            return
        }
        context.layer = layer
        let contextID = context.contextId

        self.pluginContext = context

        print("OuterframeContent: Layer registered, contextID: \(contextID)")
        sendInfraMessage(.pluginLoaded(contextID: contextID, success: true), context: "pluginLoaded")
    }
}

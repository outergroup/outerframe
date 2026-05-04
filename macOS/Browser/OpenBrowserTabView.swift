import AppKit
import Outerframe
import WebKit

private final class OpenBrowserPlaceholderView: NSView {
    override var acceptsFirstResponder: Bool {
        true
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }
}

final class OpenBrowserTabView: NSView {
    weak var delegate: OpenBrowserTabDelegate?

    private var webView: WKWebView?
    private var outerframeView: OuterframeView?
    private var placeholderView: NSView?
    private var loadGeneration = 0
    private var history: [URL] = []
    private var historyIndex = -1
    private var title: String?
    private var urlObservation: NSKeyValueObservation?
    private var titleObservation: NSKeyValueObservation?
    private var progressObservation: NSKeyValueObservation?
    private var canGoBackObservation: NSKeyValueObservation?
    private var canGoForwardObservation: NSKeyValueObservation?
    private var allowedWebNavigationURL: URL?
    private var activeOuterframeProxyRegistrationIDs: [UInt32] = []

    var currentURL: URL? {
        guard history.indices.contains(historyIndex) else { return webView?.url }
        return history[historyIndex]
    }

    var canGoBack: Bool {
        historyIndex > 0 || webView?.canGoBack == true
    }

    var canGoForward: Bool {
        (historyIndex >= 0 && historyIndex < history.count - 1) || webView?.canGoForward == true
    }

    var displayTitle: String {
        if let title, !title.isEmpty {
            return title
        }
        return currentURL?.host ?? "Outer Frame"
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        autoresizingMask = [.width, .height]
        showEmptyPage()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        autoresizingMask = [.width, .height]
        showEmptyPage()
    }

    deinit {
        unregisterActiveOuterframeProxyOrigin()
        if let outerframeView {
            Task { @MainActor in
                await outerframeView.shutdownOuterframeContent()
            }
        }
    }

    func load(_ url: URL) {
        load(url, addingToHistory: true)
    }

    func goBack() {
        if historyIndex > 0 {
            historyIndex -= 1
            load(history[historyIndex], addingToHistory: false)
            return
        }

        if webView?.canGoBack == true {
            webView?.goBack()
        }
    }

    func goForward() {
        if historyIndex >= 0, historyIndex < history.count - 1 {
            historyIndex += 1
            load(history[historyIndex], addingToHistory: false)
            return
        }

        if webView?.canGoForward == true {
            webView?.goForward()
        }
    }

    func reload() {
        if let outerframeView, outerframeView.superview === self, let url = currentURL {
            load(url, addingToHistory: false)
            return
        }

        if let url = currentURL {
            load(url, addingToHistory: false)
        }
    }

    private func load(_ url: URL, addingToHistory: Bool) {
        if addingToHistory {
            if historyIndex < history.count - 1 {
                history.removeSubrange((historyIndex + 1)..<history.count)
            }
            history.append(url)
            historyIndex = history.count - 1
        }

        title = nil
        loadGeneration += 1
        let generation = loadGeneration
        unregisterActiveOuterframeProxyOrigin()
        delegate?.browserTab(self, didUpdateURL: url)
        delegate?.browserTab(self, didUpdateTitle: nil)
        delegate?.browserTabDidUpdateNavigationState(self)
        delegate?.browserTab(self, didUpdateProgress: 0)

        if url.scheme?.lowercased() == "about" {
            delegate?.browserTab(self, didUpdateProgress: nil)
            showEmptyPage()
            return
        }

        showLoadingPage(for: url)

        Task { @MainActor [weak self] in
            guard let self else { return }

            let shouldTryOuterframe = await self.shouldTryOuterframe(for: url)
            guard self.loadGeneration == generation else { return }

            if shouldTryOuterframe {
                await self.loadOuterframe(url, generation: generation)
            } else {
                self.loadWeb(url)
            }
        }
    }

    private func shouldTryOuterframe(for url: URL) async -> Bool {
        if url.isFileURL {
            return url.pathExtension.lowercased() == "outerframe"
        }

        guard ["http", "https"].contains(url.scheme?.lowercased() ?? "") else {
            return false
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 4
        configuration.timeoutIntervalForResource = 4
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }

        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue(OuterframeView.topLevelNavigationCapabilityHeaderValue,
                         forHTTPHeaderField: OuterframeView.topLevelNavigationCapabilityHeaderField)

        do {
            let (_, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode),
                  let mimeType = httpResponse.mimeType?.lowercased() else {
                return false
            }
            return mimeType == OuterframeView.outerframeMimeType
        } catch {
            return false
        }
    }

    private func loadOuterframe(_ url: URL, generation: Int) async {
        let candidate = OuterframeView(frame: bounds)
        candidate.autoresizingMask = [.width, .height]
        candidate.delegate = self

        var pendingRegistrationIDs: [UInt32] = []
        do {
            let proxyRegistration = try await registerOuterframeProxyOrigin(for: url)
            pendingRegistrationIDs = proxyRegistration.registrationIDs
            let result = try await candidate.loadOuterframeContent(from: url,
                                                                   networkProxyEndpoint: proxyRegistration.endpoint,
                                                                   cacheContext: nil,
                                                                   bypassCache: false)
            guard loadGeneration == generation else {
                await unregisterOuterframeProxyOrigins(ids: proxyRegistration.registrationIDs)
                await candidate.shutdownOuterframeContent()
                return
            }

            switch result {
            case .loaded:
                activeOuterframeProxyRegistrationIDs = proxyRegistration.registrationIDs
                pendingRegistrationIDs = []
                showOuterframe(candidate)
                delegate?.browserTab(self, didUpdateProgress: nil)
            case .notOuterframe:
                await unregisterOuterframeProxyOrigins(ids: proxyRegistration.registrationIDs)
                await candidate.shutdownOuterframeContent()
                loadWeb(url)
            @unknown default:
                await unregisterOuterframeProxyOrigins(ids: proxyRegistration.registrationIDs)
                await candidate.shutdownOuterframeContent()
                loadWeb(url)
            }
        } catch {
            await unregisterOuterframeProxyOrigins(ids: pendingRegistrationIDs)
            guard loadGeneration == generation else {
                await candidate.shutdownOuterframeContent()
                return
            }
            await candidate.shutdownOuterframeContent()
            delegate?.browserTab(self, didUpdateProgress: nil)
            showErrorPage(url: url, error: error)
        }
    }

    private struct OuterframeProxyRegistration {
        let endpoint: OuterframeNetworkProxyEndpoint
        let registrationIDs: [UInt32]
    }

    private func registerOuterframeProxyOrigin(for url: URL) async throws -> OuterframeProxyRegistration {
        let proxy = OuterframeNetworkProxyConnection.shared
        let proxyPort = try await proxy.start()
        let username = UUID().uuidString
        let password = UUID().uuidString
        var registrationIDs: [UInt32] = []
        var registeredOrigins = Set<OriginKey>()

        do {
            let target = loopbackTarget(for: url)
            var routeURLs = [url]
            if target != nil, let syntheticURL = makeSyntheticLoopbackURL(from: url) {
                routeURLs.append(syntheticURL)
            }

            for routeURL in routeURLs {
                guard let origin = originParts(from: routeURL) else {
                    throw OpenBrowserProxyError.invalidOrigin
                }

                let originKey = OriginKey(scheme: origin.scheme, host: origin.host, port: origin.port)
                guard registeredOrigins.insert(originKey).inserted else { continue }

                let registrationID = proxy.allocateRegistrationID()
                registrationIDs.append(registrationID)

                if let target {
                    try await proxy.registerTCPOrigin(id: registrationID,
                                                      username: username,
                                                      password: password,
                                                      scheme: origin.scheme,
                                                      host: origin.host,
                                                      port: origin.port,
                                                      targetHost: target.host,
                                                      targetPort: target.port)
                } else {
                    try await proxy.registerOrigin(id: registrationID,
                                                   username: username,
                                                   password: password,
                                                   scheme: origin.scheme,
                                                   host: origin.host,
                                                   port: origin.port)
                }
            }
        } catch {
            await unregisterOuterframeProxyOrigins(ids: registrationIDs)
            throw error
        }

        let endpoint = OuterframeNetworkProxyEndpoint(host: "127.0.0.1",
                                                      port: proxyPort,
                                                      username: username,
                                                      password: password)
        return OuterframeProxyRegistration(endpoint: endpoint, registrationIDs: registrationIDs)
    }

    private func unregisterActiveOuterframeProxyOrigin() {
        guard !activeOuterframeProxyRegistrationIDs.isEmpty else { return }
        let registrationIDs = activeOuterframeProxyRegistrationIDs
        activeOuterframeProxyRegistrationIDs = []
        Task { @MainActor in
            await unregisterOuterframeProxyOrigins(ids: registrationIDs)
        }
    }

    private func unregisterOuterframeProxyOrigins(ids: [UInt32]) async {
        for id in ids {
            await OuterframeNetworkProxyConnection.shared.unregisterOrigin(id: id)
        }
    }

    private func makeSyntheticLoopbackURL(from url: URL) -> URL? {
        guard let host = url.host, isLoopbackHost(host) else { return nil }
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
        components.host = "outerframe.local"
        return components.url
    }

    private func loopbackTarget(for url: URL) -> (host: String, port: UInt16)? {
        guard let host = url.host,
              isLoopbackHost(host),
              let port = explicitOrDefaultPort(for: url) else {
            return nil
        }
        return (host, port)
    }

    private func originParts(from url: URL) -> (scheme: String, host: String, port: UInt16)? {
        guard let scheme = url.scheme?.lowercased(),
              let host = url.host,
              let port = explicitOrDefaultPort(for: url) else {
            return nil
        }
        return (scheme, host, port)
    }

    private func explicitOrDefaultPort(for url: URL) -> UInt16? {
        if let port = url.port {
            return UInt16(exactly: port)
        }
        switch url.scheme?.lowercased() {
        case "http":
            return 80
        case "https":
            return 443
        default:
            return nil
        }
    }

    private func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized == "localhost"
            || normalized == "127.0.0.1"
            || normalized == "::1"
            || normalized == "0:0:0:0:0:0:0:1"
            || normalized == "0000:0000:0000:0000:0000:0000:0000:0001"
    }

    private func loadWeb(_ url: URL) {
        let webView = self.webView ?? makeWebView()
        self.webView = webView
        allowedWebNavigationURL = url
        showContentView(webView)
        webView.load(topLevelNavigationRequest(for: url))
    }

    private func topLevelNavigationRequest(for url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        if shouldSendOuterframeCapabilityHeader(for: url) {
            request.setValue(OuterframeView.topLevelNavigationCapabilityHeaderValue,
                             forHTTPHeaderField: OuterframeView.topLevelNavigationCapabilityHeaderField)
        }
        return request
    }

    private func shouldSendOuterframeCapabilityHeader(for url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else {
            return false
        }

        return scheme == "http" || scheme == "https"
    }

    private func makeWebView() -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")

        let webView = WKWebView(frame: bounds, configuration: configuration)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true

        titleObservation = webView.observe(\.title, options: [.new]) { [weak self] webView, _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.title = webView.title
                self.delegate?.browserTab(self, didUpdateTitle: self.title)
            }
        }

        urlObservation = webView.observe(\.url, options: [.new]) { [weak self] webView, _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.delegate?.browserTab(self, didUpdateURL: webView.url)
            }
        }

        progressObservation = webView.observe(\.estimatedProgress, options: [.new]) { [weak self] webView, _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.delegate?.browserTab(self, didUpdateProgress: webView.isLoading ? webView.estimatedProgress : nil)
            }
        }

        canGoBackObservation = webView.observe(\.canGoBack, options: [.new]) { [weak self] _, _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.delegate?.browserTabDidUpdateNavigationState(self)
            }
        }

        canGoForwardObservation = webView.observe(\.canGoForward, options: [.new]) { [weak self] _, _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.delegate?.browserTabDidUpdateNavigationState(self)
            }
        }

        return webView
    }

    private func showOuterframe(_ view: OuterframeView) {
        outerframeView = view
        showContentView(view)
    }

    private func showContentView(_ view: NSView) {
        placeholderView?.removeFromSuperview()
        placeholderView = nil

        if outerframeView.map({ $0 === view }) != true {
            let previousOuterframe = outerframeView
            outerframeView = nil
            if let previousOuterframe {
                Task { @MainActor in
                    await previousOuterframe.shutdownOuterframeContent()
                }
            }
        }

        for subview in subviews where subview !== view {
            subview.removeFromSuperview()
        }

        if view.superview !== self {
            addSubview(view)
        }
        view.frame = bounds
        view.autoresizingMask = [.width, .height]
    }

    private func showEmptyPage() {
        delegate?.browserTab(self, didUpdateProgress: nil)
        showPlaceholder(title: "Outer Frame", message: "")
    }

    private func showLoadingPage(for url: URL) {
        showPlaceholder(title: "Loading", message: url.absoluteString)
    }

    private func showErrorPage(url: URL, error: Error) {
        showPlaceholder(title: "Unable to Load Page", message: "\(url.absoluteString)\n\(error.localizedDescription)")
    }

    private func showPlaceholder(title: String, message: String) {
        let container = OpenBrowserPlaceholderView(frame: bounds)
        container.autoresizingMask = [.width, .height]

        let titleLabel = NSTextField(labelWithString: title)
        titleLabel.font = NSFont.systemFont(ofSize: 22, weight: .semibold)
        titleLabel.alignment = .center
        titleLabel.translatesAutoresizingMaskIntoConstraints = false

        let messageLabel = NSTextField(labelWithString: message)
        messageLabel.font = NSFont.systemFont(ofSize: 13)
        messageLabel.textColor = .secondaryLabelColor
        messageLabel.alignment = .center
        messageLabel.maximumNumberOfLines = 0
        messageLabel.lineBreakMode = .byCharWrapping
        messageLabel.translatesAutoresizingMaskIntoConstraints = false

        let stack = NSStackView(views: [titleLabel, messageLabel])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false

        container.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            stack.leadingAnchor.constraint(greaterThanOrEqualTo: container.leadingAnchor, constant: 32),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -32),
            messageLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 720)
        ])

        showContentView(container)
        placeholderView = container
    }
}

private struct OriginKey: Hashable {
    let scheme: String
    let host: String
    let port: UInt16
}

private enum OpenBrowserProxyError: Error, LocalizedError {
    case invalidOrigin

    var errorDescription: String? {
        switch self {
        case .invalidOrigin:
            return "Unable to register a network proxy origin for this Outerframe URL."
        }
    }
}

extension OpenBrowserTabView: WKNavigationDelegate {
    func webView(_ webView: WKWebView,
                 decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard navigationAction.targetFrame?.isMainFrame != false,
              let url = navigationAction.request.url else {
            decisionHandler(.allow)
            return
        }

        if allowedWebNavigationURL == url {
            allowedWebNavigationURL = nil
            decisionHandler(.allow)
            return
        }

        guard navigationAction.navigationType == .linkActivated || navigationAction.navigationType == .formSubmitted,
              navigationAction.request.httpMethod?.uppercased() == "GET" else {
            decisionHandler(.allow)
            return
        }

        decisionHandler(.cancel)
        load(url)
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        delegate?.browserTab(self, didUpdateProgress: webView.estimatedProgress)
    }

    func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        delegate?.browserTab(self, didUpdateURL: webView.url)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        title = webView.title
        delegate?.browserTab(self, didUpdateTitle: title)
        delegate?.browserTab(self, didUpdateProgress: nil)
        delegate?.browserTabDidUpdateNavigationState(self)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        handleWebNavigationFailure(error, url: webView.url)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        handleWebNavigationFailure(error, url: webView.url)
    }

    private func handleWebNavigationFailure(_ error: Error, url: URL?) {
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
            return
        }
        delegate?.browserTab(self, didUpdateProgress: nil)
        showErrorPage(url: url ?? currentURL ?? URL(fileURLWithPath: "/"), error: error)
    }
}

extension OpenBrowserTabView: WKUIDelegate {
    func webView(_ webView: WKWebView,
                 createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url {
            delegate?.browserTab(self, didRequestOpenURL: url)
        }
        return nil
    }
}

extension OpenBrowserTabView: OuterframeViewDelegate {
    func outerframeView(_ view: OuterframeView,
                        didRequestOpenWindowWithURLString urlString: String,
                        displayString: String?,
                        preferredSize: CGSize?) {
        guard let url = URL(string: urlString) else { return }
        delegate?.browserTab(self, didRequestOpenURL: url)
    }

    func outerframeViewDidResetOuterframeContentOutput(_ view: OuterframeView) {
    }

    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStdout text: String) {
    }

    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStderr text: String) {
    }

    func outerframeViewDidTimeoutWaitingForPluginLoaded(_ view: OuterframeView) {
        delegate?.browserTab(self, didUpdateProgress: nil)
        showErrorPage(url: currentURL ?? URL(fileURLWithPath: "/"),
                      error: NSError(domain: "OuterFrame", code: 1, userInfo: [
                        NSLocalizedDescriptionKey: "Outerframe content did not finish loading."
                      ]))
    }
}

import AppKit

private extension NSToolbarItem.Identifier {
    static let openBrowserNewTab = NSToolbarItem.Identifier("OpenBrowserToolbar.newTab")
    static let openBrowserNavigation = NSToolbarItem.Identifier("OpenBrowserToolbar.navigation")
    static let openBrowserAddressBar = NSToolbarItem.Identifier("OpenBrowserToolbar.addressBar")
}

private let openBrowserAddressBarMinimumWidth: CGFloat = 220
private let openBrowserToolbarReservedWidth: CGFloat = 330
private let openBrowserDefaultWindowContentSize = NSSize(width: 1280, height: 800)

@MainActor
protocol OpenBrowserTabDelegate: AnyObject {
    func browserTab(_ tab: OpenBrowserTabView, didUpdateURL url: URL?)
    func browserTab(_ tab: OpenBrowserTabView, didUpdateTitle title: String?)
    func browserTab(_ tab: OpenBrowserTabView, didUpdateProgress progress: Double?)
    func browserTabDidUpdateNavigationState(_ tab: OpenBrowserTabView)
    func browserTab(_ tab: OpenBrowserTabView, didRequestOpenURL url: URL)
}

final class OpenBrowserWindowController: NSWindowController, NSToolbarDelegate, NSToolbarItemValidation, NSUserInterfaceValidations {
    private weak var newTabButton: OpenBrowserToolbarButton?
    private weak var newTabToolbarView: NSView?
    private weak var backButton: OpenBrowserToolbarButton?
    private weak var forwardButton: OpenBrowserToolbarButton?
    private weak var navigationToolbarView: NSView?
    private weak var reloadButton: OpenBrowserToolbarButton?
    private weak var addressBarView: OpenBrowserAddressBarView?
    private weak var addressBarBackgroundView: OpenBrowserToolbarBackgroundView?
    private weak var addressBarContainer: NSView?
    private weak var addressBarToolbarItem: NSToolbarItem?
    private var addressBarWidthConstraint: NSLayoutConstraint?
    private weak var resizeObservedWindow: NSWindow?
    private let tabView = OpenBrowserTabView()

    convenience init() {
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: openBrowserDefaultWindowContentSize),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered,
                              defer: false)
        window.center()
        self.init(window: window)
    }

    override init(window: NSWindow?) {
        super.init(window: window)
        configureWindow()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configureWindow()
    }

    override func windowDidLoad() {
        super.windowDidLoad()
        configureWindow()
    }

    private func configureWindow() {
        guard let window else { return }

        window.title = "Outer Frame"
        window.minSize = NSSize(width: 760, height: 460)
        applyDefaultWindowContentSizeIfNeeded(to: window)
        window.tabbingMode = .preferred
        window.tabbingIdentifier = NSWindow.TabbingIdentifier("OuterFrameBrowser")
        window.titleVisibility = .hidden
        window.toolbarStyle = .unified
        window.titlebarAppearsTransparent = true
        window.styleMask.insert(.unifiedTitleAndToolbar)
        window.styleMask.insert(.fullSizeContentView)

        let toolbar = NSToolbar(identifier: "OpenBrowserToolbar")
        toolbar.delegate = self
        toolbar.displayMode = .iconOnly
        window.toolbar = toolbar
        if resizeObservedWindow !== window {
            if let resizeObservedWindow {
                NotificationCenter.default.removeObserver(self,
                                                          name: NSWindow.didResizeNotification,
                                                          object: resizeObservedWindow)
            }
            NotificationCenter.default.addObserver(self,
                                                   selector: #selector(windowDidResize(_:)),
                                                   name: NSWindow.didResizeNotification,
                                                   object: window)
            resizeObservedWindow = window
        }

        let contentView = NSView()
        tabView.delegate = self
        tabView.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(tabView)
        window.contentView = contentView
        NSLayoutConstraint.activate([
            tabView.topAnchor.constraint(equalTo: contentView.safeAreaLayoutGuide.topAnchor),
            tabView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            tabView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            tabView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor)
        ])

        syncToolbarItems()
        clearAddressBar()
        updateNavigationButtons()
    }

    private func applyDefaultWindowContentSizeIfNeeded(to window: NSWindow) {
        let currentSize = window.contentLayoutRect.size
        guard currentSize.width < openBrowserDefaultWindowContentSize.width
                || currentSize.height < openBrowserDefaultWindowContentSize.height else {
            return
        }

        window.setContentSize(openBrowserDefaultWindowContentSize)
        window.center()
    }

    func load(_ url: URL) {
        tabView.load(url)
    }

    func focusAddressBar(_ sender: Any?) {
        addressBarView?.startEditing()
    }

    func createNativeTab(loading url: URL? = nil, focusAddressBar: Bool = false) {
        let controller = OpenBrowserWindowController()
        if let appDelegate = NSApp.delegate as? AppDelegate {
            appDelegate.registerWindowController(controller)
        }

        guard let currentWindow = window,
              let newWindow = controller.window else {
            controller.showWindow(nil)
            if let url {
                controller.load(url)
            } else if focusAddressBar {
                controller.focusAddressBar(nil)
            }
            return
        }

        currentWindow.addTabbedWindow(newWindow, ordered: .above)
        DispatchQueue.main.async { [weak controller, weak newWindow] in
            guard let controller, let newWindow else { return }
            newWindow.tabGroup?.selectedWindow = newWindow
            newWindow.makeKeyAndOrderFront(nil)
            if let url {
                controller.load(url)
            } else if focusAddressBar {
                controller.focusAddressBar(nil)
            }
        }
    }

    private func syncToolbarItems() {
        guard let toolbar = window?.toolbar else { return }
        let desired: [NSToolbarItem.Identifier] = [
            .openBrowserNavigation,
            .space,
            .openBrowserAddressBar,
            .space,
            .openBrowserNewTab
        ]

        if toolbar.items.map(\.itemIdentifier) == desired {
            updateAddressBarSizing()
            return
        }

        for index in stride(from: toolbar.items.count - 1, through: 0, by: -1) {
            toolbar.removeItem(at: index)
        }

        for (index, identifier) in desired.enumerated() {
            toolbar.insertItem(withItemIdentifier: identifier, at: index)
        }

        updateAddressBarSizing()
    }

    private func ensureNewTabToolbarView() -> NSView {
        if let newTabToolbarView {
            return newTabToolbarView
        }

        let config = NSImage.SymbolConfiguration(pointSize: 13, weight: .semibold)
        let button = OpenBrowserToolbarButton(image: symbolImage("plus", configuration: config),
                                              target: self,
                                              action: #selector(newTab(_:)))
        button.translatesAutoresizingMaskIntoConstraints = false
        button.toolTip = "New Tab"
        button.contentTintColor = .labelColor
        self.newTabButton = button

        let container = OpenBrowserToolbarBackgroundView()
        container.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(button)
        NSLayoutConstraint.activate([
            container.heightAnchor.constraint(equalToConstant: 36),
            button.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            button.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            button.widthAnchor.constraint(equalToConstant: 30),
            button.heightAnchor.constraint(equalToConstant: 30),
            container.widthAnchor.constraint(equalToConstant: 36)
        ])
        newTabToolbarView = container
        return container
    }

    private func ensureNavigationToolbarView() -> NSView {
        if let navigationToolbarView {
            return navigationToolbarView
        }

        let config = NSImage.SymbolConfiguration(pointSize: 13, weight: .semibold)
        let back = OpenBrowserToolbarButton(image: symbolImage("chevron.backward", configuration: config),
                                            target: self,
                                            action: #selector(goBack(_:)))
        back.translatesAutoresizingMaskIntoConstraints = false
        back.contentTintColor = .labelColor
        back.toolTip = "Go Back"
        back.isEnabled = false
        self.backButton = back

        let forward = OpenBrowserToolbarButton(image: symbolImage("chevron.forward", configuration: config),
                                               target: self,
                                               action: #selector(goForward(_:)))
        forward.translatesAutoresizingMaskIntoConstraints = false
        forward.contentTintColor = .labelColor
        forward.toolTip = "Go Forward"
        forward.isEnabled = false
        self.forwardButton = forward

        let separator = OpenBrowserSeparatorView()
        separator.translatesAutoresizingMaskIntoConstraints = false

        let container = OpenBrowserToolbarBackgroundView()
        container.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(back)
        container.addSubview(separator)
        container.addSubview(forward)

        NSLayoutConstraint.activate([
            container.heightAnchor.constraint(equalToConstant: 36),
            back.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 4),
            back.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            back.widthAnchor.constraint(equalToConstant: 30),
            back.heightAnchor.constraint(equalToConstant: 30),
            separator.leadingAnchor.constraint(equalTo: back.trailingAnchor, constant: 2),
            separator.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            separator.widthAnchor.constraint(equalToConstant: 1),
            separator.heightAnchor.constraint(equalToConstant: 18),
            forward.leadingAnchor.constraint(equalTo: separator.trailingAnchor, constant: 2),
            forward.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -4),
            forward.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            forward.widthAnchor.constraint(equalToConstant: 30),
            forward.heightAnchor.constraint(equalToConstant: 30)
        ])

        navigationToolbarView = container
        return container
    }

    private func ensureAddressBarToolbarView() -> NSView {
        if let addressBarContainer {
            return addressBarContainer
        }

        let container = OpenBrowserToolbarSizedView()
        container.translatesAutoresizingMaskIntoConstraints = false
        container.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        container.setContentHuggingPriority(.defaultLow, for: .horizontal)

        let widthConstraint = container.widthAnchor.constraint(equalToConstant: 520)
        widthConstraint.isActive = true
        container.heightAnchor.constraint(equalToConstant: 36).isActive = true
        addressBarWidthConstraint = widthConstraint

        let background = OpenBrowserToolbarBackgroundView()
        background.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(background)
        addressBarBackgroundView = background

        let addressBar = OpenBrowserAddressBarView(frame: .zero)
        addressBar.translatesAutoresizingMaskIntoConstraints = false
        addressBar.delegate = self
        addressBar.setPlaceholderText("Enter URL")
        container.addSubview(addressBar)
        addressBarView = addressBar

        let config = NSImage.SymbolConfiguration(pointSize: 12, weight: .semibold)
        let reload = OpenBrowserToolbarButton(image: symbolImage("arrow.clockwise", configuration: config),
                                              target: self,
                                              action: #selector(reload(_:)))
        reload.translatesAutoresizingMaskIntoConstraints = false
        reload.contentTintColor = .secondaryLabelColor
        reload.toolTip = "Reload"
        reload.isHidden = true
        container.addSubview(reload)
        reloadButton = reload

        NSLayoutConstraint.activate([
            background.topAnchor.constraint(equalTo: container.topAnchor),
            background.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            background.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            background.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            addressBar.topAnchor.constraint(equalTo: container.topAnchor),
            addressBar.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            addressBar.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 10),
            addressBar.trailingAnchor.constraint(equalTo: reload.leadingAnchor, constant: -6),
            reload.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -4),
            reload.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            reload.widthAnchor.constraint(equalToConstant: 28),
            reload.heightAnchor.constraint(equalToConstant: 28)
        ])

        addressBarContainer = container
        return container
    }

    private func updateAddressBarSizing() {
        guard let container = addressBarContainer,
              let widthConstraint = addressBarWidthConstraint else { return }
        let contentWidth = window?.contentView?.bounds.width ?? 1120
        let targetWidth = max(contentWidth - openBrowserToolbarReservedWidth - 12, openBrowserAddressBarMinimumWidth)
        widthConstraint.constant = targetWidth
        container.layoutSubtreeIfNeeded()
        let size = NSSize(width: ceil(targetWidth), height: 36)
        container.frame = NSRect(origin: .zero, size: size)
        if let sizedContainer = container as? OpenBrowserToolbarSizedView {
            sizedContainer.preferredSize = size
        }
        addressBarToolbarItem?.view?.frame = NSRect(origin: .zero, size: size)
    }

    private func updateNavigationButtons() {
        backButton?.isEnabled = tabView.canGoBack
        forwardButton?.isEnabled = tabView.canGoForward
    }

    private func clearAddressBar() {
        addressBarView?.setPlaceholderText("Enter URL")
        addressBarView?.setLoadProgress(nil, animated: false)
        reloadButton?.isHidden = true
    }

    private func updateAddressBar(with url: URL?) {
        guard let url else {
            clearAddressBar()
            return
        }
        addressBarView?.update(displayText: formattedDisplayString(for: url),
                               editingText: editingString(for: url))
        reloadButton?.isHidden = false
    }

    private func formattedDisplayString(for url: URL) -> String {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let host = components.host,
              !host.isEmpty else {
            return url.absoluteString
        }

        var display = host
        if let port = components.port {
            display += ":\(port)"
        }
        if !components.percentEncodedPath.isEmpty, components.percentEncodedPath != "/" {
            display += components.percentEncodedPath
        }
        if let query = components.percentEncodedQuery, !query.isEmpty {
            display += "?\(query)"
        }
        if let fragment = components.percentEncodedFragment, !fragment.isEmpty {
            display += "#\(fragment)"
        }
        return display
    }

    private func editingString(for url: URL) -> String {
        guard url.scheme?.lowercased() == "http" else {
            return url.absoluteString
        }
        return url.absoluteString.replacingOccurrences(of: "http://", with: "", options: [.anchored, .caseInsensitive])
    }

    private func normalizedURL(from value: String) -> URL? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let explicitURL = explicitURL(from: trimmed) {
            return explicitURL
        }

        let defaultScheme = defaultScheme(forAddressWithoutScheme: trimmed)
        return URL(string: "\(defaultScheme)://\(trimmed)")
    }

    private func explicitURL(from value: String) -> URL? {
        if value.range(of: "://") != nil,
           let url = URL(string: value),
           url.scheme != nil {
            return url
        }

        guard let separatorIndex = value.firstIndex(of: ":") else { return nil }

        let scheme = String(value[..<separatorIndex]).lowercased()
        let remainder = value[value.index(after: separatorIndex)...]
        if ["about", "data", "blob", "file"].contains(scheme),
           let url = URL(string: value) {
            return url
        }

        if ["http", "https"].contains(scheme),
           !remainder.hasPrefix("//"),
           let url = URL(string: "\(scheme)://\(remainder)") {
            return url
        }

        return nil
    }

    private func defaultScheme(forAddressWithoutScheme value: String) -> String {
        let lowercased = value.lowercased()
        if lowercased == "localhost"
            || lowercased.hasPrefix("localhost:")
            || lowercased.hasPrefix("127.")
            || lowercased.hasPrefix("[::1]") {
            return "http"
        }

        return "https"
    }

    private func symbolImage(_ name: String, configuration: NSImage.SymbolConfiguration) -> NSImage {
        if let image = NSImage(systemSymbolName: name, accessibilityDescription: nil)?.withSymbolConfiguration(configuration) {
            return image
        }
        return NSImage(size: NSSize(width: 16, height: 16))
    }

    func toolbar(_ toolbar: NSToolbar,
                 itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier,
                 willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? {
        switch itemIdentifier {
        case .openBrowserNewTab:
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "New Tab"
            item.paletteLabel = "New Tab"
            item.view = ensureNewTabToolbarView()
            return item
        case .openBrowserNavigation:
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "Navigation"
            item.paletteLabel = "Navigation"
            item.view = ensureNavigationToolbarView()
            return item
        case .openBrowserAddressBar:
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "Address"
            item.paletteLabel = "Address"
            item.view = ensureAddressBarToolbarView()
            addressBarToolbarItem = item
            updateAddressBarSizing()
            return item
        default:
            return nil
        }
    }

    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.openBrowserNavigation, .space, .openBrowserAddressBar, .space, .openBrowserNewTab]
    }

    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.openBrowserNewTab, .openBrowserNavigation, .openBrowserAddressBar, .space, .flexibleSpace]
    }

    func validateToolbarItem(_ item: NSToolbarItem) -> Bool {
        true
    }

    func validateUserInterfaceItem(_ item: NSValidatedUserInterfaceItem) -> Bool {
        switch item.action {
        case #selector(goBack(_:)):
            return tabView.canGoBack
        case #selector(goForward(_:)):
            return tabView.canGoForward
        case #selector(reload(_:)):
            return tabView.currentURL != nil
        default:
            return true
        }
    }

    @objc func newTab(_ sender: Any?) {
        createNativeTab(focusAddressBar: true)
    }

    @objc private func goBack(_ sender: Any?) {
        tabView.goBack()
    }

    @objc private func goForward(_ sender: Any?) {
        tabView.goForward()
    }

    @objc private func reload(_ sender: Any?) {
        tabView.reload()
    }

    @objc private func windowDidResize(_ notification: Notification) {
        updateAddressBarSizing()
    }

    deinit {
        if let resizeObservedWindow {
            NotificationCenter.default.removeObserver(self,
                                                      name: NSWindow.didResizeNotification,
                                                      object: resizeObservedWindow)
        }
    }
}

extension OpenBrowserWindowController: OpenBrowserTabDelegate {
    func browserTab(_ tab: OpenBrowserTabView, didUpdateURL url: URL?) {
        updateAddressBar(with: url)
        updateNavigationButtons()
    }

    func browserTab(_ tab: OpenBrowserTabView, didUpdateTitle title: String?) {
        let displayTitle = tab.displayTitle
        window?.title = displayTitle
        window?.tab.title = displayTitle
    }

    func browserTab(_ tab: OpenBrowserTabView, didUpdateProgress progress: Double?) {
        addressBarView?.setLoadProgress(progress)
    }

    func browserTabDidUpdateNavigationState(_ tab: OpenBrowserTabView) {
        updateNavigationButtons()
    }

    func browserTab(_ tab: OpenBrowserTabView, didRequestOpenURL url: URL) {
        createNativeTab(loading: url)
    }
}

extension OpenBrowserWindowController: OpenBrowserAddressBarViewDelegate {
    func addressBarView(_ addressBarView: OpenBrowserAddressBarView, didSubmitURL urlString: String) {
        guard let url = normalizedURL(from: urlString) else {
            NSSound.beep()
            return
        }
        tabView.load(url)
    }

    func addressBarViewDidChangeEditingState(_ addressBarView: OpenBrowserAddressBarView, isEditing: Bool) {
        addressBarBackgroundView?.isFocused = isEditing
    }

    func addressBarViewDidChangeAppearance(_ addressBarView: OpenBrowserAddressBarView) {
        addressBarBackgroundView?.refreshAppearance()
    }
}

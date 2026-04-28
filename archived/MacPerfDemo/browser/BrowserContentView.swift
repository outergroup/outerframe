//
//  BrowserContentView.swift
//

import AppKit
import OuterdocViewer

class BrowserContentView: NSView, NSToolbarDelegate, NSWindowDelegate {
    // Main content view holding the document
    private var documentView: ScrollableTextView?
    
    // Text field for URL input
    private var urlTextField: NSTextField?
    
    // Current URL being displayed
    private var currentURL: URL?
    
    // Document model managing content and plugins
    private let documentModel = DocumentModel()
    
    // Navigation history
    private var navigationHistory: [URL] = []
    private var currentHistoryIndex: Int = -1
    
    // Toolbar items that need enabling/disabling
    private weak var backButton: NSToolbarItem? {
        didSet {
            updateBackForwardState()
        }
    }
    private weak var forwardButton: NSToolbarItem? {
        didSet {
            updateBackForwardState()
        }
    }
    
    // Computed properties to determine if back/forward should be enabled
    private var canGoBack: Bool {
        return currentHistoryIndex > 0 && !navigationHistory.isEmpty
    }
    
    private var canGoForward: Bool {
        return currentHistoryIndex < navigationHistory.count - 1 && !navigationHistory.isEmpty
    }
    
    init() {
        super.init(frame: .zero)
        setupView()
        setupKeyboardShortcuts()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
        setupKeyboardShortcuts()
    }
    
    private func setupView() {
        // Create empty document view with document model
        documentView = ScrollableTextView(documentModel: documentModel)
        
        if let documentView = documentView {
            documentView.translatesAutoresizingMaskIntoConstraints = false
            addSubview(documentView)
            
            NSLayoutConstraint.activate([
                documentView.topAnchor.constraint(equalTo: topAnchor),
                documentView.leadingAnchor.constraint(equalTo: leadingAnchor),
                documentView.trailingAnchor.constraint(equalTo: trailingAnchor),
                documentView.bottomAnchor.constraint(equalTo: bottomAnchor)
            ])
        }
    }
    
    // MARK: - URL Loading
    
    func navigate(_ urlString: String) {
        guard let url = URL(string: urlString) else {
            print("Invalid URL: \(urlString)")
            // Show error to user
            let alert = NSAlert()
            alert.messageText = "Invalid URL"
            alert.informativeText = "The URL '\(urlString)' is not valid."
            alert.alertStyle = .warning
            alert.runModal()
            return
        }
        
        navigate(url)
    }
    
    private func navigate(_ url: URL) {
        // Only add to history if we're loading a new URL (not navigating)
        // Only add to history if:
        // 1. We're loading a new URL (not the current one)
        // 2. We're not currently navigating through history
        // 3. The URL is valid
        if currentURL != url && !isNavigating {
            // If we're not at the end of the history, remove forward entries
            if currentHistoryIndex < navigationHistory.count - 1 {
                print("Removing forward history from index \(currentHistoryIndex + 1)")
                navigationHistory.removeSubrange((currentHistoryIndex + 1)...)
            }
            
            // Add the URL to history
            navigationHistory.append(url)
            currentHistoryIndex = navigationHistory.count - 1

            // Update navigation state immediately after modifying history
            updateBackForwardState()
        } else {
            print("Not adding to history: currentURL=\(currentURL?.absoluteString ?? "nil"), isNavigating=\(isNavigating)")
        }
        
        // Update current URL
        currentURL = url
        
        // Update the URL text field
        urlTextField?.stringValue = url.absoluteString
        
        // Always update navigation buttons
        updateBackForwardState()
        
        // Check if it's a file URL or a network URL
        if url.isFileURL {
            loadLocalFile(url)
        } else {
            loadRemoteURL(url, cachePolicy: .useProtocolCachePolicy)
        }
    }
    
    private var isNavigating = false
    
    private func updateBackForwardState() {
        // Use performSelector(onMainThread:) for more reliable toolbar updates
        if Thread.isMainThread {
            updateButtonsImmediately()
        } else {
            self.performSelector(onMainThread: #selector(updateButtonsImmediately), with: nil, waitUntilDone: false)
        }
    }
    
    @objc private func updateButtonsImmediately() {
        // We may need to override validation methods so use a subclass approach
        // This approach works with private validation mechanism in NSToolbar
        class ToolbarItemHelper {
            static func forceUpdate(_ item: NSToolbarItem?, enabled: Bool) {
                guard let item = item else { return }
                
                // Only update if needed to prevent event loops
                if item.isEnabled != enabled {
                    item.isEnabled = enabled
                    
                    // Register for next run loop to make sure state persists
                    DispatchQueue.main.async {
                        item.isEnabled = enabled
                    }
                }
            }
        }
        
        ToolbarItemHelper.forceUpdate(backButton, enabled: canGoBack)
        ToolbarItemHelper.forceUpdate(forwardButton, enabled: canGoForward)
    }
    
    private func loadLocalFile(_ url: URL) {
        do {
            // Delegate full deserialization to documentModel
            try documentModel.read(from: url)
        } catch {
            print("Error loading file: \(error)")
        }
    }
    
    // MARK: - Keyboard Shortcuts
    
    private func setupKeyboardShortcuts() {
        // Register for key event notifications
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self = self else { return event }
            
            if event.modifierFlags.contains(.command) {
                // Check for Command-L
                if event.charactersIgnoringModifiers == "l" {
                    self.focusAddressBar()
                    return nil // Consume the event
                }
                
                // Check for Command-R
                if event.charactersIgnoringModifiers == "r" {
                    self.reloadPage()
                    return nil // Consume the event
                }
            }
            
            return event // Pass the event through
        }
    }
    
    @objc func focusAddressBar() {
        guard let urlTextField = self.urlTextField else { return }
        
        // Make the window first responder if needed
        if let window = self.window, window.firstResponder != urlTextField {
            // Select all text and focus the field
            urlTextField.selectText(nil)
            window.makeFirstResponder(urlTextField)
        }
    }
    
    // MARK: - Actions
    
    @objc func loadURLFromTextField(_ sender: NSTextField) {
        navigate(sender.stringValue)
    }
    
    // MARK: - Window Delegate
    
    func windowWillClose(_ notification: Notification) {
        documentModel.cleanupDynamicSubstrings()
    }
    
    deinit {
        documentModel.cleanupDynamicSubstrings()
    }
    
    // MARK: - Validation 
    
    // Special method that NSToolbar will call to validate toolbar items
    @objc func validateToolbarItem(_ item: NSToolbarItem) -> Bool {
        if item.itemIdentifier == Self.goBackItemIdentifier {
            return canGoBack
        } else if item.itemIdentifier == Self.goForwardItemIdentifier {
            return canGoForward
        }
        return true
    }
}

// MARK: - Responsive Address Bar Container

class ResponsiveAddressBarContainer: NSView {
    private var textField: NSTextField
    private var desiredWidthPercentage: CGFloat
    private var widthConstraint: NSLayoutConstraint?
    
    init(frame: NSRect, textField: NSTextField, desiredWidthPercentage: CGFloat) {
        self.textField = textField
        self.desiredWidthPercentage = desiredWidthPercentage
        super.init(frame: frame)
        
        addSubview(textField)
        
        // Register for window resize notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowDidResize(_:)),
            name: NSWindow.didResizeNotification,
            object: nil
        )
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        updateWidth()
    }
    
    @objc private func windowDidResize(_ notification: Notification) {
        // Only respond to our own window's resize events
        if let resizedWindow = notification.object as? NSWindow,
           resizedWindow == self.window {
            updateWidth()
        }
    }
    
    private func updateWidth() {
        guard let window = self.window else { return }
        
        // Calculate width as percentage of window width
        let newWidth = max(200, window.frame.width * desiredWidthPercentage)
        
        // Remove old constraint if it exists
        if let constraint = widthConstraint {
            removeConstraint(constraint)
        }
        
        // Create and activate new constraint
        widthConstraint = widthAnchor.constraint(equalToConstant: newWidth)
        widthConstraint?.isActive = true
        
        // Trigger layout update
        needsLayout = true
    }
}

// MARK: - Array Extension for Safe Access

extension Array {
    subscript(safe index: Int) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

// MARK: - NSToolbarDelegate

extension BrowserContentView {
    private static let toolbarIdentifier = "BrowserToolbar"
    private static let urlItemIdentifier = NSToolbarItem.Identifier("URLItem")
    private static let reloadItemIdentifier = NSToolbarItem.Identifier("ReloadItem")
    private static let goBackItemIdentifier = NSToolbarItem.Identifier("GoBackItem")
    private static let goForwardItemIdentifier = NSToolbarItem.Identifier("GoForwardItem")
    
    func toolbar(_ toolbar: NSToolbar, itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier, willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? {
        if itemIdentifier == Self.urlItemIdentifier {
            // Standard toolbar item for the URL field
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "URL"
            
            // Create URL text field - increase height to match Safari
            let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 500, height: 30))
            textField.translatesAutoresizingMaskIntoConstraints = false
            textField.placeholderString = "Enter URL"
            textField.target = self
            textField.action = #selector(loadURLFromTextField(_:))
            textField.cell?.sendsActionOnEndEditing = false
            textField.bezelStyle = .roundedBezel
            textField.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
            
            // Store reference to the text field
            urlTextField = textField
            
            // Create a custom container that will resize with the window
            let container = ResponsiveAddressBarContainer(
                frame: NSRect(x: 0, y: 0, width: 600, height: 30),
                textField: textField,
                desiredWidthPercentage: 0.6
            )
            container.translatesAutoresizingMaskIntoConstraints = false
            
            // Set constraints with horizontal padding
            NSLayoutConstraint.activate([
                // Add horizontal padding (20 points on each side)
                textField.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 20),
                textField.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -20),
                textField.topAnchor.constraint(equalTo: container.topAnchor),
                textField.bottomAnchor.constraint(equalTo: container.bottomAnchor)
            ])
            
            // Set minimum width to ensure usability on very small windows
            container.widthAnchor.constraint(greaterThanOrEqualToConstant: 200).isActive = true
            
            // Set up the toolbar item
            item.view = container
            
            return item
        } else if itemIdentifier == Self.reloadItemIdentifier {
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "Reload"
            item.toolTip = "Reload the current page"
            item.image = NSImage(systemSymbolName: "arrow.clockwise", accessibilityDescription: "Reload")
            item.target = self
            item.action = #selector(reloadPage)
            return item
        } else if itemIdentifier == Self.goBackItemIdentifier {
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "Back"
            item.toolTip = "Go back"
            item.image = NSImage(systemSymbolName: "arrow.left", accessibilityDescription: "Back")
            item.target = self
            item.action = #selector(goBack)
            item.isEnabled = false // Initial state, will be updated
            
            // Store reference and let the property observer update the state
            backButton = item
            return item
        } else if itemIdentifier == Self.goForwardItemIdentifier {
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "Forward"
            item.toolTip = "Go forward"
            item.image = NSImage(systemSymbolName: "arrow.right", accessibilityDescription: "Forward")
            item.target = self
            item.action = #selector(goForward)
            item.isEnabled = false // Initial state, will be updated
            
            // Store reference and let the property observer update the state
            forwardButton = item
            return item
        }
        
        return nil
    }
    
    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        return [
            Self.goBackItemIdentifier,
            Self.goForwardItemIdentifier,
            .flexibleSpace,
            Self.urlItemIdentifier,
            Self.reloadItemIdentifier,
            .flexibleSpace
        ]
    }
    
    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        return [
            Self.goBackItemIdentifier,
            Self.goForwardItemIdentifier,
            .flexibleSpace,
            Self.urlItemIdentifier,
            Self.reloadItemIdentifier,
            .flexibleSpace
        ]
    }
    
    @objc func reloadPage() {
        if let url = currentURL {
            if url.isFileURL {
                loadLocalFile(url)
            } else {
                loadRemoteURL(url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData)
            }
        }
    }
    
    private func loadRemoteURL(_ url: URL, cachePolicy: URLRequest.CachePolicy) {
        let request = URLRequest(url: url, cachePolicy: cachePolicy)
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }
            // Handle errors and HTTP failures
            if let error = error {
                DispatchQueue.main.async {
                    self.documentModel.textStorage.setAttributedString(
                        NSAttributedString(string: "Error loading URL: \(error.localizedDescription)"))
                }
                return
            }
            guard let data = data, let http = response as? HTTPURLResponse else {
                DispatchQueue.main.async {
                    self.documentModel.textStorage.setAttributedString(
                        NSAttributedString(string: "Error: No data received from server."))
                }
                return
            }
            guard http.statusCode == 200 else {
                DispatchQueue.main.async {
                    self.documentModel.textStorage.setAttributedString(
                        NSAttributedString(string: "Error: Server returned status code \(http.statusCode)"))
                }
                return
            }
            // Unified parse of document buffer
            var attributedStringObjC: NSAttributedString?
            let success = readOuterdocFromPackedBuffer(data, &attributedStringObjC)
            if success, let attrStr = attributedStringObjC {
                DispatchQueue.main.async {
                    // Now set the attributed string, which may trigger attachment processing
                    self.documentModel.textStorage.setAttributedString(attrStr)
                    
                    // Process any placeholder attachments for already-registered plugins
                    self.documentModel.processPlaceholderAttachments()
                }
            } else {
                DispatchQueue.main.async {
                    self.documentModel.textStorage.setAttributedString(
                        NSAttributedString(string: "Error: Could not parse response as an outerdoc document."))
                }
            }
        }
        
        task.resume()
        
        // Update current URL in navigation history
        currentURL = url
    }
    
    @objc func goBack() {
        // Check if we can go back
        guard currentHistoryIndex > 0 && !navigationHistory.isEmpty else {
            print("Cannot go back: index = \(currentHistoryIndex), history size = \(navigationHistory.count)")
            // Update navigation state
            updateBackForwardState()
            return
        }
        
        print("Going back from index \(currentHistoryIndex) to \(currentHistoryIndex-1)")
        
        // Set flag to prevent adding this navigation to history
        isNavigating = true
        
        // Update index and load URL
        currentHistoryIndex -= 1
        
        if let url = navigationHistory[safe: currentHistoryIndex] {
            print("Back: Loading URL \(url.absoluteString)")
            navigate(url)
        } else {
            print("Error: Invalid URL at history index \(currentHistoryIndex)")
        }
        
        isNavigating = false
        
        // Ensure buttons are properly updated after navigation completes
        updateBackForwardState()
    }
    
    @objc func goForward() {
        // Check if we can go forward
        guard currentHistoryIndex < navigationHistory.count - 1 && !navigationHistory.isEmpty else {
            print("Cannot go forward: index = \(currentHistoryIndex), history size = \(navigationHistory.count)")
            // Update navigation state
            updateBackForwardState()
            return
        }
        
        print("Going forward from index \(currentHistoryIndex) to \(currentHistoryIndex+1)")
        
        // Set flag to prevent adding this navigation to history
        isNavigating = true
        
        // Update index and load URL
        currentHistoryIndex += 1
        
        if let url = navigationHistory[safe: currentHistoryIndex] {
            print("Forward: Loading URL \(url.absoluteString)")
            navigate(url)
        } else {
            print("Error: Invalid URL at history index \(currentHistoryIndex)")
        }
        
        isNavigating = false
        
        // Ensure buttons are properly updated after navigation completes
        updateBackForwardState()
    }
}

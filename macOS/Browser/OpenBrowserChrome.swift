import AppKit
import QuartzCore

final class OpenBrowserToolbarSizedView: NSView {
    var preferredSize = NSSize(width: 220, height: 36) {
        didSet {
            invalidateIntrinsicContentSize()
        }
    }

    override var intrinsicContentSize: NSSize {
        preferredSize
    }
}

final class OpenBrowserToolbarButton: NSControl {
    private var isHovered = false
    private var isPressed = false
    private var hoverTrackingArea: NSTrackingArea?
    private let iconView = NSImageView()
    private var baseImage: NSImage?

    var contentTintColor: NSColor = .labelColor {
        didSet {
            updateContentAppearance()
        }
    }

    override var isEnabled: Bool {
        didSet {
            updateContentAppearance()
            updateAppearance()
        }
    }

    init(image: NSImage, target: AnyObject?, action: Selector?) {
        self.baseImage = image
        super.init(frame: .zero)
        self.target = target
        self.action = action
        configureControl()
    }

    required init?(coder: NSCoder) {
        return nil
    }

    override var intrinsicContentSize: NSSize {
        NSSize(width: 30, height: 30)
    }

    override func mouseEntered(with event: NSEvent) {
        super.mouseEntered(with: event)
        isHovered = true
        updateAppearance()
    }

    override func mouseExited(with event: NSEvent) {
        super.mouseExited(with: event)
        isHovered = false
        updateAppearance()
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()

        if let hoverTrackingArea {
            removeTrackingArea(hoverTrackingArea)
        }

        let trackingArea = NSTrackingArea(rect: .zero,
                                          options: [.activeInKeyWindow, .inVisibleRect, .mouseEnteredAndExited],
                                          owner: self,
                                          userInfo: nil)
        addTrackingArea(trackingArea)
        hoverTrackingArea = trackingArea
    }

    override func layout() {
        super.layout()
        layer?.cornerRadius = bounds.height / 2
        let iconSize = iconView.image?.size ?? NSSize(width: 16, height: 16)
        iconView.frame = CGRect(x: floor((bounds.width - iconSize.width) / 2),
                                y: floor((bounds.height - iconSize.height) / 2),
                                width: iconSize.width,
                                height: iconSize.height)
    }

    override func mouseDown(with event: NSEvent) {
        guard isEnabled, let window else { return }

        isPressed = true
        updateAppearance()

        while let nextEvent = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp]) {
            let location = convert(nextEvent.locationInWindow, from: nil)
            let isInside = bounds.contains(location)

            switch nextEvent.type {
            case .leftMouseDragged:
                if isPressed != isInside {
                    isPressed = isInside
                    updateAppearance()
                }
            case .leftMouseUp:
                isPressed = false
                updateAppearance()
                if isInside {
                    _ = sendAction(action, to: target)
                }
                return
            default:
                break
            }
        }

        isPressed = false
        updateAppearance()
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateContentAppearance()
        updateAppearance()
    }

    private func configureControl() {
        wantsLayer = true
        layer?.masksToBounds = true
        setAccessibilityRole(.button)

        iconView.imageScaling = .scaleProportionallyUpOrDown
        addSubview(iconView)

        updateContentAppearance()
        updateAppearance()
    }

    private func updateContentAppearance() {
        let effectiveTint = isEnabled ? contentTintColor : .tertiaryLabelColor
        effectiveAppearance.performAsCurrentDrawingAppearance {
            if let baseImage {
                let image = baseImage.copy() as? NSImage ?? baseImage
                image.isTemplate = true
                iconView.image = image
            }
            iconView.contentTintColor = effectiveTint
        }
        alphaValue = isEnabled ? 1 : 0.55
        needsLayout = true
    }

    private func updateAppearance() {
        let backgroundColor: NSColor
        if !isEnabled {
            backgroundColor = .clear
        } else if isPressed {
            backgroundColor = .secondarySystemFill
        } else if isHovered {
            backgroundColor = .tertiarySystemFill
        } else {
            backgroundColor = .clear
        }
        effectiveAppearance.performAsCurrentDrawingAppearance {
            layer?.backgroundColor = backgroundColor.cgColor
        }
    }
}

final class OpenBrowserToolbarBackgroundView: NSView {
    private weak var glassView: NSView?

    var isFocused = false {
        didSet {
            updateAppearance()
        }
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    override func layout() {
        super.layout()
        let radius = bounds.height / 2
        layer?.cornerRadius = radius
        if #available(macOS 26, *), let glassView = glassView as? NSGlassEffectView {
            glassView.cornerRadius = radius
        }
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateAppearance()
    }

    func refreshAppearance() {
        updateAppearance()
    }

    private func setupView() {
        wantsLayer = true
        layer?.masksToBounds = false

        if #available(macOS 26, *) {
            let glass = NSGlassEffectView()
            glass.translatesAutoresizingMaskIntoConstraints = false
            glass.style = .regular
            addSubview(glass)
            NSLayoutConstraint.activate([
                glass.leadingAnchor.constraint(equalTo: leadingAnchor),
                glass.trailingAnchor.constraint(equalTo: trailingAnchor),
                glass.topAnchor.constraint(equalTo: topAnchor),
                glass.bottomAnchor.constraint(equalTo: bottomAnchor)
            ])
            glassView = glass
        }

        updateAppearance()
    }

    private func updateAppearance() {
        effectiveAppearance.performAsCurrentDrawingAppearance {
            let backgroundColor: NSColor
            if glassView != nil {
                backgroundColor = isFocused ? NSColor.controlAccentColor.withAlphaComponent(0.08) : NSColor.white.withAlphaComponent(0.05)
            } else {
                backgroundColor = isFocused ? NSColor.controlBackgroundColor.withAlphaComponent(0.98) : NSColor.controlBackgroundColor.withAlphaComponent(0.92)
            }
            layer?.backgroundColor = backgroundColor.cgColor
            layer?.shadowColor = NSColor.black.withAlphaComponent(glassView != nil ? 0.12 : 0.08).cgColor
            layer?.shadowOpacity = 0.2
            layer?.shadowRadius = isFocused ? 14 : 10
            layer?.shadowOffset = CGSize(width: 0, height: -1)
        }
    }
}

final class OpenBrowserSeparatorView: NSView {
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        updateColor()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        wantsLayer = true
        updateColor()
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateColor()
    }

    private func updateColor() {
        effectiveAppearance.performAsCurrentDrawingAppearance {
            layer?.backgroundColor = NSColor.separatorColor.cgColor
        }
    }
}

@MainActor
protocol OpenBrowserAddressBarViewDelegate: AnyObject {
    func addressBarView(_ addressBarView: OpenBrowserAddressBarView, didSubmitURL urlString: String)
    func addressBarViewDidChangeEditingState(_ addressBarView: OpenBrowserAddressBarView, isEditing: Bool)
    func addressBarViewDidChangeAppearance(_ addressBarView: OpenBrowserAddressBarView)
}

final class OpenBrowserFocusAwareTextField: NSTextField {
    var onResignFirstResponder: (() -> Void)?

    override func resignFirstResponder() -> Bool {
        let result = super.resignFirstResponder()
        if result {
            onResignFirstResponder?()
        }
        return result
    }
}

final class OpenBrowserAddressBarView: NSView {
    private enum DisplayContent {
        case placeholder(String)
        case remote
        case userDraft(String)
    }

    private let horizontalPadding: CGFloat = 12
    private let progressHorizontalInset: CGFloat = 6
    private let progressVerticalInset: CGFloat = 4
    private let progressBarHeight: CGFloat = 2
    private let progressAnimationDuration: TimeInterval = 0.2

    private let displayLabel = NSTextField(labelWithString: "")
    private let editField = OpenBrowserFocusAwareTextField()
    private let progressView = NSView()
    private var isEditing = false
    private var isTransitioningToEdit = false
    private var displayContent: DisplayContent = .placeholder("")
    private var locationDisplayText: String?
    private var currentEditingText = ""
    private var userEditedText: String?
    private var placeholderText: String?
    private var initialEditSnapshot: String?
    private var currentProgressValue: CGFloat?
    private var hideProgressWorkItem: DispatchWorkItem?

    weak var delegate: OpenBrowserAddressBarViewDelegate?

    override init(frame: NSRect) {
        super.init(frame: frame)
        setupViews()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupViews()
    }

    override func layout() {
        super.layout()

        let availableWidth = max(bounds.width - (horizontalPadding * 2), 0)
        let editSize = editField.intrinsicContentSize
        let editHeight = max(editSize.height, 0)
        editField.frame = NSRect(x: horizontalPadding,
                                 y: ((bounds.height - editHeight) / 2).rounded(),
                                 width: availableWidth,
                                 height: editHeight.rounded())

        let displayHeight = min(displayLabel.intrinsicContentSize.height, bounds.height)
        displayLabel.frame = NSRect(x: horizontalPadding,
                                    y: ((bounds.height - displayHeight) / 2).rounded(),
                                    width: availableWidth.rounded(),
                                    height: displayHeight.rounded())

        progressView.frame = progressFrame(for: currentProgressValue ?? 0)
    }

    override func mouseDown(with event: NSEvent) {
        if !isEditing {
            startEditing()
        }
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        if !isEditing, bounds.contains(point) {
            return self
        }
        return super.hitTest(point)
    }

    override var acceptsFirstResponder: Bool {
        true
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        progressView.layer?.backgroundColor = NSColor.controlAccentColor.cgColor
        delegate?.addressBarViewDidChangeAppearance(self)
    }

    func startEditing() {
        guard !isEditing else { return }

        isTransitioningToEdit = true
        isEditing = true
        delegate?.addressBarViewDidChangeEditingState(self, isEditing: true)

        let startingText: String
        if let userEditedText, !userEditedText.isEmpty {
            startingText = userEditedText
        } else {
            switch displayContent {
            case .userDraft(let text):
                startingText = text
            case .remote:
                startingText = currentEditingText
            case .placeholder:
                startingText = ""
            }
        }

        editField.stringValue = startingText
        editField.isHidden = false
        displayLabel.isHidden = true
        window?.makeFirstResponder(editField)
        initialEditSnapshot = startingText

        DispatchQueue.main.async { [weak self] in
            guard let self, self.isEditing else { return }
            self.editField.selectText(nil)
            self.isTransitioningToEdit = false
        }
    }

    func update(displayText: String, editingText: String) {
        locationDisplayText = displayText
        currentEditingText = editingText
        userEditedText = nil
        placeholderText = nil
        updateEditFieldPlaceholder()
        displayContent = .remote
        if !isEditing {
            updateDisplay()
        }
    }

    func setPlaceholderText(_ text: String) {
        placeholderText = text
        locationDisplayText = nil
        currentEditingText = ""
        userEditedText = nil
        updateEditFieldPlaceholder()
        displayContent = .placeholder(text)
        if !isEditing {
            updateDisplay()
        }
    }

    func setLoadProgress(_ progress: Double?, animated: Bool = true) {
        hideProgressWorkItem?.cancel()
        hideProgressWorkItem = nil

        guard let progress else {
            guard !progressView.isHidden else {
                currentProgressValue = nil
                return
            }

            currentProgressValue = 1
            progressView.frame = progressFrame(for: 1)
            let workItem = DispatchWorkItem { [weak self] in
                self?.hideProgress(animated: animated)
            }
            hideProgressWorkItem = workItem
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15, execute: workItem)
            return
        }

        let clamped = CGFloat(max(0, min(progress, 1)))
        currentProgressValue = clamped
        progressView.layer?.backgroundColor = NSColor.controlAccentColor.cgColor
        progressView.isHidden = false
        progressView.alphaValue = 1
        progressView.frame = progressFrame(for: clamped)
        needsLayout = true
    }

    private func setupViews() {
        wantsLayer = true
        layer?.masksToBounds = true

        progressView.wantsLayer = true
        progressView.isHidden = true
        progressView.alphaValue = 1
        progressView.layer?.cornerRadius = progressBarHeight / 2
        progressView.layer?.backgroundColor = NSColor.controlAccentColor.cgColor
        addSubview(progressView)

        displayLabel.font = .systemFont(ofSize: 13, weight: .medium)
        displayLabel.textColor = .secondaryLabelColor
        displayLabel.lineBreakMode = .byTruncatingMiddle
        displayLabel.cell?.truncatesLastVisibleLine = true
        displayLabel.cell?.wraps = false
        displayLabel.isSelectable = false
        displayLabel.backgroundColor = .clear
        addSubview(displayLabel)

        editField.isHidden = true
        editField.isBordered = false
        editField.focusRingType = .none
        editField.backgroundColor = .clear
        editField.delegate = self
        editField.font = .systemFont(ofSize: 13, weight: .medium)
        editField.maximumNumberOfLines = 1
        editField.lineBreakMode = .byTruncatingTail
        editField.cell?.truncatesLastVisibleLine = true
        editField.cell?.isScrollable = true
        editField.cell?.wraps = false
        editField.onResignFirstResponder = { [weak self] in
            self?.handleEditFieldLostFocus()
        }
        addSubview(editField)

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(windowDidUpdate(_:)),
                                               name: NSWindow.didUpdateNotification,
                                               object: nil)
    }

    private func updateEditFieldPlaceholder() {
        guard let placeholderText else {
            editField.placeholderAttributedString = nil
            return
        }

        editField.placeholderAttributedString = NSAttributedString(
            string: placeholderText,
            attributes: [
                .foregroundColor: NSColor.placeholderTextColor,
                .font: editField.font ?? NSFont.systemFont(ofSize: NSFont.systemFontSize)
            ]
        )
    }

    private func progressFrame(for progress: CGFloat) -> NSRect {
        let clamped = max(0, min(progress, 1))
        let availableWidth = max(0, bounds.width - (progressHorizontalInset * 2))
        return NSRect(x: progressHorizontalInset,
                      y: progressVerticalInset,
                      width: availableWidth * clamped,
                      height: progressBarHeight)
    }

    private func hideProgress(animated: Bool) {
        currentProgressValue = nil
        hideProgressWorkItem = nil

        if animated {
            NSAnimationContext.runAnimationGroup { context in
                context.duration = progressAnimationDuration
                context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                progressView.animator().alphaValue = 0
            } completionHandler: { [weak self] in
                guard let self else { return }
                self.progressView.isHidden = true
                self.progressView.alphaValue = 1
                self.progressView.frame = self.progressFrame(for: 0)
            }
        } else {
            progressView.isHidden = true
            progressView.alphaValue = 1
            progressView.frame = progressFrame(for: 0)
        }
    }

    private func handleEditFieldLostFocus() {
        guard isEditing, !isTransitioningToEdit else { return }

        let text = editField.stringValue
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            userEditedText = nil
            displayContent = defaultDisplayContent() ?? .placeholder("")
        } else if let initialEditSnapshot, initialEditSnapshot == text {
            displayContent = defaultDisplayContent() ?? .userDraft(text)
            userEditedText = nil
        } else {
            displayContent = .userDraft(text)
            userEditedText = text
        }

        stopEditing()
    }

    private func stopEditing() {
        isEditing = false
        isTransitioningToEdit = false
        initialEditSnapshot = nil
        delegate?.addressBarViewDidChangeEditingState(self, isEditing: false)
        editField.isHidden = true
        displayLabel.isHidden = false
        updateDisplay()
    }

    private func defaultDisplayContent() -> DisplayContent? {
        if let locationDisplayText, !locationDisplayText.isEmpty {
            return .remote
        }
        if let placeholderText {
            return .placeholder(placeholderText)
        }
        return nil
    }

    private func updateDisplay() {
        switch displayContent {
        case .placeholder(let text):
            displayLabel.stringValue = text
            displayLabel.textColor = .placeholderTextColor
        case .remote:
            displayLabel.stringValue = locationDisplayText ?? ""
            displayLabel.textColor = .secondaryLabelColor
        case .userDraft(let text):
            displayLabel.stringValue = text
            displayLabel.textColor = .labelColor
        }
        needsLayout = true
    }

    @objc private func windowDidUpdate(_ notification: Notification) {
        if isEditing,
           window?.firstResponder !== editField,
           window?.firstResponder !== editField.currentEditor() {
            handleEditFieldLostFocus()
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}

extension OpenBrowserAddressBarView: NSTextFieldDelegate {
    func controlTextDidEndEditing(_ obj: Notification) {
        guard isEditing else { return }

        if let textMovement = obj.userInfo?["NSTextMovement"] as? Int,
           textMovement == NSTextMovement.return.rawValue {
            let urlString = editField.stringValue
            userEditedText = nil
            stopEditing()
            delegate?.addressBarView(self, didSubmitURL: urlString)
        }
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.cancelOperation(_:)) {
            userEditedText = nil
            displayContent = defaultDisplayContent() ?? .placeholder("")
            stopEditing()
            return true
        }
        return false
    }
}

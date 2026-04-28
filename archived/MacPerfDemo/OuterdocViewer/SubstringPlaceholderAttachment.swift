//
//  UntrustedFigureAttachment.swift
//

import AppKit
import MetalKit
import IOSurface

// MARK: - SubstringPlaceholderAttachment

/// This attachment serves two purposes:
/// - Shows placeholder UI before an attachment has rendered
/// - This attachment is removed from the document and replaced with the generated substring.
public class SubstringPlaceholderAttachment: NSTextAttachment {
    @objc public var contentType: String?

    var beingReplaced = false

    public override var image: NSImage? {
        get {
            return nil
        }
        set {
            fatalError("Cannot set image")
        }
    }

    public override var fileType: String? {
        get {
            return "dev.outergroup.outerdocsubstring"
        }
        set {
            fatalError()
        }
    }

    public override func viewProvider(
        for parentView: NSView?, location: NSTextLocation, textContainer: NSTextContainer?
    ) -> SubstringPlaceholderAttachmentViewProvider? {
        // Create a new view provider
        return SubstringPlaceholderAttachmentViewProvider(
            textAttachment: self, parentView: parentView,
            textLayoutManager: textContainer?.textLayoutManager, location: location)
    }

    // Must set a preferredFilename for copy-paste to work
    public override var fileWrapper: FileWrapper? {
        get {
            let wrapper = FileWrapper(regularFileWithContents: Data())
            wrapper.preferredFilename = "view.outerdocsubstring"
            return wrapper
        }
        set {
        }
    }
}


// MARK: - SubstringPlaceholderAttachmentViewProvider

public class SubstringPlaceholderAttachmentViewProvider: NSTextAttachmentViewProvider {
    private var containerView: ContainerView?
    /// Loading view displayed while awaiting plugin registration or figure creation
    private var loadingView: NSTextField?
    private var loadingTimer: Timer?


    static let defaultWidth: CGFloat = 400
    static let defaultHeight: CGFloat = 100

    // Class to act as container with proper sizing behavior
    class ContainerView: NSView {
        var isInitialState = true
        
        override var intrinsicContentSize: NSSize {
            if isInitialState {
                // Small blank space initially, roughly text attachment size
                return NSSize(width: 20, height: 20)
            } else {
                return NSSize(
                    width: defaultWidth,
                    height: defaultHeight)
            }
        }
    }

    public override init(
        textAttachment: NSTextAttachment,
        parentView: NSView?,
        textLayoutManager: NSTextLayoutManager?,
        location: any NSTextLocation
    ) {
        super.init(
            textAttachment: textAttachment, parentView: parentView,
            textLayoutManager: textLayoutManager, location: location)
        tracksTextAttachmentViewBounds = true
    }

    public override func loadView() {
        // Create container view - initially small and blank
        let containerView = ContainerView(
            frame: NSRect(x: 0, y: 0, width: 20, height: 20))
        containerView.translatesAutoresizingMaskIntoConstraints = false
        containerView.wantsLayer = true
        // Initially just a small blank space - no background or border
        self.containerView = containerView
        self.view = containerView

        let attachment = self.textAttachment as! SubstringPlaceholderAttachment

        // Schedule the loading state to appear after 100ms. (Avoid flashing if the substring appears immediately)
        loadingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: false) { [weak self] _ in
            guard let self = self, let containerView = self.containerView else { return }
            
            // Transition to loading state
            containerView.isInitialState = false
            
            // Update frame to new size
            containerView.frame = NSRect(
                x: 0, y: 0,
                width: SubstringPlaceholderAttachmentViewProvider.defaultWidth,
                height: SubstringPlaceholderAttachmentViewProvider.defaultHeight
            )
            
            // Update appearance to show loading box
            containerView.layer?.backgroundColor = NSColor.lightGray.withAlphaComponent(0.2).cgColor
            containerView.layer?.borderColor = NSColor.lightGray.cgColor
            containerView.layer?.borderWidth = 1
            containerView.layer?.cornerRadius = 8

            let loadingText = "Loading content \(attachment.contentType ?? "unknown")..."
            let loadingView = NSTextField(labelWithString: loadingText)
            self.loadingView = loadingView
            loadingView.alignment = .center
            loadingView.translatesAutoresizingMaskIntoConstraints = false
            
            containerView.addSubview(loadingView)
            
            NSLayoutConstraint.activate([
                loadingView.centerXAnchor.constraint(equalTo: containerView.centerXAnchor),
                loadingView.centerYAnchor.constraint(equalTo: containerView.centerYAnchor),
            ])
            
            // Invalidate intrinsic content size and notify text system
            containerView.invalidateIntrinsicContentSize()

            if let textLayoutManager = self.textLayoutManager {
                let location = self.location
                if let end = textLayoutManager.location(location, offsetBy: 1),
                   let range = NSTextRange(location: location, end: end) {
                    textLayoutManager.invalidateLayout(for: range)
                }
            }
        }
    }
    
    deinit {
        loadingTimer?.invalidate()
    }
}

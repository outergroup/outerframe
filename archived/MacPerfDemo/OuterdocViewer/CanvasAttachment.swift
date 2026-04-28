import AppKit
import IOSurface

/// A canvas attachment that uses NSTextAttachmentViewProvider for stable view display.
/// This attachment combines the placeholder attachment approach with the view provider system.
@objc public class CanvasAttachment: NSTextAttachment {
    /// Stable identifier for the canvas
    @objc public var canvasID: UUID
    
    /// Identifier for the substring this canvas belongs to
    @objc public var substringId: UUID
    
    /// The width of the canvas in points
    @objc public var width: CGFloat
    
    /// The height of the canvas in points
    @objc public var height: CGFloat

    @objc public var iosurface: IOSurface

    @objc public init(canvasID: UUID, substringId: UUID, width: CGFloat, height: CGFloat, iosurface: IOSurface) {
        self.canvasID = canvasID
        self.substringId = substringId
        self.width = width
        self.height = height
        self.iosurface = iosurface
        super.init(data: nil, ofType: nil)
        print("CanvasAttachment: Created with canvasID: \(canvasID), substringId: \(substringId)")
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    public override var bounds: CGRect {
        get {
            return CGRect(x: 0, y: 0, width: width, height: height)
        }
        set {
            // Update dimensions if bounds are changed
            width = newValue.width
            height = newValue.height
        }
    }

    public override var image: NSImage? {
        get {
            // Return a transparent image to prevent the default "missing image" icon
            return NSImage(size: NSSize(width: width, height: height))
        }
        set {
            fatalError("Cannot set image on CanvasAttachment")
        }
    }

    public override var fileType: String? {
        get {
            return "dev.outergroup.canvas"
        }
        set {
            fatalError("Cannot set fileType on CanvasAttachment")
        }
    }

    public override func viewProvider(
        for parentView: NSView?, location: NSTextLocation, textContainer: NSTextContainer?
    ) -> NSTextAttachmentViewProvider? {
        return CanvasAttachmentViewProvider(
            textAttachment: self,
            parentView: parentView,
            textLayoutManager: textContainer?.textLayoutManager,
            location: location
        )
    }
    
    // Must set a preferredFilename for copy-paste to work
    public override var fileWrapper: FileWrapper? {
        get {
            let wrapper = FileWrapper(regularFileWithContents: Data())
            wrapper.preferredFilename = "canvas.canvas"
            return wrapper
        }
        set {
        }
    }

    deinit {
        print("CanvasAttachment: Cleaning up canvas with ID \(canvasID)")
        GlobalCanvasRegistry.shared.removeCanvas(canvasID: canvasID)
    }
}

// MARK: - CanvasAttachmentViewProvider

public class CanvasAttachmentViewProvider: NSTextAttachmentViewProvider {
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

        // Hack: Invalidate here. (For some reason the initial replaceCharacters that inserts this isn't invalidating the layout?)
        let end = textLayoutManager!.location(location, offsetBy: 1)
        textLayoutManager!.invalidateLayout(for: .init(location: location, end: end)!)
    }
    
    public override func loadView() {
        guard let attachment = self.textAttachment as? CanvasAttachment else {
            fatalError("Expected CanvasAttachment but got \(type(of: self.textAttachment))")
        }
        
        // Create the canvas view directly
        let canvasView = CanvasAttachmentView(
            frame: NSRect(x: 0, y: 0, width: attachment.width, height: attachment.height),
            canvasID: attachment.canvasID,
            substringId: attachment.substringId,
            iosurface: attachment.iosurface
        )

        self.view = canvasView
        
        // Register this view with the global registry
        GlobalCanvasRegistry.shared.registerView(
            canvasID: attachment.canvasID,
            substringId: attachment.substringId,
            view: canvasView
        )
        
        print("CanvasAttachmentViewProvider: Created canvas view for ID \(attachment.canvasID)")
    }
}

// MARK: - CanvasAttachmentView

/// View for rendering canvas content from IOSurfaces provided by content process.
/// This is now provided directly through NSTextAttachmentViewProvider instead of overlay positioning.
class CanvasAttachmentView: NSView {
    /// Stable identifier for the canvas
    var canvasID: UUID
    
    /// Identifier for the substring this canvas belongs to
    var substringId: UUID

    /// Single IOSurface for this canvas
    var ioSurface: IOSurface

    func onIOSurfaceUpdate() {
        let layer = layer!
        layer.contents = nil
        layer.contents = ioSurface
    }

    init(
        frame: NSRect, canvasID: UUID, substringId: UUID, iosurface: IOSurface
    ) {
        self.canvasID = canvasID
        self.substringId = substringId
        self.ioSurface = iosurface

        super.init(frame: frame)

        let layer = CALayer()
        if let screen = NSScreen.main {
            layer.contentsScale = screen.backingScaleFactor
        }
        layer.frame = CGRect(origin: .zero, size: NSSize(width: frame.width, height: frame.height))
        layer.contents = iosurface
        self.layer = layer
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override var intrinsicContentSize: NSSize {
        return self.layer!.frame.size
    }
}

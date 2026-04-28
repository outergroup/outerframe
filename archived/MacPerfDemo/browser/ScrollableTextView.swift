//
//  ScrollableTextView.swift
//

import AppKit
import OuterdocViewer

class ScrollableTextView: NSView {
    var textView: NSTextView
    /// Document model providing content and plugins
    private let documentModel: DocumentModel
    var scrollView: NSScrollView

    /// Initialize with a shared DocumentModel
    init(documentModel: DocumentModel) {
        self.documentModel = documentModel
        // Create a TextKit 2 stack with NSTextLayoutManager
        let textStorage = documentModel.textStorage
        
        // Create the TextKit 2 content storage
        let textContentStorage = NSTextContentStorage()
        textContentStorage.textStorage = textStorage
        
        // Create the TextKit 2 layout manager
        let textLayoutManager = NSTextLayoutManager()
        textContentStorage.addTextLayoutManager(textLayoutManager)
        
        // Create a text container
        let textContainer = NSTextContainer(size: NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        textContainer.widthTracksTextView = true
        textLayoutManager.textContainer = textContainer
        
        // Create the text view with TextKit 2 stack
        textView = NSTextView(frame: .zero, textContainer: textContainer)
        
        // Wire document model into text view
        textView.textContainerInset = NSSize(width: 4, height: 6)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.isEditable = false // Read-only for browser
        textView.isSelectable = true
        textView.allowsUndo = false
        textView.drawsBackground = true
        
        // Create scroll view
        scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.documentView = textView
        scrollView.autoresizingMask = [.width, .height]
        
        super.init(frame: .zero)
        
        addSubview(scrollView)
        scrollView.frame = bounds
        scrollView.autoresizingMask = [.width, .height]
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    // Pass-through for frame changes
    override var frame: NSRect {
        didSet {
            scrollView.frame = bounds
        }
    }
}

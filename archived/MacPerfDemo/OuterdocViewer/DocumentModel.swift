//
//  DocumentModel.swift
//

import AppKit
import Foundation

// MARK: - NSTextStorageDelegate for tracking dynamic substring ranges

/// Model that holds document content for serialization/deserialization.
public class DocumentModel: NSObject {
    /// Underlying text storage for the document content.
    public let textStorage: NSTextStorage

    public override init() {
        textStorage = NSTextStorage()
        super.init()
    }

    /// Writes the document content to disk at the specified URL.
    /// - Parameter url: File URL to write to.
    public func write(to url: URL) throws {
        // Ensure file exists
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil, attributes: nil)
        }
        let fileHandle = try FileHandle(forWritingTo: url)
        // Build a temporary copy and replace embedded substrings with attachments
        let saveString = NSMutableAttributedString(attributedString: textStorage)
        saveString.beginEditing()
        saveString.enumerateAttribute(.embeddedSubstring,
                                      in: NSRange(location: 0, length: saveString.length),
                                      options: []) {
            value, range, _ in
            guard let fragAttr = value as? EmbeddedSubstringAttribute else { return }
            // Create attachment placeholder for this substring
            let attachment = SubstringPlaceholderAttachment()
            attachment.contentType = fragAttr.contentType

            // Serialize the outerdoc substring
            let data = serializeOuterdocAttributedString(
                fragAttr.attributedString
            ) ?? Data()
            
            let wrapper = FileWrapper(regularFileWithContents: data)
            wrapper.preferredFilename = "attachment.outerdoc"
            attachment.fileWrapper = wrapper
            let attrString = NSAttributedString(attachment: attachment)
            saveString.replaceCharacters(in: range, with: attrString)
        }
        saveString.endEditing()
        // Serialize the document with attachments in place
        writeOuterdocAttributedString(
            fileHandle.fileDescriptor,
            saveString)
    }

    /// Reads document content from disk into the model.
    /// - Parameter url: File URL to read from.
    public func read(from url: URL) throws {
        // Open file and parse serialized document
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { fileHandle.closeFile() }
        var attributedString: NSAttributedString?
        let success = readOuterdocFromPackedFile(
            fileHandle.fileDescriptor,
            &attributedString)
        guard success, let attributedString = attributedString
        else {
            throw NSError(
                domain: "DocumentModel", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to parse document."])
        }
        // Apply the read attributed string
        textStorage.setAttributedString(attributedString)
        // Replace embedded outerdoc substrings with their inline text
        let fullRange = NSRange(location: 0, length: textStorage.length)
        // Use options:.reverse to process attachments from end to beginning
        textStorage.enumerateAttribute(.attachment, in: fullRange) {
            value, range, _ in
            // Only process embedded outerdoc substrings represented by SubstringPlaceholderAttachment
            guard let ov = value as? SubstringPlaceholderAttachment,
                let fileType = ov.contentType,
                let wrapper = ov.fileWrapper,
                let data = wrapper.regularFileContents
            else {
                return
            }

            // Deserialize the outerdoc substring
            var substring: NSAttributedString?
            let ok = readOuterdocFromPackedBuffer(data, &substring)
            guard ok, let substring = substring else { return }

            // Replace attachment with attributed substring
            textStorage.replaceCharacters(in: range, with: substring)

            // Track this embedded substring for future serialization
            let substringRange = NSRange(location: range.location, length: substring.length)
            let substringId = UUID()
            
            // For now, we'll store empty arrays for elements and canvases since they're not in the packed buffer
            // They will be populated when the outerdoc is created
            let fragAttr = EmbeddedSubstringAttribute(
                contentType: fileType,
                attributedString: substring,
                substringId: substringId
            )
            textStorage.addAttribute(.embeddedSubstring, value: fragAttr, range: substringRange)

            DynamicSubstringManager.shared.register(substringId: substringId, fileType: fileType, textStorage: textStorage)

            // Create the outerdoc substring via the XPC connection
            OuterContentConnection.shared.createDynamicSubstring(substringId: substringId, contentType: fileType) { [weak self] runtimeData, iosurfacesByCanvas in
                guard let self else { return }

                if !runtimeData.attributedString.string.isEmpty {
                    // Create a mutable copy with our custom attribute
                    let mutableNewFragment = NSMutableAttributedString(attributedString: runtimeData.attributedString)

                    // Create and add the embedded substring attribute with the same ID
                    let updatedAttr = EmbeddedSubstringAttribute(
                        contentType: fileType,
                        attributedString: runtimeData.attributedString,
                        substringId: substringId
                    )
                    mutableNewFragment.addAttribute(.embeddedSubstring, value: updatedAttr,
                                                  range: NSRange(location: 0, length: mutableNewFragment.length))

                    // Replace the content at the substring range on the main thread
                    DispatchQueue.main.async {
                        self.textStorage.replaceCharacters(in: substringRange, with: mutableNewFragment)
                    }
                }
            }
        }
    }

    /// Processes existing SubstringPlaceholderAttachments for the given outerdoc types by batching a single XPC call.
    public func processPlaceholderAttachments() {
        let fullRange = NSRange(location: 0, length: textStorage.length)
        // Collect attachments and their file types
        var attachments: [(fileType: String, attachment: SubstringPlaceholderAttachment, substringId: UUID)] = []
        textStorage.enumerateAttribute(.attachment, in: fullRange, options: []) { value, _, _ in
            if let ov = value as? SubstringPlaceholderAttachment,
               !ov.beingReplaced,
               let ft = ov.contentType {
                ov.beingReplaced = true
                let substringId = UUID()
                attachments.append((fileType: ft, attachment: ov, substringId: substringId))
                DynamicSubstringManager.shared.register(substringId: substringId, fileType: ft, textStorage: self.textStorage)
            }
        }
        guard !attachments.isEmpty else { return }

        OuterContentConnection.shared.createDynamicSubstrings(substringIds: attachments.map { $0.substringId },
                                                    contentTypes: attachments.map { $0.fileType }) { runtimeOuterdocDatas, iosurfacesByDoc in
            // Apply replacements on the main thread
            DispatchQueue.main.async {
                self.textStorage.beginEditing()
                let full = NSRange(location: 0, length: self.textStorage.length)
                var processedAttachments = Set<SubstringPlaceholderAttachment>()
                
                self.textStorage.enumerateAttribute(
                    .attachment, in: full, options: []) { value, range, stop in
                        guard let ov = value as? SubstringPlaceholderAttachment,
                              let fileType = ov.contentType,
                              !processedAttachments.contains(ov),
                              let index = attachments.firstIndex(where: { $0.attachment === ov })
                        else {
                            return
                        }
                        
                        processedAttachments.insert(ov)
                        let runtimeData = runtimeOuterdocDatas[index]
                        let substringId = attachments[index].substringId
                        let iosurfaces = iosurfacesByDoc[index]

                        // Replace placeholder with substring
                        self.textStorage.replaceCharacters(in: range, with: runtimeData.attributedString)
                        let newRange = NSRange(
                            location: range.location, length: runtimeData.attributedString.length)

                        // Process element ranges
                        let elementArray = runtimeData.elements

                        // Process canvas information
                        if !runtimeData.canvases.isEmpty {

                            for (i, canvas) in runtimeData.canvases.enumerated() {
                                // Create a canvas attachment at the specified location
                                let canvasAttachment = CanvasAttachment(
                                    canvasID: canvas.canvasID,
                                    substringId: substringId,
                                    width: CGFloat(canvas.width),
                                    height: CGFloat(canvas.height),
                                    iosurface: iosurfaces[i]
                                )

                                GlobalCanvasRegistry.shared.registerCanvasMapping(canvasID: canvas.canvasID, substringId: substringId)

                                // Get existing attributes at the attachment location
                                let canvasLocation = newRange.location + Int(canvas.location)
                                let existingAttributes = self.textStorage.attributes(at: canvasLocation, effectiveRange: nil)
                                
                                // Create an attributed string with the attachment, preserving existing attributes
                                let canvasAttrString = NSMutableAttributedString(attachment: canvasAttachment)
                                canvasAttrString.addAttributes(existingAttributes, range: NSRange(location: 0, length: 1))

                                // Only insert if it's within our range
                                if canvasLocation <= newRange.location + newRange.length {
                                    print("DocumentModel: Inserting canvas placeholder at position \(canvasLocation)")
                                    self.textStorage.replaceCharacters(in: NSRange(location: canvasLocation, length: 1), with: canvasAttrString)
                                } else {
                                    print("DocumentModel: Canvas position \(canvasLocation) is outside valid range \(newRange.location) to \(newRange.location + newRange.length)")
                                }
                            }
                        }

                        for element in elementArray {
                            self.textStorage.addAttribute(
                                .element,
                                value: ElementAttribute(elementID: element.elementID),
                                range: NSRange(
                                    location: newRange.location + element.range.location,
                                    length: element.range.length
                                )
                            )
                        }

                        // Tag this range with embeddedSubstring attribute for serialization
                        let fragAttr = EmbeddedSubstringAttribute(
                            contentType: fileType,
                            attributedString: runtimeData.attributedString,
                            substringId: substringId
                        )
                        self.textStorage.addAttribute(
                            .embeddedSubstring, value: fragAttr, range: newRange)
                    }
                self.textStorage.endEditing()
                
            }
        }
    }

    deinit {
        cleanupDynamicSubstrings()
    }

    /// Cleans up all dynamic substrings associated with this document model's text storage.
    /// Call this when the document is being closed or deallocated.
    public func cleanupDynamicSubstrings() {
        // Clean up all dynamic substrings associated with this document's text storage
        DynamicSubstringManager.shared.cleanupSubstrings(for: textStorage)
    }
}

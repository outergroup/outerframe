//
//  SineWaveDemo.swift
//

import Foundation
import IOSurface
import OuterdocPlugin
import MetalKit

//import AppKit  // For NSScreen, but it causes an icon to appear in the dock


// Configuration constants
let N_WAVES = 24
let SHOW_TEXT_LABELS = false // Set to false for a cheaper version without NSAttributedString text labels
let SHOW_CANVASES = true // Set to false to show labels without canvases
let SHOW_LABELS_IN_CANVAS = true // Set to true to render a set of text labels directly in the canvas

/// Class to generate animated outerdoc fragments
class SineWaveSubstring: OuterdocDynamicSubstring {

    // Add a unique instance ID to identify each fragment instance
    private let instanceID = UUID()

    // Animation state - unique per fragment instance
    private var animationStartTime: Double = 0

    // Animation scheduler for requesting frames
    private weak var animationScheduler: AnimationScheduler?

    // Track element IDs by their type
    private var elementIds: [String: UUID]

    private var canvasIDs: [UUID]

    // Store the main fragment text ranges for elements
    private var elementRanges: [String: NSRange] = [:]

    // Cache for element contents to avoid unnecessary updates
    // We need to store the entire NSAttributedString for comparison
    // Use element ID as key to ensure uniqueness across fragments
    private var cachedElementContents: [UUID: NSAttributedString] = [:]

    // Keep track of the last time we updated each canvas
    private var lastCanvasUpdateTime: [UUID: TimeInterval] = [:]

    // Manage our own IOSurfaces per canvas for direct rendering
    private var canvasSurfaces: [IOSurface] = []           // Single IOSurface per canvas, indexed by canvas position

    // Device and command queue can be shared
    private static var sharedDevice: MTLDevice?
    private static var sharedCommandQueue: MTLCommandQueue?

    // These need to be per-instance
    private var device: MTLDevice? { return SineWaveSubstring.sharedDevice }
    private var commandQueue: MTLCommandQueue? { return SineWaveSubstring.sharedCommandQueue }
    private var pipelineState: MTLRenderPipelineState? // For progress bar
    private var textPipelineState: MTLRenderPipelineState? // For text rendering
    private var textureSamplerState: MTLSamplerState? // For text rendering

    // Cached resources for better performance - must be per-instance
    private var progressBarVertexBuffer: MTLBuffer?

    // We need separate resources per canvas to prevent interference
    private var progressBuffers: [MTLBuffer?] = []

    // Instance data buffer for SIMD rendering
    private var instanceDataBuffer: MTLBuffer?

    // Constants buffer for layout values
    private var constantsBuffer: MTLBuffer?

    // Texture array for parallel rendering
    private var textureArray: MTLTexture?

    // Text texture array for text rendering
    private var textTextureArray: MTLTexture?

    // Swift struct matching the Metal InstanceData structure
    struct InstanceData {
        var progress: Float
        var canvasIndex: UInt32
    }

    // Swift struct matching the Metal LayoutConstants structure
    struct LayoutConstants {
        var xPosMin: Float
        var xPosMax: Float
    }

    // Texture cache organized by canvas
    // Structure: [MTLTexture] indexed by [canvasIndex]
    private var textureCache: [MTLTexture] = []

    // Cached text rendering resources using arrays for faster lookup
    // Cache text textures by progress value index (0-10 for progress 0.0-1.0)
    private var textTextureCache: [MTLTexture] = []

    // Cached text vertex buffer (one per progress value)
    private var textVertexBufferCache: [MTLBuffer] = []

    // Single vertex buffer for all text instances
    private var textVertexBuffer: MTLBuffer?

    // Pre-computed formatted progress strings (0.00 to 1.00)
    static let progressStrings: [String] = {
        var strings = [String]()
        for i in 0...10 {
            let value = Double(i) / 10.0
            strings.append(String(format: "%.1f", value))
        }
        return strings
    }()

    // Reusable buffers for text rendering to avoid constant allocations
    private var textQuadVertexBuffer: MTLBuffer?
    private var textOffsetBuffer: MTLBuffer?
    private var textColorBuffer: MTLBuffer?

    // Default initializer using static IDs
    convenience init() {
        // Create IDs for all 12 sine waves
        var elementIds = [String: UUID]()

        for i in 0..<N_WAVES {
            elementIds["sine\(i)"] = UUID()
        }

        self.init(
            elementIds: elementIds,
        )
    }

    // Custom initializer that accepts element and canvas IDs
    init(elementIds: [String: UUID]) {
        self.elementIds = elementIds
        self.canvasIDs = Array(elementIds.keys).map { _ in UUID() }
        self.animationStartTime = CACurrentMediaTime()

        print("SineWaveDemoGenerator initialized with element IDs: \(elementIds) and canvas IDs: \(canvasIDs)")

        // Use predefined element ranges from the container
        let fragmentContainer = generateFragmentContainer()
        self.elementRanges = fragmentContainer.elementRanges

        // Set up Metal for canvas rendering
        setupMetal()
    }

    func getInitialSubstring() -> OuterdocSubstring {
        print("Generating initial Outerdoc fragment data")

        // Create the main container with proper content (not placeholders)
        let fragmentContainer = generateFragmentContainer()
        let container = fragmentContainer.attributedString
        print("Container string: \"\(container.string)\"")

        var elements: [ElementInfo] = []
        var canvases: [CanvasInfo] = []

        // Add all sine elements and canvases
        for i in 0..<N_WAVES {
            // Get specific IDs for this sine wave
            let sineID = elementIds["sine\(i)"]!
            let progressCanvasID = canvasIDs[i]
            print("Element ID: sine\(i)=\(sineID), Canvas ID: progress\(i)=\(progressCanvasID)")

            // Only add element info if we're showing text labels
            if SHOW_TEXT_LABELS {
                // Use the predefined element range from our container
                if let sineRange = fragmentContainer.elementRanges["sine\(i)"] {
                    // Create element info but DON'T add .element attribute
                    let elementInfo = ElementInfo()
                    elementInfo.elementID = sineID
                    elementInfo.range = sineRange
                    elements.append(elementInfo)
                }
            }

            // Only add canvas if we're showing canvases
            if SHOW_CANVASES {
                // Create a canvas for the progress bar visualization
                let canvasInfo = CanvasInfo()
                canvasInfo.canvasID = progressCanvasID

                // Find the attachment character (\uFFFC) in the string at the correct index for this line
                let attachmentStr = "\u{FFFC}"
                let containerStr = container.string

                if SHOW_TEXT_LABELS {
                    // When showing text, we need to look for the attachment character in the format: "Sine i: [attachment] value"
                    let linePrefix = ""// "Sine \(i): "

                    // Look for the i-th occurrence of the attachment character
                    var searchStartIndex = containerStr.startIndex
                    var currentIndex = 0
                    var found = false

                    while currentIndex <= i, let range = containerStr.range(of: attachmentStr, range: searchStartIndex..<containerStr.endIndex) {
                        if currentIndex == i {
                            // Convert String.Index to Int offset for NSRange
                            let location = containerStr.distance(from: containerStr.startIndex, to: range.lowerBound)
                            canvasInfo.location = Int32(location)
                            found = true
                            break
                        }
                        searchStartIndex = range.upperBound
                        currentIndex += 1
                    }

                    if !found {
                        // Fallback if attachment character not found (shouldn't happen)
                        // Calculate an approximate position based on line number
                        let lineLength = linePrefix.count + attachmentStr.count + 6 // account for value text and spacing
                        canvasInfo.location = Int32(i * (lineLength + 1) + linePrefix.count) // +1 for newline
                    }
                } else {
                    // In text-free mode, each line is simply a single attachment character + newline
                    // So we can easily calculate the position
                    canvasInfo.location = Int32(i * 2); // Each line is the attachment char (1) + newline (1)
                }

                // Make the progress bar smaller to fit nicely on the same line
                // If we're showing labels in canvas, add extra width for the text
                canvasInfo.width = SHOW_LABELS_IN_CANVAS ? 270 : 200
                canvasInfo.height = 20
                canvases.append(canvasInfo)
            }
        }

        print("Returning OuterdocSubstring with \(elements.count) elements and \(canvases.count) canvases")
        return OuterdocSubstring(
            attributedString: container,
            elements: elements,
            canvases: canvases,
            canvasSurfaces: canvasSurfaces
        )
    }

    // Set up Metal for canvas rendering
    private func setupMetal() {
        // Initialize shared device and command queue if needed
        if SineWaveSubstring.sharedDevice == nil {
            guard let device = MTLCreateSystemDefaultDevice() else {
                print("Metal is not supported on this device")
                return
            }
            SineWaveSubstring.sharedDevice = device

            // Create command queue
            guard let commandQueue = device.makeCommandQueue() else {
                print("Failed to create Metal command queue")
                return
            }
            SineWaveSubstring.sharedCommandQueue = commandQueue
        }

        // We need a device to continue
        guard let device = self.device else {
            print("No Metal device available")
            return
        }

        // Initialize reusable vertex buffer for progress bar
        // Keep progress bar at 200 pixels equivalent in normalized coordinates
        // The canvas is 270 pixels wide when SHOW_LABELS_IN_CANVAS is true

        let backingScaleFactor: CGFloat = 2.0 // NSScreen.main?.backingScaleFactor ?? 1.0

        let canvasWidth = Int(backingScaleFactor * (SHOW_LABELS_IN_CANVAS ? 270 : 200))
        let canvasHeight = Int(backingScaleFactor * 20)
        let progressBarWidth: Float = Float(SHOW_LABELS_IN_CANVAS ? (0.95 * backingScaleFactor * 200.0 / Double(canvasWidth)) : 0.95)
        let vertices: [SIMD2<Float>] = [
            SIMD2<Float>(-0.95, -0.5),           // Bottom left
            SIMD2<Float>(progressBarWidth, -0.5), // Bottom right - adjusted to leave room for text
            SIMD2<Float>(-0.95, 0.5),            // Top left
            SIMD2<Float>(progressBarWidth, 0.5)   // Top right - adjusted to leave room for text
        ]
        progressBarVertexBuffer = device.makeBuffer(bytes: vertices,
                                                   length: vertices.count * MemoryLayout<SIMD2<Float>>.stride,
                                                   options: .storageModeShared)

        // Create instance data buffer for all canvases
        instanceDataBuffer = device.makeBuffer(
            length: canvasIDs.count * MemoryLayout<InstanceData>.stride,
            options: .storageModeShared)

        // Create constants buffer with layout values
        let layoutConstants = LayoutConstants(
            xPosMin: -0.95,
            xPosMax: Float(SHOW_LABELS_IN_CANVAS ? (0.95 * backingScaleFactor * 200.0 / Double(canvasWidth)) : 0.95)
        )
        constantsBuffer = device.makeBuffer(bytes: [layoutConstants],
                                           length: MemoryLayout<LayoutConstants>.stride,
                                           options: .storageModeShared)

        // Create texture array for parallel rendering
        textureArray = device.makeTexture(descriptor: {
            let descriptor = MTLTextureDescriptor()
            descriptor.textureType = .type2DArray
            descriptor.pixelFormat = .bgra8Unorm
            descriptor.width = canvasWidth
            descriptor.height = canvasHeight
            descriptor.arrayLength = canvasIDs.count // One slice per canvas
            descriptor.usage = [.renderTarget, .shaderRead]
            descriptor.storageMode = .private
            return descriptor
        }())
        guard textureArray != nil else {
            print("Failed to create texture array")
            return
        }
        print("Created texture array: \(canvasWidth)x\(canvasHeight) with \(canvasIDs.count) slices")

        // Initialize per-canvas arrays
        let canvasCount = canvasIDs.count
        canvasSurfaces.reserveCapacity(canvasCount)
        textureCache.reserveCapacity(canvasCount)
        progressBuffers = Array(repeating: nil, count: canvasCount)

        // Initialize per-canvas resources
        for index in 0..<canvasCount {
            // Create a separate progress buffer for each canvas
            let progressBuffer = device.makeBuffer(length: MemoryLayout<Float>.size,
                                                  options: .storageModeShared)
            progressBuffers[index] = progressBuffer

            var bytesPerRow: Int = canvasWidth * 4
            // 16-byte aligned
            bytesPerRow = ((bytesPerRow + 15)) / 16 * 16

            // These properties must match what CAMetalLayer would create
            let iosurfaceProperties: [IOSurfacePropertyKey: Any] = [
                .width: canvasWidth,
                .height: canvasHeight,
                .pixelFormat: UInt32(kCVPixelFormatType_32BGRA),
                .bytesPerElement: 4,
                .bytesPerRow: bytesPerRow,
                .allocSize: bytesPerRow * canvasHeight * 4
            ]
            let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: canvasWidth,
                height: canvasHeight,
                mipmapped: false
            )
            textureDescriptor.usage = [.renderTarget, .shaderRead]
            textureDescriptor.storageMode = .shared

            // Create single surface and corresponding texture
            guard let surface = IOSurface(properties: iosurfaceProperties) else {
                print("Failed to create IOSurface")
                continue
            }
            NSLog("Created IOSurface: \(String(describing: IOSurfaceGetID(surface)))")

            guard let texture = device.makeTexture(
                descriptor: textureDescriptor,
                iosurface: surface,
                plane: 0) else {
                print("Failed to create texture for IOSurface in canvas \(index)")
                continue
            }

            // Store surface and texture
            canvasSurfaces.append(surface)
            textureCache.append(texture)

            print("Created IOSurface and texture for canvas \(index)")
        }

        // Create text texture array
        let textScale = 4.0
        let textWidth = Int(60 * textScale)
        let textHeight = Int(20 * textScale)

        let textTextureDescriptor = MTLTextureDescriptor()
        textTextureDescriptor.textureType = .type2DArray
        textTextureDescriptor.pixelFormat = .rgba8Unorm
        textTextureDescriptor.width = textWidth
        textTextureDescriptor.height = textHeight
        textTextureDescriptor.arrayLength = 11 // 0.0 to 1.0 in 0.1 increments
        textTextureDescriptor.usage = [.shaderRead]
        textTextureDescriptor.storageMode = .private

        textTextureArray = device.makeTexture(descriptor: textTextureDescriptor)
        guard textTextureArray != nil else {
            print("Failed to create text texture array")
            return
        }

        // Initialize text textures
        textTextureCache.reserveCapacity(11)
        textVertexBufferCache.reserveCapacity(11)
        for progressIndex in 0..<11 {
            let progressText = SineWaveSubstring.progressStrings[progressIndex]

            let textColor = NSColor.white
            let scale = 4.0
            let textWidth = Int(60 * scale)
            let textHeight = Int(20 * scale)
            let bytesPerRow = textWidth * 4

            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)

            guard let context = CGContext(data: nil,
                                          width: textWidth,
                                          height: textHeight,
                                          bitsPerComponent: 8,
                                          bytesPerRow: bytesPerRow,
                                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: bitmapInfo.rawValue) else {
                return
            }

            // Configure context
            context.setShouldAntialias(true)
            context.setAllowsAntialiasing(true)
            context.setShouldSmoothFonts(true)
            context.clear(CGRect(x: 0, y: 0, width: textWidth, height: textHeight))
            context.translateBy(x: 0, y: CGFloat(textHeight))
            context.scaleBy(x: 1.0, y: -1.0)

            // Create and draw text
            let fontSize = 20.0 * scale
            let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .semibold)
            let textAttributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: textColor,
            ]

            let attributedString = NSAttributedString(string: progressText, attributes: textAttributes)
            let line = CTLineCreateWithAttributedString(attributedString)

            var ascent: CGFloat = 0
            var descent: CGFloat = 0
            var leading: CGFloat = 0
            let lineWidth = CTLineGetTypographicBounds(line, &ascent, &descent, &leading)

            // This yPosition logic only works if fontSize == textHeight. I should find more robust math.
            let yPosition = descent
            let xPosition = (CGFloat(textWidth) - lineWidth) / 2.0

            context.textPosition = CGPoint(x: xPosition, y: yPosition)
            CTLineDraw(line, context)

            guard let image = context.makeImage() else { return }

            // Create Metal texture
            let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm,
                width: textWidth,
                height: textHeight,
                mipmapped: false
            )
            textureDescriptor.usage = [.shaderRead]

            guard let newTexture = device.makeTexture(descriptor: textureDescriptor),
                  let dataProvider = image.dataProvider,
                  let data = dataProvider.data,
                  let bytes = CFDataGetBytePtr(data) else { return }

            let region = MTLRegionMake2D(0, 0, textWidth, textHeight)
            newTexture.replace(region: region, mipmapLevel: 0, withBytes: bytes, bytesPerRow: bytesPerRow)

            textTextureCache.append(newTexture)

            // Copy to texture array slice
            if let commandBuffer = SineWaveSubstring.sharedCommandQueue?.makeCommandBuffer(),
               let blitEncoder = commandBuffer.makeBlitCommandEncoder(),
               let textArray = textTextureArray {
                blitEncoder.copy(from: newTexture,
                               sourceSlice: 0,
                               sourceLevel: 0,
                               sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                               sourceSize: MTLSize(width: textWidth, height: textHeight, depth: 1),
                               to: textArray,
                               destinationSlice: progressIndex,
                               destinationLevel: 0,
                               destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
                blitEncoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            }

            // Create vertex buffer with proper aspect ratio calculations

            struct TextVertex {
                var position: SIMD2<Float>
                var texCoord: SIMD2<Float>
            }

            // Calculate proper aspect ratios for text positioning
            let canvasAspectRatio = Float(canvasWidth / canvasHeight)  // 13.5 when showing labels
            let textAspectRatio = Float(textWidth) / Float(textHeight) // 3.5 at 2x scale

            // Calculate text dimensions in normalized device coordinates
            // The text should maintain its aspect ratio within the canvas
            let textWidthNDC: Float
            let textHeightNDC: Float

            // Determine available space for text (right side of canvas after progress bar)
            let availableWidthNDC = 1.0 - (progressBarWidth + 0.03)
            let availableHeightNDC: Float = 1.0 // Full height of canvas in NDC

            // Scale text to fit available space while maintaining aspect ratio
            let scaleByWidth = availableWidthNDC / (textAspectRatio / canvasAspectRatio)
            let scaleByHeight = availableHeightNDC
            let scale2 = min(scaleByWidth, scaleByHeight)

            textWidthNDC = scale2 * (textAspectRatio / canvasAspectRatio)
            textHeightNDC = scale2

            // Position text on the right side of the canvas, vertically centered
            let textLeft = progressBarWidth + 0.03
            let textRight = textLeft + textWidthNDC
            let textBottom = -textHeightNDC / 2.0
            let textTop = textHeightNDC / 2.0

            // Create vertices with proper aspect ratio positioning
            let textVertices: [TextVertex] = [
                TextVertex(position: SIMD2<Float>(textLeft, textBottom), texCoord: SIMD2<Float>(0, 0)),   // Bottom left
                TextVertex(position: SIMD2<Float>(textRight, textBottom), texCoord: SIMD2<Float>(1, 0)),  // Bottom right
                TextVertex(position: SIMD2<Float>(textLeft, textTop), texCoord: SIMD2<Float>(0, 1)),      // Top left
                TextVertex(position: SIMD2<Float>(textRight, textTop), texCoord: SIMD2<Float>(1, 1))      // Top right
            ]

            guard let newVertexBuffer = device.makeBuffer(
                bytes: textVertices,
                length: textVertices.count * MemoryLayout<TextVertex>.stride,
                options: .storageModeShared
            ) else {
                print("Failed to create vertex buffer.")
                return
            }

            textVertexBufferCache.append(newVertexBuffer)
        }

        // Create a single text vertex buffer for all instances (using the first progress index layout)
        if !textVertexBufferCache.isEmpty {
            textVertexBuffer = textVertexBufferCache[0]
        }

        // Load Metal library from bundle (same approach as ExampleFigure)
        let currentBundle = Bundle(for: type(of: self))
        var library: MTLLibrary?
        do {
            library = try device.makeDefaultLibrary(bundle: currentBundle)
            print("Successfully loaded Metal library from class bundle: \(currentBundle.bundlePath)")
        } catch {
            print("Failed to create Metal library: \(error)")
            return
        }

        // Get shader functions
        guard let library = library,
              let vertexFunction = library.makeFunction(name: "canvas_vertex_main"),
              let fragmentFunction = library.makeFunction(name: "canvas_fragment_main"),
              let textVertexFunction = library.makeFunction(name: "canvas_text_vertex_main"),
              let textFragmentFunction = library.makeFunction(name: "canvas_text_fragment_main") else {
            print("Failed to create Metal shader functions")
            return
        }

        // Create pipeline for progress bar
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction

        // Set input primitive topology for render target array support
        pipelineDescriptor.inputPrimitiveTopology = .triangle

        // Define vertex format for progress bar
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        pipelineDescriptor.vertexDescriptor = vertexDescriptor

        // Configure color attachment
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        // Enable blending for text rendering
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        // Create pipeline for text rendering
        let textPipelineDescriptor = MTLRenderPipelineDescriptor()
        textPipelineDescriptor.vertexFunction = textVertexFunction
        textPipelineDescriptor.fragmentFunction = textFragmentFunction

        // Set input primitive topology for render target array support
        textPipelineDescriptor.inputPrimitiveTopology = .triangle

        // Define vertex format for text rendering
        let textVertexDescriptor = MTLVertexDescriptor()
        // Position attribute
        textVertexDescriptor.attributes[0].format = .float2
        textVertexDescriptor.attributes[0].offset = 0
        textVertexDescriptor.attributes[0].bufferIndex = 0
        // Texture coordinate attribute
        textVertexDescriptor.attributes[1].format = .float2
        textVertexDescriptor.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        textVertexDescriptor.attributes[1].bufferIndex = 0
        // Layout
        textVertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride * 2
        textPipelineDescriptor.vertexDescriptor = textVertexDescriptor

        // Use same color attachment configuration (including blending)
        textPipelineDescriptor.colorAttachments[0] = pipelineDescriptor.colorAttachments[0]

        // Create render pipeline states
        do {
            print("Creating main pipeline state...")
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            print("Main pipeline state created successfully")

            print("Creating text pipeline state...")
            textPipelineState = try device.makeRenderPipelineState(descriptor: textPipelineDescriptor)
            print("Text pipeline state created successfully")

            // Create a sampler state for texture rendering
            let samplerDescriptor = MTLSamplerDescriptor()
            samplerDescriptor.minFilter = .linear
            samplerDescriptor.magFilter = .linear
            textureSamplerState = device.makeSamplerState(descriptor: samplerDescriptor)

            print("Created pipeline states for progress bar and text rendering")
        } catch {
            print("Failed to create render pipeline states: \(error)")
            // Print detailed error information
            if let metalError = error as NSError? {
                print("Error domain: \(metalError.domain)")
                print("Error code: \(metalError.code)")
                print("Error description: \(metalError.localizedDescription)")
                for (key, value) in metalError.userInfo {
                    print("  \(key): \(value)")
                }
            }
        }
    }

    func generateNextFrame(targetTimestamp: CFTimeInterval) -> OuterdocElementUpdates? {
        var updates: [UUID: NSAttributedString] = [:]
        if SHOW_TEXT_LABELS {
            // Update each element type (only if text labels are enabled)
            for (elementType, elementID) in elementIds {
                // Check if this is a sine element (starts with "sine")
                if elementType.hasPrefix("sine") {
                    // Extract the index from the element type if present (format: "sineN")
                    var phaseOffset = 0.0
                    let indexString = elementType.dropFirst("sine".count) // Remove "sine" prefix
                    if let index = Int(indexString) {
                        // Calculate phase offset based on index (30 degree increments)
                        phaseOffset = Double(index) * (Double.pi / 6.0)
                    }

                    // Generate the element content with the appropriate phase offset
                    let content = generateElementContent(elementType: "sine", targetTimestamp: targetTimestamp, startTime: animationStartTime, phaseOffset: phaseOffset)

                    // Check if the content or attributes have changed. (Text-based updates are expensive)
                    let cached = cachedElementContents[elementID]
                    let hasChanged = needsUpdate(newContent: content, cachedContent: cached)

                    if hasChanged {
                        cachedElementContents[elementID] = content
                        updates[elementID] = content
                    }
                }
            }
        }

        var ret: OuterdocElementUpdates? = nil
        if !updates.isEmpty {
            var elementIds = [UUID]()
            var newStrings = [NSAttributedString]()
            for (elementId, newString) in updates {
                elementIds.append(elementId)
                newStrings.append(newString)
            }
            ret = OuterdocElementUpdates(elementIds: elementIds, newStrings: newStrings)
        }

        // Request the next animation frame if any visual elements are enabled
        if SHOW_TEXT_LABELS || SHOW_CANVASES {
            animationScheduler?.requestAnimationFrame()
        }

        // Return the element IDs and their corresponding serialized content
        return ret
    }


    func updateCanvases(targetTimestamp: CFTimeInterval) {
        // If canvases are disabled, return early
        if !SHOW_CANVASES {
            return
        }

        guard let commandQueue = commandQueue,
              let pipelineState = pipelineState,
              let vertexBuffer = progressBarVertexBuffer,
              let instanceBuffer = instanceDataBuffer,
              let constantsBuffer = constantsBuffer,
              let textureArray = textureArray else {
            print("Metal not properly initialized for SIMD canvas rendering")
            return
        }

        // Prepare instance data
        var instanceData: [InstanceData] = []
        instanceData.reserveCapacity(canvasIDs.count)

        let elapsedTime = targetTimestamp - animationStartTime
        let frequency = 0.5 // Hz

        for index in 0..<canvasIDs.count {
            // Calculate progress for this canvas
            let phaseOffset = Double(index) * (Double.pi / 6.0)
            let sineValue = (sin(elapsedTime * 2.0 * .pi * frequency + phaseOffset) + 1.0) / 2.0

            // Add instance data
            instanceData.append(InstanceData(
                progress: Float(sineValue),
                canvasIndex: UInt32(index) // Use loop index directly for texture array slice
            ))
        }

        // Update instance data buffer
        let contents = instanceBuffer.contents()
        memcpy(contents, instanceData, instanceData.count * MemoryLayout<InstanceData>.stride)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Failed to create command buffer for SIMD rendering")
            return
        }

        // STEP 1: Render all canvases to texture array in parallel
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = textureArray
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        renderPassDescriptor.renderTargetArrayLength = textureArray.arrayLength

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            print("Failed to create render encoder for texture array rendering")
            return
        }

        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(instanceBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(constantsBuffer, offset: 0, index: 2)

        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: instanceData.count)

        if SHOW_LABELS_IN_CANVAS {
            guard let textPipeline = textPipelineState,
                  let samplerState = textureSamplerState,
                  let textVertexBuf = textVertexBuffer,
                  let textTexArray = textTextureArray else {
                print("Missing text pipeline state for texture array rendering")
                return
            }

            renderEncoder.setRenderPipelineState(textPipeline)
            renderEncoder.setFragmentSamplerState(samplerState, index: 0)
            renderEncoder.setVertexBuffer(textVertexBuf, offset: 0, index: 0)
            // renderEncoder.setVertexBuffer(instanceBuffer, offset: 0, index: 1)  // Already set above
            renderEncoder.setFragmentTexture(textTexArray, index: 0)

            // Render all text instances in a single draw call
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: instanceData.count)
        }

        renderEncoder.endEncoding()

        // STEP 2: Copy texture array slices to individual IOSurfaces

        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            print("Failed to create blit command encoder")
            return
        }

        for index in 0..<canvasIDs.count {
            // Copy from texture array slice to IOSurface texture
            let destinationTexture = textureCache[index]
            blitEncoder.copy(from: textureArray,
                           sourceSlice: index,
                           sourceLevel: 0,
                           sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                           sourceSize: MTLSize(width: textureArray.width,
                                               height: textureArray.height,
                                               depth: 1),
                           to: destinationTexture,
                           destinationSlice: 0,
                           destinationLevel: 0,
                           destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        }

        blitEncoder.endEncoding()

        commandBuffer.commit()
    }

    func setAnimationScheduler(scheduler: AnimationScheduler?) {
        self.animationScheduler = scheduler

        // Immediately request first animation frame
        // This will be processed on the next display link cycle
        scheduler?.requestAnimationFrame()
    }
}


/// Generates an attributed string based on animation parameters
/// - Parameters:
///   - startTime: Animation start time
///   - frameCount: Current frame count
/// - Returns: Attributed string with animation data
/// Structure to hold both the container NSAttributedString and element ranges
fileprivate struct FragmentContainer {
    var attributedString: NSAttributedString
    var elementRanges: [String: NSRange]
}

/// Generate the main fragment container with valid values instead of placeholders
/// - Parameter targetTimestamp: The intended presentation time for this frame
/// - Returns: A struct containing the attributed string and predefined element ranges
fileprivate func generateFragmentContainer(targetTimestamp: CFTimeInterval = CACurrentMediaTime()) -> FragmentContainer {
    let startTime = targetTimestamp

    // Create a mutable attributed string to build our content
    let monospacedFont = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
    let attributedString = NSMutableAttributedString()
    var elementRanges = [String: NSRange]()

    // Create N sine waves, each on its own line with a different phase
    for i in 0..<N_WAVES {
        // Calculate a different phase for each line
        let phaseOffset = Double(i) * (Double.pi / 6.0) // 30 degree increments

        // Line content depends on whether we're showing text labels
        if SHOW_TEXT_LABELS {
            // Generate the initial sine wave content with its own phase
            let sineContent = generateElementContent(elementType: "sine", targetTimestamp: targetTimestamp, startTime: startTime, phaseOffset: phaseOffset)
            let sineText = sineContent.string

            // Create the line text with actual values and layout for canvas
            let labelText = "Sine \(i): "

            // Format: "Sine i: [progress bar] 0.80" with the progress bar as an attachment character
            // \u{FFFC} is the Object Replacement Character that will be replaced with our canvas
            // We still include the attachment character even when canvases are disabled
            var lineText: String
            var sineValueLocation: Int
            if SHOW_CANVASES {
                lineText = labelText + "\u{FFFC} " + sineText
                sineValueLocation = labelText.count + 2 // +1 for attachment char, +1 for space after it
            } else {
                lineText = labelText + sineText
                sineValueLocation = labelText.count
            }

            // Create attributed string for this line
            let lineString = NSMutableAttributedString(string: lineText)

            let sineRange = NSRange(location: sineValueLocation, length: sineText.count)

            // Apply sine element attributes to just the value part
            lineString.addAttributes(sineContent.attributes(at: 0, effectiveRange: nil), range: sineRange)

            // Get the current position in the main attributed string
            let startPosition = attributedString.length

            // Append this line to the main string
            attributedString.append(lineString)

            // Always add a newline after each line
            attributedString.append(NSAttributedString(string: "\n"))

            // Adjust the range to account for position in the full string
            let adjustedSineRange = NSRange(location: startPosition + sineValueLocation, length: sineText.count)
            elementRanges["sine\(i)"] = adjustedSineRange
        } else {
            // Simplified version without text labels - just progress bars

            // Just use the attachment character for the canvas
            let attachmentChar = "\u{FFFC}"

            // Set up line attributes
            let lineAttributes: [NSAttributedString.Key: Any] = [
                .font: monospacedFont,
                .foregroundColor: NSColor.labelColor
            ]

            // Create attributed string for this line - just the canvas
            let lineString = NSMutableAttributedString(
                string: attachmentChar,
                attributes: lineAttributes
            )

            // Append this line to the main string
            attributedString.append(lineString)

            // Always add a newline after each line
            attributedString.append(NSAttributedString(string: "\n"))
        }
    }

    let paragraphAttributes: [NSAttributedString.Key: Any] = [
        .font: monospacedFont,
        .foregroundColor: NSColor.labelColor
    ]
    attributedString.addAttributes(paragraphAttributes, range: NSMakeRange(0, attributedString.length))

    return FragmentContainer(attributedString: attributedString, elementRanges: elementRanges)
}

/// Generate animated element content for a specific element
/// - Parameters:
///   - elementType: The type of element to generate (frame, time, or sine)
///   - targetTimestamp: The intended presentation time for this frame
///   - startTime: Animation start time
///   - phaseOffset: Optional phase offset for the sine wave (to create multiple different waves)
/// - Returns: Attributed string for the element
fileprivate func generateElementContent(elementType: String, targetTimestamp: CFTimeInterval = CACurrentMediaTime(), startTime: Double, phaseOffset: Double = 0.0) -> NSAttributedString {
    // Calculate elapsed time using the provided timestamp
    let elapsedTime = targetTimestamp - startTime

    // Calculate sine wave value (0.0 to 1.0)
    let frequency = 0.5 // Hz
    let sineValue = (sin(elapsedTime * 2.0 * .pi * frequency + phaseOffset) + 1.0) / 2.0

    // Get pre-computed formatted string using the progress value as an index
    let progressIndex = min(10, max(0, Int(round(sineValue * 10.0))))
    let sineValueFormatted = SineWaveSubstring.progressStrings[progressIndex]

    // Generate content based on element type - keep the same width as placeholders
    // to avoid messing up the layout
    var text = ""

    // Ensure we use a fixed-width font for proper alignment of the progress bar
    let monospacedFont = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
    let attributes: [NSAttributedString.Key: Any] = [
        .font: monospacedFont
    ]

    switch elementType {
    case "sine":
        // Show the numeric value with a progress bar
        text = sineValueFormatted
    default:
        text = "Unknown"
    }

    return NSAttributedString(string: text, attributes: attributes)
}

/// Helper function to determine if an attributed string needs to be updated
/// - Parameters:
///   - newContent: The newly generated content
///   - cachedContent: The previously cached content (if any)
/// - Returns: True if the content needs to be updated
fileprivate func needsUpdate(newContent: NSAttributedString, cachedContent: NSAttributedString?) -> Bool {
    // If there's no cached content, we definitely need to update
    guard let cached = cachedContent else {
        return true
    }

    // If the string content is different, we need to update
    if cached.string != newContent.string {
        return true
    }

    // If the string length is the same, we need to check attributes
    let length = newContent.length
    if cached.length != length {
        return true
    }

    // Check attributes throughout the string
    var cachedAttrs: [NSAttributedString.Key: Any]?
    var newAttrs: [NSAttributedString.Key: Any]?
    var cachedRange = NSRange()
    var newRange = NSRange()

    // Start at index 0
    var index = 0

    // Check attributes for each run of attributes in the string
    while index < length {
        // Get attributes at this index
        cached.attributes(at: index, effectiveRange: &cachedRange)
        newContent.attributes(at: index, effectiveRange: &newRange)

        // Get the attributes
        cachedAttrs = cached.attributes(at: index, effectiveRange: nil)
        newAttrs = newContent.attributes(at: index, effectiveRange: nil)

        // Compare attribute dictionaries (simplified comparison)
        // Check font and foreground color which are the main attributes we use
        if let cachedFont = cachedAttrs?[.font] as? NSFont,
           let newFont = newAttrs?[.font] as? NSFont,
           cachedFont != newFont {
            return true
        }

        if let cachedColor = cachedAttrs?[.foregroundColor] as? NSColor,
           let newColor = newAttrs?[.foregroundColor] as? NSColor,
           !cachedColor.isClose(to: newColor) {
            return true
        }

        // Move past this attribute run
        index = cachedRange.upperBound
    }

    // If we get here, the attributed strings are effectively the same
    return false
}

/// Extension to help compare colors with a tolerance
extension NSColor {
    func isClose(to other: NSColor, tolerance: CGFloat = 0.01) -> Bool {
        // Convert both colors to the same color space for comparison
        guard let selfRGB = self.usingColorSpace(.sRGB),
              let otherRGB = other.usingColorSpace(.sRGB) else {
            return false
        }

        // Compare RGBA components
        return abs(selfRGB.redComponent - otherRGB.redComponent) <= tolerance &&
               abs(selfRGB.greenComponent - otherRGB.greenComponent) <= tolerance &&
               abs(selfRGB.blueComponent - otherRGB.blueComponent) <= tolerance &&
               abs(selfRGB.alphaComponent - otherRGB.alphaComponent) <= tolerance
    }
}

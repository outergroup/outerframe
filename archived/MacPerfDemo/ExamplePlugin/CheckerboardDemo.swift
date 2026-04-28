//
//  CheckerboardDemo.swift
//

import Foundation
import IOSurface
import OuterdocPlugin
import MetalKit


class CheckerboardDemo: OuterdocDynamicSubstring {
    
    private let instanceID = UUID()
    private let canvasID = UUID()
    
    // Animation state
    private var animationStartTime: Double = 0
    private weak var animationScheduler: AnimationScheduler?
    
    // Canvas properties
    private var canvasSurface: IOSurface?
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var textureCache: MTLTexture?
    
    // Metal rendering resources
    private var pipelineState: MTLRenderPipelineState?
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    private var borderBuffer: MTLBuffer?
    private var colorUniformBuffer: MTLBuffer?
    private var animationUniformBuffer: MTLBuffer?
    private var borderUniformBuffer: MTLBuffer?
    
    // Animation parameters
    private let animationSpeedX: Float = -2.5 // Cells per cycle
    private let animationSpeedY: Float = -2.5 // Cells per cycle
    private let targetCellSize: Float = 40.0
    
    // Store grid dimensions
    private var gridWidth: Int = 0
    private var gridHeight: Int = 0
    
    // Canvas dimensions
    private let canvasWidth = 400
    private let canvasHeight = 300
    
    init() {
        self.animationStartTime = CACurrentMediaTime()
        setupMetal()
    }
    
    func getInitialSubstring() -> OuterdocSubstring {
        // Create a simple fragment with just an attachment character and newline
        let monospacedFont = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        let attributedString = NSAttributedString(
            string: "\u{FFFC}\n",
            attributes: [
                .font: monospacedFont,
                .foregroundColor: NSColor.labelColor
            ]
        )
        
        // Create canvas info
        var canvases: [CanvasInfo] = []
        let canvasInfo = CanvasInfo()
        canvasInfo.canvasID = canvasID
        canvasInfo.location = 0 // Position of the attachment character
        canvasInfo.width = Int32(canvasWidth)
        canvasInfo.height = Int32(canvasHeight)
        canvases.append(canvasInfo)
        
        // Get the IOSurfaces
        var surfaces: [IOSurface] = []
        if let surface = canvasSurface {
            surfaces.append(surface)
        }

        return OuterdocSubstring(
            attributedString: attributedString,
            elements: [],
            canvases: canvases,
            canvasSurfaces: surfaces
        )
    }
    
    func generateNextFrame(targetTimestamp: CFTimeInterval) -> OuterdocElementUpdates? {
        // Request next animation frame
        animationScheduler?.requestAnimationFrame()
        
        // No element updates needed
        return nil
    }
    
    func updateCanvases(targetTimestamp: CFTimeInterval) {
        guard let commandQueue = commandQueue,
              let pipelineState = pipelineState,
              let vertexBuffer = vertexBuffer,
              let indexBuffer = indexBuffer,
              let borderBuffer = borderBuffer,
              let texture = textureCache else {
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 1.0, green: 0.0, blue: 1.0, alpha: 1.0) // Magenta background
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Calculate animation
        let elapsedTimeSeconds = Float(targetTimestamp - animationStartTime)
        
        // Calculate cell sizes in normalized device coordinates
        let cellSizeX = 2.0 / Float(gridWidth - 2)
        let cellSizeY = 2.0 / Float(gridHeight - 2)
        
        // Calculate total position in cell units
        let totalPhaseX = elapsedTimeSeconds * animationSpeedX
        let totalPhaseY = elapsedTimeSeconds * animationSpeedY
        
        // Get fractional part for smooth animation within each cell
        let fractX = totalPhaseX - floor(totalPhaseX)
        let fractY = totalPhaseY - floor(totalPhaseY)
        
        // Convert to NDC space offsets (scaled to cell size)
        let offsetX = -fractX * cellSizeX  // Negative for rightward motion
        let offsetY = -fractY * cellSizeY  // Negative for upward motion
        
        // Animation uniform structure
        struct AnimationUniform {
            var offset: SIMD2<Float>
        }
        
        // Color uniform structure
        struct ColorUniform {
            var color: SIMD4<Float>
        }
        
        // DRAW CHECKERBOARD
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        
        // Set yellow color for checkerboard squares
        var yellowColor = ColorUniform(color: SIMD4<Float>(1.0, 1.0, 0.0, 1.0))
        
        // Update color uniform buffer
        if let buffer = colorUniformBuffer {
            let contents = buffer.contents()
            memcpy(contents, &yellowColor, MemoryLayout<ColorUniform>.size)
            renderEncoder.setVertexBuffer(buffer, offset: 0, index: 1)
        }
        
        // Create animation uniform
        var animUniform = AnimationUniform(offset: SIMD2<Float>(offsetX, offsetY))
        
        // Update animation uniform buffer
        if let buffer = animationUniformBuffer {
            let contents = buffer.contents()
            memcpy(contents, &animUniform, MemoryLayout<AnimationUniform>.size)
            renderEncoder.setVertexBuffer(buffer, offset: 0, index: 2)
        }
        
        // Draw checkerboard
        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexBuffer.length / MemoryLayout<UInt16>.stride,
            indexType: .uint16,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )
        
        // DRAW BORDER
        renderEncoder.setVertexBuffer(borderBuffer, offset: 0, index: 0)
        
        // Set white color for border
        var whiteColor = ColorUniform(color: SIMD4<Float>(1.0, 1.0, 1.0, 1.0))
        if let buffer = colorUniformBuffer {
            let contents = buffer.contents()
            memcpy(contents, &whiteColor, MemoryLayout<ColorUniform>.size)
        }
        
        // No animation for border (zero offset)
        var noAnimation = AnimationUniform(offset: SIMD2<Float>(0.0, 0.0))
        if let buffer = borderUniformBuffer {
            let contents = buffer.contents()
            memcpy(contents, &noAnimation, MemoryLayout<AnimationUniform>.size)
            renderEncoder.setVertexBuffer(buffer, offset: 0, index: 2)
        }
        
        // Draw border as line strip
        renderEncoder.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: 5)
        
        renderEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func setAnimationScheduler(scheduler: AnimationScheduler?) {
        self.animationScheduler = scheduler
        scheduler?.requestAnimationFrame()
    }
    
    private func setupMetal() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to create Metal command queue")
            return
        }
        self.commandQueue = commandQueue
        
        // Create IOSurface and texture
        let backingScaleFactor: CGFloat = 2.0
        let scaledWidth = Int(backingScaleFactor * CGFloat(canvasWidth))
        let scaledHeight = Int(backingScaleFactor * CGFloat(canvasHeight))
        
        var bytesPerRow: Int = scaledWidth * 4
        bytesPerRow = ((bytesPerRow + 15)) / 16 * 16  // 16-byte aligned
        
        let iosurfaceProperties: [IOSurfacePropertyKey: Any] = [
            .width: scaledWidth,
            .height: scaledHeight,
            .pixelFormat: UInt32(kCVPixelFormatType_32BGRA),
            .bytesPerElement: 4,
            .bytesPerRow: bytesPerRow,
            .allocSize: bytesPerRow * scaledHeight * 4
        ]
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: scaledWidth,
            height: scaledHeight,
            mipmapped: false
        )
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        textureDescriptor.storageMode = .shared
        
        guard let surface = IOSurface(properties: iosurfaceProperties) else {
            print("Failed to create IOSurface")
            return
        }
        
        guard let texture = device.makeTexture(
            descriptor: textureDescriptor,
            iosurface: surface,
            plane: 0) else {
            print("Failed to create texture for IOSurface")
            return
        }
        
        canvasSurface = surface
        textureCache = texture
        
        // Set up Metal rendering pipeline
        setupMetalPipeline()
        
        // Create geometry for the checkerboard
        createGeometry(width: scaledWidth, height: scaledHeight)
    }
    
    private func setupMetalPipeline() {
        guard let device = device else { return }
        
        let currentBundle = Bundle(for: type(of: self))
        var library: MTLLibrary?
        do {
            library = try device.makeDefaultLibrary(bundle: currentBundle)
            print("Successfully loaded Metal library from class bundle: \(currentBundle.bundlePath)")
        } catch {
            print("Failed to load Metal library from class bundle \(currentBundle.bundlePath). Error: \(error)")
        }
        
        guard let library = library else {
            print("Failed to load any Metal library")
            return
        }
        
        // Get shader functions from Checkerboard.metal
        guard let vertexFunction = library.makeFunction(name: "vertex_main"),
              let fragmentFunction = library.makeFunction(name: "fragment_main") else {
            print("Failed to create Metal shader functions")
            return
        }
        
        // Create render pipeline descriptor
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        
        // Define vertex format
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        // Configure color attachment
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Create render pipeline state
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create render pipeline state: \(error)")
        }
    }
    
    private func createGeometry(width: Int, height: Int) {
        guard let device = device else { return }
        
        // Calculate the number of cells
        let cellSize: Float = targetCellSize
        let numCellsX = max(Int(Float(width) / cellSize) + 2, 6)
        let numCellsY = max(Int(Float(height) / cellSize) + 2, 6)
        
        // Store grid dimensions for animation calculations
        gridWidth = numCellsX
        gridHeight = numCellsY
        
        // Normalized cell dimensions
        let cellWidth = 2.0 / Float(numCellsX - 2)
        let cellHeight = 2.0 / Float(numCellsY - 2)
        
        var vertices: [SIMD2<Float>] = []
        var indices: [UInt16] = []
        
        // Generate checkerboard pattern
        for row in -1..<(numCellsY + 1) {
            for col in -1..<(numCellsX + 1) {
                // Only create squares for alternating pattern
                if (row + col) % 2 == 0 {
                    // Calculate position in NDC space (-1 to 1)
                    let left = -1.0 + Float(col) * cellWidth
                    let right = left + cellWidth
                    let top = 1.0 - Float(row) * cellHeight
                    let bottom = top - cellHeight
                    
                    let baseVertex = UInt16(vertices.count)
                    
                    // Add four vertices for the quad
                    vertices.append(SIMD2<Float>(left, top))      // Top-left
                    vertices.append(SIMD2<Float>(right, top))     // Top-right
                    vertices.append(SIMD2<Float>(left, bottom))   // Bottom-left
                    vertices.append(SIMD2<Float>(right, bottom))  // Bottom-right
                    
                    // Add indices for two triangles
                    indices.append(baseVertex)
                    indices.append(baseVertex + 1)
                    indices.append(baseVertex + 2)
                    
                    indices.append(baseVertex + 1)
                    indices.append(baseVertex + 3)
                    indices.append(baseVertex + 2)
                }
            }
        }
        
        vertexBuffer = device.makeBuffer(bytes: vertices,
                                         length: vertices.count * MemoryLayout<SIMD2<Float>>.stride,
                                         options: .storageModeShared)
        
        indexBuffer = device.makeBuffer(bytes: indices,
                                        length: indices.count * MemoryLayout<UInt16>.stride,
                                        options: .storageModeShared)
        
        // Create border vertices (slightly inset)
        let inset: Float = 0.01
        let borderVertices: [SIMD2<Float>] = [
            SIMD2<Float>(-1.0 + inset, 1.0 - inset),   // Top left
            SIMD2<Float>(1.0 - inset, 1.0 - inset),    // Top right
            SIMD2<Float>(1.0 - inset, -1.0 + inset),   // Bottom right
            SIMD2<Float>(-1.0 + inset, -1.0 + inset),  // Bottom left
            SIMD2<Float>(-1.0 + inset, 1.0 - inset)    // Back to top left
        ]
        
        borderBuffer = device.makeBuffer(bytes: borderVertices,
                                         length: borderVertices.count * MemoryLayout<SIMD2<Float>>.stride,
                                         options: .storageModeShared)
        
        // Create uniform buffers
        struct ColorUniform {
            var color: SIMD4<Float>
        }
        
        struct AnimationUniform {
            var offset: SIMD2<Float>
        }
        
        colorUniformBuffer = device.makeBuffer(length: MemoryLayout<ColorUniform>.size,
                                               options: .storageModeShared)
        
        animationUniformBuffer = device.makeBuffer(length: MemoryLayout<AnimationUniform>.size,
                                                   options: .storageModeShared)
        
        borderUniformBuffer = device.makeBuffer(length: MemoryLayout<AnimationUniform>.size,
                                                options: .storageModeShared)
    }
}


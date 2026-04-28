//
//  ScalarDistributionListDemoSingleCanvas.swift
//

import Foundation
import IOSurface
import OuterdocPlugin
import MetalKit


// Configuration constants
fileprivate let N_DISTRIBUTIONS = 20
fileprivate let ROW_HEIGHT = 30
fileprivate let ROW_MARGIN = 4
fileprivate let CANVAS_WIDTH = 400
fileprivate let CANVAS_HEIGHT = N_DISTRIBUTIONS * (ROW_HEIGHT + ROW_MARGIN)

fileprivate func toPrecisionThrifty(_ d: Float, _ precision: Int32) -> String {
    let ptr = UnsafeMutablePointer<CChar>.allocate(capacity: 64)
    toPrecisionThriftyC(d, precision, ptr, 64)
    let ret = String(cString: ptr)
    ptr.deallocate()
    return ret
}

fileprivate enum ScaleType {
    case log
    case linear
}

class ScalarDistributionListSingleCanvasDemo: OuterdocDynamicSubstring {
    
    private let instanceID = UUID()
    private var canvasID = UUID()
    
    // Canvas rendering properties
    private var canvasSurface: IOSurface?
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var texture: MTLTexture?
    
    // Metal rendering pipelines
    private var pointRenderPipelineState: MTLRenderPipelineState?
    private var backgroundRenderPipelineState: MTLRenderPipelineState?
    private var characterRenderPipelineState: MTLRenderPipelineState?
    
    // Buffers
    private var pointVertexBuffer: MTLBuffer?
    private var quadVertexBuffer: MTLBuffer?
    private var pointUniformsBuffer: MTLBuffer?
    private var plotBoundsBuffer: MTLBuffer?
    private var backgroundUniformsBuffers: [MTLBuffer] = []  // Multiple buffers for backgrounds
    private var characterUniformsBuffer: MTLBuffer?
    private var characterInstanceBuffer: MTLBuffer?
    
    // Pre-computed point data buffers for each dataset
    private var datasetPointBuffers: [MTLBuffer] = []
    private var totalPointCount: Int = 0
    
    // Max values for each canvas at each timestep
    private var maxValuesPerTimestep: [[Float]] = []
    private var globalMaxValue: Float = 0.0
    
    let backingScaleFactor: CGFloat = 2.0
    
    // Cache layout calculations
    private struct RowLayoutCache {
        let minText: String
        let maxText: String
        let minLabelWidth: CGFloat
        let maxLabelWidth: CGFloat
        let plotWidth: CGFloat
    }
    private var layoutCache: [RowLayoutCache] = []
    
    // Character texture cache
    private struct CharacterTextureCache {
        var textureArray: [MTLTexture] = []
        var asciiToIndex: [Int] = Array(repeating: -1, count: 256)
        var characterWidth: CGFloat = 0
        var characterHeight: CGFloat = 0
        var initialized: Bool = false
    }
    private var characterTextureCache = CharacterTextureCache()
    private var characterTextureArray: MTLTexture?
    
    // Animation properties
    private weak var animationScheduler: AnimationScheduler?
    private var animationStartTime: Double = 0
    
    // Fixed dataset animation
    private let numTimesteps = 20
    private var currentTimestep: Int = 0
    private var nextTimestep: Int = 1
    private let transitionDuration: Double = 0.4
    private let timestepDuration: Double = 0.5

    // Sample toy data for ML parameter distributions
    fileprivate static let toyDistributions: [(String, (Float, Float), ScaleType, Int, Int, Float)] = [
        ("compare('log_epochs') /", (0, 3.3), .linear, 16, 80, 0.33),
        ("compare('log_batch_size') /", (0, 3.4), .linear, 16, 80, 0.33),
        ("compare('log_conv1_weight_decay') /", (0, 3.5), .linear, 16, 80, 0.33),
        ("compare('log_conv2_weight_decay') /", (0, 3.4), .linear, 16, 80, 0.33),
        ("compare('log_conv3_weight_decay') /", (0, 3.5), .linear, 16, 80, 0.33),
        ("compare('log_dense1_weight_decay') /", (0, 3.3), .linear, 16, 80, 0.33),
        ("compare('log_dense2_weight_decay') /", (0, 3.6), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_initial_lr_pct') /", (0, 3.4), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_final_lr_pct') /", (0, 3.9), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_pct_warmup') /", (0, 3.9), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_max_lr') /", (0, 1.2), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_momentum_max_damping_factor') /", (0, 0.78), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_momentum_min_damping_factor_pct') /", (0, 3.3), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_beta1_max_damping_factor') /", (0, 2.1), .linear, 16, 80, 0.33),
        ("compare('log_1cycle_beta1_min_damping_factor_pct') /", (0, 3.3), .linear, 16, 80, 0.33),
        ("compare('log_beta2_damping_factor') /", (0, 3.4), .linear, 16, 80, 0.33),
        ("compare('log_conv1_channels') /", (0, 4.3), .linear, 16, 80, 0.33),
        ("compare('log_conv2_channels') /", (0, 4.1), .linear, 16, 80, 0.33),
        ("compare('log_conv3_channels') /", (0, 3.8), .linear, 16, 80, 0.33),
        ("compare('log_dense1_units') /", (0, 3.6), .linear, 16, 80, 0.33),
    ]
    
    init() {
        setupMetal()
        initializeAnimationData()
    }
    
    func getInitialSubstring() -> OuterdocSubstring {
        // Create a simple document structure with a single canvas
        let attributedString = NSMutableAttributedString()
        
        // Just add the canvas placeholder - no text labels
        attributedString.append(NSAttributedString(string: "\u{FFFC}\n"))

        // Create canvas info
        var canvases: [CanvasInfo] = []
        let canvasInfo = CanvasInfo()
        canvasInfo.canvasID = canvasID
        canvasInfo.location = 0 // Position of the attachment character
        canvasInfo.width = Int32(CANVAS_WIDTH)
        canvasInfo.height = Int32(CANVAS_HEIGHT)
        canvases.append(canvasInfo)

        // Get canvas surfaces
        var canvasSurfaces: [IOSurface] = []
        if let surface = self.canvasSurface {
            canvasSurfaces.append(surface)
        }

        return OuterdocSubstring(
            attributedString: attributedString,
            elements: [],
            canvases: canvases,
            canvasSurfaces: canvasSurfaces
        )
    }
    
    func generateNextFrame(targetTimestamp: CFTimeInterval) -> OuterdocElementUpdates? {
        // Continue the animation by requesting the next frame
        animationScheduler?.requestAnimationFrame()
        
        // No element updates needed since we're using a single canvas
        return nil
    }
    
    func updateCanvases(targetTimestamp: CFTimeInterval) {
        var progress: Float
        do {
            // Calculate time elapsed since animation started
            let elapsedTime = targetTimestamp - animationStartTime
            
            // Total duration for one complete cycle through all timesteps
            let totalCycleDuration = Double(numTimesteps) * timestepDuration
            
            // Where are we in the current cycle? (0 to 1)
            let cycleProgress = elapsedTime.truncatingRemainder(dividingBy: totalCycleDuration) / totalCycleDuration
            
            // Which timestep pair are we between?
            let exactTimestep = cycleProgress * Double(numTimesteps)
            currentTimestep = Int(exactTimestep) % numTimesteps
            nextTimestep = (currentTimestep + 1) % numTimesteps
            
            // Progress within the current transition (0 to 1)
            let transitionProgress = exactTimestep - Double(currentTimestep)
            
            // Apply easing only to the transition portion (not the hold time)
            // Scale progress to account for transition duration vs cycle interval
            let scaledProgress = min(transitionProgress * (timestepDuration / transitionDuration), 1.0)
            progress = Float(cubicEaseInOut(scaledProgress))
        }
        
        guard let commandQueue = commandQueue,
              let texture = texture,
              let device = device else { return }
        
        // Create a single command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let canvasWidth = Int(backingScaleFactor * CGFloat(CANVAS_WIDTH))
        let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))
        
        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0) // Transparent background
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            print("Failed to create render encoder")
            return
        }
        
        // First render all backgrounds
        renderBackgrounds(renderEncoder: renderEncoder, progress: progress, device: device)
        
        // Then render all text
        renderAllText(renderEncoder: renderEncoder, progress: progress, device: device)
        
        // Finally render all points
        renderAllPoints(renderEncoder: renderEncoder, progress: progress)
        
        renderEncoder.endEncoding()
        
        // Commit the command buffer
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func setAnimationScheduler(scheduler: AnimationScheduler?) {
        self.animationScheduler = scheduler
        self.animationStartTime = CACurrentMediaTime()
        
        // Start the animation loop
        scheduler?.requestAnimationFrame()
    }
    
    private func cubicEaseInOut(_ t: Double) -> Double {
        if t < 0.5 {
            return 4 * t * t * t
        } else {
            let p = 2 * t - 2
            return 1 + p * p * p / 2
        }
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
        
        // Initialize character texture cache
        initializeCharacterTextureCache(device: device)
        createCharacterTextureArray(device: device)
        
        // Calculate total point count
        totalPointCount = 0
        for (_, _, _, numExperiments, numOutcomes, _) in ScalarDistributionListSingleCanvasDemo.toyDistributions {
            totalPointCount += numExperiments * numOutcomes
        }
        
        // Pre-calculate layout cache
        initializeLayoutCache()
        
        // Load shaders
        let currentBundle = Bundle(for: type(of: self))
        var library: MTLLibrary?
        do {
            library = try device.makeDefaultLibrary(bundle: currentBundle)
            print("Successfully loaded Metal library from class bundle: \(currentBundle.bundlePath)")
        } catch {
            print("Failed to load Metal library from class bundle \(currentBundle.bundlePath). Error: \(error)")
            return
        }
        
        guard let library = library else { return }
        
        // Create point rendering pipeline
        setupPointRenderingPipeline(device: device, library: library)
        
        // Create background rendering pipeline
        setupBackgroundRenderingPipeline(device: device, library: library)
        
        // Create character rendering pipeline
        setupCharacterRenderingPipeline(device: device, library: library)
        
        // Create buffers
        createBuffers(device: device)
        
        // Create single large IOSurface and texture
        createCanvasSurface(device: device)
    }
    
    private func initializeLayoutCache() {
        let fontSize: CGFloat = 10 * backingScaleFactor
        let labelPadding: CGFloat = 8 * backingScaleFactor
        
        layoutCache = []
        
        for (_, domain, scaleType, _, _, _) in ScalarDistributionListSingleCanvasDemo.toyDistributions {
            let minText = "0" // Always start from 0
            let maxText = switch scaleType {
            case .linear: toPrecisionThrifty(domain.1, 2)
            case .log: String(format: "%.1e", domain.1)
            }
            
            let minLabelWidth: CGFloat = CGFloat(minText.count) * fontSize * 0.6 + labelPadding * 2
            let maxLabelWidth: CGFloat = CGFloat(maxText.count) * fontSize * 0.6 + labelPadding * 2
            let plotWidth = CGFloat(CANVAS_WIDTH) * backingScaleFactor - minLabelWidth - maxLabelWidth
            
            layoutCache.append(RowLayoutCache(
                minText: minText,
                maxText: maxText,
                minLabelWidth: minLabelWidth,
                maxLabelWidth: maxLabelWidth,
                plotWidth: plotWidth
            ))
        }
    }
    
    private func setupPointRenderingPipeline(device: MTLDevice, library: MTLLibrary) {
        guard let vertexFunction = library.makeFunction(name: "single_canvas_point_vertex"),
              let fragmentFunction = library.makeFunction(name: "single_canvas_point_fragment") else {
            print("Failed to load point shader functions")
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        // Configure vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        pipelineDescriptor.inputPrimitiveTopology = .point
        
        do {
            pointRenderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create point render pipeline state: \(error)")
        }
    }
    
    private func setupBackgroundRenderingPipeline(device: MTLDevice, library: MTLLibrary) {
        guard let vertexFunction = library.makeFunction(name: "single_canvas_background_vertex"),
              let fragmentFunction = library.makeFunction(name: "single_canvas_background_fragment") else {
            print("Failed to load background shader functions")
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        
        // Configure vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        pipelineDescriptor.inputPrimitiveTopology = .triangle
        
        do {
            backgroundRenderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create background render pipeline state: \(error)")
        }
    }
    
    private func setupCharacterRenderingPipeline(device: MTLDevice, library: MTLLibrary) {
        guard let vertexFunction = library.makeFunction(name: "single_canvas_character_vertex"),
              let fragmentFunction = library.makeFunction(name: "single_canvas_character_fragment") else {
            print("Failed to load character shader functions")
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        // Configure vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        pipelineDescriptor.inputPrimitiveTopology = .triangle
        
        do {
            characterRenderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create character render pipeline state: \(error)")
        }
    }
    
    private func createBuffers(device: MTLDevice) {
        // Create point vertex buffer
        let vertices: [SIMD2<Float>] = [SIMD2<Float>(0.0, 0.0)]
        pointVertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<SIMD2<Float>>.stride, options: [])
        
        // Create quad vertex buffer for rectangles and text
        let quadVertices: [SIMD2<Float>] = [
            SIMD2<Float>(0.0, 0.0),  // Bottom-left
            SIMD2<Float>(1.0, 0.0),  // Bottom-right
            SIMD2<Float>(0.0, 1.0),  // Top-left
            SIMD2<Float>(1.0, 1.0),  // Top-right
        ]
        quadVertexBuffer = device.makeBuffer(bytes: quadVertices, length: quadVertices.count * MemoryLayout<SIMD2<Float>>.stride, options: [])
        
        // Create uniforms buffers
        pointUniformsBuffer = device.makeBuffer(length: MemoryLayout<PointUniforms>.stride, options: [])
        plotBoundsBuffer = device.makeBuffer(length: N_DISTRIBUTIONS * MemoryLayout<PlotBounds>.stride, options: [])
        
        // Create multiple background uniform buffers (2 per distribution - white and gray)
        backgroundUniformsBuffers = []
        for _ in 0..<(N_DISTRIBUTIONS * 2) {
            if let buffer = device.makeBuffer(length: MemoryLayout<BackgroundUniforms>.stride, options: []) {
                backgroundUniformsBuffers.append(buffer)
            }
        }
        
        characterUniformsBuffer = device.makeBuffer(length: MemoryLayout<CharacterUniforms>.stride, options: [])
        
        // Create character instance buffer for all characters
        let maxCharacters = N_DISTRIBUTIONS * 2 * 10 // Max characters per row * 2 labels * rows
        characterInstanceBuffer = device.makeBuffer(length: maxCharacters * MemoryLayout<CharacterInstanceData>.stride, options: [])
    }
    
    private func createCanvasSurface(device: MTLDevice) {
        let canvasWidth = Int(backingScaleFactor * CGFloat(CANVAS_WIDTH))
        let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))
        
        var bytesPerRow: Int = canvasWidth * 4
        bytesPerRow = ((bytesPerRow + 15)) / 16 * 16  // 16-byte aligned
        
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
        
        self.canvasSurface = surface
        self.texture = texture
    }
    
    // Initialize animation data and pre-compute all point positions
    private func initializeAnimationData() {
        guard let device = device else { return }
        
        datasetPointBuffers = []
        maxValuesPerTimestep = []
        globalMaxValue = 0.0
        
        // First, generate the final dataset with Gaussian sampling
        var finalDatasets: [[[Float]]] = []
        for (_, domain, scaleType, numExperiments, numOutcomes, _) in ScalarDistributionListSingleCanvasDemo.toyDistributions {
            let dataset = generateGaussianData(numExperiments: numExperiments, numOutcomes: numOutcomes, domain: domain, scaleType: scaleType)
            finalDatasets.append(dataset)
        }
        
        // First pass: compute all values and find global max
        var allValuesPerTimestep: [[[Float]]] = []
        
        for timestep in 0..<numTimesteps {
            var valuesForTimestep: [[Float]] = []
            var maxValuesForTimestep: [Float] = []
            
            for (canvasIndex, (_, _, _, numExperiments, numOutcomes, initialValue)) in ScalarDistributionListSingleCanvasDemo.toyDistributions.enumerated() {
                let finalDataset = finalDatasets[canvasIndex]
                
                // Calculate interpolation progress for this timestep
                let timestepProgress = Float(timestep) / Float(numTimesteps - 1)
                
                // Apply cubic interpolation
                let cubicProgress: Float = timestepProgress * timestepProgress * (3.0 - 2.0 * timestepProgress)
                
                var canvasValues: [Float] = []
                var canvasMaxValue: Float = 0.0
                
                for experimentIndex in 0..<numExperiments {
                    for outcomeIndex in 0..<numOutcomes {
                        // Interpolate between initial value and final value
                        let finalValue = finalDataset[experimentIndex][outcomeIndex]
                        let baseInterpolatedValue = initialValue + (finalValue - initialValue) * cubicProgress
                        
                        // Add multiplicative noise
                        let noiseIndex = timestep * numExperiments * numOutcomes + experimentIndex * numOutcomes + outcomeIndex
                        let noiseSeed = Float(noiseIndex * 2654435761) / Float(UInt32.max)
                        
                        // Combine multiple frequencies for more organic noise
                        let noise1 = sin(noiseSeed * 2 * Float.pi)
                        let noise2 = sin(noiseSeed * 4.7 * Float.pi + 1.3)
                        let noise3 = sin(noiseSeed * 9.2 * Float.pi + 2.7)
                        let combinedNoise = (noise1 + 0.5 * noise2 + 0.25 * noise3) / 1.75
                        
                        // Scale noise amount based on transition
                        let transitionWeight = sin(timestepProgress * Float.pi)
                        let noiseScale = 0.05 + 0.15 * transitionWeight
                        let noiseAmount = 1.0 + noiseScale * combinedNoise
                        
                        let interpolatedValue = baseInterpolatedValue * noiseAmount
                        canvasValues.append(interpolatedValue)
                        canvasMaxValue = max(canvasMaxValue, interpolatedValue)
                    }
                }
                
                valuesForTimestep.append(canvasValues)
                maxValuesForTimestep.append(canvasMaxValue)
                globalMaxValue = max(globalMaxValue, canvasMaxValue)
            }
            
            allValuesPerTimestep.append(valuesForTimestep)
            maxValuesPerTimestep.append(maxValuesForTimestep)
        }
        
        // Second pass: create point buffers with proper scaling
        for timestep in 0..<numTimesteps {
            var allPointsForTimestep: [PointPosition] = []
            
            for (distributionIndex, (_, _, _, numExperiments, numOutcomes, _)) in ScalarDistributionListSingleCanvasDemo.toyDistributions.enumerated() {
                guard distributionIndex < layoutCache.count else { continue }
                let layout = layoutCache[distributionIndex]
                
                let rowY = CGFloat(distributionIndex * (ROW_HEIGHT + ROW_MARGIN)) * backingScaleFactor
                let rowHeight = CGFloat(ROW_HEIGHT) * backingScaleFactor
                
                // Scale based on global max
                let scale: (Float) -> Float = { value in
                    Float(value / self.globalMaxValue)
                }
                
                let canvasValues = allValuesPerTimestep[timestep][distributionIndex]
                var valueIndex = 0
                
                for experimentIndex in 0..<numExperiments {
                    let yNormalized = (Float(experimentIndex) + 0.5) / Float(numExperiments)
                    let yPixel = Float(rowY) + yNormalized * Float(rowHeight)
                    
                    for _ in 0..<numOutcomes {
                        let interpolatedValue = canvasValues[valueIndex]
                        valueIndex += 1
                        
                        let xNormalized = scale(interpolatedValue)
                        // Clamp to ensure points don't extend beyond the plot area
                        let clampedXNormalized = min(xNormalized, 1.0)
                        let xPixel = Float(layout.minLabelWidth) + clampedXNormalized * Float(layout.plotWidth)
                        
                        let pointData = PointPosition(position: SIMD2<Float>(xPixel, yPixel))
                        allPointsForTimestep.append(pointData)
                    }
                }
            }
            
            // Create GPU buffer for this timestep
            guard let buffer = device.makeBuffer(
                bytes: allPointsForTimestep,
                length: allPointsForTimestep.count * MemoryLayout<PointPosition>.stride,
                options: []
            ) else {
                print("Failed to create buffer for timestep \(timestep)")
                continue
            }
            
            datasetPointBuffers.append(buffer)
        }
        
        // Initialize dataset indices
        currentTimestep = 0
        nextTimestep = 1
    }
    
    // Generate Gaussian-distributed data
    private func generateGaussianData(numExperiments: Int, numOutcomes: Int, domain: (Float, Float), scaleType: ScaleType) -> [[Float]] {
        var results: [[Float]] = []
        
        for experimentIndex in 0..<numExperiments {
            var experimentData: [Float] = []
            
            // Create smooth variation of mean and variance across experiments
            let experimentProgress = Float(experimentIndex) / Float(max(1, numExperiments - 1))
            
            // Vary the mean smoothly across the domain
            let meanProgress = 0.2 + 0.6 * experimentProgress
            let mean: Float
            let variance: Float
            
            switch scaleType {
            case .linear:
                mean = domain.0 + meanProgress * (domain.1 - domain.0)
                let relativeVariance = 0.05 + 0.1 * (1.0 - experimentProgress)
                variance = relativeVariance * (domain.1 - domain.0)
            case .log:
                let logMin = log(domain.0)
                let logMax = log(domain.1)
                let logMean = logMin + meanProgress * (logMax - logMin)
                mean = exp(logMean)
                let logVarianceScale = 0.3 + 0.4 * (1.0 - experimentProgress)
                variance = (logVarianceScale * logVarianceScale)
            }
            
            // Generate Gaussian samples
            for _ in 0..<numOutcomes {
                let value: Float
                switch scaleType {
                case .linear:
                    value = generateGaussianSample(mean: mean, variance: variance)
                case .log:
                    let logMean = log(mean)
                    let logStdDev = sqrt(variance)
                    let logValue = generateGaussianSample(mean: logMean, variance: logStdDev * logStdDev)
                    value = exp(logValue)
                }
                let clampedValue = max(domain.0, min(domain.1, value))
                experimentData.append(clampedValue)
            }
            
            results.append(experimentData)
        }
        
        return results
    }
    
    // Box-Muller transform for Gaussian sampling
    private func generateGaussianSample(mean: Float, variance: Float) -> Float {
        let u1 = Float.random(in: 0...1)
        let u2 = Float.random(in: 0...1)
        let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
        return mean + sqrt(variance) * z0
    }
    
    // Render backgrounds for all distributions
    private func renderBackgrounds(renderEncoder: MTLRenderCommandEncoder, progress: Float, device: MTLDevice) {
        guard let backgroundRenderPipelineState = backgroundRenderPipelineState,
              let quadVertexBuffer = quadVertexBuffer,
              !backgroundUniformsBuffers.isEmpty else { return }
        
        renderEncoder.setRenderPipelineState(backgroundRenderPipelineState)
        renderEncoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)
        
        let canvasWidth = Int(backingScaleFactor * CGFloat(CANVAS_WIDTH))
        let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))
        
        for i in 0..<N_DISTRIBUTIONS {
            let layout = layoutCache[i]
            let rowY = Float(i * (ROW_HEIGHT + ROW_MARGIN)) * Float(backingScaleFactor)
            let rowHeight = Float(ROW_HEIGHT) * Float(backingScaleFactor)
            
            // Interpolate max value
            let currentMax = maxValuesPerTimestep[currentTimestep][i]
            let nextMax = maxValuesPerTimestep[nextTimestep][i]
            let interpolatedMax = currentMax + (nextMax - currentMax) * progress
            
            // Calculate dynamic width
            let widthRatio = interpolatedMax / globalMaxValue
            let dynamicPlotWidth = Float(layout.plotWidth) * widthRatio
            let visibleWidth = Float(layout.minLabelWidth) + dynamicPlotWidth + Float(layout.maxLabelWidth)
            
            // Draw white rounded rectangle background
            var uniforms = BackgroundUniforms(
                canvasSize: SIMD2<Float>(Float(canvasWidth), Float(canvasHeight)),
                rect: SIMD4<Float>(0, rowY, visibleWidth, rowHeight),
                fillColor: SIMD4<Float>(1.0, 1.0, 1.0, 1.0),
                strokeColor: SIMD4<Float>(0.7, 0.7, 0.7, 1.0),
                cornerRadius: 5 * Float(backingScaleFactor),
                strokeWidth: 1 * Float(backingScaleFactor),
                isGrayRect: 0
            )
            
            let bufferIndex = i * 2  // White rectangle buffer
            guard bufferIndex < backgroundUniformsBuffers.count else { continue }
            let buffer = backgroundUniformsBuffers[bufferIndex]
            buffer.contents().copyMemory(from: &uniforms, byteCount: MemoryLayout<BackgroundUniforms>.stride)
            renderEncoder.setVertexBuffer(buffer, offset: 0, index: 1)
            renderEncoder.setFragmentBuffer(buffer, offset: 0, index: 1)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            
            // Draw gray plot area
            uniforms = BackgroundUniforms(
                canvasSize: SIMD2<Float>(Float(canvasWidth), Float(canvasHeight)),
                rect: SIMD4<Float>(Float(layout.minLabelWidth), rowY, dynamicPlotWidth, rowHeight),
                fillColor: SIMD4<Float>(0.9, 0.9, 0.9, 1.0),
                strokeColor: SIMD4<Float>(0.7, 0.7, 0.7, 1.0),
                cornerRadius: 0,
                strokeWidth: 1 * Float(backingScaleFactor),
                isGrayRect: 1
            )
            
            let grayBufferIndex = i * 2 + 1  // Gray rectangle buffer
            guard grayBufferIndex < backgroundUniformsBuffers.count else { continue }
            let grayBuffer = backgroundUniformsBuffers[grayBufferIndex]
            grayBuffer.contents().copyMemory(from: &uniforms, byteCount: MemoryLayout<BackgroundUniforms>.stride)
            renderEncoder.setVertexBuffer(grayBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentBuffer(grayBuffer, offset: 0, index: 1)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
    }
    
    // Render all text
    private func renderAllText(renderEncoder: MTLRenderCommandEncoder, progress: Float, device: MTLDevice) {
        guard let characterRenderPipelineState = characterRenderPipelineState,
              let quadVertexBuffer = quadVertexBuffer,
              let characterInstanceBuffer = characterInstanceBuffer,
              let characterTextureArray = characterTextureArray,
              characterTextureCache.initialized else { return }
        
        let canvasWidth = Int(backingScaleFactor * CGFloat(CANVAS_WIDTH))
        let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))
        
        // Collect all character instances
        var allInstances: [CharacterInstanceData] = []
        allInstances.reserveCapacity(N_DISTRIBUTIONS * 8) // heuristic

        for i in 0..<N_DISTRIBUTIONS {
            let layout = layoutCache[i]
            let rowY = Float(i * (ROW_HEIGHT + ROW_MARGIN)) * Float(backingScaleFactor)
            let rowHeight = Float(ROW_HEIGHT) * Float(backingScaleFactor)
            
            // Interpolate max value
            let currentMax = maxValuesPerTimestep[currentTimestep][i]
            let nextMax = maxValuesPerTimestep[nextTimestep][i]
            let interpolatedMax = currentMax + (nextMax - currentMax) * progress
            
            // Calculate dynamic width
            let widthRatio = interpolatedMax / globalMaxValue
            let dynamicPlotWidth = Float(layout.plotWidth) * widthRatio
            
            let charWidth = Float(characterTextureCache.characterWidth)
            let charHeight = Float(characterTextureCache.characterHeight)
            let textY = rowY + (rowHeight - charHeight) / 2
            
            // Min label "0" - use buffer directly like in original
            let minTextBuffer: [CChar] = [48, 0] // "0" in ASCII
            let minTextLen = 1
            
            // Calculate min text position
            let minTextWidth = Float(minTextLen) * charWidth
            let minTextX = (Float(layout.minLabelWidth) - minTextWidth) / 2
            
            // Add min text characters
            var currentX = minTextX
            for j in 0..<minTextLen {
                let ascii = Int(minTextBuffer[j])
                let textureIndex = characterTextureCache.asciiToIndex[ascii]
                if textureIndex >= 0 {
                    let transform = createTransformMatrix(
                        x: currentX,
                        y: textY,
                        width: charWidth,
                        height: charHeight,
                        canvasSize: SIMD2<Float>(Float(canvasWidth), Float(canvasHeight))
                    )
                    allInstances.append(CharacterInstanceData(transform: transform, textureIndex: UInt32(textureIndex)))
                }
                currentX += charWidth
            }
            
            // Max label - use temporary buffer for performance like in original
            withUnsafeTemporaryAllocation(of: CChar.self, capacity: 64) { maxTextBuffer in
                // Populate max text buffer
                let scaleType = ScalarDistributionListSingleCanvasDemo.toyDistributions[i].2
                let maxTextLen = switch scaleType {
                case .linear:
                    Int(toPrecisionThriftyC(interpolatedMax, 2, maxTextBuffer.baseAddress, 64))
                case .log:
                    Int(formatScientificC(interpolatedMax, maxTextBuffer.baseAddress, 64))
                }
                
                // Calculate text width using full character width
                let maxTextWidth = Float(maxTextLen) * charWidth
                var currentX = Float(layout.minLabelWidth) + dynamicPlotWidth + (Float(layout.maxLabelWidth) - maxTextWidth) / 2
                
                // Add max text characters
                for i in 0..<maxTextLen {
                    let ascii = Int(maxTextBuffer[i])
                    let textureIndex = characterTextureCache.asciiToIndex[ascii]
                    if textureIndex >= 0 {
                        let transform = createTransformMatrix(
                            x: currentX,
                            y: textY,
                            width: charWidth,
                            height: charHeight,
                            canvasSize: SIMD2<Float>(Float(canvasWidth), Float(canvasHeight))
                        )
                        allInstances.append(CharacterInstanceData(transform: transform, textureIndex: UInt32(textureIndex)))
                    }
                    currentX += charWidth
                }
            }
        }
        
        guard !allInstances.isEmpty else { return }
        
        // Update instance buffer
        characterInstanceBuffer.contents().copyMemory(
            from: allInstances,
            byteCount: allInstances.count * MemoryLayout<CharacterInstanceData>.stride
        )
        
        // Set pipeline and render
        renderEncoder.setRenderPipelineState(characterRenderPipelineState)
        renderEncoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(characterInstanceBuffer, offset: 0, index: 2)
        renderEncoder.setFragmentTexture(characterTextureArray, index: 0)
        renderEncoder.setFragmentBuffer(characterInstanceBuffer, offset: 0, index: 2)
        
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: allInstances.count)
    }
    
    // Render all points
    private func renderAllPoints(renderEncoder: MTLRenderCommandEncoder, progress: Float) {
        guard let pointRenderPipelineState = pointRenderPipelineState,
              let pointVertexBuffer = pointVertexBuffer,
              let pointUniformsBuffer = pointUniformsBuffer,
              let plotBoundsBuffer = plotBoundsBuffer,
              currentTimestep < datasetPointBuffers.count,
              nextTimestep < datasetPointBuffers.count else { return }
        
        let startBuffer = datasetPointBuffers[currentTimestep]
        let endBuffer = datasetPointBuffers[nextTimestep]
        
        let canvasWidth = Int(backingScaleFactor * CGFloat(CANVAS_WIDTH))
        let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))
        
        // Update plot bounds based on current animation state
        var plotBoundsData: [PlotBounds] = []
        for i in 0..<N_DISTRIBUTIONS {
            let layout = layoutCache[i]
            
            // Interpolate max value
            let currentMax = maxValuesPerTimestep[currentTimestep][i]
            let nextMax = maxValuesPerTimestep[nextTimestep][i]
            let interpolatedMax = currentMax + (nextMax - currentMax) * progress
            
            // Calculate dynamic width
            let widthRatio = interpolatedMax / globalMaxValue
            let dynamicPlotWidth = Float(layout.plotWidth) * widthRatio
            
            // Calculate Y bounds for this row
            let rowY = Float(i * (ROW_HEIGHT + ROW_MARGIN)) * Float(backingScaleFactor)
            let rowHeight = Float(ROW_HEIGHT) * Float(backingScaleFactor)
            
            let bounds = PlotBounds(
                minX: Float(layout.minLabelWidth),
                maxX: Float(layout.minLabelWidth) + dynamicPlotWidth,
                minY: rowY,
                maxY: rowY + rowHeight
            )
            plotBoundsData.append(bounds)
        }
        
        plotBoundsBuffer.contents().copyMemory(from: plotBoundsData, byteCount: plotBoundsData.count * MemoryLayout<PlotBounds>.stride)
        
        // Update uniforms
        let uniforms = PointUniforms(
            viewportSize: SIMD2<Float>(Float(canvasWidth), Float(canvasHeight)),
            pointRadius: 1.5 * Float(backingScaleFactor),
            pointColor: SIMD4<Float>(0.2, 0.4, 0.8, 0.7),
            progress: progress,
            rowHeight: Float(ROW_HEIGHT) * Float(backingScaleFactor),
            rowMargin: Float(ROW_MARGIN) * Float(backingScaleFactor)
        )
        
        pointUniformsBuffer.contents().copyMemory(from: [uniforms], byteCount: MemoryLayout<PointUniforms>.stride)
        
        renderEncoder.setRenderPipelineState(pointRenderPipelineState)
        renderEncoder.setVertexBuffer(pointVertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(pointUniformsBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(startBuffer, offset: 0, index: 2)
        renderEncoder.setVertexBuffer(endBuffer, offset: 0, index: 3)
        renderEncoder.setVertexBuffer(plotBoundsBuffer, offset: 0, index: 4)
        
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: 1, instanceCount: totalPointCount)
    }
    
    // Helper to create transformation matrix
    private func createTransformMatrix(x: Float, y: Float, width: Float, height: Float, canvasSize: SIMD2<Float>) -> simd_float4x4 {
        // Convert to NDC coordinates
        let ndcX = (x / canvasSize.x) * 2.0 - 1.0
        let ndcY = 1.0 - ((y + height) / canvasSize.y) * 2.0
        let ndcWidth = (width / canvasSize.x) * 2.0
        let ndcHeight = (height / canvasSize.y) * 2.0
        
        return simd_float4x4(
            SIMD4<Float>(ndcWidth, 0, 0, 0),
            SIMD4<Float>(0, ndcHeight, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(ndcX, ndcY, 0, 1)
        )
    }
    
    // Initialize character texture cache
    private func initializeCharacterTextureCache(device: MTLDevice) {
        guard !characterTextureCache.initialized else { return }
        
        let fontSize: CGFloat = 10 * backingScaleFactor
        let font = CTFontCreateWithName("Menlo" as CFString, fontSize, nil)
        
        // Get character dimensions
        let testString = "0" as CFString
        let attrString = CFAttributedStringCreate(nil, testString, [kCTFontAttributeName: font] as CFDictionary)
        let line = CTLineCreateWithAttributedString(attrString!)
        let bounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
        
        // Add some padding
        let charWidth = ceil(bounds.width) + 2
        let charHeight = ceil(bounds.height) + 4
        
        characterTextureCache.characterWidth = charWidth
        characterTextureCache.characterHeight = charHeight
        
        // Characters to cache
        let charactersToCache: [Character] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-", "e", "+"]
        
        // Create textures and populate lookup table
        characterTextureCache.textureArray.removeAll()
        for (index, char) in charactersToCache.enumerated() {
            if let texture = createCharacterTexture(character: char, width: charWidth, height: charHeight, fontSize: fontSize, device: device) {
                characterTextureCache.textureArray.append(texture)
                // Populate ASCII lookup table
                if let ascii = char.asciiValue {
                    characterTextureCache.asciiToIndex[Int(ascii)] = index
                }
            }
        }
        
        characterTextureCache.initialized = true
    }
    
    // Create texture array from individual character textures
    private func createCharacterTextureArray(device: MTLDevice) {
        guard characterTextureCache.initialized,
              !characterTextureCache.textureArray.isEmpty else { return }
        
        let width = Int(characterTextureCache.characterWidth)
        let height = Int(characterTextureCache.characterHeight)
        let arrayLength = characterTextureCache.textureArray.count
        
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type2DArray
        textureDescriptor.pixelFormat = .bgra8Unorm
        textureDescriptor.width = width
        textureDescriptor.height = height
        textureDescriptor.arrayLength = arrayLength
        textureDescriptor.mipmapLevelCount = 1
        textureDescriptor.usage = [.shaderRead]
        
        guard let textureArray = device.makeTexture(descriptor: textureDescriptor) else {
            print("Failed to create character texture array")
            return
        }
        
        // Copy each character texture into the array
        for (index, texture) in characterTextureCache.textureArray.enumerated() {
            let commandBuffer = commandQueue?.makeCommandBuffer()
            let blitEncoder = commandBuffer?.makeBlitCommandEncoder()
            
            blitEncoder?.copy(
                from: texture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: width, height: height, depth: 1),
                to: textureArray,
                destinationSlice: index,
                destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            
            blitEncoder?.endEncoding()
            commandBuffer?.commit()
        }
        
        characterTextureArray = textureArray
    }
    
    // Create a texture for a single character
    private func createCharacterTexture(character: Character, width: CGFloat, height: CGFloat, fontSize: CGFloat, device: MTLDevice) -> MTLTexture? {
        let bytesPerPixel = 4
        let bytesPerRow = Int(width) * bytesPerPixel
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: Int(width),
            height: Int(height),
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }
        
        // Clear to transparent
        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw character
        let font = CTFontCreateWithName("Menlo" as CFString, fontSize, nil)
        let textColor = CGColor(red: 0.6, green: 0.6, blue: 0.6, alpha: 1.0)
        
        let charString = String(character) as CFString
        let attrString = CFAttributedStringCreate(nil, charString, [kCTFontAttributeName: font, kCTForegroundColorAttributeName: textColor] as CFDictionary)
        let line = CTLineCreateWithAttributedString(attrString!)
        let textBounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
        
        // Center the character
        let textX = (width - textBounds.width) / 2
        let textY = (height - textBounds.height) / 2
        context.textPosition = CGPoint(x: textX, y: textY)
        CTLineDraw(line, context)
        
        // Create texture from context
        guard let data = context.data else { return nil }
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: Int(width),
            height: Int(height),
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else { return nil }
        
        texture.replace(
            region: MTLRegionMake2D(0, 0, Int(width), Int(height)),
            mipmapLevel: 0,
            withBytes: data,
            bytesPerRow: bytesPerRow
        )
        
        return texture
    }
    
    // Structs for shader data
    private struct PointUniforms {
        var viewportSize: SIMD2<Float>
        var pointRadius: Float
        var pointColor: SIMD4<Float>
        var progress: Float
        var rowHeight: Float
        var rowMargin: Float
    }
    
    private struct PlotBounds {
        var minX: Float
        var maxX: Float
        var minY: Float
        var maxY: Float
    }
    
    private struct PointPosition {
        var position: SIMD2<Float>
    }
    
    private struct BackgroundUniforms {
        var canvasSize: SIMD2<Float>
        var rect: SIMD4<Float> // x, y, width, height
        var fillColor: SIMD4<Float>
        var strokeColor: SIMD4<Float>
        var cornerRadius: Float
        var strokeWidth: Float
        var isGrayRect: Float
    }
    
    private struct CharacterUniforms {
        var canvasSize: SIMD2<Float>
    }
    
    private struct CharacterInstanceData {
        var transform: simd_float4x4
        var textureIndex: UInt32
    }
}

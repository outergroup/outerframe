//
//  ScalarDistributionDemo.swift
//

import Foundation
import IOSurface
import OuterdocPlugin
import MetalKit


// Configuration constants
fileprivate let N_DISTRIBUTIONS = 20

fileprivate let CANVAS_WIDTH = 400
fileprivate let CANVAS_HEIGHT = 12

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

class ScalarDistributionDemo: OuterdocDynamicSubstring {
    
    private let instanceID = UUID()
    private var elementIds: [String: UUID]
    private var canvasIDs: [UUID]
    private var elementRanges: [String: NSRange] = [:]
    private var cachedElementContents: [UUID: NSAttributedString] = [:]
    
    // Canvas rendering properties
    private var canvasSurfaces: [IOSurface] = []
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var textureCache: [MTLTexture] = []
    
    // Metal rendering for wandering points
    private var pointRenderPipelineState: MTLRenderPipelineState?
    private var pointVertexBuffer: MTLBuffer?
    private var pointUniformsBuffers: [MTLBuffer] = []
    private var pointInstanceBuffers: [MTLBuffer] = []
    
    // Metal rendering for UI elements
    private var instancedGrayRectPipelineState: MTLRenderPipelineState?
    private var endMarkerPipelineState: MTLRenderPipelineState?
    private var uiVertexBuffer: MTLBuffer?
    private var backgroundUniformsBuffer: MTLBuffer?
    private var backgroundInstanceBuffer: MTLBuffer?
    private var endMarkerUniformsBuffer: MTLBuffer?
    private var endMarkerInstanceBuffer: MTLBuffer?
    
    // Character-based text rendering pipeline
    private var characterRenderPipelineState: MTLRenderPipelineState?
    private var characterInstanceBuffer: MTLBuffer?
    
    // Instanced rendering properties
    private var largeTexture: MTLTexture?
    private var canvasInstanceBuffer: MTLBuffer?
    private var pointUniformsBuffer: MTLBuffer?
    private var canvasPointCounts: [Int] = []
    private var totalPointCount: Int = 0
    
    // Pre-computed point data buffers for each dataset
    private var datasetPointBuffers: [MTLBuffer] = []  // One buffer per dataset
    
    // Max values for each canvas at each timestep
    private var maxValuesPerTimestep: [[Float]] = []  // [timestep][canvas]
    private var globalMaxValue: Float = 0.0
    private var canvasActualWidths: [Int] = []  // Actual width for each canvas based on its data

    let backingScaleFactor: CGFloat = 2.0
    
    // Cache layout calculations
    private struct CanvasLayoutCache {
        let minText: String
        let maxText: String
        let minLabelWidth: CGFloat
        let maxLabelWidth: CGFloat
        let plotWidth: CGFloat
    }
    private var layoutCache: [CanvasLayoutCache] = []
    
    // Character texture cache for optimized rendering
    private struct CharacterTextureCache {
        var textureArray: [MTLTexture] = []
        var asciiToIndex: [Int] = Array(repeating: -1, count: 256)  // ASCII lookup table
        var characterWidth: CGFloat = 0
        var characterHeight: CGFloat = 0
        var initialized: Bool = false
    }
    private var characterTextureCache = CharacterTextureCache()
    private var characterTextureArray: MTLTexture?  // Texture2DArray containing all character textures
    
    // Animation properties
    private weak var animationScheduler: AnimationScheduler?
    private var animationStartTime: Double = 0
    
    // Fixed dataset animation
    private let numTimesteps = 20
    private var currentTimestep: Int = 0
    private var nextTimestep: Int = 1
    private let transitionDuration: Double = 0.4
    private let timestepDuration: Double = 0.5
    
    // Cubic easing function for smooth transitions
    private func cubicEaseInOut(_ t: Double) -> Double {
        if t < 0.5 {
            return 4 * t * t * t
        } else {
            let p = 2 * t - 2
            return 1 + p * p * p / 2
        }
    }
    
    // Sample toy data for ML parameter distributions
    // Each entry contains: (parameter key name, domain, scale type, number of values, initial value)
    fileprivate static let toyDistributions: [(String, (Float, Float), ScaleType, Int, Float)] = [
        ("'log_epochs'", (0, 3.3), .linear, 100, 0.33),
        ("'log_batch_size'", (0, 3.4), .linear, 100, 0.33),
        ("'log_conv1_weight_decay'", (0, 3.5), .linear, 100, 0.33),
        ("'log_conv2_weight_decay'", (0, 3.4), .linear, 100, 0.33),
        ("'log_conv3_weight_decay'", (0, 3.5), .linear, 100, 0.33),
        ("'log_dense1_weight_decay'", (0, 3.3), .linear, 100, 0.33),
        ("'log_dense2_weight_decay'", (0, 3.6), .linear, 100, 0.33),
        ("'log_1cycle_initial_lr_pct'", (0, 3.4), .linear, 100, 0.33),
        ("'log_1cycle_final_lr_pct'", (0, 3.9), .linear, 100, 0.33),
        ("'log_1cycle_pct_warmup'", (0, 3.9), .linear, 100, 0.33),
        ("'log_1cycle_max_lr'", (0, 1.2), .linear, 100, 0.33),
        ("'log_1cycle_momentum_max_damping_factor'", (0, 0.78), .linear, 100, 0.33),
        ("'log_1cycle_momentum_min_damping_factor_pct'", (0, 3.3), .linear, 100, 0.33),
        ("'log_1cycle_beta1_max_damping_factor'", (0, 2.1), .linear, 100, 0.33),
        ("'log_1cycle_beta1_min_damping_factor_pct'", (0, 3.3), .linear, 100, 0.33),
        ("'log_beta2_damping_factor'", (0, 3.4), .linear, 100, 0.33),
        ("'log_conv1_channels'", (0, 4.3), .linear, 100, 0.33),
        ("'log_conv2_channels'", (0, 4.1), .linear, 100, 0.33),
        ("'log_conv3_channels'", (0, 3.8), .linear, 100, 0.33),
        ("'log_dense1_units'", (0, 3.6), .linear, 100, 0.33),
    ]
    
    init() {
        var elementIds = [String: UUID]()
        for i in 0..<N_DISTRIBUTIONS {
            elementIds["dist\(i)"] = UUID()
        }
        self.elementIds = elementIds
        self.canvasIDs = Array(elementIds.keys).map { _ in UUID() }
        
        let fragmentContainer = generateFragmentContainer()
        self.elementRanges = fragmentContainer.elementRanges
        
        // Initialize Metal device first
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
        
        // Calculate layout cache early for width calculations
        initializeLayoutCache()
        
        // Initialize animation data and calculate canvas widths
        initializeAnimationData()
        calculateCanvasWidths()
        
        // Now setup Metal resources with correct canvas widths
        setupMetal()
    }
    
    func getInitialSubstring() -> OuterdocSubstring {
        let fragmentContainer = generateFragmentContainer()
        let container = fragmentContainer.attributedString

        var canvases: [CanvasInfo] = []
        
        for i in 0..<N_DISTRIBUTIONS {
            let canvasID = canvasIDs[i]
            
            // Add canvas info
            let canvasInfo = CanvasInfo()
            canvasInfo.canvasID = canvasID
            
            // Find attachment character position in the string
            let attachmentStr = "\u{FFFC}"
            let containerStr = container.string
            let keyName = ScalarDistributionDemo.toyDistributions[min(i, ScalarDistributionDemo.toyDistributions.count - 1)].0
            let labelPrefix = "compare(\(keyName)) / "
            
            // Calculate position based on line structure
            var searchStartIndex = containerStr.startIndex
            var currentIndex = 0
            var found = false
            
            while currentIndex <= i, let range = containerStr.range(of: attachmentStr, range: searchStartIndex..<containerStr.endIndex) {
                if currentIndex == i {
                    let location = containerStr.distance(from: containerStr.startIndex, to: range.lowerBound)
                    canvasInfo.location = Int32(location)
                    found = true
                    break
                }
                searchStartIndex = range.upperBound
                currentIndex += 1
            }
            
            if !found {
                // Fallback calculation
                let estimatedLineLength = labelPrefix.count + 1 + 1 // label + attachment + newline
                canvasInfo.location = Int32(i * estimatedLineLength + labelPrefix.count)
            }
            
            // Set canvas dimensions based on actual data width
            canvasInfo.width = Int32(canvasActualWidths[i])
            canvasInfo.height = Int32(CANVAS_HEIGHT)  // Height for distribution visualization
            canvases.append(canvasInfo)
        }

        return OuterdocSubstring(
            attributedString: container,
            elements: [],
            canvases: canvases,
            canvasSurfaces: self.canvasSurfaces
        )
    }
    
    func generateNextFrame(targetTimestamp: CFTimeInterval) -> OuterdocElementUpdates? {
        // Continue the animation by requesting the next frame
        animationScheduler?.requestAnimationFrame()
        
        // No element updates needed since we're using canvases
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
              let largeTexture = largeTexture else { return }

        // Create a single command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let device else { return }

        let canvasWidth = Int(backingScaleFactor * CGFloat(CANVAS_WIDTH))
        let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))

        // First, clear all textures and render dynamic content
        do {
            guard let instancedGrayRectPipelineState = instancedGrayRectPipelineState,
                  let uiVertexBuffer = uiVertexBuffer,
                  let backgroundUniformsBuffer = backgroundUniformsBuffer,
                  let backgroundInstanceBuffer = backgroundInstanceBuffer else {
                return
            }

            // Update background instance data for all canvases
            var backgroundInstances: [BackgroundInstanceData] = []
            backgroundInstances.reserveCapacity(N_DISTRIBUTIONS)

            for index in 0..<N_DISTRIBUTIONS {
                let layout = layoutCache[index]

                // Interpolate max value
                let currentMax = maxValuesPerTimestep[currentTimestep][index]
                let nextMax = maxValuesPerTimestep[nextTimestep][index]
                let interpolatedMax = currentMax + (nextMax - currentMax) * Float(progress)

                // Calculate dynamic width based on max value relative to global max
                let widthRatio = interpolatedMax / globalMaxValue
                let dynamicPlotWidth = Float(layout.plotWidth) * Float(widthRatio)

                let instance = BackgroundInstanceData(
                    canvasIndex: UInt32(index),
                    visibleWidth: Float(layout.minLabelWidth) + dynamicPlotWidth + Float(layout.maxLabelWidth),
                    plotOffset: Float(layout.minLabelWidth),
                    plotWidth: dynamicPlotWidth,
                    strokeWidth: 0.5 * Float(backingScaleFactor)
                )

                backgroundInstances.append(instance)
            }

            // Update instance buffer
            backgroundInstanceBuffer.contents().copyMemory(
                from: backgroundInstances,
                byteCount: backgroundInstances.count * MemoryLayout<BackgroundInstanceData>.stride
            )

            // Create a single render pass for the entire texture array
            let renderPassDescriptor = MTLRenderPassDescriptor()
            renderPassDescriptor.colorAttachments[0].texture = largeTexture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)  // Transparent background
            renderPassDescriptor.colorAttachments[0].storeAction = .store
            renderPassDescriptor.renderTargetArrayLength = N_DISTRIBUTIONS
            guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(
                descriptor: renderPassDescriptor) else {
                print("Failed to create render encoder for UI")
                return
            }

            // Draw all gray rectangles
            renderEncoder.setRenderPipelineState(instancedGrayRectPipelineState)
            renderEncoder.setVertexBuffer(uiVertexBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(backgroundUniformsBuffer, offset: 0, index: 1)
            renderEncoder.setVertexBuffer(backgroundInstanceBuffer, offset: 0, index: 2)
            renderEncoder.setFragmentBuffer(backgroundUniformsBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentBuffer(backgroundInstanceBuffer, offset: 0, index: 2)

            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: N_DISTRIBUTIONS)

            // Batch render text for all canvases in a single optimized pass
            renderAllCanvasText(
                currentMaxValues: maxValuesPerTimestep[currentTimestep],
                nextMaxValues: maxValuesPerTimestep[nextTimestep],
                progress: progress,
                backgroundInstances: backgroundInstances,
                renderEncoder: renderEncoder,
                device: device
            )

            // Render points to each texture array slice using instanced rendering
            guard let pipelineState = pointRenderPipelineState,
                  let pointVertexBuffer = pointVertexBuffer,
                  let pointUniformsBuffer = pointUniformsBuffer,
                  currentTimestep < datasetPointBuffers.count,
                  nextTimestep < datasetPointBuffers.count else { return }

            let startBuffer = datasetPointBuffers[currentTimestep]
            let endBuffer = datasetPointBuffers[nextTimestep]

            // Update uniforms
            let uniforms = PointUniforms(
                viewportSize: SIMD2<Float>(Float(canvasWidth), Float(canvasHeight)),
                pointRadius: 2.5 * Float(backingScaleFactor),
                pointColor: SIMD4<Float>(0.2, 0.4, 0.8, 0.4),
                progress: progress
            )

            let uniformsPointer = pointUniformsBuffer.contents().bindMemory(to: PointUniforms.self, capacity: 1)
            uniformsPointer[0] = uniforms

            renderEncoder.setRenderPipelineState(pipelineState)
            renderEncoder.setVertexBuffer(pointVertexBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(pointUniformsBuffer, offset: 0, index: 1)
            renderEncoder.setVertexBuffer(startBuffer, offset: 0, index: 2)
            renderEncoder.setVertexBuffer(endBuffer, offset: 0, index: 3)
            renderEncoder.setVertexBuffer(backgroundInstanceBuffer, offset: 0, index: 4)

            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: 1, instanceCount: totalPointCount)
            
            // Render end markers on top of everything
            guard let endMarkerPipelineState = endMarkerPipelineState,
                  let endMarkerUniformsBuffer = endMarkerUniformsBuffer,
                  let endMarkerInstanceBuffer = endMarkerInstanceBuffer else { 
                renderEncoder.endEncoding()
                return 
            }
            
            // Update end marker instance data
            var endMarkerInstances: [EndMarkerInstanceData] = []
            endMarkerInstances.reserveCapacity(N_DISTRIBUTIONS * 2)
            
            for index in 0..<N_DISTRIBUTIONS {
                let layout = layoutCache[index]
                let instance = backgroundInstances[index]
                
                let markerWidth: Float = 2.0 * Float(backingScaleFactor)  // 4 actual pixels wide
                let strokeWidth: Float = 0.5 * Float(backingScaleFactor)  // 1 actual pixel stroke
                
                // Start marker at the beginning of the gray area
                let startMarker = EndMarkerInstanceData(
                    canvasIndex: UInt32(index),
                    xPosition: Float(layout.minLabelWidth),
                    markerWidth: markerWidth,
                    strokeWidth: strokeWidth
                )
                endMarkerInstances.append(startMarker)
                
                // End marker at the end of the gray area
                let endMarker = EndMarkerInstanceData(
                    canvasIndex: UInt32(index),
                    xPosition: Float(layout.minLabelWidth) + instance.plotWidth - markerWidth,
                    markerWidth: markerWidth,
                    strokeWidth: strokeWidth
                )
                endMarkerInstances.append(endMarker)
            }
            
            // Update instance buffer
            endMarkerInstanceBuffer.contents().copyMemory(
                from: endMarkerInstances,
                byteCount: endMarkerInstances.count * MemoryLayout<EndMarkerInstanceData>.stride
            )
            
            // Render end markers
            renderEncoder.setRenderPipelineState(endMarkerPipelineState)
            renderEncoder.setVertexBuffer(uiVertexBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(endMarkerUniformsBuffer, offset: 0, index: 1)
            renderEncoder.setVertexBuffer(endMarkerInstanceBuffer, offset: 0, index: 2)
            renderEncoder.setFragmentBuffer(endMarkerUniformsBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentBuffer(endMarkerInstanceBuffer, offset: 0, index: 2)
            
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: endMarkerInstances.count)
            
            renderEncoder.endEncoding()
        }

        // Now blit the completed large texture to IOSurfaces using the same command buffer
        do {
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }

            for (index, texture) in textureCache.enumerated() {
                // The IOSurface width is what we calculated in canvasActualWidths, scaled by backingScaleFactor
                let ioSurfaceWidth = Int(backingScaleFactor * CGFloat(canvasActualWidths[index]))
                
                blitEncoder.copy(
                    from: largeTexture,
                    sourceSlice: index,
                    sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: ioSurfaceWidth, height: canvasHeight, depth: 1),
                    to: texture,
                    destinationSlice: 0,
                    destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                )
            }

            blitEncoder.endEncoding()
        }

        // Commit the single command buffer containing all operations
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func setAnimationScheduler(scheduler: AnimationScheduler?) {
        self.animationScheduler = scheduler
        self.animationStartTime = CACurrentMediaTime()
        
        // Start the animation loop
        scheduler?.requestAnimationFrame()
    }
    
    // Initialize animation data and pre-compute all point positions
    private func initializeAnimationData() {
        guard let device = device else { return }
        
        datasetPointBuffers = []
        maxValuesPerTimestep = []
        globalMaxValue = 0.0
        
        // First, generate the final dataset with varied distributions
        var finalDatasets: [[Float]] = []
        for (canvasIndex, (_, domain, scaleType, numValues, _)) in ScalarDistributionDemo.toyDistributions.enumerated() {
            let dataset = generateVariedDistribution(canvasIndex: canvasIndex, numValues: numValues, domain: domain, scaleType: scaleType)
            finalDatasets.append(dataset)
        }
        
        // First pass: compute all values and find global max
        var allValuesPerTimestep: [[[Float]]] = []  // [timestep][canvas][values]
        
        for timestep in 0..<numTimesteps {
            var valuesForTimestep: [[Float]] = []
            var maxValuesForTimestep: [Float] = []
            
            for (canvasIndex, (_, _, _, numValues, initialValue)) in ScalarDistributionDemo.toyDistributions.enumerated() {
                let finalDataset = finalDatasets[canvasIndex]
                
                // Calculate interpolation progress for this timestep
                let timestepProgress = Float(timestep) / Float(numTimesteps - 1)

                // Apply cubic interpolation for more interesting transitions
                let cubicProgress: Float = timestepProgress * timestepProgress * (3.0 - 2.0 * timestepProgress)

                var canvasValues: [Float] = []
                var canvasMaxValue: Float = 0.0
                
                for valueIndex in 0..<numValues {
                    // Interpolate between initial value and final value
                    let finalValue = finalDataset[valueIndex]
                    let baseInterpolatedValue = initialValue + (finalValue - initialValue) * cubicProgress
                    
                    // Add multiplicative noise that varies by timestep and position
                    let noiseIndex = timestep * numValues + valueIndex
                    let noiseSeed = Float(noiseIndex * 2654435761) / Float(UInt32.max)

                    // Combine multiple frequencies for more organic noise
                    let noise1 = sin(noiseSeed * 2 * Float.pi)
                    let noise2 = sin(noiseSeed * 4.7 * Float.pi + 1.3)
                    let noise3 = sin(noiseSeed * 9.2 * Float.pi + 2.7)
                    let combinedNoise = (noise1 + 0.5 * noise2 + 0.25 * noise3) / 1.75
                    
                    // Scale noise amount based on how far we are in the transition
                    let transitionWeight = sin(timestepProgress * Float.pi)
                    let noiseScale = 0.05 + 0.15 * transitionWeight
                    let noiseAmount = 1.0 + noiseScale * combinedNoise
                    
                    let interpolatedValue = baseInterpolatedValue * noiseAmount
                    canvasValues.append(interpolatedValue)
                    canvasMaxValue = max(canvasMaxValue, interpolatedValue)
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
            var allPointsForTimestep: [PointInstanceData] = []
            
            for (canvasIndex, (_, _, _, numValues, _)) in ScalarDistributionDemo.toyDistributions.enumerated() {
                guard canvasIndex < layoutCache.count else { continue }
                let layout = layoutCache[canvasIndex]
                
                let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))
                let plotHeight = CGFloat(canvasHeight)
                
                // Scale based on global max, not domain
                let scale: (Float) -> Float = { value in
                    Float(value / self.globalMaxValue)
                }
                
                let canvasValues = allValuesPerTimestep[timestep][canvasIndex]
                
                // For 1D visualization, all points are on a single horizontal line
                let yPixel = Float(plotHeight) / 2.0  // Center vertically
                
                for valueIndex in 0..<numValues {
                    let interpolatedValue = canvasValues[valueIndex]
                    
                    let xNormalized = scale(interpolatedValue)
                    let xPixel = Float(layout.minLabelWidth) + xNormalized * Float(layout.plotWidth)
                    
                    let pointData = PointInstanceData(
                        startPosition: SIMD2<Float>(xPixel, yPixel),
                        endPosition: SIMD2<Float>(xPixel, yPixel),
                        alpha: 1.0,
                        canvasIndex: UInt32(canvasIndex)
                    )
                    allPointsForTimestep.append(pointData)
                }
            }
            
            // Create GPU buffer for this timestep
            guard let buffer = device.makeBuffer(
                bytes: allPointsForTimestep,
                length: allPointsForTimestep.count * MemoryLayout<PointInstanceData>.stride,
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
    
    // Generate varied distributions for each parameter
    private func generateVariedDistribution(canvasIndex: Int, numValues: Int, domain: (Float, Float), scaleType: ScaleType) -> [Float] {
        var results: [Float] = []
        
        // Use canvas index to determine distribution type
        let isBimodal = (canvasIndex % 3) == 0
        
        if isBimodal {
            // Bimodal distribution
            let spacing = 0.2 + Float(canvasIndex % 5) * 0.1  // Distance between peaks (0.2 to 0.6)
            let center1 = domain.0 + (domain.1 - domain.0) * (0.5 - spacing/2)
            let center2 = domain.0 + (domain.1 - domain.0) * (0.5 + spacing/2)
            let variance = 0.0001 * (domain.1 - domain.0) * (domain.1 - domain.0)  // Very small variance
            
            for i in 0..<numValues {
                // Mix between the two modes
                let useFirst = (i % 2) == 0
                let center = useFirst ? center1 : center2
                let value = generateGaussianSample(mean: center, variance: variance)
                results.append(max(domain.0, min(domain.1, value)))
            }
        } else {
            // Single Gaussian with varying center and variance
            let centerPosition = 0.2 + Float(canvasIndex % 7) * 0.1  // Center varies from 0.2 to 0.8
            let center = domain.0 + (domain.1 - domain.0) * centerPosition
            
            // Variance varies but stays very small
            let varianceScale = 0.0001 + Float(canvasIndex % 4) * 0.00005  // 0.0001 to 0.0003
            let variance = varianceScale * (domain.1 - domain.0) * (domain.1 - domain.0)
            
            for _ in 0..<numValues {
                let value = generateGaussianSample(mean: center, variance: variance)
                results.append(max(domain.0, min(domain.1, value)))
            }
        }
        
        return results
    }
    
    // Box-Muller transform for Gaussian sampling
    private func generateGaussianSample(mean: Float, variance: Float) -> Float {
        let u1 = Float.random(in: 0.001...0.999)  // Avoid exact 0 or 1
        let u2 = Float.random(in: 0...1)
        let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
        return mean + sqrt(variance) * z0
    }

    private func initializeLayoutCache() {
        let fontSize: CGFloat = 10 * backingScaleFactor
        let labelPadding: CGFloat = 8 * backingScaleFactor
        
        layoutCache = []
        
        for (_, domain, scaleType, _, _) in ScalarDistributionDemo.toyDistributions {
            let minText = switch scaleType {
            case .linear: toPrecisionThrifty(domain.0, 2)
            case .log: String(format: "%.1e", domain.0)
            }
            let maxText = switch scaleType {
            case .linear: toPrecisionThrifty(domain.1, 2)
            case .log: String(format: "%.1e", domain.1)
            }
            
            let minLabelWidth: CGFloat = CGFloat(minText.count) * fontSize * 0.6 + labelPadding * 2
            let maxLabelWidth: CGFloat = CGFloat(maxText.count) * fontSize * 0.6 + labelPadding * 2
            // Use CANVAS_WIDTH as the maximum plot width for layout calculations
            let plotWidth = CGFloat(backingScaleFactor * CGFloat(CANVAS_WIDTH)) - minLabelWidth - maxLabelWidth
            
            layoutCache.append(CanvasLayoutCache(
                minText: minText,
                maxText: maxText,
                minLabelWidth: minLabelWidth,
                maxLabelWidth: maxLabelWidth,
                plotWidth: plotWidth
            ))
        }
    }
    
    // Calculate actual canvas widths based on data ranges
    private func calculateCanvasWidths() {
        guard !maxValuesPerTimestep.isEmpty, globalMaxValue > 0 else {
            // Fallback to default widths
            canvasActualWidths = Array(repeating: Int(CANVAS_WIDTH), count: N_DISTRIBUTIONS)
            return
        }
        
        canvasActualWidths = []
        
        for canvasIndex in 0..<N_DISTRIBUTIONS {
            // Find the maximum value across all timesteps for this canvas
            var canvasGlobalMax: Float = 0.0
            for timestep in 0..<numTimesteps {
                if timestep < maxValuesPerTimestep.count && canvasIndex < maxValuesPerTimestep[timestep].count {
                    canvasGlobalMax = max(canvasGlobalMax, maxValuesPerTimestep[timestep][canvasIndex])
                }
            }
            
            // Calculate the width ratio for this canvas
            let widthRatio = canvasGlobalMax / globalMaxValue
            
            // Calculate actual canvas width in logical pixels
            let layout = layoutCache[canvasIndex]
            // layout values are already in scaled pixels, so calculate the actual width in scaled pixels first
            let actualPlotWidthScaled = Float(layout.plotWidth) * widthRatio
            let actualCanvasWidthScaled = layout.minLabelWidth + CGFloat(actualPlotWidthScaled) + layout.maxLabelWidth
            // Convert to logical pixels for IOSurface creation
            let actualCanvasWidth = Int(ceil(actualCanvasWidthScaled / backingScaleFactor))
            
            canvasActualWidths.append(actualCanvasWidth)
        }
    }
    
    private func setupMetal() {
        guard let device = device else {
            print("Device not initialized")
            return
        }
        
        // Initialize character texture cache
        initializeCharacterTextureCache(device: device)
        createCharacterTextureArray(device: device)
        
        canvasPointCounts = []
        totalPointCount = 0

        for (_, _, _, numValues, _) in ScalarDistributionDemo.toyDistributions {
            canvasPointCounts.append(numValues)
            totalPointCount += numValues
        }

        let canvasHeight = Int(backingScaleFactor * CGFloat(CANVAS_HEIGHT))
        let textureArrayWidth = Int(backingScaleFactor * CGFloat(CANVAS_WIDTH))

        let currentBundle = Bundle(for: type(of: self))
        var library: MTLLibrary?
        do {
            library = try device.makeDefaultLibrary(bundle: currentBundle)
            print("Successfully loaded Metal library from class bundle: \(currentBundle.bundlePath)")
        } catch {
            print("Failed to load Metal library from class bundle \(currentBundle.bundlePath). Error: \(error)")
        }

        guard let library = library else {
            return
        }

        guard let vertexFunction = library.makeFunction(name: "scalar_dist_point_vertex"),
              let fragmentFunction = library.makeFunction(name: "scalar_dist_point_fragment") else {
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

        // Required when using render_target_array_index
        pipelineDescriptor.inputPrimitiveTopology = .point

        do {
            pointRenderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create point render pipeline state: \(error)")
            return
        }
        
        // Set up instanced gray rectangle rendering pipeline
        guard let instancedGrayRectVertexFunction = library.makeFunction(name: "scalar_dist_gray_rect_vertex"),
              let instancedGrayRectFragmentFunction = library.makeFunction(name: "scalar_dist_gray_rect_fragment") else {
            print("Failed to load instanced gray rect shader functions")
            return
        }
        
        let instancedGrayRectPipelineDescriptor = MTLRenderPipelineDescriptor()
        instancedGrayRectPipelineDescriptor.vertexFunction = instancedGrayRectVertexFunction
        instancedGrayRectPipelineDescriptor.fragmentFunction = instancedGrayRectFragmentFunction
        instancedGrayRectPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        instancedGrayRectPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        instancedGrayRectPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        instancedGrayRectPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        instancedGrayRectPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        instancedGrayRectPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        // Configure UI vertex descriptor
        let uiVertexDescriptor = MTLVertexDescriptor()
        uiVertexDescriptor.attributes[0].format = .float2
        uiVertexDescriptor.attributes[0].offset = 0
        uiVertexDescriptor.attributes[0].bufferIndex = 0
        uiVertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        instancedGrayRectPipelineDescriptor.vertexDescriptor = uiVertexDescriptor
        
        // Required when using render_target_array_index
        instancedGrayRectPipelineDescriptor.inputPrimitiveTopology = .triangle
        
        do {
            instancedGrayRectPipelineState = try device.makeRenderPipelineState(descriptor: instancedGrayRectPipelineDescriptor)
            print("Successfully created instanced gray rect render pipeline state")
        } catch {
            print("Failed to create instanced gray rect render pipeline state: \(error)")
            return
        }
        
        // Set up character-based text rendering pipeline
        guard let characterVertexFunction = library.makeFunction(name: "scalar_dist_character_vertex"),
              let characterFragmentFunction = library.makeFunction(name: "scalar_dist_character_fragment") else {
            print("Failed to load character shader functions")
            return
        }
        
        let characterPipelineDescriptor = MTLRenderPipelineDescriptor()
        characterPipelineDescriptor.vertexFunction = characterVertexFunction
        characterPipelineDescriptor.fragmentFunction = characterFragmentFunction
        characterPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        characterPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        characterPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        characterPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        characterPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        characterPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        characterPipelineDescriptor.vertexDescriptor = uiVertexDescriptor
        
        // Required when using render_target_array_index
        characterPipelineDescriptor.inputPrimitiveTopology = .triangle
        
        do {
            characterRenderPipelineState = try device.makeRenderPipelineState(descriptor: characterPipelineDescriptor)
            print("Successfully created character render pipeline state")
        } catch {
            print("Failed to create character render pipeline state: \(error)")
            return
        }
        
        // Set up end marker rendering pipeline
        guard let endMarkerVertexFunction = library.makeFunction(name: "scalar_dist_end_marker_vertex"),
              let endMarkerFragmentFunction = library.makeFunction(name: "scalar_dist_end_marker_fragment") else {
            print("Failed to load end marker shader functions")
            return
        }
        
        let endMarkerPipelineDescriptor = MTLRenderPipelineDescriptor()
        endMarkerPipelineDescriptor.vertexFunction = endMarkerVertexFunction
        endMarkerPipelineDescriptor.fragmentFunction = endMarkerFragmentFunction
        endMarkerPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        endMarkerPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        endMarkerPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        endMarkerPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        endMarkerPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        endMarkerPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        endMarkerPipelineDescriptor.vertexDescriptor = uiVertexDescriptor
        
        // Required when using render_target_array_index
        endMarkerPipelineDescriptor.inputPrimitiveTopology = .triangle
        
        do {
            endMarkerPipelineState = try device.makeRenderPipelineState(descriptor: endMarkerPipelineDescriptor)
            print("Successfully created end marker render pipeline state")
        } catch {
            print("Failed to create end marker render pipeline state: \(error)")
            return
        }

        // Create a simple quad vertex buffer for point rendering
        let vertices: [SIMD2<Float>] = [
            SIMD2<Float>(0.0, 0.0)
        ]
        pointVertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<SIMD2<Float>>.stride, options: [])

        // Create UI vertex buffer for rectangles (triangle strip order)
        let uiVertices: [SIMD2<Float>] = [
            SIMD2<Float>(0.0, 0.0),  // Bottom-left
            SIMD2<Float>(1.0, 0.0),  // Bottom-right  
            SIMD2<Float>(0.0, 1.0),  // Top-left
            SIMD2<Float>(1.0, 1.0),  // Top-right
        ]
        uiVertexBuffer = device.makeBuffer(bytes: uiVertices, length: uiVertices.count * MemoryLayout<SIMD2<Float>>.stride, options: [])
        
        // Create background uniforms buffer - use fixed CANVAS_WIDTH
        var backgroundUniforms = BackgroundUniforms(
            canvasSize: SIMD2<Float>(Float(textureArrayWidth), Float(canvasHeight)),
            grayRectFillColor: SIMD4<Float>(0.9, 0.9, 0.9, 1.0),
            strokeColor: SIMD4<Float>(0.7, 0.7, 0.7, 1.0)
        )
        
        backgroundUniformsBuffer = device.makeBuffer(bytes: &backgroundUniforms, 
                                                    length: MemoryLayout<BackgroundUniforms>.stride, 
                                                    options: [])
        
        // Create background instance buffer (will be updated each frame)
        backgroundInstanceBuffer = device.makeBuffer(length: N_DISTRIBUTIONS * MemoryLayout<BackgroundInstanceData>.stride, 
                                                    options: [])
        
        // Create character instance buffer for ALL characters across ALL canvases
        let maxTotalCharacters = N_DISTRIBUTIONS * 2 * 10
        characterInstanceBuffer = device.makeBuffer(length: maxTotalCharacters * MemoryLayout<CharacterInstanceData>.stride, options: [])
        
        // Create end marker buffers
        var endMarkerUniforms = EndMarkerUniforms(
            canvasSize: SIMD2<Float>(Float(textureArrayWidth), Float(canvasHeight)),
            fillColor: SIMD4<Float>(0.5, 0.5, 0.5, 1.0),  // Dark fill color
            strokeColor: SIMD4<Float>(0.7, 0.7, 0.7, 1.0)
        )
        endMarkerUniformsBuffer = device.makeBuffer(bytes: &endMarkerUniforms,
                                                   length: MemoryLayout<EndMarkerUniforms>.stride,
                                                   options: [])
        
        // Create end marker instance buffer (2 markers per canvas - start and end)
        endMarkerInstanceBuffer = device.makeBuffer(length: N_DISTRIBUTIONS * 2 * MemoryLayout<EndMarkerInstanceData>.stride,
                                                   options: [])

        // Create uniforms buffers for each canvas
        pointUniformsBuffers = []
        pointInstanceBuffers = []

        for _ in 0..<N_DISTRIBUTIONS {
            guard let uniformsBuffer = device.makeBuffer(length: MemoryLayout<PointUniforms>.stride, options: []) else {
                print("Failed to create uniforms buffer")
                return
            }
            pointUniformsBuffers.append(uniformsBuffer)

            // Create instance data buffer (will be updated each frame)
            let maxPoints = 1000 // Reasonable upper bound for points
            guard let instanceBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<PointInstanceData>.stride, options: []) else {
                print("Failed to create instance buffer")
                return
            }
            pointInstanceBuffers.append(instanceBuffer)
        }

        // Create texture array to hold all canvases - use fixed CANVAS_WIDTH
        let largeTextureDescriptor = MTLTextureDescriptor()
        largeTextureDescriptor.textureType = .type2DArray
        largeTextureDescriptor.pixelFormat = .bgra8Unorm
        largeTextureDescriptor.width = textureArrayWidth
        largeTextureDescriptor.height = canvasHeight
        largeTextureDescriptor.arrayLength = N_DISTRIBUTIONS
        largeTextureDescriptor.mipmapLevelCount = 1
        largeTextureDescriptor.usage = [.renderTarget, .shaderRead]

        largeTexture = device.makeTexture(descriptor: largeTextureDescriptor)

        // Create single uniforms buffer
        pointUniformsBuffer = device.makeBuffer(
            length: MemoryLayout<PointUniforms>.stride,
            options: []
        )

        // Create IOSurfaces and textures for each canvas with correct sizes
        for i in 0..<N_DISTRIBUTIONS {
            // Use the actual calculated width for this canvas
            let canvasWidth = Int(backingScaleFactor * CGFloat(canvasActualWidths[i]))
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
                continue
            }
            
            guard let texture = device.makeTexture(
                descriptor: textureDescriptor,
                iosurface: surface,
                plane: 0) else {
                print("Failed to create texture for IOSurface")
                continue
            }
            
            canvasSurfaces.append(surface)
            textureCache.append(texture)
        }
    }

    private struct PointUniforms {
        var viewportSize: SIMD2<Float>
        var pointRadius: Float
        var pointColor: SIMD4<Float>
        var progress: Float
    }
    
    private struct CharacterInstanceData {
        var transform: simd_float4x4
        var textureIndex: UInt32  // Index into character texture array or atlas
        var canvasIndex: UInt32   // Which canvas this character belongs to
    }
    
    // Instance data for background rectangles
    private struct BackgroundInstanceData {
        var canvasIndex: UInt32
        var visibleWidth: Float  // Dynamic width based on max value
        var plotOffset: Float    // X offset for gray rectangle
        var plotWidth: Float     // Dynamic width for gray rectangle
        var strokeWidth: Float   // Stroke width for the gray rectangle
    }
    
    private struct BackgroundUniforms {
        var canvasSize: SIMD2<Float>
        var grayRectFillColor: SIMD4<Float>
        var strokeColor: SIMD4<Float>
    }
    
    private struct PointInstanceData {
        var startPosition: SIMD2<Float>
        var endPosition: SIMD2<Float>
        var alpha: Float
        var canvasIndex: UInt32
    }
    
    private struct EndMarkerInstanceData {
        var canvasIndex: UInt32
        var xPosition: Float
        var markerWidth: Float
        var strokeWidth: Float
    }
    
    private struct EndMarkerUniforms {
        var canvasSize: SIMD2<Float>
        var fillColor: SIMD4<Float>
        var strokeColor: SIMD4<Float>
    }
    
    // Initialize character texture cache with digits, period, and minus sign
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
        print("Initialized character texture cache with \(characterTextureCache.textureArray.count) characters")
    }
    
    // Create a texture2d_array from individual character textures
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
        print("Created character texture array with \(arrayLength) textures")
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
    
    // Batch render all text for all canvases with optimized instanced rendering
    private func renderAllCanvasText(currentMaxValues: [Float], nextMaxValues: [Float], progress: Float, backgroundInstances: [BackgroundInstanceData], renderEncoder: MTLRenderCommandEncoder, device: MTLDevice) {
        guard let characterRenderPipelineState = characterRenderPipelineState,
              let uiVertexBuffer = uiVertexBuffer,
              let characterInstanceBuffer = characterInstanceBuffer,
              let characterTextureArray = characterTextureArray,
              characterTextureCache.initialized else { return }
        
        let charWidth = characterTextureCache.characterWidth
        let charHeight = characterTextureCache.characterHeight
        let totalCanvasWidth = CGFloat(CANVAS_WIDTH) * backingScaleFactor
        let totalCanvasHeight = CGFloat(CANVAS_HEIGHT) * backingScaleFactor
        
        // Helper function to create transformation matrix
        func createTransformMatrix(x: Float, y: Float, width: Float, height: Float) -> simd_float4x4 {
            let totalWidth = Float(totalCanvasWidth)
            let totalHeight = Float(totalCanvasHeight)
            
            // Convert to NDC coordinates
            let ndcX = (x / totalWidth) * 2.0 - 1.0
            let ndcY = 1.0 - ((y + height) / totalHeight) * 2.0
            let ndcWidth = (width / totalWidth) * 2.0
            let ndcHeight = (height / totalHeight) * 2.0
            
            return simd_float4x4(
                SIMD4<Float>(ndcWidth, 0, 0, 0),
                SIMD4<Float>(0, ndcHeight, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(ndcX, ndcY, 0, 1)
            )
        }
        
        // Collect all character instances across all canvases
        var allInstances: [CharacterInstanceData] = []
        allInstances.reserveCapacity(N_DISTRIBUTIONS * 8) // heuristic

        // Process each canvas
        for canvasIndex in 0..<N_DISTRIBUTIONS {
            guard canvasIndex < backgroundInstances.count else { break }

            let instance = backgroundInstances[canvasIndex]
            let interpolatedMax = currentMaxValues[canvasIndex] + (nextMaxValues[canvasIndex] - currentMaxValues[canvasIndex]) * Float(progress)
            let layout = layoutCache[canvasIndex]
            
            // Calculate dynamic plot width
            let dynamicPlotWidth = layout.plotWidth * CGFloat(Double(instance.plotWidth) / Double(layout.plotWidth))

            withUnsafeTemporaryAllocation(of: CChar.self, capacity: 64) { maxTextBuffer in
                // Populate max text buffer
                let scaleType = ScalarDistributionDemo.toyDistributions[canvasIndex].2
                let maxTextLen = switch scaleType {
                case .linear:
                    Int(toPrecisionThriftyC(interpolatedMax, 2, maxTextBuffer.baseAddress, 64))
                case .log:
                    Int(formatScientificC(interpolatedMax, maxTextBuffer.baseAddress, 64))
                }

                let minTextBuffer: [CChar] = [48, 0] // "0" in ASCII
                let minTextLen = 1

                // Calculate text positions
                let minTextWidth = CGFloat(minTextLen) * charWidth
                let maxTextWidth = CGFloat(maxTextLen) * charWidth
                let minTextX = (layout.minLabelWidth - minTextWidth) / 2
                let maxTextX = layout.minLabelWidth + dynamicPlotWidth + (layout.maxLabelWidth - maxTextWidth) / 2
                // Center text vertically - adjust for baseline positioning
                let textY = (totalCanvasHeight - charHeight) / 2 - charHeight * 0.1

                // Add min text characters (just "0")
                var currentX = minTextX
                for i in 0..<minTextLen {
                    let ascii = Int(minTextBuffer[i])
                    let textureIndex = characterTextureCache.asciiToIndex[ascii]
                    if textureIndex >= 0 {
                        let instanceData = CharacterInstanceData(
                            transform: createTransformMatrix(x: Float(currentX), y: Float(textY), width: Float(charWidth), height: Float(charHeight)),
                            textureIndex: UInt32(textureIndex),
                            canvasIndex: UInt32(canvasIndex)
                        )
                        allInstances.append(instanceData)
                    }
                    currentX += charWidth
                }

                // Add max text characters
                currentX = maxTextX
                for i in 0..<maxTextLen {
                    let ascii = Int(maxTextBuffer[i])
                    let textureIndex = characterTextureCache.asciiToIndex[ascii]
                    if textureIndex >= 0 {
                        let instanceData = CharacterInstanceData(
                            transform: createTransformMatrix(x: Float(currentX), y: Float(textY), width: Float(charWidth), height: Float(charHeight)),
                            textureIndex: UInt32(textureIndex),
                            canvasIndex: UInt32(canvasIndex)
                        )
                        allInstances.append(instanceData)
                    }
                    currentX += charWidth
                }
            }
        }
        
        guard !allInstances.isEmpty else { return }
        
        // Update the instance buffer with all character data
        characterInstanceBuffer.contents().copyMemory(
            from: allInstances,
            byteCount: allInstances.count * MemoryLayout<CharacterInstanceData>.stride
        )
        
        // Set up pipeline state and texture array
        renderEncoder.setRenderPipelineState(characterRenderPipelineState)
        // This is already set in "updateCanvases"
        // renderEncoder.setVertexBuffer(uiVertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(characterInstanceBuffer, offset: 0, index: 2)
        renderEncoder.setFragmentTexture(characterTextureArray, index: 0)
        renderEncoder.setFragmentBuffer(characterInstanceBuffer, offset: 0, index: 2)
        
        // Draw all instances in a single call
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: allInstances.count)
    }
}

private struct FragmentContainer {
    var attributedString: NSAttributedString
    var elementRanges: [String: NSRange]
}

private func generateFragmentContainer() -> FragmentContainer {
    let monospacedFont = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
    let attributedString = NSMutableAttributedString()
    var elementRanges = [String: NSRange]()
    
    for i in 0..<N_DISTRIBUTIONS {
        let (keyName, _, _, _, _) = ScalarDistributionDemo.toyDistributions[min(i, ScalarDistributionDemo.toyDistributions.count - 1)]
        
        // Create attributed string with brown color for the key name
        let lineString = NSMutableAttributedString()
        
        // Add "compare("
        let comparePrefix = "compare("
        lineString.append(NSAttributedString(string: comparePrefix, attributes: [
            .font: monospacedFont,
            .foregroundColor: NSColor.labelColor
        ]))
        
        // Add the key name in brown
        lineString.append(NSAttributedString(string: keyName, attributes: [
            .font: monospacedFont,
            .foregroundColor: NSColor.systemRed
        ]))
        
        // Add ") / " and the canvas placeholder
        let suffix = ") / \u{FFFC}\n"
        lineString.append(NSAttributedString(string: suffix, attributes: [
            .font: monospacedFont,
            .foregroundColor: NSColor.labelColor
        ]))
        
        let startPosition = attributedString.length
        attributedString.append(lineString)
        
        // Store element range (pointing to the canvas placeholder)
        let canvasLocation = startPosition + comparePrefix.count + keyName.count + 4  // ") / " is 4 characters
        let distRange = NSRange(location: canvasLocation, length: 1)
        elementRanges["dist\(i)"] = distRange
    }
    
    return FragmentContainer(attributedString: attributedString, elementRanges: elementRanges)
}

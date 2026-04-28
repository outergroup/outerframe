// Configuration constants matching Swift implementation
const N_DISTRIBUTIONS = 20;
const CANVAS_WIDTH = 400;
const CANVAS_HEIGHT = 30;
const ROW_PADDING = 2; // Padding between distributions

// Sample toy data for ML parameter distributions
const toyDistributions = [
    ["log_epochs", [0, 3.3], "linear", 16, 80, 0.33],
    ["log_batch_size", [0, 3.4], "linear", 16, 80, 0.33],
    ["log_conv1_weight_decay", [0, 3.5], "linear", 16, 80, 0.33],
    ["log_conv2_weight_decay", [0, 3.4], "linear", 16, 80, 0.33],
    ["log_conv3_weight_decay", [0, 3.5], "linear", 16, 80, 0.33],
    ["log_dense1_weight_decay", [0, 3.3], "linear", 16, 80, 0.33],
    ["log_dense2_weight_decay", [0, 3.6], "linear", 16, 80, 0.33],
    ["log_1cycle_initial_lr_pct", [0, 3.4], "linear", 16, 80, 0.33],
    ["log_1cycle_final_lr_pct", [0, 3.9], "linear", 16, 80, 0.33],
    ["log_1cycle_pct_warmup", [0, 3.9], "linear", 16, 80, 0.33],
    ["log_1cycle_max_lr", [0, 1.2], "linear", 16, 80, 0.33],
    ["log_1cycle_momentum_max_damping_factor", [0, 0.78], "linear", 16, 80, 0.33],
    ["log_1cycle_momentum_min_damping_factor_pct", [0, 3.3], "linear", 16, 80, 0.33],
    ["log_1cycle_beta1_max_damping_factor", [0, 2.1], "linear", 16, 80, 0.33],
    ["log_1cycle_beta1_min_damping_factor_pct", [0, 3.3], "linear", 16, 80, 0.33],
    ["log_beta2_damping_factor", [0, 3.4], "linear", 16, 80, 0.33],
    ["log_conv1_channels", [0, 4.3], "linear", 16, 80, 0.33],
    ["log_conv2_channels", [0, 4.1], "linear", 16, 80, 0.33],
    ["log_conv3_channels", [0, 3.8], "linear", 16, 80, 0.33],
    ["log_dense1_units", [0, 3.6], "linear", 16, 80, 0.33]
];

function toPrecisionThrifty(d, precision) {
    if (d >= 1.0 && (Math.log10(d) + 1) >= precision) {
        return d.toFixed(0);
    }

    const fullPrecision = d.toPrecision(precision);
    const parsedPrecise = parseFloat(fullPrecision);

    for (let i = 0; i < precision; i++) {
        const candidate = d.toPrecision(i + 1);
        if (parseFloat(candidate) === parsedPrecise) {
            return candidate;
        }
    }

    return fullPrecision;
}

class WanderingPointsVisualizationSingleCanvasWebGPU {
    constructor() {
        this.canvas = null;
        this.device = null;
        this.context = null;

        this.numTimesteps = 20;
        this.currentTimestep = 0;
        this.nextTimestep = 1;
        this.animationStartTime = 0;
        this.transitionDuration = 400;
        this.cycleInterval = 500;

        // Cache layout calculations
        this.layoutCache = [];

        // WebGPU resources
        this.pipelines = {};
        this.bindGroups = {};
        this.buffers = {};

        this.totalPointCount = 0;

        // Max values for each canvas at each timestep
        this.maxValuesPerTimestep = [];
        this.globalMaxValue = 0.0;

        // Character texture atlas
        this.characterTextureAtlas = null;
        this.characterTextureView = null;
        this.characterDimensions = null;

        // Canvas dimensions
        this.CANVAS_TOTAL_HEIGHT = N_DISTRIBUTIONS * CANVAS_HEIGHT + (N_DISTRIBUTIONS - 1) * ROW_PADDING;

        // FPS tracking
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        this.fps = 0;

        this.init();
    }

    async init() {
        if (!navigator.gpu) {
            this.showWebGPUError('WebGPU is not supported in this browser');
            return;
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            this.showWebGPUError('No GPU adapter available. WebGPU may be disabled in your browser settings.');
            return;
        }

        try {
            this.device = await adapter.requestDevice();
        } catch (error) {
            this.showWebGPUError('Failed to initialize GPU device. Error: ' + error.message);
            return;
        }

        this.createCanvas();
        this.createHTMLLabels();
        this.initializeLayoutCache();
        await this.initializeCharacterTextureAtlas();
        this.initializeData();
        await this.setupWebGPU();

        // Start animation after all initialization is complete
        this.startAnimation();
    }

    showWebGPUError(message) {
        const canvas = document.getElementById('visualization');
        const labelsColumn = document.getElementById('labels-column');
        
        // Hide the canvas and labels
        if (canvas) canvas.style.display = 'none';
        if (labelsColumn) labelsColumn.style.display = 'none';
        
        // Create error message
        const errorDiv = document.createElement('div');
        errorDiv.style.textAlign = 'center';
        errorDiv.style.padding = '40px';
        errorDiv.style.color = '#666';
        errorDiv.innerHTML = `
            <h3 style="color: #d00; margin-bottom: 20px;">WebGPU Not Available</h3>
            <p style="margin-bottom: 10px;">${message}</p>
            <p style="font-size: 0.9em; margin-top: 20px;">
                Try the <a href="ScalarDistributionList-WebGL-SingleCanvas.html">WebGL version</a> instead.
            </p>
        `;
        
        // Insert error message in the visualization wrapper
        const visualizationWrapper = document.querySelector('.visualization-wrapper');
        if (visualizationWrapper) {
            visualizationWrapper.innerHTML = '';
            visualizationWrapper.appendChild(errorDiv);
        }
    }

    createCanvas() {
        this.canvas = document.getElementById('visualization');
        const devicePixelRatio = window.devicePixelRatio || 1;
        this.canvas.width = CANVAS_WIDTH * devicePixelRatio;
        this.canvas.height = this.CANVAS_TOTAL_HEIGHT * devicePixelRatio;
        this.canvas.style.width = `${CANVAS_WIDTH}px`;
        this.canvas.style.height = `${this.CANVAS_TOTAL_HEIGHT}px`;
    }

    createHTMLLabels() {
        const labelsColumn = document.getElementById('labels-column');
        
        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [paramName] = toyDistributions[i];
            const label = document.createElement('div');
            label.className = 'label-item';
            
            // Create the formatted label with red parameter name
            label.innerHTML = `compare('<span class="param-name">${paramName}</span>') /`;
            
            labelsColumn.appendChild(label);
        }
    }

    async initializeCharacterTextureAtlas() {
        const characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', 'e', '+'];
        const fontSize = 10 * (window.devicePixelRatio || 1);
        const padding = 2 * (window.devicePixelRatio || 1);

        // Create canvas to render characters
        const measureCanvas = document.createElement('canvas');
        const measureCtx = measureCanvas.getContext('2d');
        measureCtx.font = `bold ${fontSize}px Menlo, Monaco, monospace`;

        // Measure character dimensions
        const metrics = measureCtx.measureText('0');
        const charWidth = Math.ceil(metrics.width) + padding * 2;
        const charHeight = Math.ceil(fontSize * 1.5);

        this.characterDimensions = {
            width: charWidth,
            height: charHeight,
            fontSize: fontSize
        };

        // Create texture atlas
        const atlasWidth = charWidth * characters.length;
        const atlasHeight = charHeight;

        const atlasCanvas = document.createElement('canvas');
        atlasCanvas.width = atlasWidth;
        atlasCanvas.height = atlasHeight;
        const atlasCtx = atlasCanvas.getContext('2d');

        // Fill with white background
        atlasCtx.fillStyle = 'white';
        atlasCtx.fillRect(0, 0, atlasWidth, atlasHeight);

        // Set up text rendering with bold font
        atlasCtx.font = `bold ${fontSize}px Menlo, Monaco, monospace`;
        atlasCtx.textAlign = 'center';
        atlasCtx.textBaseline = 'middle';
        atlasCtx.fillStyle = 'black';

        // Create ASCII to texture index lookup table
        this.asciiToTextureIndex = new Array(256).fill(-1);
        characters.forEach((char, index) => {
            this.asciiToTextureIndex[char.charCodeAt(0)] = index;
        });

        // Render each character
        characters.forEach((char, index) => {
            const x = index * charWidth + charWidth / 2;
            const y = atlasHeight / 2;
            atlasCtx.fillText(char, x, y);
        });


        // Create WebGPU texture from canvas
        const imageBitmap = await createImageBitmap(atlasCanvas);

        this.characterTextureAtlas = this.device.createTexture({
            size: [atlasWidth, atlasHeight, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture: this.characterTextureAtlas },
            [atlasWidth, atlasHeight]
        );

        this.characterTextureView = this.characterTextureAtlas.createView();
    }

    initializeLayoutCache() {
        const fontSize = 10;
        const labelPadding = 8;

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [, domain, scaleType] = toyDistributions[i];

            const minText = "0";
            const maxText = scaleType === "linear" ?
                toPrecisionThrifty(domain[1], 2) :
                domain[1].toExponential(1);

            const minLabelWidth = minText.length * fontSize * 0.6 + labelPadding * 2;
            const maxLabelWidth = maxText.length * fontSize * 0.6 + labelPadding * 2;
            const plotWidth = CANVAS_WIDTH - minLabelWidth - maxLabelWidth;

            this.layoutCache.push({
                minText: minText,
                maxText: maxText,
                minLabelWidth: minLabelWidth,
                maxLabelWidth: maxLabelWidth,
                plotWidth: plotWidth
            });
        }
    }

    initializeData() {
        // Similar to multi-canvas version - pre-compute all point positions
        this.maxValuesPerTimestep = [];
        this.globalMaxValue = 0.0;
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Calculate total point count
        this.totalPointCount = 0;
        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [, , , numExperiments, numOutcomes] = toyDistributions[i];
            this.totalPointCount += numExperiments * numOutcomes;
        }

        // Generate final datasets
        const finalDatasets = [];
        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [, domain, scaleType, numExperiments, numOutcomes] = toyDistributions[i];
            const dataset = this.generateGaussianData(numExperiments, numOutcomes, domain, scaleType);
            finalDatasets.push(dataset);
        }

        // First pass: compute all values and find global max
        const allValuesPerTimestep = [];

        for (let timestep = 0; timestep < this.numTimesteps; timestep++) {
            const valuesForTimestep = [];
            const maxValuesForTimestep = [];

            for (let distributionIndex = 0; distributionIndex < N_DISTRIBUTIONS; distributionIndex++) {
                const [, , , numExperiments, numOutcomes, initialValue] = toyDistributions[distributionIndex];
                const finalDataset = finalDatasets[distributionIndex];

                const timestepProgress = timestep / (this.numTimesteps - 1);
                const cubicProgress = timestepProgress * timestepProgress * (3.0 - 2.0 * timestepProgress);

                const distributionValues = [];
                let distributionMaxValue = 0.0;

                for (let experimentIndex = 0; experimentIndex < numExperiments; experimentIndex++) {
                    for (let outcomeIndex = 0; outcomeIndex < numOutcomes; outcomeIndex++) {
                        const finalValue = finalDataset[experimentIndex][outcomeIndex];
                        const baseInterpolatedValue = initialValue + (finalValue - initialValue) * cubicProgress;

                        const noiseIndex = distributionIndex * this.numTimesteps * numExperiments * numOutcomes + timestep * numExperiments * numOutcomes + experimentIndex * numOutcomes + outcomeIndex;
                        const noiseSeed = noiseIndex * 2654435761 / 4294967296;

                        const noise1 = Math.sin(noiseSeed * 2 * Math.PI);
                        const noise2 = Math.sin(noiseSeed * 4.7 * Math.PI + 1.3);
                        const noise3 = Math.sin(noiseSeed * 9.2 * Math.PI + 2.7);
                        const combinedNoise = (noise1 + 0.5 * noise2 + 0.25 * noise3) / 1.75;

                        const transitionWeight = Math.sin(timestepProgress * Math.PI);
                        const noiseScale = 0.05 + 0.15 * transitionWeight;
                        const noiseAmount = 1.0 + noiseScale * combinedNoise;

                        const interpolatedValue = baseInterpolatedValue * noiseAmount;
                        distributionValues.push(interpolatedValue);
                        distributionMaxValue = Math.max(distributionMaxValue, interpolatedValue);
                    }
                }

                valuesForTimestep.push(distributionValues);
                maxValuesForTimestep.push(distributionMaxValue);
                this.globalMaxValue = Math.max(this.globalMaxValue, distributionMaxValue);
            }

            allValuesPerTimestep.push(valuesForTimestep);
            this.maxValuesPerTimestep.push(maxValuesForTimestep);
        }

        // Store raw values for GPU upload
        this.allValuesPerTimestep = allValuesPerTimestep;
    }

    generateGaussianData(numExperiments, numOutcomes, domain, scaleType) {
        const results = [];

        for (let experimentIndex = 0; experimentIndex < numExperiments; experimentIndex++) {
            const experimentData = [];

            const experimentProgress = experimentIndex / Math.max(1, numExperiments - 1);
            const meanProgress = 0.2 + 0.6 * experimentProgress;

            let mean, variance;

            if (scaleType === "linear") {
                mean = domain[0] + meanProgress * (domain[1] - domain[0]);
                const relativeVariance = 0.05 + 0.1 * (1.0 - experimentProgress);
                variance = relativeVariance * (domain[1] - domain[0]);
            } else {
                const logMin = Math.log(domain[0]);
                const logMax = Math.log(domain[1]);
                const logMean = logMin + meanProgress * (logMax - logMin);
                mean = Math.exp(logMean);
                const logVarianceScale = 0.3 + 0.4 * (1.0 - experimentProgress);
                variance = logVarianceScale * logVarianceScale;
            }

            for (let outcomeIndex = 0; outcomeIndex < numOutcomes; outcomeIndex++) {
                let value;
                if (scaleType === "linear") {
                    value = this.generateGaussianSample(mean, variance);
                } else {
                    const logMean = Math.log(mean);
                    const logStdDev = Math.sqrt(variance);
                    const logValue = this.generateGaussianSample(logMean, logStdDev * logStdDev);
                    value = Math.exp(logValue);
                }
                const clampedValue = Math.max(domain[0], Math.min(domain[1], value));
                experimentData.push(clampedValue);
            }

            results.push(experimentData);
        }

        return results;
    }

    generateGaussianSample(mean, variance) {
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return mean + Math.sqrt(variance) * z0;
    }

    async setupWebGPU() {
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Configure canvas context
        this.context = this.canvas.getContext('webgpu');
        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: canvasFormat,
            alphaMode: 'premultiplied',
        });

        // Create pipelines
        await this.createPipelines();

        // Create buffers
        this.createBuffers();

        // Create bind groups
        this.createBindGroups();
    }

    async createPipelines() {
        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        // Background rendering pipeline with instanced rendering
        const backgroundShaderModule = this.device.createShaderModule({
            code: `
                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) uv: vec2f,
                    @location(1) rect: vec4f,
                    @location(2) fillColor: vec4f,
                    @location(3) strokeColor: vec4f,
                    @location(4) cornerRadius: f32,
                    @location(5) strokeWidth: f32,
                }

                struct Uniforms {
                    viewportSize: vec2f,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vs_main(
                    @builtin(vertex_index) vertexIndex: u32,
                    @builtin(instance_index) instanceIndex: u32,
                    @location(0) quadPos: vec2f,
                    @location(1) rect: vec4f,  // x, y, width, height
                    @location(2) fillColor: vec4f,
                    @location(3) strokeColor: vec4f,
                    @location(4) cornerRadius: f32,
                    @location(5) strokeWidth: f32
                ) -> VertexOutput {
                    var output: VertexOutput;

                    let worldPos = rect.xy + quadPos * rect.zw;
                    let ndcX = (worldPos.x / uniforms.viewportSize.x) * 2.0 - 1.0;
                    let ndcY = 1.0 - (worldPos.y / uniforms.viewportSize.y) * 2.0;

                    output.position = vec4f(ndcX, ndcY, 0.0, 1.0);
                    output.uv = quadPos;
                    output.rect = rect;
                    output.fillColor = fillColor;
                    output.strokeColor = strokeColor;
                    output.cornerRadius = cornerRadius;
                    output.strokeWidth = strokeWidth;

                    return output;
                }

                fn roundedRectSDF(p: vec2f, size: vec2f, radius: f32) -> f32 {
                    // Clamp radius to prevent artifacts
                    let r = min(radius, min(size.x, size.y));
                    let d = abs(p) - size + vec2f(r);
                    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0))) - r;
                }

                @fragment
                fn fs_main(input: VertexOutput) -> @location(0) vec4f {
                    let pixelPos = input.uv * input.rect.zw;
                    let p = pixelPos - input.rect.zw * 0.5;

                    let d = roundedRectSDF(p, input.rect.zw * 0.5, input.cornerRadius);

                    // Calculate alpha for fill and stroke
                    let fillAlpha = smoothstep(1.0, -1.0, d);
                    let strokeAlpha = smoothstep(input.strokeWidth, -input.strokeWidth, abs(d)) *
                                    (1.0 - smoothstep(-1.0, 0.0, d));

                    // Mix fill and stroke colors without conditional
                    let fillColor = input.fillColor * fillAlpha;
                    let finalColor = mix(fillColor, input.strokeColor * strokeAlpha, strokeAlpha);

                    return finalColor;
                }
            `
        });

        this.pipelines.background = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: backgroundShaderModule,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        // Quad vertices (per-vertex)
                        arrayStride: 8, // 2 floats per vertex
                        stepMode: 'vertex',
                        attributes: [{
                            format: 'float32x2',
                            offset: 0,
                            shaderLocation: 0, // quadPos
                        }],
                    },
                    {
                        // Instance data (per-instance)
                        arrayStride: 56, // 14 floats per instance (4 for rect + 4 for fillColor + 4 for strokeColor + 1 for cornerRadius + 1 for strokeWidth)
                        stepMode: 'instance',
                        attributes: [
                            {
                                format: 'float32x4',
                                offset: 0,
                                shaderLocation: 1, // rect
                            },
                            {
                                format: 'float32x4',
                                offset: 16,
                                shaderLocation: 2, // fillColor
                            },
                            {
                                format: 'float32x4',
                                offset: 32,
                                shaderLocation: 3, // strokeColor
                            },
                            {
                                format: 'float32',
                                offset: 48,
                                shaderLocation: 4, // cornerRadius
                            },
                            {
                                format: 'float32',
                                offset: 52,
                                shaderLocation: 5, // strokeWidth
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: backgroundShaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: canvasFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });

        // Simple rectangle fill pipeline with instanced rendering
        const simpleRectFillShaderModule = this.device.createShaderModule({
            code: `
                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) color: vec4f,
                }

                struct Uniforms {
                    viewportSize: vec2f,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vs_main(
                    @builtin(vertex_index) vertexIndex: u32,
                    @builtin(instance_index) instanceIndex: u32,
                    @location(0) quadPos: vec2f,
                    @location(1) rect: vec4f,  // x, y, width, height
                    @location(2) color: vec4f
                ) -> VertexOutput {
                    var output: VertexOutput;

                    let worldPos = rect.xy + quadPos * rect.zw;
                    let ndcX = (worldPos.x / uniforms.viewportSize.x) * 2.0 - 1.0;
                    let ndcY = 1.0 - (worldPos.y / uniforms.viewportSize.y) * 2.0;

                    output.position = vec4f(ndcX, ndcY, 0.0, 1.0);
                    output.color = color;

                    return output;
                }

                @fragment
                fn fs_main(input: VertexOutput) -> @location(0) vec4f {
                    return input.color;
                }
            `
        });

        this.pipelines.simpleRectFill = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: simpleRectFillShaderModule,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        // Quad vertices (per-vertex)
                        arrayStride: 8, // 2 floats per vertex
                        stepMode: 'vertex',
                        attributes: [{
                            format: 'float32x2',
                            offset: 0,
                            shaderLocation: 0, // quadPos
                        }],
                    },
                    {
                        // Instance data (per-instance)
                        arrayStride: 32, // 8 floats per instance (4 for rect + 4 for color)
                        stepMode: 'instance',
                        attributes: [
                            {
                                format: 'float32x4',
                                offset: 0,
                                shaderLocation: 1, // rect
                            },
                            {
                                format: 'float32x4',
                                offset: 16,
                                shaderLocation: 2, // color
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: simpleRectFillShaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: canvasFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });

        // Simple rectangle stroke pipeline with instanced rendering
        const simpleRectStrokeShaderModule = this.device.createShaderModule({
            code: `
                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) uv: vec2f,
                    @location(1) rect: vec4f,
                    @location(2) strokeColor: vec4f,
                    @location(3) strokeWidth: f32,
                }

                struct Uniforms {
                    viewportSize: vec2f,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vs_main(
                    @builtin(vertex_index) vertexIndex: u32,
                    @builtin(instance_index) instanceIndex: u32,
                    @location(0) quadPos: vec2f,
                    @location(1) rect: vec4f,  // x, y, width, height
                    @location(2) strokeColor: vec4f,
                    @location(3) strokeWidth: f32
                ) -> VertexOutput {
                    var output: VertexOutput;

                    let worldPos = rect.xy + quadPos * rect.zw;
                    let ndcX = (worldPos.x / uniforms.viewportSize.x) * 2.0 - 1.0;
                    let ndcY = 1.0 - (worldPos.y / uniforms.viewportSize.y) * 2.0;

                    output.position = vec4f(ndcX, ndcY, 0.0, 1.0);
                    output.uv = quadPos;
                    output.rect = rect;
                    output.strokeColor = strokeColor;
                    output.strokeWidth = strokeWidth;

                    return output;
                }

                @fragment
                fn fs_main(input: VertexOutput) -> @location(0) vec4f {
                    let pixelPos = input.uv * input.rect.zw;

                    let distFromLeft = pixelPos.x;
                    let distFromRight = input.rect.z - pixelPos.x;
                    let distFromTop = pixelPos.y;
                    let distFromBottom = input.rect.w - pixelPos.y;

                    let minDist = min(min(distFromLeft, distFromRight), min(distFromTop, distFromBottom));

                    if (minDist < input.strokeWidth) {
                        return input.strokeColor;
                    }

                    // Outside stroke area - make transparent
                    return vec4f(0.0, 0.0, 0.0, 0.0);
                }
            `
        });

        this.pipelines.simpleRectStroke = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: simpleRectStrokeShaderModule,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        // Quad vertices (per-vertex)
                        arrayStride: 8, // 2 floats per vertex
                        stepMode: 'vertex',
                        attributes: [{
                            format: 'float32x2',
                            offset: 0,
                            shaderLocation: 0, // quadPos
                        }],
                    },
                    {
                        // Instance data (per-instance)
                        arrayStride: 36, // 9 floats per instance (4 for rect + 4 for color + 1 for strokeWidth)
                        stepMode: 'instance',
                        attributes: [
                            {
                                format: 'float32x4',
                                offset: 0,
                                shaderLocation: 1, // rect
                            },
                            {
                                format: 'float32x4',
                                offset: 16,
                                shaderLocation: 2, // strokeColor
                            },
                            {
                                format: 'float32',
                                offset: 32,
                                shaderLocation: 3, // strokeWidth
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: simpleRectStrokeShaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: canvasFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });

        // Point rendering pipeline with instanced rendering
        const pointShaderModule = this.device.createShaderModule({
            code: `
                struct Uniforms {
                    viewportSize: vec2f,
                    pointRadius: f32,
                    progress: f32,
                    globalMaxValue: f32,
                }

                struct DistributionData {
                    rowY: f32,
                    minLabelWidth: f32,
                    plotWidth: f32,      // Static plot width for positioning
                    rowHeight: f32,
                    numExperiments: f32,
                    numOutcomes: f32,
                    pointStartIndex: f32, // Start index in the values arrays
                    dynamicPlotWidth: f32, // Dynamic plot width for clipping
                }

                struct PointData {
                    value: f32,
                    distIndex: f32,
                    experimentIndex: f32,
                    outcomeIndex: f32,
                }

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) pointCoord: vec2f,
                    @location(1) color: vec4f,
                    @location(2) centerPosition: vec2f,
                    @location(3) plotBounds: vec4f,  // minX, maxX, minY, maxY
                    @location(4) pointRadius: f32,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                @group(0) @binding(1) var<storage, read> startPointData: array<PointData>;
                @group(0) @binding(2) var<storage, read> endPointData: array<PointData>;
                @group(0) @binding(3) var<storage, read> distributionData: array<DistributionData>;

                @vertex
                fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
                    // Get point data directly - no loop needed!
                    let startPoint = startPointData[instanceIndex];
                    let endPoint = endPointData[instanceIndex];
                    let distIndex = u32(startPoint.distIndex);
                    let dist = distributionData[distIndex];

                    // Get interpolated value
                    let interpolatedValue = mix(startPoint.value, endPoint.value, uniforms.progress);

                    // Calculate position directly from stored indices
                    let yNormalized = (startPoint.experimentIndex + 0.5) / dist.numExperiments;
                    let yPixel = dist.rowY + yNormalized * dist.rowHeight;

                    // Scale based on global max
                    let xNormalized = interpolatedValue / uniforms.globalMaxValue;
                    let xPixel = dist.minLabelWidth + xNormalized * dist.plotWidth;

                    // Convert to NDC
                    let ndcX = (xPixel / uniforms.viewportSize.x) * 2.0 - 1.0;
                    let ndcY = 1.0 - (yPixel / uniforms.viewportSize.y) * 2.0;

                    var output: VertexOutput;
                    output.position = vec4f(ndcX, ndcY, 0.0, 1.0);
                    output.color = vec4f(0.2, 0.4, 0.8, 0.7);
                    output.centerPosition = vec2f(xPixel, yPixel);
                    // Stroke extends 0.5 pixels on each side
                    let strokeHalfWidth = 0.5;
                    output.plotBounds = vec4f(dist.minLabelWidth + strokeHalfWidth, 
                                              dist.minLabelWidth + dist.dynamicPlotWidth + strokeHalfWidth, 
                                              dist.rowY, dist.rowY + dist.rowHeight);
                    output.pointRadius = uniforms.pointRadius;

                    // Generate point quad vertices for triangle strip
                    let quadOffsets = array<vec2f, 4>(
                        vec2f(-1.0, -1.0),
                        vec2f( 1.0, -1.0),
                        vec2f(-1.0,  1.0),
                        vec2f( 1.0,  1.0)
                    );

                    let offset = quadOffsets[vertexIndex % 4u] * uniforms.pointRadius;
                    output.position.x += offset.x * 2.0 / uniforms.viewportSize.x;
                    output.position.y -= offset.y * 2.0 / uniforms.viewportSize.y;
                    output.pointCoord = (quadOffsets[vertexIndex % 4u] + 1.0) * 0.5;

                    return output;
                }

                @fragment
                fn fs_main(input: VertexOutput) -> @location(0) vec4f {
                    // Calculate the actual pixel position of this fragment
                    let offset = (input.pointCoord - vec2f(0.5)) * input.pointRadius * 2.0;
                    let fragmentPos = input.centerPosition + offset;
                    
                    // Check if fragment is outside plot bounds
                    // Stroke width adjustment is already done on CPU side
                    if (fragmentPos.x < input.plotBounds.x || fragmentPos.x > input.plotBounds.y ||
                        fragmentPos.y < input.plotBounds.z || fragmentPos.y > input.plotBounds.w) {
                        discard;
                    }

                    let center = vec2f(0.5, 0.5);
                    let dist = length(input.pointCoord - center);
                    let alpha = 1.0 - smoothstep(0.4, 0.5, dist);
                    // Further reduce alpha for better transparency
                    let finalAlpha = alpha * 0.8;

                    return vec4f(input.color.rgb, input.color.a * finalAlpha);
                }
            `
        });

        this.pipelines.points = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: pointShaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: pointShaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: canvasFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });

        // Text rendering pipeline with instancing
        const textShaderModule = this.device.createShaderModule({
            code: `
                struct TextUniforms {
                    viewportSize: vec2f,
                    atlasCharWidth: f32,
                    atlasCharHeight: f32,
                }

                struct CharacterData {
                    x: f32,
                    y: f32,
                    width: f32,
                    height: f32,
                    charIndex: f32,
                }

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) texCoord: vec2f,
                }

                @group(0) @binding(0) var<uniform> uniforms: TextUniforms;
                @group(0) @binding(1) var<storage, read> charData: array<CharacterData>;
                @group(0) @binding(2) var characterTexture: texture_2d<f32>;
                @group(0) @binding(3) var characterSampler: sampler;

                @vertex
                fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
                    var pos = array<vec2f, 4>(
                        vec2f(0.0, 0.0),
                        vec2f(1.0, 0.0),
                        vec2f(0.0, 1.0),
                        vec2f(1.0, 1.0)
                    );

                    var output: VertexOutput;
                    let p = pos[vertexIndex];

                    // Get character data for this instance
                    let char = charData[instanceIndex];

                    let worldPos = vec2f(char.x, char.y) + p * vec2f(char.width, char.height);
                    let ndcX = (worldPos.x / uniforms.viewportSize.x) * 2.0 - 1.0;
                    let ndcY = 1.0 - (worldPos.y / uniforms.viewportSize.y) * 2.0;

                    output.position = vec4f(ndcX, ndcY, 0.0, 1.0);

                    // Calculate texture coordinates for this character in the atlas
                    let charStartU = char.charIndex / 14.0; // 14 characters in atlas
                    let charWidth = 1.0 / 14.0;

                    // p is the vertex position (0,0 to 1,1), we need to map it to the character's portion of the atlas
                    let atlasU = charStartU + p.x * charWidth;
                    let atlasV = p.y;
                    output.texCoord = vec2f(atlasU, atlasV);

                    return output;
                }

                @fragment
                fn fs_main(input: VertexOutput) -> @location(0) vec4f {
                    let texColor = textureSample(characterTexture, characterSampler, input.texCoord);
                    let alpha = 1.0 - texColor.r;  // Black text on white background
                    return vec4f(0.6, 0.6, 0.6, alpha);  // Gray text (0.6, 0.6, 0.6)
                }
            `
        });

        this.pipelines.text = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: textShaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: textShaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: canvasFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });
    }

    createBuffers() {
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Create quad vertex buffer for rectangle rendering
        const quadVertices = new Float32Array([
            0, 0,  // bottom-left
            1, 0,  // bottom-right
            0, 1,  // top-left
            1, 1   // top-right
        ]);

        this.buffers.quadVertices = this.device.createBuffer({
            size: quadVertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.buffers.quadVertices, 0, quadVertices);

        // Create viewport uniform buffer for background shader
        this.buffers.backgroundViewportUniform = this.device.createBuffer({
            size: 16, // vec2f + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });


        // Create merged point data buffers for each timestep with metadata
        this.buffers.mergedPointData = [];
        for (let timestep = 0; timestep < this.numTimesteps; timestep++) {
            // Create array of PointData structs (4 elements per point: value, distIndex, experimentIndex, outcomeIndex)
            const pointData = [];
            
            for (let dist = 0; dist < N_DISTRIBUTIONS; dist++) {
                const [, , , numExperiments, numOutcomes] = toyDistributions[dist];
                const distValues = this.allValuesPerTimestep[timestep][dist];
                let valueIndex = 0;
                
                for (let exp = 0; exp < numExperiments; exp++) {
                    for (let outcome = 0; outcome < numOutcomes; outcome++) {
                        pointData.push(
                            distValues[valueIndex++], // value
                            dist,                     // distIndex (as u32)
                            exp,                      // experimentIndex (as u32)
                            outcome                   // outcomeIndex (as u32)
                        );
                    }
                }
            }

            const buffer = this.device.createBuffer({
                size: pointData.length * 4, // 4 bytes per element
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            this.device.queue.writeBuffer(buffer, 0, new Float32Array(pointData));
            this.buffers.mergedPointData.push(buffer);
        }

        // Create distribution data buffer for instanced point rendering
        const distributionData = new Float32Array(N_DISTRIBUTIONS * 8); // 8 floats per distribution (struct size)
        let pointStartIndex = 0;

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [, , , numExperiments, numOutcomes] = toyDistributions[i];
            const offset = i * 8;

            // This data will be filled in renderAllPointsBatched with dynamic values
            distributionData[offset] = 0;     // rowY - will be filled dynamically
            distributionData[offset + 1] = 0; // minLabelWidth - will be filled dynamically
            distributionData[offset + 2] = 0; // plotWidth - will be filled dynamically
            distributionData[offset + 3] = 0; // rowHeight - will be filled dynamically
            distributionData[offset + 4] = numExperiments;
            distributionData[offset + 5] = numOutcomes;
            distributionData[offset + 6] = pointStartIndex;
            distributionData[offset + 7] = 0; // padding

            pointStartIndex += numExperiments * numOutcomes;
        }

        this.buffers.distributionData = this.device.createBuffer({
            size: distributionData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.buffers.distributionData, 0, distributionData);

        // Create point uniforms buffer for instanced rendering
        this.buffers.pointUniforms = this.device.createBuffer({
            size: 32, // Smaller uniform buffer for global data only
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create character data buffer for text (max characters)
        const maxCharacters = N_DISTRIBUTIONS * 20; // Estimate max 20 chars per distribution
        this.buffers.characterData = this.device.createBuffer({
            size: maxCharacters * 5 * 4, // 5 floats per character
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Create text uniform buffer
        this.buffers.textUniforms = this.device.createBuffer({
            size: 64, // Aligned size to 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create sampler for text
        this.textSampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });
    }

    createBindGroups() {

        // Create bind groups for instanced point rendering
        this.bindGroups.pointsByTimestep = [];

        for (let currentTs = 0; currentTs < this.numTimesteps; currentTs++) {
            const nextTs = (currentTs + 1) % this.numTimesteps;

            this.bindGroups.pointsByTimestep[currentTs] = this.device.createBindGroup({
                layout: this.pipelines.points.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.buffers.pointUniforms },
                    },
                    {
                        binding: 1,
                        resource: { buffer: this.buffers.mergedPointData[currentTs] },
                    },
                    {
                        binding: 2,
                        resource: { buffer: this.buffers.mergedPointData[nextTs] },
                    },
                    {
                        binding: 3,
                        resource: { buffer: this.buffers.distributionData },
                    },
                ],
            });
        }

        // Cache for text bind group
        this.bindGroups.text = null;
    }

    collectCharactersForText(characters, text, centerX, centerY) {
        if (!this.characterDimensions) return;

        const devicePixelRatio = window.devicePixelRatio || 1;
        const charWidth = this.characterDimensions.width;
        const charHeight = this.characterDimensions.height;
        const charSpacing = charWidth * 0.6;
        const totalWidth = charWidth + (text.length - 1) * charSpacing;

        let x = centerX - totalWidth / 2;
        const y = centerY - charHeight / 2 + 1 * devicePixelRatio; // Small offset to center better

        for (let i = 0; i < text.length; i++) {
            const textureIndex = this.asciiToTextureIndex[text.charCodeAt(i)];

            if (textureIndex !== -1) {
                characters.push({
                    x: x,
                    y: y,
                    width: charWidth,
                    height: charHeight,
                    charIndex: textureIndex
                });
            }

            x += charSpacing;
        }
    }

    render(progress) {
        const devicePixelRatio = window.devicePixelRatio || 1;
        const encoder = this.device.createCommandEncoder();

        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        // Prepare data for all distributions
        const allBackgrounds = [];
        const allCharacters = [];

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const layout = this.layoutCache[i];
            const [, , scaleType, numExperiments, numOutcomes] = toyDistributions[i];
            const rowY = i * (CANVAS_HEIGHT + ROW_PADDING) * devicePixelRatio;

            // Interpolate max value
            const currentMax = this.maxValuesPerTimestep[this.currentTimestep][i];
            const nextMax = this.maxValuesPerTimestep[this.nextTimestep][i];
            const interpolatedMax = currentMax + (nextMax - currentMax) * progress;

            // Calculate dynamic width
            const widthRatio = interpolatedMax / this.globalMaxValue;
            const dynamicPlotWidth = layout.plotWidth * widthRatio;
            const visibleWidth = layout.minLabelWidth + dynamicPlotWidth + layout.maxLabelWidth;

            // Collect background rectangles
            allBackgrounds.push({
                rect: [0, rowY, visibleWidth * devicePixelRatio, CANVAS_HEIGHT * devicePixelRatio],
                fillColor: [1.0, 1.0, 1.0, 1.0],
                strokeColor: [0.7, 0.7, 0.7, 1.0],
                cornerRadius: 5 * devicePixelRatio,
                strokeWidth: 1 * devicePixelRatio
            });

            // Collect gray plot area (with fill and stroke)
            allBackgrounds.push({
                rect: [layout.minLabelWidth * devicePixelRatio, rowY,
                       dynamicPlotWidth * devicePixelRatio, CANVAS_HEIGHT * devicePixelRatio],
                fillColor: [0.9, 0.9, 0.9, 1.0],
                strokeColor: [0.7, 0.7, 0.7, 1.0],
                cornerRadius: 0,  // No rounded corners for gray area
                strokeWidth: 1 * devicePixelRatio
            });

            // Collect text characters
            const textY = rowY + (CANVAS_HEIGHT / 2) * devicePixelRatio;

            // Min text
            const minText = "0";
            const minCenterX = (layout.minLabelWidth / 2) * devicePixelRatio;
            this.collectCharactersForText(allCharacters, minText, minCenterX, textY);

            // Max text
            const maxText = scaleType === "linear" ?
                toPrecisionThrifty(interpolatedMax, 2) :
                interpolatedMax.toExponential(1);
            const maxCenterX = (layout.minLabelWidth + dynamicPlotWidth + layout.maxLabelWidth / 2) * devicePixelRatio;
            this.collectCharactersForText(allCharacters, maxText, maxCenterX, textY);
        }

        // Render all backgrounds (including gray areas with stroke)
        this.renderBatchedBackgrounds(renderPass, allBackgrounds);

        // Render all points for all distributions
        this.renderAllPointsBatched(renderPass, progress);

        // Render all text in one instanced draw call
        this.renderBatchedText(renderPass, allCharacters);

        renderPass.end();

        // Submit the command buffer
        this.device.queue.submit([encoder.finish()]);
    }

    renderBatchedBackgrounds(renderPass, backgrounds) {
        if (backgrounds.length === 0) return;

        const devicePixelRatio = window.devicePixelRatio || 1;

        // Update viewport uniform
        const viewportData = new Float32Array([
            this.canvas.width,
            this.canvas.height,
            0, 0 // padding
        ]);
        this.device.queue.writeBuffer(this.buffers.backgroundViewportUniform, 0, viewportData);

        // Create instance data buffer
        const instanceData = new Float32Array(backgrounds.length * 14); // 14 floats per instance

        for (let i = 0; i < backgrounds.length; i++) {
            const bg = backgrounds[i];
            const offset = i * 14;
            instanceData.set([
                ...bg.rect,         // 4 floats
                ...bg.fillColor,    // 4 floats
                ...bg.strokeColor,  // 4 floats
                bg.cornerRadius,    // 1 float
                bg.strokeWidth      // 1 float
            ], offset);
        }

        // Create or update instance buffer
        if (!this.buffers.backgroundInstances || this.buffers.backgroundInstances.size < instanceData.byteLength) {
            if (this.buffers.backgroundInstances) {
                this.buffers.backgroundInstances.destroy();
            }
            this.buffers.backgroundInstances = this.device.createBuffer({
                size: instanceData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
        }

        this.device.queue.writeBuffer(this.buffers.backgroundInstances, 0, instanceData);

        // Create bind group for viewport uniform if not exists
        if (!this.bindGroups.backgroundViewport) {
            this.bindGroups.backgroundViewport = this.device.createBindGroup({
                layout: this.pipelines.background.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: { buffer: this.buffers.backgroundViewportUniform },
                }],
            });
        }

        // Set pipeline and vertex buffers
        renderPass.setPipeline(this.pipelines.background);
        renderPass.setVertexBuffer(0, this.buffers.quadVertices);
        renderPass.setVertexBuffer(1, this.buffers.backgroundInstances);
        renderPass.setBindGroup(0, this.bindGroups.backgroundViewport);

        // Draw all backgrounds in one call
        renderPass.draw(4, backgrounds.length);
    }

    renderBatchedPlotAreaFills(renderPass, plotAreaFills) {
        if (plotAreaFills.length === 0) return;

        const devicePixelRatio = window.devicePixelRatio || 1;

        // Create instance data buffer
        const instanceData = new Float32Array(plotAreaFills.length * 8); // 4 for rect, 4 for color

        for (let i = 0; i < plotAreaFills.length; i++) {
            const area = plotAreaFills[i];
            instanceData.set([
                ...area.rect,
                ...area.fillColor
            ], i * 8);
        }

        // Create or update instance buffer
        if (!this.buffers.plotAreaFillInstances || this.buffers.plotAreaFillInstances.size < instanceData.byteLength) {
            if (this.buffers.plotAreaFillInstances) {
                this.buffers.plotAreaFillInstances.destroy();
            }
            this.buffers.plotAreaFillInstances = this.device.createBuffer({
                size: instanceData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
        }

        this.device.queue.writeBuffer(this.buffers.plotAreaFillInstances, 0, instanceData);

        // Create bind group for viewport uniform
        if (!this.bindGroups.simpleRectFillViewport) {
            const viewportBuffer = this.device.createBuffer({
                size: 16, // vec2f + padding
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.buffers.simpleRectFillViewport = viewportBuffer;

            const viewportData = new Float32Array([this.canvas.width, this.canvas.height, 0, 0]);
            this.device.queue.writeBuffer(viewportBuffer, 0, viewportData);

            this.bindGroups.simpleRectFillViewport = this.device.createBindGroup({
                layout: this.pipelines.simpleRectFill.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: { buffer: viewportBuffer },
                }],
            });
        }

        // Set pipeline and draw all instances in one call
        renderPass.setPipeline(this.pipelines.simpleRectFill);
        renderPass.setVertexBuffer(0, this.buffers.quadVertices);
        renderPass.setVertexBuffer(1, this.buffers.plotAreaFillInstances);
        renderPass.setBindGroup(0, this.bindGroups.simpleRectFillViewport);
        renderPass.draw(4, plotAreaFills.length);
    }

    renderBatchedPlotAreaStrokes(renderPass, plotAreaStrokes) {
        if (plotAreaStrokes.length === 0) return;

        const devicePixelRatio = window.devicePixelRatio || 1;

        // Create instance data buffer
        const instanceData = new Float32Array(plotAreaStrokes.length * 9); // 4 for rect, 4 for color, 1 for strokeWidth

        for (let i = 0; i < plotAreaStrokes.length; i++) {
            const stroke = plotAreaStrokes[i];
            instanceData.set([
                ...stroke.rect,
                ...stroke.strokeColor,
                stroke.strokeWidth
            ], i * 9);
        }

        // Create or update instance buffer
        if (!this.buffers.plotAreaStrokeInstances || this.buffers.plotAreaStrokeInstances.size < instanceData.byteLength) {
            if (this.buffers.plotAreaStrokeInstances) {
                this.buffers.plotAreaStrokeInstances.destroy();
            }
            this.buffers.plotAreaStrokeInstances = this.device.createBuffer({
                size: instanceData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
        }

        this.device.queue.writeBuffer(this.buffers.plotAreaStrokeInstances, 0, instanceData);

        // Create bind group for viewport uniform
        if (!this.bindGroups.simpleRectStrokeViewport) {
            const viewportBuffer = this.device.createBuffer({
                size: 16, // vec2f + padding
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.buffers.simpleRectStrokeViewport = viewportBuffer;

            const viewportData = new Float32Array([this.canvas.width, this.canvas.height, 0, 0]);
            this.device.queue.writeBuffer(viewportBuffer, 0, viewportData);

            this.bindGroups.simpleRectStrokeViewport = this.device.createBindGroup({
                layout: this.pipelines.simpleRectStroke.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: { buffer: viewportBuffer },
                }],
            });
        }

        // Set pipeline and draw all instances in one call
        renderPass.setPipeline(this.pipelines.simpleRectStroke);
        renderPass.setVertexBuffer(0, this.buffers.quadVertices);
        renderPass.setVertexBuffer(1, this.buffers.plotAreaStrokeInstances);
        renderPass.setBindGroup(0, this.bindGroups.simpleRectStrokeViewport);
        renderPass.draw(4, plotAreaStrokes.length);
    }

    renderAllPointsBatched(renderPass, progress) {
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Update distribution data buffer with dynamic values
        const distributionData = new Float32Array(N_DISTRIBUTIONS * 8);
        let pointStartIndex = 0;

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const layout = this.layoutCache[i];
            const [, , , numExperiments, numOutcomes] = toyDistributions[i];
            const rowY = i * (CANVAS_HEIGHT + ROW_PADDING) * devicePixelRatio;
            const offset = i * 8;

            // Interpolate max value
            const currentMax = this.maxValuesPerTimestep[this.currentTimestep][i];
            const nextMax = this.maxValuesPerTimestep[this.nextTimestep][i];
            const interpolatedMax = currentMax + (nextMax - currentMax) * progress;

            // Calculate dynamic width
            const widthRatio = interpolatedMax / this.globalMaxValue;
            const dynamicPlotWidth = layout.plotWidth * widthRatio;
            
            // Account for stroke width (extends 0.5 pixels on each side)
            const strokeWidth = 1 * devicePixelRatio;
            
            distributionData[offset] = rowY;
            distributionData[offset + 1] = layout.minLabelWidth * devicePixelRatio;
            distributionData[offset + 2] = layout.plotWidth * devicePixelRatio;  // Static width for positioning
            distributionData[offset + 3] = CANVAS_HEIGHT * devicePixelRatio;
            distributionData[offset + 4] = numExperiments;
            distributionData[offset + 5] = numOutcomes;
            distributionData[offset + 6] = pointStartIndex;
            distributionData[offset + 7] = dynamicPlotWidth * devicePixelRatio; // Dynamic width (we'll add stroke in shader)

            pointStartIndex += numExperiments * numOutcomes;
        }

        this.device.queue.writeBuffer(this.buffers.distributionData, 0, distributionData);

        // Update global uniforms
        const uniformData = new Float32Array(8); // Aligned to 32 bytes
        uniformData.set([
            this.canvas.width,
            this.canvas.height,
            1.5 * devicePixelRatio, // pointRadius
            progress,
            this.globalMaxValue,
            0, 0, 0 // padding
        ]);

        this.device.queue.writeBuffer(this.buffers.pointUniforms, 0, uniformData);

        // Use the cached bind group for this timestep
        const pointBindGroup = this.bindGroups.pointsByTimestep[this.currentTimestep];

        // Render all points in a single draw call
        renderPass.setPipeline(this.pipelines.points);
        renderPass.setBindGroup(0, pointBindGroup);
        renderPass.draw(4, this.totalPointCount); // All points from all distributions
    }

    renderBatchedText(renderPass, allCharacters) {
        if (allCharacters.length === 0) return;

        const devicePixelRatio = window.devicePixelRatio || 1;

        // Update character data in the pre-created buffer
        const charData = new Float32Array(allCharacters.length * 5);
        for (let i = 0; i < allCharacters.length; i++) {
            const char = allCharacters[i];
            charData[i * 5] = char.x;
            charData[i * 5 + 1] = char.y;
            charData[i * 5 + 2] = char.width;
            charData[i * 5 + 3] = char.height;
            charData[i * 5 + 4] = char.charIndex;
        }

        // Update the pre-created buffer (only write the actual data size)
        this.device.queue.writeBuffer(this.buffers.characterData, 0, charData);

        // Update text uniforms
        const textUniformData = new Float32Array(16);
        textUniformData.set([
            this.canvas.width,
            this.canvas.height,
            this.characterDimensions.width,
            this.characterDimensions.height,
        ]);
        this.device.queue.writeBuffer(this.buffers.textUniforms, 0, textUniformData);

        // Create bind group only once and cache it
        if (!this.bindGroups.text) {
            this.bindGroups.text = this.device.createBindGroup({
                layout: this.pipelines.text.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.buffers.textUniforms },
                    },
                    {
                        binding: 1,
                        resource: { buffer: this.buffers.characterData },
                    },
                    {
                        binding: 2,
                        resource: this.characterTextureView,
                    },
                    {
                        binding: 3,
                        resource: this.textSampler,
                    },
                ],
            });
        }

        // Render all characters in one draw call
        renderPass.setPipeline(this.pipelines.text);
        renderPass.setBindGroup(0, this.bindGroups.text);
        renderPass.draw(4, allCharacters.length);
    }

    cubicEaseInOut(t) {
        if (t < 0.5) {
            return 4 * t * t * t;
        } else {
            const p = 2 * t - 2;
            return 1 + p * p * p / 2;
        }
    }

    animate(currentTime) {
        // Update FPS counter
        this.frameCount++;
        if (currentTime - this.lastFpsUpdate >= 1000) {
            this.fps = Math.round(this.frameCount * 1000 / (currentTime - this.lastFpsUpdate));
            this.frameCount = 0;
            this.lastFpsUpdate = currentTime;
            
            // Update FPS display
            const fpsDisplay = document.getElementById('fps-display');
            if (fpsDisplay) {
                fpsDisplay.textContent = this.fps;
            }
        }

        // Calculate time elapsed since animation started
        const elapsedTime = currentTime - this.animationStartTime;

        // Total duration for one complete cycle through all timesteps
        const totalCycleDuration = this.numTimesteps * this.cycleInterval;
        
        // Where are we in the current cycle? (0 to 1)
        const cycleProgress = (elapsedTime % totalCycleDuration) / totalCycleDuration;
        
        // Which timestep pair are we between?
        const exactTimestep = cycleProgress * this.numTimesteps;
        this.currentTimestep = Math.floor(exactTimestep) % this.numTimesteps;
        this.nextTimestep = (this.currentTimestep + 1) % this.numTimesteps;
        
        // Progress within the current transition (0 to 1)
        const transitionProgress = exactTimestep - this.currentTimestep;
        
        // Apply easing only to the transition portion (not the hold time)
        // Scale progress to account for transition duration vs cycle interval
        const scaledProgress = Math.min(transitionProgress * (this.cycleInterval / this.transitionDuration), 1.0);
        const easedProgress = this.cubicEaseInOut(scaledProgress);

        // Render directly to the canvas
        this.render(easedProgress);

        requestAnimationFrame((time) => this.animate(time));
    }

    startAnimation() {
        requestAnimationFrame((time) => {
            this.animationStartTime = time;
            this.lastFpsUpdate = time;
            this.animate(time);
        });
    }
}

// Initialize the visualization when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new WanderingPointsVisualizationSingleCanvasWebGPU();
});

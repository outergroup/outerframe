// Configuration constants matching Swift implementation
const N_DISTRIBUTIONS = 20; // Now we can support all 20 using offscreen canvas approach
const CANVAS_WIDTH = 400;
const CANVAS_HEIGHT = 30;

// Sample toy data for ML parameter distributions
// Each entry contains: [parameter name, domain, scale type, number of experiments, number of outcomes per experiment, initial value]
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

class WanderingPointsVisualization {
    constructor() {
        this.canvases = [];
        this.ctx2ds = []; // 2D contexts for each canvas
        this.offscreenCanvas = null;
        this.offscreenGl = null;

        // Constants for offscreen canvas layout
        this.ROW_HEIGHT = 35;
        this.ROW_MARGIN = 10;
        this.OFFSCREEN_WIDTH = CANVAS_WIDTH;
        this.OFFSCREEN_HEIGHT = N_DISTRIBUTIONS * (this.ROW_HEIGHT + this.ROW_MARGIN);

        this.numTimesteps = 20; // Number of timesteps (interpolation steps from initial to final values)
        this.currentTimestep = 0;
        this.nextTimestep = 1;
        this.animationStartTime = 0;
        this.transitionDuration = 400;
        this.cycleInterval = 500;

        // Cache layout calculations
        this.layoutCache = [];

        // Shader programs for offscreen rendering
        this.pointShaderProgram = null;
        this.batchedRectShaderProgram = null;
        this.batchedCharacterShaderProgram = null;

        // Cached shader locations
        this.shaderLocations = {
            point: {},
            batchedRect: {},
            batchedCharacter: {}
        };

        // Pre-computed point buffers for each timestep
        this.datasetPointBuffers = [];
        this.totalPointCount = 0;

        // Max values for each canvas at each timestep
        this.maxValuesPerTimestep = []; // [timestep][canvas]
        this.globalMaxValue = 0.0;

        // Cached buffers
        this.quadVertexBuffer = null;

        // Plot bounds texture for clipping
        this.plotBoundsTexture = null;
        this.plotBoundsData = new Float32Array(N_DISTRIBUTIONS * 4); // RGBA per row

        // FPS tracking
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        this.fps = 0;

        this.createDOM();
        this.setupOffscreenCanvas();

        this.startAnimation();
    }

    initializeData() {
        // Pre-compute all point positions for each timestep
        this.datasetPointBuffers = [];
        this.maxValuesPerTimestep = [];
        this.globalMaxValue = 0.0;
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Calculate total point count
        this.totalPointCount = 0;
        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [, , , numExperiments, numOutcomes] = toyDistributions[i];
            this.totalPointCount += numExperiments * numOutcomes;
        }

        // First, generate the final dataset with Gaussian sampling
        const finalDatasets = [];
        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [, domain, scaleType, numExperiments, numOutcomes] = toyDistributions[i];
            const dataset = this.generateGaussianData(numExperiments, numOutcomes, domain, scaleType);
            finalDatasets.push(dataset);
        }

        // First pass: compute all values and find global max
        const allValuesPerTimestep = []; // [timestep][distribution][values]

        for (let timestep = 0; timestep < this.numTimesteps; timestep++) {
            const valuesForTimestep = [];
            const maxValuesForTimestep = [];

            for (let distributionIndex = 0; distributionIndex < N_DISTRIBUTIONS; distributionIndex++) {
                const [, , , numExperiments, numOutcomes, initialValue] = toyDistributions[distributionIndex];
                const finalDataset = finalDatasets[distributionIndex];

                // Calculate interpolation progress for this timestep
                const timestepProgress = timestep / (this.numTimesteps - 1);

                // Apply cubic interpolation for more interesting transitions
                const cubicProgress = timestepProgress * timestepProgress * (3.0 - 2.0 * timestepProgress);

                const distributionValues = [];
                let distributionMaxValue = 0.0;

                for (let experimentIndex = 0; experimentIndex < numExperiments; experimentIndex++) {
                    for (let outcomeIndex = 0; outcomeIndex < numOutcomes; outcomeIndex++) {
                        // Interpolate between initial value and final value
                        const finalValue = finalDataset[experimentIndex][outcomeIndex];
                        const baseInterpolatedValue = initialValue + (finalValue - initialValue) * cubicProgress;

                        // Add multiplicative noise that varies by timestep and position
                        const noiseIndex = timestep * numExperiments * numOutcomes + experimentIndex * numOutcomes + outcomeIndex;
                        const noiseSeed = noiseIndex * 2654435761 / 4294967296; // JavaScript's max safe integer approximation

                        // Combine multiple frequencies for more organic noise
                        const noise1 = Math.sin(noiseSeed * 2 * Math.PI);
                        const noise2 = Math.sin(noiseSeed * 4.7 * Math.PI + 1.3);
                        const noise3 = Math.sin(noiseSeed * 9.2 * Math.PI + 2.7);
                        const combinedNoise = (noise1 + 0.5 * noise2 + 0.25 * noise3) / 1.75;

                        // Scale noise amount based on how far we are in the transition
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

        // Second pass: create point buffers with proper scaling
        for (let timestep = 0; timestep < this.numTimesteps; timestep++) {
            const positions = new Float32Array(this.totalPointCount * 2); // x,y for each point
            let pointIndex = 0;

            for (let distributionIndex = 0; distributionIndex < N_DISTRIBUTIONS; distributionIndex++) {
                const [, , , numExperiments, numOutcomes] = toyDistributions[distributionIndex];
                const layout = this.layoutCache[distributionIndex];
                const rowY = distributionIndex * (this.ROW_HEIGHT + this.ROW_MARGIN);

                // Scale based on global max, not domain
                const scale = (value) => {
                    return value / this.globalMaxValue;
                };

                const distributionValues = allValuesPerTimestep[timestep][distributionIndex];
                let valueIndex = 0;

                for (let experimentIndex = 0; experimentIndex < numExperiments; experimentIndex++) {
                    const yNormalized = (experimentIndex + 0.5) / numExperiments;
                    const yPixel = (rowY + yNormalized * this.ROW_HEIGHT) * devicePixelRatio;

                    for (let outcomeIndex = 0; outcomeIndex < numOutcomes; outcomeIndex++) {
                        const interpolatedValue = distributionValues[valueIndex];
                        valueIndex++;

                        const xNormalized = scale(interpolatedValue);
                        const xPixel = (layout.minLabelWidth + xNormalized * layout.plotWidth) * devicePixelRatio;

                        positions[pointIndex * 2] = xPixel;
                        positions[pointIndex * 2 + 1] = yPixel;
                        pointIndex++;
                    }
                }
            }

            // Create WebGL buffer for this timestep
            const buffer = this.offscreenGl.createBuffer();
            this.offscreenGl.bindBuffer(this.offscreenGl.ARRAY_BUFFER, buffer);
            this.offscreenGl.bufferData(this.offscreenGl.ARRAY_BUFFER, positions, this.offscreenGl.STATIC_DRAW);
            this.datasetPointBuffers.push(buffer);
        }

        // Initialize timestep indices
        this.currentTimestep = 0;
        this.nextTimestep = 1;
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
            } else { // log
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
                } else { // log
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

    createDOM() {
        const distributionsContainer = document.getElementById('distributions');

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [paramName] = toyDistributions[i];

            // Add the text part
            const textSpan = document.createElement('span');
            textSpan.innerHTML = `compare('<span class="param-name">${paramName}</span>') / `;
            distributionsContainer.appendChild(textSpan);

            // Create and add canvas
            const canvas = document.createElement('canvas');
            const devicePixelRatio = window.devicePixelRatio || 1;
            canvas.width = CANVAS_WIDTH * devicePixelRatio;
            canvas.height = CANVAS_HEIGHT * devicePixelRatio;
            canvas.style.width = `${CANVAS_WIDTH}px`;
            canvas.style.height = `${CANVAS_HEIGHT}px`;

            // Get 2D context
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false; // For pixel-perfect copying

            distributionsContainer.appendChild(canvas);

            // Add newline except for last item
            if (i < N_DISTRIBUTIONS - 1) {
                distributionsContainer.appendChild(document.createTextNode('\n'));
            }

            this.canvases.push(canvas);
            this.ctx2ds.push(ctx);
        }
    }

    initializeLayoutCache() {
        const fontSize = 10; // This is in CSS pixels
        const labelPadding = 8; // This is in CSS pixels

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const [paramName, domain, scaleType] = toyDistributions[i];

            const minText = "0"; // Always start from 0
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

    setupOffscreenCanvas() {
        // Create offscreen canvas
        const devicePixelRatio = window.devicePixelRatio || 1;
        this.offscreenCanvas = new OffscreenCanvas(
            this.OFFSCREEN_WIDTH * devicePixelRatio,
            this.OFFSCREEN_HEIGHT * devicePixelRatio
        );

        this.offscreenGl = this.offscreenCanvas.getContext('webgl2') ||
                           this.offscreenCanvas.getContext('webgl');

        if (!this.offscreenGl) {
            console.error('WebGL not supported for offscreen canvas');
            return;
        }

        // Pre-calculate layout cache
        this.initializeLayoutCache();

        // Initialize data after layout cache and GL context are ready
        this.initializeData();

        // Initialize character textures
        this.initializeCharacterTextures();

        const gl = this.offscreenGl;

        // Create point shader program with interpolation
        const pointVertexShaderSource = `
            precision mediump float;
            attribute vec2 a_startPosition;
            attribute vec2 a_endPosition;
            uniform vec2 u_viewportSize;
            uniform float u_pointRadius;
            uniform vec4 u_pointColor;
            uniform float u_progress; // Eased progress from CPU

            varying vec4 v_color;
            varying vec2 v_centerPosition;
            varying float v_pointRadius;

            void main() {
                // Interpolate between start and end positions
                vec2 interpolatedPosition = mix(a_startPosition, a_endPosition, u_progress);

                // Convert pixel coordinates to normalized device coordinates
                vec2 normalizedPosition;
                normalizedPosition.x = (interpolatedPosition.x / u_viewportSize.x) * 2.0 - 1.0;
                normalizedPosition.y = 1.0 - (interpolatedPosition.y / u_viewportSize.y) * 2.0;

                gl_Position = vec4(normalizedPosition, 0.0, 1.0);
                gl_PointSize = u_pointRadius * 2.0;
                v_color = u_pointColor;
                v_centerPosition = interpolatedPosition;
                v_pointRadius = u_pointRadius;
            }
        `;

        const pointFragmentShaderSource = `
            precision mediump float;
            precision mediump sampler2D;
            varying vec4 v_color;
            varying vec2 v_centerPosition;
            varying float v_pointRadius;
            uniform sampler2D u_plotBoundsTexture;
            uniform float u_rowHeight;
            uniform float u_rowMargin;
            uniform vec2 u_viewportSize;

            void main() {
                // Calculate the actual pixel position of this fragment
                vec2 offset = (gl_PointCoord - vec2(0.5)) * v_pointRadius * 2.0;
                vec2 fragmentPos = v_centerPosition + offset;

                // Calculate which row this point belongs to
                float rowTotalHeight = u_rowHeight + u_rowMargin;
                float rowIndex = floor(v_centerPosition.y / rowTotalHeight);

                // Read plot bounds from texture (stored as minX, maxX in R and G channels as 0-255)
                vec2 texCoord = vec2(0.5, (rowIndex + 0.5) / 20.0);
                vec4 boundsData = texture2D(u_plotBoundsTexture, texCoord);
                float minX = (boundsData.r) * u_viewportSize.x;
                float maxX = (boundsData.g) * u_viewportSize.x;

                // Check if fragment is outside plot bounds
                if (fragmentPos.x < minX || fragmentPos.x > maxX) {
                    discard;
                }

                // Create circular points by checking distance from center
                vec2 center = vec2(0.5, 0.5);
                float distance = length(gl_PointCoord - center);

                // Smooth edge falloff for anti-aliasing
                float alpha = 1.0 - smoothstep(0.4, 0.5, distance);

                gl_FragColor = vec4(v_color.rgb, v_color.a * alpha);
            }
        `;

        this.pointShaderProgram = this.createShaderProgram(gl, pointVertexShaderSource, pointFragmentShaderSource);
        if (this.pointShaderProgram) {
            // Cache uniform and attribute locations for point shader
            this.shaderLocations.point = {
                uniforms: {
                    viewportSize: gl.getUniformLocation(this.pointShaderProgram, 'u_viewportSize'),
                    pointRadius: gl.getUniformLocation(this.pointShaderProgram, 'u_pointRadius'),
                    pointColor: gl.getUniformLocation(this.pointShaderProgram, 'u_pointColor'),
                    progress: gl.getUniformLocation(this.pointShaderProgram, 'u_progress'),
                    plotBoundsTexture: gl.getUniformLocation(this.pointShaderProgram, 'u_plotBoundsTexture'),
                    rowHeight: gl.getUniformLocation(this.pointShaderProgram, 'u_rowHeight'),
                    rowMargin: gl.getUniformLocation(this.pointShaderProgram, 'u_rowMargin')
                },
                attributes: {
                    startPosition: gl.getAttribLocation(this.pointShaderProgram, 'a_startPosition'),
                    endPosition: gl.getAttribLocation(this.pointShaderProgram, 'a_endPosition')
                }
            };
        }

        // Create batched rectangle rendering shaders
        const batchedRectVertexShaderSource = `
            attribute vec2 a_position;
            attribute vec4 a_rect; // x, y, width, height
            attribute vec4 a_fillColor;
            attribute vec4 a_strokeColor;
            attribute vec2 a_params; // cornerRadius, strokeWidth
            uniform vec2 u_viewportSize;

            varying vec2 v_texCoord;
            varying vec4 v_rect;
            varying vec4 v_fillColor;
            varying vec4 v_strokeColor;
            varying vec2 v_params;

            void main() {
                // Transform vertex position to rect bounds
                vec2 position = a_rect.xy + a_position * a_rect.zw;

                // Convert to NDC
                vec2 ndcPos;
                ndcPos.x = (position.x / u_viewportSize.x) * 2.0 - 1.0;
                ndcPos.y = 1.0 - (position.y / u_viewportSize.y) * 2.0;

                gl_Position = vec4(ndcPos, 0.0, 1.0);

                // Pass data to fragment shader
                v_texCoord = a_position;
                v_rect = a_rect;
                v_fillColor = a_fillColor;
                v_strokeColor = a_strokeColor;
                v_params = a_params;
            }
        `;

        const batchedRectFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            varying vec4 v_rect;
            varying vec4 v_fillColor;
            varying vec4 v_strokeColor;
            varying vec2 v_params;

            float roundedRectSDF(vec2 p, vec2 size, float radius) {
                vec2 d = abs(p) - size + radius;
                return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - radius;
            }

            void main() {
                vec2 pixelPos = v_texCoord * v_rect.zw;
                vec2 p = pixelPos - v_rect.zw * 0.5;

                float cornerRadius = v_params.x;
                float strokeWidth = v_params.y;

                float d = roundedRectSDF(p, v_rect.zw * 0.5, cornerRadius);

                float fillAlpha = 1.0 - smoothstep(-1.0, 0.0, d);
                float strokeAlpha = smoothstep(-strokeWidth - 1.0, -strokeWidth, d) *
                                   (1.0 - smoothstep(-0.5, 0.5, d));

                vec4 color = v_fillColor * fillAlpha;
                color = mix(color, v_strokeColor, strokeAlpha);
                color.a *= fillAlpha + strokeAlpha;

                gl_FragColor = color;
            }
        `;

        this.batchedRectShaderProgram = this.createShaderProgram(gl, batchedRectVertexShaderSource, batchedRectFragmentShaderSource);
        if (!this.batchedRectShaderProgram) {
            console.error('Failed to create batched rect shader program');
        } else {
            // Cache uniform and attribute locations for batched rect shader
            this.shaderLocations.batchedRect = {
                uniforms: {
                    viewportSize: gl.getUniformLocation(this.batchedRectShaderProgram, 'u_viewportSize')
                },
                attributes: {
                    position: gl.getAttribLocation(this.batchedRectShaderProgram, 'a_position'),
                    rect: gl.getAttribLocation(this.batchedRectShaderProgram, 'a_rect'),
                    fillColor: gl.getAttribLocation(this.batchedRectShaderProgram, 'a_fillColor'),
                    strokeColor: gl.getAttribLocation(this.batchedRectShaderProgram, 'a_strokeColor'),
                    params: gl.getAttribLocation(this.batchedRectShaderProgram, 'a_params')
                }
            };
        }

        // Create batched character rendering shader
        const batchedCharacterVertexShaderSource = `
            attribute vec2 a_position;
            attribute vec4 a_charRect; // x, y, width, height
            attribute float a_textureIndex;
            uniform vec2 u_viewportSize;

            varying vec2 v_texCoord;
            varying float v_textureIndex;

            void main() {
                // Transform vertex position to character bounds
                vec2 position = a_charRect.xy + a_position * a_charRect.zw;

                // Convert to NDC
                vec2 ndcPos;
                ndcPos.x = (position.x / u_viewportSize.x) * 2.0 - 1.0;
                ndcPos.y = 1.0 - (position.y / u_viewportSize.y) * 2.0;

                gl_Position = vec4(ndcPos, 0.0, 1.0);

                // Pass texture coordinates as-is (no flip needed)
                v_texCoord = a_position;
                v_textureIndex = a_textureIndex;
            }
        `;

        const batchedCharacterFragmentShaderSource = `
            precision mediump float;
            uniform sampler2D u_characterTextures[16]; // Array of character textures
            varying vec2 v_texCoord;
            varying float v_textureIndex;

            void main() {
                // Select texture based on index
                vec4 texColor;
                int index = int(v_textureIndex + 0.5); // Round to nearest integer

                // Manual texture selection (GLSL ES 1.0 doesn't support dynamic array indexing)
                if (index == 0) texColor = texture2D(u_characterTextures[0], v_texCoord);
                else if (index == 1) texColor = texture2D(u_characterTextures[1], v_texCoord);
                else if (index == 2) texColor = texture2D(u_characterTextures[2], v_texCoord);
                else if (index == 3) texColor = texture2D(u_characterTextures[3], v_texCoord);
                else if (index == 4) texColor = texture2D(u_characterTextures[4], v_texCoord);
                else if (index == 5) texColor = texture2D(u_characterTextures[5], v_texCoord);
                else if (index == 6) texColor = texture2D(u_characterTextures[6], v_texCoord);
                else if (index == 7) texColor = texture2D(u_characterTextures[7], v_texCoord);
                else if (index == 8) texColor = texture2D(u_characterTextures[8], v_texCoord);
                else if (index == 9) texColor = texture2D(u_characterTextures[9], v_texCoord);
                else if (index == 10) texColor = texture2D(u_characterTextures[10], v_texCoord);
                else if (index == 11) texColor = texture2D(u_characterTextures[11], v_texCoord);
                else if (index == 12) texColor = texture2D(u_characterTextures[12], v_texCoord);
                else if (index == 13) texColor = texture2D(u_characterTextures[13], v_texCoord);
                else texColor = vec4(0.0);

                // Use the inverse of red channel as alpha (black text on white background)
                float alpha = 1.0 - texColor.r;
                // Render gray text (0.6, 0.6, 0.6)
                gl_FragColor = vec4(0.6, 0.6, 0.6, alpha);
            }
        `;


        // Create batched character shader program
        this.batchedCharacterShaderProgram = this.createShaderProgram(gl, batchedCharacterVertexShaderSource, batchedCharacterFragmentShaderSource);
        if (!this.batchedCharacterShaderProgram) {
            console.error('Failed to create batched character shader program');
        } else {
            // Cache uniform and attribute locations for batched character shader
            this.shaderLocations.batchedCharacter = {
                uniforms: {
                    viewportSize: gl.getUniformLocation(this.batchedCharacterShaderProgram, 'u_viewportSize'),
                    characterTextures: []
                },
                attributes: {
                    position: gl.getAttribLocation(this.batchedCharacterShaderProgram, 'a_position'),
                    charRect: gl.getAttribLocation(this.batchedCharacterShaderProgram, 'a_charRect'),
                    textureIndex: gl.getAttribLocation(this.batchedCharacterShaderProgram, 'a_textureIndex')
                }
            };

            // Get uniform locations for character texture array
            for (let i = 0; i < 16; i++) {
                this.shaderLocations.batchedCharacter.uniforms.characterTextures[i] =
                    gl.getUniformLocation(this.batchedCharacterShaderProgram, `u_characterTextures[${i}]`);
            }
        }

        // Create a simple unit quad vertex buffer (0,0 to 1,1)
        const quadVertices = new Float32Array([
            0, 0,  // bottom-left
            1, 0,  // bottom-right
            0, 1,  // top-left
            1, 1   // top-right
        ]);

        this.quadVertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);

        // Create plot bounds texture for clipping
        this.plotBoundsTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.plotBoundsTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // Initialize with empty data - use UNSIGNED_BYTE for compatibility
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, N_DISTRIBUTIONS, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }

    createShaderProgram(gl, vertexSource, fragmentSource) {
        const vertexShader = this.createShader(gl, gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

        if (!vertexShader || !fragmentShader) {
            return null;
        }

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Error linking shader program:', gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Error compiling shader:', gl.getShaderInfoLog(shader));
            console.error('Shader source:', source);
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    initializeCharacterTextures() {
        this.characterTextures = [];
        this.characterMap = {};
        const characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', 'e', '+'];

        // Create ASCII to texture index lookup table
        this.asciiToTextureIndex = new Array(256).fill(-1);
        characters.forEach((char, index) => {
            this.asciiToTextureIndex[char.charCodeAt(0)] = index;
        });

        const fontSize = 10 * (window.devicePixelRatio || 1);
        const padding = 2 * (window.devicePixelRatio || 1);

        // Create a canvas to measure text dimensions
        const measureCanvas = document.createElement('canvas');
        const measureCtx = measureCanvas.getContext('2d');
        measureCtx.font = `bold ${fontSize}px Menlo, Monaco, monospace`;

        // Measure a sample character to get dimensions
        const metrics = measureCtx.measureText('0');
        const charWidth = Math.ceil(metrics.width) + padding * 2;
        const charHeight = Math.ceil(fontSize * 1.5); // Add some vertical padding

        // Store dimensions for later use
        this.characterDimensions = {
            width: charWidth,
            height: charHeight,
            fontSize: fontSize
        };

        // Create texture for each character
        const gl = this.offscreenGl;
        characters.forEach((char, index) => {
            const canvas = document.createElement('canvas');
            canvas.width = charWidth;
            canvas.height = charHeight;
            const ctx = canvas.getContext('2d');

            // Fill with white background (we'll invert in shader)
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, charWidth, charHeight);

            // Set up text rendering with bold font
            ctx.font = `bold ${fontSize}px Menlo, Monaco, monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = 'black';

            // Draw the character in black
            ctx.fillText(char, charWidth / 2, charHeight / 2);

            // Create WebGL texture
            const texture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            this.characterTextures.push(texture);
            this.characterMap[char] = index;
        });
    }


    // Cubic easing function for smooth transitions
    cubicEaseInOut(t) {
        if (t < 0.5) {
            return 4 * t * t * t;
        } else {
            const p = 2 * t - 2;
            return 1 + p * p * p / 2;
        }
    }

    // Render to the offscreen canvas
    renderOffscreen(progress) {
        const gl = this.offscreenGl;

        // Clear canvas to transparent
        gl.viewport(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
        gl.clearColor(0.0, 0.0, 0.0, 0.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        // Render backgrounds using GPU shaders
        this.renderBackgroundsGPU(progress);

        // Render all points with GPU interpolation
        this.renderAllPoints(progress);
    }

    renderBackgroundsGPU(progress) {
        const gl = this.offscreenGl;
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Collect all rectangle data for batching
        const roundedRects = [];

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const layout = this.layoutCache[i];
            const rowY = i * (this.ROW_HEIGHT + this.ROW_MARGIN);

            // Interpolate max value
            const currentMax = this.maxValuesPerTimestep[this.currentTimestep][i];
            const nextMax = this.maxValuesPerTimestep[this.nextTimestep][i];
            const interpolatedMax = currentMax + (nextMax - currentMax) * progress;

            // Calculate dynamic width based on max value relative to global max
            const widthRatio = interpolatedMax / this.globalMaxValue;
            const dynamicPlotWidth = layout.plotWidth * widthRatio;
            const visibleWidth = layout.minLabelWidth + dynamicPlotWidth + layout.maxLabelWidth;

            // Update plot bounds data for clipping
            // Account for stroke width (extends 0.5 pixels on each side)
            const strokeWidth = 1 * devicePixelRatio;
            const minX = layout.minLabelWidth * devicePixelRatio + strokeWidth * 0.5;
            const maxX = (layout.minLabelWidth + dynamicPlotWidth) * devicePixelRatio + strokeWidth * 0.5;
            this.plotBoundsData[i * 4] = minX / this.offscreenCanvas.width;
            this.plotBoundsData[i * 4 + 1] = maxX / this.offscreenCanvas.width;
            this.plotBoundsData[i * 4 + 2] = 0; // unused
            this.plotBoundsData[i * 4 + 3] = 1; // alpha

            // White rounded rectangle background
            roundedRects.push({
                rect: [0, rowY * devicePixelRatio, visibleWidth * devicePixelRatio, this.ROW_HEIGHT * devicePixelRatio],
                fillColor: [1.0, 1.0, 1.0, 1.0],
                strokeColor: [0.7, 0.7, 0.7, 1.0],
                cornerRadius: 5 * devicePixelRatio,
                strokeWidth: 1 * devicePixelRatio
            });

            // Gray plot area (use rounded rect with 0 corner radius to get stroke support)
            roundedRects.push({
                rect: [layout.minLabelWidth * devicePixelRatio, rowY * devicePixelRatio,
                       dynamicPlotWidth * devicePixelRatio, this.ROW_HEIGHT * devicePixelRatio],
                fillColor: [0.9, 0.9, 0.9, 1.0],
                strokeColor: [0.7, 0.7, 0.7, 1.0],
                cornerRadius: 0,
                strokeWidth: 1 * devicePixelRatio
            });
        }

        // Update plot bounds texture - convert to UNSIGNED_BYTE
        const plotBoundsBytes = new Uint8Array(N_DISTRIBUTIONS * 4);
        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            plotBoundsBytes[i * 4] = Math.floor(this.plotBoundsData[i * 4] * 255);
            plotBoundsBytes[i * 4 + 1] = Math.floor(this.plotBoundsData[i * 4 + 1] * 255);
            plotBoundsBytes[i * 4 + 2] = 0;
            plotBoundsBytes[i * 4 + 3] = 255;
        }
        gl.bindTexture(gl.TEXTURE_2D, this.plotBoundsTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1, N_DISTRIBUTIONS, gl.RGBA, gl.UNSIGNED_BYTE, plotBoundsBytes);

        // Render rounded rectangles in batch
        if (roundedRects.length > 0 && this.batchedRectShaderProgram) {
            this.renderBatchedRoundedRects(roundedRects);
        }

        // Render all text in batch
        this.renderBatchedText(progress);
    }

    renderBatchedRoundedRects(rects) {
        const gl = this.offscreenGl;
        const program = this.batchedRectShaderProgram;
        const locations = this.shaderLocations.batchedRect;

        gl.useProgram(program);

        // Set viewport uniform
        gl.uniform2f(locations.uniforms.viewportSize, this.offscreenCanvas.width, this.offscreenCanvas.height);

        // Bind quad vertices
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
        gl.enableVertexAttribArray(locations.attributes.position);
        gl.vertexAttribPointer(locations.attributes.position, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.position, 0); // Use same vertices for all instances

        // Create instance data arrays
        const rectData = new Float32Array(rects.length * 4);
        const fillColorData = new Float32Array(rects.length * 4);
        const strokeColorData = new Float32Array(rects.length * 4);
        const paramsData = new Float32Array(rects.length * 2);

        // Fill instance data
        for (let i = 0; i < rects.length; i++) {
            rectData.set(rects[i].rect, i * 4);
            fillColorData.set(rects[i].fillColor, i * 4);
            strokeColorData.set(rects[i].strokeColor, i * 4);
            paramsData[i * 2] = rects[i].cornerRadius;
            paramsData[i * 2 + 1] = rects[i].strokeWidth;
        }

        // Create and bind instance buffers
        const rectBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, rectBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, rectData, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(locations.attributes.rect);
        gl.vertexAttribPointer(locations.attributes.rect, 4, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.rect, 1);

        const fillColorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, fillColorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, fillColorData, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(locations.attributes.fillColor);
        gl.vertexAttribPointer(locations.attributes.fillColor, 4, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.fillColor, 1);

        const strokeColorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, strokeColorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, strokeColorData, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(locations.attributes.strokeColor);
        gl.vertexAttribPointer(locations.attributes.strokeColor, 4, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.strokeColor, 1);

        const paramsBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, paramsBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, paramsData, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(locations.attributes.params);
        gl.vertexAttribPointer(locations.attributes.params, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.params, 1);

        // Draw all instances in one call
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, rects.length);

        // Reset vertex attribute divisors
        gl.vertexAttribDivisor(locations.attributes.rect, 0);
        gl.vertexAttribDivisor(locations.attributes.fillColor, 0);
        gl.vertexAttribDivisor(locations.attributes.strokeColor, 0);
        gl.vertexAttribDivisor(locations.attributes.params, 0);

        // Cleanup
        gl.disableVertexAttribArray(locations.attributes.position);
        gl.disableVertexAttribArray(locations.attributes.rect);
        gl.disableVertexAttribArray(locations.attributes.fillColor);
        gl.disableVertexAttribArray(locations.attributes.strokeColor);
        gl.disableVertexAttribArray(locations.attributes.params);

        gl.deleteBuffer(rectBuffer);
        gl.deleteBuffer(fillColorBuffer);
        gl.deleteBuffer(strokeColorBuffer);
        gl.deleteBuffer(paramsBuffer);
    }


    renderBatchedText(progress) {
        const gl = this.offscreenGl;
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Collect all characters to render
        const characters = [];

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const layout = this.layoutCache[i];
            const rowY = i * (this.ROW_HEIGHT + this.ROW_MARGIN);

            // Interpolate max value for text
            const currentMax = this.maxValuesPerTimestep[this.currentTimestep][i];
            const nextMax = this.maxValuesPerTimestep[this.nextTimestep][i];
            const interpolatedMax = currentMax + (nextMax - currentMax) * progress;

            const widthRatio = interpolatedMax / this.globalMaxValue;
            const dynamicPlotWidth = layout.plotWidth * widthRatio;

            const textY = rowY + this.ROW_HEIGHT / 2 + 1;

            // Collect characters for min label "0"
            const minText = "0";
            const minCenterX = layout.minLabelWidth / 2;
            this.collectCharacters(characters, minText, minCenterX, textY);

            // Collect characters for max label
            const [, , scaleType] = toyDistributions[i];
            const maxText = scaleType === "linear" ?
                toPrecisionThrifty(interpolatedMax, 2) :
                interpolatedMax.toExponential(1);
            const maxCenterX = layout.minLabelWidth + dynamicPlotWidth + layout.maxLabelWidth / 2;
            this.collectCharacters(characters, maxText, maxCenterX, textY);
        }

        if (characters.length === 0) return;

        // Use WebGL2 instancing
        this.renderCharactersInstanced(characters);
    }

    collectCharacters(characters, text, centerX, centerY) {
        if (!this.characterDimensions) return;

        const devicePixelRatio = window.devicePixelRatio || 1;
        const charWidth = this.characterDimensions.width / devicePixelRatio;
        const charHeight = this.characterDimensions.height / devicePixelRatio;
        const charSpacing = charWidth * 0.6;
        const totalWidth = charWidth + (text.length - 1) * charSpacing;

        let x = centerX - totalWidth / 2;
        const y = centerY - charHeight / 2;

        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            const textureIndex = this.asciiToTextureIndex[text.charCodeAt(i)];

            if (textureIndex !== -1) {
                characters.push({
                    rect: [x * devicePixelRatio, y * devicePixelRatio,
                           charWidth * devicePixelRatio, charHeight * devicePixelRatio],
                    textureIndex: textureIndex
                });
            }

            x += charSpacing;
        }
    }

    renderCharactersInstanced(characters) {
        const gl = this.offscreenGl;
        const program = this.batchedCharacterShaderProgram;
        const locations = this.shaderLocations.batchedCharacter;

        gl.useProgram(program);

        // Set viewport uniform
        gl.uniform2f(locations.uniforms.viewportSize, this.offscreenCanvas.width, this.offscreenCanvas.height);

        // Bind character textures
        for (let i = 0; i < this.characterTextures.length && i < 16; i++) {
            gl.activeTexture(gl.TEXTURE0 + i);
            gl.bindTexture(gl.TEXTURE_2D, this.characterTextures[i]);
            gl.uniform1i(locations.uniforms.characterTextures[i], i);
        }

        // Bind quad vertices
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
        gl.enableVertexAttribArray(locations.attributes.position);
        gl.vertexAttribPointer(locations.attributes.position, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.position, 0);

        // Create instance data
        const rectData = new Float32Array(characters.length * 4);
        const textureIndexData = new Float32Array(characters.length);

        for (let i = 0; i < characters.length; i++) {
            rectData.set(characters[i].rect, i * 4);
            textureIndexData[i] = characters[i].textureIndex;
        }

        // Create and bind instance buffers
        const rectBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, rectBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, rectData, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(locations.attributes.charRect);
        gl.vertexAttribPointer(locations.attributes.charRect, 4, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.charRect, 1);

        const textureIndexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, textureIndexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, textureIndexData, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(locations.attributes.textureIndex);
        gl.vertexAttribPointer(locations.attributes.textureIndex, 1, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(locations.attributes.textureIndex, 1);

        // Draw all characters in one call
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, characters.length);

        // Reset vertex attribute divisors
        gl.vertexAttribDivisor(locations.attributes.charRect, 0);
        gl.vertexAttribDivisor(locations.attributes.textureIndex, 0);

        // Cleanup
        gl.disableVertexAttribArray(locations.attributes.position);
        gl.disableVertexAttribArray(locations.attributes.charRect);
        gl.disableVertexAttribArray(locations.attributes.textureIndex);

        gl.deleteBuffer(rectBuffer);
        gl.deleteBuffer(textureIndexBuffer);
    }


    createTransformMatrix(x, y, width, height) {
        const devicePixelRatio = window.devicePixelRatio || 1;
        const canvasWidth = this.offscreenCanvas.width;
        const canvasHeight = this.offscreenCanvas.height;

        // Convert to physical pixels
        x *= devicePixelRatio;
        y *= devicePixelRatio;
        width *= devicePixelRatio;
        height *= devicePixelRatio;

        // Create transformation matrix for NDC space
        const scaleX = (width * 2) / canvasWidth;
        const scaleY = (height * 2) / canvasHeight;
        const translateX = (x * 2 / canvasWidth) - 1;
        const translateY = 1 - ((y + height) * 2 / canvasHeight);

        // Column-major order for WebGL
        return new Float32Array([
            scaleX, 0, 0,
            0, scaleY, 0,
            translateX, translateY, 1
        ]);
    }

    renderAllPoints(progress) {
        const gl = this.offscreenGl;
        const program = this.pointShaderProgram;

        gl.useProgram(program);

        // Use cached locations
        const locations = this.shaderLocations.point;

        // Set uniforms
        const devicePixelRatio = window.devicePixelRatio || 1;
        gl.uniform2f(locations.uniforms.viewportSize, this.offscreenCanvas.width, this.offscreenCanvas.height);
        gl.uniform1f(locations.uniforms.pointRadius, 1.5 * devicePixelRatio);
        gl.uniform4f(locations.uniforms.pointColor, 0.1, 0.2, 0.6, 0.8);
        gl.uniform1f(locations.uniforms.progress, progress);
        gl.uniform1f(locations.uniforms.rowHeight, this.ROW_HEIGHT * devicePixelRatio);
        gl.uniform1f(locations.uniforms.rowMargin, this.ROW_MARGIN * devicePixelRatio);

        // Bind plot bounds texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.plotBoundsTexture);
        gl.uniform1i(locations.uniforms.plotBoundsTexture, 0);

        // Bind start positions
        gl.bindBuffer(gl.ARRAY_BUFFER, this.datasetPointBuffers[this.currentTimestep]);
        gl.enableVertexAttribArray(locations.attributes.startPosition);
        gl.vertexAttribPointer(locations.attributes.startPosition, 2, gl.FLOAT, false, 0, 0);

        // Bind end positions
        gl.bindBuffer(gl.ARRAY_BUFFER, this.datasetPointBuffers[this.nextTimestep]);
        gl.enableVertexAttribArray(locations.attributes.endPosition);
        gl.vertexAttribPointer(locations.attributes.endPosition, 2, gl.FLOAT, false, 0, 0);

        // Draw all points
        gl.drawArrays(gl.POINTS, 0, this.totalPointCount);

        // Cleanup
        gl.disableVertexAttribArray(locations.attributes.startPosition);
        gl.disableVertexAttribArray(locations.attributes.endPosition);
    }

    // Copy from offscreen canvas to individual canvases
    copyToCanvases() {
        const devicePixelRatio = window.devicePixelRatio || 1;

        for (let i = 0; i < N_DISTRIBUTIONS; i++) {
            const ctx = this.ctx2ds[i];
            const canvas = this.canvases[i];
            const rowY = i * (this.ROW_HEIGHT + this.ROW_MARGIN);

            // Clear the entire canvas first to remove stale content
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Copy the appropriate rectangle from the offscreen canvas
            ctx.drawImage(
                this.offscreenCanvas,
                0, rowY * devicePixelRatio,                    // source x, y
                CANVAS_WIDTH * devicePixelRatio,               // source width
                this.ROW_HEIGHT * devicePixelRatio,            // source height (use ROW_HEIGHT, not CANVAS_HEIGHT)
                0, 0,                                          // destination x, y
                CANVAS_WIDTH * devicePixelRatio,               // destination width
                CANVAS_HEIGHT * devicePixelRatio               // destination height
            );
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

        // Render to offscreen canvas
        this.renderOffscreen(easedProgress);

        // Copy to individual canvases
        this.copyToCanvases();

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
    new WanderingPointsVisualization();
});

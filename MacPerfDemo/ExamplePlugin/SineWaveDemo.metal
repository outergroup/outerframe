#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float xPos; // Store normalized x position for fragment shader
    float progressValue; // Pass the progress value to fragment shader
    uint renderTargetArrayIndex [[render_target_array_index]];
};

// Instance data structure for per-canvas information
struct InstanceData {
    float progress;
    uint canvasIndex;
};

// Constants structure for layout values
struct LayoutConstants {
    float xPosMin;
    float xPosMax;
};

// Main vertex shader for the progress bar with instanced rendering
vertex VertexOut canvas_vertex_main(VertexIn in [[stage_in]],
                                    constant InstanceData *instanceData [[buffer(1)]],
                                    constant LayoutConstants &constants [[buffer(2)]],
                                    uint instanceID [[instance_id]]) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);

    // Get instance-specific data
    InstanceData instance = instanceData[instanceID];

    // Store position and progress for fragment shader
    out.xPos = (in.position.x - constants.xPosMin) / (constants.xPosMax - constants.xPosMin);
    out.progressValue = instance.progress;

    // Route to the correct render target array slice
    out.renderTargetArrayIndex = instance.canvasIndex;

    // Basic color initialization
    out.color = float4(1.0, 1.0, 1.0, 1.0);

    return out;
}

// Fragment shader for progress bar (unchanged)
fragment float4 canvas_fragment_main(VertexOut in [[stage_in]]) {
    // More precise color calculation in fragment shader
    if (in.xPos <= in.progressValue) {
        // Create green to yellow gradient for filled part
        // More visible gradient that varies across the entire bar
        return float4(1.0,
                      // Scale to make gradient cover the filled area
                      0.3 + 0.7 * (in.xPos / in.progressValue),
                      0.0,
                      1.0);
    } else {
        // Dark gray for empty part
        return float4(0.3, 0.3, 0.3, 1.0);
    }
}

// ====== Text rendering shaders ======

// Structure for text vertex data
struct TextVertexIn {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

// Structure for text vertex output
struct TextVertexOut {
    float4 position [[position]];
    float2 texCoord;
    float progress; // Pass progress value to fragment shader
    uint renderTargetArrayIndex [[render_target_array_index]];
};

// Vertex shader for text rendering with instanced support
vertex TextVertexOut canvas_text_vertex_main(TextVertexIn in [[stage_in]],
                                              constant InstanceData *instanceData [[buffer(1)]],
                                              uint instanceID [[instance_id]]) {
    TextVertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;

    // Get instance-specific data and route to correct render target
    InstanceData instance = instanceData[instanceID];
    out.progress = instance.progress;
    out.renderTargetArrayIndex = instance.canvasIndex;

    return out;
}

// Fragment shader for text rendering
fragment float4 canvas_text_fragment_main(TextVertexOut in [[stage_in]],
                                          texture2d_array<float> textTextureArray [[texture(0)]],
                                          sampler textureSampler [[sampler(0)]]) {
    // Get the progress value and calculate texture slice
    uint textureSlice = uint(clamp(round(in.progress * 10.0), 0.0, 10.0));
    
    // Sample from the appropriate texture slice
    return textTextureArray.sample(textureSampler, in.texCoord, textureSlice);
}

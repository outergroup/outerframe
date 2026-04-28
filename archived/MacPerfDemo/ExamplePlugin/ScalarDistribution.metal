#include <metal_stdlib>
using namespace metal;

// Simple rectangle rendering for UI elements
struct UIVertex {
    float2 position [[attribute(0)]];
};

struct UIVertexOut {
    float4 position [[position]];
    float2 texCoord;
    uint render_target_array_index [[render_target_array_index]];
};

// Instance data for background rectangles
struct BackgroundInstanceData {
    uint canvasIndex;
    float visibleWidth;
    float plotOffset;
    float plotWidth;
    float strokeWidth;
};

// Uniforms for instanced background rendering
struct BackgroundUniforms {
    float2 canvasSize;  // Size of each canvas
    float4 grayRectFillColor;
    float4 strokeColor;
};

// Instanced gray rectangle rendering
vertex UIVertexOut scalar_dist_gray_rect_vertex(UIVertex in [[stage_in]],
                                              constant BackgroundUniforms& uniforms [[buffer(1)]],
                                              constant BackgroundInstanceData* instances [[buffer(2)]],
                                              uint instanceID [[instance_id]]) {
    UIVertexOut out;
    constant BackgroundInstanceData& instance = instances[instanceID];
    
    // Transform vertex to cover the gray rectangle area
    float2 size = float2(instance.plotWidth, uniforms.canvasSize.y);
    float2 pos = in.position * size + float2(instance.plotOffset, 0.0);
    float2 ndcPos = (pos / uniforms.canvasSize) * 2.0 - 1.0;
    ndcPos.y = -ndcPos.y; // Flip Y
    
    out.position = float4(ndcPos, 0.0, 1.0);
    out.texCoord = in.position;
    out.render_target_array_index = instanceID;
    
    return out;
}

fragment float4 scalar_dist_gray_rect_fragment(UIVertexOut in [[stage_in]],
                                            constant BackgroundUniforms& uniforms [[buffer(1)]],
                                            constant BackgroundInstanceData* instances [[buffer(2)]]) {
    constant BackgroundInstanceData& instance = instances[in.render_target_array_index];
    
    // Get the pixel position within the rectangle
    float2 rectSize = float2(instance.plotWidth, uniforms.canvasSize.y);
    float2 pixelPos = in.texCoord * rectSize;
    
    // Calculate distance from edges
    float2 edgeDist = min(pixelPos, rectSize - pixelPos);
    float minDist = min(edgeDist.x, edgeDist.y);
    
    // If we're within the stroke width from the edge, use stroke color
    if (minDist < instance.strokeWidth) {
        return uniforms.strokeColor;
    }
    
    return uniforms.grayRectFillColor;
}

// Input vertex structure for points
struct PointVertexIn {
    float2 position [[attribute(0)]];
};

// Output vertex structure (passed to fragment shader)
struct PointVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
    uint render_target_array_index [[render_target_array_index]];
    float2 centerPosition;  // Center position in screen space
    float2 plotBounds;      // Min and max X in screen space for clipping
    float pointRadius;      // Point radius in pixels
};

// Uniform structure for point rendering
struct PointUniforms {
    float2 viewportSize;
    float pointRadius;
    float4 pointColor;
    float progress;       // Animation progress (0 to 1, already eased on CPU)
};

// Per-point data for each wandering point
struct PointInstanceData {
    float2 startPosition;  // Position (when used as single buffer, both start/end are same)
    float2 endPosition;    // Position (when used as single buffer, both start/end are same)
    float alpha;           // Alpha value for this point
    uint canvasIndex;      // Which canvas this point belongs to
};

// Vertex shader for wandering points with dual buffer interpolation
vertex PointVertexOut scalar_dist_point_vertex(PointVertexIn in [[stage_in]],
                                             constant PointUniforms &uniforms [[buffer(1)]],
                                             constant PointInstanceData *startData [[buffer(2)]],
                                             constant PointInstanceData *endData [[buffer(3)]],
                                             constant BackgroundInstanceData *backgroundData [[buffer(4)]],
                                             uint instanceID [[instance_id]]) {
    PointVertexOut out;
    
    // Get data from both buffers
    PointInstanceData startInstance = startData[instanceID];
    PointInstanceData endInstance = endData[instanceID];
    
    // Interpolate between the two buffers using pre-eased progress from CPU
    float2 interpolatedPosition = mix(startInstance.startPosition, endInstance.startPosition, uniforms.progress);
    
    // Convert to normalized device coordinates 
    float2 normalizedPosition;
    normalizedPosition.x = (interpolatedPosition.x / uniforms.viewportSize.x) * 2.0 - 1.0;
    normalizedPosition.y = 1.0 - (interpolatedPosition.y / uniforms.viewportSize.y) * 2.0; // Flip Y coordinate
    
    // Get background data for this canvas to determine plot bounds
    constant BackgroundInstanceData& bgData = backgroundData[startInstance.canvasIndex];
    float plotMinX = bgData.plotOffset;
    float plotMaxX = bgData.plotOffset + bgData.plotWidth;
    
    out.position = float4(normalizedPosition, 0.0, 1.0);
    out.pointSize = uniforms.pointRadius * 2.0;
    out.color = float4(uniforms.pointColor.rgb, uniforms.pointColor.a * startInstance.alpha);
    out.render_target_array_index = startInstance.canvasIndex;
    out.centerPosition = interpolatedPosition;
    out.plotBounds = float2(plotMinX, plotMaxX);
    out.pointRadius = uniforms.pointRadius;
    
    return out;
}

// Fragment shader for wandering points
fragment float4 scalar_dist_point_fragment(PointVertexOut in [[stage_in]],
                                         float2 pointCoord [[point_coord]]) {
    // Calculate the actual pixel position of this fragment
    // pointCoord is in [0,1] range within the point sprite
    float2 offset = (pointCoord - float2(0.5)) * in.pointRadius * 2.0;
    float fragmentX = in.centerPosition.x + offset.x;
    
    // Check if this fragment's x position is outside the plot bounds
    if (fragmentX < in.plotBounds.x || fragmentX > in.plotBounds.y) {
        discard_fragment();
    }
    
    // Create circular points by checking distance from center
    float2 center = float2(0.5, 0.5);
    float distance = length(pointCoord - center);
    
    // Smooth edge falloff for anti-aliasing
    float alpha = 1.0 - smoothstep(0.4, 0.5, distance);
    
    return float4(in.color.rgb, in.color.a * alpha);
}

// Character rendering shaders
struct CharacterVertex {
    float2 position [[attribute(0)]];
};

struct CharacterVertexOut {
    float4 position [[position]];
    float2 texCoord;
    uint render_target_array_index [[render_target_array_index]];
    uint textureIndex;  // Pass texture index from vertex to fragment
};

struct CharacterInstanceData {
    float4x4 transform;
    uint textureIndex;
    uint canvasIndex;
};

vertex CharacterVertexOut scalar_dist_character_vertex(CharacterVertex in [[stage_in]],
                                          constant CharacterInstanceData* instances [[buffer(2)]],
                                          uint instanceID [[instance_id]]) {
    CharacterVertexOut out;
    constant CharacterInstanceData& instance = instances[instanceID];
    
    out.position = instance.transform * float4(in.position, 0.0, 1.0);
    out.texCoord = float2(in.position.x, 1.0 - in.position.y); // Flip Y for texture sampling
    out.render_target_array_index = instance.canvasIndex;  // Use canvas index from instance data
    out.textureIndex = instance.textureIndex;  // Pass texture index to fragment shader
    return out;
}

fragment float4 scalar_dist_character_fragment(CharacterVertexOut in [[stage_in]],
                                  texture2d_array<float> characterTextures [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    // Sample from the texture array using the texture index passed from vertex shader
    float4 textColor = characterTextures.sample(textureSampler, in.texCoord, in.textureIndex);
    
    // Premultiply alpha for correct blending
    return float4(textColor.rgb * textColor.a, textColor.a);
}

// End marker instance data
struct EndMarkerInstanceData {
    uint canvasIndex;
    float xPosition;      // X position of the marker
    float markerWidth;    // Width of the marker
    float strokeWidth;    // Stroke width
};

// End marker uniforms
struct EndMarkerUniforms {
    float2 canvasSize;
    float4 fillColor;     // Dark color for the marker
    float4 strokeColor;   // Light color for the stroke
};

// Vertex shader for end markers
vertex UIVertexOut scalar_dist_end_marker_vertex(UIVertex in [[stage_in]],
                                                constant EndMarkerUniforms& uniforms [[buffer(1)]],
                                                constant EndMarkerInstanceData* instances [[buffer(2)]],
                                                uint instanceID [[instance_id]]) {
    UIVertexOut out;
    constant EndMarkerInstanceData& instance = instances[instanceID];
    
    // Transform vertex to cover the marker area
    float2 size = float2(instance.markerWidth, uniforms.canvasSize.y);
    float2 pos = in.position * size + float2(instance.xPosition, 0.0);
    float2 ndcPos = (pos / uniforms.canvasSize) * 2.0 - 1.0;
    ndcPos.y = -ndcPos.y; // Flip Y
    
    out.position = float4(ndcPos, 0.0, 1.0);
    out.texCoord = in.position;
    out.render_target_array_index = instance.canvasIndex;
    
    return out;
}

// Fragment shader for end markers
fragment float4 scalar_dist_end_marker_fragment(UIVertexOut in [[stage_in]],
                                              constant EndMarkerUniforms& uniforms [[buffer(1)]],
                                              constant EndMarkerInstanceData* instances [[buffer(2)]]) {
    constant EndMarkerInstanceData& instance = instances[in.render_target_array_index];
    
    // Get the pixel position within the marker
    float2 markerSize = float2(instance.markerWidth, uniforms.canvasSize.y);
    float2 pixelPos = in.texCoord * markerSize;
    
    // Calculate distance from edges
    float2 edgeDist = min(pixelPos, markerSize - pixelPos);
    float minDist = min(edgeDist.x, edgeDist.y);
    
    // If we're within the stroke width from the edge, use stroke color
    if (minDist < instance.strokeWidth) {
        return uniforms.strokeColor;
    }
    
    return uniforms.fillColor;
}

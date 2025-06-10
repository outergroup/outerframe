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

struct UIUniforms {
    float4x4 transform;
    float4 color;
    uint canvasIndex;
};

struct RoundedRectUniforms {
    float4x4 transform;
    float4 fillColor;
    float4 strokeColor;
    float2 rectSize; // Size in pixels
    float cornerRadius;
    float strokeWidth;
    uint canvasIndex;
};

// Instance data for background rectangles
struct BackgroundInstanceData {
    uint canvasIndex;
    float visibleWidth;
    float cornerRadius;
    float strokeWidth;
    float plotOffset;
    float plotWidth;
};

// Uniforms for instanced background rendering
struct BackgroundUniforms {
    float2 canvasSize;  // Size of each canvas
    float4 whiteRectFillColor;
    float4 blackStrokeColor;
    float4 grayRectFillColor;
};

vertex UIVertexOut ui_vertex(UIVertex in [[stage_in]],
                             constant UIUniforms& uniforms [[buffer(1)]]) {
    UIVertexOut out;
    // Apply transform to vertex position
    out.position = uniforms.transform * float4(in.position, 0.0, 1.0);
    out.texCoord = in.position; // Pass through texture coordinates
    out.render_target_array_index = uniforms.canvasIndex;
    return out;
}

fragment float4 ui_fragment(UIVertexOut in [[stage_in]],
                           constant UIUniforms& uniforms [[buffer(1)]]) {
    return uniforms.color;
}

// Vertex shader for rounded rectangles
vertex UIVertexOut rounded_rect_vertex(UIVertex in [[stage_in]],
                                      constant RoundedRectUniforms& uniforms [[buffer(1)]]) {
    UIVertexOut out;
    out.position = uniforms.transform * float4(in.position, 0.0, 1.0);
    out.texCoord = in.position;
    out.render_target_array_index = uniforms.canvasIndex;
    return out;
}

// SDF for a rounded rectangle
float rounded_rect_sdf(float2 p, float2 size, float radius) {
    float2 d = abs(p) - size + radius;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - radius;
}

// Fragment shader for rounded rectangles with border
fragment float4 rounded_rect_fragment(UIVertexOut in [[stage_in]],
                                     constant RoundedRectUniforms& uniforms [[buffer(1)]]) {
    // Convert texture coordinates to pixel coordinates
    float2 pixelPos = in.texCoord * uniforms.rectSize;
    
    // Center the coordinate system
    float2 p = pixelPos - uniforms.rectSize * 0.5;
    
    // Calculate SDF
    float d = rounded_rect_sdf(p, uniforms.rectSize * 0.5, uniforms.cornerRadius);
    
    // Smooth edges for anti-aliasing
    float fillAlpha = 1.0 - smoothstep(-1.0, 0.0, d);
    float strokeAlpha = smoothstep(-uniforms.strokeWidth - 1.0, -uniforms.strokeWidth, d) * 
                        (1.0 - smoothstep(-0.5, 0.5, d));
    
    // Mix fill and stroke colors
    float4 color = uniforms.fillColor * fillAlpha;
    color = mix(color, uniforms.strokeColor, strokeAlpha);
    
    return color;
}

// Instanced white rounded rectangle rendering
vertex UIVertexOut instanced_white_rect_vertex(UIVertex in [[stage_in]],
                                               constant BackgroundUniforms& uniforms [[buffer(1)]],
                                               constant BackgroundInstanceData* instances [[buffer(2)]],
                                               uint instanceID [[instance_id]]) {
    UIVertexOut out;
    constant BackgroundInstanceData& instance = instances[instanceID];
    
    // Transform vertex to cover the white rectangle area
    float2 size = float2(instance.visibleWidth, uniforms.canvasSize.y);
    float2 pos = in.position * size;
    float2 ndcPos = (pos / uniforms.canvasSize) * 2.0 - 1.0;
    ndcPos.y = -ndcPos.y; // Flip Y
    
    out.position = float4(ndcPos, 0.0, 1.0);
    out.texCoord = in.position;
    out.render_target_array_index = instanceID;
    
    return out;
}

fragment float4 instanced_white_rect_fragment(UIVertexOut in [[stage_in]],
                                             constant BackgroundUniforms& uniforms [[buffer(1)]],
                                             constant BackgroundInstanceData* instances [[buffer(2)]]) {
    constant BackgroundInstanceData& instance = instances[in.render_target_array_index];
    
    // Calculate SDF for rounded rectangle
    float2 rectSize = float2(instance.visibleWidth, uniforms.canvasSize.y);
    float2 pixelPos = in.texCoord * rectSize;
    float2 p = pixelPos - rectSize * 0.5;
    
    float d = rounded_rect_sdf(p, rectSize * 0.5, instance.cornerRadius);
    
    float fillAlpha = 1.0 - smoothstep(-1.0, 0.0, d);
    float strokeAlpha = smoothstep(-instance.strokeWidth - 1.0, -instance.strokeWidth, d) * 
                        (1.0 - smoothstep(-0.5, 0.5, d));
    
    float4 color = uniforms.whiteRectFillColor * fillAlpha;
    color = mix(color, uniforms.blackStrokeColor, strokeAlpha);
    
    // Apply alpha to avoid drawing outside the rounded rect
    color.a *= fillAlpha + strokeAlpha;
    return color;
}

// Instanced gray rectangle rendering
vertex UIVertexOut instanced_gray_rect_vertex(UIVertex in [[stage_in]],
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

fragment float4 instanced_gray_rect_fragment(UIVertexOut in [[stage_in]],
                                            constant BackgroundUniforms& uniforms [[buffer(1)]],
                                            constant BackgroundInstanceData* instances [[buffer(2)]]) {
    constant BackgroundInstanceData& instance = instances[in.render_target_array_index];
    
    // Calculate SDF for rectangle (no rounded corners)
    float2 rectSize = float2(instance.plotWidth, uniforms.canvasSize.y);
    float2 pixelPos = in.texCoord * rectSize;
    float2 p = pixelPos - rectSize * 0.5;
    
    float d = rounded_rect_sdf(p, rectSize * 0.5, 0.0); // No corner radius
    
    float fillAlpha = 1.0 - smoothstep(-1.0, 0.0, d);
    float strokeAlpha = smoothstep(-instance.strokeWidth - 1.0, -instance.strokeWidth, d) * 
                        (1.0 - smoothstep(-0.5, 0.5, d));
    
    float4 color = uniforms.grayRectFillColor * fillAlpha;
    color = mix(color, uniforms.blackStrokeColor, strokeAlpha);
    
    return color;
}

// Text blending shader - for overlaying text textures onto the background
struct TextBlendVertex {
    float2 position [[attribute(0)]];
};

struct TextBlendVertexOut {
    float4 position [[position]];
    float2 texCoord;
    uint render_target_array_index [[render_target_array_index]];
};

struct TextBlendUniforms {
    float4x4 transform;
    uint canvasIndex;
};

vertex TextBlendVertexOut text_blend_vertex(TextBlendVertex in [[stage_in]],
                                           constant TextBlendUniforms& uniforms [[buffer(1)]]) {
    TextBlendVertexOut out;
    out.position = uniforms.transform * float4(in.position, 0.0, 1.0);
    out.texCoord = float2(in.position.x, 1.0 - in.position.y); // Flip Y for texture sampling
    out.render_target_array_index = uniforms.canvasIndex;
    return out;
}

fragment float4 text_blend_fragment(TextBlendVertexOut in [[stage_in]],
                                   texture2d<float> textTexture [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    float4 textColor = textTexture.sample(textureSampler, in.texCoord);
    
    // Premultiply alpha for correct blending
    return float4(textColor.rgb * textColor.a, textColor.a);
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

// Per-canvas data
struct CanvasInstanceData {
    float2 canvasSize;        // Individual canvas dimensions
    float2 plotRectOffset;    // Plot area offset within canvas
    float2 plotRectSize;      // Plot area size
    uint canvasIndex;         // Which canvas this is (0-23)
};

// Per-point data for each wandering point
struct PointInstanceData {
    float2 startPosition;  // Position (when used as single buffer, both start/end are same)
    float2 endPosition;    // Position (when used as single buffer, both start/end are same)
    float alpha;           // Alpha value for this point
    uint canvasIndex;      // Which canvas this point belongs to
};

// Vertex shader for wandering points with dual buffer interpolation
vertex PointVertexOut wandering_point_vertex(PointVertexIn in [[stage_in]],
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
fragment float4 wandering_point_fragment(PointVertexOut in [[stage_in]],
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

struct CharacterUniforms {
    float2 canvasSize;
    uint canvasIndex;
};

struct CharacterInstanceData {
    float4x4 transform;
    uint textureIndex;
    uint canvasIndex;
};

vertex CharacterVertexOut character_vertex(CharacterVertex in [[stage_in]],
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

fragment float4 character_fragment(CharacterVertexOut in [[stage_in]],
                                  texture2d_array<float> characterTextures [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    // Sample from the texture array using the texture index passed from vertex shader
    float4 textColor = characterTextures.sample(textureSampler, in.texCoord, in.textureIndex);
    
    // Premultiply alpha for correct blending
    return float4(textColor.rgb * textColor.a, textColor.a);
}

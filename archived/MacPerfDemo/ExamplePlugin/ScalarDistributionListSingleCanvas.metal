#include <metal_stdlib>
using namespace metal;

// Uniforms for point rendering
struct PointUniforms {
    float2 viewportSize;
    float pointRadius;
    float4 pointColor;
    float progress;
    float rowHeight;
    float rowMargin;
};

// Per-row plot bounds data
struct PlotBounds {
    float minX;
    float maxX;
    float minY;
    float maxY;
};

// Per-point position data
struct PointPosition {
    float2 position;
};

// Input vertex structure for points
struct PointVertexIn {
    float2 position [[attribute(0)]];
};

// Output vertex structure for points
struct PointVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
    float2 centerPosition;  // Center position in screen space
    float4 plotBounds;      // Min/max X and Y in screen space for clipping (minX, maxX, minY, maxY)
    float pointRadius;      // Point radius in pixels
};

// Vertex shader for single canvas points with interpolation
vertex PointVertexOut single_canvas_point_vertex(PointVertexIn in [[stage_in]],
                                                  constant PointUniforms &uniforms [[buffer(1)]],
                                                  constant PointPosition *startPositions [[buffer(2)]],
                                                  constant PointPosition *endPositions [[buffer(3)]],
                                                  constant PlotBounds *plotBounds [[buffer(4)]],
                                                  uint instanceID [[instance_id]]) {
    PointVertexOut out;
    
    // Get start and end positions
    float2 startPos = startPositions[instanceID].position;
    float2 endPos = endPositions[instanceID].position;
    
    // Interpolate between positions using pre-eased progress
    float2 interpolatedPosition = mix(startPos, endPos, uniforms.progress);
    
    // Convert to normalized device coordinates
    float2 normalizedPosition;
    normalizedPosition.x = (interpolatedPosition.x / uniforms.viewportSize.x) * 2.0 - 1.0;
    normalizedPosition.y = 1.0 - (interpolatedPosition.y / uniforms.viewportSize.y) * 2.0;
    
    out.position = float4(normalizedPosition, 0.0, 1.0);
    out.pointSize = uniforms.pointRadius * 2.0;
    out.color = uniforms.pointColor;
    out.centerPosition = interpolatedPosition;
    out.pointRadius = uniforms.pointRadius;
    
    // Calculate which row this point belongs to based on Y position
    // Each row is (rowHeight + rowMargin) pixels tall
    const float rowTotalHeight = uniforms.rowHeight + uniforms.rowMargin;
    uint rowIndex = uint(interpolatedPosition.y / rowTotalHeight);
    rowIndex = min(rowIndex, 19u); // Clamp to valid range (0-19)
    
    out.plotBounds = float4(plotBounds[rowIndex].minX, plotBounds[rowIndex].maxX, 
                           plotBounds[rowIndex].minY, plotBounds[rowIndex].maxY);
    
    return out;
}

// Fragment shader for single canvas points
fragment float4 single_canvas_point_fragment(PointVertexOut in [[stage_in]],
                                           float2 pointCoord [[point_coord]]) {
    // Calculate the actual pixel position of this fragment
    // pointCoord is in [0,1] range within the point sprite
    float2 offset = (pointCoord - float2(0.5)) * in.pointRadius * 2.0;
    float2 fragmentPos = in.centerPosition + offset;
    
    // Check if this fragment's position is outside the plot bounds
    if (fragmentPos.x < in.plotBounds.x || fragmentPos.x > in.plotBounds.y ||
        fragmentPos.y < in.plotBounds.z || fragmentPos.y > in.plotBounds.w) {
        discard_fragment();
    }
    
    // Create circular points by checking distance from center
    float2 center = float2(0.5, 0.5);
    float distance = length(pointCoord - center);
    
    // Smooth edge falloff for anti-aliasing
    float alpha = 1.0 - smoothstep(0.4, 0.5, distance);
    
    return float4(in.color.rgb, in.color.a * alpha);
}

// Background rendering structures
struct BackgroundVertex {
    float2 position [[attribute(0)]];
};

struct BackgroundVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct BackgroundUniforms {
    float2 canvasSize;
    float4 rect; // x, y, width, height
    float4 fillColor;
    float4 strokeColor;
    float cornerRadius;
    float strokeWidth;
    float isGrayRect;
};

// Vertex shader for background rectangles
vertex BackgroundVertexOut single_canvas_background_vertex(BackgroundVertex in [[stage_in]],
                                                          constant BackgroundUniforms &uniforms [[buffer(1)]]) {
    BackgroundVertexOut out;
    
    // Transform vertex position to rect bounds
    float2 position = uniforms.rect.xy + in.position * uniforms.rect.zw;
    
    // Convert to NDC
    float2 ndcPos;
    ndcPos.x = (position.x / uniforms.canvasSize.x) * 2.0 - 1.0;
    ndcPos.y = 1.0 - (position.y / uniforms.canvasSize.y) * 2.0;
    
    out.position = float4(ndcPos, 0.0, 1.0);
    out.texCoord = in.position;
    
    return out;
}

// SDF for rounded rectangle
float single_canvas_rounded_rect_sdf(float2 p, float2 size, float radius) {
    float2 d = abs(p) - size + radius;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - radius;
}

// Fragment shader for background rectangles
fragment float4 single_canvas_background_fragment(BackgroundVertexOut in [[stage_in]],
                                                 constant BackgroundUniforms &uniforms [[buffer(1)]]) {
    // Convert texture coordinates to pixel coordinates
    float2 pixelPos = in.texCoord * uniforms.rect.zw;
    
    // Center the coordinate system
    float2 p = pixelPos - uniforms.rect.zw * 0.5;
    
    // Calculate SDF
    float cornerRadius = uniforms.isGrayRect > 0.5 ? 0.0 : uniforms.cornerRadius;
    float d = single_canvas_rounded_rect_sdf(p, uniforms.rect.zw * 0.5, cornerRadius);
    
    // Smooth edges for anti-aliasing
    float fillAlpha = 1.0 - smoothstep(-1.0, 0.0, d);
    float strokeAlpha = smoothstep(-uniforms.strokeWidth - 1.0, -uniforms.strokeWidth, d) * 
                        (1.0 - smoothstep(-0.5, 0.5, d));
    
    // Mix fill and stroke colors
    float4 color = uniforms.fillColor * fillAlpha;
    color = mix(color, uniforms.strokeColor, strokeAlpha);
    
    // Apply alpha
    color.a *= fillAlpha + strokeAlpha;
    return color;
}

// Character rendering structures
struct CharacterVertex {
    float2 position [[attribute(0)]];
};

struct CharacterVertexOut {
    float4 position [[position]];
    float2 texCoord;
    uint textureIndex;
};

struct CharacterInstanceData {
    float4x4 transform;
    uint textureIndex;
};

// Vertex shader for character rendering
vertex CharacterVertexOut single_canvas_character_vertex(CharacterVertex in [[stage_in]],
                                                        constant CharacterInstanceData* instances [[buffer(2)]],
                                                        uint instanceID [[instance_id]]) {
    CharacterVertexOut out;
    constant CharacterInstanceData& instance = instances[instanceID];
    
    out.position = instance.transform * float4(in.position, 0.0, 1.0);
    out.texCoord = float2(in.position.x, 1.0 - in.position.y); // Flip Y for texture sampling
    out.textureIndex = instance.textureIndex;
    return out;
}

// Fragment shader for character rendering
fragment float4 single_canvas_character_fragment(CharacterVertexOut in [[stage_in]],
                                               texture2d_array<float> characterTextures [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    // Sample from the texture array
    float4 textColor = characterTextures.sample(textureSampler, in.texCoord, in.textureIndex);
    
    // Premultiply alpha for correct blending
    return float4(textColor.rgb * textColor.a, textColor.a);
}

#include <metal_stdlib>
using namespace metal;

// Input vertex structure
struct VertexIn {
    float2 position [[attribute(0)]];
};

// Output vertex structure (passed to fragment shader)
struct VertexOut {
    float4 position [[position]];
    float4 color;
};

// Uniform structure for color
struct ColorUniform {
    float4 color;
};

// Uniform structure for animation
struct AnimationUniform {
    float2 offset;
};

// Vertex shader - applies animation offset and sets color
vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant ColorUniform &colorUniform [[buffer(1)]],
                             constant AnimationUniform &animUniform [[buffer(2)]]) {
    VertexOut out;

    // Apply animation offset
    float2 animatedPosition = in.position + animUniform.offset;
    
    out.position = float4(animatedPosition, 0.0, 1.0);
    out.color = colorUniform.color;
    return out;
}

// Fragment shader - simply outputs the color
fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}

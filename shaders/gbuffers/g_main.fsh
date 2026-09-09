#version 430 core
#include "/lib/buffers/frame_data.glsl"

uniform sampler2D gtexture;
in vec2 texCoord;
in vec3 normal;
in vec3 viewPos; // view-space position (camera at origin), length = linear distance

/* RENDERTARGETS: 7,8 */ // colortex7=entity colour, colortex8=entity linear depth (positive = entity)
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 depthOut;

void main() {
    // 1. Discard terrain/block fragments — geometry handled by ray tracing.
    #if defined(GBUFFERS_TERRAIN) || defined(GBUFFERS_BLOCK)
    discard;
    #endif

    // 2. Alpha discard for transparent pixels
    vec4 texVal = texture(gtexture, texCoord);
    if (texVal.a < 0.01) {
        discard;
    }

    // 3. Simple directional lighting
    // Perspective interpolation does not preserve unit length.
    float intensity = dot(lightDir_global, normalize(normal));
    float lighting = clamp(10.0 * intensity, 0.2, 1.0) * 2.0;

    fragColor = vec4(pow(texVal.rgb, vec3(2.2)) * lighting, 1.0);

    #ifdef LIGHT
    fragColor.xyz *= 200.0;
    fragColor.a = 0.5; // semi-transparent overlay
    #else
    fragColor.a = 1.0; // opaque
    #endif

    // 4. Write positive linear distance to colortex8 (>0 = entity present)
    depthOut = vec4(length(viewPos), 0.0, 0.0, 1.0);
}

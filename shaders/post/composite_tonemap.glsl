#version 430 core

#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/constants.glsl"
#include "/lib/post_processing/tonemap.glsl"

in vec2 texCoord;

uniform sampler2D colortex0; // RT scene colour
uniform sampler2D colortex7; // entity/DH overlay colour (rgb)
uniform sampler2D colortex8; // linear depth:  >0 = entity (distance),
//               ==0 = no overlay

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

vec3 applyCameraVignette(vec3 hdrColor, vec2 uv) {
    // Reconstruct x/z and y/z from the exact asymmetric RT projection. This
    // makes the optical falloff respond to FOV, aspect ratio and lens shift.
    vec2 ndc = uv * 2.0 - 1.0;
    vec2 projectionScale = max(abs(rtProjectionParams.xy), vec2(1e-6));
    vec2 viewSlope = (ndc + rtProjectionParams.zw) / projectionScale;
    float cosSquared = 1.0 / (1.0 + dot(viewSlope, viewSlope));

    // Strength zero is the disabled state; one applies the ideal cos^4 law.
    float lensFalloff = mix(1.0, cosSquared * cosSquared,
        CAMERA_VIGNETTE_STRENGTH);
    return hdrColor * lensFalloff;
}

void main() {
    uvec2 xy = uvec2(gl_FragCoord.xy);
    vec4 scene = texture(colortex0, texCoord);
    float marker = texture(colortex8, texCoord).r; // magnitude = linear depth

    // --- No overlay at this pixel ---
    if (abs(marker) < 1e-6) {
        fragColor = vec4(applyCameraVignette(scene.rgb, texCoord), 1.0);
        return;
    }

    vec4 entity = texture(colortex7, texCoord);

    // --- Read RT linear depth (world-space hit distance from camera) ---
    float rtDist = readPrimaryDistance(xy);

    // --- Entity (marker > 0): linear depth comparison ---
    float entityDist = marker; // already world-space linear distance

    // RT hit sky → entity always visible
    if (rtDist < -0.5) {
        vec3 composited = mix(scene.rgb, entity.rgb * div_avgExposure, entity.a);
        fragColor = vec4(applyCameraVignette(composited, texCoord), 1.0);
        return;
    }

    // Depth comparison: entity vs RT first-surface distance.
    // Both are world-space linear distances — direct comparison.
    if (entityDist <= rtDist) {
        fragColor.rgb = mix(scene.rgb, entity.rgb * div_avgExposure, entity.a);
    } else {
        fragColor.rgb = scene.rgb;
    }
    fragColor.rgb = applyCameraVignette(fragColor.rgb, texCoord);
    fragColor.a = 1.0;
}

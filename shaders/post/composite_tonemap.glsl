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

void main() {
    uvec2 xy = uvec2(gl_FragCoord.xy);
    vec4 entity = texture(colortex7, texCoord);
    vec4 scene = texture(colortex0, texCoord);
    float marker = texture(colortex8, texCoord).r; // magnitude = linear depth

    // --- No overlay at this pixel ---
    if (abs(marker) < 1e-6) {
        fragColor = vec4(scene.rgb, 1.0);
        return;
    }

    // --- Read RT linear depth (world-space hit distance from camera) ---
    vec3 rtWorldPos_rel;
    float rtDist;
    readGeo0(GEO_N_GEO, xy, rtWorldPos_rel, rtDist);

    // --- Entity (marker > 0): linear depth comparison ---
    float entityDist = marker; // already world-space linear distance

    // RT hit sky → entity always visible
    if (rtDist < -0.5) {
        fragColor.rgb = mix(scene.rgb, entity.rgb * div_avgExposure, entity.a);
        fragColor.a = 1.0;
        return;
    }

    // Depth comparison: entity vs RT first-surface distance.
    // Both are world-space linear distances — direct comparison.
    if (entityDist <= rtDist) {
        fragColor.rgb = mix(scene.rgb, entity.rgb * div_avgExposure, entity.a);
    } else {
        fragColor.rgb = scene.rgb;
    }
    fragColor.a = 1.0;
}

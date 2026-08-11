#version 430 core

// Refraction packing and local variance preparation. A 16x16 workgroup loads
// one 20x20 shared tile and evaluates the 5x5 geometry-aware variance kernel.

layout(local_size_x = 16, local_size_y = 16) in;
#define REFRACT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

uniform vec2 resolution;

layout(rgba32f) uniform writeonly image2D colorimg3;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

const uint HALO = 2u;
const uint TILE = 16u + 2u * HALO;
const uint TILE_AREA = TILE * TILE;
const float hw[3] = float[](1.0, 0.66667, 0.44444);

#ifndef VAR_FILTER_NORMAL_POWER
#define VAR_FILTER_NORMAL_POWER ATROUS_NORMAL_POWER
#endif
#ifndef VAR_FILTER_POSITION_PARAM
#define VAR_FILTER_POSITION_PARAM ATROUS_POSITION_PARAM
#endif

struct TileSample {
    vec4 pos_oct;    // refraction endpoint position + packed direction
    vec4 color_vproj;
    vec4 H_rough;    // geometry normal + path roughness; roughness < 0 is sky
};
shared TileSample sm[TILE_AREA];

float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texSize = ivec2(resolution);
    uint tid = gl_LocalInvocationIndex;
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO);

    // All source planes are independent. Load them in one traversal and use a
    // single visibility barrier (the previous code used four traversals and
    // four barriers for the same 20x20 tile).
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        uint tx = i % TILE;
        uint ty = i / TILE;
        ivec2 cc = clamp(tileOrigin + ivec2(tx, ty), ivec2(0), texSize - 1);
        uvec2 xy = uvec2(cc);

        vec3 primaryPosition;
        float primaryDistance;
        readGeo0(GEO_N_GEO, xy, primaryPosition, primaryDistance);

        sm[i].pos_oct = vec4(0.0);
        sm[i].color_vproj = vec4(0.0);
        sm[i].H_rough = vec4(0.0, 0.0, 0.0, -1.0);
        if (primaryDistance > -0.5) {
            vec3 H;
            float roughnessUnused, pathRoughness;
            int materialUnused;
            readGeo1(GEO_N_NORMALS, xy, H, roughnessUnused,
                materialUnused, pathRoughness);

            vec3 endpointPosition, direction;
            readRefrGeo(xy, endpointPosition, direction);

            vec3 color;
            float virtualProjectionDistance, accumulatedWeightUnused;
            readRefrLight(xy, color, virtualProjectionDistance,
                accumulatedWeightUnused);
            if (any(isnan(color)) || any(isinf(color))) color = vec3(0.0);

            sm[i].pos_oct = vec4(endpointPosition, encodeNormal(direction));
            sm[i].color_vproj = vec4(color, virtualProjectionDistance);
            sm[i].H_rough = vec4(H, max(pathRoughness, 0.0));
        }
    }
    barrier();

    if (any(greaterThanEqual(gid, uvec2(resolution)))) return;

    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;
    TileSample c = sm[cy * TILE + cx];

    if (c.H_rough.w < 0.0) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        imageStore(colorimg4, ivec2(gid),
            uvec4(0u, 0u, packHalf2x16(vec2(-1.0, 0.0)), 0u));
        return;
    }

    vec3 cPos = c.pos_oct.xyz;
    vec3 cR = decodeNormal(c.pos_oct.w);
    vec3 cH = c.H_rough.xyz;
    vec3 cColor = c.color_vproj.xyz;
    float cVproj = c.color_vproj.w;
    float cRough = c.H_rough.w;

    // These terms are center-invariant. The former helper recomputed the
    // center distance, pixel footprint and reciprocal for every one of 25 taps.
    float resolutionY = max(resolution.y, 1.0);
    float pixelFootprint = max(length(cPos) / resolutionY, 1e-4);
    float invDepthScale = 1.0 / max(
        VAR_FILTER_POSITION_PARAM * pixelFootprint, 1e-6);
    float centerPlaneDistance = dot(cPos, cH);

    float sumW = 0.0, sumL = 0.0, sumL2 = 0.0;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            TileSample s = sm[(cy + uint(ky)) * TILE + (cx + uint(kx))];
            if (s.H_rough.w < 0.0) continue;

            float normalDot = clamp(dot(cH, s.H_rough.xyz), 0.0, 1.0);
            float planeDistance = abs(dot(s.pos_oct.xyz, cH) -
                centerPlaneDistance);
            float geometryWeight = pow(normalDot, VAR_FILTER_NORMAL_POWER) *
                exp2(-planeDistance * invDepthScale * LOG2_E);
            float w = hw[abs(kx)] * hw[abs(ky)] * geometryWeight;
            float L = luma(s.color_vproj.xyz);
            sumW += w;
            sumL += w * L;
            sumL2 += w * L * L;
        }
    }

    float centerLuma = luma(cColor);
    float inverseWeight = sumW > 1e-8 ? 1.0 / sumW : 0.0;
    float mean = sumW > 1e-8 ? sumL * inverseWeight : centerLuma;
    float variance = sumW > 1e-8
        ? max(sumL2 * inverseWeight - mean * mean, 0.0) : 0.0;

    float sigma = sqrt(variance);
    float clampedLuma = clamp(centerLuma, mean - 2.0 * sigma,
        mean + 2.0 * sigma);
    cColor *= clampedLuma / max(centerLuma, 1e-8);

    PackedLightSample packed_ = packSpecularSample(cPos, cR, cColor, cRough,
        variance, cVproj, cH);
    imageStore(colorimg3, ivec2(gid), packed_.data0);
    imageStore(colorimg4, ivec2(gid), packed_.data1);
}

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

// Refraction geometry is already 16-byte packed, while light is four FP16
// values and the G-buffer normal is oct-encoded. Preserve those source
// encodings in LDS and unpack only the fields consumed by a tap.
shared vec4 sm_refr_geo[TILE_AREA];
shared float sm_luma[TILE_AREA];
shared uvec2 sm_surface_packed[TILE_AREA]; // oct(H), fbits(path roughness)

float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

uvec2 loadTileSample(uint index, uvec2 xy) {
    float primaryDistance =
        geomBuffer.data[addr(GEO_N_GEO, xy)].w;

    sm_refr_geo[index] = vec4(0.0);
    sm_luma[index] = 0.0;
    sm_surface_packed[index] = uvec2(0u,
        floatBitsToUint(-1.0));

    uvec2 packedLight = uvec2(0u);
    if (primaryDistance > -0.5) {
        vec4 surface = geomBuffer.data[addr(GEO_N_NORMALS, xy)];
        vec4 refrGeo = refractBuffer.data[addr(SPEC_N_GEO, xy)];
        vec4 refrLight = refractBuffer.data[addr(SPEC_N_LIGHT, xy)];
        packedLight = uvec2(floatBitsToUint(refrLight.x),
            floatBitsToUint(refrLight.y));
        vec2 rg = unpackHalf2x16(packedLight.x);
        vec2 bv = unpackHalf2x16(packedLight.y);
        vec3 color = vec3(rg, bv.x);
        if (any(isnan(color)) || any(isinf(color))) {
            color = vec3(0.0);
            packedLight = uvec2(0u,
                packHalf2x16(vec2(0.0, bv.y)));
        }

        sm_refr_geo[index] = refrGeo;
        sm_luma[index] = luma(color);
        sm_surface_packed[index] = uvec2(floatBitsToUint(surface.x),
            floatBitsToUint(max(surface.w, 0.0)));
    }
    return packedLight;
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texSize = ivec2(resolution);
    uint tid = gl_LocalInvocationIndex;
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO);

    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;
    uint centerIndex = cy * TILE + cx;
    ivec2 centerCoord = clamp(ivec2(gid), ivec2(0), texSize - 1);
    uvec2 centerLightPacked =
        loadTileSample(centerIndex, uvec2(centerCoord));

    // Each lane owns its interior sample; lanes cooperate only on the halo.
    // Center RGB/vproj remains in registers while LDS stores neighbor luma.
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        uint tx = i % TILE;
        uint ty = i / TILE;
        bool interior = tx >= HALO && tx < HALO + 16u &&
            ty >= HALO && ty < HALO + 16u;
        if (interior) continue;
        ivec2 cc = clamp(tileOrigin + ivec2(tx, ty), ivec2(0), texSize - 1);
        loadTileSample(i, uvec2(cc));
    }
    barrier();

    if (any(greaterThanEqual(gid, uvec2(resolution)))) return;

    uvec2 cSurface = sm_surface_packed[centerIndex];
    float cRough = uintBitsToFloat(cSurface.y);

    if (cRough < 0.0) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        imageStore(colorimg4, ivec2(gid),
            uvec4(0u, 0u, packHalf2x16(vec2(-1.0, 0.0)), 0u));
        return;
    }

    vec4 cGeo = sm_refr_geo[centerIndex];
    vec2 cRG = unpackHalf2x16(centerLightPacked.x);
    vec2 cBV = unpackHalf2x16(centerLightPacked.y);
    vec3 cPos = cGeo.xyz;
    vec3 cR = decodeNormal(cGeo.w);
    vec3 cH = decodeNormal(uintBitsToFloat(cSurface.x));
    vec3 cColor = vec3(cRG, cBV.x);
    float cVproj = cBV.y;

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
            uint sampleIndex =
                (cy + uint(ky)) * TILE + (cx + uint(kx));
            uvec2 sampleSurface = sm_surface_packed[sampleIndex];
            if (uintBitsToFloat(sampleSurface.y) < 0.0) continue;
            vec4 sampleGeo = sm_refr_geo[sampleIndex];
            vec3 sampleH = decodeNormal(
                uintBitsToFloat(sampleSurface.x));

            float normalDot = clamp(dot(cH, sampleH), 0.0, 1.0);
            float planeDistance = abs(dot(sampleGeo.xyz, cH) -
                centerPlaneDistance);
            float geometryWeight = pow(normalDot, VAR_FILTER_NORMAL_POWER) *
                exp2(-planeDistance * invDepthScale * LOG2_E);
            float w = hw[abs(kx)] * hw[abs(ky)] * geometryWeight;
            float L = sm_luma[sampleIndex];
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

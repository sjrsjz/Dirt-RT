#ifndef MAXENT_SPECULAR_TEMPORAL_COMMON_GLSL
#define MAXENT_SPECULAR_TEMPORAL_COMMON_GLSL

// MaxEnt specular temporal geometry, reprojection and signal helpers.
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

float maxentPerceptualRoughness(float ggxAlpha) {
    return sqrt(clamp(ggxAlpha, 0.0, 1.0));
}

struct MaxEntGeometry {
    vec3 position;
    vec3 normal;
    float distance;
    float roughness;
    float pathRoughness;
    uint materialID;
    bool valid;
};

MaxEntGeometry maxentDecodeGeometry(uvec4 words, uvec2 pixel) {
    MaxEntGeometry g;
    float ggxAlpha;
    int materialID;
    unpackPrimaryGeometry(words, pixel, g.position, g.distance, g.normal,
        ggxAlpha, materialID, g.pathRoughness);
    g.roughness = maxentPerceptualRoughness(ggxAlpha);
    g.materialID = uint(max(materialID, 0));
    g.valid = g.distance >= 0.0 && !isnan(g.distance)
        && !isinf(g.distance);
    return g;
}

MaxEntGeometry maxentLoadGeometry(uvec2 pixel) {
    return maxentDecodeGeometry(readPrimaryGeometryWords(pixel), pixel);
}

vec3 maxentSafeNormalize(vec3 v, vec3 fallback) {
    float l2 = dot(v, v);
    return l2 > 1e-20 ? v * inversesqrt(l2) : fallback;
}

vec3 maxentFiniteColor(vec3 c) {
    if (any(isnan(c)) || any(isinf(c))) return vec3(0.0);
    return clamp(c, vec3(0.0), vec3(65504.0));
}

bool maxentInBounds(ivec2 p, ivec2 size) {
    return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, size));
}

vec2 maxentHash2(uvec2 pixel, uint frame) {
    uvec2 v = pixel ^ uvec2(frame * 0x9e3779b9u,
        (frame + 1u) * 0x85ebca6bu);
    v ^= v.yx >> 16u;
    v *= uvec2(0x7feb352du, 0x846ca68bu);
    v ^= v.yx >> 15u;
    return vec2(v & 0x00ffffffu) * (1.0 / 16777216.0);
}

vec2 maxentProjectPrevious(vec3 currentRelativePosition, vec3 cameraDelta) {
    vec4 clip = rtPrevViewProjection *
        vec4(currentRelativePosition + cameraDelta, 1.0);
    if (clip.w <= 1e-8 || any(isnan(clip)) || any(isinf(clip)))
        return vec2(-2.0);
    return clip.xy / clip.w * 0.5 + 0.5;
}

uint maxentPackHalf2(float a, float b) {
    return packHalf2x16(clamp(vec2(a, b), vec2(-65504.0), vec2(65504.0)));
}

SpecularMaxEnt maxentMixMaxEnt(SpecularMaxEnt a, SpecularMaxEnt b, float t) {
    SpecularMaxEnt s;
    s.maxEntY = mix(a.maxEntY, b.maxEntY, t);
    s.CoCg = mix(a.CoCg, b.CoCg, t);
    return sanitizeSpecularMaxEnt(s);
}

SpecularMaxEnt maxentWeightedMaxEnt(SpecularMaxEnt a, float wa,
        SpecularMaxEnt b, float wb) {
    SpecularMaxEnt s;
    s.maxEntY = a.maxEntY * wa + b.maxEntY * wb;
    s.CoCg = a.CoCg * wa + b.CoCg * wb;
    return s;
}

SpecularMaxEnt maxentScaleMaxEnt(SpecularMaxEnt s, float scale) {
    s.maxEntY *= scale;
    s.CoCg *= scale;
    return sanitizeSpecularMaxEnt(s);
}

// Raw temporal input: MaxEnt6 plus the reflection hit distance.
struct MaxEntSpecularInput {
    SpecularMaxEnt signal;
    float hitDistance;
};

uvec4 maxentPackSpecularInput(MaxEntSpecularInput s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, maxentPackHalf2(s.hitDistance, 0.0));
}

MaxEntSpecularInput maxentUnpackSpecularInput(uvec4 p) {
    MaxEntSpecularInput s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    s.hitDistance = max(unpackHalf2x16(p.w).x, 0.0);
    return s;
}

struct MaxEntTemporalSignal {
    SpecularMaxEnt signal;
    float secondMoment;
    float historyLength;
};

uvec4 maxentPackTemporal(MaxEntTemporalSignal s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, maxentPackHalf2(
        encodeSqrtMomentFP16(s.secondMoment), s.historyLength));
}

MaxEntTemporalSignal maxentUnpackTemporal(uvec4 p) {
    MaxEntTemporalSignal s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    vec2 momentHistory = unpackHalf2x16(p.w);
    s.secondMoment = decodeSqrtMomentFP16(momentHistory.x);
    s.historyLength = max(momentHistory.y, 0.0);
    return s;
}

uvec4 maxentPackTemporalAux(float hitDistance) {
    return uvec4(maxentPackHalf2(hitDistance, 0.0), 0u, 0u, 0u);
}

float maxentUnpackTemporalHitDistance(uvec4 p) {
    return max(unpackHalf2x16(p.x).x, 0.0);
}

float maxentSpecLobeTanHalfAngle(float roughness, float volumeFraction) {
    roughness = clamp(roughness, 0.0, 1.0);
    volumeFraction = clamp(volumeFraction, 0.0, 1.0);
    return roughness * roughness * volumeFraction /
        max(1.0 - volumeFraction, 1e-6);
}

#endif // MAXENT_SPECULAR_TEMPORAL_COMMON_GLSL

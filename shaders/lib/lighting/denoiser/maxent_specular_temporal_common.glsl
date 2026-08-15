#ifndef MAXENT_SPECULAR_TEMPORAL_COMMON_GLSL
#define MAXENT_SPECULAR_TEMPORAL_COMMON_GLSL

// MaxEnt specular temporal/prepass geometry, reprojection and signal helpers.
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

vec3 maxentMaxEntYCoCg(SpecularMaxEnt s) {
    s = sanitizeSpecularMaxEnt(s);
    return vec3(s.maxEntY.w, s.CoCg);
}

SpecularMaxEnt maxentSetMaxEntYCoCg(SpecularMaxEnt s, vec3 ycocg) {
    ycocg.x = max(ycocg.x, 0.0);
    float scale = ycocg.x / max(s.maxEntY.w, 1e-8);
    s.maxEntY.xyz *= scale;
    s.maxEntY.w = ycocg.x;
    s.CoCg = ycocg.yz;
    return sanitizeSpecularMaxEnt(s);
}

// Raw/prepass and A-trous payload: MaxEnt6 plus two scalar slots.
struct MaxEntPrepassSignal {
    SpecularMaxEnt signal;
    float hitDistance;
};

uvec4 maxentPackPrepass(MaxEntPrepassSignal s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, maxentPackHalf2(s.hitDistance, 0.0));
}

MaxEntPrepassSignal maxentUnpackPrepass(uvec4 p) {
    MaxEntPrepassSignal s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    s.hitDistance = max(unpackHalf2x16(p.w).x, 0.0);
    return s;
}

struct MaxEntSlowSignal {
    SpecularMaxEnt signal;
    float secondMoment;
    float historyLength; // slow-history Kish N_eff
};

uvec4 maxentPackSlow(MaxEntSlowSignal s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, maxentPackHalf2(
        encodeSqrtMomentFP16(s.secondMoment), s.historyLength));
}

MaxEntSlowSignal maxentUnpackSlow(uvec4 p) {
    MaxEntSlowSignal s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    vec2 momentHistory = unpackHalf2x16(p.w);
    s.secondMoment = decodeSqrtMomentFP16(momentHistory.x);
    s.historyLength = max(momentHistory.y, 0.0);
    return s;
}

struct MaxEntFastSignal {
    SpecularMaxEnt signal;
    float hitDistance;
    float historyLength; // responsive-history Kish N_eff
};

uvec4 maxentPackFast(MaxEntFastSignal s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, maxentPackHalf2(s.hitDistance, s.historyLength));
}

MaxEntFastSignal maxentUnpackFast(uvec4 p) {
    MaxEntFastSignal s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    vec2 hn = unpackHalf2x16(p.w);
    s.hitDistance = max(hn.x, 0.0);
    s.historyLength = max(hn.y, 0.0);
    return s;
}

float maxentSpecLobeTanHalfAngle(float roughness, float volumeFraction) {
    roughness = clamp(roughness, 0.0, 1.0);
    volumeFraction = clamp(volumeFraction, 0.0, 1.0);
    return roughness * roughness * volumeFraction /
        max(1.0 - volumeFraction, 1e-6);
}

float maxentSpatialPlaneExponent(vec3 centerPos, vec3 centerNormal,
        vec3 samplePos) {
    float resolutionY = max(float(resolution_global.y), 1.0);
    float centerDistance = max(length(centerPos), 0.001);
    float footprintDistance = max(centerDistance, resolutionY * 1e-5);
    float invPixelFootprint = resolutionY / max(
        MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE * footprintDistance, resolutionY * 1e-6);
    return abs(dot(samplePos, centerNormal)
        - dot(centerPos, centerNormal)) * invPixelFootprint;
}

float maxentSpatialPlaneWeight(vec3 centerPos, vec3 centerNormal,
        vec3 samplePos) {
    return exp(-maxentSpatialPlaneExponent(centerPos,
        centerNormal, samplePos));
}

vec2 maxentRoughnessWeightParams(float roughness, float fraction) {
    const float sensitivity = 0.03;
    float a = 1.0 / mix(sensitivity, 1.0,
        clamp(roughness * fraction, 0.0, 1.0));
    return vec2(a, -roughness * a);
}

float maxentExponentialWeight(float x, vec2 p) {
    return exp(-3.0 * abs(x * p.x + p.y));
}

#endif // MAXENT_SPECULAR_TEMPORAL_COMMON_GLSL

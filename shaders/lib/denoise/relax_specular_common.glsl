#ifndef RELAX_SPECULAR_COMMON_GLSL
#define RELAX_SPECULAR_COMMON_GLSL

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

float relaxPerceptualRoughness(float ggxAlpha) {
    return sqrt(clamp(ggxAlpha, 0.0, 1.0));
}

vec3 relaxSafeNormalize(vec3 v, vec3 fallback) {
    float l2 = dot(v, v);
    return l2 > 1e-20 ? v * inversesqrt(l2) : fallback;
}

vec3 relaxFiniteColor(vec3 c) {
    if (any(isnan(c)) || any(isinf(c))) return vec3(0.0);
    return clamp(c, vec3(0.0), vec3(65504.0));
}

bool relaxInBounds(ivec2 p, ivec2 size) {
    return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, size));
}

vec2 relaxHash2(uvec2 pixel, uint frame) {
    uvec2 v = pixel ^ uvec2(frame * 0x9e3779b9u,
        (frame + 1u) * 0x85ebca6bu);
    v ^= v.yx >> 16u;
    v *= uvec2(0x7feb352du, 0x846ca68bu);
    v ^= v.yx >> 15u;
    return vec2(v & 0x00ffffffu) * (1.0 / 16777216.0);
}

vec2 relaxProjectPrevious(vec3 currentRelativePosition, vec3 cameraDelta) {
    vec4 clip = rtPrevViewProjection *
        vec4(currentRelativePosition + cameraDelta, 1.0);
    if (clip.w <= 1e-8 || any(isnan(clip)) || any(isinf(clip)))
        return vec2(-2.0);
    return clip.xy / clip.w * 0.5 + 0.5;
}

uint relaxPackHalf2(float a, float b) {
    return packHalf2x16(clamp(vec2(a, b), vec2(-65504.0), vec2(65504.0)));
}

SpecularMaxEnt relaxMixMaxEnt(SpecularMaxEnt a, SpecularMaxEnt b, float t) {
    SpecularMaxEnt s;
    s.aliceY = mix(a.aliceY, b.aliceY, t);
    s.CoCg = mix(a.CoCg, b.CoCg, t);
    return sanitizeSpecularMaxEnt(s);
}

SpecularMaxEnt relaxWeightedMaxEnt(SpecularMaxEnt a, float wa,
        SpecularMaxEnt b, float wb) {
    SpecularMaxEnt s;
    s.aliceY = a.aliceY * wa + b.aliceY * wb;
    s.CoCg = a.CoCg * wa + b.CoCg * wb;
    return s;
}

SpecularMaxEnt relaxScaleMaxEnt(SpecularMaxEnt s, float scale) {
    s.aliceY *= scale;
    s.CoCg *= scale;
    return sanitizeSpecularMaxEnt(s);
}

vec3 relaxMaxEntYCoCg(SpecularMaxEnt s) {
    s = sanitizeSpecularMaxEnt(s);
    return vec3(s.aliceY.w, s.CoCg);
}

// Feature z = (Y * direction, Y), so E[|z|^2] = 2 E[Y^2].
// This is the covariance trace in the same four-dimensional units as the
// MaxEnt Bures distance used by the spatial light-field weight.
float relaxMaxEntLightFieldVariance(vec4 meanAliceY, float meanY2) {
    return max(2.0 * meanY2 - dot(meanAliceY, meanAliceY), 0.0);
}

SpecularMaxEnt relaxSetMaxEntYCoCg(SpecularMaxEnt s, vec3 ycocg) {
    ycocg.x = max(ycocg.x, 0.0);
    float scale = ycocg.x / max(s.aliceY.w, 1e-8);
    s.aliceY.xyz *= scale;
    s.aliceY.w = ycocg.x;
    s.CoCg = ycocg.yz;
    return sanitizeSpecularMaxEnt(s);
}

// Raw/prepass and A-trous payload: MaxEnt6 plus two scalar slots.
struct RelaxPrepassSignal {
    SpecularMaxEnt signal;
    float hitDistance;
};

uvec4 relaxPackPrepass(RelaxPrepassSignal s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, relaxPackHalf2(s.hitDistance, 0.0));
}

RelaxPrepassSignal relaxUnpackPrepass(uvec4 p) {
    RelaxPrepassSignal s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    s.hitDistance = max(unpackHalf2x16(p.w).x, 0.0);
    return s;
}

struct RelaxSlowSignal {
    SpecularMaxEnt signal;
    float secondMoment;
};

uvec4 relaxPackSlow(RelaxSlowSignal s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, relaxPackHalf2(
        encodeSqrtMomentFP16(s.secondMoment), 0.0));
}

RelaxSlowSignal relaxUnpackSlow(uvec4 p) {
    RelaxSlowSignal s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    s.secondMoment = decodeSqrtMomentFP16(unpackHalf2x16(p.w).x);
    return s;
}

// Responsive history intentionally carries only total YCoCg. The angular
// state remains in the slow MaxEnt record, where it is stable enough to use.
struct RelaxFastSignal {
    vec3 YCoCg;
    float hitDistance;
    float historyLength;
    float confidence;
    uint materialID;
};

uvec4 relaxPackFast(RelaxFastSignal s) {
    return uvec4(relaxPackHalf2(s.YCoCg.x, s.YCoCg.y),
        relaxPackHalf2(s.YCoCg.z, s.hitDistance),
        relaxPackHalf2(s.historyLength, s.confidence), s.materialID);
}

RelaxFastSignal relaxUnpackFast(uvec4 p) {
    RelaxFastSignal s;
    vec2 yc = unpackHalf2x16(p.x);
    vec2 ch = unpackHalf2x16(p.y);
    vec2 nc = unpackHalf2x16(p.z);
    s.YCoCg = vec3(yc, ch.x);
    s.hitDistance = max(ch.y, 0.0);
    s.historyLength = max(nc.x, 0.0);
    s.confidence = clamp(nc.y, 0.0, 1.0);
    s.materialID = p.w;
    return s;
}

// After temporal clamping AliceY lives in RGBA32F; this record carries its
// six remaining scalars in the existing RGBA32UI attachment.
struct RelaxPostSignal {
    vec2 CoCg;
    float secondMoment;
    float hitDistance;
    float historyLength;
    float confidence;
    uint materialID;
};

uvec4 relaxPackPost(RelaxPostSignal s) {
    return uvec4(relaxPackHalf2(s.CoCg.x, s.CoCg.y),
        relaxPackHalf2(encodeSqrtMomentFP16(s.secondMoment), s.hitDistance),
        relaxPackHalf2(s.historyLength, s.confidence), s.materialID);
}

RelaxPostSignal relaxUnpackPost(uvec4 p) {
    RelaxPostSignal s;
    s.CoCg = unpackHalf2x16(p.x);
    vec2 mh = unpackHalf2x16(p.y);
    vec2 nc = unpackHalf2x16(p.z);
    s.secondMoment = decodeSqrtMomentFP16(mh.x);
    s.hitDistance = max(mh.y, 0.0);
    s.historyLength = max(nc.x, 0.0);
    s.confidence = clamp(nc.y, 0.0, 1.0);
    s.materialID = p.w;
    return s;
}

struct RelaxSpatialSignal {
    SpecularMaxEnt signal;
    float variance;
    float hitDistance;
};

uvec4 relaxPackSpatial(RelaxSpatialSignal s) {
    uvec3 p = packSpecularMaxEnt(s.signal);
    return uvec4(p, relaxPackHalf2(s.variance, s.hitDistance));
}

RelaxSpatialSignal relaxUnpackSpatial(uvec4 p) {
    RelaxSpatialSignal s;
    s.signal = unpackSpecularMaxEnt(p.xyz);
    vec2 vh = unpackHalf2x16(p.w);
    s.variance = max(vh.x, 0.0);
    s.hitDistance = max(vh.y, 0.0);
    return s;
}

uint relaxPackNormalMaterial(vec3 n, uint materialID) {
    n = relaxSafeNormalize(n, vec3(0.0, 1.0, 0.0));
    vec2 p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * sign_not_zero(p);
    uint oct8 = packUnorm4x8(vec4(p * 0.5 + 0.5, 0.0, 0.0)) & 0xffffu;
    return oct8 | ((materialID & 0xffffu) << 16u);
}

void relaxUnpackNormalMaterial(uint p, out vec3 n, out uint materialID) {
    vec2 f = unpackUnorm4x8(p & 0xffffu).xy * 2.0 - 1.0;
    n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * sign_not_zero(n.xy);
    n = relaxSafeNormalize(n, vec3(0.0, 1.0, 0.0));
    materialID = p >> 16u;
}

float relaxSpecLobeTanHalfAngle(float roughness, float volumeFraction) {
    roughness = clamp(roughness, 0.0, 1.0);
    volumeFraction = clamp(volumeFraction, 0.0, 1.0);
    return roughness * roughness * volumeFraction /
        max(1.0 - volumeFraction, 1e-6);
}

float relaxSpatialPlaneExponent(vec3 centerPos, vec3 centerNormal,
        vec3 samplePos) {
    float resolutionY = max(float(resolution_global.y), 1.0);
    float centerDistance = max(length(centerPos), 0.001);
    float footprintDistance = max(centerDistance, resolutionY * 1e-5);
    float invPixelFootprint = resolutionY / max(
        ATROUS_POSITION_PARAM * footprintDistance, resolutionY * 1e-6);
    return abs(dot(samplePos, centerNormal)
        - dot(centerPos, centerNormal)) * invPixelFootprint;
}

float relaxSpatialPlaneWeight(vec3 centerPos, vec3 centerNormal,
        vec3 samplePos) {
    return exp(-relaxSpatialPlaneExponent(centerPos,
        centerNormal, samplePos));
}

vec2 relaxRoughnessWeightParams(float roughness, float fraction) {
    const float sensitivity = 0.03;
    float a = 1.0 / mix(sensitivity, 1.0,
        clamp(roughness * fraction, 0.0, 1.0));
    return vec2(a, -roughness * a);
}

float relaxExponentialWeight(float x, vec2 p) {
    return exp(-3.0 * abs(x * p.x + p.y));
}

float relaxHitDistanceWeight(float centerHitDistance,
        float sampleHitDistance, float centerRoughness) {
    float hitScale = max(max(centerHitDistance, sampleHitDistance), 1.0);
    float hitSigma = hitScale * mix(0.02, 0.5,
        clamp(centerRoughness, 0.0, 1.0)) + 1e-5;
    float similarity = exp(-abs(sampleHitDistance - centerHitDistance)
        / hitSigma);
    return mix(RELAX_MIN_HIT_DISTANCE_WEIGHT, 1.0, similarity);
}

#endif

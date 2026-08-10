#ifndef RELAX_SPECULAR_COMMON_GLSL
#define RELAX_SPECULAR_COMMON_GLSL

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

// RELAX consumes perceptual roughness. Dirt RT stores GGX alpha in the G-buffer.
float relaxPerceptualRoughness(float ggxAlpha) {
    return sqrt(clamp(ggxAlpha, 0.0, 1.0));
}

float relaxLuma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 relaxSafeNormalize(vec3 v, vec3 fallback) {
    float l2 = dot(v, v);
    return l2 > 1e-20 ? v * inversesqrt(l2) : fallback;
}

vec3 relaxFiniteColor(vec3 c) {
    bvec3 bad = bvec3(isnan(c.x) || isinf(c.x),
                      isnan(c.y) || isinf(c.y),
                      isnan(c.z) || isinf(c.z));
    if (any(bad)) return vec3(0.0);
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

vec2 relaxCurrentUv(uvec2 pixel) {
    // Vulkanite primary rays are generated at integer pixel coordinates.
    return vec2(pixel) / vec2(resolution_global);
}

vec2 relaxProjectPrevious(vec3 currentRelativePosition, vec3 cameraDelta) {
    vec3 previousRelativePosition = currentRelativePosition + cameraDelta;
    vec4 clip = rtPrevViewProjection * vec4(previousRelativePosition, 1.0);
    if (abs(clip.w) < 1e-8) return vec2(-2.0);
    return clip.xy / clip.w * 0.5 + 0.5;
}

vec2 relaxProjectPreviousRelative(vec3 previousRelativePosition) {
    vec4 clip = rtPrevViewProjection * vec4(previousRelativePosition, 1.0);
    if (abs(clip.w) < 1e-8) return vec2(-2.0);
    return clip.xy / clip.w * 0.5 + 0.5;
}

uint relaxPackHalf2(float a, float b) {
    vec2 v = clamp(vec2(a, b), vec2(-65504.0), vec2(65504.0));
    return packHalf2x16(v);
}

// Prepass transient: 8 FP16 values in the existing rgba32ui image. Endpoint
// moments are virtual-image offsets in world axes and use the common
// VPROJDIST_SKY scale. The RMS encoding is only a storage transform; all
// filtering uses decoded E[|X|^2].
struct RelaxPrepassSignal {
    vec3 radiance;
    RelaxEndpointMoments endpoint;
};

float relaxEndpointDistanceScale() {
    // Endpoint offsets are normalized before FP16 storage. Keep the common
    // linear scale representable even if the Iris option exceeds FP16 range.
    return clamp(VPROJDIST_SKY, 1.0, 65504.0);
}

float relaxEndpointMeanDistance(RelaxEndpointMoments endpoint) {
    endpoint = sanitizeRelaxEndpointMoments(endpoint);
    return relaxEndpointMomentsValid(endpoint)
        ? length(endpoint.mean) * relaxEndpointDistanceScale() : 0.0;
}

uvec4 relaxPackPrepass(RelaxPrepassSignal s) {
    s.endpoint = sanitizeRelaxEndpointMoments(s.endpoint);
    uvec2 packedEndpoint = relaxPackEndpointMoments(s.endpoint);
    return uvec4(
        relaxPackHalf2(s.radiance.r, s.radiance.g),
        relaxPackHalf2(s.radiance.b, 0.0),
        packedEndpoint);
}

RelaxPrepassSignal relaxUnpackPrepass(uvec4 p) {
    RelaxPrepassSignal s;
    vec2 rg = unpackHalf2x16(p.x);
    vec2 bh = unpackHalf2x16(p.y);
    s.radiance = relaxFiniteColor(vec3(rg, bh.x));
    s.endpoint = relaxUnpackEndpointMoments(p.zw);
    return s;
}

struct RelaxSlowSignal {
    vec3 radiance;
    float secondMoment;
    float historyLength;
    float confidence;
};

struct RelaxFastSignal {
    vec3 radiance;
    // Derived every frame from the temporally filtered four endpoint moments.
    // This is transient spatial-filter metadata, not an independent history.
    float endpointDistance;
    float historyLength;
    float confidence;
    uint materialID;
};

uvec4 relaxPackSlow(RelaxSlowSignal s) {
    return uvec4(
        relaxPackHalf2(s.radiance.r, s.radiance.g),
        relaxPackHalf2(s.radiance.b,
            encodeSqrtMomentFP16(s.secondMoment)),
        relaxPackHalf2(s.historyLength, s.confidence),
        0u);
}

RelaxSlowSignal relaxUnpackSlow(uvec4 p) {
    RelaxSlowSignal s;
    vec2 rg = unpackHalf2x16(p.x);
    vec2 bm = unpackHalf2x16(p.y);
    vec2 hc = unpackHalf2x16(p.z);
    s.radiance = vec3(rg, bm.x);
    s.secondMoment = decodeSqrtMomentFP16(bm.y);
    s.historyLength = max(hc.x, 0.0);
    s.confidence = clamp(hc.y, 0.0, 1.0);
    return s;
}

uvec4 relaxPackFast(RelaxFastSignal s) {
    return uvec4(
        relaxPackHalf2(s.radiance.r, s.radiance.g),
        relaxPackHalf2(s.radiance.b, s.endpointDistance),
        relaxPackHalf2(s.historyLength, s.confidence), s.materialID);
}

RelaxFastSignal relaxUnpackFast(uvec4 p) {
    RelaxFastSignal s;
    vec2 rg = unpackHalf2x16(p.x);
    vec2 bh = unpackHalf2x16(p.y);
    vec2 hc = unpackHalf2x16(p.z);
    s.radiance = vec3(rg, bh.x);
    s.endpointDistance = max(bh.y, 0.0);
    s.historyLength = max(hc.x, 0.0);
    s.confidence = clamp(hc.y, 0.0, 1.0);
    s.materialID = p.w;
    return s;
}

struct RelaxSpatialSignal {
    vec3 radiance;
    float roughness;
    float variance;
    float endpointDistance;
    float historyLength;
    float confidence;
};

uvec4 relaxPackSpatial(RelaxSpatialSignal s) {
    return uvec4(
        relaxPackHalf2(s.radiance.r, s.radiance.g),
        relaxPackHalf2(s.radiance.b, s.roughness),
        relaxPackHalf2(s.variance, s.endpointDistance),
        relaxPackHalf2(s.historyLength, s.confidence));
}

RelaxSpatialSignal relaxUnpackSpatial(uvec4 p) {
    RelaxSpatialSignal s;
    vec2 rg = unpackHalf2x16(p.x);
    vec2 br = unpackHalf2x16(p.y);
    vec2 vh = unpackHalf2x16(p.z);
    vec2 hc = unpackHalf2x16(p.w);
    s.radiance = vec3(rg, br.x);
    s.roughness = clamp(br.y, 0.0, 1.0);
    s.variance = max(vh.x, 0.0);
    s.endpointDistance = max(vh.y, 0.0);
    s.historyLength = max(hc.x, 0.0);
    s.confidence = clamp(hc.y, 0.0, 1.0);
    return s;
}

// 8-bit octahedral normal plus a 16-bit material identifier. NRD explicitly
// supports 8/10-bit normals; this keeps the transient geometry texture compact.
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

vec3 relaxRgbToYCoCg(vec3 c) {
    float Y = dot(c, vec3(0.25, 0.5, 0.25));
    return vec3(Y, c.r - c.b, c.g - 0.5 * (c.r + c.b));
}

vec3 relaxYCoCgToRgb(vec3 c) {
    float t = c.x - 0.5 * c.z;
    return vec3(t + 0.5 * c.y, c.x + 0.5 * c.z, t - 0.5 * c.y);
}

float relaxSpecMagicCurve(float roughness) {
    float f = 1.0 - exp2(-200.0 * roughness * roughness);
    return f * pow(clamp(roughness, 0.0, 1.0), 0.25);
}

float relaxSpecLobeTanHalfAngle(float roughness, float volumeFraction) {
    roughness = clamp(roughness, 0.0, 1.0);
    volumeFraction = clamp(volumeFraction, 0.0, 1.0);
    return roughness * roughness * volumeFraction /
        max(1.0 - volumeFraction, 1e-6);
}

float relaxPlaneWeight(vec3 centerPos, vec3 centerNormal, vec3 samplePos, float threshold) {
    return float(abs(dot(samplePos - centerPos, centerNormal)) <= threshold);
}

vec2 relaxRoughnessWeightParams(float roughness, float fraction) {
    const float sensitivity = 0.03;
    float a = 1.0 / mix(sensitivity, 1.0, clamp(roughness * fraction, 0.0, 1.0));
    return vec2(a, -roughness * a);
}

float relaxExponentialWeight(float x, vec2 p) {
    return exp(-3.0 * abs(x * p.x + p.y));
}

vec2 relaxNormalWeightParams(float roughness, float historyLength, float confidence) {
    float relaxation = clamp(historyLength / 5.0, 0.0, 1.0);
    relaxation *= mix(1.0, confidence, RELAX_NORMAL_RELAXATION);
    float angle = atan(relaxSpecLobeTanHalfAngle(roughness, RELAX_LOBE_ANGLE_FRACTION));
    angle *= 10.0 - 9.0 * relaxation;
    angle = min(0.5 * PI, angle + RELAX_LOBE_ANGLE_SLACK);
    return vec2(max(angle, 1.5 / 255.0), 0.9 + 0.1 * relaxation);
}

float relaxSpecularNormalWeight(vec2 params, vec3 n0, vec3 n, vec3 v0, vec3 v) {
    float angle = acos(clamp(min(dot(n0, n), dot(v0, v)), -1.0, 1.0));
    float t = clamp(angle / params.x, 0.0, 1.0);
    t = t * t * (3.0 - 2.0 * t);
    return clamp(1.0 - t * params.y, 0.0, 1.0);
}

#endif

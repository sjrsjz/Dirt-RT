#version 430 compatibility

// ===========================================================================
// Pass: ReSTIR 空域+时域加权蓄水池 (composite58)
// ===========================================================================
// 8 点 Poisson 盘空间采样 (无共享内存) + 降噪先验 + 时域重投影
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/common.glsl"
#include "/lib/lighting/alice.glsl"

uniform sampler2D colortex4; // atrous 降噪 ALICE (packAlice 格式)
uniform sampler2D colortex6; // temporal_diffuse validKernelWeight

layout(rgba32f) uniform writeonly image2D colorimg6;

const uint POISSON_N = 8u;
const float POISSON_R0 = 16.0;
const float GUIDE_MAX_M = 64.0;

// NRD Poisson 盘: .xy=偏移 .z=length .w=高斯权重
const vec4 POISSON[8] = {
    vec4(-0.4706069, -0.4427112, +0.6461146, +0.81170),
    vec4(-0.9057375, +0.3003471, +0.9542373, +0.63422),
    vec4(-0.3487388, +0.4037880, +0.5335386, +0.86734),
    vec4(+0.1023042, +0.6439373, +0.6520134, +0.80847),
    vec4(+0.5699277, +0.3513750, +0.6695386, +0.79925),
    vec4(+0.2939128, -0.1131226, +0.3149309, +0.95161),
    vec4(+0.7836658, -0.4208784, +0.8895339, +0.67328),
    vec4(+0.1564120, -0.8198990, +0.8346850, +0.70589)
    };

// ---------------------------------------------------------------------------
// PCG RNG
// ---------------------------------------------------------------------------
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float nextFloat(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) / 4294967296.0;
}

// ---------------------------------------------------------------------------
// 蓄水池
// ---------------------------------------------------------------------------
struct Reservoir {
    vec4 y;
    float weightSum;
    float M;
    float target;
};
void reservoirInit(out Reservoir r) {
    r.y = vec4(0.0);
    r.weightSum = 0.0;
    r.M = 0.0;
    r.target = 0.0;
}
void reservoirUpdate(inout Reservoir r, vec4 candidate, float w, float m, inout uint seed) {
    r.weightSum += w;
    r.M += m;
    if (nextFloat(seed) * r.weightSum < w) {
        r.y = candidate;
        r.target = w;
    }
}
float reservoirW(Reservoir r) {
    return r.weightSum / max(r.M * r.target, 1e-20);
}

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------
bool isSky(vec4 y) {
    return y.w <= 1e-8;
}

vec2 packAliceHalf(vec4 y) {
    return vec2(uintBitsToFloat(packHalf2x16(y.xy)), uintBitsToFloat(packHalf2x16(y.zw)));
}
vec4 unpackAliceHalf(vec2 p) {
    return vec4(unpackHalf2x16(floatBitsToUint(p.x)), unpackHalf2x16(floatBitsToUint(p.y)));
}

float guideTarget(vec4 y, vec3 N) {
    return alice_irradiance(y, N);
}

// ---------------------------------------------------------------------------
// Phase 2: 8 点 Poisson 盘空间蓄水池 (直接 SSBO 读)
// ---------------------------------------------------------------------------
Reservoir spatialReservoir(uvec2 gid, vec3 centerNormal, float centerDist, inout uint seed) {
    Reservoir r;
    reservoirInit(r);
    ivec2 texSize = ivec2(resolution_global);

    float theta = 2.0 * PI * nextFloat(seed);
    mat2 rot = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * POISSON_R0;

    for (uint k = 0u; k < POISSON_N; k++) {
        vec4 ps = POISSON[k];
        ivec2 off = ivec2(round(rot * ps.xy));
        ivec2 sc = ivec2(gid) + off;
        if (any(lessThan(sc, ivec2(0))) || any(greaterThanEqual(sc, texSize))) continue;

        uvec2 xy = uvec2(sc);
        AliceEncoding alice;
        float meanY2_unused;
        readDiffuseLightRT(xy, alice, meanY2_unused);
        vec4 y = alice.aliceY;
        if (isSky(y) || readDiffuseSurfaceMask(xy) < 0.5) continue;

        // 深度不连续拒绝
        vec3 pos;
        float dist;
        readGeo0(GEO_N_GEO, xy, pos, dist);
        float depthDiff = abs(dist - centerDist) / max(abs(centerDist) + 1e-4, 1.0);
        float geomW = exp2(-depthDiff * float(ATROUS_POSITION_PARAM));
        if (geomW <= 1e-4) continue;

        float w = guideTarget(y, centerNormal) * geomW;
        if (w > 1e-8) reservoirUpdate(r, y, w, 1.0, seed);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Phase 3: 降噪先验
// ---------------------------------------------------------------------------
void addDenoisedPrior(inout Reservoir r, ivec2 pix, vec3 centerNormal, inout uint seed) {
    vec4 raw = texelFetch(colortex4, pix, 0);
    AliceEncoding a = unpackAlice(raw.x, raw.y, raw.z);
    vec4 y = a.aliceY;
    if (isSky(y)) return;
    float w = guideTarget(y, centerNormal);
    if (w > 1e-8) reservoirUpdate(r, y, w, 1.0, seed);
}

// ---------------------------------------------------------------------------
// Phase 4: 2×2 随机重投影历史蓄水池
// ---------------------------------------------------------------------------
struct StoredReservoir {
    vec4 y;
    float W;
    float M;
};

StoredReservoir loadStored(ivec2 pix) {
    uvec4 raw = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, uvec2(pix))];
    StoredReservoir s;
    s.y = vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
    s.W = uintBitsToFloat(raw.z);
    s.M = uintBitsToFloat(raw.w);
    return s;
}

bool sampleHistory(uvec2 gxy, inout uint seed, out StoredReservoir result) {
    result.y = vec4(0.0);
    result.W = 0.0;
    result.M = 0.0;

    vec3 curPos;
    float curDist;
    readGeo0(GEO_N_GEO, gxy, curPos, curDist);
    if (curDist <= -0.5) return false;

    vec3 prevPos = curPos + camPos - prevRaytracingCamPos;
    vec4 clip = rtPrevProjection * rtPrevModelView * vec4(prevPos, 1.0);
    if (clip.w <= 1e-6) return false;
    vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) return false;

    vec2 prevCoord = uv * vec2(resolution_global);
    ivec2 p0 = ivec2(floor(prevCoord));
    vec2 f = fract(prevCoord);

    ivec2 offset = ivec2(nextFloat(seed) < f.x ? 1 : 0, nextFloat(seed) < f.y ? 1 : 0);
    result = loadStored(clamp(p0 + offset, ivec2(0), ivec2(resolution_global) - 1));
    return result.W > 0.0 && result.M > 0.0;
}

// ---------------------------------------------------------------------------
// Phase 5: 历史 RIS 合并
// ---------------------------------------------------------------------------
void combineWithHistory(inout Reservoir r, StoredReservoir hist, float temporalConf, vec3 centerNormal, inout uint seed) {
    float targetNow = guideTarget(hist.y, centerNormal);
    if (targetNow <= 1e-8) return;

    float reusedM = min(hist.M, GUIDE_MAX_M) * clamp(temporalConf, 0.0, 1.0);
    if (reusedM <= 0.0) return;

    float candidateWeight = targetNow * hist.W * reusedM;
    reservoirUpdate(r, hist.y, candidateWeight, reusedM, seed);
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    ivec2 texSize = ivec2(resolution_global);
    if (any(greaterThanEqual(gid, uvec2(texSize)))) return;

    uvec2 gxy = gid;
    ivec2 pix = ivec2(gid);
    uint seed = pcg_hash(gid.x ^ pcg_hash(gid.y ^ pcg_hash(frame_id)));

    // 中心像素数据
    vec3 centerNormal, centerPos;
    float centerDist;
    {
        readGeo0(GEO_N_GEO, gxy, centerPos, centerDist);
        float rough, pathR;
        int it;
        readGeo1(GEO_N_NORMALS, gxy, centerNormal, rough, it, pathR);
    }

    // Phase 2: Poisson 盘空间蓄水池
    Reservoir r = spatialReservoir(gid, centerNormal, centerDist, seed);

    // Phase 3: 降噪先验
    addDenoisedPrior(r, pix, centerNormal, seed);

    // Phase 4+5: 历史重投影 + RIS 合并
    StoredReservoir hist;
    if (sampleHistory(gxy, seed, hist)) {
        float temporalConf = clamp(texelFetch(colortex6, pix, 0).r, 0.0, 1.0);
        combineWithHistory(r, hist, temporalConf, centerNormal, seed);
    }

    r.M = min(r.M, GUIDE_MAX_M);
    float W = reservoirW(r);

    vec2 halfY = packAliceHalf(r.y);
    imageStore(colorimg6, pix, vec4(halfY, W, r.M));
}

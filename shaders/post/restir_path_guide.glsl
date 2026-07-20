#version 430 compatibility

// ===========================================================================
// Pass: ReSTIR 空域+时域加权蓄水池 (composite58)
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/common.glsl"

uniform sampler2D colortex4; // atrous 降噪 ALICE (packAlice 格式)
uniform sampler2D colortex6; // temporal_diffuse validKernelWeight

layout(rgba32f) uniform writeonly image2D colorimg6;

const uint HALO = 2u;
const uint TILE = 16u + 2u * HALO;
const uint TILE_AREA = TILE * TILE;
const float GUIDE_MAX_M = 64.0;

uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float nextFloat(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) * (1.0 / 4294967296.0); // 除以 2^32
}

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
    // 只有当随机命中时才更新样本，避开 0 权重的非法样本
    if (nextFloat(seed) * r.weightSum < w) {
        r.y = candidate;
        r.target = w;
    }
}

float reservoirW(Reservoir r) {
    return r.weightSum / max(r.M * r.target, 1e-20);
}

// ---------------------------------------------------------------------------
// ALICE 编码辅助函数
// ---------------------------------------------------------------------------
float aliceOmega(vec4 y) {
    return y.w;
}
bool isSky(vec4 y) {
    return y.w <= 1e-8;
}

vec2 packAliceHalf(vec4 y) {
    return vec2(uintBitsToFloat(packHalf2x16(y.xy)), uintBitsToFloat(packHalf2x16(y.zw)));
}
vec4 unpackAliceHalf(vec2 p) {
    return vec4(unpackHalf2x16(floatBitsToUint(p.x)), unpackHalf2x16(floatBitsToUint(p.y)));
}

float guideTarget(vec4 y) {
    return aliceOmega(y);
}

// ---------------------------------------------------------------------------
// 共享内存
// ---------------------------------------------------------------------------
shared vec4 sm_aliceY[TILE_AREA];

void loadTile(ivec2 texSize, uint tid) {
    // 优化：256 线程协同加载 400 个数据，逻辑保持不变但边界安全
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(i % TILE, i / TILE);
        AliceEncoding alice;
        float weight;
        readDiffuseSwap(uvec2(clamp(gc, ivec2(0), texSize - 1)), alice, weight);
        sm_aliceY[i] = alice.aliceY;
    }
    barrier();
    memoryBarrierShared();
}

// ---------------------------------------------------------------------------
// Phase 2: 5×5 WRS 空域蓄水池
// ---------------------------------------------------------------------------
Reservoir spatialReservoir(uvec2 lid, inout uint seed) {
    Reservoir r;
    reservoirInit(r);
    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;

    // 展平分支逻辑，让 GPU 跑满算力
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            vec4 y = sm_aliceY[(cy + uint(ky)) * TILE + (cx + uint(kx))];
            float w = guideTarget(y);
            // 只要 target > 0 且不是天空，就参与更新
            if (w > 1e-8) {
                reservoirUpdate(r, y, w, 1.0, seed);
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Phase 4: 2×2 随机重投影历史蓄水池 (Stochastic Bilinear Filter)
// ---------------------------------------------------------------------------
struct StoredReservoir {
    vec4 y;
    float W;
    float M;
};

StoredReservoir loadStored(ivec2 pix) {
    vec4 raw = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, pix)];
    return StoredReservoir(unpackAliceHalf(raw.xy), raw.z, raw.w);
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

    // 神奇的 2 行代码：小数部分 f 越大，越容易命中 +1 偏移
    ivec2 offset = ivec2(nextFloat(seed) < f.x ? 1 : 0,
            nextFloat(seed) < f.y ? 1 : 0);

    result = loadStored(clamp(p0 + offset, ivec2(0), ivec2(resolution_global) - 1));
    return result.W > 0.0 && result.M > 0.0;
}

// ---------------------------------------------------------------------------
// Phase 5: 历史 RIS 合并
// ---------------------------------------------------------------------------
void combineWithHistory(inout Reservoir r, StoredReservoir hist, float temporalConf, inout uint seed) {
    float targetNow = guideTarget(hist.y);
    if (targetNow <= 1e-8) return;

    float reusedM = min(hist.M, GUIDE_MAX_M) * clamp(temporalConf, 0.0, 1.0);
    if (reusedM <= 0.0) return;

    // [优化 3]: 严格无偏权重计算！
    // 既然 M 被截断为 reusedM，权重就必须用 reusedM 来算，否则能量不守恒！
    float candidateWeight = targetNow * hist.W * reusedM;

    reservoirUpdate(r, hist.y, candidateWeight, reusedM, seed);
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    ivec2 texSize = ivec2(resolution_global);

    uint seed = pcg_hash(gid.x ^ pcg_hash(gid.y ^ pcg_hash(frame_id)));

    // Phase 1
    loadTile(texSize, gl_LocalInvocationIndex);

    if (any(greaterThanEqual(gid, uvec2(texSize)))) return;

    // Phase 2
    Reservoir r = spatialReservoir(gl_LocalInvocationID.xy, seed);

    // Phase 3
    vec4 rawDenoised = texelFetch(colortex4, ivec2(gid), 0);
    AliceEncoding a = unpackAlice(rawDenoised.x, rawDenoised.y, rawDenoised.z);
    if (!isSky(a.aliceY)) {
        float wDenoised = guideTarget(a.aliceY);
        if (wDenoised > 1e-8) reservoirUpdate(r, a.aliceY, wDenoised, 1.0, seed);
    }

    // Phase 4
    StoredReservoir hist;
    if (sampleHistory(gid, seed, hist)) {
        float temporalConf = clamp(texelFetch(colortex6, ivec2(gid), 0).r, 0.0, 1.0);
        combineWithHistory(r, hist, temporalConf, seed);
    }

    // 最终规范化与输出
    r.M = min(r.M, GUIDE_MAX_M);

    imageStore(colorimg6, ivec2(gid), vec4(packAliceHalf(r.y), reservoirW(r), r.M));
}

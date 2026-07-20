#version 430 compatibility

// ===========================================================================
// Pass swap3: 漫反射缓冲交换 + 路径引导蓄水池抽样 (Diffuse Buffer Swap — Compute)
// ===========================================================================
// 管线位置: composite59，在空间滤波 (atrous_denoise_diffuse.glsl) 之后。
//
// Phase 1: 加载 N=4(时域累积 swap) 到共享内存 tile
// Phase 2: 5×5 能量加权蓄水池抽样 → 选出最优时域光照样本
// Phase 3: 正常 swap 流程 (读取滤波后 colortex，WriteDiffuse 写入降噪 N=4)
// Phase 4: 蓄水池候选 vs 降噪结果投票 → 胜者写入 N=5(pathGuide)
//
// 16×16 workgroup + halo=2 → 20×20 tile。
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/common.glsl"

uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;

// ---- 共享内存 tile ----
const uint HALO = 2u;
const uint TILE = 16u + 2u * HALO;
const uint TILE_AREA = TILE * TILE;

struct TileSample {
    vec4 aliceY;
};
shared TileSample sm_tile[TILE_AREA];

float aliceEnergy(vec4 y) { return y.w; }

void unpackLightSample(ivec2 coord, out vec3 pos, out vec3 normal, out AliceEncoding encoded, out AliceEncoding blurred_alice) {
    vec4 sample_data0 = texelFetch(colortex3, coord, 0);
    vec4 sample_data1 = texelFetch(colortex4, coord, 0);
    vec4 sample_data2 = texelFetch(colortex5, coord, 0);
    pos = sample_data0.xyz;
    normal = decodeNormal(sample_data0.w);
    encoded = unpackAlice(sample_data1.x, sample_data1.y, sample_data1.z);
    blurred_alice = unpackAlice(sample_data2.x, sample_data2.y, sample_data2.z);
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texSize = ivec2(resolution_global);
    uint tid = gl_LocalInvocationIndex;

    // =========================================================================
    // Phase 1: 加载时域累积 N=4 到共享内存（用于蓄水池抽样）
    // =========================================================================
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        uint tx = i % TILE;
        uint ty = i / TILE;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
        uvec2 xy = uvec2(cc);

        AliceEncoding alice; float weight;
        readDiffuseSwap(xy, alice, weight);
        sm_tile[i].aliceY = alice.aliceY;
    }
    barrier();
    memoryBarrierShared();

    // =========================================================================
    // Phase 2: 5×5 能量加权蓄水池抽样（从共享内存读取）
    // =========================================================================
    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;

    vec4 reservoirY = vec4(0.0);
    float reservoirW = 0.0;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            TileSample s = sm_tile[(cy + uint(ky)) * TILE + (cx + uint(kx))];
            float energy = aliceEnergy(s.aliceY);
            if (energy < 1e-8) continue;

            float w = energy;
            reservoirW += w;

            float r = rand(vec2(float(gid.x) + float(kx) * 0.13, float(gid.y) + float(ky) * 0.37));
            if (r < w / max(reservoirW, 1e-20)) {
                reservoirY = s.aliceY;
            }
        }
    }

    // =========================================================================
    // Phase 3: 正常 swap 流程
    // =========================================================================
    ivec2 pix = ivec2(gid);
    diffuseIlluminationData tmp = fetchDiffuse(pix);

    if (any(isnan(tmp.data_swap.aliceY))) tmp.data_swap.aliceY = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);

    tmp.prev_weight = tmp.weight;
    tmp.data = tmp.data_swap;

    AliceEncoding blurred_alice;
    unpackLightSample(pix, tmp.pos, tmp.normal, tmp.data_swap, blurred_alice);

    tmp.data = mix_alice(tmp.data, blurred_alice, clamp(NRD_BLEND_STRENGTH * exp(-NRD_BLEND_STRENGTH * clamp(tmp.weight, 0.0, 100.0)), 0.0, 1.0));
    WriteDiffuse(tmp, pix);

    // =========================================================================
    // Phase 4: 蓄水池候选 vs 降噪结果投票 → N=5
    // =========================================================================
    // reservoirY = 5×5 时域最优（局部高频感知）
    // tmp.data_swap.aliceY = 降噪结果（atrous 全图空间滤波，长程感知）
    uvec2 gxy = uvec2(pix);
    float denoisedEnergy = aliceEnergy(tmp.data_swap.aliceY);
    float reservoirEnergy = aliceEnergy(reservoirY);

    vec4 finalGuide;
    if (denoisedEnergy > reservoirEnergy * 1.05) {
        finalGuide = tmp.data_swap.aliceY;  // 降噪更强 → 长程可靠
    } else {
        finalGuide = reservoirY;             // 蓄水池更强 → 局部高频
    }
    writePathGuide(gxy, finalGuide);
}

#version 430 compatibility

// ===========================================================================
// Pass swap6: 折射缓冲打包 + 镜面方差预计算 (Compute, 共享内存加速)
// ===========================================================================
// 折射对应 swap4 (反射). 读 SSBO (refractIlluminationBuffer) + image (折射累积颜色)
// + denoiseBuffer, LDS 5×5 方差, H = 实际几何法线 (geometryNormal), 打包 colortex3/4 供 301 降噪.
// TileSample 紧凑打包为 3 个 vec4 (48B).
// ===========================================================================

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
    vec4 pos_oct;
    vec4 color_vproj;
    vec4 H_dist;
};
shared TileSample sm[TILE_AREA];

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float varianceGeometryWeight(vec3 cPos, vec3 cH, vec3 sPos, vec3 sH) {
    float nd = clamp(dot(cH, sH), 0.0, 1.0);
    float wNormal = pow(nd, VAR_FILTER_NORMAL_POWER);
    float distToCam = max(length(cPos), 0.01);
    float pixelFootprint = max(distToCam / max(resolution.y, 1.0), 1e-4);
    float planeDist = abs(dot(sPos - cPos, cH));
    float depthTerm = planeDist / max(VAR_FILTER_POSITION_PARAM * pixelFootprint, 1e-6);
    return wNormal * exp2(-depthTerm * LOG2_E);
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texSize = ivec2(resolution);
    uint tid = gl_LocalInvocationIndex;

    // ---- Phase 1: 协作加载 tile 到共享内存（按 N 层级拆分 barrier）----
    // Phase 1a: Binding 0 N=0 — 几何位置 + 距离
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        uint tx = i % TILE;
        uint ty = i / TILE;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
        uvec2 xy = uvec2(cc);

        vec3 dpos; float dist;
        readGeo0(GEO_N_GEO, xy, dpos, dist);
        sm[i].H_dist.w = dist;
        sm[i].pos_oct.xyz = dpos;
    }
    barrier();
    memoryBarrierShared();

    // Phase 1b: Binding 0 N=1 — 法线 + roughness（仅非天空像素）
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        if (sm[i].H_dist.w > -0.5) {
            uint tx = i % TILE;
            uint ty = i / TILE;
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
            ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
            uvec2 xy = uvec2(cc);

            vec3 H; float rough_unused; int illumType_unused;
            float _pr; readGeo1(GEO_N_NORMALS, xy, H, rough_unused, illumType_unused, _pr);
            sm[i].H_dist.xyz = H;
        }
    }
    barrier();
    memoryBarrierShared();

    // Phase 1c: Binding 4 N=0 — 折射几何（位置 + 方向 T）
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        if (sm[i].H_dist.w > -0.5) {
            uint tx = i % TILE;
            uint ty = i / TILE;
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
            ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
            uvec2 xy = uvec2(cc);

            vec3 epos, T;
            readRefrGeo(xy, epos, T);
            sm[i].pos_oct = vec4(epos, encodeNormal(T));
        }
    }
    barrier();
    memoryBarrierShared();

    // Phase 1d: Binding 4 N=1 — 折射光照（颜色 + 虚拟投射距离）
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        if (sm[i].H_dist.w > -0.5) {
            uint tx = i % TILE;
            uint ty = i / TILE;
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
            ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
            uvec2 xy = uvec2(cc);

            vec3 color; float vprojDist, accumW;
            readRefrLight(xy, color, vprojDist, accumW);
            if (any(isnan(color)) || any(isinf(color))) color = vec3(0.0);
            sm[i].color_vproj = vec4(color, vprojDist);
        }
    }
    barrier();
    memoryBarrierShared();

    if (any(greaterThanEqual(gid, uvec2(resolution)))) return;

    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;
    TileSample c = sm[cy * TILE + cx];

    if (c.H_dist.w < -0.5) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        imageStore(colorimg4, ivec2(gid), uvec4(0u, 0u, packHalf2x16(vec2(-1.0, 0.0)), 0u));
        return;
    }

    vec3 cPos = c.pos_oct.xyz;
    vec3 cR = decodeNormal(c.pos_oct.w);
    vec3 cH = c.H_dist.xyz;
    vec3 cColor = c.color_vproj.xyz;
    float cVproj = c.color_vproj.w;
    uvec2 cxy = uvec2(clamp(ivec2(gid), ivec2(0), texSize - 1));
    vec3 spec_unused, diff_unused; float cRough;
    cRough = readPathRoughness(GEO_N_NORMALS, cxy);

    float sumW = 0.0, sumL = 0.0, sumL2 = 0.0;
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            TileSample s = sm[(cy + uint(ky)) * TILE + (cx + uint(kx))];
            if (s.H_dist.w < -0.5) continue;
            float w = hw[abs(kx)] * hw[abs(ky)]
                    * varianceGeometryWeight(cPos, cH, s.pos_oct.xyz, s.H_dist.xyz);
            float L = luma(s.color_vproj.xyz);
            sumW += w; sumL += w * L; sumL2 += w * L * L;
        }
    }
    float mean = (sumW > 1e-8) ? sumL / sumW : luma(cColor);
    float variance = (sumW > 1e-8) ? max(sumL2 / sumW - mean * mean, 0.0) : 0.0;

    // ---- 2-sigma 压制 (比 3σ 更激进地压制 specular firefly) ----
    float cLuma = luma(cColor);
    float sigma = sqrt(max(variance, 0.0));
    float clampedLuma = clamp(cLuma, mean - 2.0 * sigma, mean + 2.0 * sigma);
    cColor *= clampedLuma / max(cLuma, 1e-8);

    PackedLightSample ps = packSpecularSample(cPos, cR, cColor, cRough, variance, cVproj, cH);
    imageStore(colorimg3, ivec2(gid), ps.data0);
    imageStore(colorimg4, ivec2(gid), ps.data1);
}

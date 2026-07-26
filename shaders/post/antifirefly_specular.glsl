#version 430 compatibility

// ===========================================================================
// Pass 1 CS: 镜面抗萤火虫滤波 (Anti-Firefly, RCRS)
// ===========================================================================
// 基于 NRD RELAX_AntiFirefly: Rank-Conditioned Rank-Selection
// 在时域累积前钳制离群亮点，防止拖影和闪烁
//
// 策略: 3x3 邻域 min/max 钳制
//   - 找出相同材质邻域的最大和最小亮度
//   - 如果中心像素超出 [min, max]，替换为边界值
//   - 保留方差不变（仅钳制颜色）
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

#define REFLECT_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

// 共享内存用于 3x3 邻域访问
const uint HALO = 1u;
const uint TILE = 16u + 2u * HALO;
const uint TILE_AREA = TILE * TILE;

struct TileSample {
    vec3 color;
    float materialID; // 用粗糙度作为简单的材质区分
};

shared TileSample sm[TILE_AREA];

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texSize = ivec2(resolution_global);
    uint tid = gl_LocalInvocationIndex;

    // ---- Phase 1: 加载到共享内存 ----
    // Phase 1a: Binding 0 N=0 — 距离（天空判定）
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        uint tx = i % TILE;
        uint ty = i / TILE;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
        uvec2 xy = uvec2(cc);

        vec3 pos_unused; float dist;
        readGeo0(GEO_N_GEO, xy, pos_unused, dist);
        sm[i].materialID = (dist > -0.5) ? 0.0 : -1.0;
        sm[i].color = vec3(0.0);
    }
    barrier();
    memoryBarrierShared();

    // Phase 1b: Binding 0 N=1 — roughness（材质区分）
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        if (sm[i].materialID > -0.5) {
            uint tx = i % TILE;
            uint ty = i / TILE;
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
            ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
            uvec2 xy = uvec2(cc);

            vec3 geometryNormal_unused; float rough; int illumType_unused;
            float _pr; readGeo1(GEO_N_NORMALS, xy, geometryNormal_unused, rough, illumType_unused, _pr);
            sm[i].materialID = rough;
        }
    }
    barrier();
    memoryBarrierShared();

    // Phase 1c: Binding 3 N=1 — 反射光照颜色
    for (uint i = tid; i < TILE_AREA; i += 256u) {
        if (sm[i].materialID > -0.5) {
            uint tx = i % TILE;
            uint ty = i / TILE;
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
            ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
            uvec2 xy = uvec2(cc);

            vec3 color; float vproj_unused, accumW_unused;
            readReflLight(xy, color, vproj_unused, accumW_unused);
            if (any(isnan(color)) || any(isinf(color))) color = vec3(0.0);
            sm[i].color = color;
        }
    }
    barrier();
    memoryBarrierShared();

    if (gid.x >= uint(texSize.x) || gid.y >= uint(texSize.y)) return;

    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;
    uint center_idx = cy * TILE + cx;

    TileSample center = sm[center_idx];

    // 天空直接退出
    if (center.materialID < 0.0) return;

    float centerLuma = luma(center.color);
    float maxLuma = centerLuma;
    float minLuma = centerLuma;
    vec3 maxColor = center.color;
    vec3 minColor = center.color;

    // ---- Phase 2: 3x3 邻域搜索 min/max ----
    const float materialThreshold = 0.15; // 粗糙度差异阈值

    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            if (xx == 0 && yy == 0) continue;

            int sx = int(cx) + xx;
            int sy = int(cy) + yy;
            if (sx < 0 || sy < 0 || sx >= int(TILE) || sy >= int(TILE)) continue;

            uint sample_idx = uint(sy) * TILE + uint(sx);
            TileSample s = sm[sample_idx];

            // 材质匹配检查
            if (s.materialID < 0.0) continue; // 天空
            if (abs(s.materialID - center.materialID) > materialThreshold) continue;

            float sampleLuma = luma(s.color);

            if (sampleLuma > maxLuma) {
                maxLuma = sampleLuma;
                maxColor = s.color;
            }
            if (sampleLuma < minLuma) {
                minLuma = sampleLuma;
                minColor = s.color;
            }
        }
    }

    // ---- Phase 3: RCRS 钳制 ----
    vec3 outColor = center.color;

    if (centerLuma > maxLuma) {
        outColor = maxColor;
    } else if (centerLuma < minLuma) {
        outColor = minColor;
    }

    // 写回 (仅更新颜色，保留其他字段)
    uvec2 outXY = gid;
    vec3 oldColor; float vproj, accumW;
    readReflLight(outXY, oldColor, vproj, accumW);
    writeReflLight(outXY, outColor, vproj, accumW);
}

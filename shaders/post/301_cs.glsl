#version 430 compatibility
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ===========================================================================
// Pass 301 CS: 屏幕空间模糊滤波器 (计算着色器变体) — 前 3 级 à‑trous (R0=1,2,3)
// ===========================================================================
// 参考: 301.glsl — 镜面反射/折射的屏幕空间各向异性模糊
//
// 设计动机:
//   301.glsl 为 fragment shader, 每线程通过 texelFetch 读取 18 个 vec4。
//   R0 ≤ 3 时邻域高度重叠, 共享内存可显著减少冗余读取。
//   STEP ≤ 3 全部轴对齐 (无旋转抖动), HALO = R0, tile 尺寸最小化。
//
// 共享内存使用量 (16×16 工作组, 轴对齐):
//   R0=1: 18×18 tile →  324 vec4(geom) +  324 vec4(light) = 10.1 KB
//   R0=2: 20×20 tile →  400 vec4(geom) +  400 vec4(light) = 12.5 KB
//   R0=3: 22×22 tile →  484 vec4(geom) +  484 vec4(light) = 15.1 KB
//
// 管线:
//   R0=1 → dispatch CS (R0=1, 轴对齐)
//   R0=2 → dispatch CS (R0=2, 轴对齐)
//   R0=3 → dispatch CS (R0=3, 轴对齐)
//   R0≥4 → 301.glsl (fragment shader, 旋转抖动)
//
// 天空掩码: roughness < 0 (由 swap4/swap6/301 写入)
// ===========================================================================

// ---- 工作组配置 — 16×16 = 256 线程 ------------------------------------------
layout(local_size_x = 16, local_size_y = 16) in;

// ---- 输入纹理 (只读) -------------------------------------------------------
uniform sampler2D colortex3; // 几何: pos.xyz + encoded normal
uniform sampler2D colortex4; // 镜面反射: f16(R,G)|f16(B,roughness)|weight|spare

// ---- 输出图像 --------------------------------------------------------------
layout(rgba32f) uniform image2D colorimg4;

// ---- 共享内存 — 缓存 tile 内的几何 + 光照数据 ------------------------------
// 天空掩码复用 roughness (sm_light[].y 的 second half): < 0 = 天空
// STEP ≤ 3 全部轴对齐 → HALO = R0, 无需旋转扩展
#define HALO R0
#define TILE_SIZE (16 + 2 * HALO)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
shared vec4 sm_light[TILE_AREA];

// ---- 可调参数 (与 301.glsl 一致) --------------------------------------------
const float NORMAL_PARAM = 8.0;
const float POSITION_PARAM = 1.0;

// ===========================================================================
// 辅助函数
// ===========================================================================

float computeAnisotropicAxisScale(vec3 B, vec3 A, vec3 n) {
    float an = dot(A, n);
    float bn = dot(B, n);
    vec3 x = an * B - bn * A;
    return abs(bn) * sqrt(max(1.0 - an * an, 0.0)) / max(0.01, dot(x, x));
}

float GetRoughnessWeight(float roughness0, float roughness) {
    float norm = roughness0 * roughness0 * 0.99 + 0.01;
    float w = abs(roughness0 - roughness) * (1.0 / norm);
    return clamp(1.0 - w, 0.0, 1.0);
}

// 从共享内存解包镜面反射样本 (与 denoise.glsl unpackSpecularSample 语义一致)
void unpackSpecularSampleSM(uint tile_idx, out vec3 pos, out vec3 normal,
    out vec3 radiance, out float weight, out float roughness) {
    vec4 geom = sm_geometry[tile_idx];
    vec4 light = sm_light[tile_idx];
    pos = geom.xyz;
    normal = decodeNormal(geom.w);
    // colortex4 新格式: .x=f16(R,G), .y=f16(B,roughness), .z=weight
    vec2 rg = unpackHalf2x16(floatBitsToUint(light.x));
    vec2 br = unpackHalf2x16(floatBitsToUint(light.y));
    radiance = vec3(rg.x, rg.y, br.x);
    roughness = br.y;
    weight = light.z;
}

// ===========================================================================
// 主函数
// ===========================================================================

void main() {
    // ---- 工作项标识 --------------------------------------------------------
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uvec2 local_id = gl_LocalInvocationID.xy;
    uint local_idx = gl_LocalInvocationIndex;
    uvec2 group_id = gl_WorkGroupID.xy;

    ivec2 texSize = textureSize(colortex3, 0);

    // ---- Tile 原点 (含 HALO 边框, 覆盖 à‑trous 邻域) -------------------------
    ivec2 tile_origin = ivec2(group_id * 16u) - ivec2(HALO);

    // =========================================================================
    // Phase 1: 协作加载 tile 到共享内存
    // =========================================================================
    for (uint i = local_idx; i < uint(TILE_AREA); i += 256u) {
        uint tx = i % uint(TILE_SIZE);
        uint ty = i / uint(TILE_SIZE);
        ivec2 gc = tile_origin + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);

        if (gc == cc) {
            sm_geometry[i] = texelFetch(colortex3, cc, 0);
            sm_light[i] = texelFetch(colortex4, cc, 0);
        } else {
            // 越界像素: 几何清零, roughness 标记为负 → 天空
            sm_geometry[i] = vec4(0.0);
            sm_light[i] = vec4(0.0, uintBitsToFloat(packHalf2x16(vec2(0.0, -1.0))), 0.0, 0.0);
        }
    }

    barrier();
    memoryBarrierShared();

    // =========================================================================
    // Phase 2: 中心像素有效性检查
    // =========================================================================
    uint cx = local_id.x + uint(HALO);
    uint cy = local_id.y + uint(HALO);
    uint center_idx = cy * uint(TILE_SIZE) + cx;

    // 跳过越界像素
    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    // 天空检查 — roughness < 0 复用作天空 mask
    float centerRoughCheck = unpackHalf2x16(floatBitsToUint(sm_light[center_idx].y)).y;
    if (centerRoughCheck < 0.0) return;

    // ---- 中心像素深度 (用于模糊因子计算) -----------------------------------
    uint idx = getIdx(uvec2(clamp(pix, ivec2(0), texSize - 1)));
    bufferData info_ = denoiseBuffer.data[idx];
    // 保险: 双重确认非天空像素
    if (info_.distance < -0.5) return;

    // ---- 解包中心像素 ------------------------------------------------------
    vec3 centerPos, centerNormal, centerRadiance;
    float centerWeight, centerRoughness;
    unpackSpecularSampleSM(center_idx, centerPos, centerNormal, centerRadiance,
        centerWeight, centerRoughness);

    // ---- 几何法线 (用于构建反射平面) ---------------------------------------
    vec3 geoNormal = decodeNormal(diffuseIllumiantionBuffer.data[idx].oct_n2);

    float depth = info_.distance;

    // ---- 各向异性轴计算 (基于反射平面) --------------------------------------
    vec3 planeN = -reflect(centerNormal, geoNormal);
    vec3 viewDir = cross(camX_global, camY_global);
    float axis_A = 0.75 + max(computeAnisotropicAxisScale(viewDir, camX_global, planeN), 0.0);
    float axis_B = 0.75 + max(computeAnisotropicAxisScale(viewDir, camY_global, planeN), 0.0);
    axis_A *= axis_A;
    axis_B *= axis_B;

    // ---- 动态模糊因子 ------------------------------------------------------
    float blur_factor = (1.0 - exp(-0.25 * depth)) / 3.0;
    float normal_factor = (1.0 - exp(-0.1 * depth)) * NORMAL_PARAM;

    // ---- 累积器 (中心像素权重 = 1) ------------------------------------------
    vec3 A = centerRadiance;
    float w = 1.0;

    // =========================================================================
    // Phase 3: 3×3 à‑trous 采样循环 — 全部从共享内存读取
    // =========================================================================
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            // 邻域在 tile 内的坐标 — 轴对齐步进 (小步长无需旋转抖动)
            int sx = int(cx) + i * R0;
            int sy = int(cy) + j * R0;

            // 边界裁剪 (tile 内)
            if (sx < 0 || sy < 0 || sx >= int(TILE_SIZE) || sy >= int(TILE_SIZE)) continue;

            uint sample_idx = uint(sy) * uint(TILE_SIZE) + uint(sx);

            // ---- 解包邻域样本 (共享内存) -----------------------------------
            vec3 samplePosW, sampleNormal, sampleRadiance;
            float sampleWeight, sampleRoughness;
            unpackSpecularSampleSM(sample_idx, samplePosW, sampleNormal,
                sampleRadiance, sampleWeight, sampleRoughness);

            // 天空检查 — roughness < 0
            if (sampleRoughness < 0.0) continue;

            // ---- 粗糙度权重 ------------------------------------------------
            float rW = GetRoughnessWeight(centerRoughness, sampleRoughness);

            // ---- 组合权重: 单次 exp (各向异性+位置+法线), 法线用 exp(-k(1-dot)) ----
            float w0 = rW * exp(-(blur_factor * (axis_A * float(i * i) + axis_B * float(j * j))
                                + POSITION_PARAM * abs(dot(centerPos - samplePosW, centerNormal))
                                + normal_factor * (1.0 - dot(centerNormal, sampleNormal))));

            A += sampleRadiance * w0;
            w += w0;
        }
    }

    // ---- 归一化并输出 (新压缩格式) ------------------------------------------
    if (any(isnan(A))) A = vec3(0.0);
    vec3 filteredRadiance = A / max(w, 0.01);

    imageStore(colorimg4, pix, vec4(
            uintBitsToFloat(packHalf2x16(filteredRadiance.rg)),
            uintBitsToFloat(packHalf2x16(vec2(filteredRadiance.b, centerRoughness))),
            centerWeight,
            0.0
        ));
}

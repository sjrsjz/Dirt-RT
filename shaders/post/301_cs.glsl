#version 430 compatibility
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ===========================================================================
// Pass 301 CS: 镜面 NRD 风格降噪 (计算着色器变体, 前 3 级 à-trous R0=1,2,4)
// ===========================================================================
// 参考 301.glsl. R0≤4 轴对齐, HALO=R0, LDS 缓存 tile 几何+光照, 减少 texelFetch.
// 天空 (variance<0) 早退, 保留 swap4 写入的 mask.
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // f16(R,G)|f16(B,roughness)|f16(variance,vprojdist)|oct(H)

layout(rgba32f) uniform image2D colorimg4;

#define HALO R0
#define TILE_SIZE (16 + 2 * HALO)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
shared vec4 sm_light[TILE_AREA];

const float NORMAL_PARAM = 8.0;
const float POSITION_PARAM = 1.0;

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

float luma3(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// 从共享内存解包镜面样本 (与 denoise.glsl unpackSpecularSample 语义一致)
void unpackSpecularSampleSM(uint tile_idx, out vec3 pos, out vec3 R, out vec3 radiance,
    out float roughness, out float variance, out float vprojdist, out vec3 H) {
    PackedLightSample s;
    s.data0 = sm_geometry[tile_idx];
    s.data1 = sm_light[tile_idx];
    unpackSpecularSample(s, pos, R, radiance, roughness, variance, vprojdist, H);
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uvec2 lid = gl_LocalInvocationID.xy;
    uint local_idx = gl_LocalInvocationIndex;
    uvec2 group_id = gl_WorkGroupID.xy;
    ivec2 texSize = textureSize(colortex3, 0);

    ivec2 tile_origin = ivec2(group_id * 16u) - ivec2(HALO);

    // ---- Phase 1: 协作加载 tile 到共享内存 ----
    for (uint i = local_idx; i < uint(TILE_AREA); i += 256u) {
        uint tx = i % uint(TILE_SIZE);
        uint ty = i / uint(TILE_SIZE);
        ivec2 gc = tile_origin + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
        if (gc == cc) {
            sm_geometry[i] = texelFetch(colortex3, cc, 0);
            sm_light[i] = texelFetch(colortex4, cc, 0);
        } else {
            // 越界: 天空 mask (variance<0)
            sm_geometry[i] = vec4(0.0);
            sm_light[i] = vec4(0.0, 0.0, pack2HalfClamped(-1.0, 0.0), 0.0);
        }
    }

    barrier();
    memoryBarrierShared();

    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    uint cx = lid.x + uint(HALO);
    uint cy = lid.y + uint(HALO);
    uint center_idx = cy * uint(TILE_SIZE) + cx;

    vec3 cPos, cR, cRad, cH;
    float cRough, cVar, cVproj;
    unpackSpecularSampleSM(center_idx, cPos, cR, cRad, cRough, cVar, cVproj, cH);

    // 天空: 早退, 保留 swap4 写入的 mask
    if (cVar < 0.0) return;

    vec3 V = normalize(camPos - cPos);
    vec3 planeN = cross(V, cR);
    vec3 viewDir = cross(camX_global, camY_global);
    float axis_A = 0.75 + max(computeAnisotropicAxisScale(viewDir, camX_global, planeN), 0.0);
    float axis_B = 0.75 + max(computeAnisotropicAxisScale(viewDir, camY_global, planeN), 0.0);
    axis_A *= axis_A;
    axis_B *= axis_B;

    float depth = cVproj;
    float blur_factor   = (1.0 - exp2(-0.36067376 * depth)) / 3.0;
    float normal_factor = (1.0 - exp2(-0.14426950 * depth)) * NORMAL_PARAM;

    float blur_factor2   = blur_factor   * LOG2_E;
    float pos_param2     = POSITION_PARAM * LOG2_E;
    float normal_factor2 = normal_factor * LOG2_E;
    float luma_phi2      = SVGF_PHI_L * LOG2_E * inversesqrt(max(cVar, 1e-8));
    float cLuma = luma3(cRad);

    vec3 A = cRad;          // 中心像素 (权重 = 1)
    float w = 1.0;
    float varEnergy = cVar; // 方差传播

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            int sx = int(cx) + i * R0;
            int sy = int(cy) + j * R0;
            if (sx < 0 || sy < 0 || sx >= int(TILE_SIZE) || sy >= int(TILE_SIZE)) continue;
            uint sample_idx = uint(sy) * uint(TILE_SIZE) + uint(sx);

            vec3 sPos, sR, sRad, sH;
            float sRough, sVar, sVproj;
            unpackSpecularSampleSM(sample_idx, sPos, sR, sRad, sRough, sVar, sVproj, sH);

            if (sVar < 0.0) continue; // 天空

            float rW = GetRoughnessWeight(cRough, sRough);
            float w0 = rW * exp2(-(blur_factor2 * (axis_A * float(i * i) + axis_B * float(j * j))
                                 + pos_param2 * abs(dot(cPos - sPos, cH))
                                 + normal_factor2 * (1.0 - dot(cH, sH))
                                 + luma_phi2 * abs(cLuma - luma3(sRad))));

            A += sRad * w0;
            w += w0;
            varEnergy += w0 * w0 * sVar;
        }
    }

    if (any(isnan(A))) A = vec3(0.0);
    vec3 filtered = A / max(w, 0.01);
    float outVar = varEnergy / max(w * w, 1e-8);

    imageStore(colorimg4, pix, packSpecularSample(cPos, cR, filtered, cRough, outVar, cVproj, cH).data1);
}

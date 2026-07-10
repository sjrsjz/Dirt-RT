#version 430 compatibility
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ===========================================================================
// Pass 301 CS: 镜面 NRD 风格降噪 (计算着色器变体, 前 3 级 à-trous R0=1,2,4)
// ===========================================================================
// 混合权重 (同 atrous_denoise_specular.glsl): NRD 虚拟追踪 + 表面几何边缘停止. R0≤4 轴对齐, HALO=R0.
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // f16(R,G)|f16(B,roughness)|f16(variance,virtualProjDist)|oct(H)

layout(rgba32f) uniform image2D colorimg4;

#define HALO R0
#define TILE_SIZE (16 + 2 * HALO)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
shared vec4 sm_light[TILE_AREA];

float GetRoughnessWeight(float roughness0, float roughness) {
    float norm = roughness0 * roughness0 * SPEC_ROUGH_NORM_A + SPEC_ROUGH_NORM_B;
    float w = abs(roughness0 - roughness) * (1.0 / norm);
    return clamp(1.0 - w, 0.0, 1.0);
}


// 从共享内存解包镜面样本 (与 denoise.glsl unpackSpecularSample 语义一致)
void unpackSpecularSampleSM(uint tile_idx, out vec3 pos, out vec3 R, out vec3 radiance,
    out float roughness, out float variance, out float virtualProjDist, out vec3 H) {
    PackedLightSample s;
    s.data0 = sm_geometry[tile_idx];
    s.data1 = sm_light[tile_idx];
    unpackSpecularSample(s, pos, R, radiance, roughness, variance, virtualProjDist, H);
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

    float alpha = max(cRough * cRough, SPEC_MIN_ALPHA);
    float alpha2 = alpha * alpha;
    float lobe_param2 = SPEC_BLUR_BOOST / (alpha2 * SPEC_LOBE_DIVISOR) * LOG2_E;
    float hit_dist_param2 = SPEC_BLUR_BOOST * SPEC_HIT_DIST_SENS * LOG2_E;
    float surf_pos_param2 = SPEC_BLUR_BOOST * SPEC_SURF_PARAM * LOG2_E;

    float luma_phi2 = SPEC_BLUR_BOOST * SVGF_PHI_L * LOG2_E * inversesqrt(max(cVar, 1e-8)) / (1.0 + cRough * SPEC_LUMA_ROUGH_SOFT);
    float cLuma = luma(cRad);

    // Surface geometry edge-stop (uses actual geometry normal H, not reconstructed V+R)
    float cDistToCam = max(length(cPos), 0.01);
    float cPixelFootprint = max(cDistToCam / float(texSize.y), 1e-4);

    vec3 A = cRad;          // 中心像素 (权重 = 1)
    float w = 1.0;
    float varEnergy = cVar;

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

            float surfDist = abs(dot(cPos - sPos, cH));
            float w_surf = exp2(-surf_pos_param2 * surfDist);

            float R_dot_R = max(dot(cR, sR), 0.0);
            float w_lobe = exp2(-(1.0 - R_dot_R) * lobe_param2);

            // NRD hardening: amplify sensitivity when hitDist → 0
            // Small hitDist (reflection near surface): hardFactor > 1 → selective → sharp
            // Large hitDist (distant reflection): hardFactor → 0 → permissive → blur
            float hitDistDiff = abs(cVproj - sVproj);
            float hitDistSum = cVproj + sVproj + 1e-5;
            float hardFactor = 1.0 + SPEC_HIT_DIST_HARDEN / max(max(cVproj, sVproj), 1e-5);
            float w_hitDist = exp2(-(hitDistDiff / hitDistSum) * hit_dist_param2 * hardFactor);

            float w_luma = exp2(-luma_phi2 * abs(cLuma - luma(sRad)));

            // ---- 表面几何权重 (使用实际几何法线 H, 防止跨几何边缘泄漏) ----
            float nd = clamp(dot(cH, sH), 0.0, 1.0);
            float normalTerm = SPEC_GEOM_NORMAL_POWER * (1.0 - nd);
            float planeDist = abs(dot(sPos - cPos, cH));
            float depthTerm = planeDist / max(SPEC_GEOM_DEPTH_PARAM * cPixelFootprint, 1e-6);
            float w_geom = exp2(-(normalTerm + depthTerm) * LOG2_E);

            float w0 = rW * w_surf * w_lobe * w_hitDist * w_luma * w_geom;

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

#version 430 compatibility

// ===========================================================================
// Pass swap6: 折射缓冲打包 + 镜面方差预计算 (Compute, 共享内存加速)
// ===========================================================================
// 折射对应 swap4 (反射). 读 SSBO (refractIllumiantionBuffer) + image (折射累积颜色)
// + denoiseBuffer, LDS 5×5 方差, H = normalize(V+R), 打包 colortex3/4 供 301 降噪.
// TileSample 紧凑打包为 3 个 vec4 (48B).
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define REFRACT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform vec2 resolution;

layout(rgba32f) uniform writeonly image2D colorimg3;
layout(rgba32f) uniform writeonly image2D colorimg4;

const uint HALO = 2u;
const uint TILE = 16u + 2u * HALO;
const uint TILE_AREA = TILE * TILE;

const float hw[3] = float[](1.0, 0.66667, 0.44444);

#ifndef VAR_FILTER_NORMAL_POWER
#define VAR_FILTER_NORMAL_POWER SVGF_NORMAL_POWER
#endif
#ifndef VAR_FILTER_POSITION_PARAM
#define VAR_FILTER_POSITION_PARAM SVGF_POSITION_PARAM
#endif

struct TileSample {
    vec4 pos_oct;
    vec4 color_vproj;
    vec4 H_dist;
};
shared TileSample sm[TILE_AREA];

float luma3(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

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

    for (uint i = tid; i < TILE_AREA; i += 256u) {
        uint tx = i % TILE;
        uint ty = i / TILE;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
        uint idx = getIdx(uvec2(cc));

        float dist = denoiseBuffer.data[idx].distance;
        TileSample s;
        if (dist > -0.5) {
            SpecularRTElement e = refractIllumiantionBuffer.data[idx];
            vec2 rg = unpackHalf2x16(floatBitsToUint(e.color_rg));
            float b = unpackHalf2x16(floatBitsToUint(e.color_b)).x;
            vec3 color = vec3(rg.x, rg.y, b);
            if (any(isnan(color)) || any(isinf(color))) color = vec3(0.0);
            vec3 R = decodeNormal(e.oct_dir);
            vec3 epos = vec3(e.px, e.py, e.pz);
            vec3 H = normalize(-normalize(epos) + R);
            if (any(isnan(H)) || any(isinf(H))) H = R;
            s.pos_oct = vec4(epos, e.oct_dir);
            s.color_vproj = vec4(color, e.vprojdist);
            s.H_dist = vec4(H, dist);
        } else {
            s.pos_oct = vec4(0.0);
            s.color_vproj = vec4(0.0);
            s.H_dist = vec4(0.0, 0.0, 0.0, dist);
        }
        sm[i] = s;
    }

    barrier();
    memoryBarrierShared();

    if (any(greaterThanEqual(gid, uvec2(resolution)))) return;

    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;
    TileSample c = sm[cy * TILE + cx];

    if (c.H_dist.w < -0.5) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        imageStore(colorimg4, ivec2(gid), vec4(0.0, 0.0, pack2HalfClamped(-1.0, 0.0), 0.0));
        return;
    }

    vec3 cPos = c.pos_oct.xyz;
    vec3 cR = decodeNormal(c.pos_oct.w);
    vec3 cH = c.H_dist.xyz;
    vec3 cColor = c.color_vproj.xyz;
    float cVproj = c.color_vproj.w;
    uint cidx = getIdx(uvec2(clamp(ivec2(gid), ivec2(0), texSize - 1)));
    float cRough = denoiseBuffer.data[cidx].roughness;

    float sumW = 0.0, sumL = 0.0, sumL2 = 0.0;
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            TileSample s = sm[(cy + uint(ky)) * TILE + (cx + uint(kx))];
            if (s.H_dist.w < -0.5) continue;
            float w = hw[abs(kx)] * hw[abs(ky)]
                    * varianceGeometryWeight(cPos, cH, s.pos_oct.xyz, s.H_dist.xyz);
            float L = luma3(s.color_vproj.xyz);
            sumW += w; sumL += w * L; sumL2 += w * L * L;
        }
    }
    float mean = (sumW > 1e-8) ? sumL / sumW : luma3(cColor);
    float variance = (sumW > 1e-8) ? max(sumL2 / sumW - mean * mean, 0.0) : 0.0;

    PackedLightSample ps = packSpecularSample(cPos, cR, cColor, cRough, variance, cVproj, cH);
    imageStore(colorimg3, ivec2(gid), ps.data0);
    imageStore(colorimg4, ivec2(gid), ps.data1);
}

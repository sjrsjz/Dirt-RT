#version 430 compatibility

// ===========================================================================
// Pass 102 CS: 折射时域累积 — P_virtual 虚像重投影 (完全修复版)
// ===========================================================================

#define REFRACT_BUFFER_MIN

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

uniform vec2 resolution;

// 虚像位置边缘停止: 多少米位移导致置信度减半
const float VIRT_POS_PARAM = 8.0;

vec3 cameraDelta;

vec3 reproject(vec3 pos_rel) {
    vec3 prevPlayerPos = pos_rel + cameraDelta;
    vec4 clipPos = rtPrevProjection * rtPrevModelView * vec4(prevPlayerPos, 1.0);
    vec3 ndc = clipPos.xyz / clipPos.w;
    return ndc * 0.5 + 0.5;
}

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}

bool evalVirtualPath(vec2 prevUV, vec3 curVirtual, vec3 curNormal_dir, vec3 curColor, float vproj,
    out vec3 result, out float Wnew) {
    result = curColor;
    Wnew = 1.0;

    vec2 ptc = prevUV * vec2(resolution_global);
    ivec2 pt = ivec2(floor(ptc));

    vec3 accumColor = vec3(0.0);
    float sumW = 0.0, sumPrevW = 0.0;
    float maxConf = 0.0;

    float footprint = 1.0 / max(vproj, 1e-4);

    for (int i = 0; i < 4; i++) {
        ivec2 st = pt + ivec2(i & 1, i >> 1);
        uvec2 siXY = uvec2(clamp(st, ivec2(0), ivec2(resolution_global) - 1));

        vec3 hPos, hT;
        readRefrHistGeo(siXY, hPos, hT);

        vec3 hColor; float hVproj, histWeight;
        readRefrHistLight(siXY, hColor, hVproj, histWeight);

        if (histWeight < 1e-4) continue;

        vec3 hPosCur = hPos - cameraDelta;

        vec3 hV = length(hPosCur) > 0.001 ? normalize(hPosCur) : vec3(0, 0, 1);

        vec3 hVirtual = hPosCur + hV * hVproj;

        float distDiff = length(hVirtual - curVirtual);

        // 软性置信度：随虚像距离差平滑衰减
        float posConf = exp2(-VIRT_POS_PARAM * distDiff * footprint);

        vec3 hNormal_dir = hT;
        float normDot = dot(hNormal_dir, curNormal_dir);
        // 如果法线差异过大(夹角>45度)，抛弃历史防止拖影
        if (normDot < 0.707) continue;

        // 双线性插值权重
        vec2 sc = vec2(st);
        float bw = (1.0 - abs(ptc.x - sc.x)) * (1.0 - abs(ptc.y - sc.y));
        float w = bw * posConf;

        if (any(isnan(hColor)) || any(isinf(hColor))) continue;

        accumColor += hColor * w;
        sumW += w;
        sumPrevW += w * histWeight;
        maxConf = max(maxConf, posConf);
    }

    if (sumW < 1e-8) return false;

    float historyW = sumPrevW / sumW;
    float allowed = ACCUMULATION_LENGTH * maxConf;
    float prevW = min(historyW, allowed) + 1.0;

    result = mix(accumColor / sumW, curColor, 1.0 / max(prevW, 1.0));
    Wnew = prevW;
    return true;
}

void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    vec2 texCoord = (vec2(pix) + 0.5) / vec2(resolution);

    float info_distance;
    { vec3 _pos; readGeo0(GEO_N_GEO, pix, _pos, info_distance); }

    float vproj;
    vec3IlluminationData curr_sample;

    unpackSpecularRT_Refr(pix, curr_sample.pos, curr_sample.normal, curr_sample.data_swap, vproj); // curr_sample.normal stores T (transmission direction), not a surface normal
    curr_sample.data = vec3(0.0);
    curr_sample.weight = 0.0;
    curr_sample.prev_weight = 0.0;

    // 如果几何深度无效，直接跳过时域
    if (info_distance < -0.5) {
        curr_sample.weight = 1.0;
        writeRefract(curr_sample, ivec2(pix));
        return;
    }

    cameraDelta = camPos - prevRaytracingCamPos;

    vec3 pos = curr_sample.pos;
    vec3 curNormal_dir = curr_sample.normal;

    vec3 curColor = curr_sample.data_swap;
    vec3 rd = normalize(curr_sample.pos);

    // 计算当前帧的虚拟世界坐标
    vec3 curVirtual = pos + rd * vproj;

    vec3 result = curColor;
    float Wnew = 1.0;

    // 屏蔽掉指向天空的无效历史
    bool hitSky = (vproj > 0.5 * VPROJDIST_SKY);

    if (!hitSky) {
        // 统一对虚像位置进行重投影
        vec3 vmbUV = reproject(curVirtual);

        if (!any(isnan(vmbUV)) && !any(isinf(vmbUV)) && !notInRange(vmbUV.xy)) {
            evalVirtualPath(vmbUV.xy, curVirtual, curNormal_dir, curColor, vproj, result, Wnew);
        }
    }

    if (any(isnan(result))) result = curColor;

    curr_sample.data_swap = result;
    curr_sample.weight = Wnew;
    writeRefract(curr_sample, ivec2(pix));
}

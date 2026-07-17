#version 430 compatibility

// ===========================================================================
// Pass 102 CS: 折射时域累积 — P_virtual 虚像重投影 (完全修复版)
// ===========================================================================

#define REFRACT_BUFFER_MIN

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

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

bool evalVirtualPath(vec2 prevUV, vec3 curVirtual, vec3 curNormal_dir, vec3 curColor,
    out vec3 result, out float Wnew) {
    result = curColor;
    Wnew = 1.0;

    vec2 ptc = prevUV * vec2(resolution_global);
    ivec2 pt = ivec2(floor(ptc));

    vec3 accumColor = vec3(0.0);
    float sumW = 0.0, sumPrevW = 0.0;
    float maxConf = 0.0;

    for (int i = 0; i < 4; i++) {
        ivec2 st = pt + ivec2(i & 1, i >> 1);
        uint si = getIndex(uvec2(clamp(st, ivec2(0), ivec2(resolution_global) - 1)));
        SpecularRTElement e = refractIlluminationBuffer.data[si];

        if (e.hist_weight < 1e-4) continue;

        vec3 hPos = vec3(e.hist_px, e.hist_py, e.hist_pz);
        vec3 hPosCur = hPos - cameraDelta;

        vec3 hV = length(hPosCur) > 0.001 ? normalize(hPosCur) : vec3(0, 0, 1);
        float hVproj = e.hist_vprojdist;

        vec3 hVirtual = hPosCur + hV * hVproj;

        float distDiff = length(hVirtual - curVirtual);

        // 软性置信度：随虚像距离差平滑衰减
        float posConf = exp2(-VIRT_POS_PARAM * distDiff);

        vec3 hNormal_dir = decodeNormal(e.hist_oct_dir);
        float normDot = dot(hNormal_dir, curNormal_dir);
        // 如果法线差异过大(夹角>45度)，抛弃历史防止拖影
        if (normDot < 0.707) continue;

        // 双线性插值权重
        vec2 sc = vec2(st);
        float bw = (1.0 - abs(ptc.x - sc.x)) * (1.0 - abs(ptc.y - sc.y));
        float w = bw * posConf;

        // 颜色解包
        vec2 rg = unpackHalf2x16(floatBitsToUint(e.hist_color_rg));
        float b = unpackHalf2x16(floatBitsToUint(e.hist_color_b)).x;
        vec3 hColor = vec3(rg.x, rg.y, b);

        if (any(isnan(hColor)) || any(isinf(hColor))) continue;

        accumColor += hColor * w;
        sumW += w;
        sumPrevW += w * e.hist_weight;
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

vec3IlluminationData data3;
uint idx;

void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    vec2 texCoord = (vec2(pix) + 0.5) / vec2(resolution);
    idx = getIndex(pix);
    float info_distance = denoiseBuffer.data[idx].distance;

    unpackSpecularRT(refractIlluminationBuffer.data[idx], data3.pos, data3.normal, data3.data_swap);
    data3.data = vec3(0.0);
    data3.weight = 0.0;
    data3.prev_weight = 0.0;
    data3.mixWeight = 0.0;

    // 如果几何深度无效，直接跳过时域
    if (info_distance < -0.5) {
        data3.weight = 1.0;
        WriteRefract(data3, ivec2(pix));
        return;
    }

    cameraDelta = camPos - prevRaytracingCamPos;

    vec3 pos = data3.pos;
    vec3 normal = data3.normal;
    // 解耦：normal 的长度是虚像距离(vproj)，方向才是纯法线
    float vproj = length(normal);
    vec3 curNormal_dir = vproj > 0.001 ? (normal / vproj) : vec3(0.0, 1.0, 0.0);

    vec3 curColor = data3.data_swap;
    vec3 viewDir = pos / max(length(pos), 0.001);

    // 计算当前帧的虚拟世界坐标
    // 当 vproj 非常小时（例如没有折射的粗糙表面），curVirtual 天然等于 pos。
    // 这意味着 VMB 会自动完美退化为普通的 SMB 表面重投影，无需做 IF/ELSE 分支混合！
    vec3 curVirtual = pos + viewDir * vproj;

    vec3 result = curColor;
    float Wnew = 1.0;

    // 屏蔽掉指向天空的无效历史
    bool hitSky = (vproj > 0.5 * VPROJDIST_SKY);

    if (!hitSky) {
        // 统一对虚像位置进行重投影
        vec3 vmbUV = reproject(curVirtual);

        if (!any(isnan(vmbUV)) && !any(isinf(vmbUV)) && !notInRange(vmbUV.xy)) {
            evalVirtualPath(vmbUV.xy, curVirtual, curNormal_dir, curColor, result, Wnew);
        }
    }

    if (any(isnan(result))) result = curColor;

    data3.data_swap = result;
    data3.weight = Wnew;
    data3.mixWeight = denoiseBuffer.data[idx].refractWeight;
    WriteRefract(data3, ivec2(pix));
}

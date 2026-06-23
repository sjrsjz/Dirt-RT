#version 430 compatibility

// ===========================================================================
// Pass 100: 漫反射时域累积 (Diffuse Temporal Accumulation)
// ===========================================================================
// 管线位置: 在光线追踪生成当前帧 GI 样本后，与上一帧历史混合
//
// 算法:
//   1. 重投影 (Reprojection): 利用 camera 运动矩阵，将当前像素的世界空间
//      位置反投影到上一帧的屏幕坐标
//   2. 可见性验证: 检查重投影坐标是否在屏幕范围内、几何体是否有效
//   3. 历史混合: 使用指数移动平均 (EMA) 将当前 SH 光照与历史混合，
//      混合权重由位置/法线一致性决定
//   4. 方差估计: 使用 Welford 在线算法维护时域方差，供后续 SVGF 使用
//
// 输出:
//   - diffuseIllumiantionData (通过 WriteDiffuse): 更新后的 SH + 方差 + 权重
//   - extInfoBuffer:                  (weight, variance) 供 110.glsl 使用
// ===========================================================================

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

uniform sampler2D colortex0;
uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform vec3 cameraPosition;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;

uniform float near;
uniform float far;
uniform vec2 resolution;
uniform int worldTime;

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

// 法线相似度权重指数 — 值越大，法线差异导致的拒绝越严格
#define NORMAL_PARAM TEMPORAL_NORMAL_PARAM

// 位置/深度差异的敏感度 — 控制对几何不连续性的响应
#define POSITION_PARAM TEMPORAL_POSITION_PARAM

// 亮度相关参数 (本 pass 中未直接使用，保留供后续调参)
const float LUMINANCE_PARAM = 4.0;

// ---------------------------------------------------------------------------
// SVGF 风格的边缘停止权重函数
// ---------------------------------------------------------------------------

// 法线权重: 基于法线夹角余弦，距离越远容忍度越低 (近处严格，远处宽松)
float svgfNormalWeight(vec3 centerNormal, vec3 normal, float distance) {
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM);
}

// 位置权重: 检查采样点偏离中心平面的程度，按距离归一化
float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)));
}

// ---------------------------------------------------------------------------
// 重投影 (Reprojection) 函数
// ---------------------------------------------------------------------------

// 从屏幕空间坐标重投影到上一帧 (使用完整的逆投影 + 模型变换链)
vec3 reproject(vec3 screenPos) {
    vec4 tmp = gbufferProjectionInverse * vec4(screenPos * 2.0 - 1.0, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 playerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = playerPos + cameraPosition;
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}

// 从世界空间位置直接重投影到上一帧屏幕坐标 (简化路径)
vec3 reproject2(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}

// ---------------------------------------------------------------------------
// 全局变量 (用于在不同函数间传递状态)
// ---------------------------------------------------------------------------

vec3 prevScreenPos; // 重投影后的上一帧屏幕坐标
float info_distance; // 当前像素的光线追踪距离
uint idx_l; // 重投影像素的去噪缓冲区索引
vec2 texSize; // 纹理尺寸
uint idx; // 当前像素的去噪缓冲区索引

in vec2 texCoord;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}

diffuseIllumiantionBufferData current_data; // 当前帧数据 (来自光线追踪)
diffuseIllumiantionData out_data; // 输出数据

// // ===========================================================================
// // 无偏加权 Welford 在线方差更新 (West 1979)
// // ===========================================================================
// // 与原始 SVGF 的矩估计不同，这里使用单通道标量 (亮度 Y) 的加权方差
// //
// // 参数:
// //   old_mean    : 历史加权均值 (亮度)
// //   old_var     : 历史加权方差
// //   new_val     : 当前帧亮度值
// //   old_weight  : 历史累积权重
// //   new_weight  : 当前帧权重 (通常 = 1.0)
// //
// // 返回: 更新后的方差
// float updateVariance(float old_mean, float old_var, float new_val,
//                      float old_weight, float new_weight) {
//     float total = old_weight + new_weight;
//     float delta = new_val - old_mean;
//     float new_mean = old_mean + delta * new_weight / total;
//     float new_var = (old_weight * old_var + new_weight * delta * (new_val - new_mean)) / total;
//     return max(new_var, 0.0);
// }

// Welford 在线方差更新 — 单遍扫描计算邻域 SH 的加权方差
// 参数:
//   M_n     : 当前加权和 (avg_SH)
//   D_n     : 当前方差
//   X       : 新样本
//   h_w     : 历史权重总和
//   w       : 新样本的权重
// 返回:      更新后的方差 D_{n+1}
float updateVariance(vec4 M_n, float D_n, vec4 X, float h_w, float w) {
    vec4 diff = X - M_n / h_w; // 新样本与当前均值的差
    float t = 1.0 / (h_w + w);
    return (D_n * h_w + dot(diff, diff) * w * t) * t;
}

float output_weight = 0.0;
float output_variance = 0.0;

// ===========================================================================
// 时域累积核心逻辑
// ===========================================================================
void MixDiffuse() {
    // ---- 情况 1: 重投影失败 (屏幕外 / 新增像素) ---------------------------
    // 直接用当前帧数据初始化，方差设为 1.0 (最不确定状态)
    if (notInRange(prevScreenPos.xy)) {
        output_variance = 1.0;
        output_weight = 1.0;
        out_data.data_swap = current_data.data_swap;
        return;
    }

    // ---- 情况 2: 正常重投影 — 采样历史数据 --------------------------------
    vec2 prev_screen = prevScreenPos.xy * textureSize(colortex0, 0);
    diffuseIllumiantionData data = sampleDiffuse(prev_screen);

    // 计算重投影置信度: 位置一致性 × 法线一致性 × 几何有效性
    float pos_weight = svgfPositionWeight(data.pos, current_data.pos,
            current_data.normal, info_distance);
    float normal_weight = pow(max(dot(data.normal, current_data.normal), 0.0),
            NORMAL_PARAM);

    float s = float(info_distance > -0.5) * pos_weight * normal_weight;

    // 历史权重受重投影置信度调制
    float prevW = data.prev_weight * s;

    // ---- 情况 2a: 历史数据不足 — 直接使用当前帧 ----------------------------
    if (prevW < 1e-2) {
        output_variance = max(current_data.data_swap.shY.w * 0.5, 0.5);
        output_weight = 1.0;
        out_data.data_swap = current_data.data_swap;
    }
    // ---- 情况 2b: 正常混合 — EMA 融合当前与历史 ---------------------------
    else {
        const float max_history = 1000.0;
        float old_total = prevW;
        float new_total = clamp(prevW + 1.0, 1.0, max_history);

        // 光照混合: 新帧权重 = 1/new_total, 历史 = old_total/new_total
        out_data.data_swap = mix_SH(data.data, current_data.data_swap,
                1.0 / new_total);

        // 方差更新: 基于亮度通道的 Welford 递推
        output_variance = updateVariance(
                data.data.shY, // 历史亮度均值
                data.prev_variance, // 历史方差
                current_data.data_swap.shY, // 当前亮度样本
                old_total, // 旧总权重
                1.0 // 当前帧权重 (=1)
            );

        output_weight = new_total;
    }
}

/* RENDERTARGETS: 5 */
layout(location = 0) out vec4 output_data;

layout(rgba32f) uniform image2D extInfoBuffer;

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    uvec2 pix = uvec2(gl_FragCoord.xy);
    idx = getIdx(pix);

    // 将当前帧的 reservoir 数据复制到 prevReservoirs，供下一帧重投影使用
    prevReservoirs.data[idx] = curReservoirs.data[idx];

    // ---- 读取当前像素的几何与光照数据 ------------------------------------
    info_distance = denoiseBuffer.data[idx].distance;
    current_data = diffuseIllumiantionBuffer.data[idx];

    // 初始化输出为当前帧值 (天空 / 无效几何由后续分支处理)
    out_data.data_swap = current_data.data_swap;
    out_data.data = init_SH();
    out_data.normal = current_data.normal;
    out_data.normal2 = current_data.normal2;
    out_data.pos = current_data.pos;
    output_weight = 1.0;
    output_variance = 0.0;

    // ---- 天空 / 无效几何: 直接写出，不做时域累积 -------------------------
    if (info_distance < -0.5) {
        WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));
        imageStore(extInfoBuffer, ivec2(gl_FragCoord.xy),
            vec4(output_weight, output_variance, 0.0, 0.0));
        return;
    }

    // ---- 重投影到上一帧 --------------------------------------------------
    prevScreenPos = reproject2(current_data.pos);
    idx_l = getIdx(uvec2(prevScreenPos.xy * textureSize(colortex0, 0) + 0.5));

    // ---- 执行时域混合 ----------------------------------------------------
    MixDiffuse();

    // 将 (weight, variance) 存入外部缓冲区，供 110.glsl 读取
    imageStore(extInfoBuffer, ivec2(gl_FragCoord.xy),
        vec4(output_weight, output_variance, 0.0, 0.0));

    // 写出更新后的 diffuse 数据
    WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));
}

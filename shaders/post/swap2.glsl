#version 430 compatibility
#define DIFFUSE_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4,5,6 */

layout(location = 0) out vec4 diffuseNormal;
layout(location = 1) out vec4 diffusePos;
layout(location = 2) out vec4 shY;
layout(location = 3) out vec4 CoCg;

uniform vec2 resolution;

// ===========================================================================
// ★ 预平滑滤波半径控制配置
// ===========================================================================
// RADIUS 1 = 3x3  (性能优，9次采样)
// RADIUS 2 = 5x5  (平衡点，25次采样)
// RADIUS 3 = 7x7  (极限稳定，49次采样，专治各种黑斑和时域断层)
#define VAR_FILTER_RADIUS 1 
#define VARIANCE_SCALE 10.0
// ===========================================================================

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint idx = getIdx(uvec2(pix));

    // 获取中心点几何与基础数据
    vec3 centerNormal = diffuseIllumiantionBuffer.data[idx].normal;
    vec3 centerPos    = diffuseIllumiantionBuffer.data[idx].pos;
    diffuseIllumiantionData centerData = fetchDiffuse(pix);

    // 计算用于深度容差的足迹缩放 (防止近处/远处边缘断裂)
    float dist_to_cam = max(length(centerPos - camPos), 0.01); 
    float pixel_footprint = max(dist_to_cam / float(resolution.y), 0.0001);

    // =========================================================================
    // ★ 方差宏观空域预平滑 (Large-scale Spatial Variance Pre-filtering)
    // =========================================================================
    float smoothed_variance = 0.0;
    float sum_weight = 0.0;

    // 当半径大于1时，我们需要一个基础方差用来防止除0死锁
    float r_float = max(float(VAR_FILTER_RADIUS), 1.0);

    for (int i = -VAR_FILTER_RADIUS; i <= VAR_FILTER_RADIUS; i++) {
        for (int j = -VAR_FILTER_RADIUS; j <= VAR_FILTER_RADIUS; j++) {
            
            // 1. 空间高斯权重衰减 
            // 使得 7x7 采样呈中心高边缘低的圆形扩展，而不是硬盒子
            float w_spatial = exp(-2.0 * float(i*i + j*j) / (r_float * r_float));

            ivec2 sample_pix = pix + ivec2(i, j);
            
            // 边界检查
            if (sample_pix.x < 0 || sample_pix.y < 0 || 
                sample_pix.x >= int(resolution.x) || sample_pix.y >= int(resolution.y)) {
                continue;
            }

            uint n_idx = getIdx(uvec2(sample_pix));
            vec3 sampleNormal = diffuseIllumiantionBuffer.data[n_idx].normal;
            vec3 samplePos    = diffuseIllumiantionBuffer.data[n_idx].pos;
            
            // 读取邻居原始方差
            // 只要同平面有噪点，它就会像病毒一样扩散，拔高整体宽容度
            diffuseIllumiantionData n_data = fetchDiffuse(sample_pix);

            // 2. 双边边缘保护
            // 绝对不能让强光面的方差越过墙角糊到阴影里！
            // 距离拉大后，防穿模权重必须十分严格
            float w_n = pow(max(dot(centerNormal, sampleNormal), 0.0), 64.0); // 严格锁死平滑段法线
            float plane_dist = abs(dot(samplePos - centerPos, centerNormal));
            
            // 由于采样范围增大，给一点基础容差 0.01 避免起伏网格被断开
            float w_z = exp(-plane_dist / (pixel_footprint * float(VAR_FILTER_RADIUS) + 0.01));

            // 自身权重为最高，外围综合衰减
            float w = (i == 0 && j == 0) ? 1.0 : (w_spatial * w_n * w_z);

            smoothed_variance += n_data.variance * w;
            sum_weight += w;
        }
    }
    
    // 归一化出纯净、连贯的大范围底层方差
    float final_variance = smoothed_variance / max(sum_weight, 1e-5);
    // =========================================================================

    // 使用原始未过滤的光照数据（不进行萤火虫拦截）
    SH outSH;
    outSH.shY  = centerData.data_swap.shY;
    outSH.CoCg = centerData.data_swap.CoCg;

    // 写入目标 Buffers
    diffuseNormal.xyz = centerNormal;
    if (centerData.weight <= 1.0 + 1e-3) {
        diffusePos = vec4(centerPos, 0);
    } else {
        diffusePos = vec4(centerPos, 0);
    }

    // 写入经过预平滑方差后的输出（无萤火虫过滤）
    shY  = outSH.shY;
    CoCg = vec4(outSH.CoCg, final_variance * VARIANCE_SCALE + 2000.0 * exp(- min(centerData.weight, 10.0)), centerData.weight);

    // NAN 保护
    if (any(isnan(shY)))  shY  = vec4(0);
    if (any(isnan(CoCg))) CoCg = vec4(0);
}
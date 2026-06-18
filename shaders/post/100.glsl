#version 430 compatibility

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

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

const float NORMAL_PARAM = 4.0;
const float POSITION_PARAM = 64.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal, float distance) {
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM * (0.25 + 4 * exp(-0.25 * distance)));
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    return exp(-(pow(POSITION_PARAM * abs(dot(pixelPos - centerPos, normal) / sqrt(distance)), 4)));
}

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

vec3 reproject2(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}

vec3 prevScreenPos;
float info_distance;
uint idx_l;
vec2 texSize;
uint idx;
vec3 curr_rd;

in vec2 texCoord;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}

diffuseIllumiantionBufferData current_data;
diffuseIllumiantionData out_data;

// ============================================================
// ★ 无偏加权 Welford 在线方差更新
// ============================================================
float updateVariance(float old_mean, float old_var, float new_val, float old_weight, float new_weight) {
    float total = old_weight + new_weight;
    float delta = new_val - old_mean;
    float new_mean = old_mean + delta * new_weight / total;
    // 加权方差递推 (West 1979)
    float new_var = (old_weight * old_var + new_weight * delta * (new_val - new_mean)) / total;
    return max(new_var, 0.0);
}

float output_weight = 0;
float output_variance = 0;

float visible_factor(vec3 normal, vec3 local_position, vec3 rd) {
    return 1 / (abs(dot(normal, rd)) + 1e-2) * length(local_position);
}

// ============================================================
// 时域累积逻辑（仅修改方差部分）
// ============================================================
void MixDiffuse() {
    if (notInRange(prevScreenPos.xy)) {
        output_variance = 1.0; 
        output_weight = 1.0;
        out_data.data_swap = current_data.data_swap;
        return;
    }
    vec2 prev_screen = prevScreenPos.xy * textureSize(colortex0, 0);
    diffuseIllumiantionData data = sampleDiffuse(prev_screen);

    float pos_weight = svgfPositionWeight(data.pos, current_data.pos, current_data.normal, info_distance);
    float normal_weight = pow(max(dot(data.normal, current_data.normal), 0.0), NORMAL_PARAM);
  
    float s = float(info_distance > -0.5) * pos_weight * normal_weight;
    float prevW = data.prev_weight * s;

    if (prevW < 1e-2) {
        output_variance = max(current_data.data_swap.shY.w * 0.5, 0.5);
        output_weight = 1.0;
        out_data.data_swap = current_data.data_swap;
    } else {
        float max_history = 1000.0;
        float old_total = prevW;                     // 上一帧累计的总权重（未加当前帧）
        float new_total = clamp(prevW + 1.0, 1.0, max_history);

        // 光照混合（使用新总权重）
        out_data.data_swap = mix_SH(data.data, current_data.data_swap, 1.0 / new_total);

        // 方差更新（无偏 Welford）
        output_variance = updateVariance(
            data.data.shY.w,          // 历史亮度均值
            data.prev_variance,       // 历史方差
            current_data.data_swap.shY.w,    // 当前亮度
            old_total,                // 旧总权重（尚未加 1）
            1.0                       // 当前帧权重
        );
        // output_variance = min(output_variance, 100.0);

        output_weight = new_total;
    }
}

/* RENDERTARGETS: 5 */
layout(location = 0) out vec4 output_data;

layout(rgba32f) uniform image2D extInfoBuffer;

void main() {
    uvec2 pix = uvec2(gl_FragCoord.xy);

    idx = getIdx(pix);

    info_distance = denoiseBuffer.data[idx].distance;
    curr_rd = normalize(denoiseBuffer.data[idx].rd);
    current_data = diffuseIllumiantionBuffer.data[idx];

    out_data.data_swap = current_data.data_swap;
    out_data.data = init_SH();
    out_data.normal = current_data.normal;
    out_data.normal2 = current_data.normal2;
    out_data.pos = current_data.pos;
    output_weight = 1;
    output_variance = 0;
    if (info_distance < -0.5) {
        WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));
        imageStore(extInfoBuffer, ivec2(gl_FragCoord.xy), vec4(output_weight, output_variance, 0, 0));
        return;
    }
    prevScreenPos = reproject2(current_data.pos);

    idx_l = getIdx(uvec2(prevScreenPos.xy * textureSize(colortex0, 0) + 0.5));
    MixDiffuse();

    imageStore(extInfoBuffer, ivec2(gl_FragCoord.xy), vec4(output_weight, output_variance, 0, 0));

    WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));
}
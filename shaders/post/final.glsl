#version 430

// ===========================================================================
// Pass final: 最终合成 — 色调映射 + 实体叠加 (Final Composite)
// ===========================================================================
// 管线末端 pass，负责:
//   1. 应用曝光 / HDR 色调映射
//   2. 叠加实体渲染 (entity 遮罩)
//   3. 输出到主帧缓冲 (colortex0)
//
// 实体叠加:
//   - mask = depth_texture < entity_depth_texture → entity 在前景
//   - 使用 entity.a 作为 alpha 进行 mix
// ===========================================================================

#include "/lib/buffers/frame_data.glsl"
#include "/lib/constants.glsl"
#include "/lib/post_processing/tonemap.glsl"

in vec2 texCoord;

uniform float near;
uniform float far;

uniform sampler2D colortex0;   // 主场景颜色
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex7;   // 实体渲染
uniform sampler2D colortex8;   // 实体深度 (或其他)
uniform sampler2D colortex9;   // 场景深度 (或其他)
uniform sampler2D depthtex0;

/* RENDERTARGETS: 0,8,9 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragData;
layout(location = 2) out vec4 fragData2;

void main() {
    // 实体遮罩: entity 深度 > 场景深度 → entity 在前面
    vec4 entity = texture(colortex7, texCoord);
    bool mask = texture(colortex8, texCoord).w < texture(colortex9, texCoord).w;
    vec4 data = texture(colortex0, texCoord);
    vec4 scene = data * avgExposure;
    // 叠加实体渲染 (使用 entity alpha 进行混合)
    fragColor.xyz = mask ? mix(scene.rgb, entity.rgb, entity.a) : scene.rgb;
}

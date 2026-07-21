#version 430

#include "/lib/buffers/frame_data.glsl"
#include "/lib/constants.glsl"
#include "/lib/post_processing/tonemap.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;   // 主场景颜色
uniform sampler2D colortex7;   // 实体渲染 (包含 Alpha 遮罩)

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    vec4 entity = texture(colortex7, texCoord);
    vec4 scene = texture(colortex0, texCoord);
    
    // 直接利用实体自身的 alpha 遮罩进行平滑混合，不需要任何深度对比
    // 只有在 colortex7 被实体渲染写入的地方 (entity.a > 0.0)，才会融合实体颜色
    fragColor.rgb = mix(scene.rgb, entity.rgb * div_avgExposure, entity.a);
    fragColor.a = 1.0;
}
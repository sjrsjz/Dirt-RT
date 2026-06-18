#version 430

// ===========================================================================
// Pass exposure: 曝光 / 亮度传递 (Exposure Copy)
// ===========================================================================
// 最简单的 pass: 将 colortex0 直接复制到输出。
// 通常用于曝光计算或帧缓冲传递的中间步骤。
// ===========================================================================

#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/constants.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = texture(colortex0, texCoord);
}

#version 430

// ===========================================================================
// Pass init: 初始化 / 帧缓冲复制
// ===========================================================================
// 最简单的 pass: 直接将 colortex0 复制到输出。
// 用于初始化 RENDERTARGET 0 或作为帧缓冲传递的基础步骤。
// ===========================================================================

#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/constants.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = texture(colortex0, texCoord);
}

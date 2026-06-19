#version 430 compatibility

// ===========================================================================
// Pass init_sky: 天空纹理预计算 (Sky Precomputation — Compute Shader)
// ===========================================================================
// 功能: 在 Compute Shader 中预计算天空颜色图集。
//
// 算法:
//   1. setSkyVars() 设置太阳/月亮角度、光照参数等全局变量
//   2. GenSky() 为每个像素生成对应方向的大气散射颜色
//
// 调度: 64×32 工作组 → 对应某种低分辨率天空纹理
// 输出: 直接写入天空颜色缓冲区 (由 GenSky 内部处理)
//
// 注: 这个 pass 预计算天空以便后续快速查找，避免在每个像素中
//      重复完整的大气散射计算。SampleSky() 在 fog.fsh 中使用。
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/sky_color.glsl"

const ivec3 workGroups = ivec3(64, 32, 1);

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    // 设置太阳/月亮全局变量
    setSkyVars();

    // 生成该方向的大气散射颜色
    GenSky(SunLight_global, MoonLight_global, lightDir_global, camPos, p);
}

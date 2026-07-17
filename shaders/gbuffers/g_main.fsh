#version 430
#include "/lib/buffers/frame_data.glsl"

uniform sampler2D gtexture;
in vec2 texCoord;
in vec3 normal;

/* RENDERTARGETS: 7 */ // 仅输出到 colortex7，杜绝 9 号缓冲的读写冲突
layout(location = 0) out vec4 fragColor;

void main() {
    // 1. 提前进行 Discard 判定，防止垃圾数据写入
    vec4 texVal = texture(gtexture, texCoord);
    if (texVal.a < 0.01) {
        discard;
    }

    // 2. 正常计算光照
    float intensity = dot(lightDir_global, normal);
    vec3 sun = vec3(10.0);
    vec3 final = clamp(sun * intensity, 0.2, 1.0) * 2.0;
    
    fragColor = pow(texVal, vec4(2.2, 2.2, 2.2, 1.0)) * vec4(final, 1.0);
    
    #ifdef LIGHT
    fragColor.xyz *= 200.0;
    fragColor.a = 0.5; // 写入半透明度用于后期混合
    #else
    fragColor.a = 1.0; // 实心实体写入 1.0 遮罩
    #endif
}
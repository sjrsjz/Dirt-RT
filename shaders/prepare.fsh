#version 430 core
// 清除 colortex7 的内容，避免残留上一帧的实体渲染数据（Iris 疑似存在调度问题导致清除失败，需要手动清除）

/* RENDERTARGETS: 7 */
layout(location = 0) out vec4 fragColor;

void main() {
    // 在整个帧的最开始，将 colortex7 物理擦除为纯透明黑色
    fragColor = vec4(0.0); 
}

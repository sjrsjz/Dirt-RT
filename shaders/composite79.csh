#version 430 core

// Preserve the pre-bloom synchronization point after the refraction denoiser
// was removed. This pass resets only the SPD tail-election counter.
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

#include "/lib/buffers/frame_data.glsl"

void main() {
    bloomCompletedGroups = 0u;
}

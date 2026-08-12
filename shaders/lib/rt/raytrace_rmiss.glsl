#version 460 core
#extension GL_EXT_ray_tracing : enable

#include "/lib/rt/payload.glsl"
layout(location = 6) rayPayloadInEXT Payload payload;

void main(void) {
    payload_packHitDistance(payload.data, -1.0);
}

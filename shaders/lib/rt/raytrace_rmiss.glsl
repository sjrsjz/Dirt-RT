#version 460
#extension GL_EXT_ray_tracing : enable

#include "/lib/rt/payload.glsl"
layout(location = 6) rayPayloadInEXT Payload payload;

void main(void) {
    payload_packHitPos(payload.data, vec3(0.0), -1.0);
}

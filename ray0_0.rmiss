#version 460
#extension GL_EXT_ray_tracing : enable

#include "/lib/rt/payload.glsl"

layout(location = 6) rayPayloadInEXT Payload payload;

void main(void) {
  payload.hitData = vec4(-1);
}
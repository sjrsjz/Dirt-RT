#ifndef PAYLOAD_GLSL
#define PAYLOAD_GLSL

#include "/lib/pbr/material.glsl"
#include "/lib/rt/payload_pack.glsl"

struct Payload {
    uint data[PAYLOAD_SLOTS];
};

#endif // PAYLOAD_GLSL

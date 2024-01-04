#ifndef PAYLOAD_GLSL
#define PAYLOAD_GLSL

#include "/lib/pbr/material.glsl"
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
struct Payload {
    vec4 hitData;
    vec3 geometryNormal;
    vec3 shadowTransmission;
    i16vec2 ignore_block_id; 
    Material material;  
};

#endif // PAYLOAD_GLSL
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : require

#include "/lib/rt/data.glsl"
#include "/lib/rt/payload.glsl"
#include "/lib/rt/fragment_info.glsl"
#include "/lib/rt/volume_extinction.glsl"
layout(location = 6) rayPayloadInEXT Payload payload;

hitAttributeEXT vec2 baryCoord;

layout(std430, binding = 0) uniform CameraInfo {
    vec3 corners[4];
    mat4 viewInverse;
    vec3 sunAngle;
} cam;

layout(binding = 3) uniform sampler2D blockTex;

layout(set = 1, binding = 0) buffer Quads {
    Quad quads[];
} geometryBuffers[];

Quad getRayQuad() {
    return geometryBuffers[nonuniformEXT(gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT)].quads[gl_PrimitiveID >> 1];
}

void main() {
    Quad quad = getRayQuad();
    vec2 uv = getFragmentUV(quad, baryCoord);
    vec4 texColor = texture(blockTex, uv);

    bool inside, handedness, isNEE;
    float prevDist = payload_unpackFlags(payload.data, inside, handedness, isNEE);
    vec3 shadowTrans = payload_unpackShadow(payload.data);

    int blockID = quad.vertices[0].block_id.x;
    bool isTransmissive = blockID == BLOCK_WATER || blockID == BLOCK_GLASS;

    if (inside) {
        float segDist = clamp(gl_HitTEXT - prevDist, 0.0, 100.0);
        shadowTrans = applyVolumeExtinction(shadowTrans, segDist, texColor, blockID);
    }

    // Pass through: alpha-test transparency, or NEE shadow ray passing through transmissive blocks
    if (texColor.a < 0.1 || (isNEE && isTransmissive)) {
        prevDist = gl_HitTEXT;
        inside = !inside;
        payload_packShadow(payload.data, shadowTrans, blockID);
        payload_packFlags(payload.data, prevDist, inside, handedness, isNEE);
        ignoreIntersectionEXT;
    }

    payload_packShadow(payload.data, shadowTrans, blockID);
    payload_packFlags(payload.data, prevDist, inside, handedness, isNEE);
}

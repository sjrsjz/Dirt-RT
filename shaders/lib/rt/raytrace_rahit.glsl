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

    // Unpack volume state
    bool inside; uint bounce; bool handedness; uint ignoreEnc;
    float prevDist = payload_unpackFlags(payload.data, inside, bounce, handedness, ignoreEnc);
    int blockID;
    vec3 shadowTrans = payload_unpackShadow(payload.data, blockID);

    if (inside) {
        if (quad.vertices[0].block_id.x == 1000) {
            shadowTrans *= exp(-clamp(gl_HitTEXT - prevDist, 0.0, 100.0) * vec3(0.1, 0.03, 0.04));
        } else {
            shadowTrans *= exp(-10.0 * clamp(gl_HitTEXT - prevDist, 0.0, 10.0) * (1.05 - texColor.rgb) * texColor.a);
        }
    }

    int ignoreID = payload_decodeIgnoreID(ignoreEnc);

    if (texColor.a < 0.1 || quad.vertices[0].block_id.x == ignoreID
        || (quad.vertices[0].block_id.x == 1000 && ignoreEnc != 0u)) {
        prevDist = gl_HitTEXT;
        inside = !inside;
        payload_packShadow(payload.data, shadowTrans, quad.vertices[0].block_id.x);
        payload_packFlags(payload.data, prevDist, inside, bounce, handedness, ignoreEnc);
        ignoreIntersectionEXT;
    }

    payload_packShadow(payload.data, shadowTrans, quad.vertices[0].block_id.x);
    payload_packFlags(payload.data, prevDist, inside, bounce, handedness, ignoreEnc);
}

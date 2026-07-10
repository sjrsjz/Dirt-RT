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
    bool inside; bool metal; uint bounce;
    float prevDist = payload_unpackFlags(payload.data, inside, metal, bounce);
    vec3 shadowTrans = payload_unpackShadow(payload.data);

    if (inside) {
        if (quad.vertices[0].block_id.x == 1000) {
            shadowTrans *= exp(-clamp(gl_HitTEXT - prevDist, 0, 100) * vec3(0.1, 0.03, 0.04));
        } else {
            shadowTrans *= exp(-10 * clamp(gl_HitTEXT - prevDist, 0, 10) * (1.05 - texColor.rgb) * texColor.a);
        }
    }

    int ignoreID;
    { int blockID; payload_unpackBlockIDs(payload.data, blockID, ignoreID); }

    if (texColor.a < 0.1 || quad.vertices[0].block_id.x == ignoreID || quad.vertices[0].block_id.x == 1000 && ignoreID != 0) {
        prevDist = gl_HitTEXT;
        inside = !inside;
        // Pack back before ignoring to persist volume state for next intersections
        payload_packShadow(payload.data, shadowTrans);
        payload_packFlags(payload.data, prevDist, inside, metal, bounce);
        ignoreIntersectionEXT;
    }

    payload_packShadow(payload.data, shadowTrans);
    payload_packFlags(payload.data, prevDist, inside, metal, bounce);
}

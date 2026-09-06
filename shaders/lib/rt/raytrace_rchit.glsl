#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_ARB_shader_texture_lod : enable
#extension GL_EXT_scalar_block_layout : require

#include "/lib/rt/data.glsl"
#include "/lib/rt/payload.glsl"
#include "/lib/rt/fragment_info.glsl"
#include "/lib/rt/volume_extinction.glsl"
#include "/lib/rt/mipmap.glsl"
#include "/lib/buffers/frame_data.glsl"

layout(location = 6) rayPayloadInEXT Payload payload;

hitAttributeEXT vec2 baryCoord;

layout(std430, binding = 0) uniform CameraInfo {
    vec3 corners[4];
    mat4 viewInverse;
    vec3 sunAngle;
} cam;

layout(set = 1, binding = 0) buffer Quads {
    Quad quads[];
} geometryBuffers[];

#define ENTITY_INSTANCE_FLAG 0x800000u

Quad getRayQuad() {
    uint geometryIndex = (uint(gl_InstanceCustomIndexEXT) & ~ENTITY_INSTANCE_FLAG)
        + uint(gl_GeometryIndexEXT);
    return geometryBuffers[nonuniformEXT(geometryIndex)].quads[gl_PrimitiveID >> 1];
}

void main() {
    Quad quad = getRayQuad();
    bool sideB = getTriangleSide();
    bool isSideA = !sideB;

    // === Geometric identification ===
    payload_packHitDistance(payload.data, gl_HitTEXT);
    int entityTextureIndex = quad.vertices[0].block_id.x == -2
        ? int(quad.vertices[0].block_id.y) - 1 : -1;
    payload_packQuadIDs(payload.data,
        uint(gl_InstanceCustomIndexEXT) + uint(gl_GeometryIndexEXT),
        uint(max(entityTextureIndex + 1, 0)), uint(gl_PrimitiveID));
    payload_packBarycentrics(payload.data, baryCoord);

    // === Quad-derived data for material evaluation in rgen ===
    float bitangentSign;
    int blockID;
    vec2 uv;
    vec4 atlas;
    vec3 geomN;
    vec3 tangent;
    vec3 gradientU, gradientV;
    {
        vec3 barys = vec3(1.0 - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
        uv = getFragmentUV(quad, barys, isSideA);
        atlas = entityTextureIndex >= 0
            ? getEntityTextureBox(quad)
            : getTextureAtlasBox(quad, isSideA);
        geomN = interpolateVertexNormal(quad, baryCoord, sideB);
        tangent = interpolateVertexTangent(quad, baryCoord, sideB);
        bitangentSign = float(quad.vertices[0].tangent.w) * 0.0078125;
        blockID = quad.vertices[0].block_id.x;
        vec3 tint = interpolateVertexColor(quad, baryCoord, sideB);
        float skylight = interpolateVertexLight(quad, baryCoord, sideB).y;

        payload_packQuadUV(payload.data, uv);
        payload_packAtlasBox(payload.data, atlas);
        payload_packGeomNormal(payload.data, geomN);
        payload_packTangent(payload.data, tangent);
        payload_packQuadExtras(payload.data, tint, skylight);

        bool entityGeometry = entityTextureIndex >= 0;
        uint vertex1 = isSideA ? 1u : 2u;
        uint vertex2 = isSideA ? 2u : 3u;
        vec3 objectPosition0 = decodeVertexObjectPosition(
            quad.vertices[0], entityGeometry);
        vec3 objectPosition1 = decodeVertexObjectPosition(
            quad.vertices[vertex1], entityGeometry);
        vec3 objectPosition2 = decodeVertexObjectPosition(
            quad.vertices[vertex2], entityGeometry);
        vec3 position0 = gl_ObjectToWorldEXT
            * vec4(objectPosition0, 1.0);
        vec3 position1 = gl_ObjectToWorldEXT
            * vec4(objectPosition1, 1.0);
        vec3 position2 = gl_ObjectToWorldEXT
            * vec4(objectPosition2, 1.0);
        vec2 uv0 = vec2(quad.vertices[0].block_texture)
            * 0.0000152587890625;
        vec2 uv1 = vec2(quad.vertices[vertex1].block_texture)
            * 0.0000152587890625;
        vec2 uv2 = vec2(quad.vertices[vertex2].block_texture)
            * 0.0000152587890625;
        computeTriangleTextureGradients(position0, position1, position2,
            uv0, uv1, uv2, gradientU, gradientV);
        payload_packTextureGradients(payload.data, gradientU, gradientV);
    }

    // === Volume absorption (accumulated across intersections) ===
    bool inside, handedness, isNEE, ignoreTransmissive;
    float prevDist = payload_unpackFlags(payload.data, inside, handedness,
        isNEE, ignoreTransmissive);
    // Any-hit has already accumulated the segment's volume extinction before
    // accepting this closest hit. Reapplying it here would square absorption.
    vec3 shadowTrans = payload_unpackShadow(payload.data);

    payload_packShadow(payload.data, shadowTrans, blockID);
    payload_packFlags(payload.data, prevDist, inside, bitangentSign > 0.0,
        isNEE, ignoreTransmissive);
}

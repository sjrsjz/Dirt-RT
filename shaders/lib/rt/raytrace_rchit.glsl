#version 460
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
    vec3 worldPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    Quad quad = getRayQuad();
    bool sideB = getTriangleSide();
    bool isSideA = !sideB;

    // === Geometric identification ===
    payload_packHitPos(payload.data, worldPos, gl_HitTEXT);
    payload_packQuadIDs(payload.data,
        uint(gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT), 0u, uint(gl_PrimitiveID));
    payload_packBarycentrics(payload.data, baryCoord);

    // === Quad-derived data for material evaluation in rgen ===
    float bitangentSign;
    int blockID;
    {
        vec3 barys = vec3(1.0 - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
        vec2 uv = getFragmentUV(quad, barys, isSideA);
        vec4 atlas = getTextureAtlasBox(quad, isSideA);
        vec3 geomN = interpolateVertexNormal(quad, baryCoord, sideB);
        vec3 tangent = interpolateVertexTangent(quad, baryCoord, sideB);
        bitangentSign = float(quad.vertices[0].tangent.w) * 0.0078125;
        blockID = quad.vertices[0].block_id.x;
        vec3 tint = interpolateVertexColor(quad, baryCoord, sideB);
        float skylight = interpolateVertexLight(quad, baryCoord, sideB).y;

        payload_packQuadUV(payload.data, uv);
        payload_packAtlasBox(payload.data, atlas);
        payload_packGeomNormal(payload.data, geomN);
        payload_packTangent(payload.data, tangent);
        payload_packQuadExtras(payload.data, tint, skylight);
    }

    // === Volume absorption (accumulated across intersections) ===
    bool inside, handedness, isNEE;
    float prevDist = payload_unpackFlags(payload.data, inside, handedness, isNEE);
    vec3 shadowTrans = payload_unpackShadow(payload.data);

    if (inside) {
        vec2 uv = getFragmentUV(quad, baryCoord);
        vec4 albedo = texture(blockTex, uv);
        float segDist = clamp(gl_HitTEXT - prevDist, 0.0, 100.0);
        shadowTrans = applyVolumeExtinction(shadowTrans, segDist, albedo, blockID);
    }

    payload_packShadow(payload.data, shadowTrans, blockID);
    payload_packFlags(payload.data, prevDist, inside, bitangentSign > 0.0, isNEE);
}

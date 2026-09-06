#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_shader_texture_lod : enable

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

layout(binding = 3) uniform sampler2D blockTex;
layout(binding = 6) uniform sampler2D entityTextures[256];

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
    int blockID = quad.vertices[0].block_id.x;
    bool inside, handedness, isNEE, ignoreTransmissive;
    float prevDist = payload_unpackFlags(payload.data, inside, handedness,
        isNEE, ignoreTransmissive);
    bool isTransmissive = blockID == BLOCK_WATER || blockID == BLOCK_GLASS
        || blockID == BLOCK_ICE;

    // Background visibility rays intentionally see the first opaque surface.
    // Reject water/glass before UV gradients and anisotropic texture reads.
    if (ignoreTransmissive && isTransmissive) {
        ignoreIntersectionEXT;
        return;
    }

    bool sideB = getTriangleSide();
    vec2 uv = getFragmentUV(quad, baryCoord);
    int entityTextureIndex = quad.vertices[0].block_id.x == -2
        ? int(quad.vertices[0].block_id.y) - 1 : -1;
    vec4 atlas = entityTextureIndex >= 0
        ? getEntityTextureBox(quad)
        : getTextureAtlasBox(quad, !sideB);
    vec3 geomN = interpolateVertexNormal(quad, baryCoord, sideB);
    vec3 tangent = interpolateVertexTangent(quad, baryCoord, sideB);
    float coneWidth, coneSpread;
    payload_unpackRayCone(payload.data, coneWidth, coneSpread);
    coneWidth += gl_HitTEXT * coneSpread;

    bool entityGeometry = entityTextureIndex >= 0;
    uint vertex1 = !sideB ? 1u : 2u;
    uint vertex2 = !sideB ? 2u : 3u;
    vec3 position0 = gl_ObjectToWorldEXT * vec4(
        decodeVertexObjectPosition(quad.vertices[0], entityGeometry), 1.0);
    vec3 position1 = gl_ObjectToWorldEXT * vec4(
        decodeVertexObjectPosition(quad.vertices[vertex1], entityGeometry), 1.0);
    vec3 position2 = gl_ObjectToWorldEXT * vec4(
        decodeVertexObjectPosition(quad.vertices[vertex2], entityGeometry), 1.0);
    vec2 uv0 = vec2(quad.vertices[0].block_texture)
        * 0.0000152587890625;
    vec2 uv1 = vec2(quad.vertices[vertex1].block_texture)
        * 0.0000152587890625;
    vec2 uv2 = vec2(quad.vertices[vertex2].block_texture)
        * 0.0000152587890625;
    vec3 gradientU, gradientV;
    computeTriangleTextureGradients(position0, position1, position2,
        uv0, uv1, uv2, gradientU, gradientV);
    vec3 texturePlaneNormal = cross(gradientU, gradientV);
    texturePlaneNormal = dot(texturePlaneNormal, texturePlaneNormal) > 1e-20
        ? normalize(texturePlaneNormal) : geomN;

    vec4 texColor;
    if (entityTextureIndex >= 0) {
        ivec2 textureResolution = textureSize(
            entityTextures[nonuniformEXT(entityTextureIndex)], 0);
        RtTextureFootprint footprint = rtSecondaryTextureFootprint(
            textureResolution, atlas, coneWidth,
            gl_WorldRayDirectionEXT, texturePlaneNormal, tangent,
            gradientU, gradientV);
        texColor = rtSampleAnisotropic(
            entityTextures[nonuniformEXT(entityTextureIndex)], uv, atlas,
            textureResolution, footprint, false);
    } else {
        ivec2 textureResolution = textureSize(blockTex, 0);
        RtTextureFootprint footprint = rtSecondaryTextureFootprint(
            textureResolution, atlas, coneWidth,
            gl_WorldRayDirectionEXT, texturePlaneNormal, tangent,
            gradientU, gradientV);
        texColor = rtSampleAnisotropic(blockTex, uv, atlas,
            textureResolution, footprint, true);
    }

    int mediumBlockID;
    vec3 shadowTrans = payload_unpackShadow(payload.data, mediumBlockID);

    if (inside) {
        float segDist = clamp(gl_HitTEXT - prevDist, 0.0, 100.0);
        shadowTrans = applyVolumeExtinction(shadowTrans, segDist, texColor,
            mediumBlockID);
    }

    // Alpha-tested coverage is not a volume boundary. In particular, player
    // hat/jacket layers contain transparent black texels over the base skin.
    if (texColor.a < 0.1) {
        ignoreIntersectionEXT;
        return;
    }

    // Transmissive blocks do cross a volume boundary on NEE shadow rays.
    if (isNEE && isTransmissive) {
        prevDist = gl_HitTEXT;
        inside = !inside;
        payload_packShadow(payload.data, shadowTrans, blockID);
        payload_packFlags(payload.data, prevDist, inside, handedness, isNEE,
            ignoreTransmissive);
        ignoreIntersectionEXT;
        return;
    }

    payload_packShadow(payload.data, shadowTrans, blockID);
    payload_packFlags(payload.data, prevDist, inside, handedness, isNEE,
        ignoreTransmissive);
}

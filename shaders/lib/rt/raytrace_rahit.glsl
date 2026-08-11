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
    bool sideB = getTriangleSide();
    vec2 uv = getFragmentUV(quad, baryCoord);
    int entityTextureIndex = quad.vertices[0].block_id.x == -2
        ? int(quad.vertices[0].block_id.y) - 1 : -1;
    vec4 atlas = entityTextureIndex >= 0
        ? vec4(0.0, 0.0, 1.0, 1.0)
        : getTextureAtlasBox(quad, !sideB);
    vec3 geomN = interpolateVertexNormal(quad, baryCoord, sideB);
    vec2 mipResolution = max(vec2(resolution_global),
        vec2(gl_LaunchSizeEXT.xy));
    float pixelConeSpread = rtPixelConeSpread(cam.corners[0],
        cam.corners[1], cam.corners[2], mipResolution);

    vec4 texColor;
    if (entityTextureIndex >= 0) {
        ivec2 textureResolution = textureSize(
            entityTextures[nonuniformEXT(entityTextureIndex)], 0);
        float mipLevel = rtTextureLod(textureResolution, atlas,
            gl_HitTEXT, gl_WorldRayDirectionEXT, geomN, 0u,
            pixelConeSpread);
        texColor = textureLod(
            entityTextures[nonuniformEXT(entityTextureIndex)], uv,
            mipLevel);
    } else {
        ivec2 textureResolution = textureSize(blockTex, 0);
        float mipLevel = rtTextureLod(textureResolution, atlas,
            gl_HitTEXT, gl_WorldRayDirectionEXT, geomN, 0u,
            pixelConeSpread);
        texColor = textureLod(blockTex, uv, mipLevel);
    }

    bool inside, handedness, isNEE;
    float prevDist = payload_unpackFlags(payload.data, inside, handedness, isNEE);
    vec3 shadowTrans = payload_unpackShadow(payload.data);

    int blockID = quad.vertices[0].block_id.x;
    bool isTransmissive = blockID == BLOCK_WATER || blockID == BLOCK_GLASS;

    if (inside) {
        float segDist = clamp(gl_HitTEXT - prevDist, 0.0, 100.0);
        shadowTrans = applyVolumeExtinction(shadowTrans, segDist, texColor, blockID);
    }

    // Alpha-tested coverage is not a volume boundary. In particular, player
    // hat/jacket layers contain transparent black texels over the base skin.
    if (texColor.a < 0.1) {
        ignoreIntersectionEXT;
    }

    // Transmissive blocks do cross a volume boundary on NEE shadow rays.
    if (isNEE && isTransmissive) {
        prevDist = gl_HitTEXT;
        inside = !inside;
        payload_packShadow(payload.data, shadowTrans, blockID);
        payload_packFlags(payload.data, prevDist, inside, handedness, isNEE);
        ignoreIntersectionEXT;
    }

    payload_packShadow(payload.data, shadowTrans, blockID);
    payload_packFlags(payload.data, prevDist, inside, handedness, isNEE);
}

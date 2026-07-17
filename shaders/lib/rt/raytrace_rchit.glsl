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
        vec3 geomN = decodeSNorm8Vec3(quad.vertices[0].normal);
        vec3 tangent = decodeSNorm8Vec3(quad.vertices[0].tangent.xyz);
        bitangentSign = float(quad.vertices[0].tangent.w) * 0.0078125;
        blockID = quad.vertices[0].block_id.x;
        vec3 tint = quad.vertices[0].color.rgb / 255.0;

        // Skylight from light_texture interpolation
        float AB = float(max(quad.vertices[1].position.x, quad.vertices[0].position.x)
                    - min(quad.vertices[1].position.x, quad.vertices[0].position.x));
        vec2 fA = quad.vertices[0].light_texture.xy;
        vec2 fB = quad.vertices[1].light_texture.xy;
        vec2 fC = quad.vertices[2].light_texture.xy;
        vec2 fD = quad.vertices[3].light_texture.xy;
        if (AB > 0.5) {
            fA = quad.vertices[3].light_texture.xy;
            fB = quad.vertices[0].light_texture.xy;
            fC = quad.vertices[1].light_texture.xy;
            fD = quad.vertices[2].light_texture.xy;
        }
        vec2 frag_uv = getRelativeUV(uv, atlas);
        float skylight = mix(mix(fA, fB, frag_uv.y), mix(fD, fC, frag_uv.y), frag_uv.x).y;

        payload_packQuadUV(payload.data, uv);
        payload_packAtlasBox(payload.data, atlas);
        payload_packGeomNormal(payload.data, geomN);
        payload_packTangent(payload.data, tangent);
        payload_packQuadExtras(payload.data, tint, skylight);
    }

    // === Volume absorption (accumulated across intersections) ===
    bool inside;
    uint bounce;
    bool handedness;
    uint ignoreEnc;
    float prevDist = payload_unpackFlags(payload.data, inside, bounce, handedness, ignoreEnc);
    vec3 shadowTrans = payload_unpackShadow(payload.data);

    if (inside) {
        vec2 uv = getFragmentUV(quad, baryCoord);
        vec4 albedo = texture(blockTex, uv);
        float segDist = clamp(gl_HitTEXT - prevDist, 0.0, 100.0);
        if (quad.vertices[0].block_id.x == BLOCK_WATER) {
            // Physically-based water extinction (real absorption coefficients)
            shadowTrans *= exp2(-segDist
                        * vec3(0.14426950, 0.04328085, 0.05770780));
        } else {
            // LabPBR dielectric extinction:
            //   albedo.rgb = base color (surface appearance → transmitted tint)
            //   albedo.a   = translucent  (0=opaque, 1=fully transmitting)
            //   T(d) = mix(0, albedo^d, translucent)
            float translucency = albedo.a;
            vec3 beersLambert = pow(max(albedo.rgb, 0.005), vec3(segDist));
            shadowTrans *= mix(vec3(0.0), beersLambert, translucency);
        }
    }

    payload_packShadow(payload.data, shadowTrans, blockID);
    payload_packFlags(payload.data, prevDist, inside, bounce, bitangentSign > 0.0, ignoreEnc);
}

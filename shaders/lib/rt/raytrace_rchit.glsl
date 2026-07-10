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
#include "/lib/pbr/material.glsl"
layout(location = 6) rayPayloadInEXT Payload payload;

hitAttributeEXT vec2 baryCoord;

layout(std430, binding = 0) uniform CameraInfo {
    vec3 corners[4];
    mat4 viewInverse;
    vec3 sunAngle;
} cam;

layout(binding = 3) uniform sampler2D blockTex;
layout(binding = 4) uniform sampler2D blockTexNormal;
layout(binding = 5) uniform sampler2D blockTexSpecular;

layout(set = 1, binding = 0) buffer Quads {
    Quad quads[];
} geometryBuffers[];

Quad getRayQuad() {
    return geometryBuffers[nonuniformEXT(gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT)].quads[gl_PrimitiveID >> 1];
}

#define POM_STEPS 32
#define POM_DEPTH 0.25
#define BINARY_SEARCH_STEPS 6

#define LINEAR_SAMPLING 1

// 辅助函数：转换纹理坐标
vec2 getTexCoord(vec2 coord, vec4 atlas) {
    return atlas.xy + fract(coord) * atlas.zw;
}

vec2 localUV(vec2 uv, vec4 atlas) {
    return (uv - atlas.xy) / atlas.zw;
}
vec2 globalUV(vec2 uv, vec4 atlas) {
    return uv * atlas.zw + atlas.xy;
}

float sampleHeight(sampler2D tex, vec2 coord, vec4 atlas) {
    #if LINEAR_SAMPLING == 1
    vec2 res = textureSize(tex, 0);
    vec2 pixel = globalUV(coord, atlas) * res;

    vec2 i = floor(pixel);
    vec2 f = (pixel - i);

    vec2 base_uv = localUV(i / res, atlas);
    vec2 dx = vec2(1.0 / (res.x * atlas.z), 0.0);
    vec2 dy = vec2(0.0, 1.0 / (res.y * atlas.w));

    float h00 = texture(tex, getTexCoord(base_uv, atlas)).a;
    float h10 = texture(tex, getTexCoord(base_uv + dx, atlas)).a;
    float h01 = texture(tex, getTexCoord(base_uv + dy, atlas)).a;
    float h11 = texture(tex, getTexCoord(base_uv + dx + dy, atlas)).a;

    return mix(
        mix(h00, h10, f.x),
        mix(h01, h11, f.x),
        f.y
    ) * POM_DEPTH - POM_DEPTH;
    #else
    return texture(tex, getTexCoord(coord, atlas)).a * POM_DEPTH - POM_DEPTH;
    #endif
}

vec2 computeDerivatives(vec2 coord, vec4 atlas) {
    const float offset = 0.00025;
    float x_h_L = sampleHeight(blockTexNormal, coord + vec2(-offset * 2, 0), atlas);
    float x_h_R = sampleHeight(blockTexNormal, coord + vec2(offset * 2, 0), atlas);
    float y_h_L = sampleHeight(blockTexNormal, coord + vec2(0, -offset), atlas);
    float y_h_R = sampleHeight(blockTexNormal, coord + vec2(0, offset), atlas);

    return vec2(
        (x_h_L - x_h_R) / (2 * offset),
        (y_h_L - y_h_R) / (2 * offset)
    );
}

vec4 getParallaxOffset(vec2 texCoord, vec3 viewDir, mat3 tbn, vec4 atlas) {
    vec3 V = normalize(transpose(tbn) * viewDir);

    vec2 currentTexCoord = localUV(texCoord, atlas);

    if (V.z >= 0.0) {
        vec2 realCoord = getTexCoord(currentTexCoord, atlas);
        vec2 derivatives = computeDerivatives(currentTexCoord, atlas);
        return vec4(realCoord, derivatives);
    }

    vec2 dtex = V.xy * POM_DEPTH / (-V.z * POM_STEPS);
    float currentHeight = 0;
    float stepSize = POM_DEPTH / POM_STEPS;

    float heightFromTexture = sampleHeight(blockTexNormal, currentTexCoord, atlas);
    int steps = 0;

    while (currentHeight > heightFromTexture && steps < POM_STEPS) {
        currentTexCoord += dtex;
        heightFromTexture = sampleHeight(blockTexNormal, currentTexCoord, atlas);
        currentHeight -= stepSize;
        steps++;
    }

    vec2 prevTexCoord = currentTexCoord - dtex;
    float prevHeight = currentHeight + stepSize;

    for (int i = 0; i < BINARY_SEARCH_STEPS; i++) {
        dtex *= 0.5;
        stepSize *= 0.5;

        vec2 midTexCoord = prevTexCoord + dtex;
        float midHeight = prevHeight - stepSize;
        float heightFromTexture = sampleHeight(blockTexNormal, currentTexCoord, atlas);

        if (heightFromTexture > midHeight) {
            currentTexCoord = midTexCoord;
            currentHeight = midHeight;
        } else {
            prevTexCoord = midTexCoord;
            prevHeight = midHeight;
        }
    }

    vec2 realCoord = getTexCoord(currentTexCoord, atlas);
    vec2 derivatives = computeDerivatives(currentTexCoord, atlas);
    return vec4(realCoord, derivatives);
}
void main() {
    vec3 worldPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    Quad quad = getRayQuad();

    FragmentInfo fragInfo = getFragmentInfo(quad, baryCoord);
    vec4 shadeColor = quad.vertices[0].color * 0.0039215686274509803921568627451;

    mat3 tbn = mat3(
            fragInfo.tangent,
            fragInfo.bitangent,
            fragInfo.normal
        );

    vec4 atlas = getTextureAtlasBox(quad);

    bool inside; bool metal; uint bounce;
    payload_unpackFlags(payload.data, inside, metal, bounce);

    vec2 parallaxTexCoord;
    vec2 derivatives;
    if (bounce == 0u) {
        vec4 pom = getParallaxOffset(fragInfo.uv, gl_WorldRayDirectionEXT, tbn, atlas);
        parallaxTexCoord = pom.xy;
        derivatives = pom.zw;
    } else {
        vec2 localTC = localUV(fragInfo.uv, atlas);
        parallaxTexCoord = getTexCoord(localTC, atlas);
        derivatives = computeDerivatives(localTC, atlas);
    }

    vec4 specular = texture(blockTexSpecular, parallaxTexCoord);

    vec3 normal3 = normalize(vec3(derivatives, 1));
    vec4 normal = vec4(normal3 * 0.5 + 0.5, texture(blockTexNormal, parallaxTexCoord).a);

    vec2 frag_uv = getRelativeUV(fragInfo.uv, atlas);

    vec4 albedo = texture(blockTex, parallaxTexCoord);

    albedo.rgb = pow(albedo.rgb * shadeColor.rgb, vec3(2.2));

    fragInfo.uv = fract(fragInfo.uv * vec2(64, 32));

    float AB = float(max(quad.vertices[1].position.x, quad.vertices[0].position.x) - min(quad.vertices[1].position.x, quad.vertices[0].position.x));

    vec2 A = quad.vertices[0].light_texture.xy;
    vec2 B = quad.vertices[1].light_texture.xy;
    vec2 C = quad.vertices[2].light_texture.xy;
    vec2 D = quad.vertices[3].light_texture.xy;
    if (AB > 0.5) {
        A = quad.vertices[3].light_texture.xy;
        B = quad.vertices[0].light_texture.xy;
        C = quad.vertices[1].light_texture.xy;
        D = quad.vertices[2].light_texture.xy;
    }

    // Compute skylight locally (was stored in material.light_texture.y)
    float skylight = mix(mix(A, B, fragInfo.uv.y), mix(D, C, fragInfo.uv.y), fragInfo.uv.x).y;

    // Pack hit position
    payload_packHitPos(payload.data, worldPos, gl_HitTEXT);

    // Unpack wetness params for getMaterial
    vec2 wet = payload_unpackWetness(payload.data);

    // Compute f0Channel (same as getMaterial internally) and material
    int f0Channel = int(specular.g * 255.0 + 0.5);
    Material mat = getMaterial(albedo, normal, specular, tbn, wet.x, wet.y, skylight, fragInfo.normal);

    // Pack material fields
    // Store ORIGINAL albedo.rgb (getMaterial zeros it for metals; rgen needs
    // it for custom-metal F0 reconstruction via f0Channel≥238)
    payload_packAlbedo(payload.data, albedo.rgb, mat.translucent);
    payload_setF0(payload.data, f0Channel);
    payload_packBSDF(payload.data, mat.roughness, mat.subsurface_scattering);
    payload_packMatNormal(payload.data, mat.normal);
    payload_packEmissionAO(payload.data, mat.emission, mat.ambientOcclusion);
    payload_packGeomNormal(payload.data, fragInfo.normal);
    // Preserve ignore_block_id set by rgen (update only block type)
    int curBlock, curIgnore;
    payload_unpackBlockIDs(payload.data, curBlock, curIgnore);
    payload_packBlockIDs(payload.data, quad.vertices[0].block_id.x, curIgnore);

    // Volume handling
    vec3 shadowTrans = payload_unpackShadow(payload.data);
    float prevDist = payload_unpackFlags(payload.data, inside, metal, bounce);

    if (inside) {
        if (quad.vertices[0].block_id.x == 1000) {
            shadowTrans *= exp2(-clamp(gl_HitTEXT - prevDist, 0, 100) * vec3(0.14426950, 0.04328085, 0.05770780));
        } else {
            shadowTrans *= exp2(-14.42695 * clamp(gl_HitTEXT - prevDist, 0, 10) * (1.05 - albedo.rgb) * albedo.a);
        }
    }

    // metallic flag for BSDF
    metal = mat.metallic > 0.5;

    payload_packShadow(payload.data, shadowTrans);
    payload_packFlags(payload.data, prevDist, inside, metal, bounce);
}

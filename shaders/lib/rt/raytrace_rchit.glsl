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

    // 计算实际采样坐标
    vec2 base_uv = localUV(i / res, atlas);
    vec2 dx = vec2(1.0 / (res.x * atlas.z), 0.0);
    vec2 dy = vec2(0.0, 1.0 / (res.y * atlas.w));

    // 采样四个角
    float h00 = texture(tex, getTexCoord(base_uv, atlas)).a;
    float h10 = texture(tex, getTexCoord(base_uv + dx, atlas)).a;
    float h01 = texture(tex, getTexCoord(base_uv + dy, atlas)).a;
    float h11 = texture(tex, getTexCoord(base_uv + dx + dy, atlas)).a;

    // 双线性插值
    return mix(
        mix(h00, h10, f.x),
        mix(h01, h11, f.x),
        f.y
    ) * POM_DEPTH - POM_DEPTH;
    #else
    return texture(tex, getTexCoord(coord, atlas)).a * POM_DEPTH - POM_DEPTH;
    #endif
}

// 计算高度图偏导数
vec2 computeDerivatives(vec2 coord, vec4 atlas) {
    const float offset = 0.0005;
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
    //return vec4(texCoord, 0, 0);
    vec3 V = normalize(transpose(tbn) * viewDir);

    vec2 currentTexCoord = localUV(texCoord, atlas);

    // 视线方向朝下则提前退出
    if (V.z >= 0.0) {
        vec2 realCoord = getTexCoord(currentTexCoord, atlas);
        vec2 derivatives = computeDerivatives(currentTexCoord, atlas);
        return vec4(realCoord, derivatives);
    }

    vec2 dtex = V.xy * POM_DEPTH / (-V.z * POM_STEPS);
    float currentHeight = 0;
    float stepSize = POM_DEPTH / POM_STEPS;

    // Ray marching
    float heightFromTexture = sampleHeight(blockTexNormal, currentTexCoord, atlas);
    int steps = 0;

    while (currentHeight > heightFromTexture && steps < POM_STEPS) {
        currentTexCoord += dtex;
        heightFromTexture = sampleHeight(blockTexNormal, currentTexCoord, atlas);
        currentHeight -= stepSize;
        steps++;
    }

    // 二分查找细化
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

    vec2 parallaxTexCoord;
    vec2 derivatives;
    if (payload.bounce_depth == 0u) {
        // 仅主光线做完整 POM (32步线性 + 6步二分); 次级弹射跳过
        vec4 pom = getParallaxOffset(fragInfo.uv, gl_WorldRayDirectionEXT, tbn, atlas);
        parallaxTexCoord = pom.xy;
        derivatives = pom.zw;
    } else {
        // 次级弹射: 无 POM 位移, 仍计算高度导数用于法线扰动
        vec2 localTC = localUV(fragInfo.uv, atlas);
        parallaxTexCoord = getTexCoord(localTC, atlas);
        derivatives = computeDerivatives(localTC, atlas);
    }

    vec4 specular = texture(blockTexSpecular, parallaxTexCoord);
    //vec4 normal = texture(blockTexNormal, parallaxTexCoord);

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

    payload.material.light_texture = vec3(mix(mix(A, B, fragInfo.uv.y), mix(D, C, fragInfo.uv.y), fragInfo.uv.x), 0);

    payload.hitData = vec4(worldPos, gl_HitTEXT);
    payload.geometryNormal = fragInfo.normal;
    payload.material = getMaterial(albedo, normal, specular, tbn, payload.wetStrength_global, payload.wetness_global, payload.material.light_texture.y, fragInfo.normal);
    if (payload.inside_block) {
        if (quad.vertices[0].block_id.x == 1000) {
            payload.shadowTransmission *= exp2(-clamp(gl_HitTEXT - payload.prev_distance, 0, 100) * vec3(0.14426950, 0.04328085, 0.05770780));
        } else {
            payload.shadowTransmission *= exp2(-14.42695 * clamp(gl_HitTEXT - payload.prev_distance, 0, 10) * (1.05 - albedo.rgb) * albedo.a);
        }
    }
    payload.material.block_id = quad.vertices[0].block_id;
}

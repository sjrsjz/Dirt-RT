#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "/lib/rt/data.glsl"
#include "/lib/rt/payload.glsl"
#include "/lib/rt/fragment_info.glsl"
#include "/lib/pbr/material.glsl"
layout(location = 6) rayPayloadInEXT Payload payload;

hitAttributeEXT vec2 baryCoord;

layout(std140, binding = 0) uniform CameraInfo {
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

void main() {
    vec3 worldPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    Quad quad = getRayQuad();

    FragmentInfo fragInfo = getFragmentInfo(quad, baryCoord);
    vec4 shadeColor = quad.vertices[0].color * 0.0039215686274509803921568627451;

    vec4 specular = texture(blockTexSpecular, fragInfo.uv);
    vec4 normal = texture(blockTexNormal, fragInfo.uv);
    vec4 albedo = texture(blockTex, fragInfo.uv);
    albedo.rgb = pow(albedo.rgb * shadeColor.rgb, vec3(2.2));

    fragInfo.uv=fract(fragInfo.uv*vec2(64,32));

    float AB=float(max(quad.vertices[1].position.x,quad.vertices[0].position.x)-min(quad.vertices[1].position.x,quad.vertices[0].position.x));

    vec2 A = quad.vertices[0].light_texture.xy;
    vec2 B = quad.vertices[1].light_texture.xy;
    vec2 C = quad.vertices[2].light_texture.xy;
    vec2 D = quad.vertices[3].light_texture.xy;
    if(AB>0.5){
        A = quad.vertices[3].light_texture.xy;
        B = quad.vertices[0].light_texture.xy;
        C = quad.vertices[1].light_texture.xy;
        D = quad.vertices[2].light_texture.xy;
    }


    payload.material.light_texture = vec3(mix(mix(A,B,fragInfo.uv.y),mix(D,C,fragInfo.uv.y),fragInfo.uv.x),0);

    mat3 tbn = mat3(
            fragInfo.tangent,
            fragInfo.bitangent,
            fragInfo.normal
        );

    payload.hitData = vec4(worldPos, gl_HitTEXT);
    payload.geometryNormal = fragInfo.normal;
    payload.material = getMaterial(albedo, normal, specular, tbn, payload.wetStrength_global, payload.wetness_global, payload.material.light_texture.y, fragInfo.normal);
    payload.shadowTransmission *= exp(-10*(1-albedo.rgb)* albedo.a);
    payload.material.block_id = quad.vertices[0].block_id;
    

}

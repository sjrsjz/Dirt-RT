#ifndef FRAGMENT_INFO_GLSL
#define FRAGMENT_INFO_GLSL

#include "/lib/rt/data.glsl"

struct FragmentInfo {
    vec2 uv;
    vec3 tangent;
    vec3 bitangent;
    vec3 normal;
};

vec2 getFragmentUV(Quad quad, vec3 baryCoords, bool isSideA) {
    vec2 t0 = (quad.vertices[0].block_texture) * 0.0000152587890625;
    vec2 t1 = (isSideA ? quad.vertices[1].block_texture : quad.vertices[2].block_texture) * 0.0000152587890625;
    vec2 t2 = (isSideA ? quad.vertices[2].block_texture : quad.vertices[3].block_texture) * 0.0000152587890625;
    return t0 * baryCoords.x + t1 * baryCoords.y + t2 * baryCoords.z;
}

vec2 getFragmentUV(Quad quad, vec2 baryCoords) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;
    vec3 barys = vec3(1.0 - baryCoords.x - baryCoords.y, baryCoords.x, baryCoords.y);
    return getFragmentUV(quad, barys, isSideA);
}

FragmentInfo getFragmentInfo(Quad quad, vec2 baryCoords) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;

    vec3 barys = vec3(1.0 - baryCoords.x - baryCoords.y, baryCoords.x, baryCoords.y);
    vec2 uv = getFragmentUV(quad, barys, isSideA);
    vec3 normal = quad.vertices[0].normal * 0.0078125;
    vec3 tangent = quad.vertices[0].tangent.xyz * 0.0078125;
    vec3 bitangent = cross(tangent, normal) * (quad.vertices[0].tangent.w * 0.0078125);
    return FragmentInfo(uv, tangent, bitangent, normal);
}

vec4 getTextureAtlasBox(Quad quad, bool isSideA) {
    // 获取三个顶点的UV
    vec2 t0 = quad.vertices[0].block_texture * 0.0000152587890625;
    vec2 t1 = (isSideA ? quad.vertices[1].block_texture : quad.vertices[2].block_texture) * 0.0000152587890625;
    vec2 t2 = (isSideA ? quad.vertices[2].block_texture : quad.vertices[3].block_texture) * 0.0000152587890625;
    
    // 计算UV边界
    vec2 minUV = min(min(t0, t1), t2);
    vec2 maxUV = max(max(t0, t1), t2);
    
    // 返回 (x_offset, y_offset, width, height)
    return vec4(minUV, maxUV - minUV);
}

vec4 getTextureAtlasBox(Quad quad) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;
    return getTextureAtlasBox(quad, isSideA);
}

vec2 getRelativeUV(vec2 uv, Quad quad) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;
    vec4 atlas = getTextureAtlasBox(quad, isSideA);
    
    // 转换到相对坐标 (0-1范围)
    return (uv - atlas.xy) / atlas.zw;
}

vec2 getRelativeUV(vec2 uv, vec4 atlas) {
    return (uv - atlas.xy) / atlas.zw;
}

#endif // FRAGMENT_INFO_GLSL
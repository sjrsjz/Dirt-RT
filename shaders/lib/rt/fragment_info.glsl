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

#ifndef FRAGMENT_INFO_NO_PRIMITIVE
vec2 getFragmentUV(Quad quad, vec2 baryCoords) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;
    vec3 barys = vec3(1.0 - baryCoords.x - baryCoords.y, baryCoords.x, baryCoords.y);
    return getFragmentUV(quad, barys, isSideA);
}
#endif

#ifndef FRAGMENT_INFO_NO_PRIMITIVE
FragmentInfo getFragmentInfo(Quad quad, vec2 baryCoords) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;

    vec3 barys = vec3(1.0 - baryCoords.x - baryCoords.y, baryCoords.x, baryCoords.y);
    vec2 uv = getFragmentUV(quad, barys, isSideA);
    vec3 normal = quad.vertices[0].normal * 0.0078125;
    vec3 tangent = quad.vertices[0].tangent.xyz * 0.0078125;
    vec3 bitangent = cross(tangent, normal) * (quad.vertices[0].tangent.w * 0.0078125);
    return FragmentInfo(uv, tangent, bitangent, normal);
}
#endif

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

#ifndef FRAGMENT_INFO_NO_PRIMITIVE
vec4 getTextureAtlasBox(Quad quad) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;
    return getTextureAtlasBox(quad, isSideA);
}
#endif

#ifndef FRAGMENT_INFO_NO_PRIMITIVE
vec2 getRelativeUV(vec2 uv, Quad quad) {
    bool isSideA = (gl_PrimitiveID & 1) == 0;
    vec4 atlas = getTextureAtlasBox(quad, isSideA);
    return (uv - atlas.xy) / atlas.zw;
}
#endif

vec2 getRelativeUV(vec2 uv, vec4 atlas) {
    return (uv - atlas.xy) / atlas.zw;
}

// ===========================================================================
// Barycentric vertex interpolation — replaces vertex[0]-only normal/tangent.
// Quad triangulation: each quad = 2 triangles
//   Triangle A: vertices [0, 1, 2]  (gl_PrimitiveID even)
//   Triangle B: vertices [0, 2, 3]  (gl_PrimitiveID odd)
// ===========================================================================

#ifndef FRAGMENT_INFO_NO_PRIMITIVE
bool getTriangleSide() {
    return (gl_PrimitiveID & 1) != 0;
}
#endif

// Decode snorm8 (i8vec3) vertex attribute to float vec3
vec3 decodeSNorm8Vec3(i8vec3 v) {
    return vec3(v) * 0.0078125; // 1/128
}

// Barycentric interpolation of vertex normals across the correct triangle
vec3 interpolateVertexNormal(Quad quad, vec2 bary, bool sideB) {
    vec3 n0 = decodeSNorm8Vec3(quad.vertices[0].normal);
    float w0 = 1.0 - bary.x - bary.y;
    if (!sideB) {
        // Triangle A: vertices 0, 1, 2
        vec3 n1 = decodeSNorm8Vec3(quad.vertices[1].normal);
        vec3 n2 = decodeSNorm8Vec3(quad.vertices[2].normal);
        return normalize(n0 * w0 + n1 * bary.x + n2 * bary.y);
    } else {
        // Triangle B: vertices 0, 2, 3
        vec3 n2 = decodeSNorm8Vec3(quad.vertices[2].normal);
        vec3 n3 = decodeSNorm8Vec3(quad.vertices[3].normal);
        return normalize(n0 * w0 + n2 * bary.x + n3 * bary.y);
    }
}

// Barycentric interpolation of vertex tangents across the correct triangle
vec3 interpolateVertexTangent(Quad quad, vec2 bary, bool sideB) {
    vec3 t0 = decodeSNorm8Vec3(quad.vertices[0].tangent.xyz);
    float w0 = 1.0 - bary.x - bary.y;
    if (!sideB) {
        vec3 t1 = decodeSNorm8Vec3(quad.vertices[1].tangent.xyz);
        vec3 t2 = decodeSNorm8Vec3(quad.vertices[2].tangent.xyz);
        return normalize(t0 * w0 + t1 * bary.x + t2 * bary.y);
    } else {
        vec3 t2 = decodeSNorm8Vec3(quad.vertices[2].tangent.xyz);
        vec3 t3 = decodeSNorm8Vec3(quad.vertices[3].tangent.xyz);
        return normalize(t0 * w0 + t2 * bary.x + t3 * bary.y);
    }
}

#endif // FRAGMENT_INFO_GLSL

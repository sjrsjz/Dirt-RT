#ifndef DIRT_RT_LIB_MATH_SAMPLING_GLSL
#define DIRT_RT_LIB_MATH_SAMPLING_GLSL

// Orthonormal frames and hemisphere direction sampling.
#include "/lib/constants.glsl"
#include "/lib/math/random.glsl"
// Right-handed frame for a unit normal. The signed denominator stays away
// from zero at both poles; no cross-product normalization is needed.
void orthonormalBasis(vec3 n, out vec3 tangent, out vec3 bitangent) {
    float signZ = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (signZ + n.z);
    float b = n.x * n.y * a;
    tangent = vec3(1.0 + signZ * n.x * n.x * a, signZ * b, -signZ * n.x);
    bitangent = vec3(b, signZ + n.y * n.y * a, -n.y);
}

// Historical Y-up interface used by the uniform solar-disc sampler.
void XYZ(vec3 n, out vec3 X, out vec3 Y, out vec3 Z) {
    Y = n;
    orthonormalBasis(n, X, Z);
}

vec3 DiffuseNormal(vec3 macroNormal, vec3 pos) {
    vec3 randN0, randN1;
    orthonormalBasis(macroNormal, randN0, randN1);
    vec2 xi = rand2(pos);
    float alpha = xi.x * 2 * PI;
    return sqrt(1 - xi.y) * macroNormal + sqrt(xi.y) * (cos(alpha) * randN0 + sin(alpha) * randN1);
}

vec3 SampleUniformHemisphere(vec3 geometryNormal, vec2 xi) {
    float phi = xi.x * 2.0 * PI;
    float cosTheta = xi.y; // cosTheta 在 [0, 1] 均匀分布
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 localDir = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 tangent, bitangent;
    orthonormalBasis(geometryNormal, tangent, bitangent);

    return tangent * localDir.x + bitangent * localDir.y + geometryNormal * localDir.z;
}

vec3 SampleUniformHemisphere(vec3 geometryNormal, vec3 pos) {
    return SampleUniformHemisphere(geometryNormal, rand2(pos));
}

#endif // DIRT_RT_LIB_MATH_SAMPLING_GLSL

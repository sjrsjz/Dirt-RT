#ifndef ALICE_ENCODE_GLSL
#define ALICE_ENCODE_GLSL

#include "/lib/lighting/alice.glsl"

// ===========================================================================
// ALICE light encoding — application-layer encode/decode/project/compose
//
// These are higher-level utilities built on top of the core ALICE math library
// (alice.glsl). They handle:
//   - RGB-to-ALICE encoding (radiance_to_alice)
//   - Irradiance reconstruction (project_alice_irradiance)
//   - Dual-vector packing for temporal storage
//   - AliceEncoding composition (mix, init, scale, accumulate)
//   - AliceEncoding ⇄ packed vec3 (for SSBO/texture I/O)
// ===========================================================================

struct AliceEncoding {
    vec4 aliceY;
    vec2 CoCg;
};

AliceEncoding radiance_to_alice(vec3 color, vec3 dir)
{
    AliceEncoding result;
    float Y = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float Co = 0.5 * color.r - 0.5 * color.b;
    float Cg = -0.25 * color.r + 0.5 * color.g - 0.25 * color.b;
    result.CoCg = vec2(Co, Cg);
    result.aliceY = vec4(dir * Y, Y);
    return result;
}

vec3 project_alice_irradiance(AliceEncoding encoded, vec3 N)
{
    float total_omega = encoded.aliceY.w;
    if (total_omega <= 1e-10) return vec3(0.0);

    float irradiance = alice_irradiance(encoded.aliceY, N);
    float attenuation = irradiance / total_omega;

    float Y  = total_omega;
    float Co = encoded.CoCg.x;
    float Cg = encoded.CoCg.y;

    float t = Y - Cg;
    vec3 total_RGB = vec3(t + Co, Y + Cg, t - Co);

    return max(total_RGB * attenuation, vec3(0.0));
}

// Dual-vector packing for temporal storage
vec4 packDualVector(vec3 dual_theta, float dual_beta) {
    return vec4(dual_theta, dual_beta);
}
vec4 packDualVectorFromEncoded(vec4 aliceEncoded) {
    vec4 tb = alice_theta_beta(aliceEncoded);
    return tb;
}
void unpackDualVector(vec4 packed_, out vec3 dual_theta, out float dual_beta) {
    dual_theta = packed_.xyz;
    dual_beta = packed_.w;
}

// ALICE composition primitives
AliceEncoding mix_alice(AliceEncoding a, AliceEncoding b, float s) {
    AliceEncoding result;
    result.aliceY = mix(a.aliceY, b.aliceY, s);
    result.CoCg = mix(a.CoCg, b.CoCg, s);
    return result;
}

AliceEncoding init_alice() {
    AliceEncoding result;
    result.aliceY = vec4(0.0);
    result.CoCg = vec2(0.0);
    return result;
}

AliceEncoding scale_alice(AliceEncoding A, float x) {
    AliceEncoding tmp;
    tmp.CoCg = A.CoCg * x;
    tmp.aliceY = A.aliceY * x;
    return tmp;
}

void accumulate_alice(inout AliceEncoding accum, AliceEncoding b, float scale) {
    accum.aliceY += b.aliceY * scale;
    accum.CoCg += b.CoCg * scale;
}

// AliceEncoding ⇄ packed vec3 (for 3-float SSBO/texture storage)
vec3 packAlice(AliceEncoding encoded) {
    float s0 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.x, encoded.aliceY.y)));
    float s1 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.z, encoded.aliceY.w)));
    float s2 = uintBitsToFloat(packHalf2x16(vec2(encoded.CoCg.x, encoded.CoCg.y)));
    return vec3(s0, s1, s2);
}

AliceEncoding unpackAlice(float s0, float s1, float s2) {
    AliceEncoding encoded;
    vec2 v0 = unpackHalf2x16(floatBitsToUint(s0));
    vec2 v1 = unpackHalf2x16(floatBitsToUint(s1));
    vec2 v2 = unpackHalf2x16(floatBitsToUint(s2));
    encoded.aliceY = vec4(v0.x, v0.y, v1.x, v1.y);
    encoded.CoCg = v2;
    return encoded;
}

AliceEncoding unpackAlice(vec3 s) {
    return unpackAlice(s.x, s.y, s.z);
}

#endif // ALICE_ENCODE_GLSL

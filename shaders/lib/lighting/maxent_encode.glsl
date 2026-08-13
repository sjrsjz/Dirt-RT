#ifndef MAXENT_ENCODE_GLSL
#define MAXENT_ENCODE_GLSL

#include "/lib/lighting/maxent.glsl"

// ===========================================================================
// MaxEnt light encoding — application-layer encode/decode/project/compose
//
// These are higher-level utilities built on top of the core MaxEnt math library
// (maxent.glsl). They handle:
//   - RGB-to-MaxEnt encoding (radiance_to_maxent)
//   - Irradiance reconstruction (project_maxent_irradiance)
//   - Dual-vector packing for temporal storage
//   - MaxEntEncoding composition (mix, init, scale, accumulate)
//   - MaxEntEncoding ⇄ packed vec3 (for SSBO/texture I/O)
// ===========================================================================

struct MaxEntEncoding {
    vec4 maxEntY;
    vec2 CoCg;
};

MaxEntEncoding radiance_to_maxent(vec3 color, vec3 dir)
{
    MaxEntEncoding result;
    float Y = dot(color, vec3(0.25, 0.5, 0.25));
    float Co = 0.5 * color.r - 0.5 * color.b;
    float Cg = -0.25 * color.r + 0.5 * color.g - 0.25 * color.b;
    result.CoCg = vec2(Co, Cg);
    result.maxEntY = vec4(dir * Y, Y);
    return result;
}

vec3 project_maxent_irradiance(MaxEntEncoding encoded, vec3 N)
{
    float total_omega = encoded.maxEntY.w;
    if (total_omega <= 1e-10) return vec3(0.0);

    float irradiance = maxent_irradiance(encoded.maxEntY, N);
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
vec4 packDualVectorFromEncoded(vec4 maxentEncoded) {
    vec4 tb = maxent_theta_beta(maxentEncoded);
    return tb;
}
void unpackDualVector(vec4 packed_, out vec3 dual_theta, out float dual_beta) {
    dual_theta = packed_.xyz;
    dual_beta = packed_.w;
}

// MaxEnt composition primitives
MaxEntEncoding mix_maxent(MaxEntEncoding a, MaxEntEncoding b, float s) {
    MaxEntEncoding result;
    result.maxEntY = mix(a.maxEntY, b.maxEntY, s);
    result.CoCg = mix(a.CoCg, b.CoCg, s);
    return result;
}

MaxEntEncoding init_maxent() {
    MaxEntEncoding result;
    result.maxEntY = vec4(0.0);
    result.CoCg = vec2(0.0);
    return result;
}

MaxEntEncoding scale_maxent(MaxEntEncoding A, float x) {
    MaxEntEncoding tmp;
    tmp.CoCg = A.CoCg * x;
    tmp.maxEntY = A.maxEntY * x;
    return tmp;
}

void accumulate_maxent(inout MaxEntEncoding accum, MaxEntEncoding b, float scale) {
    accum.maxEntY += b.maxEntY * scale;
    accum.CoCg += b.CoCg * scale;
}

// MaxEntEncoding ⇄ packed vec3 (for 3-float SSBO/texture storage)
vec3 packMaxEnt(MaxEntEncoding encoded) {
    float s0 = uintBitsToFloat(packHalf2x16(vec2(encoded.maxEntY.x, encoded.maxEntY.y)));
    float s1 = uintBitsToFloat(packHalf2x16(vec2(encoded.maxEntY.z, encoded.maxEntY.w)));
    float s2 = uintBitsToFloat(packHalf2x16(vec2(encoded.CoCg.x, encoded.CoCg.y)));
    return vec3(s0, s1, s2);
}

MaxEntEncoding unpackMaxEnt(float s0, float s1, float s2) {
    MaxEntEncoding encoded;
    vec2 v0 = unpackHalf2x16(floatBitsToUint(s0));
    vec2 v1 = unpackHalf2x16(floatBitsToUint(s1));
    vec2 v2 = unpackHalf2x16(floatBitsToUint(s2));
    encoded.maxEntY = vec4(v0.x, v0.y, v1.x, v1.y);
    encoded.CoCg = v2;
    return encoded;
}

MaxEntEncoding unpackMaxEnt(vec3 s) {
    return unpackMaxEnt(s.x, s.y, s.z);
}

#endif // MAXENT_ENCODE_GLSL

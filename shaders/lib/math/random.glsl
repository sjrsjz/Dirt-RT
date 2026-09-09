#ifndef DIRT_RT_LIB_MATH_RANDOM_GLSL
#define DIRT_RT_LIB_MATH_RANDOM_GLSL

// Per-invocation random streams and sequence state.
#include "/lib/math/hash.glsl"
uint iFrame = 0;

void setFrame(uint frame) {
    iFrame = frame;
}

uint wseed;
uint whash(uint seed)
{
    seed = (seed ^ uint(61)) ^ (seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> uint(4));
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> uint(15));
    return seed;
}

float randcore4()
{
    wseed = whash(wseed);

    uint m = (wseed >> 9) | 0x3F800000u;
    return uintBitsToFloat(m) - 1.0;
}

// Fixed-point phase: uint overflow supplies mod 1 before conversion to FP32.
// Large float(frame)*step values otherwise lose the fractional random bits.
const uint WEYL_FRAME = 0x9e3779b9u;
const uint WEYL_SAMPLE = 0x6a09e667u;
uint weyl_idx = 0u;

float randomUnitFloat(uint bits) {
    return uintBitsToFloat((bits >> 9u) | 0x3f800000u) - 1.0;
}

float weyl() {
    weyl_idx += 1u;
    return randomUnitFloat(iFrame * WEYL_FRAME + weyl_idx * WEYL_SAMPLE);
}

float rand(vec3 p3)
{
    p3 += fract(weyl());
    p3 = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float rand(vec2 p)
{
    p += fract(weyl());
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// -----------------------------------------------------------
// 2D Weyl sequence — dedicated for importance sampling
// -----------------------------------------------------------
// Each output dimension has a distinct fixed-point step.
// Frame: φ⁻¹ and 1/π.  Step: √2−1 and √3−1.
const uvec2 WEYL2_FRAME = uvec2(0x9e3779b9u, 0x517cc1b7u);
const uvec2 WEYL2_STEP = uvec2(0x6a09e667u, 0xbb67ae85u);

uint weyl2_idx = 0u;

vec2 weyl2() {
    weyl2_idx += 1u;
    uvec2 phase = iFrame * WEYL2_FRAME + weyl2_idx * WEYL2_STEP;
    return vec2(randomUnitFloat(phase.x), randomUnitFloat(phase.y));
}

// Dedicated 2D quasi-random sampler for importance sampling.
// Returns vec2 in [0,1)² with good 2D low-discrepancy properties,
// independent from the 1D rand() stream.
vec2 rand2(vec3 p) {
    vec2 offset = hash23(p * 32.);
    return fract(offset + weyl2());
}

uvec3 wseed3;
uvec3 whash3(uvec3 seed)
{
    seed = (seed ^ uint(61)) ^ (seed >> uvec3(16));
    seed *= uvec3(9);
    seed = seed ^ (seed >> uvec3(4));
    seed *= uvec3(0x27d4eb2d);
    seed = seed ^ (seed >> uvec3(15));
    return seed;
}

float getRandom() {
    wseed3 = whash3(wseed3.yzx);
    weyl_idx += 1u;
    return randomUnitFloat(wseed3.x + iFrame * WEYL_FRAME
        + weyl_idx * WEYL_SAMPLE);
}

#endif // DIRT_RT_LIB_MATH_RANDOM_GLSL

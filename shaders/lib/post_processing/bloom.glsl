#ifndef BLOOM_GLSL
#define BLOOM_GLSL
#include "/lib/settings.glsl"
#include "/lib/constants.glsl"

// Keep the entire bloom path in FP32. A physical 0.53-degree solar disc has
// roughly two orders of magnitude more radiance per texel than the old broad
// disc, so a half-float clamp would discard the energy before downsampling.
// 1e30 leaves ample headroom for weighted sums while remaining finite.
const float BLOOM_FLOAT_MAX = 1e30;

float bloomSafeFloat(float v) {
    if (isnan(v)) return 0.0;
    if (isinf(v)) return v > 0.0 ? BLOOM_FLOAT_MAX : 0.0;
    return clamp(v, 0.0, BLOOM_FLOAT_MAX);
}

vec3 bloomSafeFloat(vec3 v) {
    return vec3(bloomSafeFloat(v.r), bloomSafeFloat(v.g), bloomSafeFloat(v.b));
}

// Representative wavelengths for linear RGB channels. The model mixes a
// wavelength-independent large-particle component with small-particle
// Rayleigh scattering, where mu_s is proportional to lambda^-4. Gaussian
// variance is proportional to mu_s, hence pure-Rayleigh sigma is proportional
// to lambda^-2. Green is the reference; pure-Rayleigh ratios are approximately
// R=0.807, G=1.000, B=1.400.
vec3 bloomChromaticSigmaScale() {
    const vec3 rgbWavelengthNm = vec3(611.0, 549.0, 464.0);
    vec3 rayleighVariance = pow(vec3(549.0) / rgbWavelengthNm, vec3(4.0));
    // The setting is the Rayleigh fraction of scattering power, so interpolate
    // variance (not sigma) and convert back to a Gaussian standard deviation.
    return sqrt(mix(vec3(1.0), rayleighVariance, BLOOM_CHROMATIC_SCATTER));
}

vec3 bloomDiffusionSigmaScale() {
    return max(vec3(1e-3), bloomChromaticSigmaScale() * BLOOM_DIFFUSION_SCALE);
}

ivec2 bloomOrigin(int l, ivec2 sz) {
    return sz - (sz >> l);
}
ivec2 bloomSize(int l, ivec2 sz) {
    return sz >> (l + 1);
}

// 查找像素所在 LOD 区域。
// 返回 L in [0, 8]，区域边界 [rO, rM]；无效区域返回 L = -1。
void bloomFindLOD(
    ivec2 c,
    ivec2 sz,
    out int L,
    out ivec2 rO,
    out ivec2 rM
) {
    L = -1;
    rO = ivec2(0);
    rM = ivec2(0);

    // 防止 size - c == 0 时传给 findMSB(0u)。
    if (any(lessThan(c, ivec2(0))) ||
        any(greaterThanEqual(c, sz)) ||
        any(lessThanEqual(sz, ivec2(0)))) {
        return;
    }

    ivec2 d = sz - c;

    int Lx = findMSB(uint(sz.x)) - findMSB(uint(d.x));
    int Ly = findMSB(uint(sz.y)) - findMSB(uint(d.y));

    // 修正 findMSB 忽略的尾数部分。
    if (Lx > 0 && uint(d.x) > (uint(sz.x) >> uint(Lx))) {
        --Lx;
    }

    if (Ly > 0 && uint(d.y) > (uint(sz.y) >> uint(Ly))) {
        --Ly;
    }

    // 两个轴不在同一层，或者超过 Bloom Atlas 的最大层数。
    if (Lx != Ly || Lx < 0 || Lx > 8) {
        return;
    }

    ivec2 o = (Lx == 0) ? ivec2(0) : sz - (sz >> Lx);
    ivec2 s = sz >> (Lx + 1);

    if (all(greaterThanEqual(c, o)) &&
        all(lessThan(c, o + s))) {
        L = Lx;
        rO = o;
        rM = o + s - ivec2(1);
    }
}


#endif // BLOOM_GLSL

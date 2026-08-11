#ifndef COMMON_GLSL
#define COMMON_GLSL
#include "/lib/constants.glsl"
#include "/lib/math/hash.glsl"

uint iFrame = 0;

void setFrame(uint frame) {
    iFrame = frame;
}

vec2 rot(vec2 a, float theta) {
    return a.xx * vec2(cos(theta), sin(theta)) + a.yy * vec2(-sin(theta), cos(theta));
}

float hash(float n)
{
    return fract(cos(n) * 41415.92653);
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

void XYZ(vec3 n, out vec3 X, out vec3 Y, out vec3 Z) {
    Y = n;
    X = vec3(n.z, 0, n.x);
    X = abs(n.y) == 1 ? vec3(1, 0, 0) : normalize(X);
    Z = cross(n, X);
}

float ensurePositive(float x, float defaultValue) {
    return (x <= 0.0 || isnan(x) || isinf(x)) ? defaultValue : x;
}

// Weyl sequence — low-discrepancy quasi-random generator
// α = golden-ratio conjugate ≈ 0.618… for frame offset
// β = √2 − 1          ≈ 0.414… for sample offset
// w_n = (frame × α + sample_index × β) mod 1
const float WEYL_FRAME = 0.6180339887498949;
const float WEYL_SAMPLE = 0.4142135623730951;

float weyl_idx = 0.0;

float weyl() {
    weyl_idx += 1.0;
    return float(iFrame) * WEYL_FRAME + weyl_idx * WEYL_SAMPLE;
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
// Each output dimension uses an independent irrational step,
// guaranteeing low-discrepancy 2D stratification.
// Frame: φ⁻¹ and 1/π.  Step: √2−1 and √3−1.
const vec2 WEYL2_FRAME = vec2(0.6180339887498949, 0.3183098861837907);
const vec2 WEYL2_STEP  = vec2(0.4142135623730951, 0.7320508075688772);

float weyl2_idx = 0.0;

vec2 weyl2() {
    weyl2_idx += 1.0;
    return float(iFrame) * WEYL2_FRAME + weyl2_idx * WEYL2_STEP;
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
    // 1. 右移 9 位，提取 wseed3.x 中高 23 位具有随机性的比特作为浮点数尾数 (Mantissa)
    // 2. 与 0x3F800000u 进行位或。
    //    在 IEEE 754 格式中，该操作将符号位置为 0，指数位设为 127，构造出 [1.0, 2.0) 之间的浮点数
    uint m = (wseed3.x >> 9) | 0x3F800000u;
    
    // 3. 将其解释为浮点数，然后减去 1.0，得到严格在 [0.0, 1.0) 之间无损、均匀的随机数
    return fract(uintBitsToFloat(m) - 1.0 + weyl());
}

float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

vec4 reflectanceColor(vec3 c, float cosA) {
    vec3 F0 = c + (1.0 - c) * pow(1.0 - abs(cosA), 5.0);
    return vec4(F0, luma(F0));
}

float GGX_Lamda(float NoX, float a) {
    NoX = max(abs(NoX), 1e-6);
    float a2 = a * a;
    return 0.5 * (sqrt(1.0 + a2 * (1.0 / (NoX * NoX) - 1.0)) - 1.0);
}


float GGX_G2(float VoN, float LoN, float a) {
    float L1 = GGX_Lamda(VoN, a);
    float L2 = GGX_Lamda(LoN, a);
    return clamp((1 + L1) / max(1.0 + L2 + L1, 1e-5), 0, 1);
}

float GGX_G2_standard(float NoV, float NoL, float a) {
    NoV = max(NoV, 1e-6);
    NoL = max(NoL, 1e-6);

    float lambdaV = GGX_Lamda(NoV, a);
    float lambdaL = GGX_Lamda(NoL, a);

    return 1.0 / (1.0 + lambdaV + lambdaL);
}

vec3 GGXNormal(vec3 macroNormal, float roughness, vec2 xi) {
    vec3 randN0;
    randN0.y = -length(macroNormal.xz);
    if (macroNormal.y > 0.99 || macroNormal.y < -0.99)
        randN0.xz = vec2(1, 0);
    else
        randN0.xz = macroNormal.xz * macroNormal.y * inversesqrt(1 - macroNormal.y * macroNormal.y);
    vec3 randN1 = cross(macroNormal, randN0);
    float alpha = xi.x * 2 * PI;
    float cosbeta = min(sqrt(max(0., (1. - xi.y) / (1. + xi.y * (roughness * roughness - 1.)))), 1.);

    return cosbeta * macroNormal + sqrt(1 - cosbeta * cosbeta) * (cos(alpha) * randN0 + sin(alpha) * randN1);
}

vec3 GGXNormal(vec3 macroNormal, float roughness, vec3 pos) {
    return GGXNormal(macroNormal, roughness, rand2(pos));
}

vec3 DiffuseNormal(vec3 macroNormal, vec3 pos) {
    vec3 randN0;
    randN0.y = -length(macroNormal.xz);
    if (macroNormal.y > 0.99 || macroNormal.y < -0.99)
        randN0.xz = vec2(1, 0);
    else
        randN0.xz = macroNormal.xz * macroNormal.y * inversesqrt(1 - macroNormal.y * macroNormal.y);
    vec3 randN1 = cross(macroNormal, randN0);
    vec2 xi = rand2(pos);
    float alpha = xi.x * 2 * PI;
    return sqrt(1 - xi.y) * macroNormal + sqrt(xi.y) * (cos(alpha) * randN0 + sin(alpha) * randN1);
}

vec3 SampleUniformHemisphere(vec3 geometryNormal, vec2 xi) {
    float phi = xi.x * 2.0 * PI;
    float cosTheta = xi.y; // cosTheta 在 [0, 1] 均匀分布
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    
    vec3 localDir = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    
    // 构建局部正交基 (ONB)
    vec3 up = abs(geometryNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, geometryNormal));
    vec3 bitangent = cross(geometryNormal, tangent);
    
    return tangent * localDir.x + bitangent * localDir.y + geometryNormal * localDir.z;
}

vec3 SampleUniformHemisphere(vec3 geometryNormal, vec3 pos) {
    return SampleUniformHemisphere(geometryNormal, rand2(pos));
}

float GGX_D(float costheta, float a) {
    float NoH = max(costheta, 0.0);
    float alpha = max(a, 1e-4);
    float a2 = alpha * alpha;
    float b = 1.0 + (a2 - 1.0) * NoH * NoH;
    return a2 / max(PI * b * b, 1e-20);
}

// GGX normal-distribution PDF in solid angle of the half vector: D(H) * NoH.
float GGXpdf(float costheta, float fai, float a) {
    float NoH = max(costheta, 0.0);
    return GGX_D(NoH, a) * NoH;
}

// NDF-sampled reflection direction PDF for GGX.
// Assumes GGXpdf(NoH, ...) returns D(H) * NoH.
float GGX_ndf_pdf(vec3 wo, vec3 wi, vec3 macroNormal, float roughness) {
    float NoV = dot(macroNormal, wo);
    float NoL = dot(macroNormal, wi);

    if (NoV <= 1e-6 || NoL <= 1e-6)
        return 0.0;

    vec3 Hsum = wo + wi;
    float Hlen2 = dot(Hsum, Hsum);
    if (Hlen2 <= 1e-12)
        return 0.0;

    vec3 H = Hsum * inversesqrt(Hlen2);

    float NoH = dot(macroNormal, H);
    float VoH = dot(wo, H);
    if (NoH <= 1e-6 || VoH <= 1e-6)
        return 0.0;

    float D_NoH = GGXpdf(NoH, 0.0, roughness);

    // p(wi) = D(H) * NoH / (4 * VoH)
    return D_NoH / (4.0 * VoH);
}

float mixp(float F, float S) {
    return F * S / max(1 + (S - 1) * F, 1e-5);
}
float fresnel(vec3 v, vec3 n, float rs) {
    vec2 A;
    A.x = dot(v, n);
    A.y = sqrt(max(1 - (1 - A.x * A.x) * (rs * rs), 0));
    A = (A * rs - A.yx) / max(A * rs + A.yx, 1e-4);
    return 0.5 * dot(A, A);
}

vec4 noised(in vec3 x)
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    vec3 u = w * w * (3.0 - 2.0 * w);
    vec3 du = 6.0 * w * (1.0 - w);

    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    float a = hash(n + 0.0);
    float b = hash(n + 1.0);
    float c = hash(n + 157.0);
    float d = hash(n + 158.0);
    float e = hash(n + 113.0);
    float f = hash(n + 114.0);
    float g = hash(n + 270.0);
    float h = hash(n + 271.0);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return vec4(k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z,
        du * (vec3(k1, k2, k3) + u.yzx * vec3(k4, k5, k6) + u.zxy * vec3(k6, k4, k5) + k7 * u.yzx * u.zxy));
}

float valueNoise(vec3 position) {
    vec3 p = floor(position);
    vec3 f = fract(position);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return mix(mix(mix(hash11(n + 0.0), hash11(n + 1.0), f.x),
            mix(hash11(n + 157.0), hash11(n + 158.0), f.x), f.y),
        mix(mix(hash11(n + 113.0), hash11(n + 114.0), f.x),
            mix(hash11(n + 270.0), hash11(n + 271.0), f.x), f.y), f.z);
}

vec4 fbm3D(in vec3 x, int n)
{
    const float scale = 1.5;

    float a = 0.0;
    float b = 0.5;
    float f = 1.0;
    vec3 d = vec3(0.0);
    for (int i = 0; i < n; i++)
    {
        vec4 n = noised(f * x * scale);
        a += b * n.x; // accumulate values
        d += b * n.yzw * f * scale; // accumulate derivatives
        b *= 0.5 / (1 + dot(d, d)); // amplitude decrease
        f *= 2; // frequency increase
    }

    return vec4(a, d);
}
#endif // COMMON_GLSL

#ifndef COMMON_GLSL
#define COMMON_GLSL
#include "/lib/constants.glsl"

#ifdef USE_NOISE_TEXTURE
uniform sampler2D NoiseTexture;
#endif

uint iFrame = 0;
//const float PI=3.14159265358;
struct object {
    float d;
    float d2;
    int id;
    int i_id;
    int s;
};
struct material {
    vec3 Cs;
    vec3 Cd;
    vec2 S;
    vec4 R;
    vec3 light;
};
struct info {
    vec3 rd_i;
    vec3 rd_o;
    vec3 n;
    vec3 microNormal;
    vec3 macroNormal;
    vec3 p;
    material surface;
    float n_i;
    float n_o;
    float distance;
    float sampleDistance;
    vec3 color;
    vec3 color2;
    vec3 color3;
    vec3 shade;
    vec3 absorption;
    vec3 emission;
    float sampleRoughness;
    int type;
    bool inside;
};

//----------------------------------------------------------------------------------------
//  1 out, 1 in...
float hash11(float p)
{
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}


//----------------------------------------------------------------------------------------
//  1 out, 3 in...
float hash13(vec3 p3)
{
    p3 = fract(p3 * .1031);
    p3 = fract(tan(dot(p3, p3) * 20 * atan(p3)));
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 rot(vec2 a, float theata) {
    return a.xx * vec2(cos(theata), sin(theata)) + a.yy * vec2(-sin(theata), cos(theata));
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

    return float(wseed) * (1.0 / 4294967296.0);
}

void XYZ(vec3 n, out vec3 X, out vec3 Y, out vec3 Z) {
    Y = n;
    X = vec3(n.z, 0, n.x);
    X = abs(n.y) == 1 ? vec3(1, 0, 0) : normalize(X);
    Z = cross(n, X);
}
float rand_i = 0.;
float rand(vec3 p3)
{
    p3 *= 31;
    rand_i += 0.4;
    p3 += rand_i + iFrame;
    p3 = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float rand(vec2 p)
{
    rand_i += 0.4;
    p += rand_i + iFrame;
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
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

float getRnd() {
    wseed3 = whash3(wseed3.yzx);
    return fract(float(wseed3.x) * (1.0 / 4294967296.0));
}

vec3 rndS(vec3 pos) {
    return normalize(tan(vec3(rand(pos) - 0.5, rand(pos) - 0.5, rand(pos) - 0.5)));
}
float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

vec4 rColor(vec3 c, float cosA) {
    vec3 F0 = c + (1.0 - c) * pow(1.0 - abs(cosA), 5.0);
    return vec4(F0, luma(F0));
}

float GGX_Lamda(float VoN, float a) {
    return (-1 + sqrt(1 + a * a * (1. / (VoN * VoN) - 1))) * 0.5;
}

float GGX_G2(float VoN, float LoN, float a) {
    float L1 = GGX_Lamda(VoN, a);
    float L2 = GGX_Lamda(LoN, a);
    return clamp((1 + L1) / (1 + L2 + L1), 0, 1);
}
vec3 GGXNormal(vec3 normal, float roughness, vec3 pos) {
    vec3 randN0;
    randN0.y = -length(normal.xz);
    if (normal.y > 0.99 || normal.y < -0.99)
        randN0.xz = vec2(1, 0);
    else
        randN0.xz = normal.xz * normal.y * inversesqrt(1 - normal.y * normal.y);
    vec3 randN1 = cross(normal, randN0);
    float alpha = rand(pos) * 2 * PI;
    float tmp = rand(pos);
    float cosbeta = min(sqrt(max(0., (1. - tmp) / (1. + tmp * (roughness * roughness - 1.)))), 1.);

    return cosbeta * normal + sqrt(1 - cosbeta * cosbeta) * (cos(alpha) * randN0 + sin(alpha) * randN1);
}
vec3 DiffuseNormal(vec3 normal, vec3 pos) {
    vec3 randN0;
    randN0.y = -length(normal.xz);
    if (normal.y > 0.99 || normal.y < -0.99)
        randN0.xz = vec2(1, 0);
    else
        randN0.xz = normal.xz * normal.y * inversesqrt(1 - normal.y * normal.y);
    vec3 randN1 = cross(normal, randN0);
    float alpha = rand(pos) * 2 * PI;
    float tmp = rand(pos);
    return sqrt(1 - tmp) * normal + sqrt(tmp) * (cos(alpha) * randN0 + sin(alpha) * randN1);
}

// 方向重要性采样
vec4 WeightedDiffuse(vec3 normal, vec3 pos, float power) {
    // 返回采样方向和权重
    vec3 randN0;
    randN0.y = -length(normal.xz);
    if (normal.y > 0.99 || normal.y < -0.99)
        randN0.xz = vec2(1, 0);
    else
        randN0.xz = normal.xz * normal.y * inversesqrt(1 - normal.y * normal.y);
    vec3 randN1 = cross(normal, randN0);

    // 生成随机方向
    float alpha = rand(pos) * 2 * PI;
    float rnd = rand(pos);
    float tmp = 1 - 2 * pow(rnd,power);
    vec3 n = tmp * normal + sqrt(1 - tmp * tmp) * (cos(alpha) * randN0 + sin(alpha) * randN1);

    // 计算权重
    float pdf = pow(max(rnd,1e-5),1-power) / (PI*power);



    return vec4(n, pdf);
}
float GGXpdf(float costheta, float fai, float a) {
    float a2 = a * a;
    float b = 1 + (a2 - 1) * costheta * costheta;
    return a2 * costheta / (PI * b * b);
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


#ifdef USE_NOISE_TEXTURE
float sample3Dnoise(in vec3 v) {
    vec3 p = mod(floor(v), 256.0); // Add this line to make the noise repeat every 256 units
    vec3 f = fract(v);
    f = f*f*(3.-2.*f);
    
    vec2 uv = (p.xy+vec2(37.,17.)*p.z) + f.xy;
    vec2 rg = textureLod( NoiseTexture, fract((uv+.5)/256.), 0.).yx;
    return mix(rg.x, rg.y, f.z);
}
#endif
float valueNoise(vec3 position) {
#ifndef USE_NOISE_TEXTURE
    vec3 p = floor(position);
    vec3 f = fract(position);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash11(n+  0.0), hash11(n+  1.0),f.x),
                   mix( hash11(n+157.0), hash11(n+158.0),f.x),f.y),
               mix(mix( hash11(n+113.0), hash11(n+114.0),f.x),
                   mix( hash11(n+270.0), hash11(n+271.0),f.x),f.y),f.z);
#else
    // 对二维噪声纹理进行三维采样
    return sample3Dnoise(position);
#endif
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
#endif COMMON_GLSL

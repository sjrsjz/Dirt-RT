#ifndef LIGHTING_EON_GLSL
#define LIGHTING_EON_GLSL

// ============================================================================
// EON rough diffuse BRDF + four-parameter directional convolution
// ============================================================================
//
// EON is the energy-preserving Oren--Nayar model by Portsmouth, Kutz and Hill.
// It consists of a Fujii Oren--Nayar single-scattering lobe plus a reciprocal
// multiple-scattering compensation lobe.
//
// This file also supplies a LUT-assisted convolution of EON with the
// four-parameter incident-light representation
//
//     maxEntY = vec4(v, omega),  |v| <= omega .
//
// Its normalized directional energy density is
//
//   p(u) = (1-kappa^2)^2 / (4*pi*(1-kappa*dot(axis,u))^3),
//
// where kappa is exactly the normalized first-moment length. The Lambert
// term is closed form. The view-dependent FON partition and the
// multiple-scattering residual are reconstructed from Iris custom textures.
//
// Important conventions
// ---------------------
// - wi and wo point away from the surface.
// - normal, wi and wo must be normalized.
// - roughness is EON's linear r in [0,1], not GGX alpha. If the material stores
//   alpha = perceptualRoughness^2, pass sqrt(alpha).
// - rho is the EON single-scattering albedo. It is intentionally inside the
//   BRDF because EON's multiple-scattering response is nonlinear in rho.
// - totalRgb is integral L_i(wi) dOmega. eon_project_maxent() therefore returns
//   outgoing radiance, including the BRDF and NoL factors.
// - The MaxEnt overload assumes CoCg is angularly shared, as in
//   project_maxent_irradiance().
//
// The FON directional-albedo polynomial is the approximation used by EON.
// ============================================================================

// Vulkanite exposes Iris custom textures in descriptor set 2, sorted by
// sampler name. aaaRtBlueNoise occupies binding 0, so this sampler occupies
// binding 1 in ray-tracing programs. Ordinary Iris programs bind it by name.
#ifdef EON_RT_CUSTOM_TEXTURES
layout(set = 2, binding = 1) uniform sampler3D aabEonKappaLut;
#else
uniform sampler3D aabEonKappaLut;
#endif

const float EON_PI = 3.14159265358979323846;
const float EON_INV_PI = 1.0 / EON_PI;
const float EON_C1 = 0.5 - 2.0 / (3.0 * EON_PI);
const float EON_C2 = 2.0 / 3.0 - 28.0 / (15.0 * EON_PI);

float eon_fon_g_over_pi(float mu) {
    float x = 1.0 - clamp(mu, 0.0, 1.0);
    return x * (0.0571085289 + x * (0.491881867
        + x * (-0.332181442 + x * 0.0714429953)));
}

// Unit-albedo FON directional albedo E_F(mu).
float eon_fon_directional_albedo(float mu, float roughness) {
    float r = clamp(roughness, 0.0, 1.0);
    float AF = 1.0 / (1.0 + EON_C1 * r);
    return AF * (1.0 + r * eon_fon_g_over_pi(mu));
}

float eon_fon_average_albedo(float roughness) {
    float r = clamp(roughness, 0.0, 1.0);
    float AF = 1.0 / (1.0 + EON_C1 * r);
    return AF * (1.0 + EON_C2 * r);
}

float eon_fon_missing(float mu) {
    return EON_C1 - eon_fon_g_over_pi(mu);
}

// Evaluate the EON BRDF. This returns f_r, not f_r * NoL.
vec3 eon_brdf(vec3 wi, vec3 wo, vec3 normal,
        float roughness, vec3 rho) {
    float NoI = dot(normal, wi);
    float NoO = dot(normal, wo);
    if (NoI <= 0.0 || NoO <= 0.0)
        return vec3(0.0);

    float r = clamp(roughness, 0.0, 1.0);
    rho = clamp(rho, vec3(0.0), vec3(1.0));

    float AF = 1.0 / (1.0 + EON_C1 * r);
    float s = dot(wi, wo) - NoI * NoO;
    float sOverTF = s > 0.0 ? s / max(max(NoI, NoO), 1e-6) : s;
    vec3 singleScatter = rho * (EON_INV_PI * AF)
        * max(1.0 + r * sOverTF, 0.0);

    // Stable form of the compensation lobe. Writing it in terms of
    // (1-E_F)/(1-<E_F>) directly produces 0/0 at r=0; the common factors of r
    // and AF have been cancelled analytically here.
    if (r <= 1e-6)
        return singleScatter;

    float averageEF = AF * (1.0 + EON_C2 * r);
    vec3 rhoMS = rho * rho * averageEF
        / max(vec3(1.0) - rho * (1.0 - averageEF), vec3(1e-6));
    float missingProduct = eon_fon_missing(NoI) * eon_fon_missing(NoO);
    float compensationShape = AF * r * missingProduct
        / max(EON_C1 - EON_C2, 1e-6);
    vec3 multipleScatter = rhoMS * (EON_INV_PI * compensationShape);

    return max(singleScatter + multipleScatter, vec3(0.0));
}

// Directional albedo of the complete EON BRDF under uniform illumination.
// This gives the matching white-furnace result for the same FON fit.
vec3 eon_directional_albedo(vec3 rho, float roughness, float mu) {
    float r = clamp(roughness, 0.0, 1.0);
    rho = clamp(rho, vec3(0.0), vec3(1.0));
    float EF = eon_fon_directional_albedo(mu, r);
    float averageEF = eon_fon_average_albedo(r);
    vec3 rhoMS = rho * rho * averageEF
        / max(vec3(1.0) - rho * (1.0 - averageEF), vec3(1e-6));
    return max(rho * EF + rhoMS * (1.0 - EF), vec3(0.0));
}

// ============================================================================
// EON importance sampling (CLTC)
// ============================================================================
// Port of the authors' MIT-licensed reference implementation:
// https://github.com/portsmouth/EON-diffuse
//
// EON is not a microfacet BRDF, so this is not a GGX-style VNDF. Its official
// view-conditioned importance sampler is a Clipped Linearly Transformed Cosine
// (CLTC), mixed with a small uniform-hemisphere proposal for defensive
// sampling. The returned PDF is with respect to solid angle.

mat3 eon_surface_frame(vec3 normal) {
    vec3 tangent = abs(normal.z) < 0.999
        ? normalize(cross(vec3(0.0, 0.0, 1.0), normal))
        : normalize(cross(vec3(0.0, 1.0, 0.0), normal));
    return mat3(tangent, cross(normal, tangent), normal);
}

// V is expressed in a local frame whose +Z axis is the surface normal.
mat3 eon_ltc_frame(vec3 V) {
    float lengthSquared = dot(V.xy, V.xy);
    vec3 X = lengthSquared > 1e-12
        ? vec3(V.x, V.y, 0.0) * inversesqrt(lengthSquared)
        : vec3(1.0, 0.0, 0.0);
    vec3 Y = vec3(-X.y, X.x, 0.0);
    return mat3(X, Y, vec3(0.0, 0.0, 1.0));
}

void eon_ltc_coefficients(float mu, float roughness,
        out float a, out float b, out float c, out float d) {
    float r = clamp(roughness, 0.0, 1.0);
    mu = clamp(mu, 0.0, 1.0);
    a = 1.0 + r * (0.303392
        + (-0.518982 + 0.111709 * mu) * mu
        + (-0.276266 + 0.335918 * mu) * r);
    b = r * (-1.16407 + 1.15859 * mu
        + (0.150815 - 0.150105 * mu) * r)
        / (mu * mu * mu - 1.43545);
    c = 1.0 + r * (0.20013
        + (-0.506373 + 0.261777 * mu) * mu);
    d = r * (0.540852
        + (-1.01625 + 0.475392 * mu) * mu)
        / (-1.0743 + (0.0725628 + mu) * mu);
}

vec4 eon_cltc_sample_local(vec3 woLocal, float roughness, vec2 xi) {
    float a, b, c, d;
    eon_ltc_coefficients(woLocal.z, roughness, a, b, c, d);

    float radius = sqrt(clamp(xi.x, 0.0, 1.0));
    float phi = 2.0 * EON_PI * xi.y;
    float x = radius * cos(phi);
    float y = radius * sin(phi);

    float vz = inversesqrt(d * d + 1.0);
    float clippedArea = 0.5 * (1.0 + vz);
    x = -mix(sqrt(max(1.0 - y * y, 0.0)), x, clippedArea);

    vec3 wh = vec3(x, y,
        sqrt(max(1.0 - x * x - y * y, 0.0)));
    float pdfWh = wh.z / max(EON_PI * clippedArea, 1e-8);

    vec3 wiUnnormalized = vec3(
        a * wh.x + b * wh.z,
        c * wh.y,
        d * wh.x + wh.z);
    float wiLength = length(wiUnnormalized);
    float determinant = max(c * (a - b * d), 1e-8);
    float pdfWi = pdfWh * wiLength * wiLength * wiLength / determinant;

    vec3 wiLocal = normalize(eon_ltc_frame(woLocal) * wiUnnormalized);
    return vec4(wiLocal, max(pdfWi, 0.0));
}

float eon_cltc_pdf_local(vec3 woLocal, vec3 wiLocal, float roughness) {
    vec3 wi = transpose(eon_ltc_frame(woLocal)) * wiLocal;
    float a, b, c, d;
    eon_ltc_coefficients(woLocal.z, roughness, a, b, c, d);

    float determinant = max(c * (a - b * d), 1e-8);
    vec3 wh = vec3(
        c * (wi.x - b * wi.z),
        (a - b * d) * wi.y,
        -c * (d * wi.x - a * wi.z));
    float lengthSquared = max(dot(wh, wh), 1e-12);
    float vz = inversesqrt(d * d + 1.0);
    float clippedArea = 0.5 * (1.0 + vz);
    return determinant * determinant / (lengthSquared * lengthSquared)
        * max(wh.z, 0.0) / max(EON_PI * clippedArea, 1e-8);
}

vec3 eon_uniform_hemisphere_sample_local(vec2 xi) {
    float mu = clamp(xi.x, 0.0, 1.0);
    float sinTheta = sqrt(max(1.0 - mu * mu, 0.0));
    float phi = 2.0 * EON_PI * xi.y;
    return vec3(sinTheta * cos(phi), sinTheta * sin(phi), mu);
}

float eon_uniform_sampling_probability(float mu, float roughness) {
    float angularFit = 0.162925
        + (-0.372058 + (0.538233 - 0.290822 * mu) * mu) * mu;
    return clamp(pow(clamp(roughness, 0.0, 1.0), 0.1)
        * angularFit, 0.0, 0.95);
}

vec4 eon_sample_local(vec3 woLocal, float roughness, vec2 xi) {
    float uniformProbability = eon_uniform_sampling_probability(
        clamp(woLocal.z, 0.0, 1.0), roughness);
    float cltcProbability = 1.0 - uniformProbability;

    vec3 wiLocal;
    float cltcPdf;
    if (uniformProbability > 1e-6 && xi.x < uniformProbability) {
        vec2 remappedXi = vec2(xi.x / uniformProbability, xi.y);
        wiLocal = eon_uniform_hemisphere_sample_local(remappedXi);
        cltcPdf = eon_cltc_pdf_local(woLocal, wiLocal, roughness);
    } else {
        vec2 remappedXi = vec2(
            (xi.x - uniformProbability) / max(cltcProbability, 1e-6),
            xi.y);
        vec4 cltcSample = eon_cltc_sample_local(
            woLocal, roughness, remappedXi);
        wiLocal = cltcSample.xyz;
        cltcPdf = cltcSample.w;
    }

    float pdf = uniformProbability * (0.5 * EON_INV_PI)
        + cltcProbability * cltcPdf;
    return vec4(wiLocal, max(pdf, 0.0));
}

float eon_pdf_local(vec3 woLocal, vec3 wiLocal, float roughness) {
    if (woLocal.z <= 0.0 || wiLocal.z <= 0.0)
        return 0.0;
    float uniformProbability = eon_uniform_sampling_probability(
        clamp(woLocal.z, 0.0, 1.0), roughness);
    float cltcPdf = eon_cltc_pdf_local(woLocal, wiLocal, roughness);
    return uniformProbability * (0.5 * EON_INV_PI)
        + (1.0 - uniformProbability) * cltcPdf;
}

// World-space EON/CLTC sample. xyz is wi and w is its solid-angle PDF.
vec4 eon_sample_direction(vec3 wo, vec3 normal,
        float roughness, vec2 xi) {
    mat3 frame = eon_surface_frame(normal);
    vec3 woLocal = transpose(frame) * wo;
    if (woLocal.z <= 0.0)
        return vec4(0.0);
    vec4 sampleLocal = eon_sample_local(woLocal, roughness, xi);
    return vec4(normalize(frame * sampleLocal.xyz), sampleLocal.w);
}

float eon_direction_pdf(vec3 wo, vec3 wi, vec3 normal,
        float roughness) {
    mat3 frame = eon_surface_frame(normal);
    return eon_pdf_local(transpose(frame) * wo,
        transpose(frame) * wi, roughness);
}

// Exact clamped-cosine response of the cubic-reciprocal closure.
float eon_cosine_response(float kappa, float cosine) {
    float oneMinusK2 = (1.0 - kappa) * (1.0 + kappa);
    float kCosine = kappa * cosine;
    float denominator = sqrt(max(
        oneMinusK2 + kCosine * kCosine, 1.0e-20));
    if (kCosine < 0.0) {
        float sumTerm = denominator - kCosine;
        return oneMinusK2 * oneMinusK2
            / (4.0 * denominator * sumTerm * sumTerm);
    }
    return (oneMinusK2 + 2.0 * kCosine * kCosine)
        / (4.0 * denominator) + 0.5 * kCosine;
}

float eon_cubic_bernstein(vec4 control, float t) {
    float a = mix(control.x, control.y, t);
    float b = mix(control.y, control.z, t);
    float c = mix(control.z, control.w, t);
    return mix(mix(a, b, t), mix(b, c, t), t);
}

// The RGBA channels are cubic Bernstein controls in transformed kappa.
// Texture axes are axis polar angle, outgoing cosine, and four packed
// relative-azimuth segments.
float eon_lut_fon_partition(float kappa, vec3 axis,
        float muO, float sineO, float irradiance) {
    const float KAPPA_MAX = 0.999;
    const float INVERSE_ATANH_KAPPA_MAX = 0.2631439642;
    const float SEGMENT_COUNT = 4.0;

    float boundedKappa = clamp(kappa, 0.0, KAPPA_MAX);
    float transformedKappa = 0.5
        * log((1.0 + boundedKappa) / (1.0 - boundedKappa))
        * INVERSE_ATANH_KAPPA_MAX;
    float segment = min(floor(SEGMENT_COUNT * transformedKappa),
        SEGMENT_COUNT - 1.0);
    float localKappa = SEGMENT_COUNT * transformedKappa - segment;

    ivec3 size = textureSize(aabEonKappaLut, 0);
    float axisCoordinate = 1.0
        - acos(clamp(axis.z, -1.0, 1.0)) / EON_PI;
    float axisTexel = (0.5 + axisCoordinate * float(size.x - 1))
        / float(size.x);
    float outgoingTexel = (0.5 + muO * float(size.y - 1))
        / float(size.y);

    float transverse = sqrt(max(
        (1.0 - axis.z) * (1.0 + axis.z), 0.0));
    float relativeCosine = transverse > 1.0e-7
        ? clamp(axis.x / transverse, -1.0, 1.0) : 0.0;
    float relativeCoordinate = 0.5 * (relativeCosine + 1.0);
    float relativeResolution = float(size.z) / SEGMENT_COUNT;
    float packedTexel = segment * relativeResolution
        + relativeCoordinate * (relativeResolution - 1.0);
    float packedCoordinate = (packedTexel + 0.5) / float(size.z);

    vec4 control = textureLod(aabEonKappaLut,
        vec3(axisTexel, outgoingTexel, packedCoordinate), 0.0);
    return irradiance * sineO
        * eon_cubic_bernstein(control, localKappa);
}

// Closed missing-energy residual; kept arithmetic-only to avoid a second
// incoherent texture access.
float eon_missing_shape_integral_small_k(float kappa, float cosine)
{
    float c2 = cosine * cosine;
    float q0 = 0.05380804990;
    float q1 = cosine * 0.1194479670;
    float q2 = -0.04071064876 + c2 * 0.1221319463;
    float q3 = cosine * (-0.03198186694 + c2 * 0.05330311157);
    float q4 = -0.007239172103 + c2
        * (0.03168107227 - c2 * 0.01660592660);
    float q5 = cosine * (-0.01774579306 + c2
        * (0.05083183402 - c2 * 0.02975771714));
    float q6 = -0.002621532148 + c2 * (0.01161714248 + c2
        * (-0.003170355181 - c2 * 0.004317443507));
    float q7 = cosine * (-0.01108880632 + c2 * (0.04064661330 + c2
        * (-0.03859071524 + c2 * 0.009719148892)));
    float q8 = -0.001230165015 + c2 * (0.004962958342 + c2
        * (0.001746585325 + c2 * (-0.006197769745
        + c2 * 0.001007317627)));
    float q9 = cosine * (-0.007560696613 + c2 * (0.03326857278 + c2
        * (-0.04438954593 + c2 * (0.02482292180
        - c2 * 0.006052431775))));
    float q10 = -0.0006675136980 + c2 * (0.002268632956 + c2
        * (0.003498986647 + c2 * (-0.007236667336 + c2
        * (0.002589612020 - c2 * 0.0004666163294))));
    float result = q10;
    result = result * kappa + q9;
    result = result * kappa + q8;
    result = result * kappa + q7;
    result = result * kappa + q6;
    result = result * kappa + q5;
    result = result * kappa + q4;
    result = result * kappa + q3;
    result = result * kappa + q2;
    result = result * kappa + q1;
    return result * kappa + q0;
}

float eon_missing_shape_integral_axis(float kappa, float cosine)
{
    float k2 = kappa * kappa;
    float one_minus_k2 = (1.0 - kappa) * (1.0 + kappa);
    float k6 = k2 * k2 * k2;
    if (cosine >= 0.0)
    {
        float p = 42865797180.0 + kappa * (-47591289882.0 + kappa
            * (-24453317468.0 + kappa * (9856501519.0 + kappa
            * (27969867127.0 - 13756200.0 * kappa))));
        float q = -7144299530.0 + kappa * (-2784567648.0 + kappa
            * (2280134616.0 + 3300999181.0 * kappa));
        float one_plus_k = 1.0 + kappa;
        return (one_plus_k * one_plus_k * kappa * p
            - 6.0 * one_minus_k2 * one_minus_k2 * q
            * log(max(1.0 - kappa, 1.0e-20)))
            / (120000000000.0 * k6);
    }
    float p = 42865797180.0 + kappa * (47591289882.0 + kappa
        * (-24453317468.0 + kappa * (-9856501519.0 + kappa
        * (27969867127.0 + 13756200.0 * kappa))));
    float q = 7144299530.0 + kappa * (-2784567648.0 + kappa
        * (-2280134616.0 + 3300999181.0 * kappa));
    float one_minus_k = 1.0 - kappa;
    return (-one_minus_k * one_minus_k * kappa * p
        + 6.0 * one_minus_k2 * one_minus_k2 * q * log(1.0 + kappa))
        / (120000000000.0 * k6);
}

float eon_asinh_endpoint_difference(float z1, float z0, float s2)
{
    float root1 = sqrt(max(z1 * z1 + s2, 1.0e-30));
    float root0 = sqrt(max(z0 * z0 + s2, 1.0e-30));
    if (z0 >= 0.0)
        return log((z1 + root1) / (z0 + root0));
    if (z1 <= 0.0)
        return log((-z0 + root0) / (-z1 + root1));
    return log(z1 + root1) + log(-z0 + root0) - log(s2);
}

float eon_missing_shape_integral_closed(float kappa, float cosine)
{
    if (kappa <= 0.4)
        return eon_missing_shape_integral_small_k(kappa, cosine);

    float c2 = cosine * cosine;
    float one_minus_c2 = max((1.0 - abs(cosine))
        * (1.0 + abs(cosine)), 0.0);
    if (one_minus_c2 < 1.0e-4)
        return eon_missing_shape_integral_axis(kappa, cosine);

    const float s0 = -0.0004585400;
    const float s1 = 0.3300999181;
    const float s2m = 0.0760044872;
    const float s3 = -0.0464094608;
    const float s4 = -0.0714429953;
    float k2 = kappa * kappa;
    float n0 = 2.0 + k2 * (1.0 - c2);
    float n1 = -4.0 * kappa * cosine;
    float n2 = k2 * (3.0 * c2 - 1.0);
    float pm1 = s0 * n0;
    float pm2 = s0 * n1 + s1 * n0;
    float pm3 = s0 * n2 + s1 * n1 + s2m * n0;
    float pm4 = s1 * n2 + s2m * n1 + s3 * n0;
    float pm5 = s2m * n2 + s3 * n1 + s4 * n0;
    float pm6 = s3 * n2 + s4 * n1;
    float pm7 = s4 * n2;

    float inverse_k = 1.0 / kappa;
    float inverse2 = inverse_k * inverse_k;
    float inverse3 = inverse2 * inverse_k;
    float inverse4 = inverse3 * inverse_k;
    float inverse5 = inverse4 * inverse_k;
    float inverse6 = inverse5 * inverse_k;
    float inverse7 = inverse6 * inverse_k;
    float inverse8 = inverse7 * inverse_k;
    float v1 = pm1 * inverse2;
    float v2 = pm2 * inverse3;
    float v3 = pm3 * inverse4;
    float v4 = pm4 * inverse5;
    float v5 = pm5 * inverse6;
    float v6 = pm6 * inverse7;
    float v7 = pm7 * inverse8;

    float p0 = cosine * (v1 + cosine * (v2 + cosine * (v3 + cosine
        * (v4 + cosine * (v5 + cosine * (v6 + cosine * v7))))));
    float p1 = v1 + cosine * (2.0 * v2 + cosine * (3.0 * v3 + cosine
        * (4.0 * v4 + cosine * (5.0 * v5 + cosine
        * (6.0 * v6 + cosine * 7.0 * v7)))));
    float p2 = v2 + cosine * (3.0 * v3 + cosine * (6.0 * v4 + cosine
        * (10.0 * v5 + cosine * (15.0 * v6 + cosine * 21.0 * v7))));
    float p3 = v3 + cosine * (4.0 * v4 + cosine * (10.0 * v5
        + cosine * (20.0 * v6 + cosine * 35.0 * v7)));
    float p4 = v4 + cosine * (5.0 * v5 + cosine
        * (15.0 * v6 + cosine * 35.0 * v7));
    float p5 = v5 + cosine * (6.0 * v6 + cosine * 21.0 * v7);
    float p6 = v6 + cosine * 7.0 * v7;
    float p7 = v7;

    float one_minus_k2 = (1.0 - kappa) * (1.0 + kappa);
    float ss = one_minus_k2 * one_minus_c2;
    float ss2 = ss * ss;
    float ss3 = ss2 * ss;
    float r0 = (-p1 - 2.0 * ss * p3 + 8.0 * ss2 * p5
        - 16.0 * ss3 * p7) / 3.0;
    float r1 = (2.0 * p0 - 2.0 * ss2 * p4 + 5.0 * ss3 * p6)
        / (2.0 * ss);
    float r2 = -p3 + 4.0 * ss * p5 - 8.0 * ss2 * p7;
    float r3 = (2.0 * p0 + ss * p2 - 4.0 * ss2 * p4
        + 10.0 * ss3 * p6) / (3.0 * ss2);
    float r4 = p5 - 2.0 * ss * p7;
    float r5 = 0.5 * p6;
    float r6 = p7 / 3.0;
    float a = p4 - 2.5 * ss * p6;

    float z0 = -cosine;
    float z1 = kappa - cosine;
    float polynomial0 = r0 + z0 * (r1 + z0 * (r2 + z0 * (r3
        + z0 * (r4 + z0 * (r5 + z0 * r6)))));
    float polynomial1 = r0 + z1 * (r1 + z1 * (r2 + z1 * (r3
        + z1 * (r4 + z1 * (r5 + z1 * r6)))));
    float q0 = z0 * z0 + ss;
    float q1 = z1 * z1 + ss;
    float rational = polynomial1 / (q1 * sqrt(q1))
        - polynomial0 / (q0 * sqrt(q0));
    float transcendental = a * eon_asinh_endpoint_difference(z1, z0, ss);
    return 0.25 * one_minus_k2 * one_minus_k2
        * (rational + transcendental);
}


vec3 eon_channel_response(float kappa, vec3 axis,
        float muO, float sineO, float roughness, vec3 rho) {
    float irradiance = eon_cosine_response(kappa, axis.z);
    float fonPartition = eon_lut_fon_partition(
        kappa, axis, muO, sineO, irradiance);

    float AF = 1.0 / (1.0 + EON_C1 * roughness);
    vec3 single = rho * (AF * EON_INV_PI)
        * (irradiance + roughness * fonPartition);

    float average = AF * (1.0 + EON_C2 * roughness);
    vec3 rhoMS = rho * rho * average
        / max(vec3(1.0) - rho * (1.0 - average), vec3(1.0e-6));
    float shapeIntegral = eon_missing_shape_integral_closed(kappa, axis.z);
    vec3 multiple = rhoMS * EON_INV_PI * (roughness * AF)
        * (eon_fon_missing(muO) * shapeIntegral / (EON_C1 - EON_C2));
    return single + multiple;
}

// Convolve a total RGB incident-light energy with EON. totalRgb and
// maxEntY.w must describe the same signal; only their ratio/chroma differs.
vec3 eon_project_maxent(vec4 maxEntY, vec3 totalRgb,
        vec3 normal, vec3 wo, float roughness, vec3 rho) {
    float omega = maxEntY.w;
    float NoO = dot(normal, wo);
    if (omega <= 1e-8 || NoO <= 0.0)
        return vec3(0.0);

    totalRgb = max(totalRgb, vec3(0.0));
    rho = clamp(rho, vec3(0.0), vec3(1.0));
    float r = clamp(roughness, 0.0, 1.0);
    float momentLength = length(maxEntY.xyz);
    vec3 axis = momentLength > 1e-8
        ? maxEntY.xyz / momentLength : normal;
    float kappa = clamp(momentLength / omega, 0.0, 1.0);

    // FP16 storage moves directional atoms slightly inside the cone.
    if (1.0 - kappa <= 2e-3) {
        float NoI = max(dot(normal, axis), 0.0);
        return totalRgb * eon_brdf(axis, wo, normal, r, rho) * NoI;
    }

    // Uniform directional energy has an exact response.
    if (kappa <= 1e-5)
        return totalRgb * eon_directional_albedo(rho, r, NoO)
            / (4.0 * EON_PI);

    float muO = clamp(NoO, 0.0, 1.0);
    float sineO = sqrt(max((1.0 - muO) * (1.0 + muO), 0.0));
    vec3 tangent;
    if (sineO > 1.0e-6)
        tangent = (wo - normal * muO) / sineO;
    else
        tangent = eon_surface_frame(normal)[0];
    vec3 bitangent = cross(normal, tangent);
    vec3 localAxis = vec3(
        dot(axis, tangent), dot(axis, bitangent), dot(axis, normal));

    return totalRgb * eon_channel_response(
        kappa, localAxis, muO, sineO, r, rho);
}

// Convert the MaxEnt shared-chroma representation back to its total RGB energy
// and evaluate the EON convolution. This overload deliberately accepts the raw
// fields instead of MaxEntEncoding so the file has no struct/include cycle.
vec3 eon_project_maxent(vec4 maxEntY, vec2 CoCg,
        vec3 normal, vec3 wo, float roughness, vec3 rho) {
    float Y = maxEntY.w;
    float t = Y - CoCg.y;
    vec3 totalRgb = vec3(t + CoCg.x, Y + CoCg.y, t - CoCg.x);
    return eon_project_maxent(maxEntY, totalRgb,
        normal, wo, roughness, rho);
}

#endif // LIGHTING_EON_GLSL

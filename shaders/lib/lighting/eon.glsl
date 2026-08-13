#ifndef LIGHTING_EON_GLSL
#define LIGHTING_EON_GLSL

// ============================================================================
// EON rough diffuse BRDF + four-parameter MaxEnt convolution
// ============================================================================
//
// EON is the energy-preserving Oren--Nayar model by Portsmouth, Kutz and Hill.
// It consists of a Fujii Oren--Nayar single-scattering lobe plus a reciprocal
// multiple-scattering compensation lobe.
//
// This file also supplies a positive, no-ringing convolution of EON with the
// four-parameter MaxEnt/MaxEnt incident-light representation
//
//     maxEntY = vec4(v, omega),  |v| <= omega .
//
// The convolution samples the normalized MaxEnt *energy* density (g^-4) with
// its analytic inverse CDF. Every summand is non-negative, so unlike a finite
// SH reconstruction this runtime path cannot introduce negative ringing.
// The deterministic quadrature is approximate; increase
// EON_MAXENT_QUADRATURE_SAMPLES when the extra cost is acceptable.
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
// The degree-8 directional-albedo polynomial below was fitted in
// temp/lighting/fit_fon_albedo.wls. It is constrained to be exact at grazing
// incidence and exact in the projected-hemisphere average. Its maximum error
// against the analytic FON directional albedo is 5.67e-5 on 100001 test points
// (the published degree-4 EON fit is about 6.02e-4 on the same grid).
// ============================================================================

#ifndef EON_MAXENT_QUADRATURE_SAMPLES
#define EON_MAXENT_QUADRATURE_SAMPLES 16
#endif

const float EON_PI = 3.14159265358979323846;
const float EON_INV_PI = 1.0 / EON_PI;
const float EON_C1 = 0.5 - 2.0 / (3.0 * EON_PI);
const float EON_C2 = 2.0 / 3.0 - 28.0 / (15.0 * EON_PI);

// Constrained approximation of G_F(mu) / pi, expressed in t = 1-mu.
// Horner evaluation matters: the alternating high-order coefficients are not
// suitable for an FP16 polynomial evaluation.
float eon_fon_g_over_pi(float mu) {
    float t = 1.0 - clamp(mu, 0.0, 1.0);
    float p = 2.466759312119113;
    p = -10.456272593424314 + t * p;
    p = 18.308971828815313 + t * p;
    p = -17.165026239689762 + t * p;
    p = 9.377275496109155 + t * p;
    p = -3.2405670832887474 + t * p;
    p = 0.9701930884155742 + t * p;
    p = 0.02645960015447481 + t * p;
    return t * p;
}

// Unit-albedo FON directional albedo E_F(mu). The degree-8 fit is used here
// rather than acos/sqrt-heavy exact G_F.
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

// Missing unit-albedo FON energy, with a clamp that only suppresses the last
// few float32 ulps at grazing incidence. The constrained polynomial is
// non-negative in double precision over mu in [0,1].
float eon_fon_missing(float mu) {
    return max(EON_C1 - eon_fon_g_over_pi(mu), 0.0);
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
// This gives an exact white-furnace result for rho=1 up to the P8 fit error.
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

float eon_maxent_kappa(float momentLength, float omega) {
    if (omega <= 1e-8)
        return 0.0;
    float rho = clamp(momentLength / omega, 0.0, 1.0 - 1e-6);
    return 3.0 * rho
        / (2.0 + sqrt(max(4.0 - 3.0 * rho * rho, 1e-12)));
}

// Exact normalized cosine moment of the four-parameter MaxEnt energy lobe.
// Used to make the r=0 (Lambert) limit analytic rather than quadrature-limited.
float eon_maxent_cosine_moment(float kappa, float axisNoN) {
    if (kappa <= 1e-6)
        return 0.25;
    if (1.0 - kappa <= 1e-5)
        return max(axisNoN, 0.0);

    float k2 = kappa * kappa;
    float u = clamp(axisNoN, -1.0, 1.0);
    float u2 = u * u;
    float d = max(1.0 - k2 + k2 * u2, 1e-12);
    float d32 = d * sqrt(d);
    float symmetric = 3.0 + 6.0 * k2 * (-1.0 + 2.0 * u2)
        + k2 * k2 * (3.0 - 12.0 * u2 + 8.0 * u2 * u2);
    return max((symmetric + 8.0 * kappa * u * d32)
        / (4.0 * (3.0 + k2) * d32), 0.0);
}

float eon_radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u)
        | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u)
        | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u)
        | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u)
        | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

// Sample the normalized angular energy density proportional to
// (1-kappa*dot(axis,wi))^-4. The cube-root inverse is different from the
// square-root inverse used for the g^-3 directional probability density.
vec3 eon_sample_maxent_energy(vec3 axis, float kappa, vec2 xi,
        vec3 tangent, vec3 bitangent) {
    float mu;
    if (kappa <= 1e-5) {
        mu = 2.0 * xi.x - 1.0;
    } else {
        float inverseCubeMin = pow(1.0 + kappa, -3.0);
        float inverseCubeMax = pow(max(1.0 - kappa, 1e-6), -3.0);
        float inverseCube = mix(inverseCubeMin, inverseCubeMax, xi.x);
        mu = (1.0 - pow(max(inverseCube, 1e-20), -1.0 / 3.0))
            / kappa;
        mu = clamp(mu, -1.0, 1.0);
    }

    float phi = 2.0 * EON_PI * xi.y;
    float sinTheta = sqrt(max(1.0 - mu * mu, 0.0));
    return normalize(axis * mu + sinTheta
        * (tangent * cos(phi) + bitangent * sin(phi)));
}

// Convolve a total RGB incident-light energy with EON using the MaxEnt-4
// moments in maxEntY. totalRgb and maxEntY.w must describe the same signal;
// only their ratio/chroma differs.
vec3 eon_project_maxent(vec4 maxEntY, vec3 totalRgb,
        vec3 normal, vec3 wo, float roughness, vec3 rho) {
    float omega = maxEntY.w;
    if (omega <= 1e-8 || dot(normal, wo) <= 0.0)
        return vec3(0.0);

    totalRgb = max(totalRgb, vec3(0.0));
    float momentLength = length(maxEntY.xyz);
    vec3 axis = momentLength > 1e-8
        ? maxEntY.xyz / momentLength : normal;
    float momentRho = clamp(momentLength / omega, 0.0, 1.0);
    float kappa = eon_maxent_kappa(momentLength, omega);
    float r = clamp(roughness, 0.0, 1.0);

    // Lambert has a known analytic MaxEnt convolution. Besides being faster,
    // this branch guarantees bit-stable continuity with maxent_irradiance().
    if (r <= 1e-5) {
        float cosineMoment = eon_maxent_cosine_moment(
            kappa, dot(axis, normal));
        return totalRgb * clamp(rho, vec3(0.0), vec3(1.0))
            * (EON_INV_PI * cosineMoment);
    }

    // Isotropic MaxEnt is uniform over the full sphere. Its convolution is
    // totalRgb/(4*pi) times EON's directional albedo, so no quadrature is
    // required and the white-furnace case remains exact to the P8 fit.
    if (momentRho <= 1e-5) {
        return totalRgb * eon_directional_albedo(
            rho, r, max(dot(normal, wo), 0.0)) / (4.0 * EON_PI);
    }

    // A cone-boundary state is a directional atom. FP16 storage moves a true
    // atom slightly inside the cone, hence the small representable tolerance.
    if (1.0 - momentRho <= 2e-3) {
        float NoI = max(dot(normal, axis), 0.0);
        return totalRgb * eon_brdf(axis, wo, normal, r, rho) * NoI;
    }

    // Align phi=0 with the surface normal projected around the MaxEnt axis.
    // This removes arbitrary world-axis orientation from the deterministic
    // quadrature. The outgoing direction is the secondary fallback for the
    // coaxial axis/normal case.
    vec3 tangent = normal - axis * dot(axis, normal);
    float tangentLength2 = dot(tangent, tangent);
    if (tangentLength2 <= 1e-8) {
        tangent = wo - axis * dot(axis, wo);
        tangentLength2 = dot(tangent, tangent);
    }
    if (tangentLength2 <= 1e-8) {
        tangent = abs(axis.y) < 0.999
            ? cross(vec3(0.0, 1.0, 0.0), axis)
            : cross(vec3(1.0, 0.0, 0.0), axis);
        tangentLength2 = dot(tangent, tangent);
    }
    tangent *= inversesqrt(max(tangentLength2, 1e-12));
    vec3 bitangent = cross(axis, tangent);

    vec3 integral = vec3(0.0);
    for (int i = 0; i < EON_MAXENT_QUADRATURE_SAMPLES; ++i) {
        // Midpoint stratification avoids evaluating either singular CDF end.
        vec2 xi = vec2((float(i) + 0.5)
                / float(EON_MAXENT_QUADRATURE_SAMPLES),
            eon_radical_inverse(uint(i)));
        vec3 wi = eon_sample_maxent_energy(axis, kappa, xi,
            tangent, bitangent);
        float NoI = max(dot(normal, wi), 0.0);
        integral += eon_brdf(wi, wo, normal, r, rho) * NoI;
    }

    return max(totalRgb * integral
        / float(EON_MAXENT_QUADRATURE_SAMPLES), vec3(0.0));
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

#ifndef DIRT_RT_LIB_RT_RAYTRACE_BSDF_GLSL
#define DIRT_RT_LIB_RT_RAYTRACE_BSDF_GLSL

// Surface BSDFs. wo/wi point away from the vertex; PDFs use solid angle.

float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1e-30);
}

float sunSolidAngle() {
    return max(2.0 * PI * (1.0 - cosD_S), 1e-12);
}

float sunDirectionPdf(vec3 wi, vec3 lightDir) {
    vec3 sunDirection = -normalize(lightDir);
    return dot(wi, sunDirection) >= cosD_S
    ? 1.0 / sunSolidAngle() : 0.0;
}

bool isDeltaSpecular(float roughness) {
    return roughness <= SPECULAR_DELTA_ALPHA;
}

// ===========================================================================
// Half-Vector Computation
// ===========================================================================

HalfVector computeHalfVector(vec3 wo, vec3 wi) {
    vec3 Hsum = wo + wi;
    float Hlen2 = dot(Hsum, Hsum);
    float valid = float(Hlen2 > 1e-12);
    HalfVector hv;
    hv.H = Hsum * inversesqrt(max(Hlen2, 1e-12)) * valid;
    hv.valid = valid > 0.5;
    return hv;
}

// ===========================================================================
// Disney Cook-Torrance BRDF Evaluation
// ===========================================================================

// Evaluate GGX microfacet BRDF: f(wo, wi) × NoL, and VNDF sampling PDF.
// Returns false if any degenerate angle; caller treats as bsdf_weight = 0.
// The caller applies MIS weighting externally (pdfMix or pdfNDF×P_spec).
vec3 evaluateSurfaceFresnel(vec3 wo, vec3 H, vec3 Cs, float Sx,
    float transmissionSelector, float etaRatio) {
    return surfaceFresnel(wo, H, Cs, Sx, transmissionSelector, etaRatio);
}

bool evaluateSpecularBRDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 Cs, float Sx,
    float transmissionSelector, float etaRatio, float roughness,
    out vec3 fSpecTimesNoL, out float pdfNDF, out vec3 qLiResponse
) {
    qLiResponse = vec3(0.0);
    float NoV = dot(macroNormal, wo);
    float NoL = dot(macroNormal, wi);

    if (NoV <= 1e-5 || NoL <= 1e-5) {
        fSpecTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    HalfVector hv = computeHalfVector(wo, wi);
    float NoH = dot(macroNormal, hv.H);
    float VoH = abs(dot(wo, hv.H));

    bool valid = hv.valid && NoH > 1e-5 && VoH > 1e-5;

    if (valid) {
        float rough = max(roughness, 1e-4);
        float D = GGX_D(NoH, rough);
        float smithV = GGX_smithRoot(NoV, rough);
        float smithL = GGX_smithRoot(NoL, rough);
        vec3 Fh = evaluateSurfaceFresnel(
                wo, hv.H, Cs, Sx, transmissionSelector, etaRatio);

        // Common denominators for f*NoL, q_vndf and their ratio. In
        // particular, do not recompute H, D and Smith G1 through a PDF helper.
        float viewDenominator = NoV + smithV;
        float maskingDenominator = NoL * smithV + NoV * smithL;
        qLiResponse = Fh * (NoL * viewDenominator / maskingDenominator);
        pdfNDF = D / (2.0 * viewDenominator);
        fSpecTimesNoL = Fh * (D * NoL / (2.0 * maskingDenominator));
    } else {
        fSpecTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
    }

    return valid;
}

// NEE needs f*NoL and q; continuation additionally consumes the analytic
// response above. The compiler drops the unused response for this overload.
bool evaluateSpecularBRDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 Cs, float Sx,
    float transmissionSelector, float etaRatio, float roughness,
    out vec3 fSpecTimesNoL, out float pdfNDF
) {
    vec3 response;
    return evaluateSpecularBRDF(wo, wi, macroNormal, Cs, Sx,
        transmissionSelector, etaRatio, roughness,
        fSpecTimesNoL, pdfNDF, response);
}

// Burley's Disney diffuse term. surf.R.x stores GGX alpha, while the Disney
// fit is parameterized by perceptual roughness, hence the square root.
float evaluateDisneyDiffuseFactor(
    vec3 wo, vec3 wi, vec3 macroNormal, float roughness
) {
    float NoV = clamp(dot(macroNormal, wo), 0.0, 1.0);
    float NoL = clamp(dot(macroNormal, wi), 0.0, 1.0);
    float onePlusVoL = 1.0 + clamp(dot(wo, wi), -1.0, 1.0);
    if (onePlusVoL <= 5e-13 || NoV <= 0.0 || NoL <= 0.0) return 0.0;
    float perceptualRoughness = sqrt(clamp(roughness, 0.0, 1.0));
    // For unit wo/wi, 2*(wi.H)^2 = 1 + wo.wi; H need not be normalized.
    float Fd90 = 0.5 + perceptualRoughness * onePlusVoL;
    float lightScatter = mix(1.0, Fd90, pow5(1.0 - NoL));
    float viewScatter = mix(1.0, Fd90, pow5(1.0 - NoV));
    return max(lightScatter * viewScatter, 0.0);
}

bool evaluateDisneyDiffuseBRDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 geometryNormal,
    vec3 diffuseColor, float roughness,
    out vec3 fDiffuseTimesNoL, out float pdfCosine
) {
    float NoL = dot(macroNormal, wi);
    float NoV = dot(macroNormal, wo);
    if (NoL <= 1e-6 || NoV <= 1e-6
            || dot(geometryNormal, wi) <= 0.0) {
        fDiffuseTimesNoL = vec3(0.0);
        pdfCosine = 0.0;
        return false;
    }

    float Fd = evaluateDisneyDiffuseFactor(
            wo, wi, macroNormal, roughness);
    fDiffuseTimesNoL = diffuseColor * (Fd * NoL / PI);
    pdfCosine = NoL / PI;
    return Fd > 0.0;
}

// Unified diffuse evaluator for secondary vertices. The first vertex is
// handled separately because its BRDF is deferred to the MaxEnt projection in
// composite_lighting.glsl.
bool evaluateSurfaceDiffuseBRDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 geometryNormal,
    vec3 diffuseColor, float roughness,
    out vec3 fDiffuseTimesNoL, out float pdfDiffuse
) {
    #if EON_ENABLED
    float NoL = dot(macroNormal, wi);
    float NoV = dot(macroNormal, wo);
    if (NoL <= 1e-6 || NoV <= 1e-6
            || dot(geometryNormal, wi) <= 0.0) {
        fDiffuseTimesNoL = vec3(0.0);
        pdfDiffuse = 0.0;
        return false;
    }

    // Dirt RT stores GGX alpha; EON consumes linear/perceptual roughness.
    float eonRoughness = sqrt(clamp(roughness, 0.0, 1.0));
    fDiffuseTimesNoL = eon_brdf(
            wi, wo, macroNormal, eonRoughness, diffuseColor) * NoL;
    pdfDiffuse = eon_direction_pdf(
        wo, wi, macroNormal, eonRoughness);
    return any(greaterThan(fDiffuseTimesNoL, vec3(0.0)));
    #else
    return evaluateDisneyDiffuseBRDF(
        wo, wi, macroNormal, geometryNormal, diffuseColor, roughness,
        fDiffuseTimesNoL, pdfDiffuse);
    #endif
}

// VNDF-sampled rough transmission. Both f_t*abs(NoL) and its directional
// PDF contain D, VoH, abs(LiH), and the refraction Jacobian. Cancel them before
// evaluation: weight = color * (1-F) * etaRatio^2 * G2/G1(wo).
// This interface intentionally returns f/p; transmission has no sun-NEE peer.
bool sampleTransmissionWeight(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 microNormal,
    vec3 transmissionColor, float etaRatio, float roughness,
    out vec3 weight
) {
    weight = vec3(0.0);
    float NoV = dot(macroNormal, wo);
    float NoL = -dot(macroNormal, wi);
    vec3 H = dot(microNormal, macroNormal) >= 0.0
        ? microNormal : -microNormal;
    if (NoV <= 1e-5 || NoL <= 1e-5 || etaRatio <= 1e-5
            || dot(macroNormal, H) <= 1e-5 || dot(wo, H) <= 1e-5
            || dot(wi, H) >= -1e-5)
        return false;
    // Equal indices remove the interface: no microfacet masking or bending.
    if (etaRatio == 1.0) {
        weight = transmissionColor;
        return true;
    }
    float T = 1.0 - clamp(fresnel(wo, H, etaRatio), 0.0, 1.0);
    weight = transmissionColor * (T * etaRatio * etaRatio
        * GGX_G2(NoV, NoL, max(roughness, 1e-4)));
    return true;
}

#endif // DIRT_RT_LIB_RT_RAYTRACE_BSDF_GLSL

#ifndef DIRT_RT_RAYTRACE_BOUNCES_GLSL
#define DIRT_RT_RAYTRACE_BOUNCES_GLSL

// First-bounce and secondary-bounce sampling policies.

// ===========================================================================
// First-Bounce Handlers (compile-time dispatched via #if defined)
// ===========================================================================

void handleFirstBounce_Reflection(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal,
    inout vec3 microNormal, material surf, float etaRatio,
    GuideInfo guide, vec2 xi,
    out vec3 bsdf_weight, out vec3 next_rd,
    out vec3 qLiResponse, out float sampledStrategyPdf,
    out bool sampledDelta
) {
    vec3 wo = -rd_i;
    bsdf_weight = vec3(0.0);
    qLiResponse = vec3(0.0);
    next_rd = rd_i;
    sampledStrategyPdf = 0.0;
    sampledDelta = isDeltaSpecular(surf.R.x);

    if (sampledDelta) {
        next_rd = reflect(rd_i, macroNormal);
        // Geometric-hemisphere repair.  Strong mapped
        // normals must not send a visible reflection through the surface.
        if (dot(next_rd, geometryNormal) < 0.0)
            next_rd = reflect(next_rd, geometryNormal);
        bsdf_weight = evaluateSurfaceFresnel(wo, macroNormal, surf.Cs,
            surf.S.x, surf.S.y, etaRatio);
        qLiResponse = bsdf_weight;
        return;
    }

    bool useGuide = getRandom() < guide.prob;
    if (useGuide) {
        next_rd = sample_maxent_guiding(guide.axis, guide.kappa, xi);
        HalfVector sampledHalf = computeHalfVector(wo, next_rd);
        microNormal = sampledHalf.valid ? sampledHalf.H : macroNormal;
    } else {
        // Only pay for VNDF sampling when that proposal is actually chosen.
        microNormal = GGXVNDFNormal(macroNormal, wo, surf.R.x, xi);
        next_rd = reflect(rd_i, microNormal);
    }
    // The MaxEnt proposal is spherical and may select the lower geometric
    // hemisphere. Rejection is unbiased because the target BRDF is zero there.
    if (dot(next_rd, geometryNormal) <= 0.0) {
        bsdf_weight = vec3(0.0);
        return;
    }

    vec3 wi = next_rd;
    vec3 fSpecTimesNoL_val;
    float pdfNDF;
    if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
            surf.S.y, etaRatio, surf.R.x,
            fSpecTimesNoL_val, pdfNDF, qLiResponse)) {
        sampledStrategyPdf = specularGuideMixturePdf(
            guide, pdfNDF, wi);
        // q remains needed for the mixture/MIS. Cancel f/q analytically;
        // an unguided VNDF draw then has exactly qLiResponse as throughput.
        bsdf_weight = sampledStrategyPdf > 0.0
            ? qLiResponse * (pdfNDF / sampledStrategyPdf) : vec3(0.0);
    } else {
        bsdf_weight = vec3(0.0);
    }
}

void handleFirstBounce_Diffuse(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal,
    material surf, LobeProbs lobes, GuideInfo guide, vec2 xi,
    out vec3 bsdf_weight, out vec3 next_rd, out float sampledStrategyPdf
) {
    bsdf_weight = vec3(0.0);
    next_rd = rd_i;
    sampledStrategyPdf = 0.0;
    // Metals and fully transmissive interfaces cannot consume the dedicated
    // diffuse sample; avoid the guide-buffer lookup and continuation ray.
    if (lobes.P_diff <= 1e-8) return;

    float guideWeight;
    sampleDiffuseWithGuide(
        geometryNormal, macroNormal, ro_o, guide, xi, next_rd, guideWeight,
        sampledStrategyPdf);
    #if EON_ENABLED
    // Store a pure incident-radiance estimator in MaxEnt. The complete EON
    // BRDF, NoL and material albedo are applied after denoising in composite.
    // guideWeight=(1/2pi)/q, hence 2pi*guideWeight=1/q.
    bsdf_weight = vec3(2.0 * PI * guideWeight);
    #else
    // Legacy path: MaxEnt supplies NoL and albedo in composite, while the
    // Disney Fd/pi factor remains baked into the transported sample.
    float Fd = evaluateDisneyDiffuseFactor(
            -rd_i, next_rd, macroNormal, surf.R.x);
    bsdf_weight = vec3(2.0 * guideWeight * Fd);
    #endif
}

// ===========================================================================
// Secondary Bounce Handler (stochastic mixture)
// ===========================================================================

void handleSecondaryBounce(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal,
    material surf, LobeProbs lobes,
    out vec3 bsdf_weight, out vec3 next_rd, out int lobeType,
    out bool sampledSpecularLobe, out float sampledStrategyPdf,
    out bool sampledDeltaLobe, out bool neeCompatible,
    inout bool inside_state
) {
    float rnd_lobe = getRandom();
    sampledSpecularLobe = false;
    sampledStrategyPdf = 0.0;
    sampledDeltaLobe = false;
    neeCompatible = false;
    float surfaceIor = transportIorFromMaterial(surf);
    float n_i = inside_state ? surfaceIor : 1.0;
    float n_o = inside_state ? 1.0 : surfaceIor;
    float rs = n_i / n_o;

    // Lobe choice does not depend on H. Diffuse paths need no GGX sample.
    vec3 microNormal = macroNormal;
    if (rnd_lobe < lobes.P_spec + lobes.P_refr
            && !isDeltaSpecular(surf.R.x))
        microNormal = GGXVNDFNormal(macroNormal, -rd_i, surf.R.x, ro_o);

    if (rnd_lobe < lobes.P_spec) {
        // Reflection
        sampledSpecularLobe = true;
        lobeType = REFLECTION;
        sampledDeltaLobe = isDeltaSpecular(surf.R.x);
        neeCompatible = true;
        next_rd = reflect(rd_i,
                sampledDeltaLobe ? macroNormal : microNormal);
        if (dot(next_rd, geometryNormal) < 0.0) {
            if (sampledDeltaLobe)
                next_rd = reflect(next_rd, geometryNormal);
            else {
                bsdf_weight = vec3(0.0);
                return;
            }
        }
        if (dot(next_rd, macroNormal) > 0.0) {
            vec3 wo = -rd_i;
            vec3 wi = next_rd;
            if (sampledDeltaLobe) {
                vec3 F = evaluateSurfaceFresnel(
                        wo, macroNormal, surf.Cs, surf.S.x, surf.S.y, rs);
                bsdf_weight = F / max(lobes.P_spec, 1e-8);
                return;
            }
            vec3 fSpecTimesNoL_val, qLiResponse;
            float pdfNDF;
            if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
                    surf.S.y, rs, surf.R.x,
                    fSpecTimesNoL_val, pdfNDF, qLiResponse)) {
                bsdf_weight = qLiResponse / lobes.P_spec;
                sampledStrategyPdf = pdfNDF * lobes.P_spec;
            } else {
                bsdf_weight = vec3(0.0);
            }
        } else {
            bsdf_weight = vec3(0.0);
        }
    } else if (rnd_lobe < lobes.P_spec + lobes.P_refr) {
        // Refraction
        lobeType = REFRACTION;
        vec3 refract_dir = refract(rd_i, microNormal, rs);
        // A shading-normal transmission must cross the geometric interface.
        // Reject invalid draws without changing the selected lobe: relabeling
        // them as reflection could feed a zero BTDF sample into the cache's
        // integrated reflection fallback with the wrong selection probability.
        if (dot(refract_dir, refract_dir) > 0.0
                && dot(refract_dir, geometryNormal) < 0.0) {
            next_rd = refract_dir;
            vec3 transmissionWeight;
            vec3 transmissionColor = evaluateTransmissionAlbedo(surf)
                * clamp(surf.S.y, 0.0, 1.0);
            bool validTransmission;
            if (isDeltaSpecular(surf.R.x)) {
                float F = clamp(
                        fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
                bsdf_weight = transmissionColor * (1.0 - F) * (rs * rs)
                        / max(lobes.P_refr, 1e-8);
                validTransmission = true;
                sampledDeltaLobe = true;
            } else {
                validTransmission = sampleTransmissionWeight(
                        -rd_i, next_rd, macroNormal, microNormal,
                        transmissionColor, rs, surf.R.x,
                        transmissionWeight);
                bsdf_weight = validTransmission
                    ? transmissionWeight / lobes.P_refr : vec3(0.0);
            }
            inside_state = validTransmission ? !inside_state : inside_state;
        } else {
            next_rd = rd_i;
            bsdf_weight = vec3(0.0);
        }
    } else {
        // Diffuse
        lobeType = DIFFUSION;
        neeCompatible = true;
        vec3 fDiffuseTimesNoL;
        float pdfDiffuse;
        vec3 diffuseColor = surf.Cd
                * (1.0 - clamp(surf.S.y, 0.0, 1.0));
        bool validDiffuse;
        #if EON_ENABLED
        // First-bounce diffuse uses the MaxEnt proposal. Only continuation
        // vertices use EON's view-conditioned CLTC importance sampler.
        float eonRoughness = sqrt(clamp(surf.R.x, 0.0, 1.0));
        vec4 eonSample = eon_sample_direction(
            -rd_i, macroNormal, eonRoughness,
            vec2(getRandom(), getRandom()));
        next_rd = eonSample.xyz;
        float NoL = dot(macroNormal, next_rd);
        validDiffuse = NoL > 1e-6
            && dot(geometryNormal, next_rd) > 0.0
            && eonSample.w > 1e-8;
        fDiffuseTimesNoL = validDiffuse
            ? eon_brdf(next_rd, -rd_i, macroNormal,
                eonRoughness, diffuseColor) * NoL
            : vec3(0.0);
        pdfDiffuse = validDiffuse ? eonSample.w : 0.0;
        #else
        next_rd = DiffuseNormal(macroNormal, ro_o);
        validDiffuse = evaluateSurfaceDiffuseBRDF(
                -rd_i, next_rd, macroNormal, geometryNormal,
                diffuseColor, surf.R.x,
                fDiffuseTimesNoL, pdfDiffuse);
        #endif
        sampledStrategyPdf = lobes.P_diff * pdfDiffuse;
        bsdf_weight = validDiffuse && sampledStrategyPdf > 1e-8
            ? fDiffuseTimesNoL / sampledStrategyPdf : vec3(0.0);
    }
}


#endif // DIRT_RT_RAYTRACE_BOUNCES_GLSL

#ifndef DIRT_RT_RAYTRACE_BOUNCES_GLSL
#define DIRT_RT_RAYTRACE_BOUNCES_GLSL

// First-bounce and secondary-bounce sampling policies.

// ===========================================================================
// First-Bounce Handlers (compile-time dispatched via #if defined)
// ===========================================================================

void handleFirstBounce_Reflection(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, LobeProbs lobes, float etaRatio,
    out vec3 bsdf_weight, out vec3 next_rd,
    out float sampledStrategyPdf, out bool sampledDelta
) {
    vec3 wo = -rd_i;
    bsdf_weight = vec3(0.0);
    next_rd = rd_i;
    sampledStrategyPdf = 0.0;
    sampledDelta = isDeltaSpecular(surf.R.x);
    // Dedicated first-lobe passes are full-screen. Do not start a
    // continuation for a material whose reflection lobe has zero support.
    if (lobes.P_spec <= 1e-8) return;

    if (sampledDelta) {
        next_rd = reflect(rd_i, macroNormal);
        // Geometric-hemisphere repair.  Strong mapped
        // normals must not send a visible reflection through the surface.
        if (dot(next_rd, geometryNormal) < 0.0)
            next_rd = reflect(next_rd, geometryNormal);
        bsdf_weight = evaluateSurfaceFresnel(wo, macroNormal, surf.Cs,
            surf.S.x, surf.S.y, etaRatio);
        return;
    }

    // Keep the continuation strategy equal to the VNDF density returned by
    // evaluateSpecularBRDF. Mixing a separate guide here would require a
    // mixture PDF for both throughput and NEE MIS.
    next_rd = reflect(rd_i, microNormal);
    // Folding an invalid VNDF reflection about the geometry normal changes
    // the sampling density and biases f/pdf. Reject it instead.
    if (dot(next_rd, geometryNormal) <= 0.0) {
        bsdf_weight = vec3(0.0);
        return;
    }

    vec3 wi = next_rd;
    vec3 fSpecTimesNoL_val;
    float pdfNDF;
    if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
            surf.S.y, etaRatio, surf.R.x,
            fSpecTimesNoL_val, pdfNDF)) {
        sampledStrategyPdf = pdfNDF;
        bsdf_weight = (pdfNDF > 1e-8)
            ? (fSpecTimesNoL_val / pdfNDF) : vec3(0.0);
    } else {
        bsdf_weight = vec3(0.0);
    }
}

void handleFirstBounce_Refraction(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, LobeProbs lobes, float rs,
    out vec3 bsdf_weight, out vec3 next_rd, inout bool inside_state,
    out PSRResult psr, bool wasInside, int baseDepth
) {
    bsdf_weight = vec3(0.0);
    next_rd = rd_i;
    psr.virtualDist = 0.0;
    psr.pathRoughness = 0.0;
    psr.refrDir = rd_i;
    // Most primary surfaces are opaque. This guard must precede refract() and,
    // especially, tracePSRChain(), which can issue four additional RT rays.
    if (lobes.P_refr <= 1e-8) return;

    vec3 refract_dir = refract(rd_i, microNormal, rs);
    vec3 psr_refract_dir = refract(rd_i, geometryNormal, rs);

    if (dot(refract_dir, refract_dir) > 0.0) {
        next_rd = refract_dir;
        bool was_inverse_0 = inside_state;
        inside_state = !inside_state;

        bool psrEnabled = surf.R.x < PSR_ROUGHNESS_THRESHOLD;
        if (psrEnabled) {
            vec3 chain_rd = dot(psr_refract_dir, psr_refract_dir) > 0.0 ? psr_refract_dir : refract_dir;
            float savedConeWidth = rtCurrentConeWidth;
            psr = tracePSRChain(ro_o, chain_rd, geometryNormal, surf.R.x, was_inverse_0, baseDepth);
            rtCurrentConeWidth = savedConeWidth;
        } else {
            psr.virtualDist = 0.0;
            psr.pathRoughness = surf.R.x;
            psr.refrDir = dot(psr_refract_dir, psr_refract_dir) > 0.0 ? psr_refract_dir : refract_dir;
        }

        vec3 fTransmissionTimesNoL;
        float pdfTransmission;
        vec3 transmissionColor = surf.Cd * clamp(surf.S.y, 0.0, 1.0);
        if (isDeltaSpecular(surf.R.x)) {
            float F = clamp(fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
            // Radiance transport through a delta dielectric carries eta_i^2 /
            // eta_t^2. The inverse factor at the exit interface cancels it.
            bsdf_weight = transmissionColor * (1.0 - F) * (rs * rs);
        } else {
            bool validTransmission = evaluateTransmissionBSDF(
                    -rd_i, next_rd, macroNormal, microNormal,
                    transmissionColor, rs, surf.R.x,
                    fTransmissionTimesNoL, pdfTransmission);
            bsdf_weight = validTransmission && pdfTransmission > 1e-8
                ? fTransmissionTimesNoL / pdfTransmission : vec3(0.0);
        }
    } else {
        // TIR
        next_rd = reflect(rd_i, microNormal);
        psr.virtualDist = 0.0;
        psr.pathRoughness = surf.R.x;
        psr.refrDir = next_rd;
        bsdf_weight = vec3(0.0);
    }
}

void handleFirstBounce_Diffuse(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal,
    material surf, LobeProbs lobes, vec2 xi,
    out vec3 bsdf_weight, out vec3 next_rd, out float sampledStrategyPdf
) {
    bsdf_weight = vec3(0.0);
    next_rd = rd_i;
    sampledStrategyPdf = 0.0;
    // Metals and fully transmissive interfaces cannot consume the dedicated
    // diffuse sample; avoid the guide-buffer lookup and continuation ray.
    if (lobes.P_diff <= 1e-8) return;

    GuideInfo guide = computeMaxEntGuide(ro_o, PATH_GUIDING_STRENGTH);
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
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
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
    float n_i = inside_state ? REFRACTIVE_INDEX : 1.0;
    float n_o = inside_state ? 1.0 : REFRACTIVE_INDEX;
    float rs = n_i / n_o;

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
            vec3 fSpecTimesNoL_val;
            float pdfNDF;
            if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
                    surf.S.y, rs, surf.R.x,
                    fSpecTimesNoL_val, pdfNDF)) {
                bsdf_weight = (pdfNDF > 1e-8)
                    ? (fSpecTimesNoL_val / (pdfNDF * lobes.P_spec)) : vec3(0.0);
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
        if (dot(refract_dir, refract_dir) > 0.0) {
            next_rd = refract_dir;
            vec3 fTransmissionTimesNoL;
            float pdfTransmission;
            vec3 transmissionColor = surf.Cd * clamp(surf.S.y, 0.0, 1.0);
            bool validTransmission;
            if (isDeltaSpecular(surf.R.x)) {
                float F = clamp(
                        fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
                bsdf_weight = transmissionColor * (1.0 - F) * (rs * rs)
                        / max(lobes.P_refr, 1e-8);
                validTransmission = true;
                sampledDeltaLobe = true;
            } else {
                validTransmission = evaluateTransmissionBSDF(
                        -rd_i, next_rd, macroNormal, microNormal,
                        transmissionColor, rs, surf.R.x,
                        fTransmissionTimesNoL, pdfTransmission);
                bsdf_weight = validTransmission && pdfTransmission > 1e-8
                    ? fTransmissionTimesNoL
                        / (pdfTransmission * max(lobes.P_refr, 1e-8)) : vec3(0.0);
            }
            inside_state = validTransmission ? !inside_state : inside_state;
        } else {
            next_rd = reflect(rd_i, microNormal);
            lobeType = REFLECTION;
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

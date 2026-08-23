#ifndef DIRT_RT_RAYTRACE_TRANSPORT_GLSL
#define DIRT_RT_RAYTRACE_TRANSPORT_GLSL

// Transport data, BRDF/BTDF evaluation, guiding and PSR.

// ===========================================================================
// Data Structures
// ===========================================================================

struct HalfVector {
    vec3 H;
    bool valid;
};

struct LobeProbs {
    float P_spec, P_refr, P_diff;
    vec3 specWeight, refrWeight, diffWeight;
    float F;
};

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

struct MediumResult {
    vec3 absorption;
    vec3 emission;
};

struct GuideInfo {
    vec3 axis;
    float kappa;
    float prob;
    bool valid;
};

struct PSRResult {
    float virtualDist;
    float pathRoughness;
    vec3 refrDir;
    vec3 endpoint;
    vec3 endpointGeometryNormal;
    vec3 endpointMacroNormal;
    vec3 endpointDiffuseAlbedo;
    float endpointRoughness;
    vec3 endpointLight;
    vec3 transmittance;
    bool endpointValid;
    bool environment;
};

struct FirstBounceData {
    vec3 p, macro_n, geometry_n, micro_n, rd_o, rd_i, refr_dir;
    vec3 specularAlbedo, diffuseAlbedo, transmissionAlbedo;
    vec3 emission_val, light_surf, absorption;
    float t, roughness, n_i, n_o, t2_ior_adjusted, pathRoughness;
    float reflectionHitDistance;
    vec3 surfaceMotion;
    float motionValid;
    int type, materialID;
};

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
    vec3 schlickF = reflectanceColor(Cs, abs(dot(wo, H))).rgb * Sx;
    float dielectricF = fresnel(wo, H, etaRatio);
    return mix(schlickF, vec3(dielectricF),
        clamp(transmissionSelector, 0.0, 1.0));
}

bool evaluateSpecularBRDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 Cs, float Sx,
    float transmissionSelector, float etaRatio, float roughness,
    out vec3 fSpecTimesNoL, out float pdfNDF
) {
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
        float G2 = GGX_G2_standard(NoV, NoL, rough);
        vec3 Fh = evaluateSurfaceFresnel(
                wo, hv.H, Cs, Sx, transmissionSelector, etaRatio);

        fSpecTimesNoL = Fh * D * G2 * NoL / max(4.0 * NoV * NoL, 1e-8);
        pdfNDF = GGX_vndf_pdf(wo, wi, macroNormal, rough);
    } else {
        fSpecTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
    }

    return valid;
}

// Burley's Disney diffuse term. surf.R.x stores GGX alpha, while the Disney
// fit is parameterized by perceptual roughness, hence the square root.
float evaluateDisneyDiffuseFactor(
    vec3 wo, vec3 wi, vec3 macroNormal, float roughness
) {
    float NoV = clamp(dot(macroNormal, wo), 0.0, 1.0);
    float NoL = clamp(dot(macroNormal, wi), 0.0, 1.0);
    HalfVector hv = computeHalfVector(wo, wi);
    if (!hv.valid || NoV <= 0.0 || NoL <= 0.0) return 0.0;

    float LoH = clamp(dot(wi, hv.H), 0.0, 1.0);
    float perceptualRoughness = sqrt(clamp(roughness, 0.0, 1.0));
    float Fd90 = 0.5 + 2.0 * perceptualRoughness * LoH * LoH;
    float lightScatter = mix(1.0, Fd90, pow(1.0 - NoL, 5.0));
    float viewScatter = mix(1.0, Fd90, pow(1.0 - NoV, 5.0));
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

// Walter/PBRT rough-dielectric BTDF evaluated for a VNDF-sampled microfacet.
// etaRatio is eta_i / eta_t, matching GLSL refract().  The returned value is
// f_t * abs(NoL), and pdfNDF is the corresponding solid-angle PDF of wi.
bool evaluateTransmissionBSDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 microNormal,
    vec3 transmissionColor, float etaRatio, float roughness,
    out vec3 fTransmissionTimesNoL, out float pdfNDF
) {
    float NoV = dot(macroNormal, wo);
    float NoL = -dot(macroNormal, wi);
    if (NoV <= 1e-5 || NoL <= 1e-5 || etaRatio <= 1e-5) {
        fTransmissionTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    vec3 H = dot(microNormal, macroNormal) >= 0.0
        ? microNormal : -microNormal;
    float NoH = dot(macroNormal, H);
    float VoH = dot(wo, H);
    float LiH = dot(wi, H);
    if (NoH <= 1e-5 || VoH <= 1e-5 || LiH >= -1e-5) {
        fTransmissionTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    float rough = max(roughness, 1e-4);
    float etaP = 1.0 / etaRatio;
    float denom = LiH + VoH / etaP;
    float denom2 = denom * denom;
    if (denom2 <= 1e-12) {
        fTransmissionTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    float D = GGX_D(NoH, rough);
    float G2 = GGX_G2_standard(NoV, NoL, rough);
    float T = 1.0 - clamp(fresnel(wo, H, etaRatio), 0.0, 1.0);
    float btdf = T * D * G2
            * abs(LiH * VoH / max(NoL * NoV * denom2, 1e-20));
    // Radiance transport across media is non-symmetric by eta^2.
    btdf /= max(etaP * etaP, 1e-8);

    float dH_dWi = abs(LiH) / denom2;
    pdfNDF = GGX_vndf_half_pdf(wo, H, macroNormal, rough) * dH_dWi;
    fTransmissionTimesNoL = transmissionColor * btdf * NoL;
    bool finiteResult = !isnan(pdfNDF) && !isinf(pdfNDF)
            && !any(isnan(fTransmissionTimesNoL))
            && !any(isinf(fTransmissionTimesNoL));
    if (!finiteResult) {
        pdfNDF = 0.0;
        fTransmissionTimesNoL = vec3(0.0);
    }
    return finiteResult && pdfNDF > 0.0;
}

// ===========================================================================
// BSDF Lobe Probabilities
// ===========================================================================

LobeProbs computeLobeProbs(material surf, vec3 rd_i, vec3 macroNormal, float rs) {
    LobeProbs p;
    // Lobe selection must be independent of the subsequently sampled GGX
    // half-vector. This gives the continuation strategy a well-defined PDF
    // that can be evaluated for an arbitrary NEE direction.
    p.F = clamp(fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
    vec4 rC = reflectanceColor(surf.Cs, dot(rd_i, macroNormal));

    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    float diffuseSelector = 1.0 - transmissionSelector;
    float specularSelector = clamp(surf.S.x, 0.0, 1.0);

    vec3 interfaceF = mix(
            rC.rgb * specularSelector,
            vec3(p.F),
            transmissionSelector);
    // Macro-normal Fresnel is not the support of a rough microfacet lobe.  In
    // particular, F0 == 0 and NoV ~= 1 make it exactly zero even though tilted
    // GGX facets have VoH < 1 and therefore non-zero Schlick Fresnel.
    //
    // Build a half-vector-independent proposal importance from a fixed
    // cosine-weighted angular kernel.  Its Schlick moment is analytic:
    //   E[(1 - mu)^5] = integral_0^1 2 mu (1 - mu)^5 dmu = 1 / 21.
    // GGX alpha blends from the macro-normal value to that broad-kernel mean.
    // This value only selects a lobe; the sampled BRDF and its PDF below still
    // determine the Monte Carlo weight.
    float etaDenominator = max(abs(1.0 + rs), 1e-8);
    float dielectricF0 = (1.0 - rs) / etaDenominator;
    dielectricF0 *= dielectricF0;
    float dielectricF90 = (rs == 1.0) ? 0.0 : 1.0;
    vec3 interfaceF0 = mix(
            clamp(surf.Cs, vec3(0.0), vec3(1.0)) * specularSelector,
            vec3(dielectricF0), transmissionSelector);
    vec3 interfaceF90 = mix(
            vec3(specularSelector), vec3(dielectricF90),
            transmissionSelector);
    const float SCHLICK_COSINE_KERNEL_MOMENT = 1.0 / 21.0;
    vec3 cosineKernelMeanF = interfaceF0
            + (interfaceF90 - interfaceF0) * SCHLICK_COSINE_KERNEL_MOMENT;
    float proposalWidth = isDeltaSpecular(surf.R.x)
            ? 0.0 : clamp(surf.R.x, 0.0, 1.0);
    vec3 roughInterfaceImportance = max(interfaceF,
            mix(interfaceF, cosineKernelMeanF, proposalWidth));
    // Probabilities choose a non-zero transport lobe; they are not Fresnel
    // absorption probabilities. In particular, a metal has Cd == 0 and must
    // never spend samples on a zero-valued diffuse branch.
    float interfaceEnergy = clamp(luma(roughInterfaceImportance), 0.0, 1.0);
    float diffuseBaseEnergy = max(luma(max(surf.Cd, vec3(0.0))), 0.0);
    vec3 transmissionColor = evaluateTransmissionAlbedo(surf);
    float transmissionEnergy = max(luma(max(transmissionColor,
        vec3(0.0))), 0.0);
    float remainingEnergy = max(1.0 - interfaceEnergy, 0.0);
    float specImportance = interfaceEnergy;
    float refrImportance = remainingEnergy * transmissionSelector
        * transmissionEnergy;
    float diffImportance = remainingEnergy * diffuseSelector
        * diffuseBaseEnergy;
    float importanceSum = specImportance + refrImportance + diffImportance;

    if (importanceSum > 1e-8) {
        p.P_spec = specImportance / importanceSum;
        p.P_refr = refrImportance / importanceSum;
        p.P_diff = diffImportance / importanceSum;
    } else {
        p.P_spec = 0.0;
        p.P_refr = 0.0;
        p.P_diff = 0.0;
    }

    vec3 nonSpecColor = surf.Cd;
    p.specWeight = interfaceF / max(p.P_spec, 1e-5);
    p.refrWeight = transmissionColor * transmissionSelector
        / max(p.P_refr, 1e-5);
    p.diffWeight = nonSpecColor * diffuseSelector / max(p.P_diff, 1e-5);

    return p;
}

// ===========================================================================
// Volumetric Medium
// ===========================================================================

MediumResult evalMedium(float t, vec3 rd_i, float ro_i_y, bool inside,
    int mediumBlockID, vec3 mediumTint, float mediumExtinctionWeight,
    vec4 fogColor, vec3 globalEmission) {
    MediumResult m;
    if (inside) {
        if (mediumBlockID == BLOCK_WATER
                || mediumBlockID == BLOCK_GLASS
                || mediumBlockID == BLOCK_ICE) {
            m.absorption = mediumVolumeTransmittance(t, mediumBlockID,
                mediumTint, mediumExtinctionWeight);
            m.emission = vec3(0.0);
        } else {
            // Non-block camera media (currently lava) retain their emissive
            // fog model.
            m.absorption = exp2(-t * fogColor.yzw * LOG2_E);
            m.emission = (1.0 - m.absorption)
                / (fogColor.yzw + 1e-5) * globalEmission;
        }
    } else {
        m.absorption = exp2(-max(b_Q * (b_P.x - ro_i_y) * t
                        - 0.5 * b_Q * t * t * rd_i.y, 0.0) * LOG2_E);
        m.emission = vec3(0.0);
    }
    return m;
}

// ===========================================================================
// MaxEnt Path Guiding
// ===========================================================================

GuideInfo computeMaxEntGuide(vec3 ro_o, float strengthMultiplier) {
    GuideInfo g;
    g.axis = vec3(0.0, 1.0, 0.0);
    g.kappa = 0.0;
    g.prob = 0.0;
    g.valid = false;

    vec2 prev_coord = reproject(ro_o).xy;
    bool validPrev = all(greaterThanEqual(prev_coord, vec2(0.0)))
            && all(lessThanEqual(prev_coord, vec2(1.0)));
    if (!validPrev) return g;

    vec4 guideY = samplePathGuide(prev_coord * vec2(resolution_global));
    vec3 x = guideY.xyz;
    float omega = guideY.w;
    float length_x = max(length(x), 1e-20);
    omega = max(omega, length_x);
    g.axis = x / length_x;
    float rho = clamp(length_x / omega, 0.0, 1.0);
    g.kappa = maxent_kappa(length_x, omega);
    g.valid = length_x > 1e-8;
    g.prob = float(g.valid) * strengthMultiplier * rho;
    return g;
}

// ===========================================================================
// Diffuse Direction Sampling with MaxEnt MIS
// ===========================================================================

vec3 sampleDiffuseWithGuide(vec3 geometryNormal, vec3 shadingNormal,
    vec3 ro_o, GuideInfo guide, vec2 xi,
    out vec3 next_rd, out float guideWeight, out float sampledPdf) {
    sampledPdf = 0.0;
    bool useGuide = getRandom() < guide.prob;
    if (useGuide) {
        next_rd = sample_maxent_guiding(guide.axis, guide.kappa, xi);
    } else {
        next_rd = SampleUniformHemisphere(shadingNormal, xi);
    }

    float NoL = max(0.0, dot(shadingNormal, next_rd));
    float geometryNoL = dot(geometryNormal, next_rd);
    // 如果采样到了几何半球下方，直接裁切
    if (NoL <= 0.0 || geometryNoL <= 0.0) {
        guideWeight = 0.0;
        return vec3(0.0);
    }

    float pdfUniform = 1.0 / (2.0 * PI);
    float pdfMaxEnt = guide.prob > 0.0 ? maxent_guiding_pdf(next_rd, guide.axis, guide.kappa) : 0.0;
    float pdfMix = (1.0 - guide.prob) * pdfUniform + guide.prob * pdfMaxEnt;
    sampledPdf = pdfMix;
    guideWeight = (pdfMix > 1e-20) ? (pdfUniform / pdfMix) : 0.0;
    return vec3(guideWeight);
}

// ===========================================================================
// PSR (Primary Surface Replacement) Refractive Chain
// ===========================================================================

PSRResult tracePSRChain(vec3 ro, vec3 rd, vec3 geometryNormal,
    float firstRoughness, bool wasInside, int firstMediumBlockID,
    vec3 firstMediumTint, float firstMediumExtinctionWeight, int baseDepth) {
    PSRResult result;
    result.virtualDist = 0.0;
    result.pathRoughness = 0.0;
    result.refrDir = rd;
    result.endpoint = vec3(0.0);
    result.endpointGeometryNormal = vec3(0.0, 1.0, 0.0);
    result.endpointMacroNormal = vec3(0.0, 1.0, 0.0);
    result.endpointDiffuseAlbedo = vec3(0.0);
    result.endpointRoughness = 1.0;
    result.endpointLight = vec3(0.0);
    result.transmittance = vec3(1.0);
    result.endpointValid = false;
    result.environment = false;

    float r_accum2 = firstRoughness * firstRoughness;
    float firstMediumIor = transportIorFromBlock(firstMediumBlockID);
    float n_camera = wasInside ? firstMediumIor : 1.0;

    vec3 ro_chain = ro;
    vec3 rd_chain = rd;
    bool inside_chain = !wasInside;
    // This is the medium occupied by the segment about to be traced, not the
    // material at its terminal hit. Camera-originated dielectric rays have
    // already crossed the primary interface before entering this function.
    int mediumBlockID = inside_chain ? firstMediumBlockID : 0;
    vec3 mediumTint = firstMediumTint;
    float mediumExtinctionWeight = firstMediumExtinctionWeight;
    vec3 departN = geometryNormal;

    for (int i = 0; i < MAX_REFRACTIVE_BOUNCES; i++) {
        result.refrDir = rd_chain;
        float offsetSign = inside_chain ? -1.0 : 1.0;
        vec3 ro_next, rd_next;
        float t_next = raycast(ro_chain + departN * offsetSign * 0.00025,
                rd_chain, ro_next, rd_next, !inside_chain, false);

        if (t_next < -0.5) {
            result.virtualDist = VPROJDIST_SKY;
            result.environment = true;
            break;
        }

        Material hitMat = evaluateMaterial(tmp_Payload, ro_chain, rd_chain,
            uint(baseDepth + 1 + i));
        int hitBlockID;
        payload_unpackShadow(tmp_Payload.data, hitBlockID);

        // Extinction belongs to the medium occupied by this segment, never to
        // the material at its terminal hit.
        if (inside_chain) {
            result.transmittance *= mediumVolumeTransmittance(t_next,
                mediumBlockID, mediumTint, mediumExtinctionWeight);
        }
        material hitSurf = materialFromEvaluated(hitMat, hitBlockID);

        float n_segment = inside_chain
            ? transportIorFromBlock(mediumBlockID) : 1.0;
        result.virtualDist += t_next * n_camera / n_segment;

        vec3 hitGeomN = payload_unpackGeomNormal(tmp_Payload.data);
        hitGeomN = faceforward(hitGeomN, hitGeomN, rd_chain);

        if (!isTransmissiveBlock(hitBlockID)) {
            result.endpoint = ro_next;
            result.endpointGeometryNormal = hitGeomN;
            result.endpointMacroNormal = hitMat.macroNormal;
            result.endpointDiffuseAlbedo = evaluateNonSpecularAlbedo(
                hitSurf, rd_chain, hitMat.macroNormal)
                * (1.0 - clamp(hitSurf.S.y, 0.0, 1.0));
            result.endpointRoughness = hitSurf.R.x;
            result.endpointLight = max(hitSurf.light, vec3(0.0));
            result.endpointValid = true;
            break;
        }

        // Only interfaces blur the virtual image. The terminal opaque
        // surface's BRDF roughness belongs to diffuse projection, not PSR.
        r_accum2 += hitSurf.R.x * hitSurf.R.x;

        float n_from = inside_chain
            ? transportIorFromBlock(mediumBlockID) : 1.0;
        float n_to = inside_chain
            ? 1.0 : transportIorFromBlock(hitBlockID);
        float interfaceEta = n_from / n_to;
        vec3 next_refract = refract(rd_chain, hitGeomN, interfaceEta);

        if (dot(next_refract, next_refract) <= 0.0) break;

        float interfaceFresnel = clamp(fresnel(-rd_chain, hitGeomN,
            interfaceEta), 0.0, 1.0);
        vec3 interfaceColor = evaluateTransmissionAlbedo(hitSurf)
            * clamp(hitSurf.S.y, 0.0, 1.0);
        result.transmittance *= interfaceColor
            * ((1.0 - interfaceFresnel) * interfaceEta * interfaceEta);

        rd_chain = next_refract;
        result.refrDir = rd_chain;
        ro_chain = ro_next;
        inside_chain = !inside_chain;
        mediumBlockID = inside_chain ? hitBlockID : 0;
        if (inside_chain) {
            mediumTint = hitSurf.Cd;
            mediumExtinctionWeight =
                transportExtinctionWeightFromMaterial(hitSurf);
        } else {
            mediumTint = vec3(1.0);
            mediumExtinctionWeight = 0.0;
        }
        departN = hitGeomN;
    }

    result.pathRoughness = sqrt(r_accum2);
    return result;
}


#endif // DIRT_RT_RAYTRACE_TRANSPORT_GLSL

#ifndef DIRT_RT_RAYTRACE_PATH_TRACE_GLSL
#define DIRT_RT_RAYTRACE_PATH_TRACE_GLSL

// ray1/ray2 forward paths; secondary vertices may select any surface lobe.
// ray3 primary refraction uses the endpoint traversal in refraction.glsl.

// -----------------------------------------------------------------------------------
// Core: Forward Path Tracing
// -----------------------------------------------------------------------------------
void updatePathMedium(material boundarySurface, bool nowInside,
        inout int mediumBlockID, inout vec3 mediumTint,
        inout float mediumExtinctionWeight) {
    if (nowInside) {
        mediumBlockID = transportBlockFromMaterial(boundarySurface);
        mediumTint = boundarySurface.Cd;
        mediumExtinctionWeight =
            transportExtinctionWeightFromMaterial(boundarySurface);
    } else {
        mediumBlockID = 0;
        mediumTint = vec3(1.0);
        mediumExtinctionWeight = 0.0;
    }
}

void Trace(uvec2 coord, vec3 ro, vec3 rd, vec3 lightDir) {
    // === SETUP ===
    uint isEyeInWater = cam.flags & 3u;
    uvec2 xy = coord;
    bool originalInside = isEyeInWater != 0;
    bool inside = originalInside;
    int activeMediumBlockID = isEyeInWater == 1u ? BLOCK_WATER : 0;
    vec3 activeMediumTint = vec3(1.0);
    float activeMediumExtinctionWeight = 0.0;

    vec3 ro_i = ro;
    vec3 rd_i = rd;

    vec3 throughput = vec3(1.0);
    vec3 L_indirect = vec3(0.0);
    vec3 L_direct_0 = vec3(0.0);
    vec3 L_direct_0_dir = -lightDir;
    vec3 L_direct_0_incident = vec3(0.0);
    vec3 reflectionFirstQLiResponse = vec3(1.0);
    GuideInfo reflectionGuide = emptyGuideInfo();
    GuideInfo diffuseGuide = emptyGuideInfo();
    vec2 firstReflectionXi = vec2(0.5);
    float cascadedRoughness2 = 0.0;
    // Sampling metadata for MIS if the current continuation ray reaches the
    // solar disc. Delta/refraction events have no competing sun-NEE strategy.
    float lastBsdfStrategyPdf = 0.0;
    bool lastBsdfDelta = false;
    bool lastNeeCompatible = false;

    vec4 fogColor = (isEyeInWater == 2u) ? vec4(0, 0.05, 0.075, 0.1) * 5.0 : vec4(0, 0.325, 0.295, 0.3);
    vec3 globalEmission = (isEyeInWater == 2u) ? vec3(1, 0.25, 0.05) * 10.0 : vec3(0);

    float fireflyCap = FIREFLY_SUPPRESSION_MULTIPLIER * div_avgExposure;

    FirstBounceData fb = initFirstBounceData(ro, rd);

    // ===== FIRST BOUNCE =====
    material surf;
    loadPrimarySurfaceGBuffer(xy, ro, rd, fb, surf);

    #if defined(FIRST_LOBE_DIFFUSE)
    // The denoised diffuse domain represents the opaque scene behind primary
    // water/glass/ice. PSR can therefore reproject its terminal hit directly into
    // this buffer without running a separate refraction denoiser.
    if (fb.t > -0.5 && surf.S.y > 1e-4) {
        vec3 backgroundRay = fb.rd_i;
        vec3 backgroundHit, backgroundDirection;
        rtCurrentConeWidth = 0.0;
        float backgroundT = raycastIgnoreTransmissive(ro, backgroundRay,
            backgroundHit, backgroundDirection, !originalInside);

        if (backgroundT < -0.5) {
            fb = initFirstBounceData(ro, backgroundRay);
            surf = newMaterial(vec3(0.0), vec3(0.0), vec2(0.0),
                vec4(1.0, 0.0, 0.0, 0.0), vec3(0.0));
        } else {
            vec4 backgroundMotion = getPrimarySurfaceMotion(tmp_Payload);
            Material backgroundMat = evaluateMaterial(tmp_Payload, ro,
                backgroundRay, 0u);
            vec3 rawGeometryNormal =
                payload_unpackGeomNormal(tmp_Payload.data);
            vec3 backgroundGeometryNormal = faceforward(rawGeometryNormal,
                rawGeometryNormal, backgroundRay);
            int backgroundBlockID;
            payload_unpackShadow(tmp_Payload.data, backgroundBlockID);
            surf = materialFromEvaluated(backgroundMat, backgroundBlockID);
            int backgroundMaterialID = getMaxEntMaterialID(tmp_Payload,
                backgroundBlockID);
            markRadianceCacheGeometryHit(xy, backgroundHit,
                backgroundGeometryNormal);

            float backgroundNI = originalInside ? REFRACTIVE_INDEX : 1.0;
            MediumResult backgroundMedium = evalMedium(backgroundT,
                backgroundRay, ro.y, originalInside,
                activeMediumBlockID, activeMediumTint,
                activeMediumExtinctionWeight, fogColor, globalEmission);
            recordFirstBounceGBuffer(backgroundHit, ro,
                backgroundMat.macroNormal, backgroundGeometryNormal,
                backgroundMat.macroNormal, surf, backgroundMaterialID,
                backgroundRay, backgroundRay, backgroundT, backgroundNI,
                backgroundNI, -1, backgroundMedium.emission,
                backgroundMedium.absorption, fb);
            fb.surfaceMotion = backgroundMotion.xyz;
            fb.motionValid = backgroundMotion.w;
        }
    }
    #endif

    rtCurrentConeWidth = max(fb.t, 0.0) * rtCurrentConeSpread;
    vec3 ro_o = fb.p;
    vec3 rd_o = fb.rd_i;

    if (fb.t < -0.5) {
        throughput = vec3(0.0);
        #if defined(FIRST_LOBE_DIFFUSE)
        invalidatePreparedDiffuseHistory(xy, cam.frameId);
        #endif
    } else {
        vec3 geometryNormal = fb.geometry_n;
        vec3 macroNormal = fb.macro_n;
        float surfaceIor = transportIorFromMaterial(surf);
        float n_i = inside ? surfaceIor : 1.0;
        float n_o = inside ? 1.0 : surfaceIor;
        float rs = n_i / n_o;
        LobeProbs lobes = computeLobeProbs(surf, fb.rd_i, macroNormal, rs);
        #if defined(FIRST_LOBE_DIFFUSE)
        // The diffuse continuation never consumes a GGX micro-normal.
        vec3 microNormal = macroNormal;
        #elif defined(FIRST_LOBE_REFLECTION)
        firstReflectionXi = rtBlueNoise2D(xy, 0u);
        reflectionGuide = computeSpecularMaxEntGuide(
            xy, SPECULAR_PATH_GUIDING_STRENGTH);
        vec3 microNormal = macroNormal;
        #endif

        vec3 bsdf_weight = vec3(0.0);
        vec3 next_rd = fb.rd_i;
        int current_type = -1;
        #if defined(FIRST_LOBE_REFLECTION)
        current_type = REFLECTION;
        bool firstDelta;
        handleFirstBounce_Reflection(fb.rd_i, ro_o, macroNormal,
            geometryNormal, microNormal, surf, rs, reflectionGuide,
            firstReflectionXi, bsdf_weight, next_rd,
            reflectionFirstQLiResponse, lastBsdfStrategyPdf, firstDelta);
        lastBsdfDelta = firstDelta;
        lastNeeCompatible = true;
        #else
        current_type = DIFFUSION;
        prepareDiffuseDenoisedSurfaceReprojection(xy, ro_o - ro,
            geometryNormal,
            ro - prevRaytracingCamPos - fb.surfaceMotion,
            fb.motionValid, cam.frameId, rtViewProjection,
            mat3(rtCurrentModelViewLocal),
            rtCurrentProjectionParamsLocal);
        diffuseGuide = computeMaxEntGuide(xy, cam.frameId,
            PATH_GUIDING_STRENGTH);
        handleFirstBounce_Diffuse(fb.rd_i, ro_o, macroNormal,
            geometryNormal, surf, lobes, diffuseGuide,
            rtBlueNoise2D(xy, 0u),
            bsdf_weight, next_rd,
            lastBsdfStrategyPdf);
        lastBsdfDelta = false;
        lastNeeCompatible = true;
        #endif

        bool firstHasSunNee = (current_type == DIFFUSION
                    && lobes.P_diff > 1e-8)
                || (current_type == REFLECTION
                    && !isDeltaSpecular(surf.R.x));
        if (!isDarkened && firstHasSunNee) {
            vec3 sunWi, sunLi;
            float lightPdf;
            if (sampleDirectSun(ro_o, geometryNormal, macroNormal,
                    lightDir, inside,
                    rtBlueNoise2D(xy, 1u),
                    sunWi, sunLi, lightPdf)) {
                L_direct_0_dir = sunWi;
                if (current_type == DIFFUSION) {
                    float proposalPdf = (1.0 - diffuseGuide.prob)
                            * (1.0 / (2.0 * PI));
                    proposalPdf += diffuseGuide.prob * maxent_guiding_pdf(
                        sunWi, diffuseGuide.axis, diffuseGuide.kappa);
                    float misWeight = powerHeuristic(lightPdf, proposalPdf);
                    #if EON_ENABLED
                    // The deferred EON projection supplies f_r * NoL * rho.
                    L_direct_0 = max(vec3(0.0), sunLi
                                * (misWeight / max(lightPdf, 1e-20)));
                    #else
                    float Fd = evaluateDisneyDiffuseFactor(
                            -fb.rd_i, sunWi, macroNormal, surf.R.x);
                    L_direct_0 = max(vec3(0.0), sunLi
                                * (Fd * misWeight / max(PI * lightPdf, 1e-20)));
                    #endif
                } else if (current_type == REFLECTION
                        && !isDeltaSpecular(surf.R.x)) {
                    vec3 fSpecTimesNoL;
                    float pdfNDF;
                    if (evaluateSpecularBRDF(-fb.rd_i, sunWi, macroNormal,
                            surf.Cs, surf.S.x, surf.S.y, rs, surf.R.x,
                            fSpecTimesNoL, pdfNDF)) {
                        float proposalPdf = specularGuideMixturePdf(
                            reflectionGuide, pdfNDF, sunWi);
                        float misWeight = powerHeuristic(
                            lightPdf, proposalPdf);
                        // Convert the light-proposal sample to the same q*Li
                        // measure as the VNDF continuation sample.
                        L_direct_0_incident = max(vec3(0.0), sunLi *
                            (pdfNDF * misWeight
                                / max(lightPdf, 1e-20)));
                    }
                }
            }
        }

        fb.rd_o = next_rd;
        fb.micro_n = microNormal;
        fb.type = current_type;
        fb.n_i = n_i;
        fb.n_o = n_i;
        cascadedRoughness2 = current_type == DIFFUSION
            ? 1.0 : surf.R.x * surf.R.x;
        throughput *= bsdf_weight;
        ro_i = ro_o + geometryNormal * 0.001;
        rd_i = next_rd;
    }

    // ===== SECONDARY LOOP =====
    if (max(throughput.r, max(throughput.g, throughput.b)) > 0.0
            && !any(isnan(throughput)) && !any(isinf(throughput))) {
        // Track whether we arrived via specular (not just current lobe).
        // Primary specular → secondary surface should be cache-eligible at bounce 2.
        bool arrivedViaSpecular = false;
        #if defined(FIRST_LOBE_REFLECTION)
        arrivedViaSpecular = true;
        #endif
        for (int depth = 1; depth < MaxRay; depth++) {
            // --- Ray cast ---
            float t2 = raycast(ro_i, rd_i, ro_o, rd_o, !inside, false);


            // Distance of the actual noisy specular sample. This replaces the
            // unrelated extra ray previously traced along a fitted direction.
            if (depth == 1) {
                fb.reflectionHitDistance = (t2 > -0.5) ? t2 : VPROJDIST_SKY;
            }

            // --- Miss -> sky ---
            if (t2 < -0.5) {
                vec3 sky = sampleSkyNoSun(ro_i.y, rd_i, lightDir).xyz;
                vec3 sunDisc = vec3(0.0);
                sunDisc = sampleSkySunDisc(
                    ro_i.y, rd_i, lightDir).xyz;
                float discWeight = 1.0;
                float lightPdf = sunDirectionPdf(rd_i, lightDir);
                if (lastNeeCompatible && !lastBsdfDelta
                        && lightPdf > 0.0) {
                    discWeight = powerHeuristic(
                            lastBsdfStrategyPdf, lightPdf);
                }
                sky += sunDisc * discWeight;
                if (any(isnan(sky)) || any(isinf(sky))) sky = vec3(0.0);
                vec3 skyContrib = throughput * sky;
                if (depth >= 2) {
                    float lum = dot(skyContrib, vec3(0.2126, 0.7152, 0.0722));
                    skyContrib *= fireflyCap / max(lum, fireflyCap);
                }
                L_indirect += skyContrib;
                break;
            }

            // --- Material ---
            Material surfaceMat = evaluateMaterial(tmp_Payload, ro_i, rd_i,
                uint(depth));
            vec3 geomN = payload_unpackGeomNormal(tmp_Payload.data);
            vec3 geometryNormal = faceforward(geomN, geomN, rd_i);
            #if defined(FIRST_LOBE_DIFFUSE)
            markRadianceCacheGeometryHit(xy, ro_o, geometryNormal);
            #endif
            vec3 macroNormal = surfaceMat.macroNormal;
            int blockID;
            payload_unpackShadow(tmp_Payload.data, blockID);
            material surf = materialFromEvaluated(surfaceMat, blockID);
            float surfaceIor2 = transportIorFromMaterial(surf);
            float n_i2 = inside ? surfaceIor2 : 1.0;
            float n_o2 = inside ? 1.0 : surfaceIor2;
            float rs2 = n_i2 / n_o2;
            bool surfaceInside = inside;

            // --- Medium ---
            MediumResult medium = evalMedium(t2, rd_i, ro_i.y, inside,
                activeMediumBlockID, activeMediumTint,
                activeMediumExtinctionWeight, fogColor, globalEmission);
            // Segment emission is accumulated before applying this segment's
            // transmittance.  Surface terms and all following bounces use the
            // attenuated throughput.
            vec3 bounceEmission = throughput * medium.emission;
            throughput *= medium.absorption;
            bounceEmission += throughput * surf.light;
            if (depth >= 2) {
                float lum = dot(bounceEmission, vec3(0.2126, 0.7152, 0.0722));
                bounceEmission *= fireflyCap / max(lum, fireflyCap);
            }
            L_indirect += bounceEmission;

            // --- Lobe probabilities ---
            LobeProbs lobes = computeLobeProbs(surf, rd_i, macroNormal, rs2);

            // --- Secondary bounce: stochastic mixture ---
            vec3 bsdf_weight;
            vec3 next_rd;
            int current_type;
            bool sampledSpecularLobe;
            float sampledStrategyPdf;
            bool sampledDeltaLobe;
            bool neeCompatible;
            handleSecondaryBounce(rd_i, ro_o, macroNormal, geometryNormal,
                surf, lobes, bsdf_weight, next_rd,
                current_type, sampledSpecularLobe, sampledStrategyPdf,
                sampledDeltaLobe, neeCompatible, inside);
            if (current_type == REFRACTION && inside != surfaceInside) {
                updatePathMedium(surf, inside, activeMediumBlockID,
                    activeMediumTint, activeMediumExtinctionWeight);
            }

            float nextCascadedRoughness2 = current_type == DIFFUSION
                ? 1.0 : cascadedRoughness2 + surf.R.x * surf.R.x;

            // --- Sun NEE ---
            // One visibility ray evaluates every non-delta reflection lobe.
            // Lobe selection is solely a continuation strategy and must not
            // decide whether direct lighting exists at this vertex.
            bool sampledDiffuseLobe = current_type == DIFFUSION;
            bool hasDiffuseSunNee = lobes.P_diff > 1e-8;
            bool hasSpecularSunNee = lobes.P_spec > 1e-8
                    && !isDeltaSpecular(surf.R.x);
            if (!isDarkened && (hasDiffuseSunNee || hasSpecularSunNee)) {
                vec3 sunWi, sunLi;
                float lightPdf;
                vec3 sunL = vec3(0.0);
                if (sampleDirectSun(ro_o, geometryNormal, macroNormal,
                        lightDir, surfaceInside,
                        sunWi, sunLi, lightPdf)) {
                    if (hasDiffuseSunNee) {
                        vec3 diffuseColor = surf.Cd
                                * (1.0 - clamp(surf.S.y, 0.0, 1.0));
                        vec3 fDiffuseTimesNoL;
                        float pdfDiffuse;
                        if (evaluateSurfaceDiffuseBRDF(
                                -rd_i, sunWi, macroNormal, geometryNormal,
                                diffuseColor, surf.R.x,
                                fDiffuseTimesNoL, pdfDiffuse)) {
                            sunL += misLightContribution(
                                    fDiffuseTimesNoL, sunLi, lightPdf,
                                    lobes.P_diff * pdfDiffuse);
                        }
                    }

                    if (hasSpecularSunNee) {
                        vec3 fSpecTimesNoL;
                        float pdfNDF;
                        if (evaluateSpecularBRDF(
                                -rd_i, sunWi, macroNormal,
                                surf.Cs, surf.S.x, surf.S.y, rs2, surf.R.x,
                                fSpecTimesNoL, pdfNDF)) {
                            sunL += misLightContribution(
                                    fSpecTimesNoL, sunLi, lightPdf,
                                    lobes.P_spec * pdfNDF);
                        }
                    }
                }

                vec3 neeContrib = throughput * sunL;
                if (depth >= 2) {
                    float lum = dot(neeContrib, vec3(0.2126, 0.7152, 0.0722));
                    neeContrib *= fireflyCap / max(lum, fireflyCap);
                }
                L_indirect += neeContrib;
            }

            // --- Radiance-cache path termination ---
            // Query only after NEE so direct sunlight is never replaced.
            int bounceNumber = depth + 1;
            float roughThreshold2 =
                RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD
                    * RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD;
            // Cache-eligible when:
            //   Diffuse + setting (≥3 avoids corner block artifacts), OR
            //   Specular lobe (roughness-based OR sharp after 1 reflection), OR
            //   Arrived via specular — secondary surface after mirror reflection.
            bool diffuseEligible = sampledDiffuseLobe
                    && bounceNumber >= RADIANCE_CACHE_DIFFUSE_MIN_BOUNCE;
            bool specularLobeEligible = sampledSpecularLobe
                    && (nextCascadedRoughness2 >= roughThreshold2 || bounceNumber >= 2);
            bool viaSpecularEligible = arrivedViaSpecular
                    && current_type != REFRACTION && bounceNumber >= 2;
            bool cacheEligible = !inside
                    && (diffuseEligible || specularLobeEligible || viaSpecularEligible);
            if (cacheEligible) {
                RadianceCache cache;
                if (loadSecondaryRadianceCache(
                        ro_o, geometryNormal, cache)) {
                    vec3 cachedLighting = sampledDiffuseLobe
                        ? evaluateCachedDiffuseLighting(
                            cache, macroNormal, lobes) : evaluateCachedRoughSpecularLighting(
                            cache, rd_i, macroNormal, surf, lobes);
                    vec3 cacheContrib = throughput * cachedLighting;
                    if (depth >= 2) {
                        float lum = dot(
                                cacheContrib,
                                vec3(0.2126, 0.7152, 0.0722));
                        cacheContrib *=
                            fireflyCap / max(lum, fireflyCap);
                    }
                    L_indirect += cacheContrib;
                    break;
                }
            }

            // --- Throughput update ---
            throughput *= bsdf_weight;
            if (any(isnan(throughput)) || any(isinf(throughput))) {
                throughput = vec3(0.0);
                break;
            }
            if (max(throughput.r, max(throughput.g, throughput.b)) <= 0.0) break;
            cascadedRoughness2 = nextCascadedRoughness2;

            // --- Russian Roulette ---
            if (depth >= 2) {
                float p = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
                if (getRandom() >= p) break;
                throughput /= p;
            }

            // --- Advance ---
            ro_i = ro_o + geometryNormal * ((current_type == REFRACTION) ? -0.001 : 0.001);
            rd_i = next_rd;
            arrivedViaSpecular = (current_type == REFLECTION);
            lastBsdfStrategyPdf = sampledStrategyPdf;
            lastBsdfDelta = sampledDeltaLobe;
            lastNeeCompatible = neeCompatible;
        }
    }

    // ===== OUTPUT =====
    if (any(isnan(L_indirect)) || any(isinf(L_indirect)))
        L_indirect = vec3(0.0);
    if (any(isnan(L_direct_0)) || any(isinf(L_direct_0)))
        L_direct_0 = vec3(0.0);

    #if defined(FIRST_LOBE_DIFFUSE)
    writeDiffuseOutput(
        xy, fb, L_indirect,
        L_direct_0,
        L_direct_0_dir, ro);
    #elif defined(FIRST_LOBE_REFLECTION)
    writeReflectionOutput(xy, fb, L_indirect,
        L_direct_0_incident, L_direct_0_dir,
        reflectionFirstQLiResponse, ro);
    #endif
}

#endif // DIRT_RT_RAYTRACE_PATH_TRACE_GLSL

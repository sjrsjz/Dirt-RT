#ifndef DIRT_RT_RAYTRACE_GBUFFER_IO_GLSL
#define DIRT_RT_RAYTRACE_GBUFFER_IO_GLSL

// Primary G-buffer serialization and lobe output encoding.

// ===========================================================================
// G-Buffer & Output
// ===========================================================================

FirstBounceData initFirstBounceData(vec3 ro, vec3 rd) {
    FirstBounceData fb;
    fb.p = ro;
    fb.macro_n = -rd;
    fb.geometry_n = -rd;
    fb.micro_n = -rd;
    fb.rd_o = rd;
    fb.rd_i = rd;
    fb.refr_dir = rd;
    fb.specularAlbedo = vec3(0.0);
    fb.diffuseAlbedo = vec3(0.0);
    fb.transmissionAlbedo = vec3(0.0);
    fb.emission_val = vec3(0.0);
    fb.light_surf = vec3(0.0);
    fb.absorption = vec3(1.0);
    fb.t = -1.0;
    fb.roughness = 1.0;
    fb.n_i = 1.0;
    fb.n_o = 1.0;
    fb.t2_ior_adjusted = 0.0;
    fb.pathRoughness = 0.0;
    fb.reflectionHitDistance = 0.0;
    fb.surfaceMotion = vec3(0.0);
    fb.motionValid = 0.0;
    fb.type = -1;
    fb.materialID = 0;
    return fb;
}

void recordFirstBounceGBuffer(
    vec3 ro_o, vec3 ro, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, int materialID, vec3 rd_i, vec3 next_rd, float t,
    float n_i, float n_o, int lobeType, vec3 segmentEmission,
    vec3 currentAbsorption, inout FirstBounceData fb
) {
    fb.p = ro_o;
    fb.macro_n = macroNormal;
    fb.geometry_n = geometryNormal;
    fb.micro_n = microNormal;
    fb.t = t;
    fb.type = lobeType;
    fb.materialID = materialID;
    fb.n_i = n_i;
    fb.n_o = n_o;
    fb.rd_i = rd_i;
    fb.rd_o = next_rd;
    fb.emission_val = segmentEmission;
    fb.light_surf = surf.light;
    fb.roughness = surf.R.x;
    fb.absorption = currentAbsorption;

    fb.specularAlbedo = evaluateSpecularAlbedo(
            surf, rd_i, macroNormal, n_i / max(n_o, 1e-5));
    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    vec3 nonSpecularAlbedo = evaluateNonSpecularAlbedo(surf, rd_i, macroNormal);
    fb.diffuseAlbedo = nonSpecularAlbedo * (1.0 - transmissionSelector);
    fb.transmissionAlbedo = evaluateTransmissionAlbedo(surf)
        * transmissionSelector;
}

void writePrimarySurfaceGBuffer(uvec2 xy, FirstBounceData fb,
    material surf, vec3 ro) {
    writePrimaryGeometry(xy, fb.geometry_n, fb.roughness,
        fb.materialID, fb.macro_n, fb.t);
    writeAlbedosPath(GEO_N_ALBEDOS, xy,
        fb.specularAlbedo, fb.diffuseAlbedo);
    writeMiscTransport(GEO_N_MISC, xy,
        fb.transmissionAlbedo, fb.emission_val);
    writeLightAbs(GEO_N_LIGHTABS, xy, fb.light_surf, fb.absorption);
    writeSurfaceMotion(xy, fb.surfaceMotion, fb.motionValid);
    writePrimaryMaterial(xy, surf.Cs, surf.Cd, surf.S);
}

void loadPrimarySurfaceGBuffer(uvec2 xy, vec3 ro, vec3 primaryRay,
    out FirstBounceData fb, out material surf) {
    fb = initFirstBounceData(ro, vec3(0.0, 0.0, -1.0));

    uvec4 geometryWords = readPrimaryGeometryWords(xy);
    fb.t = uintBitsToFloat(geometryWords.w);
    fb.geometry_n = decodeNormalU(geometryWords.x);
    fb.roughness = unpackHalf2x16(geometryWords.y).x;
    fb.pathRoughness = fb.roughness;
    fb.materialID = int(geometryWords.y >> 16u);
    // Continuation passes run before ray4 publishes the current camera into
    // FrameData. Reuse the exact current ray already constructed from cam;
    // reconstructPrimaryRay() intentionally still sees the previous frame here.
    fb.rd_i = primaryRay;
    vec3 positionRelative = fb.t >= 0.0
        ? fb.rd_i * fb.t : vec3(0.0);
    fb.p = ro + positionRelative;
    fb.macro_n = decodeNormalU(geometryWords.z);
    #if defined(FIRST_LOBE_DIFFUSE)
    readAlbedosPath(GEO_N_ALBEDOS, xy, fb.specularAlbedo,
        fb.diffuseAlbedo);
    #elif defined(FIRST_LOBE_REFLECTION)
    fb.specularAlbedo = readPrimarySpecularAlbedo(xy);
    #endif
    fb.micro_n = fb.macro_n;
    #if defined(FIRST_LOBE_REFRACTION)
    fb.transmissionAlbedo = readPrimaryTransmission(xy);
    #endif
    fb.rd_o = fb.rd_i;
    fb.refr_dir = fb.rd_i;

    vec3 Cs, Cd;
    vec2 S;
    readPrimaryMaterial(xy, Cs, Cd, S);
    surf = newMaterial(Cs, Cd, S, vec4(fb.roughness, 0.0, 0.0, 0.0),
            vec3(0.0));
    #if defined(FIRST_LOBE_DIFFUSE)
    readSurfaceMotion(xy, fb.surfaceMotion, fb.motionValid);
    #endif
}

void writeDiffuseOutput(uvec2 xy, FirstBounceData fb, vec3 L_indirect,
    vec3 L_direct_0, vec3 L_direct_0_dir, vec3 ro) {
    MaxEntEncoding combinedMaxEnt = init_maxent();
    float mask = 0.0;
    if (fb.t > -0.5) {
        L_indirect = clamp(L_indirect, 0.0, GI_CLAMP_MAX);
        L_direct_0 = clamp(L_direct_0, 0.0, 32000.0);
        MaxEntEncoding indMaxEnt = radiance_to_maxent(L_indirect, fb.rd_o);
        MaxEntEncoding dirMaxEnt = radiance_to_maxent(
                L_direct_0, L_direct_0_dir);
        indMaxEnt.CoCg += dirMaxEnt.CoCg;
        indMaxEnt.maxEntY += dirMaxEnt.maxEntY;
        combinedMaxEnt = indMaxEnt;
        mask = 1.0;
    }
    // For one sample sqrt(E[Y²]) = |Y|. Encoded luminance is nonnegative.
    writeDiffuseLightRT(xy, combinedMaxEnt,
        max(combinedMaxEnt.maxEntY.w, 0.0));
    if (mask > 0.5) {
        writeDiffuseSurface(xy, fb.macro_n, fb.geometry_n, fb.t,
            fb.diffuseAlbedo, fb.roughness, fb.surfaceMotion,
            fb.motionValid);
    } else {
        writeDiffuseSurfaceInvalid(xy);
    }
}

vec3 recoverFirstBounceIncidentMeasure(vec3 pathContribution,
        vec3 firstQLiResponse) {
    vec3 incident = vec3(0.0);
    if (abs(firstQLiResponse.x) > 1e-8)
        incident.x = pathContribution.x / firstQLiResponse.x;
    if (abs(firstQLiResponse.y) > 1e-8)
        incident.y = pathContribution.y / firstQLiResponse.y;
    if (abs(firstQLiResponse.z) > 1e-8)
        incident.z = pathContribution.z / firstQLiResponse.z;
    // The path contribution is (f/p)*Li and firstQLiResponse is f/q.
    // Their quotient is (q/p)*Li, whose expectation under the actual mixture
    // proposal p is the q*Li measure consumed by the deferred decoder.
    // Delta mirrors retain the existing Li convention with response F.
    if (any(isnan(incident)) || any(isinf(incident))) return vec3(0.0);
    return clamp(incident, vec3(0.0), vec3(400.0 * div_avgExposure));
}

void writeReflectionOutput(uvec2 xy, FirstBounceData fb,
        vec3 indirectContribution, vec3 directIncident,
        vec3 directIncidentDirection, vec3 firstQLiResponse, vec3 ro) {
    vec3 refl_R = fb.rd_o;
    float refl_vprojdist = fb.reflectionHitDistance;
    SpecularMaxEnt signal = emptySpecularMaxEnt();
    if (fb.t > -0.5) {
        vec3 incident = recoverFirstBounceIncidentMeasure(
            indirectContribution, firstQLiResponse);
        signal = specularMaxEntFromRgbDirection(incident, refl_R);
        SpecularMaxEnt directSignal = specularMaxEntFromRgbDirection(
            clamp(directIncident, vec3(0.0),
                vec3(64000.0)), directIncidentDirection);
        signal.maxEntY += directSignal.maxEntY;
        signal.CoCg += directSignal.CoCg;
        signal = sanitizeSpecularMaxEnt(signal);
    }
    writeReflMaxEntSample(xy, signal, refl_vprojdist, refl_R);
}

#if defined(FIRST_LOBE_REFRACTION)
void TraceRefractionPSR(uvec2 xy, vec3 ro, vec3 primaryRay) {
    PSRResolveData outputData;
    outputData.endpointRelative = vec3(0.0);
    outputData.refractedDirection = vec3(0.0, 0.0, -1.0);
    outputData.geometryNormal = vec3(0.0, 1.0, 0.0);
    outputData.macroNormal = vec3(0.0, 1.0, 0.0);
    outputData.diffuseAlbedo = vec3(0.0);
    outputData.roughness = 1.0;
    outputData.pathRoughness = 0.0;
    outputData.transmittance = vec3(1.0);
    outputData.surfaceLight = vec3(0.0);
    outputData.endpointValid = false;
    outputData.environment = false;
    outputData.screenCandidate = false;

    FirstBounceData fb;
    material surf;
    loadPrimarySurfaceGBuffer(xy, ro, primaryRay, fb, surf);
    if (fb.t > -0.5 && surf.S.y > 1e-4) {
        bool wasInside = (cam.flags & 3u) != 0u;
        float surfaceIor = transportIorFromMaterial(surf);
        float nI = wasInside ? surfaceIor : 1.0;
        float nO = wasInside ? 1.0 : surfaceIor;
        vec3 chainDirection = refract(fb.rd_i, fb.geometry_n, nI / nO);
        if (dot(chainDirection, chainDirection) > 0.0) {
            rtCurrentConeWidth = max(fb.t, 0.0) * rtCurrentConeSpread;
            int firstMediumBlockID = transportBlockFromMaterial(surf);
            PSRResult psr = tracePSRChain(fb.p, chainDirection,
                fb.geometry_n, surf.R.x, wasInside,
                firstMediumBlockID, surf.Cd,
                transportExtinctionWeightFromMaterial(surf), 0);
            float firstEta = nI / nO;
            float firstFresnel = clamp(fresnel(-fb.rd_i, fb.geometry_n,
                firstEta), 0.0, 1.0);
            outputData.endpointRelative = psr.endpoint - ro;
            outputData.refractedDirection = psr.refrDir;
            outputData.geometryNormal = psr.endpointGeometryNormal;
            outputData.macroNormal = psr.endpointMacroNormal;
            outputData.diffuseAlbedo = psr.endpointDiffuseAlbedo;
            outputData.roughness = psr.endpointRoughness;
            outputData.pathRoughness = psr.pathRoughness;
            outputData.transmittance = psr.transmittance
                * ((1.0 - firstFresnel) * firstEta * firstEta);
            outputData.surfaceLight = psr.endpointLight;
            outputData.endpointValid = psr.endpointValid;
            outputData.environment = psr.environment;
            outputData.screenCandidate = psr.endpointValid
                && surf.R.x < PSR_ROUGHNESS_THRESHOLD
                && psr.pathRoughness < PSR_PATH_ROUGHNESS_THRESHOLD;
        }
    }
    writePSRResolve(xy, outputData);
}
#endif

#endif // DIRT_RT_RAYTRACE_GBUFFER_IO_GLSL
